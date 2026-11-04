"""The control plane's logic, independent of HTTP: projects and keys, requests, files, usage, audit.

One ``Platform`` object backs the HTTP API, the CLI and the MCP server. A
background thread moves running requests to their final state by polling the
executor, records usage, writes the audit chain and sends webhooks.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from . import db
from .catalog import Catalog, ModelSpec, digest
from .config import Settings
from .executor import Executor, make_executor
from .pipelines import estimate_pipeline, parse_pipeline, plan_calls, resolve, step_scope
from .storage import Storage, input_key, make_storage

STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
KEY_PREFIX = "ai"


def _now() -> float:
    return time.time()


def _id() -> str:
    return uuid.uuid4().hex[:24]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Platform:
    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None,
                 storage: Storage | None = None, executor: Executor | None = None, start: bool = True) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.validate()
        self.catalog = catalog or Catalog(self.settings.models_dir)
        self.storage = storage or make_storage(self.settings)
        self.executor = executor or make_executor(self.settings, self.storage)
        self.engine = db.connect(self.settings.database_url)
        self._stop = threading.Event()
        self._pipeline_lock = threading.Lock()  # one advance at a time; the poller and API calls both advance runs
        self._poller = threading.Thread(target=self._poll_loop, daemon=True, name="platform-poller")
        if start:
            self._poller.start()

    def close(self) -> None:
        self._stop.set()

    # --- projects and keys ------------------------------------------------------------------------

    def create_project(self, name: str, credits: float = 0.0) -> dict[str, Any]:
        project_id = _id()
        with self.engine.begin() as conn:
            conn.execute(db.projects.insert().values(id=project_id, name=name, created=_now(), credits=credits))
        self._audit(project_id, "project_created", name=name)
        return {"id": project_id, "name": name, "credits": credits}

    def create_key(self, project_id: str, name: str = "default") -> dict[str, Any]:
        """Returns the key once; only its hash is stored."""
        prefix = secrets.token_hex(4)
        key = f"{KEY_PREFIX}_{prefix}_{secrets.token_urlsafe(32)}"
        key_id = _id()
        with self.engine.begin() as conn:
            conn.execute(db.api_keys.insert().values(id=key_id, project_id=project_id, name=name, prefix=prefix,
                                                     key_hash=hash_key(key), created=_now()))
        self._audit(project_id, "key_created", key=key_id, name=name)
        return {"id": key_id, "name": name, "prefix": prefix, "key": key}

    def revoke_key(self, project_id: str, key_id: str) -> None:
        with self.engine.begin() as conn:
            n = conn.execute(db.api_keys.update().where(db.api_keys.c.id == key_id,
                                                        db.api_keys.c.project_id == project_id,
                                                        db.api_keys.c.revoked.is_(None))
                             .values(revoked=_now())).rowcount
        if not n:
            raise KeyError(key_id)
        self._audit(project_id, "key_revoked", key=key_id)

    def list_keys(self, project_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.api_keys.c.id, db.api_keys.c.name, db.api_keys.c.prefix,
                                          db.api_keys.c.created, db.api_keys.c.last_used, db.api_keys.c.revoked)
                                .where(db.api_keys.c.project_id == project_id)).mappings().all()
        return [dict(r) for r in rows]

    def authenticate(self, key: str) -> dict[str, Any] | None:
        """The project for an API key, or ``None``."""
        if not key or key.count("_") < 2:
            return None
        h = hash_key(key)
        with self.engine.begin() as conn:
            row = conn.execute(sa.select(db.api_keys.c.id, db.api_keys.c.project_id, db.api_keys.c.key_hash,
                                         db.api_keys.c.revoked)
                               .where(db.api_keys.c.prefix == key.split("_")[1])).mappings().first()
            if row is None or row["revoked"] or not hmac.compare_digest(row["key_hash"], h):
                return None
            conn.execute(db.api_keys.update().where(db.api_keys.c.id == row["id"]).values(last_used=_now()))
            project = conn.execute(sa.select(db.projects).where(db.projects.c.id == row["project_id"])).mappings().first()
        return dict(project) | {"key_id": row["id"]}

    # --- catalog ----------------------------------------------------------------------------------

    def models(self) -> list[dict[str, Any]]:
        return [s.to_public() for s in self.catalog.public()]

    def model(self, model: str) -> dict[str, Any]:
        return self.catalog.get(model).to_schema()

    def estimate(self, model: str, inputs: dict[str, Any], units: float | None = None) -> dict[str, Any]:
        spec = self.catalog.get(model)
        spec.validate_inputs(inputs)
        measured = self._measured(spec.id)
        n = units if units is not None else 1.0
        return {"model": spec.id, "version": spec.version, "unit": spec.unit, "units": n,
                "price_per_unit": spec.price_per_unit, "price": round(n * spec.price_per_unit, 6),
                "seconds_per_unit": measured, "seconds_source": "measured" if measured else "none"}

    def _measured(self, model_id: str) -> float | None:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(sa.func.sum(db.usage_events.c.gpu_seconds),
                                         sa.func.sum(db.usage_events.c.units))
                               .where(db.usage_events.c.model == model_id)).first()
        if row and row[1]:
            return row[0] / row[1]
        return None

    # --- requests ---------------------------------------------------------------------------------

    def submit(self, project: dict[str, Any], model: str, inputs: dict[str, Any], mode: str = "queue",
               webhook_url: str | None = None, idempotency_key: str | None = None, seed: int = 0,
               pipeline: tuple[str, str, int] | None = None) -> dict[str, Any]:
        spec = self.catalog.get(model)
        if mode == "run" and not spec.sync:
            raise ValueError(f"{spec.id} must be called through /queue")
        spec.validate_inputs(inputs)
        self._check_files(project["id"], spec, inputs)
        self._check_balance(project, spec)
        if idempotency_key:
            existing = self._by_idempotency(project["id"], idempotency_key)
            if existing:
                return existing
        request_id = _id()
        with self.engine.begin() as conn:
            conn.execute(db.requests.insert().values(
                id=request_id, project_id=project["id"], model=spec.id, version=spec.version, mode=mode,
                status="queued", inputs=inputs, inputs_digest=digest(inputs), webhook_url=webhook_url,
                idempotency_key=idempotency_key, created=_now(),
                pipeline_id=pipeline[0] if pipeline else None, step_id=pipeline[1] if pipeline else None,
                item_index=pipeline[2] if pipeline else None))
        self._audit(project["id"], "request_submitted", request=request_id, model=spec.id,
                    version=spec.version, inputs_digest=digest(inputs))
        try:
            job_id = self.executor.submit(spec, inputs, request_id, project["id"], seed)
            self._update(request_id, status="running", provider=self.executor.name, provider_job_id=job_id,
                         started=_now())
        except Exception as exc:  # noqa: BLE001 - the provider refused; the request fails cleanly
            self._update(request_id, status="failed", error=f"submit failed: {exc}", finished=_now())
        return self.status(project["id"], request_id)

    def _check_files(self, project_id: str, spec: ModelSpec, inputs: dict[str, Any]) -> None:
        for name in spec.file_inputs():
            file_id = inputs.get(name)
            if file_id is None:
                continue
            with self.engine.connect() as conn:
                row = conn.execute(sa.select(db.files.c.id).where(db.files.c.id == file_id,
                                                                  db.files.c.project_id == project_id)).first()
            if row is None:
                raise ValueError(f"{name}: unknown file {file_id}")
            if not self.storage.exists(input_key(project_id, file_id)):
                raise ValueError(f"{name}: file {file_id} has not been uploaded yet")

    def _check_balance(self, project: dict[str, Any], spec: ModelSpec) -> None:
        cap = project.get("spend_cap")
        if cap is None:
            return
        spent = self.usage(project["id"])["total_price"]
        if spent + spec.price_per_unit > cap:
            raise PermissionError(f"spend cap of {cap} reached")

    def _by_idempotency(self, project_id: str, key: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(db.requests.c.id).where(db.requests.c.project_id == project_id,
                                                                 db.requests.c.idempotency_key == key)).first()
        return self.status(project_id, row[0]) if row else None

    def _update(self, request_id: str, **fields: Any) -> None:
        with self.engine.begin() as conn:
            conn.execute(db.requests.update().where(db.requests.c.id == request_id).values(**fields))

    def status(self, project_id: str, request_id: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(db.requests).where(db.requests.c.id == request_id,
                                                            db.requests.c.project_id == project_id)).mappings().first()
        if row is None:
            raise KeyError(request_id)
        out = {k: row[k] for k in ("id", "model", "version", "mode", "status", "created", "started", "finished",
                                   "units", "gpu_seconds", "price", "provider", "error")}
        out["request_id"] = out.pop("id")
        if row["pipeline_id"]:
            out["pipeline"] = {"run_id": row["pipeline_id"], "step": row["step_id"], "index": row["item_index"]}
        return out

    def result(self, project_id: str, request_id: str, url_expires: int = 3600) -> dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(db.requests).where(db.requests.c.id == request_id,
                                                            db.requests.c.project_id == project_id)).mappings().first()
        if row is None:
            raise KeyError(request_id)
        if row["status"] != "succeeded":
            raise FileNotFoundError(f"request {request_id} is {row['status']}")
        outputs = dict(row["outputs"] or {})
        for name, value in outputs.items():
            if isinstance(value, dict) and "file" in value:
                outputs[name] = {**value, "url": self.storage.download_url(value["file"], url_expires)}
        return {"request_id": request_id, "model": row["model"], "version": row["version"], "status": "succeeded",
                "outputs": outputs, "units": row["units"], "gpu_seconds": row["gpu_seconds"], "price": row["price"],
                "manifest": row["manifest"]}

    def cancel(self, project_id: str, request_id: str) -> dict[str, Any]:
        st = self.status(project_id, request_id)
        if st["status"] in ("queued", "running"):
            with self.engine.connect() as conn:
                job = conn.execute(sa.select(db.requests.c.provider_job_id)
                                   .where(db.requests.c.id == request_id)).scalar()
            if job:
                self.executor.cancel(job)
            self._update(request_id, status="cancelled", finished=_now())
            self._audit(project_id, "request_cancelled", request=request_id)
        return self.status(project_id, request_id)

    def wait(self, project_id: str, request_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        while True:
            st = self.status(project_id, request_id)
            if st["status"] in ("succeeded", "failed", "cancelled") or time.time() >= deadline:
                return st
            time.sleep(0.05)

    def list_requests(self, project_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            ids = [r[0] for r in conn.execute(sa.select(db.requests.c.id).where(db.requests.c.project_id == project_id)
                                              .order_by(db.requests.c.created.desc()).limit(limit))]
        return [self.status(project_id, i) for i in ids]

    # --- pipelines --------------------------------------------------------------------------------

    def estimate_pipeline(self, spec: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_pipeline(spec)
        for step in parsed.steps:
            self.catalog.get(step.model)
        return estimate_pipeline(parsed, inputs, self.catalog)

    def submit_pipeline(self, project: dict[str, Any], spec: dict[str, Any], inputs: dict[str, Any],
                        webhook_url: str | None = None, rerun_from: tuple[str, str] | None = None) -> dict[str, Any]:
        """Start a pipeline run. ``rerun_from=(run_id, step_id)`` reuses that run's results before the step."""
        parsed = parse_pipeline(spec)
        for step in parsed.steps:
            self.catalog.get(step.model)
        run_id = _id()
        state: dict[str, Any] = {"inputs": inputs, "steps": {}, "approved": [], "outputs": None}
        if rerun_from:
            old = self._run_row(project["id"], rerun_from[0])
            if rerun_from[1] not in [st.id for st in parsed.steps]:
                raise KeyError(rerun_from[1])
            for st in parsed.steps:
                if st.id == rerun_from[1]:
                    break
                prev = old["state"]["steps"].get(st.id)
                if prev is None or prev["status"] not in ("succeeded", "skipped"):
                    raise ValueError(f"step {st.id} did not finish in run {rerun_from[0]}, cannot reuse it")
                state["steps"][st.id] = {**prev, "reused_from": rerun_from[0], "price": 0.0}  # paid for already
        with self.engine.begin() as conn:
            conn.execute(db.requests.insert().values(
                id=run_id, project_id=project["id"], model=f"pipeline:{parsed.name}", version="1", mode="queue",
                status="running", inputs=inputs, inputs_digest=digest({"spec": spec, "inputs": inputs}),
                webhook_url=webhook_url, created=_now(), started=_now()))
            conn.execute(db.pipeline_runs.insert().values(
                id=run_id, project_id=project["id"], name=parsed.name, spec=spec, state=state, status="running",
                created=_now()))
        self._audit(project["id"], "pipeline_submitted", run=run_id, name=parsed.name,
                    steps=[st.id for st in parsed.steps], spec_digest=digest(spec))
        self._advance_pipeline(run_id)
        return self.pipeline_status(project["id"], run_id)

    def _run_row(self, project_id: str, run_id: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(db.pipeline_runs).where(db.pipeline_runs.c.id == run_id,
                                                                 db.pipeline_runs.c.project_id == project_id)
                               ).mappings().first()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    def _save_run(self, run_id: str, state: dict[str, Any], status: str, current_step: str | None,
                  finished: bool = False) -> None:
        with self.engine.begin() as conn:
            conn.execute(db.pipeline_runs.update().where(db.pipeline_runs.c.id == run_id)
                         .values(state=state, status=status, current_step=current_step,
                                 finished=_now() if finished else None))

    def approve_step(self, project_id: str, run_id: str, step_id: str) -> dict[str, Any]:
        row = self._run_row(project_id, run_id)
        state = row["state"]
        if step_id not in state["approved"]:
            state["approved"].append(step_id)
            self._save_run(run_id, state, "running" if row["status"] == "waiting_approval" else row["status"],
                           row["current_step"])
            self._audit(project_id, "pipeline_step_approved", run=run_id, step=step_id)
        self._advance_pipeline(run_id)
        return self.pipeline_status(project_id, run_id)

    def cancel_pipeline(self, project_id: str, run_id: str) -> dict[str, Any]:
        row = self._run_row(project_id, run_id)
        if row["status"] in ("running", "waiting_approval"):
            with self.engine.connect() as conn:
                children = conn.execute(sa.select(db.requests.c.id).where(
                    db.requests.c.pipeline_id == run_id, db.requests.c.status.in_(("queued", "running")))).all()
            for (cid,) in children:
                self.cancel(project_id, cid)
            self._save_run(run_id, row["state"], "cancelled", row["current_step"], finished=True)
            self._update(run_id, status="cancelled", finished=_now())
            self._audit(project_id, "pipeline_cancelled", run=run_id)
        return self.pipeline_status(project_id, run_id)

    def pipeline_status(self, project_id: str, run_id: str) -> dict[str, Any]:
        row = self._run_row(project_id, run_id)
        spec = parse_pipeline(row["spec"])
        state = row["state"]
        steps = []
        price = 0.0
        for st in spec.steps:
            res = state["steps"].get(st.id, {"status": "pending"})
            entry = {"id": st.id, "model": st.model, "status": res.get("status", "pending"),
                     "approve": st.approve, "calls": len(res.get("calls", [])),
                     "kept": len(res.get("items", [])) if st.map else None,
                     "request_ids": [c["request_id"] for c in res.get("calls", []) if c.get("request_id")],
                     "price": res.get("price", 0.0), "attempts": res.get("attempts", 0),
                     "reused_from": res.get("reused_from"), "error": res.get("error")}
            price += entry["price"] or 0.0
            steps.append(entry)
        return {"run_id": run_id, "name": row["name"], "status": row["status"], "current_step": row["current_step"],
                "steps": steps, "price": round(price, 6), "outputs": state.get("outputs"),
                "created": row["created"], "finished": row["finished"]}

    def pipeline_step(self, project_id: str, run_id: str, step_id: str) -> dict[str, Any]:
        """Everything about one step: its calls, their inputs and outputs, what was kept."""
        row = self._run_row(project_id, run_id)
        spec = parse_pipeline(row["spec"])
        st = spec.step(step_id)
        res = row["state"]["steps"].get(step_id, {"status": "pending"})
        calls = []
        for c in res.get("calls", []):
            entry = {"index": c["index"], "inputs": c["inputs"], "request_id": c.get("request_id")}
            if c.get("request_id"):
                entry["status"] = self.status(project_id, c["request_id"])
            calls.append(entry)
        kept_ids = {it["request_id"] for it in res.get("items", [])}
        return {"run_id": run_id, "step": step_id, "model": st.model, "status": res.get("status"),
                "map": st.map, "keep": None if st.keep is None else st.keep.__dict__, "approve": st.approve,
                "calls": calls, "outputs": res.get("outputs"), "items": res.get("items"),
                "dropped": [it for it in res.get("all_items", []) if it["request_id"] not in kept_ids],
                "error": res.get("error")}

    def _advance_pipelines(self) -> int:
        with self.engine.connect() as conn:
            ids = [r[0] for r in conn.execute(sa.select(db.pipeline_runs.c.id)
                                              .where(db.pipeline_runs.c.status == "running"))]
        n = 0
        for run_id in ids:
            try:
                n += self._advance_pipeline(run_id)
            except Exception as exc:  # noqa: BLE001 - one broken run must not stop the others
                print(f"pipeline {run_id}: {type(exc).__name__}: {exc}", flush=True)
        return n

    def _advance_pipeline(self, run_id: str) -> int:
        """Move a run forward as far as it can go without waiting. Returns 1 if anything changed."""
        with self._pipeline_lock:
            return self._advance_pipeline_locked(run_id)

    def _advance_pipeline_locked(self, run_id: str) -> int:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(db.pipeline_runs).where(db.pipeline_runs.c.id == run_id)).mappings().first()
        if row is None or row["status"] not in ("running", "waiting_approval"):
            return 0
        project_id, state = row["project_id"], row["state"]
        spec = parse_pipeline(row["spec"])
        project = {"id": project_id, "spend_cap": None}
        changed = 0
        for st in spec.steps:
            res = state["steps"].setdefault(st.id, {"status": "pending", "calls": [], "attempts": 0})
            if res["status"] in ("succeeded", "skipped"):
                continue
            if res["status"] == "pending":
                if st.approve and st.id not in state["approved"]:
                    if row["status"] != "waiting_approval" or row["current_step"] != st.id:
                        self._save_run(run_id, state, "waiting_approval", st.id)
                        return 1
                    return changed
                calls = plan_calls(st, state)
                if not calls:
                    res.update(status="skipped", outputs={}, items=[], all_items=[])
                    changed = 1
                    continue
                for call in calls:
                    submitted = self.submit(project, st.model, call["inputs"], "queue",
                                            seed=st.seed if st.seed is not None else 0,
                                            pipeline=(run_id, st.id, call["index"]))
                    call["request_id"] = submitted["request_id"]
                res.update(status="running", calls=calls, attempts=res.get("attempts", 0) + 1)
                self._save_run(run_id, state, "running", st.id)
                return 1
            # running: look at the children
            statuses = {c["request_id"]: self.status(project_id, c["request_id"]) for c in res["calls"]}
            if any(v["status"] in ("queued", "running") for v in statuses.values()):
                if changed:
                    self._save_run(run_id, state, "running", st.id)
                return changed
            failed = [c for c in res["calls"] if statuses[c["request_id"]]["status"] != "succeeded"]
            if failed:
                if res["attempts"] <= st.retries:
                    for c in failed:
                        c["request_id"] = self.submit(project, st.model, c["inputs"], "queue",
                                                      seed=st.seed if st.seed is not None else 0,
                                                      pipeline=(run_id, st.id, c["index"]))["request_id"]
                    res["attempts"] += 1
                    self._save_run(run_id, state, "running", st.id)
                    return 1
                errors = [statuses[c["request_id"]].get("error") for c in failed]
                res.update(status="failed", error=f"{len(failed)} of {len(res['calls'])} calls failed: {errors[0]}")
                self._save_run(run_id, state, "failed", st.id, finished=True)
                self._update(run_id, status="failed", error=f"step {st.id}: {res['error']}", finished=_now())
                self._audit(project_id, "pipeline_failed", run=run_id, step=st.id, error=res["error"])
                return 1
            items = []
            for c in res["calls"]:
                r = self.result(project_id, c["request_id"], url_expires=0)
                outputs = _strip_urls(r["outputs"])
                items.append({"index": c["index"], "inputs": c["inputs"], "outputs": outputs,
                              "request_id": c["request_id"], "item": c.get("item"),
                              "checks": outputs.get("checks") if isinstance(outputs, dict) else None})
            res["price"] = round(sum(statuses[c["request_id"]].get("price") or 0.0 for c in res["calls"]), 6)
            if st.map is None:
                res.update(status="succeeded", outputs=items[0]["outputs"], items=items, all_items=items,
                           request_id=items[0]["request_id"])
            else:
                kept = st.keep.apply(items) if st.keep else items
                res.update(status="succeeded", outputs={}, items=kept, all_items=items)
            changed = 1
        # every step finished
        outputs = resolve(spec.outputs, step_scope(state)) if spec.outputs else {
            st.id: state["steps"][st.id].get("outputs") or state["steps"][st.id].get("items") for st in spec.steps}
        state["outputs"] = outputs
        total = round(sum(state["steps"][st.id].get("price", 0.0) or 0.0 for st in spec.steps), 6)
        self._save_run(run_id, state, "succeeded", None, finished=True)
        self._update(run_id, status="succeeded", outputs=outputs, price=total, units=len(spec.steps),
                     manifest={"pipeline": spec.name, "spec_digest": digest(row["spec"]),
                               "steps": {st.id: [c.get("request_id") for c in state["steps"][st.id].get("calls", [])]
                                         for st in spec.steps}, "outputs_digest": digest(outputs)},
                     finished=_now())
        self._audit(project_id, "pipeline_succeeded", run=run_id, price=total, outputs_digest=digest(outputs))
        return 1

    # --- files ------------------------------------------------------------------------------------

    def create_file(self, project_id: str, filename: str, content_type: str | None = None,
                    size: int | None = None) -> dict[str, Any]:
        file_id = _id()
        key = input_key(project_id, file_id)
        expires = _now() + self.settings.retention_days * 86400
        with self.engine.begin() as conn:
            conn.execute(db.files.insert().values(id=file_id, project_id=project_id, key=key, filename=filename,
                                                  content_type=content_type, size=size, created=_now(),
                                                  expires=expires))
        return {"file_id": file_id, "filename": filename, "upload_url": self.storage.upload_url(key, content_type),
                "expires": expires}

    def file(self, project_id: str, file_id: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(db.files).where(db.files.c.id == file_id,
                                                         db.files.c.project_id == project_id)).mappings().first()
        if row is None:
            raise KeyError(file_id)
        return {"file_id": file_id, "filename": row["filename"], "content_type": row["content_type"],
                "size": row["size"], "uploaded": self.storage.exists(row["key"]),
                "url": self.storage.download_url(row["key"]) if self.storage.exists(row["key"]) else None}

    def put_local_file(self, key: str, data: bytes) -> None:
        """Local mode only: the API receives the upload itself instead of R2."""
        self.storage.put_bytes(key, data)  # type: ignore[attr-defined]

    def read_local_file(self, key: str) -> bytes:
        return self.storage.read(key)  # type: ignore[attr-defined]

    # --- usage and audit --------------------------------------------------------------------------

    def usage(self, project_id: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.usage_events.c.model, sa.func.sum(db.usage_events.c.units),
                                          sa.func.sum(db.usage_events.c.gpu_seconds),
                                          sa.func.sum(db.usage_events.c.price), sa.func.count())
                                .where(db.usage_events.c.project_id == project_id)
                                .group_by(db.usage_events.c.model)).all()
        by_model = [{"model": m, "units": u or 0, "gpu_seconds": g or 0, "price": p or 0, "requests": n}
                    for m, u, g, p, n in rows]
        return {"by_model": by_model, "total_price": round(sum(r["price"] for r in by_model), 6),
                "total_gpu_seconds": round(sum(r["gpu_seconds"] for r in by_model), 3)}

    def _audit(self, project_id: str, event: str, **payload: Any) -> None:
        with self.engine.begin() as conn:
            prev = conn.execute(sa.select(db.audit_log.c.hash).where(db.audit_log.c.project_id == project_id)
                                .order_by(db.audit_log.c.id.desc()).limit(1)).scalar() or "0" * 64
            created = _now()
            h = digest({"project": project_id, "event": event, "payload": payload, "prev": prev, "created": created})
            conn.execute(db.audit_log.insert().values(project_id=project_id, event=event, payload=payload,
                                                      prev_hash=prev, hash=h, created=created))

    def audit(self, project_id: str, limit: int = 200) -> dict[str, Any]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.audit_log).where(db.audit_log.c.project_id == project_id)
                                .order_by(db.audit_log.c.id)).mappings().all()
        prev, intact, broken = "0" * 64, True, None
        for row in rows:
            expected = digest({"project": project_id, "event": row["event"], "payload": row["payload"],
                               "prev": prev, "created": row["created"]})
            if row["prev_hash"] != prev or row["hash"] != expected:
                intact, broken = False, row["id"]
                break
            prev = row["hash"]
        entries = [{"id": r["id"], "event": r["event"], "payload": r["payload"], "hash": r["hash"],
                    "created": r["created"]} for r in rows[-limit:]]
        return {"intact": intact, "broken_at": broken, "entries": len(rows), "recent": entries}

    # --- the poller -------------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - keep polling
                print(f"poller: {type(exc).__name__}: {exc}", flush=True)
            self._stop.wait(0.2 if self.executor.name == "local" else 2.0)

    def poll_once(self) -> int:
        """Finish every running request whose job is done. Returns how many changed."""
        with self.engine.connect() as conn:
            running = conn.execute(sa.select(db.requests.c.id, db.requests.c.project_id, db.requests.c.model,
                                             db.requests.c.provider_job_id, db.requests.c.webhook_url,
                                             db.requests.c.started)
                                   .where(db.requests.c.status == "running",
                                          db.requests.c.provider_job_id.isnot(None))).mappings().all()
        changed = 0
        for row in running:
            spec = self.catalog.get(row["model"])
            result = self.executor.poll(row["provider_job_id"])
            if result is None:
                if _now() - (row["started"] or _now()) > spec.hardware.timeout + 120:
                    result = {"status": "failed", "error": f"timed out after {spec.hardware.timeout} s"}
                    self.executor.cancel(row["provider_job_id"])
                else:
                    continue
            self._finish(row, spec, result)
            changed += 1
        changed += self._advance_pipelines()
        return changed

    def _finish(self, row: Any, spec: ModelSpec, result: dict[str, Any]) -> None:
        request_id, project_id = row["id"], row["project_id"]
        gpu_seconds = float(result.get("gpu_seconds") or 0.0)
        if result.get("status") == "succeeded":
            units = float(result.get("units") or 1.0)
            price = round(units * spec.price_per_unit, 6)
            manifest = {"model": spec.id, "version": spec.version, "fingerprint": result.get("fingerprint"),
                        "inputs_digest": result.get("inputs_digest"), "outputs_digest": digest(result["outputs"]),
                        "device": result.get("device"), "provider": self.executor.name,
                        "timings": result.get("timings"), "units": units}
            self._update(request_id, status="succeeded", outputs=result["outputs"], units=units,
                         gpu_seconds=gpu_seconds, price=price, manifest=manifest, finished=_now())
            with self.engine.begin() as conn:
                conn.execute(db.usage_events.insert().values(project_id=project_id, request_id=request_id,
                                                             model=spec.id, units=units, gpu_seconds=gpu_seconds,
                                                             price=price, created=_now()))
                conn.execute(db.projects.update().where(db.projects.c.id == project_id)
                             .values(credits=db.projects.c.credits - price))
            self._audit(project_id, "request_succeeded", request=request_id, model=spec.id, units=units,
                        price=price, outputs_digest=manifest["outputs_digest"])
        else:
            self._update(request_id, status="failed", error=result.get("error", "unknown error"),
                         gpu_seconds=gpu_seconds, finished=_now())
            self._audit(project_id, "request_failed", request=request_id, model=spec.id,
                        error=result.get("error", "unknown error"))
        if row["webhook_url"]:
            threading.Thread(target=self._webhook, args=(row["webhook_url"], project_id, request_id),
                             daemon=True).start()

    def _webhook(self, url: str, project_id: str, request_id: str) -> None:
        import httpx

        payload = self.status(project_id, request_id)
        body = json.dumps(payload).encode()
        signature = hmac.new(project_id.encode(), body, hashlib.sha256).hexdigest()
        try:
            httpx.post(url, content=body, headers={"Content-Type": "application/json",
                                                   "X-Signature": f"sha256={signature}"}, timeout=10)
        except Exception as exc:  # noqa: BLE001 - webhooks are best effort
            print(f"webhook to {url} failed: {exc}", flush=True)


def _strip_urls(outputs: Any) -> Any:
    if isinstance(outputs, dict):
        return {k: ({kk: vv for kk, vv in v.items() if kk != "url"} if isinstance(v, dict) and "file" in v
                    else _strip_urls(v)) for k, v in outputs.items()}
    return outputs


def workspace_from(root: str | Path) -> Settings:
    """Settings for a local workspace directory (the CLI's default)."""
    s = Settings.from_env()
    s.storage_dir = Path(root)
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    if s.database_url.startswith("sqlite"):
        s.database_url = f"sqlite:///{(s.storage_dir / 'platform.sqlite').as_posix()}"
    return s
