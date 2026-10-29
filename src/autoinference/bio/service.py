"""The hosted service: a job store, one runner thread that owns the GPU, and an HTTP API.

Everything a customer sends or gets back lives under one workspace directory:

    <root>/jobs.sqlite        job index
    <root>/jobs/<id>/         spec, results, manifest and poses for one campaign
    <root>/tuned/<id>.json    models fitted on the customer's assay data
    <root>/timings.json       measured seconds per candidate for each backend
    <root>/audit.jsonl        hash-chained log of every submission, result and replay

The same ``BioService`` object backs the HTTP API, the MCP server for agents
and the command line, so all three behave identically.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .backends import backend_names, create_backend
from .campaign import estimate, run_campaign
from .provenance import AuditLog, digest
from .tuning import TunedModel, fit_rescorer
from .types import CampaignSpec

DEFAULT_GPU_HOUR_PRICE = 1.0  # placeholder; set the real rental price with --gpu-hour-price


class BioService:
    def __init__(self, root: str | Path, gpu_hour_price: float = DEFAULT_GPU_HOUR_PRICE,
                 backend_options: dict[str, dict] | None = None, start: bool = True) -> None:
        self.root = Path(root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.gpu_hour_price = gpu_hour_price
        self.backend_options = backend_options or {}
        self.audit = AuditLog(self.root / "audit.jsonl")
        self._db_lock = threading.Lock()
        self._gpu_lock = threading.Lock()  # one campaign or replay on the hardware at a time
        self._db = sqlite3.connect(self.root / "jobs.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, name TEXT, "
                         "created REAL, finished REAL, error TEXT, progress TEXT)")
        self._db.commit()
        self._backends: dict[str, Any] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="bio-runner")
        if start:
            for (job_id,) in self._db.execute("SELECT id FROM jobs WHERE status IN ('queued', 'running')"):
                self._queue.put(job_id)  # resume work that was in flight when the service stopped
            self._thread.start()

    # --- helpers ----------------------------------------------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        path = (self.root / "jobs" / job_id).resolve()
        if path.parent != (self.root / "jobs").resolve():
            raise KeyError(job_id)
        return path

    def _set(self, job_id: str, **fields: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._db_lock:
            self._db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
            self._db.commit()

    def _timings(self) -> dict[str, float]:
        path = self.root / "timings.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _record_timing(self, backend: str, seconds_per_item: float | None) -> None:
        if not seconds_per_item:
            return
        timings = self._timings()
        old = timings.get(backend)
        timings[backend] = seconds_per_item if old is None else 0.7 * old + 0.3 * seconds_per_item
        (self.root / "timings.json").write_text(json.dumps(timings, indent=2))

    def _backend(self, name: str):
        if name not in self._backends:
            self._backends[name] = create_backend(name, **self.backend_options.get(name, {}))
        return self._backends[name]

    # --- API --------------------------------------------------------------------------------------

    def backends(self) -> list[dict]:
        from ..registry import REGISTRY

        out = []
        for name in backend_names():
            entry = REGISTRY.get("bio_backend", name)
            cls = entry.factory
            out.append({"name": name, "seconds_per_item": self._timings().get(name, cls.seconds_per_item),
                        "produces_structures": cls.produces_structures, "devices": list(entry.manifest.devices),
                        "requires": list(entry.manifest.deps), "doc": (cls.__doc__ or "").strip().split("\n")[0]})
        return out

    def estimate(self, spec: dict) -> dict:
        return estimate(CampaignSpec.from_dict(spec), self.gpu_hour_price, self._timings())

    def submit(self, spec: dict) -> dict:
        parsed = CampaignSpec.from_dict(spec)  # validate before accepting
        if parsed.tuned_model:
            TunedModel.load(self.root / "tuned", parsed.tuned_model)
        job_id = uuid.uuid4().hex[:12]
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True)
        (job_dir / "spec.json").write_text(json.dumps(spec, indent=2))
        quote = self.estimate(spec)
        with self._db_lock:
            self._db.execute("INSERT INTO jobs VALUES (?, 'queued', ?, ?, NULL, NULL, NULL)",
                             (job_id, parsed.name, time.time()))
            self._db.commit()
        self.audit.append("submitted", job=job_id, spec_digest=digest(parsed.to_dict()),
                          candidates=len(parsed.candidates), quoted_cost=quote["cost"])
        self._queue.put(job_id)
        return {"job_id": job_id, "status": "queued", "estimate": quote}

    def status(self, job_id: str) -> dict:
        with self._db_lock:
            row = self._db.execute("SELECT id, status, name, created, finished, error, progress FROM jobs "
                                   "WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        keys = ("job_id", "status", "name", "created", "finished", "error", "progress")
        out = dict(zip(keys, row))
        out["progress"] = json.loads(out["progress"]) if out["progress"] else None
        return out

    def jobs(self) -> list[dict]:
        with self._db_lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM jobs ORDER BY created DESC")]
        return [self.status(i) for i in ids]

    def results(self, job_id: str, top: int | None = None) -> dict:
        path = self._job_dir(job_id) / "result.json"
        if not path.exists():
            raise FileNotFoundError(f"job {job_id} has no results yet ({self.status(job_id)['status']})")
        data = json.loads(path.read_text())
        if top is not None:
            data["results"] = data["results"][:top]
        return data

    def wait(self, job_id: str, timeout: float = 3600.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.status(job_id)
            if st["status"] in ("done", "failed"):
                return st
            time.sleep(0.05)
        raise TimeoutError(job_id)

    def fit_tuned_model(self, job_id: str, labels: dict[str, int], model_id: str | None = None) -> dict:
        """Fit a re-scorer from assay outcomes for candidates scored in ``job_id``."""
        data = self.results(job_id)
        model = fit_rescorer(data["results"] + data.get("dropped", []), {k: int(v) for k, v in labels.items()},
                             model_id or f"tuned-{uuid.uuid4().hex[:8]}")
        model.save(self.root / "tuned")
        self.audit.append("tuned", model=model.id, job=job_id, data_digest=model.data_digest,
                          metrics={k: model.metrics[k] for k in ("n", "actives", "cv_auc", "best_single_auc")})
        return model.to_dict()

    def verify(self, job_id: str) -> dict:
        """Replay a finished job from its spec and seed and compare the output digests."""
        original = self.results(job_id)["manifest"]
        spec = CampaignSpec.from_dict(json.loads((self._job_dir(job_id) / "spec.json").read_text()))
        tuned = TunedModel.load(self.root / "tuned", spec.tuned_model) if spec.tuned_model else None
        with self._gpu_lock:
            replay = run_campaign(spec, self._stage_backends(spec), tuned)
        same = replay.manifest["output_digest"] == original["output_digest"]
        self.audit.append("verified", job=job_id, identical=same, digest=replay.manifest["output_digest"])
        return {"job_id": job_id, "identical": same, "original_digest": original["output_digest"],
                "replay_digest": replay.manifest["output_digest"]}

    def audit_check(self) -> dict:
        ok, detail = self.audit.verify()
        return {"intact": ok, "detail": detail, "entries": len(self.audit.entries())}

    # --- runner -----------------------------------------------------------------------------------

    def _stage_backends(self, spec: CampaignSpec) -> dict:
        names = {s.backend for s in spec.stages if not s.options} | set(spec.consensus)
        return {n: self._backend(n) for n in names}

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._execute(job_id)
            except Exception as exc:
                self._set(job_id, status="failed", finished=time.time(),
                          error=f"{exc}\n{traceback.format_exc(limit=3)}")
                self.audit.append("failed", job=job_id, error=str(exc))

    def _execute(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        spec = CampaignSpec.from_dict(json.loads((job_dir / "spec.json").read_text()))
        tuned = TunedModel.load(self.root / "tuned", spec.tuned_model) if spec.tuned_model else None
        self._set(job_id, status="running")

        def progress(backend: str, done: int, total: int) -> None:
            self._set(job_id, progress=json.dumps({"backend": backend, "done": done, "total": total}))

        with self._gpu_lock:
            result = run_campaign(spec, self._stage_backends(spec), tuned, progress)
        poses = job_dir / "poses"
        for rec in result.results + result.dropped:
            text = rec.pop("structure", None)
            if text is not None:
                poses.mkdir(exist_ok=True)
                name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in rec["id"])
                (poses / f"{name}.pdb").write_text(text)
                rec["structure_file"] = f"poses/{name}.pdb"
        for report in result.stage_reports:
            self._record_timing(report["backend"], report["seconds_per_item"])
        gpu_seconds = sum(r["seconds"] for r in result.stage_reports)
        out = result.to_dict()
        out["gpu_seconds"] = gpu_seconds
        out["cost"] = round(gpu_seconds / 3600 * self.gpu_hour_price, 6)
        (job_dir / "result.json").write_text(json.dumps(out, indent=2))
        self._set(job_id, status="done", finished=time.time())
        self.audit.append("finished", job=job_id, output_digest=result.manifest["output_digest"],
                          gpu_seconds=round(gpu_seconds, 4), results=len(result.results))


def make_handler(service: BioService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            pass

        def _json(self, status: int, payload: Any) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def _route(self, method: str) -> None:
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            query = dict(q.split("=", 1) for q in self.path.split("?", 1)[1].split("&") if "=" in q) \
                if "?" in self.path else {}
            try:
                if method == "GET" and parts == ["health"]:
                    return self._json(200, {"status": "ok"})
                if method == "GET" and parts == ["v1", "backends"]:
                    return self._json(200, service.backends())
                if method == "GET" and parts == ["v1", "jobs"]:
                    return self._json(200, service.jobs())
                if method == "GET" and parts == ["v1", "audit"]:
                    return self._json(200, service.audit_check())
                if method == "POST" and parts == ["v1", "estimate"]:
                    return self._json(200, service.estimate(self._body()))
                if method == "POST" and parts == ["v1", "campaigns"]:
                    return self._json(202, service.submit(self._body()))
                if len(parts) >= 3 and parts[:2] == ["v1", "jobs"]:
                    job_id, rest = parts[2], parts[3:]
                    if method == "GET" and not rest:
                        return self._json(200, service.status(job_id))
                    if method == "GET" and rest == ["results"]:
                        top = int(query["top"]) if "top" in query else None
                        return self._json(200, service.results(job_id, top))
                    if method == "POST" and rest == ["verify"]:
                        return self._json(200, service.verify(job_id))
                    if method == "POST" and rest == ["tune"]:
                        body = self._body()
                        return self._json(200, service.fit_tuned_model(job_id, body["labels"], body.get("model_id")))
                self._json(404, {"error": "not found"})
            except KeyError as exc:
                self._json(404, {"error": f"unknown {exc}"})
            except FileNotFoundError as exc:
                self._json(409, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": f"bad request: {exc}"})

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


def serve(service: BioService, host: str = "127.0.0.1", port: int = 8100) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"bio service on http://{host}:{port}/v1 (workspace {service.root})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
