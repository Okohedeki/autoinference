"""Cloud GPU backends: run a model on rented GPUs instead of the local machine.

The service (job store, campaigns, trust checks, tuning, audit log) runs
anywhere small. Only model calls go to the cloud. A campaign stage is split into
chunks and the chunks run on many GPUs at once, so a stage of a few hundred
candidates takes about as long as one chunk.

Two providers sit behind one ``Runner`` interface, chosen per stage with the
``provider`` option:

- ``modal`` (default): serverless GPUs with per-second billing and scale to
  zero. Workers are in ``modal_app.py`` and deploy with ``autoinference bio deploy``.
- ``runpod``: a RunPod Serverless endpoint built from ``deploy/runpod`` and
  running ``runpod_worker.py``. Usually cheaper per GPU-hour, so it suits
  large batch campaigns.

Both workers run the same :func:`run_boltz_chunk`, so the results match. SDKs
and credentials are needed only when a cloud backend first calls out.

Billing: every chunk reports the GPU seconds it used inside the container.
Those land in each prediction's ``extras["gpu_seconds"]``, so the stage's cost
is the GPU time actually billed, not the (much shorter) wall time of a
parallel stage.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from ..registry import Manifest, register
from .backends import SLOT, Backend, Boltz2
from .types import Candidate, Prediction, Target

APP_NAME = "autoinference-bio"
WORKER = "Boltz2Worker"


class Runner(Protocol):
    """Sends chunks to remote workers. Each call returns the dict built by :func:`run_boltz_chunk`."""

    def call(self, request: dict) -> dict: ...

    def map(self, requests: list[dict]) -> list[dict]: ...


class ModalRunner:
    """Calls the deployed Modal worker class, optionally on a different GPU type than it was deployed with."""

    def __init__(self, app: str = APP_NAME, worker: str = WORKER, gpu: str | None = None) -> None:
        try:
            import modal
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("the modal provider needs Modal: pip install -e \".[cloud]\", "
                               "then python -m modal setup") from exc
        cls = modal.Cls.from_name(app, worker)
        if gpu:
            cls = cls.with_options(gpu=gpu)
        self._method = cls().predict

    def call(self, request: dict) -> dict:
        return self._method.remote(**request)

    def map(self, requests: list[dict]) -> list[dict]:
        if not requests:
            return []
        keys = list(requests[0])
        return list(self._method.map(*[[r[k] for r in requests] for k in keys]))


class RunPodRunner:
    """Calls a RunPod Serverless endpoint running ``runpod_worker.py``.

    Jobs go to the endpoint's queue with ``POST /run`` and are polled with
    ``GET /status/<id>``. ``/runsync`` is not used because it keeps results
    for about a minute, and a chunk can take longer than that. The endpoint's
    GPU type and worker counts are set when the endpoint is created on RunPod.
    """

    BASE = "https://api.runpod.ai/v2"
    TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

    def __init__(self, endpoint: str, api_key: str | None = None, poll_seconds: float = 3.0,
                 timeout: float = 3600.0) -> None:
        if not endpoint:
            raise ValueError("the runpod provider needs the endpoint id: options {\"endpoint\": \"<id>\"}")
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RuntimeError("the runpod provider needs RUNPOD_API_KEY set")
        self.url = f"{self.BASE}/{endpoint}"
        self.poll_seconds = poll_seconds
        self.timeout = timeout

    def _http(self, method: str, path: str, body: dict | None = None) -> dict:
        import json
        import urllib.request

        req = urllib.request.Request(self.url + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    def submit(self, request: dict) -> str:
        return self._http("POST", "/run", {"input": request})["id"]

    def wait(self, job_ids: list[str]) -> list[dict]:
        done: dict[str, dict] = {}
        deadline = time.time() + self.timeout
        while len(done) < len(job_ids):
            if time.time() > deadline:
                for jid in job_ids:
                    if jid not in done:
                        self._http("POST", f"/cancel/{jid}")
                raise TimeoutError(f"RunPod jobs still running after {self.timeout:.0f} s")
            for jid in job_ids:
                if jid in done:
                    continue
                st = self._http("GET", f"/status/{jid}")
                if st.get("status") in self.TERMINAL:
                    if st["status"] != "COMPLETED" or not isinstance(st.get("output"), dict)                             or "error" in st["output"]:
                        raise RuntimeError(f"RunPod job {jid} {st.get('status')}: {st.get('error') or st.get('output')}")
                    done[jid] = st["output"]
            if len(done) < len(job_ids):
                time.sleep(self.poll_seconds)
        return [done[j] for j in job_ids]

    def call(self, request: dict) -> dict:
        return self.wait([self.submit(request)])[0]

    def map(self, requests: list[dict]) -> list[dict]:
        return self.wait([self.submit(r) for r in requests]) if requests else []


PROVIDERS = ("modal", "runpod")


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


@register(SLOT, "boltz2-cloud", Manifest(kind="kernel", devices=("cloud",), lossless_class="tolerance-exact",
                                         deps=("modal",), provenance="Boltz (MIT licence)", licence="MIT"))
class Boltz2Cloud(Backend):
    """Boltz-2 co-folding and affinity on cloud GPUs, fanned out across many workers.

    The first candidate runs alone so the target's alignment is computed once
    (and the model weights land in the shared cache on a fresh deployment).
    The alignment is then sent with every other chunk, and kept locally in
    ``msa_cache`` so later campaigns on the same target skip it. Scores match
    the local ``boltz2`` backend.
    """

    name = "boltz2-cloud"
    seconds_per_item = 60.0
    produces_structures = True

    def __init__(self, provider: str | None = None, gpu: str | None = None, chunk_size: int = 4,
                 diffusion_samples: int = 1, extra_args: str = "", app: str = APP_NAME,
                 endpoint: str | None = None, msa_cache: str | None = None, runner: Runner | None = None) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.provider = provider or os.environ.get("AUTOINFERENCE_CLOUD", "modal")
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown cloud provider {self.provider!r}; choose one of {', '.join(PROVIDERS)}")
        self.endpoint = endpoint or os.environ.get("AUTOINFERENCE_RUNPOD_ENDPOINT")
        self.gpu = gpu
        self.app = app
        self.chunk_size = chunk_size
        self.diffusion_samples = diffusion_samples
        self.extra_args = extra_args
        default_cache = Path(os.environ.get("AUTOINFERENCE_CACHE", Path.home() / ".cache" / "autoinference"))
        self.msa_cache = Path(msa_cache) if msa_cache else default_cache / "msa"
        self._runner = runner
        self.version = "unknown"  # the worker reports the Boltz version it ran
        self.gpu_type = gpu or "deployed default"

    @property
    def runner(self) -> Runner:
        if self._runner is None:
            if self.provider == "runpod":
                self._runner = RunPodRunner(self.endpoint or "")
            else:
                self._runner = ModalRunner(self.app, WORKER, self.gpu)
        return self._runner

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "provider": self.provider, "gpu": self.gpu_type,
                "diffusion_samples": self.diffusion_samples}

    def _msa_file(self, sequence: str) -> Path:
        return self.msa_cache / f"{hashlib.sha256(sequence.encode()).hexdigest()[:24]}.csv"

    def _request(self, target: Target, chunk: list[Candidate], seed: int, msa: str | None) -> dict:
        return {"target": {"id": target.id, "sequence": target.sequence},
                "candidates": [{"id": c.id, "smiles": c.smiles, "sequence": c.sequence} for c in chunk],
                "seed": seed, "msa": msa, "diffusion_samples": self.diffusion_samples,
                "extra_args": self.extra_args}

    def _predictions(self, reply: dict) -> list[Prediction]:
        self.version = reply.get("version", self.version)
        self.gpu_type = reply.get("gpu", self.gpu_type)
        preds = reply["predictions"]
        per_item = reply.get("gpu_seconds", 0.0) / max(len(preds), 1)
        return [Prediction(p["candidate_id"], self.name, p["scores"], structure=p.get("structure"),
                           extras={"gpu_seconds": per_item}) for p in preds]

    def predict(self, target, candidates, seed):
        if not candidates:
            return []
        msa = None
        if target.msa_path:
            msa = Path(target.msa_path).read_text()
        elif self._msa_file(target.sequence).exists():
            msa = self._msa_file(target.sequence).read_text()
        out: list[Prediction] = []
        rest = list(candidates)
        if msa is None:
            first = self.runner.call(self._request(target, rest[:1], seed, None))
            out += self._predictions(first)
            rest = rest[1:]
            msa = first.get("msa")
            if msa:
                self.msa_cache.mkdir(parents=True, exist_ok=True)
                self._msa_file(target.sequence).write_text(msa)
        replies = self.runner.map([self._request(target, chunk, seed, msa) for chunk in _chunks(rest, self.chunk_size)])
        for reply in replies:
            out += self._predictions(reply)
        return out


def run_boltz_chunk(target: dict, candidates: list[dict], seed: int, msa: str | None, diffusion_samples: int = 1,
                    extra_args: str = "", cache: str | None = None, command: str = "boltz") -> dict:
    """Runs inside a worker container: score one chunk with the ``boltz`` command and return plain data."""
    start = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix="chunk-"))
    msa_path = None
    if msa:
        msa_path = work / "target.csv"
        msa_path.write_text(msa)
    tgt = Target(target["id"], target["sequence"], str(msa_path) if msa_path else None)
    model = Boltz2(command=command, workdir=str(work), use_msa_server=msa is None,
                   diffusion_samples=diffusion_samples, extra_args=extra_args, cache=cache,
                   msa_cache=str(work / "msa-cache"))
    preds = model.predict(tgt, [Candidate(**c) for c in candidates], seed)
    computed = model._cached_msa(tgt.sequence)
    return {
        "predictions": [{"candidate_id": p.candidate_id, "scores": p.scores, "structure": p.structure} for p in preds],
        "msa": msa if msa else (computed.read_text() if computed.exists() else None),
        "gpu_seconds": time.perf_counter() - start,
        "version": model.version,
        "gpu": "unknown",  # the worker fills in the GPU it ran on
    }
