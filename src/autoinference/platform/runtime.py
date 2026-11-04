"""What runs inside a worker: load a handler once, run it per request, move files in and out.

A handler module defines::

    def load(spec) -> state            # once per container; returns whatever run needs
    def run(inputs, ctx, state) -> dict  # per request

``run`` returns ``{"outputs": {...}, "units": <float>}``. File-typed outputs are
``pathlib.Path`` values; the runtime uploads them and replaces them with
storage keys. File-typed inputs arrive as local paths. The handler never sees
a provider SDK or a credential.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import ModelSpec, digest
from .storage import Storage, input_key, output_key

_LOADED: dict[str, tuple[Any, Any]] = {}   # model id -> (module, state)


@dataclass
class Context:
    request_id: str
    project_id: str
    workdir: Path
    seed: int = 0
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.logs.append(line)
        print(line, flush=True)

    def path(self, name: str) -> Path:
        """A place for an output file."""
        out = self.workdir / "out"
        out.mkdir(parents=True, exist_ok=True)
        return out / name


def load_handler(spec: ModelSpec) -> tuple[Any, Any]:
    if spec.id in _LOADED:
        return _LOADED[spec.id]
    module_name = f"autoinference_model_{spec.worker_name}"
    module_spec = importlib.util.spec_from_file_location(module_name, spec.handler_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    state = module.load(spec) if hasattr(module, "load") else None
    _LOADED[spec.id] = (module, state)
    return module, state


def _gpu_name() -> str:
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "cpu"


def execute(spec: ModelSpec, inputs: dict[str, Any], request_id: str, project_id: str, storage: Storage,
            seed: int = 0, workroot: Path | None = None) -> dict[str, Any]:
    """Run one request. Returns a result dict that is plain JSON (safe to send across providers)."""
    started = time.perf_counter()
    if workroot is not None:
        Path(workroot).mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"req-{request_id}-", dir=workroot))
    ctx = Context(request_id, project_id, workdir, seed)
    try:
        local_inputs = dict(inputs)
        for name in spec.file_inputs():
            if name in local_inputs and local_inputs[name] is not None:
                file_id = local_inputs[name]
                local_inputs[name] = storage.get(input_key(project_id, file_id), workdir / "in" / file_id)
        module, state = load_handler(spec)
        load_seconds = time.perf_counter() - started
        result = module.run(local_inputs, ctx, state)
        if not isinstance(result, dict) or "outputs" not in result:
            raise RuntimeError("handler.run must return {'outputs': ..., 'units': ...}")
        outputs = dict(result["outputs"])
        for name in spec.file_outputs():
            value = outputs.get(name)
            if isinstance(value, (str, Path)) and Path(value).exists():
                key = output_key(project_id, request_id, f"{name}{Path(value).suffix}")
                storage.put(key, Path(value))
                outputs[name] = {"file": key, "size": Path(value).stat().st_size}
        elapsed = time.perf_counter() - started
        return {
            "status": "succeeded",
            "outputs": outputs,
            "units": float(result.get("units", 1.0)),
            "gpu_seconds": elapsed,
            "timings": {"load": load_seconds, "total": elapsed},
            "device": _gpu_name(),
            "fingerprint": spec.fingerprint(),
            "inputs_digest": digest(inputs),
            "logs": ctx.logs[-200:],
        }
    except Exception as exc:  # noqa: BLE001 - reported to the caller as a failed request
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8), "gpu_seconds": time.perf_counter() - started,
                "device": _gpu_name(), "logs": ctx.logs[-200:]}
    finally:
        if os.environ.get("AUTOINFERENCE_KEEP_WORKDIRS") != "1":
            shutil.rmtree(workdir, ignore_errors=True)
