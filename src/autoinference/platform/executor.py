"""Executors: where a request runs.

``LocalExecutor`` runs handlers in-process on a thread pool (development and
tests). ``ModalExecutor`` spawns the model's worker class on Modal and polls
the function call; the worker runs ``runtime.execute`` with the same handler.
Both return the plain result dict that ``runtime.execute`` builds, so the
service treats them identically.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from .catalog import ModelSpec
from .runtime import execute
from .storage import Storage

PENDING = "pending"


class Executor(Protocol):
    name: str

    def submit(self, spec: ModelSpec, inputs: dict[str, Any], request_id: str, project_id: str,
               seed: int) -> str:
        """Start the request; return a provider job id."""

    def poll(self, job_id: str) -> dict[str, Any] | None:
        """The result dict once finished, else ``None``."""

    def cancel(self, job_id: str) -> bool: ...


class LocalExecutor:
    name = "local"

    def __init__(self, storage: Storage, workroot: Path | None = None, workers: int = 2) -> None:
        self.storage = storage
        self.workroot = workroot
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model")
        self._jobs: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, spec, inputs, request_id, project_id, seed):
        future = self._pool.submit(execute, spec, inputs, request_id, project_id, self.storage, seed, self.workroot)
        with self._lock:
            self._jobs[request_id] = future
        return request_id

    def poll(self, job_id):
        with self._lock:
            future = self._jobs.get(job_id)
        if future is None:
            return {"status": "failed", "error": "unknown local job (the service restarted)"}
        if not future.done():
            return None
        with self._lock:
            self._jobs.pop(job_id, None)
        exc = future.exception()
        if exc is not None:
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        return future.result()

    def cancel(self, job_id):
        with self._lock:
            future = self._jobs.get(job_id)
        return bool(future and future.cancel())


class ModalExecutor:
    name = "modal"

    def __init__(self, app_name: str) -> None:
        import modal

        self.modal = modal
        self.app_name = app_name
        self._classes: dict[str, Any] = {}

    def _worker(self, spec: ModelSpec):
        if spec.id not in self._classes:
            self._classes[spec.id] = self.modal.Cls.from_name(self.app_name, spec.worker_name)
        return self._classes[spec.id]

    def submit(self, spec, inputs, request_id, project_id, seed):
        call = self._worker(spec)().run.spawn(spec.id, inputs, request_id, project_id, seed)
        return call.object_id

    def poll(self, job_id):
        call = self.modal.FunctionCall.from_id(job_id)
        try:
            return call.get(timeout=0)
        except TimeoutError:
            return None
        except Exception as exc:  # noqa: BLE001 - a crashed or expired call is a failed request
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    def cancel(self, job_id):
        try:
            self.modal.FunctionCall.from_id(job_id).cancel()
            return True
        except Exception:  # noqa: BLE001
            return False


def make_executor(settings, storage: Storage) -> Executor:
    if settings.executor == "modal":
        return ModalExecutor(settings.modal_app)
    return LocalExecutor(storage, settings.storage_dir / "work")
