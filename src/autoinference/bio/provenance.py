"""Reproducibility: run manifests, deterministic execution and a tamper-evident audit log.

A manifest records everything that decides a campaign's output: the spec, the
seed, each backend's fingerprint and the software environment. Its output
digest is a hash of every score, so replaying the manifest on the same machine
must give the same digest. The audit log chains each entry to the previous
one's hash, so editing or deleting a past entry is detectable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any


def digest(obj: Any) -> str:
    """SHA-256 of an object's canonical JSON form."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(data).hexdigest()


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    from .. import __version__

    env["autoinference"] = __version__
    return env


def make_deterministic(seed: int) -> None:
    """Seed every RNG and ask PyTorch for deterministic kernels in this process."""
    import random

    random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def output_digest(results: list[dict]) -> str:
    """Hash of the ranked ids and every score, with floats in full precision."""
    return digest([[r["id"], {k: repr(v) for k, v in sorted(r["scores"].items())}] for r in results])


class AuditLog:
    """Append-only JSON-lines log whose entries are hash-chained."""

    GENESIS = "0" * 64

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        last = self.GENESIS
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def append(self, event: str, **data: Any) -> dict:
        with self._lock:
            entry = {"time": time.time(), "event": event, "data": data, "prev": self._last_hash()}
            entry["hash"] = digest({k: entry[k] for k in ("time", "event", "data", "prev")})
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
            return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def verify(self) -> tuple[bool, str]:
        prev = self.GENESIS
        for i, e in enumerate(self.entries()):
            if e["prev"] != prev:
                return False, f"entry {i} does not follow entry {i - 1}"
            if digest({k: e[k] for k in ("time", "event", "data", "prev")}) != e["hash"]:
                return False, f"entry {i} was modified"
            prev = e["hash"]
        return True, "ok"
