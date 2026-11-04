"""RunPod Serverless worker: scores one chunk per job with the same code as the Modal worker.

Built into the image in ``deploy/runpod/Dockerfile``. A job's ``input`` holds
the keyword arguments of :func:`autoinference.bio.cloud.run_boltz_chunk`.
"""

from __future__ import annotations

import os
import subprocess

from .cloud import run_boltz_chunk

BOLTZ_CACHE = os.environ.get("BOLTZ_CACHE", "/opt/boltz-cache")
ALLOWED = {"target", "candidates", "seed", "msa", "diffusion_samples", "extra_args"}


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def handler(job: dict) -> dict:
    args = job["input"]
    unknown = set(args) - ALLOWED
    if unknown:
        return {"error": f"unknown input fields: {sorted(unknown)}"}
    reply = run_boltz_chunk(**args, cache=BOLTZ_CACHE)
    reply["gpu"] = _gpu_name()
    return reply


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
