"""The Modal deployment: one GPU worker class per model, each with its own container image.

Deploy (after ``python -m modal setup``)::

    autoinference bio deploy            # or: python -m modal deploy -m autoinference.bio.modal_app

Each model gets its own image because these models pin conflicting
dependencies. Weights are downloaded while the image builds, so a cold
container starts with them on local disk and never races other containers to
fetch them. Containers scale to zero when idle, so nothing is billed between
campaigns.

The GPU setting is a fallback list: Modal tries the types in order, which keeps
campaigns moving when the first type is scarce. A stage can ask for another
type with ``"options": {"gpu": "<type>"}``, through ``Cls.with_options``.
Valid names are in Modal's GPU guide; prices are on its pricing page.

A server process authenticates with the ``MODAL_TOKEN_ID`` and
``MODAL_TOKEN_SECRET`` environment variables instead of ``modal setup``.
"""

from __future__ import annotations

import os
import subprocess

import modal

from .cloud import APP_NAME, run_boltz_chunk

BOLTZ_VERSION = "2.2.1"
BOLTZ_CACHE = "/opt/boltz-cache"
DEFAULT_GPU = [g for g in os.environ.get("AUTOINFERENCE_MODAL_GPU", "L40S,A100").split(",") if g]
IDLE_SECONDS = int(os.environ.get("AUTOINFERENCE_MODAL_IDLE", "120"))  # stay warm this long between chunks
MAX_CONTAINERS = int(os.environ.get("AUTOINFERENCE_MODAL_MAX_CONTAINERS", "20"))  # caps a campaign's spend rate

# Weights are fetched by a plain command, not ``run_function``: the build step runs before this
# package is added to the image, so it must not import it.
_DOWNLOAD = ("python -c \"from pathlib import Path; from boltz.main import download_boltz2; "
             f"download_boltz2(Path('{BOLTZ_CACHE}'))\"")

boltz_image = (
    modal.Image.debian_slim(python_version="3.11")
    # The engine package imports these at the top level; the worker only uses the bio backends.
    .uv_pip_install(f"boltz[cuda]=={BOLTZ_VERSION}", "safetensors>=0.4", "transformers>=4.51",
                    "huggingface_hub>=0.24")
    .run_commands(f"mkdir -p {BOLTZ_CACHE}", _DOWNLOAD)
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("autoinference")  # last: added at container start, so code edits need no rebuild
)

app = modal.App(APP_NAME)


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


@app.cls(image=boltz_image, gpu=DEFAULT_GPU, timeout=3600, scaledown_window=IDLE_SECONDS,
         max_containers=MAX_CONTAINERS, retries=modal.Retries(max_retries=2, initial_delay=5.0))
class Boltz2Worker:
    @modal.method()
    def predict(self, target: dict, candidates: list[dict], seed: int, msa: str | None,
                diffusion_samples: int = 1, extra_args: str = "") -> dict:
        reply = run_boltz_chunk(target, candidates, seed, msa, diffusion_samples, extra_args, cache=BOLTZ_CACHE)
        reply["gpu"] = _gpu_name()
        return reply
