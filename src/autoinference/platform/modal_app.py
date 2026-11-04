"""The Modal deployment: the control plane as a web app, and one GPU worker class per catalog model.

    autoinference platform deploy        # or: python -m modal deploy -m autoinference.platform.modal_app

Every model gets its own image from the recipe in its ``model.yaml``, with its
weights baked in at build time. Workers scale to zero. The control plane keeps
one container warm so the API answers at once.

Both read their configuration from the Modal secret named by
``AUTOINFERENCE_SECRET`` (default ``autoinference-platform``): ``DATABASE_URL``,
``STORAGE=r2`` and the ``R2_*`` variables, ``AUTOINFERENCE_ENV``. Create it with
``modal secret create`` or in the Modal dashboard; nothing is read from files.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from .catalog import Catalog, ModelSpec
from .config import Settings

SETTINGS = Settings.from_env()
MODELS_DIR = SETTINGS.models_dir
REMOTE_MODELS = "/root/models"
SECRET_NAME = os.environ.get("AUTOINFERENCE_SECRET", "autoinference-platform")
IDLE_SECONDS = int(os.environ.get("AUTOINFERENCE_MODAL_IDLE", "120"))

app = modal.App(SETTINGS.modal_app)
secret = modal.Secret.from_name(SECRET_NAME)

PLATFORM_DEPS = ("sqlalchemy>=2.0", "psycopg[binary]>=3.1", "boto3>=1.34", "pyyaml>=6.0", "jsonschema>=4.20",
                 "httpx>=0.27")


def worker_image(spec: ModelSpec) -> modal.Image:
    img = modal.Image.debian_slim(python_version=spec.image.python)
    if spec.image.apt:
        img = img.apt_install(*spec.image.apt)
    img = img.uv_pip_install(*PLATFORM_DEPS, *spec.image.pip)
    if spec.image.env:
        img = img.env(dict(spec.image.env))
    if spec.image.commands:
        img = img.run_commands(*spec.image.commands)
    img = img.env({"PYTHONUNBUFFERED": "1", "AUTOINFERENCE_MODELS_DIR": REMOTE_MODELS, "EXECUTOR": "local"})
    # Added at container start rather than baked in, so code changes need no image rebuild.
    return (img.add_local_python_source("autoinference")
               .add_local_dir(MODELS_DIR, REMOTE_MODELS, ignore=["**/__pycache__", "**/*.pyc"]))


control_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("fastapi[standard]>=0.115", "modal>=1.6", *PLATFORM_DEPS)
    .env({"PYTHONUNBUFFERED": "1", "AUTOINFERENCE_MODELS_DIR": REMOTE_MODELS, "EXECUTOR": "modal"})
    .add_local_python_source("autoinference")
    .add_local_dir(MODELS_DIR, REMOTE_MODELS, ignore=["**/__pycache__", "**/*.pyc"])
)


class _Worker:
    """Serves one catalog model. Subclassed per model below, with ``MODEL_ID`` set."""

    MODEL_ID = ""

    @modal.enter()
    def setup(self) -> None:
        from .runtime import load_handler
        from .storage import make_storage

        self.catalog = Catalog(Path(REMOTE_MODELS))
        self.spec = self.catalog.get(self.MODEL_ID)
        load_handler(self.spec)
        self.storage = make_storage(Settings.from_env())

    @modal.method()
    def run(self, model_id: str, inputs: dict, request_id: str, project_id: str, seed: int = 0) -> dict:
        from .runtime import execute

        if model_id != self.spec.id:
            return {"status": "failed", "error": f"this worker serves {self.spec.id}, not {model_id}"}
        return execute(self.spec, inputs, request_id, project_id, self.storage, seed, Path("/tmp/work"))


def _define_workers() -> dict[str, type]:
    workers = {}
    for spec in Catalog(MODELS_DIR):
        cls = type(spec.worker_name, (_Worker,), {"MODEL_ID": spec.id, "__module__": __name__,
                                                 "__qualname__": spec.worker_name})
        hw = spec.hardware
        decorated = app.cls(
            image=worker_image(spec), secrets=[secret], gpu=list(hw.gpu) or None, timeout=hw.timeout,
            scaledown_window=IDLE_SECONDS, max_containers=hw.max_containers,
            memory=hw.memory_mb, retries=modal.Retries(max_retries=1, initial_delay=5.0),
        )(cls)
        workers[spec.id] = decorated
        globals()[spec.worker_name] = decorated   # Modal looks classes up by name in this module
    return workers


WORKERS = _define_workers()


@app.function(image=control_image, secrets=[secret], min_containers=1, scaledown_window=600, timeout=300)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="api")
def api():
    from .api import create_app

    return create_app()
