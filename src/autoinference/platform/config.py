"""Settings, read from the environment so the same code runs locally, in CI and on Modal.

Local mode (the default) needs no accounts: SQLite, a directory for files and
in-process execution. Production sets ``DATABASE_URL`` (Postgres), ``STORAGE=r2``
with the R2 variables, and ``EXECUTOR=modal``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _models_dir() -> Path:
    env = os.environ.get("AUTOINFERENCE_MODELS_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:  # the repository's models/ directory, when running from a checkout
        if (parent / "models").is_dir() and (parent / "pyproject.toml").exists():
            return parent / "models"
    return Path("/root/models")  # where the Modal images put it


@dataclass
class Settings:
    environment: str = "dev"
    database_url: str = ""
    storage: str = "local"              # local | r2
    storage_dir: Path = field(default_factory=lambda: Path("platform-workspace"))
    r2_bucket: str = ""
    r2_endpoint: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    executor: str = "local"             # local | modal
    modal_app: str = "autoinference-platform"
    models_dir: Path = field(default_factory=_models_dir)
    retention_days: int = 30
    sync_timeout: float = 60.0          # /run waits this long before telling the caller to poll
    public_url: str = "http://127.0.0.1:8200"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        s = cls(
            environment=env.get("AUTOINFERENCE_ENV", "dev"),
            database_url=env.get("DATABASE_URL", ""),
            storage=env.get("STORAGE", "local"),
            executor=env.get("EXECUTOR", "local"),
            modal_app=env.get("AUTOINFERENCE_MODAL_APP", "autoinference-platform"),
            r2_bucket=env.get("R2_BUCKET", ""),
            r2_endpoint=env.get("R2_ENDPOINT", ""),
            r2_access_key_id=env.get("R2_ACCESS_KEY_ID", ""),
            r2_secret_access_key=env.get("R2_SECRET_ACCESS_KEY", ""),
            retention_days=int(env.get("RETENTION_DAYS", "30")),
            sync_timeout=float(env.get("SYNC_TIMEOUT", "60")),
            public_url=env.get("PUBLIC_URL", "http://127.0.0.1:8200"),
        )
        if env.get("STORAGE_DIR"):
            s.storage_dir = Path(env["STORAGE_DIR"])
        if env.get("AUTOINFERENCE_MODELS_DIR"):
            s.models_dir = Path(env["AUTOINFERENCE_MODELS_DIR"])
        if not s.database_url:
            s.storage_dir.mkdir(parents=True, exist_ok=True)
            s.database_url = f"sqlite:///{(s.storage_dir / 'platform.sqlite').as_posix()}"
        return s

    def validate(self) -> None:
        if self.storage not in ("local", "r2"):
            raise ValueError(f"STORAGE must be local or r2, not {self.storage!r}")
        if self.executor not in ("local", "modal"):
            raise ValueError(f"EXECUTOR must be local or modal, not {self.executor!r}")
        if self.storage == "r2" and not all((self.r2_bucket, self.r2_endpoint, self.r2_access_key_id,
                                             self.r2_secret_access_key)):
            raise ValueError("STORAGE=r2 needs R2_BUCKET, R2_ENDPOINT, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY")
        if self.environment == "prod" and (self.storage == "local" or self.executor == "local"
                                           or self.database_url.startswith("sqlite")):
            raise ValueError("production must use Postgres, R2 and the Modal executor")
