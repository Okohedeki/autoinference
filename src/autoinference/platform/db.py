"""Tables. SQLAlchemy Core, so SQLite (tests, local mode) and Postgres (production) share one schema.

Every table that holds customer data carries ``project_id`` and every query
filters on it; that is the tenancy boundary.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text

metadata = MetaData()

projects = Table(
    "projects", metadata,
    Column("id", String(32), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("created", Float, nullable=False),
    Column("credits", Float, nullable=False, server_default="0"),   # prepaid balance; null limit = pay as you go
    Column("spend_cap", Float),
)

api_keys = Table(
    "api_keys", metadata,
    Column("id", String(32), primary_key=True),
    Column("project_id", String(32), sa.ForeignKey("projects.id"), nullable=False, index=True),
    Column("name", String(200), nullable=False),
    Column("prefix", String(16), nullable=False, index=True),
    Column("key_hash", String(64), nullable=False, unique=True),
    Column("created", Float, nullable=False),
    Column("last_used", Float),
    Column("revoked", Float),
)

requests = Table(
    "requests", metadata,
    Column("id", String(32), primary_key=True),
    Column("project_id", String(32), nullable=False, index=True),
    Column("model", String(200), nullable=False),
    Column("version", String(50), nullable=False),
    Column("mode", String(10), nullable=False),          # run | queue
    Column("status", String(20), nullable=False, index=True),
    Column("inputs", sa.JSON, nullable=False),
    Column("inputs_digest", String(64), nullable=False),
    Column("outputs", sa.JSON),
    Column("error", Text),
    Column("provider", String(20)),
    Column("provider_job_id", String(200)),
    Column("units", Float),
    Column("gpu_seconds", Float),
    Column("price", Float),
    Column("manifest", sa.JSON),
    Column("webhook_url", Text),
    Column("idempotency_key", String(200)),
    Column("pipeline_id", String(32), index=True),
    Column("step_id", String(50)),
    Column("item_index", Integer),
    Column("created", Float, nullable=False),
    Column("started", Float),
    Column("finished", Float),
    sa.UniqueConstraint("project_id", "idempotency_key", name="uq_requests_idempotency"),
)

pipeline_runs = Table(
    "pipeline_runs", metadata,
    Column("id", String(32), primary_key=True),                       # equals the parent request id
    Column("project_id", String(32), nullable=False, index=True),
    Column("name", String(200), nullable=False),
    Column("spec", sa.JSON, nullable=False),
    Column("state", sa.JSON, nullable=False),                         # inputs, per-step results, current step
    Column("status", String(20), nullable=False, index=True),         # running | waiting_approval | succeeded | failed | cancelled
    Column("current_step", String(50)),
    Column("created", Float, nullable=False),
    Column("finished", Float),
)

files = Table(
    "files", metadata,
    Column("id", String(32), primary_key=True),
    Column("project_id", String(32), nullable=False, index=True),
    Column("key", Text, nullable=False),
    Column("filename", String(500), nullable=False),
    Column("content_type", String(200)),
    Column("size", Integer),
    Column("created", Float, nullable=False),
    Column("expires", Float),
)

usage_events = Table(
    "usage_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("project_id", String(32), nullable=False, index=True),
    Column("request_id", String(32), nullable=False),
    Column("model", String(200), nullable=False),
    Column("units", Float, nullable=False),
    Column("gpu_seconds", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("created", Float, nullable=False),
)

audit_log = Table(
    "audit_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("project_id", String(32), nullable=False, index=True),
    Column("event", String(50), nullable=False),
    Column("payload", sa.JSON, nullable=False),
    Column("prev_hash", String(64), nullable=False),
    Column("hash", String(64), nullable=False),
    Column("created", Float, nullable=False),
)


def connect(database_url: str) -> sa.Engine:
    kwargs = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs["poolclass"] = sa.pool.StaticPool if ":memory:" in database_url else sa.pool.NullPool
    elif database_url.startswith("postgres://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgres://"):]
    elif database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]
    engine = sa.create_engine(database_url, pool_pre_ping=True, **kwargs)
    metadata.create_all(engine)
    return engine
