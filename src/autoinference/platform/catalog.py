"""The model contract: a directory with ``model.yaml`` and ``handler.py``.

``model.yaml`` declares the model's id, version, license, billing unit, hardware,
image recipe and the JSON Schema of its inputs and outputs. Schema fields with
``"format": "file"`` are files: the API accepts a file id there and hands the
handler a local path; the handler returns a path and the API hands the caller
a download URL. The playground, the SDK types and the MCP tools are all
generated from the same schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import jsonschema
import yaml

MODEL_ID = re.compile(r"^[a-z0-9-]+/[a-z0-9.-]+$")
UNITS = {"request", "structure", "design", "molecule", "1k_residues", "mb_scanned", "variant", "cells",
         "1k_md_steps", "relaxation", "1k_actions", "rollout", "second", "series", "table", "episode", "scene",
         "log_hour", "forecast", "tile", "simulation", "geometry", "study", "segmentation", "image", "plate",
         "1k_samples", "grid_case", "event", "recording_hour", "page", "gpu_second"}


@dataclass(frozen=True)
class Hardware:
    gpu: tuple[str, ...] = ()          # fallback order; empty means CPU only
    timeout: int = 600                 # seconds per request
    memory_mb: int | None = None
    concurrency: int = 1               # requests one container handles at once
    max_containers: int = 10


@dataclass(frozen=True)
class ImageSpec:
    python: str = "3.11"
    pip: tuple[str, ...] = ()
    apt: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()     # run at build time, e.g. weight downloads
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    version: str
    domain: str
    name: str
    description: str
    license: str
    source: str
    unit: str
    price_per_unit: float
    hardware: Hardware
    image: ImageSpec
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    path: Path
    sync: bool = True                  # may be called through /run as well as /queue
    public: bool = True
    weights: dict[str, str] = field(default_factory=dict)   # name -> digest, recorded in manifests
    tags: tuple[str, ...] = ()

    @property
    def handler_path(self) -> Path:
        return self.path / "handler.py"

    @property
    def worker_name(self) -> str:
        """A valid Python identifier for the Modal class that serves this model."""
        return "W_" + re.sub(r"[^0-9a-zA-Z]", "_", self.id)

    def file_inputs(self) -> list[str]:
        return [k for k, v in self.inputs.get("properties", {}).items() if v.get("format") == "file"]

    def file_outputs(self) -> list[str]:
        return [k for k, v in self.outputs.get("properties", {}).items() if v.get("format") == "file"]

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        try:
            jsonschema.validate(inputs, self.inputs)
        except jsonschema.ValidationError as exc:
            where = "/".join(str(p) for p in exc.absolute_path) or "inputs"
            raise ValueError(f"{where}: {exc.message}") from None

    def fingerprint(self) -> dict[str, Any]:
        return {"model": self.id, "version": self.version, "weights": dict(self.weights),
                "handler_digest": hashlib.sha256(self.handler_path.read_bytes()).hexdigest()
                if self.handler_path.exists() else None}

    def to_public(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "domain": self.domain, "name": self.name,
                "description": self.description, "license": self.license, "source": self.source,
                "unit": self.unit, "price_per_unit": self.price_per_unit, "sync": self.sync,
                "hardware": {"gpu": list(self.hardware.gpu), "timeout": self.hardware.timeout},
                "tags": list(self.tags)}

    def to_schema(self) -> dict[str, Any]:
        out = self.to_public()
        out["inputs"] = self.inputs
        out["outputs"] = self.outputs
        return out


def load_spec(path: Path) -> ModelSpec:
    data = yaml.safe_load((path / "model.yaml").read_text(encoding="utf-8"))
    missing = {"id", "version", "domain", "name", "license", "source", "unit", "price_per_unit", "inputs",
               "outputs"} - set(data)
    if missing:
        raise ValueError(f"{path / 'model.yaml'}: missing {sorted(missing)}")
    if not MODEL_ID.match(data["id"]):
        raise ValueError(f"{data['id']!r}: model ids look like <domain>/<name>, lower case")
    if data["id"].split("/")[0] != data["domain"]:
        raise ValueError(f"{data['id']!r}: the id must start with the domain {data['domain']!r}")
    if data["unit"] not in UNITS:
        raise ValueError(f"{data['id']}: unknown unit {data['unit']!r}")
    for part in ("inputs", "outputs"):
        if data[part].get("type") != "object":
            raise ValueError(f"{data['id']}: {part} schema must be an object schema")
        jsonschema.Draft202012Validator.check_schema(data[part])
    hw = data.get("hardware", {})
    img = data.get("image", {})
    gpu = hw.get("gpu", [])
    return ModelSpec(
        id=data["id"], version=str(data["version"]), domain=data["domain"], name=data["name"],
        description=data.get("description", ""), license=data["license"], source=data["source"],
        unit=data["unit"], price_per_unit=float(data["price_per_unit"]),
        hardware=Hardware(gpu=tuple([gpu] if isinstance(gpu, str) else gpu), timeout=int(hw.get("timeout", 600)),
                          memory_mb=hw.get("memory_mb"), concurrency=int(hw.get("concurrency", 1)),
                          max_containers=int(hw.get("max_containers", 10))),
        image=ImageSpec(python=str(img.get("python", "3.11")), pip=tuple(img.get("pip", [])),
                        apt=tuple(img.get("apt", [])), commands=tuple(img.get("commands", [])),
                        env=dict(img.get("env", {}))),
        inputs=data["inputs"], outputs=data["outputs"], path=path, sync=bool(data.get("sync", True)),
        public=bool(data.get("public", True)), weights={k: str(v) for k, v in data.get("weights", {}).items()},
        tags=tuple(data.get("tags", [])),
    )


class Catalog:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._specs: dict[str, ModelSpec] = {}
        self.reload()

    def reload(self) -> None:
        specs = {}
        for yaml_path in sorted(self.root.glob("*/*/model.yaml")):
            spec = load_spec(yaml_path.parent)
            if spec.id in specs:
                raise ValueError(f"duplicate model id {spec.id}")
            specs[spec.id] = spec
        self._specs = specs

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, model: str) -> ModelSpec:
        """``domain/name`` or ``domain/name@version``."""
        name, _, version = model.partition("@")
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(model)
        if version and version != spec.version:
            raise KeyError(f"{name} version {version} is not available (current: {spec.version})")
        return spec

    def public(self) -> list[ModelSpec]:
        return [s for s in self._specs.values() if s.public]


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
