"""Engine configuration: which implementation fills each slot (spec section 12.2)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

DEFAULT_PARTS: dict[str, str] = {
    "renderer": "hf-tokenizer",
    "scheduler": "fcfs",
    "overlap": "none",
    "spec-planner": "off",
    "kv-allocator": "paged",
    "prefix-cache": "none",
    "input-prep": "reference",
    "executor": "eager",
    "placement": "single-device",
    "loader": "safetensors",
    "sampler": "reference",
    "spec-drafter": "off",
}


@dataclass(frozen=True)
class OpOverride:
    """Assigns ``impl`` to ``slot`` for instances matching every given key."""

    slot: str
    impl: str
    layer_type: str | None = None
    phase: str | None = None
    device: str | None = None
    layers: tuple[int, int] | None = None  # inclusive range of layer indices

    def matches(self, slot: str, layer: int | None, layer_type: str | None, phase: str | None, device: str) -> bool:
        if slot != self.slot:
            return False
        if self.layer_type is not None and layer_type != self.layer_type:
            return False
        if self.phase is not None and phase != self.phase:
            return False
        if self.device is not None and device != self.device:
            return False
        if self.layers is not None and (layer is None or not self.layers[0] <= layer <= self.layers[1]):
            return False
        return True


@dataclass
class EngineConfig:
    device: str = "auto"
    dtype: str = "float32"
    page_size: int = 16
    num_pages: int = 0             # 0: size from memory_fraction at startup
    memory_fraction: float = 0.2   # share of free device memory given to the KV cache when num_pages is 0
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 512
    max_model_len: int = 4096
    profile: bool = False
    parts: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PARTS))
    part_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    op_default: str = "reference"
    ops: dict[str, str] = field(default_factory=dict)
    op_overrides: list[OpOverride] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        cfg = cls()
        for key, value in data.get("engine", {}).items():
            attr = key.replace("-", "_")
            if not hasattr(cfg, attr):
                raise ValueError(f"unknown engine setting {key!r}")
            setattr(cfg, attr, value)
        for part, value in data.get("parts", {}).items():
            if isinstance(value, dict):
                value = dict(value)
                cfg.parts[part] = value.pop("impl")
                cfg.part_options[part] = value
            else:
                cfg.parts[part] = value
        ops = dict(data.get("ops", {}))
        cfg.op_default = ops.pop("default", cfg.op_default)
        for raw in ops.pop("override", []):
            raw = dict(raw)
            layers = raw.pop("layers", None)
            if layers is not None:
                layers = (int(layers[0]), int(layers[1])) if isinstance(layers, list) else (int(layers), int(layers))
            cfg.op_overrides.append(OpOverride(layers=layers, **raw))
        cfg.ops.update(ops)
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "EngineConfig":
        path = Path(path)
        if not path.exists():
            bundled = resources.files("autoinference") / "configs" / path.name
            if bundled.is_file():
                return cls.from_dict(tomllib.loads(bundled.read_text(encoding="utf-8")))
            raise FileNotFoundError(path)
        return cls.from_dict(tomllib.loads(path.read_text(encoding="utf-8")))

    def op_impl(self, slot: str, layer: int | None, layer_type: str | None, phase: str | None, device: str) -> str:
        """Resolve one slot instance. Later overrides win over earlier ones."""
        impl = self.ops.get(slot, self.op_default)
        for override in self.op_overrides:
            if override.matches(slot, layer, layer_type, phase, device):
                impl = override.impl
        return impl
