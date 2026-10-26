"""The op table: one resolved implementation per slot instance.

Model code never imports an op implementation. It asks the table for the op
filling a slot at a given instance key, once at build time, and calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..config import EngineConfig
from ..observability import Profiler
from ..registry import REGISTRY
from .interfaces import OP_SLOTS


@dataclass(frozen=True)
class InstanceKey:
    slot: str
    layer: int | None = None
    layer_type: str | None = None
    phase: str | None = None
    device: str = "cpu"


class _Timed:
    __slots__ = ("fn", "label", "profiler")

    def __init__(self, fn: Callable[..., Any], label: str, profiler: Profiler) -> None:
        self.fn = fn
        self.label = label
        self.profiler = profiler

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.profiler.span(self.label):
            return self.fn(*args, **kwargs)


class OpTable:
    def __init__(self, config: EngineConfig, device: str, profiler: Profiler) -> None:
        self.config = config
        self.device = device
        self.profiler = profiler
        self._instances: dict[str, Callable[..., Any]] = {}
        self.assignments: dict[InstanceKey, str] = {}

    def get(self, slot: str, layer: int | None = None, layer_type: str | None = None, phase: str | None = None):
        if slot not in OP_SLOTS:
            raise KeyError(f"{slot!r} is not an op slot")
        key = InstanceKey(slot, layer, layer_type, phase, self.device)
        impl = self.config.op_impl(slot, layer, layer_type, phase, self.device)
        entry = REGISTRY.get(slot, impl)
        if self.device not in entry.manifest.devices:
            raise ValueError(f"{slot}/{impl} does not run on {self.device} (supports {entry.manifest.devices})")
        self.assignments[key] = impl
        # One object per (slot, impl): ops are stateless, so instances can share it.
        cache_key = f"{slot}/{impl}"
        if cache_key not in self._instances:
            self._instances[cache_key] = entry.factory()
        fn = self._instances[cache_key]
        if not self.profiler.enabled:
            return fn
        tag = ",".join(x for x in (layer_type, phase) if x)
        label = f"forward/{slot}:{impl}" + (f"[{tag}]" if tag else "")
        return _Timed(fn, label, self.profiler)

    def describe(self) -> dict[str, dict[str, int]]:
        """Implementation usage per slot, for logging which plugins a run used."""
        out: dict[str, dict[str, int]] = {}
        for key, impl in self.assignments.items():
            out.setdefault(key.slot, {}).setdefault(impl, 0)
            out[key.slot][impl] += 1
        return out
