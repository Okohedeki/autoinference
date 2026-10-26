"""Slot registry: every swappable part and op registers its implementations here.

A slot is the unit the search assigns an implementation to (see the spec,
section 3.3). Implementations register under ``(slot, name)`` with a manifest
describing what they claim, need and guarantee (spec section 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Lossless classes, strongest first (spec section 8.2).
LOSSLESS_CLASSES = ("bit-exact", "tolerance-exact", "distribution-preserving", "lossy-gated")
KINDS = ("kernel", "policy", "transform", "config", "infrastructure")


@dataclass(frozen=True)
class Manifest:
    """What an implementation declares about itself.

    The MVP records these fields but enforces only ``devices``; the rest are
    exercised so the format is stable before the harness enforces it.
    """

    kind: str = "kernel"
    devices: tuple[str, ...] = ("cpu", "cuda", "mps")
    lossless_class: str = "tolerance-exact"
    technique_id: str | None = None
    claims: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    accepts: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    assumes: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    deps: tuple[str, ...] = ("torch",)
    provenance: str = "autoinference"
    licence: str = "Apache-2.0"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.lossless_class not in LOSSLESS_CLASSES:
            raise ValueError(f"unknown lossless class {self.lossless_class!r}")


@dataclass(frozen=True)
class Entry:
    slot: str
    name: str
    factory: Callable[..., Any]
    manifest: Manifest


class Registry:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Entry]] = {}

    def register(self, slot: str, name: str, manifest: Manifest | None = None) -> Callable[[T], T]:
        def decorator(factory: T) -> T:
            impls = self._entries.setdefault(slot, {})
            if name in impls:
                raise ValueError(f"{slot}/{name} is already registered")
            impls[name] = Entry(slot, name, factory, manifest or Manifest())
            return factory

        return decorator

    def get(self, slot: str, name: str) -> Entry:
        try:
            return self._entries[slot][name]
        except KeyError:
            known = ", ".join(sorted(self._entries.get(slot, {}))) or "none"
            raise KeyError(f"no implementation {name!r} for slot {slot!r} (registered: {known})") from None

    def create(self, slot: str, name: str, **kwargs: Any) -> Any:
        return self.get(slot, name).factory(**kwargs)

    def slots(self) -> list[str]:
        return sorted(self._entries)

    def names(self, slot: str) -> list[str]:
        return sorted(self._entries.get(slot, {}))


REGISTRY = Registry()
register = REGISTRY.register
