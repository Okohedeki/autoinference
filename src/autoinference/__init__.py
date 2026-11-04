"""AutoInference: a slotted LLM inference engine whose every part is swappable, and the platform
that serves specialised models on top of it.

The engine (which needs PyTorch) is imported on first use, so the platform's
control plane and CPU-only workers can import this package without it.
"""

from __future__ import annotations

from .registry import REGISTRY, Manifest, register

__all__ = ["REGISTRY", "Engine", "EngineConfig", "Generation", "Manifest", "SamplingParams", "register"]
__version__ = "0.1.0"

_LAZY = {"EngineConfig": ".config", "Engine": ".engine", "Generation": ".engine", "SamplingParams": ".types"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
