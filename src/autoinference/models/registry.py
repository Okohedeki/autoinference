"""Architecture registry: maps a checkpoint's ``model_type`` to a model builder."""

from __future__ import annotations

from typing import Callable

_ARCHS: dict[str, Callable] = {}


def register_arch(*model_types: str) -> Callable:
    def decorator(builder: Callable) -> Callable:
        for t in model_types:
            if t in _ARCHS:
                raise ValueError(f"architecture {t!r} is already registered")
            _ARCHS[t] = builder
        return builder

    return decorator


def get_arch(model_type: str) -> Callable:
    try:
        return _ARCHS[model_type]
    except KeyError:
        raise KeyError(f"unsupported model_type {model_type!r} (supported: {', '.join(sorted(_ARCHS))})") from None


def supported_archs() -> list[str]:
    return sorted(_ARCHS)
