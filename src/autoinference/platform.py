"""Platform layer: device discovery and per-device defaults."""

from __future__ import annotations

import torch

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name.lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}; choose from {sorted(_DTYPES)}") from None


def free_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    return None
