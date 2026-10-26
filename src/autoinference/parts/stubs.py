"""Trivial implementations of parts the MVP declares but does not build yet.

Each is the identity or "off" choice for its slot, so the engine's structure
already has every part and later work only adds implementations.
"""

from __future__ import annotations

import torch

from ..registry import Manifest, register
from ..types import Request

OFF = Manifest(kind="policy", lossless_class="bit-exact")


@register("replica-balance", "single", OFF)
class SingleReplica:
    def pick(self, request: Request) -> int:
        return 0


@register("kv-tier", "none", OFF)
class NoKVTier:
    def offload(self, pages: list[int]) -> None:
        pass

    def restore(self, pages: list[int]) -> None:
        pass


@register("kv-select", "all", OFF)
class KeepAllKV:
    def select(self, request: Request, pages: list[int]) -> list[int]:
        return pages


@register("workspace", "torch", OFF)
class TorchWorkspace:
    def empty(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)


@register("expert-cache", "none", OFF)
class NoExpertCache:
    def lookup(self, layer: int, expert_ids: list[int]) -> list[bool]:
        return [True] * len(expert_ids)


@register("expert-balance", "none", OFF)
class NoExpertBalance:
    def observe(self, layer: int, expert_ids: torch.Tensor) -> None:
        pass


@register("transfer", "torch", OFF)
class TorchTransfer:
    def copy(self, src: torch.Tensor, device: torch.device) -> torch.Tensor:
        return src.to(device)


@register("offload-scheduler", "resident", OFF)
class AllResident:
    def where(self, layer: int, tokens_per_expert: list[int]) -> list[str]:
        return ["resident"] * len(tokens_per_expert)


@register("parallelism", "single", OFF)
class SingleProcess:
    world_size = 1
    rank = 0


@register("collective", "none", OFF)
class NoCollective:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x


@register("weight-format", "as-stored", Manifest(kind="transform", lossless_class="bit-exact"))
class WeightsAsStored:
    def encode(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


@register("weight-rotation", "none", Manifest(kind="transform", lossless_class="bit-exact"))
class NoRotation:
    def apply(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return weights


@register("expert-prune", "none", Manifest(kind="transform", lossless_class="bit-exact"))
class NoExpertPrune:
    def apply(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return weights


@register("adapter", "none", OFF)
class NoAdapter:
    def apply(self, name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y


@register("grammar", "none", OFF)
class NoGrammar:
    def mask(self, request: Request, vocab_size: int) -> torch.Tensor | None:
        return None


@register("mm-preprocess", "none", OFF)
class NoMultimodal:
    def process(self, item: object) -> None:
        raise NotImplementedError("multimodal inputs are not supported yet")


@register("os-config", "none", Manifest(kind="config", lossless_class="bit-exact"))
class NoOSConfig:
    def apply(self) -> None:
        pass
