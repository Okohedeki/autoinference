"""Contracts for the op-library slots (spec section 3.3, op library table).

Every op is a callable object. Shapes use T for tokens in the call, H for
hidden size, Hq/Hkv for query/key-value heads, D for head size, E for experts,
I for expert intermediate size and K for experts per token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..types import AttentionMetadata

OP_SLOTS = (
    "embedding",
    "linear",
    "norm",
    "rotary",
    "kv-write",
    "attn-prefill",
    "attn-decode",
    "activation",
    "moe-router",
    "moe-dispatch",
    "moe-experts",
    "moe-combine",
    "lm-head",
)


@dataclass(frozen=True)
class RopeSpec:
    head_dim: int
    theta: float
    rope_type: str = "default"


@dataclass
class MoEDispatch:
    """Token-expert pairs grouped by expert.

    Row r of ``hidden`` is token ``src_token[r]`` routed to expert
    ``expert_ids[r]`` with weight ``weights[r]``. Rows of one expert are
    contiguous: expert e owns rows ``offsets[e]:offsets[e + 1]``.
    """

    hidden: torch.Tensor      # [T*K, H]
    expert_ids: torch.Tensor  # [T*K]
    weights: torch.Tensor     # [T*K]
    src_token: torch.Tensor   # [T*K]
    offsets: torch.Tensor     # [E + 1], on the host
    num_tokens: int


class Embedding(Protocol):
    def __call__(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...


class Linear(Protocol):
    def __call__(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor: ...


class Norm(Protocol):
    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor: ...


class Rotary(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, spec: RopeSpec
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class KVWrite(Protocol):
    def __call__(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor
    ) -> None: ...


class Attention(Protocol):
    """Queries [T, Hq, D] for one phase; caches [pages, page_size, Hkv, D].

    Query j of a sequence sits at position ``ctx_len - q_len + j`` and attends
    causally to every cached key at or before that position.
    """

    def __call__(
        self, q: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor, meta: AttentionMetadata, scale: float
    ) -> torch.Tensor: ...


class Activation(Protocol):
    def __call__(self, gate: torch.Tensor, up: torch.Tensor, kind: str) -> torch.Tensor: ...


class MoERouter(Protocol):
    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, top_k: int, normalize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class MoEDispatcher(Protocol):
    def __call__(self, x: torch.Tensor, expert_ids: torch.Tensor, weights: torch.Tensor, num_experts: int) -> MoEDispatch: ...


class MoEExperts(Protocol):
    def __call__(self, d: MoEDispatch, gate_up: torch.Tensor, down: torch.Tensor, kind: str) -> torch.Tensor: ...


class MoECombine(Protocol):
    def __call__(self, expert_out: torch.Tensor, d: MoEDispatch) -> torch.Tensor: ...


class LMHead(Protocol):
    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
