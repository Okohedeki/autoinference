"""Contracts for the parts outside the op library (spec section 3.3).

Parts the MVP implements have a reference implementation in this package.
The rest are declared here so their place in the structure is fixed; each has
a trivial implementation registered in ``stubs.py``.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from ..types import BatchPlan, ForwardInputs, Request, StepEvents

# ---------------------------------------------------------------- serving


class Renderer(Protocol):
    eos_token_ids: tuple[int, ...]

    def encode(self, text: str) -> list[int]: ...
    def apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


# ---------------------------------------------------------------- control


class Scheduler(Protocol):
    def add(self, request: Request) -> None: ...
    def schedule(self) -> BatchPlan: ...
    def commit(self, plan: BatchPlan, sampled: dict[int, int]) -> StepEvents: ...
    def abort(self, request_id: int) -> None: ...
    def has_work(self) -> bool: ...


class Overlap(Protocol):
    """Decides whether host work for step N+1 overlaps device work for step N."""

    enabled: bool


class SpecPlanner(Protocol):
    def draft_length(self, request: Request) -> int: ...


# ---------------------------------------------------------------- memory


class KVAllocator(Protocol):
    page_size: int
    num_pages: int

    def allocate(self, n: int) -> list[int] | None: ...
    def free(self, pages: list[int]) -> None: ...
    def incref(self, pages: list[int]) -> None: ...
    def num_free(self) -> int: ...
    def pages_needed(self, num_tokens: int, have: int) -> int: ...


class PrefixCache(Protocol):
    def match(self, token_ids: list[int]) -> tuple[list[int], int]: ...
    def insert(self, request: Request) -> None: ...
    def evict(self, n: int) -> int: ...
    def forget(self, request: Request) -> None: ...


class KVTier(Protocol):
    def offload(self, pages: list[int]) -> None: ...
    def restore(self, pages: list[int]) -> None: ...


class KVSelect(Protocol):
    def select(self, request: Request, pages: list[int]) -> list[int]: ...


class Workspace(Protocol):
    def empty(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor: ...


# ---------------------------------------------------------------- execution


class InputPrep(Protocol):
    def prepare(self, plan: BatchPlan) -> ForwardInputs: ...


class Executor(Protocol):
    def run(self, forward: Any, inputs: ForwardInputs) -> torch.Tensor: ...


class Placement(Protocol):
    def device_for(self, tensor_name: str) -> torch.device: ...


class ExpertCache(Protocol):
    def lookup(self, layer: int, expert_ids: list[int]) -> list[bool]: ...


class Transfer(Protocol):
    def copy(self, src: torch.Tensor, device: torch.device) -> torch.Tensor: ...


class OffloadScheduler(Protocol):
    def where(self, layer: int, tokens_per_expert: list[int]) -> list[str]: ...


class Parallelism(Protocol):
    world_size: int
    rank: int


# ---------------------------------------------------------------- model


class Loader(Protocol):
    def load(self, model_dir: str, names: list[str] | None = None) -> dict[str, torch.Tensor]: ...


class WeightFormat(Protocol):
    def encode(self, name: str, tensor: torch.Tensor) -> torch.Tensor: ...


class Adapter(Protocol):
    def apply(self, name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------- decoding


class Sampler(Protocol):
    def sample(self, logits: torch.Tensor, requests: list[Request]) -> list[int]: ...


class Grammar(Protocol):
    def mask(self, request: Request, vocab_size: int) -> torch.Tensor | None: ...


class SpecDrafter(Protocol):
    def propose(self, requests: list[Request]) -> dict[int, list[int]]: ...


class SpecVerifier(Protocol):
    def verify(self, request: Request, draft: list[int], logits: torch.Tensor) -> list[int]: ...


# ---------------------------------------------------------------- platform


class OSConfig(Protocol):
    def apply(self) -> None: ...
