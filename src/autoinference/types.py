"""Data passed between parts (spec section 12.3)."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

import torch

_request_ids = itertools.count()


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    """One generation request and its progress.

    ``token_ids`` holds the prompt followed by generated tokens.
    ``num_computed`` counts tokens whose keys and values are in the cache.
    """

    prompt_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    request_id: int = field(default_factory=lambda: next(_request_ids))
    token_ids: list[int] = field(default_factory=list)
    num_computed: int = 0
    pages: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: str | None = None
    generator: torch.Generator | None = None

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("prompt must contain at least one token")
        if not self.token_ids:
            self.token_ids = list(self.prompt_ids)

    @property
    def output_ids(self) -> list[int]:
        return self.token_ids[len(self.prompt_ids):]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


class Phase(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class SeqPlan:
    """What one sequence does in one step."""

    request_id: int
    token_ids: tuple[int, ...]  # the new tokens whose KV this step computes
    start_pos: int              # position of the first new token
    pages: tuple[int, ...]      # page table covering positions [0, start_pos + len(token_ids))
    phase: Phase
    sample: bool                # whether this step produces a token for the sequence


@dataclass(frozen=True)
class BatchPlan:
    """The scheduler's frozen per-step plan. Prefill sequences come first."""

    seqs: tuple[SeqPlan, ...]

    @property
    def num_tokens(self) -> int:
        return sum(len(s.token_ids) for s in self.seqs)

    @property
    def empty(self) -> bool:
        return not self.seqs


@dataclass
class AttentionMetadata:
    """Per-sequence layout of one phase's tokens, as attention ops consume it."""

    q_start: list[int]       # offset of each sequence's queries within this phase's tokens
    q_len: list[int]
    ctx_len: list[int]       # total keys visible after this step's writes
    page_tables: list[list[int]]
    page_size: int

    @property
    def num_seqs(self) -> int:
        return len(self.q_len)


@dataclass
class ForwardInputs:
    """Flattened tensors the model runner and op slots consume."""

    token_ids: torch.Tensor      # [T] long
    positions: torch.Tensor      # [T] long
    slot_mapping: torch.Tensor   # [T] long: page * page_size + offset
    logits_indices: torch.Tensor # [S] long: rows of the hidden states that need logits
    num_prefill_tokens: int
    prefill: AttentionMetadata
    decode: AttentionMetadata
    sample_request_ids: list[int]


@dataclass
class ForwardOutput:
    logits: torch.Tensor          # [S, vocab]
    request_ids: list[int]
    token_ids: list[int] | None = None


@dataclass
class TokenEvent:
    request_id: int
    token_id: int
    finished: bool
    finish_reason: str | None = None


@dataclass
class StepEvents:
    tokens: list[TokenEvent] = field(default_factory=list)
    num_scheduled_tokens: int = 0
