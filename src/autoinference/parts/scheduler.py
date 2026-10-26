"""Scheduler part: admission, continuous batching, chunked prefill, preemption."""

from __future__ import annotations

from collections import deque

import torch

from ..registry import Manifest, register
from ..types import BatchPlan, Phase, Request, RequestStatus, SeqPlan, StepEvents, TokenEvent
from .kv import PagedAllocator
from .prefix_cache import NoPrefixCache


@register("scheduler", "fcfs", Manifest(kind="policy", lossless_class="tolerance-exact"))
class FCFSScheduler:
    """First come, first served, with one token budget shared by every phase.

    Each step every running sequence computes the tokens it has not cached yet,
    up to the remaining budget: one token for a sequence that is decoding, a
    chunk of its prompt for one that is prefilling. Waiting requests are admitted
    in arrival order while the budget, the sequence limit and free pages allow.
    When pages run out, the most recently admitted sequence is preempted and
    later recomputed from scratch.
    """

    def __init__(
        self,
        allocator: PagedAllocator,
        prefix_cache: NoPrefixCache,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        eos_token_ids: tuple[int, ...] = (),
    ) -> None:
        self.allocator = allocator
        self.prefix_cache = prefix_cache
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.eos_token_ids = tuple(eos_token_ids)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[int, Request] = {}
        self.num_preemptions = 0

    # ------------------------------------------------------------ requests

    def add(self, request: Request) -> None:
        if request.num_tokens >= self.max_model_len:
            raise ValueError(f"prompt of {request.num_tokens} tokens does not fit max_model_len={self.max_model_len}")
        if request.params.seed is not None:
            request.generator = torch.Generator().manual_seed(request.params.seed)
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def abort(self, request_id: int) -> None:
        req = self.requests.get(request_id)
        if req is None or req.status is RequestStatus.FINISHED:
            return
        if req in self.waiting:
            self.waiting.remove(req)
        self._finish(req, "abort")

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------ scheduling

    def _grow(self, req: Request, upto: int) -> bool:
        need = self.allocator.pages_needed(upto, len(req.pages))
        if need == 0:
            return True
        pages = self.allocator.allocate(need)
        if pages is None:
            return False
        req.pages.extend(pages)
        return True

    def _preempt(self, req: Request) -> None:
        self.allocator.free(req.pages)
        req.pages = []
        req.num_computed = 0
        req.status = RequestStatus.WAITING
        self.prefix_cache.forget(req)
        self.running.remove(req)
        self.waiting.appendleft(req)
        self.num_preemptions += 1

    def schedule(self) -> BatchPlan:
        budget = self.max_num_batched_tokens
        chosen: list[tuple[Request, int]] = []

        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed, budget)
            while not self._grow(req, req.num_computed + n):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is req:
                    break
            if req.status is not RequestStatus.RUNNING:
                continue  # req itself was preempted; running[i] is now the next one
            chosen.append((req, n))
            budget -= n
            i += 1

        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            if not req.pages:
                pages, cached = self.prefix_cache.match(req.token_ids)
                req.pages, req.num_computed = pages, cached
            n = min(req.num_tokens - req.num_computed, budget)
            if not self._grow(req, req.num_computed + n):
                self.allocator.free(req.pages)
                req.pages, req.num_computed = [], 0
                if not self.running and not chosen:
                    raise RuntimeError(
                        f"the KV cache ({self.allocator.num_pages} pages of {self.allocator.page_size}) "
                        f"cannot hold even one chunk of request {req.request_id}"
                    )
                break
            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            chosen.append((req, n))
            budget -= n

        seqs = []
        for req, n in chosen:
            start = req.num_computed
            generating = start >= len(req.prompt_ids)
            phase = Phase.DECODE if n == 1 and generating else Phase.PREFILL
            seqs.append(
                SeqPlan(
                    request_id=req.request_id,
                    token_ids=tuple(req.token_ids[start:start + n]),
                    start_pos=start,
                    pages=tuple(req.pages),
                    phase=phase,
                    sample=start + n == req.num_tokens,
                )
            )
        seqs.sort(key=lambda s: s.phase is Phase.DECODE)  # prefill first, order otherwise kept
        return BatchPlan(tuple(seqs))

    # ------------------------------------------------------------ commit

    def commit(self, plan: BatchPlan, sampled: dict[int, int]) -> StepEvents:
        events = StepEvents(num_scheduled_tokens=plan.num_tokens)
        for seq in plan.seqs:
            req = self.requests[seq.request_id]
            if req.status is not RequestStatus.RUNNING:
                continue
            req.num_computed += len(seq.token_ids)
            self.prefix_cache.insert(req)
            if not seq.sample:
                continue
            token = sampled[req.request_id]
            req.token_ids.append(token)
            reason = self._stop_reason(req, token)
            if reason is not None:
                self._finish(req, reason)
            events.tokens.append(TokenEvent(req.request_id, token, reason is not None, reason))
        return events

    def _stop_reason(self, req: Request, token: int) -> str | None:
        p = req.params
        if not p.ignore_eos and token in self.eos_token_ids:
            return "stop"
        if token in p.stop_token_ids:
            return "stop"
        if len(req.output_ids) >= p.max_tokens:
            return "length"
        if req.num_tokens >= self.max_model_len:
            return "length"
        return None

    def _finish(self, req: Request, reason: str) -> None:
        req.status = RequestStatus.FINISHED
        req.finish_reason = reason
        if req in self.running:
            self.running.remove(req)
        if req.pages:
            self.allocator.free(req.pages)
            req.pages = []
        self.prefix_cache.forget(req)
