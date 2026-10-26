"""Sampler part: logits processors, then token selection."""

from __future__ import annotations

import torch

from ..registry import Manifest, register
from ..types import Request


@register("logits-processor", "repetition-penalty", Manifest(kind="kernel", lossless_class="bit-exact"))
class RepetitionPenalty:
    """Divides positive logits (multiplies negative ones) of tokens already seen."""

    def __call__(self, logits: torch.Tensor, request: Request) -> torch.Tensor:
        penalty = request.params.repetition_penalty
        if penalty == 1.0:
            return logits
        seen = torch.tensor(sorted(set(request.token_ids)), dtype=torch.long, device=logits.device)
        picked = logits[seen]
        logits = logits.clone()
        logits[seen] = torch.where(picked > 0, picked / penalty, picked * penalty)
        return logits


@register("sampler", "reference", Manifest(kind="kernel", lossless_class="distribution-preserving"))
class ReferenceSampler:
    """Greedy, or temperature then top-k then top-p, one request at a time."""

    def __init__(self, processors: list | None = None) -> None:
        self.processors = processors if processors is not None else [RepetitionPenalty()]

    def sample(self, logits: torch.Tensor, requests: list[Request]) -> list[int]:
        out = []
        for row, req in zip(logits.float(), requests):
            for proc in self.processors:
                row = proc(row, req)
            out.append(self._pick(row, req))
        return out

    def _pick(self, row: torch.Tensor, req: Request) -> int:
        p = req.params
        if p.greedy:
            return int(torch.argmax(row))
        row = row / p.temperature
        if p.top_k > 0:
            kth = torch.topk(row, min(p.top_k, row.numel())).values[-1]
            row = row.masked_fill(row < kth, float("-inf"))
        probs = torch.softmax(row, dim=-1)
        if p.top_p < 1.0:
            sorted_probs, order = torch.sort(probs, descending=True)
            keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < p.top_p
            sorted_probs = sorted_probs * keep
            probs = torch.zeros_like(probs).scatter(0, order, sorted_probs)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs.cpu(), 1, generator=req.generator))
