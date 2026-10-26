"""Input preparation part: turn a batch plan into the tensors kernels consume."""

from __future__ import annotations

import torch

from ..registry import Manifest, register
from ..types import AttentionMetadata, BatchPlan, ForwardInputs, Phase


@register("input-prep", "reference", Manifest(kind="policy", lossless_class="bit-exact"))
class ReferenceInputPrep:
    def __init__(self, page_size: int, device: torch.device) -> None:
        self.page_size = page_size
        self.device = device

    def prepare(self, plan: BatchPlan) -> ForwardInputs:
        token_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        logits_indices: list[int] = []
        sample_ids: list[int] = []
        metas = {
            Phase.PREFILL: AttentionMetadata([], [], [], [], self.page_size),
            Phase.DECODE: AttentionMetadata([], [], [], [], self.page_size),
        }
        phase_offset = {Phase.PREFILL: 0, Phase.DECODE: 0}

        for seq in plan.seqs:
            n = len(seq.token_ids)
            meta = metas[seq.phase]
            meta.q_start.append(phase_offset[seq.phase])
            meta.q_len.append(n)
            meta.ctx_len.append(seq.start_pos + n)
            meta.page_tables.append(list(seq.pages))
            phase_offset[seq.phase] += n

            for j, tok in enumerate(seq.token_ids):
                pos = seq.start_pos + j
                token_ids.append(tok)
                positions.append(pos)
                slots.append(seq.pages[pos // self.page_size] * self.page_size + pos % self.page_size)
            if seq.sample:
                logits_indices.append(len(token_ids) - 1)
                sample_ids.append(seq.request_id)

        def as_tensor(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=self.device)

        return ForwardInputs(
            token_ids=as_tensor(token_ids),
            positions=as_tensor(positions),
            slot_mapping=as_tensor(slots),
            logits_indices=as_tensor(logits_indices),
            num_prefill_tokens=phase_offset[Phase.PREFILL],
            prefill=metas[Phase.PREFILL],
            decode=metas[Phase.DECODE],
            sample_request_ids=sample_ids,
        )
