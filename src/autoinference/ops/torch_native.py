"""Second implementations of a few op slots, built on PyTorch's own kernels.

They exist so that swapping is exercised on real alternatives, and they are
already faster than the references on accelerators.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..registry import Manifest, register
from ..types import AttentionMetadata
from .interfaces import MoEDispatch
from .reference import gather_context

TORCH = Manifest(kind="kernel", lossless_class="tolerance-exact", provenance="pytorch")


@register("linear", "torch", TORCH)
class TorchLinear:
    def __call__(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(x, weight, bias)


class SDPAAttention:
    """Per-sequence ``scaled_dot_product_attention`` over the gathered context."""

    def __call__(
        self, q: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor, meta: AttentionMetadata, scale: float
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        num_heads = q.shape[1]
        for i in range(meta.num_seqs):
            q_start, q_len, ctx_len = meta.q_start[i], meta.q_len[i], meta.ctx_len[i]
            ki = gather_context(cache_k, meta.page_tables[i], ctx_len).to(q.dtype)
            vi = gather_context(cache_v, meta.page_tables[i], ctx_len).to(q.dtype)
            group = num_heads // ki.shape[1]
            ki = ki.repeat_interleave(group, dim=1).transpose(0, 1)  # [Hq, L, D]
            vi = vi.repeat_interleave(group, dim=1).transpose(0, 1)
            qi = q[q_start:q_start + q_len].transpose(0, 1)           # [Hq, n, D]
            q_pos = torch.arange(ctx_len - q_len, ctx_len, device=q.device)
            mask = torch.arange(ctx_len, device=q.device)[None, :] <= q_pos[:, None]
            oi = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=scale)
            out[q_start:q_start + q_len] = oi.transpose(0, 1)
        return out


register("attn-prefill", "torch-sdpa", TORCH)(SDPAAttention)
register("attn-decode", "torch-sdpa", TORCH)(SDPAAttention)


@register("moe-experts", "grouped", TORCH)
class GroupedMoEExperts:
    """One matmul per expert over that expert's contiguous block of rows."""

    def __call__(self, d: MoEDispatch, gate_up: torch.Tensor, down: torch.Tensor, kind: str) -> torch.Tensor:
        if kind != "silu":
            raise NotImplementedError(f"activation {kind!r} is not implemented yet")
        inter = down.shape[-1]
        out = torch.empty_like(d.hidden)
        offsets = d.offsets.tolist()
        for e in range(len(offsets) - 1):
            lo, hi = offsets[e], offsets[e + 1]
            if lo == hi:
                continue
            h = F.linear(d.hidden[lo:hi], gate_up[e])
            out[lo:hi] = F.linear(F.silu(h[:, :inter]) * h[:, inter:], down[e])
        return out
