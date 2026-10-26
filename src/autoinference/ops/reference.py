"""Reference implementations of every op slot.

Written for obvious correctness, not speed: explicit loops where they make the
logic clearer, FP32 accumulation, outputs cast back to the input dtype. Every
other implementation of a slot is checked against these.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..registry import Manifest, register
from ..types import AttentionMetadata
from .interfaces import MoEDispatch, RopeSpec

REF = Manifest(kind="kernel", lossless_class="tolerance-exact", provenance="autoinference reference")


@register("embedding", "reference", REF)
class ReferenceEmbedding:
    def __call__(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return weight.index_select(0, token_ids)


@register("linear", "reference", REF)
class ReferenceLinear:
    def __call__(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = x.float() @ weight.float().t()
        if bias is not None:
            y = y + bias.float()
        return y.to(x.dtype)


@register("norm", "reference", REF)
class ReferenceRMSNorm:
    """RMSNorm over the last dimension; the weight multiplies after the cast back."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return weight * xf.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@register("rotary", "reference", REF)
class ReferenceRotary:
    """Rotary embedding with the two halves of each head rotated together."""

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, spec: RopeSpec
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spec.rope_type != "default":
            raise NotImplementedError(f"rope type {spec.rope_type!r} is not implemented yet")
        dim = spec.head_dim
        inv_freq = 1.0 / (spec.theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=q.device).float() / dim))
        freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)                  # [T, D]
        cos = emb.cos().to(q.dtype)[:, None, :]
        sin = emb.sin().to(q.dtype)[:, None, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


@register("kv-write", "reference", REF)
class ReferenceKVWrite:
    def __call__(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor
    ) -> None:
        page_size = cache_k.shape[1]
        pages = slot_mapping // page_size
        offsets = slot_mapping % page_size
        cache_k[pages, offsets] = k.to(cache_k.dtype)
        cache_v[pages, offsets] = v.to(cache_v.dtype)


def gather_context(cache: torch.Tensor, pages: list[int], ctx_len: int) -> torch.Tensor:
    """Keys or values of one sequence, [ctx_len, Hkv, D], in position order."""
    idx = torch.tensor(pages, dtype=torch.long, device=cache.device)
    flat = cache.index_select(0, idx).reshape(-1, cache.shape[2], cache.shape[3])
    return flat[:ctx_len]


class ReferenceAttention:
    """Causal attention over the paged cache, one sequence at a time."""

    def __call__(
        self, q: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor, meta: AttentionMetadata, scale: float
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        num_heads = q.shape[1]
        for i in range(meta.num_seqs):
            q_start, q_len, ctx_len = meta.q_start[i], meta.q_len[i], meta.ctx_len[i]
            qi = q[q_start:q_start + q_len].float()                       # [n, Hq, D]
            ki = gather_context(cache_k, meta.page_tables[i], ctx_len).float()  # [L, Hkv, D]
            vi = gather_context(cache_v, meta.page_tables[i], ctx_len).float()
            group = num_heads // ki.shape[1]
            ki = ki.repeat_interleave(group, dim=1)                       # [L, Hq, D]
            vi = vi.repeat_interleave(group, dim=1)
            scores = torch.einsum("nhd,lhd->hnl", qi, ki) * scale         # [Hq, n, L]
            q_pos = torch.arange(ctx_len - q_len, ctx_len, device=q.device)
            k_pos = torch.arange(ctx_len, device=q.device)
            visible = k_pos[None, :] <= q_pos[:, None]                    # [n, L]
            scores = scores.masked_fill(~visible[None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[q_start:q_start + q_len] = torch.einsum("hnl,lhd->nhd", probs, vi).to(q.dtype)
        return out


register("attn-prefill", "reference", REF)(ReferenceAttention)
register("attn-decode", "reference", REF)(ReferenceAttention)


@register("activation", "reference", REF)
class ReferenceActivation:
    def __call__(self, gate: torch.Tensor, up: torch.Tensor, kind: str) -> torch.Tensor:
        if kind != "silu":
            raise NotImplementedError(f"activation {kind!r} is not implemented yet")
        return F.silu(gate) * up


@register("moe-router", "reference", REF)
class ReferenceMoERouter:
    """Softmax over all experts, then top-k, optionally renormalized."""

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, top_k: int, normalize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = x.float() @ weight.float().t()
        probs = torch.softmax(logits, dim=-1)
        top_w, top_ids = torch.topk(probs, top_k, dim=-1)
        if normalize:
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        return top_ids, top_w.to(x.dtype)


@register("moe-dispatch", "reference", REF)
class ReferenceMoEDispatch:
    """Sort token-expert pairs by expert so each expert's rows are contiguous."""

    def __call__(self, x: torch.Tensor, expert_ids: torch.Tensor, weights: torch.Tensor, num_experts: int) -> MoEDispatch:
        num_tokens, top_k = expert_ids.shape
        flat_ids = expert_ids.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        src_token = order // top_k
        counts = torch.bincount(flat_ids, minlength=num_experts).cpu()
        offsets = torch.zeros(num_experts + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(counts, dim=0)
        return MoEDispatch(
            hidden=x.index_select(0, src_token),
            expert_ids=flat_ids[order],
            weights=weights.reshape(-1)[order],
            src_token=src_token,
            offsets=offsets,
            num_tokens=num_tokens,
        )


@register("moe-experts", "reference", REF)
class ReferenceMoEExperts:
    """One gated MLP per token-expert pair, computed row by row."""

    def __call__(self, d: MoEDispatch, gate_up: torch.Tensor, down: torch.Tensor, kind: str) -> torch.Tensor:
        if kind != "silu":
            raise NotImplementedError(f"activation {kind!r} is not implemented yet")
        inter = down.shape[-1]
        out = torch.empty_like(d.hidden)
        for r in range(d.hidden.shape[0]):
            e = int(d.expert_ids[r])
            h = gate_up[e].float() @ d.hidden[r].float()      # [2I]
            act = F.silu(h[:inter]) * h[inter:]
            out[r] = (down[e].float() @ act).to(out.dtype)
        return out


@register("moe-combine", "reference", REF)
class ReferenceMoECombine:
    def __call__(self, expert_out: torch.Tensor, d: MoEDispatch) -> torch.Tensor:
        out = torch.zeros(d.num_tokens, expert_out.shape[1], dtype=expert_out.dtype, device=expert_out.device)
        weighted = expert_out * d.weights[:, None].to(expert_out.dtype)
        out.index_add_(0, d.src_token, weighted)
        return out


@register("lm-head", "reference", REF)
class ReferenceLMHead:
    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x.float() @ weight.float().t()
