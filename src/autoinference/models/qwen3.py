"""Qwen3 dense and Qwen3 MoE, written as a graph of op-slot calls."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..ops import OpTable, RopeSpec
from ..parts.kv import KVStorage
from ..types import ForwardInputs
from .registry import register_arch


@dataclass(frozen=True)
class Qwen3Spec:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope: RopeSpec
    tie_word_embeddings: bool
    hidden_act: str = "silu"
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False
    moe_layers: frozenset[int] = field(default_factory=frozenset)

    @classmethod
    def from_config(cls, cfg: dict) -> "Qwen3Spec":
        rope = cfg.get("rope_parameters") or {}
        scaling = cfg.get("rope_scaling") or {}
        theta = rope.get("rope_theta", cfg.get("rope_theta", 10000.0))
        rope_type = rope.get("rope_type", scaling.get("rope_type", scaling.get("type", "default")))
        heads = cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
        num_layers = cfg["num_hidden_layers"]
        num_experts = cfg.get("num_experts") or cfg.get("num_local_experts") or 0
        moe_layers: set[int] = set()
        if num_experts:
            step = cfg.get("decoder_sparse_step", 1) or 1
            dense_only = set(cfg.get("mlp_only_layers") or [])
            moe_layers = {i for i in range(num_layers) if i not in dense_only and (i + 1) % step == 0}
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=num_layers,
            num_heads=heads,
            num_kv_heads=cfg.get("num_key_value_heads", heads),
            head_dim=head_dim,
            intermediate_size=cfg.get("intermediate_size", 0),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope=RopeSpec(head_dim=head_dim, theta=float(theta), rope_type=rope_type),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            hidden_act=cfg.get("hidden_act", "silu"),
            num_experts=num_experts,
            num_experts_per_tok=cfg.get("num_experts_per_tok", 0),
            moe_intermediate_size=cfg.get("moe_intermediate_size", 0),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", False)),
            moe_layers=frozenset(moe_layers),
        )


class _Layer:
    """One decoder layer's weights and its resolved ops."""

    def __init__(self, i: int, spec: Qwen3Spec, w: dict[str, torch.Tensor], ops: OpTable) -> None:
        p = f"model.layers.{i}."
        self.moe = i in spec.moe_layers
        self.w_in = w[p + "input_layernorm.weight"]
        self.w_post = w[p + "post_attention_layernorm.weight"]
        self.wq = w[p + "self_attn.q_proj.weight"]
        self.wk = w[p + "self_attn.k_proj.weight"]
        self.wv = w[p + "self_attn.v_proj.weight"]
        self.wo = w[p + "self_attn.o_proj.weight"]
        self.w_qn = w[p + "self_attn.q_norm.weight"]
        self.w_kn = w[p + "self_attn.k_norm.weight"]

        self.norm = ops.get("norm", layer=i, layer_type="block")
        self.head_norm = ops.get("norm", layer=i, layer_type="qk")
        self.attn_linear = ops.get("linear", layer=i, layer_type="attention")
        self.rotary = ops.get("rotary", layer=i, layer_type="full")
        self.kv_write = ops.get("kv-write", layer=i, layer_type="full")
        self.attn_prefill = ops.get("attn-prefill", layer=i, layer_type="full", phase="prefill")
        self.attn_decode = ops.get("attn-decode", layer=i, layer_type="full", phase="decode")

        if self.moe:
            self.w_router = w[p + "mlp.gate.weight"]
            self.gate_up, self.down = _expert_weights(p + "mlp.experts.", spec, w)
            self.router = ops.get("moe-router", layer=i, layer_type="moe")
            self.dispatch = ops.get("moe-dispatch", layer=i, layer_type="moe")
            self.experts = ops.get("moe-experts", layer=i, layer_type="moe")
            self.combine = ops.get("moe-combine", layer=i, layer_type="moe")
        else:
            self.w_gate = w[p + "mlp.gate_proj.weight"]
            self.w_up = w[p + "mlp.up_proj.weight"]
            self.w_down = w[p + "mlp.down_proj.weight"]
            self.mlp_linear = ops.get("linear", layer=i, layer_type="mlp")
            self.activation = ops.get("activation", layer=i, layer_type="mlp")


def _expert_weights(prefix: str, spec: Qwen3Spec, w: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert weights as [E, 2I, H] (gate rows then up rows) and [E, H, I].

    Checkpoints store one tensor per expert; some tools store them fused.
    """
    if prefix + "gate_up_proj" in w:
        return w.pop(prefix + "gate_up_proj"), w.pop(prefix + "down_proj")
    gate_up, down = [], []
    for e in range(spec.num_experts):
        gate = w.pop(f"{prefix}{e}.gate_proj.weight")
        up = w.pop(f"{prefix}{e}.up_proj.weight")
        gate_up.append(torch.cat([gate, up], dim=0))
        down.append(w.pop(f"{prefix}{e}.down_proj.weight"))
    return torch.stack(gate_up), torch.stack(down)


class Qwen3Model:
    def __init__(self, spec: Qwen3Spec, weights: dict[str, torch.Tensor], ops: OpTable) -> None:
        self.spec = spec
        self.ops = ops
        self.embed_weight = weights["model.embed_tokens.weight"]
        self.norm_weight = weights["model.norm.weight"]
        self.lm_head_weight = weights.get("lm_head.weight", self.embed_weight)
        if spec.tie_word_embeddings:
            self.lm_head_weight = self.embed_weight
        self.embedding = ops.get("embedding")
        self.final_norm = ops.get("norm", layer_type="final")
        self.lm_head = ops.get("lm-head")
        self.layers = [_Layer(i, spec, weights, ops) for i in range(spec.num_layers)]
        self.scale = spec.head_dim ** -0.5

    @property
    def num_layers(self) -> int:
        return self.spec.num_layers

    def forward(self, inputs: ForwardInputs, kv: KVStorage) -> torch.Tensor:
        s = self.spec
        T = inputs.token_ids.shape[0]
        Tp = inputs.num_prefill_tokens
        eps = s.rms_norm_eps
        h = self.embedding(inputs.token_ids, self.embed_weight)

        for i, L in enumerate(self.layers):
            x = L.norm(h, L.w_in, eps)
            q = L.attn_linear(x, L.wq).view(T, s.num_heads, s.head_dim)
            k = L.attn_linear(x, L.wk).view(T, s.num_kv_heads, s.head_dim)
            v = L.attn_linear(x, L.wv).view(T, s.num_kv_heads, s.head_dim)
            q = L.head_norm(q, L.w_qn, eps)
            k = L.head_norm(k, L.w_kn, eps)
            q, k = L.rotary(q, k, inputs.positions, s.rope)
            L.kv_write(k, v, inputs.slot_mapping, kv.keys[i], kv.values[i])

            o = torch.empty_like(q)
            if Tp:
                o[:Tp] = L.attn_prefill(q[:Tp], kv.keys[i], kv.values[i], inputs.prefill, self.scale)
            if T > Tp:
                o[Tp:] = L.attn_decode(q[Tp:], kv.keys[i], kv.values[i], inputs.decode, self.scale)
            h = h + L.attn_linear(o.reshape(T, s.num_heads * s.head_dim), L.wo)

            x = L.norm(h, L.w_post, eps)
            if L.moe:
                expert_ids, weights = L.router(x, L.w_router, s.num_experts_per_tok, s.norm_topk_prob)
                d = L.dispatch(x, expert_ids, weights, s.num_experts)
                y = L.combine(L.experts(d, L.gate_up, L.down, s.hidden_act), d)
            else:
                act = L.activation(L.mlp_linear(x, L.w_gate), L.mlp_linear(x, L.w_up), s.hidden_act)
                y = L.mlp_linear(act, L.w_down)
            h = h + y

        h = self.final_norm(h.index_select(0, inputs.logits_indices), self.norm_weight, eps)
        return self.lm_head(h, self.lm_head_weight)


@register_arch("qwen3", "qwen3_moe")
def build_qwen3(config: dict, weights: dict[str, torch.Tensor], ops: OpTable) -> Qwen3Model:
    return Qwen3Model(Qwen3Spec.from_config(config), weights, ops)
