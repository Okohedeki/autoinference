"""Shared fixtures: tiny random checkpoints saved in the Hugging Face layout."""

from __future__ import annotations

import pytest
import torch

from autoinference import Engine, EngineConfig

TINY_COMMON = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)


def _save(model_cls, config_cls, path, **extra):
    torch.manual_seed(0)
    config = config_cls(**TINY_COMMON, **extra)
    model = model_cls(config).eval()
    # Larger weights than the default init so routing and attention are not near-uniform.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.15)
    model.save_pretrained(path)
    return model


@pytest.fixture(scope="session")
def tiny_dense(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    path = tmp_path_factory.mktemp("qwen3-tiny")
    model = _save(Qwen3ForCausalLM, Qwen3Config, path)
    return path, model


@pytest.fixture(scope="session")
def tiny_moe(tmp_path_factory):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    path = tmp_path_factory.mktemp("qwen3-moe-tiny")
    model = _save(
        Qwen3MoeForCausalLM,
        Qwen3MoeConfig,
        path,
        moe_intermediate_size=32,
        num_local_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    return path, model


@pytest.fixture(params=["tiny_dense", "tiny_moe"])
def tiny_model(request):
    return request.getfixturevalue(request.param)


def make_engine(path, **overrides) -> Engine:
    cfg = EngineConfig.load("reference.toml")
    cfg.device = "cpu"
    cfg.dtype = "float32"
    cfg.max_model_len = 256
    cfg.num_pages = 128
    ops = overrides.pop("ops", {})
    parts = overrides.pop("parts", {})
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.ops.update(ops)
    cfg.parts.update(parts)
    return Engine(path, cfg)


PROMPTS = [
    [5, 17, 42, 99, 3, 250, 18],
    [7, 7, 7, 120, 64],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
]
