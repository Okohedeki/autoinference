"""Swapping an implementation through configuration leaves results unchanged."""

from __future__ import annotations

import pytest
import torch

from autoinference import REGISTRY, EngineConfig, SamplingParams
from autoinference.config import OpOverride

from .conftest import PROMPTS, make_engine

ALTERNATES = [
    (slot, name)
    for slot in ("linear", "attn-prefill", "attn-decode", "moe-experts")
    for name in REGISTRY.names(slot)
    if name != "reference"
]


@pytest.mark.parametrize("slot,impl", ALTERNATES)
def test_op_swap(tiny_model, slot, impl):
    path, _ = tiny_model
    ref = make_engine(path)
    alt = make_engine(path, ops={slot: impl})
    params = SamplingParams(max_tokens=12, ignore_eos=True)
    for prompt in PROMPTS:
        torch.testing.assert_close(alt.score(prompt), ref.score(prompt), atol=1e-4, rtol=1e-4)
    assert [g.token_ids for g in alt.generate(PROMPTS, params)] == [g.token_ids for g in ref.generate(PROMPTS, params)]


def test_bundled_torch_config_matches_reference(tiny_moe):
    path, _ = tiny_moe
    ref = make_engine(path)
    cfg = EngineConfig.load("torch.toml")
    alt = make_engine(path, ops=cfg.ops, parts=cfg.parts, op_overrides=cfg.op_overrides)
    params = SamplingParams(max_tokens=12, ignore_eos=True)
    assert [g.token_ids for g in alt.generate(PROMPTS, params)] == [g.token_ids for g in ref.generate(PROMPTS, params)]
    used = alt.describe()["ops"]["attn-decode"]
    assert used == {"reference": 2, "torch-sdpa": 1}


def test_override_resolution():
    cfg = EngineConfig()
    cfg.ops["attn-decode"] = "torch-sdpa"
    cfg.op_overrides = [
        OpOverride(slot="attn-decode", impl="reference", layers=(0, 1)),
        OpOverride(slot="attn-decode", impl="other", layers=(1, 1), device="cuda"),
    ]
    assert cfg.op_impl("attn-decode", 0, "full", "decode", "cpu") == "reference"
    assert cfg.op_impl("attn-decode", 1, "full", "decode", "cpu") == "reference"
    assert cfg.op_impl("attn-decode", 1, "full", "decode", "cuda") == "other"
    assert cfg.op_impl("attn-decode", 2, "full", "decode", "cpu") == "torch-sdpa"
    assert cfg.op_impl("linear", 2, "mlp", None, "cpu") == "reference"
