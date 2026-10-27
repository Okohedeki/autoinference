"""Greedy generation on a real checkpoint matches Hugging Face transformers.

Skipped unless AUTOINFERENCE_REAL_MODEL names a local checkpoint directory or a
Hugging Face repo id. Runs in FP32 so both sides compute the same numbers.
"""

from __future__ import annotations

import os

import pytest
import torch

from autoinference import Engine, EngineConfig, SamplingParams

MODEL = os.environ.get("AUTOINFERENCE_REAL_MODEL")
pytestmark = [pytest.mark.real_weights, pytest.mark.skipif(not MODEL, reason="AUTOINFERENCE_REAL_MODEL not set")]

QUESTIONS = [
    "Name three primary colors.",
    "What is 12 times 7?",
    "Write one sentence about the ocean.",
]


@pytest.fixture(scope="module")
def engines():
    from transformers import AutoModelForCausalLM

    device = os.environ.get("AUTOINFERENCE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    cfg = EngineConfig.load("reference.toml")
    cfg.device, cfg.dtype, cfg.max_model_len = device, "float32", 1024
    engine = Engine(MODEL, cfg)
    hf = AutoModelForCausalLM.from_pretrained(str(engine.model_dir), dtype=torch.float32).to(device).eval()
    return engine, hf


def test_greedy_matches_transformers(engines):
    engine, hf = engines
    params = SamplingParams(max_tokens=32)  # both sides stop at the end-of-turn token
    for q in QUESTIONS:
        prompt = engine.renderer.apply_chat_template([{"role": "user", "content": q}])
        ours = engine.generate([prompt], params)[0].token_ids
        ids = torch.tensor([prompt], device=hf.device)
        with torch.no_grad():
            out = hf.generate(ids, max_new_tokens=32, do_sample=False)
        assert ours == out[0, len(prompt):].tolist(), q
