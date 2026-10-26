"""The reference engine reproduces Hugging Face transformers on the same weights."""

from __future__ import annotations

import torch

from autoinference import SamplingParams

from .conftest import PROMPTS, make_engine


def hf_greedy(model, prompt: list[int], n: int) -> list[int]:
    ids = torch.tensor([prompt])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0)
    return out[0, len(prompt):].tolist()


def test_logits_match(tiny_model):
    path, hf = tiny_model
    engine = make_engine(path)
    for prompt in PROMPTS:
        ours = engine.score(prompt)
        with torch.no_grad():
            theirs = hf(torch.tensor([prompt])).logits[0].float()
        torch.testing.assert_close(ours, theirs, atol=1e-4, rtol=1e-4)


def test_greedy_match(tiny_model):
    path, hf = tiny_model
    engine = make_engine(path)
    params = SamplingParams(max_tokens=20, ignore_eos=True)
    for prompt in PROMPTS:
        ours = engine.generate([prompt], params)[0].token_ids
        assert ours == hf_greedy(hf, prompt, 20)
