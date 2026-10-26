"""Batching, chunked prefill, preemption and prefix caching do not change outputs."""

from __future__ import annotations

from autoinference import SamplingParams

from .conftest import PROMPTS, make_engine

PARAMS = SamplingParams(max_tokens=16, ignore_eos=True)


def alone(path) -> list[list[int]]:
    engine = make_engine(path, max_num_seqs=1)
    return [engine.generate([p], PARAMS)[0].token_ids for p in PROMPTS]


def test_batched_equals_alone(tiny_model):
    path, _ = tiny_model
    engine = make_engine(path)
    assert [g.token_ids for g in engine.generate(PROMPTS, PARAMS)] == alone(path)


def test_chunked_prefill(tiny_model):
    path, _ = tiny_model
    engine = make_engine(path, max_num_batched_tokens=4)  # prompts arrive in chunks of at most 4 tokens
    assert [g.token_ids for g in engine.generate(PROMPTS, PARAMS)] == alone(path)


def test_preemption(tiny_dense):
    path, _ = tiny_dense
    engine = make_engine(path, num_pages=5, page_size=8)  # room for one long request, not all three
    out = [g.token_ids for g in engine.generate(PROMPTS, PARAMS)]
    assert engine.scheduler.num_preemptions > 0
    assert out == alone(path)
    assert engine.allocator.num_free() == engine.allocator.num_pages


def test_prefix_cache_reuses_pages(tiny_dense):
    path, _ = tiny_dense
    shared = list(range(10, 50))  # 40 tokens = 5 full pages of 8
    prompts = [shared + [3, 4], shared + [5, 6, 7]]
    plain = make_engine(path, page_size=8)
    cached = make_engine(path, page_size=8, parts={"prefix-cache": "block-hash"})
    expected = [plain.generate([p], PARAMS)[0].token_ids for p in prompts]

    first = cached.generate([prompts[0]], PARAMS)[0].token_ids
    assert cached.prefix_cache.num_cached_pages >= 5
    req_id = cached.add_request(prompts[1], PARAMS)
    cached.step()  # admission: prefix matched, only the tail is computed
    assert cached.scheduler.requests[req_id].num_computed == 40 + 3
    while cached.has_work():
        cached.step()
    second = cached.scheduler.requests[req_id].output_ids
    assert [first, second] == expected


def test_finished_requests_release_pages(tiny_moe):
    path, _ = tiny_moe
    engine = make_engine(path)
    engine.generate(PROMPTS, PARAMS)
    assert engine.allocator.num_free() == engine.allocator.num_pages
