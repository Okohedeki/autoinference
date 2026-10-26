"""The engine: builds every part from the configuration and runs the step loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from . import models, ops, parts  # noqa: F401  (registers built-in implementations)
from .config import EngineConfig
from .models import get_arch
from .observability import Profiler
from .ops import OpTable
from .parts.kv import KVStorage
from .parts.loader import read_config, resolve_model_dir
from .platform import free_memory_bytes, resolve_device, resolve_dtype
from .registry import REGISTRY
from .types import BatchPlan, Phase, Request, SamplingParams, SeqPlan, StepEvents


@dataclass
class Generation:
    request_id: int
    prompt_ids: list[int]
    token_ids: list[int]
    finish_reason: str | None
    text: str | None = None


class Engine:
    def __init__(self, model: str | Path, config: EngineConfig | None = None) -> None:
        cfg = config or EngineConfig()
        self.config = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)
        self.profiler = Profiler(cfg.profile, self.device)
        self.model_dir = resolve_model_dir(str(model))
        self.hf_config = read_config(self.model_dir)

        self.placement = self._part("placement", device=self.device)
        loader = self._part("loader", dtype=self.dtype, device_for=self.placement.device_for)
        weights = loader.load(self.model_dir)

        self.ops = OpTable(cfg, self.device.type, self.profiler)
        self.model = get_arch(self.hf_config["model_type"])(self.hf_config, weights, self.ops)
        spec = self.model.spec

        num_pages = cfg.num_pages or self._auto_num_pages(spec)
        self.allocator = self._part("kv-allocator", num_pages=num_pages, page_size=cfg.page_size)
        self.kv = KVStorage.create(
            spec.num_layers, num_pages, cfg.page_size, spec.num_kv_heads, spec.head_dim, self.dtype, self.device
        )
        self.prefix_cache = self._part("prefix-cache", allocator=self.allocator)

        self.renderer = self._make_renderer()
        self.eos_token_ids = self.renderer.eos_token_ids if self.renderer else self._config_eos_ids()
        self.scheduler = self._part(
            "scheduler",
            allocator=self.allocator,
            prefix_cache=self.prefix_cache,
            max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_model_len=cfg.max_model_len,
            eos_token_ids=self.eos_token_ids,
        )
        self.input_prep = self._part("input-prep", page_size=cfg.page_size, device=self.device)
        self.executor = self._part("executor")
        self.sampler = self._part("sampler")
        # Declared parts with only an "off" choice so far; built so the configuration is validated.
        self.overlap = self._part("overlap")
        self.spec_planner = self._part("spec-planner")
        self.spec_drafter = self._part("spec-drafter")

    # ------------------------------------------------------------ construction

    def _part(self, slot: str, **kwargs: Any) -> Any:
        name = self.config.parts[slot]
        options = self.config.part_options.get(slot, {})
        return REGISTRY.create(slot, name, **kwargs, **options)

    def _auto_num_pages(self, spec) -> int:
        cfg = self.config
        wanted = -(-cfg.max_num_seqs * cfg.max_model_len // cfg.page_size) + 1
        free = free_memory_bytes(self.device)
        if free is None:
            return wanted
        per_page = KVStorage.bytes_per_page(spec.num_layers, cfg.page_size, spec.num_kv_heads, spec.head_dim, self.dtype)
        return max(1, min(wanted, int(free * cfg.memory_fraction) // per_page))

    def _make_renderer(self):
        has_tokenizer = any((self.model_dir / f).exists() for f in ("tokenizer.json", "tokenizer_config.json"))
        return self._part("renderer", model_dir=self.model_dir) if has_tokenizer else None

    def _config_eos_ids(self) -> tuple[int, ...]:
        ids: set[int] = set()
        for source in (self.hf_config, self._generation_config()):
            eos = source.get("eos_token_id")
            if isinstance(eos, int):
                ids.add(eos)
            elif isinstance(eos, list):
                ids.update(eos)
        return tuple(sorted(ids))

    def _generation_config(self) -> dict:
        path = self.model_dir / "generation_config.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    # ------------------------------------------------------------ step loop

    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> int:
        req = Request(list(prompt_ids), params or SamplingParams())
        self.scheduler.add(req)
        return req.request_id

    def abort(self, request_id: int) -> None:
        self.scheduler.abort(request_id)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> StepEvents:
        prof = self.profiler
        with prof.span("schedule"):
            plan = self.scheduler.schedule()
        if plan.empty:
            return StepEvents()
        with prof.span("input-prep"):
            inputs = self.input_prep.prepare(plan)
        with prof.span("forward"):
            logits = self.executor.run(lambda x: self.model.forward(x, self.kv), inputs)
        with prof.span("sample"):
            reqs = [self.scheduler.requests[r] for r in inputs.sample_request_ids]
            tokens = self.sampler.sample(logits, reqs)
        with prof.span("commit"):
            events = self.scheduler.commit(plan, dict(zip(inputs.sample_request_ids, tokens)))
        prof.count("steps")
        prof.count("tokens_scheduled", plan.num_tokens)
        prof.count("tokens_generated", len(events.tokens))
        return events

    def generate(
        self, prompts: Iterable[list[int] | str], params: SamplingParams | None = None
    ) -> list[Generation]:
        ids = []
        for prompt in prompts:
            if isinstance(prompt, str):
                if self.renderer is None:
                    raise ValueError("this model has no tokenizer files; pass token ids instead")
                prompt = self.renderer.encode(prompt)
            ids.append(self.add_request(prompt, params))
        while self.has_work():
            self.step()
        out = []
        for rid in ids:
            req = self.scheduler.requests.pop(rid)
            text = self.renderer.decode(req.output_ids) if self.renderer else None
            out.append(Generation(rid, req.prompt_ids, req.output_ids, req.finish_reason, text))
        return out

    def chat(self, messages: list[dict[str, str]], params: SamplingParams | None = None) -> Generation:
        if self.renderer is None:
            raise ValueError("this model has no tokenizer files")
        return self.generate([self.renderer.apply_chat_template(messages)], params)[0]

    # ------------------------------------------------------------ inspection

    def score(self, token_ids: list[int]) -> torch.Tensor:
        """Logits at every position of one sequence, computed as a single prefill."""
        n = len(token_ids)
        pages = self.allocator.allocate(self.allocator.pages_needed(n, 0))
        if pages is None:
            raise RuntimeError("not enough free KV pages to score this sequence")
        try:
            seq = SeqPlan(-1, tuple(token_ids), 0, tuple(pages), Phase.PREFILL, sample=True)
            inputs = self.input_prep.prepare(BatchPlan((seq,)))
            inputs.logits_indices = torch.arange(n, device=self.device)
            return self.executor.run(lambda x: self.model.forward(x, self.kv), inputs)
        finally:
            self.allocator.free(pages)

    def describe(self) -> dict:
        return {
            "model": str(self.model_dir),
            "model_type": self.hf_config["model_type"],
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "kv_pages": self.allocator.num_pages,
            "page_size": self.allocator.page_size,
            "parts": dict(self.config.parts),
            "ops": self.ops.describe(),
        }
