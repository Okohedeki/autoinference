"""Command line: generate, chat, serve, and list the registered parts."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .config import EngineConfig
from .registry import REGISTRY
from .types import SamplingParams


def _engine_from(args: argparse.Namespace):
    from .engine import Engine

    cfg = EngineConfig.load(args.config) if args.config else EngineConfig()
    if args.device:
        cfg.device = args.device
    if args.dtype:
        cfg.dtype = args.dtype
    if getattr(args, "profile", False):
        cfg.profile = True
    return Engine(args.model, cfg)


def _params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    engine = _engine_from(args)
    params = _params(args)
    start = time.perf_counter()
    if args.chat:
        results = [engine.chat([{"role": "user", "content": p}], params) for p in args.prompt]
    else:
        results = engine.generate(args.prompt, params)
    elapsed = time.perf_counter() - start
    for prompt, res in zip(args.prompt, results):
        print(f"--- prompt: {prompt!r}")
        print(res.text)
        print(f"[{len(res.token_ids)} tokens, finish={res.finish_reason}]")
    total = sum(len(r.token_ids) for r in results)
    print(f"\n{total} tokens in {elapsed:.2f}s ({total / max(elapsed, 1e-9):.1f} tok/s)", file=sys.stderr)
    if args.describe:
        print(json.dumps(engine.describe(), indent=2), file=sys.stderr)
    if engine.profiler.enabled:
        print(engine.profiler.summary(), file=sys.stderr)


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(_engine_from(args), args.host, args.port, model_name=args.model)


def cmd_parts(args: argparse.Namespace) -> None:
    for slot in REGISTRY.slots():
        names = []
        for name in REGISTRY.names(slot):
            m = REGISTRY.get(slot, name).manifest
            names.append(f"{name} ({m.kind}, {m.lossless_class})")
        print(f"{slot:18s} {', '.join(names)}")


def main(argv: list[str] | None = None) -> None:
    from . import models, ops, parts  # noqa: F401  (register built-ins)

    ap = argparse.ArgumentParser(prog="autoinference")
    sub = ap.add_subparsers(dest="command", required=True)

    def engine_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("model", help="local checkpoint directory or Hugging Face repo id")
        p.add_argument("--config", help="engine TOML file (or the name of a bundled one, e.g. torch.toml)")
        p.add_argument("--device", help="cpu, cuda, mps or auto")
        p.add_argument("--dtype", help="float32, bfloat16 or float16")

    g = sub.add_parser("generate", help="generate text for one or more prompts")
    engine_args(g)
    g.add_argument("--prompt", action="append", required=True)
    g.add_argument("--chat", action="store_true", help="wrap each prompt in the chat template")
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.add_argument("--seed", type=int)
    g.add_argument("--profile", action="store_true", help="print per-slot timings")
    g.add_argument("--describe", action="store_true", help="print which implementation filled each slot")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("serve", help="serve an OpenAI-compatible chat endpoint")
    engine_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    p = sub.add_parser("parts", help="list every slot and its registered implementations")
    p.set_defaults(func=cmd_parts)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
