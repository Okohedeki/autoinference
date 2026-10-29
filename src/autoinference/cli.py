"""Command line: generate, chat, serve, list the registered parts, and run the bio service."""

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


def _bio_service(args: argparse.Namespace, start: bool = True):
    from .bio import BioService

    options = {"boltz2": {"command": args.boltz_command}} if args.boltz_command else {}
    return BioService(args.root, args.gpu_hour_price, options, start=start)


def _read_spec(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_bio_serve(args: argparse.Namespace) -> None:
    from .bio.service import serve

    serve(_bio_service(args), args.host, args.port)


def cmd_bio_mcp(args: argparse.Namespace) -> None:
    from .bio.mcp_server import run_stdio

    run_stdio(_bio_service(args))


def cmd_bio_estimate(args: argparse.Namespace) -> None:
    print(json.dumps(_bio_service(args, start=False).estimate(_read_spec(args.spec)), indent=2))


def cmd_bio_run(args: argparse.Namespace) -> None:
    service = _bio_service(args)
    job = service.submit(_read_spec(args.spec))
    print(f"job {job['job_id']}: estimated {job['estimate']['gpu_hours']:.4f} GPU-hours", file=sys.stderr)
    status = service.wait(job["job_id"])
    if status["status"] != "done":
        print(status["error"], file=sys.stderr)
        sys.exit(1)
    data = service.results(job["job_id"], args.top)
    for r in data["results"]:
        pose = r.get("pose") or {}
        flags = "" if pose.get("valid", True) else "  INVALID POSE: " + "; ".join(pose["issues"])
        p = f"  p_active={r['p_active']:.3f}" if r.get("p_active") is not None else ""
        scores = ", ".join(f"{k}={v:.3f}" for k, v in r["scores"].items())
        print(f"{r['rank']:>3}. {r['id']}  {scores}{p}{flags}")
    print(f"\njob {job['job_id']}, output digest {data['manifest']['output_digest'][:16]}", file=sys.stderr)


def cmd_bio_verify(args: argparse.Namespace) -> None:
    print(json.dumps(_bio_service(args).verify(args.job), indent=2))


def cmd_bio_tune(args: argparse.Namespace) -> None:
    from .bio.tuning import load_assay

    labels = load_assay(args.assay, args.threshold, not args.lower_is_active)
    model = _bio_service(args, start=False).fit_tuned_model(args.job, labels, args.model_id)
    print(json.dumps({"id": model["id"], "metrics": model["metrics"]}, indent=2))


def cmd_bio_backends(args: argparse.Namespace) -> None:
    for b in _bio_service(args, start=False).backends():
        print(f"{b['name']:18s} {b['seconds_per_item']:>8.3f} s/candidate  {b['doc']}")


def main(argv: list[str] | None = None) -> None:
    from . import bio, models, ops, parts  # noqa: F401  (register built-ins)

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

    b = sub.add_parser("bio", help="hosted inference for biology models")
    bsub = b.add_subparsers(dest="bio_command", required=True)

    def bio_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", default="bio-workspace", help="workspace directory for jobs, models and logs")
        p.add_argument("--gpu-hour-price", type=float, default=1.0, help="GPU rental price used for cost figures")
        p.add_argument("--boltz-command", help="how to invoke boltz, if it is not on PATH")

    for name, func, help_ in (("serve", cmd_bio_serve, "serve the HTTP API"),
                              ("mcp", cmd_bio_mcp, "serve the MCP tools over stdio"),
                              ("backends", cmd_bio_backends, "list the model backends")):
        p = bsub.add_parser(name, help=help_)
        bio_args(p)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8100)
        p.set_defaults(func=func)
    p = bsub.add_parser("estimate", help="estimate the GPU time and cost of a campaign spec")
    bio_args(p)
    p.add_argument("spec")
    p.set_defaults(func=cmd_bio_estimate)
    p = bsub.add_parser("run", help="run a campaign spec and print the ranked results")
    bio_args(p)
    p.add_argument("spec")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_bio_run)
    p = bsub.add_parser("verify", help="replay a finished job and check the output is identical")
    bio_args(p)
    p.add_argument("job")
    p.set_defaults(func=cmd_bio_verify)
    p = bsub.add_parser("tune", help="fit a re-scorer on assay outcomes for a finished job")
    bio_args(p)
    p.add_argument("job")
    p.add_argument("assay", help="CSV with an id column and either active (0/1) or value")
    p.add_argument("--threshold", type=float, help="value at which a candidate counts as active")
    p.add_argument("--lower-is-active", action="store_true", help="smaller values are better, e.g. IC50")
    p.add_argument("--model-id")
    p.set_defaults(func=cmd_bio_tune)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
