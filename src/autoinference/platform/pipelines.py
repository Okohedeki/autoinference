"""Pipelines: graphs of model calls, run as one request, controllable step by step.

A pipeline spec is JSON::

    {
      "inputs": {"target": "MKT...", "ligands": ["CCO", "c1ccccc1"]},
      "steps": [
        {"id": "fold", "model": "biology/boltz-2", "inputs": {"protein": "$inputs.target"}},
        {"id": "dock", "model": "biology/diffdock", "map": "$inputs.ligands",
         "inputs": {"protein": "$fold.outputs.structure", "ligand": "$item"},
         "keep": {"top": 20, "by": "score"}},
        {"id": "affinity", "model": "biology/boltz-2", "map": "$dock.items",
         "inputs": {"protein": "$inputs.target", "ligand": "$item.inputs.ligand"},
         "approve": true}
      ],
      "outputs": {"shortlist": "$affinity.items"}
    }

Each step is one model call, or with ``map`` one call per item of a list,
fanned out across the executor. ``keep`` filters a mapped step's items by
their outputs (the funnel). ``approve`` pauses the run before the step until
it is approved. ``when`` skips a step. ``retries``, ``timeout``, ``options``
(provider, GPU) and ``checks`` thresholds are per step. Each call is a normal
request with its own id, status, logs and cost, and the pipeline run records
which request belongs to which step.

References: ``$inputs.<name>``, ``$<step>.outputs.<name>`` (single steps),
``$<step>.items`` (kept items of a mapped step, each ``{inputs, outputs,
request_id}``), ``$item`` and ``$item.<path>`` inside a mapped step, and
``$index``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .catalog import ModelSpec

REF = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)((?:\.[A-Za-z0-9_-]+)*)$")
STEP_ID = re.compile(r"^[a-z][a-z0-9_-]{0,40}$")
RESERVED = {"inputs", "item", "index", "outputs", "steps"}


class PipelineError(ValueError):
    pass


@dataclass(frozen=True)
class KeepRule:
    top: int | None = None
    fraction: float | None = None
    by: str = "score"
    descending: bool = True
    min: dict[str, float] = field(default_factory=dict)      # output name -> minimum
    max: dict[str, float] = field(default_factory=dict)
    valid_only: bool = False                                 # drop items whose checks failed

    def apply(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = []
        for it in items:
            out = it.get("outputs", {})
            if self.valid_only and it.get("checks") and not it["checks"].get("valid", True):
                continue
            if any(_num(out.get(k)) is None or _num(out.get(k)) < v for k, v in self.min.items()):
                continue
            if any(_num(out.get(k)) is None or _num(out.get(k)) > v for k, v in self.max.items()):
                continue
            kept.append(it)
        if self.top is not None or self.fraction is not None:
            kept.sort(key=lambda it: (_num(it.get("outputs", {}).get(self.by)) is None,
                                      -(_num(it.get("outputs", {}).get(self.by)) or 0.0) if self.descending
                                      else (_num(it.get("outputs", {}).get(self.by)) or 0.0),
                                      it.get("index", 0)))
            n = len(kept)
            if self.fraction is not None:
                n = min(n, max(1, int(round(len(kept) * self.fraction))))
            if self.top is not None:
                n = min(n, self.top)
            kept = kept[:n]
        return kept


@dataclass(frozen=True)
class Step:
    id: str
    model: str
    inputs: dict[str, Any]
    map: str | None = None
    keep: KeepRule | None = None
    approve: bool = False
    when: str | None = None
    retries: int = 0
    timeout: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def references(self) -> set[str]:
        found: set[str] = set()

        def walk(v: Any) -> None:
            if isinstance(v, str):
                m = REF.match(v)
                if m and m.group(1) not in RESERVED:
                    found.add(m.group(1))
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)

        walk(self.inputs)
        for text in (self.map, self.when):
            if text:
                walk(text)
        return found


@dataclass(frozen=True)
class PipelineSpec:
    steps: tuple[Step, ...]
    outputs: dict[str, Any]
    name: str = "pipeline"
    description: str = ""

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)


def parse_pipeline(data: dict[str, Any]) -> PipelineSpec:
    if not isinstance(data.get("steps"), list) or not data["steps"]:
        raise PipelineError("a pipeline needs a non-empty list of steps")
    steps: list[Step] = []
    seen: set[str] = set()
    for i, raw in enumerate(data["steps"]):
        sid = raw.get("id") or f"step{i + 1}"
        if not STEP_ID.match(sid) or sid in RESERVED:
            raise PipelineError(f"bad step id {sid!r}")
        if sid in seen:
            raise PipelineError(f"duplicate step id {sid!r}")
        if not raw.get("model"):
            raise PipelineError(f"step {sid}: model is required")
        keep = None
        if raw.get("keep") is not None:
            k = raw["keep"]
            keep = KeepRule(top=k.get("top"), fraction=k.get("fraction"), by=k.get("by", "score"),
                            descending=bool(k.get("descending", True)), min=dict(k.get("min", {})),
                            max=dict(k.get("max", {})), valid_only=bool(k.get("valid_only", False)))
            if raw.get("map") is None:
                raise PipelineError(f"step {sid}: keep needs map")
        step = Step(id=sid, model=raw["model"], inputs=dict(raw.get("inputs", {})), map=raw.get("map"), keep=keep,
                    approve=bool(raw.get("approve", False)), when=raw.get("when"), retries=int(raw.get("retries", 0)),
                    timeout=raw.get("timeout"), options=dict(raw.get("options", {})),
                    checks=dict(raw.get("checks", {})), seed=raw.get("seed"))
        for ref in step.references():
            if ref not in seen:
                raise PipelineError(f"step {sid} refers to ${ref}, which is not an earlier step")
        steps.append(step)
        seen.add(sid)
    return PipelineSpec(tuple(steps), dict(data.get("outputs", {})), data.get("name", "pipeline"),
                        data.get("description", ""))


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _lookup(root: Any, path: str) -> Any:
    cur = root
    for part in [p for p in path.split(".") if p]:
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise PipelineError(f"unknown field {part!r} in reference")
            cur = cur[part]
        else:
            raise PipelineError(f"cannot index into {type(cur).__name__} with {part!r}")
    return cur


def resolve(value: Any, scope: dict[str, Any]) -> Any:
    """Replace ``$...`` references in a value with their contents."""
    if isinstance(value, str):
        m = REF.match(value)
        if m:
            name, path = m.group(1), m.group(2)
            if name not in scope:
                raise PipelineError(f"unknown reference ${name}")
            return _lookup(scope[name], path)
        return value
    if isinstance(value, dict):
        return {k: resolve(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, scope) for v in value]
    return value


def truthy(value: Any) -> bool:
    return bool(value) and value not in ("false", "0", "no")


def step_scope(state: dict[str, Any]) -> dict[str, Any]:
    """What references can see: pipeline inputs and finished steps."""
    scope: dict[str, Any] = {"inputs": state["inputs"]}
    for sid, result in state["steps"].items():
        if result.get("status") in ("succeeded", "skipped"):
            scope[sid] = {"outputs": result.get("outputs", {}), "items": result.get("items", []),
                          "all_items": result.get("all_items", []), "request_id": result.get("request_id")}
    return scope


def plan_calls(step: Step, state: dict[str, Any]) -> list[dict[str, Any]]:
    """The model calls a step makes, with their resolved inputs."""
    scope = step_scope(state)
    if step.when is not None and not truthy(resolve(step.when, scope)):
        return []
    if step.map is None:
        return [{"index": 0, "inputs": resolve(step.inputs, scope)}]
    items = resolve(step.map, scope)
    if not isinstance(items, list):
        raise PipelineError(f"step {step.id}: map must refer to a list")
    calls = []
    for i, item in enumerate(items):
        calls.append({"index": i, "inputs": resolve(step.inputs, {**scope, "item": item, "index": i}),
                      "item": item})
    return calls


def estimate_pipeline(spec: PipelineSpec, inputs: dict[str, Any], catalog) -> dict[str, Any]:
    """Count calls per step from list lengths and keep rules. Unknown counts fall back to 1."""
    counts: dict[str, int] = {}
    stages, total = [], 0.0
    for step in spec.steps:
        n = 1
        if step.map:
            m = REF.match(step.map)
            if m and m.group(1) == "inputs":
                try:
                    n = len(_lookup(inputs, m.group(2)))
                except Exception:  # noqa: BLE001
                    n = 1
            elif m and m.group(1) in counts:
                n = counts[m.group(1)]
        model: ModelSpec = catalog.get(step.model)
        price = n * model.price_per_unit
        total += price
        kept = n
        if step.keep:
            if step.keep.fraction is not None:
                kept = max(1, int(round(n * step.keep.fraction)))
            if step.keep.top is not None:
                kept = min(kept, step.keep.top)
        counts[step.id] = kept
        stages.append({"step": step.id, "model": model.id, "calls": n, "kept_at_most": kept,
                       "unit": model.unit, "price": round(price, 6), "approve": step.approve})
    return {"steps": stages, "price": round(total, 6), "currency": "USD"}
