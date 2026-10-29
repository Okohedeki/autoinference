"""Campaigns: multi-stage funnels that spend expensive models only on promising candidates.

Each stage runs one backend on the candidates that survived the previous
stage, checks any poses it produced, then keeps the best by a chosen score.
After the last stage, optional consensus backends re-score the survivors and a
tuned model (fitted on the customer's assay data) adds a calibrated
probability of being active. Every run returns a manifest that is enough to
replay it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..registry import REGISTRY
from .backends import SLOT, Backend, create_backend
from .provenance import digest, environment, make_deterministic, output_digest
from .trust import check_pose, consensus
from .tuning import TunedModel, features_of
from .types import CampaignSpec, Candidate, Prediction

Progress = Callable[[str, int, int], None]


def _seconds_per_item(name: str, measured: dict[str, float] | None) -> tuple[float, str]:
    if measured and name in measured:
        return measured[name], "measured"
    return REGISTRY.get(SLOT, name).factory.seconds_per_item, "default"


def estimate(spec: CampaignSpec, gpu_hour_price: float, measured: dict[str, float] | None = None) -> dict:
    """Upper-bound GPU time and cost before anything runs, stage by stage."""
    n = len(spec.candidates)
    stages, total = [], 0.0
    for stage in spec.stages:
        per, source = _seconds_per_item(stage.backend, measured)
        secs = n * per
        total += secs
        kept = stage.survivors(n)
        stages.append({"backend": stage.backend, "candidates": n, "kept_at_most": kept,
                       "seconds_per_item": per, "seconds_source": source, "gpu_seconds": secs})
        n = kept
    for name in spec.consensus:
        per, source = _seconds_per_item(name, measured)
        total += n * per
        stages.append({"backend": name, "candidates": n, "consensus": True, "seconds_per_item": per,
                       "seconds_source": source, "gpu_seconds": n * per})
    flat = len(spec.candidates) * max(_seconds_per_item(s.backend, measured)[0] for s in spec.stages)
    return {"stages": stages, "gpu_seconds": total, "gpu_hours": total / 3600,
            "cost": round(total / 3600 * gpu_hour_price, 4), "gpu_hour_price": gpu_hour_price,
            "unfunnelled_gpu_seconds": flat,
            "funnel_saving": round(1 - total / flat, 4) if flat else 0.0}


@dataclass
class CampaignResult:
    results: list[dict]
    manifest: dict[str, Any]
    stage_reports: list[dict] = field(default_factory=list)
    rank_agreement: dict[str, float | None] = field(default_factory=dict)
    dropped: list[dict] = field(default_factory=list)  # candidates filtered out, with the scores they got

    def to_dict(self) -> dict:
        return {"results": self.results, "dropped": self.dropped, "manifest": self.manifest,
                "stages": self.stage_reports, "rank_agreement": self.rank_agreement}


def _batched(items: list[Candidate], size: int | None):
    if not size:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_campaign(spec: CampaignSpec, backends: dict[str, Backend] | None = None,
                 tuned: TunedModel | None = None, progress: Progress | None = None) -> CampaignResult:
    """Run a campaign. ``backends`` supplies pre-built backends by name; others are created from
    each stage's options."""
    make_deterministic(spec.seed)
    backends = dict(backends or {})
    by_id = {c.id: c for c in spec.candidates}
    records: dict[str, dict] = {c.id: {"id": c.id, "stage_scores": {}, "stage_reached": 0, "pose": None}
                                for c in spec.candidates}
    survivors = list(spec.candidates)
    reports = []

    def backend_for(name: str, options: dict) -> Backend:
        if name not in backends:
            backends[name] = create_backend(name, **options)
        return backends[name]

    for index, stage in enumerate(spec.stages, start=1):
        options = dict(stage.options)
        batch_size = options.pop("batch_size", None)
        backend = backend_for(stage.backend, options)
        start = time.perf_counter()
        preds: list[Prediction] = []
        for batch in _batched(survivors, batch_size):
            preds += backend.predict(spec.target, batch, spec.seed)
            if progress:
                progress(stage.backend, len(preds), len(survivors))
        elapsed = time.perf_counter() - start
        invalid = 0
        for p in preds:
            rec = records[p.candidate_id]
            rec["stage_scores"][stage.backend] = p.scores
            rec["stage_reached"] = index
            if p.structure is not None:
                rec["pose"] = check_pose(p.structure).to_dict()
                rec["structure"] = p.structure
                invalid += not rec["pose"]["valid"]
        pool = [p for p in preds if not (stage.drop_invalid and records[p.candidate_id]["pose"]
                                         and not records[p.candidate_id]["pose"]["valid"])]
        if stage.min_score is not None:
            pool = [p for p in pool if p.scores.get(stage.rank_by, float("-inf")) >= stage.min_score]
        pool.sort(key=lambda p: (-p.scores.get(stage.rank_by, float("-inf")), p.candidate_id))
        pool = pool[:stage.survivors(len(pool))]
        survivors = [by_id[p.candidate_id] for p in pool]
        reports.append({"backend": stage.backend, "scored": len(preds), "kept": len(survivors),
                        "invalid_poses": invalid, "seconds": round(elapsed, 4),
                        "seconds_per_item": elapsed / len(preds) if preds else None})

    agreement: dict[str, float | None] = {}
    final_ids = [c.id for c in survivors]
    if spec.consensus and survivors:
        last = spec.stages[-1]
        scores = {last.backend: {cid: records[cid]["stage_scores"][last.backend][last.rank_by] for cid in final_ids}}
        for name in spec.consensus:
            for p in backend_for(name, {}).predict(spec.target, survivors, spec.seed):
                records[p.candidate_id]["stage_scores"][f"consensus:{name}"] = p.scores
                scores.setdefault(name, {})[p.candidate_id] = p.scores.get(last.rank_by, p.scores["score"])
        per_candidate, agreement = consensus(scores)
        for cid in final_ids:
            records[cid]["consensus"] = per_candidate[cid]

    if tuned is not None:
        for rec in records.values():
            rec["p_active"] = tuned.p_active(features_of(rec))

    results = []
    for cid in final_ids:
        rec = records[cid]
        rec["scores"] = {f"{b}.{k}": v for b, s in rec["stage_scores"].items() for k, v in s.items()}
        results.append(rec)
    if tuned is not None:
        results.sort(key=lambda r: (-(r["p_active"] if r["p_active"] is not None else -1), r["id"]))
    for rank, rec in enumerate(results, start=1):
        rec["rank"] = rank
    kept = set(final_ids)
    dropped = sorted((records[cid] for cid in records if cid not in kept),
                     key=lambda r: (-r["stage_reached"], r["id"]))
    for rec in dropped:
        rec["scores"] = {f"{b}.{k}": v for b, s in rec["stage_scores"].items() for k, v in s.items()}

    manifest = {
        "spec_digest": digest(spec.to_dict()),
        "seed": spec.seed,
        "backends": {name: b.fingerprint() for name, b in backends.items()},
        "tuned_model": None if tuned is None else {"id": tuned.id, "data_digest": tuned.data_digest},
        "environment": environment(),
        "deterministic": True,
        "output_digest": output_digest(results),
    }
    return CampaignResult(results, manifest, reports, agreement, dropped)
