"""Adapting predictions to a customer's own experimental results.

``fit_rescorer`` learns, from assay outcomes, how to weigh the scores every
backend produced for the same candidates, then calibrates the combined score
to the observed hit rate. The fitted model is a small JSON file kept in the
customer's workspace; the assay data never leaves it.

This is the lightest form of tuning. Fine-tuning a backend's own weights on the
same data plugs in behind the same ``TunedModel`` interface: it produces a new
backend version plus a calibrator.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provenance import digest
from .trust import IsotonicCalibrator


def load_assay(path: str | Path, threshold: float | None = None, higher_is_active: bool = True) -> dict[str, int]:
    """Read assay outcomes from CSV with an ``id`` column and either ``active`` (0/1) or ``value``.

    With ``value``, a candidate is active when it passes ``threshold``.
    """
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("active", "") != "":
                labels[row["id"]] = int(float(row["active"]))
            elif row.get("value", "") != "":
                if threshold is None:
                    raise ValueError("a value column needs a threshold to decide what counts as active")
                v = float(row["value"])
                labels[row["id"]] = int(v >= threshold if higher_is_active else v <= threshold)
    return labels


def features_of(result: dict) -> dict[str, float]:
    """Flatten one campaign result's per-stage scores into ``backend.score`` features."""
    out = {}
    for backend, scores in result.get("stage_scores", {}).items():
        for k, v in scores.items():
            out[f"{backend}.{k}"] = float(v)
    return out


def auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@dataclass
class TunedModel:
    id: str
    features: list[str]
    means: list[float]
    stds: list[float]
    weights: list[float]
    bias: float
    calibrator: IsotonicCalibrator
    metrics: dict[str, Any]
    data_digest: str

    def raw(self, feats: dict[str, float]) -> float | None:
        if any(f not in feats for f in self.features):
            return None
        z = self.bias + sum(w * (feats[f] - m) / s for f, m, s, w in
                            zip(self.features, self.means, self.stds, self.weights))
        return 1 / (1 + math.exp(-z))

    def p_active(self, feats: dict[str, float]) -> float | None:
        r = self.raw(feats)
        return None if r is None else self.calibrator(r)

    def to_dict(self) -> dict:
        return {"id": self.id, "features": self.features, "means": self.means, "stds": self.stds,
                "weights": self.weights, "bias": self.bias, "calibrator": self.calibrator.to_dict(),
                "metrics": self.metrics, "data_digest": self.data_digest}

    @classmethod
    def from_dict(cls, d: dict) -> "TunedModel":
        return cls(d["id"], d["features"], d["means"], d["stds"], d["weights"], d["bias"],
                   IsotonicCalibrator.from_dict(d["calibrator"]), d["metrics"], d["data_digest"])

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, directory: str | Path, model_id: str) -> "TunedModel":
        return cls.from_dict(json.loads((Path(directory) / f"{model_id}.json").read_text()))


def _logistic(rows: list[list[float]], labels: list[int], l2: float = 0.1, steps: int = 500,
              lr: float = 0.5) -> tuple[list[float], float]:
    """Plain batch gradient descent, so the fit is deterministic and needs no extra packages."""
    w, b = [0.0] * len(rows[0]), 0.0
    n = len(rows)
    for _ in range(steps):
        gw, gb = [0.0] * len(w), 0.0
        for x, y in zip(rows, labels):
            p = 1 / (1 + math.exp(-(b + sum(wi * xi for wi, xi in zip(w, x)))))
            for j, xj in enumerate(x):
                gw[j] += (p - y) * xj
            gb += p - y
        w = [wi - lr * (g / n + l2 * wi) for wi, g in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def fit_rescorer(results: list[dict], labels: dict[str, int], model_id: str, folds: int = 5) -> TunedModel:
    """Fit a re-scorer on the campaign results that have an assay outcome.

    Only features every labelled candidate has are used, so candidates that
    reached the same stages are compared on the same footing. The reported
    metrics compare the best single model score with the re-scorer, both
    measured by cross-validated AUC.
    """
    labelled = [r for r in results if r["id"] in labels]
    if len(labelled) < 6:
        raise ValueError(f"need at least 6 candidates with assay outcomes, got {len(labelled)}")
    feats = [features_of(r) for r in labelled]
    names = sorted(set.intersection(*(set(f) for f in feats)))
    if not names:
        raise ValueError("the labelled candidates share no scores to learn from")
    ys = [labels[r["id"]] for r in labelled]
    if len(set(ys)) < 2:
        raise ValueError("assay outcomes need both actives and inactives")
    xs = [[f[n] for n in names] for f in feats]

    def standardise(rows):
        means = [sum(c) / len(c) for c in zip(*rows)]
        stds = [math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) or 1.0 for c, m in zip(zip(*rows), means)]
        return means, stds

    def fit(rows, ys_):
        means, stds = standardise(rows)
        z = [[(v - m) / s for v, m, s in zip(r, means, stds)] for r in rows]
        w, b = _logistic(z, ys_)
        return means, stds, w, b

    def score(model, row):
        means, stds, w, b = model
        return 1 / (1 + math.exp(-(b + sum(wi * (v - m) / s for wi, v, m, s in zip(w, row, means, stds)))))

    # Cross-validated scores, with folds assigned by a hash of the id so they do not depend on order.
    k = max(2, min(folds, len(labelled) // 3))
    fold_of = [int(digest(r["id"])[:8], 16) % k for r in labelled]
    cv = [0.0] * len(labelled)
    for f in range(k):
        train = [i for i in range(len(labelled)) if fold_of[i] != f]
        if len({ys[i] for i in train}) < 2:
            return _finish(model_id, names, xs, ys, fit, score, labelled, labels, cv_auc=None)
        model = fit([xs[i] for i in train], [ys[i] for i in train])
        for i in range(len(labelled)):
            if fold_of[i] == f:
                cv[i] = score(model, xs[i])
    return _finish(model_id, names, xs, ys, fit, score, labelled, labels, cv_auc=auc(cv, ys))


def _finish(model_id, names, xs, ys, fit, score, labelled, labels, cv_auc) -> TunedModel:
    single = {n: auc([x[j] for x in xs], ys) for j, n in enumerate(names)}
    means, stds, w, b = model = fit(xs, ys)
    calibrator = IsotonicCalibrator().fit([score(model, x) for x in xs], ys)
    best = max((v for v in single.values() if v is not None), default=None)
    metrics = {"n": len(ys), "actives": sum(ys), "cv_auc": cv_auc, "best_single_auc": best,
               "single_auc": single}
    data = digest(sorted((r["id"], labels[r["id"]]) for r in labelled))
    return TunedModel(model_id, names, means, stds, w, b, calibrator, metrics, data)
