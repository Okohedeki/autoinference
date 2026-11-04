"""Demo handler: deterministic scores from a hash, so tests and pipelines have something to rank."""

from __future__ import annotations

import hashlib
from pathlib import Path


def load(spec):
    return {"version": spec.version}


def _unit(*parts: object) -> float:
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def run(inputs: dict, ctx, state) -> dict:
    target, molecule = inputs["target"], inputs["molecule"]
    bad = inputs.get("fail_if_contains")
    if bad and bad in molecule:
        raise RuntimeError(f"molecule contains {bad!r}")
    score = round(0.7 * _unit(target, molecule) + 0.3 * _unit(molecule), 4)
    qed = round(_unit("qed", molecule), 4)
    outputs = {"score": score, "qed": qed}
    report = inputs.get("report")
    if report is not None:
        outputs["report_size"] = Path(report).stat().st_size
    pose = ctx.path("pose.pdb")
    pose.write_text(f"REMARK demo pose for {molecule}\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    outputs["pose"] = pose
    ctx.log(f"scored {molecule}: {score}")
    return {"outputs": outputs, "units": 1}
