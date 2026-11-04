"""Boltz-2 on the model contract. Reuses the bio backend that drives the ``boltz`` command."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def load(spec):
    from autoinference.bio.backends import Boltz2

    cache = os.environ.get("BOLTZ_CACHE")
    workdir = Path(os.environ.get("BOLTZ_WORKDIR", "/tmp/boltz-work"))
    workdir.mkdir(parents=True, exist_ok=True)
    return Boltz2(workdir=str(workdir), cache=cache, msa_cache=str(workdir / "msa-cache"))


def run(inputs: dict, ctx, model) -> dict:
    from autoinference.bio.trust import check_pose
    from autoinference.bio.types import Candidate, Target

    if bool(inputs.get("ligand")) == bool(inputs.get("binder")):
        raise ValueError("give exactly one of ligand or binder")
    model.diffusion_samples = int(inputs.get("diffusion_samples", 1))
    msa = inputs.get("msa")
    target = Target("target", inputs["protein"], str(msa) if msa else None)
    candidate = Candidate("c", smiles=inputs.get("ligand"), sequence=inputs.get("binder"))
    ctx.log("running boltz predict")
    pred = model.predict(target, [candidate], int(inputs.get("seed", 0)))[0]
    outputs = dict(pred.scores)
    if pred.structure:
        path = ctx.path("structure.pdb")
        path.write_text(pred.structure)
        outputs["structure"] = path
        outputs["checks"] = check_pose(pred.structure).to_dict()
    cached = model._cached_msa(target.sequence)
    if cached.exists():
        out_msa = ctx.path("msa.csv")
        shutil.copyfile(cached, out_msa)
        outputs["msa"] = out_msa
    return {"outputs": outputs, "units": 1}
