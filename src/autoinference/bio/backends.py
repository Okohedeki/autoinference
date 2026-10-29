"""Model backends: each one scores a batch of candidates against a target.

Backends register in the shared slot registry under ``bio_backend``, so
``autoinference parts`` lists them next to the engine's parts. Every backend
reports a fingerprint (name, version, weights) that goes into the run manifest,
and an estimate of GPU seconds per candidate that the cost estimator uses.

Two synthetic backends need no GPU or model weights. They score candidates
from a hidden, seeded activity function so funnels, trust checks and tuning can
be exercised and tested end to end. They are not models of anything real.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..registry import REGISTRY, Manifest, register
from .types import Candidate, Prediction, Target

SLOT = "bio_backend"


class Backend:
    name = "backend"
    version = "0"
    seconds_per_item = 1.0  # estimated GPU seconds per candidate, used before any timing is measured
    produces_structures = False

    def predict(self, target: Target, candidates: list[Candidate], seed: int) -> list[Prediction]:
        raise NotImplementedError

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}


def create_backend(name: str, **options: Any) -> Backend:
    return REGISTRY.create(SLOT, name, **options)


def backend_names() -> list[str]:
    return REGISTRY.names(SLOT)


# --- synthetic backends ---------------------------------------------------------------------------

def _unit(*parts: object) -> float:
    """A deterministic number in [0, 1) from the given parts."""
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def hidden_activity(target: Target, candidate: Candidate) -> float:
    """The synthetic ground truth both synthetic backends observe with different amounts of noise."""
    return _unit("activity", target.sequence, candidate.smiles or candidate.sequence)


class _Synthetic(Backend):
    noise = 0.0

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay  # seconds to sleep per candidate, to imitate a slow model in demos

    def _score(self, target: Target, c: Candidate, seed: int) -> float:
        noise = (_unit(self.name, "noise", seed, c.id, target.id) - 0.5) * 2 * self.noise
        return min(1.0, max(0.0, hidden_activity(target, c) + noise))

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "noise": self.noise}


@register(SLOT, "synthetic-screen", Manifest(kind="kernel", devices=("cpu",), lossless_class="bit-exact", deps=()))
class SyntheticScreen(_Synthetic):
    """Cheap and noisy, standing in for a fast first-pass filter."""

    name = "synthetic-screen"
    version = "1"
    seconds_per_item = 0.01
    noise = 0.35

    def predict(self, target, candidates, seed):
        out = []
        for c in candidates:
            if self.delay:
                time.sleep(self.delay)
            out.append(Prediction(c.id, self.name, {"score": self._score(target, c, seed)}))
        return out


@register(SLOT, "synthetic-pose", Manifest(kind="kernel", devices=("cpu",), lossless_class="bit-exact", deps=()))
class SyntheticPose(_Synthetic):
    """Expensive and accurate, standing in for a co-folding model. Emits a small PDB pose.

    About one pose in eight is deliberately broken (a clash or a ligand far from the
    protein) so the validity checks have something to catch.
    """

    name = "synthetic-pose"
    version = "1"
    seconds_per_item = 20.0
    noise = 0.08
    produces_structures = True

    def predict(self, target, candidates, seed):
        out = []
        for c in candidates:
            if self.delay:
                time.sleep(self.delay)
            score = self._score(target, c, seed)
            fault = _unit("fault", seed, c.id)
            kind = "clash" if fault < 0.0625 else "unbound" if fault < 0.125 else None
            confidence = round(0.5 + 0.5 * _unit("conf", seed, c.id), 6)
            out.append(Prediction(c.id, self.name, {"score": score, "confidence": confidence},
                                  structure=_synthetic_pose(c.id, seed, kind)))
        return out


def _synthetic_pose(cid: str, seed: int, fault: str | None) -> str:
    lines = []
    serial = 1
    for i in range(30):  # a helix-like CA trace
        x, y, z = 2.3 * math.cos(i * 1.75), 2.3 * math.sin(i * 1.75), 1.5 * i
        lines.append(_pdb_atom("ATOM", serial, "CA", "ALA", "A", i + 1, x, y, z, "C"))
        serial += 1
    cx, cy, cz = 7.0, 0.0, 22.0
    if fault == "clash":
        cx, cy, cz = 2.3 * math.cos(15 * 1.75), 2.3 * math.sin(15 * 1.75), 1.5 * 15 + 0.4
    elif fault == "unbound":
        cx, cy, cz = 40.0, 40.0, 40.0
    for j in range(8):
        dx, dy, dz = (_unit("lig", seed, cid, j, k) * 0.2 for k in range(3))  # keeps atoms ≥ 1 Å apart
        lines.append(_pdb_atom("HETATM", serial, f"C{j + 1}", "LIG", "B", 1,
                               cx + 1.4 * (j % 3) + dx, cy + 1.4 * (j // 3) + dy, cz + dz, "C"))
        serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _pdb_atom(rec, serial, name, res, chain, resi, x, y, z, element) -> str:
    return (f"{rec:<6s}{serial:>5d} {name:<4s} {res:>3s} {chain}{resi:>4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")


# --- RDKit drug-likeness filter -------------------------------------------------------------------

@register(SLOT, "rdkit-druglike", Manifest(kind="kernel", devices=("cpu",), lossless_class="bit-exact",
                                           deps=("rdkit",), licence="BSD-3-Clause"))
class RDKitDrugLike(Backend):
    """Scores small molecules by QED drug-likeness and drops ones RDKit cannot parse. CPU only."""

    name = "rdkit-druglike"
    seconds_per_item = 0.001

    def __init__(self) -> None:
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem import QED
        except ImportError as exc:
            raise RuntimeError("rdkit-druglike needs RDKit: pip install rdkit") from exc
        self._chem, self._qed = Chem, QED
        self.version = rdkit.__version__

    def predict(self, target, candidates, seed):
        out = []
        for c in candidates:
            mol = self._chem.MolFromSmiles(c.smiles) if c.smiles else None
            score = float(self._qed.qed(mol)) if mol is not None else 0.0
            out.append(Prediction(c.id, self.name, {"score": score}, extras={"parsed": mol is not None}))
        return out


# --- Boltz-2 --------------------------------------------------------------------------------------

@register(SLOT, "boltz2", Manifest(kind="kernel", devices=("cuda", "cpu"), lossless_class="tolerance-exact",
                                   deps=("boltz",), provenance="Boltz (MIT licence)", licence="MIT"))
class Boltz2(Backend):
    """Runs Boltz-2 co-folding and affinity prediction through the ``boltz`` command line.

    The command can live in a separate environment: set ``command`` (or the
    ``AUTOINFERENCE_BOLTZ`` environment variable) to its full invocation. All
    candidates in a batch go to one ``boltz predict`` call, so the model loads
    once per batch. The target's alignment is computed on the first batch and
    reused for every later one, since it depends only on the target.

    With ``wsl=True`` the command runs inside WSL (for example
    ``wsl -d Ubuntu -e /root/boltzenv/bin/boltz``) and every path passed to it is
    translated to its ``/mnt/<drive>`` form. Boltz's data files cannot be
    unpacked on native Windows, so this is the way to run it from Windows.

    Scores: ``score`` is Boltz's binder probability, ``affinity`` its predicted
    affinity value (lower binds tighter) and ``confidence`` its complex confidence.
    """

    name = "boltz2"
    seconds_per_item = 60.0
    produces_structures = True

    def __init__(self, command: str | None = None, workdir: str | None = None, use_msa_server: bool = True,
                 diffusion_samples: int = 1, extra_args: str = "", cache: str | None = None,
                 wsl: bool = False, version: str | None = None) -> None:
        cmd = command or os.environ.get("AUTOINFERENCE_BOLTZ") or shutil.which("boltz")
        if not cmd:
            raise RuntimeError("boltz2 needs the boltz command: pip install boltz, or set AUTOINFERENCE_BOLTZ")
        self.command = _split(cmd)
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="boltz-"))
        self.use_msa_server = use_msa_server
        self.diffusion_samples = diffusion_samples
        self.extra_args = _split(extra_args)
        self.cache = cache
        self.wsl = wsl
        self._msa: dict[str, str] = {}
        self.version = version or self._version()

    def _path(self, path: str | Path) -> str:
        return _wsl_path(path) if self.wsl else str(path)

    def _version(self) -> str:
        try:
            from importlib.metadata import version

            return version("boltz")
        except Exception:
            return "unknown"  # boltz lives in another environment

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "diffusion_samples": self.diffusion_samples}

    def predict(self, target, candidates, seed):
        if not candidates:
            return []
        batch = self.workdir / f"batch-{int(time.time() * 1000)}"
        inputs, outdir = batch / "inputs", batch / "out"
        inputs.mkdir(parents=True)
        msa = target.msa_path or self._msa.get(target.sequence)
        for c in candidates:
            (inputs / f"{_safe(c.id)}.yaml").write_text(_boltz_yaml(target, c, msa and self._path(msa)))
        args = [*self.command, "predict", self._path(inputs), "--out_dir", self._path(outdir), "--seed", str(seed),
                "--output_format", "pdb", "--diffusion_samples", str(self.diffusion_samples), *self.extra_args]
        if msa is None and self.use_msa_server:
            args.append("--use_msa_server")
        if self.cache:
            args += ["--cache", self._path(self.cache)]
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"boltz predict failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
        results = next(outdir.glob("boltz_results_*"), outdir)
        if msa is None:
            found = sorted((results / "msa").glob("*.csv"))
            if found:
                self._msa[target.sequence] = str(found[0])
        return [self._read(results / "predictions" / _safe(c.id), c) for c in candidates]

    def _read(self, pred_dir: Path, c: Candidate) -> Prediction:
        stem = _safe(c.id)
        conf = _load_json(pred_dir / f"confidence_{stem}_model_0.json")
        aff = _load_json(pred_dir / f"affinity_{stem}.json")
        pose = pred_dir / f"{stem}_model_0.pdb"
        scores = {"confidence": float(conf.get("confidence_score", 0.0))}
        if "iptm" in conf:
            scores["iptm"] = float(conf["iptm"])
        if aff:
            scores["score"] = float(aff.get("affinity_probability_binary", 0.0))
            scores["affinity"] = float(aff.get("affinity_pred_value", 0.0))
        else:
            scores["score"] = scores.get("iptm", scores["confidence"])
        return Prediction(c.id, self.name, scores, structure=pose.read_text() if pose.exists() else None)


def _split(command: str) -> list[str]:
    """Split a command line; on Windows keep backslashes and drop the quotes around paths."""
    if os.name != "nt":
        return shlex.split(command)
    return [t[1:-1] if len(t) > 1 and t[0] == t[-1] == '"' else t for t in shlex.split(command, posix=False)]


def _wsl_path(path: str | Path) -> str:
    """A Windows drive path becomes its ``/mnt/<drive>/...`` form; POSIX paths pass through."""
    text = str(Path(path).resolve()) if os.name == "nt" else str(path)
    if len(text) > 1 and text[1] == ":":
        return f"/mnt/{text[0].lower()}" + text[2:].replace("\\", "/")
    return text


def _safe(cid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cid)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _boltz_yaml(target: Target, c: Candidate, msa: str | None) -> str:
    protein = f"  - protein:\n      id: A\n      sequence: {target.sequence}\n"
    if msa:
        protein += f"      msa: {json.dumps(str(msa))}\n"
    if c.smiles is not None:
        other = f"  - ligand:\n      id: B\n      smiles: {json.dumps(c.smiles)}\n"
        props = "properties:\n  - affinity:\n      binder: B\n"
    else:
        other = f"  - protein:\n      id: B\n      sequence: {c.sequence}\n"
        if msa:
            other += "      msa: empty\n"
        props = ""
    return "version: 1\nsequences:\n" + protein + other + props
