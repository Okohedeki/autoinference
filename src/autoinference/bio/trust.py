"""Checks that say how far to trust a prediction.

* ``check_pose``: physical sanity of a predicted complex (clashes, overlapping
  ligand atoms, a ligand that is not touching the protein).
* ``rank_agreement`` and ``consensus``: whether independent models agree.
* ``IsotonicCalibrator``: maps a model score to the observed hit rate, fitted on
  the customer's own experimental outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

WATER = {"HOH", "WAT", "DOD"}
CLASH = 2.0  # heavy-atom distance (Å) between ligand and protein that counts as a clash
OVERLAP = 0.9  # two ligand atoms closer than this (Å) cannot both be real
CONTACT = 6.0  # a bound ligand has some atom within this distance (Å) of the protein


@dataclass
class Atom:
    record: str
    name: str
    resname: str
    chain: str
    element: str
    xyz: tuple[float, float, float]


def parse_pdb(text: str) -> list[Atom]:
    atoms = []
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        atoms.append(Atom(rec, line[12:16].strip(), line[17:20].strip(), line[21:22],
                          element.upper(), (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    return atoms


def _dist(a: Atom, b: Atom) -> float:
    return math.dist(a.xyz, b.xyz)


@dataclass
class PoseReport:
    valid: bool
    issues: list[str] = field(default_factory=list)
    clashes: int = 0
    min_contact: float | None = None
    ligand_atoms: int = 0

    def to_dict(self) -> dict:
        return {"valid": self.valid, "issues": self.issues, "clashes": self.clashes,
                "min_contact": None if self.min_contact is None else round(self.min_contact, 3),
                "ligand_atoms": self.ligand_atoms}


def check_pose(pdb_text: str, ligand_chain: str | None = None) -> PoseReport:
    """Physical sanity of a protein-ligand complex.

    The ligand is ``ligand_chain`` when given, otherwise every non-water HETATM.
    Hydrogens are ignored.
    """
    atoms = [a for a in parse_pdb(pdb_text) if a.element != "H" and a.resname not in WATER]
    if ligand_chain is not None:
        ligand = [a for a in atoms if a.chain == ligand_chain]
    else:
        ligand = [a for a in atoms if a.record == "HETATM"]
    protein = [a for a in atoms if a not in ligand and a.record == "ATOM"]
    issues = []
    if not ligand:
        return PoseReport(False, ["no ligand atoms in the pose"])
    if not protein:
        return PoseReport(False, ["no protein atoms in the pose"], ligand_atoms=len(ligand))
    clashes, min_contact = 0, math.inf
    for la in ligand:
        for pa in protein:
            d = _dist(la, pa)
            min_contact = min(min_contact, d)
            if d < CLASH:
                clashes += 1
    if clashes:
        issues.append(f"{clashes} ligand-protein contacts closer than {CLASH} Å")
    if min_contact > CONTACT:
        issues.append(f"ligand is {min_contact:.1f} Å from the protein, so it is not bound")
    overlaps = sum(1 for i, a in enumerate(ligand) for b in ligand[i + 1:] if _dist(a, b) < OVERLAP)
    if overlaps:
        issues.append(f"{overlaps} pairs of ligand atoms overlap")
    return PoseReport(not issues, issues, clashes, min_contact, len(ligand))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2  # ties share their average rank
        i = j + 1
    return ranks


def rank_agreement(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Spearman correlation between two models' scores on the candidates they share."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return None
    ra, rb = _ranks([a[k] for k in shared]), _ranks([b[k] for k in shared])
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va and vb else None


def consensus(scores_by_backend: dict[str, dict[str, float]],
              top_fraction: float = 0.5) -> tuple[dict[str, dict], dict[str, float | None]]:
    """Per candidate, whether every backend puts it in its own top ``top_fraction``; plus the
    pairwise rank agreement between backends."""
    tops = {}
    for backend, scores in scores_by_backend.items():
        ranked = sorted(scores, key=lambda k: -scores[k])
        tops[backend] = set(ranked[:max(1, math.ceil(len(ranked) * top_fraction))])
    candidates = set().union(*scores_by_backend.values()) if scores_by_backend else set()
    per_candidate = {}
    for cid in sorted(candidates):
        votes = {b: cid in tops[b] for b in scores_by_backend if cid in scores_by_backend[b]}
        per_candidate[cid] = {"agree": len(votes) == len(scores_by_backend) and all(votes.values()),
                              "top_votes": votes}
    names = list(scores_by_backend)
    agreement = {f"{x}~{y}": rank_agreement(scores_by_backend[x], scores_by_backend[y])
                 for i, x in enumerate(names) for y in names[i + 1:]}
    return per_candidate, agreement


class IsotonicCalibrator:
    """Monotone map from score to probability of an experimental hit (pool adjacent violators)."""

    def __init__(self, xs: list[float] | None = None, ys: list[float] | None = None) -> None:
        self.xs, self.ys = xs or [], ys or []

    def fit(self, scores: list[float], labels: list[int]) -> "IsotonicCalibrator":
        if len(scores) != len(labels) or not scores:
            raise ValueError("calibration needs matching, non-empty scores and labels")
        pairs = sorted(zip(scores, labels))
        blocks: list[list[float]] = []  # [sum of labels, count, max score]
        for s, y in pairs:
            blocks.append([float(y), 1.0, s])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                y2, n2, s2 = blocks.pop()
                blocks[-1][0] += y2
                blocks[-1][1] += n2
                blocks[-1][2] = s2
        self.xs = [b[2] for b in blocks]
        self.ys = [b[0] / b[1] for b in blocks]
        return self

    def __call__(self, score: float) -> float:
        if not self.xs:
            raise ValueError("calibrator is not fitted")
        for x, y in zip(self.xs, self.ys):
            if score <= x:
                return y
        return self.ys[-1]

    def to_dict(self) -> dict:
        return {"xs": self.xs, "ys": self.ys}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        return cls(list(d["xs"]), list(d["ys"]))
