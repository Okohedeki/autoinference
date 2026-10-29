"""Plain data types shared by the campaign runner, the backends and the service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Target:
    """The protein every candidate is scored against."""

    id: str
    sequence: str
    msa_path: str | None = None  # precomputed alignment; computed once and reused when absent


@dataclass(frozen=True)
class Candidate:
    """One thing to score: a small molecule (SMILES) or a protein binder (sequence)."""

    id: str
    smiles: str | None = None
    sequence: str | None = None

    def __post_init__(self) -> None:
        if (self.smiles is None) == (self.sequence is None):
            raise ValueError(f"candidate {self.id!r} needs exactly one of smiles or sequence")


@dataclass
class Prediction:
    """What one backend said about one candidate. Higher ``scores['score']`` is better."""

    candidate_id: str
    backend: str
    scores: dict[str, float]
    structure: str | None = None  # PDB text, when the backend produces a pose
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage:
    """One step of a funnel: run ``backend`` on the survivors, then keep the best of them."""

    backend: str
    rank_by: str = "score"
    keep_top: int | None = None
    keep_fraction: float | None = None
    min_score: float | None = None
    drop_invalid: bool = False  # drop candidates whose pose fails the validity checks
    options: dict[str, Any] = field(default_factory=dict)

    def survivors(self, n: int) -> int:
        """Upper bound on how many candidates leave this stage, used for cost estimates."""
        if self.keep_fraction is not None:
            n = max(1, int(round(n * self.keep_fraction))) if n else 0
        if self.keep_top is not None:
            n = min(n, self.keep_top)
        return n


@dataclass
class CampaignSpec:
    target: Target
    candidates: list[Candidate]
    stages: list[Stage]
    name: str = "campaign"
    seed: int = 0
    consensus: list[str] = field(default_factory=list)  # extra backends that re-score the final survivors
    tuned_model: str | None = None  # id of a model fitted on the customer's assay data

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a campaign needs at least one stage")
        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids must be unique")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CampaignSpec":
        return cls(
            target=Target(**d["target"]),
            candidates=[Candidate(**c) for c in d["candidates"]],
            stages=[Stage(**s) for s in d["stages"]],
            name=d.get("name", "campaign"),
            seed=int(d.get("seed", 0)),
            consensus=list(d.get("consensus", [])),
            tuned_model=d.get("tuned_model"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
