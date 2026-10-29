"""Hosted inference for biology models: funnel campaigns, trust checks, tuning on assay data,
reproducible runs, and an MCP server for agents."""

from . import backends  # noqa: F401  (registers the built-in backends)
from .campaign import CampaignResult, estimate, run_campaign
from .service import BioService
from .types import CampaignSpec, Candidate, Prediction, Stage, Target

__all__ = ["BioService", "CampaignResult", "CampaignSpec", "Candidate", "Prediction", "Stage", "Target",
           "estimate", "run_campaign"]
