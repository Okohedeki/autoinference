"""The platform: one API over a catalog of specialised models, run on serverless GPUs.

The control plane (this package) is small and CPU-only. It owns the catalog,
projects and API keys, the request queue, files, usage and the audit log. Model
code runs in the data plane: one container per model on Modal or RunPod, or
in-process in local mode for development and tests.

Design: ``reports/Platform spec.md``.
"""

from .catalog import Catalog, ModelSpec
from .config import Settings
from .service import Platform

__all__ = ["Catalog", "ModelSpec", "Platform", "Settings"]
