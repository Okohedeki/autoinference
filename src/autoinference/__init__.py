"""AutoInference: a slotted LLM inference engine whose every part is swappable."""

from .config import EngineConfig
from .engine import Engine, Generation
from .registry import REGISTRY, Manifest, register
from .types import SamplingParams

__all__ = ["REGISTRY", "Engine", "EngineConfig", "Generation", "Manifest", "SamplingParams", "register"]
__version__ = "0.1.0"
