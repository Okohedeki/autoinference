"""Model architectures: importing this package registers every built-in one."""

from . import qwen3  # noqa: F401
from .registry import get_arch, register_arch, supported_archs

__all__ = ["get_arch", "register_arch", "supported_archs"]
