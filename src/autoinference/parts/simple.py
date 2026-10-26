"""Single implementations of the execution and control parts the MVP keeps trivial."""

from __future__ import annotations

from typing import Callable

import torch

from ..registry import Manifest, register
from ..types import ForwardInputs, Request

EXACT_POLICY = Manifest(kind="policy", lossless_class="bit-exact")


@register("placement", "single-device", EXACT_POLICY)
class SingleDevicePlacement:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def device_for(self, tensor_name: str) -> torch.device:
        return self.device


@register("executor", "eager", EXACT_POLICY)
class EagerExecutor:
    def run(self, forward: Callable[[ForwardInputs], torch.Tensor], inputs: ForwardInputs) -> torch.Tensor:
        with torch.inference_mode():
            return forward(inputs)


@register("overlap", "none", EXACT_POLICY)
class NoOverlap:
    enabled = False


@register("spec-planner", "off", EXACT_POLICY)
class SpecPlannerOff:
    def draft_length(self, request: Request) -> int:
        return 0


@register("spec-drafter", "off", EXACT_POLICY)
class SpecDrafterOff:
    def propose(self, requests: list[Request]) -> dict[int, list[int]]:
        return {}


@register("spec-verifier", "off", EXACT_POLICY)
class SpecVerifierOff:
    def verify(self, request: Request, draft: list[int], logits: torch.Tensor) -> list[int]:
        return []
