"""Weight loader part: read checkpoint files into tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

from ..registry import Manifest, register

MODEL_FILE_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.jinja"]


def resolve_model_dir(path_or_id: str) -> Path:
    """A local directory as is, otherwise a Hugging Face repo id fetched to the local cache."""
    path = Path(path_or_id).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_id, allow_patterns=MODEL_FILE_PATTERNS))


def read_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


@register("loader", "safetensors", Manifest(kind="config", lossless_class="bit-exact", deps=("safetensors",)))
class SafetensorsLoader:
    """Loads every tensor, casting floating-point ones to ``dtype``.

    ``device_for`` comes from the placement part and picks each tensor's device.
    """

    def __init__(self, dtype: torch.dtype, device_for: Callable[[str], torch.device]) -> None:
        self.dtype = dtype
        self.device_for = device_for

    def files(self, model_dir: Path) -> list[Path]:
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
            return [model_dir / n for n in names]
        files = sorted(model_dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {model_dir}")
        return files

    def load(self, model_dir: str | Path) -> dict[str, torch.Tensor]:
        model_dir = Path(model_dir)
        tensors: dict[str, torch.Tensor] = {}
        for file in self.files(model_dir):
            with safe_open(str(file), framework="pt", device="cpu") as f:
                for name in f.keys():
                    t = f.get_tensor(name)
                    if t.is_floating_point():
                        t = t.to(self.dtype)
                    tensors[name] = t.to(self.device_for(name))
        return tensors
