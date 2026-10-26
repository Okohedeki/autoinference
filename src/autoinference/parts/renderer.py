"""Renderer part: chat templates, tokenization and incremental detokenization."""

from __future__ import annotations

import json
from pathlib import Path

from ..registry import Manifest, register


@register("renderer", "hf-tokenizer", Manifest(kind="infrastructure", lossless_class="bit-exact", deps=("transformers",)))
class HFTokenizerRenderer:
    """Uses the model's own tokenizer files through ``transformers``."""

    def __init__(self, model_dir: str | Path) -> None:
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.eos_token_ids = self._eos_ids()

    def _eos_ids(self) -> tuple[int, ...]:
        ids: set[int] = set()
        gen = self.model_dir / "generation_config.json"
        if gen.exists():
            eos = json.loads(gen.read_text(encoding="utf-8")).get("eos_token_id")
            if isinstance(eos, int):
                ids.add(eos)
            elif isinstance(eos, list):
                ids.update(eos)
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return tuple(sorted(ids))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def apply_chat_template(self, messages: list[dict[str, str]], enable_thinking: bool = False) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        return self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class IncrementalDetokenizer:
    """Emits only the new text each time tokens are added.

    Text ending in an incomplete multi-byte character is held back until the
    next token completes it.
    """

    def __init__(self, renderer: HFTokenizerRenderer) -> None:
        self.renderer = renderer
        self.ids: list[int] = []
        self.emitted = ""

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.renderer.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta
