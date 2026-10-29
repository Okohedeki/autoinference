"""Text embeddings with an on-disk cache.

Defaults to Qwen3-Embedding-0.6B, which runs on CUDA, Apple MPS or CPU, so the pipeline works on
Windows, macOS and Linux. Pass a larger Qwen3-Embedding model (4B, 8B) for sharper neighborhoods. Vectors are L2-normalized, so a dot product is a cosine.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

LARGE_MODEL = "Qwen/Qwen3-Embedding-8B"
SMALL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIMS = 1024  # Qwen3 embeddings are Matryoshka-trained, so a truncated prefix is still a good vector
INSTRUCTION = "Instruct: Represent this AI inference product or business so similar offerings are close\nQuery: "


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_model(device: str) -> str:
    return SMALL_MODEL


class Embedder:
    def __init__(self, model_name: str | None = None, cache_dir: Path = Path("cache")):
        self.device = pick_device()
        self.model_name = model_name or default_model(self.device)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is None:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            self._tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            self._model = AutoModel.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()

    @torch.inference_mode()
    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self._load()
        out = []
        order = np.argsort([-len(t) for t in texts])  # similar lengths per batch, less padding
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            batch = self._tok(
                [INSTRUCTION + texts[i] for i in idx],
                padding=True, truncation=True, max_length=256, return_tensors="pt",
            ).to(self.device)
            hidden = self._model(**batch).last_hidden_state
            vecs = F.normalize(hidden[:, -1, :DIMS].float(), dim=-1)  # left padding: last token is real
            out.append((idx, vecs.cpu().numpy()))
        result = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, vecs in out:
            result[idx] = vecs
        return result

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed texts, reusing cached vectors for any text seen before with the same model."""
        stem = self.model_name.replace("/", "__")
        keys_file, vecs_file = self.cache_dir / f"{stem}.keys.json", self.cache_dir / f"{stem}.npy"
        keys, vecs = [], np.zeros((0, DIMS), dtype=np.float16)
        if keys_file.exists() and vecs_file.exists():
            keys, vecs = json.loads(keys_file.read_text()), np.load(vecs_file)
        key = lambda t: hashlib.sha1(t.encode()).hexdigest()
        index = {k: i for i, k in enumerate(keys)}
        missing = sorted({t for t in texts if key(t) not in index})
        if missing:
            vecs = np.concatenate([vecs, self._encode(missing, batch_size).astype(np.float16)])
            for t in missing:
                index[key(t)] = len(keys)
                keys.append(key(t))
            np.save(vecs_file, vecs)
            keys_file.write_text(json.dumps(keys))
        out = vecs[[index[key(t)] for t in texts]].astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)
