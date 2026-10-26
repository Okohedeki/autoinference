"""KV and state allocator part: page bookkeeping and the cache tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..registry import Manifest, register

POLICY = Manifest(kind="policy", lossless_class="bit-exact")


@register("kv-allocator", "paged", POLICY)
class PagedAllocator:
    """Fixed-size pages with reference counts.

    A page is free when its count drops to zero. A prefix cache may hold its own
    reference on a page; when free pages run out the allocator asks the
    ``evictor`` to release some before giving up.
    """

    def __init__(self, num_pages: int, page_size: int) -> None:
        if num_pages <= 0 or page_size <= 0:
            raise ValueError("num_pages and page_size must be positive")
        self.num_pages = num_pages
        self.page_size = page_size
        self._refs = [0] * num_pages
        self._free = list(range(num_pages - 1, -1, -1))  # pop() hands out low ids first
        self.evictor: Callable[[int], int] | None = None

    def num_free(self) -> int:
        return len(self._free)

    def pages_needed(self, num_tokens: int, have: int) -> int:
        return max(0, -(-num_tokens // self.page_size) - have)

    def allocate(self, n: int) -> list[int] | None:
        if n > len(self._free) and self.evictor is not None:
            self.evictor(n - len(self._free))
        if n > len(self._free):
            return None
        pages = [self._free.pop() for _ in range(n)]
        for p in pages:
            self._refs[p] = 1
        return pages

    def incref(self, pages: list[int]) -> None:
        for p in pages:
            if self._refs[p] <= 0:
                raise RuntimeError(f"incref on free page {p}")
            self._refs[p] += 1

    def free(self, pages: list[int]) -> None:
        for p in pages:
            if self._refs[p] <= 0:
                raise RuntimeError(f"double free of page {p}")
            self._refs[p] -= 1
            if self._refs[p] == 0:
                self._free.append(p)

    def refcount(self, page: int) -> int:
        return self._refs[page]


@dataclass
class KVStorage:
    """Per-layer key and value pages: [num_pages, page_size, num_kv_heads, head_dim]."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    @classmethod
    def create(
        cls,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "KVStorage":
        shape = (num_pages, page_size, num_kv_heads, head_dim)
        return cls(
            keys=[torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)],
            values=[torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)],
        )

    @staticmethod
    def bytes_per_page(num_layers: int, page_size: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
        return 2 * num_layers * page_size * num_kv_heads * head_dim * torch.empty((), dtype=dtype).element_size()
