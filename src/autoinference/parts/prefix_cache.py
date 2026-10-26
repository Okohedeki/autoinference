"""Prefix cache part: find, share and evict reusable prompt prefixes."""

from __future__ import annotations

from collections import OrderedDict

from ..registry import Manifest, register
from ..types import Request
from .kv import PagedAllocator

POLICY = Manifest(kind="policy", lossless_class="tolerance-exact")


@register("prefix-cache", "none", Manifest(kind="policy", lossless_class="bit-exact"))
class NoPrefixCache:
    def __init__(self, allocator: PagedAllocator) -> None:
        self.allocator = allocator

    def match(self, token_ids: list[int]) -> tuple[list[int], int]:
        return [], 0

    def insert(self, request: Request) -> None:
        pass

    def evict(self, n: int) -> int:
        return 0

    def forget(self, request: Request) -> None:
        pass


@register("prefix-cache", "block-hash", POLICY)
class BlockHashPrefixCache:
    """Caches full pages keyed by a hash chained over every token before them.

    The cache holds one reference on each cached page, so pages outlive the
    request that computed them. Pages only the cache references are evicted
    least-recently-used first.
    """

    def __init__(self, allocator: PagedAllocator) -> None:
        self.allocator = allocator
        self.page_size = allocator.page_size
        self._by_hash: OrderedDict[int, int] = OrderedDict()  # hash -> page, in LRU order
        self._inserted: dict[int, tuple[int, int | None]] = {}  # request id -> (blocks inserted, last hash)
        allocator.evictor = self.evict

    def _block_hash(self, parent: int | None, tokens: list[int]) -> int:
        return hash((parent, tuple(tokens)))

    def match(self, token_ids: list[int]) -> tuple[list[int], int]:
        # Keep at least one token uncached so the step that follows produces logits.
        max_blocks = (len(token_ids) - 1) // self.page_size
        pages: list[int] = []
        parent: int | None = None
        for b in range(max_blocks):
            h = self._block_hash(parent, token_ids[b * self.page_size:(b + 1) * self.page_size])
            page = self._by_hash.get(h)
            if page is None:
                break
            self._by_hash.move_to_end(h)
            pages.append(page)
            parent = h
        if pages:
            self.allocator.incref(pages)
        return pages, len(pages) * self.page_size

    def insert(self, request: Request) -> None:
        done, parent = self._inserted.get(request.request_id, (0, None))
        full = request.num_computed // self.page_size
        for b in range(done, full):
            tokens = request.token_ids[b * self.page_size:(b + 1) * self.page_size]
            h = self._block_hash(parent, tokens)
            if h not in self._by_hash:
                page = request.pages[b]
                self.allocator.incref([page])
                self._by_hash[h] = page
            parent = h
        self._inserted[request.request_id] = (max(done, full), parent)

    def forget(self, request: Request) -> None:
        self._inserted.pop(request.request_id, None)

    def evict(self, n: int) -> int:
        freed = 0
        for h in list(self._by_hash):
            if freed >= n:
                break
            page = self._by_hash[h]
            if self.allocator.refcount(page) == 1:
                del self._by_hash[h]
                self.allocator.free([page])
                freed += 1
        return freed

    @property
    def num_cached_pages(self) -> int:
        return len(self._by_hash)
