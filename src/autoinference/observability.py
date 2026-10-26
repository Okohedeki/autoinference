"""Per-slot timers and counters. The search reads its signals from here."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class _Stat:
    calls: int = 0
    seconds: float = 0.0


class Profiler:
    """Accumulates wall time per label. Device work is synchronized when enabled,
    so timings are accurate but slower than an unprofiled run."""

    def __init__(self, enabled: bool = False, device: torch.device | None = None) -> None:
        self.enabled = enabled
        self.device = device
        self._stats: dict[str, _Stat] = defaultdict(_Stat)
        self.counters: dict[str, int] = defaultdict(int)

    def _sync(self) -> None:
        if self.device is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def span(self, label: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            stat = self._stats[label]
            stat.calls += 1
            stat.seconds += time.perf_counter() - start

    def count(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def reset(self) -> None:
        self._stats.clear()
        self.counters.clear()

    def to_dict(self) -> dict:
        return {
            "spans": {k: {"calls": v.calls, "seconds": v.seconds} for k, v in sorted(self._stats.items())},
            "counters": dict(sorted(self.counters.items())),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        if not self._stats and not self.counters:
            return "no profile data (enable with engine.profile = true)"
        top = [k for k in self._stats if "/" not in k]
        total = sum(self._stats[k].seconds for k in top) or 1e-12
        rows = sorted(self._stats.items(), key=lambda kv: -kv[1].seconds)
        width = max(len(k) for k, _ in rows) if rows else 10
        lines = [f"{'span'.ljust(width)}  {'calls':>7}  {'ms':>10}  {'share':>6}"]
        for label, stat in rows:
            share = stat.seconds / total if "/" not in label else float("nan")
            share_txt = f"{share:6.1%}" if share == share else "     -"
            lines.append(f"{label.ljust(width)}  {stat.calls:7d}  {stat.seconds * 1e3:10.2f}  {share_txt}")
        for name, value in sorted(self.counters.items()):
            lines.append(f"counter {name} = {value}")
        return "\n".join(lines)
