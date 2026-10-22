"""Thread-safe RAM staging ring buffer with late-bound verification.

While the user types, the syntactic gatekeeper emits closed phrases. Those
phrases are staged locally as read-only retrieval keys: nothing is sent to a
model, so staging costs zero LLM (Large Language Model) tokens. When the user
presses Enter, the final prompt is compared against every staged item and only
items at or above the similarity threshold are injected into the model context.
Everything else -- typically the abandoned half of a mid-sentence pivot -- is
silently evicted from RAM, which is what keeps wasted tokens at zero.

Time is injected through ``clock`` so tests and benchmarks can run on a virtual
timeline instead of sleeping.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, Optional, Tuple

DEFAULT_TTL_SECONDS = 30.0
DEFAULT_MAX_ITEMS = 64
DEFAULT_SIMILARITY_THRESHOLD = 0.75

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> str:
    """Lowercase and strip punctuation so casing and commas do not matter."""
    return " ".join(_WORD_RE.findall(text.lower()))


@dataclass(frozen=True)
class StagedItem:
    """One read-only context span waiting for late-bound verification."""

    item_id: int
    text: str
    kind: str
    source: str
    created_at: float
    ttl_seconds: float
    speculative_read_only: bool = True

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at


class StagingBuffer:
    """Bounded ring buffer of staged context spans, guarded by a re-entrant lock."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_items: int = DEFAULT_MAX_ITEMS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self._ttl = float(ttl_seconds)
        self._max_items = int(max_items)
        self._threshold = float(similarity_threshold)
        self._clock = clock or time.monotonic
        self._items: Deque[StagedItem] = deque()
        self._lock = threading.RLock()
        self._next_id = 0
        self._staged = 0
        self._expired = 0
        self._capacity_evictions = 0
        self._verification_evictions = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    @property
    def max_items(self) -> int:
        return self._max_items

    @property
    def similarity_threshold(self) -> float:
        return self._threshold

    def stage(
        self,
        text: str,
        kind: str = "NP",
        source: str = "gatekeeper",
        now: Optional[float] = None,
    ) -> Optional[StagedItem]:
        """Stage one span, ignoring blanks and already-staged duplicates."""
        item, _ = self._stage_one(text, kind, source, now)
        return item

    def stage_chunks(self, chunks: Iterable[object], source: str = "gatekeeper") -> int:
        """Stage every chunk exposing ``text`` and ``kind``; return the new count."""
        added = 0
        for chunk in chunks:
            _, created = self._stage_one(chunk.text, chunk.kind, source, None)
            added += 1 if created else 0
        return added

    def _stage_one(
        self,
        text: str,
        kind: str,
        source: str,
        now: Optional[float],
    ) -> Tuple[Optional[StagedItem], bool]:
        cleaned = text.strip()
        if not cleaned:
            return None, False
        with self._lock:
            current = self._clock() if now is None else now
            self._purge_locked(current)
            key = normalize(cleaned)
            for item in self._items:
                if normalize(item.text) == key:
                    return item, False
            if len(self._items) >= self._max_items:
                self._items.popleft()
                self._capacity_evictions += 1
            item = StagedItem(
                item_id=self._next_id,
                text=cleaned,
                kind=kind,
                source=source,
                created_at=current,
                ttl_seconds=self._ttl,
            )
            self._next_id += 1
            self._items.append(item)
            self._staged += 1
            return item, True

    def purge_expired(self, now: Optional[float] = None) -> int:
        """Drop items whose TTL elapsed; return how many were removed."""
        with self._lock:
            return self._purge_locked(self._clock() if now is None else now)

    def snapshot(self) -> Tuple[StagedItem, ...]:
        """Return a stable copy of all live staged items."""
        with self._lock:
            self._purge_locked(self._clock())
            return tuple(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "staged": self._staged,
                "expired": self._expired,
                "evicted_capacity": self._capacity_evictions,
                "evicted_verification": self._verification_evictions,
                "live": len(self._items),
                "max_items": self._max_items,
                "ttl_seconds": self._ttl,
                "similarity_threshold": self._threshold,
            }

    def _purge_locked(self, now: float) -> int:
        kept = deque(item for item in self._items if not item.is_expired(now))
        expired = len(self._items) - len(kept)
        if expired:
            self._items = kept
            self._expired += expired
        return expired
