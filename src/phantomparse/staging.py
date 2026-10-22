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

import math
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

DEFAULT_TTL_SECONDS = 30.0
DEFAULT_MAX_ITEMS = 64
DEFAULT_SIMILARITY_THRESHOLD = 0.75
NGRAM_SIZE = 3
VERIFICATION_BUDGET_MS = 15.0

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> str:
    """Lowercase and strip punctuation so casing and commas do not matter."""
    return " ".join(_WORD_RE.findall(text.lower()))


def char_ngrams(text: str, size: int = NGRAM_SIZE) -> List[str]:
    """Character n-grams over the normalized text, padded for word edges."""
    joined = f" {normalize(text)} "
    if not joined.strip():
        return []
    if len(joined) <= size:
        return [joined]
    return [joined[index : index + size] for index in range(len(joined) - size + 1)]


def _gram_weights(text: str) -> Dict[str, float]:
    """Sublinear term frequency, the classic damped count for short spans."""
    counts = Counter(char_ngrams(text))
    return {gram: 1.0 + math.log(count) for gram, count in counts.items()}


def _cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[gram] * right[gram] for gram in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def similarity(query: str, item: str) -> float:
    """Score how strongly ``query`` supports a staged ``item``.

    Character-trigram cosine is length sensitive: a phrase that appears
    verbatim inside a long prompt still scores low because the prompt carries
    many extra grams. Containment (the fraction of the item's grams present in
    the query) fixes that asymmetry, and the maximum of the two is the score.
    """
    item_grams = set(char_ngrams(item))
    query_grams = set(char_ngrams(query))
    if not item_grams or not query_grams:
        return 0.0
    containment = len(item_grams & query_grams) / len(item_grams)
    cosine = _cosine(_gram_weights(item), _gram_weights(query))
    return max(cosine, containment)


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


@dataclass(frozen=True)
class ScoredItem:
    """A staged item paired with its verification score."""

    item: StagedItem
    score: float
    accepted: bool


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of comparing the final prompt against the staging buffer."""

    prompt: str
    threshold: float
    latency_ms: float
    scored: Tuple[ScoredItem, ...]
    expired: int = 0

    @property
    def accepted(self) -> Tuple[ScoredItem, ...]:
        return tuple(entry for entry in self.scored if entry.accepted)

    @property
    def rejected(self) -> Tuple[ScoredItem, ...]:
        return tuple(entry for entry in self.scored if not entry.accepted)

    @property
    def accepted_text(self) -> Tuple[str, ...]:
        return tuple(entry.item.text for entry in self.accepted)

    @property
    def max_score(self) -> float:
        return max((entry.score for entry in self.scored), default=0.0)

    @property
    def within_budget(self) -> bool:
        return self.latency_ms < VERIFICATION_BUDGET_MS


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

    def verify(self, prompt: str, threshold: Optional[float] = None) -> VerificationResult:
        """Score every staged item against ``prompt`` and evict the losers."""
        limit = self._threshold if threshold is None else float(threshold)
        started = time.perf_counter()
        with self._lock:
            expired = self._purge_locked(self._clock())
            scored: List[ScoredItem] = []
            survivors: Deque[StagedItem] = deque()
            for item in self._items:
                score = similarity(prompt, item.text)
                accepted = score >= limit
                scored.append(ScoredItem(item=item, score=score, accepted=accepted))
                if accepted:
                    survivors.append(item)
                else:
                    self._verification_evictions += 1
            self._items = survivors
        latency_ms = (time.perf_counter() - started) * 1000.0
        return VerificationResult(
            prompt=prompt,
            threshold=limit,
            latency_ms=latency_ms,
            scored=tuple(scored),
            expired=expired,
        )

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
