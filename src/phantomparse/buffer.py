"""Keystroke stream segmentation into committed orthographic tokens.

The buffer solves the Byte-Pair Encoding (BPE) subword instability problem: a
half-typed word can tokenise into a completely different set of subwords than
the finished word, so nothing is emitted until its right edge is confirmed by
whitespace or terminal punctuation.

Only committed tokens are safe to parse, stage, or prefill. The trailing
fragment stays in ``pending`` and is deliberately invisible downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

WHITESPACE_BOUNDARY = "whitespace"
PUNCTUATION_BOUNDARY = "punctuation"

# Only terminal punctuation closes a token. Apostrophes and hyphens stay inside
# the word ("don't", "late-bound") because they do not end an orthographic word.
TERMINAL_PUNCTUATION = frozenset(".,;:!?")


@dataclass(frozen=True)
class CommitEvent:
    """A token whose right boundary is confirmed."""

    token: str
    char_index: int
    token_index: int
    boundary: str = WHITESPACE_BOUNDARY


@dataclass
class KeystrokeBuffer:
    """Accumulates keystrokes and emits tokens only at confirmed boundaries."""

    _chars: List[str] = field(default_factory=list)
    _pending: List[str] = field(default_factory=list)
    _pending_start: Optional[int] = None
    _tokens: List[CommitEvent] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The full raw stream, committed and pending alike."""
        return "".join(self._chars)

    @property
    def pending(self) -> str:
        """The uncommitted trailing fragment."""
        return "".join(self._pending)

    @property
    def tokens(self) -> List[CommitEvent]:
        """Committed tokens in stream order."""
        return list(self._tokens)

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    def feed_char(self, char: str) -> Optional[CommitEvent]:
        """Consume one character, returning a token when a boundary closes it."""
        if not char:
            return None
        if len(char) > 1:
            raise ValueError("feed_char accepts exactly one character")

        if char.isspace():
            self._chars.append(char)
            return self._commit(WHITESPACE_BOUNDARY)

        if char in TERMINAL_PUNCTUATION:
            if self._pending_start is None:
                self._pending_start = len(self._chars)
            self._chars.append(char)
            self._pending.append(char)
            return self._commit(PUNCTUATION_BOUNDARY)

        if self._pending_start is None:
            self._pending_start = len(self._chars)
        self._chars.append(char)
        self._pending.append(char)
        return None

    def feed_text(self, text: str) -> List[CommitEvent]:
        """Consume a run of characters and return every token it closed."""
        events: List[CommitEvent] = []
        for char in text:
            event = self.feed_char(char)
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> Optional[CommitEvent]:
        """Commit the pending fragment, as pressing Enter does."""
        return self._commit(WHITESPACE_BOUNDARY)

    def _commit(self, boundary: str) -> Optional[CommitEvent]:
        if not self._pending:
            self._pending_start = None
            return None
        token = "".join(self._pending)
        event = CommitEvent(
            token=token,
            char_index=self._pending_start if self._pending_start is not None else 0,
            token_index=len(self._tokens),
            boundary=boundary,
        )
        self._tokens.append(event)
        self._pending.clear()
        self._pending_start = None
        return event
