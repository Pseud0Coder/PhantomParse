"""Local syntactic gatekeeper: shallow Noun Phrase / Prepositional Phrase chunking.

Full constituency parsing is unnecessary and too slow for a per-keystroke loop.
What matters downstream is which closed constituents carry retrieval signal:
Noun Phrases (NP) and Prepositional Phrases (PP).

Two backends, same output shape:

1. SpaCy ``en_core_web_sm`` when it is installed and loadable.
2. A deterministic closed-class rule-based chunker with no model download, no
   network access, and a latency budget of a few milliseconds.

Both ignore incomplete constituents such as ``"in the "``, whose complement
contains only a determiner and therefore has no head noun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from phantomparse.buffer import TERMINAL_PUNCTUATION

NP = "NP"
PP = "PP"

DETERMINERS = frozenset(
    """
    the a an this that these those my your our their its his her
    """.split()
)

PREPOSITIONS = frozenset(
    """
    in on at for with from into about across over under between against within
    during by of to through after before near via
    """.split()
)

# Closed-class and unambiguous predicates. When one of these appears, the
# current chunk is considered closed. Ambiguous nouns ("test", "log", "review")
# are deliberately excluded so they can still terminate a noun phrase.
BOUNDARY_WORDS = frozenset(
    """
    is are was were be been being am has have had do does did can could should
    would will may might must
    and or but if then than when while because so that which who whom whose
    where how why not no out
    explain compare write show tell give find think know see look go come take
    put keep let need want make use get set run add remove delete create build
    implement generate describe summarize define refactor analyze suggest list
    return handle handles check
    """.split()
)

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[.,;:!?]")

GATEKEEPER_BUDGET_MS = 15.0


@dataclass(frozen=True)
class Chunk:
    """A closed syntactic constituent recovered from the committed text."""

    text: str
    kind: str
    start: int
    end: int
    head: str = ""

    @property
    def is_phrase(self) -> bool:
        return self.kind in (NP, PP)


def _is_nounish(token: str) -> bool:
    """True when a token can sit inside a noun phrase."""
    if not token or token in TERMINAL_PUNCTUATION:
        return False
    low = token.lower()
    if low in PREPOSITIONS or low in BOUNDARY_WORDS:
        return False
    return token[0].isalnum()


def _has_content(tokens: Sequence[str]) -> bool:
    return any(token.lower() not in DETERMINERS for token in tokens)


def _rule_chunks(text: str) -> List[Chunk]:
    """Deterministic closed-class chunker: the no-download fallback."""
    matches = list(TOKEN_RE.finditer(text))
    chunks: List[Chunk] = []
    index = 0

    while index < len(matches):
        token = matches[index].group()
        low = token.lower()

        if low in PREPOSITIONS:
            words: List[str] = []
            cursor = index + 1
            while cursor < len(matches) and _is_nounish(matches[cursor].group()):
                words.append(matches[cursor].group())
                cursor += 1
            if _has_content(words):
                chunks.append(
                    Chunk(
                        text=" ".join([token] + words),
                        kind=PP,
                        start=matches[index].start(),
                        end=matches[cursor - 1].end(),
                        head=low,
                    )
                )
            index = cursor
            continue

        if _is_nounish(token):
            words = []
            cursor = index
            while cursor < len(matches) and _is_nounish(matches[cursor].group()):
                words.append(matches[cursor].group())
                cursor += 1
            if _has_content(words):
                head = [word for word in words if word.lower() not in DETERMINERS][-1]
                chunks.append(
                    Chunk(
                        text=" ".join(words),
                        kind=NP,
                        start=matches[index].start(),
                        end=matches[cursor - 1].end(),
                        head=head.lower(),
                    )
                )
            index = cursor
            continue

        index += 1

    return chunks


class SyntacticGatekeeper:
    """Shallow chunker with an optional SpaCy backend."""

    def __init__(self, prefer_spacy: bool = True) -> None:
        self._nlp = _load_spacy() if prefer_spacy else None

    @property
    def backend(self) -> str:
        return "spacy" if self._nlp is not None else "rules"

    def parse(self, text: str) -> List[Chunk]:
        """Return the closed NP and PP constituents of ``text``."""
        if not text or not text.strip():
            return []
        if self._nlp is not None:
            try:
                return _spacy_chunks(self._nlp, text)
            except Exception:  # pragma: no cover - defensive: bad model state
                self._nlp = None
        return _rule_chunks(text)

    def chunks_of_kind(self, text: str, kind: str) -> List[Chunk]:
        """Convenience filter used by benchmarks and tests."""
        return [chunk for chunk in self.parse(text) if chunk.kind == kind]


def _load_spacy():
    """Load ``en_core_web_sm`` if SpaCy and the model are both available."""
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:  # model missing, incompatible, or blocked by policy
        return None


def _spacy_chunks(nlp, text: str) -> List[Chunk]:  # pragma: no cover - optional
    """Extract NP and PP constituents with SpaCy's dependency parser."""
    doc = nlp(text)
    chunks: List[Chunk] = [
        Chunk(
            text=chunk.text,
            kind=NP,
            start=chunk.start_char,
            end=chunk.end_char,
            head=chunk.root.text.lower(),
        )
        for chunk in doc.noun_chunks
    ]

    for token in doc:
        if token.pos_ != "ADP":
            continue
        for noun_chunk in doc.noun_chunks:
            if noun_chunk.start == token.i + 1:
                chunks.append(
                    Chunk(
                        text=f"{token.text} {noun_chunk.text}",
                        kind=PP,
                        start=token.idx,
                        end=noun_chunk.end_char,
                        head=token.text.lower(),
                    )
                )
                break

    chunks.sort(key=lambda chunk: (chunk.start, chunk.end))
    return chunks


_DEFAULT: Optional[SyntacticGatekeeper] = None


def default_gatekeeper() -> SyntacticGatekeeper:
    """Process-wide gatekeeper, so the SpaCy model is loaded at most once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SyntacticGatekeeper()
    return _DEFAULT


def chunk_text(text: str) -> Tuple[Chunk, ...]:
    """Module-level helper mirroring ``default_gatekeeper().parse``."""
    return tuple(default_gatekeeper().parse(text))
