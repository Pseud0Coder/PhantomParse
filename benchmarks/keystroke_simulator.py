"""Deterministic human-typing simulator.

Virtual time only: no wall-clock sleeps, so benchmarks, tests, and GIF rendering
are reproducible. Typing speed follows the standard convention of five
characters per word (WPM = Words Per Minute).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

CHARS_PER_WORD = 5

CHAR = "char"
BACKSPACE = "backspace"
PAUSE = "pause"

PIVOT_TEXT = "Check the payment failure logs in Razorpay"


@dataclass(frozen=True)
class Scenario:
    """A scripted typing session."""

    key: str
    title: str
    initial_text: str
    pivot_text: str = ""
    pivot_backspaces: int = 0
    pivot_pause_ms: float = 400.0
    wpm: int = 60

    @property
    def final_text(self) -> str:
        """The prompt that remains once the user presses Enter."""
        if self.pivot_backspaces <= 0:
            return self.initial_text
        kept = self.initial_text[: max(0, len(self.initial_text) - self.pivot_backspaces)]
        return kept + self.pivot_text


@dataclass(frozen=True)
class SimEvent:
    """A single input event on a virtual timeline."""

    kind: str
    value: str
    t_ms: float
    index: int


@dataclass(frozen=True)
class SimulationResult:
    """A scenario rendered as a timeline of events."""

    scenario: Scenario
    events: Tuple[SimEvent, ...]
    interval_ms: float

    @property
    def duration_ms(self) -> float:
        return self.events[-1].t_ms if self.events else 0.0

    @property
    def final_text(self) -> str:
        return reconstruct(self.events)

    @property
    def char_count(self) -> int:
        return sum(1 for event in self.events if event.kind == CHAR)

    @property
    def backspace_count(self) -> int:
        return sum(1 for event in self.events if event.kind == BACKSPACE)


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        key="linear",
        title="Linear technical query",
        initial_text=(
            "Explain how BPE tokenization handles out of vocabulary morphemes "
            "in transformer models"
        ),
    ),
    Scenario(
        key="pivot",
        title="Mid-sentence pivot (waste trap)",
        initial_text=PIVOT_TEXT,
        pivot_text="Write a unit test for the Firebase auth middleware.",
        pivot_backspaces=len(PIVOT_TEXT),
        pivot_pause_ms=400.0,
    ),
    Scenario(
        key="multi_clause",
        title="Multi-clause RAG query",
        initial_text=(
            "Compare the attention sink mechanism in StreamingLLM with "
            "PagedAttention in vLLM"
        ),
    ),
)


def char_interval_ms(wpm: int) -> float:
    """Milliseconds between keystrokes for a given words-per-minute rate."""
    if wpm <= 0:
        raise ValueError("wpm must be positive")
    return 60_000.0 / (wpm * CHARS_PER_WORD)


def simulate(scenario: Scenario, wpm: Optional[int] = None) -> SimulationResult:
    """Render a scenario into an event timeline at a fixed typing speed."""
    interval = char_interval_ms(wpm or scenario.wpm)
    events: List[SimEvent] = []
    t_ms = 0.0
    index = 0

    def add(kind: str, value: str) -> None:
        nonlocal index
        events.append(SimEvent(kind=kind, value=value, t_ms=t_ms, index=index))
        index += 1

    for char in scenario.initial_text:
        add(CHAR, char)
        t_ms += interval

    if scenario.pivot_backspaces:
        add(PAUSE, "")
        t_ms += scenario.pivot_pause_ms
        for _ in range(scenario.pivot_backspaces):
            add(BACKSPACE, "")
            t_ms += interval
        for char in scenario.pivot_text:
            add(CHAR, char)
            t_ms += interval

    return SimulationResult(scenario=scenario, events=tuple(events), interval_ms=interval)


def simulate_all(wpm: Optional[int] = None) -> List[SimulationResult]:
    """Render every bundled scenario."""
    return [simulate(scenario, wpm=wpm) for scenario in SCENARIOS]


def scenario_by_key(key: str) -> Scenario:
    """Look up a bundled scenario by its stable key."""
    for scenario in SCENARIOS:
        if scenario.key == key:
            return scenario
    raise KeyError(f"Unknown scenario: {key}")


def reconstruct(events: Sequence[SimEvent]) -> str:
    """Replay events to recover the visible text buffer."""
    buffer: List[str] = []
    for event in events:
        if event.kind == CHAR:
            buffer.append(event.value)
        elif event.kind == BACKSPACE and buffer:
            buffer.pop()
    return "".join(buffer)
