"""Tests for the RAM staging ring buffer and late-bound verification."""

from __future__ import annotations

import threading

import pytest

from benchmarks.keystroke_simulator import scenario_by_key, simulate
from phantomparse.buffer import KeystrokeBuffer
from phantomparse.gatekeeper import SyntacticGatekeeper
from phantomparse.staging import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TTL_SECONDS,
    VERIFICATION_BUDGET_MS,
    StagingBuffer,
    similarity,
)

PIVOT_PROMPT = "Write a unit test for the Firebase auth middleware."
ABANDONED_SPANS = (
    "the payment failure logs",
    "in Razorpay",
    "Check the payment failure logs in Razorpay",
)
FINAL_SPANS = ("a unit test", "for the Firebase auth middleware")


class FakeClock:
    """Advanceable clock so TTL behaviour is deterministic and instant."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def replay_pivot(buffer_max: int = 64, clock: FakeClock | None = None) -> StagingBuffer:
    """Replay the pivot scenario through buffer, gatekeeper and staging."""
    scenario = scenario_by_key("pivot")
    events = simulate(scenario).events
    buffer = KeystrokeBuffer()
    gatekeeper = SyntacticGatekeeper(prefer_spacy=False)
    staging = StagingBuffer(
        ttl_seconds=DEFAULT_TTL_SECONDS,
        max_items=buffer_max,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        clock=clock or FakeClock(),
    )
    for event in events:
        if event.kind == "char":
            if buffer.feed_char(event.value) is not None:
                staging.stage_chunks(gatekeeper.parse(_committed(buffer)).chunks)
        elif event.kind == "backspace":
            buffer.feed_backspace()
    buffer.flush()
    staging.stage_chunks(gatekeeper.parse(_committed(buffer)).chunks)
    return staging


def _committed(buffer: KeystrokeBuffer) -> str:
    return " ".join(event.token for event in buffer.tokens)


def test_similarity_of_the_same_span_is_one() -> None:
    assert similarity(PIVOT_PROMPT, "for the Firebase auth middleware") == pytest.approx(1.0)


def test_similarity_separates_pivot_spans_from_the_final_prompt() -> None:
    for span in ABANDONED_SPANS:
        assert similarity(PIVOT_PROMPT, span) < DEFAULT_SIMILARITY_THRESHOLD
    for span in FINAL_SPANS:
        assert similarity(PIVOT_PROMPT, span) >= DEFAULT_SIMILARITY_THRESHOLD


def test_similarity_handles_blank_input() -> None:
    assert similarity("", "anything") == 0.0
    assert similarity("anything", "   ") == 0.0


def test_similarity_ignores_case_and_punctuation() -> None:
    assert similarity("Write a unit test.", "A UNIT TEST") == pytest.approx(1.0)


def test_stage_returns_a_read_only_item_with_ttl() -> None:
    clock = FakeClock(100.0)
    staging = StagingBuffer(clock=clock)
    item = staging.stage("the Firebase auth middleware", kind="NP")

    assert item is not None
    assert item.speculative_read_only is True
    assert item.kind == "NP"
    assert item.created_at == 100.0
    assert item.expires_at == 100.0 + DEFAULT_TTL_SECONDS
    assert item.is_expired(100.0) is False
    assert item.is_expired(100.0 + DEFAULT_TTL_SECONDS) is True


def test_blank_spans_are_not_staged() -> None:
    staging = StagingBuffer()
    assert staging.stage("   ") is None
    assert len(staging) == 0


def test_duplicate_spans_are_staged_once() -> None:
    staging = StagingBuffer()
    first = staging.stage("the Firebase auth middleware")
    second = staging.stage("The Firebase auth middleware!")

    assert first is not None and second is not None
    assert first.item_id == second.item_id
    assert len(staging) == 1
    assert staging.stats()["staged"] == 1


def test_expired_items_are_purged_on_snapshot() -> None:
    clock = FakeClock()
    staging = StagingBuffer(ttl_seconds=5.0, clock=clock)
    staging.stage("the payment failure logs")

    clock.advance(4.9)
    assert len(staging.snapshot()) == 1

    clock.advance(0.2)
    assert staging.snapshot() == ()
    assert len(staging) == 0
    assert staging.stats()["expired"] == 1


def test_capacity_evicts_the_oldest_item() -> None:
    staging = StagingBuffer(max_items=3)
    for text in ("alpha one", "beta two", "gamma three", "delta four"):
        staging.stage(text)

    live = [item.text for item in staging.snapshot()]
    assert live == ["beta two", "gamma three", "delta four"]
    assert staging.stats()["evicted_capacity"] == 1


def test_stage_chunks_returns_only_new_items() -> None:
    gatekeeper = SyntacticGatekeeper(prefer_spacy=False)
    staging = StagingBuffer()
    chunks = gatekeeper.parse(PIVOT_PROMPT).chunks

    assert staging.stage_chunks(chunks) == len(chunks)
    assert staging.stage_chunks(chunks) == 0


def test_verify_accepts_spans_that_survive_into_the_final_prompt() -> None:
    staging = StagingBuffer()
    for span in FINAL_SPANS:
        staging.stage(span)

    result = staging.verify(PIVOT_PROMPT)

    assert set(result.accepted_text) == set(FINAL_SPANS)
    assert result.rejected == ()
    assert result.threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert result.max_score == pytest.approx(1.0)


def test_verify_evicts_abandoned_spans_and_keeps_survivors() -> None:
    staging = StagingBuffer()
    for span in FINAL_SPANS:
        staging.stage(span)
    for span in ABANDONED_SPANS:
        staging.stage(span)

    result = staging.verify(PIVOT_PROMPT)

    assert {entry.item.text for entry in result.rejected} == set(ABANDONED_SPANS)
    assert len(result.accepted) == len(FINAL_SPANS)
    assert len(staging) == len(FINAL_SPANS)
    assert staging.stats()["evicted_verification"] == len(ABANDONED_SPANS)


def test_mid_sentence_pivot_yields_zero_wasted_tokens() -> None:
    staging = replay_pivot()
    result = staging.verify(PIVOT_PROMPT)

    staged_texts = {entry.item.text.lower() for entry in result.scored}
    abandoned = [text for text in staged_texts if "razorpay" in text or "payment" in text]
    assert abandoned, "the abandoned clause should have been staged locally"

    for entry in result.rejected:
        assert entry.score < result.threshold

    final_normalized = PIVOT_PROMPT.lower()
    for text in result.accepted_text:
        assert text.lower() in final_normalized

    # Only accepted spans can reach the model context, so no prefetched span
    # from the abandoned clause is ever paid for.
    injected = set(result.accepted_text)
    wasted = {text for text in abandoned if text in injected}
    assert wasted == set()
    assert result.accepted, "the retyped clause must still be injected"


def test_verify_threshold_override_accepts_weaker_matches() -> None:
    staging = StagingBuffer()
    staging.stage("the payment failure logs")
    weak = similarity(PIVOT_PROMPT, "the payment failure logs")
    assert 0.0 < weak < DEFAULT_SIMILARITY_THRESHOLD

    item = staging.stage("the payment failure logs")
    assert item is not None
    result = staging.verify(PIVOT_PROMPT, threshold=weak - 0.01)
    assert len(result.accepted) == 1


def test_verify_on_an_empty_buffer_is_a_no_op() -> None:
    result = StagingBuffer().verify("anything at all")
    assert result.scored == ()
    assert result.accepted_text == ()
    assert result.max_score == 0.0


def test_verify_latency_stays_inside_the_budget() -> None:
    staging = replay_pivot()
    staging.verify(PIVOT_PROMPT)
    timings = [staging.verify(PIVOT_PROMPT).latency_ms for _ in range(25)]
    assert max(timings) < VERIFICATION_BUDGET_MS


def test_clear_empties_the_buffer() -> None:
    staging = StagingBuffer()
    staging.stage("the Firebase auth middleware")
    staging.clear()
    assert len(staging) == 0
    assert staging.snapshot() == ()


def test_stats_report_configuration_and_counters() -> None:
    staging = StagingBuffer(ttl_seconds=12.0, max_items=7, similarity_threshold=0.5)
    stats = staging.stats()

    assert stats["ttl_seconds"] == 12.0
    assert stats["max_items"] == 7
    assert stats["similarity_threshold"] == 0.5
    assert stats["staged"] == 0
    assert stats["live"] == 0


def test_concurrent_staging_respects_capacity() -> None:
    staging = StagingBuffer(max_items=20)
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            for index in range(50):
                staging.stage(f"span {worker_id} item {index}")
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(staging) == 20
    assert staging.stats()["staged"] == 400
    assert staging.stats()["evicted_capacity"] == 380


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ttl_seconds": 0}, "ttl_seconds"),
        ({"max_items": 0}, "max_items"),
        ({"similarity_threshold": 1.5}, "similarity_threshold"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StagingBuffer(**kwargs)
