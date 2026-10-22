"""End-to-end tests: simulated keystrokes in, verified prompt out.

These tests drive the full PhantomParse pipeline the way the product does --
the keystroke simulator produces the event timeline, the daemon consumes it,
and the assertions are the product promises: near-zero Time to First Token and
zero wasted LLM tokens on a mid-sentence pivot.
"""

from __future__ import annotations

from typing import List

import pytest

from benchmarks.keystroke_simulator import CHAR, PAUSE, scenario_by_key, simulate
from phantomparse.cli import main
from phantomparse.config import Config
from phantomparse.daemon import Daemon, maximal_spans
from phantomparse.overlay import Overlay
from phantomparse.prefill_adapter import BACKEND_EMULATOR, PrefillAdapter
from phantomparse.staging import VERIFICATION_BUDGET_MS

PIVOT = "Write a unit test for the Firebase auth middleware."
ABANDONED = ("Razorpay", "payment", "logs")
INJECTED = ("a unit test", "for the Firebase auth middleware")


def run_scenario(key: str) -> Daemon:
    """Feed a simulated scenario through a headless daemon."""
    config = Config()
    daemon = Daemon(
        config=config,
        prefill=PrefillAdapter(config, backend=BACKEND_EMULATOR),
        overlay=Overlay(headless=True),
    )
    result = simulate(scenario_by_key(key))
    for event in result.events:
        if event.kind == PAUSE:
            continue
        if event.kind == CHAR:
            daemon.feed_char(event.value)
        else:
            daemon.feed_backspace()
    return daemon


def test_pivot_scenario_ends_with_zero_wasted_tokens() -> None:
    daemon = run_scenario("pivot")

    result = daemon.submit()

    assert result.prompt == PIVOT
    assert result.wasted_tokens == 0
    assert result.injected == 2
    assert result.evicted == 3


def test_every_injected_span_is_verbatim_in_the_final_prompt() -> None:
    daemon = run_scenario("pivot")

    result = daemon.submit()

    assert result.injected_spans == INJECTED
    for span in result.injected_spans:
        assert span in result.prompt


def test_abandoned_pivot_spans_are_never_injected() -> None:
    daemon = run_scenario("pivot")

    result = daemon.submit()

    injected = " ".join(result.injected_spans)
    for span in ABANDONED:
        assert span not in injected


def test_rejected_spans_scored_below_the_threshold() -> None:
    daemon = run_scenario("pivot")

    result = daemon.submit()

    assert result.verification.rejected
    assert result.verification.max_score == pytest.approx(1.0)
    for scored in result.verification.rejected:
        assert scored.score < Config().similarity_threshold


def test_verification_stays_inside_its_latency_budget() -> None:
    daemon = run_scenario("pivot")

    result = daemon.submit()

    assert result.verification.within_budget is True
    assert result.verification.latency_ms < VERIFICATION_BUDGET_MS


def test_typing_pays_the_prefill_cost_so_enter_is_fast() -> None:
    daemon = run_scenario("pivot")

    result = daemon.submit()

    assert result.ttft_saved_ms >= 420.0
    assert result.prefill_latency_ms < result.ttft_saved_ms


def test_maximal_spans_collapse_growing_prefixes() -> None:
    spans = ["a unit", "a unit test", "for the Firebase", "for the Firebase auth"]

    assert maximal_spans(spans) == ("a unit test", "for the Firebase auth")


def test_maximal_spans_keep_unrelated_and_duplicate_spans_once() -> None:
    spans = ["in Razorpay", "in Razorpay", "for the Firebase auth middleware"]

    assert maximal_spans(spans) == ("in Razorpay", "for the Firebase auth middleware")


def test_linear_scenario_injects_every_staged_span() -> None:
    daemon = run_scenario("linear")

    result = daemon.submit()

    assert result.wasted_tokens == 0
    assert result.evicted == 0
    assert result.injected >= 1
    assert all(span in result.prompt for span in result.injected_spans)


def test_a_second_prompt_starts_from_a_clean_slate() -> None:
    daemon = run_scenario("pivot")
    daemon.submit()

    daemon.feed_text("Summarise the incident timeline")
    second = daemon.submit()

    assert second.prompt == "Summarise the incident timeline"
    assert second.wasted_tokens == 0
    assert daemon.submissions == 2


def test_overlay_never_shows_a_stale_stage_count_after_submit() -> None:
    daemon = run_scenario("pivot")
    before = daemon.overlay.state.staged

    daemon.submit()

    assert before > 0
    assert daemon.overlay.state.staged == 0


def test_pipeline_memory_stays_under_the_ram_budget() -> None:
    import tracemalloc

    tracemalloc.start()
    try:
        daemon = run_scenario("pivot")
        daemon.submit()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak / (1024 * 1024) < Config().max_ram_mb


def test_cli_overlay_demo_prints_the_pill(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["overlay", "--demo", "--headless", "--step", "12"]) == 0

    printed = capsys.readouterr().out
    assert "PHANTOMPARSE" in printed
    assert "injected" in printed


def test_cli_config_prints_the_effective_settings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["config"]) == 0

    printed = capsys.readouterr().out
    assert "similarity_threshold: 0.75" in printed
    assert "prefill_tokens_per_second: 240.0" in printed


def test_cli_bench_emits_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["bench", "--modes", "phantom", "--scenarios", "pivot"])

    printed = capsys.readouterr().out
    assert code == 0
    assert "| Mode | Scenario | TTFT (ms)" in printed
    assert "| PhantomParse | pivot |" in printed


def test_cli_rejects_an_unknown_scenario() -> None:
    assert main(["overlay", "--demo", "--headless", "--scenario", "nope"]) == 2


def test_full_timeline_reconstructs_the_prompt() -> None:
    events: List[str] = []
    result = simulate(scenario_by_key("pivot"))
    for event in result.events:
        if event.kind == CHAR:
            events.append(event.value)
        elif event.kind != PAUSE:
            events.pop()

    assert "".join(events) == PIVOT
