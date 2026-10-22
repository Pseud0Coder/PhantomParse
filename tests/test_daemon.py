"""Tests for the speculative daemon that wires buffer, gatekeeper and staging."""

from __future__ import annotations

from phantomparse.config import Config
from phantomparse.daemon import (
    HOTKEY_BACKEND_HEADLESS,
    Daemon,
    demo_states,
)
from phantomparse.overlay import Overlay
from phantomparse.prefill_adapter import BACKEND_EMULATOR, PrefillAdapter

INITIAL = "Check the payment failure logs in Razorpay"
PIVOT = "Write a unit test for the Firebase auth middleware."


def make_daemon() -> Daemon:
    config = Config()
    return Daemon(
        config=config,
        prefill=PrefillAdapter(config, backend=BACKEND_EMULATOR),
        overlay=Overlay(headless=True),
    )


def type_initial(daemon: Daemon, backspaces: int = len(INITIAL)) -> None:
    daemon.feed_text(INITIAL)
    for _ in range(backspaces):
        daemon.feed_backspace()
    daemon.feed_text(PIVOT)


def test_typing_a_closed_phrase_stages_it_in_ram() -> None:
    daemon = make_daemon()

    daemon.feed_text(INITIAL)

    assert daemon.buffer.text == INITIAL
    assert len(daemon.staging) >= 1
    assert "the payment failure logs" in daemon.chunk_texts()


def test_keystrokes_inside_a_word_do_not_stage_anything() -> None:
    daemon = make_daemon()

    daemon.feed_text("Chec")

    assert daemon.buffer.pending == "Chec"
    assert len(daemon.staging) == 0
    assert daemon.prefill.cached_tokens == 0


def test_backspace_counts_a_rollback_and_truncates_the_kv_cache() -> None:
    daemon = make_daemon()
    daemon.feed_text("Write a unit test ")
    cached_text = daemon.prefill.cached_text

    daemon.feed_backspace()  # deletes the boundary space, no token is invalidated
    rollback = daemon.feed_backspace()

    assert rollback is not None
    assert rollback.removed == "test"
    assert daemon.rollbacks == 1
    committed = " ".join(event.token for event in daemon.buffer.tokens)
    assert daemon.prefill.cached_text == committed
    assert len(daemon.prefill.cached_text) < len(cached_text)


def test_submit_verifies_stages_and_reports_zero_waste() -> None:
    daemon = make_daemon()
    type_initial(daemon)

    result = daemon.submit()

    assert result.prompt == PIVOT
    assert result.injected == 2
    assert result.evicted == 3
    assert result.wasted_tokens == 0
    assert result.injected_text == (
        "a unit test for the Firebase auth middleware"
    )
    assert result.injected_tokens > 0
    assert result.ttft_saved_ms > 0


def test_submit_resets_the_next_prompt() -> None:
    daemon = make_daemon()
    type_initial(daemon)

    daemon.submit()

    assert daemon.buffer.text == ""
    assert len(daemon.staging) == 0
    assert daemon.prefill.cached_tokens == 0
    assert daemon.submissions == 1


def test_overlay_state_follows_submission() -> None:
    daemon = make_daemon()
    type_initial(daemon)

    daemon.submit()
    state = daemon.overlay.state

    assert state.staged == 0
    assert state.injected == 2
    assert "2 injected" in state.status
    assert "3 evicted" in state.status


def test_toggle_pauses_and_rearms_speculation() -> None:
    daemon = make_daemon()

    assert daemon.toggle() is False
    assert daemon.active is False
    assert daemon.overlay.state.status == "paused"

    assert daemon.toggle() is True
    assert daemon.overlay.state.status == "armed"


def test_stats_expose_backend_and_counters() -> None:
    daemon = make_daemon()
    daemon.feed_text(INITIAL)

    stats = daemon.stats()

    assert stats["prefill_backend"] == BACKEND_EMULATOR
    assert stats["hotkey_backend"] == HOTKEY_BACKEND_HEADLESS
    assert stats["staged"] >= 1
    assert stats["active"] is True


def test_headless_run_and_stop_do_not_require_a_display() -> None:
    daemon = make_daemon()

    daemon.run(block=False)
    daemon.stop()

    assert daemon.overlay.headless is True


def test_demo_states_cover_typing_pivot_and_answer() -> None:
    states = demo_states(INITIAL, PIVOT, step=10)

    assert len(states) >= 4
    assert states[0].committed == "C"
    assert any(state.status.startswith("pivot") for state in states)
    assert states[-1].committed == PIVOT
    assert states[-1].injected == 2
    assert states[-1].staged == 0
