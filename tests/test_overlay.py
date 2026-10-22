"""Tests for the headless pill state machine and its deterministic text."""

from __future__ import annotations

from phantomparse.overlay import (
    CHARCOAL,
    CYAN,
    DIM,
    PURPLE,
    Overlay,
    OverlayState,
    pill_lines,
    tkinter_available,
)


def state(**overrides: object) -> OverlayState:
    base = {
        "committed": "Write a unit test",
        "chunks": ("a unit test",),
        "staged": 3,
        "injected": 0,
        "ttft_saved_ms": 466.4,
        "status": "armed",
    }
    base.update(overrides)
    return OverlayState(**base)  # type: ignore[arg-type]


def test_palette_matches_the_synthwave_spec() -> None:
    assert (CHARCOAL, CYAN, PURPLE) == ("#12131A", "#00F5D4", "#9B5DE5")
    assert DIM != CHARCOAL


def test_pill_lines_report_status_chunks_and_savings() -> None:
    lines = pill_lines(state())

    assert lines[0] == "PHANTOMPARSE   armed"
    assert lines[1] == "chunks 1   staged 3   saved 466 ms"
    assert lines[2] == "a unit test"


def test_pill_lines_join_multiple_chunks() -> None:
    lines = pill_lines(state(chunks=("a unit test", "for the Firebase auth middleware")))

    assert lines[1].startswith("chunks 2")
    assert lines[2] == "a unit test | for the Firebase auth middleware"


def test_pill_lines_placeholder_before_any_closed_phrase() -> None:
    lines = pill_lines(state(chunks=(), staged=0, ttft_saved_ms=0.0))

    assert lines[1] == "chunks 0   staged 0   saved 0 ms"
    assert lines[2] == "-- no closed phrases yet"


def test_pill_lines_truncate_long_chunk_lists_from_the_left() -> None:
    long_chunk = "the payment failure logs " * 4
    lines = pill_lines(state(chunks=(long_chunk,)), width=30)

    assert lines[2].startswith("...")
    assert len(lines[2]) == 30
    assert lines[2].endswith("failure logs ")


def test_headless_overlay_records_state_without_drawing() -> None:
    overlay = Overlay(headless=True)
    applied = state()

    lines = overlay.update(applied)

    assert overlay.headless is True
    assert overlay.state is applied
    assert lines == pill_lines(applied)


def test_headless_run_cycles_to_the_final_frame() -> None:
    overlay = Overlay(headless=True)
    frames = (state(status="armed"), state(status="responding"), state(status="done"))

    overlay.run(frames, interval_ms=1)

    assert overlay.state.status == "done"


def test_close_marks_the_overlay_closed() -> None:
    overlay = Overlay(headless=True)

    overlay.close()

    assert overlay._closed is True


def test_tkinter_availability_is_a_boolean() -> None:
    assert isinstance(tkinter_available(), bool)
