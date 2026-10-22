"""Tests for the deterministic keystroke simulator."""

from __future__ import annotations

import pytest

from benchmarks.keystroke_simulator import (
    BACKSPACE,
    CHAR,
    CHARS_PER_WORD,
    PAUSE,
    PIVOT_TEXT,
    SCENARIOS,
    Scenario,
    char_interval_ms,
    reconstruct,
    scenario_by_key,
    simulate,
    simulate_all,
)


def test_char_interval_matches_five_characters_per_word_convention() -> None:
    assert CHARS_PER_WORD == 5
    assert char_interval_ms(60) == pytest.approx(200.0)
    assert char_interval_ms(120) == pytest.approx(100.0)


def test_char_interval_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        char_interval_ms(0)
    with pytest.raises(ValueError):
        char_interval_ms(-10)


def test_linear_scenario_replays_to_its_initial_text() -> None:
    result = simulate(scenario_by_key("linear"))

    assert result.final_text == result.scenario.initial_text
    assert result.backspace_count == 0
    assert result.char_count == len(result.scenario.initial_text)


def test_virtual_timeline_advances_one_interval_per_event() -> None:
    result = simulate(scenario_by_key("linear"))

    timestamps = [event.t_ms for event in result.events]
    assert timestamps[0] == 0.0
    offsets = {round(b - a, 6) for a, b in zip(timestamps, timestamps[1:])}
    assert offsets == {round(result.interval_ms, 6)}
    assert result.duration_ms == pytest.approx(len(timestamps[:-1]) * 200.0)


def test_pivot_scenario_erases_then_retypes() -> None:
    result = simulate(scenario_by_key("pivot"))
    scenario = result.scenario

    assert result.final_text == scenario.final_text
    assert result.final_text == scenario.pivot_text
    assert result.backspace_count == len(PIVOT_TEXT)
    assert result.char_count == len(PIVOT_TEXT) + len(scenario.pivot_text)
    assert sum(1 for event in result.events if event.kind == PAUSE) == 1


def test_pivot_pause_last_longer_than_a_single_keystroke() -> None:
    result = simulate(scenario_by_key("pivot"))

    pause_index = next(i for i, event in enumerate(result.events) if event.kind == PAUSE)
    pause_gap = result.events[pause_index + 1].t_ms - result.events[pause_index].t_ms
    assert pause_gap == pytest.approx(result.scenario.pivot_pause_ms)


def test_override_wpm_changes_interval_but_not_final_text() -> None:
    slow = simulate(scenario_by_key("multi_clause"), wpm=30)
    fast = simulate(scenario_by_key("multi_clause"), wpm=120)

    assert slow.interval_ms == pytest.approx(400.0)
    assert fast.interval_ms == pytest.approx(100.0)
    assert slow.duration_ms == pytest.approx(fast.duration_ms * 4)
    assert slow.final_text == fast.final_text


def test_simulate_all_covers_every_bundled_scenario() -> None:
    results = simulate_all()

    assert len(results) == len(SCENARIOS) == 3
    assert [result.scenario.key for result in results] == [item.key for item in SCENARIOS]


def test_scenario_by_key_raises_on_unknown_key() -> None:
    with pytest.raises(KeyError):
        scenario_by_key("does-not-exist")


def test_reconstruct_ignores_backspace_on_empty_buffer() -> None:
    scenario = Scenario(
        key="edge",
        title="Backspace on empty buffer",
        initial_text="ab",
        pivot_text="",
        pivot_backspaces=5,
    )
    result = simulate(scenario)
    assert result.final_text == ""


def test_reconstruct_returns_empty_string_without_events() -> None:
    assert reconstruct([]) == ""


def test_event_kinds_are_a_closed_set() -> None:
    kinds = {event.kind for result in simulate_all() for event in result.events}
    assert kinds <= {CHAR, BACKSPACE, PAUSE}
