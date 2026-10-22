"""Tests for the keystroke stream buffer."""

from __future__ import annotations

import pytest

from phantomparse.buffer import (
    PUNCTUATION_BOUNDARY,
    WHITESPACE_BOUNDARY,
    CommitEvent,
    KeystrokeBuffer,
    RollbackEvent,
)


def test_word_is_not_committed_until_whitespace_closes_it() -> None:
    buffer = KeystrokeBuffer()

    assert buffer.feed_text("paym") == []
    assert buffer.token_count == 0
    assert buffer.pending == "paym"

    events = buffer.feed_text("ent ")

    assert events == [
        CommitEvent(
            token="payment",
            char_index=0,
            token_index=0,
            boundary=WHITESPACE_BOUNDARY,
        )
    ]
    assert buffer.pending == ""
    assert buffer.text == "payment "


def test_mid_word_pause_leaves_the_fragment_uncommitted() -> None:
    buffer = KeystrokeBuffer()

    buffer.feed_text("Fireba")
    assert buffer.token_count == 0

    # No input arrives while the user thinks; a pause must never flush a
    # half-typed word, because that is what makes subword tokenisation unstable.
    assert buffer.pending == "Fireba"

    events = buffer.feed_text("se ")

    assert [event.token for event in events] == ["Firebase"]
    assert buffer.pending == ""


def test_rapid_typing_matches_character_by_character_typing() -> None:
    text = "Check the payment failure logs in Razorpay"

    rapid = KeystrokeBuffer()
    rapid_events = rapid.feed_text(text)

    gradual = KeystrokeBuffer()
    gradual_events = [event for char in text for event in [gradual.feed_char(char)] if event]

    assert rapid_events == gradual_events
    # The final word has no right boundary yet, so it is still pending.
    assert [event.token for event in rapid_events] == text.split()[:-1]
    assert rapid.pending == "Razorpay"


def test_terminal_punctuation_closes_a_token_without_a_space() -> None:
    buffer = KeystrokeBuffer()

    events = buffer.feed_text("middleware.")

    assert len(events) == 1
    assert events[0].token == "middleware."
    assert events[0].boundary == PUNCTUATION_BOUNDARY
    assert buffer.pending == ""


def test_apostrophes_and_hyphens_stay_inside_the_word() -> None:
    buffer = KeystrokeBuffer()

    events = buffer.feed_text("don't late-bound ")

    assert [event.token for event in events] == ["don't", "late-bound"]


def test_char_indexes_point_at_the_token_start_in_the_raw_stream() -> None:
    buffer = KeystrokeBuffer()

    events = buffer.feed_text("Check the payment ")

    assert [(event.token, event.char_index) for event in events] == [
        ("Check", 0),
        ("the", 6),
        ("payment", 10),
    ]
    assert [event.token_index for event in events] == [0, 1, 2]


def test_repeated_whitespace_never_commits_an_empty_token() -> None:
    buffer = KeystrokeBuffer()

    assert buffer.feed_text("   ") == []
    assert buffer.token_count == 0


def test_flush_commits_a_trailing_fragment_once() -> None:
    buffer = KeystrokeBuffer()
    buffer.feed_text("Razorpay")

    first = buffer.flush()
    second = buffer.flush()

    assert first is not None
    assert first.token == "Razorpay"
    assert second is None
    assert buffer.token_count == 1


def test_feed_char_rejects_multi_character_input() -> None:
    with pytest.raises(ValueError):
        KeystrokeBuffer().feed_char("ab")


def test_backspace_inside_pending_fragment_emits_nothing() -> None:
    buffer = KeystrokeBuffer()
    buffer.feed_text("Fireba")

    assert buffer.feed_backspace() is None
    assert buffer.pending == "Fireb"
    assert buffer.token_count == 0


def test_backspace_deleting_whitespace_emits_no_rollback() -> None:
    buffer = KeystrokeBuffer()
    buffer.feed_text("Check ")

    assert buffer.feed_backspace() is None
    assert [event.token for event in buffer.tokens] == ["Check"]
    assert buffer.text == "Check"


def test_backspace_into_committed_token_emits_rollback_event() -> None:
    buffer = KeystrokeBuffer()
    buffer.feed_text("Check the payment ")
    buffer.feed_backspace()  # removes the trailing space

    rollback = buffer.feed_backspace()

    assert rollback == RollbackEvent(char_index=16, token_index=2, removed="payment")
    assert [event.token for event in buffer.tokens] == ["Check", "the"]
    assert buffer.pending == "paymen"


def test_multi_word_backspace_unwinds_tokens_in_reverse_order() -> None:
    buffer = KeystrokeBuffer()
    buffer.feed_text("Check the payment logs ")

    rollbacks = []
    for _ in range(13):
        event = buffer.feed_backspace()
        if event is not None:
            rollbacks.append(event)

    assert [event.removed for event in rollbacks] == ["logs", "payment"]
    assert [event.token_index for event in rollbacks] == [3, 2]
    assert [event.token for event in buffer.tokens] == ["Check", "the"]
    assert buffer.pending == ""


def test_punctuation_token_rolls_back_to_its_bare_word() -> None:
    buffer = KeystrokeBuffer()
    buffer.feed_text("Razorpay.")

    rollback = buffer.feed_backspace()

    assert rollback is not None
    assert rollback.removed == "Razorpay."
    assert buffer.tokens == []
    assert buffer.pending == "Razorpay"


def test_retyping_after_rollback_recommits_at_the_same_token_index() -> None:
    buffer = KeystrokeBuffer()
    buffer.feed_text("Check the payment ")

    for _ in range(8):  # trailing space plus the seven letters of "payment"
        buffer.feed_backspace()

    assert [event.token for event in buffer.tokens] == ["Check", "the"]
    assert buffer.pending == ""

    recommitted = buffer.feed_text("payment ")

    assert [event.token for event in recommitted] == ["payment"]
    assert recommitted[0].token_index == 2
    assert recommitted[0].char_index == 10
    assert [event.token for event in buffer.tokens] == ["Check", "the", "payment"]


def test_backspace_on_empty_buffer_is_a_no_op() -> None:
    buffer = KeystrokeBuffer()

    assert buffer.feed_backspace() is None
    assert buffer.text == ""


def test_pending_text_always_matches_the_raw_stream_suffix() -> None:
    buffer = KeystrokeBuffer()
    buffer.feed_text("Compare the attention sink ")
    buffer.feed_text("mechanism")

    assert buffer.text.endswith(buffer.pending)
    assert buffer.pending == "mechanism"

    buffer.feed_backspace()
    assert buffer.text.endswith(buffer.pending)
    assert buffer.pending == "mechanis"
