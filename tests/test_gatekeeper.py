"""Tests for the local syntactic gatekeeper."""

from __future__ import annotations

import pytest

from phantomparse.gatekeeper import (
    GATEKEEPER_BUDGET_MS,
    NP,
    PP,
    Chunk,
    SyntacticGatekeeper,
    chunk_text,
    default_gatekeeper,
)


@pytest.fixture()
def gatekeeper() -> SyntacticGatekeeper:
    return SyntacticGatekeeper(prefer_spacy=False)


def _texts(chunks) -> list:
    return [chunk.text for chunk in chunks]


def test_rule_backend_is_selected_when_spacy_is_disabled(gatekeeper) -> None:
    assert gatekeeper.backend == "rules"


def test_noun_phrase_and_prepositional_phrase_are_both_detected(gatekeeper) -> None:
    result = gatekeeper.parse("Write a unit test for the Firebase auth middleware.")

    assert _texts(result.chunks) == [
        "a unit test",
        "for the Firebase auth middleware",
    ]
    assert result.by_kind(NP)[0].text == "a unit test"
    assert result.by_kind(PP)[0].text == "for the Firebase auth middleware"


def test_pivot_clause_yields_the_retrieval_bearing_phrase(gatekeeper) -> None:
    result = gatekeeper.parse("Check the payment failure logs in Razorpay ")

    assert _texts(result.chunks) == [
        "the payment failure logs",
        "in Razorpay",
    ]
    assert result.by_kind(PP)[0].head == "in"


def test_incomplete_fragment_without_a_head_noun_is_ignored(gatekeeper) -> None:
    assert gatekeeper.parse("in the ").chunks == ()
    assert gatekeeper.parse("for a ").chunks == ()
    assert gatekeeper.parse("in the ").by_kind(PP) == ()


def test_pp_is_recovered_once_the_complement_has_content(gatekeeper) -> None:
    result = gatekeeper.parse("in the payment logs")

    assert _texts(result.chunks) == ["in the payment logs"]
    assert result.chunks[0].kind == PP


def test_multiple_prepositional_phrases_are_ordered_by_offset(gatekeeper) -> None:
    text = (
        "Compare the attention sink mechanism in StreamingLLM "
        "with PagedAttention in vLLM"
    )
    result = gatekeeper.parse(text)

    assert _texts(result.chunks) == [
        "the attention sink mechanism",
        "in StreamingLLM",
        "with PagedAttention",
        "in vLLM",
    ]
    offsets = [chunk.start for chunk in result.chunks]
    assert offsets == sorted(offsets)


def test_chunk_offsets_slice_back_to_the_chunk_text(gatekeeper) -> None:
    text = "Check the payment failure logs in Razorpay "
    for chunk in gatekeeper.parse(text).chunks:
        assert text[chunk.start:chunk.end] == chunk.text


def test_terminal_punctuation_is_not_part_of_the_chunk(gatekeeper) -> None:
    result = gatekeeper.parse("Firebase auth middleware.")

    assert _texts(result.chunks) == ["Firebase auth middleware"]
    assert result.chunks[0].kind == NP


def test_noun_phrase_head_is_its_last_content_word(gatekeeper) -> None:
    result = gatekeeper.parse("the payment failure logs ")

    assert result.chunks[0].head == "logs"


def test_every_chunk_is_tagged_speculative_read_only(gatekeeper) -> None:
    result = gatekeeper.parse("Check the payment failure logs in Razorpay ")

    assert result.chunks
    assert all(chunk.speculative_read_only is True for chunk in result.chunks)
    assert all(chunk.is_phrase for chunk in result.chunks)


def test_blank_input_returns_no_chunks(gatekeeper) -> None:
    assert gatekeeper.parse("").chunks == ()
    assert gatekeeper.parse("   \n\t ").chunks == ()


def test_chunks_of_kind_filters_by_constituent_type(gatekeeper) -> None:
    text = "Check the payment failure logs in Razorpay "

    assert _texts(gatekeeper.chunks_of_kind(text, PP)) == ["in Razorpay"]
    assert _texts(gatekeeper.chunks_of_kind(text, NP)) == ["the payment failure logs"]


def test_every_parse_finishes_inside_the_latency_budget(gatekeeper) -> None:
    text = (
        "Compare the attention sink mechanism in StreamingLLM "
        "with PagedAttention in vLLM"
    )

    latencies = [gatekeeper.parse(text).latency_ms for _ in range(25)]

    assert max(latencies) < GATEKEEPER_BUDGET_MS
    assert all(gatekeeper.parse(text).within_budget for _ in range(5))


def test_chunk_dataclass_defaults_to_read_only() -> None:
    chunk = Chunk(text="in Razorpay", kind=PP, start=0, end=11, head="in")

    assert chunk.speculative_read_only is True
    assert chunk.is_phrase is True


def test_default_gatekeeper_is_a_singleton() -> None:
    assert default_gatekeeper() is default_gatekeeper()


def test_chunk_text_helper_matches_the_default_gatekeeper() -> None:
    expected = default_gatekeeper().parse("in Razorpay").chunks

    assert chunk_text("in Razorpay") == expected
