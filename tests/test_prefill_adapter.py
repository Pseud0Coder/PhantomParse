"""Tests for the incremental prefill adapter and its emulator fallback."""

from __future__ import annotations

import pytest

from phantomparse.config import Config
from phantomparse.prefill_adapter import (
    BACKEND_EMULATOR,
    BACKEND_HTTP,
    RETRIEVAL_BASE_MS,
    PrefillAdapter,
    estimate_tokens,
    probe_endpoint,
)

PROMPT = "Write a unit test for the Firebase auth middleware."
PROMPT_TOKENS = estimate_tokens(PROMPT)


def emulator(**overrides: float) -> PrefillAdapter:
    """Adapter pinned to the deterministic emulator, never touching the network."""
    return PrefillAdapter(Config(**overrides), backend=BACKEND_EMULATOR)


class RecordingTransport:
    """Stand-in for the HTTP transport that records every prefill request."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url: str, payload: dict, timeout: float) -> None:
        self.calls.append((url, payload, timeout))
        if self.fail:
            raise OSError("connection refused")


def test_estimate_tokens_is_zero_for_blank_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \n\t ") == 0


def test_estimate_tokens_scales_with_word_count() -> None:
    assert estimate_tokens("payment") == 2
    assert estimate_tokens("payment failure logs") == 5


def test_emulator_warm_reports_tokens_and_latency() -> None:
    adapter = emulator()
    outcome = adapter.warm(PROMPT)

    assert outcome.backend == BACKEND_EMULATOR
    assert outcome.tokens == PROMPT_TOKENS == 13
    assert outcome.cached_tokens == outcome.tokens
    assert outcome.latency_ms == pytest.approx(PROMPT_TOKENS / 240.0 * 1000.0)
    assert adapter.cached_text == PROMPT
    assert adapter.warm_count == 1


def test_warm_strips_surrounding_whitespace() -> None:
    adapter = emulator()
    adapter.warm("   in Razorpay  ")

    assert adapter.cached_text == "in Razorpay"


def test_backend_auto_falls_back_when_probe_fails() -> None:
    adapter = PrefillAdapter(Config(), backend="auto", probe=lambda: False)

    assert adapter.backend == BACKEND_EMULATOR


def test_backend_auto_uses_http_when_probe_succeeds() -> None:
    transport = RecordingTransport()
    adapter = PrefillAdapter(
        Config(), backend="auto", probe=lambda: True, transport=transport
    )

    outcome = adapter.warm(PROMPT)

    assert outcome.backend == BACKEND_HTTP
    assert len(transport.calls) == 1


def test_http_warm_posts_a_single_token_prefill() -> None:
    transport = RecordingTransport()
    adapter = PrefillAdapter(Config(), backend=BACKEND_HTTP, transport=transport)
    outcome = adapter.warm(PROMPT)

    url, payload, timeout = transport.calls[0]
    assert url.endswith("/api/generate")
    assert payload["prompt"] == PROMPT
    assert payload["options"] == {"num_predict": 1, "temperature": 0.0}
    assert timeout == pytest.approx(0.5)
    assert outcome.backend == BACKEND_HTTP
    assert outcome.tokens == PROMPT_TOKENS


def test_http_failure_demotes_to_emulator() -> None:
    transport = RecordingTransport(fail=True)
    adapter = PrefillAdapter(Config(), backend=BACKEND_HTTP, transport=transport)

    outcome = adapter.warm(PROMPT)

    assert outcome.backend == BACKEND_EMULATOR
    assert adapter.backend == BACKEND_EMULATOR
    assert outcome.latency_ms == pytest.approx(PROMPT_TOKENS / 240.0 * 1000.0)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError):
        PrefillAdapter(Config(), backend="gpu")


def test_truncate_drops_the_tail_of_the_cached_prefix() -> None:
    adapter = emulator()
    adapter.warm("Write a unit test")
    before = adapter.cached_tokens

    removed = adapter.truncate(len("Write a"))

    assert removed == before - estimate_tokens("Write a")
    assert adapter.cached_text == "Write a"
    assert adapter.cached_tokens == estimate_tokens("Write a")
    assert adapter.truncated_tokens == removed


def test_truncate_past_the_end_is_a_no_op() -> None:
    adapter = emulator()
    adapter.warm("Write a unit test")

    assert adapter.truncate(999) == 0
    assert adapter.cached_text == "Write a unit test"


def test_truncate_to_zero_empties_the_cache() -> None:
    adapter = emulator()
    adapter.warm(PROMPT)

    adapter.truncate(0)

    assert adapter.cached_text == ""
    assert adapter.cached_tokens == 0


def test_reset_forgets_the_cached_prefix() -> None:
    adapter = emulator()
    adapter.warm(PROMPT)

    adapter.reset()

    assert adapter.cached_tokens == 0
    assert adapter.cached_text == ""


def test_saved_ttft_is_zero_before_any_prefill() -> None:
    assert emulator().saved_ttft_ms() == 0.0


def test_saved_ttft_includes_retrieval_and_prefill() -> None:
    adapter = emulator()
    adapter.warm(PROMPT)

    expected = RETRIEVAL_BASE_MS + adapter.cached_tokens / 240.0 * 1000.0
    assert adapter.saved_ttft_ms() == pytest.approx(expected)


def test_probe_endpoint_returns_false_for_a_dead_port() -> None:
    assert probe_endpoint("http://127.0.0.1:9", timeout=0.05) is False
