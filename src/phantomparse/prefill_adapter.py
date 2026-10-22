"""Incremental prefill adapter for local inference servers.

PhantomParse warms the model's Key-Value (KV) cache while the user is still
typing, so the only work left on ``Enter`` is the verification step and a tiny
tail prefill. Two backends implement that:

``http``
    Stream committed prefixes to a local Ollama or llama.cpp server over its
    HTTP API. Requests go to ``{endpoint}/api/generate`` with ``num_predict: 1``
    so the server prefills the prompt and emits a single token.
``emulator``
    A deterministic, high-precision latency model used when no local server is
    reachable. It bills ``tokens / prefill_tokens_per_second`` and never touches
    the network, which keeps tests and CI hermetic.

``auto`` (the default) probes the endpoint once and degrades to the emulator.

Backspacing invalidates cache state, so :meth:`PrefillAdapter.truncate` drops
everything after the edited character before the next prefix is warmed.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from phantomparse.config import Config

TOKENS_PER_WORD = 1.35
RETRIEVAL_BASE_MS = 420.0
NETWORK_ROUND_TRIP_MS = 260.0
PROBE_TIMEOUT_S = 0.25
REQUEST_TIMEOUT_S = 0.5

BACKEND_AUTO = "auto"
BACKEND_HTTP = "http"
BACKEND_EMULATOR = "emulator"
BACKENDS = (BACKEND_AUTO, BACKEND_HTTP, BACKEND_EMULATOR)

Transport = Callable[[str, Dict[str, Any], float], None]


def estimate_tokens(text: str) -> int:
    """Deterministic Byte-Pair-Encoding (BPE) style token estimate."""
    words = [word for word in text.split() if word]
    if not words:
        return 0
    return int(math.ceil(len(words) * TOKENS_PER_WORD))


def _urllib_transport(url: str, payload: Dict[str, Any], timeout: float) -> None:
    """POST ``payload`` as JSON and drain the response."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def probe_endpoint(endpoint: str, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Return True when a local inference server answers on ``endpoint``."""
    url = endpoint.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return bool(response.status == 200)
    except Exception:
        return False


@dataclass(frozen=True)
class PrefillOutcome:
    """Result of warming the KV cache with one committed prefix."""

    text: str
    tokens: int
    cached_tokens: int
    latency_ms: float
    backend: str
    truncated_tokens: int = 0


class PrefillAdapter:
    """Thread-safe KV-cache warmer with a deterministic emulator fallback."""

    def __init__(
        self,
        config: Optional[Config] = None,
        backend: str = BACKEND_AUTO,
        transport: Optional[Transport] = None,
        probe: Optional[Callable[[], bool]] = None,
        request_timeout: float = REQUEST_TIMEOUT_S,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"Unknown prefill backend: {backend}")
        self.config = config or Config()
        self._requested = backend
        self._transport = transport or _urllib_transport
        self._probe = probe or (lambda: probe_endpoint(self.config.endpoint))
        self._request_timeout = request_timeout
        self._backend: Optional[str] = None
        self._cached_text = ""
        self._cached_tokens = 0
        self._warm_count = 0
        self._truncated_tokens = 0
        self._lock = threading.RLock()

    @property
    def backend(self) -> str:
        """Resolved backend name; probing happens on first use."""
        with self._lock:
            if self._backend is None:
                self._backend = self._resolve_backend()
            return self._backend

    def _resolve_backend(self) -> str:
        if self._requested == BACKEND_EMULATOR:
            return BACKEND_EMULATOR
        try:
            reachable = bool(self._probe())
        except Exception:
            reachable = False
        return BACKEND_HTTP if reachable else BACKEND_EMULATOR

    @property
    def cached_tokens(self) -> int:
        with self._lock:
            return self._cached_tokens

    @property
    def cached_text(self) -> str:
        with self._lock:
            return self._cached_text

    @property
    def warm_count(self) -> int:
        with self._lock:
            return self._warm_count

    @property
    def truncated_tokens(self) -> int:
        with self._lock:
            return self._truncated_tokens

    def warm(self, text: str) -> PrefillOutcome:
        """Prefill ``text`` into the KV cache and report the cost."""
        cleaned = text.strip()
        tokens = estimate_tokens(cleaned)
        backend = self.backend
        started = time.perf_counter()
        if backend == BACKEND_HTTP and cleaned:
            payload = {
                "model": self.config.model,
                "prompt": cleaned,
                "stream": False,
                "options": {"num_predict": 1, "temperature": 0.0},
            }
            url = self.config.endpoint.rstrip("/") + "/api/generate"
            try:
                self._transport(url, payload, self._request_timeout)
            except Exception:
                with self._lock:
                    self._backend = BACKEND_EMULATOR
                backend = BACKEND_EMULATOR
        if backend == BACKEND_HTTP:
            latency_ms = (time.perf_counter() - started) * 1000.0
        else:
            latency_ms = self._emulated_latency_ms(tokens)
        with self._lock:
            self._cached_text = cleaned
            self._cached_tokens = tokens
            self._warm_count += 1
            return PrefillOutcome(
                text=cleaned,
                tokens=tokens,
                cached_tokens=tokens,
                latency_ms=latency_ms,
                backend=backend,
            )

    def truncate(self, char_index: int) -> int:
        """Drop everything at or after ``char_index``; return dropped tokens.

        ``char_index`` is an absolute keystroke-stream index, so a backspace in
        the trailing pending fragment is a no-op here: the cached prefix is
        already shorter than the stream.
        """
        with self._lock:
            kept = self._cached_text[: max(0, char_index)]
            kept_tokens = estimate_tokens(kept)
            removed = max(0, self._cached_tokens - kept_tokens)
            self._cached_text = kept
            self._cached_tokens = kept_tokens
            self._truncated_tokens += removed
            return removed

    def reset(self) -> None:
        """Forget all cached state, as starting a new prompt does."""
        with self._lock:
            self._cached_text = ""
            self._cached_tokens = 0

    def saved_ttft_ms(self) -> float:
        """Latency already paid during typing and therefore saved on Enter."""
        with self._lock:
            cached = self._cached_tokens
        if cached <= 0:
            return 0.0
        rate = max(self.config.prefill_tokens_per_second, 1.0)
        return RETRIEVAL_BASE_MS + cached / rate * 1000.0

    def _emulated_latency_ms(self, tokens: int) -> float:
        rate = max(self.config.prefill_tokens_per_second, 1.0)
        return tokens / rate * 1000.0
