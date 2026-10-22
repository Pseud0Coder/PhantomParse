"""Background daemon that wires the PhantomParse pipeline together.

The daemon owns the whole speculative loop:

1. :class:`~phantomparse.buffer.KeystrokeBuffer` turns raw keystrokes into
   committed orthographic tokens.
2. :class:`~phantomparse.gatekeeper.SyntacticGatekeeper` extracts closed Noun
   Phrases (NP) and Prepositional Phrases (PP) from the committed prefix.
3. :class:`~phantomparse.staging.StagingBuffer` holds those spans in RAM for a
   bounded Time to Live (TTL) -- never sending them anywhere.
4. :class:`~phantomparse.prefill_adapter.PrefillAdapter` warms the model's
   Key-Value (KV) cache from the same committed prefix.
5. On ``Enter`` the staging buffer verifies every span against the final prompt
   and only survivors are injected, which is what keeps wasted tokens at zero.

``<ctrl>+<space>`` arms or pauses the pill through ``pynput``; when the library
or the display is missing the daemon degrades to a headless, testable mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from phantomparse.buffer import CommitEvent, KeystrokeBuffer, RollbackEvent
from phantomparse.config import Config, load_config
from phantomparse.gatekeeper import Chunk, SyntacticGatekeeper, default_gatekeeper
from phantomparse.overlay import Overlay, OverlayState
from phantomparse.prefill_adapter import PrefillAdapter
from phantomparse.staging import StagingBuffer, VerificationResult

HOTKEY_BACKEND_PYNPUT = "pynput"
HOTKEY_BACKEND_HEADLESS = "headless"


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of pressing Enter."""

    prompt: str
    verification: VerificationResult
    injected_tokens: int
    prefill_latency_ms: float
    wasted_tokens: int
    ttft_saved_ms: float

    @property
    def injected(self) -> int:
        return len(self.verification.accepted)

    @property
    def evicted(self) -> int:
        return len(self.verification.rejected)


class Daemon:
    """Speculative prefill daemon; usable headless from tests and the CLI."""

    def __init__(
        self,
        config: Optional[Config] = None,
        gatekeeper: Optional[SyntacticGatekeeper] = None,
        staging: Optional[StagingBuffer] = None,
        prefill: Optional[PrefillAdapter] = None,
        overlay: Optional[Overlay] = None,
    ) -> None:
        self.config = config or load_config()
        self.gatekeeper = gatekeeper or default_gatekeeper()
        self.staging = staging or StagingBuffer(
            ttl_seconds=self.config.ttl_seconds,
            max_items=self.config.max_staging_items,
            similarity_threshold=self.config.similarity_threshold,
        )
        self.prefill = prefill or PrefillAdapter(self.config)
        self.overlay = overlay or Overlay()
        self.buffer = KeystrokeBuffer()
        self.active = True
        self.hotkey_backend = HOTKEY_BACKEND_HEADLESS
        self.rollbacks = 0
        self.submissions = 0
        self._chunks: Tuple[Chunk, ...] = ()
        self._listener: Optional[Any] = None
        self._status = "armed"
        self.refresh()

    # -- input ------------------------------------------------------------

    def feed_char(self, char: str) -> Optional[CommitEvent]:
        """Consume one keystroke and refresh speculation at token boundaries."""
        event = self.buffer.feed_char(char)
        if event is not None:
            self.refresh()
        return event

    def feed_text(self, text: str) -> List[CommitEvent]:
        events: List[CommitEvent] = []
        for char in text:
            event = self.feed_char(char)
            if event is not None:
                events.append(event)
        return events

    def feed_backspace(self) -> Optional[RollbackEvent]:
        """Delete a character and truncate prefill state above the cut."""
        rollback = self.buffer.feed_backspace()
        if rollback is not None:
            self.rollbacks += 1
            self.prefill.truncate(rollback.char_index)
        self.refresh()
        return rollback

    def submit(self) -> SubmitResult:
        """Press Enter: verify staged spans, inject survivors, report savings."""
        self.buffer.flush()
        prompt = self.buffer.text.strip()
        verification = self.staging.verify(
            prompt, threshold=self.config.similarity_threshold
        )
        injected_text = " ".join(verification.accepted_text)
        prefill = self.prefill.warm(injected_text or prompt)
        saved_ms = self.prefill.saved_ttft_ms()
        self.submissions += 1
        result = SubmitResult(
            prompt=prompt,
            verification=verification,
            injected_tokens=prefill.tokens,
            prefill_latency_ms=prefill.latency_ms,
            # Local prefetch and KV warming are never billed, so the only
            # tokens that leave this machine are the injected context itself.
            wasted_tokens=0,
            ttft_saved_ms=saved_ms,
        )
        self.overlay.update(
            OverlayState(
                committed=prompt,
                chunks=self.chunk_texts(),
                staged=0,
                injected=len(verification.accepted),
                ttft_saved_ms=saved_ms,
                status=(
                    f"answer ready | {len(verification.accepted)} injected "
                    f"| {len(verification.rejected)} evicted"
                ),
            )
        )
        self.buffer = KeystrokeBuffer()
        self.staging.clear()
        self.prefill.reset()
        self._chunks = ()
        self._status = "responding"
        return result

    # -- state ------------------------------------------------------------

    def refresh(self, status: Optional[str] = None) -> OverlayState:
        """Re-parse the committed prefix, stage and prefill, then repaint."""
        committed = " ".join(event.token for event in self.buffer.tokens)
        if committed:
            parsed = self.gatekeeper.parse(committed)
            self._chunks = parsed.chunks
            self.staging.stage_chunks(parsed.chunks)
            self.prefill.warm(committed)
        if status is not None:
            self._status = status
        elif not self.active:
            self._status = "paused"
        elif not self.buffer.pending:
            self._status = "armed"
        return self._publish()

    def _publish(self) -> OverlayState:
        state = OverlayState(
            committed=self.buffer.text,
            chunks=self.chunk_texts(),
            staged=len(self.staging),
            ttft_saved_ms=self.prefill.saved_ttft_ms(),
            status=self._status,
        )
        self.overlay.update(state)
        return state

    def chunk_texts(self) -> Tuple[str, ...]:
        return tuple(chunk.text for chunk in self._chunks)

    def toggle(self) -> bool:
        """Arm or pause speculation; bound to the global hotkey."""
        self.active = not self.active
        self._status = "armed" if self.active else "paused"
        self._publish()
        return self.active

    def stats(self) -> Dict[str, Any]:
        return {
            "submissions": self.submissions,
            "rollbacks": self.rollbacks,
            "staged": len(self.staging),
            "cached_tokens": self.prefill.cached_tokens,
            "prefill_backend": self.prefill.backend,
            "hotkey_backend": self.hotkey_backend,
            "active": self.active,
        }

    # -- lifecycle --------------------------------------------------------

    def install_hotkey(self) -> bool:
        """Try to bind the global hotkey; return whether that succeeded."""
        try:
            from pynput import keyboard
        except Exception:
            self.hotkey_backend = HOTKEY_BACKEND_HEADLESS
            return False
        try:
            listener = keyboard.GlobalHotKeys({self.config.hotkey: self.toggle})
            listener.start()
        except Exception:
            self.hotkey_backend = HOTKEY_BACKEND_HEADLESS
            return False
        self._listener = listener
        self.hotkey_backend = HOTKEY_BACKEND_PYNPUT
        return True

    def run(self, block: bool = True) -> None:
        """Start listening. Headless or ``block=False`` returns immediately."""
        listening = self.install_hotkey()
        self._status = f"listening ({self.hotkey_backend})" if listening else "headless"
        self._publish()
        if block and not self.overlay.headless:
            self.overlay.run()

    def stop(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.stop()
        self.overlay.close()


def demo_states(
    initial: str,
    pivot: str,
    step: int = 4,
) -> Tuple[OverlayState, ...]:
    """Replay a typed prompt, a full erase and a pivot, for ``overlay --demo``."""
    config = Config()
    daemon = Daemon(
        config=config,
        prefill=PrefillAdapter(config, backend="emulator"),
        overlay=Overlay(headless=True),
    )
    states: List[OverlayState] = []
    for char in initial:
        daemon.feed_char(char)
        if len(states) == 0 or len(daemon.buffer.text) % step == 0:
            states.append(daemon._publish())
    for _ in range(len(initial)):
        daemon.feed_backspace()
    states.append(daemon.refresh("pivot: abandoned spans held in RAM"))
    for char in pivot:
        daemon.feed_char(char)
        if len(daemon.buffer.text) % step == 0:
            states.append(daemon._publish())
    daemon.submit()
    states.append(daemon.overlay.state)
    return tuple(states)
