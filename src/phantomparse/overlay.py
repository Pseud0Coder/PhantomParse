"""Wispr-style floating pill overlay.

A borderless, always-on-top ``tkinter`` pill that shows what the daemon is doing
while the user types: the active syntactic chunks, how many context spans are
staged in RAM, and the Time to First Token (TTFT) already saved in milliseconds.

The pill is deliberately tiny and chrome-free, in the synthwave palette:
charcoal ``#12131A``, neon cyan ``#00F5D4`` and electric purple ``#9B5DE5``.

``headless=True`` -- also implied when no display or no ``tkinter`` is
available -- keeps the exact same state machine but renders nothing, so tests
and the GIF recorder can drive it deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

CHARCOAL = "#12131A"
CYAN = "#00F5D4"
PURPLE = "#9B5DE5"
TEXT = "#E6EAF2"
DIM = "#6C7285"

PILL_WIDTH = 470
PILL_HEIGHT = 104
CORNER_RADIUS = 26
DEFAULT_INTERVAL_MS = 140


@dataclass(frozen=True)
class OverlayState:
    """Everything the pill displays at one instant."""

    committed: str = ""
    chunks: Tuple[str, ...] = ()
    staged: int = 0
    injected: int = 0
    ttft_saved_ms: float = 0.0
    status: str = "idle"


def pill_lines(state: OverlayState, width: int = 46) -> List[str]:
    """Deterministic pill text, shared by the tkinter and headless paths."""
    chunks = " | ".join(state.chunks) if state.chunks else "-- no closed phrases yet"
    if len(chunks) > width:
        chunks = "..." + chunks[-(width - 3):]
    return [
        f"PHANTOMPARSE   {state.status}",
        f"chunks {len(state.chunks)}   staged {state.staged}   "
        f"saved {state.ttft_saved_ms:.0f} ms",
        chunks,
    ]


def tkinter_available() -> bool:
    """Return True when a Tk display can actually be opened."""
    try:
        import tkinter
    except Exception:
        return False
    try:
        root = tkinter.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class Overlay:
    """Floating pill view; in headless mode it only records state."""

    def __init__(self, headless: Optional[bool] = None) -> None:
        self.state = OverlayState()
        self.headless = (not tkinter_available()) if headless is None else bool(headless)
        self._root = None
        self._canvas = None
        self._frames: List[OverlayState] = []
        self._interval = DEFAULT_INTERVAL_MS
        self._closed = False

    def update(self, state: OverlayState) -> List[str]:
        """Apply ``state`` and return the rendered pill lines."""
        self.state = state
        lines = pill_lines(state)
        if not self.headless and self._canvas is not None:
            self._redraw(lines)
        return lines

    def run(
        self,
        frames: Sequence[OverlayState] = (),
        interval_ms: int = DEFAULT_INTERVAL_MS,
    ) -> None:
        """Show the pill. With ``frames`` it cycles them, then holds the last."""
        self._frames = list(frames)
        self._interval = max(0, int(interval_ms))
        if self.headless:
            for frame in self._frames:
                self.update(frame)
            return
        self._ensure_window()
        self._advance(0)
        self._root.mainloop()

    def close(self) -> None:
        self._closed = True
        if self._root is not None:
            self._root.destroy()
            self._root = None
            self._canvas = None

    def _ensure_window(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg=CHARCOAL)
        x = (root.winfo_screenwidth() - PILL_WIDTH) // 2
        y = root.winfo_screenheight() - PILL_HEIGHT - 120
        root.geometry(f"{PILL_WIDTH}x{PILL_HEIGHT}+{x}+{y}")
        canvas = tk.Canvas(
            root,
            width=PILL_WIDTH,
            height=PILL_HEIGHT,
            bg=CHARCOAL,
            highlightthickness=0,
        )
        canvas.pack()
        root.bind("<Escape>", lambda _event: self.close())
        self._root = root
        self._canvas = canvas
        self._redraw(pill_lines(self.state))

    def _redraw(self, lines: Sequence[str]) -> None:
        canvas = self._canvas
        if canvas is None:
            return
        canvas.delete("all")
        _rounded_rect(
            canvas,
            2,
            2,
            PILL_WIDTH - 2,
            PILL_HEIGHT - 2,
            CORNER_RADIUS,
            fill=CHARCOAL,
            outline=CYAN,
            width=2,
        )
        canvas.create_oval(24, 25, 36, 37, fill=PURPLE, outline="")
        canvas.create_text(
            46,
            31,
            text=lines[0],
            anchor="w",
            fill=CYAN,
            font=("Consolas", 11, "bold"),
        )
        canvas.create_text(
            PILL_WIDTH - 26,
            31,
            text=f"saved {self.state.ttft_saved_ms:.0f} ms",
            anchor="e",
            fill=PURPLE,
            font=("Consolas", 10),
        )
        canvas.create_text(
            26,
            60,
            text=lines[1],
            anchor="w",
            fill=TEXT,
            font=("Consolas", 10),
        )
        canvas.create_text(
            26,
            82,
            text=lines[2],
            anchor="w",
            fill=DIM,
            font=("Consolas", 9),
        )

    def _advance(self, index: int) -> None:
        if self._closed or self._root is None or not self._frames:
            return
        if index >= len(self._frames):
            return
        self.update(self._frames[index])
        self._root.after(self._interval, self._advance, index + 1)


def _rounded_rect(
    canvas: object,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius: float,
    **kwargs: object,
) -> int:
    """Draw a smooth rounded rectangle on a tkinter canvas."""
    points = [
        x0 + radius, y0,
        x1 - radius, y0,
        x1, y0,
        x1, y0 + radius,
        x1, y1 - radius,
        x1, y1,
        x1 - radius, y1,
        x0 + radius, y1,
        x0, y1,
        x0, y1 - radius,
        x0, y0 + radius,
        x0, y0,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)  # type: ignore[attr-defined]
