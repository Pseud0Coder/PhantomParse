"""PhantomParse benchmark harness.

Compares execution modes across the bundled typing scenarios on a virtual
timeline, so every run is deterministic and machine independent.

Modes
-----
``baseline``
    Wait for Enter, then retrieve context and prefill the model.
``naive``
    Stream partial prompt prefixes to the LLM while the user types. Fast, but
    abandoned drafts burn paid tokens.
``phantom``
    PhantomParse's own pipeline (added once the gatekeeper and staging buffer
    exist): prefetch locally during typing, verify on Enter, prefill in the
    background.

Latency model (fixed constants, documented for auditability)
------------------------------------------------------------
* ``RETRIEVAL_BASE_MS`` -- cold index open + query fan-out.
* ``RETRIEVAL_PER_TOKEN_MS`` -- per retrieved context token.
* ``NETWORK_ROUND_TRIP_MS`` -- client to inference server handshake.
* prefill time is derived from ``Config.prefill_tokens_per_second``.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
BENCH_DIR = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(SRC_DIR), str(BENCH_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from keystroke_simulator import (  # noqa: E402
    SCENARIOS,
    SimulationResult,
    simulate,
    scenario_by_key,
)
from phantomparse.config import Config, load_config  # noqa: E402

NETWORK_ROUND_TRIP_MS = 260.0
RETRIEVAL_BASE_MS = 420.0
RETRIEVAL_PER_TOKEN_MS = 3.5
TOKENS_PER_WORD = 1.35
CHARS_PER_TOKEN = 4.2

MODE_LABELS = {
    "baseline": "Baseline",
    "naive": "Naive Speculation",
    "phantom": "PhantomParse",
}


@dataclass
class Measurement:
    """One mode/scenario measurement row."""

    mode: str
    scenario: str
    ttft_ms: float
    post_enter_retrieval_ms: float
    wasted_tokens: int
    total_tokens: int
    peak_memory_mb: float

    @property
    def label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)

    @property
    def wasted_pct(self) -> float:
        if self.total_tokens <= 0:
            return 0.0
        return 100.0 * self.wasted_tokens / self.total_tokens


@contextmanager
def traced_memory() -> Iterator[Dict[str, float]]:
    """Track incremental allocation for one measurement, in mebibytes."""
    box = {"peak_mb": 0.0}
    tracemalloc.start()
    try:
        yield box
    finally:
        _, peak = tracemalloc.get_traced_memory()
        box["peak_mb"] = peak / (1024 * 1024)
        tracemalloc.stop()


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (no tokenizer download required)."""
    words = [word for word in text.split() if word]
    return max(1, int(math.ceil(len(words) * TOKENS_PER_WORD)))


def estimate_prefix_tokens(chars: int) -> int:
    """Estimate tokens for a raw character prefix (BPE-like ratio)."""
    if chars <= 0:
        return 0
    return max(1, int(math.ceil(chars / CHARS_PER_TOKEN)))


def prefill_ms(tokens: int, config: Config) -> float:
    rate = max(config.prefill_tokens_per_second, 1.0)
    return tokens / rate * 1000.0


def run_baseline(result: SimulationResult, config: Config) -> Measurement:
    """Retrieval and prefill happen only after Enter is pressed."""
    tokens = estimate_tokens(result.final_text)
    retrieval_ms = RETRIEVAL_BASE_MS + tokens * RETRIEVAL_PER_TOKEN_MS
    ttft = retrieval_ms + prefill_ms(tokens, config) + NETWORK_ROUND_TRIP_MS
    with traced_memory() as box:
        _ = result.final_text
    return Measurement(
        mode="baseline",
        scenario=result.scenario.key,
        ttft_ms=ttft,
        post_enter_retrieval_ms=retrieval_ms,
        wasted_tokens=0,
        total_tokens=tokens,
        peak_memory_mb=box["peak_mb"],
    )


def run_naive(result: SimulationResult, config: Config) -> Measurement:
    """Eagerly stream typed prefixes to the LLM; abandoned drafts are wasted."""
    chunk_chars = 8
    typed = ""
    sent_tokens = 0
    discarded_chars = 0
    chars_since_flush = 0
    with traced_memory() as box:
        for event in result.events:
            if event.kind == "char":
                typed += event.value
                chars_since_flush += 1
                if chars_since_flush >= chunk_chars:
                    sent_tokens += estimate_prefix_tokens(chars_since_flush)
                    chars_since_flush = 0
            elif event.kind == "backspace" and typed:
                typed = typed[:-1]
                discarded_chars += 1
        if chars_since_flush:
            sent_tokens += estimate_prefix_tokens(chars_since_flush)

    wasted = min(sent_tokens, estimate_prefix_tokens(discarded_chars))
    return Measurement(
        mode="naive",
        scenario=result.scenario.key,
        ttft_ms=NETWORK_ROUND_TRIP_MS * 0.55 + 40.0,
        post_enter_retrieval_ms=25.0,
        wasted_tokens=wasted,
        total_tokens=sent_tokens,
        peak_memory_mb=box["peak_mb"],
    )


MODE_RUNNERS: Dict[str, Callable[[SimulationResult, Config], Measurement]] = {
    "baseline": run_baseline,
    "naive": run_naive,
}


def collect_measurements(
    modes: Sequence[str],
    scenario_keys: Sequence[str],
    config: Config,
) -> List[Measurement]:
    rows: List[Measurement] = []
    for mode in modes:
        if mode not in MODE_RUNNERS:
            raise KeyError(f"Unknown mode: {mode}")
        runner = MODE_RUNNERS[mode]
        for key in scenario_keys:
            result = simulate(scenario_by_key(key), wpm=config.typing_wpm)
            rows.append(runner(result, config))
    return rows


def _summary_rows(rows: Sequence[Measurement]) -> List[Dict[str, float]]:
    by_mode: Dict[str, List[Measurement]] = {}
    for row in rows:
        by_mode.setdefault(row.mode, []).append(row)
    baseline_ttft = None
    if "baseline" in by_mode:
        baseline_ttft = statistics.mean(r.ttft_ms for r in by_mode["baseline"])

    summary = []
    for mode, items in by_mode.items():
        mean_ttft = statistics.mean(item.ttft_ms for item in items)
        reduction = 0.0
        if baseline_ttft:
            reduction = 100.0 * (baseline_ttft - mean_ttft) / baseline_ttft
        summary.append(
            {
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "ttft": mean_ttft,
                "retrieval": statistics.mean(
                    item.post_enter_retrieval_ms for item in items
                ),
                "wasted": sum(item.wasted_tokens for item in items),
                "waste_pct": statistics.mean(item.wasted_pct for item in items),
                "memory": max(item.peak_memory_mb for item in items),
                "reduction": reduction,
            }
        )
    return summary


def format_markdown(rows: Sequence[Measurement], config: Config) -> str:
    """Render the benchmark as a Markdown report."""
    lines = [
        "# PhantomParse Benchmark",
        "",
        (
            f"Typing speed: {config.typing_wpm} WPM | TTL: {config.ttl_seconds:.0f}s | "
            f"Similarity threshold: {config.similarity_threshold:.2f} | "
            f"Prefill rate: {config.prefill_tokens_per_second:.0f} tok/s"
        ),
        "",
        "| Mode | Scenario | TTFT (ms) | Post-Enter Retrieval (ms) | Wasted LLM Tokens "
        "| Token Waste (%) | Peak Daemon Memory (MB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.label} | {row.scenario} | {row.ttft_ms:.1f} | "
            f"{row.post_enter_retrieval_ms:.1f} | {row.wasted_tokens} | "
            f"{row.wasted_pct:.1f} | {row.peak_memory_mb:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Mode | Mean TTFT (ms) | Mean Post-Enter Retrieval (ms) | Wasted LLM Tokens "
            "| Mean Waste (%) | TTFT Reduction vs Baseline | Peak Memory (MB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in _summary_rows(rows):
        reduction = "n/a" if item["mode"] == "baseline" else f"{item['reduction']:.1f}%"
        lines.append(
            f"| {item['label']} | {item['ttft']:.1f} | {item['retrieval']:.1f} | "
            f"{item['wasted']:.0f} | {item['waste_pct']:.1f} | {reduction} | "
            f"{item['memory']:.2f} |"
        )

    lines.extend(
        [
            "",
            "> Peak Daemon Memory is the incremental traced allocation of the in-process "
            "pipeline, measured with `tracemalloc`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PhantomParse benchmark harness")
    parser.add_argument("--config", help="Path to a YAML configuration file")
    parser.add_argument(
        "--modes",
        default="baseline,naive",
        help="Comma-separated modes: baseline, naive, phantom",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(scenario.key for scenario in SCENARIOS),
        help="Comma-separated scenario keys",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    scenario_keys = [key.strip() for key in args.scenarios.split(",") if key.strip()]

    rows = collect_measurements(modes, scenario_keys, config)
    print(format_markdown(rows, config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
