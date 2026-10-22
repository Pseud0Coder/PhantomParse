"""Command-line entry point for PhantomParse.

Four subcommands mirror the daemon's lifecycle:

``phantomparse start``
    Attach the speculative pipeline to the terminal: every keystroke is buffered,
    parsed, staged and prefilled, while the pill repaints the live state.
``phantomparse overlay --demo``
    Play the recorded pivot scenario through the floating pill.
``phantomparse bench``
    Delegate to the benchmark harness that compares the three modes.
``phantomparse config``
    Inspect, seed or override the layered YAML configuration.

The module keeps all side effects behind ``main`` so tests can import it freely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

from phantomparse.config import (
    BUNDLED_CONFIG_PATH,
    USER_CONFIG_PATH,
    Config,
    ensure_user_config,
    load_config,
)
from phantomparse.daemon import Daemon, SubmitResult, demo_states
from phantomparse.overlay import Overlay, OverlayState, pill_lines
from phantomparse.prefill_adapter import PrefillAdapter

SUBMIT_SEPARATOR = "-" * 62


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phantomparse",
        description="Speculative context staging that removes LLM Time to First Token.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Run the speculative daemon")
    start.add_argument("--config", help="Path to a YAML configuration file")
    start.add_argument("--headless", action="store_true", help="Never open the pill")
    start.add_argument("--no-hotkey", action="store_true", help="Skip the global hotkey")
    start.add_argument("--demo", action="store_true", help="Replay the pivot scenario")
    start.add_argument("--scenario", default="pivot", help="Scenario key for --demo")
    start.add_argument("--step", type=int, default=4, help="Demo frames every N chars")
    start.add_argument("--interval-ms", type=int, default=140, help="Demo frame interval")
    start.set_defaults(handler=_cmd_start)

    overlay = subparsers.add_parser("overlay", help="Show the floating pill")
    overlay.add_argument("--demo", action="store_true", help="Cycle demo frames")
    overlay.add_argument("--headless", action="store_true", help="Print instead of drawing")
    overlay.add_argument("--scenario", default="pivot", help="Scenario key for --demo")
    overlay.add_argument("--step", type=int, default=4, help="Demo frames every N chars")
    overlay.add_argument("--interval-ms", type=int, default=140, help="Frame interval")
    overlay.set_defaults(handler=_cmd_overlay)

    bench = subparsers.add_parser("bench", help="Run the latency and waste benchmarks")
    bench.add_argument("--config", help="Path to a YAML configuration file")
    bench.add_argument("--modes", default="baseline,naive,phantom")
    bench.add_argument("--scenarios", default="linear,pivot,multi_clause")
    bench.set_defaults(handler=_cmd_bench)

    config = subparsers.add_parser("config", help="Inspect or edit configuration")
    config.add_argument("--path", help="Read configuration from this file instead")
    config.add_argument("--init", action="store_true", help="Write the user config if absent")
    config.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Persist an override into the user config (repeatable)",
    )
    config.set_defaults(handler=_cmd_config)
    return parser


# -- subcommands ----------------------------------------------------------


def _cmd_start(args: argparse.Namespace) -> int:
    if args.demo:
        return _run_demo(args.scenario, args.step, args.headless, args.interval_ms)

    config = load_config(args.config)
    daemon = Daemon(
        config=config,
        prefill=PrefillAdapter(config),
        overlay=Overlay(headless=True if args.headless else None),
    )
    if args.no_hotkey:
        daemon.refresh("terminal attach")
    else:
        daemon.run(block=False)
    daemon.overlay.open()
    print(
        f"PhantomParse {daemon.stats()['prefill_backend']} prefill | "
        f"hotkey {config.hotkey} ({daemon.hotkey_backend}) | "
        f"threshold {config.similarity_threshold}"
    )
    print("Type to speculate, Enter to submit, Esc to quit.")
    try:
        while True:
            key = _read_key()
            if key is None:
                break
            if key == "\n":
                _print_submission(daemon.submit())
            elif key == "\b":
                daemon.feed_backspace()
            else:
                event = daemon.feed_char(key)
                if event is not None:
                    _echo_state(daemon.overlay.state, daemon.overlay.headless)
            daemon.overlay.pump()
    except KeyboardInterrupt:
        print()
    finally:
        daemon.stop()
    return 0


def _cmd_overlay(args: argparse.Namespace) -> int:
    return _run_demo(args.scenario, args.step, args.headless, args.interval_ms)


def _cmd_bench(args: argparse.Namespace) -> int:
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmarks import run_bench

    argv: List[str] = ["--modes", args.modes, "--scenarios", args.scenarios]
    if args.config:
        argv.extend(["--config", args.config])
    return run_bench.main(argv)


def _cmd_config(args: argparse.Namespace) -> int:
    if args.overrides:
        merged = load_config(args.path).to_dict()
        for override in args.overrides:
            key, _, raw = override.partition("=")
            key = key.strip()
            if not key or not raw:
                print(f"ignoring malformed override: {override}", file=sys.stderr)
                continue
            if key not in Config().to_dict():
                print(f"unknown configuration key: {key}", file=sys.stderr)
                return 2
            merged[key] = raw.strip()
        saved = Config.from_mapping(merged).save(USER_CONFIG_PATH)
        print(f"wrote {saved}")

    if args.init:
        print(f"user config: {ensure_user_config()}")

    effective = load_config(args.path)
    print(f"# bundled: {BUNDLED_CONFIG_PATH}")
    print(f"# user:    {USER_CONFIG_PATH}")
    print(f"# active:  {args.path or USER_CONFIG_PATH}")
    print(yaml.safe_dump(effective.to_dict(), sort_keys=True).rstrip())
    return 0


# -- demo ----------------------------------------------------------------


def _run_demo(scenario: str, step: int, headless: bool, interval_ms: int) -> int:
    from benchmarks.keystroke_simulator import scenario_by_key

    try:
        case = scenario_by_key(scenario)
    except KeyError:
        print(f"unknown scenario: {scenario}", file=sys.stderr)
        return 2
    frames = demo_states(case.initial_text, case.pivot_text, step=max(1, step))
    overlay = Overlay(headless=True if headless else None)
    if overlay.headless:
        for frame in frames:
            _echo_state(frame, headless=True)
        _print_submission_state(frames[-1])
        return 0
    overlay.run(frames, interval_ms=max(1, interval_ms))
    return 0


# -- terminal plumbing ---------------------------------------------------


def _read_key() -> Optional[str]:
    """Read one keystroke; return None on Escape, Ctrl-C or end of input."""
    if sys.stdin is None or sys.stdin.closed:
        return None
    if sys.platform.startswith("win"):
        import msvcrt

        char = msvcrt.getwch()
    else:
        import termios
        import tty

        descriptor = sys.stdin.fileno()
        saved = termios.tcgetattr(descriptor)
        try:
            tty.setcbreak(descriptor)
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
    if char in ("\x1b", "\x03", ""):
        return None
    if char in ("\r", "\n"):
        return "\n"
    if char in ("\x08", "\x7f"):
        return "\b"
    return char


def _echo_state(state: OverlayState, headless: bool) -> None:
    if not headless:
        return
    header, metrics, chunks = pill_lines(state)
    print(f"  {header}")
    print(f"  {metrics}")
    print(f"  {chunks}")


def _print_submission(result: SubmitResult) -> None:
    print(SUBMIT_SEPARATOR)
    print(f"prompt   {result.prompt}")
    print(
        f"injected {result.injected} | evicted {result.evicted} | "
        f"wasted {result.wasted_tokens} tokens | saved {result.ttft_saved_ms:.0f} ms"
    )
    print(f"prefill  {result.prefill_latency_ms:.1f} ms | {result.injected_tokens} tokens")
    print(SUBMIT_SEPARATOR)


def _print_submission_state(state: OverlayState) -> None:
    header, metrics, chunks = pill_lines(state)
    print(SUBMIT_SEPARATOR)
    print(header)
    print(metrics)
    print(chunks)
    print(SUBMIT_SEPARATOR)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
