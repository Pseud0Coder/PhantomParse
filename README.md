# PhantomParse

**PhantomParse** is a system input plugin and background daemon that prefetches and stages
context while you type, then verifies it locally the instant you press `Enter`.

- **Near-zero TTFT** (Time to First Token): retrieval and prefill work happens during typing,
  not after submission.
- **Zero wasted LLM tokens**: speculative context is held in local RAM and only injected if it
  still matches the final, completed prompt.

## Why

A normal prompt spends its first seconds retrieving context and prefilling the model. The user
has already been typing for those seconds. PhantomParse moves that work into the typing window
without ever shipping abandoned draft context to a paid LLM.

## Architecture

```text
 keystrokes ──▶ buffer.py ──▶ gatekeeper.py ──▶ staging.py ──▶ prefill_adapter.py
               (boundary      (NP/PP chunk      (TTL ring       (speculative KV
                commits,       detection,        buffer +        cache warming)
                rollbacks)     < 15 ms)          late-bound
                                                verifier)
                                                    │
                                          Enter ────┴──▶ verified context ──▶ LLM
```

| Module | Role | Linguistics analogue |
| --- | --- | --- |
| `buffer.py` | Emits committed words/rollbacks on boundaries | Orthographic segmentation, morpheme boundaries |
| `gatekeeper.py` | Detects closed noun/prepositional phrases | Shallow chunking / constituency parsing |
| `staging.py` | TTL ring buffer + cosine relevance gate | Working memory, pragmatic filtering |
| `prefill_adapter.py` | Warms the KV cache with committed tokens | Speculative prefix encoding |

## Quickstart

```bash
pip install -r requirements.txt

# Run the benchmark harness (Baseline vs. Naive Speculation vs. PhantomParse)
python benchmarks/run_bench.py

# Run the test suite
pytest tests/

# Lint
flake8 src/ tests/ benchmarks/
```

Once installed as a package (`pip install -e .`), the CLI is available:

```bash
phantomparse start            # background daemon + hotkey listener
phantomparse overlay --demo   # render the floating pill overlay demo
phantomparse bench            # benchmark suite
phantomparse config           # show effective configuration
```

## Configuration

PhantomParse reads `~/.config/phantomparse/config.yaml`, falling back to
[`config/default_config.yaml`](config/default_config.yaml). Missing files fall back to built-in
defaults, so a fresh install works with no configuration.

```yaml
hotkey: "<ctrl>+<space>"
ttl_seconds: 30.0
similarity_threshold: 0.75
max_staging_items: 64
model: llama3
endpoint: http://localhost:11434
prefill_tokens_per_second: 240.0
typing_wpm: 60
max_ram_mb: 50.0
max_chunk_tokens: 24
```

## Status

Phase 1. All runtime heavy lifting is local: the gatekeeper degrades to a deterministic
rule-based parser when SpaCy is unavailable, and the prefill adapter degrades to a latency
emulator when no local inference server is running.
