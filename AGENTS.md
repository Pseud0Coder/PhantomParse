# AGENTS.md

Guidance for AI agents working in this repository.

## Project Overview
- **Project Name:** PhantomParse
- **Version:** 1.0.0
- **Purpose:** A system input plugin and background daemon that prefetches and stages context as you type, eliminating LLM Time-to-First-Token (TTFT) latency while ensuring zero wasted LLM tokens through local late-bound verification on `Enter`.
- **Stack:** Python 3.11+, PyTorch / ONNX (for local NER and lightweight embedding scorers like `bge-small` / `SpaCy`), pytest for testing, and custom benchmarking harnesses for latency/token tracking.
- **Entry point:** `benchmarks/run_bench.py` (initial telemetry and baseline simulation harness).

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start development / run baseline simulator
python benchmarks/run_bench.py

# Run tests
pytest tests/

# Lint / format
flake8 src/ tests/ benchmarks/
```

## Repository Structure

```text
# <root>/
#   src/          Application source (keystroke buffer, syntactic gatekeeper, staging ring buffer)
#   tests/        Test suite for components and state transitions
#   benchmarks/   Keystroke simulator and TTFT performance benchmarks
#   docs/         Documentation, specs, and feature GIFs
```

## Conventions

- Follow the existing style and structure of surrounding code.
- Keep changes focused; do not refactor unrelated code.
- Reuse existing utilities and patterns instead of introducing new abstractions.
- Add comments only where the logic is not self-evident.
- Update tests and documentation when behavior changes.

## Rules for Agents

1. Read the relevant files before editing them.
2. Do not create new files unless necessary; prefer editing existing ones.
3. Never commit secrets, credentials, or environment files.
4. Do not run destructive git commands (force push, hard reset, clean -f) unless explicitly asked.
5. Verify changes by running the build, tests, and linter where available.
6. Report blockers instead of guessing when requirements are ambiguous.

## Testing

- Unit tests are located in `tests/` and run using `pytest`. 
- Performance benchmarks are located in `benchmarks/` and must be executed to output a comparative markdown table before any feature PR is finalized.
- No external mock services or cloud API keys are required for local test suites; all speculative lookups run against local stub indices or mock vector stores during testing.

## Environment

- Required environment variables: `PYTHONPATH=.` (documented in `.env.example` when cloud LLM integrations are introduced).
- Version requirements: Python >= 3.11, Git.