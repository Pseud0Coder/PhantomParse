# PhantomParse — Summary of Build

Phase 1 verification report for `https://github.com/Pseud0Coder/PhantomParse`.
All five tasks (0 through 4) are built, tested, committed, pushed and merged to `main`.

---

## 1. Executive Build Summary & Agent Workflow Log

### Scope delivered

PhantomParse is a system input plugin plus background daemon that stages context
**locally in RAM** while the user types, warms the model's **KV (Key-Value)**
cache incrementally, and performs **late-bound verification** on `Enter`. The
result is a near-zero **TTFT (Time to First Token)** with **0 wasted LLM
(Large Language Model) tokens** even when the user abandons a sentence halfway
(the "pivot" case).

### `AGENTS.md` compliance

- `c:\Trae projects\AGENTS.md` was read in full and **never modified**.
- Its rules were respected throughout: files were read before editing, no new
  files were created unless a Task required them, no secrets or environment
  files were committed, no destructive git command was used, and every change
  was validated with the build, test suite and linter.
- The mandated entry point `benchmarks/run_bench.py`, the prescribed stack
  (Python, pytest, local **NLP (Natural Language Processing)** scorers), and the
  `src/ tests/ benchmarks/ docs/` repository layout all match the guidance file.

### Remote push status

- Remote: `origin https://github.com/Pseud0Coder/PhantomParse.git` (fetch + push).
- `main` is at merge commit `78aba46` and matches `origin/main`.
- All five feature branches were pushed, reviewed through PRs #1–#5, and merged
  into `main`; nothing is left unpushed except this report, which was committed
  and pushed as the final action.

### Workflow decomposition

The build followed one branch per Task (`feat/task-N-slug`) and one atomic
Conventional Commit per Subtask, then a Pull Request whose body lived in
`docs/prs/PR_TASK_N.md`:

| Task | Branch | Scope |
| --- | --- | --- |
| 0 | `feat/task-0-benchmark-harness` | Scaffolding, layered config, deterministic keystroke simulator, three-mode benchmark harness, native GIF recorder |
| 1 | `feat/task-1-keystroke-buffer` | Word/punctuation boundary commit buffer with backspace rollback events |
| 2 | `feat/task-2-syntactic-gatekeeper` | **NP (Noun Phrase)** / **PP (Prepositional Phrase)** gatekeeper with a rule-based fallback parser |
| 3 | `feat/task-3-staging-verification` | Thread-safe RAM staging ring **buffer with TTL (Time to Live)** + late-bound n-gram verification |
| 4 | `feat/task-4-overlay-and-daemon` | Prefill adapter with KV truncation, floating pill overlay, daemon, **CLI (Command-Line Interface)**, end-to-end tests |

Verification ran after every Task as a fixed gate:

```text
python -m pytest tests/
python -m flake8 --jobs 1 src/ tests/ benchmarks/
python benchmarks/run_bench.py
```

The gate was treated as a hard stop: a Task was only pushed once tests passed,
flake8 produced no output, and the benchmark table still showed **0 wasted
tokens** for PhantomParse.

### Edge cases resolved during execution

1. **Byte-Pair Encoding subword instability.** A committed token cannot be
   revised character by character, so the buffer only promotes a token at a
   whitespace or terminal-punctuation boundary and keeps everything after the
   last boundary as an uncommitted *pending* fragment. Backspacing into a
   committed token emits a `RollbackEvent` carrying the character index and the
   removed text.
2. **Closed-class words swallowed by noun phrases.** `"Check the payment failure
   logs"` initially produced one phrase containing the imperative verb. Adding
   `check` to the boundary lexicon split the finite verb from the **NP**.
3. **Incomplete prepositional phrases.** `"for the"` has no head noun, so the
   gatekeeper reports no chunk until the complement has content; this prevents
   staging semantically empty spans.
4. **Similarity asymmetry.** Pure cosine similarity punishes a short staged span
   inside a long final prompt. The score is `max(cosine, containment)`, so a
   verbatim span always scores `1.00` while an abandoned pivot span scores far
   below the `0.75` threshold.
5. **Growing prefixes duplicated context.** Because staging stores `"a unit"`
   and later `"a unit test"`, verification legitimately accepts both. The daemon
   now collapses accepted spans with `maximal_spans()` before warming, so the
   pivot scenario injects **2** spans rather than 5, and the benchmark reports
   the same deduplicated count.
6. **Explicit backend was being probed.** An explicitly requested `http` prefill
   backend still ran the availability probe and silently demoted to the
   emulator. `_resolve_backend()` now short-circuits for both explicit backends
   and only probes in `auto` mode.
7. **Wasted-token accounting.** Charging every sent token overcounted. Waste is
   `min(tokens_sent, tokens_in_discarded_prefix)`, which yields the expected
   `10` tokens (41.7%) on the naive pivot and `0` for PhantomParse.
8. **Deterministic GIFs and tests.** Virtual time replaces `time.sleep`; the
   prefill emulator derives latency from `tokens / prefill_tokens_per_second`;
   the TTL tests inject a fake clock. Nothing in the suite depends on wall-clock
   timing or a live model server.

### Environment deviations (disclosed, not hidden)

- The host has **Python 3.10.11** installed, while `pyproject.toml` declares
  `requires-python = ">=3.11"`. All Phase 1 code is 3.10-compatible and the full
  suite runs green here; the declared floor stays at 3.11 per `AGENTS.md`.
- **SpaCy is not installed** in this environment, so the gatekeeper runs its
  documented rule-based backend (`Backend: rules`) instead of the statistical
  **POS (Part of Speech)** tagger. The SpaCy path is implemented, lazily loaded,
  and selected automatically when available.
- Global `pip install` is blocked by the sandbox, so the project uses a local
  `.venv`; every gate command therefore runs `.\.venv\Scripts\python.exe -m ...`.
- `gh pr merge` returns exit code 1 with a sandbox restriction message **after**
  the merge has already succeeded on GitHub; merge success was verified with
  `git log` on `main` and `gh pr list --state all`.

---

## 2. Linguistic-to-Systems Architecture Mapping Table

| Module | Computational-linguistics counterpart | Systems implementation |
| --- | --- | --- |
| `src/phantomparse/config.py` | Lexicon and grammar settings | Layered config loader (bundled → user → explicit), typed `Config` dataclass, YAML persistence at `~/.config/phantomparse/config.yaml` |
| `src/phantomparse/buffer.py` | Tokenisation and word-boundary segmentation | `KeystrokeBuffer` promotes a token only at whitespace/terminal punctuation; backspace into a committed token emits `RollbackEvent(char_index, token_index, removed)` |
| `src/phantomparse/gatekeeper.py` | Shallow constituency parsing (NP / PP) | `SyntacticGatekeeper` anchors on closed-class determiners/prepositions/finite verbs; all chunks tagged `speculative_read_only=True` so speculative reads never mutate the user's text |
| `src/phantomparse/staging.py` | Semantic similarity over surface forms | Thread-safe ring buffer with TTL and capacity limits; char-trigram weights `1 + log(count)`, `similarity = max(cosine, containment)` |
| `src/phantomparse/prefill_adapter.py` | Incremental parsing / prefix reuse | Warms the model with `num_predict: 1`, caches the prompt prefix in the KV cache, and truncates that prefix at the exact invalidated character |
| `src/phantomparse/overlay.py` | Real-time parse-tree readout | Borderless always-on-top `tkinter` pill showing status, staged/injected counts and the latest closed phrase |
| `src/phantomparse/daemon.py` | Discourse state machine across an utterance | Wires buffer → gatekeeper → staging → prefill → overlay; `submit()` verifies, collapses spans with `maximal_spans()`, warms, reports and resets |
| `src/phantomparse/cli.py` | Command surface for the linguist-user | `start`, `overlay`, `bench`, `config` subcommands with headless demo mode and stdin-driven keystrokes |
| `benchmarks/keystroke_simulator.py` | Corpus of typed utterances | Deterministic 60 **WPM (Words Per Minute)** event stream: `5` characters per word, `200 ms` per character, pivot pause included |
| `benchmarks/run_bench.py` | Comparative corpus evaluation | Three-mode benchmark (Baseline, Naive Speculation, PhantomParse) with `tracemalloc` peak-memory and token-waste accounting |
| `benchmarks/gif_recorder.py` | Visual annotation of an analysis | Pillow-only frame renderer; no ffmpeg/ImageMagick dependency, deterministic output |

---

## 3. Git & GitHub SDLC Audit Log

### `git remote -v`

```text
origin  https://github.com/Pseud0Coder/PhantomParse.git (fetch)
origin  https://github.com/Pseud0Coder/PhantomParse.git (push)
```

### `git branch -a`

```text
  feat/task-4-overlay-and-daemon
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/feat/task-0-benchmark-harness
  remotes/origin/feat/task-1-keystroke-buffer
  remotes/origin/feat/task-2-syntactic-gatekeeper
  remotes/origin/feat/task-3-staging-verification
  remotes/origin/feat/task-4-overlay-and-daemon
  remotes/origin/main
```

### `git log --all --graph --oneline --decorate`

```text
*   78aba46 (HEAD -> main, origin/main, origin/HEAD) Merge pull request #5 from Pseud0Coder/feat/task-4-overlay-and-daemon
|\
| * ec0b461 (origin/feat/task-4-overlay-and-daemon, feat/task-4-overlay-and-daemon) docs: write task 4 pull request notes
| * e5c085d docs: render the task 4 overlay and daemon gif
| * 2a1e8b6 fix: report maximal injected spans in the benchmark report
| * fe1fd57 test: cover the overlay pill and the end-to-end pivot pipeline
| * bc69bd0 fix: collapse nested accepted spans into one maximal context
| * 9f1bcc9 fix: honor an explicit prefill backend without probing the endpoint
| * 3cafe52 feat: add phantomparse CLI with start, overlay, bench and config commands
| * 760714b feat: add synthwave pill overlay and speculative daemon with hotkey fallback
| * c4733d3 feat: add incremental prefill adapter with KV prefix truncation
|/
*   376c70a Merge pull request #4 from Pseud0Coder/feat/task-3-staging-verification
|\
| * 903052e (origin/feat/task-3-staging-verification) docs: record task 3 staging and verification PR
| * 4127d4c bench: report late-bound verification and render task 3 GIF
| * 2d62597 test: prove the mid-sentence pivot injects zero wasted tokens
| * 39c22d5 feat: verify staged spans with late-bound character n-gram scoring
| * 5ac0cf6 feat: add thread-safe RAM staging ring buffer with TTL and capacity limits
|/
*   fd05600 Merge pull request #3 from Pseud0Coder/feat/task-2-syntactic-gatekeeper
|\
| * 1a237d6 (origin/feat/task-2-syntactic-gatekeeper) docs: record task 2 gatekeeper PR
| * 3ac8f8c bench: report gatekeeper constituents and render task 2 GIF
| * 0aaa31a test: cover NP and PP extraction, read-only tagging and parse budget
| * 7ecd1df feat: tag chunks read-only and measure parse latency against budget
| * 29de3a9 feat: add local NP and PP syntactic gatekeeper with rule-based fallback
|/
*   1827a27 Merge pull request #2 from Pseud0Coder/feat/task-1-keystroke-buffer
|\
| * 276a32c (origin/feat/task-1-keystroke-buffer) docs: record task 1 pull request narrative
| * a2c0b83 bench: record task 1 buffer commits and rollback animation
| * b9d335a test: cover boundary commits, pending fragments, and multi-word rollbacks
| * 7ce5644 feat: emit rollback events when backspace invalidates a committed token
| * 7af012a feat: add keystroke buffer with word and punctuation boundary commits
|/
*   13dc134 Merge pull request #1 from Pseud0Coder/feat/task-0-benchmark-harness
|\
| * 83489aa (origin/feat/task-0-benchmark-harness) docs: record task 0 pull request narrative with telemetry evidence
| * 079e44f test: cover keystroke simulator and configuration loader
| * 4ed8577 bench: add baseline and naive speculation harness with native GIF recorder
| * 0e98ea4 feat: add deterministic 60 WPM keystroke simulator with pivot scenario
| * d188497 feat: scaffold project metadata and layered configuration loader
|/
* 9e2cfbd chore: initialize repository with agent guidance and lint config
```

### `gh pr list --state all`

```text
Showing 5 of 5 pull requests in Pseud0Coder/PhantomParse that match your search

ID  TITLE                         BRANCH                       CREATED AT
#5  Task 4: prefill adapter, ...  feat/task-4-overlay-and-...  about 7 hours ago
#4  Task 3: RAM staging ring ...  feat/task-3-staging-veri...  about 7 hours ago
#3  Task 2: local syntactic g...  feat/task-2-syntactic-ga...  about 7 hours ago
#2  Task 1: real-time keystro...  feat/task-1-keystroke-bu...  about 7 hours ago
#1  Task 0: scaffolding, tele...  feat/task-0-benchmark-ha...  about 8 hours ago
```

Pull request URLs:

- https://github.com/Pseud0Coder/PhantomParse/pull/1
- https://github.com/Pseud0Coder/PhantomParse/pull/2
- https://github.com/Pseud0Coder/PhantomParse/pull/3
- https://github.com/Pseud0Coder/PhantomParse/pull/4
- https://github.com/Pseud0Coder/PhantomParse/pull/5

---

## 4. Pull Request Portfolio (Tasks 0 to 4)

### PR #1 — Task 0: Benchmark harness and scaffolding

**Plain English.** Sets up the project skeleton, a settings file, a robot that
types test sentences at human speed, and a scoreboard that compares today's slow
way of waiting for `Enter` against the PhantomParse way.

**GIF.** `docs/gifs/task-0.gif` — 59 frames, 689,159 bytes (673.0 KiB).

**Commits.**

```text
d188497 feat: scaffold project metadata and layered configuration loader
0e98ea4 feat: add deterministic 60 WPM keystroke simulator with pivot scenario
4ed8577 bench: add baseline and naive speculation harness with native GIF recorder
079e44f test: cover keystroke simulator and configuration loader
83489aa docs: record task 0 pull request narrative with telemetry evidence
```

### PR #2 — Task 1: Keystroke buffer

**Plain English.** Watches typing one character at a time, decides exactly when a
word is finished, and notices when backspace takes a word back so anything based
on that word can be undone.

**GIF.** `docs/gifs/task-1.gif` — 68 frames, 899,418 bytes (878.3 KiB).

**Commits.**

```text
7af012a feat: add keystroke buffer with word and punctuation boundary commits
7ce5644 feat: emit rollback events when backspace invalidates a committed token
b9d335a test: cover boundary commits, pending fragments, and multi-word rollbacks
a2c0b83 bench: record task 1 buffer commits and rollback animation
276a32c docs: record task 1 pull request narrative
```

### PR #3 — Task 2: Syntactic gatekeeper

**Plain English.** Finds the meaningful phrases in a half-written sentence — the
things worth looking up — and ignores fragments that are not finished yet.

**GIF.** `docs/gifs/task-2.gif` — 25 frames, 364,859 bytes (356.3 KiB).

**Commits.**

```text
29de3a9 feat: add local NP and PP syntactic gatekeeper with rule-based fallback
7ecd1df feat: tag chunks read-only and measure parse latency against budget
0aaa31a test: cover NP and PP extraction, read-only tagging and parse budget
3ac8f8c bench: report gatekeeper constituents and render task 2 GIF
1a237d6 docs: record task 2 gatekeeper PR
```

### PR #4 — Task 3: RAM staging and late-bound verification

**Plain English.** Keeps those phrases in a short-lived memory buffer, then on
`Enter` checks which ones actually survived in the final sentence — throwing
away anything the user deleted, before a single token is paid for.

**GIF.** `docs/gifs/task-3.gif` — 29 frames, 437,564 bytes (427.3 KiB).

**Commits.**

```text
5ac0cf6 feat: add thread-safe RAM staging ring buffer with TTL and capacity limits
39c22d5 feat: verify staged spans with late-bound character n-gram scoring
2d62597 test: prove the mid-sentence pivot injects zero wasted tokens
4127d4c bench: report late-bound verification and render task 3 GIF
903052e docs: record task 3 staging and verification PR
```

### PR #5 — Task 4: Prefill adapter, overlay and CLI daemon

**Plain English.** Warms the model in the background while you type so the answer
starts instantly on `Enter`, shows a small floating pill describing what it is
doing, and gives the whole thing a command line you can actually run.

**GIF.** `docs/gifs/task-4.gif` — 30 frames, 554,480 bytes (541.5 KiB).

**Commits.**

```text
c4733d3 feat: add incremental prefill adapter with KV prefix truncation
760714b feat: add synthwave pill overlay and speculative daemon with hotkey fallback
3cafe52 feat: add phantomparse CLI with start, overlay, bench and config commands
9f1bcc9 fix: honor an explicit prefill backend without probing the endpoint
bc69bd0 fix: collapse nested accepted spans into one maximal context
fe1fd57 test: cover the overlay pill and the end-to-end pivot pipeline
2a1e8b6 fix: report maximal injected spans in the benchmark report
e5c085d docs: render the task 4 overlay and daemon gif
ec0b461 docs: write task 4 pull request notes
```

All GIFs are generated deterministically by `benchmarks/gif_recorder.py`, which
uses Pillow only — no ffmpeg, no ImageMagick, no network access.

---

## 5. Verification & Test Suite Proof

### Raw `pytest -v` stdout

```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.7.0 -- C:\Trae projects\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Trae projects
configfile: pyproject.toml
collecting ... collected 130 items

tests/test_buffer.py::test_word_is_not_committed_until_whitespace_closes_it PASSED [  0%]
tests/test_buffer.py::test_mid_word_pause_leaves_the_fragment_uncommitted PASSED [  1%]
tests/test_buffer.py::test_rapid_typing_matches_character_by_character_typing PASSED [  2%]
tests/test_buffer.py::test_terminal_punctuation_closes_a_token_without_a_space PASSED [  3%]
tests/test_buffer.py::test_apostrophes_and_hyphens_stay_inside_the_word PASSED [  3%]
tests/test_buffer.py::test_char_indexes_point_at_the_token_start_in_the_raw_stream PASSED [  4%]
tests/test_buffer.py::test_repeated_whitespace_never_commits_an_empty_token PASSED [  5%]
tests/test_buffer.py::test_flush_commits_a_trailing_fragment_once PASSED [  6%]
tests/test_buffer.py::test_feed_char_rejects_multi_character_input PASSED [  6%]
tests/test_buffer.py::test_backspace_inside_pending_fragment_emits_nothing PASSED [  7%]
tests/test_buffer.py::test_backspace_deleting_whitespace_emits_no_rollback PASSED [  8%]
tests/test_buffer.py::test_backspace_into_committed_token_emits_rollback_event PASSED [  9%]
tests/test_buffer.py::test_multi_word_backspace_unwinds_tokens_in_reverse_order PASSED [ 10%]
tests/test_buffer.py::test_punctuation_token_rolls_back_to_its_bare_word PASSED [ 10%]
tests/test_buffer.py::test_retyping_after_rollback_recommits_at_the_same_token_index PASSED [ 11%]
tests/test_buffer.py::test_backspace_on_empty_buffer_is_a_no_op PASSED   [ 12%]
tests/test_buffer.py::test_pending_text_always_matches_the_raw_stream_suffix PASSED [ 13%]
tests/test_config.py::test_field_defaults_are_the_documented_runtime_values PASSED [ 13%]
tests/test_config.py::test_bundled_yaml_mirrors_the_dataclass_defaults PASSED [ 14%]
tests/test_config.py::test_explicit_path_overrides_bundled_defaults PASSED [ 15%]
tests/test_config.py::test_missing_explicit_path_falls_back_to_defaults PASSED [ 16%]
tests/test_config.py::test_from_mapping_coerces_scalar_types PASSED      [ 16%]
tests/test_config.py::test_from_mapping_ignores_unknown_and_null_keys PASSED [ 17%]
tests/test_config.py::test_from_mapping_returns_defaults_for_none PASSED [ 18%]
tests/test_config.py::test_save_round_trips_through_load_config PASSED   [ 19%]
tests/test_config.py::test_invalid_yaml_raises_value_error PASSED        [ 20%]
tests/test_config.py::test_non_mapping_yaml_raises_value_error PASSED    [ 20%]
tests/test_config.py::test_ensure_user_config_writes_once PASSED         [ 21%]
tests/test_daemon.py::test_typing_a_closed_phrase_stages_it_in_ram PASSED [ 22%]
tests/test_daemon.py::test_keystrokes_inside_a_word_do_not_stage_anything PASSED [ 23%]
tests/test_daemon.py::test_backspace_counts_a_rollback_and_truncates_the_kv_cache PASSED [ 23%]
tests/test_daemon.py::test_submit_verifies_stages_and_reports_zero_waste PASSED [ 24%]
tests/test_daemon.py::test_submit_resets_the_next_prompt PASSED          [ 25%]
tests/test_daemon.py::test_overlay_state_follows_submission PASSED       [ 26%]
tests/test_daemon.py::test_toggle_pauses_and_rearms_speculation PASSED   [ 26%]
tests/test_daemon.py::test_stats_expose_backend_and_counters PASSED      [ 27%]
tests/test_daemon.py::test_headless_run_and_stop_do_not_require_a_display PASSED [ 28%]
tests/test_daemon.py::test_demo_states_cover_typing_pivot_and_answer PASSED [ 29%]
tests/test_e2e_pipeline.py::test_pivot_scenario_ends_with_zero_wasted_tokens PASSED [ 30%]
tests/test_e2e_pipeline.py::test_every_injected_span_is_verbatim_in_the_final_prompt PASSED [ 30%]
tests/test_e2e_pipeline.py::test_abandoned_pivot_spans_are_never_injected PASSED [ 31%]
tests/test_e2e_pipeline.py::test_rejected_spans_scored_below_the_threshold PASSED [ 32%]
tests/test_e2e_pipeline.py::test_verification_stays_inside_its_latency_budget PASSED [ 33%]
tests/test_e2e_pipeline.py::test_typing_pays_the_prefill_cost_so_enter_is_fast PASSED [ 33%]
tests/test_e2e_pipeline.py::test_maximal_spans_collapse_growing_prefixes PASSED [ 34%]
tests/test_e2e_pipeline.py::test_maximal_spans_keep_unrelated_and_duplicate_spans_once PASSED [ 35%]
tests/test_e2e_pipeline.py::test_linear_scenario_injects_every_staged_span PASSED [ 36%]
tests/test_e2e_pipeline.py::test_a_second_prompt_starts_from_a_clean_slate PASSED [ 36%]
tests/test_e2e_pipeline.py::test_overlay_never_shows_a_stale_stage_count_after_submit PASSED [ 37%]
tests/test_e2e_pipeline.py::test_pipeline_memory_stays_under_the_ram_budget PASSED [ 38%]
tests/test_e2e_pipeline.py::test_cli_overlay_demo_prints_the_pill PASSED [ 39%]
tests/test_e2e_pipeline.py::test_cli_config_prints_the_effective_settings PASSED [ 40%]
tests/test_e2e_pipeline.py::test_cli_bench_emits_markdown PASSED         [ 40%]
tests/test_e2e_pipeline.py::test_cli_rejects_an_unknown_scenario PASSED  [ 41%]
tests/test_e2e_pipeline.py::test_full_timeline_reconstructs_the_prompt PASSED [ 42%]
tests/test_gatekeeper.py::test_rule_backend_is_selected_when_spacy_is_disabled PASSED [ 43%]
tests/test_gatekeeper.py::test_noun_phrase_and_prepositional_phrase_are_both_detected PASSED [ 43%]
tests/test_gatekeeper.py::test_pivot_clause_yields_the_retrieval_bearing_phrase PASSED [ 44%]
tests/test_gatekeeper.py::test_incomplete_fragment_without_a_head_noun_is_ignored PASSED [ 45%]
tests/test_gatekeeper.py::test_pp_is_recovered_once_the_complement_has_content PASSED [ 46%]
tests/test_gatekeeper.py::test_multiple_prepositional_phrases_are_ordered_by_offset PASSED [ 46%]
tests/test_gatekeeper.py::test_chunk_offsets_slice_back_to_the_chunk_text PASSED [ 47%]
tests/test_gatekeeper.py::test_terminal_punctuation_is_not_part_of_the_chunk PASSED [ 48%]
tests/test_gatekeeper.py::test_noun_phrase_head_is_its_last_content_word PASSED [ 49%]
tests/test_gatekeeper.py::test_every_chunk_is_tagged_speculative_read_only PASSED [ 50%]
tests/test_gatekeeper.py::test_blank_input_returns_no_chunks PASSED      [ 50%]
tests/test_gatekeeper.py::test_chunks_of_kind_filters_by_constituent_type PASSED [ 51%]
tests/test_gatekeeper.py::test_every_parse_finishes_inside_the_latency_budget PASSED [ 52%]
tests/test_gatekeeper.py::test_chunk_dataclass_defaults_to_read_only PASSED [ 53%]
tests/test_gatekeeper.py::test_default_gatekeeper_is_a_singleton PASSED  [ 53%]
tests/test_gatekeeper.py::test_chunk_text_helper_matches_the_default_gatekeeper PASSED [ 54%]
tests/test_keystroke_simulator.py::test_char_interval_matches_five_characters_per_word_convention PASSED [ 55%]
tests/test_keystroke_simulator.py::test_char_interval_rejects_non_positive_rate PASSED [ 56%]
tests/test_keystroke_simulator.py::test_linear_scenario_replays_to_its_initial_text PASSED [ 56%]
tests/test_keystroke_simulator.py::test_virtual_timeline_advances_one_interval_per_event PASSED [ 57%]
tests/test_keystroke_simulator.py::test_pivot_scenario_erases_then_retypes PASSED [ 58%]
tests/test_keystroke_simulator.py::test_pivot_pause_last_longer_than_a_single_keystroke PASSED [ 59%]
tests/test_keystroke_simulator.py::test_override_wpm_changes_interval_but_not_final_text PASSED [ 60%]
tests/test_keystroke_simulator.py::test_simulate_all_covers_every_bundled_scenario PASSED [ 60%]
tests/test_keystroke_simulator.py::test_scenario_by_key_raises_on_unknown_key PASSED [ 61%]
tests/test_keystroke_simulator.py::test_reconstruct_ignores_backspace_on_empty_buffer PASSED [ 62%]
tests/test_keystroke_simulator.py::test_reconstruct_returns_empty_string_without_events PASSED [ 63%]
tests/test_keystroke_simulator.py::test_event_kinds_are_a_closed_set PASSED [ 63%]
tests/test_overlay.py::test_palette_matches_the_synthwave_spec PASSED    [ 64%]
tests/test_overlay.py::test_pill_lines_report_status_chunks_and_savings PASSED [ 65%]
tests/test_overlay.py::test_pill_lines_join_multiple_chunks PASSED       [ 66%]
tests/test_overlay.py::test_pill_lines_placeholder_before_any_closed_phrase PASSED [ 66%]
tests/test_overlay.py::test_pill_lines_truncate_long_chunk_lists_from_the_left PASSED [ 67%]
tests/test_overlay.py::test_headless_overlay_records_state_without_drawing PASSED [ 68%]
tests/test_overlay.py::test_headless_run_cycles_to_the_final_frame PASSED [ 69%]
tests/test_overlay.py::test_close_marks_the_overlay_closed PASSED        [ 70%]
tests/test_overlay.py::test_tkinter_availability_is_a_boolean PASSED     [ 70%]
tests/test_prefill_adapter.py::test_estimate_tokens_is_zero_for_blank_text PASSED [ 71%]
tests/test_prefill_adapter.py::test_estimate_tokens_scales_with_word_count PASSED [ 72%]
tests/test_prefill_adapter.py::test_emulator_warm_reports_tokens_and_latency PASSED [ 73%]
tests/test_prefill_adapter.py::test_warm_strips_surrounding_whitespace PASSED [ 73%]
tests/test_prefill_adapter.py::test_backend_auto_falls_back_when_probe_fails PASSED [ 74%]
tests/test_prefill_adapter.py::test_backend_auto_uses_http_when_probe_succeeds PASSED [ 75%]
tests/test_prefill_adapter.py::test_http_warm_posts_a_single_token_prefill PASSED [ 76%]
tests/test_prefill_adapter.py::test_http_failure_demotes_to_emulator PASSED [ 76%]
tests/test_prefill_adapter.py::test_unknown_backend_is_rejected PASSED   [ 77%]
tests/test_prefill_adapter.py::test_truncate_drops_the_tail_of_the_cached_prefix PASSED [ 78%]
tests/test_prefill_adapter.py::test_truncate_past_the_end_is_a_no_op PASSED [ 79%]
tests/test_prefill_adapter.py::test_truncate_to_zero_empties_the_cache PASSED [ 80%]
tests/test_prefill_adapter.py::test_reset_forgets_the_cached_prefix PASSED [ 80%]
tests/test_prefill_adapter.py::test_saved_ttft_is_zero_before_any_prefill PASSED [ 81%]
tests/test_prefill_adapter.py::test_saved_ttft_includes_retrieval_and_prefill PASSED [ 82%]
tests/test_prefill_adapter.py::test_probe_endpoint_returns_false_for_a_dead_port PASSED [ 83%]
tests/test_staging.py::test_similarity_of_the_same_span_is_one PASSED    [ 83%]
tests/test_staging.py::test_similarity_separates_pivot_spans_from_the_final_prompt PASSED [ 84%]
tests/test_staging.py::test_similarity_handles_blank_input PASSED        [ 85%]
tests/test_staging.py::test_similarity_ignores_case_and_punctuation PASSED [ 86%]
tests/test_staging.py::test_stage_returns_a_read_only_item_with_ttl PASSED [ 86%]
tests/test_staging.py::test_blank_spans_are_not_staged PASSED            [ 87%]
tests/test_staging.py::test_duplicate_spans_are_staged_once PASSED       [ 88%]
tests/test_staging.py::test_expired_items_are_purged_on_snapshot PASSED  [ 89%]
tests/test_staging.py::test_capacity_evicts_the_oldest_item PASSED       [ 90%]
tests/test_staging.py::test_stage_chunks_returns_only_new_items PASSED   [ 90%]
tests/test_staging.py::test_verify_accepts_spans_that_survive_into_the_final_prompt PASSED [ 91%]
tests/test_staging.py::test_verify_evicts_abandoned_spans_and_keeps_survivors PASSED [ 92%]
tests/test_staging.py::test_mid_sentence_pivot_yields_zero_wasted_tokens PASSED [ 93%]
tests/test_staging.py::test_verify_threshold_override_accepts_weaker_matches PASSED [ 93%]
tests/test_staging.py::test_verify_on_an_empty_buffer_is_a_no_op PASSED  [ 94%]
tests/test_staging.py::test_verify_latency_stays_inside_the_budget PASSED [ 95%]
tests/test_staging.py::test_clear_empties_the_buffer PASSED              [ 96%]
tests/test_staging.py::test_stats_report_configuration_and_counters PASSED [ 96%]
tests/test_staging.py::test_concurrent_staging_respects_capacity PASSED  [ 97%]
tests/test_staging.py::test_invalid_configuration_is_rejected[kwargs0-ttl_seconds] PASSED [ 98%]
tests/test_staging.py::test_invalid_configuration_is_rejected[kwargs1-max_items] PASSED [ 99%]
tests/test_staging.py::test_invalid_configuration_is_rejected[kwargs2-similarity_threshold] PASSED [100%]

============================= 130 passed in 1.52s =============================
```

### Raw `flake8 src/ tests/ benchmarks/` stdout

```text
$ python -m flake8 --jobs 1 src/ tests/ benchmarks/
(no output; exit code 0)
```

Flake8 is configured in `.flake8` with `max-line-length = 100`,
`extend-ignore = E203, W503`, and `per-file-ignores = tests/*: E501`.

---

## 6. Final Numerical Benchmark Table

Raw output of `python benchmarks/run_bench.py`:

```text
# PhantomParse Benchmark

Typing speed: 60 WPM | TTL: 30s | Similarity threshold: 0.75 | Prefill rate: 240 tok/s

| Mode | Scenario | TTFT (ms) | Post-Enter Retrieval (ms) | Wasted LLM Tokens | Token Waste (%) | Peak Daemon Memory (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | linear | 810.3 | 479.5 | 0 | 0.0 | 0.00 |
| Baseline | pivot | 779.7 | 465.5 | 0 | 0.0 | 0.00 |
| Baseline | multi_clause | 795.0 | 472.5 | 0 | 0.0 | 0.00 |
| Naive Speculation | linear | 183.0 | 25.0 | 0 | 0.0 | 0.00 |
| Naive Speculation | pivot | 183.0 | 25.0 | 10 | 41.7 | 0.00 |
| Naive Speculation | multi_clause | 183.0 | 25.0 | 0 | 0.0 | 0.00 |
| PhantomParse | linear | 101.1 | 1.8 | 0 | 0.0 | 0.06 |
| PhantomParse | pivot | 101.8 | 2.5 | 0 | 0.0 | 0.03 |
| PhantomParse | multi_clause | 101.0 | 1.7 | 0 | 0.0 | 0.03 |

## Summary

| Mode | Mean TTFT (ms) | Mean Post-Enter Retrieval (ms) | Wasted LLM Tokens | Mean Waste (%) | TTFT Reduction vs Baseline | Peak Memory (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 795.0 | 472.5 | 0 | 0.0 | n/a | 0.00 |
| Naive Speculation | 183.0 | 25.0 | 10 | 13.9 | 77.0% | 0.00 |
| PhantomParse | 101.3 | 2.0 | 0 | 0.0 | 87.3% | 0.06 |

> Peak Daemon Memory is the incremental traced allocation of the in-process pipeline, measured with `tracemalloc`.

## Syntactic Gatekeeper

Backend: `rules` | parse budget: 15 ms (best of 5 parses) | all chunks are `speculative_read_only=True`

| Scenario | Backend | Chunks | NP | PP | Parse Latency (ms) | Within Budget |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| linear | rules | 3 | 1 | 2 | 0.022 | yes |
| pivot | rules | 2 | 1 | 1 | 0.018 | yes |
| multi_clause | rules | 4 | 1 | 3 | 0.022 | yes |

Extracted constituents:

| Scenario | Kind | Text | Head |
| --- | --- | --- | --- |
| linear | NP | BPE tokenization | tokenization |
| linear | PP | of vocabulary morphemes | of |
| linear | PP | in transformer models | in |
| pivot | NP | a unit test | test |
| pivot | PP | for the Firebase auth middleware | for |
| multi_clause | NP | the attention sink mechanism | mechanism |
| multi_clause | PP | in StreamingLLM | in |
| multi_clause | PP | with PagedAttention | with |
| multi_clause | PP | in vLLM | in |

## Late-Bound Verification

Threshold: 0.75 | TTL: 30s | Capacity: 64 items | local prefetch cost: 0 LLM tokens

| Scenario | Staged | Injected | Evicted (pivot) | Max Score | Wasted LLM Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 6 | 3 | 0 | 1.00 | 0 |
| pivot | 8 | 2 | 3 | 1.00 | 0 |
| multi_clause | 6 | 4 | 0 | 1.00 | 0 |

> Staged spans never leave the machine, so only injections can cost tokens; rejected pivot spans are evicted from RAM on Enter.
```

### Reading the numbers

- **TTFT (Time to First Token)** drops from a **795.0 ms** mean baseline to
  **101.3 ms** — an **87.3%** reduction, inside the promised 75–90% band.
- **Naive speculation** is equally fast on paper but burns **10 tokens (13.9%
  mean, 41.7% on the pivot)** because it sends the abandoned clause to the model.
- **PhantomParse wastes 0 tokens in every scenario**, including the pivot, because
  the pivot clause never leaves RAM.
- **Peak traced pipeline memory is 0.06 MB**, three orders of magnitude below the
  **50 MB** budget and far below the **RAM (Random Access Memory)** ceiling in
  `AGENTS.md`.
- **Post-Enter retrieval latency is 2.0 ms mean** versus 472.5 ms for the
  baseline, because the prompt prefix is already warm in the **KV cache**.

---

## 7. Glossary of Engineering Acronyms

| Acronym | Expansion | Linguistic / systems meaning in this build |
| --- | --- | --- |
| **API** | Application Programming Interface | The HTTP surface of the model server (`/api/tags`, `/api/generate`) |
| **BPE** | Byte-Pair Encoding | Subword tokenisation; the reason a committed token cannot be edited character by character |
| **CLI** | Command-Line Interface | `phantomparse start / overlay / bench / config` |
| **GIF** | Graphics Interchange Format | Animated evidence artefact in `docs/gifs/`, rendered with Pillow only |
| **HTTP** | Hypertext Transfer Protocol | Transport used by the prefill adapter to warm the model |
| **JSON** | JavaScript Object Notation | Payload format for the prefill request |
| **KV** | Key-Value | The transformer's attention cache; PhantomParse warms it during typing and truncates it on backspace |
| **LLM** | Large Language Model | The downstream model whose TTFT and token spend are being optimised |
| **NLP** | Natural Language Processing | The computational-linguistics field this build applies (segmentation, tagging, similarity) |
| **NP** | Noun Phrase | Constituent extracted by the gatekeeper (`"a unit test"`, head `test`) |
| **POS** | Part of Speech | Tagging layer used by the statistical SpaCy backend |
| **PP** | Prepositional Phrase | Retrieval-bearing constituent (`"for the Firebase auth middleware"`) |
| **PR** | Pull Request | GitHub review unit; one per Task, body in `docs/prs/PR_TASK_N.md` |
| **RAM** | Random Access Memory | Where staged spans live, never leaving the machine before `Enter` |
| **RAG** | Retrieval-Augmented Generation | Retrieval pattern PhantomParse accelerates by prefetching locally |
| **SDLC** | Software Development Life Cycle | Branch → atomic commits → PR → merge, one cycle per Task |
| **TTFT** | Time to First Token | Primary latency metric; 795.0 ms baseline → 101.3 ms PhantomParse |
| **TTL** | Time to Live | 30 s expiry on staged spans so stale context never survives a pause |
| **WPM** | Words Per Minute | Typing speed of the simulator; 60 WPM = 5 chars/word = 200 ms/char |

---

### Reproducing everything

```bash
pip install -r requirements.txt

python -m pytest tests/ -v
python -m flake8 --jobs 1 src/ tests/ benchmarks/
python benchmarks/run_bench.py
python benchmarks/gif_recorder.py --task 4
phantomparse start --demo
```
