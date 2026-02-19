# Agentic Code Review Loop (Python)

This workspace contains a runnable orchestrator that iteratively:

1. Executes tests/commands against a target project.
2. Calls OpenAI to generate a JSON review response with a unified diff patch.
3. Applies the patch.
4. Re-runs until success or stop conditions are hit.

## Project Structure

- `orchestrator/` Python package (runner + loop components)
- `tools/` helper workspace folder
- `artifacts/` loop state and per-iteration outputs
- `target/` demo Python package with initially failing tests
- `scripts/` helper scripts folder
- `configs/default.yaml` default orchestrator config
- `pyproject.toml` pinned dependencies

## Setup

```bash
cd /home/Jiseong/PythonOrchestra_ws
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

## OpenAI API Key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

If `OPENAI_API_KEY` is missing, the orchestrator stops gracefully and records the error in artifacts.

## Run Orchestrator

```bash
python -m orchestrator.run --config configs/default.yaml
```

## Inspect Artifacts

- `artifacts/loop_state.json`: latest loop summary/status
- `artifacts/latest.patch`: latest generated unified diff
- `artifacts/latest_exec.json`: latest execution result
- `artifacts/iter_XXX/`: per-iteration detailed payloads (`exec.json`, `judge.json`, `review_context.json`, `review.json`, `patch.diff`, etc.)

## Demo Target

The demo target intentionally starts with a bug:

- `target/demo_pkg/math_utils.py`: `divide()` subtracts instead of dividing.
- `target/tests/test_math_utils.py`: one test should fail initially.

Run tests directly:

```bash
cd target
pytest -q
```

## Customize Goal/Requirements

Reviewer prompt templates are here:

- `orchestrator/prompts/reviewer_system.txt`
- `orchestrator/prompts/reviewer_user.txt`

Edit these files to change review behavior, constraints, and patch generation instructions.

## Stop/Safety Conditions

Configured in `configs/default.yaml`:

- `max_iters` (default `10`)
- stop when same error signature repeats `thresholds.same_error_repeats` times (default `3`)
- stop when score does not improve for `thresholds.no_progress_repeats` iterations (default `3`)
- stop when generated patch exceeds `thresholds.max_patch_changed_lines` (default `200`)
- success threshold `thresholds.success_score` (default `10`)
