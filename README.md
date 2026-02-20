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

## Usage Guide (English / 한국어)

### 1) Setup / 설치

```bash
cd /home/Jiseong/PythonOrchestra_ws
conda activate raisim
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

EN: Install dependencies in your environment before running the orchestrator.  
KR: 오케스트레이터 실행 전, 위 명령으로 의존성을 먼저 설치하세요.

### 2) OpenAI API Key / OpenAI API 키 설정

```bash
export OPENAI_API_KEY="your_api_key_here"
```

EN: If `OPENAI_API_KEY` is missing, the run stops and records the error in artifacts.  
KR: `OPENAI_API_KEY`가 없으면 실행이 중단되고 오류가 artifacts에 기록됩니다.

EN: You can also store it in `/home/Jiseong/PythonOrchestra_ws/.env` as `OPENAI_API_KEY=...`.  
KR: `/home/Jiseong/PythonOrchestra_ws/.env` 파일에 `OPENAI_API_KEY=...` 형태로 저장해도 됩니다.

### 3) Run Orchestrator / 실행 방법

```bash
python -m orchestrator.run --config configs/default.yaml
```

EN: Basic run with default config.  
KR: 기본 설정(`configs/default.yaml`)으로 실행합니다.

Stage-machine example for bolt_wrench / bolt_wrench 스테이지 머신 실행 예시:

```bash
conda activate raisim
python -m orchestrator.run --config configs/bolt_wrench_stage.yaml
```

EN: This starts from Stage 0 and advances to Stage 1 only after Stage 0 passes.  
KR: Stage 0부터 시작하며, Stage 0 통과 전에는 Stage 1로 넘어가지 않습니다.

### 4) Inspect Artifacts / 결과 확인

- `artifacts/loop_state.json`: latest loop summary/status
- `artifacts/latest.patch`: latest generated unified diff
- `artifacts/latest_exec.json`: latest execution result
- `artifacts/iter_XXX/`: per-iteration detailed payloads (`exec.json`, `judge.json`, `review_context.json`, `review.json`, `patch.diff`, etc.)

EN: Check `loop_state.json` for current stage/index/iteration when resuming.  
KR: 재시작/재개 시 현재 stage/index/iteration은 `loop_state.json`에서 확인하세요.

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
