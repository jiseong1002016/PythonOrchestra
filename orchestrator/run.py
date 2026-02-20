from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.config import Config, load_config
from orchestrator.context_collector import collect_snippets, parse_error_locations
from orchestrator.executor import Executor
from orchestrator.judge import Judge
from orchestrator.patcher import Patcher
from orchestrator.reviewer import Reviewer
from orchestrator.stages import Stage, StageStep
from orchestrator.state import ArtifactStore, LoopState


def _check_runtime_dependencies(provider: str) -> list[str]:
    missing: list[str] = []
    if provider != "openai":
        return missing
    for module_name in ("openai", "requests"):
        try:
            __import__(module_name)
        except Exception:  # noqa: BLE001
            missing.append(module_name)
    return missing


def ensure_git_repo(workspace_root: Path) -> None:
    is_git = subprocess.run(
        ["git", "-C", str(workspace_root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0

    if not is_git:
        subprocess.run(["git", "-C", str(workspace_root), "init"], check=False)

    subprocess.run(
        ["git", "-C", str(workspace_root), "config", "user.email", "codex@example.local"],
        check=False,
    )
    subprocess.run(
        ["git", "-C", str(workspace_root), "config", "user.name", "Codex"],
        check=False,
    )

    has_commit = subprocess.run(
        ["git", "-C", str(workspace_root), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0

    if not has_commit:
        subprocess.run(["git", "-C", str(workspace_root), "add", "-A"], check=False)
        subprocess.run(
            ["git", "-C", str(workspace_root), "commit", "-m", "baseline commit"],
            check=False,
        )


def _json_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _has_glob(text: str) -> bool:
    return any(ch in text for ch in "*?[]")


def _resolve_matches(workspace_root: Path, value: str) -> list[Path]:
    candidate = Path(value)
    if candidate.is_absolute():
        pattern = str(candidate)
    else:
        pattern = str((workspace_root / value).resolve())

    if _has_glob(pattern):
        return [Path(p) for p in sorted(glob.glob(pattern))]

    p = Path(pattern)
    return [p] if p.exists() else []


def _check_required_files(workspace_root: Path, required_files: list[Any]) -> list[str]:
    failures: list[str] = []

    for entry in required_files:
        if isinstance(entry, str):
            if not _resolve_matches(workspace_root, entry):
                failures.append(f"missing_required_file:{entry}")
            continue

        if isinstance(entry, list):
            options = [x for x in entry if isinstance(x, str)]
            if options and any(_resolve_matches(workspace_root, opt) for opt in options):
                continue
            failures.append(f"missing_required_file_any_of:{'|'.join(options)}")
            continue

        failures.append("invalid_required_file_entry")

    return failures


def _check_required_regex(workspace_root: Path, required_regex: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []

    for rule in required_regex:
        target = str(rule.get("path", "")).strip()
        if not target:
            failures.append("required_regex_missing_path")
            continue

        patterns_raw = rule.get("patterns")
        if isinstance(patterns_raw, list):
            patterns = [str(p) for p in patterns_raw if str(p).strip()]
        else:
            one = str(rule.get("pattern", "")).strip()
            patterns = [one] if one else []

        if not patterns:
            failures.append(f"required_regex_missing_pattern:{target}")
            continue

        flags = re.MULTILINE
        if bool(rule.get("ignore_case", False)):
            flags |= re.IGNORECASE

        files = [p for p in _resolve_matches(workspace_root, target) if p.is_file()]
        if not files:
            failures.append(f"required_regex_file_not_found:{target}")
            continue

        matched = False
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if all(re.search(pattern, text, flags=flags) for pattern in patterns):
                matched = True
                break

        if not matched:
            failures.append(f"required_regex_not_satisfied:{target}")

    return failures


def _check_required_globs(
    workspace_root: Path,
    required_globs: list[str],
    stage_start_time: float | None = None,
    require_new_files_since_stage_start: bool = False,
) -> list[str]:
    failures: list[str] = []
    for pattern in required_globs:
        value = str(pattern).strip()
        if not value:
            failures.append("required_glob_empty")
            continue
        matches = [p for p in _resolve_matches(workspace_root, value) if p.is_file()]
        if not matches:
            failures.append(f"missing_required_glob:{value}")
            continue
        if require_new_files_since_stage_start and stage_start_time is not None:
            has_new = False
            for path in matches:
                try:
                    if path.stat().st_mtime >= stage_start_time:
                        has_new = True
                        break
                except OSError:
                    continue
            if not has_new:
                failures.append(f"stale_required_glob:{value}")
    return failures


def _check_csv_header_checks(workspace_root: Path, checks: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for rule in checks:
        pattern = str(rule.get("glob", "")).strip()
        if not pattern:
            failures.append("csv_header_check_missing_glob")
            continue

        required_columns_raw = rule.get("required_columns")
        if not isinstance(required_columns_raw, list):
            failures.append(f"csv_header_check_missing_required_columns:{pattern}")
            continue
        required_columns = [str(c).strip() for c in required_columns_raw if str(c).strip()]
        if not required_columns:
            failures.append(f"csv_header_check_empty_required_columns:{pattern}")
            continue

        files = [p for p in _resolve_matches(workspace_root, pattern) if p.is_file()]
        if not files:
            failures.append(f"csv_header_no_match:{pattern}")
            continue

        for file_path in files:
            try:
                first_line = file_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
            except Exception:  # noqa: BLE001
                failures.append(f"csv_header_read_error:{file_path}")
                continue
            header_cols = [col.strip() for col in first_line.split(",")]
            missing = [col for col in required_columns if col not in header_cols]
            if missing:
                failures.append(f"csv_header_missing_columns:{file_path}:{'|'.join(missing)}")
    return failures


def _evaluate_stage_checks(
    stage: Stage,
    workspace_root: Path,
    stage_start_time: float | None = None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    failures.extend(_check_required_files(workspace_root, stage.pass_checks.required_files))
    failures.extend(
        _check_required_globs(
            workspace_root,
            stage.pass_checks.required_globs,
            stage_start_time=stage_start_time,
            require_new_files_since_stage_start=stage.pass_checks.require_new_files_since_stage_start,
        )
    )
    failures.extend(_check_csv_header_checks(workspace_root, stage.pass_checks.csv_header_checks))
    failures.extend(_check_required_regex(workspace_root, stage.pass_checks.required_regex))
    return len(failures) == 0, failures


def _build_reviewer_context(
    config: Config,
    iteration: int,
    latest_exec: dict[str, Any],
    previous_patch: str,
    stage: Stage,
    step: StageStep | None,
) -> dict[str, Any]:
    combined_output = f"{latest_exec.get('stdout', '')}\n{latest_exec.get('stderr', '')}".strip()
    locations = parse_error_locations(combined_output)
    snippets = collect_snippets(config.workspace_root, locations, radius=30, max_chars=12_000)

    return {
        "iteration": iteration,
        "command": str(latest_exec.get("command") or config.executor.command),
        "target_dir": str(config.target_dir),
        "stage_name": stage.name,
        "step_name": step.name if step else "pass_checks",
        "stage_goal": stage.description,
        "code_context": snippets,
        "latest_exec": latest_exec,
        "previous_patch": previous_patch,
    }


def _load_previous_patch(artifacts_dir: Path) -> str:
    path = artifacts_dir / "latest.patch"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _fallback_stage(command: str, max_iters: int) -> list[Stage]:
    return [
        Stage(
            name="single",
            description="Fallback single-command stage",
            steps=[StageStep(name="default", command=command)],
            max_iters=max_iters,
        )
    ]


def run_loop(config: Config) -> int:
    artifacts_dir = config.paths.artifacts_dir if config.paths else (config.workspace_root / "artifacts")
    prompt_dir = config.paths.prompt_dir if config.paths else (config.workspace_root / "orchestrator" / "prompts")

    artifacts = ArtifactStore(artifacts_dir)
    ensure_git_repo(config.workspace_root)

    executor = Executor(
        config.target_dir,
        config.executor.command,
        timeout_sec=config.executor.timeout_sec,
    )
    reviewer = Reviewer(
        provider=config.reviewer.provider,
        model=config.reviewer.model,
        temperature=config.reviewer.temperature,
        max_output_tokens=config.reviewer.max_output_tokens,
        prompt_dir=prompt_dir,
        command=config.reviewer.command,
        command_timeout_sec=config.reviewer.command_timeout_sec,
    )
    judge = Judge()
    patcher = Patcher(config.workspace_root)

    stages = config.stages or _fallback_stage(config.executor.command, config.max_iters)

    resume = artifacts.read_loop_state()
    start_stage_index = 0
    start_stage_iter = 0
    iteration = 0
    best_score = -10_000.0
    score = 0.0

    if isinstance(resume, dict) and not bool(resume.get("done", False)):
        metadata = resume.get("metadata", {}) if isinstance(resume.get("metadata"), dict) else {}
        start_stage_index = int(metadata.get("current_stage_index", 0) or 0)
        start_stage_iter = int(metadata.get("stage_iter", 0) or 0)
        iteration = int(resume.get("iteration", 0) or 0)
        best_score = float(resume.get("best_score", -10_000.0) or -10_000.0)
        score = float(resume.get("score", 0.0) or 0.0)

    if start_stage_index < 0 or start_stage_index >= len(stages):
        start_stage_index = 0
        start_stage_iter = 0

    state = LoopState(
        status="running",
        iteration=iteration,
        max_iters=config.max_iters,
        score=score,
        best_score=best_score,
        done=False,
        metadata={
            "workspace_root": str(config.workspace_root),
            "target_dir": str(config.target_dir),
            "command": config.executor.command,
            "current_stage_name": stages[start_stage_index].name,
            "current_stage_index": start_stage_index,
            "stage_iter": start_stage_iter,
            "stage_count": len(stages),
            "resumed": bool(iteration > 0),
        },
    )
    artifacts.write_loop_state(state)

    previous_patch = _load_previous_patch(artifacts_dir)
    previous_warnings = 0
    last_error_signature = ""
    repeated_error_count = 0
    no_progress_count = 0

    for stage_index in range(start_stage_index, len(stages)):
        stage = stages[stage_index]
        stage_limit = max(int(stage.max_iters or config.max_iters), 1)
        stage_iter = start_stage_iter if stage_index == start_stage_index else 0
        stage_passed = False
        prev_stage_name = str(state.metadata.get("stage_start_name", ""))
        prev_stage_time = state.metadata.get("stage_start_time_epoch")
        if (
            stage_index == start_stage_index
            and stage_iter > 0
            and prev_stage_name == stage.name
            and isinstance(prev_stage_time, (int, float))
        ):
            stage_start_time = float(prev_stage_time)
        else:
            stage_start_time = time.time()
        state.metadata["stage_start_name"] = stage.name
        state.metadata["stage_start_time_epoch"] = stage_start_time

        while stage_iter < stage_limit:
            stage_iter += 1
            state.metadata["current_stage_name"] = stage.name
            state.metadata["current_stage_index"] = stage_index
            state.metadata["stage_iter"] = stage_iter
            state.metadata["stage_start_name"] = stage.name
            state.metadata["stage_start_time_epoch"] = stage_start_time
            artifacts.write_loop_state(state)

            failed_exec: dict[str, Any] | None = None
            failed_step: StageStep | None = None

            for step in stage.steps:
                iteration += 1
                state.iteration = iteration

                exec_result = executor.run(
                    command=step.command,
                    stage_name=stage.name,
                    step_name=step.name,
                )
                judge_result = judge.score(exec_result, previous_warnings=previous_warnings)
                previous_warnings = int(judge_result.get("warnings", 0))
                score = float(judge_result.get("score", 0.0))

                signature = str(exec_result.get("parsed", {}).get("error_signature", ""))
                if signature == last_error_signature:
                    repeated_error_count += 1
                else:
                    repeated_error_count = 1
                    last_error_signature = signature

                if score > state.best_score:
                    state.best_score = score
                    no_progress_count = 0
                else:
                    no_progress_count += 1

                state.score = score

                iter_dir = artifacts.iteration_dir(iteration)
                artifacts.write_latest_exec(exec_result)
                artifacts.write_iteration_exec(iteration, exec_result)
                artifacts.write_iteration_payload(iteration, "judge.json", judge_result)

                print(
                    f"iter={iteration} stage={stage.name} step={step.name} rc={exec_result['return_code']} artifacts={iter_dir}",
                    flush=True,
                )

                if int(exec_result.get("return_code", 1)) != 0:
                    failed_exec = exec_result
                    failed_step = step
                    break

            if failed_exec is None:
                checks_passed, check_failures = _evaluate_stage_checks(
                    stage,
                    config.workspace_root,
                    stage_start_time=stage_start_time,
                )
                if checks_passed:
                    stage_passed = True
                    break

                iteration += 1
                state.iteration = iteration
                failed_step = None
                failed_exec = {
                    "command": "stage_pass_checks",
                    "cwd": str(config.target_dir),
                    "stage_name": stage.name,
                    "step_name": "pass_checks",
                    "return_code": 1,
                    "stdout": "",
                    "stderr": "\n".join(check_failures),
                    "duration_sec": 0.0,
                    "parsed": {
                        "failed_tests": None,
                        "warnings": None,
                        "error_signature": "|".join(check_failures) if check_failures else "pass_checks_failed",
                    },
                }
                judge_result = judge.score(failed_exec, previous_warnings=previous_warnings)
                score = float(judge_result.get("score", 0.0))
                state.score = score

                iter_dir = artifacts.iteration_dir(iteration)
                artifacts.write_latest_exec(failed_exec)
                artifacts.write_iteration_exec(iteration, failed_exec)
                artifacts.write_iteration_payload(iteration, "judge.json", judge_result)
                artifacts.write_iteration_payload(
                    iteration,
                    "pass_checks.json",
                    {"ok": False, "failures": check_failures},
                )

                print(
                    f"iter={iteration} stage={stage.name} step=pass_checks rc=1 artifacts={iter_dir}",
                    flush=True,
                )

            review_context = _build_reviewer_context(
                config=config,
                iteration=iteration,
                latest_exec=failed_exec,
                previous_patch=previous_patch,
                stage=stage,
                step=failed_step,
            )
            artifacts.write_iteration_payload(iteration, "review_context.json", review_context)

            max_retries = max(int(config.reviewer.max_retries), 0)
            retry_backoff = config.reviewer.retry_backoff_sec or [10, 30]
            review_result = reviewer.review(review_context)
            retry_attempt = 0
            while (not review_result.ok) and bool(review_result.error_meta.get("retriable", False)) and retry_attempt < max_retries:
                sleep_sec = retry_backoff[min(retry_attempt, len(retry_backoff) - 1)]
                retry_attempt += 1
                print(
                    f"reviewer retry {retry_attempt}/{max_retries} after {sleep_sec}s: class={review_result.error_meta.get('api_error_class', 'unknown')} status={review_result.error_meta.get('http_status')}",
                    flush=True,
                )
                time.sleep(max(0, int(sleep_sec)))
                review_result = reviewer.review(review_context)

            if not review_result.ok:
                error_meta = review_result.error_meta or {}
                artifacts.write_iteration_payload(
                    iteration,
                    "review_error.json",
                    {
                        "timestamp_utc": _json_now(),
                        "error_type": error_meta.get("error_type", "reviewer_api_error"),
                        "api_error_class": error_meta.get("api_error_class", "unknown"),
                        "retriable": bool(error_meta.get("retriable", False)),
                        "http_status": error_meta.get("http_status"),
                        "error_code": error_meta.get("error_code"),
                        "retry_attempts": retry_attempt,
                        "error": review_result.error,
                        "model": config.reviewer.model,
                    },
                )
                state.status = "error"
                state.done = True
                state.reason = "reviewer_api_failed"
                state.metadata["review_error"] = review_result.error
                state.metadata["api_error_class"] = error_meta.get("api_error_class", "unknown")
                state.metadata["retriable"] = bool(error_meta.get("retriable", False))
                state.metadata["retry_attempts"] = retry_attempt
                artifacts.write_loop_state(state)
                api_class = str(error_meta.get("api_error_class", "unknown"))
                retriable = bool(error_meta.get("retriable", False))
                http_status = error_meta.get("http_status")

                summary = {
                    "event": "reviewer_failure",
                    "api_error_class": api_class,
                    "retriable": retriable,
                    "http_status": http_status,
                    "retry_attempts": retry_attempt,
                }
                print(
                    f"reviewer failed: class={api_class} retriable={retriable} status={http_status} retries={retry_attempt}",
                    flush=True,
                )
                print(json.dumps(summary, ensure_ascii=True), flush=True)

                if api_class == "insufficient_quota":
                    return 12
                if retriable:
                    return 13
                return 4

            artifacts.write_iteration_payload(iteration, "review.json", review_result.payload)
            artifacts.write_iteration_payload(iteration, "review_raw.txt", review_result.raw_text)

            patch_text = str(review_result.payload.get("patch", ""))
            artifacts.write_latest_patch(patch_text)
            artifacts.write_iteration_payload(iteration, "patch.diff", patch_text)

            changed_lines = patcher.changed_line_count(patch_text)
            if changed_lines > config.thresholds.max_patch_changed_lines:
                state.status = "stopped"
                state.done = True
                state.reason = "patch_too_large"
                state.metadata["changed_lines"] = changed_lines
                artifacts.write_loop_state(state)
                return 5

            patch_result = patcher.apply(patch_text)
            artifacts.write_iteration_payload(
                iteration,
                "patch_apply.json",
                {"applied": patch_result.applied, "message": patch_result.message},
            )
            if not patch_result.applied:
                state.status = "error"
                state.done = True
                state.reason = "patch_apply_failed"
                state.metadata["patch_error"] = patch_result.message
                artifacts.write_loop_state(state)
                return 6

            previous_patch = patch_text
            artifacts.write_loop_state(state)

        if not stage_passed:
            state.status = "stopped"
            state.done = True
            state.reason = "stage_max_iters_reached"
            state.metadata["failed_stage_name"] = stage.name
            state.metadata["failed_stage_index"] = stage_index
            artifacts.write_loop_state(state)
            return 1

        if config.stop_after_stage and stage.name == config.stop_after_stage:
            state.status = "stopped_after_stage"
            state.done = True
            state.reason = "stop_after_stage"
            state.metadata["completed_up_to_stage"] = stage.name
            state.metadata["completed_up_to_stage_index"] = stage_index
            artifacts.write_loop_state(state)
            return 0

        repeated_error_count = 0
        no_progress_count = 0
        previous_warnings = 0
        last_error_signature = ""
        start_stage_iter = 0

    state.status = "completed"
    state.done = True
    state.reason = "all_stages_passed"
    state.metadata["completed_at"] = _json_now()
    artifacts.write_loop_state(state)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic code review loop orchestrator")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    _load_dotenv(Path.cwd() / ".env")
    config = load_config(args.config)
    missing_modules = _check_runtime_dependencies(config.reviewer.provider)
    if missing_modules:
        print(
            json.dumps(
                {
                    "exit_code": 2,
                    "error": "missing_runtime_dependencies",
                    "missing_modules": missing_modules,
                    "hint": "Install with: pip install -e .",
                },
                ensure_ascii=True,
            )
        )
        raise SystemExit(2)

    code = run_loop(config)
    print(json.dumps({"exit_code": code}, ensure_ascii=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
