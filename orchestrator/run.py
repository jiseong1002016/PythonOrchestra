from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.config import Config, load_config
from orchestrator.executor import Executor
from orchestrator.judge import Judge
from orchestrator.patcher import Patcher
from orchestrator.reviewer import Reviewer
from orchestrator.state import ArtifactStore, LoopState


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


def _build_reviewer_context(
    config: Config,
    iteration: int,
    latest_exec: dict[str, Any],
    previous_patch: str,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "command": config.executor.command,
        "target_dir": str(config.target_dir),
        "latest_exec": latest_exec,
        "previous_patch": previous_patch,
    }


def _json_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_loop(config: Config) -> int:
    artifacts = ArtifactStore(config.workspace_root / "artifacts")
    state = LoopState(
        status="running",
        iteration=0,
        max_iters=config.max_iters,
        score=0.0,
        best_score=-10_000.0,
        done=False,
        metadata={
            "workspace_root": str(config.workspace_root),
            "target_dir": str(config.target_dir),
            "command": config.executor.command,
        },
    )
    artifacts.write_loop_state(state)

    ensure_git_repo(config.workspace_root)

    executor = Executor(config.target_dir, config.executor.command)
    reviewer = Reviewer(
        model=config.reviewer.model,
        temperature=config.reviewer.temperature,
        max_output_tokens=config.reviewer.max_output_tokens,
        prompt_dir=config.workspace_root / "orchestrator" / "prompts",
    )
    judge = Judge()
    patcher = Patcher(config.workspace_root)

    last_error_signature = ""
    repeated_error_count = 0
    no_progress_count = 0
    previous_patch = ""

    for i in range(1, config.max_iters + 1):
        state.iteration = i

        exec_result = executor.run()
        judge_result = judge.score(exec_result)
        score = float(judge_result["score"])
        passed = bool(judge_result["passed"])
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
        iter_dir = artifacts.iteration_dir(i)
        artifacts.write_latest_exec(exec_result)
        artifacts.write_iteration_payload(i, "exec.json", exec_result)
        artifacts.write_iteration_payload(i, "judge.json", judge_result)

        print(
            f"iter={i} score={score} passed={passed} artifacts={iter_dir}",
            flush=True,
        )

        if passed or score >= config.thresholds.success_score:
            state.status = "completed"
            state.done = True
            state.reason = "success_threshold_reached"
            state.metadata["completed_at"] = _json_now()
            artifacts.write_loop_state(state)
            return 0

        if repeated_error_count >= config.thresholds.same_error_repeats:
            state.status = "stopped"
            state.done = True
            state.reason = "same_error_signature_repeated"
            state.metadata["error_signature"] = signature
            artifacts.write_loop_state(state)
            return 2

        if no_progress_count >= config.thresholds.no_progress_repeats:
            state.status = "stopped"
            state.done = True
            state.reason = "no_score_improvement"
            artifacts.write_loop_state(state)
            return 3

        review_context = _build_reviewer_context(config, i, exec_result, previous_patch)
        artifacts.write_iteration_payload(i, "review_context.json", review_context)

        review_result = reviewer.review(review_context)
        if not review_result.ok:
            error_payload = {
                "timestamp_utc": _json_now(),
                "error": review_result.error,
                "model": config.reviewer.model,
            }
            artifacts.write_iteration_payload(i, "review_error.json", error_payload)
            state.status = "error"
            state.done = True
            state.reason = "reviewer_api_failed"
            state.metadata["review_error"] = review_result.error
            artifacts.write_loop_state(state)
            return 4

        artifacts.write_iteration_payload(i, "review.json", review_result.payload)
        artifacts.write_iteration_payload(i, "review_raw.txt", review_result.raw_text)

        patch_text = str(review_result.payload.get("patch", ""))
        artifacts.write_latest_patch(patch_text)
        artifacts.write_iteration_payload(i, "patch.diff", patch_text)

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
            i,
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

    state.status = "stopped"
    state.done = True
    state.reason = "max_iters_reached"
    artifacts.write_loop_state(state)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic code review loop orchestrator")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    code = run_loop(config)
    print(json.dumps({"exit_code": code}, ensure_ascii=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
