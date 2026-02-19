from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExecutorConfig:
    command: str = "pytest -q"


@dataclass
class ReviewerConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.1
    max_output_tokens: int = 4000


@dataclass
class ThresholdConfig:
    success_score: float = 10.0
    same_error_repeats: int = 3
    no_progress_repeats: int = 3
    max_patch_changed_lines: int = 200


@dataclass
class Config:
    workspace_root: Path
    target_dir: Path
    max_iters: int = 10
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    judge: dict[str, Any] = field(default_factory=dict)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)


def _deep_get(data: dict[str, Any], path: list[str], default: Any) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    workspace_root = Path(raw.get("workspace_root", config_path.parent.parent)).resolve()
    target_dir = (workspace_root / raw.get("target_dir", "target")).resolve()

    return Config(
        workspace_root=workspace_root,
        target_dir=target_dir,
        max_iters=int(raw.get("max_iters", 10)),
        executor=ExecutorConfig(
            command=str(_deep_get(raw, ["executor", "command"], "pytest -q")),
        ),
        reviewer=ReviewerConfig(
            model=str(_deep_get(raw, ["reviewer", "model"], "gpt-4.1")),
            temperature=float(_deep_get(raw, ["reviewer", "temperature"], 0.1)),
            max_output_tokens=int(
                _deep_get(raw, ["reviewer", "max_output_tokens"], 4000)
            ),
        ),
        judge=_deep_get(raw, ["judge"], {}) or {},
        thresholds=ThresholdConfig(
            success_score=float(
                _deep_get(raw, ["thresholds", "success_score"], 10.0)
            ),
            same_error_repeats=int(
                _deep_get(raw, ["thresholds", "same_error_repeats"], 3)
            ),
            no_progress_repeats=int(
                _deep_get(raw, ["thresholds", "no_progress_repeats"], 3)
            ),
            max_patch_changed_lines=int(
                _deep_get(raw, ["thresholds", "max_patch_changed_lines"], 200)
            ),
        ),
    )
