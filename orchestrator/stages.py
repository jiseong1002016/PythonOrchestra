from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageStep:
    name: str
    command: str


@dataclass
class StageCheck:
    required_files: list[Any] = field(default_factory=list)
    required_globs: list[str] = field(default_factory=list)
    require_new_files_since_stage_start: bool = False
    csv_header_checks: list[dict[str, Any]] = field(default_factory=list)
    required_regex: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Stage:
    name: str
    description: str = ""
    steps: list[StageStep] = field(default_factory=list)
    pass_checks: StageCheck = field(default_factory=StageCheck)
    max_iters: int | None = None


def load_stages_from_config(raw: dict[str, Any], default_max_iters: int) -> list[Stage]:
    stage_items = raw.get("stages")
    if not isinstance(stage_items, list):
        return []

    stages: list[Stage] = []
    for index, item in enumerate(stage_items):
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or f"stage_{index}")
        description = str(item.get("description") or "")

        steps_data = item.get("steps")
        steps: list[StageStep] = []
        if isinstance(steps_data, list):
            for sidx, step_item in enumerate(steps_data):
                if not isinstance(step_item, dict):
                    continue
                command = str(step_item.get("command") or "").strip()
                if not command:
                    continue
                step_name = str(step_item.get("name") or f"step_{sidx}")
                steps.append(StageStep(name=step_name, command=command))

        checks_raw = item.get("checks")
        if not isinstance(checks_raw, dict):
            checks_raw = item.get("pass_checks")
        check = StageCheck()
        if isinstance(checks_raw, dict):
            rf = checks_raw.get("required_files")
            if isinstance(rf, list):
                check.required_files = rf

            rg = checks_raw.get("required_globs")
            if isinstance(rg, list):
                check.required_globs = [str(g) for g in rg if str(g).strip()]

            check.require_new_files_since_stage_start = bool(
                checks_raw.get("require_new_files_since_stage_start", False)
            )

            ch = checks_raw.get("csv_header_checks")
            if isinstance(ch, list):
                check.csv_header_checks = [c for c in ch if isinstance(c, dict)]

            rr = checks_raw.get("required_regex")
            if isinstance(rr, list):
                check.required_regex = [r for r in rr if isinstance(r, dict)]

        max_iters_raw = item.get("max_iters")
        if max_iters_raw is None:
            stage_max_iters: int | None = default_max_iters
        else:
            stage_max_iters = max(int(max_iters_raw), 1)

        stages.append(
            Stage(
                name=name,
                description=description,
                steps=steps,
                pass_checks=check,
                max_iters=stage_max_iters,
            )
        )

    return stages
