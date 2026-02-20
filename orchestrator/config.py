from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from orchestrator.stages import Stage, load_stages_from_config


@dataclass
class ExecutorConfig:
    command: str = "pytest -q"
    timeout_sec: int = 300


@dataclass
class ReviewerConfig:
    provider: str = "openai"
    model: str = "gpt-4.1"
    temperature: float = 0.1
    max_output_tokens: int = 4000
    command: str | None = None
    command_timeout_sec: int = 120
    max_retries: int = 2
    retry_backoff_sec: list[int] = field(default_factory=lambda: [10, 30])


@dataclass
class ThresholdConfig:
    success_score: float = 10.0
    same_error_repeats: int = 3
    no_progress_repeats: int = 3
    max_patch_changed_lines: int = 200


@dataclass
class PathsConfig:
    artifacts_dir: Path
    prompt_dir: Path


@dataclass
class Config:
    workspace_root: Path
    target_dir: Path
    max_iters: int = 10
    stop_after_stage: str | None = None
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    judge: dict[str, Any] = field(default_factory=dict)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    paths: PathsConfig | None = None
    stages: list[Stage] = field(default_factory=list)


def _deep_get(data: dict[str, Any], path: list[str], default: Any) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_under(base: Path, raw_path: str | None, default_rel: str) -> Path:
    value = (raw_path or "").strip()
    if not value:
        return (base / default_rel).resolve()
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def _to_int_list(value: Any, default: list[int]) -> list[int]:
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:  # noqa: BLE001
                continue
        return out or default
    try:
        return [int(value)]
    except Exception:  # noqa: BLE001
        return default


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = _load_yaml_like(config_path.read_text(encoding="utf-8"))

    workspace_root = Path(raw.get("workspace_root", config_path.parent.parent)).resolve()
    target_dir = (workspace_root / raw.get("target_dir", "target")).resolve()

    paths_raw = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}
    artifacts_dir = _resolve_under(
        workspace_root,
        str(paths_raw.get("artifacts_dir")) if paths_raw.get("artifacts_dir") is not None else None,
        "artifacts",
    )
    prompt_dir = _resolve_under(
        workspace_root,
        str(paths_raw.get("prompt_dir")) if paths_raw.get("prompt_dir") is not None else None,
        "orchestrator/prompts",
    )

    max_iters = int(raw.get("max_iters", 10))

    return Config(
        workspace_root=workspace_root,
        target_dir=target_dir,
        max_iters=max_iters,
        stop_after_stage=(str(raw.get("stop_after_stage", "")).strip() or None),
        executor=ExecutorConfig(
            command=str(_deep_get(raw, ["executor", "command"], "pytest -q")),
            timeout_sec=int(_deep_get(raw, ["executor", "timeout_sec"], 300)),
        ),
        reviewer=ReviewerConfig(
            provider=str(_deep_get(raw, ["reviewer", "provider"], "openai")),
            model=str(_deep_get(raw, ["reviewer", "model"], "gpt-4.1")),
            temperature=float(_deep_get(raw, ["reviewer", "temperature"], 0.1)),
            max_output_tokens=int(
                _deep_get(raw, ["reviewer", "max_output_tokens"], 4000)
            ),
            command=(
                str(_deep_get(raw, ["reviewer", "command"], "")).strip() or None
            ),
            command_timeout_sec=int(
                _deep_get(raw, ["reviewer", "command_timeout_sec"], 120)
            ),
            max_retries=int(_deep_get(raw, ["reviewer", "max_retries"], 2)),
            retry_backoff_sec=_to_int_list(
                _deep_get(raw, ["reviewer", "retry_backoff_sec"], [10, 30]),
                [10, 30],
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
        paths=PathsConfig(artifacts_dir=artifacts_dir, prompt_dir=prompt_dir),
        stages=load_stages_from_config(raw, default_max_iters=max_iters),
    )


def _load_yaml_like(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        pass

    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)

    return _simple_yaml_load(text)


def _simple_yaml_load(text: str) -> dict[str, Any]:
    lines = _yaml_lines(text)
    if not lines:
        return {}
    value, _ = _parse_yaml_block(lines, 0, lines[0][0])
    return value if isinstance(value, dict) else {}


def _yaml_lines(text: str) -> list[tuple[int, str]]:
    parsed: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line = _strip_inline_comment(raw.rstrip())
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        parsed.append((indent, line.strip()))
    return parsed


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    out: list[str] = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_yaml_block(
    lines: list[tuple[int, str]],
    idx: int,
    indent: int,
) -> tuple[Any, int]:
    if idx >= len(lines):
        return {}, idx
    _, content = lines[idx]
    if content.startswith("- "):
        return _parse_yaml_list(lines, idx, indent)
    return _parse_yaml_dict(lines, idx, indent)


def _parse_yaml_dict(
    lines: list[tuple[int, str]],
    idx: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    i = idx
    while i < len(lines):
        line_indent, content = lines[i]
        if line_indent < indent:
            break
        if line_indent > indent:
            i += 1
            continue
        if content.startswith("- "):
            break
        if ":" not in content:
            i += 1
            continue

        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value:
            out[key] = _parse_scalar(value)
            i += 1
            continue

        next_i = i + 1
        if next_i >= len(lines) or lines[next_i][0] <= line_indent:
            out[key] = {}
            i += 1
            continue

        nested, consumed = _parse_yaml_block(lines, next_i, lines[next_i][0])
        out[key] = nested
        i = consumed
    return out, i


def _parse_yaml_list(
    lines: list[tuple[int, str]],
    idx: int,
    indent: int,
) -> tuple[list[Any], int]:
    out: list[Any] = []
    i = idx
    while i < len(lines):
        line_indent, content = lines[i]
        if line_indent < indent:
            break
        if line_indent != indent or not content.startswith("- "):
            break

        item_content = content[2:].strip()
        if not item_content:
            next_i = i + 1
            if next_i < len(lines) and lines[next_i][0] > line_indent:
                nested, consumed = _parse_yaml_block(lines, next_i, lines[next_i][0])
                out.append(nested)
                i = consumed
            else:
                out.append(None)
                i += 1
            continue

        if ":" in item_content:
            key, value = item_content.split(":", 1)
            item_dict: dict[str, Any] = {key.strip(): _parse_scalar(value.strip()) if value.strip() else {}}
            next_i = i + 1
            while next_i < len(lines) and lines[next_i][0] > line_indent:
                nested_indent = lines[next_i][0]
                nested, consumed = _parse_yaml_block(lines, next_i, nested_indent)
                if isinstance(nested, dict):
                    item_dict.update(nested)
                next_i = consumed
            out.append(item_dict)
            i = next_i
            continue

        out.append(_parse_scalar(item_content))
        i += 1
    return out, i


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts: list[str] = []
        token: list[str] = []
        in_single = False
        in_double = False
        for ch in inner:
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            if ch == "," and not in_single and not in_double:
                parts.append("".join(token).strip())
                token = []
                continue
            token.append(ch)
        if token:
            parts.append("".join(token).strip())
        return [_parse_scalar(part) for part in parts if part]

    low = value.lower()
    if low in {"null", "none"}:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value
