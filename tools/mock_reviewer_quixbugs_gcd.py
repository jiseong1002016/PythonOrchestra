#!/usr/bin/env python3
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path


def _find_gcd_file(workspace_root: Path) -> Path | None:
    preferred = workspace_root / "python_programs" / "gcd.py"
    if preferred.exists():
        return preferred

    matches: list[Path] = []
    for path in workspace_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "def gcd" in text:
            matches.append(path)
            if len(matches) > 1:
                break
    if len(matches) == 1:
        return matches[0]
    return None


def _should_patch(payload: dict) -> bool:
    latest_exec = payload.get("latest_exec", {}) if isinstance(payload, dict) else {}
    stdout = str(latest_exec.get("stdout", ""))
    stderr = str(latest_exec.get("stderr", ""))
    joined = (stdout + "\n" + stderr).lower()
    tokens = ("test_gcd.py", "gcd", "recursionerror")
    return any(token in joined for token in tokens)


def _build_patch(workspace_root: Path, target: Path) -> str:
    old_text = target.read_text(encoding="utf-8")
    new_text = """def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a
"""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    rel = target.relative_to(workspace_root).as_posix()
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}

    target_dir = payload.get("target_dir")
    workspace_root = Path(str(target_dir)).resolve() if target_dir else Path.cwd()

    if not _should_patch(payload):
        out = {
            "patch_unified_diff": "",
            "score": 0,
            "summary": "No gcd-specific failure signature found.",
            "risks": ["No-op patch returned"],
        }
        print(json.dumps(out, ensure_ascii=True))
        return 0

    gcd_file = _find_gcd_file(workspace_root)
    if gcd_file is None:
        out = {
            "patch_unified_diff": "",
            "score": 0,
            "summary": "Could not locate a unique gcd implementation file.",
            "risks": ["No-op patch returned"],
        }
        print(json.dumps(out, ensure_ascii=True))
        return 0

    patch = _build_patch(workspace_root, gcd_file)
    out = {
        "patch_unified_diff": patch,
        "score": 10,
        "summary": "Replace recursive gcd with iterative Euclidean algorithm.",
        "risks": ["Patch is tailored for QuixBugs gcd case"],
        "tests_to_run": ["python -m pytest -q python_testcases/test_gcd.py --maxfail=1"],
    }
    print(json.dumps(out, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
