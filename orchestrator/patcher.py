from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchResult:
    applied: bool
    message: str


class Patcher:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def changed_line_count(self, patch_text: str) -> int:
        count = 0
        for line in patch_text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+") or line.startswith("-"):
                count += 1
        return count

    def apply(self, patch_text: str) -> PatchResult:
        if not patch_text.strip():
            return PatchResult(False, "empty patch")

        if self._is_git_repo():
            return self._apply_with_git(patch_text)
        return self._apply_with_python(patch_text)

    def _is_git_repo(self) -> bool:
        proc = subprocess.run(
            ["git", "-C", str(self.workspace_root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0

    def _apply_with_git(self, patch_text: str) -> PatchResult:
        proc = subprocess.run(
            ["git", "-C", str(self.workspace_root), "apply", "--whitespace=nowarn", "-"],
            input=patch_text,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return PatchResult(True, "applied with git apply")
        return PatchResult(False, f"git apply failed: {proc.stderr.strip()}")

    def _apply_with_python(self, patch_text: str) -> PatchResult:
        try:
            files = self._parse_unified_diff(patch_text)
            for rel_path, hunks in files:
                self._apply_file_hunks(rel_path, hunks)
            return PatchResult(True, "applied with python patcher")
        except Exception as exc:  # noqa: BLE001
            return PatchResult(False, f"python patcher failed: {exc}")

    def _parse_unified_diff(self, patch_text: str) -> list[tuple[str, list[dict[str, object]]]]:
        lines = patch_text.splitlines()
        idx = 0
        parsed: list[tuple[str, list[dict[str, object]]]] = []

        while idx < len(lines):
            line = lines[idx]
            if not line.startswith("--- "):
                idx += 1
                continue
            if idx + 1 >= len(lines) or not lines[idx + 1].startswith("+++ "):
                raise ValueError("invalid diff header")
            new_path = lines[idx + 1][4:].split("\t", 1)[0].strip()
            rel = self._normalize_diff_path(new_path)
            idx += 2
            hunks: list[dict[str, object]] = []

            while idx < len(lines) and lines[idx].startswith("@@"):
                header = lines[idx]
                m = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", header)
                if not m:
                    raise ValueError(f"invalid hunk header: {header}")
                idx += 1
                hunk_lines: list[str] = []
                while idx < len(lines) and not lines[idx].startswith("@@") and not lines[idx].startswith("--- "):
                    hunk_lines.append(lines[idx])
                    idx += 1
                hunks.append(
                    {
                        "old_start": int(m.group(1)),
                        "old_count": int(m.group(2)),
                        "new_start": int(m.group(3)),
                        "new_count": int(m.group(4)),
                        "lines": hunk_lines,
                    }
                )
            parsed.append((rel, hunks))

        if not parsed:
            raise ValueError("no file patches found")
        return parsed

    def _apply_file_hunks(self, rel_path: str, hunks: list[dict[str, object]]) -> None:
        path = self.workspace_root / rel_path
        if path.exists():
            content = path.read_text(encoding="utf-8").splitlines()
        else:
            content = []

        out: list[str] = []
        src_idx = 0

        for hunk in hunks:
            old_start = int(hunk["old_start"])
            lines = hunk["lines"]
            target_idx = old_start - 1
            if target_idx < src_idx:
                raise ValueError("overlapping hunks")
            out.extend(content[src_idx:target_idx])
            src_idx = target_idx

            assert isinstance(lines, list)
            for raw in lines:
                if raw.startswith("\\"):
                    continue
                prefix, body = raw[:1], raw[1:]
                if prefix == " ":
                    if src_idx >= len(content) or content[src_idx] != body:
                        raise ValueError("context mismatch")
                    out.append(content[src_idx])
                    src_idx += 1
                elif prefix == "-":
                    if src_idx >= len(content) or content[src_idx] != body:
                        raise ValueError("delete mismatch")
                    src_idx += 1
                elif prefix == "+":
                    out.append(body)
                else:
                    raise ValueError(f"unknown line prefix: {prefix}")

        out.extend(content[src_idx:])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(out) + "\n", encoding="utf-8")

    @staticmethod
    def _normalize_diff_path(path_text: str) -> str:
        path_text = path_text.strip()
        if path_text.startswith("a/") or path_text.startswith("b/"):
            return path_text[2:]
        return path_text
