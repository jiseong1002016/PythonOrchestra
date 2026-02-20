from __future__ import annotations

import re
from pathlib import Path

TRACEBACK_RE = re.compile(r'File "([^"]+)", line (\d+)')
CSTYLE_ERROR_RE = re.compile(r"([^\s:][^:\n]*):(\d+):(\d+):\s+error")


def parse_error_locations(text: str, max_locations: int = 3) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    for pattern in (TRACEBACK_RE, CSTYLE_ERROR_RE):
        for match in pattern.finditer(text):
            path = match.group(1).strip()
            try:
                line_no = int(match.group(2))
            except ValueError:
                continue
            key = (path, line_no)
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
            if len(found) >= max_locations:
                return found

    return found


def collect_snippets(
    workspace_root: Path,
    locations: list[tuple[str, int]],
    radius: int = 30,
    max_chars: int = 12_000,
) -> str:
    if max_chars <= 0:
        return ""

    chunks: list[str] = []
    used = 0

    for raw_path, line_no in locations:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (workspace_root / candidate).resolve()
        if not candidate.exists() or not candidate.is_file():
            continue

        try:
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        if not lines:
            continue

        start = max(line_no - radius, 1)
        end = min(line_no + radius, len(lines))

        body_lines = []
        for idx in range(start, end + 1):
            marker = ">" if idx == line_no else " "
            body_lines.append(f"{marker}{idx:5d} | {lines[idx - 1]}")
        body = "\n".join(body_lines)

        rel = _best_relpath(candidate, workspace_root)
        section = f"FILE: {rel}:{line_no}\n{body}\n"

        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = section[: max(remaining - 1, 0)]
        if not section:
            break

        chunks.append(section)
        used += len(section)
        if used >= max_chars:
            break

    return "\n---\n".join(chunks)


def _best_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
