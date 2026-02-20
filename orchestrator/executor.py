from __future__ import annotations

import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

FAILED_RE = re.compile(r"(\d+)\s+failed")
WARNING_RE = re.compile(r"(\d+)\s+warning")


class Executor:
    def __init__(self, target_dir: Path, command: str, timeout_sec: int = 300) -> None:
        self.target_dir = target_dir
        self.command = command
        self.timeout_sec = timeout_sec

    def run(
        self,
        command: str | None = None,
        stage_name: str | None = None,
        step_name: str | None = None,
    ) -> dict[str, Any]:
        actual_command = command or self.command
        start = time.monotonic()
        try:
            proc = subprocess.run(
                shlex.split(actual_command),
                cwd=self.target_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
            )
            return_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            return_code = 124
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        except FileNotFoundError as exc:
            return_code = 127
            stdout = ""
            stderr = str(exc)
        duration = time.monotonic() - start
        combined = f"{stdout}\n{stderr}".strip()

        failed_tests = self._parse_count(FAILED_RE, combined)
        warnings = self._parse_count(WARNING_RE, combined)

        return {
            "command": actual_command,
            "cwd": str(self.target_dir),
            "stage_name": stage_name,
            "step_name": step_name,
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "duration_sec": round(duration, 4),
            "parsed": {
                "failed_tests": failed_tests,
                "warnings": warnings,
                "error_signature": self._error_signature(combined),
            },
        }

    @staticmethod
    def _parse_count(pattern: re.Pattern[str], output: str) -> int | None:
        match = pattern.search(output)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _error_signature(output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        failed_lines = [line for line in lines if line.startswith("FAILED ")]
        if failed_lines:
            return "|".join(failed_lines[:5])
        return "|".join(lines[-5:]) if lines else "no-output"
