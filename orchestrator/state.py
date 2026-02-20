from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LoopState:
    status: str = "initialized"
    iteration: int = 0
    max_iters: int = 0
    score: float = 0.0
    best_score: float = -10_000.0
    done: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def write_loop_state(self, state: LoopState) -> None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": state.status,
            "iteration": state.iteration,
            "max_iters": state.max_iters,
            "score": state.score,
            "best_score": state.best_score,
            "done": state.done,
            "reason": state.reason,
            "metadata": state.metadata,
        }
        self._write_json(self.artifacts_dir / "loop_state.json", payload)

    def read_loop_state(self) -> dict[str, Any] | None:
        path = self.artifacts_dir / "loop_state.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def write_latest_exec(self, payload: dict[str, Any]) -> None:
        self._write_json(self.artifacts_dir / "latest_exec.json", payload)

    def write_iteration_exec(self, iteration: int, payload: dict[str, Any]) -> None:
        idir = self.iteration_dir(iteration)
        self._write_json(idir / "exec.json", payload)
        self._write_json(idir / "latest_exec.json", payload)

    def write_latest_patch(self, patch_text: str) -> None:
        (self.artifacts_dir / "latest.patch").write_text(patch_text, encoding="utf-8")

    def iteration_dir(self, iteration: int) -> Path:
        path = self.artifacts_dir / f"iter_{iteration:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_iteration_payload(
        self,
        iteration: int,
        filename: str,
        payload: dict[str, Any] | str,
    ) -> None:
        idir = self.iteration_dir(iteration)
        out_path = idir / filename
        if isinstance(payload, str):
            out_path.write_text(payload, encoding="utf-8")
        else:
            self._write_json(out_path, payload)
