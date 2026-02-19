from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class ReviewerResult:
    ok: bool
    payload: dict[str, Any]
    raw_text: str = ""
    error: str = ""


class Reviewer:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_output_tokens: int,
        prompt_dir: Path,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_dir = prompt_dir
        self.system_prompt = (prompt_dir / "reviewer_system.txt").read_text(encoding="utf-8")
        self.user_template = (prompt_dir / "reviewer_user.txt").read_text(encoding="utf-8")

    def review(self, context: dict[str, Any]) -> ReviewerResult:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return ReviewerResult(
                ok=False,
                payload={},
                error="OPENAI_API_KEY is not set",
            )

        prompt = self.user_template.format(
            iteration=context["iteration"],
            command=context["command"],
            target_dir=context["target_dir"],
            latest_exec_json=json.dumps(context["latest_exec"], indent=2, ensure_ascii=True),
            previous_patch=context.get("previous_patch", ""),
        )

        sdk_error: str | None = None
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                input=[
                    {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ],
            )
            text = getattr(response, "output_text", "") or ""
            if not text:
                text = self._extract_text_from_sdk_response(response)
            payload = self._parse_json_response(text)
            return ReviewerResult(ok=True, payload=payload, raw_text=text)
        except Exception as exc:  # noqa: BLE001
            sdk_error = f"openai_sdk_error: {exc}"

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            payload = self._parse_json_response(text)
            return ReviewerResult(ok=True, payload=payload, raw_text=text)
        except Exception as exc:  # noqa: BLE001
            return ReviewerResult(
                ok=False,
                payload={},
                error=f"{sdk_error or 'openai_request_failed'}; requests_error: {exc}",
            )

    @staticmethod
    def _extract_text_from_sdk_response(response: Any) -> str:
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return ""
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for c in content:
                text = getattr(c, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        data = json.loads(text)
        required = ["diagnosis", "plan", "patch", "tests_to_run", "risk", "confidence"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"missing_reviewer_keys: {', '.join(missing)}")
        if not isinstance(data["plan"], list) or not isinstance(data["tests_to_run"], list):
            raise ValueError("plan/tests_to_run must be arrays")
        return data
