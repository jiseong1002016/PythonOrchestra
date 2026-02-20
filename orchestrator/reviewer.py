from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReviewerResult:
    ok: bool
    payload: dict[str, Any]
    raw_text: str = ""
    error: str = ""
    error_meta: dict[str, Any] = field(default_factory=dict)


class Reviewer:
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        prompt_dir: Path,
        command: str | None = None,
        command_timeout_sec: int = 120,
    ) -> None:
        self.provider = provider.strip().lower() or "openai"
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_dir = prompt_dir
        self.command = command
        self.command_timeout_sec = command_timeout_sec
        self.system_prompt = (prompt_dir / "reviewer_system.txt").read_text(encoding="utf-8")
        self.user_template = (prompt_dir / "reviewer_user.txt").read_text(encoding="utf-8")

    def review(self, context: dict[str, Any]) -> ReviewerResult:
        if self.provider == "command":
            return self._review_command(context)
        return self._review_openai(context)

    def _review_command(self, context: dict[str, Any]) -> ReviewerResult:
        if not self.command:
            return ReviewerResult(
                ok=False,
                payload={},
                error="reviewer.command is required when provider=command",
                error_meta={
                    "error_type": "reviewer_config_error",
                    "api_error_class": "missing_command",
                    "retriable": False,
                    "http_status": None,
                },
            )

        input_json = json.dumps(context, ensure_ascii=True)
        try:
            proc = subprocess.run(
                self.command,
                input=input_json,
                capture_output=True,
                text=True,
                shell=True,
                check=False,
                timeout=self.command_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return ReviewerResult(
                ok=False,
                payload={},
                error=f"reviewer_command_timeout: {exc}",
                error_meta={
                    "error_type": "reviewer_command_error",
                    "api_error_class": "command_timeout",
                    "retriable": False,
                    "http_status": None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ReviewerResult(
                ok=False,
                payload={},
                error=f"reviewer_command_exec_error: {exc}",
                error_meta={
                    "error_type": "reviewer_command_error",
                    "api_error_class": "command_exec_error",
                    "retriable": False,
                    "http_status": None,
                },
            )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return ReviewerResult(
                ok=False,
                payload={},
                error=f"reviewer_command_failed rc={proc.returncode} stderr={stderr}",
                error_meta={
                    "error_type": "reviewer_command_error",
                    "api_error_class": "command_nonzero_exit",
                    "retriable": False,
                    "http_status": None,
                },
            )
        if not stdout:
            return ReviewerResult(
                ok=False,
                payload={},
                error=f"reviewer_command_empty_stdout stderr={stderr}",
                error_meta={
                    "error_type": "reviewer_command_error",
                    "api_error_class": "command_empty_output",
                    "retriable": False,
                    "http_status": None,
                },
            )

        try:
            raw_payload = json.loads(stdout)
        except Exception as exc:  # noqa: BLE001
            return ReviewerResult(
                ok=False,
                payload={},
                error=f"reviewer_command_invalid_json: {exc}; stdout={stdout}; stderr={stderr}",
                error_meta={
                    "error_type": "reviewer_command_error",
                    "api_error_class": "command_invalid_json",
                    "retriable": False,
                    "http_status": None,
                },
            )

        try:
            payload = self._parse_command_json_response(raw_payload)
        except Exception as exc:  # noqa: BLE001
            return ReviewerResult(
                ok=False,
                payload={},
                error=f"reviewer_command_invalid_payload: {exc}; stdout={stdout}; stderr={stderr}",
                error_meta={
                    "error_type": "reviewer_command_error",
                    "api_error_class": "command_invalid_payload",
                    "retriable": False,
                    "http_status": None,
                },
            )
        return ReviewerResult(ok=True, payload=payload, raw_text=stdout)

    def _review_openai(self, context: dict[str, Any]) -> ReviewerResult:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return ReviewerResult(
                ok=False,
                payload={},
                error="OPENAI_API_KEY is not set",
                error_meta={
                    "error_type": "reviewer_config_error",
                    "api_error_class": "missing_api_key",
                    "retriable": False,
                    "http_status": None,
                },
            )

        prompt = self.user_template.format(
            iteration=context["iteration"],
            command=context["command"],
            target_dir=context["target_dir"],
            stage_name=context.get("stage_name", ""),
            step_name=context.get("step_name", ""),
            stage_goal=context.get("stage_goal", ""),
            code_context=context.get("code_context", ""),
            latest_exec_json=json.dumps(context["latest_exec"], indent=2, ensure_ascii=True),
            previous_patch=context.get("previous_patch", ""),
        )

        sdk_error: str | None = None
        sdk_meta: dict[str, Any] | None = None
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            if hasattr(client, "responses"):
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
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = response.choices[0].message.content or ""
            payload = self._parse_json_response(text)
            return ReviewerResult(ok=True, payload=payload, raw_text=text)
        except Exception as exc:  # noqa: BLE001
            sdk_error = f"openai_sdk_error: {exc}"
            sdk_meta = self._classify_api_error(exc)

        try:
            import requests

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
            req_meta = self._classify_api_error(exc)
            merged_meta = req_meta if req_meta.get("api_error_class") != "unknown" else (sdk_meta or req_meta)
            return ReviewerResult(
                ok=False,
                payload={},
                error=f"{sdk_error or 'openai_request_failed'}; requests_error: {exc}",
                error_meta=merged_meta,
            )

    @staticmethod
    def _classify_api_error(exc: Exception) -> dict[str, Any]:
        text = str(exc)
        text_lower = text.lower()

        http_status = getattr(exc, "status_code", None)
        error_code: str | None = None

        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if isinstance(status, int):
                http_status = status
            try:
                data = response.json()
                if isinstance(data, dict):
                    err = data.get("error")
                    if isinstance(err, dict):
                        code_value = err.get("code")
                        if code_value is not None:
                            error_code = str(code_value)
            except Exception:  # noqa: BLE001
                pass

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                code_value = err.get("code")
                if code_value is not None:
                    error_code = str(code_value)

        api_error_class = "unknown"
        retriable = False

        if error_code == "insufficient_quota" or "insufficient_quota" in text_lower:
            api_error_class = "insufficient_quota"
            retriable = False
        elif (http_status == 429) or ("rate limit" in text_lower) or ("too many requests" in text_lower):
            api_error_class = "rate_limit"
            retriable = True
        elif (http_status in {500, 502, 503, 504}) or ("temporarily unavailable" in text_lower) or ("overloaded" in text_lower):
            api_error_class = "temporarily_unavailable"
            retriable = True

        return {
            "error_type": "reviewer_api_error",
            "api_error_class": api_error_class,
            "retriable": retriable,
            "http_status": http_status,
            "error_code": error_code,
        }

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

    @staticmethod
    def _parse_command_json_response(data: dict[str, Any]) -> dict[str, Any]:
        # Accept a lightweight command-reviewer schema and convert to existing reviewer payload.
        if not isinstance(data, dict):
            raise ValueError("command reviewer output must be a JSON object")

        if "patch_unified_diff" not in data:
            raise ValueError("missing required key: patch_unified_diff")
        patch = data.get("patch_unified_diff")
        if not isinstance(patch, str):
            raise ValueError("patch_unified_diff must be a string")

        summary = data.get("summary", "")
        score = data.get("score", 0)
        risks = data.get("risks", [])
        tests = data.get("tests_to_run", [])

        risk_list: list[str]
        if isinstance(risks, list):
            risk_list = [str(x) for x in risks]
        elif risks is None:
            risk_list = []
        else:
            risk_list = [str(risks)]

        tests_list: list[str]
        if isinstance(tests, list):
            tests_list = [str(x) for x in tests]
        elif tests is None:
            tests_list = []
        else:
            tests_list = [str(tests)]

        try:
            confidence = float(score)
        except Exception:  # noqa: BLE001
            confidence = 0.0

        return {
            "diagnosis": str(summary),
            "plan": [str(summary)] if str(summary).strip() else [],
            "patch": patch,
            "tests_to_run": tests_list,
            "risk": risk_list,
            "confidence": confidence,
        }
