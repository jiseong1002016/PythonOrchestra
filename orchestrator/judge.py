from __future__ import annotations

from typing import Any


class Judge:
    def score(self, exec_result: dict[str, Any]) -> dict[str, Any]:
        rc = int(exec_result.get("return_code", 1))
        parsed = exec_result.get("parsed", {})
        failed_tests = parsed.get("failed_tests")
        warnings = parsed.get("warnings") or 0

        score = 0.0
        reasons: list[str] = []

        if rc == 0:
            score += 10.0
            reasons.append("tests_passed")
        else:
            if isinstance(failed_tests, int):
                penalty = failed_tests * 2
                score -= float(penalty)
                reasons.append(f"failed_tests_penalty:{penalty}")
            else:
                score -= 5.0
                reasons.append("nonzero_return_penalty:5")

        if warnings:
            warning_penalty = min(float(warnings) * 0.25, 2.0)
            score -= warning_penalty
            reasons.append(f"warnings_penalty:{warning_penalty}")

        return {
            "score": round(score, 3),
            "passed": rc == 0,
            "reasons": reasons,
        }
