import re
from typing import Tuple, List, Dict, Any, Optional
from src.llm.factory import get_llm_for_task
from src.llm.prompts import INPUT_GUARDRAIL_SYSTEM_PROMPT
from src.schemas.trace import GuardrailCheckResult


class InputGuardrail:
    """Layered Input Security Guardrail for prompt injection and policy bypass detection."""

    def __init__(self):
        self.injection_patterns = [
            r"ignore\s+all\s+(previous\s+)?instructions",
            r"override\s+system\s+policy",
            r"bypass\s+safety\s+rules",
            r"claim\s+.*100%\s+safe",
            r"pretend\s+the\s+wind\s+is",
            r"fake\s+policy",
            r"sop-999",
            r"disregard\s+sop",
        ]

    def validate_input(self, user_query: str) -> GuardrailCheckResult:
        """Validates user query against prompt injection and security policies."""
        q_lower = user_query.lower()
        violations: List[str] = []

        # 1. Deterministic Regex Pattern Match
        for pattern in self.injection_patterns:
            if re.search(pattern, q_lower):
                violations.append(f"Prompt injection pattern detected: '{pattern}'")

        # 2. Heuristic check for adversarial override attempts
        if "override" in q_lower and ("system" in q_lower or "safety" in q_lower or "rule" in q_lower):
            violations.append("Adversarial system override attempt detected.")

        passed = len(violations) == 0
        details = "Input security check PASSED." if passed else f"Input security check FAILED: {', '.join(violations)}"

        return GuardrailCheckResult(
            guardrail_type="input",
            passed=passed,
            details=details,
            violations=violations,
        )
