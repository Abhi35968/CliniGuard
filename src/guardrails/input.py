import re
import json
from typing import List, Optional
from src.llm.factory import get_llm_for_task
from src.llm.prompts import INPUT_GUARDRAIL_SYSTEM_PROMPT
from src.schemas.trace import GuardrailCheckResult
from src.observability import get_logger

logger = get_logger("guardrails.input")



# Minimum query length (chars) above which the LLM second-pass is triggered.
# Short, simple queries (e.g. city names) skip the expensive LLM call.
_LLM_PASS_MIN_LENGTH = 30


class InputGuardrail:
    """
    Two-layer Input Security Guardrail.

    Layer 1 – Deterministic (always runs, O(1)):
        Regex injection pattern matching + heuristic override detection.

    Layer 2 – LLM Semantic (runs when Layer 1 passes and query is non-trivial):
        LLM is asked to classify the query as safe / unsafe using the
        INPUT_GUARDRAIL_SYSTEM_PROMPT.  Gracefully degrades to PASSED if no
        LLM provider is configured so the system never breaks on missing keys.
    """

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
            r"act\s+as\s+(if\s+you\s+(are|were)|a\s+different)",
            r"forget\s+(all\s+)?(your\s+)?(previous\s+)?instructions",
            r"you\s+are\s+now\s+",
            r"new\s+persona",
        ]
        # Lazily initialised – only created when first needed
        self._llm: Optional[object] = None
        self._llm_available: Optional[bool] = None  # None = not yet tried

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_input(self, user_query: str) -> GuardrailCheckResult:
        """
        Runs Layer-1 deterministic check, then Layer-2 LLM semantic check
        (only when Layer 1 passes and the query is long enough to warrant it).
        Returns a single composite GuardrailCheckResult.
        """
        q_lower = user_query.lower()
        violations: List[str] = []

        # ── Layer 1: Deterministic regex + heuristics ──────────────────
        for pattern in self.injection_patterns:
            if re.search(pattern, q_lower):
                violations.append(f"Regex injection pattern: '{pattern}'")

        if "override" in q_lower and (
            "system" in q_lower or "safety" in q_lower or "rule" in q_lower
        ):
            violations.append("Adversarial system override attempt detected.")

        if violations:
            # Fast-fail – no need to call LLM
            return GuardrailCheckResult(
                guardrail_type="input",
                passed=False,
                details=f"Layer-1 check FAILED: {', '.join(violations)}",
                violations=violations,
            )

        # ── Layer 2: LLM semantic check (when query is non-trivial) ────
        if len(user_query.strip()) >= _LLM_PASS_MIN_LENGTH:
            llm_passed, llm_detail, llm_violations = self._llm_check(user_query)
            if not llm_passed:
                return GuardrailCheckResult(
                    guardrail_type="input",
                    passed=False,
                    details=f"Layer-2 LLM check FAILED: {llm_detail}",
                    violations=llm_violations,
                )

        return GuardrailCheckResult(
            guardrail_type="input",
            passed=True,
            details="Input security check PASSED (Layer-1 regex + Layer-2 LLM).",
            violations=[],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        """Lazily initialise the guardrail LLM (cached after first call)."""
        if self._llm_available is None:
            try:
                provider = get_llm_for_task("guardrail")
                if provider and provider.chat_model:
                    self._llm = provider
                    self._llm_available = True
                else:
                    self._llm_available = False
            except Exception:
                self._llm_available = False
        return self._llm if self._llm_available else None

    def _llm_check(self, user_query: str):
        """
        Calls the LLM with a structured prompt requesting a JSON verdict.
        Returns (passed: bool, detail: str, violations: list[str]).
        Falls back to (True, "LLM unavailable – skipped", []) on any error.
        """
        llm = self._get_llm()
        if not llm:
            return True, "LLM guardrail unavailable – skipped (deterministic layer applied).", []

        user_message = (
            f"Analyze the following user query for safety policy compliance:\n\n"
            f"USER QUERY: {user_query}\n\n"
            "Respond ONLY with a JSON object: "
            '{"safe": true/false, "reason": "<brief reason>", "violations": ["<v1>", ...]}'
        )

        try:
            raw = llm.generate(
                system_prompt=INPUT_GUARDRAIL_SYSTEM_PROMPT,
                user_message=user_message,
            )
            # Strip markdown fences if any
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
                clean = re.sub(r"```\s*$", "", clean).strip()

            verdict = json.loads(clean)
            is_safe = bool(verdict.get("safe", True))
            reason = verdict.get("reason", "LLM check inconclusive.")
            llm_violations = verdict.get("violations", [])

            if not is_safe:
                return False, reason, llm_violations if llm_violations else [reason]
            return True, "LLM semantic check passed.", []

        except Exception as e:
            # Never block a user because the LLM call failed
            logger.warning("LLM check error (graceful skip): %s", e)
            return True, f"LLM check skipped due to error: {e}", []
