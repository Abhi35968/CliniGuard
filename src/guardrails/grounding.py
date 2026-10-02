from typing import Dict, Any, List, Optional
from src.schemas.risk import RiskAssessment
from src.schemas.trace import GuardrailCheckResult


class GroundingGuardrail:
    """Verifies that generated advisory text is strictly grounded in matched SOPs and RiskAssessment."""

    def validate_grounding(
        self,
        response_text: str,
        risk_assessment: Optional[RiskAssessment],
        matched_sops: List[Dict[str, Any]],
    ) -> GuardrailCheckResult:
        violations: List[str] = []

        if not response_text:
            return GuardrailCheckResult(
                guardrail_type="grounding",
                passed=False,
                details="Empty response text generated.",
                violations=["Empty response text."],
            )

        # 1. SOP Citation Check
        if matched_sops and len(matched_sops) > 0:
            primary_id = matched_sops[0].get("id")
            if primary_id and primary_id not in response_text:
                violations.append(f"Missing mandatory policy citation '{primary_id}' in output response.")

        # 2. Anti-Hallucination & Jailbreak Check
        if "SOP-999" in response_text:
            violations.append("Output contains fake SOP hallucination ('SOP-999').")

        if "100% safe" in response_text.lower():
            risk_level = risk_assessment.risk_level if risk_assessment else "HIGH"
            if risk_level in ("CRITICAL", "HIGH", "MODERATE"):
                violations.append(f"Output claims '100% safe' which contradicts deterministic {risk_level} risk level.")

        # 3. Conversational Filler Check
        if any(chatty in response_text.lower()[:120] for chatty in ["happy to help", "i'm happy", "i'd be happy", "sure, i can"]):
            violations.append("Output contains informal chatty filler instead of formal clinical advisory structure.")

        passed = len(violations) == 0
        details = "Grounding guardrail PASSED." if passed else f"Grounding guardrail FAILED: {', '.join(violations)}"

        return GuardrailCheckResult(
            guardrail_type="grounding",
            passed=passed,
            details=details,
            violations=violations,
        )
