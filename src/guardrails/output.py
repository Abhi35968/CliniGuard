from typing import Dict, Any, List, Optional, Tuple
from src.schemas.risk import RiskAssessment
from src.schemas.trace import GuardrailCheckResult
from src.guardrails.grounding import GroundingGuardrail
from src.guardrails.numeric import NumericConsistencyGuardrail


class OutputGuardrailPipeline:
    """Composite Output Guardrail Pipeline enforcing grounding, numeric consistency, and policy alignment."""

    def __init__(self):
        self.grounding_guardrail = GroundingGuardrail()
        self.numeric_guardrail = NumericConsistencyGuardrail()

    def validate_output(
        self,
        response_text: str,
        risk_assessment: Optional[RiskAssessment],
        matched_sops: List[Dict[str, Any]],
        telemetry: Dict[str, Any],
    ) -> Tuple[bool, List[GuardrailCheckResult]]:
        """
        Executes output guardrails.
        Returns (overall_passed_bool, list_of_check_results).
        """
        results: List[GuardrailCheckResult] = []

        # 1. Grounding Validation
        res_grounding = self.grounding_guardrail.validate_grounding(
            response_text=response_text,
            risk_assessment=risk_assessment,
            matched_sops=matched_sops,
        )
        results.append(res_grounding)

        # 2. Numeric Consistency Validation
        res_numeric = self.numeric_guardrail.validate_numeric_consistency(
            response_text=response_text,
            telemetry=telemetry,
        )
        results.append(res_numeric)

        overall_passed = all(r.passed for r in results)
        return overall_passed, results
