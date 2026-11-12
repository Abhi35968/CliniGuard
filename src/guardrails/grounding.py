import re
from typing import Dict, Any, List, Optional
from src.schemas.risk import RiskAssessment
from src.schemas.trace import GuardrailCheckResult


class GroundingGuardrail:
    """
    Verifies that generated advisory text is strictly grounded in matched SOPs and RiskAssessment.
    Enforces deterministic safety invariants:
      1. Mandatory governing SOP citation.
      2. No fabricated/hallucinated policy IDs (e.g. 'SOP-999').
      3. No risk downgrades (e.g. claiming LOW/MODERATE or APPROVED under CRITICAL/HIGH risk).
      4. No unnegated permissive claims (e.g. 'safe to proceed', 'completely safe') under CRITICAL/HIGH risk.
      5. No recommendations violating explicit policy prohibitions.
      6. Mandatory hazard/precaution warnings present under CRITICAL and HIGH severity.
      7. Formal clinical advisory structure without casual conversational filler.
    """

    # Unnegated permissive phrases that contradict a HIGH or CRITICAL risk assessment
    PERMISSIVE_PATTERNS = [
        r"\b(?:it\s+is|it\'s|is|are|conditions\s+are)\s+(?:completely\s+|perfectly\s+|totally\s+|entirely\s+|100%\s+)?safe\s+to\s+\w+\b",
        r"\bsafe\s+to\s+(?:proceed|continue|ride|cycle|jog|run|walk|hike|drive|commute|travel|go|conduct|exercise|plan)\b",
        r"\b(?:completely|perfectly|totally|entirely|100%)\s+safe\b",
        r"\b(?:it\s+is|it\'s)\s+safe\b",
        r"\b(?:no|zero)\s+(?:risk|danger|hazard)\b(?!\s*[-–]\s*free)",
        r"\b(?:ideal|perfect)\s+conditions\s+for\b",
    ]

    # Preceding negation words in the same clause indicating the activity is NOT safe
    NEGATION_PATTERN = re.compile(
        r"\b(?:not|no|never|neither|nor|cannot|can\'t|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|won\'t|hardly|unlikely|unsafe|prohibited|prohibits)\b",
        re.IGNORECASE,
    )

    # Permissive recommendation verbs that contradict policy prohibitions
    RECOMMENDATION_PATTERN = re.compile(
        r"\b(?:recommend|recommended|suggest|suggested|encourage|encouraged|feel\s+free\s+to|advisable\s+to|safe\s+to|you\s+can|okay\s+to|permitted\s+to|allowed\s+to|good\s+to)\s+(?:to\s+)?(?:use|using|take|taking|visit|visiting|ride|riding|cycle|cycling|drive|driving|proceed\s+with|enjoy|enjoying|transit\s+on|travel\s+on|cross|crossing)\s+([^.!?\n]+)",
        re.IGNORECASE,
    )

    def validate_grounding(
        self,
        response_text: str,
        risk_assessment: Optional[RiskAssessment] = None,
        matched_sops: Optional[List[Dict[str, Any]]] = None,
        active_sop: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GuardrailCheckResult:
        violations: List[str] = []

        if matched_sops is None:
            matched_sops = [active_sop] if active_sop else []
        elif active_sop and active_sop not in matched_sops:
            matched_sops = [active_sop] + matched_sops

        if not response_text or not response_text.strip():
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

        # 2. Anti-Hallucination & Fake / Unauthorized SOP Check
        allowed_sop_ids = set()
        if matched_sops:
            allowed_sop_ids.update(s.get("id") for s in matched_sops if isinstance(s, dict) and s.get("id"))
        if risk_assessment:
            if risk_assessment.primary_sop_id:
                allowed_sop_ids.add(risk_assessment.primary_sop_id)
            if risk_assessment.triggered_policies:
                allowed_sop_ids.update(risk_assessment.triggered_policies)

        if "SOP-999" in response_text:
            violations.append("Output contains fake SOP hallucination ('SOP-999').")

        cited_sop_ids = set(re.findall(r"\bSOP-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", response_text))
        for cited in cited_sop_ids:
            if allowed_sop_ids and cited not in allowed_sop_ids:
                violations.append(
                    f"Advisory cites unauthorized or invented policy identifier '{cited}' not in matched policies."
                )

        risk_level = (risk_assessment.risk_level if risk_assessment else "INFORMATIONAL").upper()

        # 3. Severity Level Alignment & Contradiction Detection for Severe/Elevated Risks
        if risk_level in ("CRITICAL", "HIGH"):
            # 3A: Risk Downgrade Detection
            downgrade_pattern = re.compile(
                r"\b(?:severity\s*level|risk\s*level|risk\s*rating|severity)\s*[:=]\s*[`'\"]?(?:LOW|MODERATE|INFORMATIONAL|NONE|MINIMAL)\b",
                re.IGNORECASE,
            )
            if downgrade_pattern.search(response_text):
                violations.append(
                    f"Output claims a downgraded risk status that contradicts the deterministic {risk_level} risk assessment."
                )

            if re.search(r"STATUS:\s*APPROVED\b", response_text, re.IGNORECASE):
                violations.append(
                    f"Output claims 'STATUS: APPROVED' which contradicts deterministic {risk_level} risk level."
                )

            # 3B: Unnegated Permissive Language Contradiction
            unnegated_claim = self._find_unnegated_permissive_claim(response_text)
            if unnegated_claim:
                violations.append(
                    f"Output contains permissive claim ('{unnegated_claim}') contradicting deterministic {risk_level} risk assessment."
                )

            # 3C: Prohibited Action Violation Detection
            if risk_assessment and risk_assessment.prohibited_actions:
                violations.extend(
                    self._check_prohibited_recommendations(
                        response_text=response_text,
                        prohibited_actions=risk_assessment.prohibited_actions,
                    )
                )

            # 3D: Missing Mandatory Warning / Restriction Check
            if risk_level == "CRITICAL":
                critical_warning_pattern = re.compile(
                    r"\b(?:CRITICAL|HAZARD|PROHIBITED|DANGER|POSTPONE|AVOID|RESTRICT|SEVERE|WARNING|ALERT)\b",
                    re.IGNORECASE,
                )
                if not critical_warning_pattern.search(response_text):
                    violations.append(
                        "Output omits mandatory warning or restriction required for CRITICAL severity."
                    )
            elif risk_level == "HIGH":
                high_warning_pattern = re.compile(
                    r"\b(?:HIGH\s*RISK|HIGH\s*SEVERITY|CAUTION|POSTPONE|AVOID|RESTRICT|WARNING|ALERT|HAZARD|UNSAFE)\b",
                    re.IGNORECASE,
                )
                if not high_warning_pattern.search(response_text):
                    violations.append(
                        "Output omits mandatory warning or restriction required for HIGH severity."
                    )

        # 4. Moderate Risk 100% Safe Check
        elif risk_level == "MODERATE":
            if "100% safe" in response_text.lower() or "completely safe" in response_text.lower():
                violations.append("Output claims absolute safety which contradicts deterministic MODERATE risk level.")

        # 5. Conversational Filler Check
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

    def _find_unnegated_permissive_claim(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for pattern in self.PERMISSIVE_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                start = match.start()
                # Inspect lookback window up to 45 chars in the same sentence/clause
                clause_start = max(0, start - 45)
                for punct in [".", ";", "!", "?", "\n"]:
                    last_p = text_lower.rfind(punct, clause_start, start)
                    if last_p != -1:
                        clause_start = max(clause_start, last_p + 1)
                window = text_lower[clause_start:start]
                if not self.NEGATION_PATTERN.search(window):
                    return match.group().strip()
        return None

    def _check_prohibited_recommendations(
        self, response_text: str, prohibited_actions: List[str]
    ) -> List[str]:
        violations = []
        text_lower = response_text.lower()

        forbidden_terms = [
            "overpass", "overpasses", "bridge", "bridges", "coastal road", "coastal roads", "causeway", "causeways",
            "two-wheeler", "two-wheelers", "scooter", "motorcycle", "bike", "cycling",
            "flood water", "flood waters", "flooded", "open-field", "trees",
            "outdoor activity", "outdoor activities", "outdoor exercise", "outdoor walk", "outdoor gathering",
            "non-essential cycling", "non-essential travel", "non-essential outdoor",
        ]

        for match in self.RECOMMENDATION_PATTERN.finditer(text_lower):
            recommended_clause = match.group(0)
            for p in prohibited_actions:
                p_lower = p.lower()
                for term in forbidden_terms:
                    if term in p_lower and term in recommended_clause:
                        violations.append(
                            f"Output recommends prohibited action: '{recommended_clause.strip()}' "
                            f"which violates policy restriction: '{p}'."
                        )
        return violations
