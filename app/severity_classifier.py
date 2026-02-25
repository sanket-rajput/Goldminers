"""
Hybrid Severity Classifier.

Two-layer approach:
  Layer 1: Rule-based red-flag escalation (instant, deterministic)
  Layer 2: LLM severity validation (contextual, dynamic)

Severity is NOT static — it evolves as the conversation progresses
and new information (answers to follow-up questions) arrives.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from app.llm_client import LLMClient
from app.symptom_extractor import ExtractionResult
from app.models import RetrievalResult
from app.utils import setup_logger, normalize_symptom

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════
# Red-Flag Rules (immediate escalation)
# ═══════════════════════════════════════════════════════════

RED_FLAG_RULES: list[dict] = [
    {
        "patterns": ["chest pain", "chest tightness", "crushing chest"],
        "severity": "severe",
        "reason": "Possible cardiac emergency",
        "action": "Seek immediate medical attention or call emergency services",
    },
    {
        "patterns": [
            "difficulty breathing", "shortness of breath", "cant breathe",
            "can't breathe", "severe breathing difficulty", "breathlessness",
        ],
        "severity": "severe",
        "reason": "Respiratory distress — may indicate pneumonia, asthma attack, or PE",
        "action": "Seek immediate medical care",
    },
    {
        "patterns": ["stiff neck", "neck stiffness", "nuchal rigidity"],
        "severity": "severe",
        "reason": "Possible meningitis when combined with fever",
        "action": "Go to emergency room immediately",
    },
    {
        "patterns": [
            "blood in stool", "melena", "rectal bleeding",
            "blood in vomit", "hematemesis",
        ],
        "severity": "severe",
        "reason": "Gastrointestinal bleeding — needs urgent evaluation",
        "action": "See a doctor today or visit emergency room",
    },
    {
        "patterns": ["coughing blood", "hemoptysis", "blood in sputum"],
        "severity": "severe",
        "reason": "Pulmonary hemorrhage — needs urgent workup",
        "action": "See a doctor immediately",
    },
    {
        "patterns": [
            "confusion", "altered mental status", "disorientation",
            "loss of consciousness", "fainting", "syncope",
        ],
        "severity": "severe",
        "reason": "Neurological emergency — possible stroke, infection, or metabolic crisis",
        "action": "Call emergency services immediately",
    },
    {
        "patterns": [
            "slurred speech", "facial drooping", "sudden weakness one side",
            "sudden numbness", "hemiplegia",
        ],
        "severity": "severe",
        "reason": "Possible stroke — time-critical",
        "action": "Call emergency services NOW. Remember FAST: Face, Arms, Speech, Time",
    },
    {
        "patterns": ["seizure", "convulsion", "fitting"],
        "severity": "severe",
        "reason": "Seizure disorder or neurological emergency",
        "action": "Call emergency services if first-time or prolonged seizure",
    },
    {
        "patterns": ["suicidal", "want to die", "kill myself", "suicidal thoughts"],
        "severity": "severe",
        "reason": "Mental health crisis",
        "action": "Contact crisis helpline immediately. You are not alone.",
    },
    {
        "patterns": ["anaphylaxis", "throat swelling", "can't swallow"],
        "severity": "severe",
        "reason": "Possible anaphylactic reaction",
        "action": "Use epinephrine if available, call emergency services",
    },
    {
        "patterns": ["severe abdominal pain", "worst pain ever"],
        "severity": "severe",
        "reason": "May indicate appendicitis, perforation, or ectopic pregnancy",
        "action": "Visit emergency room immediately",
    },
    {
        "patterns": ["persistent vomiting", "vomiting blood", "can't keep anything down"],
        "severity": "severe",
        "reason": "Risk of severe dehydration or internal bleeding",
        "action": "Seek medical attention today",
    },
]

# Temperature-based escalation
TEMP_SEVERE_PATTERNS = [
    re.compile(r"(?:fever|temperature)\s*(?:of|around|about|is)?\s*(10[3-9]|1[1-9]\d)", re.IGNORECASE),
    re.compile(r"(10[3-9]|1[1-9]\d)\s*(?:°?\s*f|degrees?\s*f)", re.IGNORECASE),
    re.compile(r"(39\.[5-9]|4[0-2]\.\d)\s*(?:°?\s*c|degrees?\s*c)", re.IGNORECASE),
    re.compile(r"\bhigh fever\b", re.IGNORECASE),
]

# Severity escalation from answer context
ANSWER_ESCALATION_PATTERNS = {
    "severe": [
        re.compile(r"\b(10|9|8)\s*/?\s*10\b"),  # pain scale 8-10/10
        re.compile(r"\b(can'?t|cannot|unable)\s+(walk|eat|sleep|work|move|stand|breathe)\b", re.IGNORECASE),
        re.compile(r"\b(getting\s+worse|worsening|deteriorating|rapidly)\b", re.IGNORECASE),
        re.compile(r"\b(emergency|unbearable|excruciating|worst)\b", re.IGNORECASE),
        re.compile(r"\b(multiple\s+days|won'?t\s+stop)\b", re.IGNORECASE),
    ],
    "moderate": [
        re.compile(r"\b(5|6|7)\s*/?\s*10\b"),  # pain scale 5-7/10
        re.compile(r"\b(moderate|significant|quite|fairly)\b", re.IGNORECASE),
        re.compile(r"\b(off\s+and\s+on|comes\s+and\s+goes|intermittent)\b", re.IGNORECASE),
        re.compile(r"\b(few\s+days|couple\s+of\s+days|several\s+days)\b", re.IGNORECASE),
        re.compile(r"\b(affecting|interfering|disrupting)\b", re.IGNORECASE),
    ],
    "mild": [
        re.compile(r"\b(1|2|3|4)\s*/?\s*10\b"),  # pain scale 1-4/10
        re.compile(r"\b(mild|slight|little|minor|manageable)\b", re.IGNORECASE),
        re.compile(r"\b(just\s+started|started\s+today|this\s+morning)\b", re.IGNORECASE),
    ],
}


# ═══════════════════════════════════════════════════════════
# Classification result
# ═══════════════════════════════════════════════════════════

class SeverityResult:
    """Structured severity assessment."""

    def __init__(
        self,
        level: str = "undetermined",
        confidence: float = 0.0,
        reasoning: str = "",
        red_flags_found: list[str] | None = None,
        recommended_action: str = "",
        is_emergency: bool = False,
    ) -> None:
        self.level = level                       # mild | moderate | severe | undetermined
        self.confidence = confidence             # 0.0–1.0
        self.reasoning = reasoning
        self.red_flags_found = red_flags_found or []
        self.recommended_action = recommended_action
        self.is_emergency = is_emergency

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "red_flags_found": self.red_flags_found,
            "recommended_action": self.recommended_action,
            "is_emergency": self.is_emergency,
        }

    def escalate(self, new_level: str, reason: str) -> None:
        """Only escalate upward (mild→moderate→severe), never downward."""
        hierarchy = {"undetermined": 0, "mild": 1, "moderate": 2, "severe": 3}
        if hierarchy.get(new_level, 0) > hierarchy.get(self.level, 0):
            self.level = new_level
            self.reasoning += f" | Escalated: {reason}"
            if new_level == "severe":
                self.is_emergency = True


# ═══════════════════════════════════════════════════════════
# LLM Severity Prompt
# ═══════════════════════════════════════════════════════════

LLM_SEVERITY_PROMPT = """\
You are a clinical severity assessment assistant.

Patient symptoms: {symptoms}
Duration: {duration}
Self-reported intensity: {intensity}
Age: {age}
Gender: {gender}

Top candidate conditions:
{candidates}

Rule-based assessment: {rule_severity}
Red flags detected: {red_flags}

Classify the overall severity as exactly one of: mild, moderate, severe.
Provide a SHORT (1-2 sentence) clinical reasoning.
Do NOT invent symptoms the patient hasn't reported.
Do NOT claim certainty.

Return ONLY valid JSON:
{{
  "severity": "mild|moderate|severe",
  "reasoning": "short reasoning",
  "recommended_action": "what the patient should do"
}}"""


# ═══════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════

class SeverityClassifier:
    """
    Hybrid severity classification with dynamic escalation.

    Severity updates every turn as the patient provides more answers.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    # ─── Full assessment ─────────────────────────────────

    async def classify(
        self,
        extraction: ExtractionResult,
        candidates: list[RetrievalResult],
        conversation_history: list[dict] | None = None,
        previous_severity: SeverityResult | None = None,
    ) -> SeverityResult:
        """
        Dynamic severity classification that:
        1. Runs rule-based checks
        2. Analyzes patient answers for escalation signals
        3. Validates with LLM
        4. Never downgrades from a previous assessment
        """
        # Start from previous assessment or fresh
        result = previous_severity or SeverityResult()

        # Layer 1: Rule-based red flag check
        self._apply_rules(extraction, result)

        # Layer 1b: Dynamic escalation from conversation answers
        if conversation_history:
            self._analyze_answers(conversation_history, result)

        # Layer 2: LLM validation
        await self._llm_validate(extraction, candidates, result)

        # Ensure we never downgrade
        if previous_severity:
            hierarchy = {"undetermined": 0, "mild": 1, "moderate": 2, "severe": 3}
            if hierarchy.get(result.level, 0) < hierarchy.get(previous_severity.level, 0):
                result.level = previous_severity.level
                result.is_emergency = previous_severity.is_emergency

        logger.info(
            "Severity: %s (confidence %.2f, emergency=%s) — %s",
            result.level,
            result.confidence,
            result.is_emergency,
            result.reasoning[:80],
        )
        return result

    # ─── Layer 1: Rules ──────────────────────────────────

    def _apply_rules(
        self, extraction: ExtractionResult, result: SeverityResult
    ) -> None:
        """Deterministic red-flag and pattern checks."""
        combined_text = " ".join(extraction.symptoms).lower()

        # Check red flag patterns
        for rule in RED_FLAG_RULES:
            for pattern in rule["patterns"]:
                if pattern in combined_text:
                    result.red_flags_found.append(pattern)
                    result.escalate("severe", rule["reason"])
                    result.recommended_action = rule["action"]

        # Temperature check
        for pat in TEMP_SEVERE_PATTERNS:
            if pat.search(combined_text):
                result.escalate("severe", "High fever detected (>103°F / 39.5°C)")
                result.recommended_action = (
                    result.recommended_action or "Seek medical attention for high fever"
                )

        # Intensity from extraction
        if extraction.intensity == "severe":
            result.escalate("severe", "Self-reported severe symptoms")
        elif extraction.intensity == "moderate":
            result.escalate("moderate", "Self-reported moderate symptoms")
        elif extraction.intensity == "mild" and result.level == "undetermined":
            result.level = "mild"
            result.reasoning = "Self-reported mild symptoms"

        # Multiple symptoms may indicate higher severity
        if len(extraction.symptoms) >= 5 and result.level in ("undetermined", "mild"):
            result.escalate("moderate", f"Multiple symptoms reported ({len(extraction.symptoms)})")

    # ─── Layer 1b: Dynamic answer analysis ───────────────

    def _analyze_answers(
        self, history: list[dict], result: SeverityResult
    ) -> None:
        """
        Inspect patient's answers to follow-up questions for severity signals.
        This makes severity DYNAMIC — it evolves with each answer.
        """
        # Get only user messages
        user_messages = [
            m["content"] for m in history
            if m.get("role") == "user" and m.get("content")
        ]

        combined_answers = " ".join(user_messages[-5:])  # last 5 answers

        for level, patterns in ANSWER_ESCALATION_PATTERNS.items():
            for pat in patterns:
                match = pat.search(combined_answers)
                if match:
                    result.escalate(level, f"Answer analysis: '{match.group(0)}'")

        logger.debug(
            "Answer analysis on %d messages → %s", len(user_messages), result.level
        )

    # ─── Layer 2: LLM Validation ─────────────────────────

    async def _llm_validate(
        self,
        extraction: ExtractionResult,
        candidates: list[RetrievalResult],
        result: SeverityResult,
    ) -> None:
        """LLM-based severity refinement."""
        candidate_text = "\n".join(
            f"  - {r.disease.disease_name} (score: {r.final_score:.2f}, "
            f"symptoms: {', '.join(r.disease.symptoms[:5])})"
            for r in candidates[:3]
        ) or "  None matched yet"

        try:
            response = await self.llm.agenerate(
                LLM_SEVERITY_PROMPT.format(
                    symptoms=", ".join(extraction.symptoms) or "not specified",
                    duration=extraction.duration or "not specified",
                    intensity=extraction.intensity or "not specified",
                    age=extraction.age or "not specified",
                    gender=extraction.gender or "not specified",
                    candidates=candidate_text,
                    rule_severity=result.level,
                    red_flags=", ".join(result.red_flags_found) or "none",
                ),
                system_prompt="You are a clinical severity assistant. Return ONLY JSON.",
            )
            parsed = self._parse(response)
            llm_level = parsed.get("severity", "").lower()
            llm_reason = parsed.get("reasoning", "")
            llm_action = parsed.get("recommended_action", "")

            # LLM can only escalate, not downgrade
            if llm_level in ("mild", "moderate", "severe"):
                result.escalate(llm_level, f"LLM: {llm_reason}")
                result.confidence = max(result.confidence, 0.7)
                if llm_action:
                    result.recommended_action = llm_action

            # If still undetermined after LLM
            if result.level == "undetermined":
                result.level = llm_level or "mild"
                result.reasoning = llm_reason or "Insufficient data for confident assessment"
                result.confidence = 0.4

        except Exception as exc:
            logger.warning("LLM severity validation failed: %s", exc)
            if result.level == "undetermined":
                result.level = "mild"
                result.confidence = 0.3
                result.reasoning = "Default mild — LLM validation unavailable"

    # ─── Helpers ─────────────────────────────────────────

    @staticmethod
    def _parse(response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text.strip())
