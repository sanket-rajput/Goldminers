"""
Conversation Intelligence Engine.

Dynamic, non-scripted follow-up logic:
  1. Inspect what data is missing (duration, intensity, age, red flags)
  2. Query the retriever to get candidate diseases
  3. Identify differentiating symptoms between top candidates
  4. Generate ONE high-value follow-up question per turn via LLM
"""

from __future__ import annotations

from app.llm_client import LLMClient
from app.symptom_extractor import ExtractionResult
from app.models import RetrievalResult
from app.utils import setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════
# Question-Generation Prompt Templates
# ═══════════════════════════════════════════════════════════

MISSING_FIELD_PROMPT = """\
You are a caring medical intake assistant.
The patient has reported these symptoms: {symptoms}

The following critical information is still missing: {missing_fields}

Generate ONE short, natural, empathetic follow-up question to gather the most \
important missing piece.
Do NOT ask about multiple things at once.
Do NOT use clinical jargon the patient won't understand.

Return ONLY the question text, nothing else."""

DIFFERENTIAL_PROMPT = """\
You are an experienced clinical reasoning assistant.

Current symptoms reported by the patient: {symptoms}
Duration: {duration}
Severity: {intensity}

Top candidate conditions based on analysis:
1. {disease_1} (confidence: {score_1:.0%})
2. {disease_2} (confidence: {score_2:.0%})
3. {disease_3} (confidence: {score_3:.0%})

Key discriminating symptoms between these conditions that the patient \
has NOT yet mentioned:
{discriminators}

Generate ONE specific, natural-language follow-up question that would best \
help differentiate between these conditions.
The question should be easy for a layperson to understand.
Ask about only ONE thing.

Return ONLY the question text, nothing else."""

SEVERITY_FOLLOWUP_PROMPT = """\
You are a caring medical intake assistant.
The patient has reported: {symptoms}
Current severity assessment: {severity}

Based on their symptoms, generate ONE follow-up question to better \
understand how severe their condition is.
Focus on functional impact (can they work, eat, sleep, walk?), \
pain scale, or worsening pattern.
Keep it conversational and empathetic.

Return ONLY the question text, nothing else."""

DYNAMIC_DEPTH_PROMPT = """\
You are a clinical reasoning assistant deciding if enough patient information \
has been gathered for a preliminary assessment.

Symptoms so far: {symptoms}
Duration known: {duration}
Severity known: {intensity}
Age known: {age}
Turn number: {turn_number}
Missing data: {missing}
Candidate diseases: {candidates}
Last question asked: {last_question}
Patient's last answer: {last_answer}

=== CONVERSATION SO FAR ===
{conversation_history}

DECISION CRITERIA — you should say READY only when:
- At least 2-3 symptoms are reported
- Duration is known (how long they've had symptoms)
- Some indication of severity is available
- You have enough context to make a reasonable differential

If ANY critical piece is still missing that would significantly change \
the assessment, ask ONE more question.

Rules:
- Maximum 5 follow-up questions total
- If turn >= 5, ALWAYS respond READY
- NEVER repeat a question already asked in the conversation above
- Be empathetic and natural

Return EITHER the word "READY" or a single follow-up question. Nothing else."""


class ConversationEngine:
    """Generates adaptive, context-aware follow-up questions."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def should_ask_followup(
        self,
        extraction: ExtractionResult,
        candidates: list[RetrievalResult],
        turn_number: int,
        last_question: str,
        last_answer: str,
        conversation_history: list[dict] | None = None,
    ) -> str | None:
        """
        Decide whether to ask a follow-up or proceed to diagnosis.

        Returns:
            - A follow-up question string, OR
            - None if ready to give assessment
        """
        missing = self._get_missing_fields(extraction)
        symptoms_str = ", ".join(extraction.symptoms) if extraction.symptoms else "none"

        # ── Dynamic depth decision via LLM ───────────────
        candidate_summary = ", ".join(
            f"{r.disease.disease_name}({r.final_score:.2f})"
            for r in candidates[:3]
        ) if candidates else "none found yet"

        # Build compact conversation history so LLM won't repeat questions
        history_lines = []
        if conversation_history:
            for msg in conversation_history[-8:]:
                role = msg.get("role", "?")
                content = msg.get("content", "")[:120]
                history_lines.append(f"  {role}: {content}")
        history_str = "\n".join(history_lines) if history_lines else "No conversation yet."

        try:
            decision = await self.llm.agenerate(
                DYNAMIC_DEPTH_PROMPT.format(
                    symptoms=symptoms_str,
                    duration=extraction.duration or "unknown",
                    intensity=extraction.intensity or "unknown",
                    age=extraction.age or "unknown",
                    turn_number=turn_number,
                    missing=", ".join(missing) if missing else "none",
                    candidates=candidate_summary,
                    last_question=last_question or "none",
                    last_answer=last_answer or "none",
                    conversation_history=history_str,
                ),
                system_prompt="You are a clinical flow controller. Be concise.",
            )
            decision = decision.strip()

            if decision.upper().startswith("READY"):
                logger.info("ConversationEngine → READY for assessment (turn %d)", turn_number)
                return None

            logger.info("ConversationEngine → follow-up: %s", decision[:80])
            return decision

        except Exception as exc:
            logger.warning("Dynamic depth LLM failed: %s", exc)
            # Fallback: use deterministic logic
            return await self._fallback_question(extraction, candidates, missing)

    async def generate_question_for_missing(
        self,
        extraction: ExtractionResult,
        missing_fields: list[str],
    ) -> str:
        """Generate a question targeting missing diagnostic fields."""
        try:
            response = await self.llm.agenerate(
                MISSING_FIELD_PROMPT.format(
                    symptoms=", ".join(extraction.symptoms) or "not yet specified",
                    missing_fields=", ".join(missing_fields),
                ),
                system_prompt="You are a caring medical intake assistant.",
            )
            return response.strip()
        except Exception as exc:
            logger.warning("Missing-field question LLM failed: %s", exc)
            return self._static_question_for_field(missing_fields[0])

    async def generate_differential_question(
        self,
        extraction: ExtractionResult,
        candidates: list[RetrievalResult],
    ) -> str:
        """Generate a question to differentiate between top candidates."""
        if len(candidates) < 2:
            return await self.generate_question_for_missing(
                extraction, self._get_missing_fields(extraction)
            )

        # Find discriminating symptoms
        user_syms = set(s.lower() for s in extraction.symptoms)
        discriminators: list[str] = []
        for r in candidates[:3]:
            for s in r.disease.symptoms[:10]:
                if s.lower() not in user_syms and s not in discriminators:
                    discriminators.append(s)

        top = candidates[:3]
        try:
            response = await self.llm.agenerate(
                DIFFERENTIAL_PROMPT.format(
                    symptoms=", ".join(extraction.symptoms),
                    duration=extraction.duration or "unknown",
                    intensity=extraction.intensity or "unknown",
                    disease_1=top[0].disease.disease_name,
                    score_1=top[0].final_score,
                    disease_2=top[1].disease.disease_name if len(top) > 1 else "N/A",
                    score_2=top[1].final_score if len(top) > 1 else 0,
                    disease_3=top[2].disease.disease_name if len(top) > 2 else "N/A",
                    score_3=top[2].final_score if len(top) > 2 else 0,
                    discriminators=", ".join(discriminators[:10]) or "none identified",
                ),
                system_prompt="You are an experienced clinical reasoning assistant.",
            )
            return response.strip()
        except Exception as exc:
            logger.warning("Differential question LLM failed: %s", exc)
            if discriminators:
                return f"Have you noticed any {discriminators[0]}?"
            return "Can you tell me more about when these symptoms started?"

    async def generate_severity_followup(
        self,
        extraction: ExtractionResult,
        current_severity: str,
    ) -> str:
        """Generate a question to better assess severity dynamically."""
        try:
            response = await self.llm.agenerate(
                SEVERITY_FOLLOWUP_PROMPT.format(
                    symptoms=", ".join(extraction.symptoms) or "unspecified",
                    severity=current_severity or "undetermined",
                ),
                system_prompt="You are a caring, thorough medical intake assistant.",
            )
            return response.strip()
        except Exception as exc:
            logger.warning("Severity follow-up LLM failed: %s", exc)
            return "On a scale of 1-10, how would you rate the discomfort?"

    # ─── Internal helpers ────────────────────────────────

    def _get_missing_fields(self, extraction: ExtractionResult) -> list[str]:
        missing: list[str] = []
        if not extraction.symptoms:
            missing.append("symptoms")
        if not extraction.duration:
            missing.append("duration")
        if not extraction.intensity:
            missing.append("intensity/severity")
        if not extraction.age:
            missing.append("age")
        return missing

    async def _fallback_question(
        self,
        extraction: ExtractionResult,
        candidates: list[RetrievalResult],
        missing: list[str],
    ) -> str | None:
        """Deterministic fallback when LLM is unavailable."""
        if missing:
            return self._static_question_for_field(missing[0])
        if candidates and len(candidates) >= 2:
            return await self.generate_differential_question(extraction, candidates)
        return None

    @staticmethod
    def _static_question_for_field(field: str) -> str:
        """Last-resort static questions for missing fields."""
        mapping = {
            "symptoms": "Could you describe what you're experiencing in a bit more detail?",
            "duration": "How long have you been experiencing these symptoms?",
            "intensity": "How would you describe the severity — mild, moderate, or severe?",
            "intensity/severity": "How would you describe the severity — mild, moderate, or severe?",
            "age": "Could you share your age? It helps me give you better guidance.",
        }
        return mapping.get(field, "Is there anything else about your symptoms you'd like to share?")
