"""
Diagnosis Engine — the central orchestrator.

Two-phase conversation flow:

  PHASE 1 — GATHERING (conversational):
    The bot has a warm, natural conversation with the patient.
    It asks follow-up questions one at a time to understand:
      • What symptoms they have
      • How long they've had them
      • How severe they feel
      • Age / gender
      • Any red-flag signals
    During this phase the bot NEVER shows diagnoses, severity
    charts, medication lists, or remedy lists. It just talks
    like a caring doctor taking an intake.

  PHASE 2 — DIAGNOSIS (structured):
    Only when enough information has been gathered does the bot
    deliver a full structured assessment:
      • Top-3 possible conditions
      • Severity assessment
      • Safe OTC medication suggestions
      • Home remedies
      • Red-flag warnings
      • Important reminder / disclaimer
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from app.llm_client import LLMClient
from app.retriever import HybridRetriever
from app.symptom_extractor import SymptomExtractor, ExtractionResult
from app.conversation_engine import ConversationEngine
from app.severity_classifier import SeverityClassifier, SeverityResult
from app.medicine_policy import MedicinePolicy, DISCLAIMER
from app.remedy_loader import RemedyLoader
from app.firebase_manager import FirebaseManager
from app.models import SessionData, RetrievalResult
from app.utils import setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════
# PHASE 1 — Gathering prompt (conversational, NO diagnosis)
# ═══════════════════════════════════════════════════════════

GATHERING_SYSTEM_PROMPT = """\
You are BluCare, a warm and empathetic medical intake assistant.
Your job RIGHT NOW is to GATHER information — NOT to diagnose.

STRICT RULES:
1. NEVER mention possible diseases, conditions, or diagnoses yet.
2. NEVER show severity assessments, medication suggestions, or remedies.
3. NEVER give a structured diagnostic report.
4. DO acknowledge what the patient said warmly and naturally.
5. DO ask exactly ONE clear follow-up question directly related to their symptoms.
6. Keep your response SHORT — 2-3 sentences max.
7. Sound like a caring human doctor, NOT a robot filling a form.
8. Your follow-up question MUST be specific to the symptoms they described.
   For example: if they say "headache" → ask about location, type of pain, triggers.
   If they say "stomach pain" → ask about when it happens, relation to food, etc.
   NEVER ask generic questions like "can you describe your symptoms" when they already told you."""

GATHERING_PROMPT_TEMPLATE = """\
{system_prompt}

=== WHAT WE KNOW SO FAR ===
Symptoms: {symptoms}
Duration: {duration}
Severity: {intensity}
Age: {age}  |  Gender: {gender}
Turn: {turn_number}

=== STILL MISSING (prioritize asking about these) ===
{missing_info}

=== CONVERSATION HISTORY ===
{history}

=== PATIENT'S LATEST MESSAGE ===
{user_message}

=== INSTRUCTIONS ===
Write a SHORT response (2-3 sentences):
1. Briefly acknowledge what the patient just said.
2. Ask ONE specific follow-up question about their symptoms.

Your question MUST be directly relevant to the symptoms they mentioned.
Good examples:
- "headache" → "Is the pain on one side or both? Does light or noise make it worse?"
- "stomach pain" → "Does the pain come after eating, or is it constant?"
- "cough" → "Is it a dry cough or are you coughing up anything?"

If we're missing duration/severity/age from the list above, weave that into
your symptom-specific question naturally (e.g. "How long has this throbbing
headache been going on?").

Do NOT repeat a question already asked in the conversation history.
Do NOT list diagnoses, medications, or remedies.
Keep it short and human."""


# ═══════════════════════════════════════════════════════════
# PHASE 2 — Full Diagnosis prompt (structured output)
# ═══════════════════════════════════════════════════════════

DIAGNOSIS_SYSTEM_PROMPT = """\
You are BluCare, an empathetic, knowledgeable medical assistant.
You help patients understand their symptoms and guide them toward appropriate care.

RULES — you MUST follow:
1. NEVER claim a confirmed diagnosis. Always say "Based on the information provided…"
2. If severity is SEVERE → strongly recommend hospital / emergency care FIRST.
3. Use warm, supportive language. The patient may be scared.
4. Present information in a clear, structured way.
5. Include the mandatory disclaimer about seeking professional medical advice.
6. Avoid medical jargon unless necessary; explain terms simply."""

DIAGNOSIS_PROMPT_TEMPLATE = """\
{system_prompt}

=== PATIENT INFORMATION ===
Symptoms: {symptoms}
Duration: {duration}
Severity (self-reported): {intensity}
Age: {age}  |  Gender: {gender}

=== RAG RESULTS (Top-3 conditions) ===
{disease_context}

=== SEVERITY ASSESSMENT ===
Level: {severity_level}
Reasoning: {severity_reasoning}
Red flags: {red_flags}
Recommended action: {recommended_action}

=== SAFE MEDICATION SUGGESTIONS ===
{medication_text}

=== HOME REMEDIES ===
{remedy_text}

=== CONVERSATION HISTORY ===
{history}

=== CURRENT USER MESSAGE ===
{user_message}

=== INSTRUCTIONS ===
Now that you have gathered enough information, provide a COMPLETE structured assessment.
Use markdown formatting with these sections:

1. **Possible Conditions** — Top 3 conditions ranked by likelihood, with a brief
   explanation for each. Start with "Based on the information you've shared…"
   NEVER present as a confirmed diagnosis.

2. **Severity Assessment** — How serious this appears and why.

3. **Safe Suggestions** — OTC medications from the list above ONLY.
   Include dosage info. Do not invent medications not in the list.

4. **Home Remedies** — From the list above only. Practical and easy to follow.

5. **Red Flag Warning** — If any red flags detected, emphasize urgency.
   If severity is SEVERE, put this section FIRST before everything else.

6. **⚕️ Important Reminder** — Always end with a reminder to consult a
   qualified healthcare professional. This is not a substitute for medical advice.

Be thorough but clear. The patient has been waiting through several questions —
now give them a comprehensive, helpful answer."""


# ═══════════════════════════════════════════════════════════
# Turn State — serializable session-level diagnostic state
# ═══════════════════════════════════════════════════════════

class DiagnosticState:
    """Tracks per-session diagnostic reasoning state."""

    def __init__(self) -> None:
        self.turn_number: int = 0
        self.extraction: ExtractionResult = ExtractionResult()
        self.severity: SeverityResult | None = None
        self.candidates: list[RetrievalResult] = []
        self.last_question: str = ""

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "extraction": self.extraction.to_dict(),
            "severity": self.severity.to_dict() if self.severity else None,
            "last_question": self.last_question,
        }

    @classmethod
    def from_session(cls, session: SessionData) -> DiagnosticState:
        """Reconstruct diagnostic state from session data."""
        state = cls()
        state.turn_number = len([
            m for m in session.previous_messages if m.get("role") == "user"
        ])
        state.extraction = ExtractionResult()
        state.extraction.symptoms = list(session.extracted_symptoms)

        # Reconstruct severity from session metadata
        meta = session.disease_candidates
        if meta and isinstance(meta, list) and len(meta) > 0:
            last = meta[-1] if isinstance(meta[-1], dict) else {}
            sev = last.get("_severity")
            if sev and isinstance(sev, dict):
                state.severity = SeverityResult(
                    level=sev.get("level", "undetermined"),
                    confidence=sev.get("confidence", 0),
                    reasoning=sev.get("reasoning", ""),
                    red_flags_found=sev.get("red_flags_found", []),
                    recommended_action=sev.get("recommended_action", ""),
                    is_emergency=sev.get("is_emergency", False),
                )
        state.last_question = (
            session.previous_messages[-1].get("content", "")
            if session.previous_messages
            and session.previous_messages[-1].get("role") == "assistant"
            else ""
        )
        return state


# ═══════════════════════════════════════════════════════════
# Diagnosis Engine
# ═══════════════════════════════════════════════════════════

class DiagnosisEngine:
    """
    Central orchestrator with two-phase conversation flow.

    Phase 1 (Gathering): Warm conversation, ask questions, collect info.
    Phase 2 (Diagnosis):  Full structured assessment once confident.
    """

    # Minimum requirements before we give a diagnosis
    MIN_SYMPTOMS_FOR_DIAGNOSIS = 2  # at least 2 symptoms
    MIN_TURNS_FOR_DIAGNOSIS = 3     # at least 3 user turns
    MAX_GATHERING_TURNS = 5         # force diagnosis after this many turns

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: HybridRetriever,
        firebase_mgr: FirebaseManager,
    ) -> None:
        self.llm = llm_client
        self.retriever = retriever
        self.firebase = firebase_mgr

        self.extractor = SymptomExtractor(llm_client)
        self.conversation = ConversationEngine(llm_client)
        self.severity_clf = SeverityClassifier(llm_client)
        self.medicine = MedicinePolicy()
        self.remedies = RemedyLoader()
        self.remedies.load()

    async def process_turn(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> AsyncGenerator[str, None]:
        """
        Execute a diagnostic turn with streaming output.

        Phase 1: If still gathering → warm conversational reply + follow-up.
        Phase 2: If ready → full structured diagnosis.
        """
        # ── Load session ────────────────────────────────────
        session = self.firebase.get_session(user_id, session_id)
        if not session:
            yield self._sse({"error": "Session not found"})
            return

        # Save user message
        self.firebase.add_message(user_id, session_id, "user", message)
        session = self.firebase.get_session(user_id, session_id)  # reload

        # ── Reconstruct state ───────────────────────────────
        state = DiagnosticState.from_session(session)
        state.turn_number += 1

        logger.info(
            "=== Turn %d | user=%s session=%s ===",
            state.turn_number, user_id[:12], session_id[:8],
        )

        # ── Step 1: Extract symptoms ────────────────────────
        extraction = await self.extractor.extract(
            message, previous_symptoms=session.extracted_symptoms
        )
        state.extraction = extraction

        # Persist symptoms
        self.firebase.update_symptoms(user_id, session_id, extraction.symptoms)

        # ── Step 2: Decide phase — GATHERING or DIAGNOSIS ───
        is_ready = await self._is_ready_for_diagnosis(
            extraction=extraction,
            state=state,
            message=message,
            session=session,
        )

        if is_ready:
            logger.info("📋 PHASE 2 → Delivering full diagnosis (turn %d)", state.turn_number)
            async for chunk in self._run_diagnosis_phase(
                user_id, session_id, message, extraction, state, session
            ):
                yield chunk
        else:
            logger.info("💬 PHASE 1 → Gathering more info (turn %d)", state.turn_number)
            async for chunk in self._run_gathering_phase(
                user_id, session_id, message, extraction, state, session
            ):
                yield chunk

    # ═══════════════════════════════════════════════════════
    # Readiness check
    # ═══════════════════════════════════════════════════════

    async def _is_ready_for_diagnosis(
        self,
        extraction: ExtractionResult,
        state: DiagnosticState,
        message: str,
        session: SessionData = None,
    ) -> bool:
        """
        Determine if we have enough info to deliver a diagnosis.

        Hard rules:
        - Need at least MIN_SYMPTOMS_FOR_DIAGNOSIS symptoms
        - Need at least MIN_TURNS_FOR_DIAGNOSIS turns
        - OR severity is SEVERE (emergency — skip gathering)

        Then ask the ConversationEngine (LLM) for confirmation.
        """
        sym_count = len(extraction.symptoms)
        turn = state.turn_number

        # Emergency override — if we detect severe red flags, diagnose immediately
        if state.severity and state.severity.is_emergency:
            logger.info("🚨 Emergency detected — skipping to diagnosis")
            return True

        # Not enough data yet — stay in gathering
        if sym_count < self.MIN_SYMPTOMS_FOR_DIAGNOSIS:
            logger.info(
                "Not enough symptoms (%d < %d) — continue gathering",
                sym_count, self.MIN_SYMPTOMS_FOR_DIAGNOSIS,
            )
            return False

        if turn < self.MIN_TURNS_FOR_DIAGNOSIS:
            logger.info(
                "Too few turns (%d < %d) — continue gathering",
                turn, self.MIN_TURNS_FOR_DIAGNOSIS,
            )
            return False

        # Hard cap — never gather forever
        if turn >= self.MAX_GATHERING_TURNS:
            logger.info("⏰ Max gathering turns (%d) — forcing diagnosis", turn)
            return True

        # We have enough basic data — ask the ConversationEngine
        # to decide if we need more or should proceed
        candidates = []
        if extraction.symptoms:
            symptom_query = ", ".join(extraction.symptoms)
            candidates = self.retriever.retrieve(
                query=symptom_query,
                user_symptoms=extraction.symptoms,
                top_k_vector=10,
                top_k_final=3,
            )

        followup = await self.conversation.should_ask_followup(
            extraction=extraction,
            candidates=candidates,
            turn_number=turn,
            last_question=state.last_question,
            last_answer=message,
            conversation_history=session.previous_messages if session else None,
        )

        # If ConversationEngine returns None → READY
        # If it returns a question → not ready yet (but store it for later)
        if followup is None:
            return True

        # Store the follow-up for the gathering step
        state._pending_followup = followup
        return False

    # ═══════════════════════════════════════════════════════
    # PHASE 1 — Gathering (conversational)
    # ═══════════════════════════════════════════════════════

    async def _run_gathering_phase(
        self,
        user_id: str,
        session_id: str,
        message: str,
        extraction: ExtractionResult,
        state: DiagnosticState,
        session: SessionData,
    ) -> AsyncGenerator[str, None]:
        """Generate a warm conversational response with a follow-up question."""

        # Build gathering prompt (LLM generates symptom-specific follow-up)
        prompt = self._build_gathering_prompt(
            user_message=message,
            extraction=extraction,
            session=session,
            state=state,
        )

        # Stream the response
        full_response = ""
        async for token in self.llm.stream_generate(prompt, system_prompt=""):
            full_response += token
            yield self._sse({"token": token})

        yield self._sse({
            "done": True,
            "phase": "gathering",
            "turn": state.turn_number,
            "symptoms_collected": extraction.symptoms,
        })

        # Persist
        self.firebase.add_message(user_id, session_id, "assistant", full_response)

        logger.info(
            "Gathering turn %d complete — symptoms so far: %s",
            state.turn_number,
            extraction.symptoms,
        )

    # ═══════════════════════════════════════════════════════
    # PHASE 2 — Full Diagnosis
    # ═══════════════════════════════════════════════════════

    async def _run_diagnosis_phase(
        self,
        user_id: str,
        session_id: str,
        message: str,
        extraction: ExtractionResult,
        state: DiagnosticState,
        session: SessionData,
    ) -> AsyncGenerator[str, None]:
        """Generate the full structured diagnostic response."""

        # ── Hybrid retrieval ────────────────────────────────
        candidates: list[RetrievalResult] = []
        if extraction.symptoms:
            symptom_query = ", ".join(extraction.symptoms)
            candidates = self.retriever.retrieve(
                query=symptom_query,
                user_symptoms=extraction.symptoms,
                top_k_vector=10,
                top_k_final=3,
            )
        state.candidates = candidates

        # ── Severity classification ─────────────────────────
        severity = await self.severity_clf.classify(
            extraction=extraction,
            candidates=candidates,
            conversation_history=session.previous_messages,
            previous_severity=state.severity,
        )
        state.severity = severity

        # ── Medications & remedies ──────────────────────────
        med_suggestions = self.medicine.get_suggestions(
            symptoms=extraction.symptoms,
            age=extraction.age,
            severity=severity.level,
        )
        med_text = self.medicine.format_for_response(
            med_suggestions, severity=severity.level
        )

        disease_names = [r.disease.disease_name for r in candidates]
        remedy_matches = self.remedies.get_remedies(
            disease_names=disease_names,
            symptoms=extraction.symptoms,
        )
        remedy_text = self.remedies.format_for_response(remedy_matches)

        # ── Build diagnosis prompt ──────────────────────────
        prompt = self._build_diagnosis_prompt(
            user_message=message,
            extraction=extraction,
            candidates=candidates,
            severity=severity,
            med_text=med_text,
            remedy_text=remedy_text,
            session=session,
        )

        # Stream the response
        full_response = ""
        async for token in self.llm.stream_generate(prompt, system_prompt=""):
            full_response += token
            yield self._sse({"token": token})

        yield self._sse({
            "done": True,
            "phase": "diagnosis",
            "severity": severity.to_dict(),
            "candidates": [
                {
                    "disease": r.disease.disease_name,
                    "score": r.final_score,
                }
                for r in candidates[:3]
            ],
            "extraction": extraction.to_dict(),
        })

        # ── Persist state ───────────────────────────────────
        self.firebase.add_message(user_id, session_id, "assistant", full_response)

        candidate_data = [
            {
                "disease_name": r.disease.disease_name,
                "final_score": r.final_score,
                "symptoms": r.disease.symptoms[:10],
            }
            for r in candidates
        ]
        candidate_data.append({"_severity": severity.to_dict()})
        self.firebase.update_disease_candidates(
            user_id, session_id, candidate_data
        )

        logger.info(
            "Diagnosis turn %d complete — severity=%s, candidates=%d",
            state.turn_number,
            severity.level,
            len(candidates),
        )

    # ═══════════════════════════════════════════════════════
    # Prompt builders
    # ═══════════════════════════════════════════════════════

    def _build_gathering_prompt(
        self,
        user_message: str,
        extraction: ExtractionResult,
        session: SessionData,
        state: DiagnosticState,
    ) -> str:
        """Build the conversational gathering prompt."""
        history_parts: list[str] = []
        for msg in session.previous_messages[-6:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:200]
            history_parts.append(f"{role}: {content}")
        history = "\n".join(history_parts) or "First message."

        # Build missing info hints
        missing_parts: list[str] = []
        if not extraction.duration:
            missing_parts.append("- Duration: how long they've had these specific symptoms")
        if not extraction.intensity:
            missing_parts.append("- Severity: how bad the symptoms feel (mild/moderate/severe)")
        if not extraction.age:
            missing_parts.append("- Age: patient's age (helps narrow down conditions)")
        if len(extraction.symptoms) < 2:
            missing_parts.append("- More symptom details: any other accompanying symptoms?")
        missing_info = "\n".join(missing_parts) if missing_parts else "All key info collected — ask about anything else relevant to their symptoms."

        return GATHERING_PROMPT_TEMPLATE.format(
            system_prompt=GATHERING_SYSTEM_PROMPT,
            symptoms=", ".join(extraction.symptoms) or "none yet",
            duration=extraction.duration or "not mentioned",
            intensity=extraction.intensity or "not mentioned",
            age=extraction.age or "not mentioned",
            gender=extraction.gender or "not mentioned",
            turn_number=state.turn_number,
            missing_info=missing_info,
            history=history,
            user_message=user_message,
        )

    def _build_diagnosis_prompt(
        self,
        user_message: str,
        extraction: ExtractionResult,
        candidates: list[RetrievalResult],
        severity: SeverityResult,
        med_text: str,
        remedy_text: str,
        session: SessionData,
    ) -> str:
        """Build the full structured diagnosis prompt."""
        # Disease context block
        if candidates:
            disease_parts: list[str] = []
            for i, r in enumerate(candidates[:3], 1):
                d = r.disease
                block = (
                    f"{i}. {d.disease_name} (score: {r.final_score:.2f})\n"
                    f"   Symptoms: {', '.join(d.symptoms[:10])}\n"
                    f"   Red flags: {', '.join(d.red_flags[:5]) or 'none'}\n"
                    f"   Treatments: {', '.join(d.treatments[:5]) or 'standard care'}\n"
                    f"   Overlap: {r.keyword_score:.0%} | "
                    f"Semantic: {r.semantic_score:.2f}"
                )
                disease_parts.append(block)
            disease_context = "\n".join(disease_parts)
        else:
            disease_context = "No strong matches found in the knowledge base."

        # History
        history_parts: list[str] = []
        for msg in session.previous_messages[-8:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:250]
            history_parts.append(f"{role}: {content}")
        history = "\n".join(history_parts) or "No history."

        return DIAGNOSIS_PROMPT_TEMPLATE.format(
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            symptoms=", ".join(extraction.symptoms) or "not specified",
            duration=extraction.duration or "not specified",
            intensity=extraction.intensity or "not specified",
            age=extraction.age or "not specified",
            gender=extraction.gender or "not specified",
            disease_context=disease_context,
            severity_level=severity.level,
            severity_reasoning=severity.reasoning,
            red_flags=", ".join(severity.red_flags_found) or "none",
            recommended_action=severity.recommended_action or "monitor symptoms",
            medication_text=med_text or "No specific OTC suggestions at this time.",
            remedy_text=remedy_text or "No specific home remedies matched.",
            history=history,
            user_message=user_message,
        )

    @staticmethod
    def _sse(data: dict) -> str:
        """Format dict as SSE data line."""
        return f"data: {json.dumps(data)}\n\n"
