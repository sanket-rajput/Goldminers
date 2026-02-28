"""
Diagnosis Engine — the central orchestrator.

Three-phase conversation flow:

  PHASE 0 — GREETING:
    Detect greeting messages (hi, hello, hey, etc.) and collect
    patient context: name, age, gender, country/region. Warm and
    empathetic with emojis. ONE question at a time.

  PHASE 1 — GATHERING (conversational):
    Warm, emoji-rich conversation with the patient.
    Ask follow-up questions one at a time to understand:
      • What symptoms they have
      • How long they've had them
      • How severe they feel
      • Any red-flag signals
    During this phase the bot NEVER shows diagnoses, severity
    charts, medication lists, or remedy lists. It just talks
    like a caring doctor taking an intake.

  PHASE 2 — DIAGNOSIS (structured):
    Full structured assessment with:
      • Top-3 conditions ranked by confidence %
      • Severity assessment
      • Safe OTC medication suggestions
      • Home remedies (region-aware)
      • Test recommendations
      • Red-flag warnings
      • Mandatory disclaimer

Safety Rules:
  - NEVER suggest antibiotics, antivirals, or Rx-only drugs
  - NEVER say "You have X" — always "Based on the information…"
  - NEVER downplay possible emergencies
  - Always recommend professional medical consultation
"""

from __future__ import annotations

import json
import re
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
# Greeting Detection
# ═══════════════════════════════════════════════════════════

GREETING_PATTERNS = re.compile(
    r"^\s*(?:hi|hello|hey|hola|namaste|good\s*(?:morning|afternoon|evening)|"
    r"howdy|sup|yo|greetings|assalam|salam|hii+|helo+|hey+)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════
# PHASE 0 — Greeting prompt
# ═══════════════════════════════════════════════════════════

GREETING_SYSTEM_PROMPT = """\
You are BluCare 💙, a warm and caring AI health companion.
The patient just greeted you or is starting a new conversation.

RULES:
1. Respond warmly with emojis (💛, 🌟, 😊).
2. Introduce yourself briefly as their health companion.
3. Ask for the NEXT missing piece of context (see below).
4. Keep it SHORT — 2-3 sentences max.
5. Sound like a friendly, caring human — NOT a robot.
6. NEVER jump to medical questions yet."""

GREETING_PROMPT_TEMPLATE = """\
{system_prompt}

=== PATIENT CONTEXT COLLECTED SO FAR ===
Name: {name}
Age: {age}
Gender: {gender}
Country/Region: {country}

=== WHAT TO ASK NEXT ===
{next_question}

=== CONVERSATION HISTORY ===
{history}

=== PATIENT'S LATEST MESSAGE ===
{user_message}

=== INSTRUCTIONS ===
Write a SHORT warm response (2-3 sentences):
1. Acknowledge what they said with warmth and emojis.
2. Ask for the NEXT missing context item naturally.
3. If ALL context is collected, warmly transition to asking about symptoms.

Examples of good questions:
- Missing age: "Could you share your age? It helps me give better guidance! 😊"
- Missing country: "Which country or region are you from? This helps me suggest relevant advice 🌍"
- All collected: "Now tell me — what's been bothering you? Take your time 💛"

Keep it natural, SHORT, and empathetic."""


# ═══════════════════════════════════════════════════════════
# PHASE 1 — Gathering prompt (conversational, NO diagnosis)
# ═══════════════════════════════════════════════════════════

GATHERING_SYSTEM_PROMPT = """\
You are BluCare 💙, a warm and empathetic medical intake assistant.
Your job RIGHT NOW is to GATHER information — NOT to diagnose.

STRICT SAFETY RULES:
1. NEVER mention possible diseases, conditions, or diagnoses yet.
2. NEVER show severity assessments, medication suggestions, or remedies.
3. NEVER give a structured diagnostic report.
4. NEVER say "You have X" or "It sounds like X disease."
5. DO acknowledge what the patient said warmly with emojis (💛, 🌡️, 😊).
6. DO ask exactly ONE clear follow-up question directly related to their symptoms.
7. Keep your response SHORT — 2-3 sentences max.
8. Sound like a caring human doctor, NOT a robot filling a form.
9. Your follow-up question MUST be specific to the symptoms they described.
   For example: if they say "headache" → ask about location, type of pain, triggers.
   If they say "stomach pain" → ask about when it happens, relation to food, etc.
   NEVER ask generic questions like "can you describe your symptoms" when they already told you.
10. Use emojis naturally but don't overdo it (1-2 per response)."""

GATHERING_PROMPT_TEMPLATE = """\
{system_prompt}

=== PATIENT CONTEXT ===
Name: {name}  |  Age: {age}  |  Gender: {gender}  |  Region: {country}

=== WHAT WE KNOW SO FAR ===
Symptoms: {symptoms}
Duration: {duration}
Severity: {intensity}
Turn: {turn_number}

=== STILL MISSING (prioritize asking about these) ===
{missing_info}

=== CONVERSATION HISTORY ===
{history}

=== PATIENT'S LATEST MESSAGE ===
{user_message}

=== INSTRUCTIONS ===
Write a SHORT response (2-3 sentences):
1. Briefly acknowledge what the patient just said with warmth.
2. Ask ONE specific follow-up question about their symptoms.

Your question MUST be directly relevant to the symptoms they mentioned.
Good examples:
- "headache" → "Is the pain on one side or both? Does light or noise make it worse? 🌡️"
- "stomach pain" → "Does the pain come after eating, or is it constant? 💛"
- "cough" → "Is it a dry cough or are you coughing up anything?"

If we're missing duration/severity from the list above, weave that into
your symptom-specific question naturally (e.g. "How long has this throbbing
headache been going on? 🌡️").

Do NOT repeat a question already asked in the conversation history.
Do NOT list diagnoses, medications, or remedies.
Keep it short, human, and empathetic with 1-2 emojis."""


# ═══════════════════════════════════════════════════════════
# PHASE 2 — Full Diagnosis prompt (structured output)
# ═══════════════════════════════════════════════════════════

# ── Shared safety prompt for assessment & detail phases ───

DIAGNOSIS_SYSTEM_PROMPT = """\
You are BluCare , an empathetic, knowledgeable AI health companion.
You help patients understand their symptoms and guide them toward appropriate care.

STRICT SAFETY RULES — you MUST follow:
1. NEVER claim a confirmed diagnosis. Always say "Based on the information provided…"
2. NEVER say "You have X disease" — use "This could possibly indicate…" or "One possibility is…"
3. NEVER suggest antibiotics, antivirals, antifungals, or any prescription-only drugs.
4. If severity is SEVERE → strongly recommend hospital / emergency care FIRST.
5. Use warm, supportive language with emojis. The patient may be scared.
6. Include confidence percentages for each condition.
7. NEVER return "Unknown" as a diagnosis — always provide at least 1 named condition.
8. Avoid medical jargon unless necessary; explain terms simply."""


# ─────────────── ASSESSMENT-ONLY prompt (diseases + confidence + menu) ───

ASSESSMENT_PROMPT_TEMPLATE = """\
{system_prompt}

=== PATIENT INFORMATION ===
Name: {name}  |  Age: {age}  |  Gender: {gender}  |  Region: {country}
Symptoms: {symptoms}
Duration: {duration}
Severity (self-reported): {intensity}

=== CONDITIONS (from knowledge base + medical reasoning) ===
{disease_context}

=== SEVERITY ASSESSMENT ===
Level: {severity_level}
Reasoning: {severity_reasoning}
Red flags: {red_flags}
Recommended action: {recommended_action}

=== CONVERSATION HISTORY ===
{history}

=== CURRENT USER MESSAGE ===
{user_message}

=== INSTRUCTIONS ===
Provide a concise, warm assessment using this EXACT structure:

---

{severity_first_note}

 **Assessment Summary**

Based on the information you've shared, {name}, here are the most likely possibilities:

🔹 **[Condition 1]** — Confidence: [X]%
   Brief explanation of why this matches, in simple language.

🔹 **[Condition 2]** — Confidence: [X]%
   Brief explanation.

🔹 **[Condition 3]** — Confidence: [X]%
   Brief explanation.

---

 **Severity Assessment**
How serious this appears and why. Address the patient warmly.
{age_dosage_note}

---

⚠️ **Red Flag Warning** (ONLY if red flags were detected)
Emphasize urgency with clear action steps.

---

Then GENTLY offer the next options (EXACT wording):

"Would you like me to share:
💊 **Safe medicine suggestions**
🌿 **Home remedies**
🧪 **Recommended diagnostic tests**

Or would you like **all of the above**?
Just let me know! 💛"

---

CRITICAL RULES:
- Use the confidence percentages from the conditions above.
- ❌ Do NOT show medications, remedies, or test lists yet — ONLY offer the menu.
- ❌ NEVER return "Unknown" or "Undetermined" — always infer a named condition.
- If the knowledge base had no strong matches, use your medical expertise to
  infer the most likely conditions based on symptoms, duration, age, gender, region.
- Be warm and empathetic with emojis.
{severity_first_note}"""


# ─────────────── DETAIL prompt (meds / remedies / tests) ─────────────

DETAIL_SYSTEM_PROMPT = """\
You are BluCare , continuing to help a patient who already received
their assessment. Now provide the specific information they requested.

STRICT SAFETY RULES:
1. NEVER suggest antibiotics, antivirals, or prescription-only drugs.
2. Use warm, empathetic language with emojis.
3. Only show the sections the patient requested.
4. Always end with the medical disclaimer.
5. NEVER re-list the disease assessment — the patient already saw it."""

DETAIL_PROMPT_TEMPLATE = """\
{system_prompt}

=== PATIENT CONTEXT ===
Name: {name}  |  Age: {age}  |  Gender: {gender}  |  Region: {country}
Symptoms: {symptoms}
Conditions identified: {conditions}
Severity: {severity_level}

=== SECTIONS THE PATIENT REQUESTED ===
{requested_sections_list}

=== DATA FOR REQUESTED SECTIONS ===
{section_data}

=== CONVERSATION HISTORY (last few messages) ===
{history}

=== PATIENT'S LATEST MESSAGE ===
{user_message}

=== INSTRUCTIONS ===
Present ONLY the sections the patient asked for, using the data above.
Use this formatting for each section:

{format_instructions}

After providing the information, end with:

---

⚕️ **Important Reminder**
"This is for informational purposes only and NOT a substitute
for professional medical advice. Please consult a healthcare
provider for proper diagnosis and treatment. 💙"

---

Then warmly ask:
"Is there anything else you'd like to know? I'm here for you 💛"

Be warm, clear, and helpful. Use emojis naturally."""


# ═══════════════════════════════════════════════════════════
# Test Recommendation Logic
# ═══════════════════════════════════════════════════════════

COMMON_TESTS: dict[str, list[str]] = {
    "fever": ["Complete Blood Count (CBC)", "Widal Test", "Dengue NS1 Antigen"],
    "prolonged_fever": [
        "CBC", "Widal Test", "Blood Culture",
        "Malaria Parasite Test", "Dengue NS1/IgM",
    ],
    "cough": ["Chest X-ray", "Sputum Culture"],
    "prolonged_cough": [
        "Chest X-ray", "Sputum AFB (TB test)", "Pulmonary Function Test",
    ],
    "abdominal_pain": ["Ultrasound Abdomen", "Liver Function Test (LFT)"],
    "diarrhea": ["Stool Routine & Microscopy", "Stool Culture"],
    "urinary": ["Urine Routine & Microscopy", "Urine Culture"],
    "joint_pain": ["ESR", "CRP", "Rheumatoid Factor", "Uric Acid"],
    "headache": ["Blood Pressure check", "Eye examination"],
    "chest_pain": ["ECG", "Troponin Test", "Chest X-ray"],
    "skin_rash": ["CBC", "IgE levels", "Skin scraping"],
    "fatigue": [
        "CBC", "Thyroid Function Test (TSH)", "Blood Sugar (Fasting)",
    ],
    "breathing": [
        "Chest X-ray", "Pulmonary Function Test", "SpO2 monitoring",
    ],
}


def _get_test_recommendations(
    symptoms: list[str],
    duration: str,
    severity: str,
) -> str:
    """Generate test recommendations based on symptoms, duration, severity."""
    recommended: list[str] = []
    seen: set[str] = set()

    is_prolonged = _is_prolonged_duration(duration)
    symptom_text = " ".join(s.lower() for s in symptoms)

    for key, tests in COMMON_TESTS.items():
        should_add = False

        if key == "prolonged_fever" and is_prolonged and "fever" in symptom_text:
            should_add = True
        elif key == "prolonged_cough" and is_prolonged and "cough" in symptom_text:
            should_add = True
        elif key == "fever" and "fever" in symptom_text and not is_prolonged:
            should_add = True
        elif key == "cough" and "cough" in symptom_text and not is_prolonged:
            should_add = True
        elif key == "abdominal_pain" and any(
            w in symptom_text for w in ["abdominal", "stomach", "belly"]
        ):
            should_add = True
        elif key == "diarrhea" and any(
            w in symptom_text for w in ["diarrhea", "loose"]
        ):
            should_add = True
        elif key == "urinary" and any(
            w in symptom_text for w in ["urination", "urine", "burning urin"]
        ):
            should_add = True
        elif key == "joint_pain" and any(
            w in symptom_text for w in ["joint", "arthritis"]
        ):
            should_add = True
        elif key == "headache" and "headache" in symptom_text:
            should_add = True
        elif key == "chest_pain" and "chest" in symptom_text:
            should_add = True
        elif key == "skin_rash" and any(
            w in symptom_text for w in ["rash", "itching", "hives"]
        ):
            should_add = True
        elif key == "fatigue" and any(
            w in symptom_text for w in ["fatigue", "weakness", "tired"]
        ):
            should_add = True
        elif key == "breathing" and any(
            w in symptom_text for w in ["breath", "wheez"]
        ):
            should_add = True

        if should_add:
            for test in tests:
                if test not in seen:
                    seen.add(test)
                    recommended.append(test)

    # If severe, add general tests
    if severity == "severe" and not recommended:
        recommended = [
            "Complete Blood Count (CBC)", "Blood Pressure", "Blood Sugar",
        ]

    if not recommended:
        return (
            "No specific tests recommended at this time. "
            "If symptoms persist beyond 3-5 days, consult a doctor."
        )

    lines = ["Based on your symptoms and duration, consider these tests:"]
    for test in recommended[:6]:
        lines.append(f"  • {test}")

    if is_prolonged:
        lines.append(
            "\n⏳ Since your symptoms have persisted, "
            "these tests are especially important."
        )
    if severity == "severe":
        lines.append(
            "\n🚨 Given the severity, please get these tests done urgently."
        )

    return "\n".join(lines)


def _is_prolonged_duration(duration: str) -> bool:
    """Check if duration exceeds ~5 days."""
    if not duration:
        return False
    d = duration.lower()
    if any(w in d for w in ["week", "month", "year"]):
        return True
    match = re.search(r"(\d+)\s*day", d)
    if match and int(match.group(1)) > 5:
        return True
    if any(w in d for w in ["long time", "several days", "more than a week"]):
        return True
    return False


# ═══════════════════════════════════════════════════════════
# Region-Aware Notes
# ═══════════════════════════════════════════════════════════

REGION_FOOD_SUGGESTIONS: dict[str, str] = {
    "india": (
        "Consider light foods like khichdi, dal rice, or curd rice. "
        "Avoid oily & spicy food. Drink warm water with tulsi or ginger."
    ),
    "pakistan": (
        "Try light foods like khichdi or daliya. "
        "Warm water with honey and ginger can help. Avoid fried foods."
    ),
    "bangladesh": (
        "Light dal-bhat with minimal spice. "
        "Warm water and rest. Avoid heavy curries."
    ),
    "nepal": "Warm dal-bhat, ginger tea, and adequate rest. Keep hydrated.",
    "south asia": (
        "Consider light foods like khichdi, dal rice. "
        "Warm ginger water. Avoid heavy, oily food."
    ),
    "default": (
        "Stick to light, easily digestible foods. "
        "Stay well hydrated. Avoid heavy or spicy meals."
    ),
}

SOUTH_ASIAN_COUNTRIES = {
    "india", "pakistan", "bangladesh", "nepal",
    "sri lanka", "bhutan", "afghanistan",
}


def _get_region_food_note(country: str) -> str:
    """Get region-specific dietary advice."""
    if not country:
        return REGION_FOOD_SUGGESTIONS["default"]
    c = country.lower().strip()
    if c in REGION_FOOD_SUGGESTIONS:
        return REGION_FOOD_SUGGESTIONS[c]
    if c in SOUTH_ASIAN_COUNTRIES:
        return REGION_FOOD_SUGGESTIONS["south asia"]
    return REGION_FOOD_SUGGESTIONS["default"]


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
        self.phase: str = "greeting"

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "extraction": self.extraction.to_dict(),
            "severity": self.severity.to_dict() if self.severity else None,
            "last_question": self.last_question,
            "phase": self.phase,
        }

    @classmethod
    def from_session(cls, session: SessionData) -> DiagnosticState:
        """Reconstruct diagnostic state from session data."""
        state = cls()
        state.turn_number = len([
            m for m in session.previous_messages if m.get("role") == "user"
        ])
        state.phase = session.phase or "greeting"
        state.extraction = ExtractionResult()
        state.extraction.symptoms = list(session.extracted_symptoms)
        state.extraction.age = session.patient_age
        state.extraction.gender = session.patient_gender

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
    Central orchestrator with three-phase conversation flow.

    Phase 0 (Greeting):   Warm introduction, collect age/gender/country.
    Phase 1 (Gathering):  Warm conversation, ask questions, collect symptoms.
    Phase 2 (Diagnosis):  Full structured assessment with confidence %.
    """

    MIN_SYMPTOMS_FOR_DIAGNOSIS = 2
    MIN_GATHERING_TURNS = 3
    MAX_GATHERING_TURNS = 5

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

    # ═══════════════════════════════════════════════════════
    # Main entry point
    # ═══════════════════════════════════════════════════════

    async def process_turn(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> AsyncGenerator[str, None]:
        """
        Execute a diagnostic turn with streaming output.

        Phase 0: Greeting → collect context (age/gender/country).
        Phase 1: Gathering → warm conversational reply + follow-up.
        Phase 2: Diagnosis → full structured assessment with confidence %.
        """
        # ── Load session ────────────────────────────────────
        session = self.firebase.get_session(user_id, session_id)
        if not session:
            yield self._sse({"error": "Session not found"})
            return

        # Save user message
        self.firebase.add_message(user_id, session_id, "user", message)
        session = self.firebase.get_session(user_id, session_id)

        # ── Reconstruct state ───────────────────────────────
        state = DiagnosticState.from_session(session)
        state.turn_number += 1

        logger.info(
            "=== Turn %d | phase=%s | user=%s session=%s ===",
            state.turn_number, state.phase, user_id[:12], session_id[:8],
        )

        # ── Route to appropriate phase ──────────────────────

        # --- PHASE 0: GREETING ---
        if state.phase == "greeting":
            context_update = self._extract_context_from_message(message, session)
            if context_update:
                for field, value in context_update.items():
                    self.firebase.update_session(
                        user_id, session_id, {field: value}
                    )
                    setattr(session, field, value)

            has_symptoms = self._message_has_symptoms(message)
            context_complete = self._is_context_complete(session)

            if has_symptoms:
                # User jumped to symptoms — transition to gathering
                state.phase = "gathering"
                self.firebase.update_session(
                    user_id, session_id, {"phase": "gathering"}
                )
                logger.info(
                    "💬 Symptoms detected in greeting — GATHERING"
                )
                # Fall through to gathering below
            elif context_complete:
                # All context collected — warm transition
                state.phase = "gathering"
                self.firebase.update_session(
                    user_id, session_id, {"phase": "gathering"}
                )
                logger.info("✅ Context complete — GATHERING")
                async for chunk in self._run_greeting_phase(
                    user_id, session_id, message, session, state, transition=True
                ):
                    yield chunk
                return
            else:
                # Still collecting context
                logger.info(
                    "👋 PHASE 0 → Greeting (turn %d)", state.turn_number
                )
                async for chunk in self._run_greeting_phase(
                    user_id, session_id, message, session, state
                ):
                    yield chunk
                return

        # --- PHASE 1: GATHERING ---
        if state.phase == "gathering":
            extraction = await self.extractor.extract(
                message, previous_symptoms=session.extracted_symptoms
            )
            state.extraction = extraction

            # Persist age/gender if newly found
            if extraction.age and not session.patient_age:
                self.firebase.update_session(
                    user_id, session_id, {"patient_age": extraction.age}
                )
                session.patient_age = extraction.age
            if extraction.gender and not session.patient_gender:
                self.firebase.update_session(
                    user_id, session_id, {"patient_gender": extraction.gender}
                )
                session.patient_gender = extraction.gender

            self.firebase.update_symptoms(
                user_id, session_id, extraction.symptoms
            )

            gathering_turns = self._count_gathering_turns(session)

            is_ready = await self._is_ready_for_diagnosis(
                extraction=extraction,
                state=state,
                message=message,
                session=session,
                gathering_turns=gathering_turns,
            )

            if is_ready:
                state.phase = "diagnosis"
                self.firebase.update_session(
                    user_id, session_id, {"phase": "diagnosis"}
                )
                logger.info(
                    "📋 PHASE 2 → Assessment (turn %d)", state.turn_number
                )
                async for chunk in self._run_assessment_phase(
                    user_id, session_id, message, extraction, state, session
                ):
                    yield chunk
                # After assessment, move to awaiting_choice
                self.firebase.update_session(
                    user_id, session_id, {"phase": "awaiting_choice"}
                )
            else:
                logger.info(
                    "💬 PHASE 1 → Gathering (turn %d)", state.turn_number
                )
                async for chunk in self._run_gathering_phase(
                    user_id, session_id, message, extraction, state, session
                ):
                    yield chunk
            return

        # --- PHASE: AWAITING_CHOICE (user picks what to see) ---
        if state.phase == "awaiting_choice":
            choices = self._detect_user_choice(message)
            has_new_symptoms = self._message_has_symptoms(message)

            if has_new_symptoms and not any(choices.values()):
                # User described new symptoms → back to gathering
                state.phase = "gathering"
                self.firebase.update_session(
                    user_id, session_id, {"phase": "gathering"}
                )
                extraction = await self.extractor.extract(
                    message, previous_symptoms=session.extracted_symptoms
                )
                self.firebase.update_symptoms(
                    user_id, session_id, extraction.symptoms
                )
                state.extraction = extraction
                logger.info(
                    "🔄 New symptoms in choice phase → GATHERING"
                )
                async for chunk in self._run_gathering_phase(
                    user_id, session_id, message, extraction, state, session
                ):
                    yield chunk
                return

            if any(choices.values()):
                logger.info(
                    "📦 DETAIL requested: meds=%s remedies=%s tests=%s",
                    choices["medications"],
                    choices["remedies"],
                    choices["tests"],
                )
                async for chunk in self._run_detail_phase(
                    user_id, session_id, message, choices, state, session
                ):
                    yield chunk
                return

            # Unclear response → gently re-offer the menu
            logger.info("❓ Unclear choice — re-offering menu")
            gentle = (
                "No worries! 💛 Just let me know what you'd like:\n\n"
                "💊 **Safe medicine suggestions**\n"
                "🌿 **Home remedies**\n"
                "🧪 **Recommended diagnostic tests**\n\n"
                "Or just say **\"all\"** for everything! 😊"
            )
            self.firebase.add_message(
                user_id, session_id, "user", message
            )
            for token in gentle.split(" "):
                yield self._sse({"token": token + " "})
            yield self._sse({
                "done": True,
                "phase": "awaiting_choice",
            })
            self.firebase.add_message(
                user_id, session_id, "assistant", gentle
            )
            return

        # --- PHASE 2: DIAGNOSIS (follow-up / re-assessment) ---
        if state.phase == "diagnosis":
            extraction = await self.extractor.extract(
                message, previous_symptoms=session.extracted_symptoms
            )
            self.firebase.update_symptoms(
                user_id, session_id, extraction.symptoms
            )
            state.extraction = extraction

            logger.info(
                "📋 PHASE 2 → Follow-up assessment (turn %d)",
                state.turn_number,
            )
            async for chunk in self._run_assessment_phase(
                user_id, session_id, message, extraction, state, session
            ):
                yield chunk
            self.firebase.update_session(
                user_id, session_id, {"phase": "awaiting_choice"}
            )
            return

        # --- PHASE: COMPLETED (session finished) ---
        if state.phase == "completed":
            completed_msg = (
                "This consultation session has been completed. 💙\n\n"
                "You can **download your PDF report** using the button above, "
                "or start a **New Session** for a new consultation.\n\n"
                "Take care and stay healthy! 🌟"
            )
            for token in completed_msg.split(" "):
                yield self._sse({"token": token + " "})
            yield self._sse({
                "done": True,
                "phase": "completed",
            })
            self.firebase.add_message(
                user_id, session_id, "assistant", completed_msg
            )
            return

    # ═══════════════════════════════════════════════════════
    # PHASE 0 — Greeting / Context Collection
    # ═══════════════════════════════════════════════════════

    async def _run_greeting_phase(
        self,
        user_id: str,
        session_id: str,
        message: str,
        session: SessionData,
        state: DiagnosticState,
        transition: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Generate a warm greeting response or ask for next context item."""

        next_q = self._get_next_context_question(session, transition)

        history_parts: list[str] = []
        for msg in session.previous_messages[-4:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:150]
            history_parts.append(f"{role}: {content}")
        history = "\n".join(history_parts) or "First message."

        prompt = GREETING_PROMPT_TEMPLATE.format(
            system_prompt=GREETING_SYSTEM_PROMPT,
            name=session.patient_name or "not yet shared",
            age=session.patient_age or "not yet shared",
            gender=session.patient_gender or "not yet shared",
            country=session.patient_country or "not yet shared",
            next_question=next_q,
            history=history,
            user_message=message,
        )

        full_response = ""
        async for token in self.llm.stream_generate(prompt, system_prompt=""):
            full_response += token
            yield self._sse({"token": token})

        yield self._sse({
            "done": True,
            "phase": "greeting",
            "turn": state.turn_number,
        })

        self.firebase.add_message(
            user_id, session_id, "assistant", full_response
        )

    # ═══════════════════════════════════════════════════════
    # Context helpers
    # ═══════════════════════════════════════════════════════

    def _extract_context_from_message(
        self, message: str, session: SessionData
    ) -> dict:
        """Extract age, gender, country from user's message."""
        updates = {}
        lower = message.lower().strip()

        # Age extraction
        if not session.patient_age:
            age_pats = [
                re.compile(
                    r"(?:i am|i'm|age|aged|im)\s*(\d{1,3})", re.IGNORECASE
                ),
                re.compile(
                    r"(\d{1,3})\s*(?:years?\s*old|yrs?|yo|y/?o)", re.IGNORECASE
                ),
                re.compile(r"^(\d{1,3})$"),
            ]
            for pat in age_pats:
                match = pat.search(lower)
                if match:
                    age_val = match.group(1)
                    if 1 <= int(age_val) <= 120:
                        updates["patient_age"] = age_val
                        break

        # Gender extraction
        if not session.patient_gender:
            gender_pats = [
                (re.compile(r"\b(?:male|man|boy|he)\b", re.IGNORECASE), "male"),
                (
                    re.compile(r"\b(?:female|woman|girl|she)\b", re.IGNORECASE),
                    "female",
                ),
            ]
            for pat, gender in gender_pats:
                if pat.search(lower):
                    updates["patient_gender"] = gender
                    break

        # Country extraction
        if not session.patient_country:
            countries = [
                "india", "pakistan", "bangladesh", "nepal", "sri lanka",
                "usa", "united states", "uk", "united kingdom", "canada",
                "australia", "nigeria", "kenya", "south africa", "brazil",
                "germany", "france", "indonesia", "philippines", "egypt",
                "saudi arabia", "uae", "dubai", "mexico", "japan", "china",
                "thailand", "vietnam", "turkey", "iran", "iraq",
                "afghanistan",
            ]
            for country in countries:
                if country in lower:
                    updates["patient_country"] = country.title()
                    break

        return updates

    def _message_has_symptoms(self, message: str) -> bool:
        """Quick check if message contains symptom-like content."""
        lower = message.lower()
        symptom_keywords = {
            "pain", "ache", "fever", "cough", "cold", "headache", "nausea",
            "vomiting", "diarrhea", "rash", "itching", "dizzy", "fatigue",
            "weak", "sore", "hurt", "swelling", "breathing", "congestion",
            "stomach", "throat", "chest", "sick", "ill", "unwell",
            "not feeling well", "feeling bad", "symptoms",
        }
        return any(kw in lower for kw in symptom_keywords)

    def _is_context_complete(self, session: SessionData) -> bool:
        """Check if enough context to move to gathering (need at least age)."""
        return bool(session.patient_age)

    def _get_next_context_question(
        self, session: SessionData, transition: bool = False
    ) -> str:
        """Determine the next context question to ask."""
        if transition:
            return (
                "ALL context is collected! Warmly transition to asking about "
                "their symptoms. Say something like "
                "'Now, tell me what's been bothering you? Take your time 💛'"
            )
        if not session.patient_age:
            return (
                "Ask for their AGE. Example: "
                "'Could you share your age? It helps me give better guidance! 😊'"
            )
        if not session.patient_gender:
            return (
                "Ask for their GENDER. Example: "
                "'Are you male or female? This helps me personalize the advice 🌟'"
            )
        if not session.patient_country:
            return (
                "Ask for their COUNTRY/REGION. Example: "
                "'Which country are you from? It helps me suggest relevant advice 🌍'"
            )
        return (
            "ALL context is collected! Warmly transition to asking about "
            "their symptoms. Say something like "
            "'Now, tell me what's been bothering you? Take your time 💛'"
        )

    # ═══════════════════════════════════════════════════════
    # Readiness check
    # ═══════════════════════════════════════════════════════

    async def _is_ready_for_diagnosis(
        self,
        extraction: ExtractionResult,
        state: DiagnosticState,
        message: str,
        session: SessionData,
        gathering_turns: int = 0,
    ) -> bool:
        """Determine if we have enough info to deliver a diagnosis."""
        sym_count = len(extraction.symptoms)

        # Emergency override
        if state.severity and state.severity.is_emergency:
            logger.info("🚨 Emergency detected — skipping to diagnosis")
            return True

        if sym_count < self.MIN_SYMPTOMS_FOR_DIAGNOSIS:
            logger.info(
                "Not enough symptoms (%d < %d) — gathering",
                sym_count, self.MIN_SYMPTOMS_FOR_DIAGNOSIS,
            )
            return False

        if gathering_turns < self.MIN_GATHERING_TURNS:
            logger.info(
                "Too few gathering turns (%d < %d) — gathering",
                gathering_turns, self.MIN_GATHERING_TURNS,
            )
            return False

        # Hard cap
        if gathering_turns >= self.MAX_GATHERING_TURNS:
            logger.info(
                "⏰ Max gathering turns (%d) — forcing diagnosis",
                gathering_turns,
            )
            return True

        # Ask ConversationEngine
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
            turn_number=gathering_turns,
            last_question=state.last_question,
            last_answer=message,
            conversation_history=(
                session.previous_messages if session else None
            ),
        )

        if followup is None:
            return True

        state._pending_followup = followup
        return False

    def _count_gathering_turns(self, session: SessionData) -> int:
        """Count user turns that occurred during the gathering phase."""
        count = 0
        in_gathering = False
        for msg in session.previous_messages:
            if msg.get("role") == "user":
                if in_gathering or session.phase == "gathering":
                    count += 1
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                if any(
                    w in content
                    for w in [
                        "symptom", "experiencing", "feeling",
                        "pain", "how long",
                    ]
                ):
                    in_gathering = True
        all_user = [
            m for m in session.previous_messages if m.get("role") == "user"
        ]
        if len(all_user) > 2:
            return max(count, len(all_user) - 2)
        return count

    # ═══════════════════════════════════════════════════════
    # PHASE 1 — Gathering
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

        prompt = self._build_gathering_prompt(
            user_message=message,
            extraction=extraction,
            session=session,
            state=state,
        )

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

        self.firebase.add_message(
            user_id, session_id, "assistant", full_response
        )

        logger.info(
            "Gathering turn %d complete — symptoms: %s",
            state.turn_number,
            extraction.symptoms,
        )

    # ═══════════════════════════════════════════════════════
    # PHASE 2 — Assessment (diseases + confidence only)
    # ═══════════════════════════════════════════════════════

    async def _run_assessment_phase(
        self,
        user_id: str,
        session_id: str,
        message: str,
        extraction: ExtractionResult,
        state: DiagnosticState,
        session: SessionData,
    ) -> AsyncGenerator[str, None]:
        """Generate the assessment-only response (no meds/remedies/tests)."""

        # ── Hybrid retrieval
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

        # ── Severity classification
        severity = await self.severity_clf.classify(
            extraction=extraction,
            candidates=candidates,
            conversation_history=session.previous_messages,
            previous_severity=state.severity,
        )
        state.severity = severity

        # ── Build assessment-only prompt (no meds/remedies/tests)
        prompt = self._build_assessment_prompt(
            user_message=message,
            extraction=extraction,
            candidates=candidates,
            severity=severity,
            session=session,
        )

        # Stream the response
        full_response = ""
        async for token in self.llm.stream_generate(prompt, system_prompt=""):
            full_response += token
            yield self._sse({"token": token})

        yield self._sse({
            "done": True,
            "phase": "awaiting_choice",
            "severity": severity.to_dict(),
            "candidates": [
                {
                    "disease": r.disease.disease_name,
                    "score": round(r.final_score * 100),
                    "confidence_pct": f"{r.final_score * 100:.0f}%",
                }
                for r in candidates[:3]
            ],
            "extraction": extraction.to_dict(),
        })

        # ── Persist
        self.firebase.add_message(
            user_id, session_id, "assistant", full_response
        )

        # Save diagnosis summary for PDF / sessions page
        self.firebase.update_session(
            user_id, session_id, {"diagnosis_summary": full_response}
        )

        candidate_data = [
            {
                "disease_name": r.disease.disease_name,
                "final_score": r.final_score,
                "confidence_pct": f"{r.final_score * 100:.0f}%",
                "symptoms": r.disease.symptoms[:10],
            }
            for r in candidates
        ]
        candidate_data.append({"_severity": severity.to_dict()})
        self.firebase.update_disease_candidates(
            user_id, session_id, candidate_data
        )

        logger.info(
            "Assessment turn %d complete — severity=%s, candidates=%d",
            state.turn_number,
            severity.level,
            len(candidates),
        )

    # ═══════════════════════════════════════════════════════
    # PHASE 3 — Detail (meds / remedies / tests on request)
    # ═══════════════════════════════════════════════════════

    async def _run_detail_phase(
        self,
        user_id: str,
        session_id: str,
        message: str,
        choices: dict,
        state: DiagnosticState,
        session: SessionData,
    ) -> AsyncGenerator[str, None]:
        """Show the specific sections the user requested."""

        # Track raw structured data for PDF / session storage
        raw_medications = []
        raw_remedies = []
        raw_tests = ""

        # Re-extract symptoms from session
        symptoms = list(session.extracted_symptoms)
        age = session.patient_age
        severity_level = "undetermined"

        # Reconstruct severity from stored candidates
        meta = session.disease_candidates
        if meta and isinstance(meta, list):
            last = meta[-1] if isinstance(meta[-1], dict) else {}
            sev = last.get("_severity")
            if sev and isinstance(sev, dict):
                severity_level = sev.get("level", "undetermined")

        # Reconstruct disease names from candidates
        disease_names = [
            c.get("disease_name", "")
            for c in (meta or [])
            if isinstance(c, dict) and "disease_name" in c
        ]
        conditions_str = ", ".join(disease_names) if disease_names else "as assessed"

        # ── Build section data based on choices ──
        section_data_parts: list[str] = []
        requested_labels: list[str] = []
        fmt_parts: list[str] = []

        if choices.get("medications"):
            requested_labels.append("💊 Safe Medicine Suggestions")
            med_suggestions = self.medicine.get_suggestions(
                symptoms=symptoms,
                age=age,
                severity=severity_level,
            )
            raw_medications = med_suggestions
            med_text = self.medicine.format_for_response(
                med_suggestions, severity=severity_level
            )
            section_data_parts.append(
                f"=== SAFE MEDICATIONS ===\n{med_text or 'No specific OTC suggestions.'}"
            )
            age_note = self._get_age_dosage_note(age)
            fmt_parts.append(
                f"💊 **Safe Medicine Suggestions**\n"
                f"List OTC medications with dosages from the data above.\n"
                f"NEVER suggest antibiotics or prescription drugs.\n{age_note}"
            )

        if choices.get("remedies"):
            requested_labels.append("🌿 Home Remedies")
            remedy_matches = self.remedies.get_remedies(
                disease_names=disease_names,
                symptoms=symptoms,
            )
            raw_remedies = remedy_matches
            remedy_text = self.remedies.format_for_response(remedy_matches)
            region_food = _get_region_food_note(session.patient_country)
            section_data_parts.append(
                f"=== HOME REMEDIES ===\n{remedy_text or 'No specific remedies.'}\n"
                f"Regional dietary advice: {region_food}"
            )
            fmt_parts.append(
                "🌿 **Home Remedies**\n"
                "Practical, easy-to-follow remedies from the data above.\n"
                f"🌍 Include regional food suggestion: {region_food}"
            )

        if choices.get("tests"):
            requested_labels.append("🧪 Recommended Tests")
            extraction = ExtractionResult()
            extraction.symptoms = symptoms
            # Get duration from conversation
            duration = ""
            for m in reversed(session.previous_messages):
                if m.get("role") == "user":
                    # Try to find duration mentions
                    lower = m.get("content", "").lower()
                    if any(w in lower for w in ["day", "week", "month", "hour"]):
                        duration = m.get("content", "")[:100]
                        break
            test_recs = _get_test_recommendations(
                symptoms=symptoms,
                duration=duration,
                severity=severity_level,
            )
            raw_tests = test_recs
            section_data_parts.append(
                f"=== TEST RECOMMENDATIONS ===\n{test_recs}"
            )
            fmt_parts.append(
                "🧪 **Recommended Diagnostic Tests**\n"
                "List relevant tests from the data above."
            )

        section_data = "\n\n".join(section_data_parts)
        format_instructions = "\n\n---\n\n".join(fmt_parts)
        requested_sections_list = ", ".join(requested_labels)

        # History
        history_parts: list[str] = []
        for msg in session.previous_messages[-6:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:200]
            history_parts.append(f"{role}: {content}")
        history = "\n".join(history_parts) or "No history."

        prompt = DETAIL_PROMPT_TEMPLATE.format(
            system_prompt=DETAIL_SYSTEM_PROMPT,
            name=session.patient_name or "friend",
            age=session.patient_age or "not specified",
            gender=session.patient_gender or "not specified",
            country=session.patient_country or "not specified",
            symptoms=", ".join(symptoms) or "not specified",
            conditions=conditions_str,
            severity_level=severity_level,
            requested_sections_list=requested_sections_list,
            section_data=section_data,
            history=history,
            user_message=message,
            format_instructions=format_instructions,
        )

        # Stream
        full_response = ""
        async for token in self.llm.stream_generate(prompt, system_prompt=""):
            full_response += token
            yield self._sse({"token": token})

        # Build detail data for PDF / session storage
        detail_data = {
            "medications": raw_medications,
            "remedies": raw_remedies,
            "tests": raw_tests,
            "sections_shown": [k for k, v in choices.items() if v],
        }

        # Mark session as completed
        self.firebase.update_session(user_id, session_id, {
            "phase": "completed",
            "detail_data": detail_data,
        })

        yield self._sse({
            "done": True,
            "phase": "completed",
            "detail_data": detail_data,
        })

        self.firebase.add_message(
            user_id, session_id, "assistant", full_response
        )

        logger.info(
            "Detail phase complete — sections: %s", requested_sections_list
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

        missing_parts: list[str] = []
        if not extraction.duration:
            missing_parts.append(
                "- Duration: how long they've had these specific symptoms"
            )
        if not extraction.intensity:
            missing_parts.append(
                "- Severity: how bad the symptoms feel (mild/moderate/severe)"
            )
        if len(extraction.symptoms) < 2:
            missing_parts.append(
                "- More symptom details: any other accompanying symptoms?"
            )
        missing_info = (
            "\n".join(missing_parts)
            if missing_parts
            else "All key info collected — ask about anything else relevant."
        )

        return GATHERING_PROMPT_TEMPLATE.format(
            system_prompt=GATHERING_SYSTEM_PROMPT,
            name=session.patient_name or "friend",
            age=session.patient_age or "not mentioned",
            gender=session.patient_gender or "not mentioned",
            country=session.patient_country or "not mentioned",
            symptoms=", ".join(extraction.symptoms) or "none yet",
            duration=extraction.duration or "not mentioned",
            intensity=extraction.intensity or "not mentioned",
            turn_number=state.turn_number,
            missing_info=missing_info,
            history=history,
            user_message=user_message,
        )

    def _build_assessment_prompt(
        self,
        user_message: str,
        extraction: ExtractionResult,
        candidates: list[RetrievalResult],
        severity: SeverityResult,
        session: SessionData,
    ) -> str:
        """Build the assessment-only prompt (diseases + confidence + menu)."""
        # Disease context with confidence percentages
        if candidates:
            disease_parts: list[str] = []
            for i, r in enumerate(candidates[:3], 1):
                d = r.disease
                pct = r.final_score * 100
                block = (
                    f"{i}. 🔹 {d.disease_name} — Confidence: {pct:.0f}%\n"
                    f"   Score breakdown: symptom_match={r.keyword_score:.0%}"
                    f" | semantic={r.semantic_score:.2f}"
                    f" | red_flag={r.red_flag_weight:.2f}"
                    f" | prevalence={r.prevalence_weight:.2f}\n"
                    f"   Symptoms: {', '.join(d.symptoms[:10])}\n"
                    f"   Red flags: "
                    f"{', '.join(d.red_flags[:5]) or 'none'}\n"
                    f"   Treatments: "
                    f"{', '.join(d.treatments[:5]) or 'standard care'}"
                )
                disease_parts.append(block)
            disease_context = "\n".join(disease_parts)
        else:
            # ❗ Never return "Unknown" — instruct LLM to infer
            disease_context = (
                "The knowledge base did not find strong matches.\n"
                "IMPORTANT: Use your medical expertise to INFER the top 3\n"
                "most likely conditions based on the patient's symptoms,\n"
                "duration, age, gender, and region.\n"
                "Assign realistic confidence percentages.\n"
                "NEVER use 'Unknown' or 'Undetermined' as a condition name.\n"
                "Always provide at least 1 named disease with a confidence %."
            )

        # History
        history_parts: list[str] = []
        for msg in session.previous_messages[-8:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:250]
            history_parts.append(f"{role}: {content}")
        history = "\n".join(history_parts) or "No history."

        # Age-specific dosage note
        age_note = self._get_age_dosage_note(session.patient_age)

        # Severity-first note
        severity_first = ""
        if severity.level == "severe" or severity.is_emergency:
            severity_first = (
                "🚨 CRITICAL: Severity is SEVERE — put Red Flag Warning "
                "section FIRST before everything else. Emphasize urgency."
            )

        return ASSESSMENT_PROMPT_TEMPLATE.format(
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            name=session.patient_name or "friend",
            age=session.patient_age or "not specified",
            gender=session.patient_gender or "not specified",
            country=session.patient_country or "not specified",
            symptoms=", ".join(extraction.symptoms) or "not specified",
            duration=extraction.duration or "not specified",
            intensity=extraction.intensity or "not specified",
            disease_context=disease_context,
            severity_level=severity.level,
            severity_reasoning=severity.reasoning,
            red_flags=", ".join(severity.red_flags_found) or "none",
            recommended_action=(
                severity.recommended_action or "monitor symptoms"
            ),
            history=history,
            user_message=user_message,
            age_dosage_note=age_note,
            severity_first_note=severity_first,
        )

    @staticmethod
    def _get_age_dosage_note(age: str) -> str:
        """Return age-specific dosage warning."""
        if not age:
            return ""
        try:
            age_int = int(age)
            if age_int < 12:
                return (
                    "⚠️ Patient is a CHILD — use pediatric dosing only. "
                    "Consult a pediatrician."
                )
            elif age_int < 18:
                return (
                    "⚠️ Patient is a teenager — verify adult vs. "
                    "pediatric dosing."
                )
            elif age_int > 65:
                return (
                    "⚠️ Patient is elderly — lower dosages may be "
                    "appropriate. Consult a doctor."
                )
        except ValueError:
            pass
        return ""

    @staticmethod
    def _detect_user_choice(message: str) -> dict:
        """Detect which detail sections the user wants to see."""
        lower = message.lower().strip()
        choices = {
            "medications": False,
            "remedies": False,
            "tests": False,
        }

        # "All" / affirmative patterns
        all_patterns = [
            "all", "everything", "all of the above", "all three",
            "show me all", "tell me all", "yes please", "yes all",
            "show everything", "all of them",
        ]
        if any(p in lower for p in all_patterns):
            return {"medications": True, "remedies": True, "tests": True}

        # Individual sections
        if any(w in lower for w in [
            "medicine", "medication", "drug", "💊", "med ",
            "meds", "safe suggestion", "tablet", "pill",
        ]):
            choices["medications"] = True
        if any(w in lower for w in [
            "remedy", "remedies", "home remed", "natural", "🌿",
            "herbal", "home cure", "food",
        ]):
            choices["remedies"] = True
        if any(w in lower for w in [
            "test", "diagnostic", "lab", "🧪", "check-up",
            "checkup", "investigation", "scan", "x-ray", "blood",
        ]):
            choices["tests"] = True

        # Simple affirmative → give all
        if not any(choices.values()):
            if any(w in lower for w in [
                "yes", "yeah", "yep", "ok", "okay", "please",
                "sure", "go ahead", "do it", "y",
            ]):
                return {"medications": True, "remedies": True, "tests": True}

        return choices

    @staticmethod
    def _sse(data: dict) -> str:
        """Format dict as SSE data line."""
        return f"data: {json.dumps(data)}\n\n"
