"""
Hybrid Symptom Extraction Engine.

Layer 1: Rule-based medical NER (regex + medical vocabulary)
  - Extracts symptoms, body parts, duration, severity indicators
  - No heavy ML dependency — CPU-friendly

Layer 2: LLM refinement
  - Normalizes synonyms (e.g. "body pain" → "myalgia")
  - Structures output into canonical form
  - Merges with session-accumulated symptoms
"""

from __future__ import annotations

import json
import re
from typing import Optional

from app.llm_client import LLMClient
from app.utils import setup_logger, normalize_symptom

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════
# Medical Vocabulary (expanded)
# ═══════════════════════════════════════════════════════════

SYMPTOM_LEXICON: set[str] = {
    # General
    "fever", "chills", "fatigue", "weakness", "malaise", "lethargy",
    "weight loss", "weight gain", "night sweats", "loss of appetite",
    "body pain", "myalgia", "body ache",

    # Head & neuro
    "headache", "migraine", "dizziness", "vertigo", "confusion",
    "fainting", "syncope", "seizure", "convulsion", "tremor",
    "blurred vision", "double vision", "memory loss", "numbness",
    "tingling", "slurred speech",

    # Respiratory
    "cough", "dry cough", "productive cough", "shortness of breath",
    "wheezing", "breathlessness", "chest tightness", "chest pain",
    "sore throat", "throat pain", "nasal congestion", "runny nose",
    "sneezing", "hemoptysis", "coughing blood", "difficulty breathing",

    # GI
    "nausea", "vomiting", "diarrhea", "constipation", "bloating",
    "abdominal pain", "stomach pain", "heartburn", "acid reflux",
    "blood in stool", "melena", "loss of appetite", "indigestion",
    "flatulence", "cramping",

    # Cardiac
    "palpitations", "chest pain", "rapid heartbeat", "tachycardia",
    "bradycardia", "irregular heartbeat", "swelling legs", "edema",

    # MSK
    "joint pain", "back pain", "neck pain", "muscle pain",
    "stiffness", "swelling", "knee pain", "shoulder pain",
    "leg pain", "arm pain",

    # Skin
    "rash", "itching", "hives", "skin rash", "redness", "swelling",
    "blisters", "bruising", "discoloration", "dry skin", "acne",
    "boils", "ulcers",

    # Urinary
    "burning urination", "frequent urination", "blood in urine",
    "dark urine", "painful urination", "urgency", "incontinence",

    # ENT
    "ear pain", "hearing loss", "tinnitus", "nasal congestion",
    "sinus pressure", "jaw pain", "toothache",

    # Psych
    "anxiety", "depression", "insomnia", "irritability", "mood swings",
    "panic attack", "suicidal thoughts",

    # Eyes
    "eye pain", "red eye", "watery eyes", "eye strain",
    "vision loss", "floaters",

    # Infections
    "swollen lymph nodes", "pus", "discharge", "wound infection",
}

BODY_PARTS: set[str] = {
    "head", "neck", "throat", "chest", "abdomen", "stomach", "back",
    "spine", "shoulder", "arm", "elbow", "wrist", "hand", "finger",
    "hip", "thigh", "knee", "leg", "ankle", "foot", "toe",
    "eye", "ear", "nose", "mouth", "jaw", "tongue", "skin",
    "liver", "kidney", "heart", "lung", "brain", "intestine",
    "bladder", "groin", "pelvis",
}

SEVERITY_KEYWORDS: dict[str, str] = {
    "mild": "mild",
    "slight": "mild",
    "a little": "mild",
    "moderate": "moderate",
    "significant": "moderate",
    "severe": "severe",
    "extreme": "severe",
    "unbearable": "severe",
    "terrible": "severe",
    "excruciating": "severe",
    "intense": "severe",
    "worst": "severe",
    "very bad": "severe",
    "really bad": "severe",
    "sharp": "moderate",
    "dull": "mild",
    "throbbing": "moderate",
    "constant": "moderate",
    "intermittent": "mild",
    "persistent": "moderate",
    "crushing": "severe",
}

DURATION_PATTERNS = [
    re.compile(
        r"(?:for|since|past|last|about|around|approximately)\s+"
        r"(\d+\s*(?:day|week|month|year|hour|minute|hr|min|d|w|mo|yr)s?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+\s*(?:day|week|month|year|hour|minute)s?)\s+(?:ago|now|back)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:started|began|onset)\s+(\d+\s*(?:day|week|month|year|hour)s?\s*ago)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(today|yesterday|this morning|last night|few days|couple of days|"
        r"a week|a month|recently|just now|suddenly|gradually)",
        re.IGNORECASE,
    ),
]

AGE_PATTERNS = [
    re.compile(r"(?:i am|i'm|age|aged)\s+(\d{1,3})\s*(?:years?\s*old|yrs?|yo)?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*(?:year\s*old|yr\s*old|yo|y/?o)\b", re.IGNORECASE),
]

GENDER_PATTERNS = [
    re.compile(r"\b(male|female|man|woman|boy|girl)\b", re.IGNORECASE),
    re.compile(r"\b(i am a?)\s*(male|female|man|woman)\b", re.IGNORECASE),
]

# Synonym normalization map
SYNONYM_MAP: dict[str, str] = {
    "body pain": "myalgia",
    "body ache": "myalgia",
    "body aches": "myalgia",
    "high temperature": "fever",
    "high fever": "fever",
    "low grade fever": "fever",
    "running nose": "runny nose",
    "blocked nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "loose stools": "diarrhea",
    "loose motions": "diarrhea",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "tummy pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "stomach ache": "abdominal pain",
    "cant sleep": "insomnia",
    "can't sleep": "insomnia",
    "trouble sleeping": "insomnia",
    "difficulty sleeping": "insomnia",
    "breathing difficulty": "shortness of breath",
    "hard to breathe": "shortness of breath",
    "cant breathe": "shortness of breath",
    "can't breathe": "shortness of breath",
    "feeling dizzy": "dizziness",
    "feeling tired": "fatigue",
    "always tired": "fatigue",
    "no energy": "fatigue",
    "feeling weak": "weakness",
    "chest tightness": "chest pain",
    "burning pee": "burning urination",
    "painful peeing": "burning urination",
    "peeing blood": "blood in urine",
    "seeing blood in stool": "blood in stool",
    "pooping blood": "blood in stool",
    "heart racing": "palpitations",
    "fast heartbeat": "tachycardia",
    "feeling sad": "depression",
    "feeling anxious": "anxiety",
    "feeling nervous": "anxiety",
    "panic attacks": "panic attack",
    "skin itching": "itching",
    "scratching a lot": "itching",
}


# ═══════════════════════════════════════════════════════════
# Extraction Result
# ═══════════════════════════════════════════════════════════

class ExtractionResult:
    """Structured output of symptom extraction."""

    def __init__(self) -> None:
        self.symptoms: list[str] = []
        self.body_parts: list[str] = []
        self.duration: str = ""
        self.intensity: str = ""
        self.age: str = ""
        self.gender: str = ""
        self.raw_entities: list[str] = []

    def to_dict(self) -> dict:
        return {
            "symptoms": self.symptoms,
            "body_parts": self.body_parts,
            "duration": self.duration,
            "intensity": self.intensity,
            "age": self.age,
            "gender": self.gender,
        }

    def merge(self, other: ExtractionResult) -> ExtractionResult:
        """Merge another result into this one, deduplicating."""
        self.symptoms = list(dict.fromkeys(self.symptoms + other.symptoms))
        self.body_parts = list(dict.fromkeys(self.body_parts + other.body_parts))
        if not self.duration and other.duration:
            self.duration = other.duration
        if not self.intensity and other.intensity:
            self.intensity = other.intensity
        if not self.age and other.age:
            self.age = other.age
        if not self.gender and other.gender:
            self.gender = other.gender
        return self


# ═══════════════════════════════════════════════════════════
# Layer 1: Rule-Based Extractor
# ═══════════════════════════════════════════════════════════

class RuleBasedExtractor:
    """Fast regex + vocabulary symptom extractor."""

    def extract(self, text: str) -> ExtractionResult:
        result = ExtractionResult()
        lower = text.lower().strip()

        # Symptoms
        result.symptoms = self._match_symptoms(lower)

        # Body parts
        for part in BODY_PARTS:
            if re.search(rf"\b{re.escape(part)}\b", lower):
                result.body_parts.append(part)

        # Duration
        for pat in DURATION_PATTERNS:
            match = pat.search(lower)
            if match:
                result.duration = match.group(1) if match.lastindex else match.group(0)
                result.duration = result.duration.strip()
                break

        # Severity / intensity
        for keyword, level in SEVERITY_KEYWORDS.items():
            if keyword in lower:
                result.intensity = level
                break

        # Age
        for pat in AGE_PATTERNS:
            match = pat.search(lower)
            if match:
                result.age = match.group(1).strip()
                break

        # Gender
        for pat in GENDER_PATTERNS:
            match = pat.search(lower)
            if match:
                g = match.group(match.lastindex or 1).lower()
                if g in ("male", "man", "boy"):
                    result.gender = "male"
                elif g in ("female", "woman", "girl"):
                    result.gender = "female"
                break

        logger.info(
            "Rule-based extraction: %d symptoms, duration=%s, intensity=%s",
            len(result.symptoms),
            result.duration or "?",
            result.intensity or "?",
        )
        return result

    def _match_symptoms(self, text: str) -> list[str]:
        found: list[str] = []
        # First try synonym map (longer phrases first)
        for phrase, canonical in sorted(
            SYNONYM_MAP.items(), key=lambda x: -len(x[0])
        ):
            if phrase in text:
                if canonical not in found:
                    found.append(canonical)

        # Then match lexicon
        for symptom in SYMPTOM_LEXICON:
            if re.search(rf"\b{re.escape(symptom)}\b", text):
                norm = SYNONYM_MAP.get(symptom, symptom)
                if norm not in found:
                    found.append(norm)

        return found


# ═══════════════════════════════════════════════════════════
# Layer 2: LLM Refinement
# ═══════════════════════════════════════════════════════════

LLM_EXTRACTION_PROMPT = """\
You are a medical symptom extraction assistant.

User message: "{message}"
Rule-based extracted data: {extracted}

Refine and complete the extraction. Normalize medical synonyms.
Examples: "body pain" → "myalgia", "high temperature" → "fever"

Return ONLY valid JSON (no markdown, no explanation):
{{
  "symptoms": ["normalized symptom1", "normalized symptom2"],
  "duration": "duration expression or empty string",
  "intensity": "mild|moderate|severe or empty string",
  "age": "numeric age or empty string",
  "gender": "male|female or empty string"
}}
"""


class LLMRefinementExtractor:
    """Uses LLM to normalize and refine rule-based extraction."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def refine(
        self, message: str, rule_result: ExtractionResult
    ) -> ExtractionResult:
        """Async LLM refinement of extracted symptoms."""
        try:
            prompt = LLM_EXTRACTION_PROMPT.format(
                message=message,
                extracted=json.dumps(rule_result.to_dict()),
            )
            response = await self.llm.agenerate(
                prompt,
                system_prompt=(
                    "You are a precise medical NER assistant. "
                    "Return ONLY valid JSON. No explanations."
                ),
            )
            parsed = self._parse(response)
            refined = ExtractionResult()
            refined.symptoms = parsed.get("symptoms", [])
            refined.duration = parsed.get("duration", "")
            refined.intensity = parsed.get("intensity", "")
            refined.age = parsed.get("age", "")
            refined.gender = parsed.get("gender", "")
            return refined
        except Exception as exc:
            logger.warning("LLM refinement failed: %s — using rule-based only.", exc)
            return rule_result

    @staticmethod
    def _parse(response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text.strip())


# ═══════════════════════════════════════════════════════════
# Combined Extractor (orchestration)
# ═══════════════════════════════════════════════════════════

class SymptomExtractor:
    """Dual-layer extraction: Rule-based → LLM refinement → merge."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.rule_extractor = RuleBasedExtractor()
        self.llm_extractor = LLMRefinementExtractor(llm_client)

    def extract_sync(self, message: str) -> ExtractionResult:
        """Synchronous extraction (rule-based only)."""
        return self.rule_extractor.extract(message)

    async def extract(
        self,
        message: str,
        previous_symptoms: list[str] | None = None,
    ) -> ExtractionResult:
        """Full async extraction with LLM refinement + session merge."""
        # Layer 1
        rule_result = self.rule_extractor.extract(message)

        # Layer 2
        refined = await self.llm_extractor.refine(message, rule_result)

        # Merge: rule + LLM
        merged = ExtractionResult()
        merged.symptoms = list(dict.fromkeys(
            rule_result.symptoms + refined.symptoms
        ))
        merged.body_parts = rule_result.body_parts
        merged.duration = refined.duration or rule_result.duration
        merged.intensity = refined.intensity or rule_result.intensity
        merged.age = refined.age or rule_result.age
        merged.gender = refined.gender or rule_result.gender

        # Merge with previous session symptoms
        if previous_symptoms:
            prev_norm = [normalize_symptom(s) for s in previous_symptoms]
            for s in merged.symptoms:
                ns = normalize_symptom(s)
                if ns not in prev_norm:
                    prev_norm.append(ns)
            merged.symptoms = list(dict.fromkeys(
                previous_symptoms + merged.symptoms
            ))

        logger.info(
            "Combined extraction: %d symptoms, dur=%s, int=%s, age=%s",
            len(merged.symptoms),
            merged.duration or "?",
            merged.intensity or "?",
            merged.age or "?",
        )
        return merged

    def get_missing_fields(self, result: ExtractionResult) -> list[str]:
        """Return list of critical diagnostic fields that are still missing."""
        missing: list[str] = []
        if not result.symptoms:
            missing.append("symptoms")
        if not result.duration:
            missing.append("duration")
        if not result.intensity:
            missing.append("intensity")
        if not result.age:
            missing.append("age")
        return missing
