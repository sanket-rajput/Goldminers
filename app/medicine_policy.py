"""
Safe Medication Policy.

Only allows low-risk OTC suggestions with:
  - Strict rules: no antibiotics, no Rx-only drugs
  - Dosage limits (adult vs. children)
  - Mandatory disclaimers
"""

from __future__ import annotations

from app.utils import setup_logger, normalize_symptom

logger = setup_logger(__name__)

DISCLAIMER = (
    "⚕️ **Disclaimer**: This is not a substitute for professional medical advice. "
    "Always consult a qualified healthcare provider before taking any medication."
)


# ═══════════════════════════════════════════════════════════
# Safe OTC Medication Database
# ═══════════════════════════════════════════════════════════

OTC_MEDICATIONS: list[dict] = [
    {
        "name": "Paracetamol (Acetaminophen)",
        "indications": [
            "fever", "headache", "body pain", "myalgia", "body ache",
            "toothache", "back pain", "joint pain", "sore throat",
            "mild pain", "pyrexia",
        ],
        "adult_dosage": "500–650 mg every 4–6 hours as needed. Maximum 4000 mg (4 g) per 24 hours.",
        "child_note": "Dosage depends on weight. Consult a pediatrician for children's dosing.",
        "warnings": [
            "Do not exceed 4 g/day — risk of liver damage",
            "Avoid if you have liver disease",
            "Do not combine with alcohol",
        ],
    },
    {
        "name": "Ibuprofen",
        "indications": [
            "headache", "body pain", "myalgia", "back pain",
            "joint pain", "toothache", "menstrual cramps",
            "muscle pain", "mild pain", "inflammation",
        ],
        "adult_dosage": "200–400 mg every 4–6 hours as needed. Maximum 1200 mg per 24 hours (OTC).",
        "child_note": "5–10 mg/kg every 6–8 hours. Use only for children >6 months. Consult pediatrician.",
        "warnings": [
            "Take with food to reduce stomach irritation",
            "Avoid if you have stomach ulcers or kidney disease",
            "Not recommended during pregnancy (especially 3rd trimester)",
            "Do NOT combine with other NSAIDs or aspirin",
            "Avoid if you have asthma triggered by NSAIDs",
        ],
    },
    {
        "name": "Oral Rehydration Solution (ORS)",
        "indications": [
            "diarrhea", "vomiting", "dehydration", "loose stools",
            "gastroenteritis", "loose motions", "fever", "heat exhaustion",
        ],
        "adult_dosage": "Dissolve 1 sachet in 1 litre of clean drinking water. Sip frequently.",
        "child_note": "Give small sips every 1–2 minutes. 50–100 ml after each loose stool for under-5s.",
        "warnings": [
            "Use within 24 hours of preparation",
            "Do not boil the solution after mixing",
        ],
    },
    {
        "name": "Steam Inhalation",
        "indications": [
            "nasal congestion", "runny nose", "cold", "cough",
            "sinusitis", "sore throat", "common cold", "bronchitis",
        ],
        "adult_dosage": "Inhale steam from hot water (not boiling) for 10–15 min, 2–3 times daily. Optionally add eucalyptus oil.",
        "child_note": "Supervised steam for children >5 years. Avoid direct hot water near young children.",
        "warnings": [
            "Maintain safe distance to avoid burns",
            "Not recommended for children under 5 without supervision",
        ],
    },
    {
        "name": "Hydration & Rest",
        "indications": [
            "fever", "cold", "fatigue", "weakness", "dehydration",
            "headache", "diarrhea", "vomiting", "malaise",
        ],
        "adult_dosage": "Drink 8–10 glasses of water daily. Coconut water and clear soups are excellent. Rest adequately.",
        "child_note": "Encourage frequent small sips. Watch for signs of dehydration.",
        "warnings": [],
    },
    {
        "name": "Cetirizine / Loratadine (Antihistamine)",
        "indications": [
            "allergy", "allergic rhinitis", "itching", "hives",
            "urticaria", "sneezing", "watery eyes", "skin rash",
            "hay fever",
        ],
        "adult_dosage": "Cetirizine 10 mg once daily OR Loratadine 10 mg once daily.",
        "child_note": "Cetirizine 5 mg for ages 2–6, 10 mg for >6 years. Consult pediatrician.",
        "warnings": [
            "May cause drowsiness (cetirizine more than loratadine)",
            "Avoid operating heavy machinery if drowsy",
        ],
    },
    {
        "name": "Antacid (Calcium carbonate / Magnesium hydroxide)",
        "indications": [
            "heartburn", "acid reflux", "acidity", "hyperacidity",
            "indigestion", "gastritis", "GERD",
        ],
        "adult_dosage": "1–2 tablets chewed after meals or as needed. Follow package directions.",
        "child_note": "Not recommended for children under 12 without medical advice.",
        "warnings": [
            "Do not use for more than 2 weeks without consulting a doctor",
            "Can interact with other medications — space doses by 2 hours",
        ],
    },
    {
        "name": "Oral Zinc Supplement",
        "indications": ["diarrhea", "loose motions", "gastroenteritis"],
        "adult_dosage": "20 mg zinc once daily for 10–14 days (WHO recommendation for diarrhea).",
        "child_note": "10 mg/day for infants <6 months, 20 mg/day for older children. With ORS.",
        "warnings": [
            "Take with food to avoid nausea",
            "Do not exceed recommended duration without medical advice",
        ],
    },
    {
        "name": "Throat Lozenges",
        "indications": ["sore throat", "throat pain", "cough", "pharyngitis"],
        "adult_dosage": "Dissolve 1 lozenge slowly in mouth every 2–3 hours as needed.",
        "child_note": "Not suitable for children under 5 (choking risk).",
        "warnings": ["Do not exceed 8–10 lozenges per day"],
    },
]

# Strictly prohibited suggestions
PROHIBITED_CATEGORIES: set[str] = {
    "antibiotics", "antiviral", "antifungal", "corticosteroid",
    "opioid", "benzodiazepine", "insulin", "anticoagulant",
    "antidepressant", "antipsychotic", "immunosuppressant",
    "chemotherapy", "hormonal",
}


class MedicinePolicy:
    """Safe OTC medication suggestions with guardrails."""

    def get_suggestions(
        self,
        symptoms: list[str],
        age: str = "",
        severity: str = "mild",
    ) -> list[dict]:
        """
        Return safe OTC suggestions matching the patient's symptoms.

        Rules enforced:
        - Only low-risk OTC medications
        - Age-appropriate dosing notes
        - Disclaimer always included
        """
        if severity == "severe":
            logger.info("Severity=severe → minimal OTC, emphasize emergency care")

        matched: list[dict] = []
        norm_symptoms = {normalize_symptom(s) for s in symptoms}

        for med in OTC_MEDICATIONS:
            med_indications = {normalize_symptom(i) for i in med["indications"]}

            # Check for any overlap
            overlap = False
            for user_s in norm_symptoms:
                for med_i in med_indications:
                    if user_s in med_i or med_i in user_s:
                        overlap = True
                        break
                if overlap:
                    break

            if not overlap:
                continue

            entry: dict = {
                "name": med["name"],
                "dosage": med["adult_dosage"],
                "warnings": med["warnings"],
            }

            # Age-specific adjustments
            is_child = self._is_child(age)
            if is_child:
                entry["dosage"] = med.get("child_note", "Consult a pediatrician.")
                entry["age_note"] = "Pediatric dosing — always confirm with a doctor."

            matched.append(entry)

        # Deduplicate by name
        seen = set()
        unique: list[dict] = []
        for m in matched:
            if m["name"] not in seen:
                seen.add(m["name"])
                unique.append(m)

        logger.info(
            "Medicine policy: %d OTC matches for %d symptoms",
            len(unique),
            len(symptoms),
        )
        return unique

    def format_for_response(
        self,
        suggestions: list[dict],
        severity: str = "mild",
    ) -> str:
        """Format medication suggestions into readable text with emojis."""
        if not suggestions:
            return ""

        lines: list[str] = ["**💊 Safe OTC Suggestions:**\n"]

        if severity == "severe":
            lines.append(
                "🚨 Given the severity, please prioritize seeing a doctor. "
                "These are only for temporary relief:\n"
            )

        for i, med in enumerate(suggestions, 1):
            lines.append(f"{i}. **{med['name']}**")
            lines.append(f"   💉 Dosage: {med['dosage']}")
            if med.get("age_note"):
                lines.append(f"   ⚠️ {med['age_note']}")
            for w in med.get("warnings", []):
                lines.append(f"   ⚠️ {w}")
            lines.append("")

        lines.append(f"\n{DISCLAIMER}")
        return "\n".join(lines)

    @staticmethod
    def _is_child(age: str) -> bool:
        """Determine if patient is a child based on age string."""
        if not age:
            return False
        try:
            return int(age) < 18
        except ValueError:
            return False
