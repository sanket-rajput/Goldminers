"""
Weighted disease scoring engine.

Scoring formula:
  final = 0.5 × semantic_similarity
        + 0.3 × symptom_overlap_ratio
        + 0.1 × red_flag_weight
        + 0.1 × prevalence_weight
"""

from __future__ import annotations

from app.models import DiseaseNode, RetrievalResult
from app.utils import setup_logger, normalize_symptom

logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════
# Reference Tables
# ═══════════════════════════════════════════════════════════

HIGH_PREVALENCE_DISEASES: set[str] = {
    "common cold", "influenza", "flu", "hypertension", "diabetes",
    "diabetes mellitus", "asthma", "allergic rhinitis", "rhinitis",
    "urinary tract infection", "uti", "gastritis", "migraine",
    "tension headache", "headache", "diarrhea", "acute gastroenteritis",
    "bronchitis", "pneumonia", "anemia", "iron deficiency anemia",
    "obesity", "depression", "anxiety", "generalized anxiety disorder",
    "back pain", "low back pain", "arthritis", "osteoarthritis",
    "eczema", "acne", "dengue", "malaria", "typhoid", "chickenpox",
}

CRITICAL_RED_FLAGS: set[str] = {
    "chest pain", "shortness of breath", "difficulty breathing",
    "sudden severe headache", "worst headache of life",
    "loss of consciousness", "syncope", "seizure", "convulsion",
    "hemoptysis", "coughing blood", "blood in stool", "melena",
    "sudden vision loss", "sudden weakness one side",
    "high fever above 104", "neck stiffness", "nuchal rigidity",
    "severe abdominal pain", "uncontrolled bleeding",
    "confusion", "altered mental status", "slurred speech",
    "facial drooping", "hemiplegia", "anaphylaxis",
    "suicidal ideation", "poisoning", "drug overdose",
}


# ═══════════════════════════════════════════════════════════
# Scoring Functions
# ═══════════════════════════════════════════════════════════

def compute_symptom_overlap(
    disease_symptoms: list[str],
    user_symptoms: list[str],
) -> float:
    """Fraction of user symptoms that match the disease's symptom list (substring match)."""
    if not user_symptoms:
        return 0.0

    norm_disease = {normalize_symptom(s) for s in disease_symptoms if s}
    norm_user = {normalize_symptom(s) for s in user_symptoms if s}

    if not norm_user:
        return 0.0

    matched = 0
    for u in norm_user:
        for d in norm_disease:
            if u in d or d in u:
                matched += 1
                break

    return matched / len(norm_user)


def compute_red_flag_weight(
    disease_red_flags: list[str],
    user_symptoms: list[str],
) -> float:
    """Weight based on how many user symptoms match known red flags."""
    if not user_symptoms:
        return 0.0

    all_flags = {normalize_symptom(f) for f in disease_red_flags if f}
    all_flags |= {normalize_symptom(f) for f in CRITICAL_RED_FLAGS}

    norm_user = {normalize_symptom(s) for s in user_symptoms if s}

    matched = 0
    for u in norm_user:
        for f in all_flags:
            if u in f or f in u:
                matched += 1
                break

    return min(matched / max(len(norm_user), 1), 1.0)


def compute_prevalence_weight(disease_name: str) -> float:
    """Heuristic prevalence estimate (0.0–1.0). Higher = more common."""
    lower = disease_name.lower()
    for common in HIGH_PREVALENCE_DISEASES:
        if common in lower or lower in common:
            return 0.8
    return 0.5


def compute_final_score(
    semantic_similarity: float,
    symptom_overlap: float,
    red_flag_weight: float,
    prevalence_weight: float,
) -> float:
    """
    Weighted combination:
      0.5 × semantic + 0.3 × overlap + 0.1 × red_flag + 0.1 × prevalence
    """
    return (
        0.5 * semantic_similarity
        + 0.3 * symptom_overlap
        + 0.1 * red_flag_weight
        + 0.1 * prevalence_weight
    )


# ═══════════════════════════════════════════════════════════
# Ranking
# ═══════════════════════════════════════════════════════════

def rank_diseases(
    vector_results: list[tuple[DiseaseNode, float]],
    user_symptoms: list[str],
    top_k: int = 3,
) -> list[RetrievalResult]:
    """Score, rank, and return the top-k diseases."""
    scored: list[RetrievalResult] = []

    for node, sem_score in vector_results:
        overlap = compute_symptom_overlap(node.symptoms, user_symptoms)
        red_flag = compute_red_flag_weight(node.red_flags, user_symptoms)
        prevalence = compute_prevalence_weight(node.disease_name)
        final = compute_final_score(sem_score, overlap, red_flag, prevalence)

        scored.append(
            RetrievalResult(
                disease=node,
                semantic_score=round(sem_score, 4),
                keyword_score=round(overlap, 4),
                red_flag_weight=round(red_flag, 4),
                prevalence_weight=round(prevalence, 4),
                final_score=round(final, 4),
            )
        )

    scored.sort(key=lambda r: r.final_score, reverse=True)
    top = scored[:top_k]

    logger.info(
        "Ranked %d → top %d: %s",
        len(scored),
        len(top),
        ", ".join(f"{r.disease.disease_name}({r.final_score})" for r in top),
    )
    return top
