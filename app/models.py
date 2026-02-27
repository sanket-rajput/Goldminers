"""Pydantic models and data schemas for the medical assistant."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
# Disease / Knowledge Base Models
# ═══════════════════════════════════════════════════════════

class DiseaseNode(BaseModel):
    """Structured disease entry extracted from medical text."""

    disease_name: str = ""
    raw_text: str = ""
    symptoms: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    treatments: list[str] = Field(default_factory=list)
    complications: list[str] = Field(default_factory=list)
    source_file: str = ""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class RetrievalResult(BaseModel):
    """Single result from the hybrid retrieval pipeline."""

    disease: DiseaseNode
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    red_flag_weight: float = 0.0
    prevalence_weight: float = 0.5
    final_score: float = 0.0


# ═══════════════════════════════════════════════════════════
# Session Models
# ═══════════════════════════════════════════════════════════

class SessionData(BaseModel):
    """Session state stored in Firebase / in-memory fallback."""

    user_id: str
    session_id: str
    extracted_symptoms: list[str] = Field(default_factory=list)
    previous_messages: list[dict] = Field(default_factory=list)
    disease_candidates: list[dict] = Field(default_factory=list)
    severity_state: dict = Field(default_factory=dict)
    last_question: str = ""
    turn_count: int = 0
    patient_name: str = ""
    patient_age: str = ""
    patient_gender: str = ""
    patient_country: str = ""
    phase: str = "greeting"  # greeting | gathering | diagnosis | awaiting_choice | completed
    diagnosis_summary: str = ""
    detail_data: dict = Field(default_factory=dict)
    is_temporary: bool = False
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ═══════════════════════════════════════════════════════════
# API Request / Response Models
# ═══════════════════════════════════════════════════════════

class StartSessionRequest(BaseModel):
    user_id: str
    name: str = ""
    age: str = ""
    gender: str = ""
    country: str = ""
    temporary: bool = False


class StartSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    diseases: list[RetrievalResult] = Field(default_factory=list)
