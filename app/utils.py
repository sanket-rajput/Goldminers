"""Utility functions: configuration, logging, and text processing."""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

class Settings:
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    FIREBASE_CREDENTIALS_PATH: str = os.getenv(
        "FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json"
    )
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./cache")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct"
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# ═══════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════

def setup_logger(name: str) -> logging.Logger:
    """Create a consistently-formatted logger."""
    settings = get_settings()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger


# ═══════════════════════════════════════════════════════════
# Text Processing
# ═══════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Normalize whitespace and line endings."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_symptom(symptom: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    s = symptom.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_symptoms_from_text(text: str) -> list[str]:
    """Heuristic regex extraction of symptoms from raw medical text."""
    patterns = [
        r"(?:symptoms?|signs?|manifestations?|presents?\s+with|characterized\s+by)"
        r"[:\s]+(.*?)(?:\.\s|\n|$)",
        r"(?:patient\s+(?:may\s+)?(?:experiences?|shows?|exhibits?|has|develops?))"
        r"[:\s]+(.*?)(?:\.\s|\n|$)",
        r"(?:clinical\s+features?|cardinal\s+features?)[:\s]+(.*?)(?:\.\s|\n|$)",
    ]

    symptoms: list[str] = []
    lower = text.lower()
    for pat in patterns:
        for match in re.findall(pat, lower, re.MULTILINE):
            items = re.split(r"[,;]|\band\b", match)
            symptoms.extend(
                t.strip() for t in items if 2 < len(t.strip()) < 60
            )
    return list(dict.fromkeys(symptoms))  # deduplicate, preserve order


def tokenize_symptoms(text: str) -> list[str]:
    """Extract individual symptom phrases from a user message."""
    t = text.lower()

    # Strip conversational fluff
    stop_phrases = [
        "i have", "i am experiencing", "i feel", "i've been having",
        "suffering from", "diagnosed with", "please help", "what is",
        "can you", "tell me", "about", "doctor", "hey", "hi",
        "i think i have", "my symptoms are", "i'm having",
    ]
    for phrase in stop_phrases:
        t = t.replace(phrase, "")

    tokens = re.split(r"[,;]|\band\b|\balso\b|\bwith\b|\bplus\b", t)
    return [tok.strip() for tok in tokens if len(tok.strip()) > 2]


def ensure_dir(path: str | Path) -> Path:
    """Guarantee a directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
