"""
Firebase session management.

Provides Firestore-backed session storage with an automatic
in-memory fallback when Firebase credentials are unavailable.

Firestore layout:
  users/{user_id}/sessions/{session_id}
    ├── extracted_symptoms   []
    ├── previous_messages    []
    ├── disease_candidates   []
    ├── created_at           ISO-8601
    └── updated_at           ISO-8601
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from app.models import SessionData
from app.utils import get_settings, setup_logger

logger = setup_logger(__name__)

# Conditional Firebase import — graceful degradation
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("firebase-admin not installed — using in-memory sessions.")


class FirebaseManager:
    """Session CRUD backed by Firestore (or in-memory fallback)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._db = None
        self._initialized = False
        # In-memory fallback: {user_id: {session_id: SessionData}}
        self._mem: dict[str, dict[str, SessionData]] = {}

    # ═══════════════════════════════════════════════════════
    # Initialisation
    # ═══════════════════════════════════════════════════════

    def initialize(self) -> bool:
        """Attempt to connect to Firestore. Returns True on success."""
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase SDK missing. In-memory mode active.")
            return False

        import os

        cred_path = self.settings.FIREBASE_CREDENTIALS_PATH
        if not os.path.exists(cred_path):
            logger.warning(
                "Credentials file not found (%s). In-memory mode active.",
                cred_path,
            )
            return False

        try:
            cred = credentials.Certificate(cred_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            self._db = firestore.client()
            self._initialized = True
            logger.info("Firebase Firestore initialised ✓")
            return True
        except Exception as exc:
            logger.error("Firebase init failed: %s — falling back to memory.", exc)
            return False

    @property
    def is_firebase_active(self) -> bool:
        return self._initialized and self._db is not None

    # ═══════════════════════════════════════════════════════
    # Session CRUD
    # ═══════════════════════════════════════════════════════

    def create_session(self, user_id: str) -> str:
        """Create a fresh session. Returns session_id."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        session = SessionData(
            user_id=user_id,
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )

        if self.is_firebase_active:
            try:
                ref = (
                    self._db.collection("users")
                    .document(user_id)
                    .collection("sessions")
                    .document(session_id)
                )
                ref.set(session.model_dump())
                logger.info("Session created (Firestore): %s/%s", user_id, session_id)
            except Exception as exc:
                logger.error("Firestore write failed: %s — saving to memory.", exc)
                self._mem_set(user_id, session)
        else:
            self._mem_set(user_id, session)
            logger.info("Session created (memory): %s/%s", user_id, session_id)

        return session_id

    def get_session(self, user_id: str, session_id: str) -> Optional[SessionData]:
        """Retrieve session. Returns None if not found."""
        if self.is_firebase_active:
            try:
                doc = (
                    self._db.collection("users")
                    .document(user_id)
                    .collection("sessions")
                    .document(session_id)
                    .get()
                )
                if doc.exists:
                    return SessionData(**doc.to_dict())
                return None
            except Exception as exc:
                logger.error("Firestore read failed: %s", exc)
                return self._mem_get(user_id, session_id)
        return self._mem_get(user_id, session_id)

    def update_session(self, user_id: str, session_id: str, data: dict) -> None:
        """Merge-update specific session fields."""
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        if self.is_firebase_active:
            try:
                (
                    self._db.collection("users")
                    .document(user_id)
                    .collection("sessions")
                    .document(session_id)
                    .update(data)
                )
                return
            except Exception as exc:
                logger.error("Firestore update failed: %s", exc)

        # Fallback / in-memory
        session = self._mem_get(user_id, session_id)
        if session:
            for k, v in data.items():
                if hasattr(session, k):
                    setattr(session, k, v)

    # ─── Convenience helpers ─────────────────────────────

    def add_message(
        self, user_id: str, session_id: str, role: str, content: str
    ) -> None:
        """Append a message to session history."""
        session = self.get_session(user_id, session_id)
        if not session:
            logger.warning("Session not found: %s/%s", user_id, session_id)
            return

        session.previous_messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.update_session(
            user_id,
            session_id,
            {"previous_messages": session.previous_messages},
        )

    def update_symptoms(
        self, user_id: str, session_id: str, symptoms: list[str]
    ) -> None:
        """Merge new symptoms into the session's accumulated list."""
        session = self.get_session(user_id, session_id)
        if not session:
            return
        merged = list(dict.fromkeys(session.extracted_symptoms + symptoms))
        self.update_session(
            user_id, session_id, {"extracted_symptoms": merged}
        )

    def update_disease_candidates(
        self, user_id: str, session_id: str, candidates: list[dict]
    ) -> None:
        """Overwrite disease candidates for the session."""
        self.update_session(
            user_id, session_id, {"disease_candidates": candidates}
        )

    # ═══════════════════════════════════════════════════════
    # In-memory fallback
    # ═══════════════════════════════════════════════════════

    def _mem_set(self, user_id: str, session: SessionData) -> None:
        self._mem.setdefault(user_id, {})[session.session_id] = session

    def _mem_get(self, user_id: str, session_id: str) -> Optional[SessionData]:
        return self._mem.get(user_id, {}).get(session_id)
