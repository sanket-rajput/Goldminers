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

import hashlib
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
        # In-memory accounts fallback: {username: {name, password_hash, created_at}}
        self._mem_accounts: dict[str, dict] = {}

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
    # User Account Management
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _hash_password(password: str) -> str:
        """SHA-256 hash of the password."""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def register_user(self, name: str, password: str) -> dict:
        """Create a new user account. Returns {success, user_id, message}."""
        username = name.strip().lower().replace(" ", "_")
        password_hash = self._hash_password(password)
        now = datetime.now(timezone.utc).isoformat()

        if self.is_firebase_active:
            try:
                doc_ref = self._db.collection("accounts").document(username)
                doc = doc_ref.get()
                if doc.exists:
                    return {"success": False, "user_id": "", "message": "Account already exists. Please login."}
                doc_ref.set({
                    "name": name.strip(),
                    "password_hash": password_hash,
                    "created_at": now,
                })
                logger.info("Account created (Firestore): %s", username)
                return {"success": True, "user_id": username, "message": "Account created successfully."}
            except Exception as exc:
                logger.error("Firestore register failed: %s — using memory.", exc)

        # In-memory fallback
        if username in self._mem_accounts:
            return {"success": False, "user_id": "", "message": "Account already exists. Please login."}
        self._mem_accounts[username] = {
            "name": name.strip(),
            "password_hash": password_hash,
            "created_at": now,
        }
        logger.info("Account created (memory): %s", username)
        return {"success": True, "user_id": username, "message": "Account created successfully."}

    def authenticate_user(self, name: str, password: str) -> dict:
        """Authenticate a user. Returns {success, user_id, display_name, message}."""
        username = name.strip().lower().replace(" ", "_")
        password_hash = self._hash_password(password)

        if self.is_firebase_active:
            try:
                doc = self._db.collection("accounts").document(username).get()
                if not doc.exists:
                    return {"success": False, "user_id": "", "display_name": "", "message": "Account not found. Please register first."}
                data = doc.to_dict()
                if data.get("password_hash") != password_hash:
                    return {"success": False, "user_id": "", "display_name": "", "message": "Incorrect password."}
                logger.info("User authenticated (Firestore): %s", username)
                return {"success": True, "user_id": username, "display_name": data.get("name", name), "message": "Login successful."}
            except Exception as exc:
                logger.error("Firestore auth failed: %s — trying memory.", exc)

        # In-memory fallback
        account = self._mem_accounts.get(username)
        if not account:
            return {"success": False, "user_id": "", "display_name": "", "message": "Account not found. Please register first."}
        if account["password_hash"] != password_hash:
            return {"success": False, "user_id": "", "display_name": "", "message": "Incorrect password."}
        logger.info("User authenticated (memory): %s", username)
        return {"success": True, "user_id": username, "display_name": account.get("name", name), "message": "Login successful."}

    def clear_all_records(self) -> dict:
        """Delete ALL data from Firestore (accounts + users/sessions). Returns summary."""
        deleted_accounts = 0
        deleted_sessions = 0

        if self.is_firebase_active:
            try:
                # Delete all accounts
                for doc in self._db.collection("accounts").stream():
                    doc.reference.delete()
                    deleted_accounts += 1

                # Delete all users and their sessions
                for user_doc in self._db.collection("users").stream():
                    # Delete sub-collection sessions
                    for sess_doc in user_doc.reference.collection("sessions").stream():
                        sess_doc.reference.delete()
                        deleted_sessions += 1
                    user_doc.reference.delete()

                logger.info("Cleared all records: %d accounts, %d sessions", deleted_accounts, deleted_sessions)
            except Exception as exc:
                logger.error("Firestore clear failed: %s", exc)

        # Clear in-memory
        mem_accounts = len(self._mem_accounts)
        mem_sessions = sum(len(s) for s in self._mem.values())
        self._mem_accounts.clear()
        self._mem.clear()

        return {
            "deleted_accounts": deleted_accounts + mem_accounts,
            "deleted_sessions": deleted_sessions + mem_sessions,
        }

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

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        deleted = False

        if self.is_firebase_active:
            try:
                (
                    self._db.collection("users")
                    .document(user_id)
                    .collection("sessions")
                    .document(session_id)
                    .delete()
                )
                logger.info("Session deleted (Firestore): %s/%s", user_id, session_id)
                deleted = True
            except Exception as exc:
                logger.error("Firestore delete failed: %s", exc)

        # Also remove from memory (may exist as fallback)
        user_sessions = self._mem.get(user_id, {})
        if session_id in user_sessions:
            del user_sessions[session_id]
            logger.info("Session deleted (memory): %s/%s", user_id, session_id)
            deleted = True

        return deleted

    def list_sessions(self, user_id: str) -> list[dict]:
        """Return a summary list of all sessions for a user, newest first.
        Temporary sessions are excluded."""
        sessions: list[dict] = []

        if self.is_firebase_active:
            try:
                docs = (
                    self._db.collection("users")
                    .document(user_id)
                    .collection("sessions")
                    .order_by("created_at", direction="DESCENDING")
                    .stream()
                )
                for doc in docs:
                    d = doc.to_dict()
                    if d.get("is_temporary", False):
                        continue
                    sessions.append(self._session_summary(d))
                return sessions
            except Exception as exc:
                logger.error("Firestore list_sessions failed: %s", exc)

        # In-memory fallback
        user_sessions = self._mem.get(user_id, {})
        for sid, session in user_sessions.items():
            if session.is_temporary:
                continue
            sessions.append(self._session_summary(session.model_dump()))

        # Sort newest first
        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions

    @staticmethod
    def _session_summary(data: dict) -> dict:
        """Extract a compact summary from full session data."""
        # Get the top disease from candidates
        candidates = data.get("disease_candidates", [])
        top_disease = ""
        for c in candidates:
            if isinstance(c, dict) and "disease_name" in c:
                top_disease = c["disease_name"]
                break

        return {
            "session_id": data.get("session_id", ""),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "phase": data.get("phase", "greeting"),
            "patient_name": data.get("patient_name", ""),
            "top_disease": top_disease,
            "symptom_count": len(data.get("extracted_symptoms", [])),
            "message_count": len(data.get("previous_messages", [])),
        }

    # ═══════════════════════════════════════════════════════
    # In-memory fallback
    # ═══════════════════════════════════════════════════════

    def _mem_set(self, user_id: str, session: SessionData) -> None:
        self._mem.setdefault(user_id, {})[session.session_id] = session

    def _mem_get(self, user_id: str, session_id: str) -> Optional[SessionData]:
        return self._mem.get(user_id, {}).get(session_id)
