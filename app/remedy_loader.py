"""
Home Remedy Loader.

Loads remedies.json and provides disease-matched remedies
for the final response.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.utils import get_settings, setup_logger, normalize_symptom

logger = setup_logger(__name__)


class RemedyLoader:
    """Loads and queries the home remedies knowledge base."""

    def __init__(self, remedies_path: str | None = None) -> None:
        settings = get_settings()
        self.path = Path(remedies_path or settings.DATA_DIR) / "remedies.json"
        self.remedies: list[dict] = []
        self._loaded = False

    def load(self) -> bool:
        """Load remedies from JSON file. Returns True on success."""
        if self._loaded:
            return True

        if not self.path.exists():
            logger.warning("remedies.json not found at %s", self.path)
            return False

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.remedies = data.get("remedies", [])
            self._loaded = True
            logger.info("Loaded %d remedy entries from %s", len(self.remedies), self.path)
            return True
        except Exception as exc:
            logger.error("Failed to load remedies: %s", exc)
            return False

    def get_remedies(
        self,
        disease_names: list[str],
        symptoms: list[str] | None = None,
    ) -> list[dict]:
        """
        Find matching remedies for given diseases/symptoms.

        Returns list of {condition, remedies[]} dicts.
        """
        if not self._loaded:
            self.load()

        if not self.remedies:
            return []

        # Combine disease names + symptoms for broader matching
        search_terms = {normalize_symptom(d) for d in disease_names if d}
        if symptoms:
            search_terms |= {normalize_symptom(s) for s in symptoms if s}

        matched: list[dict] = []
        seen_conditions: set[str] = set()

        for entry in self.remedies:
            conditions = entry.get("conditions", [])
            norm_conditions = {normalize_symptom(c) for c in conditions}

            # Check for overlap
            for term in search_terms:
                hit = False
                for cond in norm_conditions:
                    if term in cond or cond in term:
                        hit = True
                        break
                if hit:
                    condition_key = conditions[0] if conditions else "unknown"
                    if condition_key not in seen_conditions:
                        seen_conditions.add(condition_key)
                        matched.append({
                            "condition": condition_key,
                            "remedies": entry.get("remedies", []),
                        })
                    break

        logger.info(
            "Remedy lookup: %d matches for terms=%s",
            len(matched),
            list(search_terms)[:5],
        )
        return matched

    def format_for_response(self, matched_remedies: list[dict]) -> str:
        """Format matched remedies into readable text."""
        if not matched_remedies:
            return ""

        lines: list[str] = ["**🌿 Home Remedies:**\n"]

        for entry in matched_remedies[:3]:  # cap at 3 conditions
            condition = entry["condition"].title()
            lines.append(f"For **{condition}**:")
            for remedy in entry["remedies"][:4]:  # cap at 4 per condition
                lines.append(f"  • {remedy}")
            lines.append("")

        return "\n".join(lines)
