"""
Data ingestion pipeline.

Responsibilities:
  1. Load all .txt / .json files from the data directory
  2. Detect disease-level sections via heuristic heading detection
  3. (Optionally) use LLM to extract structured fields
  4. Cache structured DiseaseNode list to JSON
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from app.models import DiseaseNode
from app.llm_client import LLMClient
from app.utils import (
    get_settings,
    setup_logger,
    clean_text,
    extract_symptoms_from_text,
    ensure_dir,
)

logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════
# LLM Structuring Prompt
# ═══════════════════════════════════════════════════════════

STRUCTURING_PROMPT = """\
You are a medical data extraction assistant.
Given the following medical text about a disease or condition, extract structured information.

Text:
{text}

Return ONLY valid JSON (no markdown fences, no explanation) with this exact structure:
{{
  "disease_name": "name of the disease or condition",
  "symptoms": ["symptom1", "symptom2"],
  "red_flags": ["urgent symptom requiring immediate attention"],
  "treatments": ["treatment1", "treatment2"],
  "complications": ["complication1", "complication2"]
}}

If a field cannot be determined, use an empty list [].
Return ONLY the JSON object."""


# ═══════════════════════════════════════════════════════════
# 1) Book Loader
# ═══════════════════════════════════════════════════════════

class BookLoader:
    """Recursively loads .txt and .json files from the data directory."""

    def __init__(self, data_dir: str | None = None) -> None:
        self.data_dir = Path(data_dir or get_settings().DATA_DIR)
        logger.info("BookLoader → data_dir: %s", self.data_dir)

    def load_all_files(self) -> list[tuple[str, str]]:
        """Return list of (filename, raw_content) pairs."""
        files: list[tuple[str, str]] = []

        if not self.data_dir.exists():
            logger.warning("Data directory not found: %s", self.data_dir)
            return files

        for ext in ("*.txt", "*.json"):
            for filepath in sorted(self.data_dir.rglob(ext)):
                try:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                    if len(content.strip()) < 100:
                        continue
                    cleaned = clean_text(content) if ext == "*.txt" else content
                    files.append((filepath.name, cleaned))
                    logger.info(
                        "  Loaded %-50s (%s chars)",
                        filepath.name[:50],
                        f"{len(content):,}",
                    )
                except Exception as exc:
                    logger.error("Failed to load %s: %s", filepath, exc)

        logger.info("Total files loaded: %d", len(files))
        return files


# ═══════════════════════════════════════════════════════════
# 2) Disease Section Detector
# ═══════════════════════════════════════════════════════════

class DiseaseSectionDetector:
    """Heuristic detection of disease-level sections in unstructured text."""

    HEADING_PATTERNS = [
        # ALL CAPS line (3–120 chars)
        re.compile(r"^([A-Z][A-Z\s\-\/\(\)\',]{2,120})$", re.MULTILINE),
        # Line ending with colon
        re.compile(r"^(.{5,80})\s*:\s*$", re.MULTILINE),
        # Numbered section:  "123 - DISEASE NAME"
        re.compile(
            r"^\d{1,4}[\s\-\.]+([A-Z][A-Za-z\s\-\/\(\)\',]{3,80})$", re.MULTILINE
        ),
        # "Chapter X - Title" / "Part X - Title"
        re.compile(
            r"^(?:Chapter|Part|Section)\s+[\dIVXLCDM]+[\s\-:]+(.+)$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ]

    # ── Public API ───────────────────────────────────────

    def detect_sections(
        self, text: str, source_file: str
    ) -> list[dict[str, str]]:
        """Return list of {title, text, source_file}."""
        sections = self._detect_by_headings(text, source_file)

        if len(sections) < 3:
            logger.info(
                "Few headings in %s (%d). Falling back to chunk-based splitting.",
                source_file,
                len(sections),
            )
            sections = self._detect_by_chunks(text, source_file)

        logger.info("  %s → %d sections", source_file[:40], len(sections))
        return sections

    def parse_json_file(
        self, content: str, source_file: str
    ) -> list[dict[str, str]]:
        """Parse structured JSON into sections."""
        sections: list[dict[str, str]] = []
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error in %s: %s", source_file, exc)
            return sections

        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue

            name = (
                item.get("plant_name")
                or item.get("disease_name")
                or item.get("name")
                or "Unknown"
            )
            parts = [f"Name: {name}"]

            # Structured fields
            for key in ("botanical_name", "family", "description", "definition"):
                if key in item:
                    parts.append(f"{key}: {item[key]}")

            # Medicinal uses (list of {ailment, treatment})
            for use in item.get("medicinal_uses", []):
                ailment = use.get("ailment", "")
                treatment = use.get("treatment", "")
                parts.append(f"Ailment: {ailment} — Treatment: {treatment}")

            # Catch-all for remaining string fields
            for k, v in item.items():
                if isinstance(v, str) and k not in (
                    "id", "plant_name", "disease_name", "name",
                    "botanical_name", "family", "description", "definition",
                ):
                    parts.append(f"{k}: {v}")

            sections.append({
                "title": name,
                "text": "\n".join(parts),
                "source_file": source_file,
            })

        return sections

    # ── Private helpers ──────────────────────────────────

    def _detect_by_headings(
        self, text: str, source_file: str
    ) -> list[dict[str, str]]:
        headings: list[tuple[int, str]] = []
        for pattern in self.HEADING_PATTERNS:
            for m in pattern.finditer(text):
                title = (m.group(1) if m.lastindex else m.group(0)).strip()
                if 3 <= len(title) <= 120:
                    headings.append((m.start(), title))

        if not headings:
            return []

        # Sort by position, deduplicate headings that are too close
        headings.sort(key=lambda h: h[0])
        headings = self._deduplicate(headings, min_gap=50)

        sections: list[dict[str, str]] = []
        for i, (pos, title) in enumerate(headings):
            end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            body = text[pos:end].strip()
            if len(body) < 100:
                continue
            sections.append({
                "title": title,
                "text": body,
                "source_file": source_file,
            })
        return sections

    def _detect_by_chunks(
        self, text: str, source_file: str, chunk_size: int = 2000
    ) -> list[dict[str, str]]:
        paragraphs = text.split("\n\n")
        sections: list[dict[str, str]] = []
        buf = ""
        idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            buf += para + "\n\n"
            if len(buf) >= chunk_size:
                sections.append({
                    "title": f"{source_file}_chunk_{idx}",
                    "text": buf.strip(),
                    "source_file": source_file,
                })
                buf = para + "\n\n"  # overlap last paragraph
                idx += 1

        if len(buf.strip()) > 100:
            sections.append({
                "title": f"{source_file}_chunk_{idx}",
                "text": buf.strip(),
                "source_file": source_file,
            })
        return sections

    @staticmethod
    def _deduplicate(
        headings: list[tuple[int, str]], min_gap: int
    ) -> list[tuple[int, str]]:
        if not headings:
            return []
        result = [headings[0]]
        for pos, title in headings[1:]:
            if pos - result[-1][0] >= min_gap:
                result.append((pos, title))
        return result


# ═══════════════════════════════════════════════════════════
# 3) LLM-Assisted Structurer
# ═══════════════════════════════════════════════════════════

class DiseaseStructurer:
    """Uses LLM to extract structured fields from raw text sections."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def structure_section(
        self, title: str, text: str, source_file: str
    ) -> DiseaseNode:
        truncated = text[:4000]
        try:
            response = self.llm.generate(
                STRUCTURING_PROMPT.format(text=truncated),
                system_prompt=(
                    "You are a precise medical data extraction assistant. "
                    "Return only valid JSON."
                ),
            )
            parsed = self._parse_response(response)
            return DiseaseNode(
                disease_name=parsed.get("disease_name", title),
                raw_text=text,
                symptoms=parsed.get("symptoms", []),
                red_flags=parsed.get("red_flags", []),
                treatments=parsed.get("treatments", []),
                complications=parsed.get("complications", []),
                source_file=source_file,
            )
        except Exception as exc:
            logger.warning(
                "LLM structuring failed for '%s': %s — using heuristic fallback.",
                title[:40],
                exc,
            )
            return DiseaseNode(
                disease_name=title,
                raw_text=text,
                symptoms=extract_symptoms_from_text(text),
                source_file=source_file,
            )

    @staticmethod
    def _parse_response(response: str) -> dict:
        text = response.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        text = text.strip()
        return json.loads(text)


# ═══════════════════════════════════════════════════════════
# 4) Orchestrator
# ═══════════════════════════════════════════════════════════

class IngestionPipeline:
    """End-to-end: load → detect → structure → cache."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        data_dir: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self.loader = BookLoader(data_dir)
        self.detector = DiseaseSectionDetector()
        self.llm_client = llm_client
        self.structurer = DiseaseStructurer(llm_client) if llm_client else None
        self.cache_dir = ensure_dir(self.settings.CACHE_DIR)

    def run(
        self, use_llm: bool = True, force_refresh: bool = False
    ) -> list[DiseaseNode]:
        """Execute the full pipeline. Returns structured DiseaseNode list."""
        cache_path = self.cache_dir / "disease_nodes.json"

        # ── Cache hit ────────────────────────────────────
        if not force_refresh and cache_path.exists():
            logger.info("Loading disease nodes from cache …")
            return self._load_cache(cache_path)

        # ── Load files ───────────────────────────────────
        logger.info("=== Ingestion pipeline starting ===")
        files = self.loader.load_all_files()
        if not files:
            logger.warning("No files found in %s", self.loader.data_dir)
            return []

        # ── Detect sections ──────────────────────────────
        all_sections: list[dict[str, str]] = []
        for filename, content in files:
            if filename.lower().endswith(".json"):
                sections = self.detector.parse_json_file(content, filename)
            else:
                sections = self.detector.detect_sections(content, filename)
            all_sections.extend(sections)

        logger.info("Total sections: %d", len(all_sections))

        # ── Structure ────────────────────────────────────
        nodes: list[DiseaseNode] = []

        if use_llm and self.structurer:
            for i, sec in enumerate(all_sections):
                try:
                    node = self.structurer.structure_section(
                        sec["title"], sec["text"], sec["source_file"]
                    )
                    nodes.append(node)
                    if (i + 1) % 50 == 0:
                        logger.info("  Structured %d / %d", i + 1, len(all_sections))
                        self._save_cache(nodes, cache_path)  # checkpoint
                except Exception as exc:
                    logger.error("Section '%s' failed: %s", sec["title"][:40], exc)
                    nodes.append(
                        DiseaseNode(
                            disease_name=sec["title"],
                            raw_text=sec["text"],
                            source_file=sec["source_file"],
                        )
                    )
        else:
            for sec in all_sections:
                nodes.append(
                    DiseaseNode(
                        disease_name=sec["title"],
                        raw_text=sec["text"],
                        symptoms=extract_symptoms_from_text(sec["text"]),
                        source_file=sec["source_file"],
                    )
                )

        # ── Persist cache ────────────────────────────────
        self._save_cache(nodes, cache_path)
        logger.info("=== Ingestion complete: %d disease nodes ===", len(nodes))
        return nodes

    # ── Cache I/O ─────────────────────────────────────────

    def _save_cache(self, nodes: list[DiseaseNode], path: Path) -> None:
        data = [n.model_dump() for n in nodes]
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Cache saved → %s (%d nodes)", path, len(nodes))

    def _load_cache(self, path: Path) -> list[DiseaseNode]:
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = [DiseaseNode(**item) for item in data]
        logger.info("Cache loaded ← %s (%d nodes)", path, len(nodes))
        return nodes
