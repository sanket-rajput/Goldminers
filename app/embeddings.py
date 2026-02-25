"""
Embedding pipeline — intfloat/e5-base-v2 + FAISS IndexFlatIP.

Handles:
  • Model loading (CPU-optimized)
  • Document embedding with "passage: " prefix
  • Query embedding with "query: " prefix
  • FAISS index build / search / persist / load
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.models import DiseaseNode
from app.utils import get_settings, setup_logger, ensure_dir

logger = setup_logger(__name__)


class EmbeddingManager:
    """Manages e5 embeddings and the FAISS inner-product index."""

    def __init__(self, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self.dimension: int = 768  # e5-base-v2 default
        self.cache_dir = ensure_dir(self.settings.CACHE_DIR)

    # ─── Lazy model loading ──────────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Model ready  (dim=%d)", self.dimension)
        return self._model

    # ═══════════════════════════════════════════════════════
    # Embedding
    # ═══════════════════════════════════════════════════════

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed documents with the e5 'passage: ' prefix. Returns (N, D) float32."""
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query with the e5 'query: ' prefix. Returns (1, D) float32."""
        embedding = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)

    # ═══════════════════════════════════════════════════════
    # Index management
    # ═══════════════════════════════════════════════════════

    def build_index(self, disease_nodes: list[DiseaseNode]) -> None:
        """Build FAISS IndexFlatIP from a list of DiseaseNode objects."""
        if not disease_nodes:
            logger.warning("No nodes to index.")
            return

        logger.info("Building FAISS index for %d nodes …", len(disease_nodes))

        texts: list[str] = []
        self.metadata = []

        for node in disease_nodes:
            # Compose a rich passage for embedding
            parts = [node.disease_name]
            if node.symptoms:
                parts.append("Symptoms: " + ", ".join(node.symptoms[:20]))
            if node.red_flags:
                parts.append("Red flags: " + ", ".join(node.red_flags[:10]))
            if node.treatments:
                parts.append("Treatments: " + ", ".join(node.treatments[:10]))
            if node.raw_text:
                parts.append(node.raw_text[:500])

            texts.append(" | ".join(parts))
            self.metadata.append(node.model_dump())

        embeddings = self.embed_passages(texts)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info("FAISS index built — %d vectors", self.index.ntotal)
        self._save_index()

    def search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[DiseaseNode, float]]:
        """Search the FAISS index. Returns (DiseaseNode, cosine_similarity) pairs."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not built.")
            return []

        q_emb = self.embed_query(query)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)

        results: list[tuple[DiseaseNode, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            node = DiseaseNode(**self.metadata[idx])
            results.append((node, float(score)))

        return results

    # ═══════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════

    def _save_index(self) -> None:
        idx_path = self.cache_dir / "faiss_index.bin"
        meta_path = self.cache_dir / "faiss_metadata.json"

        if self.index:
            faiss.write_index(self.index, str(idx_path))
            logger.info("Index saved  → %s", idx_path)

        meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Metadata saved → %s", meta_path)

    def load_index(self) -> bool:
        """Load persisted FAISS index + metadata. Returns True on success."""
        idx_path = self.cache_dir / "faiss_index.bin"
        meta_path = self.cache_dir / "faiss_metadata.json"

        if not idx_path.exists() or not meta_path.exists():
            logger.info("No persisted index found at %s", self.cache_dir)
            return False

        try:
            self.index = faiss.read_index(str(idx_path))
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            logger.info(
                "Index loaded ← %d vectors, %d metadata entries",
                self.index.ntotal,
                len(self.metadata),
            )
            return True
        except Exception as exc:
            logger.error("Failed to load index: %s", exc)
            return False
