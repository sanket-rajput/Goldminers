"""
Hybrid retriever — combines FAISS vector search with keyword overlap scoring.
"""

from __future__ import annotations

from app.embeddings import EmbeddingManager
from app.scoring import rank_diseases
from app.models import RetrievalResult
from app.utils import setup_logger, tokenize_symptoms

logger = setup_logger(__name__)


class HybridRetriever:
    """Orchestrates: embed query → FAISS top-k → hybrid re-rank."""

    def __init__(self, embedding_manager: EmbeddingManager) -> None:
        self.emb = embedding_manager

    def retrieve(
        self,
        query: str,
        user_symptoms: list[str] | None = None,
        top_k_vector: int = 10,
        top_k_final: int = 3,
    ) -> list[RetrievalResult]:
        """
        Full hybrid retrieval pipeline:
          1. Extract symptoms from query (if not supplied)
          2. FAISS semantic search → top_k_vector results
          3. Re-rank with weighted scoring → top_k_final results
        """
        if user_symptoms is None:
            user_symptoms = tokenize_symptoms(query)

        logger.info(
            "Retrieve  query='%s…'  symptoms=%s",
            query[:60],
            user_symptoms,
        )

        # 1. Vector search
        vector_hits = self.emb.search(query, top_k=top_k_vector)
        if not vector_hits:
            logger.warning("No vector results for query.")
            return []

        logger.info("Vector search → %d hits", len(vector_hits))

        # 2. Hybrid re-rank
        ranked = rank_diseases(vector_hits, user_symptoms, top_k=top_k_final)
        return ranked
