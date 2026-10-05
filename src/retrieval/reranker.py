"""
Cross-Encoder Reranker module for the HR Help Desk Chatbot.
Provides fine-grained semantic relevance scoring of (query, candidate_chunk) pairs
to re-order candidate pools (e.g. Top-20 from Hybrid/RRF) down to the final Top-K (e.g. 5).
"""

from typing import List, Dict, Any, Optional
import copy


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ALTERNATIVE_RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """
    Reranks candidate chunks retrieved from an upstream retriever (FAISS, BM25, or Hybrid/RRF)
    using a neural cross-encoder model.

    Unlike dual-encoders (bi-encoders) that compare pre-computed independent embeddings,
    cross-encoders feed (query, chunk_text) simultaneously through cross-attention layers,
    yielding significantly higher semantic relevance scoring at the cost of compute latency.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        model: Optional[Any] = None
    ):
        """
        Initializes the cross-encoder reranker.

        Args:
            model_name (str): HuggingFace model identifier (default: BAAI/bge-reranker-base).
            device (Optional[str]): Device to run inference on ('cpu', 'cuda', etc.).
            model (Optional[Any]): Injected CrossEncoder model instance (used for deterministic unit testing).
        """
        self.model_name = model_name
        self.device = device

        if model is not None:
            self.model = model
        else:
            from sentence_transformers import CrossEncoder
            print(f"Loading CrossEncoder model: '{self.model_name}'...")
            self.model = CrossEncoder(self.model_name, device=self.device)
            print(f"CrossEncoder model '{self.model_name}' loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Scores and ranks candidate chunks against a query.

        Args:
            query (str): The search query or employee question.
            candidates (List[Dict[str, Any]]): List of candidate chunk dictionaries.
            top_k (int): Number of top candidates to return after reranking.

        Returns:
            List[Dict[str, Any]]: Re-ordered and truncated list of candidate chunks with
                                 updated 'rank' (1-indexed) and 'rerank_score' metadata.
        """
        if not candidates or top_k <= 0:
            return []

        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            # If query is empty, return original candidates truncated to top_k with rank updated
            truncated = [copy.deepcopy(c) for c in candidates[:top_k]]
            for i, c in enumerate(truncated):
                c["rank"] = i + 1
                c["rerank_score"] = float(c.get("score", 0.0))
            return truncated

        # 1. Prepare (query, text) pairs for cross-encoder inference
        pairs = [[cleaned_query, c.get("text", "") or ""] for c in candidates]

        # 2. Compute cross-encoder relevance scores
        raw_scores = self.model.predict(pairs)

        # 3. Create enriched copies of candidates with rerank_score
        enriched_candidates = []
        for idx, candidate in enumerate(candidates):
            c_copy = copy.deepcopy(candidate)
            score_val = raw_scores[idx]
            # Handle numpy scalar / ndarray / list / float
            try:
                c_copy["rerank_score"] = float(score_val)
            except (TypeError, ValueError):
                c_copy["rerank_score"] = float(score_val[0])
            enriched_candidates.append(c_copy)

        # 4. Sort descending by rerank_score
        enriched_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 5. Truncate to top_k and re-assign 1-based ranks
        final_candidates = enriched_candidates[:top_k]
        for rank_idx, c in enumerate(final_candidates):
            c["rank"] = rank_idx + 1

        return final_candidates
