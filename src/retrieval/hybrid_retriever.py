"""
Hybrid Retriever module for the HR Help Desk RAG pipeline.
Combines dense semantic retrieval (FAISS IndexFlatIP) and lexical keyword retrieval (BM25Okapi)
using Reciprocal Rank Fusion (RRF).
"""

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.bm25_store import BM25Store

DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 5


def reciprocal_rank_fusion(
    faiss_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """
    Combines two ranked result lists using Reciprocal Rank Fusion (RRF).
    
    RRF score formula:
        RRF_score(d) = Σ [ 1 / (rrf_k + rank_i(d)) ]

    Args:
        faiss_results (List[Dict[str, Any]]): Ranked chunks from FAISS vector search.
        bm25_results (List[Dict[str, Any]]): Ranked chunks from BM25 lexical search.
        rrf_k (int): Smoothing constant (standard default: 60).
        top_k (int): Number of top fused chunks to return.

    Returns:
        List[Dict[str, Any]]: Final ranked chunks sorted by RRF score descending.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be a positive integer, got {rrf_k}")

    # Map chunk_id -> aggregate tracking dictionary
    fused: Dict[str, Dict[str, Any]] = {}

    # 1. Process FAISS rankings
    for r in faiss_results:
        cid = r["chunk_id"]
        rank = r["rank"]
        contrib = 1.0 / (rrf_k + rank)
        if cid not in fused:
            fused[cid] = {
                "chunk_id": cid,
                "text": r["text"],
                "source": r["source"],
                "page": r["page"],
                "document_title": r.get("document_title", ""),
                "chunk": r.get("chunk"),
                "rrf_score": contrib,
                "faiss_rank": rank,
                "faiss_score": r.get("score"),
                "bm25_rank": None,
                "bm25_score": None
            }
        else:
            fused[cid]["rrf_score"] += contrib
            fused[cid]["faiss_rank"] = rank
            fused[cid]["faiss_score"] = r.get("score")

    # 2. Process BM25 rankings
    for r in bm25_results:
        cid = r["chunk_id"]
        rank = r["rank"]
        contrib = 1.0 / (rrf_k + rank)
        if cid not in fused:
            fused[cid] = {
                "chunk_id": cid,
                "text": r["text"],
                "source": r["source"],
                "page": r["page"],
                "document_title": r.get("document_title", ""),
                "chunk": r.get("chunk"),
                "rrf_score": contrib,
                "faiss_rank": None,
                "faiss_score": None,
                "bm25_rank": rank,
                "bm25_score": r.get("score")
            }
        else:
            fused[cid]["rrf_score"] += contrib
            fused[cid]["bm25_rank"] = rank
            fused[cid]["bm25_score"] = r.get("score")

    # 3. Sort by RRF score descending
    sorted_items = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    # 4. Limit to top_k and assign final rank
    final_results: List[Dict[str, Any]] = []
    for rank, item in enumerate(sorted_items[:top_k], start=1):
        item_copy = dict(item)
        item_copy["rank"] = rank
        item_copy["score"] = round(item_copy["rrf_score"], 6)
        final_results.append(item_copy)

    return final_results


class HybridRetriever:
    """
    Coordinates dense semantic retrieval (FAISS) and lexical retrieval (BM25),
    fusing the two result lists using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_store: BM25Store,
        embedder: EmbeddingManager,
        rrf_k: int = DEFAULT_RRF_K,
        default_top_k: int = DEFAULT_TOP_K
    ):
        """
        Initializes the HybridRetriever.

        Args:
            vector_store (FAISSVectorStore): Initialized FAISS vector store.
            bm25_store (BM25Store): Initialized BM25 lexical store.
            embedder (EmbeddingManager): Initialized embedding manager.
            rrf_k (int): RRF constant parameter (default: 60).
            default_top_k (int): Default number of chunks to return (default: 5).
        """
        if default_top_k <= 0:
            raise ValueError(f"default_top_k must be a positive integer, got {default_top_k}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got {rrf_k}")

        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.default_top_k = default_top_k

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval for a query string:
          1. Dense search via FAISS
          2. Lexical search via BM25
          3. RRF fusion over candidates

        Args:
            query (str): The search query.
            top_k (Optional[int]): Final number of chunks to return (defaults to self.default_top_k).
            candidate_k (Optional[int]): Number of candidate chunks to fetch from each retriever
                                         before fusion (defaults to max(k * 2, 20)).

        Returns:
            List[Dict[str, Any]]: Ranked fused chunks with metadata and RRF scores.
        """
        if not query or not query.strip():
            return []

        k = top_k if (top_k is not None and top_k > 0) else self.default_top_k
        candidates = candidate_k if (candidate_k is not None and candidate_k > 0) else max(k * 2, 20)

        # 1. FAISS retrieval
        query_vec = self.embedder.embed_query(query, normalize=True)
        faiss_results = self.vector_store.search(query_vec, top_k=candidates)

        # 2. BM25 retrieval
        bm25_results = self.bm25_store.search(query, top_k=candidates)

        # 3. Reciprocal Rank Fusion
        return reciprocal_rank_fusion(
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            rrf_k=self.rrf_k,
            top_k=k
        )

    @classmethod
    def build_from_chunks(
        cls,
        chunks: List[Document],
        embedder: EmbeddingManager,
        rrf_k: int = DEFAULT_RRF_K,
        default_top_k: int = DEFAULT_TOP_K
    ) -> "HybridRetriever":
        """
        Factory helper to build both FAISS and BM25 stores from raw chunks
        and return a ready-to-search HybridRetriever.

        Args:
            chunks (List[Document]): The chunked documents to index.
            embedder (EmbeddingManager): Embedding component.
            rrf_k (int): RRF constant parameter.
            default_top_k (int): Default top_k.

        Returns:
            HybridRetriever: Initialized hybrid retriever.
        """
        vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)
        bm25_store = BM25Store.build_from_chunks(chunks)
        return cls(
            vector_store=vector_store,
            bm25_store=bm25_store,
            embedder=embedder,
            rrf_k=rrf_k,
            default_top_k=default_top_k
        )
