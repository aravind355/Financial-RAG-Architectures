"""
BasicRAG Dense Retriever & Cross-Encoder Reranker
=================================================
Performs single-stage dense vector search over Qdrant using BGE-M3,
followed by MS-MARCO Cross-Encoder reranking.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QDRANT_PATH  = "data/qdrant"
COLLECTION   = "finance_rag"

class Retriever:
    """Dense retriever and cross-encoder reranker for BasicRAG baseline."""

    def __init__(self):
        """Initialize BGE-M3 embedder, MS-MARCO reranker, and Qdrant client connection."""
        print("Loading embedder...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print("Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.client = QdrantClient(path=QDRANT_PATH)
        info = self.client.get_collection(COLLECTION)
        print(f"Connected to Qdrant — {info.points_count} chunks\n")

    def retrieve(self, query: str, top_k: int = 20, top_n: int = 5) -> list:
        """Retrieve top_k dense candidates from Qdrant, rerank with CrossEncoder, return top_n.

        Args:
            query (str): User natural language question.
            top_k (int): Number of dense vector search candidates to fetch.
            top_n (int): Number of final cross-encoder reranked candidates to return.

        Returns:
            list: List of dictionary candidates sorted by rerank_score descending.
        """
        # ── Step 1: dense retrieval ──────────────────────────────
        query_emb = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        count = self.client.get_collection(COLLECTION).points_count
        limit = min(top_k, count)

        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_emb,
            limit=limit,
            with_payload=True
        ).points

        candidates = []
        for scored_point in results:
            payload = scored_point.payload
            doc = payload.get("content", "")
            meta = {
                "type": payload.get("type", "text"),
                "page": payload.get("page", 0),
                "source": payload.get("source", "unknown"),
                "chunk_id": payload.get("chunk_id", "")
            }
            candidates.append({
                "content":  doc,
                "metadata": meta,
                "dense_score": round(float(scored_point.score), 4)
            })

        # ── Step 2: cross-encoder reranking ──────────────────────
        pairs         = [[query, c["content"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = round(float(score), 4)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_n]