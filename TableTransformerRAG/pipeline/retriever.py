"""
pipeline/retriever.py
=====================
Three-level hierarchical retrieval (Qdrant backend) — identical to HierFinRAG.

Level 1: Section-level dense retrieval (BGE-M3 via Qdrant).
Level 2: Hybrid BM25 + dense retrieval over section children (RRF fusion).
Level 3: Table-aware row/column cell selection.
Post:    Cross-encoder reranking (BGE-reranker-v2-m3).

Retrieval parameters are identical to HierFinRAG so metrics are directly
comparable.  The only upstream difference is the parser.
"""

import json
import re
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

import config


EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

RRF_K = 60  # Reciprocal Rank Fusion smoothing constant


# ---------------------------------------------------------------------------
# Fix 2: Query expansion for Numerical/Comparison queries
# ---------------------------------------------------------------------------

_FINANCIAL_TABLE_KEYWORDS = [
    "table", "total", "net", "revenue", "income", "expense", "sales",
    "assets", "liabilities", "equity", "earnings", "operating", "gross",
    "margin", "profit", "loss", "cash", "debt", "lease", "depreciation",
    "segment", "consolidated", "statements", "financial",
]

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _expand_numerical_query(query: str) -> str:
    """Expand Numerical/Comparison queries with year tokens and financial keywords
    to shift the embedding toward table/cell chunks in vector space."""
    years       = " ".join(_YEAR_RE.findall(query))
    query_lower = query.lower()
    matched_kws = " ".join(kw for kw in _FINANCIAL_TABLE_KEYWORDS if kw in query_lower)
    expansion   = " ".join(filter(None, [years, matched_kws])).strip()
    return f"{query} {expansion}" if expansion else query


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    sparse_ranked: List[str],
    dense_ranked:  List[str],
    k: int = RRF_K,
) -> List[str]:
    """Merge two ranked chunk-id lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(sparse_ranked, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(dense_ranked, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HierarchicalRetriever:
    """Three-level hierarchical retrieval: Section → Para/Table → Cell,
    followed by cross-encoder reranking.

    Retrieval parameters are identical to HierFinRAG for fair comparison.

    Args:
        level1_k : Number of sections retrieved at Level 1.
        level2_k : Number of para/table candidates per section at Level 2.
        top_n    : Final number of reranked chunks returned to the generator.
    """

    def __init__(self, level1_k: int = 5, level2_k: int = 10, top_n: int = 5):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print("Loading cross-encoder reranker...")
        self.reranker  = CrossEncoder(RERANK_MODEL)
        self.client    = QdrantClient(path=config.QDRANT_PATH)
        info = self.client.get_collection(config.COLLECTION_NAME)
        print(f"Connected to Qdrant — {info.points_count} chunks\n")
        self.level1_k = level1_k
        self.level2_k = level2_k
        self.top_n    = top_n

    def retrieve(
        self,
        query:  str,
        intent: str = "Lookup",
        source: Optional[str] = None,
    ) -> List[dict]:
        """Full hierarchical retrieval pipeline for a single query."""
        query_str = query
        effective_level1_k = self.level1_k
        if intent in ("Numerical", "Comparison"):
            query_str = _expand_numerical_query(query)
            effective_level1_k = min(int(self.level1_k * 1.6) + 1, 12)

        query_emb = self.embedder.encode(query_str, normalize_embeddings=True).tolist()

        top_sections = self._level1_retrieve(
            query_emb, source=source, limit=effective_level1_k
        )

        candidates: List[dict] = []
        seen_ids: set          = set()

        for sec in top_sections:
            candidates.extend(self._level2_retrieve(query, query_emb, sec, seen_ids))

        # For Numerical/Comparison -- direct table search to guarantee table coverage
        if intent in ("Numerical", "Comparison"):
            candidates.extend(
                self._retrieve_tables_direct(query_emb, source, seen_ids)
            )

        for c in list(candidates):
            if c["metadata"]["type"] == "table":
                candidates.extend(self._level3_retrieve(query, query_emb, c, intent, seen_ids))

        # Change 10: For Lookup/Summarization, ALWAYS augment with direct
        # text+table search (not just as a fallback when empty).  This ensures
        # semantically relevant text content is always in the candidate pool,
        # even when pseudo-sections have weak children links.
        if intent in ("Lookup", "Summarization"):
            candidates.extend(self._retrieve_text_direct(query_emb, source, seen_ids))

        if not candidates:
            return []

        cell_cands  = [c for c in candidates if c["metadata"]["type"] == "cell"]
        other_cands = [c for c in candidates if c["metadata"]["type"] != "cell"]

        cell_cands.sort(key=lambda x: x.get("dense_score", 0.0), reverse=True)
        top_cells = cell_cands[:5]

        if other_cands:
            pairs  = [[query, c["content"]] for c in other_cands]
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                other_cands[i]["rerank_score"] = round(float(score), 4)

        # Change 18: For Numerical, give cells a bigger score boost so they
        # always rank above text/section chunks. The LLM needs table data first.
        cell_boost = 3.0 if intent == "Numerical" else 2.0

        for c in top_cells:
            c["rerank_score"] = c.get("rerank_score", 0.0) + cell_boost

        candidates = other_cands + top_cells
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Change 15: Lookup reverts to default top_n (10) since 15 overwhelms
        # the 7B model. Summarization keeps 15 — it benefits from broader context.
        effective_top_n = 15 if intent == "Summarization" else self.top_n
        return candidates[:effective_top_n]

    # ── Level 1 ──────────────────────────────────────────────────────────────

    def _level1_retrieve(
        self,
        query_emb: list,
        source:    Optional[str] = None,
        limit:     Optional[int] = None,
    ) -> List[dict]:
        """Dense section retrieval with optional source filter."""
        k = limit if limit is not None else self.level1_k

        must = [FieldCondition(key="type", match=MatchValue(value="section"))]
        if source:
            must.append(FieldCondition(key="source", match=MatchValue(value=source)))

        results = self.client.query_points(
            collection_name = config.COLLECTION_NAME,
            query           = query_emb,
            query_filter    = Filter(must=must),
            limit           = k,
            with_payload    = True,
        )
        return [r.payload for r in results.points]

    # ── Level 2 ──────────────────────────────────────────────────────────────

    def _level2_retrieve(
        self,
        query:     str,
        query_emb: list,
        section:   dict,
        seen_ids:  set,
    ) -> List[dict]:
        """Hybrid BM25 + dense retrieval over section children, fused with RRF.

        Change 9: Also includes the section's own text content as a candidate.
        This is critical for pseudo-sections (text chunks promoted to section
        type) where the section text itself IS the answer, not just a heading.
        """
        candidates = []

        # Always include the section's own content as a candidate
        sec_id = section.get("chunk_id", "")
        if sec_id and sec_id not in seen_ids:
            seen_ids.add(sec_id)
            candidates.append(self._make_candidate(section, 0.0))

        children_ids: List[str] = section.get("children_ids", [])
        if not children_ids:
            return candidates

        children_items = self._fetch_chunks_by_ids(children_ids, with_vectors=True)
        if not children_items:
            return candidates

        payloads = [item["payload"] for item in children_items]
        vectors  = [item["vector"]  for item in children_items]
        texts    = [p.get("content", "") for p in payloads]

        tokenized     = [t.lower().split() for t in texts]
        bm25          = BM25Okapi(tokenized)
        bm25_scores   = bm25.get_scores(query.lower().split())
        sparse_ranked = [payloads[i]["chunk_id"] for i in np.argsort(bm25_scores)[::-1]]

        child_embs   = np.array(vectors, dtype=np.float32)
        q_arr        = np.array(query_emb, dtype=np.float32)
        sims         = child_embs @ q_arr
        dense_ranked = [payloads[i]["chunk_id"] for i in np.argsort(sims)[::-1]]

        fused_ids   = _reciprocal_rank_fusion(sparse_ranked, dense_ranked)
        payload_map = {p["chunk_id"]: p for p in payloads}
        sim_map     = {payloads[i]["chunk_id"]: float(sims[i]) for i in range(len(payloads))}

        for cid in fused_ids[: self.level2_k]:
            if cid in seen_ids:
                continue
            pay = payload_map.get(cid)
            if pay is None:
                continue
            seen_ids.add(cid)
            candidates.append(self._make_candidate(pay, sim_map.get(cid, 0.0)))

        return candidates

    # ── Direct table retrieval (Numerical/Comparison) ────────────────────────

    def _retrieve_tables_direct(
        self,
        query_emb: list,
        source:    Optional[str],
        seen_ids:  set,
    ) -> List[dict]:
        """Directly retrieve the top TABLE chunks for Numerical/Comparison queries."""
        must = [FieldCondition(key="type", match=MatchValue(value="table"))]
        if source:
            must.append(FieldCondition(key="source", match=MatchValue(value=source)))

        results = self.client.query_points(
            collection_name = config.COLLECTION_NAME,
            query           = query_emb,
            query_filter    = Filter(must=must),
            limit           = 20,
            with_payload    = True,
        )

        new_candidates = []
        for r in results.points:
            cid = r.payload.get("chunk_id", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            new_candidates.append(self._make_candidate(r.payload, r.score))
        return new_candidates

    def _retrieve_text_direct(
        self,
        query_emb: list,
        source:    Optional[str],
        seen_ids:  set,
    ) -> List[dict]:
        """Fallback: dense search over text chunks for Lookup/Summarization.

        Used when Level 1 section retrieval returns nothing (e.g. no section
        chunks were created by the parser).  Also retrieves table chunks so
        that Lookup questions about numeric facts still have a chance.
        """
        candidates = []
        for chunk_type in ("text", "table"):
            must = [FieldCondition(key="type", match=MatchValue(value=chunk_type))]
            if source:
                must.append(FieldCondition(key="source", match=MatchValue(value=source)))

            results = self.client.query_points(
                collection_name = config.COLLECTION_NAME,
                query           = query_emb,
                query_filter    = Filter(must=must),
                limit           = 15,
                with_payload    = True,
            )
            for r in results.points:
                cid = r.payload.get("chunk_id", "")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                candidates.append(self._make_candidate(r.payload, r.score))
        return candidates

    # ── Level 3 ──────────────────────────────────────────────────────────────

    def _level3_retrieve(
        self,
        query:       str,
        query_emb:   list,
        table_chunk: dict,
        intent:      str,
        seen_ids:    set,
    ) -> List[dict]:
        """Cell extraction from a table via stored row-vector scoring."""
        pay         = table_chunk["metadata"]
        rows        = pay.get("rows") or []
        col_headers = pay.get("col_headers") or []
        table_id    = pay.get("chunk_id", "")

        if not rows or not col_headers:
            if not hasattr(self, "_all_chunks_cache"):
                import os
                chunks_path = os.path.join("data", "extracted", "chunks.json")
                if os.path.exists(chunks_path):
                    with open(chunks_path, encoding="utf-8") as f:
                        self._all_chunks_cache = json.load(f)
                else:
                    self._all_chunks_cache = []
            for c in self._all_chunks_cache:
                if c.get("chunk_id") == table_id:
                    rows        = c.get("rows") or []
                    col_headers = c.get("col_headers") or []
                    break

        if not rows or not col_headers:
            return []

        row_cell_ids = [f"{table_id}_r{r}_c0" for r in range(len(rows))]
        row_items    = self._fetch_chunks_by_ids(row_cell_ids, with_vectors=True)
        if not row_items:
            return []

        row_vecs: dict = {}
        for item in row_items:
            cid = item["payload"].get("chunk_id", "")
            try:
                r_idx = int(cid.split("_r")[-1].split("_c")[0])
                row_vecs[r_idx] = np.array(item["vector"], dtype=np.float32)
            except (ValueError, IndexError):
                continue

        q_arr = np.array(query_emb, dtype=np.float32)
        row_sims: dict = {r: float(vec @ q_arr) for r, vec in row_vecs.items()}
        top_row_idxs = sorted(row_sims, key=row_sims.get, reverse=True)[:5]
        top_col_idxs = self._select_columns(query, col_headers, intent)

        cell_ids_to_fetch = [
            f"{table_id}_r{r}_c{c}"
            for r in top_row_idxs for c in top_col_idxs
            if f"{table_id}_r{r}_c{c}" not in seen_ids
        ]

        cell_candidates = []
        if cell_ids_to_fetch:
            items      = self._fetch_chunks_by_ids(cell_ids_to_fetch)
            items_dict = {item["payload"]["chunk_id"]: item for item in items}
            for r in top_row_idxs:
                for c in top_col_idxs:
                    cid = f"{table_id}_r{r}_c{c}"
                    if cid in items_dict:
                        seen_ids.add(cid)
                        cell_candidates.append(
                            self._make_candidate(items_dict[cid]["payload"], row_sims.get(r, 0.0))
                        )
        return cell_candidates

    def _select_columns(self, query: str, col_headers: List[str], intent: str) -> List[int]:
        years = re.findall(r"(20\d{2})", query)
        if intent in ("Numerical", "Comparison") and years:
            year_cols = [i for i, h in enumerate(col_headers)
                         if any(year in str(h) for year in years)]
            if year_cols:
                return year_cols
        return list(range(len(col_headers)))

    # ── Fetch helpers ────────────────────────────────────────────────────────

    def _fetch_chunks_by_ids(
        self,
        chunk_ids:    List[str],
        with_vectors: bool = False,
    ) -> List[dict]:
        """Retrieve chunk data from Qdrant by chunk_id string."""
        if not chunk_ids:
            return []
        results, _ = self.client.scroll(
            collection_name = config.COLLECTION_NAME,
            scroll_filter   = Filter(
                should=[
                    FieldCondition(key="chunk_id", match=MatchValue(value=cid))
                    for cid in chunk_ids
                ]
            ),
            limit        = len(chunk_ids),
            with_payload = True,
            with_vectors = with_vectors,
        )
        if with_vectors:
            return [{"payload": r.payload, "vector": r.vector} for r in results]
        return [{"payload": r.payload} for r in results]

    @staticmethod
    def _make_candidate(payload: dict, dense_score: float) -> dict:
        """Wrap a Qdrant payload into the standard chunk format."""
        return {
            "content": payload.get("content", ""),
            "metadata": {
                "type":           payload.get("type", "text"),
                "page":           payload.get("page", 0),
                "source":         payload.get("source", "unknown"),
                "chunk_id":       payload.get("chunk_id", ""),
                "parent_id":      payload.get("parent_id"),
                "parent_section": payload.get("parent_section"),
                "col_header":     payload.get("col_header"),
                "row_header":     payload.get("row_header"),
                "value":          payload.get("value"),
            },
            "dense_score":  round(float(dense_score), 4),
            "rerank_score": 0.0,
        }
