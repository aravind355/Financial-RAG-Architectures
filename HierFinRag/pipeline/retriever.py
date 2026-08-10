"""
pipeline/retriever.py
=====================
Three-level hierarchical retrieval pipeline (ChromaDB backend).

Level 1: Section-level dense retrieval via BGE-M3 cosine similarity.
Level 2: Hybrid BM25 + dense retrieval over section children, fused with RRF.
Level 3: Table-aware row/column selection returning individual cell chunks.
Post:    Cross-encoder reranking of all candidates before returning top-N.

The retriever accepts an optional `source` filter so that queries about
a specific company (e.g. Apple) only search that company's indexed chunks,
preventing cross-document contamination when multiple PDFs share a collection.
"""

import json
import re
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CHROMA_PATH  = "data/chroma"
COLLECTION   = "finance_rag"

RRF_K = 60  # Reciprocal Rank Fusion smoothing constant (original RRF paper)


# ---------------------------------------------------------------------------
# Metadata deserializer (mirrors embedder._deserialize_metadata)
# ---------------------------------------------------------------------------

_LIST_FIELDS = ("children_ids", "col_headers", "row_headers", "rows")


def _deser(meta: dict) -> dict:
    """Deserialize ChromaDB metadata: JSON-decode list fields, convert '' → None."""
    result = {}
    for k, v in meta.items():
        if k in _LIST_FIELDS and isinstance(v, str):
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                result[k] = []
        elif v == "":
            result[k] = None
        else:
            result[k] = v
    return result


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
    """Expand a Numerical/Comparison query with year tokens and financial
    table keywords to shift its embedding closer to number-heavy table chunks.

    Example::
        Input:  'What is the % change in net sales from 2022 to 2023?'
        Output: 'What is the % change in net sales from 2022 to 2023?
                 2022 2023 net sales table revenue income statements'
    """
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
    """Merge two ranked chunk-id lists using Reciprocal Rank Fusion.
    score(d) = Σ 1 / (k + rank_i(d)) across all ranked lists."""
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

    Args:
        level1_k : Number of sections retrieved at Level 1.
        level2_k : Number of para/table candidates per section at Level 2.
        top_n    : Final number of reranked chunks returned to the generator.
    """

    def __init__(self, level1_k: int = 5, level2_k: int = 10, top_n: int = 5):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print("Loading cross-encoder reranker...")
        self.reranker   = CrossEncoder(RERANK_MODEL)
        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(name=COLLECTION)
        count = self.collection.count()
        print(f"Connected to ChromaDB — {count} chunks\n")
        self.level1_k = level1_k
        self.level2_k = level2_k
        self.top_n    = top_n

    def retrieve(
        self,
        query:  str,
        intent: str = "Lookup",
        source: Optional[str] = None,
    ) -> List[dict]:
        """Run the full hierarchical retrieval pipeline for a query.

        Args:
            query  : Natural-language financial question.
            intent : Classified intent ('Numerical'|'Comparison'|'Lookup'|'Summarization').
            source : PDF filename to restrict retrieval to (e.g. 'apple_2023.pdf').
                     When set, only chunks from that source are searched, preventing
                     cross-document contamination in a multi-PDF collection.

        Returns:
            List of top-N chunk dicts with content, metadata, dense_score, rerank_score.
        """
        query_str = query
        # Fix 2: Expand Numerical/Comparison queries before embedding so the
        # resulting vector sits closer to number-heavy table/cell chunks.
        effective_level1_k = self.level1_k
        if intent in ("Numerical", "Comparison"):
            query_str = _expand_numerical_query(query)
            # Fix 3: Search more sections for Numerical/Comparison queries
            # because two operands often live in different sections.
            effective_level1_k = min(int(self.level1_k * 1.6) + 1, 12)

        query_emb = self.embedder.encode(query_str, normalize_embeddings=True).tolist()

        top_sections = self._level1_retrieve(
            query_emb, source=source, limit=effective_level1_k
        )

        candidates: List[dict] = []
        seen_ids: set          = set()

        for sec in top_sections:
            candidates.extend(self._level2_retrieve(query, query_emb, sec, seen_ids))

        # For Numerical and Comparison queries, text description chunks tend to
        # outrank table chunks in Level 2, leaving the actual financial tables
        # below the top-K cutoff. This direct search guarantees the most
        # relevant TABLE chunks are included so Level 3 can extract cells.
        if intent in ("Numerical", "Comparison"):
            candidates.extend(
                self._retrieve_tables_direct(query_emb, source, seen_ids)
            )

        for c in list(candidates):
            if c["metadata"]["type"] == "table":
                candidates.extend(self._level3_retrieve(query, query_emb, c, intent, seen_ids))

        if not candidates:
            return []

        # Separate cells from other candidates
        cell_cands  = [c for c in candidates if c["metadata"]["type"] == "cell"]
        other_cands = [c for c in candidates if c["metadata"]["type"] != "cell"]

        # Sort cells by their Bi-Encoder score and keep top-5
        cell_cands.sort(key=lambda x: x.get("dense_score", 0.0), reverse=True)
        top_cells = cell_cands[:5]

        # Rerank other candidates using Cross-Encoder
        if other_cands:
            pairs  = [[query, c["content"]] for c in other_cands]
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                other_cands[i]["rerank_score"] = round(float(score), 4)

        # Boost the top cells so they always appear in the top-N LLM prompt
        for c in top_cells:
            c["rerank_score"] = c.get("rerank_score", 0.0) + 2.0

        candidates = other_cands + top_cells
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[: self.top_n]

    # ── Level 1: dense section retrieval ────────────────────────────────────

    def _level1_retrieve(
        self,
        query_emb: list,
        source:    Optional[str] = None,
        limit:     Optional[int] = None,
    ) -> List[dict]:
        """Retrieve top-K section chunks via dense similarity.
        When source is set, retrieval is restricted to that PDF filename.
        The limit parameter overrides self.level1_k (used for adaptive K).
        """
        k = limit if limit is not None else self.level1_k

        # Build ChromaDB where filter
        where: dict = {"type": {"$eq": "section"}}
        if source:
            where = {"$and": [{"type": {"$eq": "section"}}, {"source": {"$eq": source}}]}

        # Cap n_results to actual collection size to prevent ChromaDB error
        total = self.collection.count()
        results = self.collection.query(
            query_embeddings = [query_emb],
            n_results        = min(k, max(1, total)),
            where            = where,
            include          = ["metadatas", "documents", "distances"],
        )

        payloads = []
        for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
            p = _deser(meta)
            p.setdefault("content", doc)
            payloads.append(p)
        return payloads

    # ── Level 2: hybrid BM25 + dense over section children ──────────────────

    def _level2_retrieve(
        self,
        query:     str,
        query_emb: list,
        section:   dict,
        seen_ids:  set,
    ) -> List[dict]:
        """Hybrid BM25 + dense retrieval over a section's direct children,
        fused with Reciprocal Rank Fusion.

        Dense similarity is computed using the pre-stored BGE-M3 vectors fetched
        directly from ChromaDB — no model inference at retrieval time.
        """
        children_ids: List[str] = section.get("children_ids", [])
        if not children_ids:
            return []

        # Fetch payload AND the pre-stored embedding vector for each child.
        children_items = self._fetch_chunks_by_ids(children_ids, with_vectors=True)
        if not children_items:
            return []

        payloads = [item["payload"] for item in children_items]
        vectors  = [item["vector"]  for item in children_items]
        texts    = [p.get("content", "") for p in payloads]

        # Sparse ranking — BM25 over child text (no model needed)
        tokenized     = [t.lower().split() for t in texts]
        bm25          = BM25Okapi(tokenized)
        bm25_scores   = bm25.get_scores(query.lower().split())
        sparse_ranked = [payloads[i]["chunk_id"] for i in np.argsort(bm25_scores)[::-1]]

        # Dense ranking — cosine similarity using stored vectors.
        # Convert each vector to a plain Python list first to handle both
        # numpy ndarrays and lists returned by ChromaDB get().
        child_embs   = np.array([list(v) for v in vectors], dtype=np.float32)
        q_arr        = np.array(query_emb, dtype=np.float32)
        sims         = child_embs @ q_arr
        dense_ranked = [payloads[i]["chunk_id"] for i in np.argsort(sims)[::-1]]

        fused_ids   = _reciprocal_rank_fusion(sparse_ranked, dense_ranked)
        payload_map = {p["chunk_id"]: p for p in payloads}
        sim_map     = {payloads[i]["chunk_id"]: float(sims[i]) for i in range(len(payloads))}
        candidates  = []

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
        """Directly retrieve the top-8 TABLE chunks most similar to the query.

        Used for Numerical and Comparison queries where financial statement
        tables must be retrieved even when text descriptions outrank them in
        the standard Level 2 hybrid search.
        """
        where: dict = {"type": {"$eq": "table"}}
        if source:
            where = {"$and": [{"type": {"$eq": "table"}}, {"source": {"$eq": source}}]}

        # Cap n_results to actual number of table chunks to avoid ChromaDB error
        table_count = self.collection.count()
        results = self.collection.query(
            query_embeddings = [query_emb],
            n_results        = min(100, max(1, table_count)),
            where            = where,
            include          = ["metadatas", "documents", "distances"],
        )

        new_candidates = []
        for meta, doc, dist in zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0],
        ):
            p = _deser(meta)
            p.setdefault("content", doc)
            cid = p.get("chunk_id", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            # ChromaDB returns cosine distance (0 = identical); convert to similarity
            score = 1.0 - dist
            new_candidates.append(self._make_candidate(p, score))
        return new_candidates

    # ── Level 3: cell extraction ─────────────────────────────────────────────

    def _level3_retrieve(
        self,
        query:       str,
        query_emb:   list,
        table_chunk: dict,
        intent:      str,
        seen_ids:    set,
    ) -> List[dict]:
        """Select the most relevant rows and columns from a table and return
        their intersection as individual cell chunks.

        Row relevance is scored using the pre-stored cell vectors fetched from
        ChromaDB — no model inference at retrieval time.
        """
        pay         = table_chunk["metadata"]
        rows        = pay.get("rows") or []
        col_headers = pay.get("col_headers") or []
        row_headers = pay.get("row_headers") or []
        table_id    = pay.get("chunk_id", "")

        # Fallback: ChromaDB payload sometimes loses multi-dimensional arrays.
        # If they are missing, fetch them directly from the source JSON.
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
                    row_headers = c.get("row_headers") or []
                    break

        if not rows or not col_headers:
            return []

        # Build the cell IDs for the first cell of each row (column 0).
        # Fetch their pre-stored vectors to score row relevance without
        # re-encoding the row texts with BGE-M3.
        row_cell_ids = [f"{table_id}_r{r}_c0" for r in range(len(rows))]
        row_items    = self._fetch_chunks_by_ids(row_cell_ids, with_vectors=True)

        if not row_items:
            return []

        # Map row index back from chunk_id (format: <table_id>_r<r>_c0)
        row_vecs: dict[int, np.ndarray] = {}
        for item in row_items:
            cid = item["payload"].get("chunk_id", "")
            try:
                r_idx = int(cid.split("_r")[-1].split("_c")[0])
                row_vecs[r_idx] = np.array(item["vector"], dtype=np.float32)
            except (ValueError, IndexError):
                continue

        q_arr = np.array(query_emb, dtype=np.float32)
        row_sims: dict[int, float] = {
            r: float(vec @ q_arr) for r, vec in row_vecs.items()
        }

        # Select top-5 rows by cosine similarity
        top_row_idxs = sorted(row_sims, key=row_sims.get, reverse=True)[:5]
        top_col_idxs = self._select_columns(query, col_headers, intent)

        cell_ids_to_fetch = []
        for r_idx in top_row_idxs:
            for c_idx in top_col_idxs:
                cell_id = f"{table_id}_r{r_idx}_c{c_idx}"
                if cell_id not in seen_ids:
                    cell_ids_to_fetch.append(cell_id)

        cell_candidates = []
        if cell_ids_to_fetch:
            items      = self._fetch_chunks_by_ids(cell_ids_to_fetch)
            items_dict = {item["payload"]["chunk_id"]: item for item in items}

            for r_idx in top_row_idxs:
                for c_idx in top_col_idxs:
                    cell_id = f"{table_id}_r{r_idx}_c{c_idx}"
                    if cell_id in items_dict:
                        seen_ids.add(cell_id)
                        cell_candidates.append(
                            self._make_candidate(
                                items_dict[cell_id]["payload"],
                                row_sims.get(r_idx, 0.0),
                            )
                        )

        return cell_candidates

    # ── Column selection ────────────────────────────────────────────────────

    def _select_columns(self, query: str, col_headers: List[str], intent: str) -> List[int]:
        """Select relevant column indices based on intent and year mentions.

        All years mentioned in the query are extracted (not just the first one).
        This is critical for percentage-change questions that reference two years
        (e.g. "from 2022 to 2023") — both year columns must be returned so the
        LLM receives values for both periods and can compute the change correctly.

        Numerical   → columns matching any year mentioned in the query.
        Comparison  → all year columns found in headers.
        Otherwise   → all columns.
        """
        years = re.findall(r"(20\d{2})", query)

        if intent in ("Numerical", "Comparison") and years:
            year_cols = [
                i for i, h in enumerate(col_headers)
                if any(year in str(h) for year in years)
            ]
            if year_cols:
                return year_cols

        return list(range(len(col_headers)))

    # ── Fetch by chunk_id ────────────────────────────────────────────────────

    def _fetch_chunks_by_ids(
        self,
        chunk_ids:    List[str],
        with_vectors: bool = False,
    ) -> List[dict]:
        """Retrieve chunk data from ChromaDB by chunk_id.

        Args:
            chunk_ids    : List of chunk_id strings to look up.
            with_vectors : When True, also return the stored embedding vector
                           so callers can compute cosine similarity without
                           running BGE-M3 again.

        Returns:
            List of dicts.  Each dict always has a ``'payload'`` key.
            When ``with_vectors=True``, each dict also has a ``'vector'`` key
            containing the pre-stored 1024-dim float list.
        """
        if not chunk_ids:
            return []

        include = ["metadatas", "documents", "embeddings"] if with_vectors else ["metadatas", "documents"]

        results = self.collection.get(
            ids     = chunk_ids,
            include = include,
        )

        items = []
        metas    = results.get("metadatas") or []
        docs     = results.get("documents") or []
        embs_raw = results.get("embeddings")
        # ChromaDB 1.5+ may return embeddings as a numpy ndarray.
        # Using `or` on a numpy array raises 'truth value is ambiguous'.
        # Use an explicit None / length check instead.
        if embs_raw is None or (hasattr(embs_raw, '__len__') and len(embs_raw) == 0):
            embs = [None] * len(metas)
        else:
            embs = embs_raw

        for meta, doc, emb in zip(metas, docs, embs):
            p = _deser(meta)
            p.setdefault("content", doc)
            entry = {"payload": p}
            if with_vectors and emb is not None:
                entry["vector"] = emb
            items.append(entry)
        return items

    # ── Candidate wrapper ───────────────────────────────────────────────────

    @staticmethod
    def _make_candidate(payload: dict, dense_score: float) -> dict:
        """Wrap a payload dict into the standard chunk format used by the generator."""
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