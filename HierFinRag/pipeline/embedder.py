"""
pipeline/embedder.py
====================
Vector store builder for the HierFinRAG pipeline.

Encodes all document chunks produced by parser.py with the BGE-M3 embedding
model and upserts them into a local ChromaDB collection.  The full hierarchical
metadata (parent_id, children_ids, row/column headers, cell values) is stored
directly in the ChromaDB metadata payload so the retriever can reconstruct the
document tree without re-reading the raw JSON file.

Run standalone
--------------
    python -m pipeline.embedder
    python -m pipeline.embedder --chunks path/to/chunks.json
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL     = "BAAI/bge-m3"    # 1024-dimensional multilingual model
EMBED_DIM       = 1024             # Must match ChromaDB embedding dimension
COLLECTION_NAME = "finance_rag"
CHROMA_PATH     = "data/chroma"    # Local persistent storage directory
BATCH_SIZE      = 64               # Chunks per embedding batch


# ---------------------------------------------------------------------------
# Metadata serializer
# ---------------------------------------------------------------------------

def _serialize_metadata(raw: dict) -> dict:
    """Flatten chunk metadata into ChromaDB-compatible scalar/string values.

    ChromaDB metadata values must be str, int, float, or bool — no lists,
    dicts, or None.  Complex fields (children_ids, col_headers, row_headers,
    rows) are JSON-encoded to strings so they survive the round-trip and can
    be decoded by the retriever.
    """
    LIST_FIELDS = ("children_ids", "col_headers", "row_headers", "rows")
    result = {}
    for k, v in raw.items():
        if v is None:
            result[k] = ""           # ChromaDB rejects None
        elif k in LIST_FIELDS:
            result[k] = json.dumps(v) if v else "[]"
        elif isinstance(v, (list, dict)):
            result[k] = json.dumps(v)
        else:
            result[k] = v
    return result


def _deserialize_metadata(meta: dict) -> dict:
    """Reverse _serialize_metadata: JSON-decode list fields back to Python objects."""
    LIST_FIELDS = ("children_ids", "col_headers", "row_headers", "rows")
    result = {}
    for k, v in meta.items():
        if k in LIST_FIELDS and isinstance(v, str):
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
# Main builder
# ---------------------------------------------------------------------------

def build_vector_store(
    chunks_path:       str  = "data/extracted/chunks.json",
    return_embeddings: bool = False,
) -> int | tuple:
    """Embed all chunks and upsert them into the ChromaDB collection.

    The function performs a full rebuild on every call: the existing collection
    is deleted before inserting fresh vectors.  This keeps the store
    consistent with the latest parse output.

    Args:
        chunks_path       : Path to the JSON file produced by parser.py.
        return_embeddings : If True, also return a ``{chunk_id: np.ndarray}``
                            mapping used by GraphBuilder to construct the
                            TTGNN graph edges.

    Returns:
        Number of points stored, or a ``(num_points, embeddings_dict)`` tuple
        when ``return_embeddings=True``.
    """
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Image chunks have no text content — skip them until vision captioning
    # is integrated.
    valid_chunks = [c for c in all_chunks if c.get("content", "").strip()]
    skipped      = len(all_chunks) - len(valid_chunks)
    print(f"Loaded {len(all_chunks)} chunks — skipping {skipped} empty (image) chunks.")
    print(f"Embedding {len(valid_chunks)} chunks with {EMBED_MODEL}.\n")

    model = SentenceTransformer(EMBED_MODEL)

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Full rebuild: delete existing collection so the store matches the latest parse
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection — rebuilding from scratch.")

    # ChromaDB uses cosine distance internally when embeddings are L2-normalised.
    # We pass embeddings manually (not via a ChromaDB embedding function) since
    # BGE-M3 is run locally via sentence-transformers.
    collection = client.create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
    )

    total_inserted  = 0
    embeddings_dict = {}  # chunk_id → np.ndarray; returned when requested

    for i in tqdm(range(0, len(valid_chunks), BATCH_SIZE), desc="Embedding"):
        batch = valid_chunks[i : i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        # Use chunk_id directly as the ChromaDB document ID (must be unique str)
        ids = [c["chunk_id"] for c in batch]

        # Build and serialize metadata payload
        raw_payloads = [
            {
                # Core identification
                "chunk_id":       c["chunk_id"],
                "type":           c["type"],
                "page":           c["page"],
                "source":         c["source"],
                "content":        c["content"],
                # Hierarchy links (used by Level-1 → Level-2 traversal)
                "parent_id":      c.get("parent_id"),
                "children_ids":   c.get("children_ids", []),
                "parent_section": c.get("parent_section"),
                # Table-specific metadata (None for non-table chunks)
                "col_headers":    c.get("col_headers"),
                "row_headers":    c.get("row_headers"),
                "rows":           c.get("rows"),
                # Cell-specific metadata (None for non-cell chunks)
                "row_idx":        c.get("row_idx"),
                "col_idx":        c.get("col_idx"),
                "row_header":     c.get("row_header"),
                "col_header":     c.get("col_header"),
                "value":          c.get("value"),
            }
            for c in batch
        ]
        metadatas = [_serialize_metadata(p) for p in raw_payloads]

        # L2-normalised vectors → cosine similarity = dot product at query time
        embeddings = model.encode(
            texts,
            normalize_embeddings = True,
            show_progress_bar    = False,
            batch_size           = BATCH_SIZE,
        )

        for chunk, emb in zip(batch, embeddings):
            embeddings_dict[chunk["chunk_id"]] = emb

        collection.add(
            ids        = ids,
            embeddings = [emb.tolist() for emb in embeddings],
            documents  = texts,
            metadatas  = metadatas,
        )
        total_inserted += len(batch)

    count = collection.count()
    print(f"\nDone — {total_inserted} chunks stored in ChromaDB at '{CHROMA_PATH}'.")
    print(f"Collection '{COLLECTION_NAME}': {count} points.")

    if return_embeddings:
        return count, embeddings_dict
    return count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB vector store from HierFinRAG chunks."
    )
    parser.add_argument(
        "--chunks",
        default="data/extracted/chunks.json",
        help="Path to the chunks JSON file produced by parser.py "
             "(default: data/extracted/chunks.json).",
    )
    args = parser.parse_args()
    build_vector_store(args.chunks)