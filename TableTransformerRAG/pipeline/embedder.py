"""
pipeline/embedder.py
====================
Vector store builder for the TableTransformerRAG pipeline (Qdrant backend).

Identical to HierFinRAG's original embedder — encodes all chunks with BGE-M3
and upserts them into a local Qdrant collection.

Run standalone
--------------
    python -m pipeline.embedder
    python -m pipeline.embedder --chunks path/to/chunks.json
"""

import argparse
import json
import os
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config


EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM   = 1024
BATCH_SIZE  = 64


def build_vector_store(
    chunks_path:       str  = "data/extracted/chunks.json",
    return_embeddings: bool = False,
) -> int | tuple:
    """Embed all chunks and upsert them into the Qdrant collection.

    Performs a full rebuild on every call (delete + recreate) to keep the
    store consistent with the latest parse output.

    Returns:
        Number of points stored, or (num_points, embeddings_dict) when
        return_embeddings=True.
    """
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)

    valid_chunks = [c for c in all_chunks if c.get("content", "").strip()]
    skipped      = len(all_chunks) - len(valid_chunks)
    print(f"Loaded {len(all_chunks)} chunks — skipping {skipped} empty chunks.")
    print(f"Embedding {len(valid_chunks)} chunks with {EMBED_MODEL}.\n")

    model = SentenceTransformer(EMBED_MODEL)

    os.makedirs(config.QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=config.QDRANT_PATH)

    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)
        print("Deleted existing collection — rebuilding from scratch.")

    client.create_collection(
        collection_name = config.COLLECTION_NAME,
        vectors_config  = VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    total_inserted  = 0
    embeddings_dict = {}

    for i in tqdm(range(0, len(valid_chunks), BATCH_SIZE), desc="Embedding"):
        batch = valid_chunks[i : i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])) for c in batch]

        payloads = [
            {
                "chunk_id":       c["chunk_id"],
                "type":           c["type"],
                "page":           c["page"],
                "source":         c["source"],
                "content":        c["content"],
                "parent_id":      c.get("parent_id"),
                "children_ids":   c.get("children_ids", []),
                "parent_section": c.get("parent_section"),
                "col_headers":    c.get("col_headers"),
                "row_headers":    c.get("row_headers"),
                "rows":           c.get("rows"),
                "row_idx":        c.get("row_idx"),
                "col_idx":        c.get("col_idx"),
                "row_header":     c.get("row_header"),
                "col_header":     c.get("col_header"),
                "value":          c.get("value"),
            }
            for c in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings = True,
            show_progress_bar    = False,
            batch_size           = BATCH_SIZE,
        )

        for chunk, emb in zip(batch, embeddings):
            embeddings_dict[chunk["chunk_id"]] = emb

        points = [
            PointStruct(id=pid, vector=emb.tolist(), payload=pay)
            for pid, emb, pay in zip(ids, embeddings, payloads)
        ]
        client.upsert(collection_name=config.COLLECTION_NAME, points=points)
        total_inserted += len(batch)

    info = client.get_collection(config.COLLECTION_NAME)
    print(f"\nDone — {total_inserted} chunks stored in Qdrant at '{config.QDRANT_PATH}'.")
    print(f"Collection '{config.COLLECTION_NAME}': {info.points_count} points.")

    if return_embeddings:
        return info.points_count, embeddings_dict
    return info.points_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Qdrant vector store from TableTransformerRAG chunks."
    )
    parser.add_argument("--chunks", default="data/extracted/chunks.json",
                        help="Path to chunks JSON (default: data/extracted/chunks.json)")
    args = parser.parse_args()
    build_vector_store(args.chunks)
