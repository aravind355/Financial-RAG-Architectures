"""
BasicRAG Dense Embedder & Qdrant Store Builder
==============================================
Encodes extracted flat text and table chunks into 1024-dimensional dense vectors
using BAAI/bge-m3, creating a local persistent Qdrant collection.
"""

import json
import os
import uuid
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "finance_rag"
QDRANT_PATH = "data/qdrant"

def build_vector_store(chunks_path: str = "data/extracted/chunks.json") -> int:
    """Load parsed JSON chunks, embed with BGE-M3, and persist to local Qdrant vector DB.

    Args:
        chunks_path (str): Path to JSON file containing extracted chunks.

    Returns:
        int: Total number of points stored in Qdrant vector collection.
    """
    # Load chunks
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # Drop image chunks with no caption yet (empty content)
    valid_chunks = [c for c in chunks if c.get("content", "").strip()]
    skipped = len(chunks) - len(valid_chunks)
    print(f"Loaded {len(chunks)} chunks, skipping {skipped} with empty content")
    print(f"Embedding {len(valid_chunks)} chunks with model: {EMBED_MODEL}\n")

    # Load embedding model (downloads once, cached after)
    model = SentenceTransformer(EMBED_MODEL)

    # Setup Qdrant local persistent storage
    os.makedirs(QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    # Fresh start — delete old collection if exists
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print("Deleted old collection, rebuilding...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    # Embed and insert in batches of 64
    batch_size = 64
    total_inserted = 0

    for i in tqdm(range(0, len(valid_chunks), batch_size), desc="Embedding"):
        batch = valid_chunks[i:i + batch_size]

        texts     = [c["content"] for c in batch]
        # Generate deterministic UUIDs from chunk_id strings
        ids       = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])) for c in batch]
        payloads = [{
            "type":     c["type"],
            "page":     c["page"],
            "source":   c["source"],
            "chunk_id": c["chunk_id"],
            "content":  c["content"]  # Qdrant stores document text in payload
        } for c in batch]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        points = [
            PointStruct(id=p_id, vector=emb, payload=pay)
            for p_id, emb, pay in zip(ids, embeddings, payloads)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        total_inserted += len(batch)

    print(f"\nDone. {total_inserted} chunks stored in Qdrant at '{QDRANT_PATH}'")
    # Return collection count
    info = client.get_collection(COLLECTION_NAME)
    return info.points_count


if __name__ == "__main__":
    build_vector_store()