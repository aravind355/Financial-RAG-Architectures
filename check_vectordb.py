from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

model  = SentenceTransformer("BAAI/bge-m3")
client = QdrantClient(path="data/qdrant")
COLLECTION = "finance_rag"

info = client.get_collection(COLLECTION)
print(f"Total vectors in DB: {info.points_count}\n")

# Test 3 different queries
queries = [
    "What was Apple's total revenue in 2023?",
    "What are the main risk factors for the business?",
    "capital expenditure and investments",
]

for query in queries:
    embedding = model.encode(query, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=COLLECTION,
        query=embedding,
        limit=3,
        with_payload=True
    ).points

    print(f"Query: {query}")
    print("-" * 55)
    for scored_point in results:
        payload = scored_point.payload
        doc = payload.get("content", "")
        print(f"  [{payload.get('type', 'N/A'):5s}] p{str(payload.get('page', '?')):>3s}  score={scored_point.score:.3f}")
        print(f"  {doc[:120].strip()}")
        print()
    print()