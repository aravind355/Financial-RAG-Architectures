"""
BasicRAG Interactive CLI Application
=====================================
Interactive command-line query interface for the BasicRAG baseline.
Retrieves top-N context chunks from Qdrant and generates answers using local Ollama.

Usage:
    python main.py
"""

from pipeline.retriever import Retriever
from pipeline.generator import generate

def ask(query: str, retriever: Retriever) -> tuple[str, list]:
    """Execute end-to-end question answering pipeline for a single query.

    Args:
        query (str): User natural language question.
        retriever (Retriever): Initialized Retriever instance.

    Returns:
        tuple[str, list]: (Generated answer text, List of retrieved candidate chunks).
    """
    print(f"\n{'='*65}")
    print(f"  Q: {query}")
    print('='*65)

    chunks = retriever.retrieve(query, top_k=20, top_n=5)

    print("\nRetrieved chunks:")
    for i, c in enumerate(chunks):
        m = c["metadata"]
        print(f"  [{i+1}] Page {m['page']:>3} | {m['type']:5} | "
              f"dense={c['dense_score']:.3f} | rerank={c['rerank_score']:.3f}")
        print(f"       {c['content'][:90].strip()}...")

    print("\nGenerating answer...\n")
    answer = generate(query, chunks)
    print(f"Answer:\n{answer}\n")
    return answer, chunks

if __name__ == "__main__":
    retriever = Retriever()

    print("\n" + "="*65)
    print("  Financial RAG Assistant")
    print("  Type your question below. Type 'exit' or 'quit' to close.")
    print("="*65)

    while True:
        try:
            query = input("\n>> ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            ask(query, retriever)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break