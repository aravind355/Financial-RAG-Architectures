"""
main.py
=======
TableTransformerRAG — interactive financial document Q&A assistant.

Identical pipeline flow to HierFinRAG; only the parser changes.

Usage
-----
    python main.py
"""

import config

from pipeline.llm_client  import get_qwen_client
from pipeline.retriever   import HierarchicalRetriever
from pipeline.router      import FinancialIntentClassifier
from pipeline.attribution import generate_with_attribution


def ask(
    query: str,
    retriever: HierarchicalRetriever,
    router: FinancialIntentClassifier,
    llm
) -> dict:
    """Run full TableTransformerRAG pipeline for a single query.

    Args:
        query (str): User question.
        retriever (HierarchicalRetriever): Retriever instance.
        router (FinancialIntentClassifier): Intent classifier instance.
        llm: LLM client instance.

    Returns:
        dict: Attribution result containing answer text, supporting evidence, and confidence score.
    """
    print(f"\n{'=' * 65}")
    print(f"  Q: {query}")
    print("=" * 65)

    intent = router.classify(query)
    print(f"\n  Intent  : {intent}")

    chunks = retriever.retrieve(query, intent=intent)

    print(f"\n  Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        print(
            f"    [{i+1}] Page {meta['page']:>3} | {meta['type']:7} | "
            f"dense={c['dense_score']:.3f}  rerank={c['rerank_score']:.3f}"
        )
        print(f"         {c['content'][:90].strip()}...")

    print("\n  Generating answer...\n")
    result = generate_with_attribution(
        query            = query,
        retrieved_chunks = chunks,
        intent           = intent,
        llm              = llm,
    )

    print(f"Answer:\n{result['answer_text']}\n")
    print(f"  Confidence : {result['confidence']:.2%}")

    if result["supporting_cells"]:
        print("\n  Supporting Table Cells:")
        for cell in result["supporting_cells"][:5]:
            print(f"    • {cell['row_header']} | {cell['col_header']} = {cell['value']}")

    if result["supporting_sentences"]:
        print("\n  Supporting Text:")
        for sent in result["supporting_sentences"][:3]:
            print(f"    • [p{sent['page']}] {sent['content_snippet'][:100]}...")

    return result


if __name__ == "__main__":
    llm       = get_qwen_client()
    retriever = HierarchicalRetriever(
        level1_k = config.RETRIEVER_LEVEL1_K,
        level2_k = config.RETRIEVER_LEVEL2_K,
        top_n    = config.RETRIEVER_TOP_N,
    )
    router = FinancialIntentClassifier(llm=llm)

    print("\n" + "=" * 65)
    print("  TableTransformerRAG — Financial Document Q&A")
    print("  (TATR visual table extraction)")
    print("  Type your question. Type 'exit' or 'quit' to stop.")
    print("=" * 65)

    while True:
        try:
            query = input("\n>> ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break
            ask(query, retriever, router, llm)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
