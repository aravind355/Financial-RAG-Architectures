"""
main.py
=======
HierFinRAG — interactive financial document Q&A assistant.

Runs the complete pipeline for every user query:

    Step 1  Intent classification  — FinancialIntentClassifier routes the query
                                     to the correct reasoning mode.
    Step 2  Hierarchical retrieval — HierarchicalRetriever runs three levels of
                                     retrieval (Section → Para/Table → Cell),
                                     source-filtered to the correct PDF.
    Step 3  Answer generation      — Symbolic-Neural Fusion generator produces
                                     the answer (arithmetic DSL or pure LLM).
    Step 4  Attribution            — Each claim is mapped to supporting chunks
                                     and a calibrated confidence score is computed.

Usage
-----
    python main.py
"""

import config

from pipeline.llm_client  import get_qwen_client
from pipeline.retriever   import HierarchicalRetriever
from pipeline.router      import FinancialIntentClassifier
from pipeline.attribution import generate_with_attribution


# ---------------------------------------------------------------------------
# Single-query pipeline
# ---------------------------------------------------------------------------

def ask(
    query:     str,
    retriever: HierarchicalRetriever,
    router:    FinancialIntentClassifier,
    llm,
) -> dict:
    """Run the full HierFinRAG pipeline for a single financial question.

    Args:
        query     : The user's natural-language financial question.
        retriever : An initialized HierarchicalRetriever instance.
        router    : An initialized FinancialIntentClassifier instance.
        llm       : An initialized QwenClient instance.

    Returns:
        Attribution dict containing:
            answer_text, supporting_cells, supporting_sentences,
            reasoning_steps, confidence, attribution_map.
    """
    print(f"\n{'=' * 65}")
    print(f"  Q: {query}")
    print("=" * 65)

    # Step 1 — Classify intent
    intent = router.classify(query)
    print(f"\n  Intent  : {intent}")

    # Step 2 — Hierarchical retrieval (no source filter in interactive mode;
    #           the user may ask about either document in the collection)
    chunks = retriever.retrieve(query, intent=intent)

    print(f"\n  Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        print(
            f"    [{i+1}] Page {meta['page']:>3} | {meta['type']:7} | "
            f"dense={c['dense_score']:.3f}  rerank={c['rerank_score']:.3f}"
        )
        print(f"         {c['content'][:90].strip()}...")

    # Step 3+4 — Generate answer with attribution
    print("\n  Generating answer...\n")
    result = generate_with_attribution(
        query            = query,
        retrieved_chunks = chunks,
        intent           = intent,
        llm              = llm,
    )

    # Display results
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


# ---------------------------------------------------------------------------
# Interactive CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize all pipeline components once; they are reused across queries.
    # Retrieval parameters are read from config.py for centralised tuning.
    llm       = get_qwen_client()
    retriever = HierarchicalRetriever(
        level1_k = config.RETRIEVER_LEVEL1_K,
        level2_k = config.RETRIEVER_LEVEL2_K,
        top_n    = config.RETRIEVER_TOP_N,
    )
    router    = FinancialIntentClassifier(llm=llm)

    print("\n" + "=" * 65)
    print("  HierFinRAG — Hierarchical Financial Document Q&A")
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