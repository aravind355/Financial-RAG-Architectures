"""
pipeline/__init__.py
====================
HierFinRAG pipeline package.

Submodules:
    llm_client  — QwenClient (Ollama local LLM)
    parser      — Hierarchical PDF → chunk tree parser
    embedder    — BGE-M3 embedding + Qdrant vector store builder
    graph       — GraphBuilder + TTGNN (PyG GATv2)
    retriever   — HierarchicalRetriever (Level 1–3 + reranking, source-filtered)
    router      — FinancialIntentClassifier
    generator   — Symbolic-Neural Fusion generator
    symbolic    — FinQA DSL arithmetic executor
    attribution — Claim attribution + confidence scoring
"""
