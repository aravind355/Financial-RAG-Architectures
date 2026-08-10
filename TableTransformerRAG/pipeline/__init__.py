"""
pipeline/__init__.py
====================
TableTransformerRAG pipeline package.

Submodules:
    llm_client  — QwenClient (Ollama local LLM)
    parser      — TATR-based PDF parser (table detection + structure recognition)
    embedder    — BGE-M3 embedding + Qdrant vector store builder
    retriever   — HierarchicalRetriever (Level 1-3 + reranking, source-filtered)
    router      — FinancialIntentClassifier
    generator   — Symbolic-Neural Fusion generator
    symbolic    — FinQA DSL arithmetic executor
    attribution — Claim attribution + confidence scoring
"""
