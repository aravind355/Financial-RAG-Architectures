"""
config.py
=========
Central configuration for the HierFinRAG pipeline.

All pipeline components import their settings from here.  Change the model
name or endpoint once — every module picks up the new value automatically.

Prerequisites
-------------
1.  Install Ollama:  https://ollama.com
2.  Start the server:  ollama serve
3.  Pull the model:    ollama pull qwen2.5:7b
4.  Vector DB: ChromaDB (pip install chromadb) — local persistent store at data/chroma/
"""

# ---------------------------------------------------------------------------
# Ollama local LLM endpoint
# ---------------------------------------------------------------------------
# Ollama exposes an OpenAI-compatible REST API so the pipeline can use the
# standard openai-python client without any modification.
# No API key or internet connection is required.

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
OLLAMA_MODEL = "qwen2.5:7b"

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
# Increasing these values improves recall at the cost of slightly more
# LLM context tokens.  Recommended values for a 7B model:

RETRIEVER_LEVEL1_K = 5    # Top-K sections retrieved at Level 1
RETRIEVER_LEVEL2_K = 15   # Top-K para/table candidates per section at Level 2
RETRIEVER_TOP_N    = 10   # Final reranked chunks passed to the generator

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

CHROMA_PATH      = "data/chroma"    # Local ChromaDB persistent storage
COLLECTION_NAME  = "finance_rag"    # ChromaDB collection name
