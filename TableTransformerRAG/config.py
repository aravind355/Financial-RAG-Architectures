"""
config.py
=========
Central configuration for the TableTransformerRAG pipeline.

Identical retrieval parameters to HierFinRAG for a fair apples-to-apples
comparison.  The only architectural difference is the parser (TATR vs pdfplumber).

Prerequisites
-------------
1.  Install Ollama:  https://ollama.com
2.  Start the server:  ollama serve
3.  Pull the model:    ollama pull qwen2.5:7b
4.  Vector DB: Qdrant (pip install qdrant-client) — local store at data/qdrant/
5.  TATR models: python scripts/download_models.py
"""

# ---------------------------------------------------------------------------
# Ollama local LLM endpoint
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
OLLAMA_MODEL    = "qwen2.5:7b"

# ---------------------------------------------------------------------------
# Retrieval parameters  ← IDENTICAL to HierFinRAG for fair comparison
# ---------------------------------------------------------------------------
RETRIEVER_LEVEL1_K = 5     # Top-K sections at Level 1
RETRIEVER_LEVEL2_K = 15    # Top-K para/table candidates per section at Level 2
RETRIEVER_TOP_N    = 10    # Final reranked chunks passed to the generator

# ---------------------------------------------------------------------------
# Vector store (Qdrant — local persistent)
# ---------------------------------------------------------------------------
QDRANT_PATH     = "data/qdrant"
COLLECTION_NAME = "tatr_rag"

# ---------------------------------------------------------------------------
# Table Transformer (TATR) settings
# ---------------------------------------------------------------------------
TATR_DETECTION_MODEL  = "microsoft/table-transformer-detection"
TATR_STRUCTURE_MODEL  = "microsoft/table-transformer-structure-recognition-v1.1-all"
# Local paths (populated by scripts/download_models.py)
TATR_DETECTION_LOCAL  = "data/models/detection"
TATR_STRUCTURE_LOCAL  = "data/models/structure"
TATR_DET_THRESHOLD    = 0.85   # Confidence threshold for table detection (raised from 0.7 to reduce false positives)
TATR_STR_THRESHOLD    = 0.6    # Confidence threshold for structure recognition
TATR_RENDER_DPI       = 300    # PDF page render resolution

# ---------------------------------------------------------------------------
# Parser settings
# ---------------------------------------------------------------------------
PARSER_MIN_TEXT_LEN = 40       # Minimum chars for a text chunk to be kept
PARSER_CHUNK_SIZE   = 500      # Target word count for text splitting
