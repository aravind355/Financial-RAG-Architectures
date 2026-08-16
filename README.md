# Financial Document Intelligence RAG Systems
## IITH Summer Internship 2026 · IIT Hyderabad & University of Hyderabad

This repository contains the complete implementation, evaluation framework, and academic research paper for three Retrieval-Augmented Generation (RAG) architectures evaluated on complex financial document question-answering over corporate 10-K filings (Apple 2023 and Alphabet 2025 Form 10-Ks).

---

## 📌 Repository Structure

```
rag/
├── Basic_Rag/              # Baseline single-stage flat RAG (BGE-M3 + Qdrant)
├── HierFinRag/             # 3-level hierarchical RAG (RRF + Symbolic DSL)
├── TableTransformerRAG/    # DETR-based visual table RAG (Microsoft TATR)
├── ARCHITECTURE.md         # Detailed pipeline architecture & dataflow diagrams
├── IITH_paper.tex          # IEEE conference standard research paper (LaTeX)
├── IITH_paper.pdf          # Compiled 6-page IEEE research paper PDF
└── requirements.txt        # Unified Python dependencies file
```

---

## 🚀 Systems Overview

| System | Table Parsing | Hierarchy | Arithmetic | Execution Acc. (EM) | Recall@10 |
|---|---|---|---|---|---|
| **BasicRAG** | Flat `pdfplumber` | Single-stage | LLM direct | **3.0%** | 80.3% |
| **HierFinRAG** | Structural `pdfplumber` | 4-level tree | Symbolic DSL | **40.0%** | **93.4%** |
| **TableTransformerRAG** | DETR Vision (TATR) | 4-level tree | Symbolic DSL | **48.2%** | 89.7% |

---

## 🏗️ Pipeline Architecture Diagrams

Detailed visual data flow and component architecture diagrams for all three systems are documented in **[ARCHITECTURE.md](ARCHITECTURE.md)**:
- **[BasicRAG Pipeline Workflow](ARCHITECTURE.md#1-basicrag--flat-baseline)**: Flat parsing, dense embeddings, vector search, and direct LLM generation.
- **[HierFinRAG Pipeline Workflow](ARCHITECTURE.md#2-hierfinrag--hierarchical-symbolic-neural-fusion)**: 4-level document tree, 2-stage intent router, 3-level hybrid search (BM25 + BGE-M3 + RRF), and deterministic FinQA DSL execution.
- **[TableTransformerRAG Pipeline Workflow](ARCHITECTURE.md#3-tabletransformerrag--vision-based-detr-detection)**: Vision-based DETR table detection, 300-DPI crop-to-PDF coordinate mapping, and two-pass zero-shot extraction.

---

## 🛠️ Quick Start & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

Ensure local [Ollama](https://ollama.com/) service is running with Qwen 2.5:
```bash
ollama pull qwen2.5:7b
```

### 2. Running Evaluation
To evaluate any system against the 110-question FinQA benchmark:

```bash
# BasicRAG Evaluation
cd Basic_Rag
python evaluate.py

# HierFinRAG Evaluation
cd ../HierFinRag
python evaluate.py

# TableTransformerRAG Evaluation
cd ../TableTransformerRAG
python evaluate.py
```

### 3. Interactive CLI Assistant
To launch the interactive terminal query interface:
```bash
python main.py
```

### 4. Compiling Research Paper PDF
To compile the IEEE-format research paper:
```bash
pdflatex IITH_paper.tex
```

---

## 📄 Key Research Output
- **IEEE Research Paper**: [`IITH_paper.pdf`](IITH_paper.pdf)
- **Architecture & System Workflows**: [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## 👤 Author
**Aravind**  
School of Computer and Information Sciences, University of Hyderabad  
*Work conducted during Summer Research Internship at IIT Hyderabad (IITH)*