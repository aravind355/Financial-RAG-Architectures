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
├── Table_Extraction/       # Vision table detection experimental benchmarks
├── IITH_paper.tex          # IEEE conference standard research paper (LaTeX)
├── IITH_paper.pdf          # Compiled 5-page IEEE research paper PDF
├── IITH.tex                # Comprehensive 18-page technical project report
├── abstract.tex            # Standalone abstract document
├── report_25_07_2026.tex   # TableTransformerRAG 4-round optimization analysis
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
- **Technical Report**: [`IITH.pdf`](IITH.pdf)
- **Abstract PDF**: [`abstract.pdf`](abstract.pdf)

---

## 👤 Author
**Aravind**  
School of Computer and Information Sciences, University of Hyderabad  
*Work conducted during Summer Research Internship at IIT Hyderabad (IITH)*
