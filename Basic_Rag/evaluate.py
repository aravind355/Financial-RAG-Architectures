"""
evaluate.py (Basic RAG)
=======================
Basic RAG baseline pipeline evaluation.

Uses the same exact dataset and scoring functions (Execution Accuracy, F1) 
as the main hierarchical RAG to ensure a fair, apples-to-apples comparison.

Usage:
    python evaluate.py
    python evaluate.py --max 50
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Optional

from pipeline.retriever import Retriever
from pipeline.generator import generate

# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def _clean_number(s) -> Optional[float]:
    """Parse a value as float, stripping financial formatting ($, %, commas).
    Also extracts the leading number from strings like '10.6 years'."""
    try:
        if isinstance(s, list):
            s = str(s[0])
        s = str(s).replace("$", "").replace(",", "").replace("%", "").replace("€", "").strip()
        try:
            return float(s)
        except ValueError:
            m = re.match(r"^(-?\d+\.?\d*)", s)
            return float(m.group(1)) if m else None
    except (ValueError, TypeError, IndexError):
        return None

def _normalize_text(s: str) -> str:
    """Lowercase and strip financial symbols, punctuation, and extra whitespace."""
    s = str(s).lower().strip()
    s = re.sub(r"[$,%\-,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _words_contained(expected: str, predicted: str) -> bool:
    """Return True if all significant words in 'expected' appear in 'predicted'."""
    _STOP = {"and", "the", "or", "of", "in", "to", "a", "an", "is",
             "are", "was", "were", "for", "by", "at", "on", "as"}
    exp_words = [w for w in _normalize_text(expected).split()
                 if len(w) >= 3 and w not in _STOP]
    if len(exp_words) < 2:
        return False
    pred_norm = _normalize_text(predicted)
    return all(w in pred_norm for w in exp_words)

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def execution_accuracy(predicted: Optional[str], expected) -> bool:
    """Exact match after normalization (primary FinQA metric)."""
    if predicted is None:
        return False

    pred_str = str(predicted).strip()
    exp_str  = str(expected).strip()
    pred_has_pct = "%" in pred_str
    exp_has_pct  = "%" in exp_str

    pred_num = _clean_number(predicted)
    exp_num  = _clean_number(expected)

    if pred_num is not None and exp_num is not None:
        if pred_has_pct and not exp_has_pct and abs(exp_num) < 1:
            pred_num /= 100.0
        elif exp_has_pct and not pred_has_pct and abs(pred_num) < 1:
            exp_num /= 100.0

        if exp_num == 0:
            return abs(pred_num) < 1e-4
        rel_error = abs(pred_num - exp_num) / max(abs(exp_num), 1e-9)
        return rel_error <= 0.01 or abs(pred_num - exp_num) < 0.01

    return (
        _normalize_text(predicted) == _normalize_text(exp_str)
        or _normalize_text(exp_str) in _normalize_text(predicted)
        or _words_contained(exp_str, predicted)
    )

def scaled_f1(predicted: Optional[str], expected) -> float:
    """Scaled F1 for numeric answers: max(0, 1 - |pred - gold| / |gold|)."""
    if predicted is None:
        return 0.0

    pred_str = str(predicted).strip()
    exp_str  = str(expected).strip()
    pred_has_pct = "%" in pred_str
    exp_has_pct  = "%" in exp_str

    pred_num = _clean_number(predicted)
    exp_num  = _clean_number(expected)

    if pred_num is None or exp_num is None:
        return 0.0

    if pred_has_pct and not exp_has_pct and abs(exp_num) < 1:
            pred_num /= 100.0
    elif exp_has_pct and not pred_has_pct and abs(pred_num) < 1:
        exp_num /= 100.0

    if exp_num == 0:
        return 1.0 if pred_num == 0 else 0.0

    return max(0.0, 1.0 - abs(pred_num - exp_num) / abs(exp_num))

def extract_predicted_value(answer_text: str, is_symbolic: bool) -> Optional[str]:
    """Extract the final predicted value from a generated answer string."""
    text = str(answer_text)

    if is_symbolic:
        match = re.search(r"=\s*\**(-?[\d,.]+%?)\**", text)
        if match:
            return match.group(1).replace(",", "").strip()

    match = re.search(
        r"(?:answer|result|total|value)\s*(?:is|:|=)\s*\$?\s*([-\d,.]+%?)",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "").strip()

    pct_match = re.search(r"(-?\d+\.\d+%)", text)
    if pct_match:
        return pct_match.group(1).strip()

    if is_symbolic:
        numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))

        def _is_year(n: str) -> bool:
            try:
                v = int(float(n))
                return 1990 <= v <= 2099 and float(n) == v
            except (ValueError, OverflowError):
                return False

        non_year = [n for n in numbers if not _is_year(n)]
        if non_year:
            return non_year[-1]

    return text.strip()

def retrieval_recall(chunks: list, gold_text: str, k: int = 5) -> float:
    """Fraction of significant gold-context words found in the top-K chunks."""
    if not gold_text or not chunks:
        return 0.0
    gold_words = {w.lower() for w in gold_text.split() if len(w) >= 4}
    if not gold_words:
        return 0.0
    combined = " ".join(c["content"] for c in chunks[:k]).lower()
    return sum(1 for w in gold_words if w in combined) / len(gold_words)

def load_dataset(path: str = "../data/datasets/eval_dataset.json", max_items: int = 0) -> list:
    """Load the evaluation dataset from a FinQA-format JSON file."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[:max_items] if max_items else data

def _format_gold_context(item: dict) -> str:
    """Serialize a FinQA item's gold pre_text + table + post_text into a string."""
    parts = []
    pre = "\n".join(item.get("pre_text", []))
    if pre:
        parts.append(pre)
    for row in item.get("table", []):
        parts.append(" | ".join(str(c) for c in row))
    post = "\n".join(item.get("post_text", []))
    if post:
        parts.append(post)
    return " ".join(parts)

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_path: str = "../data/datasets/eval_dataset.json",
    max_items: int = 0,
    top_n: int = 10,
) -> None:
    print("\nInitializing Basic RAG pipeline...")
    retriever = Retriever()

    dataset = load_dataset(path=dataset_path, max_items=max_items)
    if not dataset:
        print(f"No data found at: {dataset_path}")
        return

    # Summarize dataset by source
    sources: dict = {}
    for item in dataset:
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    src_summary = "  |  ".join(f"{s}: {n}Q" for s, n in sources.items())

    print(f"\nEvaluation — {len(dataset)} questions (Basic RAG Dense Retrieval)")
    print(f"  Sources : {src_summary}")
    print(f"  Top-N   : {top_n} chunks per question\n")
    print("=" * 65)

    results      = []
    correct      = 0
    f1_total     = 0.0
    recall_total = 0.0
    
    # We will track stats by intent and source to match the main eval output
    route_stats: dict = {
        "Numerical":     {"total": 0, "correct": 0},
        "Comparison":    {"total": 0, "correct": 0},
        "Lookup":        {"total": 0, "correct": 0},
        "Summarization": {"total": 0, "correct": 0},
    }
    source_stats: dict = {}
    total_start = time.time()

    for i, item in enumerate(dataset):
        q_start  = time.time()
        question = item["qa"]["question"]
        expected = item["qa"]["exe_ans"]
        src      = item.get("source", "unknown")

        # The basic pipeline doesn't have an intent classifier, but we track
        # the gold intent to compare performance by category fairly.
        intent = item.get("intent", "Lookup")
        
        route_stats.setdefault(intent, {"total": 0, "correct": 0})
        route_stats[intent]["total"] += 1

        if src not in source_stats:
            source_stats[src] = {"total": 0, "correct": 0, "recall_sum": 0.0}
        source_stats[src]["total"] += 1

        print(f"[{i+1:>3}/{len(dataset)}] [{intent[:3].upper()}] [{src.replace('.pdf','')}] {question[:60]}...")

        try:
            # Basic RAG retriever doesn't take source or intent filters
            chunks = retriever.retrieve(query=question, top_k=20, top_n=top_n)

            # Retrieval recall vs gold context (proxy metric)
            gold_ctx = _format_gold_context(item)
            recall   = retrieval_recall(chunks, gold_ctx, k=top_n)
            recall_total += recall
            source_stats[src]["recall_sum"] += recall

            answer    = generate(query=question, chunks=chunks)
            
            is_sym    = intent in ("Numerical", "Comparison")
            predicted = extract_predicted_value(answer, is_symbolic=is_sym)

            is_correct = execution_accuracy(predicted, expected)
            f1         = scaled_f1(predicted, expected)

            if is_correct:
                correct += 1
                route_stats[intent]["correct"] += 1
                source_stats[src]["correct"] += 1
            f1_total += f1

        except Exception as e:
            print(f"  [ERROR] {e}")
            answer, predicted, is_correct, f1, recall = str(e), None, False, 0.0, 0.0

        elapsed = time.time() - q_start
        status  = "[PASS]" if is_correct else "[FAIL]"
        print(f"  {status} Expected: {expected!s:<15} Predicted: {predicted!s:<15} "
              f"Recall: {recall:.0%}  {elapsed:.1f}s")

        results.append({
            "id":           item.get("id", f"q_{i}"),
            "source":       src,
            "question":     question,
            "intent":       intent,
            "expected":     expected,
            "predicted":    predicted,
            "is_correct":   is_correct,
            "scaled_f1":    round(f1, 4),
            "recall":       round(recall, 4),
            "n_chunks":     len(chunks) if "chunks" in dir() else 0,
            "time_seconds": round(elapsed, 2),
        })

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    n          = len(dataset)
    total_time = time.time() - total_start
    em_acc     = correct / n
    avg_f1     = f1_total / n
    avg_recall = recall_total / n

    print(f"\n{'=' * 65}")
    print(f"  EXECUTION ACCURACY (EM): {em_acc:.1%}  ({correct}/{n})")
    print(f"  AVERAGE SCALED F1:       {avg_f1:.3f}")
    print(f"  AVG RETRIEVAL RECALL@{top_n}:  {avg_recall:.1%}")
    print(f"  Total time: {total_time/60:.1f} min  ({total_time/n:.1f}s/question)")
    print(f"{'=' * 65}")

    print("\n  Per-Intent Breakdown:")
    for intent_key, stats in route_stats.items():
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        print(f"    {intent_key:<15}: {acc:.1%}  ({stats['correct']}/{stats['total']})")

    print("\n  Per-Source Breakdown:")
    for src_key, stats in source_stats.items():
        if stats["total"] == 0:
            continue
        acc    = stats["correct"] / stats["total"]
        recall = stats["recall_sum"] / stats["total"]
        label  = src_key.replace(".pdf", "")
        print(f"    {label:<25}: EM={acc:.1%}  Recall={recall:.1%}  "
              f"({stats['correct']}/{stats['total']})")

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs("data/evals", exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/evals/basic_eval_{ts}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode":                 "basic_baseline",
            "execution_accuracy":   em_acc,
            "average_scaled_f1":    avg_f1,
            "avg_retrieval_recall": avg_recall,
            "total_cases":          n,
            "correct_cases":        correct,
            "total_time_seconds":   round(total_time, 2),
            "route_breakdown":      route_stats,
            "source_breakdown": {
                k: {
                    "em":     round(v["correct"] / (v["total"] or 1), 4),
                    "recall": round(v["recall_sum"] / (v["total"] or 1), 4),
                    "n":      v["total"],
                }
                for k, v in source_stats.items()
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basic RAG end-to-end evaluation"
    )
    parser.add_argument(
        "--dataset", default="../data/datasets/eval_dataset.json",
        help="Path to eval dataset JSON"
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="Max questions to evaluate, 0 = all (default: 0)"
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Chunks to retrieve per question (default: 10)"
    )
    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        max_items=args.max,
        top_n=args.top_n,
    )