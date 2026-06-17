import json
import os
from datetime import datetime
from pipeline.retriever import Retriever
from pipeline.generator import generate

TEST_CASES = [
    {
        "id": "Q01",
        "question": "What was Apple's total net sales revenue in fiscal year 2023?",
        "expected_keywords": ["383", "383,285", "383.3"],
        "expected_type": "text",
        "expected_page_range": [28, 35],
        "difficulty": "easy"
    },
    {
        "id": "Q02",
        "question": "How much did Apple spend on research and development in 2023?",
        "expected_keywords": ["29,915", "29915"],
        "expected_type": "table",
        "expected_page_range": [28, 52],
        "difficulty": "easy"
    },
    {
        "id": "Q03",
        "question": "What is Apple's net income for fiscal year 2023?",
        "expected_keywords": ["96,995", "96995"],
        "expected_type": "text",
        "expected_page_range": [28, 40],
        "difficulty": "easy"
    },
    {
        "id": "Q04",
        "question": "What were Apple's iPhone net sales in 2023?",
        "expected_keywords": ["200,583", "200583"],
        "expected_type": "text",
        "expected_page_range": [35, 45],
        "difficulty": "medium"
    },
    {
        "id": "Q05",
        "question": "What is Apple's total assets value on the balance sheet?",
        "expected_keywords": ["352,583", "352583"],
        "expected_type": "text",
        "expected_page_range": [30, 40],
        "difficulty": "medium"
    },
    {
        "id": "Q06",
        "question": "What are the main categories of risk factors Apple identifies?",
        "expected_keywords": ["macroeconomic", "financial", "legal", "risk"],
        "expected_type": "text",
        "expected_page_range": [5, 20],
        "difficulty": "medium"
    },
    {
        "id": "Q07",
        "question": "How much did Apple pay in dividends in 2023?",
        "expected_keywords": ["dividend", "14,996", "15,025"],
        "expected_type": "table",
        "expected_page_range": [30, 55],
        "difficulty": "hard"
    },
    {
        "id": "Q08",
        "question": "What is Apple's operating income for 2023?",
        "expected_keywords": ["114,301", "114301"],
        "expected_type": "text",
        "expected_page_range": [28, 40],
        "difficulty": "hard"
    },
    {
        "id": "Q09",
        "question": "What segments does Apple report revenue by geographically?",
        "expected_keywords": ["Americas", "Europe", "China", "Japan", "rest of asia"],
        "expected_type": "text",
        "expected_page_range": [35, 55],
        "difficulty": "medium"
    },
    {
        "id": "Q10",
        "question": "What were Apple's total liabilities in 2023?",
        "expected_keywords": ["290,437", "290437"],
        "expected_type": "table",
        "expected_page_range": [30, 40],
        "difficulty": "hard"
    },
]

def evaluate_answer(answer, test_case):
    """Score a single answer against ground truth."""
    answer_lower = answer.lower()

    # Keyword hit rate
    hits = [kw for kw in test_case["expected_keywords"]
            if kw.lower() in answer_lower]
    keyword_score = len(hits) / len(test_case["expected_keywords"])

    return {
        "keyword_score": round(keyword_score, 3),
        "keywords_found": hits,
        "keywords_missed": [kw for kw in test_case["expected_keywords"]
                            if kw.lower() not in answer_lower]
    }

def evaluate_retrieval(chunks, test_case):
    """Score retrieval quality."""
    retrieved_types  = [c["metadata"]["type"] for c in chunks]
    retrieved_pages  = [int(c["metadata"].get("page", 0)) for c in chunks]
    page_range       = test_case["expected_page_range"]

    right_type = test_case["expected_type"] in retrieved_types
    right_page = any(page_range[0] <= p <= page_range[1]
                     for p in retrieved_pages)

    top1_page_hit = (page_range[0]
                     <= int(chunks[0]["metadata"]["page"])
                     <= page_range[1]) if chunks else False

    return {
        "right_type_in_top5": right_type,
        "right_page_in_top5": right_page,
        "top1_page_correct":  top1_page_hit,
        "retrieved_pages":    retrieved_pages,
        "retrieved_types":    retrieved_types
    }

def run_evaluation():
    retriever = Retriever()
    results   = []

    print("\n" + "="*65)
    print("  BASELINE EVALUATION — Multimodal RAG on Apple 10-K")
    print("="*65 + "\n")

    for tc in TEST_CASES:
        print(f"[{tc['id']}] {tc['question'][:60]}...")

        # Retrieve
        chunks = retriever.retrieve(tc["question"], top_k=20, top_n=5)

        # Generate
        answer = generate(tc["question"], chunks)

        # Score
        answer_eval    = evaluate_answer(answer, tc)
        retrieval_eval = evaluate_retrieval(chunks, tc)

        result = {
            "id":               tc["id"],
            "question":         tc["question"],
            "difficulty":       tc["difficulty"],
            "answer":           answer,
            "answer_score":     answer_eval,
            "retrieval_score":  retrieval_eval,
            "top_chunks": [{
                "page":         c["metadata"]["page"],
                "type":         c["metadata"]["type"],
                "dense_score":  c["dense_score"],
                "rerank_score": c["rerank_score"],
                "preview":      c["content"][:100]
            } for c in chunks]
        }
        results.append(result)

        # Print per-question summary
        ks  = answer_eval["keyword_score"]
        rp  = retrieval_eval["right_page_in_top5"]
        rt  = retrieval_eval["right_type_in_top5"]
        t1  = retrieval_eval["top1_page_correct"]
        print(f"  answer_keywords={ks:.0%}  page_hit={rp}  type_hit={rt}  top1_correct={t1}")
        if answer_eval["keywords_missed"]:
            print(f"  missed: {answer_eval['keywords_missed']}")
        print()

    # ── Aggregate metrics ────────────────────────────────────────
    n = len(results)
    avg_kw    = sum(r["answer_score"]["keyword_score"]            for r in results) / n
    pct_page  = sum(r["retrieval_score"]["right_page_in_top5"]   for r in results) / n
    pct_type  = sum(r["retrieval_score"]["right_type_in_top5"]   for r in results) / n
    pct_top1  = sum(r["retrieval_score"]["top1_page_correct"]    for r in results) / n

    print("="*65)
    print("  BASELINE RESULTS")
    print("="*65)
    print(f"  Avg keyword hit rate   : {avg_kw:.1%}")
    print(f"  Right page in top-5    : {pct_page:.1%}")
    print(f"  Right chunk type       : {pct_type:.1%}")
    print(f"  Top-1 chunk correct    : {pct_top1:.1%}")
    print("="*65)

    # ── Save full results ────────────────────────────────────────
    os.makedirs("data/evals", exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/evals/baseline_{timestamp}.json"

    summary = {
        "timestamp":    timestamp,
        "model":        "llava:7b-v1.6-mistral-q4_K_M",
        "embed_model":  "BAAI/bge-m3",
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "vector_db":    "qdrant",
        "num_chunks":   retriever.client.get_collection("finance_rag").points_count,
        "metrics": {
            "avg_keyword_hit_rate": round(avg_kw, 3),
            "right_page_in_top5":  round(pct_page, 3),
            "right_type_in_top5":  round(pct_type, 3),
            "top1_correct":        round(pct_top1, 3)
        },
        "per_question": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Full results saved → {output_path}")

if __name__ == "__main__":
    run_evaluation()