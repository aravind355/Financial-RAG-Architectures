"""
pipeline/attribution.py
=======================
Answer attribution and confidence scoring for the HierFinRAG pipeline.

After the generator produces an answer this module:

    1. Decomposes the answer into atomic claims (individual factual statements).
    2. Maps each claim to supporting chunks from the retrieved context using
       keyword overlap (a lightweight, model-free heuristic).
    3. Computes a calibrated confidence score from three orthogonal signals:
           retrieval_quality     — mean cosine similarity of retrieved chunks
           reasoning_validity    — whether symbolic computations completed without error
           attribution_coverage  — fraction of claims with at least one supporting chunk
    4. Returns the structured result with full evidence provenance.

Output schema
-------------
    {
        "answer_text"          : str,
        "supporting_cells"     : list[{chunk_id, row_header, col_header, value}],
        "supporting_sentences" : list[{chunk_id, page, content_snippet}],
        "reasoning_steps"      : list[str],
        "confidence"           : float  (in [0, 1]),
        "attribution_map"      : dict[claim_str, list[chunk_id]],
    }
"""

import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def extract_atomic_claims(answer_text: str) -> List[str]:
    """Split an answer string into atomic, independently verifiable claims.

    Splits on sentence-ending punctuation and discards fragments shorter than
    five words (headings, labels, lone numbers) that cannot be meaningfully
    verified against a retrieved chunk.

    Args:
        answer_text : The full generated answer string.

    Returns:
        List of claim strings.  If no sentence-length claims are found, the
        entire answer is returned as a single-element list.
    """
    raw = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    claims = [s.strip() for s in raw if len(s.split()) >= 5]
    return claims if claims else [answer_text.strip()]


# ---------------------------------------------------------------------------
# Evidence search
# ---------------------------------------------------------------------------

def find_supporting_evidence(
    claim:            str,
    retrieved_chunks: List[dict],
) -> List[str]:
    """Find retrieved chunks that support a given claim.

    Uses token-level intersection: a chunk is considered supporting if at
    least two significant tokens (length ≥ 4 characters) from the claim
    appear in the chunk's content.

    Args:
        claim            : A single atomic claim from the answer.
        retrieved_chunks : Chunks returned by HierarchicalRetriever.retrieve().

    Returns:
        List of chunk_id strings for chunks that support this claim.
    """
    claim_tokens = set(w.lower() for w in re.findall(r"\b\w{4,}\b", claim))

    supporting_ids = []
    for chunk in retrieved_chunks:
        chunk_tokens = set(
            w.lower() for w in re.findall(r"\b\w{4,}\b", chunk.get("content", ""))
        )
        if len(claim_tokens & chunk_tokens) >= 2:
            supporting_ids.append(chunk["metadata"]["chunk_id"])

    return supporting_ids


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def compute_confidence(
    retrieval_quality:    float,
    reasoning_validity:   bool,
    attribution_coverage: float,
) -> float:
    """Compute a calibrated answer confidence score from three quality signals.

    Formula (HierFinRAG Section 3.5)::

        confidence = 0.3 × retrieval_quality
                   + 0.4 × reasoning_validity
                   + 0.3 × attribution_coverage

    Args:
        retrieval_quality    : Mean dense cosine similarity of retrieved chunks (0–1).
        reasoning_validity   : True if all symbolic steps executed without error.
        attribution_coverage : Fraction of claims with ≥1 supporting chunk (0–1).

    Returns:
        Confidence score clamped to [0, 1].
    """
    score = (
        0.3 * retrieval_quality
        + 0.4 * (1.0 if reasoning_validity else 0.0)
        + 0.3 * attribution_coverage
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Structured evidence extractors
# ---------------------------------------------------------------------------

def extract_supporting_cells(
    attribution_map:  Dict[str, List[str]],
    retrieved_chunks: List[dict],
) -> List[dict]:
    """Collect all cell-type chunks cited in the attribution map.

    Args:
        attribution_map  : Mapping of claim → list of supporting chunk_ids.
        retrieved_chunks : Full list of retrieved chunks.

    Returns:
        Deduplicated list of dicts with keys:
        chunk_id, row_header, col_header, value.
    """
    chunk_map = {
        c["metadata"]["chunk_id"]: c
        for c in retrieved_chunks
        if c["metadata"].get("chunk_id")
    }

    cells, seen = [], set()
    for chunk_ids in attribution_map.values():
        for cid in chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            chunk = chunk_map.get(cid)
            if chunk and chunk["metadata"].get("type") == "cell":
                meta = chunk["metadata"]
                cells.append({
                    "chunk_id":   cid,
                    "row_header": meta.get("row_header", ""),
                    "col_header": meta.get("col_header", ""),
                    "value":      meta.get("value", ""),
                })
    return cells


def extract_supporting_sentences(
    attribution_map:  Dict[str, List[str]],
    retrieved_chunks: List[dict],
) -> List[dict]:
    """Collect all text/section-type chunks cited in the attribution map.

    Args:
        attribution_map  : Mapping of claim → list of supporting chunk_ids.
        retrieved_chunks : Full list of retrieved chunks.

    Returns:
        Deduplicated list of dicts with keys:
        chunk_id, page, content_snippet (first 200 characters).
    """
    chunk_map = {
        c["metadata"]["chunk_id"]: c
        for c in retrieved_chunks
        if c["metadata"].get("chunk_id")
    }

    sentences, seen = [], set()
    for chunk_ids in attribution_map.values():
        for cid in chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            chunk = chunk_map.get(cid)
            if chunk and chunk["metadata"].get("type") in ("text", "section"):
                sentences.append({
                    "chunk_id":        cid,
                    "page":            chunk["metadata"].get("page", "?"),
                    "content_snippet": chunk["content"][:200],
                })
    return sentences


# ---------------------------------------------------------------------------
# Main attribution pipeline
# ---------------------------------------------------------------------------

def generate_with_attribution(
    query:               str,
    retrieved_chunks:    List[dict],
    intent:              str,
    llm,
    reasoning_had_error: bool = False,
) -> Dict[str, Any]:
    """Generate an answer and attach full evidence attribution.

    Runs the complete HierFinRAG generation + attribution loop:
        1. Generate the answer via pipeline.generator.generate().
        2. Decompose the answer into atomic claims.
        3. Map each claim to supporting chunks.
        4. Compute three quality signals and derive the confidence score.
        5. Extract structured cell and sentence evidence.

    Args:
        query               : The user's financial question.
        retrieved_chunks    : Output of HierarchicalRetriever.retrieve().
        intent              : Classified intent string.
        llm                 : QwenClient instance.
        reasoning_had_error : Set to True if the symbolic calculator raised
                              an error; reduces the confidence score.

    Returns:
        Dict with keys: answer_text, supporting_cells, supporting_sentences,
        reasoning_steps, confidence, attribution_map.
    """
    from .generator import generate

    # Step 1 — Generate answer
    answer_text = generate(query, retrieved_chunks, intent=intent, llm=llm)

    # Step 2 — Decompose into verifiable claims
    claims = extract_atomic_claims(answer_text)

    # Step 3 — Map each claim to supporting chunk IDs
    attribution_map: Dict[str, List[str]] = {
        claim: find_supporting_evidence(claim, retrieved_chunks)
        for claim in claims
    }

    # Step 4 — Compute quality signals
    dense_scores      = [c.get("dense_score", 0.0) for c in retrieved_chunks]
    retrieval_quality = sum(dense_scores) / len(dense_scores) if dense_scores else 0.0

    covered              = sum(1 for ev in attribution_map.values() if ev)
    attribution_coverage = covered / len(claims) if claims else 0.0

    confidence = compute_confidence(
        retrieval_quality    = retrieval_quality,
        reasoning_validity   = not reasoning_had_error,
        attribution_coverage = attribution_coverage,
    )

    # Step 5 — Extract structured evidence
    supporting_cells     = extract_supporting_cells(attribution_map, retrieved_chunks)
    supporting_sentences = extract_supporting_sentences(attribution_map, retrieved_chunks)
    reasoning_steps      = re.findall(r"\*\*(.+?)\*\*", answer_text)

    return {
        "answer_text":          answer_text,
        "supporting_cells":     supporting_cells,
        "supporting_sentences": supporting_sentences,
        "reasoning_steps":      reasoning_steps,
        "confidence":           confidence,
        "attribution_map": {
            claim: ev_list
            for claim, ev_list in attribution_map.items()
            if ev_list  # only include claims that have supporting evidence
        },
    }
