"""
pipeline/router.py
==================
Financial intent classifier for the HierFinRAG pipeline.

Classifies each incoming query into one of four reasoning intents that drive
both retrieval behaviour and answer generation mode:

    Numerical     — requires a single arithmetic result
                    e.g. "What was the percentage change in revenue from 2022 to 2023?"

    Comparison    — requires comparing values across time periods or entities
                    e.g. "How did Apple's gross margin change between 2021 and 2023?"

    Lookup        — factual retrieval, no calculation needed
                    e.g. "What is Apple's fiscal year end date?"

    Summarization — open-ended explanation or qualitative summary
                    e.g. "What risks does Apple disclose regarding supply chains?"

Classification uses a two-stage strategy to minimise latency:

    Stage 1 — Rule-based  : keyword scoring + regex signals.  Fast (no LLM call).
    Stage 2 — LLM fallback: single-token Ollama call for ambiguous queries.

The intent string returned by classify() is consumed by:
    - retriever.py  : guides column selection (Numerical → year-matching column)
    - generator.py  : selects reasoning mode (symbolic / hybrid / neural)
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Keyword vocabularies for rule-based classification
# ---------------------------------------------------------------------------

_NUMERICAL_KEYWORDS = [
    "percentage", "percent", "change", "growth", "increase", "decrease",
    "ratio", "margin", "rate", "calculate", "compute", "how much",
    "total", "sum", "average", "mean", "difference", "subtract", "add",
    "multiply", "divide", "proportion",
]

_COMPARISON_KEYWORDS = [
    "compare", "versus", "vs", "higher", "lower", "better", "worse",
    "than", "between", "both", "each", "across", "trend", "year-over-year",
    "yoy", "quarter-over-quarter", "qoq",
]

_LOOKUP_KEYWORDS = [
    "what is", "what was", "who is", "when did", "where is",
    "which", "how many", "list", "name",
]

_SUMMARIZATION_KEYWORDS = [
    "summarize", "explain", "describe", "overview", "discuss",
    "why", "how does", "what are the", "tell me about",
    "risks", "risk", "factors", "exposure", "impact",
    "strategy", "policy", "concern", "challenge",
    "qualitative", "nature of", "type of",
]


# ---------------------------------------------------------------------------
# FinancialIntentClassifier
# ---------------------------------------------------------------------------

class FinancialIntentClassifier:
    """Two-stage financial query intent classifier.

    Stage 1 uses weighted keyword scoring with math-operator signals.
    Stage 2 falls back to a single low-latency LLM call when Stage 1 is
    inconclusive (all scores zero, or top two intents are tied).

    Args:
        llm : Optional QwenClient instance.  When None, only rule-based
              classification is used and the LLM fallback is skipped.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def classify(self, query: str) -> str:
        """Classify a query into a financial reasoning intent.

        Returns:
            One of: ``'Numerical'``, ``'Comparison'``, ``'Lookup'``,
            ``'Summarization'``.
        """
        intent = self._rule_based(query)

        if intent is None and self.llm is not None:
            intent = self._llm_classify(query)

        return intent or "Lookup"

    # ---------------------------------------------------------------------------
    # Stage 1 — Rule-based classification
    # ---------------------------------------------------------------------------

    def _rule_based(self, query: str) -> Optional[str]:
        """Score the query against keyword vocabularies.

        Math operators (+, -, *, /) and numeric literals each add 2 extra
        points to the Numerical score to handle questions that contain an
        explicit calculation signal without using keyword vocabulary.

        Returns:
            Intent string, or ``None`` when the result is ambiguous (all
            scores are zero, or the top two intents are within 1 point).
        """
        q = query.lower()

        has_math_op = bool(re.search(r"[\+\-\*/]", query))
        
        has_number = False
        for num_str in re.findall(r"\b\d+\.?\d*\b", query):
            try:
                v = int(float(num_str))
                if not (1990 <= v <= 2099 and float(num_str) == v):
                    has_number = True
                    break
            except (ValueError, OverflowError):
                has_number = True
                break

        num_score = sum(1 for kw in _NUMERICAL_KEYWORDS     if kw in q)
        cmp_score = sum(1 for kw in _COMPARISON_KEYWORDS    if kw in q)
        lkp_score = sum(1 for kw in _LOOKUP_KEYWORDS        if kw in q)
        sum_score = sum(1 for kw in _SUMMARIZATION_KEYWORDS if kw in q)

        if has_math_op or has_number:
            num_score += 2

        scores = {
            "Numerical":     num_score,
            "Comparison":    cmp_score,
            "Lookup":        lkp_score,
            "Summarization": sum_score,
        }

        best_intent = max(scores, key=lambda k: scores[k])
        best_score  = scores[best_intent]

        if best_score == 0:
            return None  # No signal — defer to LLM

        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] <= 1:
            # Tied at the top — Numerical gets priority if it is the winner
            if best_intent == "Numerical":
                return "Numerical"
            return None  # Defer to LLM for other ties

        return best_intent

    # ---------------------------------------------------------------------------
    # Stage 2 — LLM fallback
    # ---------------------------------------------------------------------------

    def _llm_classify(self, query: str) -> str:
        """Ask the LLM to classify an ambiguous query.

        The model is constrained to a single token response (max_tokens=5)
        to minimise latency.  The response is matched case-insensitively
        against the four valid intent strings.

        Returns:
            Intent string, or ``'Lookup'`` on any error.
        """
        system = (
            "You are a financial query intent classifier. "
            "Respond with exactly one word: Numerical, Comparison, Lookup, or Summarization."
        )
        prompt = (
            f"Classify this financial question:\n\"{query}\"\n\n"
            "Options:\n"
            "  Numerical     — requires arithmetic calculation\n"
            "  Comparison    — compares multiple values or time periods\n"
            "  Lookup        — factual retrieval, no calculation\n"
            "  Summarization — open-ended explanation\n\n"
            "Answer (one word only):"
        )

        try:
            answer = self.llm.generate(
                prompt        = prompt,
                system_prompt = system,
                temperature   = 0.0,
                max_tokens    = 5,
            ).strip()

            for valid in ("Numerical", "Comparison", "Lookup", "Summarization"):
                if valid.lower() in answer.lower():
                    return valid

        except Exception as exc:
            print(f"  [Router] LLM fallback failed: {exc}")

        return "Lookup"

    # ---------------------------------------------------------------------------
    # Legacy alias
    # ---------------------------------------------------------------------------

    def route(self, query: str) -> str:
        """Return ``'symbolic'`` or ``'neural'`` for backwards compatibility.

        New code should call :meth:`classify` and inspect the intent directly.
        """
        intent = self.classify(query)
        return "symbolic" if intent in ("Numerical", "Comparison") else "neural"
