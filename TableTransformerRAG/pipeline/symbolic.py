"""
pipeline/symbolic.py
====================
Symbolic DSL calculator for the HierFinRAG pipeline.

Executes the arithmetic Domain-Specific Language (DSL) that the LLM generates
for Numerical and Comparison queries.  Using a DSL instead of eval() gives
precise, auditable results and eliminates the risk of arbitrary code execution.

Supported operations
--------------------
    add(a, b)                           →  a + b
    subtract(a, b)                      →  a - b
    multiply(a, b)                      →  a * b
    divide(a, b)                        →  a / b
    percentage_change(old_val, new_val) →  (new_val - old_val) / old_val
    percentage(part, whole)             →  (part / whole) * 100

Operations can be nested to arbitrary depth::

    divide(subtract(120, 100), 100)     →  (120 - 100) / 100  →  0.2

The calculator uses iterative regex substitution to handle nesting, then
evaluates the resulting arithmetic expression with numexpr (no eval()).
"""

import re
import numexpr as ne


class SymbolicCalculator:
    """Converts a FinQA DSL program string into a numeric result.

    Args:
        precision : Number of decimal places to round the final result to.
    """

    def __init__(self, precision: int = 4):
        self.precision = precision

    def compute(self, program: str) -> float | None:
        """Execute a DSL program and return the numeric result.

        Args:
            program : A DSL expression, e.g. ``"percentage_change(78129, 85200)"``.

        Returns:
            Float result rounded to ``self.precision`` decimal places,
            or ``None`` if parsing or evaluation fails.
        """
        try:
            expr   = self._parse_dsl(program.strip())
            result = ne.evaluate(expr)
            return round(float(result), self.precision)
        except Exception:
            return None  # Caller handles fallback to neural generation

    # ---------------------------------------------------------------------------
    # DSL → arithmetic expression converter
    # ---------------------------------------------------------------------------

    def _parse_dsl(self, expr: str) -> str:
        """Convert a DSL expression to a plain arithmetic string.

        Iteratively replaces the innermost function calls with their arithmetic
        equivalents until no more substitutions are possible (convergence).
        Also sanitises identifier tokens — if the LLM outputs a variable name
        instead of a number, it is replaced with 0 to prevent injection.

        Args:
            expr : Raw DSL string (possibly nested).

        Returns:
            Arithmetic expression string safe to pass to numexpr.evaluate().
        """
        _ALLOWED = {
            "add", "subtract", "multiply", "divide",
            "percentage_change", "percentage",
        }

        # Replace any identifier that is not a known DSL function with 0
        expr = re.sub(
            r"([a-zA-Z_]\w*)",
            lambda m: m.group(1) if m.group(1) in _ALLOWED else "0",
            expr,
        )

        # Iteratively expand innermost function calls (max depth = 15)
        for _ in range(15):
            prev = expr

            expr = re.sub(r"add\(([^()]+),\s*([^()]+)\)",                  r"(\1 + \2)",          expr)
            expr = re.sub(r"subtract\(([^()]+),\s*([^()]+)\)",             r"(\1 - \2)",          expr)
            expr = re.sub(r"multiply\(([^()]+),\s*([^()]+)\)",             r"(\1 * \2)",          expr)
            expr = re.sub(r"divide\(([^()]+),\s*([^()]+)\)",               r"(\1 / \2)",          expr)
            expr = re.sub(r"percentage_change\(([^()]+),\s*([^()]+)\)",    r"((\2 - \1) / \1)",   expr)
            expr = re.sub(r"percentage\(([^()]+),\s*([^()]+)\)",           r"(\1 / \2 * 100)",    expr)

            if expr == prev:
                break  # Converged — no further substitutions possible

        return expr
