"""
BasicRAG Generator (Local Ollama Integration)
==============================================
Formats retrieved flat text/table chunks into prompt context and invokes
local Ollama LLM (Qwen-2.5-7B) with zero temperature.
"""

import ollama

MODEL = "qwen2.5:7b"

SYSTEM = """You are a financial analyst assistant.
Answer questions using ONLY the context provided from financial documents.
Always mention which page and source you are drawing from.
If the context lacks enough information, say: "The provided documents do not contain enough information to answer this."
Never invent numbers or facts."""

def format_context(chunks: list) -> str:
    """Format retrieved document chunks into structured text blocks with source metadata headers.

    Args:
        chunks (list): List of retrieved chunk dictionaries.

    Returns:
        str: Formatted context string for prompt injection.
    """
    parts = []
    for i, c in enumerate(chunks):
        m = c["metadata"]
        header = f"[Source {i+1} | {m['source']} | Page {m['page']} | Type: {m['type']}]"
        parts.append(f"{header}\n{c['content']}")
    return "\n\n---\n\n".join(parts)

def generate(query: str, chunks: list, **kwargs) -> str:
    """Generate natural language answer via local Ollama LLM call.

    Args:
        query (str): User question.
        chunks (list): Context chunks from retriever.

    Returns:
        str: Generated LLM response text.
    """
    context = format_context(chunks)
    user_msg = f"""Here is context from financial documents:

{context}

---

Question: {query}

Answer based only on the context above. Cite the source number and page."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg}
        ],
        options={"temperature": 0.0}
    )
    return response.message.content