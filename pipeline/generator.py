import ollama

MODEL = "llava:7b-v1.6-mistral-q4_K_M"

SYSTEM = """You are a financial analyst assistant.
Answer questions using ONLY the context provided from financial documents.
Always mention which page and source you are drawing from.
If the context lacks enough information, say: "The provided documents do not contain enough information to answer this."
Never invent numbers or facts."""

def format_context(chunks):
    parts = []
    for i, c in enumerate(chunks):
        m = c["metadata"]
        header = f"[Source {i+1} | {m['source']} | Page {m['page']} | Type: {m['type']}]"
        parts.append(f"{header}\n{c['content']}")
    return "\n\n---\n\n".join(parts)

def generate(query, chunks):
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
        ]
    )
    return response.message.content