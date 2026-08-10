"""
pipeline/llm_client.py
======================
Local LLM client for the HierFinRAG pipeline, backed by Ollama.

Ollama serves any GGUF-quantized model through an OpenAI-compatible REST API,
so the standard openai-python client works without modification.  No API keys,
no rate limits, and no internet connection are required.

All pipeline modules (generator, router, attribution) import the client from
this module via the get_qwen_client() factory function.
"""

import json
from openai import OpenAI


class QwenClient:
    """Thin wrapper around the OpenAI client pointed at a local Ollama endpoint.

    The class is deliberately lightweight: it only handles the two call
    patterns needed by the pipeline (plain text and JSON-only responses).

    Args:
        base_url : Ollama API base URL.  Default: http://localhost:11434/v1
        model    : Ollama model tag.     Default: mistral:7b
        api_key  : Ignored by Ollama; required by the openai client. Any
                   non-empty string is accepted (default: "ollama").
    """

    def __init__(self, base_url: str, model: str, api_key: str = "ollama"):
        self.model   = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ---------------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------------

    def generate(
        self,
        prompt:        str,
        system_prompt: str   = "You are a helpful financial analysis assistant.",
        temperature:   float = 0.0,
        max_tokens:    int   = 2048,
    ) -> str:
        """Send a chat completion request and return the model's text response.

        Args:
            prompt        : User-facing instruction or question.
            system_prompt : Behavioral instruction for the model (role prompt).
            temperature   : Sampling temperature.  0.0 = fully deterministic.
            max_tokens    : Maximum number of tokens in the response.

        Returns:
            The model's response as a plain string.  Never raises on None
            content — an empty string is returned instead.
        """
        response = self._client.chat.completions.create(
            model       = self.model,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    def generate_json(
        self,
        prompt:        str,
        system_prompt: str   = "You are a precise JSON-generating assistant. Output only valid JSON.",
        temperature:   float = 0.0,
    ) -> dict:
        """Send a chat completion request and parse the response as JSON.

        Strips markdown code fences (```json ... ```) before parsing so the
        model does not need to be reminded to omit them.

        Returns:
            Parsed dict.  On JSON parse failure, returns {"raw": <raw_text>}
            so callers can inspect the unparseable response.
        """
        raw = self.generate(
            prompt        = prompt,
            system_prompt = system_prompt,
            temperature   = temperature,
            max_tokens    = 512,
        )

        # Strip potential ``` or ```json code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw": raw}


# ---------------------------------------------------------------------------
# Factory function — preferred import in all pipeline modules
# ---------------------------------------------------------------------------

def get_qwen_client() -> QwenClient:
    """Instantiate a QwenClient from the values in config.py.

    This is the single import point for all pipeline modules so that the
    model and endpoint are configured in exactly one place.

    Example::

        from pipeline.llm_client import get_qwen_client
        llm = get_qwen_client()
    """
    import config
    return QwenClient(
        base_url = config.OLLAMA_BASE_URL,
        api_key  = config.OLLAMA_API_KEY,
        model    = config.OLLAMA_MODEL,
    )
