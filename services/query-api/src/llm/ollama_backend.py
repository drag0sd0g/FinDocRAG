"""Ollama LLM backend — calls local Ollama REST API.

References:
  - TDD: Section 5.2.3 (Ollama backend, mistral:7b default)
  - TDD: Section 7.1 (timeout after 300s, return 504 with partial response)
"""

from __future__ import annotations

import aiohttp
import structlog

from src.llm.backend import LLMResponse

logger = structlog.get_logger()

DEFAULT_OLLAMA_URL = "http://ollama:11434"
DEFAULT_MODEL = "mistral:7b"

# Ollama defaults num_ctx to 4096 and silently discards whatever does not fit —
# from the *start* of the prompt, which is exactly where the system prompt and
# the citation instruction live.  A top_k=5 RAG prompt is already ~3 K tokens
# (5 × 512-token chunks plus headers), and injected XBRL facts push it further,
# so the default is not enough headroom.  8192 covers top_k up to ~10 while
# keeping the KV cache small enough for the 8 GB local-LLM budget in the README;
# raise it (and the RAM allowance) if you routinely query with a larger top_k.
DEFAULT_NUM_CTX = 8192

# Rough chars-per-token ratio for English prose, used only to warn when a
# prompt is about to overflow the context window.
_CHARS_PER_TOKEN = 4


class OllamaBackend:
    """Calls the local Ollama server's /api/generate endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 300.0,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._num_ctx = num_ctx

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def num_ctx(self) -> int:
        return self._num_ctx

    def _warn_if_context_exceeded(self, prompt: str, max_tokens: int) -> None:
        """Log when prompt + answer budget cannot fit in the context window.

        Ollama truncates silently, so without this the only symptom is an
        answer that ignores its instructions or cites nothing.
        """
        estimated = len(prompt) // _CHARS_PER_TOKEN
        if estimated + max_tokens <= self._num_ctx:
            return
        logger.warning(
            "ollama_prompt_may_exceed_context",
            estimated_prompt_tokens=estimated,
            max_tokens=max_tokens,
            num_ctx=self._num_ctx,
            advice="Raise OLLAMA_NUM_CTX or lower top_k; Ollama truncates the "
            "start of the prompt, dropping the system instructions.",
        )

    async def generate(self, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        """Call Ollama /api/generate and return the response.

        Raises aiohttp.ClientError or asyncio.TimeoutError on failure
        (caller handles graceful degradation).
        """
        self._warn_if_context_exceeded(prompt, max_tokens)

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            # Suppress chain-of-thought on reasoning models (e.g. Qwen3): we want
            # the grounded answer in `response`, not thinking tokens that would
            # otherwise consume the num_predict budget and leave `response` empty.
            # Ignored by non-reasoning models (e.g. mistral).
            "think": False,
            "options": {
                "num_predict": max_tokens,
                # Without an explicit num_ctx, Ollama falls back to its own
                # small default and truncates the RAG context (see above).
                "num_ctx": self._num_ctx,
            },
        }

        async with aiohttp.ClientSession(timeout=self._timeout) as session, session.post(
            f"{self._base_url}/api/generate",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        text = data.get("response", "")
        # Ollama provides token counts in some versions
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        logger.info(
            "ollama_generate_complete",
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return LLMResponse(
            text=text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
