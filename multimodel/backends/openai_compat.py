"""Local OpenAI-compatible backend
"""

from __future__ import annotations

import httpx

from .base import retry_with_backoff


class OpenAICompatibleDecoder:
    """Generic OpenAI-compatible chat-completions client for local models."""

    def __init__(
        self,
        *,
        name: str,
        family: str,
        scale: str,
        model_id: str = "local-model",
        base_url: str = "[INSERT YOUR URL HERE]",
        api_key: str = "EMPTY",          # most local servers ignore this
        max_tokens: int = 2048,          # higher than the API default: most current
                                         # local models (Qwen3, MiniMax M2, DeepSeek
                                         # R-series, Kimi K2 think) are thinking
                                         # models that consume output budget on
                                         # internal reasoning. 300 leaves nothing
                                         # visible. 2048 is the safe floor.
        temperature: float = 0.0,
        timeout_seconds: float = 600.0,  # large local models can be slow
        extra_body: dict | None = None,  # merged into the request payload — pass
                                         # server-specific knobs here, e.g.
                                         # {"enable_thinking": False} for vLLM-Qwen3
    ) -> None:
        self.name = name
        self.family = family
        self.scale = scale
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.extra_body = dict(extra_body) if extra_body else {}

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=15.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def generate(self, query: str, context: str) -> str:
        if context:
            user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_msg = f"Question: {query}"

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": user_msg}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.extra_body:
            payload.update(self.extra_body)

        async def _call():
            r = await self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()

        data = await retry_with_backoff(_call, label=f"local:{self.name}")
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
