"""OpenAI backend (openai async client).

Default model: gpt-5.4. Override at construction time if you want a different
snapshot or tier.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .base import retry_with_backoff


_DEFAULT_MODEL = "gpt-5.4"


def _load_openai_key() -> str:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY.")
    return key


class OpenAIDecoder:
    """Response decoder for OpenAI GPT via the openai async client."""

    def __init__(
        self,
        *,
        name: str = "gpt-5.4",
        family: str = "openai",
        scale: str = "5.4",
        model_id: str = _DEFAULT_MODEL,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> None:
        from openai import AsyncOpenAI

        self.name = name
        self.family = family
        self.scale = scale
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._client = AsyncOpenAI(api_key=_load_openai_key())

    async def generate(self, query: str, context: str) -> str:
        if context:
            user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_msg = f"Question: {query}"

        async def _call():
            return await self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": user_msg}],
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        resp = await retry_with_backoff(_call, label=f"openai:{self.model_id}")
        return resp.choices[0].message.content or ""

    async def close(self) -> None:
        await self._client.close()
