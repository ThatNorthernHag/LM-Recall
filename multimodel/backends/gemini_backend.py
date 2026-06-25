"""Gemini backend (google-genai SDK, sync-wrapped-in-thread).

Default model: gemini-3.1-pro-preview. Override at construction time if you
want a different snapshot or tier.

WORKAROUND NOTE — sync path, not async path
-------------------------------------------
The google-genai SDK's async client (client.aio.models.generate_content)
fails with a ClientConnectorDNSError on Windows + Python 3.13:
aiohttp's resolver inside the SDK cannot resolve
generativelanguage.googleapis.com even when the system resolver returns
correct IPv4 + IPv6 records. The sync client works perfectly against
the same endpoint. To keep the async pipeline shape, this backend wraps
the sync call in asyncio.to_thread(). Each call costs a thread; the
pipeline's gen_concurrency semaphore still bounds parallelism.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import retry_with_backoff


_DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _load_gemini_key() -> str:
    load_dotenv()
    key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return key


class GeminiDecoder:
    """Response decoder for Google Gemini via the sync SDK + thread pool."""

    def __init__(
        self,
        *,
        name: str = "gemini-3.1-pro",
        family: str = "gemini",
        scale: str = "pro-3.1",
        model_id: str = _DEFAULT_MODEL,
        max_tokens: int = 2048,    # higher than other decoders: see thinking note below
        temperature: float = 0.0,
    ) -> None:
        from google import genai
        from google.genai import types

        self.name = name
        self.family = family
        self.scale = scale
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Gemini 3.x is a thinking model: by default it spends part of
        # max_output_tokens on internal reasoning, leaving the visible
        # answer truncated. For this experiment we want a direct answer,
        # not chain-of-thought, so we set thinking_level=MINIMAL. The
        # other decoders are not thinking models, so they were unaffected
        # by 300; for Gemini we also bump max_tokens to match what the
        # visible answer alone needs.
        self._client = genai.Client(api_key=_load_gemini_key())
        self._gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(
                # Pro 3.1 rejects MINIMAL with "not supported for this model";
                # LOW is the lowest accepted level. include_thoughts=False
                # suppresses the thinking from the visible response so the
                # 300-token budget covers the answer only.
                thinking_level=types.ThinkingLevel.LOW,
                include_thoughts=False,
            ),
        )

    def _sync_call(self, user_msg: str):
        return self._client.models.generate_content(
            model=self.model_id,
            contents=user_msg,
            config=self._gen_config,
        )

    async def generate(self, query: str, context: str) -> str:
        if context:
            user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_msg = f"Question: {query}"

        async def _call():
            # Run the sync call in a worker thread; asyncio.to_thread is
            # how we keep concurrency without the broken aiohttp path.
            return await asyncio.to_thread(self._sync_call, user_msg)

        resp = await retry_with_backoff(_call, label=f"gemini:{self.model_id}")
        return resp.text or ""

    async def close(self) -> None:
        return None
