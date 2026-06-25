"""Anthropic backend: response decoder for Claude + Sonnet-as-judge helper.

Two roles, one client object, one set of credentials:

  - `AnthropicDecoder` is the response decoder (used for Haiku 4.5 and
    Sonnet 4.6 in the multi-model roster).
  - `make_judge_via_sonnet(client)` produces the `judge` callable that
    pipeline.py uses across *every* decoder's response set. The judge is
    always Sonnet 4.6 — kept constant across the table (judge is a
    constant, not a variable, except where the response
    model IS Sonnet, which is the disclosed self-evaluation cell).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv

from .base import Decoder, retry_with_backoff


_DEFAULT_HAIKU = "claude-haiku-4-5-20251001"
_JUDGE_MODEL = "claude-sonnet-4-6"

_JUDGE_PROMPT_TEMPLATE = (
    "You are evaluating whether a response correctly conveys a specific piece of information.\n\n"
    "The user asked: {query}\n\n"
    "The response given was:\n"
    "---\n"
    "{response_text}\n"
    "---\n\n"
    'Does the response convey or contain the following information: "{fact}"?\n\n'
    "Consider both literal mentions and accurate paraphrasings. The fact is satisfied "
    "if the response demonstrates awareness of this specific information, even if "
    "expressed differently. The fact is NOT satisfied if the response only mentions "
    "the general topic without conveying this specific piece of information.\n\n"
    "Answer with only YES or NO."
)


def _load_anthropic_key() -> str:
    load_dotenv()
    key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No Anthropic API key found. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY."
        )
    return key


def make_async_anthropic_client():
    import anthropic
    return anthropic.AsyncAnthropic(api_key=_load_anthropic_key())


class AnthropicDecoder:
    """Response decoder for Claude models (Haiku or Sonnet).

    For the Sonnet-as-decoder run, this still uses the same backend; the
    Sonnet-judges-Sonnet self-evaluation cell is disclosed in the
    cross-decoder table.
    """

    def __init__(
        self,
        *,
        name: str,
        family: str = "claude",
        scale: str,
        model_id: str = _DEFAULT_HAIKU,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> None:
        self.name = name
        self.family = family
        self.scale = scale
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = make_async_anthropic_client()

    async def generate(self, query: str, context: str) -> str:
        if context:
            user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_msg = f"Question: {query}"

        async def _call():
            return await self._client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": user_msg}],
            )

        resp = await retry_with_backoff(_call, label=f"anthropic:{self.model_id}")
        return resp.content[0].text

    async def close(self) -> None:
        # Anthropic SDK manages connection pools internally; nothing to close.
        return None


def make_judge_via_sonnet(
    client=None,
    *,
    judge_model: str = _JUDGE_MODEL,
) -> Callable[[str, str, str], Awaitable[float]]:
    """Build the Sonnet judge callable used by pipeline.py.

    The returned async function: given (query, response_text, fact),
    returns 1.0 if Sonnet answers YES, else 0.0. Sonnet sees the same
    prompt template as the single-decoder MVP — preserves continuity with
    the published Sonnet-judge numbers.
    """
    if client is None:
        client = make_async_anthropic_client()

    async def judge(query: str, response_text: str, fact: str) -> float:
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            query=query, response_text=response_text, fact=fact
        )

        async def _call():
            return await client.messages.create(
                model=judge_model,
                max_tokens=5,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

        resp = await retry_with_backoff(_call, label=f"sonnet-judge:{judge_model}")
        answer = resp.content[0].text.strip().upper()
        return 1.0 if answer.startswith("YES") else 0.0

    return judge


