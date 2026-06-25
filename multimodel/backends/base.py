"""Decoder protocol and shared backend utilities.

Every backend (Anthropic, local OpenAI-compatible, Gemini, OpenAI) must
implement the Decoder protocol below. pipeline.run_decoder() accepts any
Decoder and orchestrates the experiment around it.

Identity fields (name/family/scale) are persisted in the results JSON so
cross-decoder aggregation can group and label rows correctly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

log = logging.getLogger("multimodel.backends")


class DecoderError(Exception):
    """Raised by a backend when generation fails after retries."""


@runtime_checkable
class Decoder(Protocol):
    """Async generation interface common to every backend.

    Attributes:
        name:   short tag used in output filenames (e.g. 'qwen-3.5-397b').
        family: provider/lineage tag (e.g. 'qwen', 'claude', 'gemini',
                'openai', 'kimi', 'deepseek', 'minimax').
        scale:  size descriptor (e.g. '397B', '800B-q', 'flash', 'haiku',
                'sonnet', 'pro-3.1').

    Methods:
        generate: produce a single response given (query, context). Both
                  strings; context may be empty. Returns the response text.
        close:    optional async cleanup (close HTTP sessions, etc.).
    """

    name: str
    family: str
    scale: str

    async def generate(self, query: str, context: str) -> str:
        ...

    async def close(self) -> None:
        ...


# ─────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────

async def retry_with_backoff(
    coro_factory,
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    label: str = "<unnamed>",
):
    """Retry an async operation with exponential backoff + jitter.

    Pass a *factory* (zero-arg callable returning a fresh coroutine), not
    a coroutine, since coroutines cannot be awaited twice.
    """
    import random as _r
    delay = base_delay
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_err = e
            if attempt == max_attempts:
                log.error("%s: gave up after %d attempts (%s)", label, attempt, e)
                raise DecoderError(f"{label}: {e}") from e
            sleep = min(max_delay, delay) * (0.5 + _r.random())
            log.warning("%s: attempt %d failed (%s); retrying in %.2fs",
                        label, attempt, e, sleep)
            await asyncio.sleep(sleep)
            delay *= 2.0
    # unreachable
    raise DecoderError(f"{label}: {last_err}")
