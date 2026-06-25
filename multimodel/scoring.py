"""Scoring functions: substring matching + LLM-judge.

- `score_facts_substring` is carried over unchanged from the MVP: model-independent, free,
  used as the primary cross-decoder metric.
- `score_facts_sonnet` calls the Sonnet judge once per expected fact and
  returns the mean YES fraction. Async so it can be batched concurrently
  by the pipeline.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Sequence


JudgeCallable = Callable[[str, str, str], Awaitable[float]]
"""Signature: (query, response_text, fact) -> 1.0 if YES else 0.0.

The pipeline passes a closure over the Anthropic async client; the judge
backend implementation lives in backends/anthropic_backend.py.
"""


def score_facts_substring(response_text: str, expected_facts: Sequence[str]) -> float:
    if not expected_facts:
        return 0.0
    response_lower = response_text.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in response_lower)
    return hits / len(expected_facts)


async def score_facts_sonnet(
    judge: JudgeCallable,
    query: str,
    response_text: str,
    expected_facts: Sequence[str],
) -> tuple[float, list[float]]:
    """Per-fact judge YES/NO, returns (mean, per_fact_scores).

    Calls judge sequentially per fact within a single (item, level, run) cell;
    cross-cell parallelism is handled by the pipeline. This keeps the
    per-fact judge requests grouped for a given response, which makes the
    progress reporting and rate-limiting easier to reason about.
    """
    if not expected_facts:
        return 0.0, []
    per_fact = [await judge(query, response_text, fact) for fact in expected_facts]
    return sum(per_fact) / len(per_fact), per_fact
