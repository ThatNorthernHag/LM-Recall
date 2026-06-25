"""Sentence-level compression for the structure-function protocol.

Ported unchanged from the single-decoder MVP. Do not modify the logic — every
decoder in the multi-model roster must see identical compression behavior
for cross-decoder comparability.
"""

from __future__ import annotations

import random
import re

COMPRESSION_LEVELS = (1.0, 0.75, 0.50, 0.25, 0.0)
RUNS_PER_LEVEL = 3
SEED = 42


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def compress(text: str, level: float, rng: random.Random) -> str:
    """Compress `text` to `level` fraction by random sentence sampling.

    level=1.0 returns the original text; level=0.0 returns the empty string.
    Sentence indices are drawn without replacement using the supplied rng,
    which is seeded per-run so the same compression decisions reproduce.
    """
    if level <= 0.0:
        return ""
    if level >= 1.0:
        return text
    sentences = split_sentences(text)
    n_keep = max(1, round(len(sentences) * level))
    kept = sorted(rng.sample(range(len(sentences)), n_keep))
    return " ".join(sentences[i] for i in kept)


def make_rng_for_run(run_index: int) -> random.Random:
    """Per-run RNG: deterministic, reproducible, identical across decoders."""
    return random.Random(SEED + run_index)
