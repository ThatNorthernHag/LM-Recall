"""Phase 2.3 API run: OpenAI GPT 5.4.

Model id: gpt-5.4 (provided by user from current OpenAI provider docs
as of 2026-06-02). Override at construction time if a different
snapshot is preferred.

This script waits for the user's green light.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends.anthropic_backend import make_judge_via_sonnet
from backends.openai_backend import OpenAIDecoder
from corpus_v3 import corpus
from pipeline import run_decoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


async def main(suffix: str):
    decoder = OpenAIDecoder(
        name="gpt-5.4",
        family="openai",
        scale="5.4",
        model_id="gpt-5.4",
    )
    judge = make_judge_via_sonnet()
    out_path = await run_decoder(
        decoder,
        corpus=corpus,
        results_dir=RESULTS_DIR,
        sonnet_judge=judge,
        gen_concurrency=24,
        judge_concurrency=24,
        force_suffix=suffix,
    )
    print(f"\nDONE: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suffix", default="")
    args = p.parse_args()
    asyncio.run(main(suffix=args.suffix))
