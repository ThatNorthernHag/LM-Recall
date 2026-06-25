"""Phase 2.2 local-model run: Kimi K2.6 (~800B quantized).

BEFORE YOU RUN:
  1. Load Kimi K2.6 into your local server at [INSERT YOUR LOCAL SERVER URL HERE].
  2. If the server expects a specific `model` string, edit `model_id=` below.
  3. Run this script. Do not touch the server while it runs.

Output: multimodel/results/mvp_raw_results_kimi_k26q_v3_{ts}.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends.anthropic_backend import make_judge_via_sonnet
from backends.openai_compat import OpenAICompatibleDecoder
from corpus_v3 import corpus
from pipeline import run_decoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"

DECODER_KWARGS = dict(
    name="kimi-k2.6",
    family="kimi",
    scale="800Bq",
    model_id="local-model",   # EDIT IF YOUR SERVER NEEDS A SPECIFIC ID
    base_url="[INSERT YOUR LOCAL SERVER URL HERE]",
    max_tokens=2048,          # safe default for thinking-capable models
)


def _print_preflight():
    print("=" * 65)
    print("PRE-FLIGHT CHECK — read before pressing Enter")
    print("=" * 65)
    print()
    print("This script will hit the local server expecting Kimi K2.6 (quantized).")
    print(f"  endpoint:  {DECODER_KWARGS['base_url']}")
    print(f"  model_id:  {DECODER_KWARGS['model_id']}")
    print()
    print("Make sure the server has Kimi K2.6 loaded, then press Enter to start.")
    print()


async def main(skip_preflight: bool, suffix: str):
    if not skip_preflight:
        _print_preflight()
        input("[Enter to start, Ctrl-C to cancel]  ")
    decoder = OpenAICompatibleDecoder(**DECODER_KWARGS)
    judge = make_judge_via_sonnet()
    out_path = await run_decoder(
        decoder,
        corpus=corpus,
        results_dir=RESULTS_DIR,
        sonnet_judge=judge,
        gen_concurrency=6,
        judge_concurrency=24,
        force_suffix=suffix,
    )
    print(f"\nDONE: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--suffix", default="")
    args = p.parse_args()
    asyncio.run(main(skip_preflight=args.yes, suffix=args.suffix))
