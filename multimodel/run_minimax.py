"""Phase 2.2 local-model run: MiniMax 2.7 (~220B).

MiniMax has a known server-side bug — run it LAST.
If the local server fails to load it, skip this script; the cross-decoder
table will note the gap honestly.

BEFORE YOU RUN:
  1. Load MiniMax 2.7 into your local server at [INSERT YOUR LOCAL SERVER URL HERE].
  2. If the server expects a specific `model` string, edit `model_id=` below.
  3. Run this script. Do not touch the server while it runs.

Output: multimodel/results/mvp_raw_results_minimax_220b_v3_{ts}.json
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
    name="minimax-2.7",
    family="minimax",
    scale="220B",
    model_id="local-model",   # EDIT IF YOUR SERVER NEEDS A SPECIFIC ID
    base_url="[INSERT YOUR LOCAL SERVER URL HERE]",
    max_tokens=2048,          # MiniMax M2 is a thinking model — 300 yielded
                              # 54% empty responses; 2048 gives thinking room
                              # plus visible answer
)


def _print_preflight():
    print("=" * 65)
    print("PRE-FLIGHT CHECK — read before pressing Enter")
    print("=" * 65)
    print()
    print("This script will hit the local server expecting MiniMax 2.7.")
    print(f"  endpoint:  {DECODER_KWARGS['base_url']}")
    print(f"  model_id:  {DECODER_KWARGS['model_id']}")
    print()
    print()
    print("Make sure the server has MiniMax 2.7 loaded, then press Enter.")
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
