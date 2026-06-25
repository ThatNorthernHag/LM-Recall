"""Phase 2.3 API run: Claude Sonnet 4.6 as response decoder.

NOTE — self-evaluation cell: Sonnet is also the fixed judge across
every decoder in the roster (judge is a constant).
That means for this script, the Sonnet-judges-Sonnet cell in the
cross-decoder table is self-evaluation. Substring scoring is the
load-bearing column for cross-decoder comparison; the judge-scored
column is reported with an asterisk in CROSS_DECODER_TABLE.md noting
"judge = response model (self-evaluation)".

This script waits for the user's green light — Claude does not run it
on its own initiative.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends.anthropic_backend import AnthropicDecoder, make_judge_via_sonnet
from corpus_v3 import corpus
from pipeline import run_decoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


async def main(suffix: str):
    decoder = AnthropicDecoder(
        name="claude-sonnet-4-6",
        family="claude",
        scale="sonnet",
        model_id="claude-sonnet-4-6",
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
