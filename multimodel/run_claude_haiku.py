"""Phase 2.1 sanity-gate run: Claude Haiku 4.5 on the v3 corpus.

This is the first script that must pass — it sanity-gates the whole
multi-model harness. If the peak advantage at budget=15 doesn't land
within ~3 pp of the v2 Sonnet-judge number (+32.5%), something in the
refactor changed behavior and the rest of the roster shouldn't run.

The corpus is v3 (Clarke relabeled to known), so a small shift from
v2's +32.5% is expected (1-2 pp typically). A shift larger than ~5 pp
is a bug, not the relabel.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Make multimodel/ importable when run as a script.
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


async def main():
    decoder = AnthropicDecoder(
        name="claude-haiku-4-5",
        family="claude",
        scale="haiku",
        model_id="claude-haiku-4-5-20251001",
    )
    judge = make_judge_via_sonnet()
    out_path = await run_decoder(
        decoder,
        corpus=corpus,
        results_dir=RESULTS_DIR,
        sonnet_judge=judge,
        gen_concurrency=24,
        judge_concurrency=24,
    )
    print(f"\nDONE: {out_path}")
    print("\nNext step: run the stratified bootstrap on this file and check that")
    print("the peak advantage at budget=15 lands within ~3 pp of +32.5%.")
    print(f"  python bootstrap.py {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
