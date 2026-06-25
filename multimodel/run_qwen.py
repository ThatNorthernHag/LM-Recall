"""Phase 2.2 local-model run: Qwen 3.5 (~397B).

BEFORE YOU RUN:
  1. Load Qwen 3.5 into your local server at [INSERT YOUR LOCAL SERVER URL HERE].
  2. If the server expects a specific `model` string in chat completions,
     edit the `model_id=` argument below. The default placeholder
     "local-model" works on most stacks (vLLM, llama.cpp server, Ollama)
     that ignore the field when only one model is loaded.
  3. Run this script.
  4. While the script is running, do NOT touch the local server. When the
     script finishes, swap to the next model and run the next script.

Output: multimodel/results/mvp_raw_results_qwen_397b_v3_{ts}.json
The script will refuse to overwrite an existing file. Wait one minute
(timestamp ticks) or pass --suffix to add a disambiguator if you need
to re-run within the same minute.
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
    name="qwen-3.5-397b",
    family="qwen",
    scale="397B",
    model_id="local-model",   # EDIT IF YOUR SERVER NEEDS A SPECIFIC ID
    base_url="[INSERT YOUR LOCAL SERVER URL HERE]",
    max_tokens=2048,          # Qwen 3.x is a thinking model — 300 leaves nothing
    extra_body={              # If your stack supports it, this suppresses thinking:
        # "enable_thinking": False,
        # "chat_template_kwargs": {"enable_thinking": False},
    },
)


def _print_preflight():
    print("=" * 65)
    print("PRE-FLIGHT CHECK — read before pressing Enter")
    print("=" * 65)
    print()
    print("This script will hit the local server expecting Qwen 3.5 (397B)")
    print(f"  endpoint:  {DECODER_KWARGS['base_url']}")
    print(f"  model_id:  {DECODER_KWARGS['model_id']}")
    print()
    print("Make sure the server has Qwen 3.5 loaded, then press Enter to start.")
    print("Press Ctrl-C to cancel.")
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
        gen_concurrency=6,         # conservative for local server
        judge_concurrency=24,
        force_suffix=suffix,
    )
    print(f"\nDONE: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yes", action="store_true",
                   help="skip the pre-flight Enter-to-start prompt")
    p.add_argument("--suffix", default="",
                   help="extra tag appended to output filename (e.g. 'retry')")
    args = p.parse_args()
    asyncio.run(main(skip_preflight=args.yes, suffix=args.suffix))
