"""
Phase 1: Empirical ground truth measurement.

Measures compression-retention curves for all 30 corpus items using the
same protocol as the MVP (mvp_structure_function.ipynb):
  - 5 compression levels: 1.0, 0.75, 0.50, 0.25, 0.0
  - 3 runs per level (average out sentence-sampling randomness)
  - Frozen decoder: Claude Haiku 4.5
  - Scoring: keyword match (fraction of expected_facts found in response)

Outputs: verification/ground_truth_results.json

For the 10 MVP items, we also load existing results from
mvp/mvp_raw_results.json to avoid redundant API calls.
"""

import json
import os
import random
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from corpus import corpus, KNOWN_INDICES, NOVEL_INDICES, MIXED_INDICES

# ── Config ──
load_dotenv()
api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
assert api_key, "No API key found. Set CLAUDE_API_KEY in .env"

client = anthropic.Anthropic(api_key=api_key)
MODEL_ID = "claude-haiku-4-5-20251001"
TEMPERATURE = 0
COMPRESSION_LEVELS = [1.0, 0.75, 0.50, 0.25, 0.0]
RUNS_PER_LEVEL = 3
SEED = 42
random.seed(SEED)

OUT_DIR = Path(__file__).parent
OUT_FILE = OUT_DIR / "ground_truth_results.json"
MVP_RESULTS_FILE = Path(__file__).resolve().parent.parent / "mvp" / "mvp_raw_results.json"

# MVP item names (in order) — these already have ground truth
MVP_ITEM_NAMES = [
    "Sabine Hossenfelder", "Python decorators", "Marie Curie", "Knitting basics",
    "Silvermoor game postmortem", "Orenbrook dig", "Zephyr sprint review", "Kestrel Green cooperative",
    "MerchantPath fraud GNN", "Brambleweed Rust runtime",
]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple but sufficient."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def compress_content(content: str, level: float) -> str:
    """Compress by randomly removing sentences."""
    if level >= 1.0:
        return content
    if level <= 0.0:
        return ""
    sentences = split_sentences(content)
    n_keep = max(1, round(len(sentences) * level))
    kept = sorted(random.sample(range(len(sentences)), n_keep))
    return " ".join(sentences[i] for i in kept)


def score_response(response_text: str, expected_facts: list[str]) -> float:
    """Score: fraction of expected facts found in response (case-insensitive)."""
    response_lower = response_text.lower()
    found = sum(1 for fact in expected_facts if fact.lower() in response_lower)
    return found / len(expected_facts)


def query_model(content: str, query: str) -> str:
    """Send content + query to the frozen decoder."""
    if content:
        system_msg = f"You have the following information:\n\n{content}\n\nAnswer the question based on this information."
    else:
        system_msg = "Answer the following question to the best of your ability."

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=300,
        temperature=TEMPERATURE,
        system=system_msg,
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text


def measure_item(item, item_idx: int) -> dict:
    """Measure compression-retention curve for one item."""
    print(f"  [{item_idx:2d}] {item.name} ({item.category})")
    curves = {}

    for level in COMPRESSION_LEVELS:
        scores = []
        for run in range(RUNS_PER_LEVEL):
            compressed = compress_content(item.content, level)
            response = query_model(compressed, item.query)
            score = score_response(response, item.expected_facts)
            scores.append(score)
            time.sleep(0.3)  # rate limit courtesy

        mean_score = sum(scores) / len(scores)
        curves[str(level)] = {
            "mean": round(mean_score, 4),
            "scores": [round(s, 4) for s in scores],
        }
        print(f"       level={level:.2f}: mean={mean_score:.3f} scores={scores}")

    return {
        "name": item.name,
        "category": item.category,
        "index": item_idx,
        "curves": curves,
    }


def load_mvp_results() -> dict:
    """Load existing MVP results and map to our corpus indices."""
    if not MVP_RESULTS_FILE.exists():
        print("WARNING: MVP results file not found, will re-measure all items")
        return {}

    with open(MVP_RESULTS_FILE) as f:
        mvp_data = json.load(f)

    # Map MVP results to our corpus indices
    mvp_mapped = {}
    for i, mvp_result in enumerate(mvp_data["results"]):
        mvp_name = MVP_ITEM_NAMES[i]
        # Find this item in our corpus
        for j, item in enumerate(corpus):
            if item.name == mvp_name:
                mvp_mapped[j] = {
                    "name": item.name,
                    "category": item.category,
                    "index": j,
                    "curves": mvp_result["curves"],
                    "source": "mvp_raw_results.json",
                }
                break

    print(f"Loaded {len(mvp_mapped)} existing MVP results")
    return mvp_mapped


def main():
    print(f"=== Ground Truth Measurement ===")
    print(f"Model: {MODEL_ID}")
    print(f"Corpus: {len(corpus)} items")
    print(f"Compression levels: {COMPRESSION_LEVELS}")
    print(f"Runs per level: {RUNS_PER_LEVEL}")
    print()

    # Test API connection
    test = client.messages.create(
        model=MODEL_ID, max_tokens=10,
        messages=[{"role": "user", "content": "ping"}],
    )
    print(f"API connection OK: {MODEL_ID}")
    print()

    # Load existing MVP results
    existing = load_mvp_results()

    # Measure new items only
    results = {}
    new_count = 0
    for i, item in enumerate(corpus):
        if i in existing:
            results[i] = existing[i]
            print(f"  [{i:2d}] {item.name} — using existing MVP result")
        else:
            results[i] = measure_item(item, i)
            new_count += 1

    print(f"\nMeasured {new_count} new items, reused {len(existing)} MVP results")

    # Assemble output
    output = {
        "model": MODEL_ID,
        "compression_levels": COMPRESSION_LEVELS,
        "runs_per_level": RUNS_PER_LEVEL,
        "seed": SEED,
        "corpus_size": len(corpus),
        "results": [results[i] for i in range(len(corpus))],
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUT_FILE}")

    # Print summary
    print("\n=== Summary ===")
    for cat, indices in [("known", KNOWN_INDICES), ("novel", NOVEL_INDICES), ("mixed", MIXED_INDICES)]:
        means_at_100 = []
        means_at_25 = []
        for i in indices:
            r = results[i]
            means_at_100.append(r["curves"]["1.0"]["mean"])
            means_at_25.append(r["curves"]["0.25"]["mean"])
        avg_100 = sum(means_at_100) / len(means_at_100)
        avg_25 = sum(means_at_25) / len(means_at_25)
        drop = avg_100 - avg_25
        print(f"  {cat:6s}: retention@1.0={avg_100:.3f}  retention@0.25={avg_25:.3f}  drop={drop:.3f}")


if __name__ == "__main__":
    main()
