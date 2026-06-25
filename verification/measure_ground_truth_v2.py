"""
Ground truth measurement for the expanded 30-item corpus (all 5+ sentences).
Reuses MVP results for the 10 original items, measures the 20 new ones.
Output: ground_truth_results_v2.json
"""

import json
import os
import random
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from new_corpus import corpus, KNOWN_INDICES, NOVEL_INDICES, MIXED_INDICES

load_dotenv()
api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
assert api_key, "No API key found"

client = anthropic.Anthropic(api_key=api_key)
MODEL_ID = "claude-haiku-4-5-20251001"
TEMPERATURE = 0
COMPRESSION_LEVELS = [1.0, 0.75, 0.50, 0.25, 0.0]
RUNS_PER_LEVEL = 3
SEED = 42
random.seed(SEED)

OUT_FILE = Path(__file__).parent / "ground_truth_results_v2.json"
MVP_RESULTS_FILE = Path(__file__).resolve().parent.parent / "mvp" / "mvp_raw_results.json"

MVP_ITEM_NAMES = [
    "Sabine Hossenfelder", "Python decorators", "Marie Curie", "Knitting basics",
    "Silvermoor game postmortem", "Orenbrook dig", "Zephyr sprint review",
    "Kestrel Green cooperative", "MerchantPath fraud GNN", "Brambleweed Rust runtime",
]


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def compress_content(content, level):
    if level >= 1.0:
        return content
    if level <= 0.0:
        return ""
    sentences = split_sentences(content)
    n_keep = max(1, round(len(sentences) * level))
    kept = sorted(random.sample(range(len(sentences)), n_keep))
    return " ".join(sentences[i] for i in kept)


def score_response(response_text, expected_facts):
    response_lower = response_text.lower()
    found = sum(1 for fact in expected_facts if fact.lower() in response_lower)
    return found / len(expected_facts)


def query_model(content, query):
    if content:
        system_msg = f"You have the following information:\n\n{content}\n\nAnswer the question based on this information."
    else:
        system_msg = "Answer the following question to the best of your ability."
    response = client.messages.create(
        model=MODEL_ID, max_tokens=300, temperature=TEMPERATURE,
        system=system_msg,
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text


def measure_item(item, item_idx):
    print(f"  [{item_idx:2d}] {item.name} ({item.category})")
    curves = {}
    for level in COMPRESSION_LEVELS:
        scores = []
        for run in range(RUNS_PER_LEVEL):
            compressed = compress_content(item.content, level)
            response = query_model(compressed, item.query)
            score = score_response(response, item.expected_facts)
            scores.append(score)
            time.sleep(0.3)
        mean_score = sum(scores) / len(scores)
        curves[str(level)] = {"mean": round(mean_score, 4), "scores": [round(s, 4) for s in scores]}
        print(f"       level={level:.2f}: mean={mean_score:.3f}")
    return {"name": item.name, "category": item.category, "index": item_idx, "curves": curves}


def load_mvp_results():
    if not MVP_RESULTS_FILE.exists():
        return {}
    with open(MVP_RESULTS_FILE) as f:
        mvp_data = json.load(f)
    mvp_mapped = {}
    for i, mvp_result in enumerate(mvp_data["results"]):
        mvp_name = MVP_ITEM_NAMES[i]
        for j, item in enumerate(corpus):
            if item.name == mvp_name:
                mvp_mapped[j] = {
                    "name": item.name, "category": item.category,
                    "index": j, "curves": mvp_result["curves"],
                    "source": "mvp_raw_results.json",
                }
                break
    print(f"Loaded {len(mvp_mapped)} existing MVP results")
    return mvp_mapped


def main():
    print(f"=== Ground Truth Measurement v2 (expanded corpus) ===")
    print(f"Model: {MODEL_ID}")
    print(f"Corpus: {len(corpus)} items (all 5+ sentences)")
    print()

    test = client.messages.create(
        model=MODEL_ID, max_tokens=10,
        messages=[{"role": "user", "content": "ping"}],
    )
    print(f"API connection OK: {MODEL_ID}")

    existing = load_mvp_results()

    results = {}
    new_count = 0
    for i, item in enumerate(corpus):
        if i in existing:
            results[i] = existing[i]
            print(f"  [{i:2d}] {item.name} -- reusing MVP result")
        else:
            results[i] = measure_item(item, i)
            new_count += 1

    print(f"\nMeasured {new_count} new items, reused {len(existing)} MVP results")

    output = {
        "model": MODEL_ID,
        "compression_levels": COMPRESSION_LEVELS,
        "runs_per_level": RUNS_PER_LEVEL,
        "seed": SEED,
        "corpus_size": len(corpus),
        "corpus_version": "v2_expanded_5plus_sentences",
        "results": [results[i] for i in range(len(corpus))],
    }
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")

    print("\n=== Summary ===")
    for cat, indices in [("known", KNOWN_INDICES), ("novel", NOVEL_INDICES), ("mixed", MIXED_INDICES)]:
        m100 = [results[i]["curves"]["1.0"]["mean"] for i in indices]
        m25 = [results[i]["curves"]["0.25"]["mean"] for i in indices]
        avg100 = sum(m100) / len(m100)
        avg25 = sum(m25) / len(m25)
        print(f"  {cat:6s}: ret@1.0={avg100:.3f}  ret@0.25={avg25:.3f}  drop={avg100-avg25:.3f}")


if __name__ == "__main__":
    main()
