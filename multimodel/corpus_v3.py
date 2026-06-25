"""
Corpus v3 — 30-item verification corpus with Clarke item relabeled.

Difference from new_corpus.py:
  "Sci-fi reading + Clarke predictions" was in `mixed_items[7]`. At zero
  context Haiku 4.5 recalled 67% of the Clarke facts because Clarke's
  geostationary-satellite prediction is parameter-resident — its framing
  half is encyclopedic, not personal-novel. Per the contamination audit
  (2026-05-29), this item is
  relabeled to known rather than dropped, keeping N=30.

Final category split:
  Known: 11 items (original 10 + Clarke moved in)
  Novel: 10 items (unchanged)
  Mixed:  9 items (Clarke removed)

The 30 ContentItem definitions themselves are imported verbatim from
`verification/new_corpus.py` — this file does not duplicate the content,
only the categorisation decision.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

# Make verification/ importable. Resolved relative to this file, so the repo
# can live anywhere as long as verification/ sits beside multimodel/.
_VERIFICATION_DIR = Path(__file__).resolve().parent.parent / "verification"
if str(_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_DIR))

from new_corpus import (  # noqa: E402
    ContentItem,
    known_items as _known_items_v2,
    novel_items as _novel_items_v2,
    mixed_items as _mixed_items_v2,
)


CORPUS_VERSION = "v3_clarke_relabeled_known"

_CLARKE_NAME = "Sci-fi reading + Clarke predictions"

# Locate Clarke in the v2 mixed list, copy it with category="known"
_clarke_v2 = next(item for item in _mixed_items_v2 if item.name == _CLARKE_NAME)
_clarke_v3 = replace(_clarke_v2, category="known")

known_items = list(_known_items_v2) + [_clarke_v3]
novel_items = list(_novel_items_v2)
mixed_items = [item for item in _mixed_items_v2 if item.name != _CLARKE_NAME]

corpus = known_items + novel_items + mixed_items

KNOWN_INDICES = list(range(0, 11))
NOVEL_INDICES = list(range(11, 21))
MIXED_INDICES = list(range(21, 30))


if __name__ == "__main__":
    import re

    def count_sentences(text: str) -> int:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return len([p for p in parts if p.strip()])

    print(f"Corpus version: {CORPUS_VERSION}")
    print(f"Total items:    {len(corpus)}")
    print(f"  Known: {len(known_items)} (was 10, +1 from Clarke relabel)")
    print(f"  Novel: {len(novel_items)} (was 10, unchanged)")
    print(f"  Mixed: {len(mixed_items)} (was 10, -1 from Clarke relabel)")
    print()
    short = 0
    for i, item in enumerate(corpus):
        n = count_sentences(item.content)
        marker = ""
        if item.name == _CLARKE_NAME:
            marker = "   *** RELABELED FROM MIXED ***"
        if n < 5:
            short += 1
            marker += "   *** SHORT ***"
        print(f"  [{i:2d}] [{item.category:6s}] {n} sents  {item.name}{marker}")
    print()
    print(f"Items with <5 sentences: {short}")
    print(f"Items with 5+ sentences: {len(corpus) - short}")
    print()
    print(f"KNOWN_INDICES = {KNOWN_INDICES}")
    print(f"NOVEL_INDICES = {NOVEL_INDICES}")
    print(f"MIXED_INDICES = {MIXED_INDICES}")
