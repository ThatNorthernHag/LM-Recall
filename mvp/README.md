# MVP Pilot — Observable Structure Functions in LLM Compression

This is the original proof-of-concept pilot (April 2026) that started the project and
motivated the paper *Context Selection as Approximate Sufficiency: Le Cam Deficiency and
Kolmogorov's Structure Function Against a Fixed Decoder* (Alpila 2026). It is kept here
as the runnable starting point, not as the project's current state.

**The complete study is the paper and the `multimodel/` experiment one level up.** The
pilot uses 10 items and one decoder (Claude Haiku 4.5); the full study uses 30 items
across seven frozen decoders in four families, and adds the capacity floor, the
finite-sample generalization guarantee, cross-decoder generality, and the
emulation-vs-task separation. If you want the result, read those. If you want to *see the
effect appear in six minutes for about ten cents*, run this.

- Paper (Zenodo): https://doi.org/10.5281/zenodo.20800175
- Full code and data: the `multimodel/` directory in this repository

## What the pilot shows

Compress an item's context at five levels (100 / 75 / 50 / 25 / 0 %), measure how much
of its facts survive, and the retention curves come out in three distinct shapes by
content type:

- **Known** content (saturated in training) — flat curves; the model reconstructs from
  weights even at zero context.
- **Novel** content (zero-contamination fabricated or personal facts) — steep curves;
  every sentence removed is information the model cannot recover.
- **Mixed** content (known frame, novel specifics) — steepest of all when the queried
  facts are the novel parts.

A curve-aware allocator that reads these shapes and spends budget where the curves are
steep beats uniform allocation by **+35%** on the same budget (Haiku 4.5, retaining
6.40 vs 4.73 facts). The three shapes are the empirical signature the Vereshchagin–Vitányi
staircase theorem predicts; the allocation gain is the practical consequence.

## Relationship to the full study's numbers

Pilot and paper measure on different corpora, so some headline numbers differ — expected,
not a contradiction. The pilot reports a 75% known-content zero-context floor for Haiku
(4 known items); the paper reports 67% for the same decoder (11 known items, v3 corpus).
Same decoder, different corpus, different mean. The pilot's job is to show the curve
shapes are real and measurable; the full study is where the effect is pinned down at
scale.

## Contents

| File | What it is |
|------|------------|
| `mvp_structure_function.ipynb` | The experiment notebook — self-contained; run all cells to reproduce |
| `mvp_structure_function_executed.ipynb` | The same notebook with outputs preserved |
| `mvp_report.md` | The experiment report and results write-up |
| `mvp_theory.md` | Mathematical foundations: Blackwell/Le Cam (outer), Vereshchagin–Vitányi (inner), and the approximation chain from non-computable theory to a measurable curve |
| `mvp_raw_results.json` | Raw per-item, per-level retention means |
| `structure_function_curves.png` | Retention curves by category |
| `structure_function_overlay.png` | All item curves overlaid with category averages |
| `selector_comparison.png` | Curve-aware vs uniform allocation |

## Running it

Install the repo requirements (`pip install -r ../requirements.txt`, Python 3.13+) and an
Anthropic API key in a `.env` file, then open `mvp_structure_function.ipynb` and run all
cells in order to reproduce the measurements, regenerate the plots, and write the raw
data. Sentence-sampling seeds are fixed (base seed = 42) and responses run at temperature
0, so the run is deterministic up to the random sentence sampler. Budget: ~150 API calls,
~6 minutes, a few cents on Haiku. (An executed copy of the notebook is already included,
so you can read the outputs without running anything.)

## License

Code is MIT; corpus, text, and figures are CC BY 4.0 — matching the repository this pilot
belongs to.
