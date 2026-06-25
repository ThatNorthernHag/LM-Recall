"""Phase 3 cross-decoder aggregation.

Reads every results JSON in multimodel/results/ and produces, in a
dated subdirectory under multimodel/cross_decoder/run_{ts}/:

  - CROSS_DECODER_TABLE_{ts}.md
  - fig_cross_decoder_floors_{ts}.{pdf,png}
  - fig_cross_decoder_advantage_{ts}.{pdf,png}

Substring scoring is the headline cross-decoder column (model-independent,
free, validated to agree with Sonnet within ~1pp). The judge-scored
column is reported alongside, with the Sonnet-self-judge cell marked
with an asterisk.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import matplotlib.pyplot as plt  # noqa: E402

from bootstrap import run_bootstrap  # noqa: E402


RESULTS_DIR = THIS_DIR / "results"
CROSS_DIR = THIS_DIR / "cross_decoder"


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_all_results(results_dir: Path) -> list[dict]:
    """Load every mvp_raw_results_*_v3_*.json under results_dir."""
    paths = sorted(results_dir.glob("mvp_raw_results_*_v3_*.json"))
    out = []
    for p in paths:
        d = json.loads(p.read_text())
        out.append({"path": p, "data": d})
    return out


def category_floors(results: list[dict], scoring_key: str = "curves") -> dict[str, float]:
    """Mean retention at level 0.0 within each category."""
    cats = {"known": [], "novel": [], "mixed": []}
    for r in results:
        floor = r[scoring_key]["0.0"]["mean"]
        cats[r["category"]].append(floor)
    return {c: float(np.mean(v)) if v else float("nan") for c, v in cats.items()}


def peak_at_budget(sweep: list[dict], budget: float) -> dict:
    for s in sweep:
        if abs(s["budget"] - budget) < 1e-9:
            return s
    return {"budget": budget, "pct_improvement": float("nan"),
            "uniform_total": float("nan"), "curve_total": float("nan")}


# ─────────────────────────────────────────────────────────────────────
# Per-decoder summary
# ─────────────────────────────────────────────────────────────────────

def summarize_decoder(entry: dict, budget: float = 15.0) -> dict:
    d = entry["data"]
    md = d["decoder_metadata"]
    judge_md = d.get("judge_metadata", {})
    is_self = bool(judge_md.get("self_evaluation"))

    # Floors
    sub_floors = category_floors(d["results"], "curves_substring")
    jud_floors = category_floors(d["results"], "curves")  # judge primary curves
    sub_ratio = (
        sub_floors["known"] / sub_floors["novel"]
        if sub_floors["novel"] and sub_floors["novel"] > 1e-6
        else float("inf")
    )

    # Peak at budget (use substring sweep — model-independent)
    sub_sweep = d.get("selector_sweep_substring", [])
    jud_sweep = d.get("selector_sweep", [])
    sub_peak = peak_at_budget(sub_sweep, budget)
    jud_peak = peak_at_budget(jud_sweep, budget)

    # Bootstrap CI on each
    sub_boot = run_bootstrap_for_scoring(d, scoring="substring", budget=budget)
    jud_boot = run_bootstrap_for_scoring(d, scoring="judge", budget=budget)

    return {
        "path": str(entry["path"]),
        "name": md["name"],
        "family": md["family"],
        "scale": md["scale"],
        "self_judge": is_self,
        "substring": {
            "floors": sub_floors,
            "known_over_novel_ratio": sub_ratio,
            "peak": sub_peak,
            "bootstrap": sub_boot,
        },
        "judge": {
            "floors": jud_floors,
            "peak": jud_peak,
            "bootstrap": jud_boot,
        },
    }


def run_bootstrap_for_scoring(
    decoder_data: dict, *, scoring: str, budget: float = 15.0
) -> dict:
    """Run the Appendix-B bootstrap on either substring or judge curves."""
    curves_key = "curves_substring" if scoring == "substring" else "curves"
    shadow = {
        "results": [
            {
                "name": r["name"],
                "category": r["category"],
                "curves": r[curves_key],
            }
            for r in decoder_data["results"]
        ],
    }
    return run_bootstrap(shadow, budget=budget, n_resamples=8000, seed=123)


# ─────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────

def write_table(summaries: list[dict], out_path: Path, *, budget: float = 15.0) -> None:
    """Write CROSS_DECODER_TABLE_{ts}.md."""
    lines = []
    lines.append(f"# Cross-Decoder Table — v3 corpus, budget = {budget}")
    lines.append("")
    lines.append("Substring scoring is the cross-model headline column "
                 "(model-independent, free). The Sonnet-judge column is a "
                 "parallel semantic check. **The Sonnet-judges-Sonnet cell is "
                 "marked with `*` — self-evaluation, disclosed.**")
    lines.append("")
    lines.append("Bootstrap CIs: stratified within-category, 8000 resamples, "
                 "seed=123, identical to Appendix B of the paper.")
    lines.append("")

    # Substring table
    lines.append("## Substring scoring (headline)")
    lines.append("")
    lines.append("| family | scale | known floor | novel floor | known/novel | "
                 f"peak adv % (b={budget:g}) | 95% CI | P(>0) |")
    lines.append("|---|---|--:|--:|--:|--:|---|--:|")
    for s in summaries:
        sub = s["substring"]
        b = sub["bootstrap"]
        ratio = sub["known_over_novel_ratio"]
        ratio_str = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        lines.append(
            f"| {s['family']} | {s['scale']} | "
            f"{sub['floors']['known']:.2f} | "
            f"{sub['floors']['novel']:.2f} | "
            f"{ratio_str} | "
            f"+{sub['peak']['pct_improvement']:.1f}% | "
            f"[{b['ci_95_low']:+.1f}, {b['ci_95_high']:+.1f}] | "
            f"{100 * b['p_positive']:.0f}% |"
        )
    lines.append("")

    # Judge table
    lines.append("## Sonnet-judge scoring (parallel semantic check)")
    lines.append("")
    lines.append("| family | scale | known floor | novel floor | "
                 f"peak adv % (b={budget:g}) | 95% CI | P(>0) | note |")
    lines.append("|---|---|--:|--:|--:|---|--:|---|")
    for s in summaries:
        jud = s["judge"]
        b = jud["bootstrap"]
        note = "* self-judge" if s["self_judge"] else ""
        lines.append(
            f"| {s['family']} | {s['scale']} | "
            f"{jud['floors']['known']:.2f} | "
            f"{jud['floors']['novel']:.2f} | "
            f"+{jud['peak']['pct_improvement']:.1f}% | "
            f"[{b['ci_95_low']:+.1f}, {b['ci_95_high']:+.1f}] | "
            f"{100 * b['p_positive']:.0f}% | {note} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def fig_floors(summaries: list[dict], out_base: Path) -> None:
    """Zero-context floor per decoder, known vs novel.

    Tests the pre-registered scaling prediction: known floor should rise
    with scale (more training coverage), novel floor should stay near
    zero for every decoder. The known-novel gap should widen with scale.
    """
    families = [s["family"] for s in summaries]
    scales = [s["scale"] for s in summaries]
    labels = [f"{f}\n{sc}" for f, sc in zip(families, scales)]
    x = np.arange(len(summaries))
    known = [s["substring"]["floors"]["known"] for s in summaries]
    novel = [s["substring"]["floors"]["novel"] for s in summaries]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(summaries) + 4.0), 5.5))
    ax.plot(
        x, known, marker="o", markersize=10, linewidth=2,
        label="Known content (zero-context floor)", alpha=0.95,
    )
    ax.plot(
        x, novel, marker="s", markersize=10, linewidth=2,
        label="Novel content (zero-context floor)", alpha=0.95,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Zero-context retention", fontsize=11)
    ax.set_xlabel("Decoder (substring scoring)", fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=10, loc="best", frameon=True,
              framealpha=0.95, edgecolor="#cccccc")
    ax.set_title("Pre-registered scaling prediction:\n"
                 "known floor rises with scale, novel floor stays near zero",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_advantage(summaries: list[dict], out_base: Path, *, budget: float = 15.0) -> None:
    """Curve-aware selector peak advantage per decoder, with bootstrap CI bars."""
    families = [s["family"] for s in summaries]
    scales = [s["scale"] for s in summaries]
    labels = [f"{f}\n{sc}" for f, sc in zip(families, scales)]
    x = np.arange(len(summaries))
    points = [s["substring"]["peak"]["pct_improvement"] for s in summaries]
    lows = [s["substring"]["bootstrap"]["ci_95_low"] for s in summaries]
    highs = [s["substring"]["bootstrap"]["ci_95_high"] for s in summaries]
    err_low = [p - lo for p, lo in zip(points, lows)]
    err_high = [hi - p for p, hi in zip(points, highs)]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(summaries) + 4.0), 5.5))
    ax.bar(
        x, points,
        alpha=0.85, edgecolor="black", linewidth=0.6,
    )
    ax.errorbar(
        x, points, yerr=[err_low, err_high],
        fmt="none", ecolor="black", capsize=6, capthick=1.2,
        elinewidth=1.2, zorder=5,
    )
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f"Curve-aware improvement over uniform, % (budget = {budget:g})",
                  fontsize=11)
    ax.set_xlabel("Decoder (substring scoring; CI = stratified bootstrap)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    for xi, p in zip(x, points):
        ax.text(xi, p + 1.0, f"+{p:.1f}%", ha="center", fontweight="bold", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

# Preferred display order if all decoders are present. Anything not in
# this list falls to the end, alphabetised. Local models then API models.
_SCALE_ORDER = [
    "flash",   # DeepSeek 4 flash — smallest
    "220B",    # MiniMax 2.7
    "397B",    # Qwen 3.5
    "800Bq",   # Kimi K2.6 (quantized)
    "haiku",   # Claude Haiku 4.5
    "sonnet",  # Claude Sonnet 4.6
    "pro-3.1", # Gemini 3.1 Pro
    "5.4",     # GPT 5.4
]


def _sort_key(summary: dict) -> tuple:
    try:
        return (_SCALE_ORDER.index(summary["scale"]), summary["family"])
    except ValueError:
        return (len(_SCALE_ORDER), summary["family"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=float, default=15.0,
                   help="budget at which to report peak advantage (default 15)")
    p.add_argument("--results-dir", default=str(RESULTS_DIR),
                   help=f"directory of per-decoder JSONs (default {RESULTS_DIR})")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    entries = load_all_results(results_dir)
    if not entries:
        print(f"No mvp_raw_results_*_v3_*.json found under {results_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(entries)} decoder result file(s):")
    for e in entries:
        print(f"  {e['path'].name}")
    print()

    print("Summarising and bootstrapping ...")
    summaries = [summarize_decoder(e, budget=args.budget) for e in entries]
    summaries.sort(key=_sort_key)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_dir = CROSS_DIR / f"run_{ts}"
    if out_dir.exists():
        print(f"Refusing to overwrite {out_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True)

    table_path = out_dir / f"CROSS_DECODER_TABLE_{ts}.md"
    floors_base = out_dir / f"fig_cross_decoder_floors_{ts}"
    advantage_base = out_dir / f"fig_cross_decoder_advantage_{ts}"

    write_table(summaries, table_path, budget=args.budget)
    fig_floors(summaries, floors_base)
    fig_advantage(summaries, advantage_base, budget=args.budget)

    summary_json = out_dir / f"cross_decoder_summary_{ts}.json"
    summary_json.write_text(json.dumps({
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "budget": args.budget,
        "decoders": summaries,
    }, indent=2))

    print(f"\nWrote:")
    print(f"  {table_path}")
    print(f"  {floors_base}.pdf / .png")
    print(f"  {advantage_base}.pdf / .png")
    print(f"  {summary_json}")


if __name__ == "__main__":
    main()
