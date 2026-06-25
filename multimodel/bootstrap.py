"""Stratified bootstrap CI on the curve-aware selector advantage.

Implements the stratified resampler from Appendix B of the paper,
parametrized by input JSON path. Within-category resampling preserves the
design-fixed category balance; a naive cross-corpus bootstrap is wrong for
that reason.

CLI:
    python bootstrap.py path/to/mvp_raw_results_*.json [--budget 15] [--n 8000]

Library:
    from bootstrap import run_bootstrap
    summary = run_bootstrap(path_or_dict, budget=15, n_resamples=8000)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path_or_dict) -> list[dict]:
    if isinstance(path_or_dict, (str, Path)):
        with open(path_or_dict) as f:
            d = json.load(f)
    else:
        d = path_or_dict
    # cast str-keyed curves back to float for selector arithmetic
    results = []
    for r in d["results"]:
        curves = {float(k): v for k, v in r["curves"].items()}
        results.append({
            "name": r["name"],
            "category": r["category"],
            "curves": curves,
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# Bootstrap-specific allocator (matches Appendix B of the paper).
#
# Why we don't reuse selector.py here:
#   The production allocator in selector.py keys its plan by item name
#   (preserves v2's dict-based implementation). Bootstrap resamples
#   contain duplicate items; the dict-keyed allocator silently collapses
#   them. The list-of-dict allocator below treats duplicates as separate
#   plan entries, which is what's required for resampling. The production
#   allocator must not be modified, so we keep them as two separate
#   implementations — production for sweep over
#   unique items, this one for bootstrap.
# ─────────────────────────────────────────────────────────────────────


def _get_ret(r, l):
    if l in r["curves"]:
        return r["curves"][l]["mean"]
    ls = sorted(r["curves"].keys())
    for i in range(len(ls) - 1):
        if ls[i] <= l <= ls[i + 1]:
            t = (l - ls[i]) / (ls[i + 1] - ls[i])
            return (
                r["curves"][ls[i]]["mean"]
                + t * (r["curves"][ls[i + 1]]["mean"] - r["curves"][ls[i]]["mean"])
            )
    return 0.0


def _thr(res, b):
    a = min(1.0, b / len(res))
    return sum(_get_ret(r, a) for r in res)


def _cur(res, b):
    mf = min(0.25, b / len(res))
    plan = [
        {"r": r, "alloc": mf, "drop": _get_ret(r, 1.0) - _get_ret(r, 0.25)}
        for r in res
    ]
    rem = b - mf * len(res)
    for it in sorted(plan, key=lambda x: x["drop"], reverse=True):
        if rem <= 0:
            break
        uf, uh = 1.0 - it["alloc"], 0.5 - it["alloc"]
        if it["drop"] > 0.3 and rem >= uf and uf > 0:
            it["alloc"] = 1.0
            rem -= uf
        elif it["drop"] > 0.15 and rem >= uh and uh > 0:
            it["alloc"] = 0.5
            rem -= uh
    return sum(_get_ret(it["r"], it["alloc"]) for it in plan)


def _peak_advantage(results: list[dict], budget: float) -> float:
    ta = _thr(results, budget)
    tb = _cur(results, budget)
    if ta <= 0:
        return 0.0
    return (tb - ta) / ta * 100.0


def run_bootstrap(
    path_or_dict,
    *,
    budget: float = 15.0,
    n_resamples: int = 8000,
    seed: int = 123,
    categories: tuple[str, ...] = ("known", "novel", "mixed"),
) -> dict:
    """Stratified within-category bootstrap. Returns point, CI, P(>0).

    Uses the legacy np.random.seed/randint scheme from Appendix B of the
    paper so the published CI numbers reproduce bit-for-bit on the same
    input.
    """
    results = _load(path_or_dict)

    point_pct = _peak_advantage(results, budget)

    cats = {c: [r for r in results if r["category"] == c] for c in categories}
    np.random.seed(seed)

    boot = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        samp = []
        for c in categories:
            pool = cats[c]
            n = len(pool)
            if n == 0:
                continue
            idxs = np.random.randint(0, n, size=n)
            samp.extend(pool[i] for i in idxs)
        boot[k] = _peak_advantage(samp, budget)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "budget": budget,
        "n_resamples": n_resamples,
        "categories": list(categories),
        "stratified": True,
        "point_pct": float(point_pct),
        "boot_mean": float(boot.mean()),
        "boot_std": float(boot.std()),
        "ci_95_low": float(lo),
        "ci_95_high": float(hi),
        "p_positive": float((boot > 0).mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path")
    p.add_argument("--budget", type=float, default=15.0)
    p.add_argument("--n", type=int, default=8000, dest="n_resamples")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    s = run_bootstrap(
        args.path,
        budget=args.budget,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    print(
        f"budget={s['budget']:.1f}  point=+{s['point_pct']:.2f}%  "
        f"boot_mean=+{s['boot_mean']:.2f}%  "
        f"95% CI [{s['ci_95_low']:+.2f}, {s['ci_95_high']:+.2f}]  "
        f"P(>0)={100 * s['p_positive']:.1f}%"
    )


if __name__ == "__main__":
    main()
