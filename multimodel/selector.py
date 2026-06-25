"""Allocator under test: uniform threshold vs curve-aware.

Ported unchanged from the single-decoder MVP.

DO NOT MODIFY the allocation logic — every decoder in the multi-model
roster runs against the same allocator, otherwise cross-decoder numbers
are not comparable.

The curve-aware allocator works in two phases:
  1. Every item gets min_frac = min(0.25, budget/N) — a minimum viable
     inclusion floor.
  2. Items are sorted by curve steepness (retention drop from 100% to
     25% compression). The steepest items are upgraded first: drop>0.3
     promotes to 1.0 (full preservation), drop>0.15 promotes to 0.5.
     Upgrades stop when remaining budget is exhausted.
"""

from __future__ import annotations

BUDGET_SWEEP = (5.0, 10.0, 15.0, 20.0, 25.0)


def get_retention(result: dict, level: float) -> float:
    """Look up or linearly interpolate retention at a given compression level."""
    if level in result["curves"]:
        return result["curves"][level]["mean"]
    levels = sorted(result["curves"].keys())
    for i in range(len(levels) - 1):
        if levels[i] <= level <= levels[i + 1]:
            t = (level - levels[i]) / (levels[i + 1] - levels[i])
            lo = result["curves"][levels[i]]["mean"]
            hi = result["curves"][levels[i + 1]]["mean"]
            return lo + t * (hi - lo)
    return 0.0


def selector_threshold(results, budget):
    alloc_per_item = min(1.0, budget / len(results))
    return [
        (r["name"], alloc_per_item, get_retention(r, alloc_per_item))
        for r in results
    ]


def selector_curve_aware(results, budget):
    min_frac = min(0.25, budget / len(results))
    allocations = []
    for r in results:
        full_ret = get_retention(r, 1.0)
        low_ret = get_retention(r, 0.25)
        drop = full_ret - low_ret
        allocations.append({"result": r, "drop": drop})

    plan = {}
    for info in allocations:
        name = info["result"]["name"]
        plan[name] = {"result": info["result"], "alloc": min_frac, "drop": info["drop"]}

    remaining = budget - min_frac * len(allocations)
    by_steepness = sorted(plan.values(), key=lambda x: x["drop"], reverse=True)

    for item in by_steepness:
        if remaining <= 0:
            break
        name = item["result"]["name"]
        upgrade_full = 1.0 - plan[name]["alloc"]
        upgrade_half = 0.5 - plan[name]["alloc"]
        if item["drop"] > 0.3 and remaining >= upgrade_full and upgrade_full > 0:
            plan[name]["alloc"] = 1.0
            remaining -= upgrade_full
        elif item["drop"] > 0.15 and remaining >= upgrade_half and upgrade_half > 0:
            plan[name]["alloc"] = 0.5
            remaining -= upgrade_half

    selected = []
    for name, info in plan.items():
        ret = get_retention(info["result"], info["alloc"])
        selected.append((name, info["alloc"], ret))
    return selected


def compute_sweep(results) -> list[dict]:
    """Run both allocators at each budget in BUDGET_SWEEP. Returns the same
    `selector_sweep` schema as the single-decoder MVP so downstream code is drop-in compatible."""
    sweep = []
    for budget in BUDGET_SWEEP:
        sel_a = selector_threshold(results, budget)
        sel_b = selector_curve_aware(results, budget)
        total_a = sum(r for _, _, r in sel_a)
        total_b = sum(r for _, _, r in sel_b)
        budget_a = sum(a for _, a, _ in sel_a)
        budget_b = sum(a for _, a, _ in sel_b)
        delta = total_b - total_a
        pct = (delta / total_a * 100) if total_a > 0 else 0.0
        sweep.append({
            "budget": budget,
            "uniform_alloc_used": budget_a,
            "curve_alloc_used": budget_b,
            "uniform_total": total_a,
            "curve_total": total_b,
            "delta": delta,
            "pct_improvement": pct,
        })
    return sweep
