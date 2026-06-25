"""Canonical async run pipeline for one decoder × the v3 corpus.

`run_decoder(decoder, ...)` does, in order:

  1. Determinism check (one item, three repeats at compression=1.0).
  2. Generation pass — all (item × level × run) calls async with
     semaphore-bounded concurrency. Substring scores computed inline.
  3. Save intermediate JSON with response texts (so re-judging is always
     a re-scoring pass, never a re-run).
  4. Sonnet rescoring pass — all (item × level × run × fact) judge calls
     concurrently up to judge_concurrency.
  5. Selector sweep computed in-process from the substring + judge
     curves.
  6. Write final JSON. Output path is timestamped at run start and
     refuses to overwrite an existing file.

JSON schema extends the single-decoder MVP with `response_texts`,
`determinism_check`, `decoder_metadata`, `judge_metadata`,
`corpus_version`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from tqdm.asyncio import tqdm as tqdm_async

from backends.base import Decoder, DecoderError
from compression import (
    COMPRESSION_LEVELS,
    RUNS_PER_LEVEL,
    SEED,
    compress,
    make_rng_for_run,
)
from scoring import score_facts_sonnet, score_facts_substring
from selector import compute_sweep

log = logging.getLogger("multimodel.pipeline")

JudgeCallable = Callable[[str, str, str], Awaitable[float]]

CORPUS_VERSION = "v3_clarke_relabeled_known"
JUDGE_MODEL_NAME = "claude-sonnet-4-6"


# ─────────────────────────────────────────────────────────────────────
# Output path resolution
# ─────────────────────────────────────────────────────────────────────

def make_output_path(
    results_dir: Path,
    family: str,
    scale: str,
    timestamp: str,
    *,
    force_suffix: str = "",
) -> Path:
    """`mvp_raw_results_{family}_{scale}_v3_{timestamp}.json`, never overwrites."""
    results_dir.mkdir(parents=True, exist_ok=True)
    scale_tag = scale.replace(".", "").replace("-", "").lower()
    family_tag = family.replace(".", "").replace("-", "").lower()
    name = f"mvp_raw_results_{family_tag}_{scale_tag}_v3_{timestamp}"
    if force_suffix:
        name += f"_{force_suffix}"
    path = results_dir / f"{name}.json"
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}\n"
            f"Wait one minute (timestamp ticks) or pass force_suffix=..."
        )
    return path


# ─────────────────────────────────────────────────────────────────────
# Generation + scoring pass
# ─────────────────────────────────────────────────────────────────────

async def _one_generation(
    decoder: Decoder,
    item,
    level: float,
    run_index: int,
    sem: asyncio.Semaphore,
) -> tuple[float, int, str, float]:
    """Return (level, run_index, response_text, substring_score)."""
    rng = make_rng_for_run(run_index)
    compressed = compress(item.content, level, rng)
    async with sem:
        response_text = await decoder.generate(item.query, compressed)
    s = score_facts_substring(response_text, item.expected_facts)
    return level, run_index, response_text, s


async def _generation_pass(
    decoder: Decoder,
    corpus,
    gen_concurrency: int,
) -> list[dict]:
    """Issue all generation calls concurrently. Returns one dict per item."""
    sem = asyncio.Semaphore(gen_concurrency)

    # Build the task list, grouped by item so we can re-stitch them into
    # the per-item result dicts at the end.
    per_item_tasks: list[list] = [[] for _ in corpus]
    coros = []
    for i, item in enumerate(corpus):
        for level in COMPRESSION_LEVELS:
            for run in range(RUNS_PER_LEVEL):
                coro = _one_generation(decoder, item, level, run, sem)
                per_item_tasks[i].append((level, run, coro))
                coros.append(coro)

    log.info("generation pass: %d total calls, concurrency=%d", len(coros), gen_concurrency)

    # Run them all; tqdm just for progress.
    completed_coros = await tqdm_async.gather(
        *coros, desc=f"gen ({decoder.name})", file=sys.stderr
    )

    # Re-bin results into the per-item structure. completed_coros came back
    # in submission order, which matches our coros list, which matches the
    # i × level × run nested loop above.
    results = []
    cursor = 0
    for i, item in enumerate(corpus):
        curves_substring: dict[float, dict] = {}
        responses: dict[float, list[str]] = {}
        for level in COMPRESSION_LEVELS:
            scores = []
            texts = []
            for _run in range(RUNS_PER_LEVEL):
                lvl, run, text, s = completed_coros[cursor]
                cursor += 1
                assert lvl == level
                scores.append(s)
                texts.append(text)
            mean = sum(scores) / len(scores)
            var = sum((x - mean) ** 2 for x in scores) / len(scores)
            curves_substring[level] = {
                "mean": float(mean),
                "std": float(var ** 0.5),
                "scores": [float(x) for x in scores],
            }
            responses[level] = texts
        results.append({
            "name": item.name,
            "category": item.category,
            "curves_substring": curves_substring,
            "responses": responses,
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# Judge rescoring pass
# ─────────────────────────────────────────────────────────────────────

async def _one_judge_cell(
    judge: JudgeCallable,
    item,
    level: float,
    run_index: int,
    response_text: str,
    sem: asyncio.Semaphore,
) -> tuple[float, int, float, list[float]]:
    """Score one (item, level, run) cell via Sonnet judge over all expected facts."""
    async with sem:
        mean_score, per_fact = await score_facts_sonnet(
            judge, item.query, response_text, item.expected_facts
        )
    return level, run_index, mean_score, per_fact


async def _judge_pass(
    judge: JudgeCallable,
    corpus,
    intermediate_results: list[dict],
    judge_concurrency: int,
) -> None:
    """Compute Sonnet-judge curves in place on intermediate_results."""
    sem = asyncio.Semaphore(judge_concurrency)

    coros = []
    coro_meta = []  # parallel list: (i, level, run) per coro
    for i, item in enumerate(corpus):
        rdict = intermediate_results[i]
        for level in COMPRESSION_LEVELS:
            for run, response_text in enumerate(rdict["responses"][level]):
                coro = _one_judge_cell(judge, item, level, run, response_text, sem)
                coros.append(coro)
                coro_meta.append((i, level, run))

    log.info("judge pass: %d total calls, concurrency=%d", len(coros), judge_concurrency)
    completed = await tqdm_async.gather(*coros, desc="judge (sonnet)", file=sys.stderr)

    # Aggregate into curves per item
    judge_curves: dict[int, dict[float, list[float]]] = {
        i: {level: [None] * RUNS_PER_LEVEL for level in COMPRESSION_LEVELS}
        for i in range(len(corpus))
    }
    for (i, level, run), (lvl, run2, mean_s, _per_fact) in zip(coro_meta, completed):
        assert lvl == level and run2 == run
        judge_curves[i][level][run] = mean_s

    for i, rdict in enumerate(intermediate_results):
        curves: dict[float, dict] = {}
        for level in COMPRESSION_LEVELS:
            scores = judge_curves[i][level]
            mean = sum(scores) / len(scores)
            var = sum((x - mean) ** 2 for x in scores) / len(scores)
            curves[level] = {
                "mean": float(mean),
                "std": float(var ** 0.5),
                "scores": [float(x) for x in scores],
            }
        rdict["curves"] = curves   # judge is the primary "curves" field, mirroring the MVP


# ─────────────────────────────────────────────────────────────────────
# Determinism check
# ─────────────────────────────────────────────────────────────────────

async def _determinism_check(decoder: Decoder, corpus) -> dict:
    """One item, three uncompressed generations. Log byte-level identity."""
    item = corpus[0]
    log.info("determinism check on item: %s", item.name)
    outs = []
    for _ in range(3):
        out = await decoder.generate(item.query, item.content)
        outs.append(out)
    deterministic = outs[0] == outs[1] == outs[2]
    max_pair_diff = 0
    if not deterministic:
        # Measure approximate byte-level disagreement
        def diff(a, b):
            n = min(len(a), len(b))
            d = sum(1 for x, y in zip(a[:n], b[:n]) if x != y) + abs(len(a) - len(b))
            return d
        max_pair_diff = max(diff(outs[0], outs[1]),
                            diff(outs[0], outs[2]),
                            diff(outs[1], outs[2]))
    return {
        "deterministic": deterministic,
        "max_pairwise_byte_diff": max_pair_diff,
        "item_used": item.name,
        "n_repeats": 3,
        "notes": "" if deterministic else "responses varied across repeats; "
                                          "curves include sampling noise.",
    }


# ─────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────

async def run_decoder(
    decoder: Decoder,
    *,
    corpus,
    results_dir: Path,
    sonnet_judge: JudgeCallable | None,
    gen_concurrency: int = 24,
    judge_concurrency: int = 24,
    do_determinism_check: bool = True,
    force_suffix: str = "",
) -> Path:
    """End-to-end run for one decoder. Returns the output path."""
    start_dt = datetime.now(timezone.utc)
    timestamp = start_dt.strftime("%Y%m%d_%H%M")
    out_path = make_output_path(
        results_dir, decoder.family, decoder.scale, timestamp,
        force_suffix=force_suffix,
    )

    log.info("=" * 60)
    log.info("run_decoder: %s (family=%s scale=%s)",
             decoder.name, decoder.family, decoder.scale)
    log.info("output:      %s", out_path)
    log.info("gen_concurrency=%d  judge_concurrency=%d",
             gen_concurrency, judge_concurrency)
    log.info("=" * 60)

    t0 = time.perf_counter()

    det = None
    if do_determinism_check:
        det = await _determinism_check(decoder, corpus)
        log.info("determinism: %s", det)

    intermediate = await _generation_pass(decoder, corpus, gen_concurrency)

    # Persist intermediate response texts immediately, before judging.
    # This way if the judge pass crashes the generation work isn't lost.
    intermediate_path = out_path.with_suffix(".pregjudge.json")
    intermediate_payload = {
        "stage": "pre_judge",
        "decoder_metadata": _decoder_metadata(decoder, start_dt),
        "determinism_check": det,
        "corpus_version": CORPUS_VERSION,
        "compression_levels": list(COMPRESSION_LEVELS),
        "runs_per_level": RUNS_PER_LEVEL,
        "seed": SEED,
        "results": _serialize_results(intermediate),
    }
    intermediate_path.write_text(json.dumps(intermediate_payload, indent=2))
    log.info("wrote pre-judge intermediate: %s", intermediate_path)

    if sonnet_judge is not None:
        await _judge_pass(sonnet_judge, corpus, intermediate, judge_concurrency)
    else:
        # No judge: copy substring curves into the primary `curves` field
        # so downstream consumers always have one canonical set.
        for r in intermediate:
            r["curves"] = {lvl: dict(v) for lvl, v in r["curves_substring"].items()}

    # Selector sweep on judge curves (primary) and substring curves (secondary)
    sweep_judge = compute_sweep(intermediate)
    sweep_substring = compute_sweep(_swap_to_substring(intermediate))

    finished_dt = datetime.now(timezone.utc)
    wall = time.perf_counter() - t0

    final = {
        "decoder_metadata": _decoder_metadata(decoder, start_dt, finished_dt, wall),
        "determinism_check": det,
        "judge_metadata": {
            "model": JUDGE_MODEL_NAME if sonnet_judge is not None else None,
            "scoring": "sonnet_inline_per_decoder" if sonnet_judge is not None else "substring_only",
            "self_evaluation": _is_self_eval(decoder),
        },
        "corpus_version": CORPUS_VERSION,
        "compression_levels": list(COMPRESSION_LEVELS),
        "runs_per_level": RUNS_PER_LEVEL,
        "seed": SEED,
        "corpus_size": len(corpus),
        "results": _serialize_results(intermediate),
        "selector_sweep": sweep_judge,
        "selector_sweep_substring": sweep_substring,
    }

    out_path.write_text(json.dumps(final, indent=2))
    log.info("wrote final results: %s  (%.1fs total wall time)", out_path, wall)

    # Best-effort cleanup of intermediate file if final wrote OK
    try:
        intermediate_path.unlink()
    except OSError:
        pass

    await decoder.close()
    return out_path


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _decoder_metadata(
    decoder: Decoder,
    started_dt,
    finished_dt=None,
    wall=None,
) -> dict:
    md = {
        "name": decoder.name,
        "family": decoder.family,
        "scale": decoder.scale,
        "model_id": getattr(decoder, "model_id", None),
        "base_url": getattr(decoder, "base_url", None),
        "sampling": {
            "temperature": getattr(decoder, "temperature", None),
            "max_tokens": getattr(decoder, "max_tokens", None),
            "seed": SEED,
        },
        "started_at_iso": started_dt.isoformat(),
    }
    if finished_dt is not None:
        md["finished_at_iso"] = finished_dt.isoformat()
    if wall is not None:
        md["wall_seconds"] = wall
    return md


def _is_self_eval(decoder: Decoder) -> bool:
    """True iff the decoder is the same model as the judge (Sonnet 4.6)."""
    return getattr(decoder, "model_id", "") == JUDGE_MODEL_NAME


def _serialize_results(intermediate: Sequence[dict]) -> list[dict]:
    """Stringify dict keys and round floats for JSON friendliness."""
    out = []
    for r in intermediate:
        entry = {"name": r["name"], "category": r["category"]}
        if "curves" in r:
            entry["curves"] = {
                str(lvl): _round_curve(v) for lvl, v in r["curves"].items()
            }
        entry["curves_substring"] = {
            str(lvl): _round_curve(v) for lvl, v in r["curves_substring"].items()
        }
        entry["responses"] = {
            str(lvl): list(texts) for lvl, texts in r["responses"].items()
        }
        out.append(entry)
    return out


def _round_curve(v: dict) -> dict:
    return {
        "mean": round(float(v["mean"]), 4),
        "std": round(float(v["std"]), 4),
        "scores": [round(float(x), 4) for x in v["scores"]],
    }


def _swap_to_substring(results: list[dict]) -> list[dict]:
    """Build a shallow shadow of `results` where `curves` IS the substring curves.

    Used so compute_sweep can run a second time against substring scoring.
    """
    out = []
    for r in results:
        out.append({
            "name": r["name"],
            "category": r["category"],
            "curves": r["curves_substring"],
        })
    return out
