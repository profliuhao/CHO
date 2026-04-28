"""
run_evaluation.py: end-to-end CHO evaluation on Med-HALT FCT.

Pipeline:
  1. Load Med-HALT FCT examples (HuggingFace or fixture).
  2. For each example, extract claims from the (question + proposed answer).
  3. Compute CHO metrics: OHR, RIS, wHAL.
  4. Compute baseline: ROUGE-L (1 - F1 between question and answer).
  5. Aggregate per-example scores into a results table.
  6. Compute discriminative validity via AUROC of each metric against
     the ground-truth `is_correct` label (False = hallucination class).
  7. Write results to evaluation/results/.

The output is two files:
  - results.json:   per-example scores
  - summary.json:   aggregate statistics, including AUROC, mean / std
                    per metric per class, and confusion-style counts.
  - results_table.md:  human-readable Markdown summary
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# allow imports from sibling packages without installing
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from extraction.claim_extraction import extract_claims
from metrics.ohr import compute_ohr
from metrics.ris import compute_ris
from metrics.whal import compute_whal
from baselines.rouge_baseline import rouge_l_lexical_overlap_score

import medhalt_loader
import medhallu_loader

# `loader` is rebound in main() based on --benchmark; default keeps the
# historical Med-HALT behavior for any code that imports this module.
loader = medhalt_loader


# =====================================================
# AUROC (no sklearn dependency)
# =====================================================

def auroc(scores: list[float], labels: list[int]) -> float:
    """Compute the area under the ROC curve.

    `scores`: a list of metric values, higher = predicted positive class
    `labels`: a list of binary labels (0 or 1), 1 = positive class

    Uses the pair-counting formulation: AUROC = P(score(pos) > score(neg)).
    Handles ties as 0.5 contributions. Returns NaN if either class
    is empty."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
            total += 1
    return wins / total


# =====================================================
# Per-example scoring
# =====================================================

def score_example(ex) -> dict:
    # `ex` is duck-typed: must expose qid, question, proposed_answer,
    # is_correct, composed_text. Both medhalt_loader.FCTExample and
    # medhallu_loader.MedHalluExample satisfy this.
    text = ex.composed_text
    claims = extract_claims(text)

    ohr = compute_ohr(claims)
    ris = compute_ris(claims)
    whal = compute_whal(claims, ohr_result=ohr, ris_result=ris)
    rouge = rouge_l_lexical_overlap_score(ex.question, ex.proposed_answer)

    out = {
        "qid": ex.qid,
        "is_correct": ex.is_correct,
        "is_hallucination": int(not ex.is_correct),
        "n_entities": len(claims.entities),
        "n_relations": len(claims.relations),
        "n_numerics": len(claims.numerics),
        "ohr_score": _nan_safe(ohr.score),
        "ohr_coverage": ohr.coverage,
        "ohr_violations": ohr.violations,
        "ohr_verifiable": ohr.verifiable,
        "ris_score": _nan_safe(ris.score),
        "ris_inverted": ris.inverted,
        "ris_total": ris.total_relations,
        "whal_score": whal.score,
        "whal_counts": whal.counts,
        "whal_axioms": whal.detail_by_axiom,
        "rouge_l_divergence": rouge,
    }
    # Optional MedHallu metadata (None on FCT examples)
    for opt in ("source_qid", "category", "difficulty"):
        if hasattr(ex, opt):
            out[opt] = getattr(ex, opt)
    return out


def _nan_safe(x):
    return None if (x is None or x != x) else x


# =====================================================
# Aggregation
# =====================================================

def aggregate(results: list[dict]) -> dict:
    """Compute summary statistics across all examples."""
    n = len(results)
    n_pos = sum(1 for r in results if r["is_hallucination"])
    n_neg = n - n_pos

    summary = {
        "n_examples": n,
        "n_hallucinated": n_pos,
        "n_correct": n_neg,
        "metrics": {},
    }

    metrics = ["ohr_score", "ris_score", "whal_score", "rouge_l_divergence"]
    for m in metrics:
        # Replace NaN with 0 for AUROC purposes only (NaN means metric
        # had no coverage on this example, which is informative as a
        # 'no signal' value rather than positive evidence of hallucination)
        scores = [r[m] if r[m] is not None else 0.0 for r in results]
        labels = [r["is_hallucination"] for r in results]

        positives = [s for s, l in zip(scores, labels) if l == 1]
        negatives = [s for s, l in zip(scores, labels) if l == 0]

        summary["metrics"][m] = {
            "auroc": auroc(scores, labels),
            "mean_on_hallucinated": (statistics.mean(positives)
                                     if positives else None),
            "mean_on_correct": (statistics.mean(negatives)
                                if negatives else None),
            "stdev_on_hallucinated": (statistics.stdev(positives)
                                      if len(positives) >= 2 else None),
            "stdev_on_correct": (statistics.stdev(negatives)
                                 if len(negatives) >= 2 else None),
            "n_with_signal": sum(1 for s in scores if s > 0),
        }

    # Coverage stats
    coverages = [r["ohr_coverage"] for r in results]
    summary["coverage"] = {
        "mean": statistics.mean(coverages) if coverages else 0.0,
        "fraction_with_any_verifiable_claim": (
            sum(1 for r in results if r["ohr_verifiable"] > 0) / n if n else 0.0
        ),
    }

    # Axiom breakdown
    axiom_counts = {}
    for r in results:
        for axiom, c in (r.get("whal_axioms") or {}).items():
            axiom_counts[axiom] = axiom_counts.get(axiom, 0) + c
    summary["axiom_violation_counts"] = axiom_counts

    return summary


# =====================================================
# Markdown report
# =====================================================

def write_markdown(summary: dict, out_path: Path):
    n = summary["n_examples"]
    n_pos = summary["n_hallucinated"]
    n_neg = summary["n_correct"]
    bench_label = summary.get("benchmark", "Med-HALT FCT")

    lines = []
    lines.append(f"# CHO Evaluation Results on {bench_label}")
    lines.append("")
    lines.append(f"- Examples evaluated: **{n}**")
    lines.append(f"- Hallucinated (label=incorrect): **{n_pos}**")
    lines.append(f"- Correct: **{n_neg}**")
    lines.append(f"- Mean ontology coverage per example: "
                 f"**{summary['coverage']['mean']:.2%}**")
    lines.append(f"- Examples with at least one verifiable claim: "
                 f"**{summary['coverage']['fraction_with_any_verifiable_claim']:.2%}**")
    lines.append("")

    lines.append("## Discriminative validity (AUROC, hallucinated vs correct)")
    lines.append("")
    lines.append("| Metric | AUROC | Mean (hallucinated) | Mean (correct) | Examples with signal |")
    lines.append("|---|---:|---:|---:|---:|")
    metric_labels = {
        "ohr_score": "OHR (CHO)",
        "ris_score": "RIS (CHO)",
        "whal_score": "wHAL (CHO)",
        "rouge_l_divergence": "ROUGE-L 1-F1 (baseline)",
    }
    for m, label in metric_labels.items():
        s = summary["metrics"][m]
        a = s["auroc"]
        a_str = f"{a:.3f}" if a == a else "N/A"
        mp = s["mean_on_hallucinated"]
        mn = s["mean_on_correct"]
        mp_str = f"{mp:.3f}" if mp is not None else "N/A"
        mn_str = f"{mn:.3f}" if mn is not None else "N/A"
        lines.append(f"| {label} | {a_str} | {mp_str} | {mn_str} | {s['n_with_signal']}/{n} |")
    lines.append("")

    if summary["axiom_violation_counts"]:
        lines.append("## CHO axiom violations detected")
        lines.append("")
        lines.append("| Axiom | Count |")
        lines.append("|---|---:|")
        for axiom, count in sorted(summary["axiom_violation_counts"].items()):
            lines.append(f"| {axiom} | {count} |")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


# =====================================================
# CLI
# =====================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",
                        choices=["medhalt", "medhallu"],
                        default="medhalt",
                        help="which benchmark loader to use")
    parser.add_argument("--medhallu-config",
                        choices=["pqa_labeled", "pqa_artificial"],
                        default="pqa_labeled",
                        help="MedHallu HF config (only used when "
                             "--benchmark medhallu)")
    parser.add_argument("--medhallu-category",
                        action="append", default=None,
                        help="Filter to specific Category of Hallucination "
                             "(repeatable). E.g. "
                             "--medhallu-category 'Mechanism and Pathway "
                             "Misattribution'")
    parser.add_argument("--medhallu-difficulty",
                        action="append", default=None,
                        choices=["easy", "medium", "hard"],
                        help="Filter to specific Difficulty Level "
                             "(repeatable).")
    parser.add_argument("--source", choices=["auto", "huggingface", "fixture"],
                        default="auto",
                        help="auto: try HF, fall back to fixture")
    parser.add_argument("-n", type=int, default=None,
                        help="limit number of source rows (default: all). "
                             "For MedHallu each row yields 2 examples "
                             "(correct + hallucinated).")
    parser.add_argument("--results-dir", type=str,
                        default=str(HERE / "results"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    global loader
    if args.benchmark == "medhallu":
        loader = medhallu_loader
        examples = loader.load(source=args.source, n=args.n,
                               config=args.medhallu_config,
                               categories=args.medhallu_category,
                               difficulties=args.medhallu_difficulty)
        bench_label = f"MedHallu ({args.medhallu_config})"
        if args.medhallu_category:
            bench_label += " | cat=[" + ", ".join(args.medhallu_category) + "]"
        if args.medhallu_difficulty:
            bench_label += " | diff=[" + ", ".join(args.medhallu_difficulty) + "]"
    else:
        loader = medhalt_loader
        examples = loader.load(source=args.source, n=args.n)
        bench_label = "Med-HALT FCT"

    if not args.quiet:
        print(f"Loaded {len(examples)} {bench_label} examples.")
        print("Scoring with CHO metrics (OHR, RIS, wHAL) and ROUGE-L baseline...")

    results = [score_example(ex) for ex in examples]
    summary = aggregate(results)
    summary["benchmark"] = bench_label

    out_dir = Path(args.results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    write_markdown(summary, out_dir / "results_table.md")

    if not args.quiet:
        print()
        print("Discriminative validity (AUROC, higher = better separation):")
        for m in ("ohr_score", "ris_score", "whal_score", "rouge_l_divergence"):
            a = summary["metrics"][m]["auroc"]
            mp = summary["metrics"][m]["mean_on_hallucinated"]
            mn = summary["metrics"][m]["mean_on_correct"]
            mp_s = f"{mp:.3f}" if mp is not None else "N/A"
            mn_s = f"{mn:.3f}" if mn is not None else "N/A"
            a_s = f"{a:.3f}" if a == a else "N/A"
            print(f"  {m:24s}  AUROC={a_s}  hall={mp_s}  corr={mn_s}")
        print()
        print(f"Per-example results:  {out_dir / 'results.json'}")
        print(f"Aggregate summary:    {out_dir / 'summary.json'}")
        print(f"Markdown report:      {out_dir / 'results_table.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
