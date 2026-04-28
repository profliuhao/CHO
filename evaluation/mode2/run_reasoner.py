"""
run_reasoner.py

Mode 2 evaluation: run the ABox translator + HermiT reasoner on the gold
set produced by curate_gold.py, compare reasoner output to expected
classes and harm tiers, and report classification metrics.

Reports:
  - Per-class precision / recall / F1 across all CHO axiom classes
  - Confusion matrix (asserted vs predicted classes, including "no
    classification" for examples the reasoner did not assert)
  - Harm-tier accuracy (does the reasoner-inferred set of harm tiers
    contain the expected one?)
  - Per-source-bucket breakdown (synthetic_contraindication vs
    medhallu_pqa_artificial:... etc.)
  - A few worked examples for the paper

Output: evaluation/results/mode2/{summary.json,per_example.json,report.md}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
EVAL_ROOT = HERE.parent
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))

from extraction.claim_extraction import extract_claims
from mode2.abox_translator import (
    AboxTranslator,
    run_reasoner_and_read,
)


GOLD_PATH = HERE / "gold_labels.json"
OUT_DIR = REPO_ROOT / "evaluation" / "results" / "mode2"


# All CHO classes the reasoner can output (used for confusion matrix dims)
CHO_CLASSES = [
    "FabricatedEntity",
    "ForbiddenRelation",
    "InvertedRelation",
    "ThresholdViolation",
    "ModifierOmission",
    None,    # "no classification" (reasoner asserted nothing)
]


def safe_id(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)[:40]


def evaluate(examples: list[dict], quiet: bool = False) -> dict:
    translator = AboxTranslator()
    per_example: list[dict] = []

    if not quiet:
        print(f"Running Mode 2 reasoner on {len(examples)} gold examples...")

    for i, ex in enumerate(examples):
        text = ex["text"]
        gold_class = ex["expected_class"]    # may be None for negative controls
        gold_harm_tiers = set(ex["expected_harm_tiers"])

        claims = extract_claims(text)
        onto, translated = translator.translate(text, claims,
                                                source_id=safe_id(ex["id"]))

        if not translated:
            predicted_class = None
            inferred_classes = []
            inferred_harm_tiers: set[str] = set()
        else:
            inferred = run_reasoner_and_read(onto, translated)
            # If multiple individuals were asserted, choose the first
            # (synthetic gold examples produce 0 or 1; MedHallu items can
            # produce >1 — we report all but score on the most-specific).
            top = inferred[0]
            inferred_classes = sorted(set(c for r in inferred
                                          for c in r.inferred_classes
                                          if c != "Thing"))
            inferred_harm_tiers = set()
            for r in inferred:
                inferred_harm_tiers.update(r.inferred_harm_tiers)
            predicted_class = top.asserted_class

        # Harm-tier match: reasoner-inferred set should INCLUDE the
        # expected one (existential semantics — "could be T3 or T4")
        if gold_harm_tiers:
            harm_tier_match = bool(gold_harm_tiers & inferred_harm_tiers)
        else:
            # Negative control: should produce no harm-tier inference
            harm_tier_match = (len(inferred_harm_tiers) == 0)

        per_example.append({
            "id": ex["id"],
            "source": ex["source"],
            "text": text,
            "expected_class": gold_class,
            "predicted_class": predicted_class,
            "inferred_classes": inferred_classes,
            "expected_harm_tiers": sorted(gold_harm_tiers),
            "inferred_harm_tiers": sorted(inferred_harm_tiers),
            "class_match": (predicted_class == gold_class),
            "harm_tier_match": harm_tier_match,
            "is_hallucination_gold": ex.get("is_hallucination", True),
        })

    # ----- Aggregate -----
    # Per-class precision / recall / F1
    per_class: dict = {}
    for cls in [c for c in CHO_CLASSES if c is not None]:
        tp = sum(1 for r in per_example
                 if r["predicted_class"] == cls and r["expected_class"] == cls)
        fp = sum(1 for r in per_example
                 if r["predicted_class"] == cls and r["expected_class"] != cls)
        fn = sum(1 for r in per_example
                 if r["predicted_class"] != cls and r["expected_class"] == cls)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision == precision and recall == recall and
                  (precision + recall)) else float("nan"))
        per_class[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    # Confusion matrix
    confusion: dict = defaultdict(lambda: defaultdict(int))
    for r in per_example:
        confusion[r["expected_class"]][r["predicted_class"]] += 1
    confusion_dict = {
        (k or "negative_control"): dict(
            (kk or "no_assertion", vv) for kk, vv in v.items()
        )
        for k, v in confusion.items()
    }

    # Harm-tier accuracy
    n_harm_correct = sum(1 for r in per_example if r["harm_tier_match"])
    harm_tier_accuracy = n_harm_correct / len(per_example)

    # Negative-control specificity (negative controls correctly UN-classified)
    neg_total = sum(1 for r in per_example if r["expected_class"] is None)
    neg_correct = sum(1 for r in per_example
                      if r["expected_class"] is None and
                      r["predicted_class"] is None)
    specificity = neg_correct / neg_total if neg_total else float("nan")

    # Per-source-bucket breakdown
    bucket_acc: dict = defaultdict(lambda: {"n": 0, "class_match": 0,
                                            "harm_match": 0})
    for r in per_example:
        b = r["source"].split(":", 1)[0]
        bucket_acc[b]["n"] += 1
        bucket_acc[b]["class_match"] += int(r["class_match"])
        bucket_acc[b]["harm_match"] += int(r["harm_tier_match"])

    summary = {
        "n_examples": len(examples),
        "overall": {
            "class_match_rate": sum(1 for r in per_example
                                    if r["class_match"]) / len(per_example),
            "harm_tier_accuracy": harm_tier_accuracy,
            "negative_control_specificity": specificity,
        },
        "per_class": per_class,
        "confusion": confusion_dict,
        "per_source_bucket": {b: dict(v) for b, v in bucket_acc.items()},
    }

    return {"summary": summary, "per_example": per_example}


def write_markdown(summary: dict, per_example: list[dict], out_path: Path):
    n = summary["n_examples"]
    o = summary["overall"]
    lines = [
        "# Mode 2 Evaluation: Reasoner-Based CHO Classification",
        "",
        f"Examples evaluated: **{n}**",
        f"Overall class-match rate: **{o['class_match_rate']:.1%}**",
        f"Harm-tier accuracy:       **{o['harm_tier_accuracy']:.1%}**",
        f"Negative-control specificity: **{o['negative_control_specificity']:.1%}**",
        "",
        "## Per-class precision / recall / F1",
        "",
        "| Class | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cls, m in summary["per_class"].items():
        def fmt(x):
            return f"{x:.3f}" if (x == x) else "N/A"
        lines.append(f"| {cls} | {m['tp']} | {m['fp']} | {m['fn']} | "
                     f"{fmt(m['precision'])} | {fmt(m['recall'])} | "
                     f"{fmt(m['f1'])} |")

    lines += ["", "## Per-source-bucket accuracy", "",
              "| Bucket | n | class_match | harm_tier_match |",
              "|---|---:|---:|---:|"]
    for b, v in summary["per_source_bucket"].items():
        cm = v["class_match"] / v["n"]
        hm = v["harm_match"] / v["n"]
        lines.append(f"| {b} | {v['n']} | {cm:.1%} | {hm:.1%} |")

    lines += ["", "## Confusion (gold → predicted)", "",
              "| Gold ↓ / Predicted → | " + " | ".join(
                  c or "no_assert" for c in CHO_CLASSES) + " |",
              "|---" + "|---:" * len(CHO_CLASSES) + "|"]
    for g in [c for c in CHO_CLASSES if c is not None] + [None]:
        row_label = g or "negative_control"
        cells = []
        for p in CHO_CLASSES:
            v = summary["confusion"].get(g or "negative_control", {}).get(
                p or "no_assertion", 0)
            cells.append(str(v))
        lines.append(f"| **{row_label}** | " + " | ".join(cells) + " |")

    lines += ["", "## Worked examples (paper-ready)", ""]
    pos_correct = [r for r in per_example
                   if r["class_match"] and r["expected_class"]][:3]
    neg_correct = [r for r in per_example
                   if r["class_match"] and r["expected_class"] is None][:2]
    misses = [r for r in per_example if not r["class_match"]][:3]

    for tag, sample in [("Correctly classified hallucinations", pos_correct),
                        ("Correctly rejected normal claims", neg_correct),
                        ("Misclassifications", misses)]:
        lines.append(f"### {tag}")
        lines.append("")
        for r in sample:
            lines.append(f"- _{r['text']}_")
            lines.append(f"  - gold: `{r['expected_class']}` "
                         f"({', '.join(r['expected_harm_tiers']) or '—'})")
            lines.append(f"  - reasoner: `{r['predicted_class']}` "
                         f"({', '.join(r['inferred_harm_tiers']) or '—'})")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(GOLD_PATH))
    parser.add_argument("--results-dir", default=str(OUT_DIR))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text())
    examples = gold["examples"]

    result = evaluate(examples, quiet=args.quiet)
    summary, per_example = result["summary"], result["per_example"]

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_example.json").write_text(json.dumps(per_example, indent=2))
    write_markdown(summary, per_example, out_dir / "report.md")

    if not args.quiet:
        print(f"\nOverall class-match rate:        "
              f"{summary['overall']['class_match_rate']:.1%}")
        print(f"Harm-tier accuracy:              "
              f"{summary['overall']['harm_tier_accuracy']:.1%}")
        print(f"Negative-control specificity:    "
              f"{summary['overall']['negative_control_specificity']:.1%}")
        print()
        print("Per-class F1:")
        for cls, m in summary["per_class"].items():
            f1 = m["f1"]
            f1_s = f"{f1:.3f}" if (f1 == f1) else "N/A"
            print(f"  {cls:25s}  P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  F1={f1_s}  "
                  f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']})")
        print()
        print(f"Per-example: {out_dir / 'per_example.json'}")
        print(f"Summary:     {out_dir / 'summary.json'}")
        print(f"Report:      {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
