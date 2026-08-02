"""
Compute imbalance-robust Mode 3 metrics from saved predictions.

Reviewers noted that the 365-example gold set is positive-heavy
(315 hallucinations, 50 controls), so accuracy and F1 can be misleading.
This script adds prevalence-insensitive summaries without rerunning any
LLM experiments:

  - balanced accuracy = (recall + specificity) / 2
  - Matthews correlation coefficient (MCC)
  - delta over an always-hallucination baseline

Outputs:
  evaluation/results/mode3/imbalance_robust_metrics.json
  evaluation/results/mode3/imbalance_robust_metrics.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "evaluation" / "results" / "mode3"


MODEL_ORDER = [
    "gemma4_31b",
    "nemotron3_33b",
    "qwen3.5_35b",
    "qwen3.6_35b",
    "gpt-oss_120b",
    "qwen3.5_122b",
    "deepseek-v4-flash",
]

DISPLAY_NAME = {
    "gemma4_31b": "Gemma 4 31B",
    "nemotron3_33b": "Nemotron 3 33B",
    "qwen3.5_35b": "Qwen 3.5 35B",
    "qwen3.6_35b": "Qwen 3.6 35B",
    "gpt-oss_120b": "gpt-oss 120B",
    "qwen3.5_122b": "Qwen 3.5 122B",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
}


def _load_gold_counts() -> tuple[int, int]:
    gold_path = REPO_ROOT / "evaluation" / "mode2" / "gold_labels.json"
    data = json.loads(gold_path.read_text())
    examples = data["examples"]
    pos = sum(1 for e in examples if e["is_hallucination"])
    neg = sum(1 for e in examples if not e["is_hallucination"])
    return pos, neg


def _confusion_from_rates(n_pos: int, n_neg: int, recall: float,
                          specificity: float) -> dict:
    tp = recall * n_pos
    fn = n_pos - tp
    tn = specificity * n_neg
    fp = n_neg - tn
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _mcc(tp: float, fp: float, tn: float, fn: float) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denom


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom == 0 else (2 * precision * recall) / denom


def main() -> int:
    n_pos, n_neg = _load_gold_counts()
    total = n_pos + n_neg

    always_positive = {
        "accuracy": n_pos / total,
        "precision": n_pos / total,
        "recall": 1.0,
        "specificity": 0.0,
        "f1": _f1(n_pos / total, 1.0),
        "balanced_accuracy": 0.5,
        "mcc": 0.0,
    }

    rows = []
    for model in MODEL_ORDER:
        summary_path = RESULTS_ROOT / model / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        for arm, metrics in summary["per_arm"].items():
            recall = metrics["recall"]
            specificity = metrics["specificity"]
            bal_acc = (recall + specificity) / 2.0
            conf = _confusion_from_rates(n_pos, n_neg, recall, specificity)
            mcc = _mcc(**conf)
            rows.append({
                "model": model,
                "display_model": DISPLAY_NAME.get(model, model),
                "arm": arm,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "recall": recall,
                "specificity": specificity,
                "balanced_accuracy": bal_acc,
                "mcc": mcc,
                "delta_balanced_accuracy_vs_always_positive":
                    bal_acc - always_positive["balanced_accuracy"],
                "delta_mcc_vs_always_positive": mcc,
            })

    by_model = {}
    for model in MODEL_ORDER:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue
        arms = {r["arm"]: r for r in model_rows}
        if "cho_tools" in arms and "generic_knowledge" in arms:
            by_model[model] = {
                "display_model": DISPLAY_NAME.get(model, model),
                "cho_minus_generic_balanced_accuracy":
                    arms["cho_tools"]["balanced_accuracy"]
                    - arms["generic_knowledge"]["balanced_accuracy"],
                "cho_minus_generic_mcc":
                    arms["cho_tools"]["mcc"] - arms["generic_knowledge"]["mcc"],
                "cho_tools": arms["cho_tools"],
                "generic_knowledge": arms["generic_knowledge"],
            }

    out = {
        "gold_counts": {
            "hallucinated": n_pos,
            "negative_controls": n_neg,
            "total": total,
        },
        "always_hallucination_baseline": always_positive,
        "rows": rows,
        "cho_vs_generic_by_model": by_model,
    }

    json_path = RESULTS_ROOT / "imbalance_robust_metrics.json"
    md_path = RESULTS_ROOT / "imbalance_robust_metrics.md"
    json_path.write_text(json.dumps(out, indent=2))

    lines = [
        "# Imbalance-Robust Mode 3 Metrics",
        "",
        f"Gold set: {n_pos} hallucinated examples and {n_neg} negative controls.",
        "",
        "Always-hallucination baseline:",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {always_positive['accuracy']:.3f} |",
        f"| F1 | {always_positive['f1']:.3f} |",
        f"| Balanced accuracy | {always_positive['balanced_accuracy']:.3f} |",
        f"| MCC | {always_positive['mcc']:.3f} |",
        "",
        "CHO tools versus generic lookup:",
        "",
        "| Auditor | Delta balanced accuracy | Delta MCC | CHO bal. acc. | Generic bal. acc. |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        item = by_model.get(model)
        if not item:
            continue
        lines.append(
            f"| {item['display_model']} | "
            f"{item['cho_minus_generic_balanced_accuracy']:+.3f} | "
            f"{item['cho_minus_generic_mcc']:+.3f} | "
            f"{item['cho_tools']['balanced_accuracy']:.3f} | "
            f"{item['generic_knowledge']['balanced_accuracy']:.3f} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
