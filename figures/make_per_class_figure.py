"""
make_per_class_figure.py

Generates the per-CHO-class accuracy figure (replacement for Table 4 in
the paper). 2x2 faceted grouped bar chart, one panel per CHO class.

Reads from evaluation/results/mode3/<auditor>/per_example.json for the
seven auditors reported in main_v4.tex and writes a PDF (vector format
for LaTeX inclusion) to figures/per_class_accuracy.pdf.

Usage:
    python figures/make_per_class_figure.py

Re-run any time the underlying mode3 results change.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "mode3"
OUT_PATH = REPO_ROOT / "figures" / "per_class_accuracy.pdf"

# Reported auditors (parameter-ordered to match the paper's tables).
AUDITORS = [
    ("gemma4_31b",        "Gemma 31B"),
    ("nemotron3_33b",     "Nemo 33B"),
    ("qwen3.5_35b",       "Q3.5 35B"),
    ("qwen3.6_35b",       "Q3.6 35B"),
    ("gpt-oss_120b",      "gpt-oss 120B"),
    ("qwen3.5_122b",      "Q3.5 122B"),
    ("deepseek-v4-flash", "DS V4"),
]

ARMS = [
    ("no_tools",          "no_tools"),
    ("generic_knowledge", "generic"),
    ("cho_tools",         "cho_tools"),
]

# Display order and human labels for the four CHO classes shown in the paper.
CLASSES = [
    ("ForbiddenRelation",  "ForbiddenRelation (n=200)"),
    ("ThresholdViolation", "ThresholdViolation (n=54)"),
    ("InvertedRelation",   "InvertedRelation (n=36)"),
    ("FabricatedEntity",   "FabricatedEntity (n=24)"),
]

# Sequential blue palette: light gray → mid blue → dark blue.
PALETTE = {
    "no_tools":  "#cccccc",
    "generic":   "#7baaff",
    "cho_tools": "#1f4e8c",
}


def per_class_accuracy(per_example: dict) -> dict:
    """Return {(arm, class): acc} for one auditor's per_example.json."""
    out = {}
    for arm_dir, arm_label in ARMS:
        examples = per_example[arm_dir]
        by_cls = Counter()
        correct = Counter()
        for r in examples:
            gc = r.get("gold_cho_class")
            if gc is None:
                continue
            by_cls[gc] += 1
            if r.get("cho_class_pred") == gc:
                correct[gc] += 1
        for cls_key, _ in CLASSES:
            if by_cls[cls_key]:
                out[(arm_label, cls_key)] = correct[cls_key] / by_cls[cls_key]
    return out


def build_dataframe() -> pd.DataFrame:
    rows = []
    for audit_dir, audit_label in AUDITORS:
        path = RESULTS_DIR / audit_dir / "per_example.json"
        if not path.exists():
            print(f"  [warn] missing {path}; skipping")
            continue
        per = json.loads(path.read_text())
        accs = per_class_accuracy(per)
        for (arm_label, cls_key), acc in accs.items():
            rows.append({
                "auditor":   audit_label,
                "arm":       arm_label,
                "cho_class": dict(CLASSES)[cls_key],
                "accuracy":  acc,
            })
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.9)

    auditor_order = [label for _, label in AUDITORS]
    arm_order     = [label for _, label in ARMS]
    class_order   = [label for _, label in CLASSES]

    g = sns.catplot(
        data=df,
        x="auditor", y="accuracy", hue="arm",
        col="cho_class", col_order=class_order, col_wrap=2,
        order=auditor_order, hue_order=arm_order,
        kind="bar", height=2.8, aspect=1.6,
        palette=[PALETTE[a] for a in arm_order],
        legend=False,
    )

    # 100% reference line per panel + cosmetics
    for ax in g.axes.flat:
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_ylim(0.5, 1.04)
        ax.set_xlabel("")
        ax.set_ylabel("Per-class accuracy")
        ax.set_title(ax.get_title().replace("cho_class = ", ""), fontsize=10)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    # Shared legend at top
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[a]) for a in arm_order
    ]
    g.fig.legend(
        handles, arm_order,
        loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        frameon=False, fontsize=10,
    )

    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.90)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Wrote {out_path}")


def main():
    df = build_dataframe()
    if df.empty:
        raise SystemExit("No data found; check evaluation/results/mode3/.")
    print(f"Built dataframe with {len(df)} rows "
          f"({df['auditor'].nunique()} auditors x "
          f"{df['arm'].nunique()} arms x "
          f"{df['cho_class'].nunique()} classes)")
    make_figure(df, OUT_PATH)


if __name__ == "__main__":
    main()
