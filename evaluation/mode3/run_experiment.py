"""
run_experiment.py

Mode 3 experiment orchestrator. Runs the auditor across a benchmark
sample × four arms × one or more LLMs, then aggregates per-arm metrics.

Pilot (default n=50): one auditor, four arms = 200 LLM calls.
Full (n=500ish): one or more auditors, four arms.

Usage (from the user's machine, with the local LLM running on
localhost:29999):

    OPENAI_BASE_URL=http://localhost:29999/v1 \\
    OPENAI_API_KEY="$LOCAL_LLM_API_KEY" \\
    python evaluation/mode3/run_experiment.py \\
        --auditor-model qwen3.5-122b \\
        --pilot

For the full run:
    OPENAI_BASE_URL=http://localhost:29999/v1 \\
    OPENAI_API_KEY=... \\
    python evaluation/mode3/run_experiment.py \\
        --auditor-model qwen3.5-122b \\
        --full --n 500

When the second auditor endpoint is available (placeholder for the
weaker-tier auditor that the scalable-oversight framing wants), pass
--auditor-2-base-url and --auditor-2-api-key and --auditor-2-model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
EVAL_ROOT = HERE.parent
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_ROOT))

from mode3.auditor import (
    audit_one,
    AuditResult,
    OpenAICompatibleClient,
    MockLLMClient,
)


GOLD_PATH = REPO_ROOT / "evaluation" / "mode2" / "gold_labels.json"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "mode3"

ARMS = ["no_tools", "generic_knowledge", "cho_tools", "cho_ablated"]


# =====================================================
# Sampling
# =====================================================

def sample_examples(gold_path: Path, n: int,
                    seed: int = 42) -> list[dict]:
    """Take a stratified sample from the Mode 2 gold set: balance
    hallucinated vs negative-control, and within hallucinated balance
    by expected_class so each arm sees the same per-class distribution."""
    import random
    rng = random.Random(seed)
    gold = json.loads(gold_path.read_text())
    examples = gold["examples"]

    # Bucket by class
    by_class: dict = defaultdict(list)
    for e in examples:
        by_class[e["expected_class"]].append(e)

    # Target: roughly equal per class (with negative_control half the
    # total to ensure healthy specificity check)
    n_neg = max(1, n // 4)
    n_per_pos_class = max(1, (n - n_neg) // max(1, len(by_class) - 1))

    sample: list[dict] = []
    # Negative controls
    if None in by_class:
        sample += rng.sample(by_class[None],
                             min(n_neg, len(by_class[None])))
    # Hallucination classes
    for cls, items in by_class.items():
        if cls is None:
            continue
        sample += rng.sample(items, min(n_per_pos_class, len(items)))

    # Top-up: when small classes cap out (e.g. ModifierOmission has 1 ex),
    # the stratified pass falls short of n. Refill from the largest
    # non-exhausted hallucination pools so we hit the requested n.
    if len(sample) < n:
        already = {id(e) for e in sample}
        donors: list[dict] = []
        for cls, items in by_class.items():
            if cls is None:
                continue
            extras = [e for e in items if id(e) not in already]
            donors.extend(extras)
        rng.shuffle(donors)
        sample.extend(donors[: n - len(sample)])

    rng.shuffle(sample)
    return sample[:n]


# =====================================================
# Per-arm runs
# =====================================================

def run_arm(client, examples: list[dict], arm: str,
            verbose: bool = False) -> list[dict]:
    out = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        r = audit_one(client, ex["text"], arm,
                      example_id=ex["id"])
        # Attach gold for downstream comparison
        rec = r.to_dict()
        rec["gold_is_hallucination"] = ex.get("is_hallucination", True)
        rec["gold_cho_class"] = ex.get("expected_class")
        rec["gold_source"] = ex.get("source")
        out.append(rec)
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{arm}] {i+1}/{len(examples)} "
                  f"({rate:.1f} examples/s)", file=sys.stderr)
    return out


# =====================================================
# Aggregation
# =====================================================

def aggregate(per_example_by_arm: dict[str, list[dict]]) -> dict:
    """Compute per-arm metrics + within-subjects deltas across arms."""
    summary: dict = {"per_arm": {}, "deltas": {}}

    for arm, recs in per_example_by_arm.items():
        n = len(recs)
        n_completed = sum(1 for r in recs if r.get("error") is None)

        # Hallucination detection
        tp = sum(1 for r in recs if r["is_hallucination_pred"] is True
                 and r["gold_is_hallucination"] is True)
        fp = sum(1 for r in recs if r["is_hallucination_pred"] is True
                 and r["gold_is_hallucination"] is False)
        fn = sum(1 for r in recs if r["is_hallucination_pred"] is False
                 and r["gold_is_hallucination"] is True)
        tn = sum(1 for r in recs if r["is_hallucination_pred"] is False
                 and r["gold_is_hallucination"] is False)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec == prec and rec == rec and (prec + rec) else float("nan"))
        acc = (tp + tn) / n if n else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")

        # CHO class accuracy on positive examples
        cho_class_correct = sum(
            1 for r in recs
            if r["gold_is_hallucination"] is True
            and r["cho_class_pred"] == r["gold_cho_class"]
        )
        n_positive = sum(1 for r in recs if r["gold_is_hallucination"] is True)
        cho_class_acc = (cho_class_correct / n_positive
                         if n_positive else float("nan"))

        # Tool-use stats
        n_with_tool_calls = sum(1 for r in recs if r["tool_calls"])
        avg_tool_calls = (sum(len(r["tool_calls"]) for r in recs) / n
                          if n else 0)
        tool_call_breakdown = Counter()
        for r in recs:
            for tc in r["tool_calls"]:
                tool_call_breakdown[tc["name"]] += 1

        # Errors
        errors = Counter()
        for r in recs:
            if r.get("error"):
                errors[r["error"]] += 1

        summary["per_arm"][arm] = {
            "n": n,
            "n_completed": n_completed,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "specificity": spec,
            "cho_class_accuracy_on_hallucinated": cho_class_acc,
            "tool_use": {
                "examples_using_tools": n_with_tool_calls,
                "fraction_using_tools": n_with_tool_calls / n if n else 0,
                "avg_tool_calls_per_example": avg_tool_calls,
                "by_tool": dict(tool_call_breakdown),
            },
            "errors": dict(errors),
        }

    # Within-subjects deltas: cho_tools vs no_tools, cho_tools vs generic
    if "cho_tools" in per_example_by_arm and "no_tools" in per_example_by_arm:
        a = summary["per_arm"]["cho_tools"]
        b = summary["per_arm"]["no_tools"]
        summary["deltas"]["cho_vs_baseline_acc"] = a["accuracy"] - b["accuracy"]
        summary["deltas"]["cho_vs_baseline_f1"] = a["f1"] - b["f1"]
    if "cho_tools" in per_example_by_arm and "generic_knowledge" in per_example_by_arm:
        a = summary["per_arm"]["cho_tools"]
        b = summary["per_arm"]["generic_knowledge"]
        summary["deltas"]["cho_vs_generic_acc"] = a["accuracy"] - b["accuracy"]
        summary["deltas"]["cho_vs_generic_f1"] = a["f1"] - b["f1"]
    if "cho_tools" in per_example_by_arm and "cho_ablated" in per_example_by_arm:
        a = summary["per_arm"]["cho_tools"]
        b = summary["per_arm"]["cho_ablated"]
        summary["deltas"]["reasoner_contribution_acc"] = a["accuracy"] - b["accuracy"]
        summary["deltas"]["reasoner_contribution_f1"] = a["f1"] - b["f1"]

    return summary


def write_markdown(summary: dict, out_path: Path):
    lines = ["# Mode 3 Experiment: LLM Auditor with CHO Tools", ""]
    lines.append("## Per-arm metrics")
    lines.append("")
    lines.append("| Arm | n | Acc | F1 | Precision | Recall | Specificity | "
                 "CHO-class acc (on hall) | % using tools | avg tool calls |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    def fmt(x):
        if x is None:
            return "N/A"
        try:
            f = float(x)
            return "N/A" if f != f else f"{f:.3f}"
        except (TypeError, ValueError):
            return str(x)

    def fmt_pct(x):
        if x is None:
            return "N/A"
        try:
            f = float(x)
            return "N/A" if f != f else f"{f:.1%}"
        except (TypeError, ValueError):
            return str(x)

    for arm, m in summary["per_arm"].items():
        tu = m.get("tool_use", {})
        lines.append(
            f"| **{arm}** | {m.get('n', 'N/A')} | {fmt(m.get('accuracy'))} | "
            f"{fmt(m.get('f1'))} | {fmt(m.get('precision'))} | "
            f"{fmt(m.get('recall'))} | {fmt(m.get('specificity'))} | "
            f"{fmt(m.get('cho_class_accuracy_on_hallucinated'))} | "
            f"{fmt_pct(tu.get('fraction_using_tools'))} | "
            f"{fmt(tu.get('avg_tool_calls_per_example'))} |"
        )
    lines.append("")

    if summary.get("deltas"):
        lines.append("## Within-subjects deltas")
        lines.append("")
        for k, v in summary["deltas"].items():
            sign = "+" if v >= 0 else ""
            lines.append(f"- **{k}**: {sign}{v:.3f}")
        lines.append("")

    # Tool-call breakdown
    lines.append("## Tool-call breakdown (CHO arms)")
    lines.append("")
    for arm in ("cho_tools", "cho_ablated"):
        if arm in summary["per_arm"]:
            tu = summary["per_arm"][arm]["tool_use"]
            lines.append(f"### {arm}")
            for tool, count in sorted(tu["by_tool"].items(),
                                      key=lambda kv: -kv[1]):
                lines.append(f"  - `{tool}`: {count}")
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


# =====================================================
# Main
# =====================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--auditor-model", required=False,
                   help="Model name for the primary auditor "
                        "(e.g. qwen3.5-122b). Required unless --mock.")
    p.add_argument("--auditor-base-url",
                   default=os.environ.get("OPENAI_BASE_URL"),
                   help="OpenAI-compatible base URL "
                        "(default: $OPENAI_BASE_URL).")
    p.add_argument("--auditor-api-key",
                   default=os.environ.get("OPENAI_API_KEY"),
                   help="API key (default: $OPENAI_API_KEY).")
    p.add_argument("--auditor-2-model", default=None,
                   help="Optional second auditor (weaker tier) for "
                        "scalable-oversight comparisons.")
    p.add_argument("--auditor-2-base-url", default=None)
    p.add_argument("--auditor-2-api-key", default=None)
    p.add_argument("--gold", default=str(GOLD_PATH))
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--n", type=int, default=50,
                   help="Number of gold examples to sample (default: 50).")
    p.add_argument("--pilot", action="store_true",
                   help="Pilot mode: n=50, single auditor (default behavior).")
    p.add_argument("--full", action="store_true",
                   help="Full mode: requires --n explicitly.")
    p.add_argument("--arms", nargs="+", default=ARMS,
                   help="Subset of arms to run (default: all four).")
    p.add_argument("--mock", action="store_true",
                   help="Use MockLLMClient (no real LLM calls).")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.full and args.n == 50:
        print("WARNING: --full with default n=50; pass --n explicitly for "
              "the full run.", file=sys.stderr)

    if not args.mock and not args.auditor_model:
        print("ERROR: --auditor-model required unless --mock.", file=sys.stderr)
        return 2

    examples = sample_examples(Path(args.gold), n=args.n, seed=args.seed)
    print(f"Sampled {len(examples)} gold examples for the experiment.")

    auditors: list[tuple[str, object]] = []
    if args.mock:
        auditors.append(("mock", MockLLMClient()))
    else:
        auditors.append((
            args.auditor_model,
            OpenAICompatibleClient(
                model=args.auditor_model,
                base_url=args.auditor_base_url,
                api_key=args.auditor_api_key,
            ),
        ))
        if args.auditor_2_model:
            auditors.append((
                args.auditor_2_model,
                OpenAICompatibleClient(
                    model=args.auditor_2_model,
                    base_url=args.auditor_2_base_url,
                    api_key=args.auditor_2_api_key,
                ),
            ))

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save sampled examples for reproducibility
    (out_dir / "sampled_examples.json").write_text(
        json.dumps(examples, indent=2))

    for auditor_name, client in auditors:
        print(f"\n=== Auditor: {auditor_name} ===")
        per_example_by_arm: dict[str, list[dict]] = {}
        for arm in args.arms:
            print(f"  Running arm '{arm}' ({len(examples)} examples)...")
            t0 = time.time()
            per_example_by_arm[arm] = run_arm(
                client, examples, arm, verbose=args.verbose
            )
            dt = time.time() - t0
            print(f"    arm '{arm}' done in {dt:.1f}s "
                  f"({len(examples)/dt:.2f} examples/s)")

        summary = aggregate(per_example_by_arm)

        # Sanitize for filesystem: '/', ':' (Ollama uses both), spaces
        safe = auditor_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        auditor_dir = out_dir / safe
        auditor_dir.mkdir(parents=True, exist_ok=True)

        # Write each artifact independently so a failure in one (e.g.,
        # markdown formatting) doesn't lose the others. Without this,
        # a crash in write_markdown leaves summary.json unwritten too.
        for label, write_fn in [
            ("per_example.json", lambda: (auditor_dir / "per_example.json")
             .write_text(json.dumps(per_example_by_arm, indent=2,
                                    default=str))),
            ("summary.json", lambda: (auditor_dir / "summary.json")
             .write_text(json.dumps(summary, indent=2, default=str))),
            ("report.md", lambda: write_markdown(
                summary, auditor_dir / "report.md")),
        ]:
            try:
                write_fn()
            except Exception as e:
                print(f"  [warn] failed to write {label}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        print(f"\n  Results for {auditor_name}:")
        for arm, m in summary["per_arm"].items():
            f1 = m["f1"]
            f1_s = f"{f1:.3f}" if f1 == f1 else "N/A"
            print(f"    {arm:20s} acc={m['accuracy']:.3f}  F1={f1_s}  "
                  f"tool_use={m['tool_use']['fraction_using_tools']:.0%}")
        if summary.get("deltas"):
            print(f"\n  Deltas:")
            for k, v in summary["deltas"].items():
                print(f"    {k}: {v:+.3f}")
        print(f"\n  Wrote {auditor_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
