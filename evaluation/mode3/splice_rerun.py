"""Splice a targeted Mode 3 re-run into the saved results and recompute summaries.

After the lab-unit fix, only the threshold examples change, so re-running all 365
examples on all seven auditors is mostly wasted compute. This merges the re-run rows
into each auditor's saved per_example.json, keyed by (arm, example_id), then
recomputes summary.json and report.md.

Summaries are recomputed with run_experiment.aggregate, the same function the runner
uses, so spliced results and freshly run results are produced by identical code. Do
not reimplement the metrics here.

Refuses to splice unless the re-run ids are a strict subset of the saved ids and the
arms match, because either mismatch means the two runs are not comparable.

Usage:
    python evaluation/mode3/splice_rerun.py \\
        --saved evaluation/results/mode3 \\
        --rerun evaluation/results/mode3_threshold_rerun \\
        --backup-suffix .prefix-units
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import aggregate, write_markdown  # noqa: E402


def splice_auditor(saved_dir: Path, rerun_dir: Path, backup_suffix: str,
                   dry_run: bool) -> tuple[bool, str]:
    saved_path = saved_dir / "per_example.json"
    rerun_path = rerun_dir / "per_example.json"
    if not rerun_path.exists():
        return False, "no re-run results"
    if not saved_path.exists():
        return False, "no saved results"

    saved = json.loads(saved_path.read_text())
    rerun = json.loads(rerun_path.read_text())

    if set(rerun) - set(saved):
        return False, f"re-run has arms not in saved: {sorted(set(rerun) - set(saved))}"

    replaced = 0
    for arm, rows in rerun.items():
        by_id = {r["example_id"]: r for r in rows}
        saved_ids = {r["example_id"] for r in saved[arm]}
        unknown = set(by_id) - saved_ids
        if unknown:
            return False, f"arm {arm}: {len(unknown)} re-run ids absent from saved"
        merged = []
        for row in saved[arm]:
            new = by_id.get(row["example_id"])
            if new is not None:
                merged.append(new)
                replaced += 1
            else:
                merged.append(row)
        saved[arm] = merged

    if dry_run:
        return True, f"would replace {replaced} rows"

    shutil.copy2(saved_path, saved_path.with_suffix(f".json{backup_suffix}"))
    saved_path.write_text(json.dumps(saved, indent=2, default=str))

    summary = aggregate(saved)
    summary_path = saved_dir / "summary.json"
    if summary_path.exists():
        shutil.copy2(summary_path, summary_path.with_suffix(f".json{backup_suffix}"))
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    try:
        write_markdown(summary, saved_dir / "report.md")
    except Exception as exc:  # report.md is a convenience, not a result
        return True, f"replaced {replaced} rows; report.md failed: {exc}"
    return True, f"replaced {replaced} rows"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved", required=True, help="results/mode3")
    ap.add_argument("--rerun", required=True, help="results dir from the subset run")
    ap.add_argument("--backup-suffix", default=".prefix-units")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    saved_root, rerun_root = Path(args.saved), Path(args.rerun)
    auditors = sorted(d.name for d in rerun_root.iterdir()
                      if d.is_dir() and (d / "per_example.json").exists())
    if not auditors:
        print(f"no auditor result directories under {rerun_root}", file=sys.stderr)
        return 2

    print(f"{'auditor':<24} {'status'}")
    failures = 0
    for name in auditors:
        ok, msg = splice_auditor(saved_root / name, rerun_root / name,
                                 args.backup_suffix, args.dry_run)
        print(f"{name:<24} {'OK  ' if ok else 'SKIP'} {msg}")
        failures += 0 if ok else 1

    if failures:
        print(f"\n{failures} auditor(s) not spliced. Resolve before trusting Mode 3.",
              file=sys.stderr)
    elif not args.dry_run:
        print("\nSpliced. Now recompute the imbalance-robust table:")
        print("  python evaluation/mode3/analyze_imbalance_robust_metrics.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
