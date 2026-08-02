"""Compare two gold-label files and emit the subset that actually changed.

Regenerating the gold set after the lab-unit fix should change the text of the
threshold examples and nothing else. This script proves that rather than assuming
it, and writes a subset file containing only the changed examples so Mode 3 can be
re-run on them alone.

Stop and investigate if the composition summary differs, or if changed ids fall
outside the threshold sources. Either means the regeneration did more than fix
units, and every saved auditor result is then stale, not just the changed subset.

Usage:
    python evaluation/mode2/diff_gold.py --old gold_labels.backup.json \\
        --new evaluation/mode2/gold_labels.json \\
        --subset-out evaluation/mode2/gold_labels.changed_subset.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text())
    return payload, {e["id"]: e for e in payload["examples"]}


def composition(examples: dict) -> dict:
    return {
        "n": len(examples),
        "n_hallucinated": sum(1 for e in examples.values() if e["is_hallucination"]),
        "n_negative": sum(1 for e in examples.values() if not e["is_hallucination"]),
        "by_class": dict(Counter(e["expected_class"] for e in examples.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--subset-out", default=None)
    args = ap.parse_args()

    _, old = load(Path(args.old))
    new_payload, new = load(Path(args.new))

    co, cn = composition(old), composition(new)
    same_composition = co == cn
    print("composition old:", json.dumps(co, sort_keys=True))
    print("composition new:", json.dumps(cn, sort_keys=True))
    print("composition identical:", same_composition)

    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(
        i for i in (set(old) & set(new))
        if old[i]["text"] != new[i]["text"]
    )
    relabeled = sorted(
        i for i in (set(old) & set(new))
        if old[i]["expected_class"] != new[i]["expected_class"]
        or old[i]["is_hallucination"] != new[i]["is_hallucination"]
    )

    print(f"\nadded ids:      {len(added)}")
    print(f"removed ids:    {len(removed)}")
    print(f"text changed:   {len(changed)}")
    print(f"label changed:  {len(relabeled)}")

    if changed:
        srcs = Counter(new[i]["source"] for i in changed)
        print("\nchanged-by-source:", dict(srcs))
        off_target = [i for i in changed if "threshold" not in new[i]["source"]]
        if off_target:
            print(f"WARNING: {len(off_target)} changed examples are not threshold "
                  f"examples: {off_target[:10]}")
        print("\nexample text change:")
        i = changed[0]
        print("  old:", json.dumps(old[i]["text"]))
        print("  new:", json.dumps(new[i]["text"]))

    clean = same_composition and not added and not removed and not relabeled
    if clean and changed:
        print("\nOK: unit fix only. A targeted re-run of the changed ids is valid.")
    elif not changed:
        print("\nNOTHING CHANGED. The fix may not have been applied, or the gold "
              "file was not regenerated.")
    else:
        print("\nDO NOT SPLICE. The regeneration altered composition or labels, so "
              "every saved auditor result is stale. Re-run Mode 3 in full.")

    if args.subset_out and changed:
        subset = {
            "_metadata": {
                **new_payload.get("_metadata", {}),
                "description": (
                    "Subset of the regenerated gold set whose text changed after the "
                    "lab-unit fix. For targeted Mode 3 re-runs only; not a benchmark."
                ),
                "n_examples": len(changed),
                "derived_from": args.new,
                "changed_ids": changed,
            },
            "examples": [new[i] for i in changed],
        }
        Path(args.subset_out).write_text(json.dumps(subset, indent=2))
        print(f"\nwrote subset of {len(changed)} examples to {args.subset_out}")

    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
