"""Sample a small MIMIC-IV-BHC validation set for the CHO real-output audit.

Construction follows the hospital-course summarization setup: the input is the
discharge note with its Brief Hospital Course section removed, and the reference
is that removed section. A local model later writes its own hospital course from
the input, and CHO audits the generated text against the source note.

PHI handling. Note text is written only under evaluation/results/, which the
repository .gitignore excludes. Nothing here prints note text to stdout, and no
note text may appear in the paper, in figures, or in any published artifact.
Only counts, distributions, and de-identified identifiers leave this script.

Usage:
    python evaluation/mimic_bhc/sample_bhc.py --n 60 --seed 20260802
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg2

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "mimic_bhc"

# The section header is consistently spelled this way in MIMIC-IV discharge notes.
BHC_HEADER = "Brief Hospital Course:"
# Sections that commonly follow the hospital course. The first one that appears
# after the header terminates it.
NEXT_HEADERS = [
    "Medications on Admission:",
    "Discharge Medications:",
    "Discharge Disposition:",
    "Discharge Diagnosis:",
    "Discharge Condition:",
    "Discharge Instructions:",
    "Followup Instructions:",
    "Physical Exam:",
    "Pertinent Results:",
]


def split_bhc(text: str) -> tuple[str, str] | None:
    """Return (input_without_bhc, bhc_section), or None if it cannot be split."""
    start = text.find(BHC_HEADER)
    if start == -1:
        return None
    body_start = start + len(BHC_HEADER)
    ends = [text.find(h, body_start) for h in NEXT_HEADERS]
    ends = [e for e in ends if e != -1]
    end = min(ends) if ends else len(text)
    bhc = text[body_start:end].strip()
    if not bhc:
        return None
    remainder = (text[:start] + text[end:]).strip()
    return remainder, bhc


def connect():
    return psycopg2.connect(
        host=os.environ.get("MIMIC_DB_HOST", "130.68.124.64"),
        port=int(os.environ.get("MIMIC_DB_PORT", 5432)),
        dbname=os.environ.get("MIMIC_DB_NAME", "mimiciv"),
        user=os.environ.get("MIMIC_DB_USER", "postgres"),
        password=os.environ["MIMIC_DB_PASSWORD"],
        connect_timeout=15,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="admissions to sample")
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--min-chars", type=int, default=4000)
    ap.add_argument("--max-chars", type=int, default=20000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()
    cur = conn.cursor()

    # One note per subject so no patient contributes twice, length-bounded to keep
    # generation tractable and to avoid the truncated-note tail of the distribution.
    # setseed makes random() reproducible for a given --seed.
    cur.execute("select setseed(%s)", (((args.seed % 1000) / 1000.0) - 0.5,))
    cur.execute(
        """
        with eligible as (
            select distinct on (subject_id)
                   note_id, subject_id, hadm_id, charttime, length(text) as n_chars, text
              from mimiciv_note.discharge
             where text like %s
               and length(text) between %s and %s
             order by subject_id, note_id
        )
        select note_id, subject_id, hadm_id, charttime, n_chars, text
          from eligible
         order by random()
         limit %s
        """,
        (f"%{BHC_HEADER}%", args.min_chars, args.max_chars, args.n * 3),
    )
    rows = cur.fetchall()

    kept, skipped = [], 0
    for note_id, subject_id, hadm_id, charttime, n_chars, text in rows:
        parts = split_bhc(text)
        if parts is None:
            skipped += 1
            continue
        source, bhc = parts
        # A hospital course too short to summarize carries no auditable claims.
        if len(bhc) < 400:
            skipped += 1
            continue
        kept.append(
            {
                "note_id": note_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "charttime": str(charttime),
                "n_chars_note": n_chars,
                "n_chars_source": len(source),
                "n_chars_bhc": len(bhc),
                "source_text": source,
                "reference_bhc": bhc,
            }
        )
        if len(kept) >= args.n:
            break

    conn.close()

    out = OUT_DIR / "bhc_sample.json"
    payload = {
        "_metadata": {
            "description": (
                "MIMIC-IV-BHC style validation sample for the CHO real-output audit. "
                "source_text is the discharge note with the Brief Hospital Course "
                "section removed; reference_bhc is that section. CREDENTIALED DATA: "
                "do not redistribute, do not quote in the paper."
            ),
            "seed": args.seed,
            "n_requested": args.n,
            "n_kept": len(kept),
            "n_skipped_unsplittable": skipped,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "source_table": "mimiciv_note.discharge",
        },
        "examples": kept,
    }
    out.write_text(json.dumps(payload, indent=1))

    lens = sorted(e["n_chars_bhc"] for e in kept)
    src = sorted(e["n_chars_source"] for e in kept)
    print(f"kept {len(kept)} admissions ({skipped} skipped as unsplittable/too short)")
    print(f"  distinct subjects: {len({e['subject_id'] for e in kept})}")
    if lens:
        print(f"  source chars   min/median/max: {src[0]:,} / {src[len(src)//2]:,} / {src[-1]:,}")
        print(f"  reference BHC  min/median/max: {lens[0]:,} / {lens[len(lens)//2]:,} / {lens[-1]:,}")
    print(f"  wrote {out}  (gitignored)")


if __name__ == "__main__":
    main()
