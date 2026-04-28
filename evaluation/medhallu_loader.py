"""
medhallu_loader.py: Download and parse the MedHallu benchmark.

MedHallu is a medical-hallucination discrimination dataset built on top of
PubMedQA. Each source row provides:
  - a biomedical Question and supporting Knowledge passages (PubMedQA-style),
  - a human/expert-validated Ground Truth answer,
  - a deliberately-constructed Hallucinated Answer,
  - the Category of Hallucination (e.g. "Mechanism and Pathway Misattribution",
    "Incomplete Information", "Methodological and Evidence Fabrication"),
  - a Difficulty Level (easy / medium / hard).

For our evaluation we expand each source row into TWO MedHalluExample records
sharing the same source_qid:
  - one with proposed_answer = Ground Truth        and is_correct = True
  - one with proposed_answer = Hallucinated Answer and is_correct = False

This guarantees a balanced two-class problem (n_correct == n_hallucinated)
so AUROC against the `is_hallucination` label is well-defined.

The dataclass surface (qid, question, proposed_answer, is_correct,
composed_text) matches medhalt_loader.FCTExample so the orchestrator
in run_evaluation.py can consume either loader interchangeably.

Two HF configs are supported:
  - 'pqa_labeled'    : 1,000 expert-curated examples (default; primary)
  - 'pqa_artificial' : 9,000 model-generated hallucinations (larger)

Citation:
  Pandit S. et al. (2025). MedHallu: A Comprehensive Benchmark for Detecting
  Medical Hallucinations in Large Language Models.
  https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MedHalluExample:
    qid: str                       # unique per emitted record (e.g. "row42-correct")
    source_qid: str                # shared by the (correct, hallucinated) pair
    question: str
    proposed_answer: str           # the answer being evaluated for hallucination
    is_correct: bool               # True for the Ground Truth variant
    knowledge: list[str] = field(default_factory=list)
    category: Optional[str] = None      # MedHallu hallucination category (or None)
    difficulty: Optional[str] = None    # easy / medium / hard

    @property
    def composed_text(self) -> str:
        """The text we treat as 'model output' for CHO scoring.
        We concatenate question and answer so the extractor sees both
        the topic context and the claims being made about it."""
        q = self.question.strip().rstrip("?.!") + "?"
        return f"{q} {self.proposed_answer.strip()}"


# =====================================================
# Loaders
# =====================================================

def load_from_huggingface(config: str = "pqa_labeled",
                          split: str = "train",
                          n: Optional[int] = None,
                          categories: Optional[list[str]] = None,
                          difficulties: Optional[list[str]] = None,
                          ) -> list[MedHalluExample]:
    """Load MedHallu from Hugging Face. Requires network access.

    `n` is interpreted as the maximum number of *source rows* to consume
    after any category/difficulty filtering; the returned list has up to
    2*n examples (one correct + one hallucinated per source row).

    `categories` filters source rows by Category of Hallucination
    (e.g. ['Mechanism and Pathway Misattribution']); a row is kept if its
    category matches any entry. `difficulties` works the same way against
    Difficulty Level.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'datasets' library: pip install datasets"
        ) from exc

    if config not in ("pqa_labeled", "pqa_artificial"):
        raise ValueError(
            f"Unknown MedHallu config {config!r}; "
            f"expected 'pqa_labeled' or 'pqa_artificial'."
        )

    ds = load_dataset("UTAustin-AIHealth/MedHallu", config, split=split)

    cat_set = {c.strip().lower() for c in (categories or [])}
    diff_set = {d.strip().lower() for d in (difficulties or [])}

    out: list[MedHalluExample] = []
    kept = 0
    for i, row in enumerate(ds):
        if cat_set:
            row_cat = str(row.get("Category of Hallucination", "")).strip().lower()
            if row_cat not in cat_set:
                continue
        if diff_set:
            row_diff = str(row.get("Difficulty Level", "")).strip().lower()
            if row_diff not in diff_set:
                continue
        if n is not None and kept >= n:
            break
        out.extend(_row_to_examples(row, fallback_id=str(i)))
        kept += 1
    return out


def _row_to_examples(row: dict, fallback_id: str) -> list[MedHalluExample]:
    """Expand one MedHallu row into a (correct, hallucinated) pair."""
    source_qid = str(row.get("id", fallback_id))
    question = str(row.get("Question", "")).strip()
    knowledge_raw = row.get("Knowledge", []) or []
    if isinstance(knowledge_raw, str):
        knowledge = [knowledge_raw]
    else:
        knowledge = [str(k) for k in knowledge_raw]
    ground_truth = str(row.get("Ground Truth", "")).strip()
    hallucinated = str(row.get("Hallucinated Answer", "")).strip()
    category = row.get("Category of Hallucination")
    difficulty = row.get("Difficulty Level")

    # Skip degenerate rows with missing answer text on either side; we need
    # both to maintain class balance.
    if not ground_truth or not hallucinated:
        return []

    correct_ex = MedHalluExample(
        qid=f"{source_qid}-correct",
        source_qid=source_qid,
        question=question,
        proposed_answer=ground_truth,
        is_correct=True,
        knowledge=knowledge,
        category=None,                  # ground truth has no hallucination category
        difficulty=str(difficulty) if difficulty is not None else None,
    )
    hallucinated_ex = MedHalluExample(
        qid=f"{source_qid}-hallucinated",
        source_qid=source_qid,
        question=question,
        proposed_answer=hallucinated,
        is_correct=False,
        knowledge=knowledge,
        category=str(category) if category is not None else None,
        difficulty=str(difficulty) if difficulty is not None else None,
    )
    return [correct_ex, hallucinated_ex]


def load_from_fixture(fixture_path: Path) -> list[MedHalluExample]:
    """Load examples from a bundled JSON fixture (offline smoke tests).

    The fixture is a list of pre-expanded example dicts (already split into
    correct / hallucinated records), matching the MedHalluExample fields.
    """
    with open(fixture_path) as fh:
        rows = json.load(fh)
    return [
        MedHalluExample(
            qid=r["qid"],
            source_qid=r.get("source_qid", r["qid"]),
            question=r["question"],
            proposed_answer=r["proposed_answer"],
            is_correct=bool(r["is_correct"]),
            knowledge=list(r.get("knowledge") or []),
            category=r.get("category"),
            difficulty=r.get("difficulty"),
        )
        for r in rows
    ]


def load(source: str = "auto",
         n: Optional[int] = None,
         fixture_path: Optional[Path] = None,
         config: str = "pqa_labeled",
         categories: Optional[list[str]] = None,
         difficulties: Optional[list[str]] = None,
         ) -> list[MedHalluExample]:
    """Unified loader.

    source:
      'huggingface' - require HF; fail if unreachable
      'fixture'     - require fixture file; fail if missing
      'auto'        - try HF first, fall back to fixture

    n: max number of source rows to consume from HF (so max 2*n examples).
       For fixture mode, n caps the returned list directly.
    categories / difficulties: optional MedHallu metadata filters.
    """
    here = Path(__file__).parent
    default_fixture = here / "data" / "medhallu_smoke.json"
    fp = fixture_path or default_fixture

    if source == "huggingface":
        return load_from_huggingface(config=config, n=n,
                                     categories=categories,
                                     difficulties=difficulties)
    if source == "fixture":
        examples = load_from_fixture(fp)
        if n is not None:
            examples = examples[: 2 * n]   # n source rows == 2n examples
        return examples
    # auto
    try:
        return load_from_huggingface(config=config, n=n,
                                     categories=categories,
                                     difficulties=difficulties)
    except Exception as exc:
        print(f"[medhallu_loader] HF unreachable ({type(exc).__name__}); "
              f"falling back to fixture at {fp}", file=sys.stderr)
        examples = load_from_fixture(fp)
        return examples[: 2 * n] if n else examples


# =====================================================
# CLI (smoke test / inspection)
# =====================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "huggingface", "fixture"],
                        default="auto")
    parser.add_argument("--config", choices=["pqa_labeled", "pqa_artificial"],
                        default="pqa_labeled")
    parser.add_argument("-n", type=int, default=5,
                        help="number of source rows to inspect")
    parser.add_argument("--fixture", type=str, default=None)
    args = parser.parse_args()

    fp = Path(args.fixture) if args.fixture else None
    examples = load(args.source, n=args.n, fixture_path=fp, config=args.config)

    print(f"Loaded {len(examples)} MedHallu examples "
          f"(source={args.source!r}, config={args.config!r}).")
    print(f"  correct:      {sum(1 for e in examples if e.is_correct)}")
    print(f"  hallucinated: {sum(1 for e in examples if not e.is_correct)}")
    print()
    for i, ex in enumerate(examples[:6]):
        tag = "CORRECT" if ex.is_correct else "HALLUCINATED"
        print(f"--- Example {i+1}  [{tag}]  qid={ex.qid}  "
              f"category={ex.category!r}  difficulty={ex.difficulty!r} ---")
        print(f"Q: {ex.question[:100]}{'...' if len(ex.question) > 100 else ''}")
        print(f"Proposed: {ex.proposed_answer[:140]}"
              f"{'...' if len(ex.proposed_answer) > 140 else ''}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
