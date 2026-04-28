"""
medhalt_loader.py: Download and parse the Med-HALT False Confidence Test
(FCT) subset from Hugging Face.

The FCT is a labeled hallucination-discrimination task: each example has a
medical question, a CORRECT answer (one of the multiple-choice options),
and a PROPOSED answer that may or may not be correct. The model is asked
to evaluate whether the proposed answer is correct.

For our evaluation we treat the (question + proposed answer) as the
generated medical text, and the binary correctness label as ground truth.
A CHO metric is a useful hallucination detector if its score is
significantly higher on incorrect-answer examples than on correct ones.

Citation:
  Pal A., Umapathi L. K., Sankarasubbu M. (2023).
  Med-HALT: Medical Domain Hallucination Test for Large Language Models.
  CoNLL 2023. https://aclanthology.org/2023.conll-1.21/

The script also supports loading a small bundled fixture (data/medhalt_fct_smoke.json)
when network access is unavailable, so the smoke test can run end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FCTExample:
    qid: str
    question: str
    options: list[str]
    correct_index: int          # index of the actually correct option in `options`
    proposed_answer: str        # the option presented to the model for evaluation
    is_correct: bool            # True if proposed_answer matches correct option

    @property
    def composed_text(self) -> str:
        """The text we treat as 'model output' for the purpose of CHO scoring.
        We construct a declarative claim from the question stem and the
        proposed answer."""
        return f"{self.question.strip()} The answer is: {self.proposed_answer.strip()}."


def load_from_huggingface(subset: str = "reasoning_FCT",
                          split: str = "train",
                          n: Optional[int] = None) -> list[FCTExample]:
    """Load Med-HALT from Hugging Face. Requires network access."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'datasets' library: pip install datasets"
        ) from exc

    ds = load_dataset("openlifescienceai/Med-HALT", subset, split=split)
    out = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        out.append(_row_to_example(row, fallback_id=str(i)))
    return out


def _row_to_example(row: dict, fallback_id: str) -> FCTExample:
    """Map an HF row to FCTExample.

    The Med-HALT FCT schema (per the GitHub repo and HF dataset card)
    has these fields per example: question, options (list), correct_index
    (int), proposed_answer, plus metadata like exam_country, subject.
    Field names vary slightly across the seven Med-HALT subsets; this
    function handles the common cases."""
    qid = row.get("id", fallback_id)
    question = row.get("question") or row.get("Question") or ""

    # `options` may be a list, or (in reasoning_FCT) a stringified dict whose
    # keys are option indices ("0".."3") plus a "correct answer" key. Coerce
    # both shapes into an ordered list of option strings.
    raw_options = (row.get("options") if row.get("options") is not None
                   else row.get("Options"))
    if raw_options is None:
        raw_options = [row.get(k, "") for k in ("A", "B", "C", "D") if k in row]
    if isinstance(raw_options, str):
        try:
            import ast
            raw_options = ast.literal_eval(raw_options)
        except (ValueError, SyntaxError):
            raw_options = [raw_options]
    if isinstance(raw_options, dict):
        numeric_items = [(k, v) for k, v in raw_options.items()
                         if str(k).isdigit()]
        numeric_items.sort(key=lambda kv: int(kv[0]))
        options = [str(v) for _, v in numeric_items]
    else:
        options = [str(o) for o in raw_options]

    correct_index = (row.get("correct_index")
                     if row.get("correct_index") is not None
                     else row.get("answer_idx"))
    if correct_index is None:
        correct_index = row.get("correct_option_idx", 0)
    if isinstance(correct_index, str) and correct_index.isalpha():
        correct_index = ord(correct_index.upper()) - ord("A")
    correct_index = int(correct_index)

    # The "proposed" answer is the candidate being evaluated for hallucination.
    # In reasoning_FCT this is the `student_answer` field (with `student_index`
    # as the corresponding option index). Fall back to other Med-HALT field
    # names where applicable.
    proposed = (row.get("student_answer")
                if row.get("student_answer") is not None
                else row.get("proposed_answer"))
    if proposed is None:
        proposed = row.get("Proposed Answer")
    if proposed is None:
        proposed = (options[correct_index]
                    if 0 <= correct_index < len(options) else "")

    student_index = row.get("student_index")
    if student_index is not None:
        try:
            is_correct = int(student_index) == correct_index
        except (TypeError, ValueError):
            is_correct = (0 <= correct_index < len(options) and
                          options[correct_index].strip() == str(proposed).strip())
    else:
        is_correct = (0 <= correct_index < len(options) and
                      options[correct_index].strip() == str(proposed).strip())
    return FCTExample(
        qid=str(qid), question=str(question), options=options,
        correct_index=correct_index, proposed_answer=str(proposed),
        is_correct=is_correct,
    )


def load_from_fixture(fixture_path: Path) -> list[FCTExample]:
    """Load examples from a bundled JSON fixture (for offline smoke tests)."""
    with open(fixture_path) as fh:
        rows = json.load(fh)
    return [
        FCTExample(
            qid=r["qid"], question=r["question"], options=r["options"],
            correct_index=r["correct_index"],
            proposed_answer=r["proposed_answer"], is_correct=r["is_correct"],
        )
        for r in rows
    ]


def load(source: str = "auto", n: Optional[int] = None,
         fixture_path: Optional[Path] = None) -> list[FCTExample]:
    """Unified loader.

    source:
      'huggingface'  - require HF; fail if unreachable
      'fixture'      - require fixture file; fail if missing
      'auto'         - try HF first, fall back to fixture
    """
    here = Path(__file__).parent
    default_fixture = here / "data" / "medhalt_fct_smoke.json"
    fp = fixture_path or default_fixture

    if source == "huggingface":
        return load_from_huggingface(n=n)
    if source == "fixture":
        examples = load_from_fixture(fp)
        if n is not None:
            examples = examples[:n]
        return examples
    # auto
    try:
        return load_from_huggingface(n=n)
    except Exception as exc:
        print(f"[medhalt_loader] HF unreachable ({type(exc).__name__}); "
              f"falling back to fixture at {fp}", file=sys.stderr)
        return load_from_fixture(fp)[:n] if n else load_from_fixture(fp)


# =====================================================
# CLI
# =====================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "huggingface", "fixture"],
                        default="auto")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--fixture", type=str, default=None)
    args = parser.parse_args()

    fp = Path(args.fixture) if args.fixture else None
    examples = load(args.source, n=args.n, fixture_path=fp)

    print(f"Loaded {len(examples)} examples from source={args.source!r}.")
    print()
    for i, ex in enumerate(examples[:5]):
        print(f"--- Example {i+1} (qid={ex.qid}, correct={ex.is_correct}) ---")
        print(f"Q: {ex.question[:100]}{'...' if len(ex.question) > 100 else ''}")
        print(f"Proposed: {ex.proposed_answer}")
        print(f"Composed text for CHO scoring: {ex.composed_text[:120]}...")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
