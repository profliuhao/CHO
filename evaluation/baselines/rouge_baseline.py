"""
rouge_baseline.py: ROUGE-L baseline.

Pure-Python implementation of ROUGE-L F1 (longest common subsequence).
Used as a surface-form baseline against which CHO metrics are compared.
The expectation is that ROUGE-L cannot distinguish hallucinated medical
content from correct medical content when both share clinical vocabulary,
which is exactly the gap CHO metrics fill.
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Length of longest common subsequence using DP."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # 1D DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l_f1(reference: str, hypothesis: str, beta: float = 1.0) -> float:
    """Compute ROUGE-L F1 score between two strings.

    Returns a value in [0, 1]. beta=1.0 gives standard F1; ROUGE original
    uses beta=1.2 but F1 is the most commonly reported variant.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    beta_sq = beta * beta
    f = ((1 + beta_sq) * precision * recall /
         (recall + beta_sq * precision))
    return f


def rouge_l_lexical_overlap_score(reference: str, hypothesis: str) -> float:
    """A proxy 'hallucination indicator' from ROUGE-L: 1 - F1.
    Higher means more divergence from the reference, which is the
    naive surface-form analog of hallucination."""
    return 1.0 - rouge_l_f1(reference, hypothesis)


if __name__ == "__main__":
    pairs = [
        ("identical",
         "Patient has type 2 diabetes mellitus on metformin.",
         "Patient has type 2 diabetes mellitus on metformin."),
        ("paraphrase no hallucination",
         "Patient has diabetes and is taking metformin.",
         "The patient takes metformin for diabetes."),
        ("subtle hallucination",
         "Metformin treats diabetes.",
         "Metformin treats hypertension."),  # CHO would catch; ROUGE may miss
        ("totally different",
         "Patient has fever.",
         "Patient went hiking yesterday."),
    ]
    for label, ref, hyp in pairs:
        f1 = rouge_l_f1(ref, hyp)
        print(f"{label:30s} ROUGE-L F1 = {f1:.3f}  (1-F1={1-f1:.3f})")
