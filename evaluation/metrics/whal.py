"""
whal.py: Harm-Weighted Hallucination Score.

wHAL(G) = sum_i w_i * n_i(G)

where n_i is the count of tier-i hallucinations and w_i is a weight
calibrated so that w_4 >> w_3 >> w_2 >> w_1.

This metric is the integration of the harm-tier axioms in the CHO
ontology with concrete violation counts from OHR, RIS, and the curated
omission detector. Each detected violation is mapped to a CHO class
via its detection axiom, and that class's harm tier (computed by
the OWL reasoner from the SubClassOf restrictions in cho.owl) is then
used to look up the weight.

Default weights are illustrative and labeled as such in the result.
The paper specifies weight calibration via ordinal regression of expert
harm-severity ratings on hallucination counts per tier (Section 7);
for the empirical evaluation the weights are CALIBRATED rather than
assumed, but defaults are provided for end-to-end testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.claim_extraction import ExtractedClaims
from metrics.ohr import compute_ohr
from metrics.ris import compute_ris


# Default illustrative weights. The paper calls for empirical calibration
# (Section 7); these defaults are placeholders for end-to-end testing.
DEFAULT_WEIGHTS = {1: 1.0, 2: 4.0, 3: 16.0, 4: 64.0}


# Mapping from CHO axiom to harm tier, derived from the SubClassOf
# restrictions in cho.owl. Reasoner-derived; encoded here as a lookup
# table to avoid running the reasoner inside the metric loop.
AXIOM_TO_TIER = {
    "A1_ForbiddenRelation": 3,
    "A2_ThresholdViolation": 4,
    "A5_InvertedRelation": 3,
    "A3_ModifierOmission": 3,
    "A4_CriticalOmission": 3,
    "A6_FabricatedEntity": 2,
}


@dataclass
class WHALResult:
    score: float
    counts: dict = field(default_factory=dict)   # {tier: count}
    weights: dict = field(default_factory=dict)
    detail_by_axiom: dict = field(default_factory=dict)  # {axiom: count}

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "counts": self.counts,
            "weights": self.weights,
            "detail_by_axiom": self.detail_by_axiom,
        }


def compute_whal(claims: ExtractedClaims,
                 weights: dict | None = None,
                 ohr_result=None, ris_result=None) -> WHALResult:
    """Compute wHAL by aggregating violations from OHR (axioms A1, A2)
    and RIS (axiom A5). ModifierOmission and CriticalOmission detectors
    are not implemented in this lightweight pipeline; they require a
    paired source text, and so contribute zero to wHAL when called on
    a single text. They are exercised in the smoke test which has
    paired examples."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Compute (or reuse) OHR and RIS to get violation counts
    if ohr_result is None:
        ohr_result = compute_ohr(claims)
    if ris_result is None:
        ris_result = compute_ris(claims)

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    by_axiom = {}

    for v in ohr_result.violation_details:
        axiom = v["axiom"]
        tier = AXIOM_TO_TIER.get(axiom)
        if tier is None:
            continue
        counts[tier] += 1
        by_axiom[axiom] = by_axiom.get(axiom, 0) + 1

    for inv in ris_result.inversion_details:
        axiom = "A5_InvertedRelation"
        tier = AXIOM_TO_TIER[axiom]
        counts[tier] += 1
        by_axiom[axiom] = by_axiom.get(axiom, 0) + 1

    score = sum(weights[t] * c for t, c in counts.items())
    return WHALResult(score=score, counts=counts, weights=weights,
                      detail_by_axiom=by_axiom)


def whal_with_omissions(source_claims: ExtractedClaims,
                        target_claims: ExtractedClaims,
                        weights: dict | None = None) -> WHALResult:
    """Variant that takes a (source, target) pair and adds modifier-
    omission detection (e.g., negation drop)."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    base = compute_whal(target_claims, weights=weights)

    # Detect modifier omission: for each source entity that is negated,
    # check if the same canonical appears in the target as non-negated.
    src_negated = {e.canonical for e in source_claims.entities if e.negated}
    tgt_unnegated = {e.canonical for e in target_claims.entities if not e.negated}
    dropped = src_negated & tgt_unnegated

    for canon in dropped:
        axiom = "A3_ModifierOmission"
        tier = AXIOM_TO_TIER[axiom]
        base.counts[tier] += 1
        base.detail_by_axiom[axiom] = base.detail_by_axiom.get(axiom, 0) + 1

    base.score = sum(weights[t] * c for t, c in base.counts.items())
    return base


if __name__ == "__main__":
    from extraction.claim_extraction import extract_claims

    print("--- single-text wHAL ---")
    examples = [
        ("clean", "Metformin treats type 2 diabetes mellitus."),
        ("forbidden", "Metformin treats hypertension."),
        ("threshold", "Glucose was 5000 mg/dL."),
        ("inversion", "Hypoglycemia causes insulin."),
        ("multi", "Metformin treats hypertension. Hypoglycemia causes insulin. Potassium 25 mEq/L."),
    ]
    for label, txt in examples:
        c = extract_claims(txt)
        r = compute_whal(c)
        print(f"{label:15s} wHAL={r.score:7.1f}  counts={r.counts}  axioms={r.detail_by_axiom}")

    print("\n--- paired (source, target) wHAL with omission detection ---")
    pairs = [
        ("negation drop",
         "Patient has no fever and no cough.",
         "Patient has fever and cough."),
        ("clean preservation",
         "Patient has no fever.",
         "No fever was noted."),
    ]
    for label, src, tgt in pairs:
        sc = extract_claims(src)
        tc = extract_claims(tgt)
        r = whal_with_omissions(sc, tc)
        print(f"{label:25s} wHAL={r.score:7.1f}  counts={r.counts}  axioms={r.detail_by_axiom}")
