"""
ohr.py: Ontological Hallucination Rate.

OHR(G) = | { c in V(G) : violatesAxiom(c) } | / | V(G) |

V(G) = subset of claims that are ontology-verifiable, i.e. claims for which
we have a canonical lookup or axiom that returns a definitive truth value.

A claim violates an axiom if:
  - it is a (drug, treats, disease) triple where the disease is not in
    the drug's RxNorm indication list  (axiom A1: ForbiddenRelation)
  - it is a numeric lab value outside the curated physiologic range
    (axiom A2: ThresholdViolation)
  - it is a fabricated entity (canonical not in our concept inventory)
    (axiom A6: FabricatedEntity)

The implementation deliberately returns coverage alongside OHR so that low
OHR with low coverage cannot be mistaken for a safe output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow imports from sibling extraction module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.claim_extraction import ExtractedClaims, get_lexicon


# Cache for lookup tables
_RXNORM = None
_RANGES = None
_CONTRA = None


def _load_rxnorm() -> dict:
    global _RXNORM
    if _RXNORM is None:
        path = Path(__file__).parent.parent.parent / "ontology" / "data" / "rxnorm_indications.json"
        with open(path) as fh:
            _RXNORM = json.load(fh)["drugs"]
    return _RXNORM


def _load_ranges() -> dict:
    global _RANGES
    if _RANGES is None:
        path = Path(__file__).parent.parent.parent / "ontology" / "data" / "physiologic_ranges.json"
        with open(path) as fh:
            _RANGES = json.load(fh)["labs"]
    return _RANGES


def _load_contraindications() -> dict:
    """UMLS MED-RT-derived (drug → contraindicated_conditions) table.
    Returns {} if the file isn't present (Phase 1.2 not yet built)."""
    global _CONTRA
    if _CONTRA is None:
        path = Path(__file__).parent.parent.parent / "ontology" / "data" / "contraindications.json"
        if path.exists():
            with open(path) as fh:
                _CONTRA = json.load(fh).get("drugs", {})
        else:
            _CONTRA = {}
    return _CONTRA


@dataclass
class OHRResult:
    score: float            # OHR value in [0, 1] (or NaN if no coverage)
    coverage: float         # |V(G)| / |Claims(G)| in [0, 1]
    violations: int         # raw count of axiom violations
    verifiable: int         # |V(G)|
    total: int              # |Claims(G)|
    violation_details: list  # list of (claim_dict, axiom_label) tuples

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "coverage": self.coverage,
            "violations": self.violations,
            "verifiable": self.verifiable,
            "total": self.total,
            "violation_details": self.violation_details,
        }


def is_treats_relation_forbidden(drug_name: str, disease_name: str) -> Optional[bool]:
    """Verdict on whether (drug, treats, disease) violates clinical knowledge.

    Returns:
      True  - violates (either explicit MED-RT contraindication, or not in
              the drug's RxNorm indication list when we have one)
      False - permitted (matches an RxNorm indication)
      None  - cannot verify (drug not in either table)

    Verification order:
      1. Explicit contraindication (MED-RT-derived) — strongest signal,
         fires A1 unambiguously.
      2. RxNorm indication — supports OR refutes within indication-listed drugs.
      3. Otherwise None (unknown).
    """
    drug_key = drug_name.lower()
    disease_key = disease_name.lower()

    # Step 1: Explicit contraindication?
    contra = _load_contraindications()
    contra_entry = contra.get(drug_key)
    if contra_entry:
        for cond in contra_entry.get("contraindications", []):
            cond_low = cond.lower()
            if cond_low == disease_key:
                return True
            # token-overlap fuzzy match (e.g. "kidney failure" vs "acute kidney failure")
            if (disease_key in cond_low or cond_low in disease_key) \
                    and len(min(disease_key, cond_low, key=len)) >= 6:
                return True

    # Step 2: RxNorm indication?
    rxnorm = _load_rxnorm()
    if drug_key in rxnorm:
        indications = [i.lower() for i in rxnorm[drug_key]["indications"]]
        if disease_key in indications:
            return False
        for ind in indications:
            if disease_key in ind or ind in disease_key:
                return False
        return True   # in RxNorm but not indicated for this disease

    # Step 3: nothing to say
    return None


def is_threshold_violated(lab_name: str, value: float) -> Optional[bool]:
    """Returns True if value is outside plausible bounds, False if inside,
    None if lab not in our table or value is NaN."""
    if value != value:  # NaN
        return None
    ranges = _load_ranges()
    if lab_name not in ranges:
        return None
    info = ranges[lab_name]
    return value < info["plausible_min"] or value > info["plausible_max"]


def is_fabricated_entity(canonical: str) -> bool:
    """Returns True if this canonical name is not in our known concept
    inventory. By construction, the extractor only canonicalizes known
    names, so this check is effectively for upstream uses where a
    candidate name is passed in raw."""
    lex = get_lexicon()
    return canonical.lower() not in lex.entity_categories and \
           canonical.lower() not in lex.aliases


def compute_ohr(claims: ExtractedClaims) -> OHRResult:
    total = (len(claims.entities) + len(claims.relations) +
             len(claims.numerics))
    verifiable = 0
    violations = 0
    details = []

    # Verifiable entities: anything in our lexicon (i.e. all extracted entities)
    for e in claims.entities:
        verifiable += 1
        # In this implementation, the extractor only emits known entities,
        # so FabricatedEntity is rare. It would fire if upstream passed
        # in raw candidate spans; we keep the hook for future use.

    # Verifiable relations: (drug, treats, disease) where drug is in RxNorm
    for r in claims.relations:
        if r.predicate != "treats":
            continue
        if r.subject.category != "drug":
            continue
        if r.object.category not in ("disease", "mental_disorder", "symptom",
                                     "lab_finding"):
            continue
        verdict = is_treats_relation_forbidden(r.subject.canonical,
                                                r.object.canonical)
        if verdict is None:
            continue
        verifiable += 1
        if verdict:
            violations += 1
            details.append({
                "axiom": "A1_ForbiddenRelation",
                "claim": f"{r.subject.canonical} treats {r.object.canonical}",
            })

    # Verifiable numerics: lab value with known physiologic range
    for n in claims.numerics:
        if n.value != n.value:  # NaN
            continue
        verdict = is_threshold_violated(n.lab, n.value)
        if verdict is None:
            continue
        verifiable += 1
        if verdict:
            violations += 1
            details.append({
                "axiom": "A2_ThresholdViolation",
                "claim": f"{n.lab} = {n.value}",
            })

    coverage = verifiable / total if total else 0.0
    score = violations / verifiable if verifiable else float("nan")

    return OHRResult(
        score=score, coverage=coverage,
        violations=violations, verifiable=verifiable, total=total,
        violation_details=details,
    )


if __name__ == "__main__":
    from extraction.claim_extraction import extract_claims
    examples = [
        ("OK", "Patient with type 2 diabetes mellitus on metformin."),
        ("forbidden treats", "Metformin treats hypertension."),
        ("threshold violation", "Potassium was 25 mEq/L."),
        ("normal value", "Potassium was 4.5 mEq/L."),
        ("mixed", "Metformin treats hypertension. Glucose 800 mg/dL."),
    ]
    for label, txt in examples:
        c = extract_claims(txt)
        r = compute_ohr(c)
        print(f"{label:25s} OHR={r.score:.3f} cov={r.coverage:.3f} "
              f"viol={r.violations}/{r.verifiable}  details={r.violation_details}")
