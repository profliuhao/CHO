"""
ris.py: Relational Inversion Score.

RIS(G) = | { (a, r, b) in R(G) : (b, r, a) is in K and (a, r, b) is not } |
         / | R(G) |

where R(G) is the set of binary relations extracted from G and K is the
reference knowledge graph (here, the RxNorm indications + curated
causal relations between conditions and treatments).

In plain language: count the relations whose direction is provably
backwards. The canonical clinical example is "hypoglycemia causes
insulin," which has no support but whose reverse "insulin causes
hypoglycemia" is true.

Detection logic for a (a, treats, b) triple:
  - If (b, treats, a) is supported by RxNorm (i.e., b is the drug and a
    is in b's indication list) AND (a, treats, b) is NOT supported
    (a is the drug and b is in a's indication list, OR a is not a drug
    at all), then the relation is inverted.

For (a, causes, b):
  - We use a small curated table of high-confidence drug -> side-effect
    causal claims (insulin -> hypoglycemia; warfarin -> bleeding; etc.),
    plus disease -> finding causal claims (MI -> elevated troponin; etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.claim_extraction import ExtractedClaims, Relation


# Curated causal knowledge graph. These are well-established
# pharmacology/pathology relations used as the inversion oracle.
CAUSAL_KG = {
    # drug -> side effect
    ("insulin", "causes", "hypoglycemia"),
    ("warfarin", "causes", "bleeding"),
    ("heparin", "causes", "bleeding"),
    ("aspirin", "causes", "bleeding"),
    ("metformin", "causes", "lactic acidosis"),
    ("furosemide", "causes", "hypokalemia"),
    ("furosemide", "causes", "dehydration"),
    ("prednisone", "causes", "hyperglycemia"),
    ("amlodipine", "causes", "edema"),
    ("ace inhibitor", "causes", "cough"),
    ("lisinopril", "causes", "cough"),
    # disease/condition -> finding/lab
    ("myocardial infarction", "causes", "elevated troponin"),
    ("myocardial infarction", "causes", "chest pain"),
    ("heart failure", "causes", "edema"),
    ("heart failure", "causes", "shortness of breath"),
    ("sepsis", "causes", "elevated lactate"),
    ("type 2 diabetes mellitus", "causes", "hyperglycemia"),
    ("type 1 diabetes mellitus", "causes", "hyperglycemia"),
    ("diabetes mellitus", "causes", "hyperglycemia"),
    ("hypothyroidism", "causes", "fatigue"),
    ("hypothyroidism", "causes", "weight gain"),
    ("hyperthyroidism", "causes", "weight loss"),
    ("anemia", "causes", "fatigue"),
    ("pneumonia", "causes", "cough"),
    ("pneumonia", "causes", "fever"),
    ("asthma", "causes", "wheezing"),
    ("asthma", "causes", "shortness of breath"),
    ("chronic kidney disease", "causes", "elevated creatinine"),
    ("acute kidney injury", "causes", "elevated creatinine"),
    ("liver disease", "causes", "elevated alt"),
    ("hepatitis", "causes", "elevated alt"),
}


_RXNORM = None


def _load_rxnorm() -> dict:
    global _RXNORM
    if _RXNORM is None:
        path = Path(__file__).parent.parent.parent / "ontology" / "data" / "rxnorm_indications.json"
        with open(path) as fh:
            _RXNORM = json.load(fh)["drugs"]
    return _RXNORM


@dataclass
class RISResult:
    score: float
    inverted: int
    total_relations: int
    inversion_details: list

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "inverted": self.inverted,
            "total_relations": self.total_relations,
            "inversion_details": self.inversion_details,
        }


def _treats_supported(drug: str, disease: str) -> bool:
    """Is (drug, treats, disease) supported by RxNorm?"""
    rxnorm = _load_rxnorm()
    drug_key = drug.lower()
    if drug_key not in rxnorm:
        return False
    indications = [i.lower() for i in rxnorm[drug_key]["indications"]]
    disease_key = disease.lower()
    if disease_key in indications:
        return True
    for ind in indications:
        if disease_key in ind or ind in disease_key:
            return True
    return False


def _causal_supported(subj: str, obj: str) -> bool:
    """Is (subj, causes, obj) in the curated causal KG?"""
    return (subj.lower(), "causes", obj.lower()) in CAUSAL_KG


def is_relation_inverted(rel: Relation) -> bool:
    """A relation is inverted iff its reverse is supported by the KG and
    its forward direction is not."""
    a, b = rel.subject.canonical, rel.object.canonical
    if rel.predicate == "treats":
        forward = _treats_supported(a, b)
        reverse = _treats_supported(b, a)
        return reverse and not forward
    if rel.predicate == "causes":
        forward = _causal_supported(a, b)
        reverse = _causal_supported(b, a)
        return reverse and not forward
    return False


def compute_ris(claims: ExtractedClaims) -> RISResult:
    inverted_count = 0
    details = []
    for r in claims.relations:
        if is_relation_inverted(r):
            inverted_count += 1
            details.append({
                "predicate": r.predicate,
                "claim": f"{r.subject.canonical} {r.predicate} {r.object.canonical}",
                "supported_reverse": f"{r.object.canonical} {r.predicate} {r.subject.canonical}",
            })
    score = inverted_count / len(claims.relations) if claims.relations else float("nan")
    return RISResult(
        score=score, inverted=inverted_count,
        total_relations=len(claims.relations), inversion_details=details,
    )


if __name__ == "__main__":
    from extraction.claim_extraction import extract_claims
    examples = [
        ("correct", "Insulin causes hypoglycemia."),
        ("inverted", "Hypoglycemia causes insulin."),
        ("correct treats", "Metformin treats type 2 diabetes mellitus."),
        ("inverted treats", "Hypertension treats lisinopril."),
        ("no relation", "Patient is on metformin."),
    ]
    for label, txt in examples:
        c = extract_claims(txt)
        r = compute_ris(c)
        print(f"{label:20s} RIS={r.score!s:>6}  inv={r.inverted}/{r.total_relations}  "
              f"details={r.inversion_details}")
