"""
curate_gold.py

Build the Mode 2 gold-label set.

Two sources of gold-labeled hallucination examples:

1. SYNTHETIC: deterministically generated from our knowledge tables.
   - ForbiddenRelation: "{drug} treats {contraindicated_condition}"
     where (drug, condition) is a MED-RT contraindication pair.
   - InvertedRelation: known causal pair (a, causes, b) flipped to (b, causes, a).
   - ThresholdViolation: lab value outside curated physiologic bounds.
   - ModifierOmission: negated-cue example with modifier dropped.
   - Negative controls: structurally similar but correct claims.

   These exercise CHO's axiom classes by construction — the gold label
   is deterministic. Demonstrates "the formal axioms classify their
   target distribution correctly" with high precision.

2. CURATED MEDHALLU: small subset (~30) of in-clinical-scope MedHallu
   examples where the hallucination cleanly maps onto a CHO axiom.
   Drafted automatically from MedHallu's `Category of Hallucination`
   field plus heuristic content matching; flagged for user review.
   Demonstrates external validity on real-world hallucinations.

Output: evaluation/mode2/gold_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
EVAL_ROOT = HERE.parent
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))

OUT_PATH = HERE / "gold_labels.json"
CONTRA_PATH = REPO_ROOT / "ontology" / "data" / "contraindications.json"
RANGES_PATH = REPO_ROOT / "ontology" / "data" / "physiologic_ranges.json"


# Curated causal pairs (drug → side effect, disease → finding) used to
# generate InvertedRelation examples. Drawn from the existing
# metrics/ris.py CAUSAL_KG so the reasoner's KG matches.
CAUSAL_PAIRS = [
    ("insulin", "hypoglycemia"),
    ("warfarin", "bleeding"),
    ("heparin", "bleeding"),
    ("aspirin", "bleeding"),
    ("furosemide", "hypokalemia"),
    ("furosemide", "dehydration"),
    ("prednisone", "hyperglycemia"),
    ("amlodipine", "edema"),
    ("lisinopril", "cough"),
    ("metformin", "lactic acidosis"),
    ("myocardial infarction", "elevated troponin"),
    ("myocardial infarction", "chest pain"),
    ("heart failure", "edema"),
    ("sepsis", "elevated lactate"),
    ("type 2 diabetes mellitus", "hyperglycemia"),
    ("hypothyroidism", "fatigue"),
    ("anemia", "fatigue"),
    ("pneumonia", "fever"),
    ("asthma", "wheezing"),
    ("acute kidney injury", "elevated creatinine"),
]


# =====================================================
# Synthetic generators
# =====================================================

def gen_forbidden_relation(max_per_drug: int = 2,
                           total_cap: int = 50) -> list[dict]:
    """One example per (drug, contraindicated_condition) pair."""
    out = []
    if not CONTRA_PATH.exists():
        return out
    contra_data = json.loads(CONTRA_PATH.read_text())
    drugs = contra_data.get("drugs", {})
    for drug_name, info in drugs.items():
        for cond in info.get("contraindications", [])[:max_per_drug]:
            out.append({
                "id": f"syn_forb_{len(out):04d}",
                "source": "synthetic_contraindication",
                "text": f"{drug_name.capitalize()} treats {cond}.",
                "expected_class": "ForbiddenRelation",
                "expected_harm_tiers": ["T3_Actionable", "T4_LifeThreatening"],
                "expected_evidence_kind": "forbidden_relation",
                "is_hallucination": True,
            })
            if len(out) >= total_cap:
                return out
    return out


def gen_inverted_relation() -> list[dict]:
    """Flip each curated causal pair."""
    out = []
    for subj, obj in CAUSAL_PAIRS:
        # Flipped form: "<obj> causes <subj>" (e.g. "Hypoglycemia causes insulin")
        out.append({
            "id": f"syn_inv_{len(out):04d}",
            "source": "synthetic_inversion",
            "text": f"{obj.capitalize()} causes {subj}.",
            "expected_class": "InvertedRelation",
            "expected_harm_tiers": ["T3_Actionable", "T4_LifeThreatening"],
            "expected_evidence_kind": "inverted_relation",
            "is_hallucination": True,
            "underlying_correct": f"{subj.capitalize()} causes {obj}.",
        })
    return out


def gen_threshold_violation() -> list[dict]:
    """Generate lab values outside the physiologic range."""
    out = []
    if not RANGES_PATH.exists():
        return out
    data = json.loads(RANGES_PATH.read_text())
    labs = data.get("labs", {})
    for lab_name, info in labs.items():
        unit = info.get("unit") or info.get("units", "")
        # Above plausible_max
        high = info.get("plausible_max", 0) * 1.5 + 1
        out.append({
            "id": f"syn_thr_{len(out):04d}",
            "source": "synthetic_threshold_high",
            "text": f"Patient's {lab_name} was {high} {unit}.",
            "expected_class": "ThresholdViolation",
            "expected_harm_tiers": ["T4_LifeThreatening"],
            "expected_evidence_kind": "threshold_violation",
            "is_hallucination": True,
        })
        # Below plausible_min (when meaningful)
        low_min = info.get("plausible_min", 0)
        if low_min > 0:
            low = max(0, low_min - low_min * 0.5)
            out.append({
                "id": f"syn_thr_{len(out):04d}",
                "source": "synthetic_threshold_low",
                "text": f"Patient's {lab_name} was {low} {unit}.",
                "expected_class": "ThresholdViolation",
                "expected_harm_tiers": ["T4_LifeThreatening"],
                "expected_evidence_kind": "threshold_violation",
                "is_hallucination": True,
            })
    return out


def gen_modifier_omission() -> list[dict]:
    """ModifierOmission requires source-vs-output alignment to be detectable
    (you can't know a fever-claim is wrong without seeing the source said
    'no fever'). Single-text Mode 2/3 evaluation cannot meaningfully score
    it; we keep one smoke example so the class is represented in the
    gold set but the experiments will not weight it."""
    return [
        {
            "id": "syn_mod_0001",
            "source": "synthetic_negation",
            "text": "Patient has fever.",
            "expected_class": "ModifierOmission",
            "expected_harm_tiers": ["T3_Actionable", "T4_LifeThreatening"],
            "expected_evidence_kind": "modifier_omission",
            "is_hallucination": True,
            "underlying_correct": "Patient has no fever.",
            "review_status": "single_text_unevaluable",
        },
    ]


def gen_fabricated_entity() -> list[dict]:
    """Synthetic clinical claims involving plausible-sounding-but-fake
    drug/condition/lab names. An LLM auditor with reasonable clinical
    knowledge should flag these; the regex+lexicon extractor cannot
    (Mode 2 fails by design here, which is also informative).

    Names below were constructed to look pharmaceutical/medical but
    correspond to no real UMLS CUI."""
    fake_drugs = [
        ("dipenazol", "hypertension"),
        ("vorthamine", "atrial fibrillation"),
        ("benazoquine", "type 2 diabetes mellitus"),
        ("lormetacin", "rheumatoid arthritis"),
        ("kelvostatin", "hypercholesterolemia"),
        ("flexotinib", "non-small cell lung cancer"),
        ("zalmotrexate", "psoriasis"),
        ("epicortin", "asthma"),
    ]
    fake_conditions = [
        ("amoxicillin", "hyperthalmic syndrome"),
        ("metformin", "Krenner-Wells disease"),
        ("aspirin", "endopulmonary necrosis"),
        ("furosemide", "lactimic acidosis"),  # close to but not 'lactic'
        ("warfarin", "calcinemic shock"),
        ("prednisone", "myelodysplastic vasculopathy syndrome"),
    ]
    fake_labs = [
        ("zorinase", 8.4, "U/L"),
        ("kalogenase", 142.0, "ng/mL"),
        ("trophenin-X", 0.32, "mg/dL"),
        ("cardiopeptide-3", 56.0, "pg/mL"),
        ("hemoxidase", 0.07, "mmol/L"),
    ]
    fake_procedures = [
        "transverse splenotomy",
        "endoscopic auriculogram",
        "vascular metaplasty",
        "laparoscopic neuropexy",
        "cardiopulmonary debridement",
    ]

    out = []
    for drug, cond in fake_drugs:
        out.append({
            "id": f"syn_fab_drug_{len(out):04d}",
            "source": "synthetic_fake_drug",
            "text": f"{drug.capitalize()} treats {cond}.",
            "expected_class": "FabricatedEntity",
            "expected_harm_tiers": ["T2_Misleading", "T3_Actionable",
                                    "T4_LifeThreatening"],
            "expected_evidence_kind": "fabricated_entity",
            "is_hallucination": True,
            "fabricated_term": drug,
        })
    for drug, cond in fake_conditions:
        out.append({
            "id": f"syn_fab_cond_{len(out):04d}",
            "source": "synthetic_fake_condition",
            "text": f"{drug.capitalize()} treats {cond}.",
            "expected_class": "FabricatedEntity",
            "expected_harm_tiers": ["T2_Misleading", "T3_Actionable",
                                    "T4_LifeThreatening"],
            "expected_evidence_kind": "fabricated_entity",
            "is_hallucination": True,
            "fabricated_term": cond,
        })
    for lab, value, unit in fake_labs:
        out.append({
            "id": f"syn_fab_lab_{len(out):04d}",
            "source": "synthetic_fake_lab",
            "text": f"Patient's {lab} was {value} {unit}.",
            "expected_class": "FabricatedEntity",
            "expected_harm_tiers": ["T2_Misleading", "T3_Actionable",
                                    "T4_LifeThreatening"],
            "expected_evidence_kind": "fabricated_entity",
            "is_hallucination": True,
            "fabricated_term": lab,
        })
    for proc in fake_procedures:
        out.append({
            "id": f"syn_fab_proc_{len(out):04d}",
            "source": "synthetic_fake_procedure",
            "text": f"The patient underwent {proc} without complication.",
            "expected_class": "FabricatedEntity",
            "expected_harm_tiers": ["T2_Misleading", "T3_Actionable",
                                    "T4_LifeThreatening"],
            "expected_evidence_kind": "fabricated_entity",
            "is_hallucination": True,
            "fabricated_term": proc,
        })
    return out


def gen_more_inverted_relations() -> list[dict]:
    """Additional InvertedRelation flips from a wider clinically-known
    causal KG, beyond the original CAUSAL_PAIRS. Each entry asserts the
    REVERSED direction — the gold class is InvertedRelation."""
    extra_pairs = [
        # drug → adverse effect
        ("opioid", "respiratory depression"),
        ("statin", "myopathy"),
        ("ssri", "serotonin syndrome"),
        ("cisplatin", "nephrotoxicity"),
        ("vancomycin", "red man syndrome"),
        ("amiodarone", "pulmonary fibrosis"),
        # disease → finding
        ("diabetic ketoacidosis", "anion gap acidosis"),
        ("pulmonary embolism", "tachypnea"),
        ("pneumothorax", "decreased breath sounds"),
        ("hyperkalemia", "peaked t waves"),
        ("hypocalcemia", "tetany"),
        ("addisonian crisis", "hypotension"),
        ("hypothyroidism", "bradycardia"),
        ("cushing's syndrome", "moon facies"),
        ("renal failure", "elevated bun"),
        ("hemolysis", "elevated bilirubin"),
    ]
    out = []
    for subj, obj in extra_pairs:
        out.append({
            "id": f"syn_inv2_{len(out):04d}",
            "source": "synthetic_inversion_extended",
            "text": f"{obj.capitalize()} causes {subj}.",
            "expected_class": "InvertedRelation",
            "expected_harm_tiers": ["T3_Actionable", "T4_LifeThreatening"],
            "expected_evidence_kind": "inverted_relation",
            "is_hallucination": True,
            "underlying_correct": f"{subj.capitalize()} causes {obj}.",
        })
    return out


def gen_more_threshold_violations() -> list[dict]:
    """Additional ThresholdViolation examples with multiple values per lab
    and varied phrasing — boosts the bucket from ~24 to ~40+."""
    if not RANGES_PATH.exists():
        return []
    data = json.loads(RANGES_PATH.read_text())
    labs = data.get("labs", {})
    out = []
    for lab_name, info in labs.items():
        unit = info.get("unit") or info.get("units", "")
        # Two extreme-high values
        high1 = info.get("plausible_max", 0) * 2 + 5
        high2 = info.get("plausible_max", 0) * 4 + 20
        variants = [
            (high1, f"{lab_name.capitalize()} measured {high1} {unit}."),
            (high2, f"Lab results showed {lab_name} of {high2} {unit}."),
        ]
        for _val, phrasing in variants:
            out.append({
                "id": f"syn_thr2_{len(out):04d}",
                "source": "synthetic_threshold_high_extra",
                "text": phrasing,
                "expected_class": "ThresholdViolation",
                "expected_harm_tiers": ["T4_LifeThreatening"],
                "expected_evidence_kind": "threshold_violation",
                "is_hallucination": True,
            })
    return out


def gen_negative_controls(n: int = 60) -> list[dict]:
    """Correct clinical claims — the reasoner should NOT classify these
    as any hallucination class. Expanded to ~60 examples drawn from
    correct indications, correct causal relations, normal lab values,
    and correct mechanisms."""
    correct_indications = [
        ("metformin", "type 2 diabetes mellitus"),
        ("lisinopril", "hypertension"),
        ("atorvastatin", "hypercholesterolemia"),
        ("metoprolol", "hypertension"),
        ("furosemide", "heart failure"),
        ("warfarin", "atrial fibrillation"),
        ("insulin", "type 1 diabetes mellitus"),
        ("aspirin", "myocardial infarction"),
        ("heparin", "deep vein thrombosis"),
        ("prednisone", "rheumatoid arthritis"),
        ("amlodipine", "hypertension"),
        ("losartan", "hypertension"),
        ("simvastatin", "hypercholesterolemia"),
        ("atenolol", "hypertension"),
        ("hydrochlorothiazide", "hypertension"),
        ("spironolactone", "heart failure"),
        ("acetaminophen", "fever"),
        ("ibuprofen", "pain"),
        ("amoxicillin", "bacterial infection"),
        ("ceftriaxone", "pneumonia"),
    ]
    correct_causal = [
        ("insulin", "hypoglycemia"),
        ("myocardial infarction", "elevated troponin"),
        ("sepsis", "elevated lactate"),
        ("heart failure", "edema"),
        ("pneumonia", "fever"),
        ("warfarin", "bleeding"),
        ("furosemide", "hypokalemia"),
        ("hypothyroidism", "fatigue"),
        ("anemia", "fatigue"),
        ("asthma", "wheezing"),
        ("acute kidney injury", "elevated creatinine"),
        ("type 2 diabetes mellitus", "hyperglycemia"),
        ("prednisone", "hyperglycemia"),
        ("lisinopril", "cough"),
        ("aspirin", "bleeding"),
    ]
    # Correct lab values within normal range
    correct_labs = [
        ("Patient's potassium was 4.0 mEq/L.", "potassium"),
        ("Patient's sodium was 140 mEq/L.", "sodium"),
        ("Glucose 95 mg/dL.", "glucose"),
        ("Creatinine 1.0 mg/dL.", "creatinine"),
        ("Hemoglobin was 14.0 g/dL.", "hemoglobin"),
        ("Patient's BUN was 12 mg/dL.", "bun"),
        ("Lactate 1.2 mmol/L.", "lactate"),
    ]
    # Correct mechanism statements
    correct_mech = [
        "Insulin lowers blood glucose by promoting cellular uptake.",
        "ACE inhibitors block conversion of angiotensin I to angiotensin II.",
        "Beta blockers reduce heart rate by antagonizing beta-1 receptors.",
        "Aspirin irreversibly inhibits cyclooxygenase.",
        "Furosemide inhibits sodium-potassium-chloride cotransport in the loop of Henle.",
        "Warfarin inhibits vitamin K epoxide reductase.",
        "Statins inhibit HMG-CoA reductase.",
        "Heparin enhances the activity of antithrombin III.",
    ]
    out = []
    for drug, dis in correct_indications:
        out.append({
            "id": f"neg_treats_{len(out):04d}",
            "source": "synthetic_correct_indication",
            "text": f"{drug.capitalize()} treats {dis}.",
            "expected_class": None,
            "expected_harm_tiers": [],
            "expected_evidence_kind": None,
            "is_hallucination": False,
        })
    for subj, obj in correct_causal:
        out.append({
            "id": f"neg_causes_{len(out):04d}",
            "source": "synthetic_correct_causal",
            "text": f"{subj.capitalize()} causes {obj}.",
            "expected_class": None,
            "expected_harm_tiers": [],
            "expected_evidence_kind": None,
            "is_hallucination": False,
        })
    for text, _lab in correct_labs:
        out.append({
            "id": f"neg_lab_{len(out):04d}",
            "source": "synthetic_correct_lab",
            "text": text,
            "expected_class": None,
            "expected_harm_tiers": [],
            "expected_evidence_kind": None,
            "is_hallucination": False,
        })
    for text in correct_mech:
        out.append({
            "id": f"neg_mech_{len(out):04d}",
            "source": "synthetic_correct_mechanism",
            "text": text,
            "expected_class": None,
            "expected_harm_tiers": [],
            "expected_evidence_kind": None,
            "is_hallucination": False,
        })
    return out[:n] if n else out


# =====================================================
# MedHallu curation (heuristic; flagged for review)
# =====================================================

def curate_medhallu_subset(n_target: int = 30) -> list[dict]:
    """Pull a small set of MedHallu examples where the hallucinated answer
    plausibly matches a CHO axiom class.

    Heuristic mapping from MedHallu's Category of Hallucination:
      - "Mechanism and Pathway Misattribution" → tentative InvertedRelation
        (these often invert a regulatory/causal direction). FLAG_REVIEW.
      - "Methodological and Evidence Fabrication" → tentative FabricatedEntity
        or ForbiddenRelation. FLAG_REVIEW.
      - Everything else: skip.

    Each entry is stamped review_status='draft' so the user knows to
    spot-check before treating it as gold.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[curate_medhallu_subset] datasets library missing; skipping")
        return []

    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial",
                      split="train")
    out: list[dict] = []
    cat_map = {
        "Mechanism and Pathway Misattribution": ("InvertedRelation",
                                                  ["T3_Actionable",
                                                   "T4_LifeThreatening"]),
        "Methodological and Evidence Fabrication": ("FabricatedEntity",
                                                     ["T2_Misleading",
                                                      "T3_Actionable",
                                                      "T4_LifeThreatening"]),
    }
    for i, row in enumerate(ds):
        cat = row.get("Category of Hallucination", "")
        if cat not in cat_map:
            continue
        expected_class, harm_tiers = cat_map[cat]
        text = (row.get("Hallucinated Answer") or "").strip()
        if not text or len(text) < 30 or len(text) > 400:
            continue
        out.append({
            "id": f"medhallu_{i}",
            "source": f"medhallu_pqa_artificial:{cat}",
            "text": text,
            "expected_class": expected_class,
            "expected_harm_tiers": harm_tiers,
            "expected_evidence_kind": None,    # let reasoner decide kind
            "is_hallucination": True,
            "underlying_correct": (row.get("Ground Truth") or "").strip(),
            "medhallu_category": cat,
            "medhallu_difficulty": row.get("Difficulty Level"),
            "review_status": "draft",
        })
        if len(out) >= n_target:
            break
    return out


# =====================================================
# Main
# =====================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--medhallu", action="store_true",
                        help="Also include curated MedHallu subset (requires "
                             "the datasets library and HF network access).")
    parser.add_argument("--medhallu-n", type=int, default=30)
    parser.add_argument("--max-forbidden", type=int, default=200,
                        help="Cap synthetic ForbiddenRelation examples.")
    parser.add_argument("--n-negative-controls", type=int, default=60)
    args = parser.parse_args()

    examples: list[dict] = []
    examples += gen_forbidden_relation(total_cap=args.max_forbidden)
    examples += gen_inverted_relation()
    examples += gen_more_inverted_relations()
    examples += gen_threshold_violation()
    examples += gen_more_threshold_violations()
    examples += gen_modifier_omission()
    examples += gen_fabricated_entity()
    examples += gen_negative_controls(n=args.n_negative_controls)
    if args.medhallu:
        print("Pulling MedHallu subset (review_status='draft' — spot-check before using)...")
        examples += curate_medhallu_subset(n_target=args.medhallu_n)

    # Stats
    by_class: dict[str | None, int] = {}
    for e in examples:
        by_class[e["expected_class"]] = by_class.get(e["expected_class"], 0) + 1
    print(f"Generated {len(examples)} gold examples:")
    for k, v in sorted(by_class.items(), key=lambda kv: (kv[0] is None, kv[0])):
        print(f"  {v:>4}  {k or '<negative control>'}")

    out = {
        "_metadata": {
            "description": "Mode 2 gold-label set: synthetic CHO-axiom-targeted "
                           "examples + (optional) curated MedHallu subset. "
                           "The reasoner is evaluated on classification "
                           "precision/recall against `expected_class`.",
            "version": "v1",
            "n_examples": len(examples),
            "n_by_class": {k or "negative_control": v
                           for k, v in by_class.items()},
        },
        "examples": examples,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
