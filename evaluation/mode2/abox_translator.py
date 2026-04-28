"""
abox_translator.py

Translate an ExtractedClaims structure (from claim_extraction.py) into an
OWL ABox layered on top of the CHO TBox. Each candidate hallucination
becomes a NamedIndividual with:

  - Asserted parent class (FabricatedEntity / ForbiddenRelation /
    InvertedRelation / ThresholdViolation / ModifierOmission) derived
    from the same external-data lookups OHR/RIS would use.
  - Asserted data properties carrying the evidence (source text,
    numeric value, negation flag, etc.).

The reasoner then propagates the TBox SubClassOf restrictions on
ForbiddenRelation / InvertedRelation / ... to assert the harm tier
(hasHarmTier some T3 ∪ T4, etc.).

Architectural note: the translator does the external-knowledge lookups
(RxNorm, contraindications, physiologic ranges) because the reasoner
cannot itself query external graphs. The reasoner's contribution is the
class-hierarchy traversal and harm-tier propagation — a chain that
would not be derivable without the formal TBox axioms.

Output: a fresh owlready2 ontology containing the cho.owl TBox plus the
asserted ABox individuals, ready to feed to sync_reasoner_hermit().
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import owlready2 as owl

# Allow imports from sibling packages
HERE = Path(__file__).parent
EVAL_ROOT = HERE.parent
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))

from extraction.claim_extraction import ExtractedClaims, get_lexicon
from metrics.ohr import (
    is_treats_relation_forbidden,
    is_threshold_violated,
    _load_contraindications,
)
from metrics.ris import is_relation_inverted

CHO_OWL_PATH = REPO_ROOT / "ontology" / "cho.owl"


# =====================================================
# Result dataclass
# =====================================================

@dataclass
class TranslatedHallucination:
    """One asserted hallucination instance (pre-reasoner)."""
    individual_name: str
    asserted_class: str         # CHO class name
    evidence_kind: str          # 'forbidden_relation' / 'inverted_relation' /
                                # 'threshold_violation' / 'modifier_omission' /
                                # 'fabricated_entity'
    source_span: str            # human-readable text snippet
    detail: dict                # type-specific evidence


# =====================================================
# Translator
# =====================================================

class AboxTranslator:
    """Stateful translator: loads the TBox once, creates fresh ABox per call.

    Reusing the TBox across calls is faster than re-loading cho.owl per
    example. Each translate() returns a fresh world so individuals from
    different texts don't pollute each other.
    """

    def __init__(self, tbox_path: Path = CHO_OWL_PATH):
        self.tbox_path = tbox_path
        # Load once into a "template" world; we'll clone for each call.
        # Owlready2's quad-store is mutable across worlds, so per-call we
        # use a fresh in-memory world.
        self._tbox_iri = "file://" + str(tbox_path.resolve())

    def _fresh_world(self) -> owl.World:
        w = owl.World()
        w.get_ontology(self._tbox_iri).load()
        return w

    def translate(self, text: str, claims: ExtractedClaims,
                  source_id: str = "doc") -> tuple[owl.Ontology, list[TranslatedHallucination]]:
        """Build an ABox for one piece of generated text.

        Returns (ontology, list of TranslatedHallucination). The
        ontology is ready to feed to sync_reasoner_hermit().
        """
        world = self._fresh_world()
        onto = world.get_ontology(self._tbox_iri)

        # Resolve TBox classes by name (owlready2 puts them on `onto`).
        tbox = {
            "ClinicalHallucination": onto.ClinicalHallucination,
            "FabricatedEntity": onto.FabricatedEntity,
            "ForbiddenRelation": onto.ForbiddenRelation,
            "InvertedRelation": onto.InvertedRelation,
            "ThresholdViolation": onto.ThresholdViolation,
            "ModifierOmission": onto.ModifierOmission,
            # Properties
            "hasNumericValue": onto.hasNumericValue,
            "hasUnit": onto.hasUnit,
            "derivedFromSourceText": onto.derivedFromSourceText,
            "hasNegation": onto.hasNegation,
        }

        translated: list[TranslatedHallucination] = []
        idx = 0   # individual counter for unique naming

        with onto:
            lex = get_lexicon()

            # -------- Fabricated entities --------
            # An entity is "fabricated" if its surface didn't map to any
            # known canonical (the regex extractor only emits known ones,
            # so this is rare; we keep the hook for the scispacy/UMLS
            # linker upgrade where unknown candidates can flow through).
            for ent in claims.entities:
                lookup = lex.lookup(ent.canonical)
                if lookup is None:
                    name = f"{source_id}_fab_{idx}"; idx += 1
                    ind = tbox["FabricatedEntity"](name)
                    ind.derivedFromSourceText = text[max(0, ent.span[0]-20):
                                                     min(len(text), ent.span[1]+20)]
                    translated.append(TranslatedHallucination(
                        individual_name=name,
                        asserted_class="FabricatedEntity",
                        evidence_kind="fabricated_entity",
                        source_span=ent.surface,
                        detail={"canonical": ent.canonical},
                    ))

            # -------- Forbidden relations (A1) --------
            for r in claims.relations:
                if r.predicate != "treats":
                    continue
                if r.subject.category != "drug":
                    continue
                if r.object.category not in ("disease", "mental_disorder",
                                             "symptom", "lab_finding"):
                    continue
                verdict = is_treats_relation_forbidden(r.subject.canonical,
                                                       r.object.canonical)
                if verdict is True:
                    name = f"{source_id}_forb_{idx}"; idx += 1
                    ind = tbox["ForbiddenRelation"](name)
                    ind.derivedFromSourceText = (
                        f"{r.subject.canonical} treats {r.object.canonical}"
                    )
                    ind.hasNegation = bool(r.negated)
                    translated.append(TranslatedHallucination(
                        individual_name=name,
                        asserted_class="ForbiddenRelation",
                        evidence_kind="forbidden_relation",
                        source_span=text[r.span[0]:r.span[1]],
                        detail={"drug": r.subject.canonical,
                                "disease": r.object.canonical},
                    ))

            # -------- Inverted relations (A5) --------
            for r in claims.relations:
                if not is_relation_inverted(r):
                    continue
                name = f"{source_id}_inv_{idx}"; idx += 1
                ind = tbox["InvertedRelation"](name)
                ind.derivedFromSourceText = (
                    f"{r.subject.canonical} {r.predicate} {r.object.canonical}"
                )
                ind.hasNegation = bool(r.negated)
                translated.append(TranslatedHallucination(
                    individual_name=name,
                    asserted_class="InvertedRelation",
                    evidence_kind="inverted_relation",
                    source_span=text[r.span[0]:r.span[1]],
                    detail={"subject": r.subject.canonical,
                            "predicate": r.predicate,
                            "object": r.object.canonical},
                ))

            # -------- Threshold violations (A2) --------
            for n in claims.numerics:
                if n.value != n.value:        # NaN
                    continue
                verdict = is_threshold_violated(n.lab, n.value)
                if verdict is True:
                    name = f"{source_id}_thr_{idx}"; idx += 1
                    ind = tbox["ThresholdViolation"](name)
                    ind.hasNumericValue = float(n.value)
                    if n.unit:
                        ind.hasUnit = n.unit
                    ind.hasNegation = bool(n.negated)
                    ind.derivedFromSourceText = (
                        f"{n.lab} = {n.value}{(' ' + n.unit) if n.unit else ''}"
                    )
                    translated.append(TranslatedHallucination(
                        individual_name=name,
                        asserted_class="ThresholdViolation",
                        evidence_kind="threshold_violation",
                        source_span=text[n.span[0]:n.span[1]],
                        detail={"lab": n.lab, "value": n.value,
                                "unit": n.unit},
                    ))

            # -------- Modifier omission (A3) --------
            # A negated entity in extracted claims paired with a positive
            # surface form elsewhere in the text counts as potential
            # modifier loss. (Tighter detection requires source/output
            # alignment, deferred until we have both texts in Mode 3.)
            # Here we only fire when the extractor explicitly flags a
            # negated entity on the *output* side.
            for ent in claims.entities:
                if not ent.negated:
                    continue
                name = f"{source_id}_mod_{idx}"; idx += 1
                ind = tbox["ModifierOmission"](name)
                ind.derivedFromSourceText = text[max(0, ent.span[0]-30):
                                                 min(len(text), ent.span[1]+30)]
                ind.hasNegation = True
                translated.append(TranslatedHallucination(
                    individual_name=name,
                    asserted_class="ModifierOmission",
                    evidence_kind="modifier_omission",
                    source_span=ent.surface,
                    detail={"entity": ent.canonical},
                ))

        return onto, translated


# =====================================================
# Reasoner-driven inference reading
# =====================================================

@dataclass
class InferredHarmTier:
    individual_name: str
    asserted_class: str
    inferred_classes: list[str]   # all classes the reasoner placed it in
    inferred_harm_tiers: list[str]   # harm-tier classes (T2/T3/T4)


def _harm_tier_classes_from_restriction(restriction) -> list[str]:
    """Walk an owlready2 restriction and pull out the harm-tier class names
    it constrains hasHarmTier to.

    The TBox encodes axioms like:
        ForbiddenRelation.is_a.append(hasHarmTier.some(T3_Actionable | T4_LifeThreatening))

    The owlready2 representation is a Restriction with .property=hasHarmTier
    and .value either a single class or a class expression like Or([T3, T4]).
    """
    prop = getattr(restriction, "property", None)
    if prop is None or getattr(prop, "name", None) != "hasHarmTier":
        return []
    val = getattr(restriction, "value", None)
    if val is None:
        return []
    # Single class case
    if hasattr(val, "name") and not hasattr(val, "Classes"):
        return [val.name]
    # Class expression (Or, And, etc.)
    classes_attr = getattr(val, "Classes", None)
    if classes_attr is not None:
        return [c.name for c in classes_attr if hasattr(c, "name")]
    # Some union forms expose .instances or .get_class_properties; fall back
    return []


def _walk_ancestors(cls) -> list:
    """Yield the class and all its ancestors in the TBox."""
    seen = set()
    out = []
    stack = [cls]
    while stack:
        c = stack.pop()
        if c in seen or not hasattr(c, "is_a"):
            continue
        seen.add(c)
        out.append(c)
        for parent in (c.is_a or []):
            if hasattr(parent, "is_a"):  # only walk into named classes, not restrictions
                stack.append(parent)
    return out


def run_reasoner_and_read(
    onto: owl.Ontology,
    translated: list[TranslatedHallucination],
) -> list[InferredHarmTier]:
    """Run HermiT on the ABox and read out each individual's inferred
    classes plus the harm-tier classes their TBox restrictions imply.

    For an existential like `hasHarmTier some (T3 ∪ T4)`, OWL DL semantics
    cannot pick a single value, so we report the FULL set of harm-tier
    classes consistent with the TBox restriction. This is the correct
    'inference' to make from an existential — the alternative (forcing
    a single value) would require additional asserted evidence per
    individual.
    """
    with onto:
        owl.sync_reasoner_hermit(infer_property_values=True)

    out: list[InferredHarmTier] = []
    harm_tier_names = {"T1_Cosmetic", "T2_Misleading",
                       "T3_Actionable", "T4_LifeThreatening"}

    for t in translated:
        ind = onto.search_one(iri=f"*{t.individual_name}")
        if ind is None:
            out.append(InferredHarmTier(
                individual_name=t.individual_name,
                asserted_class=t.asserted_class,
                inferred_classes=[],
                inferred_harm_tiers=[],
            ))
            continue

        cls_names = [c.name for c in ind.is_a if hasattr(c, "name")]
        harm_tiers: set[str] = set()

        # 1. Direct property values (works for single-target restrictions
        #    like ThresholdViolation -> T4 only)
        try:
            for v in (ind.hasHarmTier or []):
                if hasattr(v, "name") and v.name in harm_tier_names:
                    harm_tiers.add(v.name)
        except AttributeError:
            pass

        # 2. TBox-side restrictions reachable from inferred classes (handles
        #    the union case: ForbiddenRelation -> T3 ∪ T4 yields {T3, T4})
        for c in ind.is_a:
            if not hasattr(c, "is_a"):
                continue   # restriction, not a class
            for ancestor in _walk_ancestors(c):
                for restriction in (ancestor.is_a or []):
                    harm_tiers.update(
                        _harm_tier_classes_from_restriction(restriction)
                    )

        out.append(InferredHarmTier(
            individual_name=t.individual_name,
            asserted_class=t.asserted_class,
            inferred_classes=cls_names,
            inferred_harm_tiers=sorted(harm_tiers),
        ))
    return out


# =====================================================
# Self-test
# =====================================================

if __name__ == "__main__":
    from extraction.claim_extraction import extract_claims

    samples = [
        ("metformin treats kidney failure",
         "Metformin treats kidney failure."),
        ("hypoglycemia causes insulin (inverted)",
         "Hypoglycemia causes insulin."),
        ("potassium overdose (threshold)",
         "Potassium was 25 mEq/L."),
        ("normal claim",
         "Lisinopril treats hypertension."),
    ]
    translator = AboxTranslator()
    for label, txt in samples:
        print(f"\n=== {label} ===")
        print(f"  text: {txt}")
        claims = extract_claims(txt)
        onto, translated = translator.translate(txt, claims,
                                                source_id=label.replace(" ", "_"))
        if not translated:
            print("  (no hallucination individuals asserted)")
            continue
        print(f"  asserted {len(translated)} individuals:")
        for t in translated:
            print(f"    - {t.individual_name} : {t.asserted_class}  "
                  f"(span={t.source_span!r})")
        inferred = run_reasoner_and_read(onto, translated)
        print(f"  reasoner output:")
        for r in inferred:
            print(f"    - {r.individual_name}: classes={r.inferred_classes} "
                  f"harm_tiers={r.inferred_harm_tiers}")
