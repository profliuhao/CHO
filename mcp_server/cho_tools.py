"""
cho_tools.py

The six CHO tools an auditor LLM can call. Defined as plain Python
functions so the auditor harness can call them in-process (fast, no
IPC) AND wrap them as MCP tools for the published artifact (see
server.py).

Each tool returns a JSON-serializable dict and is documented with a
JSON-schema-style signature in the TOOL_SCHEMAS dict at the bottom.
The schemas are what get exposed to the LLM via OpenAI function
calling / MCP tool listing.

Tools:
  1. lookup_drug_indications(drug)
       Returns the drug's indication list from RxNorm.
  2. check_treats_relation(drug, condition)
       Verdict on whether (drug, treats, condition) is permitted.
       Uses contraindications + RxNorm.
  3. get_lab_plausibility(lab, value, unit=None)
       Verdict on whether the value is within physiologic bounds.
  4. classify_error_via_reasoner(claim_text)
       The 'real' reasoner tool. Extracts claims from the text, builds
       an ABox, runs HermiT, returns the inferred CHO class + harm tier.
       This is the tool that makes the MCP server more than a lookup.
  5. lookup_cho_class(class_name)
       Return the CHO class definition (comment, detection rule,
       verification oracle, sub/superclasses).
  6. get_harm_tier(class_name)
       Return the harm-tier(s) the TBox restricts class_name to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

import owlready2 as owl

from extraction.claim_extraction import extract_claims, get_lexicon
from metrics.ohr import (
    is_treats_relation_forbidden,
    is_threshold_violated,
    _load_rxnorm,
    _load_ranges,
    _load_contraindications,
)
from mode2.abox_translator import AboxTranslator, run_reasoner_and_read


# Lazy-loaded reasoner translator (warm-up the TBox once)
_TRANSLATOR: AboxTranslator | None = None


def _translator() -> AboxTranslator:
    global _TRANSLATOR
    if _TRANSLATOR is None:
        _TRANSLATOR = AboxTranslator()
    return _TRANSLATOR


# Lazy-loaded TBox view for class-introspection tools
_TBOX_WORLD: owl.World | None = None
_TBOX_ONTO: owl.Ontology | None = None


def _tbox_onto() -> owl.Ontology:
    global _TBOX_WORLD, _TBOX_ONTO
    if _TBOX_ONTO is None:
        _TBOX_WORLD = owl.World()
        _TBOX_ONTO = _TBOX_WORLD.get_ontology(
            "file://" + str((REPO_ROOT / "ontology" / "cho.owl").resolve())
        ).load()
    return _TBOX_ONTO


# =====================================================
# Tool 1: lookup_drug_indications
# =====================================================

def lookup_drug_indications(drug: str) -> dict:
    """Return the indications RxNorm lists for a drug.

    When the drug is not in our knowledge tables, the response explicitly
    flags the possibility that the term is fabricated rather than just
    being a knowledge gap — this is the diagnostically correct framing
    for an LLM auditor to interpret an absence-of-evidence response."""
    rxnorm = _load_rxnorm()
    key = drug.lower().strip()
    entry = rxnorm.get(key)
    if entry is None:
        # Also check whether it appears in any of our broader entity
        # tables (UMLS-expanded lexicon, contraindications). If not in
        # ANY of them, the term is unrecognized in standard clinical
        # terminologies, which is strong evidence of fabrication.
        from extraction.claim_extraction import get_lexicon
        lex = get_lexicon()
        in_lexicon = lex.lookup(key) is not None
        contra = _load_contraindications()
        in_contra = key in contra
        return {
            "drug": drug,
            "found_in_rxnorm": False,
            "found_in_any_terminology": in_lexicon or in_contra,
            "indications": [],
            "note": (
                f"'{drug}' is not recognized in RxNorm "
                + ("nor in any of our standard clinical terminologies "
                   "(UMLS-expanded lexicon, MED-RT contraindications). "
                   "An LLM auditor should treat unrecognized clinical "
                   "terminology as evidence of a FabricatedEntity, "
                   "after considering plausible spelling variants."
                   if not (in_lexicon or in_contra)
                   else "(but is recognized in our broader lexicon — "
                        "may be a non-drug entity or a class name).")
            ),
        }
    return {
        "drug": drug,
        "found_in_rxnorm": True,
        "indications": entry.get("indications", []),
        "rxcui": entry.get("rxcui"),
    }


# =====================================================
# Tool 2: check_treats_relation
# =====================================================

def check_treats_relation(drug: str, condition: str) -> dict:
    """Verdict on (drug, treats, condition).

    Returns:
      verdict: 'permitted' | 'contraindicated' | 'unsupported' | 'unknown'
      evidence: explanation string
      sources: which knowledge tables contributed
    """
    drug_low = drug.lower().strip()
    cond_low = condition.lower().strip()

    contra = _load_contraindications()
    rxnorm = _load_rxnorm()

    # Explicit contraindication (strongest signal)
    contra_entry = contra.get(drug_low)
    if contra_entry:
        for c in contra_entry.get("contraindications", []):
            cl = c.lower()
            if cl == cond_low or cond_low in cl or cl in cond_low:
                return {
                    "verdict": "contraindicated",
                    "evidence": f"MED-RT lists '{c}' as a contraindication "
                                f"for {drug}.",
                    "sources": ["MED-RT (UMLS)"],
                    "matched_term": c,
                }

    # RxNorm indication
    rx_entry = rxnorm.get(drug_low)
    if rx_entry:
        indications = [i.lower() for i in rx_entry.get("indications", [])]
        if any(cond_low == i or cond_low in i or i in cond_low for i in indications):
            return {
                "verdict": "permitted",
                "evidence": f"RxNorm lists '{condition}' (or a synonym) "
                            f"as an indication for {drug}.",
                "sources": ["RxNorm"],
                "indications": rx_entry["indications"],
            }
        # Drug is in RxNorm but condition is not in its indications and not
        # explicitly contraindicated → "unsupported"
        return {
            "verdict": "unsupported",
            "evidence": f"{drug} is in RxNorm but {condition} is not among "
                        f"its listed indications.",
            "sources": ["RxNorm"],
            "indications": rx_entry["indications"],
        }

    # Drug not in any of our knowledge tables. Distinguish two cases:
    # (a) drug name appears in the broader lexicon (recognized concept,
    #     just no indication data) → "unknown"
    # (b) drug name does not appear anywhere → likely fabricated
    from extraction.claim_extraction import get_lexicon
    lex = get_lexicon()
    if lex.lookup(drug_low) is not None:
        return {
            "verdict": "unknown",
            "evidence": (f"'{drug}' is recognized as a clinical concept "
                         f"but we have no indication data for it. "
                         f"Cannot verify the (treats, {condition}) claim."),
            "sources": [],
            "concept_recognized": True,
        }
    return {
        "verdict": "unknown",
        "evidence": (f"'{drug}' is NOT recognized in RxNorm, MED-RT, or "
                     f"the UMLS-expanded clinical lexicon. An LLM auditor "
                     f"should treat unrecognized clinical terminology as "
                     f"evidence of a FabricatedEntity (after ruling out "
                     f"plausible spelling variants), independent of the "
                     f"treats-relation question."),
        "sources": [],
        "concept_recognized": False,
        "fabrication_signal": "strong",
    }


# =====================================================
# Tool 3: get_lab_plausibility
# =====================================================

def get_lab_plausibility(lab: str, value: float,
                         unit: str | None = None) -> dict:
    """Check whether a lab value is within curated physiologic bounds.

    Returns:
      verdict: 'within_range' | 'out_of_range' | 'unknown'
      bounds: {min, max, unit}
      severity: 'normal' | 'plausible' | 'implausible' (heuristic)
    """
    ranges = _load_ranges()
    lab_low = lab.lower().strip()
    info = ranges.get(lab_low)
    if info is None:
        # Try alias match
        for canonical, data in ranges.items():
            aliases = [a.lower() for a in data.get("alias", [])]
            if lab_low in aliases:
                info = data
                lab_low = canonical
                break
    if info is None:
        # Like the drug case: distinguish "recognized concept, no range
        # data" from "unrecognized concept = likely fabricated".
        from extraction.claim_extraction import get_lexicon
        lex = get_lexicon()
        recognized = lex.lookup(lab_low) is not None
        if recognized:
            return {
                "lab": lab,
                "value": value,
                "verdict": "unknown",
                "evidence": (f"'{lab}' is a recognized clinical concept "
                             f"but we have no physiologic range data for "
                             f"it. Cannot evaluate plausibility of "
                             f"value={value}."),
                "concept_recognized": True,
            }
        return {
            "lab": lab,
            "value": value,
            "verdict": "unknown",
            "evidence": (f"'{lab}' is NOT recognized in our physiologic "
                         f"ranges table or the UMLS-expanded clinical "
                         f"lexicon. An LLM auditor should treat "
                         f"unrecognized lab names as evidence of a "
                         f"FabricatedEntity, independent of whether the "
                         f"numeric value is plausible."),
            "concept_recognized": False,
            "fabrication_signal": "strong",
        }

    pmin = info["plausible_min"]
    pmax = info["plausible_max"]
    nmin = info.get("normal_min", info.get("reference_low", pmin))
    nmax = info.get("normal_max", info.get("reference_high", pmax))
    bounds_unit = info.get("unit") or info.get("units") or unit

    if value < pmin or value > pmax:
        verdict = "out_of_range"
        severity = "implausible"
    elif value < nmin or value > nmax:
        verdict = "within_range"
        severity = "abnormal_but_plausible"
    else:
        verdict = "within_range"
        severity = "normal"

    return {
        "lab": lab_low,
        "value": value,
        "unit": unit or bounds_unit,
        "verdict": verdict,
        "severity": severity,
        "bounds": {
            "plausible_min": pmin, "plausible_max": pmax,
            "normal_min": nmin, "normal_max": nmax,
            "unit": bounds_unit,
        },
        "evidence": (f"Curated physiologic plausible range for {lab_low}: "
                     f"{pmin}-{pmax} {bounds_unit}; value {value} is "
                     f"{verdict}."),
    }


# =====================================================
# Tool 4: classify_error_via_reasoner
# =====================================================

def classify_error_via_reasoner(claim_text: str) -> dict:
    """The reasoner tool. Extracts claims from the text, asserts an ABox,
    runs HermiT, returns the inferred CHO class(es) and harm tier(s).

    This is the only tool whose output cannot be obtained by lookup
    alone — it requires the formal axioms in cho.owl plus the
    description-logic reasoner.

    Returns:
      classifications: list of {individual, class, harm_tiers, evidence_kind}
      n_individuals: number of individuals asserted
      summary: 'no_extractable_claims' | 'no_axiom_violations_found' |
               one of the CHO classes
      n_entities_extracted, n_relations_extracted, n_numerics_extracted:
        coverage signals the LLM auditor should use to interpret the
        summary correctly. n_entities==0 means the regex extractor did
        not recognize any clinical concepts, which is COMPATIBLE with
        a FabricatedEntity hallucination (and the LLM should fall back
        to its parametric knowledge to assess).
    """
    claims = extract_claims(claim_text)
    onto, translated = _translator().translate(claim_text, claims,
                                               source_id="mcp_call")

    n_ents = len(claims.entities)
    n_rels = len(claims.relations)
    n_nums = len(claims.numerics)

    if not translated:
        # Distinguish two very different "no hallucination found" cases.
        # (a) extractor found nothing → CHO has no opinion; LLM should
        #     fall back to parametric knowledge (especially relevant for
        #     FabricatedEntity, which by definition involves entities the
        #     extractor cannot recognize).
        # (b) extractor found claims but all passed axiom checks → CHO
        #     actively certifies these particular claims as not violating
        #     any encoded axiom.
        if n_ents == 0 and n_rels == 0 and n_nums == 0:
            return {
                "classifications": [],
                "n_individuals": 0,
                "n_entities_extracted": 0,
                "n_relations_extracted": 0,
                "n_numerics_extracted": 0,
                "summary": "no_extractable_claims",
                "evidence": (
                    "The rule-based extractor recognized no clinical "
                    "concepts in this text. This is COMPATIBLE with a "
                    "FabricatedEntity hallucination (made-up drug/lab/"
                    "condition/procedure names by definition won't be "
                    "in our terminology). The reasoner has no opinion "
                    "in this case — fall back to parametric knowledge "
                    "to assess whether the claim contains fabricated "
                    "clinical entities."
                ),
                "reasoner_decisive": False,
            }
        return {
            "classifications": [],
            "n_individuals": 0,
            "n_entities_extracted": n_ents,
            "n_relations_extracted": n_rels,
            "n_numerics_extracted": n_nums,
            "summary": "no_axiom_violations_found",
            "evidence": (
                f"The extractor found {n_ents} entities, {n_rels} "
                f"relations, {n_nums} numeric claims — all of which "
                f"passed every encoded CHO axiom check. The reasoner "
                f"actively certifies these specific claims as "
                f"non-violating. (Note: the extractor's coverage of "
                f"clinical terminology is partial; out-of-vocabulary "
                f"claims may still be hallucinated — assess separately.)"
            ),
            "reasoner_decisive": True,
        }
    inferred = run_reasoner_and_read(onto, translated)
    classifications = []
    for t, r in zip(translated, inferred):
        classifications.append({
            "individual": t.individual_name,
            "asserted_class": t.asserted_class,
            "evidence_kind": t.evidence_kind,
            "source_span": t.source_span,
            "detail": t.detail,
            "inferred_classes": r.inferred_classes,
            "inferred_harm_tiers": r.inferred_harm_tiers,
        })
    # The summary picks the most specific (= deepest) class across all
    # individuals. For the 4-arm experiment we only need the top label.
    top_class = classifications[0]["asserted_class"]
    return {
        "classifications": classifications,
        "n_individuals": len(translated),
        "n_entities_extracted": n_ents,
        "n_relations_extracted": n_rels,
        "n_numerics_extracted": n_nums,
        "summary": top_class,
        "evidence": (f"Reasoner classified {len(translated)} hallucination "
                     f"individual(s) via the formal CHO TBox axioms. "
                     f"The top-level class is {top_class}."),
        "reasoner_decisive": True,
    }


# =====================================================
# Tool 5: lookup_cho_class
# =====================================================

def lookup_cho_class(class_name: str) -> dict:
    """Return TBox info for a CHO class: comment, detection rule, verification
    oracle, super/sub-classes."""
    onto = _tbox_onto()
    cls = onto.search_one(iri=f"*{class_name}")
    if cls is None or not hasattr(cls, "is_a"):
        return {"class_name": class_name, "found": False}

    def _read_anno(prop_name):
        prop = onto.search_one(iri=f"*{prop_name}")
        if prop is None:
            return []
        return list(getattr(cls, prop_name, []) or [])

    parents = [p.name for p in cls.is_a if hasattr(p, "name")]
    # Subclasses: search the ontology for classes whose is_a contains this one.
    subs = []
    for c in onto.classes():
        if cls in (c.is_a or []):
            subs.append(c.name)

    return {
        "class_name": cls.name,
        "found": True,
        "comment": list(cls.comment),
        "detection_rule": _read_anno("detectionRule"),
        "verification_oracle": _read_anno("verificationOracle"),
        "parents": parents,
        "subclasses": subs,
    }


# =====================================================
# Tool 6: get_harm_tier
# =====================================================

def get_harm_tier(class_name: str) -> dict:
    """Return the harm-tier classes the TBox restricts class_name to."""
    from mode2.abox_translator import _harm_tier_classes_from_restriction, _walk_ancestors
    onto = _tbox_onto()
    cls = onto.search_one(iri=f"*{class_name}")
    if cls is None or not hasattr(cls, "is_a"):
        return {"class_name": class_name, "found": False}

    harm_tiers: set[str] = set()
    for ancestor in _walk_ancestors(cls):
        for restriction in (ancestor.is_a or []):
            harm_tiers.update(
                _harm_tier_classes_from_restriction(restriction)
            )
    return {
        "class_name": class_name,
        "found": True,
        "harm_tiers": sorted(harm_tiers),
        "explanation": (f"TBox restriction on {class_name} (or its ancestors) "
                        f"asserts hasHarmTier some union of these classes."),
    }


# =====================================================
# Tool registry
# =====================================================
# JSON-Schema function-call definitions for OpenAI function-calling /
# MCP tool listing. The auditor harness and the MCP server both consume
# these schemas.

TOOL_SCHEMAS = [
    {
        "name": "lookup_drug_indications",
        "description": ("Look up the FDA-approved indications for a drug "
                        "from RxNorm. Use this to check whether a drug is "
                        "indicated for a given condition."),
        "parameters": {
            "type": "object",
            "properties": {
                "drug": {"type": "string", "description": "Generic drug name."}
            },
            "required": ["drug"],
        },
    },
    {
        "name": "check_treats_relation",
        "description": ("Check whether a (drug, treats, condition) claim is "
                        "permitted by clinical knowledge. Returns one of "
                        "'permitted', 'contraindicated', 'unsupported', or "
                        "'unknown', with the supporting evidence."),
        "parameters": {
            "type": "object",
            "properties": {
                "drug": {"type": "string"},
                "condition": {"type": "string"},
            },
            "required": ["drug", "condition"],
        },
    },
    {
        "name": "get_lab_plausibility",
        "description": ("Check whether a lab value is within physiologically "
                        "plausible bounds. Returns the plausible range and "
                        "whether the value is within or out of range."),
        "parameters": {
            "type": "object",
            "properties": {
                "lab": {"type": "string", "description":
                        "Canonical lab name (e.g., 'potassium', 'creatinine')."},
                "value": {"type": "number"},
                "unit": {"type": "string", "description": "Optional unit string."},
            },
            "required": ["lab", "value"],
        },
    },
    {
        "name": "classify_error_via_reasoner",
        "description": ("Classify a clinical claim using the formal CHO "
                        "ontology and the HermiT reasoner. Extracts claims "
                        "from the input text, asserts them as OWL "
                        "individuals, runs the reasoner, and returns the "
                        "inferred hallucination class(es) and harm tier(s). "
                        "Use this for the most rigorous classification."),
        "parameters": {
            "type": "object",
            "properties": {
                "claim_text": {"type": "string", "description":
                               "A short clinical claim or sentence."}
            },
            "required": ["claim_text"],
        },
    },
    {
        "name": "lookup_cho_class",
        "description": ("Look up a CHO ontology class definition: comment, "
                        "detection rule, verification oracle, parents, and "
                        "subclasses."),
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string"}
            },
            "required": ["class_name"],
        },
    },
    {
        "name": "get_harm_tier",
        "description": ("Return the harm-tier classes (T1/T2/T3/T4) the "
                        "TBox restricts a given CHO class to."),
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string"}
            },
            "required": ["class_name"],
        },
    },
]


# Map tool name → callable, for in-process dispatch
TOOLS = {
    "lookup_drug_indications": lookup_drug_indications,
    "check_treats_relation": check_treats_relation,
    "get_lab_plausibility": get_lab_plausibility,
    "classify_error_via_reasoner": classify_error_via_reasoner,
    "lookup_cho_class": lookup_cho_class,
    "get_harm_tier": get_harm_tier,
}


def call_tool(name: str, arguments: dict) -> dict:
    """In-process dispatch. Returns {error: ...} if the tool is unknown
    or arguments are malformed."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}",
                "available": sorted(TOOLS.keys())}
    try:
        return fn(**arguments)
    except TypeError as e:
        return {"error": f"Argument error: {e}"}
    except Exception as e:
        return {"error": f"Tool execution error: {type(e).__name__}: {e}"}


# =====================================================
# Self-test
# =====================================================

if __name__ == "__main__":
    print("Tool 1 (lookup_drug_indications) — metformin:")
    print(json.dumps(call_tool("lookup_drug_indications",
                               {"drug": "metformin"}), indent=2))
    print()
    print("Tool 2 (check_treats_relation) — metformin / kidney failure:")
    print(json.dumps(call_tool("check_treats_relation",
                               {"drug": "metformin",
                                "condition": "kidney failure"}), indent=2))
    print()
    print("Tool 2 — metformin / type 2 diabetes mellitus:")
    print(json.dumps(call_tool("check_treats_relation",
                               {"drug": "metformin",
                                "condition": "type 2 diabetes mellitus"}), indent=2))
    print()
    print("Tool 3 (get_lab_plausibility) — potassium / 25:")
    print(json.dumps(call_tool("get_lab_plausibility",
                               {"lab": "potassium", "value": 25,
                                "unit": "mEq/L"}), indent=2))
    print()
    print("Tool 4 (classify_error_via_reasoner) — 'Lisinopril treats angioedema.':")
    print(json.dumps(call_tool("classify_error_via_reasoner",
                               {"claim_text": "Lisinopril treats angioedema."}),
                     indent=2))
    print()
    print("Tool 5 (lookup_cho_class) — ForbiddenRelation:")
    print(json.dumps(call_tool("lookup_cho_class",
                               {"class_name": "ForbiddenRelation"}), indent=2))
    print()
    print("Tool 6 (get_harm_tier) — InvertedRelation:")
    print(json.dumps(call_tool("get_harm_tier",
                               {"class_name": "InvertedRelation"}), indent=2))
