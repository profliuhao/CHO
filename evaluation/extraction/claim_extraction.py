"""
claim_extraction.py

Lightweight claim extractor for clinical text. No scispacy or external NLP
models required: uses regex + a curated medical lexicon. Trades coverage
for portability so the full evaluation pipeline runs on a laptop.

For production use, replace this module with scispacy (en_core_sci_lg) +
QuickUMLS or MedCAT. The downstream metric code consumes the same
ExtractedClaim dataclass so substitution is drop-in.

Three extraction stages:
  1. EntityExtractor: find concept mentions and map to canonical names.
  2. RelationExtractor: find (subject, relation, object) triples.
  3. NumericExtractor: find lab values with units.
  Plus negation detection on extracted spans.

Usage:
    from claim_extraction import extract_claims
    claims = extract_claims("Patient on metformin for diabetes.")
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


SEED_LEXICON_PATH = Path(__file__).parent / "medical_lexicon.json"
UMLS_LEXICON_PATH = Path(__file__).parent / "medical_lexicon_umls.json"
RELATION_LEXICON_PATH = Path(__file__).parent / "relation_lexicon.json"
CONTRAINDICATIONS_PATH = (Path(__file__).parent.parent.parent / "ontology"
                          / "data" / "contraindications.json")


def _default_lexicon_path() -> Path:
    """Resolve which lexicon to use.

    Priority:
      1. CHO_LEXICON_PATH env var (explicit override).
      2. medical_lexicon_umls.json if present (UMLS-expanded).
      3. medical_lexicon.json (curated seed lexicon).
    """
    env = os.environ.get("CHO_LEXICON_PATH")
    if env:
        return Path(env)
    if UMLS_LEXICON_PATH.exists():
        return UMLS_LEXICON_PATH
    return SEED_LEXICON_PATH


LEXICON_PATH = _default_lexicon_path()


# =====================================================
# Data classes returned by the extractor
# =====================================================

@dataclass
class Entity:
    """A concept mention with its canonical normalized form."""
    surface: str            # text as it appeared
    canonical: str          # normalized canonical name
    category: str           # drug / disease / symptom / lab_finding / anatomy / procedure / mental_disorder
    span: tuple[int, int]   # character offsets in source
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "surface": self.surface, "canonical": self.canonical,
            "category": self.category, "span": list(self.span),
            "negated": self.negated,
        }


@dataclass
class Relation:
    """A binary relation between two entities."""
    subject: Entity
    predicate: str          # canonical predicate name (treats / causes /
                            # activates / inhibits / regulates / ...)
    object: Entity
    span: tuple[int, int]   # span of the predicate phrase
    negated: bool = False
    category: str = "unknown"   # therapeutic / causal / mechanistic_positive
                                # / mechanistic_negative / regulatory /
                                # structural / diagnostic

    def to_dict(self) -> dict:
        return {
            "subject": self.subject.to_dict(),
            "predicate": self.predicate,
            "object": self.object.to_dict(),
            "span": list(self.span),
            "negated": self.negated,
            "category": self.category,
        }


@dataclass
class NumericClaim:
    """A numeric clinical claim: a lab name, value, and optional unit."""
    lab: str                # canonical lab name
    value: float
    unit: Optional[str]
    direction: Optional[str]   # 'high' / 'low' / 'normal' / None
    span: tuple[int, int]
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "lab": self.lab, "value": self.value, "unit": self.unit,
            "direction": self.direction, "span": list(self.span),
            "negated": self.negated,
        }


@dataclass
class ExtractedClaims:
    """Container for everything extracted from one text."""
    text: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    numerics: list[NumericClaim] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "numerics": [n.to_dict() for n in self.numerics],
        }


# =====================================================
# Lexicon loading
# =====================================================

class Lexicon:
    """Loads the curated medical vocabulary once, reuses across calls."""

    def __init__(self, path: Path = LEXICON_PATH):
        with open(path) as fh:
            data = json.load(fh)

        self.entity_categories = {}
        # Build category -> list of canonical names
        for cat in ("drugs", "diseases", "mental_disorders", "symptoms",
                    "lab_findings", "anatomy", "procedures",
                    "infections_and_conditions"):
            for term in data.get(cat, []):
                cat_norm = {
                    "drugs": "drug", "diseases": "disease",
                    "mental_disorders": "mental_disorder",
                    "symptoms": "symptom", "lab_findings": "lab_finding",
                    "anatomy": "anatomy", "procedures": "procedure",
                    "infections_and_conditions": "disease",
                }[cat]
                self.entity_categories[term.lower()] = (term, cat_norm)

        # Aliases route to canonical
        self.aliases = {}
        for alias, canonical in data.get("drug_aliases", {}).items():
            self.aliases[alias.lower()] = (canonical, "drug")
        for alias, canonical in data.get("disease_aliases", {}).items():
            self.aliases[alias.lower()] = (canonical, "disease")

        # Auto-merge contraindication condition names as disease entities.
        # This ensures the extractor recognizes any condition named in the
        # MED-RT-derived contraindications table, so OHR's A1 check has a
        # (drug, treats, disease) triple to evaluate against.
        if CONTRAINDICATIONS_PATH.exists():
            with open(CONTRAINDICATIONS_PATH) as fh:
                contra_data = json.load(fh)
            for drug_info in contra_data.get("drugs", {}).values():
                for cond in drug_info.get("contraindications", []):
                    cond_low = cond.lower().strip()
                    if cond_low and cond_low not in self.entity_categories:
                        self.entity_categories[cond_low] = (cond, "disease")

        self.treats_relations = data.get("treats_relations", [])
        self.causes_relations = data.get("causes_relations", [])
        self.negation_cues = data.get("negation_cues", [])

        # Mechanism-predicate vocabulary (loaded from relation_lexicon.json
        # if present; otherwise empty and the extractor falls back to just
        # treats/causes via the seed lexicon).
        self.mechanism_predicates: dict[str, dict] = {}  # canonical -> info
        # surface_form (lowercased) -> canonical predicate name
        self.surface_to_predicate: dict[str, str] = {}
        if RELATION_LEXICON_PATH.exists():
            with open(RELATION_LEXICON_PATH) as fh:
                rel_data = json.load(fh)
            self.mechanism_predicates = rel_data.get("predicates", {})
            for canonical, info in self.mechanism_predicates.items():
                for sf in info.get("surface_forms", []):
                    self.surface_to_predicate[sf.lower()] = canonical

        # Pre-compile regexes ranked by length so longer phrases match first
        self._build_regexes()

    def _build_regexes(self):
        # Concept regex: longest-first to handle "type 2 diabetes mellitus"
        # before "diabetes mellitus".
        all_terms = list(self.entity_categories.keys()) + list(self.aliases.keys())
        all_terms.sort(key=len, reverse=True)
        # Word-boundary match, case-insensitive
        pattern = r"\b(" + "|".join(re.escape(t) for t in all_terms) + r")\b"
        self.concept_re = re.compile(pattern, re.IGNORECASE)

        # Relation phrases
        treats_pat = "|".join(re.escape(p) for p in
                              sorted(self.treats_relations, key=len, reverse=True))
        causes_pat = "|".join(re.escape(p) for p in
                              sorted(self.causes_relations, key=len, reverse=True))
        self.treats_re = re.compile(r"\b(" + treats_pat + r")\b", re.IGNORECASE)
        self.causes_re = re.compile(r"\b(" + causes_pat + r")\b", re.IGNORECASE)

        # Mechanism predicates: one big regex matching any surface form.
        # Longest-first so "is induced by" beats "induces", etc.
        if self.surface_to_predicate:
            mech_forms = sorted(self.surface_to_predicate.keys(),
                                key=len, reverse=True)
            mech_pat = "|".join(re.escape(p) for p in mech_forms)
            self.mechanism_re = re.compile(r"\b(" + mech_pat + r")\b",
                                           re.IGNORECASE)
        else:
            self.mechanism_re = None

        # Negation cues: word boundaries on left only since "non-" has hyphen
        neg_pat = "|".join(re.escape(c) for c in
                           sorted(self.negation_cues, key=len, reverse=True))
        self.negation_re = re.compile(r"\b(" + neg_pat + r")", re.IGNORECASE)

        # Number with optional unit
        # e.g. "potassium 5.8 mEq/L", "glucose of 250 mg/dL", "Na 130"
        self.number_unit_re = re.compile(
            r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:/[a-zA-Z]+)?(?:/[a-zA-Z]+)?|%)?",
            re.IGNORECASE
        )

    def lookup(self, surface: str) -> Optional[tuple[str, str]]:
        """Returns (canonical, category) for a surface form, or None."""
        s = surface.lower()
        if s in self.entity_categories:
            return self.entity_categories[s]
        if s in self.aliases:
            return self.aliases[s]
        return None


# =====================================================
# Negation
# =====================================================

def is_negated(text: str, span: tuple[int, int], lexicon: Lexicon,
               window: int = 30) -> bool:
    """Negex-lite: check if a negation cue precedes the span within a
    window of characters in the same sentence-ish unit."""
    start = max(0, span[0] - window)
    pre_text = text[start:span[0]]
    # Limit to the current clause: stop at sentence punctuation
    last_break = max(pre_text.rfind("."), pre_text.rfind(";"),
                     pre_text.rfind("!"), pre_text.rfind("?"))
    if last_break >= 0:
        pre_text = pre_text[last_break + 1:]
    return bool(lexicon.negation_re.search(pre_text))


# =====================================================
# Extractors
# =====================================================

def extract_entities(text: str, lexicon: Lexicon) -> list[Entity]:
    entities = []
    for m in lexicon.concept_re.finditer(text):
        surface = m.group(1)
        result = lexicon.lookup(surface)
        if result is None:
            continue
        canonical, category = result
        ent = Entity(
            surface=surface, canonical=canonical, category=category,
            span=(m.start(), m.end()),
        )
        ent.negated = is_negated(text, ent.span, lexicon)
        entities.append(ent)
    # Drop nested duplicates (longer matches subsume shorter ones at same start)
    entities = _dedupe_overlaps(entities)
    return entities


def _dedupe_overlaps(entities: list[Entity]) -> list[Entity]:
    """Keep the longest match for any overlapping span."""
    if not entities:
        return []
    sorted_ents = sorted(entities, key=lambda e: (e.span[0], -(e.span[1] - e.span[0])))
    kept = [sorted_ents[0]]
    for e in sorted_ents[1:]:
        if e.span[0] < kept[-1].span[1]:
            # overlap; keep the longer one
            if (e.span[1] - e.span[0]) > (kept[-1].span[1] - kept[-1].span[0]):
                kept[-1] = e
        else:
            kept.append(e)
    return kept


def extract_relations(text: str, entities: list[Entity],
                      lexicon: Lexicon) -> list[Relation]:
    """Extract (entity, treats|causes, entity) triples by looking at
    subject-predicate-object patterns in the window between IMMEDIATELY
    adjacent entities. Skipping over an intervening entity would
    cross-contaminate relations across sentences."""
    relations = []
    if len(entities) < 2:
        return relations

    ents = sorted(entities, key=lambda e: e.span[0])

    for i in range(len(ents) - 1):
        subj = ents[i]
        obj = ents[i + 1]
        if obj.span[0] <= subj.span[1]:
            continue

        window_start = subj.span[1]
        window_end = obj.span[0]
        if window_end - window_start > 60:  # too far apart
            continue
        window_text = text[window_start:window_end]
        # If the window crosses a sentence boundary, skip
        if any(ch in window_text for ch in (".", "?", "!", ";")):
            continue

        treats_m = lexicon.treats_re.search(window_text)
        causes_m = lexicon.causes_re.search(window_text)
        mech_m = (lexicon.mechanism_re.search(window_text)
                  if lexicon.mechanism_re else None)

        if treats_m:
            negated = is_negated(text, (window_start + treats_m.start(),
                                        window_start + treats_m.end()),
                                 lexicon)
            relations.append(Relation(
                subject=subj, predicate="treats", object=obj,
                span=(window_start + treats_m.start(),
                      window_start + treats_m.end()),
                negated=negated, category="therapeutic",
            ))
        elif causes_m:
            negated = is_negated(text, (window_start + causes_m.start(),
                                        window_start + causes_m.end()),
                                 lexicon)
            relations.append(Relation(
                subject=subj, predicate="causes", object=obj,
                span=(window_start + causes_m.start(),
                      window_start + causes_m.end()),
                negated=negated, category="causal",
            ))
        elif mech_m:
            surface = mech_m.group(1).lower()
            canonical = lexicon.surface_to_predicate.get(surface, "unknown")
            cat = (lexicon.mechanism_predicates.get(canonical, {})
                            .get("category", "unknown"))
            negated = is_negated(text, (window_start + mech_m.start(),
                                        window_start + mech_m.end()),
                                 lexicon)
            relations.append(Relation(
                subject=subj, predicate=canonical, object=obj,
                span=(window_start + mech_m.start(),
                      window_start + mech_m.end()),
                negated=negated, category=cat,
            ))
    return relations


def extract_numerics(text: str, entities: list[Entity],
                     lexicon: Lexicon) -> list[NumericClaim]:
    """Find lab values like 'potassium 5.8 mEq/L' or 'glucose of 250'."""
    numerics = []

    # We use the physiologic_ranges file to know which entity names are labs
    ranges_path = Path(__file__).parent.parent.parent / "ontology" / "data" / "physiologic_ranges.json"
    if ranges_path.exists():
        with open(ranges_path) as fh:
            ranges_data = json.load(fh)
        lab_names = {}
        for canonical, info in ranges_data["labs"].items():
            lab_names[canonical] = canonical
            for alias in info.get("alias", []):
                lab_names[alias.lower()] = canonical
    else:
        lab_names = {}

    # Search for "<lab name> ... <number> <unit>" within ~30 chars
    for lab_surface, lab_canonical in lab_names.items():
        for m in re.finditer(rf"\b{re.escape(lab_surface)}\b", text, re.IGNORECASE):
            # Look ahead up to 30 chars
            window_start = m.end()
            window_end = min(len(text), window_start + 30)
            window = text[window_start:window_end]
            num_m = lexicon.number_unit_re.search(window)
            if num_m is None:
                continue
            try:
                value = float(num_m.group(1))
            except (TypeError, ValueError):
                continue
            unit = num_m.group(2) if num_m.group(2) else None
            full_span = (m.start(), window_start + num_m.end())
            negated = is_negated(text, full_span, lexicon)
            numerics.append(NumericClaim(
                lab=lab_canonical, value=value, unit=unit,
                direction=None, span=full_span, negated=negated,
            ))

    # Also search for direction-only mentions: "elevated potassium", "low glucose"
    direction_pattern = re.compile(
        r"\b(elevated|high|raised|increased|decreased|low|reduced|normal)\s+"
        r"(\w+(?:\s+\w+)?)\b", re.IGNORECASE)
    for m in direction_pattern.finditer(text):
        direction = m.group(1).lower()
        if direction in ("elevated", "high", "raised", "increased"):
            direction = "high"
        elif direction in ("decreased", "low", "reduced"):
            direction = "low"
        else:
            direction = "normal"
        cand_lab = m.group(2).lower()
        if cand_lab in lab_names:
            full_span = (m.start(), m.end())
            negated = is_negated(text, full_span, lexicon)
            numerics.append(NumericClaim(
                lab=lab_names[cand_lab], value=float("nan"), unit=None,
                direction=direction, span=full_span, negated=negated,
            ))

    return numerics


# =====================================================
# Public API
# =====================================================

_LEXICON_SINGLETON: Optional[Lexicon] = None


def get_lexicon() -> Lexicon:
    global _LEXICON_SINGLETON
    if _LEXICON_SINGLETON is None:
        _LEXICON_SINGLETON = Lexicon()
    return _LEXICON_SINGLETON


def extract_claims(text: str) -> ExtractedClaims:
    """Run the full extraction pipeline on a single text."""
    lexicon = get_lexicon()
    entities = extract_entities(text, lexicon)
    relations = extract_relations(text, entities, lexicon)
    numerics = extract_numerics(text, entities, lexicon)
    return ExtractedClaims(text=text, entities=entities,
                           relations=relations, numerics=numerics)


# =====================================================
# Self-test
# =====================================================

if __name__ == "__main__":
    samples = [
        "Patient with type 2 diabetes mellitus on metformin and insulin.",
        "Metformin treats hypertension.",
        "No fever or cough on examination.",
        "Potassium 5.8 mEq/L; creatinine 2.1.",
        "Elevated troponin and chest pain suggest myocardial infarction.",
        "Hypoglycemia causes insulin in this case.",  # inverted relation
    ]
    for s in samples:
        print(f"\n>>> {s}")
        c = extract_claims(s)
        for e in c.entities:
            neg = " [NEG]" if e.negated else ""
            print(f"   ent: {e.canonical} ({e.category}){neg}")
        for r in c.relations:
            neg = " [NEG]" if r.negated else ""
            print(f"   rel: {r.subject.canonical} --{r.predicate}--> "
                  f"{r.object.canonical}{neg}")
        for n in c.numerics:
            unit = f" {n.unit}" if n.unit else ""
            direction = f" [{n.direction}]" if n.direction else ""
            value = "" if n.value != n.value else f" = {n.value}"  # NaN check
            print(f"   num: {n.lab}{value}{unit}{direction}")
