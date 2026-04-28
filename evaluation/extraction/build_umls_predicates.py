"""
build_umls_predicates.py

Build a mechanism-predicate lexicon for the CHO claim extractor.

The original extractor only recognized two relation predicates: 'treats' and
'causes'. MedHallu (and biomedical text in general) hallucinations are
expressed in much richer predicate vocabulary — regulates, activates,
inhibits, induces, expresses, phosphorylates, etc. Without these, the
relation extractor produces zero relations for mechanism-style claims and
the RIS metric has nothing to score.

This script writes evaluation/extraction/relation_lexicon.json with:
  - canonical predicates (~50)
  - surface forms (lexical realizations) per predicate
  - the inverse predicate (for A5_InvertedRelation detection)
  - the predicate category (therapeutic / causal / mechanistic_positive /
    mechanistic_negative / regulatory / structural)
  - optional cross-reference to UMLS RELA values (enrichment, not required)

The curated vocabulary below is the primary content. UMLS enrichment via the
optional --enrich-umls flag adds synonymous RELA codes by sampling relations
attached to seed CUIs.

Usage:
    # Curated only:
    python evaluation/extraction/build_umls_predicates.py

    # With UMLS enrichment (validates predicates against real RELA values):
    UMLS_API_KEY=... python evaluation/extraction/build_umls_predicates.py \\
        --enrich-umls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
OUT_PATH = HERE / "relation_lexicon.json"
CACHE_DIR = HERE / ".umls_cache"

API_BASE = "https://uts-ws.nlm.nih.gov/rest"


# ---------------------------------------------------------------------------
# Curated mechanism predicate vocabulary
# ---------------------------------------------------------------------------
# Each entry:
#   surface_forms : phrases the regex extractor will match. Active voice only;
#                   the extractor handles the inverse direction by matching the
#                   inverse predicate's surface forms.
#   inverse       : canonical name of the inverse predicate (or None).
#   category      : grouping used by the OHR/RIS metrics for axiom selection.
#
# Categories:
#   therapeutic           - drug → condition (treats, indicated_for, ...)
#   causal                - X → Y (causes, leads_to, results_in, ...)
#   mechanistic_positive  - molecular activation (activates, induces, ...)
#   mechanistic_negative  - molecular inhibition (inhibits, blocks, ...)
#   regulatory            - bidirectional control (regulates, modulates, ...)
#   structural            - composition / location (binds_to, expressed_in, ...)
#   diagnostic            - tests and signs (diagnoses, indicates, ...)

CURATED: dict[str, dict] = {

    # -- Therapeutic / clinical --------------------------------------------
    "treats": {
        "surface_forms": [
            "treats", "is used to treat", "is indicated for",
            "is prescribed for", "is given for", "is the treatment for",
            "manages", "controls", "alleviates",
        ],
        "inverse": "treated_by",
        "category": "therapeutic",
    },
    "treated_by": {
        "surface_forms": [
            "is treated with", "is treated by", "is managed with",
            "responds to treatment with", "improves with",
        ],
        "inverse": "treats",
        "category": "therapeutic",
    },
    "contraindicated_in": {
        "surface_forms": [
            "is contraindicated in", "should not be given in",
            "must be avoided in", "is not recommended in",
        ],
        "inverse": None,
        "category": "therapeutic",
    },
    "indicated_for": {
        "surface_forms": [
            "is indicated for", "has indication for",
            "is approved for", "is licensed for",
        ],
        "inverse": None,
        "category": "therapeutic",
    },

    # -- Causal -------------------------------------------------------------
    "causes": {
        "surface_forms": [
            "causes", "leads to", "results in", "produces",
            "gives rise to", "brings about", "triggers",
        ],
        "inverse": "caused_by",
        "category": "causal",
    },
    "caused_by": {
        "surface_forms": [
            "is caused by", "is due to", "results from",
            "is the result of", "is secondary to",
        ],
        "inverse": "causes",
        "category": "causal",
    },
    "prevents": {
        "surface_forms": [
            "prevents", "protects against", "reduces the risk of",
            "decreases the incidence of",
        ],
        "inverse": None,
        "category": "causal",
    },
    "predisposes_to": {
        "surface_forms": [
            "predisposes to", "increases the risk of",
            "is a risk factor for",
        ],
        "inverse": None,
        "category": "causal",
    },

    # -- Mechanistic positive ----------------------------------------------
    "activates": {
        "surface_forms": [
            "activates", "stimulates", "triggers",
            "brings about activation of",
        ],
        "inverse": "activated_by",
        "category": "mechanistic_positive",
    },
    "activated_by": {
        "surface_forms": [
            "is activated by", "is stimulated by", "is triggered by",
        ],
        "inverse": "activates",
        "category": "mechanistic_positive",
    },
    "induces": {
        "surface_forms": [
            "induces", "drives", "promotes the formation of",
            "leads to the induction of",
        ],
        "inverse": "induced_by",
        "category": "mechanistic_positive",
    },
    "induced_by": {
        "surface_forms": [
            "is induced by", "is driven by",
        ],
        "inverse": "induces",
        "category": "mechanistic_positive",
    },
    "upregulates": {
        "surface_forms": [
            "upregulates", "increases expression of",
            "elevates levels of", "enhances expression of",
        ],
        "inverse": "upregulated_by",
        "category": "mechanistic_positive",
    },
    "upregulated_by": {
        "surface_forms": ["is upregulated by", "is elevated by"],
        "inverse": "upregulates",
        "category": "mechanistic_positive",
    },
    "agonizes": {
        "surface_forms": [
            "is an agonist of", "agonizes", "acts as an agonist at",
        ],
        "inverse": None,
        "category": "mechanistic_positive",
    },
    "stimulates": {
        "surface_forms": [
            "stimulates", "stimulates the release of",
            "promotes the release of",
        ],
        "inverse": None,
        "category": "mechanistic_positive",
    },

    # -- Mechanistic negative ----------------------------------------------
    "inhibits": {
        "surface_forms": [
            "inhibits", "blocks", "suppresses",
            "interferes with", "prevents the action of",
        ],
        "inverse": "inhibited_by",
        "category": "mechanistic_negative",
    },
    "inhibited_by": {
        "surface_forms": [
            "is inhibited by", "is blocked by", "is suppressed by",
        ],
        "inverse": "inhibits",
        "category": "mechanistic_negative",
    },
    "antagonizes": {
        "surface_forms": [
            "is an antagonist of", "antagonizes",
            "acts as an antagonist at",
        ],
        "inverse": None,
        "category": "mechanistic_negative",
    },
    "downregulates": {
        "surface_forms": [
            "downregulates", "decreases expression of",
            "reduces levels of", "suppresses expression of",
        ],
        "inverse": "downregulated_by",
        "category": "mechanistic_negative",
    },
    "downregulated_by": {
        "surface_forms": ["is downregulated by", "is suppressed by"],
        "inverse": "downregulates",
        "category": "mechanistic_negative",
    },
    "degrades": {
        "surface_forms": [
            "degrades", "breaks down", "catabolizes",
        ],
        "inverse": "degraded_by",
        "category": "mechanistic_negative",
    },
    "degraded_by": {
        "surface_forms": ["is degraded by", "is broken down by"],
        "inverse": "degrades",
        "category": "mechanistic_negative",
    },

    # -- Regulatory (bidirectional) ----------------------------------------
    "regulates": {
        "surface_forms": [
            "regulates", "controls", "governs",
        ],
        "inverse": "regulated_by",
        "category": "regulatory",
    },
    "regulated_by": {
        "surface_forms": [
            "is regulated by", "is controlled by",
        ],
        "inverse": "regulates",
        "category": "regulatory",
    },
    "modulates": {
        "surface_forms": [
            "modulates", "modifies the activity of",
            "fine-tunes",
        ],
        "inverse": "modulated_by",
        "category": "regulatory",
    },
    "modulated_by": {
        "surface_forms": ["is modulated by"],
        "inverse": "modulates",
        "category": "regulatory",
    },
    "mediates": {
        "surface_forms": [
            "mediates", "is responsible for",
        ],
        "inverse": "mediated_by",
        "category": "regulatory",
    },
    "mediated_by": {
        "surface_forms": ["is mediated by", "occurs through"],
        "inverse": "mediates",
        "category": "regulatory",
    },
    "phosphorylates": {
        "surface_forms": ["phosphorylates", "adds a phosphate to"],
        "inverse": "phosphorylated_by",
        "category": "regulatory",
    },
    "phosphorylated_by": {
        "surface_forms": ["is phosphorylated by"],
        "inverse": "phosphorylates",
        "category": "regulatory",
    },
    "catalyzes": {
        "surface_forms": [
            "catalyzes", "catalyses",
            "facilitates the conversion of",
        ],
        "inverse": "catalyzed_by",
        "category": "regulatory",
    },
    "catalyzed_by": {
        "surface_forms": ["is catalyzed by", "is catalysed by"],
        "inverse": "catalyzes",
        "category": "regulatory",
    },

    # -- Structural / compositional ----------------------------------------
    "binds_to": {
        "surface_forms": [
            "binds to", "binds", "is bound to",
            "associates with", "interacts with",
        ],
        "inverse": "bound_by",
        "category": "structural",
    },
    "bound_by": {
        "surface_forms": ["is bound by"],
        "inverse": "binds_to",
        "category": "structural",
    },
    "expressed_in": {
        "surface_forms": [
            "is expressed in", "is found in",
            "is present in",
        ],
        "inverse": "expresses",
        "category": "structural",
    },
    "expresses": {
        "surface_forms": [
            "expresses", "produces",
        ],
        "inverse": "expressed_in",
        "category": "structural",
    },
    "located_in": {
        "surface_forms": [
            "is located in", "is found in", "resides in",
        ],
        "inverse": "contains",
        "category": "structural",
    },
    "contains": {
        "surface_forms": ["contains", "comprises", "includes"],
        "inverse": "located_in",
        "category": "structural",
    },
    "secretes": {
        "surface_forms": ["secretes", "releases"],
        "inverse": "secreted_by",
        "category": "structural",
    },
    "secreted_by": {
        "surface_forms": ["is secreted by", "is released by"],
        "inverse": "secretes",
        "category": "structural",
    },
    "transports": {
        "surface_forms": ["transports", "carries"],
        "inverse": "transported_by",
        "category": "structural",
    },
    "transported_by": {
        "surface_forms": ["is transported by", "is carried by"],
        "inverse": "transports",
        "category": "structural",
    },
    "synthesizes": {
        "surface_forms": ["synthesizes", "synthesises", "produces"],
        "inverse": "synthesized_by",
        "category": "structural",
    },
    "synthesized_by": {
        "surface_forms": ["is synthesized by", "is synthesised by"],
        "inverse": "synthesizes",
        "category": "structural",
    },

    # -- Diagnostic --------------------------------------------------------
    "diagnoses": {
        "surface_forms": [
            "diagnoses", "is used to diagnose",
            "establishes the diagnosis of",
        ],
        "inverse": "diagnosed_by",
        "category": "diagnostic",
    },
    "diagnosed_by": {
        "surface_forms": [
            "is diagnosed by", "is identified by",
            "is established by",
        ],
        "inverse": "diagnoses",
        "category": "diagnostic",
    },
    "indicates": {
        "surface_forms": [
            "indicates", "suggests", "is indicative of",
            "is suggestive of",
        ],
        "inverse": None,
        "category": "diagnostic",
    },

    # -- Apoptosis / cell death (common in MedHallu Mech+Pathway) ----------
    "induces_apoptosis_in": {
        "surface_forms": [
            "induces apoptosis in", "triggers apoptosis in",
            "causes apoptosis in", "induces cell death in",
        ],
        "inverse": None,
        "category": "mechanistic_positive",
    },
    "required_for": {
        "surface_forms": [
            "is required for", "is necessary for",
            "is essential for",
        ],
        "inverse": "requires",
        "category": "regulatory",
    },
    "requires": {
        "surface_forms": ["requires", "needs", "depends on"],
        "inverse": "required_for",
        "category": "regulatory",
    },
}


# ---------------------------------------------------------------------------
# Optional UMLS enrichment
# ---------------------------------------------------------------------------
# Each canonical predicate has a list of UMLS RELA codes we consider
# semantically equivalent. Used to validate the curated vocabulary against
# real biomedical relation types and (optionally) for downstream tools that
# want CUI-grounded reasoning.

UMLS_RELA_HINTS = {
    "treats": ["may_treat", "treats", "therapeutic_class_of"],
    "treated_by": ["may_be_treated_by"],
    "contraindicated_in": ["contraindicated_with_disease",
                           "ci_with"],
    "causes": ["cause_of", "may_be_caused_by", "induces"],
    "caused_by": ["caused_by", "due_to"],
    "prevents": ["may_prevent", "prevents"],
    "activates": ["activates", "induces"],
    "inhibits": ["inhibits", "negatively_regulates"],
    "regulates": ["regulates"],
    "modulates": ["modulates"],
    "binds_to": ["binds_to", "interacts_with"],
    "expressed_in": ["expressed_in"],
    "secretes": ["produces"],
    "phosphorylates": ["phosphorylates"],
    "induces_apoptosis_in": ["induces_apoptosis_in", "induces"],
    "diagnoses": ["may_diagnose"],
    "agonizes": ["agonist_of"],
    "antagonizes": ["antagonist_of"],
    "upregulates": ["positively_regulates", "upregulates"],
    "downregulates": ["negatively_regulates", "downregulates"],
}


# ---------------------------------------------------------------------------
# Optional UMLS RELA validation (light-weight; samples relations from a
# handful of well-known CUIs to see which RELA values actually appear)
# ---------------------------------------------------------------------------

PROBE_CUIS = [
    "C0025598",   # metformin (drug)
    "C0011849",   # diabetes mellitus (disease)
    "C0020538",   # hypertension
    "C0021641",   # insulin
    "C0007634",   # cell
    "C0026237",   # mitochondrion
]


def _http_get_json(url: str, retries: int = 3, backoff: float = 0.5) -> dict:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cho-eval/0.1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"GET failed: {url} ({last})")


def _cache_get(name: str):
    p = CACHE_DIR / f"{name}.json"
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: return None
    return None


def _cache_put(name: str, value: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{name}.json").write_text(json.dumps(value))


def sample_relas(api_key: str) -> dict[str, int]:
    """Return RELA value counts observed across PROBE_CUIS."""
    counts: dict[str, int] = {}
    for cui in PROBE_CUIS:
        cache_name = f"relations__{cui}"
        cached = _cache_get(cache_name)
        if cached is not None:
            rels = cached.get("relations", [])
        else:
            page = 1
            rels = []
            while True:
                url = f"{API_BASE}/content/current/CUI/{cui}/relations?" + urllib.parse.urlencode({
                    "apiKey": api_key, "pageSize": 100, "pageNumber": page,
                })
                try:
                    data = _http_get_json(url)
                except RuntimeError:
                    break
                page_rels = data.get("result", []) or []
                if not page_rels: break
                rels.extend(page_rels)
                if len(page_rels) < 100: break
                page += 1
                if page > 4: break
            _cache_put(cache_name, {"relations": rels})
        for r in rels:
            rela = (r.get("additionalRelationLabel") or "").strip()
            if rela:
                counts[rela] = counts.get(rela, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--enrich-umls", action="store_true",
                        help="Validate curated predicates against observed "
                             "UMLS RELA values (requires UMLS_API_KEY).")
    args = parser.parse_args()

    out = {
        "_metadata": {
            "description": "Mechanism-predicate lexicon for CHO claim "
                           "extraction. Curated canonical predicates + "
                           "surface forms + inverse mappings + optional "
                           "UMLS RELA cross-references.",
            "version": "v1",
            "n_predicates": len(CURATED),
        },
        "predicates": {},
    }

    observed_relas: dict[str, int] = {}
    if args.enrich_umls:
        api_key = os.environ.get("UMLS_API_KEY")
        if not api_key:
            print("ERROR: --enrich-umls requires UMLS_API_KEY env var.",
                  file=sys.stderr)
            return 2
        print("Sampling UMLS RELA values from probe CUIs...")
        observed_relas = sample_relas(api_key)
        print(f"  observed {len(observed_relas)} distinct RELA values "
              f"across {len(PROBE_CUIS)} probe concepts.")

    for canonical, info in CURATED.items():
        rela_hints = UMLS_RELA_HINTS.get(canonical, [])
        observed_match = sorted(set(rela_hints) & set(observed_relas.keys())) \
            if observed_relas else []
        out["predicates"][canonical] = {
            "surface_forms": info["surface_forms"],
            "inverse": info["inverse"],
            "category": info["category"],
            "umls_rela_hints": rela_hints,
            "umls_rela_observed": observed_match,
        }

    # Categories index for downstream metric code
    cats: dict[str, list[str]] = {}
    for name, info in CURATED.items():
        cats.setdefault(info["category"], []).append(name)
    out["categories"] = {k: sorted(v) for k, v in cats.items()}

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(CURATED)} predicates to {args.out}")
    print(f"  categories: {', '.join(sorted(cats))}")
    if observed_relas:
        # Report which UMLS RELAs we DIDN'T cover (potential additions)
        covered = set()
        for v in UMLS_RELA_HINTS.values():
            covered.update(v)
        uncovered = sorted([(r, c) for r, c in observed_relas.items()
                            if r not in covered], key=lambda kv: -kv[1])
        if uncovered:
            print("\nTop UMLS RELA values observed but NOT in curated vocab "
                  "(consider adding):")
            for rela, count in uncovered[:15]:
                print(f"  {count:>4}  {rela}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
