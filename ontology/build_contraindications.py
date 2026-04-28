"""
build_contraindications.py

Mine drug→disease contraindications from UMLS (sourced from MED-RT, the
Medication Reference Terminology that succeeded NDFRT) and write a
structured contraindications.json that the OHR metric uses to fire
A1_ForbiddenRelation violations.

The seed CHO ontology only encodes a dozen forbidden (drug, treats,
condition) pairs. MedHallu hallucinations like "lisinopril treats
pregnancy" or "metformin treats acute kidney injury" should fire A1 but
don't because the ontology has nothing to say. Pulling MED-RT
contraindications gives us hundreds-to-thousands of clinically validated
forbidden pairs.

Inputs:
  - evaluation/extraction/medical_lexicon_umls.json  (drugs we care about)
  - .umls_cache/  (search cache from build_umls_lexicon.py — we reuse the
                   drug→CUI mappings if available)

Outputs:
  - ontology/data/contraindications.json
      { "_metadata": {...},
        "drugs": {
          "<drug_canonical>": {
            "cui": "C....",
            "contraindications": ["disease 1", "disease 2", ...]
          },
          ...
        }
      }

Usage:
    UMLS_API_KEY=... python ontology/build_contraindications.py \\
        --workers 8 --max-drugs 250
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

REPO_ROOT = Path(__file__).parent.parent
LEXICON_PATH = REPO_ROOT / "evaluation" / "extraction" / "medical_lexicon_umls.json"
CACHE_DIR = REPO_ROOT / "evaluation" / "extraction" / ".umls_cache"
OUT_PATH = REPO_ROOT / "ontology" / "data" / "contraindications.json"

API_BASE = "https://uts-ws.nlm.nih.gov/rest"

# Contraindication-flavored RELA values we accept (case-insensitive).
CONTRA_RELAS = {
    "contraindicated_with_disease",
    "ci_with",
    "may_be_contraindicated_for",
    "contraindicated_in",
}


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


def _safe_key(s: str) -> str:
    return urllib.parse.quote(s, safe="")[:160]


def get_cui_for_drug(drug: str, api_key: str) -> str | None:
    """Return the top-ranked CUI for a drug name, using build_umls_lexicon's
    cached search results when available."""
    cache_name = "search__" + _safe_key(drug.lower())
    cached = _cache_get(cache_name)
    if cached is not None:
        results = cached.get("results", [])
    else:
        url = f"{API_BASE}/search/current?" + urllib.parse.urlencode({
            "string": drug, "apiKey": api_key, "pageSize": 5,
            "searchType": "exact",
        })
        try:
            data = _http_get_json(url)
        except RuntimeError:
            return None
        results = data.get("result", {}).get("results", []) or []
        _cache_put(cache_name, {"results": results})
    for r in results:
        ui = r.get("ui")
        if ui and ui not in ("NONE", "", None):
            return ui
    return None


def get_contraindications(cui: str, api_key: str) -> list[str]:
    """Pull contraindicated_with_disease and related relations for a CUI.

    Returns a deduplicated list of disease names (lowercased)."""
    cache_name = f"contra__{cui}"
    cached = _cache_get(cache_name)
    if cached is not None:
        return cached.get("conditions", [])

    conditions: set[str] = set()
    page = 1
    while True:
        url = f"{API_BASE}/content/current/CUI/{cui}/relations?" + urllib.parse.urlencode({
            "apiKey": api_key, "pageSize": 100, "pageNumber": page,
        })
        try:
            data = _http_get_json(url)
        except RuntimeError:
            break
        rels = data.get("result", []) or []
        if not rels:
            break
        for rel in rels:
            rela = (rel.get("additionalRelationLabel") or "").strip().lower()
            if rela not in CONTRA_RELAS:
                continue
            name = (rel.get("relatedIdName") or "").strip().lower()
            if name and 2 <= len(name) <= 80:
                conditions.add(name)
        if len(rels) < 100:
            break
        page += 1
        if page > 10:
            break

    out = sorted(conditions)
    _cache_put(cache_name, {"conditions": out})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-drugs", type=int, default=None,
                        help="Cap the number of drugs processed (debug).")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    api_key = os.environ.get("UMLS_API_KEY")
    if not api_key:
        print("ERROR: UMLS_API_KEY env var not set.", file=sys.stderr)
        return 2

    if not LEXICON_PATH.exists():
        print(f"ERROR: {LEXICON_PATH} not found. Run build_umls_lexicon.py first.",
              file=sys.stderr)
        return 2

    lex = json.loads(LEXICON_PATH.read_text())
    drugs = lex.get("drugs", [])
    if args.max_drugs:
        drugs = drugs[: args.max_drugs]
    print(f"Processing {len(drugs)} drugs from UMLS-expanded lexicon...")

    # Step 1: drug → CUI (uses cache from build_umls_lexicon.py if available)
    drug_to_cui: dict[str, str] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(get_cui_for_drug, d, api_key): d for d in drugs}
        for f in as_completed(futures):
            d = futures[f]
            try:
                cui = f.result()
            except Exception:
                cui = None
            if cui:
                drug_to_cui[d] = cui
    print(f"  resolved {len(drug_to_cui)}/{len(drugs)} drugs to CUIs "
          f"({time.time()-t0:.1f}s)")

    # Step 2: per-CUI contraindication fetch
    out_drugs: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(get_contraindications, c, api_key): (d, c)
                   for d, c in drug_to_cui.items()}
        for f in as_completed(futures):
            drug, cui = futures[f]
            try:
                conditions = f.result()
            except Exception as e:
                print(f"  [warn] {drug} ({cui}): {e}", file=sys.stderr)
                conditions = []
            if conditions:
                out_drugs[drug] = {"cui": cui, "contraindications": conditions}

    n_total_pairs = sum(len(v["contraindications"]) for v in out_drugs.values())
    print(f"  fetched contraindications for {len(out_drugs)} drugs, "
          f"{n_total_pairs} total (drug, condition) pairs ({time.time()-t0:.1f}s)")

    out = {
        "_metadata": {
            "description": "Drug → contraindicated-condition pairs mined from "
                           "UMLS MED-RT via REST API. Powers A1_ForbiddenRelation "
                           "violations in OHR.",
            "version": "v1",
            "n_drugs": len(out_drugs),
            "n_pairs": n_total_pairs,
            "source_relas": sorted(CONTRA_RELAS),
        },
        "drugs": out_drugs,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
