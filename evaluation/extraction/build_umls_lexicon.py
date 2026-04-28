"""
build_umls_lexicon.py

Expand the small curated medical_lexicon.json into a much larger
lexicon by mapping each seed term to its UMLS CUI(s) and pulling all
atoms (synonyms across source vocabularies, including MeSH, SNOMED,
RxNorm, ICD-10) for the matched concepts.

Output: medical_lexicon_umls.json, schema-compatible with the seed
lexicon. The downstream extractor (claim_extraction.py) loads the
UMLS-expanded file when present.

Usage:
    UMLS_API_KEY=... python evaluation/extraction/build_umls_lexicon.py \
        --workers 8

Caching: per-CUI atoms responses and per-term search responses are
cached in evaluation/extraction/.umls_cache/ so reruns and resumes
are cheap. Delete the cache directory to force fresh fetches.

Notes:
  - Modern UMLS REST (uts-ws.nlm.nih.gov/rest) accepts apiKey as a
    query parameter. No ticket-granting-ticket flow required.
  - We restrict atoms to ENG-language preferred names from broad
    sources (SNOMEDCT_US, RXNORM, MSH, NCI, MTH, ICD10CM) to keep the
    lexicon clean.
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
from typing import Optional

HERE = Path(__file__).parent
SEED_PATH = HERE / "medical_lexicon.json"
OUT_PATH = HERE / "medical_lexicon_umls.json"
CACHE_DIR = HERE / ".umls_cache"

# UMLS source vocabularies we accept atoms from. MTH is the metathesaurus
# preferred name. SNOMEDCT_US, RXNORM, MSH cover most clinical usage.
KEEP_SABS = {"MTH", "SNOMEDCT_US", "RXNORM", "MSH", "NCI", "ICD10CM",
             "MEDLINEPLUS", "MDR", "NDFRT"}

# Seed-category → UMLS semantic-type filter (for sanity-checking that the
# top search hit is the right kind of concept). We use this as a soft
# guard, not a hard filter, since search ranking is usually fine.
CATEGORY_SEMANTIC_TYPES = {
    "drugs": {"T121", "T200", "T103", "T195", "T197", "T123"},
    "diseases": {"T047", "T191", "T046", "T019", "T020"},
    "mental_disorders": {"T048", "T047"},
    "symptoms": {"T184", "T033"},
    "infections_and_conditions": {"T047", "T191", "T037", "T046"},
    "lab_findings": {"T034", "T201", "T184"},
    "anatomy": {"T023", "T029", "T024", "T030", "T022"},
    "procedures": {"T060", "T061", "T059", "T058"},
}

CATEGORIES_TO_EXPAND = list(CATEGORY_SEMANTIC_TYPES.keys())

API_BASE = "https://uts-ws.nlm.nih.gov/rest"


# =====================================================
# HTTP helpers
# =====================================================

def _http_get_json(url: str, retries: int = 3, backoff: float = 0.5) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cho-eval/0.1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"GET failed after {retries} retries: {url} ({last_err})")


def _cache_get(name: str) -> Optional[dict]:
    p = CACHE_DIR / f"{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _cache_put(name: str, value: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{name}.json"
    p.write_text(json.dumps(value))


def _safe_cache_key(s: str) -> str:
    # filename-safe, length-bounded
    return urllib.parse.quote(s, safe="")[:160]


# =====================================================
# UMLS calls
# =====================================================

def search_term(term: str, api_key: str) -> list[dict]:
    cache_name = "search__" + _safe_cache_key(term.lower())
    cached = _cache_get(cache_name)
    if cached is not None:
        return cached.get("results", [])
    url = f"{API_BASE}/search/current?" + urllib.parse.urlencode({
        "string": term, "apiKey": api_key, "pageSize": 5,
        "searchType": "exact",
    })
    data = _http_get_json(url)
    results = data.get("result", {}).get("results", []) or []
    # If exact search yielded nothing, fall back to "words" search.
    if not results or (len(results) == 1 and results[0].get("ui") in ("NONE", "", None)):
        url2 = f"{API_BASE}/search/current?" + urllib.parse.urlencode({
            "string": term, "apiKey": api_key, "pageSize": 5,
            "searchType": "words",
        })
        data2 = _http_get_json(url2)
        results = data2.get("result", {}).get("results", []) or []
    _cache_put(cache_name, {"results": results})
    return results


def get_concept_semtypes(cui: str, api_key: str) -> list[str]:
    """Return TUI codes (e.g. ['T121','T109']) for the concept."""
    cache_name = f"concept__{cui}"
    cached = _cache_get(cache_name)
    if cached is not None:
        return cached.get("tuis", [])
    url = f"{API_BASE}/content/current/CUI/{cui}?" + urllib.parse.urlencode({
        "apiKey": api_key,
    })
    try:
        data = _http_get_json(url)
    except RuntimeError:
        return []
    sts = [t.get("name", "") for t in
           (data.get("result", {}).get("semanticTypes", []) or [])]
    # we want TUI codes too; the search response embeds them differently.
    # Fall back: extract from URI like .../semantic-network/current/TUI/T047
    tui_codes = []
    for t in (data.get("result", {}).get("semanticTypes", []) or []):
        uri = t.get("uri", "")
        if "/TUI/" in uri:
            tui_codes.append(uri.rsplit("/TUI/", 1)[1])
    _cache_put(cache_name, {"semanticTypes": sts, "tuis": tui_codes})
    return tui_codes


def get_atoms(cui: str, api_key: str) -> list[str]:
    """Pull preferred ENG atoms for a CUI, restricted to KEEP_SABS."""
    cache_name = f"atoms__{cui}"
    cached = _cache_get(cache_name)
    if cached is not None:
        return cached.get("names", [])

    names: list[str] = []
    seen: set[str] = set()
    page = 1
    while True:
        url = f"{API_BASE}/content/current/CUI/{cui}/atoms?" + urllib.parse.urlencode({
            "apiKey": api_key, "language": "ENG",
            "ttys": "PT,SY,FN,IN,SCN",   # preferred terms, synonyms, fully-specified, ingredient, scientific name
            "pageSize": 100, "pageNumber": page,
        })
        try:
            data = _http_get_json(url)
        except RuntimeError:
            break
        atoms = data.get("result", []) or []
        if not atoms:
            break
        for a in atoms:
            sab = a.get("rootSource", "")
            if sab not in KEEP_SABS:
                continue
            name = (a.get("name") or "").strip()
            if not name:
                continue
            low = name.lower()
            if low in seen:
                continue
            seen.add(low)
            names.append(name)
        # UMLS atoms paginate; if the response is short, stop.
        if len(atoms) < 100:
            break
        page += 1
        if page > 6:           # cap at ~600 atoms per CUI
            break
    _cache_put(cache_name, {"names": names})
    return names


# =====================================================
# Pipeline
# =====================================================

def expand_category(category: str, seeds: list[str], api_key: str,
                    workers: int) -> list[str]:
    """Map seeds → CUIs → atoms; return a sorted unique list of names."""
    expected_tuis = CATEGORY_SEMANTIC_TYPES.get(category, set())

    # Step 1: search each seed for top CUI(s)
    seed_to_cuis: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(search_term, s, api_key): s for s in seeds}
        for f in as_completed(futures):
            s = futures[f]
            try:
                hits = f.result()
            except Exception as e:
                print(f"  [warn] search failed for {s!r}: {e}", file=sys.stderr)
                hits = []
            cuis = [h.get("ui") for h in hits if h.get("ui") not in (None, "", "NONE")]
            seed_to_cuis[s] = cuis[:3]   # keep top 3 per seed

    all_cuis: set[str] = set()
    for cuis in seed_to_cuis.values():
        all_cuis.update(cuis)

    # Step 2: for each CUI, fetch atoms (with optional semantic-type guard)
    expanded: set[str] = set(s.lower() for s in seeds)  # always include seeds
    cui_list = sorted(all_cuis)

    def fetch_one(cui: str) -> list[str]:
        if expected_tuis:
            tuis = set(get_concept_semtypes(cui, api_key))
            if tuis and not (tuis & expected_tuis):
                return []   # skip off-topic CUIs
        return get_atoms(cui, api_key)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, c): c for c in cui_list}
        for f in as_completed(futures):
            cui = futures[f]
            try:
                names = f.result()
            except Exception as e:
                print(f"  [warn] atoms failed for {cui}: {e}", file=sys.stderr)
                names = []
            for n in names:
                if 2 <= len(n) <= 80:        # reject tiny / runaway strings
                    expanded.add(n.lower())

    return sorted(expanded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Subset of categories to expand "
                             "(default: all expandable).")
    args = parser.parse_args()

    api_key = os.environ.get("UMLS_API_KEY")
    if not api_key:
        print("ERROR: UMLS_API_KEY env var not set.", file=sys.stderr)
        return 2

    seed_lex = json.loads(SEED_PATH.read_text())
    out_lex: dict = {
        "_metadata": {
            "description": "UMLS-expanded clinical lexicon for CHO claim "
                           "extraction. Built from medical_lexicon.json "
                           "seeds via UMLS REST API atom expansion.",
            "version": "umls-expanded-v1",
            "source_vocabs": sorted(KEEP_SABS),
        },
    }

    cats = args.categories or CATEGORIES_TO_EXPAND
    for cat in cats:
        seeds = seed_lex.get(cat, [])
        if not isinstance(seeds, list) or not seeds:
            print(f"[{cat}] no seeds, skipping")
            continue
        t0 = time.time()
        expanded = expand_category(cat, seeds, api_key, args.workers)
        dt = time.time() - t0
        print(f"[{cat}] {len(seeds)} seeds → {len(expanded)} expanded "
              f"({dt:.1f}s)")
        out_lex[cat] = expanded

    # Carry over non-expanded categories (relations, negation cues, aliases)
    for k, v in seed_lex.items():
        if k in out_lex:
            continue
        out_lex[k] = v

    Path(args.out).write_text(json.dumps(out_lex, indent=2))
    print(f"\nWrote expanded lexicon to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
