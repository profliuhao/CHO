"""
odi.py: Ontological Drift Index.

ODI(G) = (1/|P(G)|) * sum_{(c_s, c_t) in P(G)} d_H(c_s, c_t)

where P(G) is the set of source-output concept pairs and d_H(c_s, c_t)
is the normalized taxonomic distance between c_s and c_t in the UMLS
is-a hierarchy.

In this lightweight implementation we use the curated semantic-types
hierarchy (semantic_types.json) instead of the full UMLS graph. Drift
is measured as:

  d_H(c_s, c_t) = 0           if same semantic type
                = depth-diff / max_depth   if same branch but different depth
                = (depth_s + depth_t - 2*depth_lca) / max_depth   otherwise

The high-impact case the metric is designed to flag: simplifying
"elevated troponin" (LaboratoryFinding) into "myocardial infarction"
(Disease). These are in different branches of the hierarchy and the
normalized path length is large.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.claim_extraction import ExtractedClaims, get_lexicon


_SEMTYPES = None
_MAX_DEPTH = None


def _load_semtypes() -> dict:
    global _SEMTYPES, _MAX_DEPTH
    if _SEMTYPES is None:
        path = Path(__file__).parent.parent.parent / "ontology" / "data" / "semantic_types.json"
        with open(path) as fh:
            data = json.load(fh)
        _SEMTYPES = data
        _MAX_DEPTH = max(t["depth"] for t in data["semantic_types"].values())
    return _SEMTYPES


def _ancestors(semtype: str) -> list[str]:
    """Return semtype + all its ancestors up to the root, in order."""
    types = _load_semtypes()["semantic_types"]
    chain = [semtype]
    cur = semtype
    while True:
        node = types.get(cur)
        if node is None:
            break
        parent = node.get("parent")
        if parent is None:
            break
        chain.append(parent)
        cur = parent
    return chain


def lca_depth(t1: str, t2: str) -> int:
    """Lowest common ancestor depth in the semantic-type tree."""
    a1 = _ancestors(t1)
    a2_set = set(_ancestors(t2))
    types = _load_semtypes()["semantic_types"]
    for a in a1:
        if a in a2_set:
            return types[a]["depth"]
    return 0  # both rooted at Entity


def concept_to_semtype(concept_name: str) -> str | None:
    data = _load_semtypes()
    return data["concept_to_semtype"].get(concept_name.lower())


def hierarchical_distance(c1: str, c2: str) -> float:
    """Normalized taxonomic distance in [0, 1]. Returns NaN if either
    concept has no known semantic type."""
    t1 = concept_to_semtype(c1)
    t2 = concept_to_semtype(c2)
    if t1 is None or t2 is None:
        return float("nan")
    if t1 == t2:
        return 0.0
    types = _load_semtypes()["semantic_types"]
    d1 = types[t1]["depth"]
    d2 = types[t2]["depth"]
    lca_d = lca_depth(t1, t2)
    path_length = (d1 - lca_d) + (d2 - lca_d)
    return path_length / (2.0 * _MAX_DEPTH)


@dataclass
class ODIResult:
    score: float
    pair_count: int
    pair_distances: list  # [(c_s, c_t, distance), ...]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "pair_count": self.pair_count,
            "pair_distances": self.pair_distances,
        }


def align_concepts(source_claims: ExtractedClaims,
                   target_claims: ExtractedClaims) -> list[tuple[str, str]]:
    """Align source and target entities by greedy matching: source entities
    that don't appear verbatim in the target are paired with the
    closest target entity that shares a related semantic type or shares
    a partial string overlap. Each source entity contributes at most one
    pair."""
    source_canon = [e.canonical for e in source_claims.entities]
    target_canon = [e.canonical for e in target_claims.entities]
    target_set = set(target_canon)

    pairs = []
    used_targets = set()

    for sc in source_canon:
        if sc in target_set and sc not in used_targets:
            # Identical match: distance is zero, doesn't contribute drift
            pairs.append((sc, sc))
            used_targets.add(sc)
            continue
        # Otherwise pair with the most semantically related target
        best_target = None
        best_dist = float("inf")
        sc_type = concept_to_semtype(sc)
        for tc in target_canon:
            if tc in used_targets:
                continue
            if tc == sc:
                continue
            d = hierarchical_distance(sc, tc)
            if d != d:  # NaN
                continue
            if d < best_dist:
                best_dist = d
                best_target = tc
        if best_target is not None:
            pairs.append((sc, best_target))
            used_targets.add(best_target)

    return pairs


def compute_odi(source_claims: ExtractedClaims,
                target_claims: ExtractedClaims) -> ODIResult:
    """Compute ODI for a (source, target) text pair."""
    pairs = align_concepts(source_claims, target_claims)
    if not pairs:
        return ODIResult(score=float("nan"), pair_count=0, pair_distances=[])
    distances = []
    for s, t in pairs:
        d = hierarchical_distance(s, t)
        if d == d:  # not NaN
            distances.append((s, t, d))
    if not distances:
        return ODIResult(score=float("nan"), pair_count=0, pair_distances=[])
    score = sum(d for _, _, d in distances) / len(distances)
    return ODIResult(
        score=score, pair_count=len(distances),
        pair_distances=distances,
    )


if __name__ == "__main__":
    from extraction.claim_extraction import extract_claims
    examples = [
        ("identical",
         "Patient has elevated troponin.",
         "Patient has elevated troponin."),
        ("symptom -> disease drift",
         "Patient has elevated troponin and chest pain.",
         "Patient has myocardial infarction."),
        ("drug -> disease drift",
         "Patient on metformin and insulin.",
         "Patient has type 2 diabetes mellitus and hyperglycemia."),
        ("same semantic type",
         "Patient has hypertension.",
         "Patient has heart failure."),
    ]
    for label, src, tgt in examples:
        sc = extract_claims(src)
        tc = extract_claims(tgt)
        r = compute_odi(sc, tc)
        print(f"{label:30s} ODI={r.score:.3f}  pairs={r.pair_count}")
        for s, t, d in r.pair_distances:
            print(f"    {s!r:30s} -> {t!r:30s} d={d:.3f}")
