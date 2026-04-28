# CHO Evaluation Implementation Plan

Pivoting from "rule-based AUROC on MedHallu/Med-HALT as the headline result" to a
three-mode evaluation paradigm. The current Mode 1 pipeline becomes the deterministic
baseline; Mode 2 (reasoner-based classification) and Mode 3 (LLM auditor with CHO tool
access) become the headline contributions.

## Why we're pivoting

The original pipeline evaluates CHO as a hardcoded rulebook: load `cho.owl` once,
extract axioms into Python data structures, score generated text. The OWL ontology
plays no live role — it could be replaced by a Python dict without changing AUROC. A
hostile reviewer would notice this in thirty seconds. We're restructuring the
evaluation so the ontology pulls weight in three independent ways.

## The three modes

**Mode 1 — Hardcoded rulebook.** What we already have. Underbuilt as of last run
(chance AUROC on MedHallu; bottleneck is relation-predicate vocabulary and
axiom-rule density). Phase 1 finishes this honestly so it's a credible baseline,
not a strawman.

**Mode 2 — Live reasoner backend.** Translate generated text into OWL ABox individuals
and let HermiT derive the CHO class assignment via the formal axioms. The most
defensible "the ontology actually does work" demonstration. Headline metric:
classification precision/recall per CHO axiom class against an expert-curated gold set.

**Mode 3 — LLM-queryable infrastructure.** Wrap CHO as an MCP server exposing ~6 tools.
Run a tool-augmented vs baseline auditor experiment with four arms (no-tools,
generic-knowledge-tools, CHO-tools, CHO-ablated). Headline metric: classification
accuracy delta showing that CHO specifically (not just structured knowledge in general)
improves an LLM auditor's ability to evaluate other LLMs.

## Decisions locked in

- Mode 3 design: 4 arms — no-tools, generic-knowledge, CHO-tools, CHO-ablated
- Headline benchmark: MedHallu `pqa_artificial` filtered to Mechanism+Pathway and
  Methodological-Fabrication categories (~1,142 source rows → ~2,284 examples)
- Appendix replication: MedHallu `pqa_labeled` all-categories (1,000 source rows)
- Auditor 1: Qwen 3.5 122B served locally at `localhost:29999` (OpenAI-compatible)
- Auditor 2: TBD — placeholder; user to provide endpoint info before Phase 3 runs

## Open inputs needed

- Endpoint info for the second auditor model (Phase 3, before full experiment runs)
- ~30 minutes of spot-checking on ~50 of the Mode 2 draft gold labels (end of Phase 2)

## Phase plan

### Phase 0 — Preflight (~10 min)
- P0.1 Verify local LLM endpoint reachable and OpenAI-compatible
- P0.2 Verify Qwen 3.5 122B function-calling support on this serving stack
- P0.3 Install owlready2 + verify `cho.owl` loads and HermiT runs

### Phase 1 — Finish Mode 1 honestly (~1 day)
- P1.1 Build UMLS REL/RELA-derived mechanism predicate vocabulary
  → `evaluation/extraction/build_umls_predicates.py`
  → `evaluation/extraction/relation_lexicon.json`
- P1.2 Mine drug contraindications into A1_ForbiddenRelation axiom pairs
  → `ontology/build_contraindications.py`
  → contraindication triples written into ontology data
- P1.3 Wire expanded extractor + ontology, rerun MedHallu, write
  `evaluation/results/mode1_expanded/` with before/after comparison

### Phase 2 — Mode 2 reasoner-based evaluation (~2 days)
- P2.1 Build ABox translator (extracted-claims → OWL individuals)
  → `evaluation/mode2/abox_translator.py`
- P2.2 Curate ~150 in-clinical-scope MedHallu examples + draft gold labels
  → `evaluation/mode2/gold_labels.json`
- P2.3 Run HermiT-based classification, compare to gold, produce confusion matrix
  → `evaluation/mode2/run_reasoner.py`
  → `evaluation/results/mode2/`

### Phase 3 — Mode 3 MCP server + auditor experiment (~3-5 days)
- P3.1 Build CHO MCP server with 6 tools (lookup_drug_indications,
  check_treats_relation, get_lab_plausibility, classify_error_via_reasoner,
  lookup_cho_class, get_harm_tier)
  → `mcp_server/cho_tools.py`, `mcp_server/server.py`
- P3.2 Build auditor harness supporting 4 arms
  → `evaluation/mode3/auditor.py`
- P3.3 Pilot run: n=50, single auditor, 4 arms (200 LLM calls)
  → go/no-go decision before full run
- P3.4 Full experiment: n≈500, both auditors (when 2nd available), 4 arms
  → `evaluation/mode3/run_experiment.py`
  → `evaluation/results/mode3/`

## Reproducibility / artifact structure

Final paper artifact will publish:
- `cho.owl` + SWRL rules + reasoner report (the ontology itself)
- `mcp_server/` (the live infrastructure that exposes CHO to LLMs)
- `evaluation/` (all three modes' code + results)
- `requirements.txt` updated with: owlready2, openai (for LLM client), java runtime
  for HermiT, `mcp[cli]` (MCP SDK)

Mode 1 is the deterministic baseline. Mode 2 is the formal-validity demonstration.
Mode 3 is the headline contribution. The MCP server is the live artifact reviewers
can run themselves.

## Phase 3 runbook (user runs locally)

The Mode 3 experiment requires the LLM endpoint, which is on the user's machine.
The auditor harness and experiment runner are code-complete in the sandbox; the
user runs the actual LLM-call sweep locally.

```bash
# 1. Install dependencies (one-time)
pip install owlready2 openai datasets "mcp[cli]"

# 2. Pilot run (n=50, four arms, ~200 LLM calls; ~5-15 min on Qwen 122B)
OPENAI_BASE_URL=http://localhost:29999/v1 \
OPENAI_API_KEY=$LOCAL_LLM_API_KEY \
python evaluation/mode3/run_experiment.py \
    --auditor-model qwen3.5-122b \
    --n 50 --verbose

# 3. Inspect pilot results
cat evaluation/results/mode3/qwen3.5-122b/report.md

# 4. If pilot looks good, full run (n=500, four arms = 2000 LLM calls)
OPENAI_BASE_URL=http://localhost:29999/v1 \
OPENAI_API_KEY=$LOCAL_LLM_API_KEY \
python evaluation/mode3/run_experiment.py \
    --auditor-model qwen3.5-122b \
    --full --n 500 --verbose

# 5. Add second auditor (when endpoint info supplied):
python evaluation/mode3/run_experiment.py \
    --auditor-model qwen3.5-122b \
    --auditor-2-model <weaker-model> \
    --auditor-2-base-url http://localhost:<port>/v1 \
    --auditor-2-api-key <key> \
    --full --n 500 --verbose
```

Pilot decision criteria (before scaling to n=500):
- `cho_tools` arm should beat `no_tools` by at least a few accuracy points
- `cho_tools` arm should also beat `generic_knowledge` (defends against
  "any structured knowledge would do")
- `cho_tools` should beat `cho_ablated` if the reasoner is doing real work;
  if not, the headline becomes "lookup tools help, reasoner doesn't add"
- Tool-invocation rate in CHO arms should be >50% (otherwise the model is
  ignoring its tools — likely a function-calling support issue with the
  serving stack)

If function calling fails on Qwen 3.5, fall back to JSON-mode prompting
(small change in `auditor.py` — mark me to wire that up if needed).

## Status snapshot

- Phase 0: complete (HermiT validated; LLM endpoint unreachable from sandbox,
  Phase 3 deferred to user's machine)
- Phase 1: complete; Mode 1 expanded but AUROC stays ~0.5 on MedHallu — the
  bottleneck is closed-vocabulary entity recognition. Documented finding.
- Phase 2: complete; reasoner achieves F1=0.96 on ForbiddenRelation,
  F1=1.00 on ThresholdViolation, 100% specificity on negative controls.
  See `evaluation/results/mode2/report.md`.
- Phase 3: code complete (MCP server + auditor harness + experiment runner +
  smoke-tested with mock); user runs the actual LLM sweep locally.
