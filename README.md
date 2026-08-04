# CHO: Clinical Hallucination Ontology

A formal evaluation framework for clinical hallucinations in large language models, grounded in an OWL 2 DL ontology with reasoner-validated axioms.

This repository contains the artifact for the a paper submission *CHO: An Ontology-Grounded Evaluation Framework for Clinical Hallucinations in Large Language Models*.

## What's here

The artifact has three layers, each evaluated in its own *mode*:

**Mode 1 — Hardcoded rulebook.** A regex-and-lexicon claim extractor plus a Python implementation of three CHO metrics (OHR, RIS, wHAL) plus a ROUGE-L baseline. Runs on CPU, no GPU. Used as the deterministic-baseline detector against MedHallu and Med-HALT FCT.

**Mode 2 — Live reasoner backend.** Each extracted claim becomes an OWL ABox individual. HermiT classifies the individual via the formal axioms in `cho.owl`, deriving harm-tier assignments through SubClassOf restrictions. The headline result is per-class precision/recall on a curated 365-example gold set.

**Mode 3 — LLM-queryable infrastructure.** CHO is exposed as a six-tool MCP server. An auditor LLM queries it to classify clinical claims, evaluated under a four-arm experimental design (no-tools / generic-knowledge / cho-tools / cho-ablated) that isolates CHO's contribution from "any structured knowledge access." The headline result is the within-subjects accuracy delta of `cho_tools` over `generic_knowledge` across multiple auditor model tiers.

The implementation plan and three-mode rationale are in `IMPLEMENTATION_PLAN.md`.

## Repository layout

```
cho_test/
├── ontology/
│   ├── cho.owl                            OWL 2 DL ontology (41 classes, 8 properties)
│   ├── build_cho.py                       Programmatic ontology builder (owlready2)
│   ├── reasoner_check.py                  HermiT consistency check
│   ├── reasoner_report.txt                Reasoner output, committed for audit
│   ├── build_contraindications.py         UMLS MED-RT contraindication miner
│   └── data/
│       ├── physiologic_ranges.json        LOINC-anchored plausibility bounds
│       ├── rxnorm_indications.json        Curated drug→indication table
│       ├── semantic_types.json            UMLS Semantic Network slice
│       └── contraindications.json         (regenerated; see Reproducing below)
├── evaluation/
│   ├── medhalt_loader.py                  Med-HALT FCT loader (HuggingFace)
│   ├── medhallu_loader.py                 MedHallu loader (HuggingFace)
│   ├── run_evaluation.py                  Mode 1 orchestrator (rulebook eval)
│   ├── extraction/
│   │   ├── claim_extraction.py            Regex+lexicon claim extractor
│   │   ├── medical_lexicon.json           Seed lexicon (171 terms, hand-curated)
│   │   ├── build_umls_lexicon.py          UMLS-expansion of the seed lexicon
│   │   └── build_umls_predicates.py       UMLS REL/RELA mechanism-predicate builder
│   ├── metrics/                           OHR, RIS, wHAL, ODI implementations
│   ├── baselines/rouge_baseline.py        ROUGE-L baseline
│   ├── mode2/
│   │   ├── abox_translator.py             Claims → OWL individuals
│   │   ├── curate_gold.py                 Generates the gold-label set
│   │   ├── run_reasoner.py                Mode 2 reasoner-based evaluation
│   │   └── gold_labels.json               365 curated examples
│   └── mode3/
│       ├── auditor.py                     Four-arm auditor harness
│       └── run_experiment.py              Mode 3 experiment orchestrator
├── mcp_server/
│   ├── cho_tools.py                       Six CHO tools (in-process callable)
│   └── server.py                          MCP stdio server wrapping the tools
├── IMPLEMENTATION_PLAN.md                 Full design rationale, status snapshot, runbook
├── requirements.txt
└── README.md                              this file
```

## Reproducing each mode

**Setup once:**
```bash
pip install -r requirements.txt
# OR with uv:
#   uv venv && uv pip install -r requirements.txt
# HermiT (the OWL reasoner) is bundled with owlready2 but needs a JRE on PATH:
#   brew install openjdk@17        # macOS
#   apt install default-jre        # Ubuntu
```

**Build the UMLS-derived resources** (one-time; requires a free UMLS license at https://uts.nlm.nih.gov/uts/):
```bash
export UMLS_API_KEY=<your-umls-api-key>
python evaluation/extraction/build_umls_lexicon.py            # → medical_lexicon_umls.json (~1k terms)
python evaluation/extraction/build_umls_predicates.py         # → relation_lexicon.json (51 mechanism predicates)
python ontology/build_contraindications.py                    # → ontology/data/contraindications.json (~600 pairs)
```

**Mode 1** — deterministic baseline against MedHallu / Med-HALT:
```bash
python evaluation/run_evaluation.py --benchmark medhallu \
    --medhallu-config pqa_labeled --source huggingface
python evaluation/run_evaluation.py --benchmark medhallu \
    --medhallu-config pqa_artificial \
    --medhallu-category 'Mechanism and Pathway Misattribution' \
    --medhallu-category 'Methodological and Evidence Fabrication' \
    --source huggingface
```

**Mode 2** — formal reasoner-based classification on the curated gold set:
```bash
python evaluation/mode2/curate_gold.py                        # → evaluation/mode2/gold_labels.json
python evaluation/mode2/run_reasoner.py                       # → evaluation/results/mode2/
```

**Mode 3** — LLM auditor with CHO MCP tools, four-arm design. Requires an OpenAI-compatible LLM endpoint (vLLM, Ollama, OpenAI, etc.):
```bash
export OPENAI_BASE_URL=http://localhost:11434/v1     # e.g. local Ollama
export OPENAI_API_KEY=ollama                          # any string for local stacks
python evaluation/mode3/run_experiment.py \
    --auditor-model qwen3.5:122b \
    --n 365 --verbose
```

To run the cross-tier auditor sweep (scalable-oversight comparison), see `IMPLEMENTATION_PLAN.md`.

The MCP server can also be run standalone for use with Claude Desktop, Cline, Open WebUI, etc.:
```bash
python -m mcp_server.server                           # stdio transport
```

## Credentials and external services

| Service | Why | How to get it |
|---|---|---|
| UMLS Metathesaurus API | Mode 1 lexicon and contraindication build | Free for research at https://uts.nlm.nih.gov/uts/ |
| HuggingFace `datasets` | MedHallu and Med-HALT FCT downloads | Anonymous works; `HF_TOKEN` suppresses rate-limit warnings |
| LLM endpoint (Mode 3) | Auditor calls | Any OpenAI-compatible endpoint (Ollama, vLLM, OpenAI, Together, etc.) |

No service credentials are committed to this repository. UMLS-derived resources (`medical_lexicon_umls.json`, `contraindications.json`) are excluded from version control per UMLS license posture and rebuilt locally from your API key.

## Status

See `IMPLEMENTATION_PLAN.md` for the full status snapshot. Headline numbers from the most recent runs:

- **Mode 1 (full MedHallu, n=2000)**: AUROC ≈ 0.50 across CHO metrics. Bottleneck is closed-vocabulary entity recognition over PubMedQA-style biology prose. Reported as a deterministic-baseline finding.
- **Mode 2 (curated gold, n=170)**: F1 = 0.96 on ForbiddenRelation, 1.00 on ThresholdViolation, 100% specificity on negative controls.
- **Mode 3 (auditor = Qwen 3.5 122B, n=365)**: cho_tools accuracy = 0.956, +0.099 over no_tools baseline, +0.082 over generic-knowledge baseline. Reasoner contribution +0.005 over lookup-only ablation, replicating across two runs.

## License

CHO ontology design: CC-BY-4.0.
Code: MIT (open an issue if you'd prefer a different choice).

## Citation

Citation forthcoming once the paper is accepted.
