# CHO: Clinical Hallucination Ontology — original artifact description

> **Note:** This document describes the artifact at its initial design state.
> For the current state (three-mode evaluation, MCP server, expanded
> resources), see `README.md` and `IMPLEMENTATION_PLAN.md`. This file is
> retained as historical context.

Companion artifact for the NeurIPS 2026 Evaluations & Datasets Track submission *CHO: An Ontology-Grounded Evaluation Framework for Clinical Hallucinations in Large Language Models*.

This repository contains four things:

1. The CHO ontology, a reasoner-validated OWL 2 file (`ontology/cho.owl`).
2. Supporting curated lookup tables for the harm-tier axioms (`ontology/data/`).
3. An end-to-end evaluation pipeline that runs the CHO metrics on the Med-HALT False Confidence Test, with no GPU and no UMLS license required (`evaluation/`).
4. The paper source itself (`main.tex`, `references.bib`).

## Status of the artifact

| Component | Status |
|---|---|
| OWL 2 ontology (41 classes, 18 properties) | reasoner-validated by HermiT |
| RxNorm-style indication table (20 drugs) | hand-curated, public sources |
| Physiologic range table (15 labs, LOINC-anchored) | hand-curated, public sources |
| UMLS-style semantic-type tree (30 types) | hand-curated, derived from UMLS Semantic Network |
| Lightweight claim extractor (no scispacy needed) | runs on a laptop, ~80% precision on smoke fixture |
| OHR / RIS / ODI / wHAL metrics | implemented and unit-tested |
| ROUGE-L baseline | implemented |
| Med-HALT FCT loader | runs against Hugging Face hub |
| Smoke fixture (24 hand-built examples) | bundled, runs offline |
| End-to-end orchestrator | runs in under one minute on CPU |

## Quick start

```bash
pip install -r requirements.txt
cd ontology
python build_cho.py cho.owl                 # build the ontology
python reasoner_check.py                     # run HermiT consistency check
cd ../evaluation
python run_evaluation.py --source fixture    # offline smoke run
python run_evaluation.py --source huggingface # full Med-HALT run (needs net)
```

## Smoke-test results (offline, 24 examples)

| Metric | AUROC | Mean on hallucinated | Mean on correct |
|---|---:|---:|---:|
| **wHAL (CHO)** | **0.783** | 21.82 | 9.85 |
| OHR (CHO) | 0.668 | 0.27 | 0.15 |
| RIS (CHO) | 0.636 | 0.27 | 0.00 |
| ROUGE-L 1-F1 (baseline) | 0.259 | 0.73 | 0.87 |

ROUGE-L scores below 0.5, meaning surface-form lexical overlap is not just uninformative on this task, it is **anti-informative**. Hallucinated answers tend to share more vocabulary with their question stems than correct answers do. This is exactly the gap that motivates ontology-grounded scoring.

CHO, by contrast, gives a clean separation. wHAL has the strongest signal because it integrates all three axiom-based detectors with severity weighting.

## Repository layout

```
cho_neurips2026_eandd/
|-- main.tex                                 # paper source (NeurIPS 2026 ED)
|-- references.bib                           # 18 verified citations
|-- README.md                                # paper-side README
|-- requirements.txt
|-- ontology/
|   |-- build_cho.py                         # ontology constructor
|   |-- reasoner_check.py                    # HermiT/Pellet consistency check
|   |-- cho.owl                              # the artifact (generated)
|   |-- reasoner_report.txt                  # consistency check output
|   |-- README.md                            # ontology-side instructions
|   `-- data/
|       |-- rxnorm_indications.json          # 20 drugs and their indications
|       |-- physiologic_ranges.json          # 15 labs with plausibility bounds
|       `-- semantic_types.json              # UMLS semantic-type tree
`-- evaluation/
    |-- medhalt_loader.py                    # HF loader + offline fixture
    |-- run_evaluation.py                    # orchestrator
    |-- extraction/
    |   |-- medical_lexicon.json             # bundled clinical vocab
    |   `-- claim_extraction.py              # entities, relations, numerics
    |-- metrics/
    |   |-- ohr.py                           # Ontological Hallucination Rate
    |   |-- ris.py                           # Relational Inversion Score
    |   |-- odi.py                           # Ontological Drift Index
    |   `-- whal.py                          # Harm-Weighted Hallucination
    |-- baselines/
    |   `-- rouge_baseline.py                # ROUGE-L F1
    |-- data/
    |   `-- medhalt_fct_smoke.json           # 24 offline examples
    `-- results/
        |-- results.json                     # per-example scores (generated)
        |-- summary.json                     # aggregates (generated)
        `-- results_table.md                 # markdown report (generated)
```

## Honest limitations of the laptop-runnable pipeline

The artifact above is designed for portability, not for production strength. Three honest caveats:

**Extraction precision.** The bundled lexicon covers ~150 clinical terms. A real deployment should use scispacy with `en_core_sci_lg` plus QuickUMLS or MedCAT. Substituting them is a drop-in replacement for `extraction/claim_extraction.py`; the metric modules consume the same dataclass.

**Knowledge graph depth.** RxNorm, LOINC, and the UMLS semantic network are all represented by curated subsets. Loading the full graphs adds coverage and reduces false negatives. The `_load_*` functions can swap to a SPARQL endpoint without API changes.

**Implicit-claim detection.** The relation extractor only fires on explicit verbs (`treats`, `causes`). When a model answers a `which medication treats X` question with a bare drug name, no relation is built and OHR cannot fire. A natural extension is a question-aware post-processor that constructs implied triples from a question stem plus an answer span. This is left for the production version.

**Text vs task correctness.** CHO scores hallucination at the level of the **output text**, not at the level of task accuracy. For most generated clinical text these align; in edge cases (a Med-HALT FCT question asking which lab value is incompatible with life, where the correct answer names the impossible value) CHO will correctly fire on the text even though the answer is task-correct. The smoke fixture contains two such cases (smoke-014, smoke-015), which inflate the false-positive rate of OHR slightly. We consider this a feature rather than a bug, and the paper discusses it explicitly.

## What you need to run the full evaluation

Same machine that runs `pip install -r requirements.txt` plus internet access to Hugging Face for the full Med-HALT FCT (1,000+ examples). Expect the run to finish in 2 to 5 minutes on a 2020-or-later laptop. No GPU.

## Reproducing the paper results section

Once you run `python run_evaluation.py --source huggingface` and get the full FCT numbers, the produced `results/results_table.md` and `results/summary.json` populate the empirical content of Section 8 (Results) and Section 9 (Discussion) of the paper. We will add the corresponding LaTeX with real numbers in the next pass.

## Citation

If you use CHO in your research, please cite the paper (template):

```bibtex
@inproceedings{cho2026,
  title  = {CHO: An Ontology-Grounded Evaluation Framework for Clinical
            Hallucinations in Large Language Models},
  author = {Anonymous Authors},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},
  year   = {2026}
}
```

## License

- Code (this repository): MIT
- CHO ontology and curated tables: CC-BY-4.0
- Med-HALT data: see the dataset's own license at https://huggingface.co/datasets/openlifescienceai/Med-HALT
