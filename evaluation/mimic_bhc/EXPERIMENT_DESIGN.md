# Real-output validation on MIMIC-IV-BHC: experiment design

Written 2026-08-02. Answers the NeurIPS metareview's lead criticism: that the
evaluation runs entirely on a synthetic, CHO-specific benchmark, so it is unclear
how the framework behaves on naturally occurring clinical hallucinations.

## The problem this design has to solve

Synthetic data gives labels for free. We construct the violation, so we know its
class and harm tier. Real generated text does not work that way. A hospital-course
summary contains claims that no one has labeled, and most of them are ordinary
correct statements. Everything below is organized around one question: where does
the ground truth come from, and how much of it can be trusted.

Getting this wrong in either direction is fatal. Label everything by hand and the
study is too small to say anything. Label everything automatically with CHO and the
study is circular, because CHO would be judged only on the claims CHO can already
check.

## Stage 0. The sample

`sample_bhc.py` draws 60 admissions from `mimiciv_note.discharge`, one note per
subject, each 4,000 to 20,000 characters, each containing a `Brief Hospital Course:`
section long enough to summarize. Seed 20260802. Realized sample: 60 admissions, 60
distinct subjects, source text median 6,850 characters, reference hospital course
median 1,861 characters, zero unsplittable.

The input is the discharge note with the hospital course removed. The reference is
the removed section. This is the MIMIC-IV-BHC construction.

**Credentialed data.** Note text lives only under `evaluation/results/`, which is
gitignored. It must not be quoted in the paper, in figures, in the appendix, or in
any released artifact. Auditors and generators run on local endpoints only, so no
note text reaches a third-party API.

## Stage 1. Generation, and the leave-one-out audit design

All seven models write their own hospital course from the source text, at
temperature 0. 60 admissions times 7 generators gives 420 generated summaries.

**Auditing is leave-one-out.** Each model audits the summaries written by the other
six, never its own. That is 42 generator-auditor pairings, it uses every model on
both sides, and it removes the self-preference confound structurally rather than by
excluding models.

**Self-audit is then run as a separate condition.** Each model also audits its own
output, reported apart from the main results. The difference between a model's
verdicts on its own summaries and on other models' summaries is a measurement of
self-preference, which is a result rather than a confound to apologize for.

A hosted proprietary auditor was considered and rejected. MIMIC-IV is credentialed
under a data use agreement that forbids sending the data to third-party services,
and generated summaries inherit that restriction because they carry patient
specifics. Independently, the paper's own rationale for open-weight models is
reproducible local serving without dependence on opaque external APIs, and a
proprietary auditor would contradict it.

## Stage 2. Claim extraction

Run the existing CHO extraction pipeline over each generated summary. Record, per
claim: the source span, the extracted triple or value, and whether the claim falls
inside CHO verification scope.

Expect on the order of 10 to 30 claims per summary, so roughly 1,200 to 3,600 claims
in total. Claim count, not note count, is the effective sample size. This is the
answer to "60 notes is too small."

## Stage 3. Labeling, in three tiers

**L1, CHO-covered automatic.** Claims that hit a CHO verification source get an
automatic verdict: forbidden relation, threshold violation, inverted relation,
fabricated entity. High precision, low coverage.

**L2, source-note support.** For every claim, check whether its content is present
in the source note: entity match, value match, or neither. This yields supported,
unsupported, or not checkable. L2 catches fabrication, which is likely the dominant
real hallucination mode and which L1 largely cannot see.

**L3, clinician adjudication. DEFERRED TO FUTURE WORK.** No clinician is available
before the submission deadline. The design is specified in the clinician-review
protocol section of the paper and is not executed here: a stratified subsample of 120
to 150 claims, stratified by CHO class and harm tier, deliberately oversampling the
claims where L1 and L2 disagree, adjudicated by more than one reviewer with reported
agreement.

**The consequence has to be stated plainly, not buried.** L3 is what would license L1
and L2. Without it the automatic labels are unvalidated, so this study cannot report
accuracy against truth on real output. It must not present L1 or L2 counts as if they
were adjudicated findings, and it must not report L3 numbers that do not exist.

## Stage 4. What gets measured

Because L3 is deferred, this is a coverage and behavior study, not an accuracy study.
Items 1 through 3 and 5 below are valid without any gold labels. Item 4 is deferred
with L3.

1. **Coverage.** The fraction of real generated claims that fall inside CHO
   verification scope. Report this first and prominently. It will be far below the
   synthetic set's coverage, and that gap is the honest headline, not a
   disappointment. The reviewers already asked that uncovered claims never be
   treated as safe.
2. **OHR and wHAL on covered claims**, always printed next to coverage.
3. **Class distribution, real versus synthetic.** The synthetic gold set is 63%
   ForbiddenRelation. If real summaries are dominated by unsupported detail instead,
   the synthetic benchmark is not prevalence-representative and the paper must say
   so. This single comparison is the most reviewer-relevant analysis in the study.
4. **Auditor performance against adjudicated labels.** Deferred with L3. This would
   have been the real generalization number, and its absence is the study's main
   limitation.
5. **Harm-tier distribution** of detected violations in real output.
6. **Inter-auditor agreement on real output.** Needs no gold labels at all, so it is
   fully available now. It also connects to the paper's existing within-family
   oversight analysis.
7. **Self-preference.** Each model's verdicts on its own summaries versus on other
   models' summaries.

## Stage 5. The transfer question. DEFERRED WITH L3

Does auditor ranking on the synthetic set predict auditor ranking on real output?
Answering it properly needs real-output ground truth, so it defers with L3. State the
question and the commitment now rather than quietly dropping it.

The pre-commitment still holds and should be written into the paper before the numbers
exist. If ranking transfers, the synthetic benchmark is validated as a cheap proxy and
that is a genuine contribution. If it does not, the synthetic results are a stress test
of the audit machinery and not evidence about auditor quality in deployment.

A label-free substitute is available but weak: correlate synthetic balanced accuracy
with how often each auditor agrees with the majority of the other auditors on real
output. Agreement with the majority is not correctness, as this paper's own results
show, so this can be reported as suggestive at most and must not be presented as the
transfer result.

## What a reviewer will attack

| Objection | Answer built into the design |
|---|---|
| 60 notes is too small | Claim-level N is 1,200 to 3,600. Framed as a validation probe, not a benchmark. State the power. |
| Automatic labels are circular: CHO grades what CHO can check | Coverage is reported separately rather than folded into accuracy, and no accuracy claim is made on real output. L3 would settle it and is specified but deferred. |
| This is not prevalence | The sample is not prevalence-matched to any deployed system. Stated as a limitation, not implied away. |
| No clinician validation | Acknowledged as the main limitation. The protocol is specified and deferred, and no adjudicated number is reported. |
| The generator audits itself | Leave-one-out by construction, with self-audit reported separately as a measured effect. |
| PHI exposure | Local endpoints only, gitignored storage, no note text in any published artifact. |

## Status and dependencies

- Stage 0 is done. `evaluation/results/mimic_bhc/bhc_sample.json`, 60 admissions.
- Stages 1 through 3 need local model endpoints. `localhost:11434` and `localhost:29999`
  were both unreachable on 2026-08-02.
- Stage 3 L3 is deferred. No clinician is available before the deadline. The paper
  specifies the protocol, reports L1 and L2 as automatic and unvalidated, and states
  the limitation plainly.
- Model serving is on deepthink.csam.montclair.edu (10.8.8.18). SSH is open; the model
  ports are not directly exposed, so 11434 and 29999 need forwarding.
