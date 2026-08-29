# Frozen RefCOCO+ Transfer Preregistration

## Purpose

This experiment tests whether the coverage-aware candidate-order stability
framework transfers from unseen RefCOCO samples to RefCOCO+ without changing
the model adapters, output contract, probe distribution, estimators, or
analysis rules.

## Data

- Dataset: RefCOCO+ UNC testA and testB.
- Sample size: 500 unique images from testA and 500 unique images from testB.
- Sampling unit: one referring expression per unique image.
- Exclusion: zero image overlap with all prior RefCOCO development,
  calibration, and evaluation manifests.
- The testA and testB samples must also have zero image overlap.

## Frozen protocol

- Models: GroundingDINO-Tiny and YOLO-World-S.
- Clean tracked candidates: 5 spatially distinct boxes.
- Perturbed exposed candidates: 20 boxes.
- Duplicate IoU: 0.70.
- Association IoU: 0.15.
- Threatening-birth duplicate IoU: 0.70.
- Probe families: blur, brightness, JPEG, resolution, and Gaussian noise.
- Diagnostic probes: 40, balanced across the five families.
- Independent reference probes: 80, balanced across the five families.
- Reported diagnostic budgets: 5, 10, 20, and 40.
- Per-sample intervals: exact 95% Clopper--Pearson intervals, defined only for
  clean-eligible samples.

## Primary estimand

Operational candidate-order stability is the probability that the registered
candidate contract remains covered and the clean winner remains ahead of all
tracked competitors under a probe drawn from the frozen distribution.

The primary full-manifest estimate assigns zero operational stability to clean
outputs that do not expose at least two spatially distinct candidates. The
eligible-only estimate is secondary and must not replace the primary result.

## Registered transfer questions

1. Does the finite diagnostic estimate remain predictive of the independent
   80-probe reference on RefCOCO+?
2. Does conditional ranking stability remain optimistically biased relative
   to operational stability because of incomplete candidate coverage?
3. Are the diagnostic perturbation-family risk shares reproducible in the
   independent reference probes?
4. Does the same output-level framework remain executable and interpretable
   for both grounding architectures?
5. How do the absolute estimands and failure mechanisms shift from RefCOCO to
   RefCOCO+, and between testA and testB?

## Registered analyses

- Full-manifest and eligible-only operational stability.
- Candidate coverage and conditional ranking stability.
- Exact coverage/ranking risk decomposition.
- Diagnostic-to-reference MAE, bias, and eligible-sample Spearman correlation.
- Eligible-only Clopper--Pearson reference inclusion rate and interval width.
- Failure-cause shares and culprit competitor rank.
- Perturbation-family stability and risk shares.
- Tie-aware selective-risk analysis.
- 2,000-repetition bootstrap intervals for model means, model differences,
  and RefCOCO+ minus RefCOCO shifts.
- Separate testA, testB, and pooled target summaries.
- Source-to-target family-profile Spearman correlation, cosine similarity,
  and mean absolute risk-share shift.

## Interpretation boundary

The experiment evaluates stability under the registered probe distribution,
not semantic correctness. A difference between RefCOCO and RefCOCO+ is a
dataset-shift result, not evidence that one dataset is intrinsically harder.
Perturbation-family risk shares localise observed input-side vulnerability and
must not be described as causal internal-module attribution.

No parameter or analysis rule may be changed after the freeze record is
created in response to target-model outputs. Any post-primary sensitivity
analysis must be labelled explicitly.
