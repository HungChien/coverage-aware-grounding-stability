# Frozen Ref-L4 Transfer Preregistration

## Purpose

This experiment is a stronger external validation of the coverage-aware
candidate-order stability framework.  It tests whether the same output
contract, probe distribution, estimands, and finite-probe conclusions remain
usable for longer expressions, a broader category vocabulary, smaller target
instances, and images from both COCO and Objects365.

## Data and sampling

- Dataset: official Ref-L4 test split.
- Sample size: 1,000 different images and one expression per image.
- Source-stratified design: 600 COCO-source images and 400 Objects365-source
  images.  The Objects365 stratum is intentionally oversampled to provide a
  well-powered external visual-domain check; the pooled result therefore
  describes this registered benchmark manifest rather than the unweighted
  Ref-L4 population.
- COCO exclusion: no overlap with images used by prior RefCOCO development,
  calibration, unseen-test, or RefCOCO+ transfer manifests.
- Selection is deterministic from metadata and is completed before either
  model is run.
- Ref-L4 val is not used because the official val and test tables share many
  images and target instances.

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
- Per-sample uncertainty: exact 95% Clopper--Pearson intervals for eligible
  samples.
- GroundingDINO computational batch size is four because Ref-L4 contains very
  high-resolution images and the prior batch size of eight reached the 16 GB
  device limit.  Batch size is an execution parameter and does not alter the
  candidate contract, probes, estimator, or model weights.  This value is set
  before Ref-L4 inference.

## Primary estimand

Operational candidate-order stability is the probability that the registered
candidate set remains covered and the clean winner remains ahead of all
tracked competitors under a probe drawn from the frozen distribution.

Clean outputs with fewer than two spatially distinct candidates receive zero
full-manifest operational stability.  Eligible-only results are secondary.
The benchmark measures stability under the registered visual probe
distribution, not semantic correctness.

## Registered questions

1. Does a finite diagnostic probe budget predict an independent 80-probe
   reference on Ref-L4 for both architectures?
2. Does the coverage-aware decomposition continue to expose instability hidden
   by conditional persistence?
3. Do failure mechanisms and perturbation-family risk profiles transfer from
   RefCOCO and RefCOCO+ to Ref-L4?
4. Are the conclusions consistent in the COCO and Objects365 source strata?
5. How do stability and eligibility vary with registered query-length and
   target-scale strata?

## Registered analyses

- Full-manifest and eligible-only operational stability.
- Candidate coverage, conditional ranking stability, and their exact risk
  decomposition.
- Diagnostic-to-reference MAE, bias, eligible-sample Spearman correlation,
  interval width, and reference inclusion.
- Failure-cause and perturbation-family risk shares.
- Tie-aware selective-risk analysis.
- 2,000-repetition bootstrap intervals for model means, model differences,
  Ref-L4 minus RefCOCO, and Ref-L4 minus RefCOCO+.
- Separate pooled, COCO-source, and Objects365-source summaries.
- Fixed query-length strata: at most 18 words, 19--29 words, and at least 30
  words.
- Fixed target-scale strata using square-root box area: small below 32 pixels,
  medium from 32 through 96 pixels, and large above 96 pixels.
- Cross-dataset perturbation-family profile Spearman correlation, cosine
  similarity, and mean absolute risk-share shift.
- Post-primary candidate-contract sensitivity is explicitly labelled and does
  not replace the frozen primary setting.

## Interpretation boundary

The Objects365 stratum is the strongest visual-domain check.  The COCO stratum
tests transfer to much longer expressions and a different annotation design
while controlling the broad image source.  Family-level risk shares localise
observed input-side vulnerability; they are descriptive, not causal internal
module attribution.  No parameter, sample, threshold, or analysis rule may be
changed in response to Ref-L4 model outputs after the freeze record is created.
