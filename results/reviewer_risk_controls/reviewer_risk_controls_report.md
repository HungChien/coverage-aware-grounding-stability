# Reviewer-risk controls

This analysis addresses four interpretation risks using the frozen reference
traces. It does not rerun a grounding model. All intervals use 2,000
image-query-pair bootstrap resamples with seed 20260901.

## Candidate-pool truncation

Clean and perturbed cap hits have different implications. In all nine
dataset-model groups, every clean pool that reached 20 had already supplied
five spatially distinct tracked candidates. The clean cap therefore cannot
change the registered `Kt=5` endpoint.

For perturbed pools, the audit gives a deliberately pessimistic bound: every
failed cap-hit reference trial is assumed to become stable under a wider pool,
and no stable trial is allowed to deteriorate. The largest correction is
0.02945 for YOLO-World on Ref-L4, giving an upper bound of 0.565775. This
remains below OWLv2 (0.767688) and GroundingDINO (0.870313).

The bound is not a correction estimate. It shows that exposure truncation can
affect absolute stability and failure labels, but cannot explain the observed
model ordering. The completed raw-pool-50 run directly tests all 38,679
selected cap-hit probes. Relative to fresh exposure 20 from the same inference,
exposure 50 changes full-manifest stability by 0 to 0.001450 across the nine
groups. Every clean top-five prefix matches and model ordering is unchanged.
The summary is in
`results/cap50_confirmatory/summary/cap50_stability_summary.csv`.

## Model-dependent eligibility

The three-model common-eligibility sets contain 362 RefCOCO, 746 RefCOCO+ and
780 Ref-L4 pairs. On those identical pairs, YOLO-World competitor loss remains
56%, 60% and 59% of failed reference probes. GroundingDINO rank reversal
remains 67%, 62% and 84%. Thus the broad descriptive profiles persist after
holding the pair set fixed. They remain output symptoms rather than identified
causal architecture mechanisms.

The accompanying full-manifest analysis avoids conditioning entirely. Its
denominator is every manifest pair-probe slot and it includes clean
ineligibility as a mutually exclusive outcome.

## Severity response

Native severities are mapped to a within-family distortion coordinate in
[0,1] and divided into five equal-width bins. Blur and noise increase with
their native parameters; JPEG and resolution are reversed; brightness uses
absolute distance from one. The coordinate supports within-family comparison
only.

Blur declines strongly across all groups. On RefCOCO, for example,
GroundingDINO falls from 0.9293 in the mildest blur bin to 0.6460 in the
strongest. Brightness is flatter, while JPEG and noise show larger
model-dependent declines. Full tables, trial counts and pair-clustered
intervals are in `severity_stability.csv`.

## Family-weight sensitivity

For a fixed dataset, a model gap under any non-negative family mixture is a
weighted average of the five family-specific gaps. The attainable minimum and
maximum therefore occur at single-family endpoints. All nine minimum gaps are
positive. The narrowest is 0.048125 for GroundingDINO minus OWLv2 on RefCOCO.

This proves mixture-weight invariance over the five measured families. It does
not establish transport to an unmeasured deployment distribution or to shifts
outside the registered severity ranges.

## Reproduction

```powershell
python scripts\analyse_reviewer_risk_controls.py
```

The script writes the CSV files, four figure sets and `analysis_audit.json` to
this directory.
