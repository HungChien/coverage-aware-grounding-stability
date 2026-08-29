# Coverage-Aware Candidate-Order Stability: Large-Scale Results

## Scope and estimand

This benchmark estimates **operational candidate-order stability**, not semantic correctness. A probe is operationally stable only when the tracked clean candidates remain observable and the clean winner remains ahead of every tracked competitor. Candidate disappearance and threatening candidate birth are explicit failures, never imputed scores.

The exact decomposition checked in every eligible sample is:

`operational risk = coverage risk + coverage × conditional ranking risk`.

## Main finite-probe results

| model | sample_count | eligible_count | clean_eligibility | diagnostic_full_manifest_mean | reference_full_manifest_mean | mae_all | spearman_all | mean_diagnostic_cp_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | 500 | 499 | 0.9980 | 0.8566 | 0.8554 | 0.0253 | 0.9554 | 0.1786 |
| YOLO-World | 500 | 364 | 0.7280 | 0.5098 | 0.5118 | 0.0322 | 0.9880 | 0.2425 |

## Association quality

| model | mean_probe_coverage | mean_matched_iou | mean_perturbed_candidate_count |
| --- | --- | --- | --- |
| GroundingDINO | 0.9449 | 0.9636 | 12.5578 |
| YOLO-World | 0.8071 | 0.9651 | 4.8167 |

## Why coverage-aware stability is not direct persistence

| model | eligible_samples | probe_count | coverage | conditional_ranking | operational_stability | conditional_minus_operational | predicted_gap_from_identity | gap_identity_residual | operational_risk | coverage_risk | conditional_ranking_risk | risk_identity_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | 499 | 39920 | 0.9449 | 0.9070 | 0.8571 | 0.0499 | 0.0499 | 0.0000 | 0.1429 | 0.0551 | 0.0879 | 0.0000 |
| YOLO-World | 364 | 29120 | 0.8071 | 0.8710 | 0.7030 | 0.1680 | 0.1680 | 0.0000 | 0.2970 | 0.1929 | 0.1041 | 0.0000 |

The conditional-minus-operational column is the instability hidden by conditioning on successful candidate association. Its equality to conditional ranking multiplied by one minus coverage is checked numerically.

## Perturbation-family risk attribution

| model | family | trials | operational_stability | risk_share |
| --- | --- | --- | --- | --- |
| GroundingDINO | blur | 7984 | 0.7776 | 0.3113 |
| GroundingDINO | gaussian_noise | 7984 | 0.8030 | 0.2757 |
| GroundingDINO | brightness | 7984 | 0.9533 | 0.0654 |
| GroundingDINO | resolution | 7984 | 0.8320 | 0.2351 |
| GroundingDINO | jpeg | 7984 | 0.9196 | 0.1125 |
| YOLO-World | blur | 5824 | 0.5170 | 0.3252 |
| YOLO-World | gaussian_noise | 5824 | 0.7610 | 0.1609 |
| YOLO-World | brightness | 5824 | 0.9047 | 0.0642 |
| YOLO-World | resolution | 5824 | 0.5745 | 0.2865 |
| YOLO-World | jpeg | 5824 | 0.7577 | 0.1631 |

## Diagnostic-to-reference family profile reproducibility

| model | family | operational_stability_diagnostic | risk_share_diagnostic | operational_stability_reference | risk_share_reference | absolute_stability_error | absolute_risk_share_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | gaussian_noise | 0.8046 | 0.2758 | 0.8030 | 0.2757 | 0.0016 | 0.0001 |
| GroundingDINO | blur | 0.7728 | 0.3207 | 0.7776 | 0.3113 | 0.0048 | 0.0094 |
| GroundingDINO | resolution | 0.8384 | 0.2281 | 0.8320 | 0.2351 | 0.0064 | 0.0070 |
| GroundingDINO | brightness | 0.9484 | 0.0728 | 0.9533 | 0.0654 | 0.0049 | 0.0075 |
| GroundingDINO | jpeg | 0.9274 | 0.1025 | 0.9196 | 0.1125 | 0.0078 | 0.0100 |
| YOLO-World | gaussian_noise | 0.7682 | 0.1547 | 0.7610 | 0.1609 | 0.0072 | 0.0062 |
| YOLO-World | blur | 0.5103 | 0.3268 | 0.5170 | 0.3252 | 0.0067 | 0.0016 |
| YOLO-World | resolution | 0.5683 | 0.2881 | 0.5745 | 0.2865 | 0.0062 | 0.0016 |
| YOLO-World | brightness | 0.9069 | 0.0621 | 0.9047 | 0.0642 | 0.0022 | 0.0021 |
| YOLO-World | jpeg | 0.7479 | 0.1682 | 0.7577 | 0.1631 | 0.0098 | 0.0051 |

## Primary failure causes

| model | cause | count | share_among_failures |
| --- | --- | --- | --- |
| GroundingDINO | winner_missing | 2 | 0.0004 |
| GroundingDINO | competitor_missing | 1848 | 0.3239 |
| GroundingDINO | threatening_birth | 348 | 0.0610 |
| GroundingDINO | ranking_reversal | 3507 | 0.6147 |
| YOLO-World | winner_missing | 482 | 0.0557 |
| YOLO-World | competitor_missing | 4905 | 0.5671 |
| YOLO-World | threatening_birth | 230 | 0.0266 |
| YOLO-World | ranking_reversal | 3032 | 0.3506 |

## Stability is not correctness

| model | clean_wrong_samples | mean_reference_stability_when_clean_wrong |
| --- | --- | --- |
| GroundingDINO | 250 | 0.8280 |
| YOLO-World | 318 | 0.4441 |

This contextual check prevents a category error: a stable output may still be semantically wrong. Correctness is reported only as an external descriptor.

## Tie-aware selective risk

| model | diagnostic_budget | tie_aware_aurc | overall_reference_risk | distinct_diagnostic_scores |
| --- | --- | --- | --- | --- |
| GroundingDINO | 5 | 0.0530 | 0.1446 | 6 |
| GroundingDINO | 10 | 0.0387 | 0.1446 | 11 |
| GroundingDINO | 20 | 0.0330 | 0.1446 | 20 |
| GroundingDINO | 40 | 0.0308 | 0.1446 | 37 |
| YOLO-World | 5 | 0.2023 | 0.4882 | 6 |
| YOLO-World | 10 | 0.1938 | 0.4882 | 11 |
| YOLO-World | 20 | 0.1882 | 0.4882 | 20 |
| YOLO-World | 40 | 0.1867 | 0.4882 | 38 |

Ties are retained as groups, so a discrete small-budget estimator cannot obtain artificial ranking credit from arbitrary ordering within equal scores.

## Paired hierarchical bootstrap

| comparison | model_or_pair | diagnostic_budget | statistic | estimate | lower_95 | upper_95 | bootstrap_repetitions | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paired_cross_model | groundingdino-yoloworld | 40 | diagnostic_difference | 0.3471 | 0.3098 | 0.3840 | 2000 | 500 |
| paired_cross_model | groundingdino-yoloworld | 40 | reference_difference | 0.3441 | 0.3072 | 0.3800 | 2000 | 500 |
| single_model | groundingdino | 40 | diagnostic_mean | 0.8565 | 0.8377 | 0.8739 | 2000 | 500 |
| single_model | groundingdino | 40 | reference_mean | 0.8553 | 0.8372 | 0.8721 | 2000 | 500 |
| single_model | yoloworld | 40 | diagnostic_mean | 0.5094 | 0.4747 | 0.5437 | 2000 | 500 |
| single_model | yoloworld | 40 | reference_mean | 0.5112 | 0.4764 | 0.5444 | 2000 | 500 |

## Interpretation rules

- Full-manifest stability assigns zero to clean outputs that do not expose two distinct candidates.
- Eligible-only stability is diagnostic and is never substituted for the primary full-manifest estimate.
- Family risk shares localise *where* instability is observed under the registered probe distribution; they are not causal effects.
- The 80-probe estimate is an independent finite reference, not an unknowable exact population probability.
- Clopper-Pearson intervals quantify finite-probe uncertainty for each Bernoulli estimand.

## Reproducibility artifacts

The result directory contains the frozen configuration, manifest hash, complete compressed probe traces, row-level summaries, statistical tables, figures, and this report.
