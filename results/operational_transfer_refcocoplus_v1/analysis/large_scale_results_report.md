# Coverage-Aware Candidate-Order Stability: Large-Scale Results

## Scope and estimand

This benchmark estimates **operational candidate-order stability**, not semantic correctness. A probe is operationally stable only when the tracked clean candidates remain observable and the clean winner remains ahead of every tracked competitor. Candidate disappearance and threatening candidate birth are explicit failures, never imputed scores.

The exact decomposition checked in every eligible sample is:

`operational risk = coverage risk + coverage × conditional ranking risk`.

## Main finite-probe results

| model | sample_count | eligible_count | clean_eligibility | diagnostic_full_manifest_mean | reference_full_manifest_mean | mae_all | spearman_all | mean_diagnostic_cp_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | 1000 | 1000 | 1.0000 | 0.8520 | 0.8512 | 0.0276 | 0.9587 | 0.1850 |
| OWLv2 | 1000 | 968 | 0.9680 | 0.7347 | 0.7332 | 0.0396 | 0.9668 | 0.2227 |
| YOLO-World | 1000 | 757 | 0.7570 | 0.5028 | 0.5018 | 0.0310 | 0.9897 | 0.2472 |

## Association quality

| model | mean_probe_coverage | mean_matched_iou | mean_perturbed_candidate_count |
| --- | --- | --- | --- |
| GroundingDINO | 0.9374 | 0.9531 | 12.9160 |
| YOLO-World | 0.7604 | 0.9587 | 5.1649 |
| OWLv2 | 0.8701 | 0.9316 | 13.9363 |

## Why coverage-aware stability is not direct persistence

| model | eligible_samples | probe_count | coverage | conditional_ranking | operational_stability | conditional_minus_operational | predicted_gap_from_identity | gap_identity_residual | operational_risk | coverage_risk | conditional_ranking_risk | risk_identity_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | 1000 | 80000 | 0.9374 | 0.9080 | 0.8512 | 0.0568 | 0.0568 | 0.0000 | 0.1488 | 0.0626 | 0.0862 | 0.0000 |
| YOLO-World | 757 | 60560 | 0.7604 | 0.8717 | 0.6628 | 0.2088 | 0.2088 | 0.0000 | 0.3372 | 0.2396 | 0.0976 | 0.0000 |
| OWLv2 | 968 | 77440 | 0.8701 | 0.8705 | 0.7574 | 0.1131 | 0.1131 | 0.0000 | 0.2426 | 0.1299 | 0.1127 | 0.0000 |

The conditional-minus-operational column is the instability hidden by conditioning on successful candidate association. Its equality to conditional ranking multiplied by one minus coverage is checked numerically.

## Perturbation-family risk attribution

| model | family | trials | operational_stability | risk_share |
| --- | --- | --- | --- | --- |
| GroundingDINO | resolution | 16000 | 0.8371 | 0.2189 |
| GroundingDINO | brightness | 16000 | 0.9605 | 0.0531 |
| GroundingDINO | jpeg | 16000 | 0.7764 | 0.3004 |
| GroundingDINO | blur | 16000 | 0.7682 | 0.3115 |
| GroundingDINO | gaussian_noise | 16000 | 0.9136 | 0.1161 |
| YOLO-World | resolution | 12112 | 0.5263 | 0.2810 |
| YOLO-World | brightness | 12112 | 0.9089 | 0.0540 |
| YOLO-World | jpeg | 12112 | 0.6287 | 0.2202 |
| YOLO-World | blur | 12112 | 0.4622 | 0.3190 |
| YOLO-World | gaussian_noise | 12112 | 0.7881 | 0.1257 |
| OWLv2 | resolution | 15488 | 0.7302 | 0.2225 |
| OWLv2 | brightness | 15488 | 0.8922 | 0.0888 |
| OWLv2 | jpeg | 15488 | 0.7523 | 0.2042 |
| OWLv2 | blur | 15488 | 0.6173 | 0.3155 |
| OWLv2 | gaussian_noise | 15488 | 0.7951 | 0.1690 |

## Diagnostic-to-reference family profile reproducibility

| model | family | operational_stability_diagnostic | risk_share_diagnostic | operational_stability_reference | risk_share_reference | absolute_stability_error | absolute_risk_share_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | jpeg | 0.7732 | 0.3065 | 0.7764 | 0.3004 | 0.0032 | 0.0061 |
| GroundingDINO | gaussian_noise | 0.9166 | 0.1127 | 0.9136 | 0.1161 | 0.0030 | 0.0034 |
| GroundingDINO | resolution | 0.8416 | 0.2141 | 0.8371 | 0.2189 | 0.0046 | 0.0049 |
| GroundingDINO | blur | 0.7660 | 0.3163 | 0.7682 | 0.3115 | 0.0022 | 0.0048 |
| GroundingDINO | brightness | 0.9627 | 0.0504 | 0.9605 | 0.0531 | 0.0022 | 0.0027 |
| YOLO-World | jpeg | 0.6288 | 0.2210 | 0.6287 | 0.2202 | 0.0001 | 0.0008 |
| YOLO-World | gaussian_noise | 0.7835 | 0.1289 | 0.7881 | 0.1257 | 0.0045 | 0.0032 |
| YOLO-World | resolution | 0.5277 | 0.2812 | 0.5263 | 0.2810 | 0.0014 | 0.0002 |
| YOLO-World | blur | 0.4680 | 0.3168 | 0.4622 | 0.3190 | 0.0058 | 0.0022 |
| YOLO-World | brightness | 0.9126 | 0.0520 | 0.9089 | 0.0540 | 0.0037 | 0.0020 |
| OWLv2 | jpeg | 0.7509 | 0.2067 | 0.7523 | 0.2042 | 0.0014 | 0.0025 |
| OWLv2 | gaussian_noise | 0.7969 | 0.1686 | 0.7951 | 0.1690 | 0.0018 | 0.0004 |
| OWLv2 | resolution | 0.7368 | 0.2184 | 0.7302 | 0.2225 | 0.0067 | 0.0041 |
| OWLv2 | blur | 0.6154 | 0.3191 | 0.6173 | 0.3155 | 0.0019 | 0.0036 |
| OWLv2 | brightness | 0.8949 | 0.0872 | 0.8922 | 0.0888 | 0.0026 | 0.0016 |

## Primary failure causes

| model | cause | count | share_among_failures |
| --- | --- | --- | --- |
| GroundingDINO | winner_missing | 5 | 0.0004 |
| GroundingDINO | competitor_missing | 4128 | 0.3467 |
| GroundingDINO | threatening_birth | 874 | 0.0734 |
| GroundingDINO | ranking_reversal | 6900 | 0.5795 |
| YOLO-World | winner_missing | 1721 | 0.0843 |
| YOLO-World | competitor_missing | 12110 | 0.5931 |
| YOLO-World | threatening_birth | 677 | 0.0332 |
| YOLO-World | ranking_reversal | 5910 | 0.2895 |
| OWLv2 | winner_missing | 492 | 0.0262 |
| OWLv2 | competitor_missing | 7133 | 0.3797 |
| OWLv2 | threatening_birth | 2436 | 0.1297 |
| OWLv2 | ranking_reversal | 8724 | 0.4644 |

## Stability is not correctness

| model | clean_wrong_samples | mean_reference_stability_when_clean_wrong |
| --- | --- | --- |
| GroundingDINO | 505 | 0.8345 |
| OWLv2 | 628 | 0.6965 |
| YOLO-World | 633 | 0.4518 |

This contextual check prevents a category error: a stable output may still be semantically wrong. Correctness is reported only as an external descriptor.

## Tie-aware selective risk

| model | diagnostic_budget | tie_aware_aurc | overall_reference_risk | distinct_diagnostic_scores |
| --- | --- | --- | --- | --- |
| GroundingDINO | 5 | 0.0556 | 0.1488 | 6 |
| GroundingDINO | 10 | 0.0427 | 0.1488 | 11 |
| GroundingDINO | 20 | 0.0367 | 0.1488 | 19 |
| GroundingDINO | 40 | 0.0343 | 0.1488 | 35 |
| OWLv2 | 5 | 0.1063 | 0.2668 | 6 |
| OWLv2 | 10 | 0.0968 | 0.2668 | 11 |
| OWLv2 | 20 | 0.0874 | 0.2668 | 21 |
| OWLv2 | 40 | 0.0842 | 0.2668 | 41 |
| YOLO-World | 5 | 0.2192 | 0.4982 | 6 |
| YOLO-World | 10 | 0.2092 | 0.4982 | 11 |
| YOLO-World | 20 | 0.2039 | 0.4982 | 21 |
| YOLO-World | 40 | 0.2016 | 0.4982 | 40 |

Ties are retained as groups, so a discrete small-budget estimator cannot obtain artificial ranking credit from arbitrary ordering within equal scores.

## Paired hierarchical bootstrap

| comparison | model_or_pair | diagnostic_budget | statistic | estimate | lower_95 | upper_95 | bootstrap_repetitions | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paired_cross_model | groundingdino-owlv2 | 40 | diagnostic_difference | 0.1172 | 0.0960 | 0.1381 | 2000 | 1000 |
| paired_cross_model | groundingdino-owlv2 | 40 | reference_difference | 0.1177 | 0.0972 | 0.1386 | 2000 | 1000 |
| paired_cross_model | groundingdino-yoloworld | 40 | diagnostic_difference | 0.3498 | 0.3258 | 0.3754 | 2000 | 1000 |
| paired_cross_model | groundingdino-yoloworld | 40 | reference_difference | 0.3497 | 0.3265 | 0.3757 | 2000 | 1000 |
| paired_cross_model | owlv2-yoloworld | 40 | diagnostic_difference | 0.2326 | 0.2060 | 0.2612 | 2000 | 1000 |
| paired_cross_model | owlv2-yoloworld | 40 | reference_difference | 0.2321 | 0.2055 | 0.2597 | 2000 | 1000 |
| single_model | groundingdino | 40 | diagnostic_mean | 0.8523 | 0.8392 | 0.8648 | 2000 | 1000 |
| single_model | groundingdino | 40 | reference_mean | 0.8513 | 0.8384 | 0.8633 | 2000 | 1000 |
| single_model | owlv2 | 40 | diagnostic_mean | 0.7351 | 0.7169 | 0.7519 | 2000 | 1000 |
| single_model | owlv2 | 40 | reference_mean | 0.7336 | 0.7163 | 0.7495 | 2000 | 1000 |
| single_model | yoloworld | 40 | diagnostic_mean | 0.5025 | 0.4796 | 0.5246 | 2000 | 1000 |
| single_model | yoloworld | 40 | reference_mean | 0.5015 | 0.4784 | 0.5229 | 2000 | 1000 |

## Interpretation rules

- Full-manifest stability assigns zero to clean outputs that do not expose two distinct candidates.
- Eligible-only stability is diagnostic and is never substituted for the primary full-manifest estimate.
- Family risk shares localise *where* instability is observed under the registered probe distribution; they are not causal effects.
- The 80-probe estimate is an independent finite reference, not an unknowable exact population probability.
- Clopper-Pearson intervals quantify finite-probe uncertainty for each Bernoulli estimand.

## Reproducibility artifacts

The result directory contains the frozen configuration, manifest hash, complete compressed probe traces, row-level summaries, statistical tables, figures, and this report.
