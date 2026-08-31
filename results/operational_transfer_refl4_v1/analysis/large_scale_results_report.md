# Coverage-Aware Candidate-Order Stability: Large-Scale Results

## Scope and estimand

This benchmark estimates **operational candidate-order stability**, not semantic correctness. A probe is operationally stable only when the tracked clean candidates remain observable and the clean winner remains ahead of every tracked competitor. Candidate disappearance and threatening candidate birth are explicit failures, never imputed scores.

The exact decomposition checked in every eligible sample is:

`operational risk = coverage risk + coverage × conditional ranking risk`.

## Main finite-probe results

| model | sample_count | eligible_count | clean_eligibility | diagnostic_full_manifest_mean | reference_full_manifest_mean | mae_all | spearman_all | mean_diagnostic_cp_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | 1000 | 1000 | 1.0000 | 0.8699 | 0.8703 | 0.0256 | 0.9549 | 0.1723 |
| OWLv2 | 1000 | 998 | 0.9980 | 0.7682 | 0.7677 | 0.0399 | 0.9612 | 0.2137 |
| YOLO-World | 1000 | 781 | 0.7810 | 0.5386 | 0.5363 | 0.0319 | 0.9882 | 0.2428 |

## Association quality

| model | mean_probe_coverage | mean_matched_iou | mean_perturbed_candidate_count |
| --- | --- | --- | --- |
| GroundingDINO | 0.9776 | 0.9703 | 14.6205 |
| YOLO-World | 0.7786 | 0.9475 | 8.9866 |
| OWLv2 | 0.8935 | 0.9149 | 16.0131 |

## Why coverage-aware stability is not direct persistence

| model | eligible_samples | probe_count | coverage | conditional_ranking | operational_stability | conditional_minus_operational | predicted_gap_from_identity | gap_identity_residual | operational_risk | coverage_risk | conditional_ranking_risk | risk_identity_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | 1000 | 80000 | 0.9776 | 0.8902 | 0.8703 | 0.0199 | 0.0199 | 0.0000 | 0.1297 | 0.0224 | 0.1073 | 0.0000 |
| YOLO-World | 781 | 62480 | 0.7786 | 0.8820 | 0.6867 | 0.1953 | 0.1953 | 0.0000 | 0.3133 | 0.2214 | 0.0919 | 0.0000 |
| OWLv2 | 998 | 79840 | 0.8935 | 0.8609 | 0.7692 | 0.0917 | 0.0917 | 0.0000 | 0.2308 | 0.1065 | 0.1242 | 0.0000 |

The conditional-minus-operational column is the instability hidden by conditioning on successful candidate association. Its equality to conditional ranking multiplied by one minus coverage is checked numerically.

## Perturbation-family risk attribution

| model | family | trials | operational_stability | risk_share |
| --- | --- | --- | --- | --- |
| GroundingDINO | blur | 16000 | 0.7913 | 0.3219 |
| GroundingDINO | brightness | 16000 | 0.9496 | 0.0777 |
| GroundingDINO | gaussian_noise | 16000 | 0.9106 | 0.1379 |
| GroundingDINO | resolution | 16000 | 0.8614 | 0.2137 |
| GroundingDINO | jpeg | 16000 | 0.8387 | 0.2488 |
| YOLO-World | blur | 12496 | 0.4894 | 0.3259 |
| YOLO-World | brightness | 12496 | 0.9062 | 0.0599 |
| YOLO-World | gaussian_noise | 12496 | 0.7714 | 0.1459 |
| YOLO-World | resolution | 12496 | 0.5813 | 0.2673 |
| YOLO-World | jpeg | 12496 | 0.6852 | 0.2010 |
| OWLv2 | blur | 15968 | 0.6450 | 0.3076 |
| OWLv2 | brightness | 15968 | 0.8872 | 0.0977 |
| OWLv2 | gaussian_noise | 15968 | 0.7982 | 0.1749 |
| OWLv2 | resolution | 15968 | 0.7344 | 0.2302 |
| OWLv2 | jpeg | 15968 | 0.7812 | 0.1896 |

## Diagnostic-to-reference family profile reproducibility

| model | family | operational_stability_diagnostic | risk_share_diagnostic | operational_stability_reference | risk_share_reference | absolute_stability_error | absolute_risk_share_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | brightness | 0.9491 | 0.0782 | 0.9496 | 0.0777 | 0.0005 | 0.0005 |
| GroundingDINO | jpeg | 0.8377 | 0.2493 | 0.8387 | 0.2488 | 0.0009 | 0.0006 |
| GroundingDINO | gaussian_noise | 0.9119 | 0.1354 | 0.9106 | 0.1379 | 0.0013 | 0.0025 |
| GroundingDINO | resolution | 0.8618 | 0.2124 | 0.8614 | 0.2137 | 0.0003 | 0.0012 |
| GroundingDINO | blur | 0.7887 | 0.3246 | 0.7913 | 0.3219 | 0.0025 | 0.0027 |
| YOLO-World | brightness | 0.9040 | 0.0619 | 0.9062 | 0.0599 | 0.0022 | 0.0020 |
| YOLO-World | jpeg | 0.6877 | 0.2012 | 0.6852 | 0.2010 | 0.0026 | 0.0003 |
| YOLO-World | gaussian_noise | 0.7726 | 0.1466 | 0.7714 | 0.1459 | 0.0011 | 0.0007 |
| YOLO-World | resolution | 0.5930 | 0.2623 | 0.5813 | 0.2673 | 0.0117 | 0.0050 |
| YOLO-World | blur | 0.4910 | 0.3280 | 0.4894 | 0.3259 | 0.0016 | 0.0021 |
| OWLv2 | brightness | 0.8884 | 0.0969 | 0.8872 | 0.0977 | 0.0012 | 0.0008 |
| OWLv2 | jpeg | 0.7848 | 0.1869 | 0.7812 | 0.1896 | 0.0036 | 0.0027 |
| OWLv2 | gaussian_noise | 0.7950 | 0.1781 | 0.7982 | 0.1749 | 0.0033 | 0.0032 |
| OWLv2 | resolution | 0.7358 | 0.2294 | 0.7344 | 0.2302 | 0.0014 | 0.0008 |
| OWLv2 | blur | 0.6445 | 0.3087 | 0.6450 | 0.3076 | 0.0005 | 0.0011 |

## Primary failure causes

| model | cause | count | share_among_failures |
| --- | --- | --- | --- |
| GroundingDINO | winner_missing | 6 | 0.0006 |
| GroundingDINO | competitor_missing | 942 | 0.0908 |
| GroundingDINO | threatening_birth | 842 | 0.0812 |
| GroundingDINO | ranking_reversal | 8585 | 0.8275 |
| YOLO-World | winner_missing | 1124 | 0.0574 |
| YOLO-World | competitor_missing | 11558 | 0.5905 |
| YOLO-World | threatening_birth | 1152 | 0.0589 |
| YOLO-World | ranking_reversal | 5740 | 0.2932 |
| OWLv2 | winner_missing | 142 | 0.0077 |
| OWLv2 | competitor_missing | 4358 | 0.2365 |
| OWLv2 | threatening_birth | 4006 | 0.2174 |
| OWLv2 | ranking_reversal | 9919 | 0.5383 |

## Stability is not correctness

| model | clean_wrong_samples | mean_reference_stability_when_clean_wrong |
| --- | --- | --- |
| GroundingDINO | 613 | 0.8449 |
| OWLv2 | 611 | 0.7390 |
| YOLO-World | 644 | 0.5067 |

This contextual check prevents a category error: a stable output may still be semantically wrong. Correctness is reported only as an external descriptor.

## Tie-aware selective risk

| model | diagnostic_budget | tie_aware_aurc | overall_reference_risk | distinct_diagnostic_scores |
| --- | --- | --- | --- | --- |
| GroundingDINO | 5 | 0.0402 | 0.1297 | 6 |
| GroundingDINO | 10 | 0.0329 | 0.1297 | 11 |
| GroundingDINO | 20 | 0.0284 | 0.1297 | 19 |
| GroundingDINO | 40 | 0.0252 | 0.1297 | 35 |
| OWLv2 | 5 | 0.0926 | 0.2323 | 6 |
| OWLv2 | 10 | 0.0794 | 0.2323 | 11 |
| OWLv2 | 20 | 0.0733 | 0.2323 | 21 |
| OWLv2 | 40 | 0.0687 | 0.2323 | 40 |
| YOLO-World | 5 | 0.1958 | 0.4637 | 6 |
| YOLO-World | 10 | 0.1866 | 0.4637 | 11 |
| YOLO-World | 20 | 0.1806 | 0.4637 | 20 |
| YOLO-World | 40 | 0.1786 | 0.4637 | 41 |

Ties are retained as groups, so a discrete small-budget estimator cannot obtain artificial ranking credit from arbitrary ordering within equal scores.

## Paired hierarchical bootstrap

| comparison | model_or_pair | diagnostic_budget | statistic | estimate | lower_95 | upper_95 | bootstrap_repetitions | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paired_cross_model | groundingdino-owlv2 | 40 | diagnostic_difference | 0.1016 | 0.0814 | 0.1214 | 2000 | 1000 |
| paired_cross_model | groundingdino-owlv2 | 40 | reference_difference | 0.1024 | 0.0827 | 0.1215 | 2000 | 1000 |
| paired_cross_model | groundingdino-yoloworld | 40 | diagnostic_difference | 0.3314 | 0.3070 | 0.3576 | 2000 | 1000 |
| paired_cross_model | groundingdino-yoloworld | 40 | reference_difference | 0.3340 | 0.3098 | 0.3607 | 2000 | 1000 |
| paired_cross_model | owlv2-yoloworld | 40 | diagnostic_difference | 0.2298 | 0.2037 | 0.2579 | 2000 | 1000 |
| paired_cross_model | owlv2-yoloworld | 40 | reference_difference | 0.2317 | 0.2056 | 0.2598 | 2000 | 1000 |
| single_model | groundingdino | 40 | diagnostic_mean | 0.8698 | 0.8573 | 0.8816 | 2000 | 1000 |
| single_model | groundingdino | 40 | reference_mean | 0.8702 | 0.8581 | 0.8817 | 2000 | 1000 |
| single_model | owlv2 | 40 | diagnostic_mean | 0.7683 | 0.7523 | 0.7837 | 2000 | 1000 |
| single_model | owlv2 | 40 | reference_mean | 0.7679 | 0.7524 | 0.7832 | 2000 | 1000 |
| single_model | yoloworld | 40 | diagnostic_mean | 0.5385 | 0.5145 | 0.5607 | 2000 | 1000 |
| single_model | yoloworld | 40 | reference_mean | 0.5362 | 0.5128 | 0.5584 | 2000 | 1000 |

## Interpretation rules

- Full-manifest stability assigns zero to clean outputs that do not expose two distinct candidates.
- Eligible-only stability is diagnostic and is never substituted for the primary full-manifest estimate.
- Family risk shares localise *where* instability is observed under the registered probe distribution; they are not causal effects.
- The 80-probe estimate is an independent finite reference, not an unknowable exact population probability.
- Clopper-Pearson intervals quantify finite-probe uncertainty for each Bernoulli estimand.

## Reproducibility artifacts

The result directory contains the frozen configuration, manifest hash, complete compressed probe traces, row-level summaries, statistical tables, figures, and this report.
