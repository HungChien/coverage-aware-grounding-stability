# Adequacy of the Frozen 80-Probe Reference

## Decision

The 80-probe reference satisfies both frozen model-level adequacy criteria in 9 of 9 dataset-model groups.

It is therefore adequate for estimating and comparing model-level operational stability under the frozen probe distribution. At sample level it should remain described as a finite, noisy reference rather than exact ground truth.

## Frozen model-level criteria

1. Finite probes must contribute no more than 5% of model-level variance at R = 80.
2. The 95th percentile absolute difference between family-balanced 40/40 half-reference model means must be no larger than 0.01.

| dataset | model | probe_variance_share_R80 | required_R_for_5pct_probe_share | split_half_p95_absolute_model_difference | relative_se_excess_over_infinite_R | model_level_adequate |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 0.02844 | 45 | 0.00445 | 0.01453 | True |
| RefCOCO | OWLv2 | 0.02403 | 38 | 0.00585 | 0.01224 | True |
| RefCOCO | YOLO-World | 0.00976 | 15 | 0.00505 | 0.00491 | True |
| RefCOCO+ | GroundingDINO | 0.03124 | 50 | 0.00318 | 0.01600 | True |
| RefCOCO+ | OWLv2 | 0.02133 | 34 | 0.00407 | 0.01084 | True |
| RefCOCO+ | YOLO-World | 0.01152 | 18 | 0.00365 | 0.00581 | True |
| Ref-L4 | GroundingDINO | 0.02766 | 44 | 0.00317 | 0.01412 | True |
| Ref-L4 | OWLv2 | 0.02438 | 38 | 0.00398 | 0.01242 | True |
| Ref-L4 | YOLO-World | 0.01178 | 19 | 0.00365 | 0.00594 | True |

## Sample-level precision

| dataset | model | clean_eligibility_rate | average_eligible_sample_probe_rmse_R80 | sample_rmse_criterion_pass |
| --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 0.99800 | 0.03296 | True |
| RefCOCO | OWLv2 | 0.97600 | 0.04047 | True |
| RefCOCO | YOLO-World | 0.72800 | 0.04351 | True |
| RefCOCO+ | GroundingDINO | 1.00000 | 0.03379 | True |
| RefCOCO+ | OWLv2 | 0.96800 | 0.04007 | True |
| RefCOCO+ | YOLO-World | 0.75700 | 0.04464 | True |
| Ref-L4 | GroundingDINO | 1.00000 | 0.03131 | True |
| Ref-L4 | OWLv2 | 0.99800 | 0.03859 | True |
| Ref-L4 | YOLO-World | 0.78100 | 0.04409 | True |

| dataset | model | eligible_sample_count | mean_interval_width | median_interval_width | p90_interval_width | p95_interval_width | maximum_interval_width | fraction_width_le_0_10 | fraction_width_le_0_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 499 | 0.11896 | 0.09790 | 0.22246 | 0.22624 | 0.22790 | 0.50701 | 0.77756 |
| RefCOCO | OWLv2 | 488 | 0.15407 | 0.17238 | 0.22684 | 0.22764 | 0.22790 | 0.27049 | 0.64344 |
| RefCOCO | YOLO-World | 364 | 0.16636 | 0.19632 | 0.22754 | 0.22784 | 0.22790 | 0.22802 | 0.52747 |
| RefCOCO+ | GroundingDINO | 1000 | 0.12340 | 0.11925 | 0.22116 | 0.22624 | 0.22790 | 0.45300 | 0.78100 |
| RefCOCO+ | OWLv2 | 968 | 0.15324 | 0.16738 | 0.22624 | 0.22731 | 0.22790 | 0.26446 | 0.66632 |
| RefCOCO+ | YOLO-World | 757 | 0.17167 | 0.20521 | 0.22731 | 0.22764 | 0.22790 | 0.19287 | 0.47556 |
| Ref-L4 | GroundingDINO | 1000 | 0.11304 | 0.08437 | 0.21972 | 0.22624 | 0.22790 | 0.53600 | 0.80700 |
| Ref-L4 | OWLv2 | 998 | 0.14622 | 0.16203 | 0.22463 | 0.22731 | 0.22790 | 0.31563 | 0.69840 |
| Ref-L4 | YOLO-World | 781 | 0.16952 | 0.19948 | 0.22684 | 0.22764 | 0.22790 | 0.20230 | 0.50704 |

The Clopper--Pearson intervals, under the iid-Q Bernoulli working model, demonstrate why an 80-probe sample value must not be called exact. Precision depends strongly on whether the latent probability is near zero, one, or one half.

## Family-balanced 40/40 split-half agreement on eligible samples

| dataset | model | metric | mean | median | lower_2_5 | upper_97_5 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | sample_mae | 0.03008 | 0.03011 | 0.02730 | 0.03272 | 0.03231 |
| RefCOCO | GroundingDINO | sample_rmse | 0.05105 | 0.05105 | 0.04648 | 0.05570 | 0.05493 |
| RefCOCO | GroundingDINO | sample_spearman | 0.94704 | 0.94723 | 0.93808 | 0.95492 | 0.95363 |
| RefCOCO | OWLv2 | sample_mae | 0.04780 | 0.04780 | 0.04421 | 0.05128 | 0.05077 |
| RefCOCO | OWLv2 | sample_rmse | 0.06875 | 0.06875 | 0.06381 | 0.07417 | 0.07309 |
| RefCOCO | OWLv2 | sample_spearman | 0.95372 | 0.95378 | 0.94702 | 0.95988 | 0.95907 |
| RefCOCO | YOLO-World | sample_mae | 0.04689 | 0.04691 | 0.04306 | 0.05117 | 0.05034 |
| RefCOCO | YOLO-World | sample_rmse | 0.06625 | 0.06629 | 0.06071 | 0.07195 | 0.07104 |
| RefCOCO | YOLO-World | sample_spearman | 0.96489 | 0.96495 | 0.95908 | 0.97025 | 0.96934 |
| RefCOCO+ | GroundingDINO | sample_mae | 0.03200 | 0.03198 | 0.03012 | 0.03403 | 0.03372 |
| RefCOCO+ | GroundingDINO | sample_rmse | 0.05239 | 0.05236 | 0.04921 | 0.05583 | 0.05527 |
| RefCOCO+ | GroundingDINO | sample_spearman | 0.95040 | 0.95048 | 0.94452 | 0.95596 | 0.95502 |
| RefCOCO+ | OWLv2 | sample_mae | 0.04734 | 0.04734 | 0.04476 | 0.04987 | 0.04951 |
| RefCOCO+ | OWLv2 | sample_rmse | 0.06725 | 0.06723 | 0.06372 | 0.07082 | 0.07029 |
| RefCOCO+ | OWLv2 | sample_spearman | 0.95120 | 0.95116 | 0.94619 | 0.95621 | 0.95549 |
| RefCOCO+ | YOLO-World | sample_mae | 0.04723 | 0.04723 | 0.04432 | 0.05000 | 0.04967 |
| RefCOCO+ | YOLO-World | sample_rmse | 0.06623 | 0.06624 | 0.06244 | 0.07007 | 0.06943 |
| RefCOCO+ | YOLO-World | sample_spearman | 0.96801 | 0.96798 | 0.96451 | 0.97172 | 0.97112 |
| Ref-L4 | GroundingDINO | sample_mae | 0.02908 | 0.02908 | 0.02718 | 0.03093 | 0.03057 |
| Ref-L4 | GroundingDINO | sample_rmse | 0.05031 | 0.05029 | 0.04714 | 0.05359 | 0.05300 |
| Ref-L4 | GroundingDINO | sample_spearman | 0.94010 | 0.94016 | 0.93285 | 0.94689 | 0.94580 |
| Ref-L4 | OWLv2 | sample_mae | 0.04412 | 0.04406 | 0.04181 | 0.04662 | 0.04622 |
| Ref-L4 | OWLv2 | sample_rmse | 0.06409 | 0.06405 | 0.06053 | 0.06765 | 0.06703 |
| Ref-L4 | OWLv2 | sample_spearman | 0.95100 | 0.95110 | 0.94598 | 0.95563 | 0.95491 |
| Ref-L4 | YOLO-World | sample_mae | 0.04786 | 0.04789 | 0.04507 | 0.05083 | 0.05038 |
| Ref-L4 | YOLO-World | sample_rmse | 0.06727 | 0.06725 | 0.06361 | 0.07134 | 0.07062 |
| Ref-L4 | YOLO-World | sample_spearman | 0.96674 | 0.96680 | 0.96289 | 0.97011 | 0.96954 |

## Independent diagnostic-40 versus reference-80 agreement

| dataset | model | absolute_model_mean_difference | sample_mae | sample_rmse | predicted_rmse_under_common_Q | observed_to_predicted_rmse_ratio | sample_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 0.00123 | 0.02533 | 0.04330 | 0.05709 | 0.75850 | 0.95512 |
| RefCOCO | OWLv2 | 0.00095 | 0.04342 | 0.06085 | 0.07009 | 0.86808 | 0.96009 |
| RefCOCO | YOLO-World | 0.00264 | 0.04420 | 0.06047 | 0.07536 | 0.80242 | 0.96950 |
| RefCOCO+ | GroundingDINO | 0.00089 | 0.02756 | 0.04359 | 0.05852 | 0.74483 | 0.95869 |
| RefCOCO+ | OWLv2 | 0.00156 | 0.04087 | 0.05834 | 0.06941 | 0.84051 | 0.96337 |
| RefCOCO+ | YOLO-World | 0.00129 | 0.04095 | 0.05553 | 0.07732 | 0.71814 | 0.97644 |
| Ref-L4 | GroundingDINO | 0.00046 | 0.02564 | 0.04369 | 0.05423 | 0.80555 | 0.95488 |
| Ref-L4 | OWLv2 | 0.00049 | 0.03994 | 0.05740 | 0.06685 | 0.85874 | 0.96099 |
| Ref-L4 | YOLO-World | 0.00294 | 0.04081 | 0.05672 | 0.07637 | 0.74268 | 0.97542 |

If the observed-to-predicted RMSE ratio is close to one, disagreement is consistent with finite probe noise under a common probe law. Larger ratios indicate additional registry composition or probe-severity differences.

## Conclusion

The reference depth is sufficient for model-level conclusions because almost all remaining uncertainty is between image-query pairs, not within the 80 probes. Doubling the probe budget would therefore produce little model-level precision gain compared with sampling more independent pairs.

For per-sample prediction, 80 probes provide a useful but noisy continuous target. Reported analyses must retain finite-reference uncertainty and must not describe the 80-probe value as the latent probability itself.

## Boundary of the claim

This result validates Monte Carlo depth under the frozen empirical probe registry. It does not establish that the five registered perturbation families cover every possible real-world distribution shift.
