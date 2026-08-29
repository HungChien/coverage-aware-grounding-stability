# Adequacy of the Frozen 80-Probe Reference

## Decision

The 80-probe reference satisfies both frozen model-level adequacy criteria in 6 of 6 dataset-model groups.

It is therefore adequate for estimating and comparing model-level operational stability under the frozen probe distribution. At sample level it should remain described as a finite, noisy reference rather than exact ground truth.

## Frozen model-level criteria

1. Finite probes must contribute no more than 5% of model-level variance at R = 80.
2. The 95th percentile absolute difference between family-balanced 40/40 half-reference model means must be no larger than 0.01.

| dataset | model | probe_variance_share_R80 | required_R_for_5pct_probe_share | split_half_p95_absolute_model_difference | relative_se_excess_over_infinite_R | model_level_adequate |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 0.02844 | 45 | 0.00445 | 0.01453 | True |
| RefCOCO | YOLO-World | 0.00976 | 15 | 0.00505 | 0.00491 | True |
| RefCOCO+ | GroundingDINO | 0.03124 | 50 | 0.00318 | 0.01600 | True |
| RefCOCO+ | YOLO-World | 0.01152 | 18 | 0.00360 | 0.00581 | True |
| Ref-L4 | GroundingDINO | 0.02766 | 44 | 0.00317 | 0.01412 | True |
| Ref-L4 | YOLO-World | 0.01178 | 19 | 0.00380 | 0.00594 | True |

## Sample-level precision

| dataset | model | clean_eligibility_rate | average_eligible_sample_probe_rmse_R80 | sample_rmse_criterion_pass |
| --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 0.99800 | 0.03296 | True |
| RefCOCO | YOLO-World | 0.72800 | 0.04351 | True |
| RefCOCO+ | GroundingDINO | 1.00000 | 0.03379 | True |
| RefCOCO+ | YOLO-World | 0.75700 | 0.04464 | True |
| Ref-L4 | GroundingDINO | 1.00000 | 0.03131 | True |
| Ref-L4 | YOLO-World | 0.78100 | 0.04409 | True |

| dataset | model | eligible_sample_count | mean_interval_width | median_interval_width | p90_interval_width | p95_interval_width | maximum_interval_width | fraction_width_le_0_10 | fraction_width_le_0_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 499 | 0.11896 | 0.09790 | 0.22246 | 0.22624 | 0.22790 | 0.50701 | 0.77756 |
| RefCOCO | YOLO-World | 364 | 0.16636 | 0.19632 | 0.22754 | 0.22784 | 0.22790 | 0.22802 | 0.52747 |
| RefCOCO+ | GroundingDINO | 1000 | 0.12340 | 0.11925 | 0.22116 | 0.22624 | 0.22790 | 0.45300 | 0.78100 |
| RefCOCO+ | YOLO-World | 757 | 0.17167 | 0.20521 | 0.22731 | 0.22764 | 0.22790 | 0.19287 | 0.47556 |
| Ref-L4 | GroundingDINO | 1000 | 0.11304 | 0.08437 | 0.21972 | 0.22624 | 0.22790 | 0.53600 | 0.80700 |
| Ref-L4 | YOLO-World | 781 | 0.16952 | 0.19948 | 0.22684 | 0.22764 | 0.22790 | 0.20230 | 0.50704 |

The Clopper--Pearson intervals, under the iid-Q Bernoulli working model, demonstrate why an 80-probe sample value must not be called exact. Precision depends strongly on whether the latent probability is near zero, one, or one half.

## Family-balanced 40/40 split-half agreement on eligible samples

| dataset | model | metric | mean | median | lower_2_5 | upper_97_5 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | sample_mae | 0.03008 | 0.03011 | 0.02730 | 0.03272 | 0.03231 |
| RefCOCO | GroundingDINO | sample_rmse | 0.05105 | 0.05105 | 0.04648 | 0.05570 | 0.05493 |
| RefCOCO | GroundingDINO | sample_spearman | 0.94704 | 0.94723 | 0.93808 | 0.95492 | 0.95363 |
| RefCOCO | YOLO-World | sample_mae | 0.04678 | 0.04677 | 0.04279 | 0.05103 | 0.05034 |
| RefCOCO | YOLO-World | sample_rmse | 0.06606 | 0.06605 | 0.06060 | 0.07176 | 0.07075 |
| RefCOCO | YOLO-World | sample_spearman | 0.96505 | 0.96510 | 0.95933 | 0.97029 | 0.96952 |
| RefCOCO+ | GroundingDINO | sample_mae | 0.03200 | 0.03198 | 0.03012 | 0.03403 | 0.03372 |
| RefCOCO+ | GroundingDINO | sample_rmse | 0.05239 | 0.05236 | 0.04921 | 0.05583 | 0.05527 |
| RefCOCO+ | GroundingDINO | sample_spearman | 0.95040 | 0.95048 | 0.94452 | 0.95596 | 0.95502 |
| RefCOCO+ | YOLO-World | sample_mae | 0.04719 | 0.04723 | 0.04432 | 0.05007 | 0.04960 |
| RefCOCO+ | YOLO-World | sample_rmse | 0.06617 | 0.06616 | 0.06233 | 0.07000 | 0.06943 |
| RefCOCO+ | YOLO-World | sample_spearman | 0.96804 | 0.96807 | 0.96432 | 0.97159 | 0.97102 |
| Ref-L4 | GroundingDINO | sample_mae | 0.02908 | 0.02908 | 0.02718 | 0.03093 | 0.03057 |
| Ref-L4 | GroundingDINO | sample_rmse | 0.05031 | 0.05029 | 0.04714 | 0.05359 | 0.05300 |
| Ref-L4 | GroundingDINO | sample_spearman | 0.94010 | 0.94016 | 0.93285 | 0.94689 | 0.94580 |
| Ref-L4 | YOLO-World | sample_mae | 0.04786 | 0.04782 | 0.04494 | 0.05083 | 0.05038 |
| Ref-L4 | YOLO-World | sample_rmse | 0.06725 | 0.06724 | 0.06333 | 0.07110 | 0.07057 |
| Ref-L4 | YOLO-World | sample_spearman | 0.96674 | 0.96678 | 0.96295 | 0.97013 | 0.96958 |

## Independent diagnostic-40 versus reference-80 agreement

| dataset | model | absolute_model_mean_difference | sample_mae | sample_rmse | predicted_rmse_under_common_Q | observed_to_predicted_rmse_ratio | sample_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 0.00123 | 0.02533 | 0.04330 | 0.05709 | 0.75850 | 0.95512 |
| RefCOCO | YOLO-World | 0.00264 | 0.04420 | 0.06047 | 0.07536 | 0.80242 | 0.96950 |
| RefCOCO+ | GroundingDINO | 0.00089 | 0.02756 | 0.04359 | 0.05852 | 0.74483 | 0.95869 |
| RefCOCO+ | YOLO-World | 0.00129 | 0.04095 | 0.05553 | 0.07732 | 0.71814 | 0.97644 |
| Ref-L4 | GroundingDINO | 0.00046 | 0.02564 | 0.04369 | 0.05423 | 0.80555 | 0.95488 |
| Ref-L4 | YOLO-World | 0.00294 | 0.04081 | 0.05672 | 0.07637 | 0.74268 | 0.97542 |

If the observed-to-predicted RMSE ratio is close to one, disagreement is consistent with finite probe noise under a common probe law. Larger ratios indicate additional registry composition or probe-severity differences.

## Conclusion

The reference depth is sufficient for model-level conclusions because almost all remaining uncertainty is between image-query pairs, not within the 80 probes. Doubling the probe budget would therefore produce little model-level precision gain compared with sampling more independent pairs.

For per-sample prediction, 80 probes provide a useful but noisy continuous target. Reported analyses must retain finite-reference uncertainty and must not describe the 80-probe value as the latent probability itself.

## Boundary of the claim

This result validates Monte Carlo depth under the frozen empirical probe registry. It does not establish that the five registered perturbation families cover every possible real-world distribution shift.
