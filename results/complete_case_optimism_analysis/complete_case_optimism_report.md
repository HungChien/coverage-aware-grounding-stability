# Complete-Case Persistence Optimism: Cross-Model and Cross-Dataset Analysis

## Executive result

Across RefCOCO, RefCOCO+, and Ref-L4, complete-case persistence is an optimistic estimand whenever clean candidate eligibility or perturbed-candidate coverage is imperfect. The effect is algebraically necessary and empirically non-negligible. It is small-to-moderate for GroundingDINO and large for YOLO-World because the latter loses candidate eligibility and coverage much more often.

The analysis uses every completed trace from both architectures: 2,500 unique image-query pairs, 5,000 model-sample records, and 352,080 eligible reference-probe outcomes, together with the corresponding diagnostic probes at budgets 5, 10, 20, and 40.

## Formal estimands

Let `Gamma` be clean eligibility, `theta_cov` candidate coverage, and `theta_cc` complete-case persistence. Full-manifest operational stability is

`Theta_op = Gamma * theta_cov * theta_cc`.

The exact optimistic overstatement made by complete-case persistence is

`D_total = theta_cc - Theta_op = theta_cc * (1 - Gamma * theta_cov)`.

It separates into two non-negative terms:

`D_coverage = theta_cc * (1 - theta_cov)`

and

`D_eligibility = (1 - Gamma) * theta_cov * theta_cc`.

Consequently, more probes cannot make complete-case persistence converge to operational stability unless both eligibility and coverage equal one. More data only estimates the conditional estimand more precisely.

## Primary cross-dataset results

| dataset | model | sample_count | eligible_count | clean_eligibility | coverage | complete_case_persistence | eligible_operational_stability | full_manifest_operational_stability | coverage_optimism | eligibility_optimism | total_optimism |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 500 | 499 | 0.9980 | 0.9449 | 0.9070 | 0.8571 | 0.8554 | 0.0499 | 0.0017 | 0.0517 |
| RefCOCO | YOLO-World | 500 | 364 | 0.7280 | 0.8071 | 0.8710 | 0.7030 | 0.5118 | 0.1680 | 0.1912 | 0.3592 |
| RefCOCO+ | GroundingDINO | 1000 | 1000 | 1.0000 | 0.9374 | 0.9080 | 0.8512 | 0.8512 | 0.0568 | 0.0000 | 0.0568 |
| RefCOCO+ | YOLO-World | 1000 | 757 | 0.7570 | 0.7604 | 0.8717 | 0.6628 | 0.5018 | 0.2088 | 0.1611 | 0.3699 |
| Ref-L4 | GroundingDINO | 1000 | 1000 | 1.0000 | 0.9776 | 0.8902 | 0.8703 | 0.8703 | 0.0199 | 0.0000 | 0.0199 |
| Ref-L4 | YOLO-World | 1000 | 781 | 0.7810 | 0.7786 | 0.8820 | 0.6867 | 0.5363 | 0.1953 | 0.1504 | 0.3457 |

## Hierarchical bootstrap intervals for total optimism

| dataset | model | point_estimate | lower_95 | upper_95 | bootstrap_repetitions | sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 0.0517 | 0.0412 | 0.0623 | 2000 | 500 |
| RefCOCO | YOLO-World | 0.3592 | 0.3273 | 0.3904 | 2000 | 500 |
| RefCOCO+ | GroundingDINO | 0.0568 | 0.0504 | 0.0636 | 2000 | 1000 |
| RefCOCO+ | YOLO-World | 0.3699 | 0.3486 | 0.3918 | 2000 | 1000 |
| Ref-L4 | GroundingDINO | 0.0199 | 0.0166 | 0.0235 | 2000 | 1000 |
| Ref-L4 | YOLO-World | 0.3457 | 0.3235 | 0.3680 | 2000 | 1000 |

The bootstrap resamples image-query pairs and then resamples coverage and ranking outcomes within each selected pair. Intervals therefore include both finite-image and finite-reference-probe uncertainty.

## Fair predictive comparison on identical complete cases

| dataset | model | predictor | predictor_availability | evaluated_samples | bias | mae | rmse | reference_probe_brier | spearman | tie_aware_aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | complete_case_persistence | 0.9980 | 499 | 0.0513 | 0.0635 | 0.1361 | 0.1044 | 0.7161 | 0.0726 |
| RefCOCO | GroundingDINO | coverage_aware_operational | 0.9980 | 499 | 0.0012 | 0.0253 | 0.0433 | 0.0877 | 0.9551 | 0.0306 |
| RefCOCO | YOLO-World | complete_case_persistence | 0.7280 | 364 | 0.1715 | 0.1848 | 0.2771 | 0.2263 | 0.4456 | 0.2369 |
| RefCOCO | YOLO-World | coverage_aware_operational | 0.7280 | 364 | -0.0026 | 0.0442 | 0.0605 | 0.1532 | 0.9695 | 0.1098 |
| RefCOCO+ | GroundingDINO | complete_case_persistence | 1.0000 | 1000 | 0.0585 | 0.0689 | 0.1312 | 0.1074 | 0.6904 | 0.0780 |
| RefCOCO+ | GroundingDINO | coverage_aware_operational | 1.0000 | 1000 | 0.0009 | 0.0276 | 0.0436 | 0.0921 | 0.9587 | 0.0343 |
| RefCOCO+ | YOLO-World | complete_case_persistence | 0.7570 | 757 | 0.2175 | 0.2250 | 0.3230 | 0.2618 | 0.3775 | 0.2835 |
| RefCOCO+ | YOLO-World | coverage_aware_operational | 0.7570 | 757 | 0.0013 | 0.0410 | 0.0555 | 0.1605 | 0.9764 | 0.1322 |
| Ref-L4 | GroundingDINO | complete_case_persistence | 1.0000 | 1000 | 0.0187 | 0.0357 | 0.0714 | 0.0826 | 0.8788 | 0.0355 |
| Ref-L4 | GroundingDINO | coverage_aware_operational | 1.0000 | 1000 | -0.0005 | 0.0256 | 0.0437 | 0.0794 | 0.9549 | 0.0252 |
| Ref-L4 | YOLO-World | complete_case_persistence | 0.7810 | 781 | 0.1998 | 0.2052 | 0.2950 | 0.2406 | 0.4374 | 0.2523 |
| Ref-L4 | YOLO-World | coverage_aware_operational | 0.7810 | 781 | 0.0029 | 0.0408 | 0.0567 | 0.1568 | 0.9754 | 0.1198 |

Both predictors are evaluated on exactly the samples for which complete-case persistence exists. The target is independent 80-probe operational stability. The full-manifest coverage-aware results are additionally saved in the CSV outputs; complete-case persistence is left missing outside its observable cohort rather than being assigned an invented fallback score.

## Paired bootstrap advantage at 40 probes

| dataset | model | statistic | estimate | lower_95 | upper_95 | paired_sample_count |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | mae_advantage | 0.0383 | 0.0292 | 0.0481 | 499 |
| RefCOCO | GroundingDINO | brier_advantage | 0.0167 | 0.0110 | 0.0233 | 499 |
| RefCOCO | GroundingDINO | absolute_bias_advantage | 0.0497 | 0.0390 | 0.0606 | 499 |
| RefCOCO | YOLO-World | mae_advantage | 0.1408 | 0.1207 | 0.1613 | 364 |
| RefCOCO | YOLO-World | brier_advantage | 0.0731 | 0.0596 | 0.0869 | 364 |
| RefCOCO | YOLO-World | absolute_bias_advantage | 0.1682 | 0.1444 | 0.1914 | 364 |
| RefCOCO+ | GroundingDINO | mae_advantage | 0.0414 | 0.0353 | 0.0479 | 1000 |
| RefCOCO+ | GroundingDINO | brier_advantage | 0.0154 | 0.0122 | 0.0188 | 1000 |
| RefCOCO+ | GroundingDINO | absolute_bias_advantage | 0.0573 | 0.0502 | 0.0645 | 1000 |
| RefCOCO+ | YOLO-World | mae_advantage | 0.1839 | 0.1681 | 0.1998 | 757 |
| RefCOCO+ | YOLO-World | brier_advantage | 0.1011 | 0.0900 | 0.1123 | 757 |
| RefCOCO+ | YOLO-World | absolute_bias_advantage | 0.2154 | 0.1983 | 0.2321 | 757 |
| Ref-L4 | GroundingDINO | mae_advantage | 0.0101 | 0.0071 | 0.0133 | 1000 |
| Ref-L4 | GroundingDINO | brier_advantage | 0.0032 | 0.0019 | 0.0050 | 1000 |
| Ref-L4 | GroundingDINO | absolute_bias_advantage | 0.0175 | 0.0122 | 0.0222 | 1000 |
| Ref-L4 | YOLO-World | mae_advantage | 0.1644 | 0.1500 | 0.1790 | 781 |
| Ref-L4 | YOLO-World | brier_advantage | 0.0837 | 0.0741 | 0.0939 | 781 |
| Ref-L4 | YOLO-World | absolute_bias_advantage | 0.1968 | 0.1813 | 0.2130 | 781 |

Positive values favour the coverage-aware estimator because each statistic is defined as complete-case error minus coverage-aware error.

## Probe-budget analysis

| dataset | model | probe_budget | complete_case_persistence | full_manifest_operational_stability | total_optimism |
| --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 5 | 0.9085 | 0.8616 | 0.0469 |
| RefCOCO | GroundingDINO | 10 | 0.9068 | 0.8558 | 0.0510 |
| RefCOCO | GroundingDINO | 20 | 0.9091 | 0.8576 | 0.0515 |
| RefCOCO | GroundingDINO | 40 | 0.9078 | 0.8566 | 0.0512 |
| RefCOCO | YOLO-World | 5 | 0.8709 | 0.5072 | 0.3637 |
| RefCOCO | YOLO-World | 10 | 0.8676 | 0.5096 | 0.3580 |
| RefCOCO | YOLO-World | 20 | 0.8706 | 0.5113 | 0.3593 |
| RefCOCO | YOLO-World | 40 | 0.8703 | 0.5099 | 0.3604 |
| RefCOCO+ | GroundingDINO | 5 | 0.9116 | 0.8542 | 0.0574 |
| RefCOCO+ | GroundingDINO | 10 | 0.9103 | 0.8540 | 0.0563 |
| RefCOCO+ | GroundingDINO | 20 | 0.9093 | 0.8528 | 0.0565 |
| RefCOCO+ | GroundingDINO | 40 | 0.9091 | 0.8520 | 0.0571 |
| RefCOCO+ | YOLO-World | 5 | 0.8710 | 0.5012 | 0.3698 |
| RefCOCO+ | YOLO-World | 10 | 0.8730 | 0.5016 | 0.3714 |
| RefCOCO+ | YOLO-World | 20 | 0.8742 | 0.5022 | 0.3720 |
| RefCOCO+ | YOLO-World | 40 | 0.8736 | 0.5028 | 0.3709 |
| Ref-L4 | GroundingDINO | 5 | 0.8876 | 0.8654 | 0.0222 |
| Ref-L4 | GroundingDINO | 10 | 0.8917 | 0.8700 | 0.0217 |
| Ref-L4 | GroundingDINO | 20 | 0.8916 | 0.8708 | 0.0208 |
| Ref-L4 | GroundingDINO | 40 | 0.8904 | 0.8699 | 0.0206 |
| Ref-L4 | YOLO-World | 5 | 0.8803 | 0.5352 | 0.3451 |
| Ref-L4 | YOLO-World | 10 | 0.8818 | 0.5348 | 0.3470 |
| Ref-L4 | YOLO-World | 20 | 0.8857 | 0.5390 | 0.3468 |
| Ref-L4 | YOLO-World | 40 | 0.8848 | 0.5386 | 0.3462 |

## Perturbation-family analysis

| dataset | model | family | coverage | complete_case_persistence | full_manifest_operational_stability | total_optimism |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | blur | 0.9089 | 0.8554 | 0.7760 | 0.0794 |
| RefCOCO | GroundingDINO | brightness | 0.9826 | 0.9702 | 0.9514 | 0.0188 |
| RefCOCO | GroundingDINO | gaussian_noise | 0.9102 | 0.8822 | 0.8014 | 0.0808 |
| RefCOCO | GroundingDINO | resolution | 0.9518 | 0.8742 | 0.8304 | 0.0438 |
| RefCOCO | GroundingDINO | jpeg | 0.9712 | 0.9469 | 0.9177 | 0.0291 |
| RefCOCO | YOLO-World | blur | 0.6696 | 0.7721 | 0.3764 | 0.3957 |
| RefCOCO | YOLO-World | brightness | 0.9425 | 0.9599 | 0.6586 | 0.3013 |
| RefCOCO | YOLO-World | gaussian_noise | 0.8632 | 0.8816 | 0.5540 | 0.3276 |
| RefCOCO | YOLO-World | resolution | 0.7163 | 0.8020 | 0.4183 | 0.3838 |
| RefCOCO | YOLO-World | jpeg | 0.8439 | 0.8979 | 0.5516 | 0.3462 |
| RefCOCO+ | GroundingDINO | blur | 0.8944 | 0.8589 | 0.7682 | 0.0907 |
| RefCOCO+ | GroundingDINO | brightness | 0.9899 | 0.9703 | 0.9605 | 0.0098 |
| RefCOCO+ | GroundingDINO | gaussian_noise | 0.9751 | 0.9370 | 0.9136 | 0.0234 |
| RefCOCO+ | GroundingDINO | resolution | 0.9439 | 0.8868 | 0.8371 | 0.0498 |
| RefCOCO+ | GroundingDINO | jpeg | 0.8838 | 0.8785 | 0.7764 | 0.1021 |
| RefCOCO+ | YOLO-World | blur | 0.5913 | 0.7816 | 0.3499 | 0.4318 |
| RefCOCO+ | YOLO-World | brightness | 0.9451 | 0.9617 | 0.6881 | 0.2737 |
| RefCOCO+ | YOLO-World | gaussian_noise | 0.8736 | 0.9021 | 0.5966 | 0.3055 |
| RefCOCO+ | YOLO-World | resolution | 0.6532 | 0.8058 | 0.3984 | 0.4074 |
| RefCOCO+ | YOLO-World | jpeg | 0.7390 | 0.8507 | 0.4759 | 0.3748 |
| Ref-L4 | GroundingDINO | blur | 0.9361 | 0.8453 | 0.7913 | 0.0540 |
| Ref-L4 | GroundingDINO | brightness | 0.9959 | 0.9535 | 0.9496 | 0.0039 |
| Ref-L4 | GroundingDINO | gaussian_noise | 0.9916 | 0.9183 | 0.9106 | 0.0077 |
| Ref-L4 | GroundingDINO | resolution | 0.9876 | 0.8723 | 0.8614 | 0.0108 |
| Ref-L4 | GroundingDINO | jpeg | 0.9769 | 0.8585 | 0.8387 | 0.0198 |
| Ref-L4 | YOLO-World | blur | 0.5980 | 0.8184 | 0.3822 | 0.4362 |
| Ref-L4 | YOLO-World | brightness | 0.9461 | 0.9579 | 0.7077 | 0.2501 |
| Ref-L4 | YOLO-World | gaussian_noise | 0.8680 | 0.8887 | 0.6025 | 0.2862 |
| Ref-L4 | YOLO-World | resolution | 0.6954 | 0.8359 | 0.4540 | 0.3819 |
| Ref-L4 | YOLO-World | jpeg | 0.7854 | 0.8724 | 0.5351 | 0.3373 |

Family-level gaps demonstrate whether complete-case conditioning hides the same amount of instability under different input degradations. These are descriptive properties under the registered probe mixture, not internal causal attributions.

## Correctness strata

| dataset | model | clean_correct | sample_count | clean_eligibility | complete_case_persistence | full_manifest_operational_stability | total_optimism |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 0 | 250 | 0.9960 | 0.8738 | 0.8280 | 0.0457 |
| RefCOCO | GroundingDINO | 1 | 250 | 1.0000 | 0.9406 | 0.8827 | 0.0579 |
| RefCOCO | YOLO-World | 0 | 347 | 0.6081 | 0.8493 | 0.4070 | 0.4424 |
| RefCOCO | YOLO-World | 1 | 153 | 1.0000 | 0.8992 | 0.7495 | 0.1497 |
| RefCOCO+ | GroundingDINO | 0 | 505 | 1.0000 | 0.8789 | 0.8345 | 0.0445 |
| RefCOCO+ | GroundingDINO | 1 | 495 | 1.0000 | 0.9384 | 0.8682 | 0.0702 |
| RefCOCO+ | YOLO-World | 0 | 682 | 0.6437 | 0.8664 | 0.4193 | 0.4471 |
| RefCOCO+ | YOLO-World | 1 | 318 | 1.0000 | 0.8788 | 0.6787 | 0.2001 |
| Ref-L4 | GroundingDINO | 0 | 613 | 1.0000 | 0.8647 | 0.8449 | 0.0198 |
| Ref-L4 | GroundingDINO | 1 | 387 | 1.0000 | 0.9306 | 0.9106 | 0.0200 |
| Ref-L4 | YOLO-World | 0 | 702 | 0.6880 | 0.8642 | 0.4649 | 0.3993 |
| Ref-L4 | YOLO-World | 1 | 298 | 1.0000 | 0.9112 | 0.7047 | 0.2065 |

Correctness is used only as an external audit stratum. Neither persistence nor operational stability is interpreted as semantic correctness.

## Output-contract sensitivity

| dataset | model | settings | min_complete_case_persistence | max_complete_case_persistence | min_full_manifest_operational | max_full_manifest_operational | min_total_optimism | max_total_optimism |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 27 | 0.9043 | 0.9310 | 0.8505 | 0.9011 | 0.0197 | 0.0568 |
| RefCOCO | YOLO-World | 27 | 0.8683 | 0.9003 | 0.5090 | 0.5845 | 0.3091 | 0.3620 |
| RefCOCO+ | GroundingDINO | 27 | 0.9047 | 0.9349 | 0.8425 | 0.9012 | 0.0212 | 0.0660 |
| RefCOCO+ | YOLO-World | 27 | 0.8675 | 0.8979 | 0.4993 | 0.5981 | 0.2958 | 0.3731 |
| Ref-L4 | GroundingDINO | 27 | 0.8885 | 0.9268 | 0.8676 | 0.8813 | 0.0161 | 0.0478 |
| Ref-L4 | YOLO-World | 27 | 0.8760 | 0.9100 | 0.5305 | 0.6365 | 0.2687 | 0.3530 |

The primary contract was frozen before inference. This post-primary analysis asks whether the optimism conclusion survives reasonable candidate-count and association threshold changes. Absolute magnitudes remain contract-defined.

## Registered Ref-L4 strata

| model | dimension | level | sample_count | clean_eligibility | coverage | conditional_ranking | full_manifest_operational | total_optimism |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GroundingDINO | source_split | coco | 600 | 1.0000 | 0.9784 | 0.8719 | 0.8530 | 0.0189 |
| GroundingDINO | source_split | objects365 | 400 | 1.0000 | 0.9765 | 0.9178 | 0.8962 | 0.0215 |
| GroundingDINO | query_length_stratum | long_ge30 | 218 | 1.0000 | 0.9815 | 0.8893 | 0.8728 | 0.0165 |
| GroundingDINO | query_length_stratum | medium_19_29 | 504 | 1.0000 | 0.9755 | 0.8852 | 0.8635 | 0.0216 |
| GroundingDINO | query_length_stratum | short_le18 | 278 | 1.0000 | 0.9784 | 0.9001 | 0.8806 | 0.0195 |
| GroundingDINO | target_scale_stratum | large_gt96 | 894 | 1.0000 | 0.9779 | 0.8853 | 0.8658 | 0.0195 |
| GroundingDINO | target_scale_stratum | medium_32_96 | 105 | 1.0000 | 0.9748 | 0.9314 | 0.9079 | 0.0235 |
| GroundingDINO | target_scale_stratum | small_lt32 | 1 | 1.0000 | 1.0000 | 0.9875 | 0.9875 | 0.0000 |
| YOLO-World | source_split | coco | 600 | 0.7883 | 0.7693 | 0.8719 | 0.5288 | 0.3431 |
| YOLO-World | source_split | objects365 | 400 | 0.7700 | 0.7928 | 0.8971 | 0.5476 | 0.3494 |
| YOLO-World | query_length_stratum | long_ge30 | 218 | 0.7294 | 0.7675 | 0.8813 | 0.4933 | 0.3880 |
| YOLO-World | query_length_stratum | medium_19_29 | 504 | 0.7937 | 0.7694 | 0.8764 | 0.5352 | 0.3413 |
| YOLO-World | query_length_stratum | short_le18 | 278 | 0.7986 | 0.8032 | 0.8921 | 0.5722 | 0.3199 |
| YOLO-World | target_scale_stratum | large_gt96 | 894 | 0.7852 | 0.7781 | 0.8814 | 0.5385 | 0.3429 |
| YOLO-World | target_scale_stratum | medium_32_96 | 105 | 0.7429 | 0.7803 | 0.8856 | 0.5133 | 0.3723 |
| YOLO-World | target_scale_stratum | small_lt32 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Claim supported by the evidence

Complete-case persistence is not a noisy version of operational stability. It is a different conditional estimand with a non-negative optimism gap that is exactly determined by clean eligibility and candidate coverage. Existing traces show that the gap persists across datasets, budgets, perturbation families, and output contracts, while its magnitude is architecture dependent.

## Interpretation boundary

- The theorem concerns candidate-order operational stability, not semantic correctness.
- The 80-probe reference is finite, so bootstrap intervals quantify rather than erase its uncertainty.
- Cross-dataset replication supports transfer of the finding, not a universal claim over every grounding architecture.
- Probe-family localisation is descriptive under the registered distribution and is not a causal neural-module diagnosis.
- Full-manifest comparison is primary; eligible-only and complete-case quantities remain diagnostic.
