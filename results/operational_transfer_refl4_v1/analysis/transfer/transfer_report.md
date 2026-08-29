# Frozen Ref-L4 Transfer Analysis

All candidate, probe, estimator, and analysis definitions were registered before Ref-L4 model inference.

## Source and target estimands

| model | scope | sample_count | eligible_count | clean_eligibility | full_manifest_operational | eligible_operational | coverage | conditional_ranking | conditional_minus_operational | full_manifest_lower_95 | full_manifest_upper_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundingdino | source_refcoco | 500 | 499 | 0.9980 | 0.8554 | 0.8571 | 0.9449 | 0.9070 | 0.0499 | 0.8371 | 0.8727 |
| groundingdino | source_refcocoplus | 1000 | 1000 | 1.0000 | 0.8512 | 0.8512 | 0.9374 | 0.9080 | 0.0568 | 0.8385 | 0.8636 |
| groundingdino | target_refl4_pooled | 1000 | 1000 | 1.0000 | 0.8703 | 0.8703 | 0.9776 | 0.8902 | 0.0199 | 0.8587 | 0.8818 |
| groundingdino | target_refl4_coco | 600 | 600 | 1.0000 | 0.8530 | 0.8530 | 0.9784 | 0.8719 | 0.0189 | 0.8366 | 0.8676 |
| groundingdino | target_refl4_objects365 | 400 | 400 | 1.0000 | 0.8962 | 0.8962 | 0.9765 | 0.9178 | 0.0215 | 0.8796 | 0.9116 |
| yoloworld | source_refcoco | 500 | 364 | 0.7280 | 0.5118 | 0.7030 | 0.8071 | 0.8710 | 0.1680 | 0.4783 | 0.5449 |
| yoloworld | source_refcocoplus | 1000 | 757 | 0.7570 | 0.5018 | 0.6628 | 0.7604 | 0.8717 | 0.2088 | 0.4786 | 0.5239 |
| yoloworld | target_refl4_pooled | 1000 | 781 | 0.7810 | 0.5363 | 0.6867 | 0.7786 | 0.8820 | 0.1953 | 0.5137 | 0.5594 |
| yoloworld | target_refl4_coco | 600 | 473 | 0.7883 | 0.5288 | 0.6708 | 0.7693 | 0.8719 | 0.2011 | 0.4995 | 0.5574 |
| yoloworld | target_refl4_objects365 | 400 | 308 | 0.7700 | 0.5476 | 0.7112 | 0.7928 | 0.8971 | 0.1859 | 0.5116 | 0.5842 |

## Dataset-shift bootstrap

| model | comparison | full_manifest_delta | lower_95 | upper_95 |
| --- | --- | --- | --- | --- |
| groundingdino | Ref-L4 minus RefCOCO | 0.0149 | -0.0071 | 0.0369 |
| groundingdino | Ref-L4 minus RefCOCO+ | 0.0192 | 0.0021 | 0.0352 |
| yoloworld | Ref-L4 minus RefCOCO | 0.0246 | -0.0153 | 0.0641 |
| yoloworld | Ref-L4 minus RefCOCO+ | 0.0346 | 0.0042 | 0.0668 |

## Finite-probe transfer

| model | scope | diagnostic_budget | sample_count | eligible_count | full_manifest_bias | eligible_mae | eligible_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| groundingdino | pooled | 5 | 1000 | 1000 | -0.0049 | 0.0600 | 0.8259 |
| groundingdino | pooled | 10 | 1000 | 1000 | -0.0003 | 0.0441 | 0.8844 |
| groundingdino | pooled | 20 | 1000 | 1000 | 0.0004 | 0.0335 | 0.9233 |
| groundingdino | pooled | 40 | 1000 | 1000 | -0.0005 | 0.0256 | 0.9549 |
| groundingdino | coco | 5 | 600 | 600 | -0.0070 | 0.0656 | 0.8409 |
| groundingdino | coco | 10 | 600 | 600 | 0.0003 | 0.0490 | 0.8949 |
| groundingdino | coco | 20 | 600 | 600 | 0.0012 | 0.0370 | 0.9296 |
| groundingdino | coco | 40 | 600 | 600 | -0.0008 | 0.0285 | 0.9589 |
| groundingdino | objects365 | 5 | 400 | 400 | -0.0017 | 0.0516 | 0.7976 |
| groundingdino | objects365 | 10 | 400 | 400 | -0.0012 | 0.0367 | 0.8650 |
| groundingdino | objects365 | 20 | 400 | 400 | -0.0008 | 0.0284 | 0.9130 |
| groundingdino | objects365 | 40 | 400 | 400 | -0.0000 | 0.0214 | 0.9471 |
| yoloworld | pooled | 5 | 1000 | 781 | -0.0011 | 0.0952 | 0.8875 |
| yoloworld | pooled | 10 | 1000 | 781 | -0.0015 | 0.0718 | 0.9333 |
| yoloworld | pooled | 20 | 1000 | 781 | 0.0026 | 0.0530 | 0.9622 |
| yoloworld | pooled | 40 | 1000 | 781 | 0.0023 | 0.0408 | 0.9754 |
| yoloworld | coco | 5 | 600 | 473 | 0.0022 | 0.0973 | 0.8861 |
| yoloworld | coco | 10 | 600 | 473 | 0.0029 | 0.0724 | 0.9310 |
| yoloworld | coco | 20 | 600 | 473 | 0.0055 | 0.0542 | 0.9602 |
| yoloworld | coco | 40 | 600 | 473 | 0.0045 | 0.0434 | 0.9727 |
| yoloworld | objects365 | 5 | 400 | 308 | -0.0061 | 0.0920 | 0.8865 |
| yoloworld | objects365 | 10 | 400 | 308 | -0.0081 | 0.0710 | 0.9363 |
| yoloworld | objects365 | 20 | 400 | 308 | -0.0018 | 0.0511 | 0.9649 |
| yoloworld | objects365 | 40 | 400 | 308 | -0.0011 | 0.0369 | 0.9772 |

## Registered Ref-L4 strata

| model | dimension | level | sample_count | eligible_count | clean_eligibility | full_manifest_operational | eligible_operational | coverage | conditional_ranking | conditional_minus_operational |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundingdino | source_split | coco | 600 | 600 | 1.0000 | 0.8530 | 0.8530 | 0.9784 | 0.8719 | 0.0189 |
| groundingdino | source_split | objects365 | 400 | 400 | 1.0000 | 0.8962 | 0.8962 | 0.9765 | 0.9178 | 0.0215 |
| groundingdino | query_length_stratum | long_ge30 | 218 | 218 | 1.0000 | 0.8728 | 0.8728 | 0.9815 | 0.8893 | 0.0165 |
| groundingdino | query_length_stratum | medium_19_29 | 504 | 504 | 1.0000 | 0.8635 | 0.8635 | 0.9755 | 0.8852 | 0.0216 |
| groundingdino | query_length_stratum | short_le18 | 278 | 278 | 1.0000 | 0.8806 | 0.8806 | 0.9784 | 0.9001 | 0.0195 |
| groundingdino | target_scale_stratum | large_gt96 | 894 | 894 | 1.0000 | 0.8658 | 0.8658 | 0.9779 | 0.8853 | 0.0195 |
| groundingdino | target_scale_stratum | medium_32_96 | 105 | 105 | 1.0000 | 0.9079 | 0.9079 | 0.9748 | 0.9314 | 0.0235 |
| groundingdino | target_scale_stratum | small_lt32 | 1 | 1 | 1.0000 | 0.9875 | 0.9875 | 1.0000 | 0.9875 | 0.0000 |
| yoloworld | source_split | coco | 600 | 473 | 0.7883 | 0.5288 | 0.6708 | 0.7693 | 0.8719 | 0.2011 |
| yoloworld | source_split | objects365 | 400 | 308 | 0.7700 | 0.5476 | 0.7112 | 0.7928 | 0.8971 | 0.1859 |
| yoloworld | query_length_stratum | long_ge30 | 218 | 159 | 0.7294 | 0.4933 | 0.6763 | 0.7675 | 0.8813 | 0.2049 |
| yoloworld | query_length_stratum | medium_19_29 | 504 | 400 | 0.7937 | 0.5352 | 0.6743 | 0.7694 | 0.8764 | 0.2021 |
| yoloworld | query_length_stratum | short_le18 | 278 | 222 | 0.7986 | 0.5722 | 0.7165 | 0.8032 | 0.8921 | 0.1756 |
| yoloworld | target_scale_stratum | large_gt96 | 894 | 702 | 0.7852 | 0.5385 | 0.6858 | 0.7781 | 0.8814 | 0.1956 |
| yoloworld | target_scale_stratum | medium_32_96 | 105 | 78 | 0.7429 | 0.5133 | 0.6910 | 0.7803 | 0.8856 | 0.1946 |
| yoloworld | target_scale_stratum | small_lt32 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Perturbation-family profile transfer

| baseline | model | family_count | spearman_risk_share | cosine_risk_share | mean_absolute_risk_share_shift |
| --- | --- | --- | --- | --- | --- |
| RefCOCO | groundingdino | 5 | 0.6000 | 0.9207 | 0.0637 |
| RefCOCO | yoloworld | 5 | 1.0000 | 0.9958 | 0.0154 |
| RefCOCO+ | groundingdino | 5 | 1.0000 | 0.9925 | 0.0228 |
| RefCOCO+ | yoloworld | 5 | 1.0000 | 0.9979 | 0.0132 |

## Interpretation boundary

The benchmark estimates operational candidate-order stability under the registered visual probe distribution, not semantic correctness.  The source-stratified pooled result describes the fixed 600/400 manifest.  Family attribution is descriptive rather than causal.