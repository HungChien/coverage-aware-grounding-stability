# Frozen RefCOCO+ Transfer Analysis

All parameters, candidate contracts, probes, and estimators are inherited without target-data tuning.

## Source and target estimands

| model | scope | sample_count | eligible_count | clean_eligibility | full_manifest_operational | eligible_operational | coverage | conditional_ranking | conditional_minus_operational | full_manifest_lower_95 | full_manifest_upper_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundingdino | source_refcoco | 500 | 499 | 0.9980 | 0.8554 | 0.8571 | 0.9449 | 0.9070 | 0.0499 | 0.8385 | 0.8719 |
| groundingdino | target_refcocoplus_pooled | 1000 | 1000 | 1.0000 | 0.8512 | 0.8512 | 0.9374 | 0.9080 | 0.0568 | 0.8395 | 0.8627 |
| groundingdino | target_refcocoplus_testA | 500 | 500 | 1.0000 | 0.8595 | 0.8595 | 0.9299 | 0.9242 | 0.0648 | 0.8437 | 0.8750 |
| groundingdino | target_refcocoplus_testB | 500 | 500 | 1.0000 | 0.8429 | 0.8429 | 0.9449 | 0.8920 | 0.0492 | 0.8251 | 0.8607 |
| yoloworld | source_refcoco | 500 | 364 | 0.7280 | 0.5118 | 0.7030 | 0.8071 | 0.8710 | 0.1680 | 0.4780 | 0.5450 |
| yoloworld | target_refcocoplus_pooled | 1000 | 757 | 0.7570 | 0.5018 | 0.6628 | 0.7604 | 0.8717 | 0.2088 | 0.4797 | 0.5245 |
| yoloworld | target_refcocoplus_testA | 500 | 353 | 0.7060 | 0.4696 | 0.6652 | 0.7540 | 0.8821 | 0.2170 | 0.4384 | 0.5011 |
| yoloworld | target_refcocoplus_testB | 500 | 404 | 0.8080 | 0.5340 | 0.6608 | 0.7660 | 0.8627 | 0.2018 | 0.5038 | 0.5640 |

## Dataset-shift bootstrap

| model | comparison | full_manifest_delta | lower_95 | upper_95 |
| --- | --- | --- | --- | --- |
| groundingdino | RefCOCO+ minus RefCOCO | -0.0042 | -0.0256 | 0.0165 |
| yoloworld | RefCOCO+ minus RefCOCO | -0.0100 | -0.0518 | 0.0292 |

## Finite-probe transfer

| scope | diagnostic_budget | sample_count | eligible_count | full_manifest_bias | eligible_mae | eligible_spearman | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 5 | 1000 | 1000 | 0.0030 | 0.0681 | 0.8119 | groundingdino |
| pooled | 10 | 1000 | 1000 | 0.0028 | 0.0495 | 0.8955 | groundingdino |
| pooled | 20 | 1000 | 1000 | 0.0016 | 0.0361 | 0.9388 | groundingdino |
| pooled | 40 | 1000 | 1000 | 0.0009 | 0.0276 | 0.9587 | groundingdino |
| testA | 5 | 500 | 500 | 0.0050 | 0.0658 | 0.8114 | groundingdino |
| testA | 10 | 500 | 500 | 0.0038 | 0.0482 | 0.8854 | groundingdino |
| testA | 20 | 500 | 500 | 0.0024 | 0.0349 | 0.9332 | groundingdino |
| testA | 40 | 500 | 500 | 0.0021 | 0.0275 | 0.9530 | groundingdino |
| testB | 5 | 500 | 500 | 0.0011 | 0.0704 | 0.8117 | groundingdino |
| testB | 10 | 500 | 500 | 0.0019 | 0.0509 | 0.9044 | groundingdino |
| testB | 20 | 500 | 500 | 0.0008 | 0.0374 | 0.9440 | groundingdino |
| testB | 40 | 500 | 500 | -0.0004 | 0.0276 | 0.9636 | groundingdino |
| pooled | 5 | 1000 | 757 | -0.0006 | 0.1024 | 0.8784 | yoloworld |
| pooled | 10 | 1000 | 757 | -0.0002 | 0.0726 | 0.9307 | yoloworld |
| pooled | 20 | 1000 | 757 | 0.0005 | 0.0533 | 0.9609 | yoloworld |
| pooled | 40 | 1000 | 757 | 0.0010 | 0.0410 | 0.9764 | yoloworld |
| testA | 5 | 500 | 353 | 0.0060 | 0.1038 | 0.8711 | yoloworld |
| testA | 10 | 500 | 353 | 0.0048 | 0.0714 | 0.9302 | yoloworld |
| testA | 20 | 500 | 353 | 0.0040 | 0.0508 | 0.9610 | yoloworld |
| testA | 40 | 500 | 353 | 0.0021 | 0.0394 | 0.9763 | yoloworld |
| testB | 5 | 500 | 404 | -0.0071 | 0.1012 | 0.8854 | yoloworld |
| testB | 10 | 500 | 404 | -0.0051 | 0.0736 | 0.9317 | yoloworld |
| testB | 20 | 500 | 404 | -0.0030 | 0.0554 | 0.9606 | yoloworld |
| testB | 40 | 500 | 404 | -0.0001 | 0.0423 | 0.9762 | yoloworld |

## Perturbation-family profile transfer

| model | family_count | spearman_risk_share | cosine_risk_share | mean_absolute_risk_share_shift |
| --- | --- | --- | --- | --- |
| groundingdino | 5 | 0.6000 | 0.8768 | 0.0752 |
| yoloworld | 5 | 1.0000 | 0.9905 | 0.0228 |

## Interpretation boundary

The benchmark estimates operational candidate-order stability, not semantic correctness. Family-profile agreement is descriptive evidence of diagnostic transfer and not a causal claim.