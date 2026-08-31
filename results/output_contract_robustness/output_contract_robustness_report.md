# Output-Contract Robustness Report

## Scope

This report replays the frozen candidate-order event on the full RefCOCO, RefCOCO+, and Ref-L4 traces. It evaluates all 108 registered combinations of tracked count, exposed count, match IoU, and birth-novelty IoU, with duplicate suppression fixed at 0.70.

The evidence is a deterministic finite-grid robustness result, not a claim over every real-valued threshold.

## Model-ranking robustness

| Dataset | Comparison | Minimum same-contract gap | 95% paired-bootstrap interval | Invariant | Strong envelope separation |
|---|---|---:|---:|:---:|:---:|
| RefCOCO | GroundingDINO - OWLv2 | 0.0913 | [0.0657, 0.1155] | yes | yes |
| RefCOCO | GroundingDINO - YOLO-World | 0.3147 | [0.2764, 0.3507] | yes | yes |
| RefCOCO | OWLv2 - YOLO-World | 0.1981 | [0.1617, 0.2346] | yes | yes |
| RefCOCO+ | GroundingDINO - OWLv2 | 0.0985 | [0.0793, 0.1164] | yes | yes |
| RefCOCO+ | GroundingDINO - YOLO-World | 0.3009 | [0.2741, 0.3282] | yes | yes |
| RefCOCO+ | OWLv2 - YOLO-World | 0.1904 | [0.1625, 0.2154] | yes | yes |
| Ref-L4 | GroundingDINO - OWLv2 | 0.0616 | [0.0435, 0.0789] | yes | yes |
| Ref-L4 | GroundingDINO - YOLO-World | 0.2448 | [0.2174, 0.2724] | yes | yes |
| Ref-L4 | OWLv2 - YOLO-World | 0.1777 | [0.1501, 0.2054] | yes | yes |

A positive minimum gap means that the first model ranks above the second at every registered identical contract. Each paired bootstrap repetition resamples image-query units and recomputes the worst setting.

## Absolute-value sensitivity envelopes

| Dataset | Model | Default | Minimum | Maximum | Width | Maximum departure |
|---|---|---:|---:|---:|---:|---:|
| RefCOCO | GroundingDINO | 0.8554 | 0.8332 | 0.9012 | 0.0680 | 0.0458 |
| RefCOCO | OWLv2 | 0.7395 | 0.7053 | 0.8095 | 0.1042 | 0.0701 |
| RefCOCO | YOLO-World | 0.5118 | 0.5072 | 0.5845 | 0.0773 | 0.0727 |
| RefCOCO+ | GroundingDINO | 0.8512 | 0.8225 | 0.9012 | 0.0788 | 0.0501 |
| RefCOCO+ | OWLv2 | 0.7332 | 0.6873 | 0.8016 | 0.1143 | 0.0684 |
| RefCOCO+ | YOLO-World | 0.5018 | 0.4969 | 0.5981 | 0.1012 | 0.0963 |
| Ref-L4 | GroundingDINO | 0.8703 | 0.8526 | 0.8813 | 0.0287 | 0.0177 |
| Ref-L4 | OWLv2 | 0.7677 | 0.7061 | 0.8195 | 0.1134 | 0.0615 |
| Ref-L4 | YOLO-World | 0.5363 | 0.5202 | 0.6365 | 0.1163 | 0.1001 |

These ranges are contract sensitivity envelopes, not confidence intervals.

## One-factor sensitivity around the default

| Dataset | Model | Parameter | Tested values | Range width |
|---|---|---|---|---:|
| RefCOCO | GroundingDINO | tracked_candidate_count | 2;3;4;5 | 0.0416 |
| RefCOCO | GroundingDINO | exposed_candidate_count | 10;15;20 | 0.0147 |
| RefCOCO | GroundingDINO | match_iou_threshold | 0.1;0.15;0.25 | 0.0082 |
| RefCOCO | GroundingDINO | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0007 |
| RefCOCO | OWLv2 | tracked_candidate_count | 2;3;4;5 | 0.0661 |
| RefCOCO | OWLv2 | exposed_candidate_count | 10;15;20 | 0.0218 |
| RefCOCO | OWLv2 | match_iou_threshold | 0.1;0.15;0.25 | 0.0124 |
| RefCOCO | OWLv2 | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0020 |
| RefCOCO | YOLO-World | tracked_candidate_count | 2;3;4;5 | 0.0698 |
| RefCOCO | YOLO-World | exposed_candidate_count | 10;15;20 | 0.0012 |
| RefCOCO | YOLO-World | match_iou_threshold | 0.1;0.15;0.25 | 0.0043 |
| RefCOCO | YOLO-World | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0002 |
| RefCOCO+ | GroundingDINO | tracked_candidate_count | 2;3;4;5 | 0.0458 |
| RefCOCO+ | GroundingDINO | exposed_candidate_count | 10;15;20 | 0.0172 |
| RefCOCO+ | GroundingDINO | match_iou_threshold | 0.1;0.15;0.25 | 0.0134 |
| RefCOCO+ | GroundingDINO | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0010 |
| RefCOCO+ | OWLv2 | tracked_candidate_count | 2;3;4;5 | 0.0652 |
| RefCOCO+ | OWLv2 | exposed_candidate_count | 10;15;20 | 0.0315 |
| RefCOCO+ | OWLv2 | match_iou_threshold | 0.1;0.15;0.25 | 0.0137 |
| RefCOCO+ | OWLv2 | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0015 |
| RefCOCO+ | YOLO-World | tracked_candidate_count | 2;3;4;5 | 0.0930 |
| RefCOCO+ | YOLO-World | exposed_candidate_count | 10;15;20 | 0.0020 |
| RefCOCO+ | YOLO-World | match_iou_threshold | 0.1;0.15;0.25 | 0.0045 |
| RefCOCO+ | YOLO-World | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0006 |
| Ref-L4 | GroundingDINO | tracked_candidate_count | 2;3;4;5 | 0.0087 |
| Ref-L4 | GroundingDINO | exposed_candidate_count | 10;15;20 | 0.0129 |
| Ref-L4 | GroundingDINO | match_iou_threshold | 0.1;0.15;0.25 | 0.0040 |
| Ref-L4 | GroundingDINO | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0008 |
| Ref-L4 | OWLv2 | tracked_candidate_count | 2;3;4;5 | 0.0417 |
| Ref-L4 | OWLv2 | exposed_candidate_count | 10;15;20 | 0.0407 |
| Ref-L4 | OWLv2 | match_iou_threshold | 0.1;0.15;0.25 | 0.0195 |
| Ref-L4 | OWLv2 | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0045 |
| Ref-L4 | YOLO-World | tracked_candidate_count | 2;3;4;5 | 0.0939 |
| Ref-L4 | YOLO-World | exposed_candidate_count | 10;15;20 | 0.0096 |
| Ref-L4 | YOLO-World | match_iou_threshold | 0.1;0.15;0.25 | 0.0078 |
| Ref-L4 | YOLO-World | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0015 |

## Candidate-pool diagnostics

| Dataset | Model | Clean pool at cap 20 | Perturbed post-dedup pool at cap 20 | Median perturbed count |
|---|---|---:|---:|---:|
| RefCOCO | GroundingDINO | 0.744 | 0.009 | 13.0 |
| RefCOCO | OWLv2 | 0.584 | 0.097 | 16.0 |
| RefCOCO | YOLO-World | 0.020 | 0.022 | 3.0 |
| RefCOCO+ | GroundingDINO | 0.862 | 0.011 | 13.0 |
| RefCOCO+ | OWLv2 | 0.565 | 0.106 | 16.0 |
| RefCOCO+ | YOLO-World | 0.049 | 0.050 | 3.0 |
| Ref-L4 | GroundingDINO | 0.994 | 0.024 | 15.0 |
| Ref-L4 | OWLv2 | 0.841 | 0.099 | 17.0 |
| Ref-L4 | YOLO-World | 0.165 | 0.191 | 6.0 |

A cap hit indicates possible truncation, not a known error. It motivates the prospectively registered raw pool of 50 candidates.

## Duplicate-threshold boundary

Existing perturbed traces were already deduplicated at IoU 0.70. Therefore only clean eligibility is reported under alternative duplicate thresholds; full operational duplicate sensitivity is not identifiable from the old traces.

| Dataset | Model | Duplicate IoU | Clean eligibility | Mean retained candidates |
|---|---|---:|---:|---:|
| RefCOCO | GroundingDINO | 0.50 | 0.9960 | 4.810 |
| RefCOCO | GroundingDINO | 0.60 | 0.9980 | 4.882 |
| RefCOCO | GroundingDINO | 0.70 | 0.9980 | 4.906 |
| RefCOCO | GroundingDINO | 0.80 | 0.9980 | 4.916 |
| RefCOCO | GroundingDINO | 0.85 | 0.9980 | 4.918 |
| RefCOCO | OWLv2 | 0.50 | 0.9760 | 4.684 |
| RefCOCO | OWLv2 | 0.60 | 0.9760 | 4.732 |
| RefCOCO | OWLv2 | 0.70 | 0.9760 | 4.756 |
| RefCOCO | OWLv2 | 0.80 | 0.9760 | 4.762 |
| RefCOCO | OWLv2 | 0.85 | 0.9760 | 4.766 |
| RefCOCO | YOLO-World | 0.50 | 0.7100 | 2.606 |
| RefCOCO | YOLO-World | 0.60 | 0.7200 | 2.672 |
| RefCOCO | YOLO-World | 0.70 | 0.7280 | 2.778 |
| RefCOCO | YOLO-World | 0.80 | 0.7280 | 2.778 |
| RefCOCO | YOLO-World | 0.85 | 0.7280 | 2.778 |
| RefCOCO+ | GroundingDINO | 0.50 | 1.0000 | 4.832 |
| RefCOCO+ | GroundingDINO | 0.60 | 1.0000 | 4.937 |
| RefCOCO+ | GroundingDINO | 0.70 | 1.0000 | 4.968 |
| RefCOCO+ | GroundingDINO | 0.80 | 1.0000 | 4.979 |
| RefCOCO+ | GroundingDINO | 0.85 | 1.0000 | 4.982 |
| RefCOCO+ | OWLv2 | 0.50 | 0.9640 | 4.613 |
| RefCOCO+ | OWLv2 | 0.60 | 0.9650 | 4.649 |
| RefCOCO+ | OWLv2 | 0.70 | 0.9680 | 4.681 |
| RefCOCO+ | OWLv2 | 0.80 | 0.9690 | 4.696 |
| RefCOCO+ | OWLv2 | 0.85 | 0.9690 | 4.708 |
| RefCOCO+ | YOLO-World | 0.50 | 0.7320 | 2.671 |
| RefCOCO+ | YOLO-World | 0.60 | 0.7430 | 2.776 |
| RefCOCO+ | YOLO-World | 0.70 | 0.7570 | 2.882 |
| RefCOCO+ | YOLO-World | 0.80 | 0.7570 | 2.882 |
| RefCOCO+ | YOLO-World | 0.85 | 0.7570 | 2.882 |
| Ref-L4 | GroundingDINO | 0.50 | 1.0000 | 4.990 |
| Ref-L4 | GroundingDINO | 0.60 | 1.0000 | 4.995 |
| Ref-L4 | GroundingDINO | 0.70 | 1.0000 | 5.000 |
| Ref-L4 | GroundingDINO | 0.80 | 1.0000 | 5.000 |
| Ref-L4 | GroundingDINO | 0.85 | 1.0000 | 5.000 |
| Ref-L4 | OWLv2 | 0.50 | 0.9980 | 4.959 |
| Ref-L4 | OWLv2 | 0.60 | 0.9980 | 4.972 |
| Ref-L4 | OWLv2 | 0.70 | 0.9980 | 4.978 |
| Ref-L4 | OWLv2 | 0.80 | 0.9980 | 4.979 |
| Ref-L4 | OWLv2 | 0.85 | 0.9980 | 4.981 |
| Ref-L4 | YOLO-World | 0.50 | 0.7680 | 3.283 |
| Ref-L4 | YOLO-World | 0.60 | 0.7770 | 3.375 |
| Ref-L4 | YOLO-World | 0.70 | 0.7810 | 3.467 |
| Ref-L4 | YOLO-World | 0.80 | 0.7810 | 3.468 |
| Ref-L4 | YOLO-World | 0.85 | 0.7810 | 3.468 |

## Why the default contract is retained

- `tracked_candidate_count = 5` observes more than a top-two contest while keeping one-to-one association auditable.
- `exposed_candidate_count = 20` exposes four times the tracked universe and is the largest value identifiable in v1 traces.
- `duplicate_iou_threshold = 0.70` treats near-overlapping localisation variants as duplicates and is shared with birth novelty.
- `match_iou_threshold = 0.15` tolerates corruption-induced displacement but rejects negligible spatial continuity.
- `birth_duplicate_iou_threshold = 0.70` makes suppression and novelty use one geometric convention.
- Hungarian one-to-one association prevents two clean candidates from claiming the same perturbed output.
- Strict score order treats ties as unstable and avoids implementation-specific tie breaking.
- Missing candidates and threatening births are coverage failures; no artificial scores are imputed.

## Preregistration and trace upgrade

Protocol: `output-contract-2.0.0`. The enhanced grid was frozen before this replay analysis.

The runner now supports `raw_candidate_pool_size` and stores both pre-contract and post-contract perturbed candidates. Future confirmatory runs use a pool of 50, enabling complete offline replay of duplicate suppression and wider exposure settings.
