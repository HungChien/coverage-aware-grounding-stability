# Output-Contract Robustness Report

## Scope

This report replays the frozen candidate-order event on the full RefCOCO, RefCOCO+, and Ref-L4 traces. It evaluates all 108 registered combinations of tracked count, exposed count, match IoU, and birth-novelty IoU, with duplicate suppression fixed at 0.70.

The evidence is a deterministic finite-grid robustness result, not a claim over every real-valued threshold.

## Model-ranking robustness

| Dataset | Minimum same-contract gap | 95% paired-bootstrap interval | Invariant | Strong envelope separation |
|---|---:|---:|:---:|:---:|
| RefCOCO | 0.3147 | [0.2772, 0.3502] | yes | yes |
| RefCOCO+ | 0.3009 | [0.2738, 0.3272] | yes | yes |
| Ref-L4 | 0.2448 | [0.2173, 0.2709] | yes | yes |

A positive minimum gap proves that GroundingDINO ranks above YOLO-World at every registered identical contract. The paired bootstrap recomputes the worst setting in every repetition.

## Absolute-value sensitivity envelopes

| Dataset | Model | Default | Minimum | Maximum | Width | Maximum departure |
|---|---|---:|---:|---:|---:|---:|
| RefCOCO | groundingdino | 0.8554 | 0.8332 | 0.9012 | 0.0680 | 0.0458 |
| RefCOCO | yoloworld | 0.5118 | 0.5072 | 0.5845 | 0.0773 | 0.0727 |
| RefCOCO+ | groundingdino | 0.8512 | 0.8225 | 0.9012 | 0.0788 | 0.0501 |
| RefCOCO+ | yoloworld | 0.5018 | 0.4969 | 0.5981 | 0.1012 | 0.0963 |
| Ref-L4 | groundingdino | 0.8703 | 0.8526 | 0.8813 | 0.0287 | 0.0177 |
| Ref-L4 | yoloworld | 0.5363 | 0.5202 | 0.6365 | 0.1163 | 0.1001 |

These ranges are contract sensitivity envelopes, not confidence intervals.

## One-factor sensitivity around the default

| Dataset | Model | Parameter | Tested values | Range width |
|---|---|---|---|---:|
| RefCOCO | groundingdino | tracked_candidate_count | 2;3;4;5 | 0.0416 |
| RefCOCO | groundingdino | exposed_candidate_count | 10;15;20 | 0.0147 |
| RefCOCO | groundingdino | match_iou_threshold | 0.1;0.15;0.25 | 0.0082 |
| RefCOCO | groundingdino | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0007 |
| RefCOCO | yoloworld | tracked_candidate_count | 2;3;4;5 | 0.0698 |
| RefCOCO | yoloworld | exposed_candidate_count | 10;15;20 | 0.0012 |
| RefCOCO | yoloworld | match_iou_threshold | 0.1;0.15;0.25 | 0.0043 |
| RefCOCO | yoloworld | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0002 |
| RefCOCO+ | groundingdino | tracked_candidate_count | 2;3;4;5 | 0.0458 |
| RefCOCO+ | groundingdino | exposed_candidate_count | 10;15;20 | 0.0172 |
| RefCOCO+ | groundingdino | match_iou_threshold | 0.1;0.15;0.25 | 0.0134 |
| RefCOCO+ | groundingdino | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0010 |
| RefCOCO+ | yoloworld | tracked_candidate_count | 2;3;4;5 | 0.0930 |
| RefCOCO+ | yoloworld | exposed_candidate_count | 10;15;20 | 0.0020 |
| RefCOCO+ | yoloworld | match_iou_threshold | 0.1;0.15;0.25 | 0.0045 |
| RefCOCO+ | yoloworld | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0006 |
| Ref-L4 | groundingdino | tracked_candidate_count | 2;3;4;5 | 0.0087 |
| Ref-L4 | groundingdino | exposed_candidate_count | 10;15;20 | 0.0129 |
| Ref-L4 | groundingdino | match_iou_threshold | 0.1;0.15;0.25 | 0.0040 |
| Ref-L4 | groundingdino | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0008 |
| Ref-L4 | yoloworld | tracked_candidate_count | 2;3;4;5 | 0.0939 |
| Ref-L4 | yoloworld | exposed_candidate_count | 10;15;20 | 0.0096 |
| Ref-L4 | yoloworld | match_iou_threshold | 0.1;0.15;0.25 | 0.0078 |
| Ref-L4 | yoloworld | birth_duplicate_iou_threshold | 0.5;0.7;0.85 | 0.0015 |

## Candidate-pool diagnostics

| Dataset | Model | Clean pool at cap 20 | Perturbed post-dedup pool at cap 20 | Median perturbed count |
|---|---|---:|---:|---:|
| RefCOCO | groundingdino | 0.744 | 0.009 | 13.0 |
| RefCOCO | yoloworld | 0.020 | 0.022 | 3.0 |
| RefCOCO+ | groundingdino | 0.862 | 0.011 | 13.0 |
| RefCOCO+ | yoloworld | 0.049 | 0.050 | 3.0 |
| Ref-L4 | groundingdino | 0.994 | 0.024 | 15.0 |
| Ref-L4 | yoloworld | 0.165 | 0.191 | 6.0 |

A cap hit indicates possible truncation, not a known error. It motivates the prospectively registered raw pool of 50 candidates.

## Duplicate-threshold boundary

Existing perturbed traces were already deduplicated at IoU 0.70. Therefore only clean eligibility is reported under alternative duplicate thresholds; full operational duplicate sensitivity is not identifiable from the old traces.

| Dataset | Model | Duplicate IoU | Clean eligibility | Mean retained candidates |
|---|---|---:|---:|---:|
| RefCOCO | groundingdino | 0.50 | 0.9960 | 4.810 |
| RefCOCO | groundingdino | 0.60 | 0.9980 | 4.882 |
| RefCOCO | groundingdino | 0.70 | 0.9980 | 4.906 |
| RefCOCO | groundingdino | 0.80 | 0.9980 | 4.916 |
| RefCOCO | groundingdino | 0.85 | 0.9980 | 4.918 |
| RefCOCO | yoloworld | 0.50 | 0.7100 | 2.606 |
| RefCOCO | yoloworld | 0.60 | 0.7200 | 2.672 |
| RefCOCO | yoloworld | 0.70 | 0.7280 | 2.778 |
| RefCOCO | yoloworld | 0.80 | 0.7280 | 2.778 |
| RefCOCO | yoloworld | 0.85 | 0.7280 | 2.778 |
| RefCOCO+ | groundingdino | 0.50 | 1.0000 | 4.832 |
| RefCOCO+ | groundingdino | 0.60 | 1.0000 | 4.937 |
| RefCOCO+ | groundingdino | 0.70 | 1.0000 | 4.968 |
| RefCOCO+ | groundingdino | 0.80 | 1.0000 | 4.979 |
| RefCOCO+ | groundingdino | 0.85 | 1.0000 | 4.982 |
| RefCOCO+ | yoloworld | 0.50 | 0.7320 | 2.671 |
| RefCOCO+ | yoloworld | 0.60 | 0.7430 | 2.776 |
| RefCOCO+ | yoloworld | 0.70 | 0.7570 | 2.882 |
| RefCOCO+ | yoloworld | 0.80 | 0.7570 | 2.882 |
| RefCOCO+ | yoloworld | 0.85 | 0.7570 | 2.882 |
| Ref-L4 | groundingdino | 0.50 | 1.0000 | 4.990 |
| Ref-L4 | groundingdino | 0.60 | 1.0000 | 4.995 |
| Ref-L4 | groundingdino | 0.70 | 1.0000 | 5.000 |
| Ref-L4 | groundingdino | 0.80 | 1.0000 | 5.000 |
| Ref-L4 | groundingdino | 0.85 | 1.0000 | 5.000 |
| Ref-L4 | yoloworld | 0.50 | 0.7680 | 3.283 |
| Ref-L4 | yoloworld | 0.60 | 0.7770 | 3.375 |
| Ref-L4 | yoloworld | 0.70 | 0.7810 | 3.467 |
| Ref-L4 | yoloworld | 0.80 | 0.7810 | 3.468 |
| Ref-L4 | yoloworld | 0.85 | 0.7810 | 3.468 |

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
