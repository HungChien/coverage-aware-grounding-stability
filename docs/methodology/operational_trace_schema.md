# Operational Trace Schema

The formal runner saves one compressed JSON record per model-sample pair in
`sample_traces.jsonl.gz`. The trace is the source of truth for all later
statistics. Compact CSV summaries and figures can be regenerated without
running either grounding model again.

## Sample-level fields

| Field | Meaning |
|---|---|
| `model` | Stable model key |
| `image_id`, `ref_id` | RefCOCO sample identity |
| `query` | Natural-language referring expression |
| `target_box` | Ground-truth box retained for contextual correctness analysis |
| `category_id` | COCO category identifier |
| `clean_eligible` | Whether at least two spatially distinct clean candidates exist |
| `clean_correct` | Contextual top-one IoU correctness, not the stability target |
| `clean_best_iou` | Top-one IoU with the target box |
| `raw_clean_candidates` | Model candidates before spatial distinctness selection |
| `tracked_clean_candidates` | Frozen clean candidate universe used by stability analysis |
| `diagnostic_profile` | Complete 40-probe profile |
| `reference_profile` | Complete independent 80-probe profile |
| `probes` | Ordered diagnostic and reference probe records |

## Probe specification

Every probe record contains the exact sampled transformation:

```json
{
  "family": "blur",
  "severity": 1.25,
  "seed": 123456
}
```

The seed is essential for Gaussian-noise reproduction. Deterministic probe
families retain it to preserve a uniform schema.

## Probe outcome

Each prospective v2 probe record stores two candidate lists before the outcome:

| Field | Meaning |
|---|---|
| `raw_candidates` | Larger pre-contract perturbed candidate pool returned by inference |
| `candidates` | Post-contract, spatially distinct perturbed candidates used by the frozen outcome |

This separation makes exposed count and duplicate suppression replayable
offline. V1 traces contain only `candidates`, capped at 20 and already
deduplicated at IoU 0.70; analyses of those traces must hold duplicate
suppression fixed and cannot identify exposure above 20.

| Field | Meaning |
|---|---|
| `coverage` | One only when tracked candidates are observable and no threatening birth exists |
| `rank_stable` | Ranking result conditional on coverage; `null` if coverage fails |
| `operational_stable` | Product of coverage and conditional ranking stability |
| `primary_failure` | Exactly one of stable, winner missing, competitor missing, threatening birth, or ranking reversal |
| `culprit_clean_index` | Clean competitor producing the minimum perturbed gap in a ranking reversal |
| `mapping` | Clean-index to perturbed-index one-to-one association |
| `matched_ious` | IoU of every accepted association |
| `matched_scores` | Scores of associated perturbed candidates; missing candidates remain `null` |
| `matched_boxes` | Boxes of associated perturbed candidates |
| `gaps` | Winner-to-competitor gaps under complete coverage |
| `missing_clean_indices` | Tracked candidates without an accepted match |
| `threatening_birth_indices` | Unmatched spatially novel high-score candidates |
| `threatening_birth_scores` | Scores of threatening candidates |
| `perturbed_candidate_count` | Number of spatially distinct exposed perturbed candidates |

## Why missing values are not scores

When a tracked candidate disappears, its score is unknown under the output
contract. Assigning zero or negative infinity would manufacture a numerical
gap and could make the ranking appear stable. The trace instead records
`null`, sets coverage to zero, and assigns a coverage-failure label.

## Deterministic failure priority

Multiple structural issues can occur in one perturbed output. For auditability,
the runner assigns one primary label in this order:

1. winner missing;
2. any tracked competitor missing;
3. threatening candidate birth;
4. covered ranking reversal;
5. stable.

All low-level evidence remains in the trace, so alternative secondary analyses
can inspect co-occurring issues without changing the preregistered primary
label.

## Resume behaviour

The trace is appended before the summary CSV is updated atomically. On
`--resume`, a sample is considered complete only when both its trace record and
all four registered diagnostic-budget rows are present. If execution stops
between these operations, the sample is recomputed; trace loading keeps the
last record for a duplicated key. This prevents a summary-only sample from
being silently skipped.

## Git storage policy

Complete traces are retained locally because they can exceed normal GitHub
artifact limits. The repository commits compact result tables, figures,
metadata, hashes, and reports. A published archival version should store the
compressed traces in a release asset or institutional data repository and
record its checksum in the paper artifact manifest.
