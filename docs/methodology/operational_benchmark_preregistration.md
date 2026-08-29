# Coverage-Aware Candidate-Order Stability Benchmark v1

## Frozen primary question

Can a common, coverage-aware candidate-output contract estimate operational
candidate-order stability from a finite probe budget, diagnose observable risk
sources, and reproduce across heterogeneous grounding architectures on unseen
image-query pairs?

## Scope

The benchmark studies output stability, not semantic correctness.  Its domain
is candidate-producing, natural-language-conditioned grounding models that
expose bounding boxes and sortable candidate scores.

For a registered probe `U`, define `C(U)=1` when every tracked clean candidate
can be associated and no unresolved, spatially novel candidate can beat the
matched clean winner.  Given the matched score vector `s(U)` and the
winner-versus-competitor contrast matrix `A`, define

```text
G(U) = A s(U).
```

The primary event is

```text
Y(U) = 1{C(U)=1 and G(U)>0 componentwise}.
```

The primary instance-level estimand is `theta_op = E[Y(U)]`.  Candidate
coverage and conditional ranking stability are reported separately, with the
exact identity

```text
1 - theta_op
  = (1 - theta_cov) + theta_cov (1 - theta_rank).
```

## Dataset and models

- 500 RefCOCO image-query pairs not used by either previous 100-pair split;
- at most one expression per image;
- GroundingDINO and YOLO-World are required;
- identical image-query pairs and probe realisations are used for both models.

Clean task correctness is reported only as a contextual control.  It is not a
target inferred by the stability framework.

## Probe registry

Five meaning-preserving visual families are balanced exactly:

- Gaussian blur;
- brightness;
- JPEG compression;
- resolution reduction;
- Gaussian noise.

Every sample receives 40 diagnostic probes (8 per family) and 80 independent
reference probes (16 per family).  Prefix budgets are 5, 10, 20, and 40, with
one probe per family in every complete block.  Family, severity, seed, and
split are stored for every inference.

## Output contract

- expose at most 20 candidates;
- retain at most five spatially distinct clean candidates;
- remove duplicate candidates at IoU 0.70;
- use one-to-one maximum-IoU association with minimum IoU 0.15;
- use strict score inequalities; a tie is not stable;
- never replace a missing candidate by a numeric score;
- treat an unresolved, spatially novel candidate that can beat the matched
  clean winner as a coverage failure;
- include clean outputs with fewer than two distinct candidates in full-model
  coverage rather than silently removing them.

## Primary statistics

For each diagnostic budget, report exact fixed-budget binomial intervals for
coverage, conditional ranking stability, and operational stability.  Validate
the finite-budget estimate against the independent 80-probe reference using
MAE, RMSE, Spearman correlation, interval inclusion of the reference estimate,
and selective-risk curves.

At model level, use hierarchical bootstrap resampling of image-query pairs and
probes.  Cross-model comparisons are paired because both models receive the
same samples and probes.

## Diagnostic validation

The law of total probability gives the registered-family contributions

```text
1 - theta_op = sum_f pi_f (1 - theta_f).
```

The benchmark tests whether diagnostic family-risk shares agree with reference
shares.  Failure causes are made disjoint by the fixed precedence

```text
clean ineligible > winner missing > competitor missing
> threatening birth > ranking reversal > stable.
```

For a covered ranking reversal, the culprit is the competitor with the minimum
gap, using the smallest clean index to break exact ties.

## Claims that are explicitly excluded

- stability is not semantic correctness;
- family attribution is not internal neural causal identification;
- the benchmark gives no guarantee outside the registered probe distribution;
- raw confidence scores are not assumed comparable across models;
- no scalar score is required to outperform direct event-frequency estimation.
