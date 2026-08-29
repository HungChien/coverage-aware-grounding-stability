# Output-Contract Parameters, Robustness, and Preregistration

## 1. Why an output contract is necessary

Grounding architectures expose different numbers of boxes, use different score
scales, and produce different patterns of duplicate hypotheses. Candidate-order
stability is therefore undefined until a common observable candidate universe,
association rule, and failure policy are specified.

The output contract is not a nuisance preprocessing choice. It defines the
measurement operation. Absolute stability is consequently reported together
with a contract sensitivity envelope, while model comparison is evaluated for
invariance over a registered parameter domain.

## 2. Engineering meaning of every parameter

### Tracked candidate count \(K_t\)

The clean output retains at most \(K_t\) spatially distinct, score-ordered
candidates. It defines the competitor universe that must remain observable.

- Small \(K_t\): cheaper and easier to associate, but can hide ambiguity below
  the first competitor.
- Large \(K_t\): captures more competing hypotheses, but increases eligibility
  and association burden.
- Default \(K_t=5\): observes more than a binary top-two contest while keeping
  one-to-one association auditable.

### Exposed perturbed candidate count \(K_e\)

The perturbed output exposes at most \(K_e\) candidates before association.

- Small \(K_e\): can hide a newly born high-score candidate and can make a
  tracked candidate appear missing only because the output was truncated.
- Large \(K_e\): reduces truncation bias but increases association cost and
  reveals more threatening births.
- Default \(K_e=20\): four times the tracked set, giving the association stage
  room to observe displaced candidates and high-score births.

The contract requires \(K_e\geq K_t\).

### Duplicate IoU threshold \(\tau_{\mathrm{dup}}\)

A score-ordered candidate is retained only when its IoU with every previously
retained candidate is below \(\tau_{\mathrm{dup}}\).

- Lower threshold: stronger suppression and fewer spatial hypotheses.
- Higher threshold: retains more overlapping hypotheses that may represent
  localisation variants rather than distinct objects.
- Default 0.70: suppresses near-duplicate localisation hypotheses while
  retaining spatially distinct competitors.

### Match IoU threshold \(\tau_{\mathrm{match}}\)

Hungarian matching first maximises total clean-to-perturbed IoU. An assigned
pair is accepted only when its IoU is at least
\(\tau_{\mathrm{match}}\).

- Lower threshold: tolerates larger displacement but risks semantically weak
  associations.
- Higher threshold: demands stronger spatial continuity but converts more
  displaced candidates into coverage failures.
- Default 0.15: intentionally allows severe corruption-induced movement while
  rejecting negligible overlap.

### Birth novelty threshold \(\tau_{\mathrm{birth}}\)

An unmatched candidate is spatially novel only when its IoU with every matched
candidate is below \(\tau_{\mathrm{birth}}\). A novel candidate whose score is
not below the matched clean winner causes a threatening-birth failure.

- Lower threshold: only strongly separated candidates count as novel.
- Higher threshold: more unmatched candidates count as distinct births.
- Default 0.70: uses the same geometric notion of near duplication as the
  candidate suppression rule.

### Fixed non-numeric rules

- At least two clean candidates are required because ranking stability needs a
  winner and competitor.
- Matching is one-to-one maximum-total-IoU Hungarian assignment.
- A score tie is unstable; the winner must remain strictly above every tracked
  competitor.
- Missing candidates are coverage failures and never receive invented numeric
  scores.
- Failure precedence is fixed so that diagnostic categories are mutually
  exclusive.

## 3. Formal robustness claim

Let \(\Lambda\) be the finite preregistered replayable parameter grid and let

\[
\Theta_m(\lambda)
\]

be full-manifest operational stability for model \(m\) under contract
\(\lambda\in\Lambda\).

For two models \(a\) and \(b\), define the same-contract gap

\[
D_{a,b}(\lambda)
=
\Theta_a(\lambda)-\Theta_b(\lambda).
\]

### Proposition 1: finite-domain ranking invariance

If

\[
\boxed{
\Delta_{a,b}^{\min}
:=
\min_{\lambda\in\Lambda}D_{a,b}(\lambda)>0,
}
\]

then model \(a\) ranks above model \(b\) at every registered contract setting.

**Proof.** By the definition of a minimum, for every
\(\lambda\in\Lambda\),

\[
D_{a,b}(\lambda)
\geq
\Delta_{a,b}^{\min}>0.
\]

Therefore \(\Theta_a(\lambda)>\Theta_b(\lambda)\) for every registered
setting. \(\square\)

### Corollary 1: strong envelope separation

Define

\[
L_a=\min_{\lambda\in\Lambda}\Theta_a(\lambda),
\qquad
U_b=\max_{\lambda\in\Lambda}\Theta_b(\lambda).
\]

If \(L_a>U_b\), then model \(a\) remains above model \(b\) even when the two
models are evaluated under different registered settings. This is stronger
than same-contract ranking invariance.

These are deterministic statements over the finite registered grid. They do
not establish invariance over every real-valued threshold between grid points.

## 4. Statistical uncertainty for the worst-case gap

The two models receive the same image-query manifest. Let

\[
\widehat D_{a,b}^{\min}
=
\min_{\lambda\in\Lambda}
\left[
\widehat\Theta_a(\lambda)-
\widehat\Theta_b(\lambda)
\right].
\]

Paired bootstrap resampling of image-query pairs is used to estimate the
sampling distribution of this minimum. The minimum is recomputed inside every
bootstrap repetition, so uncertainty includes data-dependent selection of the
worst contract setting. A lower 95% percentile bound above zero supports the
finite-grid ranking claim with sampling uncertainty included.

## 5. Absolute-value sensitivity

Ranking invariance does not imply that absolute values are contract invariant.
For each model and dataset, report the contract envelope

\[
\boxed{
\mathcal I_m^{\mathrm{contract}}
=
\left[
\min_{\lambda\in\Lambda}\widehat\Theta_m(\lambda),
\max_{\lambda\in\Lambda}\widehat\Theta_m(\lambda)
\right].
}
\]

Also report the default estimate, maximum absolute departure from the default,
and one-factor-at-a-time ranges. The envelope is a deterministic sensitivity
range, not a probabilistic confidence interval.

## 6. Replayability boundary of the existing traces

The v1 trace stores clean candidates before spatial duplicate removal, but
perturbed candidates were saved after the frozen 0.70 duplicate rule and were
capped at 20. Consequently:

- \(K_t\), \(K_e\leq20\), \(\tau_{\mathrm{match}}\), and
  \(\tau_{\mathrm{birth}}\) are replayable;
- clean eligibility can be inspected under alternative duplicate thresholds;
- full operational sensitivity to \(\tau_{\mathrm{dup}}\) is not identifiable;
- exposure above 20 is not identifiable.

This limitation is reported rather than filled with assumptions. Future traces
must save a larger pre-deduplication candidate pool for every clean and
perturbed inference. The runner is updated to support this prospective
requirement.

## 7. Preregistration status

The machine-readable parameter definitions, reasonable ranges, default values,
robustness claim, bootstrap rule, replayability limits, and prohibited tuning
operations are stored in
`config/output_contract_preregistration_v2.json`.

This v2 document is a prospective freeze for future confirmatory runs. The
enhanced replay grid is frozen before executing the enhanced analysis, but it
must not be described as having been preregistered before the original v1
experiments.
