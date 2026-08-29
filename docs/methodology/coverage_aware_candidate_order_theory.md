# Coverage-Aware Candidate-Order Stability

## 1. Research object

For an image-query pair \(x=(I,T)\), a language-conditioned grounding model
returns an ordered finite candidate list

\[
\mathcal O_m(x)=\{(B_i,s_i)\}_{i=1}^{K},
\qquad s_1\geq s_2\geq\cdots\geq s_K.
\]

The benchmark does **not** treat scores from different architectures as
calibrated probabilities. It studies whether the clean candidate decision is
preserved when a probe \(U\) is drawn from a preregistered distribution
\(Q\). This makes the estimand applicable to heterogeneous candidate-producing
grounding architectures.

The clean candidate universe contains the first \(K_t\) spatially distinct
candidates. A perturbed inference exposes \(K_e\geq K_t\) candidates, so that
a new high-scoring candidate cannot be hidden by a narrow top-\(K_t\) view.

## 2. Coverage and association

Let \(\pi_U(i)\) be the one-to-one association from clean candidate \(i\) to a
perturbed candidate. The benchmark obtains \(\pi_U\) by maximum-IoU Hungarian
matching, followed by a fixed minimum-IoU acceptance threshold.

Define the coverage event

\[
C(U)=1
\]

if and only if:

1. every tracked clean candidate has an accepted association; and
2. no unmatched, spatially novel perturbed candidate scores at least as highly
   as the matched clean winner.

Candidate disappearance and threatening candidate birth are therefore
observable outcomes. They are not converted into an invented score such as
zero or negative infinity.

## 3. Candidate-order event

When \(C(U)=1\), define the perturbed gap vector

\[
\mathbf G(U)=
\begin{bmatrix}
s'_{\pi_U(1)}-s'_{\pi_U(2)}\\
\vdots\\
s'_{\pi_U(1)}-s'_{\pi_U(K_t)}
\end{bmatrix}.
\]

The conditional ranking event is

\[
S(U)=\mathbb 1\{\min_j G_j(U)>0\},
\]

and the operational event is

\[
Y(U)=C(U)S(U).
\]

Thus, an operational success requires both an observable candidate universe and
a preserved clean winner.

## 4. Primary and secondary estimands

For \(U\sim Q\), define

\[
\theta_{\mathrm{cov}}=\Pr(C=1),
\]

\[
\theta_{\mathrm{rank}}=\Pr(S=1\mid C=1),
\]

and

\[
\theta_{\mathrm{op}}=\Pr(Y=1)
=\theta_{\mathrm{cov}}\theta_{\mathrm{rank}}.
\]

The primary estimand is \(\theta_{\mathrm{op}}\). Coverage and conditional
ranking stability are secondary diagnostic estimands.

## 5. Proposition 1: exact risk decomposition

Let \(r_{\mathrm{op}}=1-\theta_{\mathrm{op}}\). Then

\[
r_{\mathrm{op}}
=(1-\theta_{\mathrm{cov}})
+\theta_{\mathrm{cov}}(1-\theta_{\mathrm{rank}}).
\]

**Proof.** Substitute
\(\theta_{\mathrm{op}}=\theta_{\mathrm{cov}}\theta_{\mathrm{rank}}\):

\[
1-\theta_{\mathrm{cov}}\theta_{\mathrm{rank}}
=1-\theta_{\mathrm{cov}}
+\theta_{\mathrm{cov}}
-\theta_{\mathrm{cov}}\theta_{\mathrm{rank}}.
\]

Collecting terms gives the result. The two terms are disjoint: failure to
establish coverage, and ranking failure after coverage has been established.

## 6. Proposition 2: why coverage-aware stability differs from direct persistence

Because \(0\leq\theta_{\mathrm{cov}}\leq1\),

\[
\theta_{\mathrm{op}}
=\theta_{\mathrm{cov}}\theta_{\mathrm{rank}}
\leq\theta_{\mathrm{rank}}.
\]

The excess of conditional persistence over operational stability is exactly

\[
\theta_{\mathrm{rank}}-\theta_{\mathrm{op}}
=\theta_{\mathrm{rank}}(1-\theta_{\mathrm{cov}}).
\]

Therefore, a method that evaluates ranking only when candidates remain
matchable can hide instability created by candidate disappearance or birth.
The difference is zero only under perfect coverage or zero conditional ranking
stability.

## 7. Proposition 3: perturbation-family attribution

Let the registered probe distribution be the mixture

\[
Q=\sum_{f=1}^{F}\pi_fQ_f,
\qquad \pi_f\geq0,
\qquad \sum_f\pi_f=1.
\]

If \(\theta_f=\Pr(Y=1\mid f)\), the law of total probability gives

\[
1-\theta_{\mathrm{op}}
=\sum_{f=1}^{F}\pi_f(1-\theta_f).
\]

When total risk is non-zero, the descriptive risk share of family \(f\) is

\[
\rho_f=
\frac{\pi_f(1-\theta_f)}{1-\theta_{\mathrm{op}}},
\qquad \sum_f\rho_f=1.
\]

These shares locate instability under the registered probe distribution. They
must not be interpreted as causal effects because the probe families may not
represent all real deployment shifts.

## 8. Proposition 4: finite-probe estimation

For a fixed sample, independent registered probes produce Bernoulli variables

\[
Y_1,\ldots,Y_n\overset{\mathrm{iid}}{\sim}
\mathrm{Bernoulli}(\theta_{\mathrm{op}}).
\]

The estimator

\[
\widehat\theta_{\mathrm{op}}
=\frac{1}{n}\sum_{r=1}^{n}Y_r
\]

is unbiased and has variance

\[
\operatorname{Var}(\widehat\theta_{\mathrm{op}})
=\frac{\theta_{\mathrm{op}}(1-\theta_{\mathrm{op}})}{n}
\leq\frac{1}{4n}.
\]

Hoeffding's inequality gives the distribution-free statement

\[
\Pr\left(
\left|\widehat\theta_{\mathrm{op}}-	heta_{\mathrm{op}}\right|
\geq\varepsilon
\right)
\leq2e^{-2n\varepsilon^2}.
\]

Hence a sufficient budget for error at most \(\varepsilon\) with confidence
at least \(1-\alpha\) is

\[
n\geq\frac{\log(2/\alpha)}{2\varepsilon^2}.
\]

The benchmark reports exact Clopper-Pearson intervals for observed Bernoulli
counts. The independent 80-probe estimate is a higher-budget finite reference,
not an exact population probability.

## 9. Proposition 5: cross-architecture order comparability

Let \(h_m\) be any strictly increasing transformation of model \(m\)'s exposed
scores. Conditional on the same candidate set,

\[
s_i>s_j \Longleftrightarrow h_m(s_i)>h_m(s_j).
\]

Therefore \(S(U)\), \(Y(U)\), and their Bernoulli probabilities are invariant
to strictly increasing score rescaling. This permits comparisons of operational
stability without pretending that a score of 0.8 has the same probabilistic
meaning in GroundingDINO and YOLO-World.

This invariance does not remove candidate-generation differences. Those
differences are deliberately retained through clean eligibility and coverage,
because they are part of the operational behaviour being benchmarked.

## 10. Identifiability boundary

Let \(H\) contain the complete observable history of clean and perturbed model
outputs, and let \(Y^*\) denote the latent semantic-correctness label. Without
annotations or an assumption linking \(H\) to \(Y^*\), correctness is not
identifiable from output probes alone. To see this, construct two data-generating
worlds with the same distribution of \(H\) but opposite values of \(Y^*\) for a
subset of examples. Every estimator measurable with respect to \(H\) has the
same distribution in both worlds, although correctness differs. No such
estimator can therefore recover correctness in both worlds.

In contrast, \(C(U)\), \(S(U)\), and \(Y(U)\) are functions of the observable
trace under the output contract. Their probabilities under the registered
probe distribution are identifiable and consistently estimated by repeated
registered probes. This is why the thesis deliberately targets stability.

## 11. Sample-level and model-level quantities

For sample \(x_n\), let \(\theta_{m,n}\) denote operational stability. The
full-manifest model quantity is

\[
\Theta_m=\frac{1}{N}\sum_{n=1}^{N}E_{m,n}\theta_{m,n},
\]

where \(E_{m,n}=1\) when the clean output exposes at least two spatially
distinct candidates and zero otherwise. The benchmark separately reports

\[
\Gamma_m=\frac{1}{N}\sum_{n=1}^{N}E_{m,n}
\]

as clean candidate eligibility, and an eligible-only conditional mean. This
prevents selective omission of difficult or sparse-output samples.

## 12. Failure localisation

Each failed probe receives exactly one primary label:

1. clean winner missing;
2. tracked competitor missing;
3. unmatched threatening candidate birth; or
4. ranking reversal under complete coverage.

For a ranking reversal, the culprit competitor is

\[
j^*(U)=\arg\min_jG_j(U).
\]

This converts a single stability estimate into an auditable failure profile.
It says whether instability was dominated by candidate observability, candidate
birth, or an actual swap among tracked candidates.

## 13. Empirical hypotheses

- **H1 — finite-budget convergence:** MAE to the independent 80-probe reference
  decreases as the balanced diagnostic budget grows from 5 to 40.
- **H2 — non-redundancy of coverage:** conditional ranking persistence exceeds
  operational stability whenever coverage is imperfect, with the exact gap
  predicted by Proposition 2.
- **H3 — transferable definition:** the same output contract and Bernoulli
  estimands can be computed for GroundingDINO and YOLO-World without comparing
  their raw score scales.
- **H4 — diagnostic value:** the registered perturbation families and disjoint
  primary causes produce non-trivial, model-specific risk profiles.
- **H5 — reproducibility:** frozen data, probes, thresholds, seeds, model
  checkpoints, and complete per-probe traces reproduce all reported tables.

## 14. Limits of the claim

The framework estimates stability under a specified probe distribution. It
does not prove semantic correctness, global robustness to arbitrary shifts, or
causal responsibility of a corruption family. Those are different estimands.
The contribution is a coverage-aware, cross-architecture and finite-sample
auditable benchmark for candidate-order stability.
