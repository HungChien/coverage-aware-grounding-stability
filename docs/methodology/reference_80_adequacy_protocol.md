# Frozen Adequacy Protocol for the 80-Probe Reference

## Question

Is the frozen 80-probe reference sufficiently stable for its intended role in
the candidate-order benchmark?

"Sufficient" must be tied to an inferential purpose. This protocol separates:

1. **model-level adequacy**: estimating the mean operational stability of a
   model on a target dataset and registered probe distribution;
2. **sample-level resolution**: approximating the latent operational stability
   of one image-query pair.

The same probe count can be adequate for the first purpose and only moderately
precise for the second.

## Frozen decision criteria

The 80-probe reference is considered adequate for model-level estimation when
both conditions hold for every frozen dataset-model group:

1. the finite-probe term contributes at most 5% of the estimated variance of
   an 80-probe image-query mean;
2. across 2,000 family-balanced 40/40 splits of the reference registry, the
   95th percentile of the absolute difference between the two model-level
   half-reference means is no larger than 0.01.

These thresholds express a one-percentage-point model-level resolution and a
design in which at least 95% of remaining uncertainty comes from sampling new
image-query units rather than adding probes to existing units.

Sample-level adequacy is not reduced to one binary statement. It is described
using:

- the average finite-probe root mean squared error
  \(\sqrt{B_m/80}\);
- 95% Clopper--Pearson interval widths under the iid-
  \(Q\) Bernoulli working model;
- family-balanced 40/40 split-half MAE, RMSE, and rank agreement;
- agreement between the independent 40-probe diagnostic registry and the
  80-probe reference registry.

An average probe RMSE no larger than 0.05 is treated as adequate for using the
80-probe value as a noisy continuous validation target. It is not interpreted
as an exact per-sample ground truth.

## Family-balanced split-half construction

The reference registry contains five perturbation families. Each repetition
randomly divides the registered probes within every family into two equal
halves. The halves therefore preserve the same family mixture and differ only
in the concrete registered probes assigned to each half.

For half means \(\widehat\Theta^{(a)}_{40}\) and
\(\widehat\Theta^{(b)}_{40}\), the model-level discrepancy is

\[
D_{40/40}
=
\left|
\widehat\Theta^{(a)}_{40}
-
\widehat\Theta^{(b)}_{40}
\right|.
\]

At sample level, the same disjoint but complementary halves produce MAE, RMSE,
and Spearman agreement. They are not treated as two newly sampled independent
registries. Full-manifest and clean-eligible scopes are reported separately so
that deterministic zeros from clean ineligibility do not inflate agreement.

## Theory-based adequacy quantities

Using the two-stage variance components

\[
A_m=\operatorname{Var}_X(g_m(X)),
\qquad
B_m=\mathbb E_X[\operatorname{Var}(W_m(X,U)\mid X)],
\]

the finite-probe variance share is

\[
\phi_m(R)
=
\frac{B_m/R}{A_m+B_m/R}.
\]

The average sample-level finite-probe RMSE is

\[
\operatorname{RMSE}_{\mathrm{probe}}(R)
=
\sqrt{\frac{B_m}{R}}.
\]

For precision conditional on clean eligibility, let
\(\Gamma_m=\Pr(E_m(X)=1)\). Because ineligible samples contribute zero
within-unit variance, the eligible-sample quantity is

\[
\operatorname{RMSE}_{\mathrm{probe}\mid E=1}(R)
=
\sqrt{\frac{B_m}{\Gamma_mR}}.
\]

The standard error at the observed sample size is

\[
\operatorname{SE}_R
=
\sqrt{
\frac{A_m}{N}
+
\frac{B_m}{NR}
}.
\]

The limiting standard error as the probe count tends to infinity is

\[
\operatorname{SE}_{\infty}
=
\sqrt{\frac{A_m}{N}}.
\]

The relative excess

\[
\frac{\operatorname{SE}_{80}}{\operatorname{SE}_{\infty}}-1
\]

directly measures how much model-level precision is still lost because the
reference uses 80 rather than infinitely many probes.

## Interpretation boundary

This analysis validates precision under the frozen empirical data and probe
registries. It does not prove that the registered probe distribution covers
every possible real-world perturbation. Adequacy of Monte Carlo depth is
different from adequacy of probe-distribution breadth.
