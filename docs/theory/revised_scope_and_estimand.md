# Revised Scope and Stability Estimand

## Research scope

This project studies candidate-order stability in vision-language
grounding. It does not use stability as a synonym for semantic correctness.

For a model (m), image (I), query (T), and a registered perturbation
distribution (U\sim\mathcal P_U), the clean model output is reduced to a fixed
set of spatially distinct candidates:

\[
\mathcal O_m(I,T)=\{(B_i,s_i)\}_{i=1}^{K},
\qquad s_1\geq s_2\geq\cdots\geq s_K.
\]

The primary pairwise estimand is

\[
\theta_j(I,T)
=
P_{U\sim\mathcal P_U}
\left(s_1^U>s_j^U\right),
\qquad j>1.
\]

The full tracked-candidate estimand is

\[
\theta_{\mathrm{all}}(I,T)
=
P_{U\sim\mathcal P_U}
\left(s_1^U>s_j^U\text{ for every }j>1\right).
\]

These quantities describe whether the clean winner retains its ordering under
the declared perturbation distribution. They do not describe whether the clean
winner corresponds to the intended semantic target.

## Diagnostic probe

A diagnostic probe is one concrete transformation sampled from, or explicitly
registered as part of, the perturbation protocol. A perturbation family is a
rule such as Gaussian blur. A probe is a realised transformation such as
Gaussian blur with radius 1.37 and a recorded random seed.

For probe (r), candidate (i) has score change

\[
\epsilon_i^{(r)}=s_i^{(r)}-s_i.
\]

For competitor (j), the relative score change is

\[
Z_j^{(r)}
=
\epsilon_j^{(r)}-\epsilon_1^{(r)}.
\]

A reversal occurs exactly when

\[
Z_j^{(r)}\geq M_j,
\qquad M_j=s_1-s_j.
\]

Probes are local stress tests, not training examples. Their purpose is to
observe how the model's current candidate ordering responds to a controlled
input neighbourhood.

## Two valid interpretations

### Fixed registry

If the same finite set of transformations is always used, the result describes
stability over that registry. It should not be interpreted as a population
probability for real-world perturbations.

### Random probe distribution

If probes are independently sampled from a declared distribution
\(\mathcal P_U\), their observations can estimate population moments and
stability under that distribution. The distribution and its severity ranges
are part of the scientific claim and must be reported.

## Explicit exclusions

The current scope does not claim:

- semantic correctness from candidate-order stability;
- guarantees outside the registered perturbation distribution;
- semantic identity for candidates matched only by geometry;
- causal identification of an internal neural-network failure;
- finite-sample certification from uncorrected plug-in moments.

The present contribution is a local, output-level stability analysis. Semantic
alignment, candidate coverage and causal diagnosis remain separate research
axes.
