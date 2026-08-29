# Two-Stage Statistical Theory for Model-Level Operational Stability

## 1. Scope and estimand

The benchmark samples two sources of variation:

1. an image-query unit from a target data distribution;
2. a registered perturbation from a probe distribution.

Fix a grounding model (m). Let

\[
X_n \overset{\mathrm{iid}}{\sim} P,
\qquad n=1,\ldots,N,
\]

where (X_n) is an image-query pair, and let

\[
U_{nr}\mid X_n \overset{\mathrm{iid}}{\sim} Q,
\qquad r=1,\ldots,R.
\]

Different image-query units and their probe collections are independent. The
probe law (Q) is part of the benchmark contract; changing (Q) changes the
estimand.

Let (E_m(x)\in\{0,1\}) indicate whether the clean output is eligible under
the frozen candidate contract. For an eligible sample, let

\[
Z_m(x,u)\in\{0,1\}
\]

denote success under the registered coverage-and-order event. The
full-manifest event is

\[
W_m(x,u):=E_m(x)Z_m(x,u).
\]

This definition removes an important ambiguity. If a symbol (Y_m) already
includes clean eligibility, then (Y_m=W_m) and eligibility must not be
multiplied a second time. In this document, (Z_m) is the eligible-sample
event and (W_m) is the full operational event.

Define the sample-specific stability function

\[
\theta_m(x):=\mathbb E_{U\sim Q}[Z_m(x,U)\mid X=x]
\]

and

\[
g_m(x):=E_m(x)\theta_m(x).
\]

The model-level operational stability on target distribution (P), under
registered probe law (Q), is

\[
\boxed{
\Theta_{m,P,Q}
:=
\mathbb E_{X\sim P}\mathbb E_{U\sim Q}[W_m(X,U)]
=
\mathbb E_{X\sim P}[g_m(X)].
}
\]

The target is stability under the specified data and probe distributions. It
is not semantic correctness and it is not a property of the architecture in
the absence of (P), (Q), and the output contract.

## 2. Balanced two-stage estimator

For (N) sampled units and (R) probes per unit, define

\[
W_{mnr}:=W_m(X_n,U_{nr})
\]

and the per-unit probe mean

\[
\widehat\theta_{m,n}^{\mathrm{op}}
:=
\frac{1}{R}\sum_{r=1}^{R}W_{mnr}.
\]

The model-level estimator is

\[
\boxed{
\widehat\Theta_m
:=
\frac{1}{N}\sum_{n=1}^{N}\widehat\theta_{m,n}^{\mathrm{op}}
=
\frac{1}{NR}\sum_{n=1}^{N}\sum_{r=1}^{R}W_{mnr}.
}
\]

Writing (W=EZ) recovers the requested form

\[
\widehat\Theta_m
=
\frac{1}{N}\sum_{n=1}^{N}E_m(X_n)
\left(
\frac{1}{R}\sum_{r=1}^{R}Z_m(X_n,U_{nr})
\right).
\]

## 3. Assumptions

The basic results require the following conditions.

**A1. Target sampling.** The image-query units (X_1,\ldots,X_N) are
independent draws from (P).

**A2. Registered probe sampling.** Conditional on (X_n), the probes
(U_{n1},\ldots,U_{nR}) are independent draws from the same law (Q).

**A3. Cross-unit independence.** Probe collections belonging to different
sampled units are independent.

**A4. Frozen measurement contract.** Eligibility, candidate association,
coverage, and order success are deterministic measurable functions of the
model output and the preregistered benchmark contract.

**A5. Non-adaptive balanced budget.** The number (R) is chosen before
observing probe outcomes. Outcome-dependent stopping requires a separate
sequential-sampling analysis.

Because (W\in\{0,1\}), all moments used below exist automatically.

## 4. Unbiasedness

### Theorem 1: unbiased model-level estimation

Under A1--A5,

\[
\boxed{
\mathbb E[\widehat\Theta_m]=\Theta_{m,P,Q}.
}
\]

**Proof.** By iterated expectation and conditional identical distribution,

\[
\begin{aligned}
\mathbb E[\widehat\Theta_m]
&=
\frac{1}{NR}\sum_{n=1}^{N}\sum_{r=1}^{R}
\mathbb E_X\left[
\mathbb E_U[W_m(X_n,U_{nr})\mid X_n]
\right]\\
&=
\frac{1}{NR}\sum_{n=1}^{N}\sum_{r=1}^{R}
\mathbb E_X[g_m(X_n)]\\
&=
\Theta_{m,P,Q}.
\end{aligned}
\]

No equality between stability and semantic correctness is used. The theorem
only concerns the operational event defined by the benchmark. \(\square\)

## 5. Exact variance decomposition

Define the two population variance components

\[
\boxed{
A_m:=\operatorname{Var}_{X\sim P}(g_m(X))
}
\]

and

\[
\boxed{
B_m:=
\mathbb E_{X\sim P}
\left[
E_m(X)\theta_m(X)(1-\theta_m(X))
\right].
}
\]

(A_m) is between-unit heterogeneity: it measures how strongly operational
stability changes across image-query pairs. (B_m) is within-unit probe
uncertainty: it measures how much registered probe outcomes fluctuate for a
fixed pair.

### Theorem 2: exact two-stage variance

Under A1--A5,

\[
\boxed{
\operatorname{Var}(\widehat\Theta_m)
=
\frac{A_m}{N}
+
\frac{B_m}{NR}.
}
\]

**Proof.** Let

\[
\overline W_{m,n}=\frac{1}{R}\sum_{r=1}^{R}W_{mnr}.
\]

The (overline W_{m,n}) are independent and identically distributed across
(n), hence

\[
\operatorname{Var}(\widehat\Theta_m)
=
\frac{1}{N}\operatorname{Var}(\overline W_{m,1}).
\]

By the law of total variance,

\[
\operatorname{Var}(\overline W_{m,1})
=
\operatorname{Var}_X
\left(
\mathbb E[\overline W_{m,1}\mid X]
\right)
+
\mathbb E_X
\left[
\operatorname{Var}(\overline W_{m,1}\mid X)
\right].
\]

The first term is

\[
\operatorname{Var}_X(g_m(X))=A_m.
\]

Conditional on (X=x), the (R) outcomes are iid Bernoulli with mean
(g_m(x)=E_m(x)\theta_m(x)). If (E_m(x)=0), every operational outcome is
zero. If (E_m(x)=1), the conditional variance is
(	heta_m(x)(1-\theta_m(x))). Therefore,

\[
\operatorname{Var}(\overline W_{m,1}\mid X=x)
=
\frac{E_m(x)\theta_m(x)(1-\theta_m(x))}{R}.
\]

Taking expectation gives (B_m/R), which proves the result. \(\square\)

### Interpretation

The decomposition gives four immediate consequences.

1. Increasing (N) reduces both sources of uncertainty.
2. Increasing (R) reduces only the within-unit component.
3. The variance floor as (R\to\infty) is (A_m/N).
4. Treating all (NR) outcomes as independent is generally anti-conservative.

## 6. Dependence, design effect, and effective sample size

Let (W_1) and (W_2) be two independent probes applied to the same randomly
sampled image-query pair. Conditional independence does not imply marginal
independence because both outcomes share (X). In fact,

\[
\operatorname{Cov}(W_1,W_2)
=
\operatorname{Var}_X(g_m(X))
=A_m.
\]

By total variance for one Bernoulli outcome,

\[
\Theta_{m,P,Q}(1-\Theta_{m,P,Q})=A_m+B_m.
\]

Therefore the intraclass correlation is

\[
\boxed{
\rho_m
=
\frac{A_m}{A_m+B_m}.
}
\]

Substituting into Theorem 2 yields

\[
\boxed{
\operatorname{Var}(\widehat\Theta_m)
=
\frac{Theta_m(1-\Theta_m)}{NR}
\left[1+(R-1)\rho_m\right].
}
\]

The multiplier

\[
\boxed{
\operatorname{DE}_m(R)=1+(R-1)\rho_m
}
\]

is the repeated-probe design effect. The corresponding effective number of
independent Bernoulli observations is

\[
\boxed{
N_{\mathrm{eff}}
=
\frac{NR}{1+(R-1)\rho_m}.
}
\]

When (ho_m>0), (N_{\mathrm{eff}}<NR). As (R\to\infty),

\[
N_{\mathrm{eff}}\longrightarrow \frac{N}{\rho_m},
\]

so repeated probing cannot create unlimited information about the target data
distribution.

## 7. Diminishing returns and the probe crossover

The fraction of the cluster-mean variance attributable to finite probes is

\[
\boxed{
\phi_m(R)
=
\frac{B_m/R}{A_m+B_m/R}
=
\frac{B_m}{A_mR+B_m}.
}
\]

The between-unit and within-unit contributions are equal at

\[
\boxed{
R_m^{\star}=\frac{B_m}{A_m}
}
\]

when (A_m>0). To ensure that probe uncertainty contributes at most
(eta\in(0,1)) of cluster-mean variance, it is sufficient and necessary that

\[
\boxed{
R
\geq
\frac{B_m(1-\eta)}{A_m\eta}.
}
\]

These equations convert the qualitative statement "more probes eventually
have diminishing returns" into an estimable design rule.

## 8. Consistency and asymptotic normality

For fixed (R\geq1), the cluster means (overline W_{m,n}\in[0,1]) are iid
with mean (Theta_m) and variance (A_m+B_m/R). The strong law of large
numbers gives

\[
\widehat\Theta_m\xrightarrow{\mathrm{a.s.}}\Theta_m.
\]

If (A_m+B_m/R>0), the classical central limit theorem gives

\[
\boxed{
\sqrt{N}
(\widehat\Theta_m-\Theta_m)
\xrightarrow{d}
\mathcal N\left(0,A_m+\frac{B_m}{R}\right).
}
\]

Thus an asymptotic confidence interval can be built from the sample variance of
the (N) cluster means. The image-query unit, not the individual probe, is the
independent unit for this interval.

## 9. Finite-sample concentration

Decompose the estimation error as

\[
\widehat\Theta_m-\Theta_m
=
\underbrace{
\frac{1}{N}\sum_{n=1}^{N}(g_m(X_n)-\Theta_m)
}_{\text{target-sampling error}}
+
\underbrace{
\frac{1}{NR}\sum_{n=1}^{N}\sum_{r=1}^{R}
(W_{mnr}-g_m(X_n))
}_{\text{probe Monte Carlo error}}.
\]

Applying Hoeffding's inequality to the first term and conditionally to the
second term gives, for any (arepsilon_X,arepsilon_U>0),

\[
\boxed{
\Pr\left(
|\widehat\Theta_m-\Theta_m|
\geq
\varepsilon_X+\varepsilon_U
\right)
\leq
2e^{-2N\varepsilon_X^2}
+
2e^{-2NR\varepsilon_U^2}.
}
\]

Consequently, with probability at least (1-\delta),

\[
\boxed{
|\widehat\Theta_m-\Theta_m|
\leq
\sqrt{\frac{\log(4/\delta)}{2N}}
+
\sqrt{\frac{\log(4/\delta)}{2NR}}.
}
\]

This bound is distribution free but conservative. A variance-sensitive
Bernstein bound follows by treating cluster means as independent bounded
variables. With

\[
V_R=A_m+\frac{B_m}{R},
\]

\[
\Pr(|\widehat\Theta_m-\Theta_m|\geq\varepsilon)
\leq
2\exp\left(
-\frac{N\varepsilon^2}{2V_R+2\varepsilon/3}
\right).
\]

The two bounds answer different questions: Hoeffding gives assumption-light
finite-sample protection, while the variance decomposition provides sharper
planning when (A_m) and (B_m) can be estimated.

## 10. Estimating the variance components

For the observed balanced matrix (W\in\{0,1\}^{N\times R}), define

\[
\overline W_n=\frac{1}{R}\sum_{r=1}^{R}W_{nr},
\qquad
\overline W=\frac{1}{N}\sum_{n=1}^{N}\overline W_n.
\]

The unbiased within-unit sample variance is

\[
s_n^2
=
\frac{1}{R-1}\sum_{r=1}^{R}(W_{nr}-\overline W_n)^2.
\]

Define

\[
\boxed{
\widehat B_m=\frac{1}{N}\sum_{n=1}^{N}s_n^2.
}
\]

Then (mathbb E[\widehat B_m]=B_m). Let

\[
S_{\overline W}^2
=
\frac{1}{N-1}\sum_{n=1}^{N}(\overline W_n-\overline W)^2.
\]

Because

\[
\mathbb E[S_{\overline W}^2]=A_m+\frac{B_m}{R},
\]

the method-of-moments estimator

\[
\boxed{
\widehat A_m^{\mathrm{raw}}
=
S_{\overline W}^2-rac{\widehat B_m}{R}
}
\]

is unbiased. Finite-sample noise can make it negative even though (A_m\geq0).
For operational planning, use

\[
\widehat A_m=\max(0,\widehat A_m^{\mathrm{raw}})
\]

while reporting the raw estimate for auditability.

An important identity is

\[
\frac{\widehat A_m^{\mathrm{raw}}}{N}
+
\frac{\widehat B_m}{NR}
=
\frac{S_{\overline W}^2}{N}.
\]

Thus the usual cluster-mean variance estimator is already valid for the
observed design. Decomposition is needed to understand the source of
uncertainty and to extrapolate to a different probe budget (R).

## 11. Sample-size planning

Using the normal approximation, a two-sided confidence interval with nominal
critical value (z_{1-\alpha/2}) has approximate half-width

\[
h
\approx
z_{1-\alpha/2}
\sqrt{
\frac{A_m+B_m/R}{N}
}.
\]

For a target half-width (h), the required number of image-query units is

\[
\boxed{
N
\geq
\frac{z_{1-\alpha/2}^2}{h^2}
\left(A_m+\frac{B_m}{R}\right).
}
\]

Planning values of (A_m) and (B_m) should come from an independent pilot or
a conservative upper envelope across models and datasets. Reusing the final
evaluation data for both design and confirmatory claims should be disclosed.

## 12. Cost-optimal allocation of images and probes

Let (c_X>0) be the cost of acquiring and processing one new image-query unit,
and let (c_U>0) be the marginal cost of one additional probe. For a total
budget (C),

\[
C=N(c_X+c_UR).
\]

Substituting (N=C/(c_X+c_UR)) into the exact variance gives

\[
\operatorname{Var}(\widehat\Theta_m)
=
\frac{(c_X+c_UR)(A_m+B_m/R)}{C}.
\]

Differentiating the numerator with respect to continuous (R>0),

\[
\frac{d}{dR}
\left[(c_X+c_UR)(A_m+B_m/R)\right]
=
A_mc_U-rac{B_mc_X}{R^2}.
\]

For (A_m,B_m>0), the unique stationary point is

\[
\boxed{
R_{\mathrm{opt}}
=
\sqrt{\frac{B_mc_X}{A_mc_U}}.
}
\]

The second derivative is (2B_mc_X/R^3>0), so this point is the unique global
minimum. In practice, round to a feasible integer and respect complete probe
blocks required by the registered family mixture.

The formula has a direct engineering interpretation:

- more heterogeneous data ((A_m\) large) favour more image-query units;
- more within-unit probe volatility ((B_m\) large) favour more probes;
- expensive new images ((c_X\) large) favour deeper probing;
- expensive model inference ((c_U\) large) favours shallower probing.

## 13. Unequal and adaptive probe budgets

If fixed probe counts (R_n) are chosen independently of the observed outcomes,

\[
\widehat\Theta_m
=
\frac{1}{N}\sum_{n=1}^{N}
\frac{1}{R_n}\sum_{r=1}^{R_n}W_{mnr}
\]

remains unbiased and

\[
\operatorname{Var}(\widehat\Theta_m)
=
\frac{A_m}{N}
+
\frac{1}{N^2}\sum_{n=1}^{N}
\mathbb E
\left[
\frac{E_m(X_n)\theta_m(X_n)(1-\theta_m(X_n))}{R_n}
\right].
\]

If (R_n) is selected after observing early successes or failures, the naive
sample proportion can be biased. Such designs require inverse-probability
weighting, martingale confidence sequences, or a preregistered stopping rule.
They are outside the primary balanced-design claim.

## 14. Dependence between sampled image-query units

The iid unit in the basic theorem is an image-query pair. If several referring
expressions share the same image, pair-level independence can fail. A rigorous
analysis must then either:

1. sample at most one query per image;
2. define the image as the outer cluster and use image-cluster bootstrap or a
   cluster-robust variance estimator; or
3. extend the hierarchy to image, query, and probe levels.

The empirical analysis reports both pair counts and unique image counts so that
this assumption is visible rather than hidden.

## 15. What the theory contributes

The two-stage theory contributes more than an error bar around a mean.

1. It identifies the correct independent sampling unit.
2. It proves an exact separation of target-sampling and probe-sampling error.
3. It quantifies the information loss caused by repeated-probe clustering.
4. It gives an effective sample size rather than reporting the misleading raw
   count (NR).
5. It gives estimable rules for diminishing returns, sample size, and
   cost-optimal allocation.
6. It remains architecture agnostic because it uses the common binary
   operational event rather than incomparable raw model scores.

## 16. Empirical verification plan

The frozen 80-probe traces from RefCOCO, RefCOCO+, and Ref-L4 are used for both
GroundingDINO and YOLO-World. For each dataset-model pair, the analysis will:

1. reconstruct the full-manifest (N\times80) operational matrix, assigning
   zero to clean-ineligible samples as required by the estimand;
2. estimate (A_m), (B_m), (ho_m), the design effect, effective sample
   size, and the probe crossover;
3. compare the predicted variance (A_m/N+B_m/(NR)) with nested resampling over
   a grid of image and probe budgets;
4. quantify diminishing returns from increasing (R);
5. produce sample-size and cost-allocation tables;
6. repeat the identical frozen analysis across all three datasets and both
   architectures.

The resampling validation uses the empirical 80-probe distribution as a finite
approximation to (Q). It validates the statistical design equations and
their usefulness for the observed benchmark. It does not claim that 80 probes
fully identify every possible real-world perturbation distribution.
