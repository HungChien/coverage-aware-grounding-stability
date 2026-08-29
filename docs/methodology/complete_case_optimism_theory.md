# Complete-Case Persistence and Operational Optimism

## Research question

Repeated-probe grounding studies often evaluate candidate-order persistence only
when the clean winner and its competitors can still be associated with
perturbed outputs. This complete-case quantity is useful, but it is not the
same estimand as full operational stability.

## Three-stage observable process

For model (m), sample (X), and probe (U), define:

- (E=1) when the clean output exposes the registered number of spatially
  distinct candidates;
- (C=1) when the tracked clean candidates remain associated and no unmatched
  threatening candidate is born;
- (S=1) when the clean winner remains ahead of every tracked competitor,
  conditional on (C=1).

Let

\[
\Gamma=\Pr(E=1),
\qquad
\theta_{\mathrm{cov}}=\Pr(C=1\mid E=1),
\qquad
\theta_{\mathrm{cc}}=\Pr(S=1\mid C=1,E=1).
\]

Complete-case persistence estimates (	heta_{\mathrm{cc}}). Full-manifest
operational stability is

\[
\Theta_{\mathrm{op}}
=
\Pr(ECS=1)
=
\Gamma\theta_{\mathrm{cov}}\theta_{\mathrm{cc}}.
\]

## Theorem 1: non-negative complete-case optimism

The optimistic overstatement of full operational stability by complete-case
persistence is

\[
D_{\mathrm{total}}
=
\theta_{\mathrm{cc}}-\Theta_{\mathrm{op}}
=
\theta_{\mathrm{cc}}
\left(1-\Gamma\theta_{\mathrm{cov}}\right)
\geq 0.
\]

Equality holds if and only if either (	heta_{\mathrm{cc}}=0), or both clean
eligibility and perturbed-candidate coverage are perfect.

**Proof.** Substitute
(Theta_{\mathrm{op}}=\Gamma\theta_{\mathrm{cov}}	heta_{\mathrm{cc}})
and factor out (	heta_{\mathrm{cc}}). Every factor lies in ([0,1]), so the
result is non-negative.

## Corollary 1: interpretable optimism decomposition

The total gap separates exactly into coverage conditioning and clean-sample
exclusion:

\[
D_{\mathrm{total}}
=
\underbrace{
\theta_{\mathrm{cc}}(1-\theta_{\mathrm{cov}})
}_{D_{\mathrm{coverage}}}
+
\underbrace{
(1-\Gamma)\theta_{\mathrm{cov}}\theta_{\mathrm{cc}}
}_{D_{\mathrm{eligibility}}}.
\]

The first term is the instability hidden by evaluating rank only when candidate
association succeeds. The second is the additional overstatement caused by
excluding clean outputs that do not expose enough distinct candidates.

## Corollary 2: more data cannot remove estimand mismatch

Under ordinary consistency conditions, a complete-case estimator converges to
(	heta_{\mathrm{cc}}), while a full-manifest operational estimator converges
to (Theta_{\mathrm{op}}). Therefore, increasing the number of images or probes
does not remove (D_{\mathrm{total}}). It only estimates each distinct
quantity more precisely.

## Finite estimator

For (N) sampled image-query pairs and (R) registered probes, define

\[
\widehat\Theta_{\mathrm{op}}
=
\frac{1}{NR}
\sum_{n=1}^{N}
E_n
\sum_{r=1}^{R}C_{nr}S_{nr}.
\]

The complete-case estimator is

\[
\widehat\theta_{\mathrm{cc}}
=
\frac{
\sum_{n,r}E_nC_{nr}S_{nr}
}{
\sum_{n,r}E_nC_{nr}
},
\]

when the denominator is non-zero. These estimators intentionally have
different denominators because they answer different questions.

## Empirical validation protocol

The frozen RefCOCO, RefCOCO+, and Ref-L4 traces are used to:

1. verify every algebraic identity to numerical precision;
2. estimate the magnitude of the gap for GroundingDINO and YOLO-World;
3. quantify image- and probe-level uncertainty with hierarchical bootstrap;
4. test whether the gap persists at diagnostic budgets 5, 10, 20, and 40;
5. attribute the gap across registered perturbation families;
6. compare both estimators on identical complete cases against independent
   80-probe operational outcomes;
7. audit sensitivity to the preregistered candidate-output contract.

The theorem does not claim semantic correctness or internal causal diagnosis.
It establishes that conditioning on successful candidate observation changes
the stability estimand in a predictable and measurable direction.
