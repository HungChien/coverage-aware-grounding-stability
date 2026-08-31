# Two-Stage Model-Level Stability Analysis

## Question

How accurately can a finite number of image-query pairs and a finite number of registered probes estimate the stability of an entire grounding model on a target data and probe distribution?

## Mathematical result

For between-pair heterogeneity `A`, within-pair probe uncertainty `B`, `N` sampled pairs, and `R` probes per pair, the exact variance is:

$$
\operatorname{Var}(\widehat{\Theta}_m)
=\frac{A_m}{N}+\frac{B_m}{NR}.
$$

The first term is a variance floor with respect to additional probes. Only more independent image-query units reduce it. The complete proof, assumptions, concentration bounds, and allocation theorem are in `docs/methodology/two_stage_model_stability_theory.md`.

## Estimated components from all frozen traces

| dataset | model | pair_count | unique_image_count | theta_hat | between_A | within_B | icc_rho | crossover_R | design_effect_R80 | effective_trials_R80 | se_understatement_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 500 | 500 | 0.85537 | 0.03704 | 0.08675 | 0.29921 | 2.34219 | 24.63721 | 1623.56058 | 4.96512 |
| RefCOCO | OWLv2 | 500 | 500 | 0.73947 | 0.06491 | 0.12787 | 0.33672 | 1.96987 | 27.60050 | 1449.24927 | 5.25543 |
| RefCOCO | YOLO-World | 500 | 500 | 0.51177 | 0.13989 | 0.11026 | 0.55922 | 0.78819 | 45.17874 | 885.37219 | 6.72531 |
| RefCOCO+ | GroundingDINO | 1000 | 1000 | 0.85116 | 0.03540 | 0.09132 | 0.27935 | 2.57979 | 23.06834 | 3467.95723 | 4.80364 |
| RefCOCO+ | OWLv2 | 1000 | 1000 | 0.73319 | 0.07134 | 0.12436 | 0.36454 | 1.74321 | 29.79834 | 2684.71295 | 5.45980 |
| RefCOCO+ | YOLO-World | 1000 | 1000 | 0.50177 | 0.12944 | 0.12069 | 0.51749 | 0.93239 | 41.88199 | 1910.12912 | 6.47332 |
| Ref-L4 | GroundingDINO | 1000 | 1000 | 0.87031 | 0.03447 | 0.07843 | 0.30531 | 2.27533 | 25.11969 | 3184.75300 | 5.01274 |
| Ref-L4 | OWLv2 | 1000 | 1000 | 0.76769 | 0.05948 | 0.11892 | 0.33343 | 1.99917 | 27.34063 | 2926.04851 | 5.22972 |
| Ref-L4 | YOLO-World | 1000 | 1000 | 0.53633 | 0.12734 | 0.12147 | 0.51179 | 0.95391 | 41.43171 | 1930.88797 | 6.43841 |

`A` measures genuine between-pair heterogeneity. `B` measures remaining probe volatility for a fixed pair. The intraclass correlation quantifies dependence between two probes sharing the same pair. The standard-error understatement factor compares the correct cluster-based uncertainty with the incorrect assumption that all `N x R` probe outcomes are independent.

## Nested-bootstrap uncertainty of the components

| dataset | model | metric | point_estimate | lower_95 | upper_95 |
| --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | theta_hat | 0.85537 | 0.83787 | 0.87250 |
| RefCOCO | GroundingDINO | between_A | 0.03704 | 0.03166 | 0.04487 |
| RefCOCO | GroundingDINO | within_B | 0.08675 | 0.07727 | 0.09449 |
| RefCOCO | GroundingDINO | icc_rho | 0.29921 | 0.27245 | 0.34390 |
| RefCOCO | OWLv2 | theta_hat | 0.73947 | 0.71647 | 0.76258 |
| RefCOCO | OWLv2 | between_A | 0.06491 | 0.05752 | 0.07557 |
| RefCOCO | OWLv2 | within_B | 0.12787 | 0.11780 | 0.13445 |
| RefCOCO | OWLv2 | icc_rho | 0.33672 | 0.30895 | 0.37905 |
| RefCOCO | YOLO-World | theta_hat | 0.51177 | 0.47892 | 0.54593 |
| RefCOCO | YOLO-World | between_A | 0.13989 | 0.13141 | 0.15116 |
| RefCOCO | YOLO-World | within_B | 0.11026 | 0.09908 | 0.11835 |
| RefCOCO | YOLO-World | icc_rho | 0.55922 | 0.52644 | 0.60411 |
| RefCOCO+ | GroundingDINO | theta_hat | 0.85116 | 0.83935 | 0.86252 |
| RefCOCO+ | GroundingDINO | between_A | 0.03540 | 0.03231 | 0.04106 |
| RefCOCO+ | GroundingDINO | within_B | 0.09132 | 0.08453 | 0.09605 |
| RefCOCO+ | GroundingDINO | icc_rho | 0.27935 | 0.26533 | 0.31173 |
| RefCOCO+ | OWLv2 | theta_hat | 0.73319 | 0.71621 | 0.74973 |
| RefCOCO+ | OWLv2 | between_A | 0.07134 | 0.06588 | 0.07974 |
| RefCOCO+ | OWLv2 | within_B | 0.12436 | 0.11681 | 0.12883 |
| RefCOCO+ | OWLv2 | icc_rho | 0.36454 | 0.34463 | 0.39844 |
| RefCOCO+ | YOLO-World | theta_hat | 0.50177 | 0.47958 | 0.52402 |
| RefCOCO+ | YOLO-World | between_A | 0.12944 | 0.12417 | 0.13717 |
| RefCOCO+ | YOLO-World | within_B | 0.12069 | 0.11288 | 0.12575 |
| RefCOCO+ | YOLO-World | icc_rho | 0.51749 | 0.49664 | 0.54861 |
| Ref-L4 | GroundingDINO | theta_hat | 0.87031 | 0.85770 | 0.88261 |
| Ref-L4 | GroundingDINO | between_A | 0.03447 | 0.03103 | 0.04009 |
| Ref-L4 | GroundingDINO | within_B | 0.07843 | 0.07139 | 0.08338 |
| Ref-L4 | GroundingDINO | icc_rho | 0.30531 | 0.29042 | 0.33879 |
| Ref-L4 | OWLv2 | theta_hat | 0.76769 | 0.75252 | 0.78336 |
| Ref-L4 | OWLv2 | between_A | 0.05948 | 0.05528 | 0.06635 |
| Ref-L4 | OWLv2 | within_B | 0.11892 | 0.11162 | 0.12299 |
| Ref-L4 | OWLv2 | icc_rho | 0.33343 | 0.31849 | 0.36348 |
| Ref-L4 | YOLO-World | theta_hat | 0.53633 | 0.51392 | 0.55865 |
| Ref-L4 | YOLO-World | between_A | 0.12734 | 0.12206 | 0.13563 |
| Ref-L4 | YOLO-World | within_B | 0.12147 | 0.11364 | 0.12623 |
| Ref-L4 | YOLO-World | icc_rho | 0.51179 | 0.49198 | 0.54427 |

These intervals resample image-query units and then probe outcomes within each selected unit. They quantify finite outer- and inner-stage uncertainty under the empirical probe law.

## Observed 80-probe design

| dataset | model | pair_count | standard_error | probe_variance_share | design_effect | effective_trials |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 500 | 0.00873 | 0.02844 | 24.63721 | 1623.56058 |
| RefCOCO | OWLv2 | 500 | 0.01153 | 0.02403 | 27.60050 | 1449.24927 |
| RefCOCO | YOLO-World | 500 | 0.01681 | 0.00976 | 45.17874 | 885.37219 |
| RefCOCO+ | GroundingDINO | 1000 | 0.00604 | 0.03124 | 23.06834 | 3467.95723 |
| RefCOCO+ | OWLv2 | 1000 | 0.00854 | 0.02133 | 29.79834 | 2684.71295 |
| RefCOCO+ | YOLO-World | 1000 | 0.01144 | 0.01152 | 41.88199 | 1910.12912 |
| Ref-L4 | GroundingDINO | 1000 | 0.00595 | 0.02766 | 25.11969 | 3184.75300 |
| Ref-L4 | OWLv2 | 1000 | 0.00781 | 0.02438 | 27.34063 | 2926.04851 |
| Ref-L4 | YOLO-World | 1000 | 0.01135 | 0.01178 | 41.43171 | 1930.88797 |

## Verification by nested resampling

The analysis evaluated 294 combinations of dataset, model, image budget, and probe budget. Each combination used independent nested resampling. The median absolute relative difference between empirical and predicted variance was 3.090%; the mean difference was 3.779%. The maximum absolute Monte Carlo bias of the model-level mean was 0.003616.

| dataset | model | scenarios | mean_absolute_bias | median_variance_ratio | mean_absolute_relative_variance_error | maximum_absolute_relative_variance_error |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 28 | 0.00045 | 1.01250 | 0.03875 | 0.12776 |
| RefCOCO | OWLv2 | 28 | 0.00070 | 0.99921 | 0.03785 | 0.09090 |
| RefCOCO | YOLO-World | 28 | 0.00080 | 1.01451 | 0.03915 | 0.14058 |
| RefCOCO+ | GroundingDINO | 35 | 0.00039 | 1.00718 | 0.03174 | 0.10076 |
| RefCOCO+ | OWLv2 | 35 | 0.00048 | 1.00412 | 0.03545 | 0.08835 |
| RefCOCO+ | YOLO-World | 35 | 0.00103 | 0.99909 | 0.04656 | 0.14525 |
| Ref-L4 | GroundingDINO | 35 | 0.00052 | 0.99094 | 0.02946 | 0.10247 |
| Ref-L4 | OWLv2 | 35 | 0.00063 | 0.98390 | 0.04274 | 0.13219 |
| Ref-L4 | YOLO-World | 35 | 0.00090 | 1.00758 | 0.03889 | 0.14622 |

This is an empirical check of the variance equation under the finite empirical probe law represented by the 80 frozen outcomes. It does not assume that the empirical probe law exhausts every real-world perturbation.

## Example sample-size planning at 20 probes

| dataset | model | probe_count | target_half_width_95 | required_pair_count |
| --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 20 | 0.02000 | 398 |
| RefCOCO | GroundingDINO | 20 | 0.03000 | 177 |
| RefCOCO | OWLv2 | 20 | 0.02000 | 685 |
| RefCOCO | OWLv2 | 20 | 0.03000 | 305 |
| RefCOCO | YOLO-World | 20 | 0.02000 | 1397 |
| RefCOCO | YOLO-World | 20 | 0.03000 | 621 |
| RefCOCO+ | GroundingDINO | 20 | 0.02000 | 384 |
| RefCOCO+ | GroundingDINO | 20 | 0.03000 | 171 |
| RefCOCO+ | OWLv2 | 20 | 0.02000 | 745 |
| RefCOCO+ | OWLv2 | 20 | 0.03000 | 332 |
| RefCOCO+ | YOLO-World | 20 | 0.02000 | 1302 |
| RefCOCO+ | YOLO-World | 20 | 0.03000 | 579 |
| Ref-L4 | GroundingDINO | 20 | 0.02000 | 369 |
| Ref-L4 | GroundingDINO | 20 | 0.03000 | 164 |
| Ref-L4 | OWLv2 | 20 | 0.02000 | 629 |
| Ref-L4 | OWLv2 | 20 | 0.03000 | 280 |
| Ref-L4 | YOLO-World | 20 | 0.02000 | 1282 |
| Ref-L4 | YOLO-World | 20 | 0.03000 | 570 |

## Example cost allocation

For illustration, the following table assumes that acquiring and processing one new image-query pair costs 100 times one additional probe inference. The formula can be recomputed for any engineering cost ratio.

| dataset | model | unit_to_probe_cost_ratio | continuous_optimal_R | bounded_integer_optimal_R | probe_share_at_optimum |
| --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 100 | 15.30421 | 15 | 0.13506 |
| RefCOCO | OWLv2 | 100 | 14.03520 | 14 | 0.12335 |
| RefCOCO | YOLO-World | 100 | 8.87801 | 9 | 0.08052 |
| RefCOCO+ | GroundingDINO | 100 | 16.06172 | 16 | 0.13885 |
| RefCOCO+ | OWLv2 | 100 | 13.20308 | 13 | 0.11824 |
| RefCOCO+ | YOLO-World | 100 | 9.65604 | 10 | 0.08529 |
| Ref-L4 | GroundingDINO | 100 | 15.08421 | 15 | 0.13171 |
| Ref-L4 | OWLv2 | 100 | 14.13920 | 14 | 0.12495 |
| Ref-L4 | YOLO-World | 100 | 9.76684 | 10 | 0.08708 |

## Main conclusions

1. Model-level stability can be estimated without treating repeated probes as independent data.
2. The exact variance has separate data-sampling and probe-sampling components.
3. Additional probes have a measurable variance floor; additional pairs reduce both components.
4. The intraclass correlation, design effect, and effective sample size quantify how much information is lost by repeated probing of the same pair.
5. Estimated variance components support explicit sample-size and cost-optimal budget decisions.
6. The same theory and frozen analysis apply to all three tested model families because the observable event does not compare raw confidence scales.

## Interpretation boundaries

- The target is operational candidate-order stability under the frozen distributions `P` and `Q`, not correctness.
- The primary iid unit is an image-query pair. Shared images require image-cluster robust inference.
- Outcome-dependent early stopping is not covered by the unbiased balanced-design theorem.
- The 80-probe traces estimate, rather than eliminate, uncertainty about the full probe distribution.
