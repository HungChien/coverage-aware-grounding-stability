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
| RefCOCO | YOLO-World | 500 | 500 | 0.51177 | 0.13989 | 0.11026 | 0.55922 | 0.78819 | 45.17874 | 885.37219 | 6.72531 |
| RefCOCO+ | GroundingDINO | 1000 | 1000 | 0.85116 | 0.03540 | 0.09132 | 0.27935 | 2.57979 | 23.06834 | 3467.95723 | 4.80364 |
| RefCOCO+ | YOLO-World | 1000 | 1000 | 0.50177 | 0.12944 | 0.12069 | 0.51749 | 0.93239 | 41.88199 | 1910.12912 | 6.47332 |
| Ref-L4 | GroundingDINO | 1000 | 1000 | 0.87031 | 0.03447 | 0.07843 | 0.30531 | 2.27533 | 25.11969 | 3184.75300 | 5.01274 |
| Ref-L4 | YOLO-World | 1000 | 1000 | 0.53633 | 0.12734 | 0.12147 | 0.51179 | 0.95391 | 41.43171 | 1930.88797 | 6.43841 |

`A` measures genuine between-pair heterogeneity. `B` measures remaining probe volatility for a fixed pair. The intraclass correlation quantifies dependence between two probes sharing the same pair. The standard-error understatement factor compares the correct cluster-based uncertainty with the incorrect assumption that all `N x R` probe outcomes are independent.

## Nested-bootstrap uncertainty of the components

| dataset | model | metric | point_estimate | lower_95 | upper_95 |
| --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | theta_hat | 0.85537 | 0.83787 | 0.87250 |
| RefCOCO | GroundingDINO | between_A | 0.03704 | 0.03166 | 0.04487 |
| RefCOCO | GroundingDINO | within_B | 0.08675 | 0.07727 | 0.09449 |
| RefCOCO | GroundingDINO | icc_rho | 0.29921 | 0.27245 | 0.34390 |
| RefCOCO | YOLO-World | theta_hat | 0.51177 | 0.47752 | 0.54456 |
| RefCOCO | YOLO-World | between_A | 0.13989 | 0.13136 | 0.15035 |
| RefCOCO | YOLO-World | within_B | 0.11026 | 0.09974 | 0.11830 |
| RefCOCO | YOLO-World | icc_rho | 0.55922 | 0.52619 | 0.60125 |
| RefCOCO+ | GroundingDINO | theta_hat | 0.85116 | 0.84004 | 0.86358 |
| RefCOCO+ | GroundingDINO | between_A | 0.03540 | 0.03221 | 0.04088 |
| RefCOCO+ | GroundingDINO | within_B | 0.09132 | 0.08408 | 0.09575 |
| RefCOCO+ | GroundingDINO | icc_rho | 0.27935 | 0.26655 | 0.31014 |
| RefCOCO+ | YOLO-World | theta_hat | 0.50177 | 0.47790 | 0.52430 |
| RefCOCO+ | YOLO-World | between_A | 0.12944 | 0.12425 | 0.13756 |
| RefCOCO+ | YOLO-World | within_B | 0.12069 | 0.11250 | 0.12572 |
| RefCOCO+ | YOLO-World | icc_rho | 0.51749 | 0.49706 | 0.54993 |
| Ref-L4 | GroundingDINO | theta_hat | 0.87031 | 0.85829 | 0.88200 |
| Ref-L4 | GroundingDINO | between_A | 0.03447 | 0.03109 | 0.03987 |
| Ref-L4 | GroundingDINO | within_B | 0.07843 | 0.07179 | 0.08326 |
| Ref-L4 | GroundingDINO | icc_rho | 0.30531 | 0.28952 | 0.33825 |
| Ref-L4 | YOLO-World | theta_hat | 0.53633 | 0.51367 | 0.55908 |
| Ref-L4 | YOLO-World | between_A | 0.12734 | 0.12211 | 0.13547 |
| Ref-L4 | YOLO-World | within_B | 0.12147 | 0.11362 | 0.12637 |
| Ref-L4 | YOLO-World | icc_rho | 0.51179 | 0.49178 | 0.54409 |

These intervals resample image-query units and then probe outcomes within each selected unit. They quantify finite outer- and inner-stage uncertainty under the empirical probe law.

## Observed 80-probe design

| dataset | model | pair_count | standard_error | probe_variance_share | design_effect | effective_trials |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 500 | 0.00873 | 0.02844 | 24.63721 | 1623.56058 |
| RefCOCO | YOLO-World | 500 | 0.01681 | 0.00976 | 45.17874 | 885.37219 |
| RefCOCO+ | GroundingDINO | 1000 | 0.00604 | 0.03124 | 23.06834 | 3467.95723 |
| RefCOCO+ | YOLO-World | 1000 | 0.01144 | 0.01152 | 41.88199 | 1910.12912 |
| Ref-L4 | GroundingDINO | 1000 | 0.00595 | 0.02766 | 25.11969 | 3184.75300 |
| Ref-L4 | YOLO-World | 1000 | 0.01135 | 0.01178 | 41.43171 | 1930.88797 |

## Verification by nested resampling

The analysis evaluated 196 combinations of dataset, model, image budget, and probe budget. Each combination used independent nested resampling. The median absolute relative difference between empirical and predicted variance was 3.007%; the mean difference was 3.603%. The maximum absolute Monte Carlo bias of the model-level mean was 0.003427.

| dataset | model | scenarios | mean_absolute_bias | median_variance_ratio | mean_absolute_relative_variance_error | maximum_absolute_relative_variance_error |
| --- | --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 28 | 0.00045 | 1.01250 | 0.03875 | 0.12776 |
| RefCOCO | YOLO-World | 28 | 0.00092 | 0.99676 | 0.04222 | 0.10207 |
| RefCOCO+ | GroundingDINO | 35 | 0.00051 | 1.00308 | 0.03853 | 0.15418 |
| RefCOCO+ | YOLO-World | 35 | 0.00089 | 0.99964 | 0.03294 | 0.12107 |
| Ref-L4 | GroundingDINO | 35 | 0.00041 | 1.01462 | 0.02945 | 0.10104 |
| Ref-L4 | YOLO-World | 35 | 0.00085 | 0.98567 | 0.03608 | 0.09174 |

This is an empirical check of the variance equation under the finite empirical probe law represented by the 80 frozen outcomes. It does not assume that the empirical probe law exhausts every real-world perturbation.

## Example sample-size planning at 20 probes

| dataset | model | probe_count | target_half_width_95 | required_pair_count |
| --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 20 | 0.02000 | 398 |
| RefCOCO | GroundingDINO | 20 | 0.03000 | 177 |
| RefCOCO | YOLO-World | 20 | 0.02000 | 1397 |
| RefCOCO | YOLO-World | 20 | 0.03000 | 621 |
| RefCOCO+ | GroundingDINO | 20 | 0.02000 | 384 |
| RefCOCO+ | GroundingDINO | 20 | 0.03000 | 171 |
| RefCOCO+ | YOLO-World | 20 | 0.02000 | 1302 |
| RefCOCO+ | YOLO-World | 20 | 0.03000 | 579 |
| Ref-L4 | GroundingDINO | 20 | 0.02000 | 369 |
| Ref-L4 | GroundingDINO | 20 | 0.03000 | 164 |
| Ref-L4 | YOLO-World | 20 | 0.02000 | 1282 |
| Ref-L4 | YOLO-World | 20 | 0.03000 | 570 |

## Example cost allocation

For illustration, the following table assumes that acquiring and processing one new image-query pair costs 100 times one additional probe inference. The formula can be recomputed for any engineering cost ratio.

| dataset | model | unit_to_probe_cost_ratio | continuous_optimal_R | bounded_integer_optimal_R | probe_share_at_optimum |
| --- | --- | --- | --- | --- | --- |
| RefCOCO | GroundingDINO | 100 | 15.30421 | 15 | 0.13506 |
| RefCOCO | YOLO-World | 100 | 8.87801 | 9 | 0.08052 |
| RefCOCO+ | GroundingDINO | 100 | 16.06172 | 16 | 0.13885 |
| RefCOCO+ | YOLO-World | 100 | 9.65604 | 10 | 0.08529 |
| Ref-L4 | GroundingDINO | 100 | 15.08421 | 15 | 0.13171 |
| Ref-L4 | YOLO-World | 100 | 9.76684 | 10 | 0.08708 |

## Main conclusions

1. Model-level stability can be estimated without treating repeated probes as independent data.
2. The exact variance has separate data-sampling and probe-sampling components.
3. Additional probes have a measurable variance floor; additional pairs reduce both components.
4. The intraclass correlation, design effect, and effective sample size quantify how much information is lost by repeated probing of the same pair.
5. Estimated variance components support explicit sample-size and cost-optimal budget decisions.
6. The same theory and frozen analysis apply to both grounding architectures because the observable operational event is architecture agnostic.

## Interpretation boundaries

- The target is operational candidate-order stability under the frozen distributions `P` and `Q`, not correctness.
- The primary iid unit is an image-query pair. Shared images require image-cluster robust inference.
- Outcome-dependent early stopping is not covered by the unbiased balanced-design theorem.
- The 80-probe traces estimate, rather than eliminate, uncertainty about the full probe distribution.
