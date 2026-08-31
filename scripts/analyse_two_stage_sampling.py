from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyse_complete_case_optimism import (  # noqa: E402
    DATASETS,
    DATASET_LABELS,
    MODELS,
    MODEL_LABELS,
    CompactRecord,
    load_compact_trace,
)
from src.two_stage_sampling import (  # noqa: E402
    crossover_probe_count,
    design_effect,
    effective_independent_trials,
    estimate_variance_components,
    optimal_probe_count,
    probe_variance_share,
    probes_for_variance_share,
    required_unit_count,
    two_stage_hoeffding_radius,
    variance_of_mean,
)


PROBE_BUDGETS = (1, 2, 5, 10, 20, 40, 80)
TARGET_HALF_WIDTHS = (0.01, 0.02, 0.03, 0.05)
COST_RATIOS = (5, 10, 25, 50, 100, 250, 500, 1000)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    if frame.empty:
        return "_No observations._"

    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def full_manifest_matrix(records: dict[tuple[int, int], CompactRecord]) -> tuple[np.ndarray, np.ndarray]:
    ordered = list(records.values())
    probe_counts = {len(record.reference_operational) for record in ordered if record.eligible}
    if probe_counts != {80}:
        raise ValueError(f"expected exactly 80 reference probes for eligible samples, got {probe_counts}")
    matrix = np.zeros((len(ordered), 80), dtype=np.int8)
    image_ids = np.empty(len(ordered), dtype=np.int64)
    for index, record in enumerate(ordered):
        image_ids[index] = record.key[0]
        if record.eligible:
            matrix[index, :] = record.reference_operational
    return matrix, image_ids


def component_table(
    all_matrices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (dataset, model), (matrix, image_ids) in all_matrices.items():
        estimate = estimate_variance_components(matrix)
        theta = estimate.theta_hat
        marginal_variance = theta * (1.0 - theta)
        rho = estimate.between / (estimate.between + estimate.within) if estimate.between + estimate.within else 0.0
        cluster_se = math.sqrt(variance_of_mean(estimate.between, estimate.within, len(matrix), 80))
        naive_se = math.sqrt(marginal_variance / (len(matrix) * 80))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "pair_count": len(matrix),
                "unique_image_count": len(np.unique(image_ids)),
                "queries_per_image": len(matrix) / len(np.unique(image_ids)),
                "probe_count": 80,
                "theta_hat": theta,
                "between_A_raw": estimate.between_raw,
                "between_A": estimate.between,
                "within_B": estimate.within,
                "A_plus_B": estimate.between + estimate.within,
                "bernoulli_variance": marginal_variance,
                "icc_rho": rho,
                "crossover_R": crossover_probe_count(estimate.between, estimate.within),
                "probe_share_R80": probe_variance_share(estimate.between, estimate.within, 80),
                "design_effect_R80": design_effect(estimate.between, estimate.within, 80),
                "effective_trials_R80": effective_independent_trials(
                    estimate.between, estimate.within, len(matrix), 80
                ),
                "raw_probe_trials": len(matrix) * 80,
                "cluster_se_R80": cluster_se,
                "naive_independent_se": naive_se,
                "se_understatement_factor": cluster_se / naive_se if naive_se > 0 else math.nan,
                "hoeffding_radius_95": two_stage_hoeffding_radius(len(matrix), 80, 0.05),
            }
        )
    return pd.DataFrame(rows)


def budget_table(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component in components.itertuples(index=False):
        previous_variance = None
        for probe_count in PROBE_BUDGETS:
            between_term = component.between_A / component.pair_count
            probe_term = component.within_B / (component.pair_count * probe_count)
            total_variance = between_term + probe_term
            reduction = (
                1.0 - total_variance / previous_variance
                if previous_variance is not None and previous_variance > 0
                else math.nan
            )
            rows.append(
                {
                    "dataset": component.dataset,
                    "model": component.model,
                    "pair_count": component.pair_count,
                    "probe_count": probe_count,
                    "between_variance_term": between_term,
                    "probe_variance_term": probe_term,
                    "total_variance": total_variance,
                    "standard_error": math.sqrt(total_variance),
                    "probe_variance_share": probe_variance_share(
                        component.between_A, component.within_B, probe_count
                    ),
                    "design_effect": design_effect(
                        component.between_A, component.within_B, probe_count
                    ),
                    "effective_trials": effective_independent_trials(
                        component.between_A,
                        component.within_B,
                        component.pair_count,
                        probe_count,
                    ),
                    "variance_reduction_from_previous_budget": reduction,
                }
            )
            previous_variance = total_variance
    return pd.DataFrame(rows)


def planning_tables(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows = []
    allocation_rows = []
    for component in components.itertuples(index=False):
        for probe_count in PROBE_BUDGETS:
            for half_width in TARGET_HALF_WIDTHS:
                sample_rows.append(
                    {
                        "dataset": component.dataset,
                        "model": component.model,
                        "probe_count": probe_count,
                        "target_half_width_95": half_width,
                        "required_pair_count": required_unit_count(
                            component.between_A,
                            component.within_B,
                            probe_count,
                            half_width,
                        ),
                    }
                )
        for ratio in COST_RATIOS:
            optimum = optimal_probe_count(
                component.between_A,
                component.within_B,
                float(ratio),
                1.0,
                minimum=1,
                maximum=80,
            )
            continuous = (
                math.sqrt(component.within_B * ratio / component.between_A)
                if component.between_A > 0
                else math.inf
            )
            allocation_rows.append(
                {
                    "dataset": component.dataset,
                    "model": component.model,
                    "unit_to_probe_cost_ratio": ratio,
                    "continuous_optimal_R": continuous,
                    "bounded_integer_optimal_R": optimum,
                    "probe_share_at_optimum": probe_variance_share(
                        component.between_A, component.within_B, optimum
                    ),
                }
            )
    return pd.DataFrame(sample_rows), pd.DataFrame(allocation_rows)


def empirical_plugin_components(matrix: np.ndarray) -> tuple[float, float, float]:
    probabilities = matrix.mean(axis=1)
    theta = float(probabilities.mean())
    between = float(probabilities.var(ddof=0))
    within = float(np.mean(probabilities * (1.0 - probabilities)))
    return theta, between, within


def validate_by_nested_resampling(
    all_matrices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for group_index, ((dataset, model), (matrix, _)) in enumerate(all_matrices.items()):
        rng = np.random.default_rng(seed + group_index * 1009)
        probabilities = matrix.mean(axis=1)
        theta, between, within = empirical_plugin_components(matrix)
        unit_grid = sorted(set(value for value in (50, 100, 250, 500, len(matrix)) if value <= len(matrix)))
        for unit_count in unit_grid:
            for probe_count in PROBE_BUDGETS:
                estimates = np.empty(repetitions, dtype=float)
                for repetition in range(repetitions):
                    selected = rng.integers(0, len(probabilities), size=unit_count)
                    counts = rng.binomial(probe_count, probabilities[selected])
                    estimates[repetition] = float(counts.sum() / (unit_count * probe_count))
                empirical_variance = float(estimates.var(ddof=1))
                predicted_variance = variance_of_mean(between, within, unit_count, probe_count)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "unit_count": unit_count,
                        "probe_count": probe_count,
                        "repetitions": repetitions,
                        "plugin_theta": theta,
                        "plugin_between_A": between,
                        "plugin_within_B": within,
                        "empirical_mean": float(estimates.mean()),
                        "empirical_bias": float(estimates.mean() - theta),
                        "empirical_variance": empirical_variance,
                        "predicted_variance": predicted_variance,
                        "variance_ratio_empirical_to_predicted": (
                            empirical_variance / predicted_variance if predicted_variance > 0 else math.nan
                        ),
                        "absolute_relative_variance_error": (
                            abs(empirical_variance - predicted_variance) / predicted_variance
                            if predicted_variance > 0
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_component_intervals(
    all_matrices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    components: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Nested percentile intervals for the estimand and variance components."""

    rows = []
    metric_names = (
        "theta_hat",
        "between_A",
        "within_B",
        "icc_rho",
        "crossover_R",
        "standard_error_R80",
    )
    for group_index, ((dataset, model), (matrix, _)) in enumerate(all_matrices.items()):
        rng = np.random.default_rng(seed + group_index * 2027)
        probabilities = matrix.mean(axis=1)
        unit_count, probe_count = matrix.shape
        draws = {name: np.empty(repetitions, dtype=float) for name in metric_names}
        for repetition in range(repetitions):
            selected = rng.integers(0, unit_count, size=unit_count)
            counts = rng.binomial(probe_count, probabilities[selected])
            unit_means = counts / probe_count
            within_variances = (
                probe_count / (probe_count - 1.0) * unit_means * (1.0 - unit_means)
            )
            within = float(within_variances.mean())
            between_raw = float(unit_means.var(ddof=1) - within / probe_count)
            between = max(0.0, between_raw)
            rho = between / (between + within) if between + within > 0 else 0.0
            draws["theta_hat"][repetition] = float(unit_means.mean())
            draws["between_A"][repetition] = between
            draws["within_B"][repetition] = within
            draws["icc_rho"][repetition] = rho
            draws["crossover_R"][repetition] = (
                within / between if between > 0 else math.inf
            )
            draws["standard_error_R80"][repetition] = math.sqrt(
                variance_of_mean(between, within, unit_count, probe_count)
            )
        point_row = components[
            (components["dataset"] == dataset) & (components["model"] == model)
        ].iloc[0]
        point_values = {
            "theta_hat": float(point_row["theta_hat"]),
            "between_A": float(point_row["between_A"]),
            "within_B": float(point_row["within_B"]),
            "icc_rho": float(point_row["icc_rho"]),
            "crossover_R": float(point_row["crossover_R"]),
            "standard_error_R80": float(point_row["cluster_se_R80"]),
        }
        for metric in metric_names:
            finite = draws[metric][np.isfinite(draws[metric])]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "point_estimate": point_values[metric],
                    "lower_95": float(np.quantile(finite, 0.025)),
                    "upper_95": float(np.quantile(finite, 0.975)),
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def save_figures(
    components: pd.DataFrame,
    budgets: pd.DataFrame,
    validation: pd.DataFrame,
    allocation: pd.DataFrame,
    output: Path,
) -> None:
    labels = [
        f"{DATASET_LABELS[row.dataset]}\n{MODEL_LABELS[row.model]}"
        for row in components.itertuples(index=False)
    ]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(x, components["between_A"], label="Between-unit component A", color="#315b7d")
    axis.bar(
        x,
        components["within_B"] / 80,
        bottom=components["between_A"],
        label="Finite-probe component B / 80",
        color="#ed9b40",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Variance of an 80-probe unit mean")
    axis.set_title("Two-stage variance decomposition")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "variance_component_decomposition.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for (dataset, model), group in budgets.groupby(["dataset", "model"], sort=False):
        label = f"{DATASET_LABELS[dataset]} / {MODEL_LABELS[model]}"
        axes[0].plot(group["probe_count"], group["standard_error"], marker="o", label=label)
        axes[1].plot(group["probe_count"], group["probe_variance_share"], marker="o", label=label)
    axes[0].set_ylabel("Estimated standard error")
    axes[0].set_title("Diminishing uncertainty with more probes")
    axes[1].set_ylabel("Finite-probe share of variance")
    axes[1].set_title("What additional probes can still reduce")
    for axis in axes:
        axis.set_xlabel("Probes per image-query pair")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "probe_budget_diminishing_returns.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 7))
    for (dataset, model), group in validation.groupby(["dataset", "model"], sort=False):
        axis.scatter(
            group["predicted_variance"],
            group["empirical_variance"],
            s=24,
            alpha=0.75,
            label=f"{DATASET_LABELS[dataset]} / {MODEL_LABELS[model]}",
        )
    positive = validation[["predicted_variance", "empirical_variance"]].to_numpy()
    lower = float(positive[positive > 0].min())
    upper = float(positive.max())
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", label="Exact agreement")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Predicted variance: A/N + B/(NR)")
    axis.set_ylabel("Nested-resampling variance")
    axis.set_title("Empirical verification of the exact variance equation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "predicted_vs_empirical_variance.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6))
    raw = components["raw_probe_trials"].to_numpy(float)
    effective = components["effective_trials_R80"].to_numpy(float)
    width = 0.36
    axis.bar(x - width / 2, raw, width=width, label="Raw N x R count", color="#b9c7d5")
    axis.bar(x + width / 2, effective, width=width, label="Effective independent trials", color="#315b7d")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Trial count")
    axis.set_title("Repeated probes are clustered, not independent")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "effective_sample_size.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6))
    for (dataset, model), group in allocation.groupby(["dataset", "model"], sort=False):
        axis.plot(
            group["unit_to_probe_cost_ratio"],
            group["bounded_integer_optimal_R"],
            marker="o",
            label=f"{DATASET_LABELS[dataset]} / {MODEL_LABELS[model]}",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Cost of one new pair / cost of one additional probe")
    axis.set_ylabel("Cost-optimal probes per pair (bounded at 80)")
    axis.set_title("Theory-guided allocation of a fixed evaluation budget")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "cost_optimal_probe_allocation.png", dpi=180)
    plt.close(fig)


def render_report(
    components: pd.DataFrame,
    budgets: pd.DataFrame,
    validation: pd.DataFrame,
    sample_sizes: pd.DataFrame,
    allocation: pd.DataFrame,
    component_intervals: pd.DataFrame,
    output: Path,
) -> None:
    display_components = components.copy()
    display_components["dataset"] = display_components["dataset"].map(DATASET_LABELS)
    display_components["model"] = display_components["model"].map(MODEL_LABELS)
    validation_summary = (
        validation.groupby(["dataset", "model"], as_index=False)
        .agg(
            scenarios=("predicted_variance", "size"),
            mean_absolute_bias=("empirical_bias", lambda values: float(np.mean(np.abs(values)))),
            median_variance_ratio=("variance_ratio_empirical_to_predicted", "median"),
            mean_absolute_relative_variance_error=("absolute_relative_variance_error", "mean"),
            maximum_absolute_relative_variance_error=("absolute_relative_variance_error", "max"),
        )
    )
    validation_summary["dataset"] = validation_summary["dataset"].map(DATASET_LABELS)
    validation_summary["model"] = validation_summary["model"].map(MODEL_LABELS)
    budget_80 = budgets[budgets["probe_count"] == 80].copy()
    budget_80["dataset"] = budget_80["dataset"].map(DATASET_LABELS)
    budget_80["model"] = budget_80["model"].map(MODEL_LABELS)
    planning_20 = sample_sizes[
        (sample_sizes["probe_count"] == 20)
        & (sample_sizes["target_half_width_95"].isin([0.02, 0.03]))
    ].copy()
    planning_20["dataset"] = planning_20["dataset"].map(DATASET_LABELS)
    planning_20["model"] = planning_20["model"].map(MODEL_LABELS)
    allocation_100 = allocation[allocation["unit_to_probe_cost_ratio"] == 100].copy()
    allocation_100["dataset"] = allocation_100["dataset"].map(DATASET_LABELS)
    allocation_100["model"] = allocation_100["model"].map(MODEL_LABELS)
    interval_display = component_intervals[
        component_intervals["metric"].isin(["theta_hat", "between_A", "within_B", "icc_rho"])
    ].copy()
    interval_display["dataset"] = interval_display["dataset"].map(DATASET_LABELS)
    interval_display["model"] = interval_display["model"].map(MODEL_LABELS)

    median_error = float(validation["absolute_relative_variance_error"].median())
    mean_error = float(validation["absolute_relative_variance_error"].mean())
    max_bias = float(validation["empirical_bias"].abs().max())
    lines = [
        "# Two-Stage Model-Level Stability Analysis",
        "",
        "## Question",
        "",
        "How accurately can a finite number of image-query pairs and a finite number of "
        "registered probes estimate the stability of an entire grounding model on a target "
        "data and probe distribution?",
        "",
        "## Mathematical result",
        "",
        "For between-pair heterogeneity `A`, within-pair probe uncertainty `B`, `N` sampled "
        "pairs, and `R` probes per pair, the exact variance is:",
        "",
        "$$",
        "\\operatorname{Var}(\\widehat{\\Theta}_m)",
        "=\\frac{A_m}{N}+\\frac{B_m}{NR}.",
        "$$",
        "",
        "The first term is a variance floor with respect to additional probes. Only more "
        "independent image-query units reduce it. The complete proof, assumptions, concentration "
        "bounds, and allocation theorem are in `docs/methodology/two_stage_model_stability_theory.md`.",
        "",
        "## Estimated components from all frozen traces",
        "",
        markdown_table(
            display_components[
                [
                    "dataset",
                    "model",
                    "pair_count",
                    "unique_image_count",
                    "theta_hat",
                    "between_A",
                    "within_B",
                    "icc_rho",
                    "crossover_R",
                    "design_effect_R80",
                    "effective_trials_R80",
                    "se_understatement_factor",
                ]
            ]
        ),
        "",
        "`A` measures genuine between-pair heterogeneity. `B` measures remaining probe "
        "volatility for a fixed pair. The intraclass correlation quantifies dependence between "
        "two probes sharing the same pair. The standard-error understatement factor compares "
        "the correct cluster-based uncertainty with the incorrect assumption that all `N x R` "
        "probe outcomes are independent.",
        "",
        "## Nested-bootstrap uncertainty of the components",
        "",
        markdown_table(
            interval_display[
                ["dataset", "model", "metric", "point_estimate", "lower_95", "upper_95"]
            ]
        ),
        "",
        "These intervals resample image-query units and then probe outcomes within each selected "
        "unit. They quantify finite outer- and inner-stage uncertainty under the empirical probe law.",
        "",
        "## Observed 80-probe design",
        "",
        markdown_table(
            budget_80[
                [
                    "dataset",
                    "model",
                    "pair_count",
                    "standard_error",
                    "probe_variance_share",
                    "design_effect",
                    "effective_trials",
                ]
            ]
        ),
        "",
        "## Verification by nested resampling",
        "",
        f"The analysis evaluated {len(validation)} combinations of dataset, model, image budget, "
        f"and probe budget. Each combination used independent nested resampling. The median "
        f"absolute relative difference between empirical and predicted variance was {median_error:.3%}; "
        f"the mean difference was {mean_error:.3%}. The maximum absolute Monte Carlo bias of the "
        f"model-level mean was {max_bias:.6f}.",
        "",
        markdown_table(validation_summary),
        "",
        "This is an empirical check of the variance equation under the finite empirical probe "
        "law represented by the 80 frozen outcomes. It does not assume that the empirical probe "
        "law exhausts every real-world perturbation.",
        "",
        "## Example sample-size planning at 20 probes",
        "",
        markdown_table(planning_20),
        "",
        "## Example cost allocation",
        "",
        "For illustration, the following table assumes that acquiring and processing one new "
        "image-query pair costs 100 times one additional probe inference. The formula can be "
        "recomputed for any engineering cost ratio.",
        "",
        markdown_table(
            allocation_100[
                [
                    "dataset",
                    "model",
                    "unit_to_probe_cost_ratio",
                    "continuous_optimal_R",
                    "bounded_integer_optimal_R",
                    "probe_share_at_optimum",
                ]
            ]
        ),
        "",
        "## Main conclusions",
        "",
        "1. Model-level stability can be estimated without treating repeated probes as independent data.",
        "2. The exact variance has separate data-sampling and probe-sampling components.",
        "3. Additional probes have a measurable variance floor; additional pairs reduce both components.",
        "4. The intraclass correlation, design effect, and effective sample size quantify how much "
        "information is lost by repeated probing of the same pair.",
        "5. Estimated variance components support explicit sample-size and cost-optimal budget decisions.",
        "6. The same theory and frozen analysis apply to all three tested model families because "
        "the observable event does not compare raw confidence scales.",
        "",
        "## Interpretation boundaries",
        "",
        "- The target is operational candidate-order stability under the frozen distributions `P` and `Q`, not correctness.",
        "- The primary iid unit is an image-query pair. Shared images require image-cluster robust inference.",
        "- Outcome-dependent early stopping is not covered by the unbiased balanced-design theorem.",
        "- The 80-probe traces estimate, rather than eliminate, uncertainty about the full probe distribution.",
    ]
    (output / "two_stage_sampling_theory_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "two_stage_sampling_analysis",
    )
    parser.add_argument("--resampling-repetitions", type=int, default=1000)
    parser.add_argument("--component-bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_matrices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    input_files = []
    for dataset, root in DATASETS.items():
        for model in MODELS:
            path = root / model / "sample_traces.jsonl.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"Loading {dataset}/{model}", flush=True)
            records = load_compact_trace(path)
            all_matrices[(dataset, model)] = full_manifest_matrix(records)
            input_files.append(path)

    print("Estimating two-stage variance components", flush=True)
    components = component_table(all_matrices)
    budgets = budget_table(components)
    sample_sizes, allocation = planning_tables(components)
    print("Running nested image/probe resampling validation", flush=True)
    validation = validate_by_nested_resampling(
        all_matrices, args.resampling_repetitions, args.seed
    )
    print("Computing nested-bootstrap component intervals", flush=True)
    component_intervals = bootstrap_component_intervals(
        all_matrices,
        components,
        args.component_bootstrap_repetitions,
        args.seed + 50001,
    )

    components.to_csv(args.output / "variance_component_estimates.csv", index=False)
    budgets.to_csv(args.output / "variance_budget_table.csv", index=False)
    sample_sizes.to_csv(args.output / "sample_size_planning.csv", index=False)
    allocation.to_csv(args.output / "allocation_tradeoff.csv", index=False)
    validation.to_csv(args.output / "subsampling_variance_validation.csv", index=False)
    component_intervals.to_csv(args.output / "component_bootstrap_intervals.csv", index=False)
    save_figures(components, budgets, validation, allocation, args.output)
    render_report(
        components,
        budgets,
        validation,
        sample_sizes,
        allocation,
        component_intervals,
        args.output,
    )

    artifacts = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name not in {"analysis_audit.json", "artifact_manifest.json"}:
            artifacts.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    (args.output / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2) + "\n", encoding="utf-8"
    )

    audit = {
        "status": "complete",
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "resampling_repetitions": args.resampling_repetitions,
        "component_bootstrap_repetitions": args.component_bootstrap_repetitions,
        "seed": args.seed,
        "validation_scenario_count": len(validation),
        "median_absolute_relative_variance_error": float(
            validation["absolute_relative_variance_error"].median()
        ),
        "mean_absolute_relative_variance_error": float(
            validation["absolute_relative_variance_error"].mean()
        ),
        "maximum_absolute_empirical_bias": float(validation["empirical_bias"].abs().max()),
        "minimum_between_component_raw": float(components["between_A_raw"].min()),
        "input_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in input_files
        },
        "outputs": sorted(path.name for path in args.output.iterdir() if path.is_file()),
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
