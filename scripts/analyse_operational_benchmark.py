from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


MODEL_LABELS = {
    "groundingdino": "GroundingDINO",
    "yoloworld": "YOLO-World",
    "owlv2": "OWLv2",
}
CAUSE_ORDER = [
    "winner_missing",
    "competitor_missing",
    "threatening_birth",
    "ranking_reversal",
]


def load_trace(path: Path) -> dict[tuple[int, int], dict]:
    """Load a resumable trace and keep the last complete record per sample."""

    records: dict[tuple[int, int], dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[(int(record["image_id"]), int(record["ref_id"]))] = record
    return records


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""

    if frame.empty:
        return "_No observations._"
    columns = [str(column) for column in frame.columns]

    def render(value) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def interval_width(successes: np.ndarray, trials: int) -> np.ndarray:
    # Wilson width is used only as a descriptive finite-budget precision curve;
    # the saved per-sample intervals remain exact Clopper-Pearson intervals.
    z = 1.959963984540054
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    half = (
        z
        * np.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return 2.0 * half


def evaluate_budget_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, budget), subset in frame.groupby(["model", "diagnostic_budget"]):
        diagnostic = subset["diagnostic_operational"].to_numpy(float)
        reference = subset["reference_operational"].to_numpy(float)
        eligible = subset["clean_eligible"].to_numpy(int).astype(bool)
        error = diagnostic - reference
        eligible_error = error[eligible]
        rows.append(
            {
                "model": model,
                "diagnostic_budget": int(budget),
                "sample_count": len(subset),
                "eligible_count": int(eligible.sum()),
                "clean_eligibility": float(eligible.mean()),
                "diagnostic_full_manifest_mean": float(diagnostic.mean()),
                "reference_full_manifest_mean": float(reference.mean()),
                "diagnostic_eligible_mean": (
                    float(diagnostic[eligible].mean()) if eligible.any() else math.nan
                ),
                "reference_eligible_mean": (
                    float(reference[eligible].mean()) if eligible.any() else math.nan
                ),
                "mae_all": float(np.abs(error).mean()),
                "rmse_all": float(np.sqrt(np.mean(error * error))),
                "bias_all": float(error.mean()),
                "spearman_all": safe_spearman(diagnostic, reference),
                "mae_eligible": (
                    float(np.abs(eligible_error).mean())
                    if eligible_error.size
                    else math.nan
                ),
                "spearman_eligible": (
                    safe_spearman(diagnostic[eligible], reference[eligible])
                    if eligible.any()
                    else math.nan
                ),
                # Clean-ineligible samples have no Bernoulli probe experiment
                # and therefore no Clopper--Pearson interval.  Interval
                # coverage is a conditional finite-probe diagnostic and must
                # only be evaluated on rows for which the interval exists.
                "reference_inside_diagnostic_cp_interval_eligible": (
                    float(
                        (
                            (
                                reference[eligible]
                                >= subset.loc[
                                    eligible, "diagnostic_operational_lower"
                                ].to_numpy(float)
                            )
                            & (
                                reference[eligible]
                                <= subset.loc[
                                    eligible, "diagnostic_operational_upper"
                                ].to_numpy(float)
                            )
                        ).mean()
                    )
                    if eligible.any()
                    else math.nan
                ),
                "mean_diagnostic_cp_width": float(
                    (
                        subset["diagnostic_operational_upper"]
                        - subset["diagnostic_operational_lower"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "diagnostic_budget"])


def selective_risk_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute tie-aware risk--coverage curves from finite-probe estimates."""

    curve_rows = []
    summary_rows = []
    for (model, budget), subset in frame.groupby(["model", "diagnostic_budget"]):
        values = subset[
            ["diagnostic_operational", "reference_operational"]
        ].copy()
        values["reference_risk"] = 1.0 - values["reference_operational"]
        thresholds = sorted(values["diagnostic_operational"].unique(), reverse=True)
        points = []
        for threshold in thresholds:
            retained = values[values["diagnostic_operational"] >= threshold]
            coverage = len(retained) / len(values)
            risk = float(retained["reference_risk"].mean())
            points.append((coverage, risk, float(threshold), len(retained)))
            curve_rows.append(
                {
                    "model": model,
                    "diagnostic_budget": int(budget),
                    "threshold": float(threshold),
                    "retained_samples": len(retained),
                    "coverage": coverage,
                    "reference_risk": risk,
                }
            )
        coverages = np.asarray([point[0] for point in points], dtype=float)
        risks = np.asarray([point[1] for point in points], dtype=float)
        if len(points) == 1:
            aurc = float(risks[0])
        else:
            # Include the origin at the risk of the highest-score tied group.
            aurc = float(
                np.trapezoid(
                    np.concatenate(([risks[0]], risks)),
                    np.concatenate(([0.0], coverages)),
                )
            )
        summary_rows.append(
            {
                "model": model,
                "diagnostic_budget": int(budget),
                "tie_aware_aurc": aurc,
                "overall_reference_risk": float(values["reference_risk"].mean()),
                "distinct_diagnostic_scores": len(thresholds),
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def compare_family_profiles(
    diagnostic: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    columns = ["model", "family", "operational_stability", "risk_share"]
    merged = diagnostic[columns].merge(
        reference[columns],
        on=["model", "family"],
        suffixes=("_diagnostic", "_reference"),
        validate="one_to_one",
    )
    merged["absolute_stability_error"] = (
        merged["operational_stability_diagnostic"]
        - merged["operational_stability_reference"]
    ).abs()
    merged["absolute_risk_share_error"] = (
        merged["risk_share_diagnostic"] - merged["risk_share_reference"]
    ).abs()
    return merged


def family_profiles_by_budget(
    traces: dict[str, dict[tuple[int, int], dict]],
    budgets: list[int],
    reference_family: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows = []
    for model, records in traces.items():
        for budget in budgets:
            counts: dict[str, Counter] = defaultdict(Counter)
            for record in records.values():
                if not record.get("clean_eligible"):
                    continue
                diagnostic = [
                    probe for probe in record["probes"]
                    if probe["split"] == "diagnostic"
                ][:budget]
                for probe in diagnostic:
                    family = probe["spec"]["family"]
                    counts[family]["trials"] += 1
                    counts[family]["stable"] += int(
                        probe["outcome"]["operational_stable"]
                    )
            total_trials = sum(value["trials"] for value in counts.values())
            temporary = []
            for family, values in counts.items():
                theta = values["stable"] / values["trials"]
                weight = values["trials"] / total_trials
                contribution = weight * (1.0 - theta)
                temporary.append((family, values["trials"], theta, contribution))
            total_risk = sum(item[3] for item in temporary)
            for family, trials, theta, contribution in temporary:
                profile_rows.append(
                    {
                        "model": model,
                        "diagnostic_budget": budget,
                        "family": family,
                        "trials": trials,
                        "operational_stability": theta,
                        "risk_share": 0.0 if total_risk == 0 else contribution / total_risk,
                    }
                )
    profiles = pd.DataFrame(profile_rows)
    reference = reference_family[
        ["model", "family", "operational_stability", "risk_share"]
    ].rename(
        columns={
            "operational_stability": "reference_operational_stability",
            "risk_share": "reference_risk_share",
        }
    )
    comparison = profiles.merge(
        reference, on=["model", "family"], validate="many_to_one"
    )
    comparison["absolute_stability_error"] = (
        comparison["operational_stability"]
        - comparison["reference_operational_stability"]
    ).abs()
    comparison["absolute_risk_share_error"] = (
        comparison["risk_share"] - comparison["reference_risk_share"]
    ).abs()
    summary = (
        comparison.groupby(["model", "diagnostic_budget"], as_index=False)
        .agg(
            mean_family_stability_error=("absolute_stability_error", "mean"),
            mean_family_risk_share_error=("absolute_risk_share_error", "mean"),
        )
        .sort_values(["model", "diagnostic_budget"])
    )
    return comparison, summary


def trace_diagnostics(
    traces: dict[str, dict[tuple[int, int], dict]], split: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    cause_counts: dict[str, Counter] = defaultdict(Counter)
    culprit_counts: dict[str, Counter] = defaultdict(Counter)
    quality: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for model, records in traces.items():
        for record in records.values():
            if not record.get("clean_eligible"):
                continue
            for probe in record["probes"]:
                if probe["split"] != split:
                    continue
                family = probe["spec"]["family"]
                outcome = probe["outcome"]
                family_counts[(model, family)]["trials"] += 1
                family_counts[(model, family)]["stable"] += int(
                    outcome["operational_stable"]
                )
                cause_counts[model][outcome["primary_failure"]] += 1
                if outcome["primary_failure"] == "ranking_reversal":
                    culprit_counts[model][str(outcome["culprit_clean_index"])] += 1
                quality[model]["coverage"].append(float(outcome["coverage"]))
                quality[model]["candidate_count"].append(
                    float(outcome["perturbed_candidate_count"])
                )
                quality[model]["matched_iou"].extend(
                    float(value)
                    for value in outcome["matched_ious"]
                    if value is not None
                )

    family_rows = []
    for (model, family), counts in family_counts.items():
        trials = counts["trials"]
        theta = counts["stable"] / trials
        family_rows.append(
            {
                "model": model,
                "family": family,
                "trials": trials,
                "operational_stability": theta,
                "family_risk": 1.0 - theta,
            }
        )
    family = pd.DataFrame(family_rows)
    for model, subset in family.groupby("model"):
        total = subset["trials"].sum()
        indices = subset.index
        family.loc[indices, "probe_weight"] = subset["trials"] / total
        family.loc[indices, "risk_contribution"] = (
            family.loc[indices, "probe_weight"]
            * family.loc[indices, "family_risk"]
        )
        total_risk = family.loc[indices, "risk_contribution"].sum()
        family.loc[indices, "risk_share"] = (
            0.0
            if total_risk == 0.0
            else family.loc[indices, "risk_contribution"] / total_risk
        )

    cause_rows = []
    for model, counts in cause_counts.items():
        failures = sum(value for key, value in counts.items() if key != "stable")
        for cause in CAUSE_ORDER:
            count = counts[cause]
            cause_rows.append(
                {
                    "model": model,
                    "cause": cause,
                    "count": count,
                    "share_among_failures": 0.0 if failures == 0 else count / failures,
                }
            )
    causes = pd.DataFrame(cause_rows)

    culprit_rows = []
    for model, counts in culprit_counts.items():
        total = sum(counts.values())
        for culprit, count in sorted(counts.items(), key=lambda item: int(item[0])):
            culprit_rows.append(
                {
                    "model": model,
                    "competitor_rank": int(culprit) + 1,
                    "count": count,
                    "share": count / total,
                }
            )
    culprits = pd.DataFrame(culprit_rows)

    quality_rows = []
    for model, values in quality.items():
        quality_rows.append(
            {
                "model": model,
                "mean_probe_coverage": float(np.mean(values["coverage"])),
                "mean_matched_iou": (
                    float(np.mean(values["matched_iou"]))
                    if values["matched_iou"]
                    else math.nan
                ),
                "mean_perturbed_candidate_count": float(
                    np.mean(values["candidate_count"])
                ),
            }
        )
    return family, causes, culprits, pd.DataFrame(quality_rows)


def aggregate_estimands(
    traces: dict[str, dict[tuple[int, int], dict]], split: str
) -> pd.DataFrame:
    """Pool eligible probe events and verify both coverage identities."""

    rows = []
    for model, records in traces.items():
        coverage = []
        operational = []
        covered_ranking = []
        eligible_samples = 0
        for record in records.values():
            if not record.get("clean_eligible"):
                continue
            eligible_samples += 1
            for probe in record["probes"]:
                if probe["split"] != split:
                    continue
                outcome = probe["outcome"]
                coverage.append(int(outcome["coverage"]))
                operational.append(int(outcome["operational_stable"]))
                if outcome["coverage"]:
                    covered_ranking.append(int(outcome["rank_stable"]))
        theta_cov = float(np.mean(coverage)) if coverage else 0.0
        theta_rank = float(np.mean(covered_ranking)) if covered_ranking else 0.0
        theta_op = float(np.mean(operational)) if operational else 0.0
        observed_gap = theta_rank - theta_op
        predicted_gap = theta_rank * (1.0 - theta_cov)
        risk = 1.0 - theta_op
        coverage_risk = 1.0 - theta_cov
        ranking_risk = theta_cov * (1.0 - theta_rank)
        rows.append(
            {
                "model": model,
                "eligible_samples": eligible_samples,
                "probe_count": len(operational),
                "coverage": theta_cov,
                "conditional_ranking": theta_rank,
                "operational_stability": theta_op,
                "conditional_minus_operational": observed_gap,
                "predicted_gap_from_identity": predicted_gap,
                "gap_identity_residual": abs(observed_gap - predicted_gap),
                "operational_risk": risk,
                "coverage_risk": coverage_risk,
                "conditional_ranking_risk": ranking_risk,
                "risk_identity_residual": abs(risk - coverage_risk - ranking_risk),
            }
        )
    return pd.DataFrame(rows)


def build_probe_arrays(
    records: dict[tuple[int, int], dict],
    keys: list[tuple[int, int]],
    diagnostic_count: int,
    reference_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagnostic = np.zeros((len(keys), diagnostic_count), dtype=np.int8)
    reference = np.zeros((len(keys), reference_count), dtype=np.int8)
    eligible = np.zeros(len(keys), dtype=np.int8)
    for row, key in enumerate(keys):
        record = records[key]
        eligible[row] = int(record.get("clean_eligible", 0))
        if not eligible[row]:
            continue
        diag_values = [
            probe["outcome"]["operational_stable"]
            for probe in record["probes"]
            if probe["split"] == "diagnostic"
        ]
        ref_values = [
            probe["outcome"]["operational_stable"]
            for probe in record["probes"]
            if probe["split"] == "reference"
        ]
        if len(diag_values) != diagnostic_count or len(ref_values) != reference_count:
            raise ValueError(f"incomplete trace for sample {key}")
        diagnostic[row] = diag_values
        reference[row] = ref_values
    return diagnostic, reference, eligible


def hierarchical_bootstrap(
    traces: dict[str, dict[tuple[int, int], dict]],
    budgets: list[int],
    diagnostic_count: int,
    reference_count: int,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    models = sorted(traces)
    common_keys = sorted(set.intersection(*(set(traces[m]) for m in models)))
    arrays = {
        model: build_probe_arrays(
            traces[model], common_keys, diagnostic_count, reference_count
        )
        for model in models
    }
    rng = np.random.default_rng(seed)
    result: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    paired_differences: dict[
        tuple[int, str], list[tuple[str, str, float]]
    ] = defaultdict(list)
    n = len(common_keys)

    for _ in range(repetitions):
        sample_indices = rng.integers(0, n, size=n)
        reference_indices = rng.integers(0, reference_count, size=(n, reference_count))
        reference_means = {}
        diagnostic_means_by_budget = {}
        for budget in budgets:
            diagnostic_indices = rng.integers(0, budget, size=(n, budget))
            diagnostic_means_by_budget[budget] = {}
            for model in models:
                diagnostic, reference, _ = arrays[model]
                diag_resampled = np.take_along_axis(
                    diagnostic[:, :budget], diagnostic_indices, axis=1
                ).mean(axis=1)
                ref_resampled = np.take_along_axis(
                    reference, reference_indices, axis=1
                ).mean(axis=1)
                diag_value = float(diag_resampled[sample_indices].mean())
                ref_value = float(ref_resampled[sample_indices].mean())
                diagnostic_means_by_budget[budget][model] = diag_value
                reference_means[model] = ref_value
                result[(model, budget, "diagnostic_mean")].append(diag_value)
                result[(model, budget, "reference_mean")].append(ref_value)
            for left, right in itertools.combinations(models, 2):
                paired_differences[(budget, "diagnostic_difference")].append(
                    (
                        left,
                        right,
                        diagnostic_means_by_budget[budget][left]
                        - diagnostic_means_by_budget[budget][right],
                    )
                )
                paired_differences[(budget, "reference_difference")].append(
                    (left, right, reference_means[left] - reference_means[right])
                )

    rows = []
    for (model, budget, statistic), values in result.items():
        array = np.asarray(values)
        rows.append(
            {
                "comparison": "single_model",
                "model_or_pair": model,
                "diagnostic_budget": budget,
                "statistic": statistic,
                "estimate": float(array.mean()),
                "lower_95": float(np.quantile(array, 0.025)),
                "upper_95": float(np.quantile(array, 0.975)),
                "bootstrap_repetitions": repetitions,
                "sample_count": n,
            }
        )
    for (budget, statistic), values in paired_differences.items():
        for left, right in itertools.combinations(models, 2):
            array = np.asarray(
                [value for a, b, value in values if a == left and b == right]
            )
            rows.append(
                {
                    "comparison": "paired_cross_model",
                    "model_or_pair": f"{left}-{right}",
                    "diagnostic_budget": budget,
                    "statistic": statistic,
                    "estimate": float(array.mean()),
                    "lower_95": float(np.quantile(array, 0.025)),
                    "upper_95": float(np.quantile(array, 0.975)),
                    "bootstrap_repetitions": repetitions,
                    "sample_count": n,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["comparison", "model_or_pair", "diagnostic_budget", "statistic"]
    )


def save_figures(
    budget: pd.DataFrame,
    family: pd.DataFrame,
    causes: pd.DataFrame,
    estimands: pd.DataFrame,
    family_comparison: pd.DataFrame,
    family_budget_summary: pd.DataFrame,
    selective_curve: pd.DataFrame,
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "groundingdino": "#3B82F6",
        "yoloworld": "#F97316",
        "owlv2": "#10B981",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model, subset in budget.groupby("model"):
        label = MODEL_LABELS.get(model, model)
        axes[0].plot(
            subset["diagnostic_budget"], subset["mae_all"], "o-",
            label=label, color=colors.get(model),
        )
        axes[1].plot(
            subset["diagnostic_budget"], subset["spearman_all"], "o-",
            label=label, color=colors.get(model),
        )
    axes[0].set(xlabel="Diagnostic probe budget", ylabel="MAE to 80-probe reference")
    axes[1].set(xlabel="Diagnostic probe budget", ylabel="Sample-level Spearman correlation")
    axes[0].legend(); axes[1].legend()
    fig.suptitle("Finite-probe estimation quality")
    fig.tight_layout()
    fig.savefig(output / "finite_probe_estimation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if not family.empty:
        pivot = family.pivot(index="family", columns="model", values="risk_share")
        pivot = pivot.reindex(sorted(pivot.index))
        ax = pivot.rename(columns=MODEL_LABELS).plot(
            kind="bar", figsize=(9, 4.5), color=[colors.get(c) for c in pivot.columns]
        )
        ax.set(ylabel="Share of operational risk", xlabel="Probe family")
        ax.set_title("Which perturbation families explain instability?")
        ax.legend(title="")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output / "reference_family_risk_share.png", dpi=220, bbox_inches="tight")
        plt.close()

    if not causes.empty:
        pivot = causes.pivot(index="cause", columns="model", values="share_among_failures")
        pivot = pivot.reindex(CAUSE_ORDER).fillna(0.0)
        ax = pivot.rename(columns=MODEL_LABELS).plot(
            kind="bar", figsize=(9, 4.5), color=[colors.get(c) for c in pivot.columns]
        )
        ax.set(ylabel="Share among observed failures", xlabel="Primary failure cause")
        ax.set_title("Operational failure decomposition")
        ax.legend(title="")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output / "reference_failure_causes.png", dpi=220, bbox_inches="tight")
        plt.close()

    if not estimands.empty:
        plotted = estimands.set_index("model")[
            ["coverage", "conditional_ranking", "operational_stability"]
        ]
        plotted.index = [MODEL_LABELS.get(index, index) for index in plotted.index]
        ax = plotted.plot(
            kind="bar",
            figsize=(8.5, 4.5),
            color=["#10B981", "#8B5CF6", "#2563EB"],
        )
        ax.set(
            ylabel="Estimated probability",
            xlabel="Model",
            ylim=(0.0, 1.05),
        )
        ax.set_title("Coverage, conditional persistence, and operational stability")
        ax.legend(title="")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output / "coverage_ranking_operational.png", dpi=220, bbox_inches="tight")
        plt.close()

    if not family_comparison.empty:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        for model, subset in family_comparison.groupby("model"):
            ax.scatter(
                subset["risk_share_diagnostic"],
                subset["risk_share_reference"],
                label=MODEL_LABELS.get(model, model),
                color=colors.get(model),
                s=55,
                alpha=0.85,
            )
            for row in subset.itertuples(index=False):
                ax.annotate(
                    row.family,
                    (row.risk_share_diagnostic, row.risk_share_reference),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.plot([0, 1], [0, 1], "--", color="#6B7280", linewidth=1)
        ax.set(
            xlabel="40-probe diagnostic risk share",
            ylabel="80-probe reference risk share",
            title="Perturbation-family profile reproducibility",
            xlim=(0, max(0.3, family_comparison["risk_share_diagnostic"].max() * 1.15)),
            ylim=(0, max(0.3, family_comparison["risk_share_reference"].max() * 1.15)),
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "family_risk_profile_transfer.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if not family_budget_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for model, subset in family_budget_summary.groupby("model"):
            label = MODEL_LABELS.get(model, model)
            color = colors.get(model)
            axes[0].plot(
                subset["diagnostic_budget"],
                subset["mean_family_stability_error"],
                "o-",
                color=color,
                label=label,
            )
            axes[1].plot(
                subset["diagnostic_budget"],
                subset["mean_family_risk_share_error"],
                "o-",
                color=color,
                label=label,
            )
        axes[0].set(xlabel="Diagnostic probe budget", ylabel="Mean family stability MAE")
        axes[1].set(xlabel="Diagnostic probe budget", ylabel="Mean family risk-share MAE")
        axes[0].legend(); axes[1].legend()
        fig.suptitle("Finite-probe localisation of perturbation-family risk")
        fig.tight_layout()
        fig.savefig(output / "family_profile_by_budget.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if not selective_curve.empty:
        fig, axes = plt.subplots(1, len(selective_curve["model"].unique()), figsize=(11, 4.3), squeeze=False)
        for axis, (model, model_rows) in zip(axes[0], selective_curve.groupby("model")):
            for budget_value, subset in model_rows.groupby("diagnostic_budget"):
                subset = subset.sort_values("coverage")
                axis.plot(
                    subset["coverage"], subset["reference_risk"], marker=".",
                    label=f"{int(budget_value)} probes",
                )
            axis.set(
                xlabel="Retained sample coverage",
                ylabel="80-probe reference risk",
                title=MODEL_LABELS.get(model, model),
                xlim=(0, 1.02),
                ylim=(0, 1.02),
            )
            axis.legend(fontsize=8)
        fig.suptitle("Tie-aware selective risk from finite-probe operational stability")
        fig.tight_layout()
        fig.savefig(output / "selective_risk_curves.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def render_report(
    budget: pd.DataFrame,
    family: pd.DataFrame,
    causes: pd.DataFrame,
    quality: pd.DataFrame,
    estimands: pd.DataFrame,
    family_comparison: pd.DataFrame,
    selective_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    max_budget = int(budget["diagnostic_budget"].max())
    final = budget[budget["diagnostic_budget"] == max_budget]
    wrong_stable_rows = []
    final_summary = summary[summary["diagnostic_budget"] == max_budget]
    for model, subset in final_summary.groupby("model"):
        wrong = subset[subset["clean_correct"] == 0]
        wrong_stable_rows.append(
            {
                "model": MODEL_LABELS.get(model, model),
                "clean_wrong_samples": len(wrong),
                "mean_reference_stability_when_clean_wrong": (
                    float(wrong["reference_operational"].mean())
                    if len(wrong)
                    else math.nan
                ),
            }
        )

    lines = [
        "# Coverage-Aware Candidate-Order Stability: Large-Scale Results",
        "",
        "## Scope and estimand",
        "",
        "This benchmark estimates **operational candidate-order stability**, not semantic correctness. "
        "A probe is operationally stable only when the tracked clean candidates remain observable "
        "and the clean winner remains ahead of every tracked competitor. Candidate disappearance "
        "and threatening candidate birth are explicit failures, never imputed scores.",
        "",
        "The exact decomposition checked in every eligible sample is:",
        "",
        "`operational risk = coverage risk + coverage × conditional ranking risk`.",
        "",
        "## Main finite-probe results",
        "",
        markdown_table(final.rename(columns={"model": "model_key"}).assign(
            model=lambda x: x["model_key"].map(MODEL_LABELS)
        )[[
            "model", "sample_count", "eligible_count", "clean_eligibility",
            "diagnostic_full_manifest_mean", "reference_full_manifest_mean",
            "mae_all", "spearman_all", "mean_diagnostic_cp_width",
        ]]),
        "",
        "## Association quality",
        "",
        markdown_table(quality.assign(model=lambda x: x["model"].map(MODEL_LABELS))),
        "",
        "## Why coverage-aware stability is not direct persistence",
        "",
        markdown_table(estimands.assign(model=lambda x: x["model"].map(MODEL_LABELS))),
        "",
        "The conditional-minus-operational column is the instability hidden by "
        "conditioning on successful candidate association. Its equality to "
        "conditional ranking multiplied by one minus coverage is checked numerically.",
        "",
        "## Perturbation-family risk attribution",
        "",
        markdown_table(family.assign(model=lambda x: x["model"].map(MODEL_LABELS))[
            ["model", "family", "trials", "operational_stability", "risk_share"]
        ]),
        "",
        "## Diagnostic-to-reference family profile reproducibility",
        "",
        markdown_table(
            family_comparison.assign(
                model=lambda x: x["model"].map(MODEL_LABELS)
            )
        ),
        "",
        "## Primary failure causes",
        "",
        markdown_table(causes.assign(model=lambda x: x["model"].map(MODEL_LABELS))),
        "",
        "## Stability is not correctness",
        "",
        markdown_table(pd.DataFrame(wrong_stable_rows)),
        "",
        "This contextual check prevents a category error: a stable output may still be "
        "semantically wrong. Correctness is reported only as an external descriptor.",
        "",
        "## Tie-aware selective risk",
        "",
        markdown_table(
            selective_summary.assign(
                model=lambda x: x["model"].map(MODEL_LABELS)
            )
        ),
        "",
        "Ties are retained as groups, so a discrete small-budget estimator cannot "
        "obtain artificial ranking credit from arbitrary ordering within equal scores.",
        "",
        "## Paired hierarchical bootstrap",
        "",
        markdown_table(bootstrap[bootstrap["diagnostic_budget"] == max_budget]),
        "",
        "## Interpretation rules",
        "",
        "- Full-manifest stability assigns zero to clean outputs that do not expose two distinct candidates.",
        "- Eligible-only stability is diagnostic and is never substituted for the primary full-manifest estimate.",
        "- Family risk shares localise *where* instability is observed under the registered probe distribution; they are not causal effects.",
        "- The 80-probe estimate is an independent finite reference, not an unknowable exact population probability.",
        "- Clopper-Pearson intervals quantify finite-probe uncertainty for each Bernoulli estimand.",
        "",
        "## Reproducibility artifacts",
        "",
        "The result directory contains the frozen configuration, manifest hash, complete compressed "
        "probe traces, row-level summaries, statistical tables, figures, and this report.",
    ]
    (output / "large_scale_results_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results" / "operational_benchmark_v1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "operational_benchmark_v1.json",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = list(config["models"])
    summaries = []
    traces = {}
    for model in models:
        model_dir = args.result_root / model
        summary_path = model_dir / "sample_budget_summary.csv"
        trace_path = model_dir / "sample_traces.jsonl.gz"
        if not summary_path.exists() or not trace_path.exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(f"missing completed result for {model}")
        frame = pd.read_csv(summary_path)
        completed = frame[["image_id", "ref_id"]].drop_duplicates().shape[0]
        if completed != int(config["sample_count"]) and not args.allow_partial:
            raise ValueError(f"{model} has {completed} rather than {config['sample_count']} samples")
        summaries.append(frame)
        traces[model] = load_trace(trace_path)
    if not summaries:
        raise ValueError("no result tables found")

    output = args.result_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    budget = evaluate_budget_table(summary)
    family, causes, culprits, quality = trace_diagnostics(traces, "reference")
    diagnostic_family, _, _, _ = trace_diagnostics(traces, "diagnostic")
    family_comparison = compare_family_profiles(diagnostic_family, family)
    estimands = aggregate_estimands(traces, "reference")
    selective_curve, selective_summary = selective_risk_tables(summary)

    budgets = [int(value) for value in config["reported_diagnostic_budgets"]]
    family_budget_comparison, family_budget_summary = family_profiles_by_budget(
        traces, budgets, family
    )
    diagnostic_count = int(config["diagnostic_probes_per_family"]) * 5
    reference_count = int(config["reference_probes_per_family"]) * 5
    bootstrap = hierarchical_bootstrap(
        traces,
        budgets,
        diagnostic_count,
        reference_count,
        int(config["bootstrap_repetitions"]),
        int(config["base_seed"]) + 991,
    )

    budget.to_csv(output / "finite_probe_budget_metrics.csv", index=False)
    family.to_csv(output / "reference_family_risk.csv", index=False)
    diagnostic_family.to_csv(output / "diagnostic_family_risk.csv", index=False)
    family_comparison.to_csv(output / "family_profile_comparison.csv", index=False)
    family_budget_comparison.to_csv(
        output / "family_profile_by_budget.csv", index=False
    )
    family_budget_summary.to_csv(
        output / "family_profile_budget_summary.csv", index=False
    )
    causes.to_csv(output / "reference_failure_causes.csv", index=False)
    culprits.to_csv(output / "reference_ranking_culprits.csv", index=False)
    quality.to_csv(output / "association_quality.csv", index=False)
    estimands.to_csv(output / "reference_aggregate_estimands.csv", index=False)
    selective_curve.to_csv(output / "selective_risk_curves.csv", index=False)
    selective_summary.to_csv(output / "selective_risk_summary.csv", index=False)
    bootstrap.to_csv(output / "hierarchical_bootstrap.csv", index=False)
    save_figures(
        budget,
        family,
        causes,
        estimands,
        family_comparison,
        family_budget_summary,
        selective_curve,
        output,
    )
    render_report(
        budget,
        family,
        causes,
        quality,
        estimands,
        family_comparison,
        selective_summary,
        bootstrap,
        summary,
        output,
    )

    audit = {
        "models_analysed": sorted(traces),
        "samples_per_model": {
            model: len(records) for model, records in traces.items()
        },
        "summary_rows": len(summary),
        "maximum_saved_risk_decomposition_error": float(
            np.nanmax(
                np.concatenate(
                    [
                        summary["diagnostic_risk_decomposition_error"].to_numpy(float),
                        summary["reference_risk_decomposition_error"].to_numpy(float),
                    ]
                )
            )
        ),
        "maximum_pooled_risk_identity_residual": float(
            estimands["risk_identity_residual"].max()
        ),
        "maximum_pooled_gap_identity_residual": float(
            estimands["gap_identity_residual"].max()
        ),
        "analysis_complete": len(traces) == len(models),
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
