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
from scipy.stats import beta, spearmanr


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
    estimate_variance_components,
    probe_variance_share,
    probes_for_variance_share,
    variance_of_mean,
)


MODEL_PROBE_SHARE_LIMIT = 0.05
MODEL_SPLIT_DIFFERENCE_LIMIT = 0.01
SAMPLE_RMSE_LIMIT = 0.05
SPLIT_REPETITIONS = 2000


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


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    return float(spearmanr(left, right).statistic)


def clopper_pearson_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return an exact two-sided binomial confidence interval."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial count")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def trace_arrays(
    records: dict[tuple[int, int], CompactRecord]
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    ordered = list(records.values())
    reference = np.zeros((len(ordered), 80), dtype=np.int8)
    diagnostic = np.zeros((len(ordered), 40), dtype=np.int8)
    eligible = np.asarray([record.eligible for record in ordered], dtype=bool)
    correct = np.asarray([record.clean_correct for record in ordered], dtype=np.int8)
    keys = [record.key for record in ordered]
    eligible_records = [record for record in ordered if record.eligible]
    if not eligible_records:
        raise ValueError("trace contains no eligible records")
    family_names = tuple(sorted(set(eligible_records[0].reference_family)))
    family_successes = np.zeros((len(ordered), len(family_names)), dtype=np.int16)
    family_totals = np.zeros(len(family_names), dtype=np.int16)
    for family_index, family in enumerate(family_names):
        family_totals[family_index] = sum(
            value == family for value in eligible_records[0].reference_family
        )
    if np.any(family_totals % 2):
        raise ValueError("reference family counts must be even for a balanced half split")
    for index, record in enumerate(ordered):
        if not record.eligible:
            continue
        if len(record.reference_operational) != 80 or len(record.diagnostic_operational) != 40:
            raise ValueError("unexpected probe count")
        reference[index] = record.reference_operational
        diagnostic[index] = record.diagnostic_operational
        for family_index, family in enumerate(family_names):
            family_mask = np.asarray(record.reference_family, dtype=object) == family
            if int(family_mask.sum()) != int(family_totals[family_index]):
                raise ValueError("reference family counts differ across samples")
            family_successes[index, family_index] = int(
                record.reference_operational[family_mask].sum()
            )
    return (
        reference,
        diagnostic,
        eligible,
        correct,
        keys,
        family_successes,
        family_totals,
        family_names,
    )


def family_balanced_split_indices(
    families: tuple[str, ...], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split every probe family equally into two disjoint halves."""

    left: list[int] = []
    right: list[int] = []
    family_array = np.asarray(families, dtype=object)
    for family in sorted(set(families)):
        indices = np.flatnonzero(family_array == family)
        if len(indices) % 2:
            raise ValueError(f"family {family} has an odd probe count")
        shuffled = rng.permutation(indices)
        midpoint = len(indices) // 2
        left.extend(shuffled[:midpoint].tolist())
        right.extend(shuffled[midpoint:].tolist())
    return np.asarray(sorted(left), dtype=int), np.asarray(sorted(right), dtype=int)


def paired_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = left - right
    return {
        "mean_left": float(left.mean()),
        "mean_right": float(right.mean()),
        "signed_model_mean_difference": float(difference.mean()),
        "absolute_model_mean_difference": float(abs(difference.mean())),
        "sample_mae": float(np.mean(np.abs(difference))),
        "sample_rmse": float(np.sqrt(np.mean(difference**2))),
        "sample_spearman": safe_spearman(left, right),
    }


def split_half_analysis(
    dataset: str,
    model: str,
    reference: np.ndarray,
    eligible: np.ndarray,
    family_successes: np.ndarray,
    family_totals: np.ndarray,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    half_probe_count = int(family_totals.sum() // 2)
    for repetition in range(repetitions):
        left_successes = np.zeros(len(reference), dtype=np.int16)
        for family_index, family_total in enumerate(family_totals):
            good = family_successes[:, family_index]
            bad = int(family_total) - good
            left_successes += rng.hypergeometric(good, bad, int(family_total) // 2)
        total_successes = family_successes.sum(axis=1)
        right_successes = total_successes - left_successes
        left = left_successes / half_probe_count
        right = right_successes / half_probe_count
        for scope, mask in (("full_manifest", np.ones(len(reference), dtype=bool)), ("clean_eligible", eligible)):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "scope": scope,
                    "repetition": repetition,
                    "half_probe_count": half_probe_count,
                    **paired_metrics(left[mask], right[mask]),
                }
            )
    return pd.DataFrame(rows)


def summarise_split_half(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = (
        "absolute_model_mean_difference",
        "sample_mae",
        "sample_rmse",
        "sample_spearman",
    )
    for keys, group in raw.groupby(["dataset", "model", "scope"], sort=False):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            rows.append(
                {
                    "dataset": keys[0],
                    "model": keys[1],
                    "scope": keys[2],
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "lower_2_5": float(np.quantile(values, 0.025)),
                    "upper_97_5": float(np.quantile(values, 0.975)),
                    "p95": float(np.quantile(values, 0.95)),
                    "repetitions": len(values),
                }
            )
    return pd.DataFrame(rows)


def sample_precision_table(
    dataset: str,
    model: str,
    reference: np.ndarray,
    eligible: np.ndarray,
    correct: np.ndarray,
    keys: list[tuple[int, int]],
) -> pd.DataFrame:
    rows = []
    for index, key in enumerate(keys):
        if eligible[index]:
            successes = int(reference[index].sum())
            lower, upper = clopper_pearson_interval(successes, 80)
            mean = successes / 80
        else:
            successes = 0
            lower = upper = mean = 0.0
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "image_id": key[0],
                "ref_id": key[1],
                "clean_eligible": int(eligible[index]),
                "clean_correct": int(correct[index]),
                "reference_successes": successes,
                "reference_trials": 80 if eligible[index] else 0,
                "reference_mean": mean,
                "lower_95": lower,
                "upper_95": upper,
                "interval_width": upper - lower,
            }
        )
    return pd.DataFrame(rows)


def interval_summary(sample_precision: pd.DataFrame) -> pd.DataFrame:
    eligible = sample_precision[sample_precision["clean_eligible"] == 1]
    rows = []
    for (dataset, model), group in eligible.groupby(["dataset", "model"], sort=False):
        widths = group["interval_width"].to_numpy(float)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "eligible_sample_count": len(group),
                "mean_interval_width": float(widths.mean()),
                "median_interval_width": float(np.median(widths)),
                "p90_interval_width": float(np.quantile(widths, 0.90)),
                "p95_interval_width": float(np.quantile(widths, 0.95)),
                "maximum_interval_width": float(widths.max()),
                "fraction_width_le_0_10": float(np.mean(widths <= 0.10)),
                "fraction_width_le_0_20": float(np.mean(widths <= 0.20)),
            }
        )
    return pd.DataFrame(rows)


def independent_registry_comparison(
    dataset: str,
    model: str,
    diagnostic: np.ndarray,
    reference: np.ndarray,
    eligible: np.ndarray,
    within_B: float,
) -> pd.DataFrame:
    diagnostic_mean = diagnostic.mean(axis=1)
    reference_mean = reference.mean(axis=1)
    eligibility_rate = float(eligible.mean())
    rows = []
    for scope, mask in (("full_manifest", np.ones(len(reference), dtype=bool)), ("clean_eligible", eligible)):
        metrics = paired_metrics(diagnostic_mean[mask], reference_mean[mask])
        scope_within = within_B if scope == "full_manifest" else within_B / eligibility_rate
        predicted_rmse = math.sqrt(scope_within * (1 / 40 + 1 / 80))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "scope": scope,
                "diagnostic_probe_count": 40,
                "reference_probe_count": 80,
                "predicted_rmse_under_common_Q": predicted_rmse,
                "observed_to_predicted_rmse_ratio": metrics["sample_rmse"] / predicted_rmse,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def model_adequacy_table(
    components: pd.DataFrame,
    split_summary: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for component in components.itertuples(index=False):
        split_row = split_summary[
            (split_summary["dataset"] == component.dataset)
            & (split_summary["model"] == component.model)
            & (split_summary["scope"] == "full_manifest")
            & (split_summary["metric"] == "absolute_model_mean_difference")
        ].iloc[0]
        interval_row = intervals[
            (intervals["dataset"] == component.dataset)
            & (intervals["model"] == component.model)
        ].iloc[0]
        eligibility_rate = interval_row["eligible_sample_count"] / component.pair_count
        probe_share = probe_variance_share(component.between_A, component.within_B, 80)
        se80 = math.sqrt(
            variance_of_mean(component.between_A, component.within_B, component.pair_count, 80)
        )
        se160 = math.sqrt(
            variance_of_mean(component.between_A, component.within_B, component.pair_count, 160)
        )
        se_floor = math.sqrt(component.between_A / component.pair_count)
        sample_rmse = math.sqrt(component.within_B / (eligibility_rate * 80))
        share_pass = probe_share <= MODEL_PROBE_SHARE_LIMIT
        split_pass = split_row["p95"] <= MODEL_SPLIT_DIFFERENCE_LIMIT
        rows.append(
            {
                "dataset": component.dataset,
                "model": component.model,
                "pair_count": component.pair_count,
                "theta_hat": component.theta_hat,
                "probe_variance_share_R80": probe_share,
                "required_R_for_5pct_probe_share": probes_for_variance_share(
                    component.between_A, component.within_B, MODEL_PROBE_SHARE_LIMIT
                ),
                "standard_error_R80": se80,
                "standard_error_R160": se160,
                "standard_error_infinite_R": se_floor,
                "relative_se_reduction_80_to_160": (se80 - se160) / se80,
                "relative_se_excess_over_infinite_R": se80 / se_floor - 1,
                "clean_eligibility_rate": eligibility_rate,
                "average_eligible_sample_probe_rmse_R80": sample_rmse,
                "split_half_p95_absolute_model_difference": float(split_row["p95"]),
                "probe_share_criterion_pass": bool(share_pass),
                "split_half_criterion_pass": bool(split_pass),
                "sample_rmse_criterion_pass": bool(sample_rmse <= SAMPLE_RMSE_LIMIT),
                "model_level_adequate": bool(share_pass and split_pass),
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    adequacy: pd.DataFrame,
    split_summary: pd.DataFrame,
    sample_precision: pd.DataFrame,
    output: Path,
) -> None:
    labels = [
        f"{DATASET_LABELS[row.dataset]}\n{MODEL_LABELS[row.model]}"
        for row in adequacy.itertuples(index=False)
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(x, adequacy["probe_variance_share_R80"], color="#315b7d")
    axes[0].axhline(MODEL_PROBE_SHARE_LIMIT, color="#b43c36", linestyle="--", label="Frozen 5% limit")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Finite-probe share of model-level variance")
    axes[0].set_title("Does R = 80 reach the variance floor?")
    axes[0].legend()
    split = split_summary[
        (split_summary["scope"] == "full_manifest")
        & (split_summary["metric"] == "absolute_model_mean_difference")
    ]
    split = split.set_index(["dataset", "model"]).loc[
        [(row.dataset, row.model) for row in adequacy.itertuples(index=False)]
    ].reset_index()
    axes[1].bar(x, split["p95"], color="#ed9b40")
    axes[1].axhline(
        MODEL_SPLIT_DIFFERENCE_LIMIT,
        color="#b43c36",
        linestyle="--",
        label="Frozen 0.01 limit",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("95th percentile absolute 40/40 mean difference")
    axes[1].set_title("Disjoint balanced half-reference agreement")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "model_level_adequacy.png", dpi=180)
    plt.close(fig)

    eligible = sample_precision[sample_precision["clean_eligible"] == 1].copy()
    group_order = [(row.dataset, row.model) for row in adequacy.itertuples(index=False)]
    data = [
        eligible[(eligible["dataset"] == dataset) & (eligible["model"] == model)][
            "interval_width"
        ].to_numpy(float)
        for dataset, model in group_order
    ]
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.boxplot(data, tick_labels=labels, showfliers=False)
    axis.axhline(0.20, color="#b43c36", linestyle="--", label="Width 0.20")
    axis.set_ylabel("Exact 95% interval width for one eligible sample")
    axis.set_title("Sample-level resolution of the 80-probe reference")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "sample_level_interval_widths.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(x - 0.18, adequacy["standard_error_R80"], width=0.36, label="R = 80", color="#315b7d")
    axis.bar(
        x + 0.18,
        adequacy["standard_error_infinite_R"],
        width=0.36,
        label="R tends to infinity",
        color="#9ab2c6",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Model-level standard error")
    axis.set_title("How much precision remains beyond 80 probes?")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "reference_80_vs_variance_floor.png", dpi=180)
    plt.close(fig)


def render_report(
    adequacy: pd.DataFrame,
    split_summary: pd.DataFrame,
    intervals: pd.DataFrame,
    independent: pd.DataFrame,
    output: Path,
) -> None:
    adequate_count = int(adequacy["model_level_adequate"].sum())
    sample_count = len(adequacy)
    display = adequacy.copy()
    display["dataset"] = display["dataset"].map(DATASET_LABELS)
    display["model"] = display["model"].map(MODEL_LABELS)
    split_display = split_summary[
        (split_summary["scope"] == "clean_eligible")
        & (split_summary["metric"].isin(["sample_mae", "sample_rmse", "sample_spearman"]))
    ].copy()
    split_display["dataset"] = split_display["dataset"].map(DATASET_LABELS)
    split_display["model"] = split_display["model"].map(MODEL_LABELS)
    interval_display = intervals.copy()
    interval_display["dataset"] = interval_display["dataset"].map(DATASET_LABELS)
    interval_display["model"] = interval_display["model"].map(MODEL_LABELS)
    independent_display = independent[independent["scope"] == "clean_eligible"].copy()
    independent_display["dataset"] = independent_display["dataset"].map(DATASET_LABELS)
    independent_display["model"] = independent_display["model"].map(MODEL_LABELS)
    lines = [
        "# Adequacy of the Frozen 80-Probe Reference",
        "",
        "## Decision",
        "",
        f"The 80-probe reference satisfies both frozen model-level adequacy criteria in "
        f"{adequate_count} of {sample_count} dataset-model groups.",
        "",
        "It is therefore adequate for estimating and comparing model-level operational "
        "stability under the frozen probe distribution. At sample level it should remain "
        "described as a finite, noisy reference rather than exact ground truth.",
        "",
        "## Frozen model-level criteria",
        "",
        "1. Finite probes must contribute no more than 5% of model-level variance at R = 80.",
        "2. The 95th percentile absolute difference between family-balanced 40/40 half-reference "
        "model means must be no larger than 0.01.",
        "",
        markdown_table(
            display[
                [
                    "dataset",
                    "model",
                    "probe_variance_share_R80",
                    "required_R_for_5pct_probe_share",
                    "split_half_p95_absolute_model_difference",
                    "relative_se_excess_over_infinite_R",
                    "model_level_adequate",
                ]
            ]
        ),
        "",
        "## Sample-level precision",
        "",
        markdown_table(
            display[
                [
                    "dataset",
                    "model",
                    "clean_eligibility_rate",
                    "average_eligible_sample_probe_rmse_R80",
                    "sample_rmse_criterion_pass",
                ]
            ]
        ),
        "",
        markdown_table(interval_display),
        "",
        "The Clopper--Pearson intervals, under the iid-Q Bernoulli working model, demonstrate "
        "why an 80-probe sample value must not be called exact. Precision depends strongly on "
        "whether the latent probability is near zero, one, or one half.",
        "",
        "## Family-balanced 40/40 split-half agreement on eligible samples",
        "",
        markdown_table(
            split_display[
                [
                    "dataset",
                    "model",
                    "metric",
                    "mean",
                    "median",
                    "lower_2_5",
                    "upper_97_5",
                    "p95",
                ]
            ]
        ),
        "",
        "## Independent diagnostic-40 versus reference-80 agreement",
        "",
        markdown_table(
            independent_display[
                [
                    "dataset",
                    "model",
                    "absolute_model_mean_difference",
                    "sample_mae",
                    "sample_rmse",
                    "predicted_rmse_under_common_Q",
                    "observed_to_predicted_rmse_ratio",
                    "sample_spearman",
                ]
            ]
        ),
        "",
        "If the observed-to-predicted RMSE ratio is close to one, disagreement is consistent "
        "with finite probe noise under a common probe law. Larger ratios indicate additional "
        "registry composition or probe-severity differences.",
        "",
        "## Conclusion",
        "",
        "The reference depth is sufficient for model-level conclusions because almost all "
        "remaining uncertainty is between image-query pairs, not within the 80 probes. Doubling "
        "the probe budget would therefore produce little model-level precision gain compared "
        "with sampling more independent pairs.",
        "",
        "For per-sample prediction, 80 probes provide a useful but noisy continuous target. "
        "Reported analyses must retain finite-reference uncertainty and must not describe the "
        "80-probe value as the latent probability itself.",
        "",
        "## Boundary of the claim",
        "",
        "This result validates Monte Carlo depth under the frozen empirical probe registry. It "
        "does not establish that the five registered perturbation families cover every possible "
        "real-world distribution shift.",
    ]
    (output / "reference_80_adequacy_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reference_80_adequacy",
    )
    parser.add_argument("--split-repetitions", type=int, default=SPLIT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    component_path = ROOT / "results" / "two_stage_sampling_analysis" / "variance_component_estimates.csv"
    if not component_path.exists():
        raise FileNotFoundError(component_path)
    components = pd.read_csv(component_path)
    split_frames = []
    precision_frames = []
    independent_frames = []
    input_files = [component_path]
    for group_index, (dataset, root) in enumerate(DATASETS.items()):
        for model_index, model in enumerate(MODELS):
            path = root / model / "sample_traces.jsonl.gz"
            print(f"Loading and analysing {dataset}/{model}", flush=True)
            records = load_compact_trace(path)
            (
                reference,
                diagnostic,
                eligible,
                correct,
                keys,
                family_successes,
                family_totals,
                _,
            ) = trace_arrays(records)
            split_frames.append(
                split_half_analysis(
                    dataset,
                    model,
                    reference,
                    eligible,
                    family_successes,
                    family_totals,
                    args.split_repetitions,
                    args.seed + group_index * 1009 + model_index * 101,
                )
            )
            precision_frames.append(
                sample_precision_table(dataset, model, reference, eligible, correct, keys)
            )
            component = components[
                (components["dataset"] == dataset) & (components["model"] == model)
            ].iloc[0]
            independent_frames.append(
                independent_registry_comparison(
                    dataset,
                    model,
                    diagnostic,
                    reference,
                    eligible,
                    float(component["within_B"]),
                )
            )
            input_files.append(path)

    split_raw = pd.concat(split_frames, ignore_index=True)
    split_summary = summarise_split_half(split_raw)
    sample_precision = pd.concat(precision_frames, ignore_index=True)
    intervals = interval_summary(sample_precision)
    independent = pd.concat(independent_frames, ignore_index=True)
    adequacy = model_adequacy_table(components, split_summary, intervals)

    split_raw.to_csv(args.output / "balanced_split_half_raw.csv", index=False)
    split_summary.to_csv(args.output / "balanced_split_half_summary.csv", index=False)
    sample_precision.to_csv(args.output / "sample_reference_precision.csv", index=False)
    intervals.to_csv(args.output / "sample_interval_summary.csv", index=False)
    independent.to_csv(args.output / "diagnostic40_vs_reference80.csv", index=False)
    adequacy.to_csv(args.output / "model_level_adequacy.csv", index=False)
    save_figures(adequacy, split_summary, sample_precision, args.output)
    render_report(adequacy, split_summary, intervals, independent, args.output)

    artifacts = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name not in {"analysis_audit.json", "artifact_manifest.json"}:
            artifacts.append(
                {"path": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            )
    (args.output / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "status": "complete",
        "split_repetitions": args.split_repetitions,
        "seed": args.seed,
        "frozen_criteria": {
            "maximum_probe_variance_share": MODEL_PROBE_SHARE_LIMIT,
            "maximum_p95_absolute_split_half_model_mean_difference": MODEL_SPLIT_DIFFERENCE_LIMIT,
            "maximum_average_sample_probe_rmse": SAMPLE_RMSE_LIMIT,
        },
        "adequate_model_level_groups": int(adequacy["model_level_adequate"].sum()),
        "total_groups": len(adequacy),
        "input_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in input_files},
        "outputs": sorted(path.name for path in args.output.iterdir() if path.is_file()),
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
