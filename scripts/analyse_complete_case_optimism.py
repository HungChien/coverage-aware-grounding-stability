from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


DATASETS = {
    "refcoco": ROOT / "results" / "operational_benchmark_v1",
    "refcocoplus": ROOT / "results" / "operational_transfer_refcocoplus_v1",
    "refl4": ROOT / "results" / "operational_transfer_refl4_v1",
}
DATASET_LABELS = {
    "refcoco": "RefCOCO",
    "refcocoplus": "RefCOCO+",
    "refl4": "Ref-L4",
}
MODEL_LABELS = {
    "groundingdino": "GroundingDINO",
    "owlv2": "OWLv2",
    "yoloworld": "YOLO-World",
}
MODELS = tuple(MODEL_LABELS)
BUDGETS = (5, 10, 20, 40)
FAMILY_ORDER = ("blur", "brightness", "gaussian_noise", "resolution", "jpeg")


@dataclass(frozen=True)
class CompactRecord:
    key: tuple[int, int]
    eligible: int
    clean_correct: int
    diagnostic_family: tuple[str, ...]
    diagnostic_coverage: np.ndarray
    diagnostic_operational: np.ndarray
    reference_family: tuple[str, ...]
    reference_coverage: np.ndarray
    reference_operational: np.ndarray


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    return float(spearmanr(left, right).statistic)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_compact_trace(path: Path) -> dict[tuple[int, int], CompactRecord]:
    records: dict[tuple[int, int], CompactRecord] = {}
    duplicate_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            key = (int(raw["image_id"]), int(raw["ref_id"]))
            if key in records:
                duplicate_count += 1
            diagnostic = [probe for probe in raw.get("probes", []) if probe["split"] == "diagnostic"]
            reference = [probe for probe in raw.get("probes", []) if probe["split"] == "reference"]

            def arrays(probes: list[dict]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
                families = tuple(str(probe["spec"]["family"]) for probe in probes)
                coverage = np.asarray(
                    [int(probe["outcome"]["coverage"]) for probe in probes], dtype=np.int8
                )
                operational = np.asarray(
                    [int(probe["outcome"]["operational_stable"]) for probe in probes],
                    dtype=np.int8,
                )
                if np.any(operational > coverage):
                    raise ValueError(f"operational success without coverage in {path}: {key}")
                return families, coverage, operational

            diag_family, diag_coverage, diag_operational = arrays(diagnostic)
            ref_family, ref_coverage, ref_operational = arrays(reference)
            eligible = int(raw.get("clean_eligible", 0))
            if eligible and (len(diagnostic) != 40 or len(reference) != 80):
                raise ValueError(
                    f"eligible sample {key} in {path} has {len(diagnostic)}/"
                    f"{len(reference)} diagnostic/reference probes"
                )
            if not eligible and (diagnostic or reference):
                raise ValueError(f"clean-ineligible sample {key} unexpectedly has probes")
            records[key] = CompactRecord(
                key=key,
                eligible=eligible,
                clean_correct=int(raw.get("clean_correct", 0)),
                diagnostic_family=diag_family,
                diagnostic_coverage=diag_coverage,
                diagnostic_operational=diag_operational,
                reference_family=ref_family,
                reference_coverage=ref_coverage,
                reference_operational=ref_operational,
            )
    if duplicate_count:
        raise ValueError(f"{path} contains {duplicate_count} duplicate resumable records")
    return records


def pooled_estimands(
    records: Iterable[CompactRecord], split: str, budget: int | None = None
) -> dict[str, float]:
    records = list(records)
    sample_count = len(records)
    eligible_count = sum(record.eligible for record in records)
    covered = 0
    stable = 0
    eligible_trials = 0
    per_sample_trials = 80 if split == "reference" else int(budget or 40)
    for record in records:
        if not record.eligible:
            continue
        coverage = getattr(record, f"{split}_coverage")[:per_sample_trials]
        operational = getattr(record, f"{split}_operational")[:per_sample_trials]
        eligible_trials += len(coverage)
        covered += int(coverage.sum())
        stable += int(operational.sum())
    gamma = safe_ratio(eligible_count, sample_count)
    theta_cov = safe_ratio(covered, eligible_trials)
    theta_cc = safe_ratio(stable, covered)
    theta_eligible_op = safe_ratio(stable, eligible_trials)
    theta_full_op = safe_ratio(stable, sample_count * per_sample_trials)
    coverage_optimism = theta_cc - theta_eligible_op
    eligibility_optimism = theta_eligible_op - theta_full_op
    total_optimism = theta_cc - theta_full_op
    predicted_coverage_optimism = theta_cc * (1.0 - theta_cov)
    predicted_eligibility_optimism = (1.0 - gamma) * theta_cov * theta_cc
    predicted_total_optimism = theta_cc * (1.0 - gamma * theta_cov)
    return {
        "sample_count": sample_count,
        "eligible_count": eligible_count,
        "probe_budget": per_sample_trials,
        "eligible_probe_trials": eligible_trials,
        "covered_probe_trials": covered,
        "stable_probe_trials": stable,
        "clean_eligibility": gamma,
        "coverage": theta_cov,
        "complete_case_persistence": theta_cc,
        "eligible_operational_stability": theta_eligible_op,
        "full_manifest_operational_stability": theta_full_op,
        "coverage_optimism": coverage_optimism,
        "eligibility_optimism": eligibility_optimism,
        "total_optimism": total_optimism,
        "predicted_coverage_optimism": predicted_coverage_optimism,
        "predicted_eligibility_optimism": predicted_eligibility_optimism,
        "predicted_total_optimism": predicted_total_optimism,
        "coverage_identity_residual": abs(coverage_optimism - predicted_coverage_optimism),
        "eligibility_identity_residual": abs(
            eligibility_optimism - predicted_eligibility_optimism
        ),
        "total_identity_residual": abs(total_optimism - predicted_total_optimism),
    }


def sample_predictions(
    records: Iterable[CompactRecord], budget: int
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for record in records:
        if record.eligible:
            diag_cov = record.diagnostic_coverage[:budget]
            diag_op = record.diagnostic_operational[:budget]
            ref_cov = record.reference_coverage
            ref_op = record.reference_operational
            diag_covered = int(diag_cov.sum())
            ref_covered = int(ref_cov.sum())
            direct = safe_ratio(float(diag_op.sum()), float(diag_covered))
            diag_operational = float(diag_op.mean())
            ref_direct = safe_ratio(float(ref_op.sum()), float(ref_covered))
            ref_operational = float(ref_op.mean())
        else:
            diag_covered = 0
            ref_covered = 0
            direct = math.nan
            ref_direct = math.nan
            diag_operational = 0.0
            ref_operational = 0.0
        rows.append(
            {
                "image_id": record.key[0],
                "ref_id": record.key[1],
                "clean_eligible": record.eligible,
                "clean_correct": record.clean_correct,
                "diagnostic_covered_probes": diag_covered,
                "reference_covered_probes": ref_covered,
                "direct_persistence": direct,
                "diagnostic_operational": diag_operational,
                "reference_direct_persistence": ref_direct,
                "reference_operational": ref_operational,
            }
        )
    return pd.DataFrame(rows)


def tie_aware_aurc(score: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(score) & np.isfinite(target)
    score = score[mask]
    target = target[mask]
    if len(score) == 0:
        return math.nan
    risk = 1.0 - target
    thresholds = sorted(np.unique(score), reverse=True)
    points = []
    for threshold in thresholds:
        selected = score >= threshold
        points.append((float(selected.mean()), float(risk[selected].mean())))
    coverage = np.asarray([point[0] for point in points])
    retained_risk = np.asarray([point[1] for point in points])
    if len(points) == 1:
        return float(retained_risk[0])
    return float(
        np.trapezoid(
            np.concatenate(([retained_risk[0]], retained_risk)),
            np.concatenate(([0.0], coverage)),
        )
    )


def prediction_metrics(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(score) & np.isfinite(target)
    score = score[mask]
    target = target[mask]
    error = score - target
    # Averaging (p-Y)^2 over independent Bernoulli reference outcomes is
    # p^2 - 2 p E[Y] + E[Y], because Y^2=Y.
    brier = score * score - 2.0 * score * target + target
    return {
        "evaluated_samples": len(score),
        "bias": float(error.mean()) if len(error) else math.nan,
        "mae": float(np.abs(error).mean()) if len(error) else math.nan,
        "rmse": float(np.sqrt(np.mean(error * error))) if len(error) else math.nan,
        "reference_probe_brier": float(brier.mean()) if len(brier) else math.nan,
        "spearman": safe_spearman(score, target),
        "tie_aware_aurc": tie_aware_aurc(score, target),
    }


def predictive_tables(
    all_records: dict[tuple[str, str], dict[tuple[int, int], CompactRecord]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    calibration_rows = []
    sample_rows = []
    for (dataset, model), records in all_records.items():
        for budget in BUDGETS:
            samples = sample_predictions(records.values(), budget)
            samples.insert(0, "model", model)
            samples.insert(0, "dataset", dataset)
            samples.insert(2, "diagnostic_budget", budget)
            sample_rows.append(samples)

            full_op = prediction_metrics(
                samples["diagnostic_operational"].to_numpy(float),
                samples["reference_operational"].to_numpy(float),
            )
            metric_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "diagnostic_budget": budget,
                    "scope": "full_manifest",
                    "predictor": "coverage_aware_operational",
                    "predictor_availability": 1.0,
                    **full_op,
                }
            )

            paired = samples[np.isfinite(samples["direct_persistence"])].copy()
            availability = len(paired) / len(samples)
            for predictor, column in (
                ("complete_case_persistence", "direct_persistence"),
                ("coverage_aware_operational", "diagnostic_operational"),
            ):
                metrics = prediction_metrics(
                    paired[column].to_numpy(float),
                    paired["reference_operational"].to_numpy(float),
                )
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "diagnostic_budget": budget,
                        "scope": "paired_complete_cases",
                        "predictor": predictor,
                        "predictor_availability": availability,
                        **metrics,
                    }
                )

                scores = paired[column].to_numpy(float)
                targets = paired["reference_operational"].to_numpy(float)
                bin_index = np.minimum((scores * 10).astype(int), 9)
                for value in sorted(np.unique(bin_index)):
                    selected = bin_index == value
                    calibration_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "diagnostic_budget": budget,
                            "scope": "paired_complete_cases",
                            "predictor": predictor,
                            "bin_lower": value / 10.0,
                            "bin_upper": (value + 1) / 10.0,
                            "sample_count": int(selected.sum()),
                            "mean_prediction": float(scores[selected].mean()),
                            "mean_reference_operational": float(targets[selected].mean()),
                            "calibration_gap": float(
                                scores[selected].mean() - targets[selected].mean()
                            ),
                        }
                    )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(calibration_rows),
        pd.concat(sample_rows, ignore_index=True),
    )


def family_estimands(
    all_records: dict[tuple[str, str], dict[tuple[int, int], CompactRecord]]
) -> pd.DataFrame:
    rows = []
    for (dataset, model), records in all_records.items():
        for family in FAMILY_ORDER:
            sample_count = len(records)
            eligible_count = sum(record.eligible for record in records.values())
            eligible_trials = covered = stable = 0
            for record in records.values():
                if not record.eligible:
                    continue
                selected = np.asarray(
                    [value == family for value in record.reference_family], dtype=bool
                )
                eligible_trials += int(selected.sum())
                covered += int(record.reference_coverage[selected].sum())
                stable += int(record.reference_operational[selected].sum())
            gamma = safe_ratio(eligible_count, sample_count)
            cov = safe_ratio(covered, eligible_trials)
            cc = safe_ratio(stable, covered)
            eligible_op = safe_ratio(stable, eligible_trials)
            probes_per_sample = safe_ratio(eligible_trials, eligible_count)
            full_op = safe_ratio(stable, sample_count * probes_per_sample)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "family": family,
                    "sample_count": sample_count,
                    "eligible_count": eligible_count,
                    "clean_eligibility": gamma,
                    "eligible_probe_trials": eligible_trials,
                    "coverage": cov,
                    "complete_case_persistence": cc,
                    "eligible_operational_stability": eligible_op,
                    "full_manifest_operational_stability": full_op,
                    "coverage_optimism": cc - eligible_op,
                    "eligibility_optimism": eligible_op - full_op,
                    "total_optimism": cc - full_op,
                    "total_identity_residual": abs(
                        (cc - full_op) - cc * (1.0 - gamma * cov)
                    ),
                }
            )
    return pd.DataFrame(rows)


def correctness_strata(
    all_records: dict[tuple[str, str], dict[tuple[int, int], CompactRecord]]
) -> pd.DataFrame:
    rows = []
    for (dataset, model), records in all_records.items():
        for correct in (0, 1):
            subset = [record for record in records.values() if record.clean_correct == correct]
            if not subset:
                continue
            values = pooled_estimands(subset, "reference")
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "clean_correct": correct,
                    **values,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_aggregate(
    dataset: str,
    model: str,
    records: dict[tuple[int, int], CompactRecord],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    ordered = list(records.values())
    n = len(ordered)
    reference_count = 80
    eligible = np.asarray([record.eligible for record in ordered], dtype=np.int8)
    coverage_count = np.asarray(
        [record.reference_coverage.sum() if record.eligible else 0 for record in ordered],
        dtype=int,
    )
    operational_count = np.asarray(
        [record.reference_operational.sum() if record.eligible else 0 for record in ordered],
        dtype=int,
    )
    coverage_probability = coverage_count / reference_count
    cc_probability = np.divide(
        operational_count,
        coverage_count,
        out=np.zeros(n, dtype=float),
        where=coverage_count > 0,
    )
    rng = np.random.default_rng(seed)
    statistics = {
        name: np.empty(repetitions, dtype=float)
        for name in (
            "clean_eligibility",
            "coverage",
            "complete_case_persistence",
            "eligible_operational_stability",
            "full_manifest_operational_stability",
            "coverage_optimism",
            "eligibility_optimism",
            "total_optimism",
        )
    }
    for repetition in range(repetitions):
        selected = rng.integers(0, n, size=n)
        selected_eligible = eligible[selected]
        cov_draw = rng.binomial(reference_count, coverage_probability[selected])
        cov_draw *= selected_eligible
        op_draw = rng.binomial(cov_draw, cc_probability[selected])
        eligible_count = int(selected_eligible.sum())
        eligible_trials = eligible_count * reference_count
        covered = int(cov_draw.sum())
        stable = int(op_draw.sum())
        gamma = eligible_count / n
        cov = safe_ratio(covered, eligible_trials)
        cc = safe_ratio(stable, covered)
        eligible_op = safe_ratio(stable, eligible_trials)
        full_op = stable / (n * reference_count)
        statistics["clean_eligibility"][repetition] = gamma
        statistics["coverage"][repetition] = cov
        statistics["complete_case_persistence"][repetition] = cc
        statistics["eligible_operational_stability"][repetition] = eligible_op
        statistics["full_manifest_operational_stability"][repetition] = full_op
        statistics["coverage_optimism"][repetition] = cc - eligible_op
        statistics["eligibility_optimism"][repetition] = eligible_op - full_op
        statistics["total_optimism"][repetition] = cc - full_op

    point = pooled_estimands(ordered, "reference")
    rows = []
    for statistic, values in statistics.items():
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "statistic": statistic,
                "point_estimate": point[statistic],
                "bootstrap_mean": float(finite.mean()),
                "lower_95": float(np.quantile(finite, 0.025)),
                "upper_95": float(np.quantile(finite, 0.975)),
                "bootstrap_repetitions": repetitions,
                "sample_count": n,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_predictor_difference(
    sample_predictions_frame: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for (dataset, model, budget), subset in sample_predictions_frame.groupby(
        ["dataset", "model", "diagnostic_budget"]
    ):
        paired = subset[np.isfinite(subset["direct_persistence"])].reset_index(drop=True)
        direct = paired["direct_persistence"].to_numpy(float)
        operational = paired["diagnostic_operational"].to_numpy(float)
        target = paired["reference_operational"].to_numpy(float)
        n = len(paired)
        if n == 0:
            continue
        values = {"mae_advantage": [], "brier_advantage": [], "absolute_bias_advantage": []}
        for _ in range(repetitions):
            index = rng.integers(0, n, size=n)
            d = direct[index]
            o = operational[index]
            t = target[index]
            d_error = d - t
            o_error = o - t
            d_brier = np.mean(d * d - 2.0 * d * t + t)
            o_brier = np.mean(o * o - 2.0 * o * t + t)
            values["mae_advantage"].append(
                float(np.mean(np.abs(d_error)) - np.mean(np.abs(o_error)))
            )
            values["brier_advantage"].append(float(d_brier - o_brier))
            values["absolute_bias_advantage"].append(
                float(abs(d_error.mean()) - abs(o_error.mean()))
            )
        for statistic, raw_values in values.items():
            array = np.asarray(raw_values)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "diagnostic_budget": int(budget),
                    "statistic": statistic,
                    "definition": "complete_case_persistence minus coverage_aware_operational",
                    "estimate": float(array.mean()),
                    "lower_95": float(np.quantile(array, 0.025)),
                    "upper_95": float(np.quantile(array, 0.975)),
                    "bootstrap_repetitions": repetitions,
                    "paired_sample_count": n,
                }
            )
    return pd.DataFrame(rows)


def contract_sensitivity(dataset_roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset, root in dataset_roots.items():
        path = root / "analysis" / "contract_sensitivity" / "contract_sensitivity.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for raw in frame.itertuples(index=False):
            gamma = float(raw.clean_eligibility)
            eligible_op = float(raw.eligible_only_operational)
            full_op = float(raw.full_manifest_operational)
            failure_rate = 1.0 - eligible_op
            coverage_failure_share = (
                float(raw.winner_missing_share)
                + float(raw.competitor_missing_share)
                + float(raw.threatening_birth_share)
            )
            coverage_risk = failure_rate * coverage_failure_share
            cov = 1.0 - coverage_risk
            ranking_risk = failure_rate * float(raw.ranking_reversal_share)
            cc = 1.0 - safe_ratio(ranking_risk, cov)
            rows.append(
                {
                    "dataset": dataset,
                    "model": raw.model,
                    "tracked_candidate_count": int(raw.tracked_candidate_count),
                    "match_iou_threshold": float(raw.match_iou_threshold),
                    "birth_duplicate_iou_threshold": float(raw.birth_duplicate_iou_threshold),
                    "clean_eligibility": gamma,
                    "coverage": cov,
                    "complete_case_persistence": cc,
                    "eligible_operational_stability": eligible_op,
                    "full_manifest_operational_stability": full_op,
                    "total_optimism": cc - full_op,
                    "identity_residual": abs(
                        (cc - full_op) - cc * (1.0 - gamma * cov)
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows = []
    if not detail.empty:
        for (dataset, model), subset in detail.groupby(["dataset", "model"]):
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "settings": len(subset),
                    "min_complete_case_persistence": subset[
                        "complete_case_persistence"
                    ].min(),
                    "max_complete_case_persistence": subset[
                        "complete_case_persistence"
                    ].max(),
                    "min_full_manifest_operational": subset[
                        "full_manifest_operational_stability"
                    ].min(),
                    "max_full_manifest_operational": subset[
                        "full_manifest_operational_stability"
                    ].max(),
                    "min_total_optimism": subset["total_optimism"].min(),
                    "max_total_optimism": subset["total_optimism"].max(),
                }
            )
    return detail, pd.DataFrame(summary_rows)


def registered_strata(root: Path) -> pd.DataFrame:
    path = root / "analysis" / "transfer" / "registered_stratum_estimands.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["coverage_optimism"] = (
        frame["conditional_ranking"] - frame["eligible_operational"]
    )
    frame["eligibility_optimism"] = (
        frame["eligible_operational"] - frame["full_manifest_operational"]
    )
    frame["total_optimism"] = (
        frame["conditional_ranking"] - frame["full_manifest_operational"]
    )
    frame["identity_residual"] = (
        frame["total_optimism"]
        - frame["conditional_ranking"]
        * (1.0 - frame["clean_eligibility"] * frame["coverage"])
    ).abs()
    return frame


def save_figures(
    aggregate: pd.DataFrame,
    budget: pd.DataFrame,
    family: pd.DataFrame,
    predictive: pd.DataFrame,
    bootstrap: pd.DataFrame,
    contract_summary: pd.DataFrame,
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "complete_case_persistence": "#8B5CF6",
        "eligible_operational_stability": "#10B981",
        "full_manifest_operational_stability": "#2563EB",
    }

    plotted = aggregate.copy()
    plotted["group"] = plotted.apply(
        lambda row: f"{DATASET_LABELS[row.dataset]}\n{MODEL_LABELS[row.model]}", axis=1
    )
    values = plotted.set_index("group")[[
        "complete_case_persistence",
        "eligible_operational_stability",
        "full_manifest_operational_stability",
    ]]
    ax = values.plot(kind="bar", figsize=(12, 5.2), color=[colors[column] for column in values])
    ax.set(
        ylabel="Estimated stability",
        xlabel="Dataset and model",
        ylim=(0.0, 1.03),
        title="Complete-case persistence versus coverage-aware operational stability",
    )
    ax.legend(["Complete-case persistence", "Eligible operational", "Full-manifest operational"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "complete_case_vs_operational.png", dpi=240, bbox_inches="tight")
    plt.close()

    components = aggregate.copy()
    components["group"] = components.apply(
        lambda row: f"{DATASET_LABELS[row.dataset]}\n{MODEL_LABELS[row.model]}", axis=1
    )
    stacked = components.set_index("group")[["coverage_optimism", "eligibility_optimism"]]
    ax = stacked.plot(
        kind="bar", stacked=True, figsize=(11.5, 5.0), color=["#F59E0B", "#EF4444"]
    )
    ax.set(
        ylabel="Optimistic overstatement",
        xlabel="Dataset and model",
        title="Why complete-case persistence overstates full-manifest stability",
    )
    ax.legend(["Conditioning on candidate coverage", "Excluding clean-ineligible samples"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "optimism_decomposition.png", dpi=240, bbox_inches="tight")
    plt.close()

    # Publication-facing replacement for the two separate complete-case and
    # stacked-decomposition charts.  RefCOCO is used as the representative
    # panel because it contains the two gap values highlighted in the main
    # text; the complete three-dataset table remains the quantitative record.
    representative = (
        aggregate[aggregate["dataset"] == "refcoco"]
        .set_index("model")
        .loc[list(MODELS)]
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13.8, 5.1), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    stage_labels = ["Complete\ncase", "Coverage\nloss", "Eligibility\nloss", "Full\nmanifest"]
    for axis, row in zip(axes, representative.itertuples(index=False)):
        start = float(row.complete_case_persistence)
        coverage_loss = float(row.coverage_optimism)
        eligibility_loss = float(row.eligibility_optimism)
        after_coverage = start - coverage_loss
        final = float(row.full_manifest_operational_stability)
        axis.bar(0, start, color="#7C3AED", width=0.68)
        axis.bar(1, -coverage_loss, bottom=start, color="#F59E0B", width=0.68)
        axis.bar(2, -eligibility_loss, bottom=after_coverage, color="#EF4444", width=0.68)
        axis.bar(3, final, color="#2563EB", width=0.68)
        axis.plot([0.34, 0.66], [start, start], color="#6B7280", linewidth=1)
        axis.plot([1.34, 1.66], [after_coverage, after_coverage], color="#6B7280", linewidth=1)
        axis.plot([2.34, 2.66], [final, final], color="#6B7280", linewidth=1)
        axis.text(0, start + 0.025, f"{start:.4f}", ha="center", va="bottom", fontsize=10, weight="bold")
        axis.text(1, start - coverage_loss / 2, f"-{coverage_loss:.4f}", ha="center", va="center", fontsize=9)
        axis.text(2, after_coverage - eligibility_loss / 2, f"-{eligibility_loss:.4f}", ha="center", va="center", fontsize=9)
        axis.text(3, final + 0.025, f"{final:.4f}", ha="center", va="bottom", fontsize=10, weight="bold")
        axis.annotate(
            f"gap = {float(row.total_optimism):.4f}",
            xy=(3, final), xytext=(1.5, 0.12),
            arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.0},
            ha="center", va="center", fontsize=10, color="#111827",
        )
        axis.set_title(MODEL_LABELS[row.model], fontsize=12, weight="bold")
        axis.set_xticks(range(4), stage_labels, fontsize=9)
        axis.set_ylim(0.0, 1.03)
        axis.grid(axis="y", alpha=0.22)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Estimated stability", fontsize=11)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#7C3AED"),
        plt.Rectangle((0, 0), 1, 1, color="#F59E0B"),
        plt.Rectangle((0, 0), 1, 1, color="#EF4444"),
        plt.Rectangle((0, 0), 1, 1, color="#2563EB"),
    ]
    fig.legend(
        handles,
        ["Complete case", "Coverage loss", "Eligibility loss", "Full manifest"],
        loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4,
        frameon=False, fontsize=10,
    )
    fig.suptitle(
        "Where complete-case optimism comes from on RefCOCO",
        fontsize=14, weight="bold", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output / "complete_case_waterfall.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 4.8), sharex=True)
    if len(MODELS) == 1:
        axes = [axes]
    for axis, model in zip(axes, MODELS):
        for dataset, subset in budget[budget["model"] == model].groupby("dataset"):
            subset = subset.sort_values("probe_budget")
            axis.plot(
                subset["probe_budget"], subset["total_optimism"], marker="o",
                label=DATASET_LABELS[dataset],
            )
        axis.set(
            xlabel="Diagnostic probe budget",
            ylabel="Complete-case optimism gap",
            title=MODEL_LABELS[model],
        )
        axis.legend()
    fig.suptitle("Finite-probe persistence of the optimism gap")
    fig.tight_layout()
    fig.savefig(output / "optimism_by_probe_budget.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    pivot = family.pivot_table(
        index=["dataset", "model"], columns="family", values="total_optimism"
    ).reindex(columns=FAMILY_ORDER)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    image = ax.imshow(pivot.to_numpy(float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    labels = [f"{DATASET_LABELS[d]} / {MODEL_LABELS[m]}" for d, m in pivot.index]
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(pivot)):
        for column in range(len(pivot.columns)):
            ax.text(column, row, f"{pivot.iloc[row, column]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Complete-case optimism by perturbation family")
    fig.colorbar(image, ax=ax, label="Optimism gap")
    fig.tight_layout()
    fig.savefig(output / "family_optimism_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    paired = predictive[
        (predictive["scope"] == "paired_complete_cases")
        & (predictive["diagnostic_budget"] == 40)
    ].copy()
    paired["group"] = paired.apply(
        lambda row: f"{DATASET_LABELS[row.dataset]}\n{MODEL_LABELS[row.model]}", axis=1
    )
    mae = paired.pivot(index="group", columns="predictor", values="mae")
    mae = mae[["complete_case_persistence", "coverage_aware_operational"]]
    ax = mae.plot(kind="bar", figsize=(11.5, 4.8), color=["#8B5CF6", "#2563EB"])
    ax.set(
        ylabel="MAE to independent 80-probe operational reference",
        xlabel="Dataset and model",
        title="Fair paired prediction comparison at 40 diagnostic probes",
    )
    ax.legend(["Complete-case persistence", "Coverage-aware operational"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "predictive_mae_comparison.png", dpi=240, bbox_inches="tight")
    plt.close()

    optimism_ci = bootstrap[bootstrap["statistic"] == "total_optimism"].copy()
    optimism_ci["group"] = optimism_ci.apply(
        lambda row: f"{DATASET_LABELS[row.dataset]}\n{MODEL_LABELS[row.model]}", axis=1
    )
    x = np.arange(len(optimism_ci))
    y = optimism_ci["point_estimate"].to_numpy(float)
    lower = y - optimism_ci["lower_95"].to_numpy(float)
    upper = optimism_ci["upper_95"].to_numpy(float) - y
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o", capsize=4, color="#B91C1C")
    ax.axhline(0.0, color="#6B7280", linewidth=1)
    ax.set_xticks(x, optimism_ci["group"], rotation=0)
    ax.set(ylabel="Total optimism gap", title="Hierarchical bootstrap uncertainty")
    fig.tight_layout()
    fig.savefig(output / "optimism_bootstrap_intervals.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    if not contract_summary.empty:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        positions = np.arange(len(contract_summary))
        centers = (
            contract_summary["min_total_optimism"].to_numpy(float)
            + contract_summary["max_total_optimism"].to_numpy(float)
        ) / 2.0
        widths = (
            contract_summary["max_total_optimism"].to_numpy(float)
            - contract_summary["min_total_optimism"].to_numpy(float)
        ) / 2.0
        labels = [
            f"{DATASET_LABELS[row.dataset]}\n{MODEL_LABELS[row.model]}"
            for row in contract_summary.itertuples(index=False)
        ]
        ax.errorbar(positions, centers, yerr=widths, fmt="o", capsize=5, color="#7C3AED")
        ax.set_xticks(positions, labels)
        ax.set(
            ylabel="Min-max optimism across 27 contracts",
            title="Post-primary output-contract sensitivity",
        )
        fig.tight_layout()
        fig.savefig(output / "contract_optimism_sensitivity.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def render_report(
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    budget: pd.DataFrame,
    family: pd.DataFrame,
    predictive: pd.DataFrame,
    predictor_bootstrap: pd.DataFrame,
    correctness: pd.DataFrame,
    contract_summary: pd.DataFrame,
    strata: pd.DataFrame,
    output: Path,
) -> None:
    total_model_sample_records = int(aggregate["sample_count"].sum())
    unique_image_query_pairs = int(
        aggregate.groupby("dataset")["sample_count"].max().sum()
    )
    total_reference_probe_trials = int(aggregate["eligible_probe_trials"].sum())
    display_aggregate = aggregate.copy()
    display_aggregate["dataset"] = display_aggregate["dataset"].map(DATASET_LABELS)
    display_aggregate["model"] = display_aggregate["model"].map(MODEL_LABELS)
    display_aggregate = display_aggregate[[
        "dataset", "model", "sample_count", "eligible_count", "clean_eligibility",
        "coverage", "complete_case_persistence", "eligible_operational_stability",
        "full_manifest_operational_stability", "coverage_optimism",
        "eligibility_optimism", "total_optimism",
    ]]

    optimism_ci = bootstrap[bootstrap["statistic"] == "total_optimism"].copy()
    optimism_ci["dataset"] = optimism_ci["dataset"].map(DATASET_LABELS)
    optimism_ci["model"] = optimism_ci["model"].map(MODEL_LABELS)
    optimism_ci = optimism_ci[[
        "dataset", "model", "point_estimate", "lower_95", "upper_95",
        "bootstrap_repetitions", "sample_count",
    ]]

    predictive_40 = predictive[
        (predictive["diagnostic_budget"] == 40)
        & (predictive["scope"] == "paired_complete_cases")
    ].copy()
    predictive_40["dataset"] = predictive_40["dataset"].map(DATASET_LABELS)
    predictive_40["model"] = predictive_40["model"].map(MODEL_LABELS)
    predictive_40 = predictive_40[[
        "dataset", "model", "predictor", "predictor_availability", "evaluated_samples",
        "bias", "mae", "rmse", "reference_probe_brier", "spearman", "tie_aware_aurc",
    ]]

    advantage_40 = predictor_bootstrap[
        predictor_bootstrap["diagnostic_budget"] == 40
    ].copy()
    advantage_40["dataset"] = advantage_40["dataset"].map(DATASET_LABELS)
    advantage_40["model"] = advantage_40["model"].map(MODEL_LABELS)

    family_summary = family[[
        "dataset", "model", "family", "coverage", "complete_case_persistence",
        "full_manifest_operational_stability", "total_optimism",
    ]].copy()
    family_summary["dataset"] = family_summary["dataset"].map(DATASET_LABELS)
    family_summary["model"] = family_summary["model"].map(MODEL_LABELS)

    correctness_display = correctness[[
        "dataset", "model", "clean_correct", "sample_count", "clean_eligibility",
        "complete_case_persistence", "full_manifest_operational_stability", "total_optimism",
    ]].copy()
    correctness_display["dataset"] = correctness_display["dataset"].map(DATASET_LABELS)
    correctness_display["model"] = correctness_display["model"].map(MODEL_LABELS)

    lines = [
        "# Complete-Case Persistence Optimism: Cross-Model and Cross-Dataset Analysis",
        "",
        "## Executive result",
        "",
        "Across RefCOCO, RefCOCO+, and Ref-L4, complete-case persistence is an "
        "optimistic estimand whenever clean candidate eligibility or perturbed-candidate "
        "coverage is imperfect. The effect is algebraically necessary and empirically "
        "non-negligible. It is small-to-moderate for GroundingDINO and large for "
        "YOLO-World because the latter loses candidate eligibility and coverage much more often.",
        "",
        f"The analysis uses every completed trace from all three tested models: "
        f"{unique_image_query_pairs:,} unique image-query pairs, "
        f"{total_model_sample_records:,} model-sample records, and "
        f"{total_reference_probe_trials:,} eligible reference-probe outcomes, together "
        "with the corresponding diagnostic probes at budgets 5, 10, 20, and 40.",
        "",
        "## Formal estimands",
        "",
        "Let `Gamma` be clean eligibility, `theta_cov` candidate coverage, and "
        "`theta_cc` complete-case persistence. Full-manifest operational stability is",
        "",
        "`Theta_op = Gamma * theta_cov * theta_cc`.",
        "",
        "The exact optimistic overstatement made by complete-case persistence is",
        "",
        "`D_total = theta_cc - Theta_op = theta_cc * (1 - Gamma * theta_cov)`.",
        "",
        "It separates into two non-negative terms:",
        "",
        "`D_coverage = theta_cc * (1 - theta_cov)`",
        "",
        "and",
        "",
        "`D_eligibility = (1 - Gamma) * theta_cov * theta_cc`.",
        "",
        "Consequently, more probes cannot make complete-case persistence converge to "
        "operational stability unless both eligibility and coverage equal one. More data "
        "only estimates the conditional estimand more precisely.",
        "",
        "## Primary cross-dataset results",
        "",
        markdown_table(display_aggregate),
        "",
        "## Hierarchical bootstrap intervals for total optimism",
        "",
        markdown_table(optimism_ci),
        "",
        "The bootstrap resamples image-query pairs and then resamples coverage and ranking "
        "outcomes within each selected pair. Intervals therefore include both finite-image "
        "and finite-reference-probe uncertainty.",
        "",
        "## Fair predictive comparison on identical complete cases",
        "",
        markdown_table(predictive_40),
        "",
        "Both predictors are evaluated on exactly the samples for which complete-case "
        "persistence exists. The target is independent 80-probe operational stability. "
        "The full-manifest coverage-aware results are additionally saved in the CSV outputs; "
        "complete-case persistence is left missing outside its observable cohort rather than "
        "being assigned an invented fallback score.",
        "",
        "## Paired bootstrap advantage at 40 probes",
        "",
        markdown_table(advantage_40[[
            "dataset", "model", "statistic", "estimate", "lower_95", "upper_95",
            "paired_sample_count",
        ]]),
        "",
        "Positive values favour the coverage-aware estimator because each statistic is "
        "defined as complete-case error minus coverage-aware error.",
        "",
        "## Probe-budget analysis",
        "",
        markdown_table(budget[[
            "dataset", "model", "probe_budget", "complete_case_persistence",
            "full_manifest_operational_stability", "total_optimism",
        ]].assign(
            dataset=lambda frame: frame["dataset"].map(DATASET_LABELS),
            model=lambda frame: frame["model"].map(MODEL_LABELS),
        )),
        "",
        "## Perturbation-family analysis",
        "",
        markdown_table(family_summary),
        "",
        "Family-level gaps demonstrate whether complete-case conditioning hides the same "
        "amount of instability under different input degradations. These are descriptive "
        "properties under the registered probe mixture, not internal causal attributions.",
        "",
        "## Correctness strata",
        "",
        markdown_table(correctness_display),
        "",
        "Correctness is used only as an external audit stratum. Neither persistence nor "
        "operational stability is interpreted as semantic correctness.",
        "",
        "## Output-contract sensitivity",
        "",
        markdown_table(contract_summary.assign(
            dataset=lambda frame: frame["dataset"].map(DATASET_LABELS),
            model=lambda frame: frame["model"].map(MODEL_LABELS),
        )),
        "",
        "The primary contract was frozen before inference. This post-primary analysis asks "
        "whether the optimism conclusion survives reasonable candidate-count and association "
        "threshold changes. Absolute magnitudes remain contract-defined.",
    ]
    if not strata.empty:
        strata_display = strata.copy()
        strata_display["model"] = strata_display["model"].map(MODEL_LABELS)
        lines.extend(
            [
                "",
                "## Registered Ref-L4 strata",
                "",
                markdown_table(strata_display[[
                    "model", "dimension", "level", "sample_count", "clean_eligibility",
                    "coverage", "conditional_ranking", "full_manifest_operational",
                    "total_optimism",
                ]]),
            ]
        )
    lines.extend(
        [
            "",
            "## Claim supported by the evidence",
            "",
            "Complete-case persistence is not a noisy version of operational stability. It is "
            "a different conditional estimand with a non-negative optimism gap that is exactly "
            "determined by clean eligibility and candidate coverage. Existing traces show that "
            "the gap persists across datasets, budgets, perturbation families, and output "
            "contracts, while its magnitude is architecture dependent.",
            "",
            "## Interpretation boundary",
            "",
            "- The theorem concerns candidate-order operational stability, not semantic correctness.",
            "- The 80-probe reference is finite, so bootstrap intervals quantify rather than erase its uncertainty.",
            "- Cross-dataset replication supports transfer of the finding, not a universal claim over every grounding architecture.",
            "- Probe-family localisation is descriptive under the registered distribution and is not a causal neural-module diagnosis.",
            "- Full-manifest comparison is primary; eligible-only and complete-case quantities remain diagnostic.",
        ]
    )
    (output / "complete_case_optimism_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "complete_case_optimism_analysis",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_records: dict[tuple[str, str], dict[tuple[int, int], CompactRecord]] = {}
    input_files = []
    for dataset, root in DATASETS.items():
        for model in MODELS:
            path = root / model / "sample_traces.jsonl.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"Loading {dataset}/{model}: {path}", flush=True)
            all_records[(dataset, model)] = load_compact_trace(path)
            input_files.append(path)

    aggregate_rows = []
    budget_rows = []
    for (dataset, model), records in all_records.items():
        aggregate_rows.append(
            {"dataset": dataset, "model": model, **pooled_estimands(records.values(), "reference")}
        )
        for budget in BUDGETS:
            budget_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    **pooled_estimands(records.values(), "diagnostic", budget),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    budget = pd.DataFrame(budget_rows)
    family = family_estimands(all_records)
    correctness = correctness_strata(all_records)
    predictive, calibration, sample_prediction_frame = predictive_tables(all_records)

    print("Running hierarchical aggregate bootstrap", flush=True)
    bootstrap_frames = []
    for index, ((dataset, model), records) in enumerate(all_records.items()):
        bootstrap_frames.append(
            bootstrap_aggregate(
                dataset,
                model,
                records,
                args.bootstrap_repetitions,
                args.seed + index * 101,
            )
        )
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    print("Running paired predictor bootstrap", flush=True)
    predictor_bootstrap = bootstrap_predictor_difference(
        sample_prediction_frame,
        args.bootstrap_repetitions,
        args.seed + 10001,
    )

    contract_detail, contract_summary = contract_sensitivity(DATASETS)
    strata = registered_strata(DATASETS["refl4"])

    aggregate.to_csv(args.output / "aggregate_optimism.csv", index=False)
    bootstrap.to_csv(args.output / "aggregate_optimism_bootstrap.csv", index=False)
    budget.to_csv(args.output / "budget_optimism.csv", index=False)
    family.to_csv(args.output / "family_optimism.csv", index=False)
    correctness.to_csv(args.output / "correctness_strata_optimism.csv", index=False)
    predictive.to_csv(args.output / "predictive_comparison.csv", index=False)
    calibration.to_csv(args.output / "predictive_calibration.csv", index=False)
    predictor_bootstrap.to_csv(args.output / "predictive_comparison_bootstrap.csv", index=False)
    contract_detail.to_csv(args.output / "contract_sensitivity_optimism.csv", index=False)
    contract_summary.to_csv(args.output / "contract_sensitivity_summary.csv", index=False)
    if not strata.empty:
        strata.to_csv(args.output / "refl4_registered_strata_optimism.csv", index=False)

    save_figures(
        aggregate,
        budget,
        family,
        predictive,
        bootstrap,
        contract_summary,
        args.output,
    )
    render_report(
        aggregate,
        bootstrap,
        budget,
        family,
        predictive,
        predictor_bootstrap,
        correctness,
        contract_summary,
        strata,
        args.output,
    )

    artifact_rows = []
    for path in sorted(args.output.iterdir()):
        if not path.is_file() or path.name in {"analysis_audit.json", "artifact_manifest.json"}:
            continue
        artifact_rows.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    (args.output / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": artifact_rows}, indent=2) + "\n", encoding="utf-8"
    )

    max_identity_residual = float(
        np.nanmax(
            np.concatenate(
                [
                    aggregate["total_identity_residual"].to_numpy(float),
                    budget["total_identity_residual"].to_numpy(float),
                    family["total_identity_residual"].to_numpy(float),
                    correctness["total_identity_residual"].to_numpy(float),
                    contract_detail["identity_residual"].to_numpy(float),
                    strata["identity_residual"].to_numpy(float) if not strata.empty else np.asarray([0.0]),
                ]
            )
        )
    )
    audit = {
        "status": "complete",
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "sample_counts": {
            f"{dataset}/{model}": len(records)
            for (dataset, model), records in all_records.items()
        },
        "total_unique_model_sample_records": sum(len(records) for records in all_records.values()),
        "total_reference_probe_trials_on_eligible_samples": int(
            aggregate["eligible_probe_trials"].sum()
        ),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "maximum_optimism_identity_residual": max_identity_residual,
        "input_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in input_files},
        "outputs": sorted(path.name for path in args.output.iterdir() if path.is_file()),
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if max_identity_residual > 1e-10:
        raise AssertionError(f"optimism identity residual too large: {max_identity_residual}")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
