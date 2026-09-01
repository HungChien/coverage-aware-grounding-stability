from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("groundingdino", "owlv2", "yoloworld")
MODEL_LABELS = {
    "groundingdino": "GroundingDINO",
    "owlv2": "OWLv2",
    "yoloworld": "YOLO-World",
}
FAMILIES = ("blur", "brightness", "jpeg", "resolution", "gaussian_noise")
FAMILY_LABELS = {
    "blur": "Blur",
    "brightness": "Brightness",
    "jpeg": "JPEG",
    "resolution": "Resolution",
    "gaussian_noise": "Noise",
}
FAILURE_LABELS = (
    "winner_missing",
    "competitor_missing",
    "threatening_birth",
    "ranking_reversal",
)
FULL_LABELS = ("clean_ineligible", *FAILURE_LABELS, "stable")
DISPLAY_LABELS = {
    "clean_ineligible": "Clean ineligible",
    "winner_missing": "Winner loss",
    "competitor_missing": "Competitor loss",
    "threatening_birth": "Candidate birth",
    "ranking_reversal": "Rank reversal",
    "stable": "Stable",
}
COLORS = {
    "clean_ineligible": "#777777",
    "winner_missing": "#d95f02",
    "competitor_missing": "#e6ab02",
    "threatening_birth": "#7570b3",
    "ranking_reversal": "#1b9e77",
    "stable": "#4c78a8",
}
MODEL_COLORS = {
    "groundingdino": "#1f77b4",
    "owlv2": "#9467bd",
    "yoloworld": "#d95f02",
}
DATASETS = (
    (
        "RefCOCO",
        ROOT / "config" / "operational_benchmark_owlv2_control_v1.json",
        ROOT / "results" / "operational_benchmark_v1",
    ),
    (
        "RefCOCO+",
        ROOT / "config" / "operational_transfer_refcocoplus_owlv2_control_v1.json",
        ROOT / "results" / "operational_transfer_refcocoplus_v1",
    ),
    (
        "Ref-L4",
        ROOT / "config" / "operational_transfer_refl4_owlv2_control_v1.json",
        ROOT / "results" / "operational_transfer_refl4_v1",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_latest_traces(path: Path) -> dict[tuple[int, int], dict]:
    records: dict[tuple[int, int], dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (int(record["image_id"]), int(record["ref_id"]))
            records[key] = record
    return records


def reference_probes(record: dict) -> list[dict]:
    return [probe for probe in record.get("probes", []) if probe["split"] == "reference"]


def normalised_strength(family: str, severity: float) -> float:
    if family == "blur":
        value = (severity - 0.3) / (2.5 - 0.3)
    elif family == "brightness":
        value = abs(severity - 1.0) / 0.3
    elif family == "jpeg":
        value = (95.0 - severity) / (95.0 - 40.0)
    elif family == "resolution":
        value = (1.0 - severity) / (1.0 - 0.5)
    elif family == "gaussian_noise":
        value = severity / 0.04
    else:
        raise ValueError(f"unknown family {family}")
    return float(np.clip(value, 0.0, 1.0))


def bootstrap_mean_rows(
    matrix: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    values = np.empty((repetitions, matrix.shape[1]), dtype=float)
    for start in range(0, repetitions, 100):
        width = min(100, repetitions - start)
        indices = rng.integers(0, n, size=(width, n))
        values[start : start + width] = matrix[indices].mean(axis=1)
    return np.quantile(values, 0.025, axis=0), np.quantile(values, 0.975, axis=0)


def bootstrap_ratio_rows(
    numerators: np.ndarray,
    denominators: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = numerators.shape[0]
    values = np.empty((repetitions, numerators.shape[1]), dtype=float)
    for start in range(0, repetitions, 100):
        width = min(100, repetitions - start)
        indices = rng.integers(0, n, size=(width, n))
        num = numerators[indices].sum(axis=1)
        den = denominators[indices].sum(axis=1)[:, None]
        values[start : start + width] = np.divide(
            num, den, out=np.zeros_like(num), where=den > 0
        )
    return np.quantile(values, 0.025, axis=0), np.quantile(values, 0.975, axis=0)


def load_dataset(
    dataset: str, config_path: Path, result_root: Path
) -> tuple[list[tuple[int, int]], dict[str, dict[tuple[int, int], dict]], dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = [(int(row["image_id"]), int(row["ref_id"])) for row in manifest]
    traces = {
        model: load_latest_traces(result_root / model / "sample_traces.jsonl.gz")
        for model in MODELS
    }
    for model, records in traces.items():
        missing = set(keys) - set(records)
        if missing:
            raise AssertionError(f"{dataset} {model}: {len(missing)} manifest keys missing")
    return keys, traces, config


def analyse(
    output: Path, repetitions: int, seed: int
) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)
    cap_rows: list[dict] = []
    full_rows: list[dict] = []
    common_rows: list[dict] = []
    common_count_rows: list[dict] = []
    selection_ci_rows: list[dict] = []
    severity_rows: list[dict] = []
    family_rows: list[dict] = []

    for dataset, config_path, result_root in DATASETS:
        keys, traces, config = load_dataset(dataset, config_path, result_root)
        n = len(keys)
        reference_count = int(config["reference_probes_per_family"]) * len(FAMILIES)
        eligible_sets = {
            model: {key for key in keys if int(traces[model][key]["clean_eligible"]) == 1}
            for model in MODELS
        }
        common = set.intersection(*(eligible_sets[model] for model in MODELS))
        common_count_rows.append(
            {
                "dataset": dataset,
                "manifest_pairs": n,
                "all_three_eligible_pairs": len(common),
                "all_three_eligible_fraction": len(common) / n,
                **{
                    f"{model}_eligible_pairs": len(eligible_sets[model])
                    for model in MODELS
                },
            }
        )

        per_model_family: dict[str, dict[str, float]] = {}
        for model in MODELS:
            records = traces[model]
            eligible_vector = np.asarray(
                [int(records[key]["clean_eligible"]) for key in keys], dtype=float
            )
            full_matrix = np.zeros((n, len(FULL_LABELS)), dtype=float)
            common_num = np.zeros((len(common), len(FAILURE_LABELS)), dtype=float)
            common_den = np.zeros(len(common), dtype=float)
            common_keys = sorted(common)
            severity_success = np.zeros((n, len(FAMILIES), 5), dtype=float)
            severity_trials = np.zeros((n, len(FAMILIES), 5), dtype=float)
            family_success = Counter()
            family_trials = Counter()
            cap_hit_trials = 0
            cap_hit_failures = 0
            cap_failure_counts = Counter()
            stable_total = 0
            clean_cap_hits = 0
            clean_cap_hits_under_five = 0

            for row_index, key in enumerate(keys):
                record = records[key]
                raw_clean_count = len(record.get("raw_clean_candidates", []))
                if raw_clean_count >= 20:
                    clean_cap_hits += 1
                    if len(record.get("tracked_clean_candidates", [])) < 5:
                        clean_cap_hits_under_five += 1
                if not int(record["clean_eligible"]):
                    full_matrix[row_index, FULL_LABELS.index("clean_ineligible")] = 1.0
                    continue
                probes = reference_probes(record)
                if len(probes) != reference_count:
                    raise AssertionError(
                        f"{dataset} {model} {key}: expected {reference_count} reference probes"
                    )
                counts = Counter(probe["outcome"]["primary_failure"] for probe in probes)
                for label in FULL_LABELS[1:]:
                    full_matrix[row_index, FULL_LABELS.index(label)] = counts[label] / reference_count
                stable_total += counts["stable"]
                for probe in probes:
                    family = str(probe["spec"]["family"])
                    outcome = int(probe["outcome"]["operational_stable"])
                    family_success[family] += outcome
                    family_trials[family] += 1
                    strength = normalised_strength(family, float(probe["spec"]["severity"]))
                    bin_index = min(4, int(math.floor(strength * 5.0)))
                    fi = FAMILIES.index(family)
                    severity_success[row_index, fi, bin_index] += outcome
                    severity_trials[row_index, fi, bin_index] += 1
                    if len(probe.get("candidates", [])) >= 20:
                        cap_hit_trials += 1
                        if not outcome:
                            cap_hit_failures += 1
                            cap_failure_counts[probe["outcome"]["primary_failure"]] += 1

            lower, upper = bootstrap_mean_rows(full_matrix, repetitions, rng)
            full_point = full_matrix.mean(axis=0)
            for index, label in enumerate(FULL_LABELS):
                full_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "category": label,
                        "risk": full_point[index],
                        "lower": lower[index],
                        "upper": upper[index],
                        "denominator_pair_probe_slots": n * reference_count,
                    }
                )

            for row_index, key in enumerate(common_keys):
                probes = reference_probes(records[key])
                counts = Counter(
                    probe["outcome"]["primary_failure"]
                    for probe in probes
                    if not int(probe["outcome"]["operational_stable"])
                )
                for label_index, label in enumerate(FAILURE_LABELS):
                    common_num[row_index, label_index] = counts[label]
                common_den[row_index] = sum(counts.values())
            c_lower, c_upper = bootstrap_ratio_rows(
                common_num, common_den, repetitions, rng
            )
            common_point = common_num.sum(axis=0) / common_den.sum()
            for index, label in enumerate(FAILURE_LABELS):
                row = {
                    "dataset": dataset,
                    "model": model,
                    "category": label,
                    "share": common_point[index],
                    "lower": c_lower[index],
                    "upper": c_upper[index],
                    "common_eligible_pairs": len(common),
                    "failed_reference_probes": int(common_den.sum()),
                }
                common_rows.append(row)
                selection_ci_rows.append({"analysis": "common_support", **row})

            observed = stable_total / (n * reference_count)
            correction = cap_hit_failures / (n * reference_count)
            cap_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "manifest_pairs": n,
                    "clean_cap20_fraction": clean_cap_hits / n,
                    "clean_cap20_under5_tracked_fraction": clean_cap_hits_under_five / n,
                    "perturbed_cap20_reference_trials": cap_hit_trials,
                    "perturbed_cap20_reference_fraction": cap_hit_trials
                    / (len(eligible_sets[model]) * reference_count),
                    "cap20_failed_reference_trials": cap_hit_failures,
                    "observed_full_manifest_stability": observed,
                    "pessimistic_max_correction": correction,
                    "pessimistic_stability_upper_bound": observed + correction,
                    **{
                        f"cap20_failed_{label}": cap_failure_counts[label]
                        for label in FAILURE_LABELS
                    },
                }
            )

            eligibility = len(eligible_sets[model]) / n
            per_model_family[model] = {}
            for family_index, family in enumerate(FAMILIES):
                family_value = eligibility * family_success[family] / family_trials[family]
                per_model_family[model][family] = family_value
                family_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "family": family,
                        "full_manifest_family_stability": family_value,
                        "eligibility": eligibility,
                        "eligible_reference_trials": family_trials[family],
                    }
                )
                for bin_index in range(5):
                    success = severity_success[:, family_index, bin_index]
                    trials = severity_trials[:, family_index, bin_index]
                    point = eligibility * success.sum() / trials.sum()
                    boot = np.empty(repetitions, dtype=float)
                    for start in range(0, repetitions, 100):
                        width = min(100, repetitions - start)
                        indices = rng.integers(0, n, size=(width, n))
                        ebar = eligible_vector[indices].mean(axis=1)
                        num = success[indices].sum(axis=1)
                        den = trials[indices].sum(axis=1)
                        boot[start : start + width] = ebar * np.divide(
                            num, den, out=np.zeros_like(num), where=den > 0
                        )
                    severity_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "family": family,
                            "severity_bin": bin_index + 1,
                            "strength_low": bin_index / 5,
                            "strength_high": (bin_index + 1) / 5,
                            "strength_mid": (bin_index + 0.5) / 5,
                            "full_manifest_stability": point,
                            "lower": float(np.quantile(boot, 0.025)),
                            "upper": float(np.quantile(boot, 0.975)),
                            "eligible_probe_trials": int(trials.sum()),
                        }
                    )

        comparisons = (
            ("groundingdino", "owlv2"),
            ("groundingdino", "yoloworld"),
            ("owlv2", "yoloworld"),
        )
        for model_a, model_b in comparisons:
            gaps = {
                family: per_model_family[model_a][family]
                - per_model_family[model_b][family]
                for family in FAMILIES
            }
            family_rows.append(
                {
                    "dataset": dataset,
                    "model": f"{model_a}_minus_{model_b}",
                    "family": "mixture_gap_summary",
                    "full_manifest_family_stability": np.mean(list(gaps.values())),
                    "eligibility": np.nan,
                    "eligible_reference_trials": np.nan,
                    "minimum_arbitrary_mixture_gap": min(gaps.values()),
                    "maximum_arbitrary_mixture_gap": max(gaps.values()),
                    "minimum_family": min(gaps, key=gaps.get),
                    "maximum_family": max(gaps, key=gaps.get),
                    "ranking_invariant_for_all_nonnegative_family_weights": all(
                        gap > 0 for gap in gaps.values()
                    ),
                }
            )

    tables = {
        "cap_truncation_audit.csv": cap_rows,
        "common_eligibility_counts.csv": common_count_rows,
        "full_manifest_failure_risk.csv": full_rows,
        "common_support_failure_composition.csv": common_rows,
        "failure_selection_bootstrap.csv": selection_ci_rows,
        "severity_stability.csv": severity_rows,
        "family_weight_sensitivity.csv": family_rows,
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / name, index=False)
    return tables


def plot_failure_selection(output: Path) -> None:
    full = pd.read_csv(output / "full_manifest_failure_risk.csv")
    common = pd.read_csv(output / "common_support_failure_composition.csv")
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.0), sharey="col")
    for row_index, (dataset, _, _) in enumerate(DATASETS):
        for col_index, (frame, categories, value, title) in enumerate(
            (
                (full, FULL_LABELS, "risk", "All manifest pair-probe slots"),
                (
                    common,
                    FAILURE_LABELS,
                    "share",
                    "Common-eligibility failed probes",
                ),
            )
        ):
            ax = axes[row_index, col_index]
            subset = frame[frame["dataset"] == dataset]
            bottom = np.zeros(len(MODELS))
            for category in categories:
                values = np.asarray(
                    [
                        subset[
                            (subset["model"] == model)
                            & (subset["category"] == category)
                        ][value].iloc[0]
                        for model in MODELS
                    ]
                )
                ax.bar(
                    np.arange(len(MODELS)),
                    values,
                    bottom=bottom,
                    color=COLORS[category],
                    label=DISPLAY_LABELS[category],
                    width=0.72,
                )
                bottom += values
            ax.set_xticks(np.arange(len(MODELS)), [MODEL_LABELS[m] for m in MODELS], rotation=15)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(f"{dataset}: {title}", fontsize=10)
            if col_index == 0:
                ax.set_ylabel("Share")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Failure profiles before and after common-support restriction", y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    fig.savefig(output / "failure_selection_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_severity(output: Path) -> None:
    frame = pd.read_csv(output / "severity_stability.csv")
    for dataset, _, _ in DATASETS:
        fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.3), sharex=True, sharey=True)
        axes = axes.ravel()
        subset = frame[frame["dataset"] == dataset]
        for family_index, family in enumerate(FAMILIES):
            ax = axes[family_index]
            for model in MODELS:
                values = subset[
                    (subset["family"] == family) & (subset["model"] == model)
                ].sort_values("severity_bin")
                x = values["strength_mid"].to_numpy()
                y = values["full_manifest_stability"].to_numpy()
                ax.plot(x, y, marker="o", linewidth=1.8, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
                ax.fill_between(
                    x,
                    values["lower"].to_numpy(),
                    values["upper"].to_numpy(),
                    color=MODEL_COLORS[model],
                    alpha=0.12,
                    linewidth=0,
                )
            ax.set_title(FAMILY_LABELS[family])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
        axes[-1].axis("off")
        for ax in axes[3:5]:
            ax.set_xlabel("Normalised within-family distortion")
        axes[0].set_ylabel("Full-manifest stability")
        axes[3].set_ylabel("Full-manifest stability")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.11), frameon=False)
        fig.suptitle(f"{dataset}: severity-response curves", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        slug = dataset.lower().replace("+", "plus").replace("-", "")
        fig.savefig(output / f"severity_stability_{slug}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_family_weight(output: Path) -> None:
    frame = pd.read_csv(output / "family_weight_sensitivity.csv")
    summary = frame[frame["family"] == "mixture_gap_summary"].copy()
    comparisons = (
        ("groundingdino_minus_owlv2", "GDINO - OWLv2"),
        ("groundingdino_minus_yoloworld", "GDINO - YOLO"),
        ("owlv2_minus_yoloworld", "OWLv2 - YOLO"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    for ax, (dataset, _, _) in zip(axes, DATASETS):
        subset = summary[summary["dataset"] == dataset]
        for index, (comparison, label) in enumerate(comparisons):
            row = subset[subset["model"] == comparison].iloc[0]
            ax.hlines(
                index,
                row["minimum_arbitrary_mixture_gap"],
                row["maximum_arbitrary_mixture_gap"],
                color="#4c78a8",
                linewidth=5,
            )
            ax.plot(
                row["full_manifest_family_stability"],
                index,
                "o",
                color="black",
                markersize=5,
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(dataset)
        ax.set_xlabel("Stability gap")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(range(len(comparisons)), [label for _, label in comparisons])
    fig.suptitle("Model gaps under arbitrary non-negative family mixtures\n(line: attainable range; dot: equal-weight mixture)")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(output / "family_weight_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_audit(output: Path, repetitions: int, seed: int) -> None:
    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.name == "analysis_audit.json" or not path.is_file():
            continue
        artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    cap = pd.read_csv(output / "cap_truncation_audit.csv")
    common = pd.read_csv(output / "common_eligibility_counts.csv")
    family = pd.read_csv(output / "family_weight_sensitivity.csv")
    summaries = family[family["family"] == "mixture_gap_summary"]
    audit = {
        "analysis": "reviewer_risk_controls",
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "clean_cap_hit_with_fewer_than_five_tracked_max": float(
            cap["clean_cap20_under5_tracked_fraction"].max()
        ),
        "maximum_pessimistic_cap_correction": float(
            cap["pessimistic_max_correction"].max()
        ),
        "common_eligibility_pairs": common.set_index("dataset")[
            "all_three_eligible_pairs"
        ].to_dict(),
        "all_family_mixture_rankings_invariant": bool(
            summaries["ranking_invariant_for_all_nonnegative_family_weights"].all()
        ),
        "artifacts": artifacts,
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reviewer_risk_controls",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    analyse(args.output, args.bootstrap_repetitions, args.seed)
    plot_failure_selection(args.output)
    plot_severity(args.output)
    plot_family_weight(args.output)
    write_audit(args.output, args.bootstrap_repetitions, args.seed)
    print(f"Saved reviewer-risk controls to {args.output}")


if __name__ == "__main__":
    main()
