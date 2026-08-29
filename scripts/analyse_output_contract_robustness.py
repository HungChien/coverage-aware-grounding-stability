from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.operational_stability import select_spatially_distinct
from src.output_contract_robustness import (
    ContractSetting,
    contract_envelope,
    default_setting_index,
    finite_grid_ranking,
    paired_bootstrap_minimum_gap,
    registered_settings,
)
from src.reliability import Candidate, iou_xyxy


DATASETS = {
    "RefCOCO": ROOT / "results" / "operational_benchmark_v1",
    "RefCOCO+": ROOT / "results" / "operational_transfer_refcocoplus_v1",
    "Ref-L4": ROOT / "results" / "operational_transfer_refl4_v1",
}
MODELS = ("groundingdino", "yoloworld")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_candidates(payload: list[dict]) -> list[Candidate]:
    return [
        Candidate(
            box=tuple(float(value) for value in item["box"]),
            score=float(item["score"]),
            label=str(item.get("label", "")),
        )
        for item in payload
    ]


def load_traces(path: Path) -> dict[tuple[int, int], dict]:
    records: dict[tuple[int, int], dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (int(record["image_id"]), int(record["ref_id"]))
            records[key] = record
    return records


def associate_all(
    clean: list[Candidate], perturbed: list[Candidate]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unconstrained Hungarian mapping and assigned IoUs."""

    mapping = np.full(len(clean), -1, dtype=int)
    assigned = np.zeros(len(clean), dtype=float)
    if not clean or not perturbed:
        return mapping, assigned
    overlaps = np.asarray(
        [
            [iou_xyxy(source.box, target.box) for target in perturbed]
            for source in clean
        ],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(1.0 - overlaps)
    mapping[rows] = columns
    assigned[rows] = overlaps[rows, columns]
    return mapping, assigned


def outcome_from_assignment(
    clean: list[Candidate],
    perturbed: list[Candidate],
    unconstrained_mapping: np.ndarray,
    assigned_ious: np.ndarray,
    match_threshold: float,
    birth_threshold: float,
) -> tuple[int, int, int | None, str]:
    mapping = unconstrained_mapping.copy()
    mapping[assigned_ious < match_threshold] = -1
    missing = np.flatnonzero(mapping < 0)
    used = {int(value) for value in mapping if value >= 0}

    threatening_birth = False
    if mapping[0] >= 0:
        winner_score = float(perturbed[int(mapping[0])].score)
        matched = [perturbed[index] for index in sorted(used)]
        for index, candidate in enumerate(perturbed):
            if index in used or candidate.score < winner_score:
                continue
            if all(
                iou_xyxy(candidate.box, accepted.box) < birth_threshold
                for accepted in matched
            ):
                threatening_birth = True
                break

    coverage = int(missing.size == 0 and not threatening_birth)
    rank_stable: int | None = None
    operational = 0
    if coverage:
        matched_scores = np.asarray(
            [perturbed[int(index)].score for index in mapping], dtype=float
        )
        rank_stable = int(np.all(matched_scores[0] - matched_scores[1:] > 0.0))
        operational = rank_stable

    if coverage and rank_stable:
        failure = "stable"
    elif 0 in missing:
        failure = "winner_missing"
    elif missing.size:
        failure = "competitor_missing"
    elif threatening_birth:
        failure = "threatening_birth"
    else:
        failure = "ranking_reversal"
    return coverage, operational, rank_stable, failure


def replay_model(
    records: dict[tuple[int, int], dict],
    settings: list[ContractSetting],
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    keys = sorted(records)
    setting_lookup = {
        (
            setting.tracked_candidate_count,
            setting.exposed_candidate_count,
            setting.match_iou_threshold,
            setting.birth_duplicate_iou_threshold,
        ): index
        for index, setting in enumerate(settings)
    }
    tracked_values = sorted({value.tracked_candidate_count for value in settings})
    exposed_values = sorted({value.exposed_candidate_count for value in settings})
    match_values = sorted({value.match_iou_threshold for value in settings})
    birth_values = sorted({value.birth_duplicate_iou_threshold for value in settings})
    duplicate = settings[0].duplicate_iou_threshold

    sample_values = np.zeros((len(settings), len(keys)), dtype=np.float32)
    eligible_counts = np.zeros(len(settings), dtype=np.int64)
    coverage_successes = np.zeros(len(settings), dtype=np.int64)
    rank_successes = np.zeros(len(settings), dtype=np.int64)
    rank_trials = np.zeros(len(settings), dtype=np.int64)
    probe_trials = np.zeros(len(settings), dtype=np.int64)
    failures = {
        cause: np.zeros(len(settings), dtype=np.int64)
        for cause in (
            "winner_missing",
            "competitor_missing",
            "threatening_birth",
            "ranking_reversal",
        )
    }
    clean_cap_hits = 0
    reference_candidate_counts: list[int] = []
    raw_probe_field_count = 0
    reference_probe_count = 0

    for sample_index, key in enumerate(keys):
        record = records[key]
        raw_clean = to_candidates(record["raw_clean_candidates"])
        clean_cap_hits += int(len(raw_clean) >= 20)
        clean_maximum = select_spatially_distinct(
            raw_clean,
            maximum=max(tracked_values),
            duplicate_iou_threshold=duplicate,
        )
        if len(clean_maximum) < 2:
            continue
        eligible_counts += 1

        reference_probes = [
            probe for probe in record["probes"] if probe["split"] == "reference"
        ]
        sample_successes = np.zeros(len(settings), dtype=np.int64)
        for probe in reference_probes:
            reference_probe_count += 1
            raw_probe_field_count += int("raw_candidates" in probe)
            # V1 traces expose only this post-0.70-deduplication field.  The
            # registered replay grid therefore holds duplicate IoU fixed.
            perturbed_pool = to_candidates(probe["candidates"])
            reference_candidate_counts.append(len(perturbed_pool))
            for tracked in tracked_values:
                clean = clean_maximum[:tracked]
                for exposed in exposed_values:
                    perturbed = perturbed_pool[:exposed]
                    mapping, assigned = associate_all(clean, perturbed)
                    for match in match_values:
                        for birth in birth_values:
                            index = setting_lookup[(tracked, exposed, match, birth)]
                            coverage, operational, rank_stable, failure = (
                                outcome_from_assignment(
                                    clean,
                                    perturbed,
                                    mapping,
                                    assigned,
                                    match,
                                    birth,
                                )
                            )
                            probe_trials[index] += 1
                            coverage_successes[index] += coverage
                            if coverage:
                                rank_trials[index] += 1
                                rank_successes[index] += int(rank_stable)
                            if failure != "stable":
                                failures[failure][index] += 1
                            sample_successes[index] += operational
        if reference_probes:
            sample_values[:, sample_index] = (
                sample_successes / len(reference_probes)
            ).astype(np.float32)

    rows = []
    for index, setting in enumerate(settings):
        total_failures = int(sum(values[index] for values in failures.values()))
        row = {
            "setting_index": index,
            "setting_id": setting.setting_id,
            "tracked_candidate_count": setting.tracked_candidate_count,
            "exposed_candidate_count": setting.exposed_candidate_count,
            "duplicate_iou_threshold": setting.duplicate_iou_threshold,
            "match_iou_threshold": setting.match_iou_threshold,
            "birth_duplicate_iou_threshold": setting.birth_duplicate_iou_threshold,
            "sample_count": len(keys),
            "eligible_count": int(eligible_counts[index]),
            "clean_eligibility": float(eligible_counts[index] / len(keys)),
            "reference_probe_count": int(probe_trials[index]),
            "coverage": (
                float(coverage_successes[index] / probe_trials[index])
                if probe_trials[index]
                else 0.0
            ),
            "conditional_ranking": (
                float(rank_successes[index] / rank_trials[index])
                if rank_trials[index]
                else np.nan
            ),
            "full_manifest_operational": float(sample_values[index].mean()),
            "eligible_only_operational": (
                float(sample_values[index].sum() / eligible_counts[index])
                if eligible_counts[index]
                else 0.0
            ),
        }
        for cause, values in failures.items():
            row[f"{cause}_share"] = (
                0.0 if total_failures == 0 else float(values[index] / total_failures)
            )
        rows.append(row)

    saturation = {
        "sample_count": len(keys),
        "raw_clean_cap20_fraction": clean_cap_hits / len(keys),
        "reference_probe_count": reference_probe_count,
        "post_dedup_probe_cap20_fraction": (
            float(np.mean(np.asarray(reference_candidate_counts) >= 20))
            if reference_candidate_counts
            else 0.0
        ),
        "median_post_dedup_probe_candidate_count": (
            float(np.median(reference_candidate_counts))
            if reference_candidate_counts
            else 0.0
        ),
        "raw_perturbed_field_fraction": (
            raw_probe_field_count / reference_probe_count
            if reference_probe_count
            else 0.0
        ),
        "sample_keys": [f"{image_id}:{ref_id}" for image_id, ref_id in keys],
    }
    return sample_values, pd.DataFrame(rows), saturation


def one_factor_rows(
    frame: pd.DataFrame, preregistration: dict, dataset: str, model: str
) -> list[dict]:
    default = preregistration["default_output_contract"]
    parameters = (
        "tracked_candidate_count",
        "exposed_candidate_count",
        "match_iou_threshold",
        "birth_duplicate_iou_threshold",
    )
    rows = []
    for parameter in parameters:
        subset = frame.copy()
        for fixed in parameters:
            if fixed != parameter:
                subset = subset[subset[fixed] == default[fixed]]
        subset = subset.sort_values(parameter)
        values = subset["full_manifest_operational"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "parameter": parameter,
                "tested_values": ";".join(str(value) for value in subset[parameter]),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "range_width": float(values.max() - values.min()),
                "default": float(
                    subset.loc[
                        subset[parameter] == default[parameter],
                        "full_manifest_operational",
                    ].iloc[0]
                ),
            }
        )
    return rows


def duplicate_eligibility_rows(
    records: dict[tuple[int, int], dict], preregistration: dict, dataset: str, model: str
) -> list[dict]:
    rows = []
    maximum = int(
        preregistration["default_output_contract"]["tracked_candidate_count"]
    )
    for threshold in preregistration["partial_clean_only_grid"][
        "duplicate_iou_threshold"
    ]:
        eligible = []
        retained = []
        for record in records.values():
            clean = select_spatially_distinct(
                to_candidates(record["raw_clean_candidates"]),
                maximum=maximum,
                duplicate_iou_threshold=float(threshold),
            )
            eligible.append(int(len(clean) >= 2))
            retained.append(len(clean))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "duplicate_iou_threshold": float(threshold),
                "clean_eligibility": float(np.mean(eligible)),
                "mean_retained_clean_candidates": float(np.mean(retained)),
                "scope": "clean_eligibility_only",
            }
        )
    return rows


def write_report(
    output: Path,
    envelope: pd.DataFrame,
    ranking: pd.DataFrame,
    one_factor: pd.DataFrame,
    saturation: pd.DataFrame,
    duplicate: pd.DataFrame,
    preregistration: dict,
) -> None:
    lines = [
        "# Output-Contract Robustness Report",
        "",
        "## Scope",
        "",
        (
            "This report replays the frozen candidate-order event on the full "
            "RefCOCO, RefCOCO+, and Ref-L4 traces. It evaluates all 108 "
            "registered combinations of tracked count, exposed count, match "
            "IoU, and birth-novelty IoU, with duplicate suppression fixed at 0.70."
        ),
        "",
        "The evidence is a deterministic finite-grid robustness result, not a claim over every real-valued threshold.",
        "",
        "## Model-ranking robustness",
        "",
        "| Dataset | Minimum same-contract gap | 95% paired-bootstrap interval | Invariant | Strong envelope separation |",
        "|---|---:|---:|:---:|:---:|",
    ]
    for row in ranking.to_dict(orient="records"):
        lines.append(
            f"| {row['dataset']} | {row['minimum_same_contract_gap']:.4f} | "
            f"[{row['bootstrap_lower_95']:.4f}, {row['bootstrap_upper_95']:.4f}] | "
            f"{'yes' if row['same_contract_ranking_invariant'] else 'no'} | "
            f"{'yes' if row['strong_interval_separation'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                "A positive minimum gap proves that GroundingDINO ranks above "
                "YOLO-World at every registered identical contract. The paired "
                "bootstrap recomputes the worst setting in every repetition."
            ),
            "",
            "## Absolute-value sensitivity envelopes",
            "",
            "| Dataset | Model | Default | Minimum | Maximum | Width | Maximum departure |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in envelope.to_dict(orient="records"):
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['default']:.4f} | "
            f"{row['minimum']:.4f} | {row['maximum']:.4f} | {row['width']:.4f} | "
            f"{row['maximum_absolute_departure']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These ranges are contract sensitivity envelopes, not confidence intervals.",
            "",
            "## One-factor sensitivity around the default",
            "",
            "| Dataset | Model | Parameter | Tested values | Range width |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in one_factor.to_dict(orient="records"):
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['parameter']} | "
            f"{row['tested_values']} | {row['range_width']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate-pool diagnostics",
            "",
            "| Dataset | Model | Clean pool at cap 20 | Perturbed post-dedup pool at cap 20 | Median perturbed count |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in saturation.to_dict(orient="records"):
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['raw_clean_cap20_fraction']:.3f} | "
            f"{row['post_dedup_probe_cap20_fraction']:.3f} | "
            f"{row['median_post_dedup_probe_candidate_count']:.1f} |"
        )
    lines.extend(
        [
            "",
            (
                "A cap hit indicates possible truncation, not a known error. It "
                "motivates the prospectively registered raw pool of 50 candidates."
            ),
            "",
            "## Duplicate-threshold boundary",
            "",
            (
                "Existing perturbed traces were already deduplicated at IoU 0.70. "
                "Therefore only clean eligibility is reported under alternative "
                "duplicate thresholds; full operational duplicate sensitivity is "
                "not identifiable from the old traces."
            ),
            "",
            "| Dataset | Model | Duplicate IoU | Clean eligibility | Mean retained candidates |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in duplicate.to_dict(orient="records"):
        lines.append(
            f"| {row['dataset']} | {row['model']} | "
            f"{row['duplicate_iou_threshold']:.2f} | {row['clean_eligibility']:.4f} | "
            f"{row['mean_retained_clean_candidates']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Why the default contract is retained",
            "",
            "- `tracked_candidate_count = 5` observes more than a top-two contest while keeping one-to-one association auditable.",
            "- `exposed_candidate_count = 20` exposes four times the tracked universe and is the largest value identifiable in v1 traces.",
            "- `duplicate_iou_threshold = 0.70` treats near-overlapping localisation variants as duplicates and is shared with birth novelty.",
            "- `match_iou_threshold = 0.15` tolerates corruption-induced displacement but rejects negligible spatial continuity.",
            "- `birth_duplicate_iou_threshold = 0.70` makes suppression and novelty use one geometric convention.",
            "- Hungarian one-to-one association prevents two clean candidates from claiming the same perturbed output.",
            "- Strict score order treats ties as unstable and avoids implementation-specific tie breaking.",
            "- Missing candidates and threatening births are coverage failures; no artificial scores are imputed.",
            "",
            "## Preregistration and trace upgrade",
            "",
            f"Protocol: `{preregistration['protocol_version']}`. The enhanced grid was frozen before this replay analysis.",
            "",
            (
                "The runner now supports `raw_candidate_pool_size` and stores both "
                "pre-contract and post-contract perturbed candidates. Future "
                "confirmatory runs use a pool of 50, enabling complete offline "
                "replay of duplicate suppression and wider exposure settings."
            ),
        ]
    )
    (output / "output_contract_robustness_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT / "config" / "output_contract_preregistration_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "output_contract_robustness",
    )
    parser.add_argument("--bootstrap-repetitions", type=int)
    args = parser.parse_args()

    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    settings = registered_settings(preregistration)
    default_index = default_setting_index(settings, preregistration)
    repetitions = int(
        args.bootstrap_repetitions
        or preregistration["robustness_statistics"][
            "paired_image_bootstrap_repetitions"
        ]
    )
    seed = int(preregistration["robustness_statistics"]["bootstrap_seed"])
    args.output.mkdir(parents=True, exist_ok=True)

    grid_frames = []
    sample_matrices: dict[tuple[str, str], np.ndarray] = {}
    sample_keys: dict[tuple[str, str], list[str]] = {}
    saturation_rows = []
    duplicate_rows = []
    input_hashes = {}

    for dataset, result_root in DATASETS.items():
        for model in MODELS:
            trace_path = result_root / model / "sample_traces.jsonl.gz"
            if not trace_path.exists():
                raise FileNotFoundError(trace_path)
            print(f"Replaying {dataset} / {model}: {trace_path}", flush=True)
            records = load_traces(trace_path)
            matrix, frame, saturation = replay_model(records, settings)
            frame.insert(0, "model", model)
            frame.insert(0, "dataset", dataset)
            grid_frames.append(frame)
            sample_matrices[(dataset, model)] = matrix
            sample_keys[(dataset, model)] = saturation.pop("sample_keys")
            saturation_rows.append({"dataset": dataset, "model": model, **saturation})
            duplicate_rows.extend(
                duplicate_eligibility_rows(
                    records, preregistration, dataset, model
                )
            )
            input_hashes[str(trace_path.relative_to(ROOT))] = sha256(trace_path)

    grid = pd.concat(grid_frames, ignore_index=True)
    grid.to_csv(args.output / "contract_grid_results.csv", index=False)
    saturation = pd.DataFrame(saturation_rows)
    saturation.to_csv(args.output / "candidate_pool_saturation.csv", index=False)
    duplicate = pd.DataFrame(duplicate_rows)
    duplicate.to_csv(
        args.output / "duplicate_clean_eligibility_sensitivity.csv", index=False
    )

    envelope_rows = []
    one_factor = []
    ranking_rows = []
    bootstrap_payload = {}
    for dataset in DATASETS:
        model_means = {}
        for model in MODELS:
            subset = grid[(grid["dataset"] == dataset) & (grid["model"] == model)]
            subset = subset.sort_values("setting_index")
            values = subset["full_manifest_operational"].to_numpy(dtype=float)
            model_means[model] = values
            summary = contract_envelope(values, default_index)
            envelope_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    **{key: value for key, value in summary.items() if not key.endswith("_index")},
                    "minimum_setting_id": settings[int(summary["minimum_index"])].setting_id,
                    "maximum_setting_id": settings[int(summary["maximum_index"])].setting_id,
                }
            )
            one_factor.extend(
                one_factor_rows(subset, preregistration, dataset, model)
            )

        keys_a = sample_keys[(dataset, MODELS[0])]
        keys_b = sample_keys[(dataset, MODELS[1])]
        if keys_a != keys_b:
            raise ValueError(f"paired sample keys differ for {dataset}")
        ranking = finite_grid_ranking(
            model_means["groundingdino"], model_means["yoloworld"]
        )
        bootstrap = paired_bootstrap_minimum_gap(
            sample_matrices[(dataset, "groundingdino")],
            sample_matrices[(dataset, "yoloworld")],
            repetitions=repetitions,
            seed=seed,
        )
        bootstrap_payload[dataset] = bootstrap
        ranking_rows.append(
            {
                "dataset": dataset,
                **ranking,
                "worst_setting_id": settings[
                    int(ranking["worst_setting_index"])
                ].setting_id,
                "bootstrap_repetitions": repetitions,
                "bootstrap_lower_95": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_median": float(np.median(bootstrap)),
                "bootstrap_upper_95": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_probability_minimum_gap_positive": float(
                    np.mean(bootstrap > 0.0)
                ),
            }
        )

    envelope = pd.DataFrame(envelope_rows)
    envelope.to_csv(args.output / "absolute_sensitivity_envelopes.csv", index=False)
    one_factor_frame = pd.DataFrame(one_factor)
    one_factor_frame.to_csv(
        args.output / "one_factor_sensitivity.csv", index=False
    )
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(args.output / "model_ranking_robustness.csv", index=False)

    npz_payload = {"setting_ids": np.asarray([item.setting_id for item in settings])}
    for (dataset, model), matrix in sample_matrices.items():
        prefix = dataset.lower().replace("+", "plus").replace("-", "") + "_" + model
        npz_payload[prefix + "_values"] = matrix
        npz_payload[prefix + "_sample_keys"] = np.asarray(sample_keys[(dataset, model)])
    for dataset, values in bootstrap_payload.items():
        prefix = dataset.lower().replace("+", "plus").replace("-", "")
        npz_payload[prefix + "_bootstrap_minimum_gaps"] = values
    np.savez_compressed(args.output / "sample_contract_values.npz", **npz_payload)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    positions = np.arange(len(DATASETS))
    width = 0.34
    for model_index, model in enumerate(MODELS):
        subset = envelope[envelope["model"] == model].set_index("dataset").loc[list(DATASETS)]
        offset = (model_index - 0.5) * width
        centers = subset["default"].to_numpy(dtype=float)
        lower = centers - subset["minimum"].to_numpy(dtype=float)
        upper = subset["maximum"].to_numpy(dtype=float) - centers
        ax.errorbar(
            positions + offset,
            centers,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=5,
            label=model,
        )
    ax.set_xticks(positions, list(DATASETS))
    ax.set_ylabel("Full-manifest operational stability")
    ax.set_title("Default estimate and registered output-contract envelope")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "contract_envelopes.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(ranking))
    point = ranking["minimum_same_contract_gap"].to_numpy(dtype=float)
    lower = point - ranking["bootstrap_lower_95"].to_numpy(dtype=float)
    upper = ranking["bootstrap_upper_95"].to_numpy(dtype=float) - point
    ax.errorbar(x, point, yerr=np.vstack([lower, upper]), fmt="o", capsize=6)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, ranking["dataset"])
    ax.set_ylabel("Worst GroundingDINO - YOLO-World gap")
    ax.set_title("Finite-grid ranking invariance with paired sampling uncertainty")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "worst_contract_model_gap.png", dpi=220)
    plt.close(fig)

    write_report(
        args.output,
        envelope,
        ranking,
        one_factor_frame,
        saturation,
        duplicate,
        preregistration,
    )

    output_hashes = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            output_hashes[path.name] = sha256(path)
    audit = {
        "protocol_version": preregistration["protocol_version"],
        "preregistration_path": str(args.preregistration.relative_to(ROOT)),
        "preregistration_sha256": sha256(args.preregistration),
        "analysis_script_sha256": sha256(Path(__file__)),
        "registered_setting_count": len(settings),
        "default_setting_index": default_index,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "input_trace_sha256": input_hashes,
        "output_sha256": output_hashes,
        "identifiability_statement": (
            "V1 traces identify tracked count, exposed count up to 20, match IoU, "
            "and birth novelty IoU. Full operational duplicate-IoU sensitivity "
            "and exposure above 20 require prospective raw perturbed pools."
        ),
    }
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in audit.items() if "sha256" not in key}, indent=2))


if __name__ == "__main__":
    main()
