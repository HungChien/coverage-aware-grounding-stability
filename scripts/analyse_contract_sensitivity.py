from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverage_aware_grounding_stability.operational_stability import (
    OutputContract,
    evaluate_probe_outcome,
    select_spatially_distinct,
)
from coverage_aware_grounding_stability.candidates import Candidate


def candidates(payload: list[dict]) -> list[Candidate]:
    return [
        Candidate(
            box=tuple(float(value) for value in item["box"]),
            score=float(item["score"]),
            label=str(item.get("label", "")),
        )
        for item in payload
    ]


def load_trace(path: Path) -> dict[tuple[int, int], dict]:
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[(int(record["image_id"]), int(record["ref_id"]))] = record
    return records


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
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    primary = config["output_contract"]
    settings = [
        (tracked, match, birth)
        for tracked in (2, 3, 5)
        for match in (0.10, 0.15, 0.25)
        for birth in (0.50, 0.70, 0.85)
    ]
    rows = []
    for model in config["models"]:
        trace_path = args.result_root / model / "sample_traces.jsonl.gz"
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        records = load_trace(trace_path)
        if len(records) != int(config["sample_count"]):
            raise ValueError(f"{model} trace is incomplete: {len(records)} samples")

        for tracked, match_threshold, birth_threshold in settings:
            sample_values = []
            eligible_values = []
            failure_counts: defaultdict[str, int] = defaultdict(int)
            probe_count = 0
            for record in records.values():
                clean = select_spatially_distinct(
                    candidates(record["raw_clean_candidates"]),
                    maximum=tracked,
                    duplicate_iou_threshold=float(primary["duplicate_iou_threshold"]),
                )
                eligible = int(len(clean) >= 2)
                eligible_values.append(eligible)
                if not eligible:
                    sample_values.append(0.0)
                    continue
                contract = OutputContract(
                    tracked_candidate_count=tracked,
                    exposed_candidate_count=int(primary["exposed_candidate_count"]),
                    duplicate_iou_threshold=float(primary["duplicate_iou_threshold"]),
                    match_iou_threshold=match_threshold,
                    birth_duplicate_iou_threshold=birth_threshold,
                )
                values = []
                for probe in record["probes"]:
                    if probe["split"] != "reference":
                        continue
                    outcome, _ = evaluate_probe_outcome(
                        clean, candidates(probe["candidates"]), contract
                    )
                    values.append(outcome.operational_stable)
                    failure_counts[outcome.primary_failure] += int(
                        outcome.primary_failure != "stable"
                    )
                probe_count += len(values)
                sample_values.append(float(np.mean(values)))
            operational = np.asarray(sample_values)
            eligible_array = np.asarray(eligible_values, dtype=bool)
            total_failures = sum(failure_counts.values())
            rows.append(
                {
                    "model": model,
                    "tracked_candidate_count": tracked,
                    "match_iou_threshold": match_threshold,
                    "birth_duplicate_iou_threshold": birth_threshold,
                    "sample_count": len(records),
                    "eligible_count": int(eligible_array.sum()),
                    "clean_eligibility": float(eligible_array.mean()),
                    "reference_probe_count": probe_count,
                    "full_manifest_operational": float(operational.mean()),
                    "eligible_only_operational": (
                        float(operational[eligible_array].mean())
                        if eligible_array.any()
                        else 0.0
                    ),
                    **{
                        f"{cause}_share": (
                            0.0
                            if total_failures == 0
                            else failure_counts[cause] / total_failures
                        )
                        for cause in (
                            "winner_missing",
                            "competitor_missing",
                            "threatening_birth",
                            "ranking_reversal",
                        )
                    },
                }
            )

    output = args.result_root / "analysis" / "contract_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "contract_sensitivity.csv", index=False)

    primary_subset = frame[
        (frame["tracked_candidate_count"] == int(primary["tracked_candidate_count"]))
        & (frame["birth_duplicate_iou_threshold"] == float(primary["birth_duplicate_iou_threshold"]))
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for model, subset in primary_subset.groupby("model"):
        subset = subset.sort_values("match_iou_threshold")
        ax.plot(
            subset["match_iou_threshold"],
            subset["full_manifest_operational"],
            marker="o",
            label=model,
        )
    ax.set(
        xlabel="Candidate association IoU threshold",
        ylabel="Full-manifest reference stability",
        title="Output-contract sensitivity around the primary association rule",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "match_threshold_sensitivity.png", dpi=220)
    plt.close(fig)

    audit = {
        "models": sorted(frame["model"].unique()),
        "settings_per_model": len(settings),
        "total_rows": len(frame),
        "primary_setting": primary,
        "note": (
            "This is a post-primary sensitivity analysis. Perturbed candidates "
            "were saved after the frozen 0.70 duplicate-suppression rule; the "
            "analysis therefore varies tracked count, association threshold, "
            "and threatening-birth threshold, but not upstream candidate generation."
        ),
    }
    (output / "sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
