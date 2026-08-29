from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_trace(path: Path) -> dict[tuple[int, int], dict]:
    records = {}
    duplicates = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (int(record["image_id"]), int(record["ref_id"]))
            duplicates += int(key in records)
            records[key] = record
    records["__duplicate_count__"] = duplicates
    return records


def assert_close(left: float, right: float, message: str) -> None:
    if not np.isclose(left, right, atol=1e-12, rtol=0.0):
        raise AssertionError(f"{message}: {left} != {right}")


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
    manifest_path = ROOT / config["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        (int(item["image_id"]), int(item["ref_id"])) for item in manifest
    }
    if len(manifest) != int(config["sample_count"]):
        raise AssertionError("manifest count does not match frozen sample count")
    if len({int(item["image_id"]) for item in manifest}) != len(manifest):
        raise AssertionError("formal manifest does not contain unique images")

    budgets = [int(value) for value in config["reported_diagnostic_budgets"]]
    families = list(
        json.loads(
            (ROOT / "config" / "random_probe_distribution.json").read_text(
                encoding="utf-8"
            )
        )["families"]
    )
    diagnostic_per_family = int(config["diagnostic_probes_per_family"])
    reference_per_family = int(config["reference_probes_per_family"])
    diagnostic_count = diagnostic_per_family * len(families)
    reference_count = reference_per_family * len(families)
    tracked_maximum = int(config["output_contract"]["tracked_candidate_count"])
    exposed_maximum = int(config["output_contract"]["exposed_candidate_count"])

    audits = {}
    traces_by_model = {}
    for model in config["models"]:
        model_dir = args.result_root / model
        summary = pd.read_csv(model_dir / "sample_budget_summary.csv")
        records_with_meta = load_trace(model_dir / "sample_traces.jsonl.gz")
        duplicate_trace_count = records_with_meta.pop("__duplicate_count__")
        records = records_with_meta
        traces_by_model[model] = records

        if set(records) != expected_keys:
            raise AssertionError(f"{model}: trace keys do not equal manifest keys")
        summary_keys = {
            (int(row.image_id), int(row.ref_id))
            for row in summary.itertuples(index=False)
        }
        if summary_keys != expected_keys:
            raise AssertionError(f"{model}: summary keys do not equal manifest keys")
        observed_pairs = Counter(
            (int(row.image_id), int(row.ref_id), int(row.diagnostic_budget))
            for row in summary.itertuples(index=False)
        )
        if any(count != 1 for count in observed_pairs.values()):
            raise AssertionError(f"{model}: duplicate sample-budget summary rows")
        if set(summary["diagnostic_budget"].unique()) != set(budgets):
            raise AssertionError(f"{model}: incomplete diagnostic budget registry")
        if len(summary) != len(manifest) * len(budgets):
            raise AssertionError(f"{model}: unexpected summary row count")

        eligible_count = 0
        diagnostic_probe_total = 0
        reference_probe_total = 0
        failure_counts = Counter()
        maximum_decomposition_residual = 0.0
        for key, record in records.items():
            eligible = int(record["clean_eligible"])
            tracked = record.get("tracked_clean_candidates", [])
            if len(tracked) > tracked_maximum:
                raise AssertionError(f"{model} {key}: too many tracked candidates")
            probes = record.get("probes", [])
            if not eligible:
                if probes:
                    raise AssertionError(f"{model} {key}: ineligible trace has probes")
                continue
            eligible_count += 1
            if len(tracked) < 2:
                raise AssertionError(f"{model} {key}: eligible trace has fewer than two candidates")
            diagnostic = [probe for probe in probes if probe["split"] == "diagnostic"]
            reference = [probe for probe in probes if probe["split"] == "reference"]
            if len(diagnostic) != diagnostic_count or len(reference) != reference_count:
                raise AssertionError(f"{model} {key}: incomplete probe registry")
            diagnostic_probe_total += len(diagnostic)
            reference_probe_total += len(reference)
            if Counter(probe["spec"]["family"] for probe in diagnostic) != Counter(
                {family: diagnostic_per_family for family in families}
            ):
                raise AssertionError(f"{model} {key}: diagnostic families are imbalanced")
            if Counter(probe["spec"]["family"] for probe in reference) != Counter(
                {family: reference_per_family for family in families}
            ):
                raise AssertionError(f"{model} {key}: reference families are imbalanced")
            for probe in probes:
                outcome = probe["outcome"]
                if len(probe["candidates"]) > exposed_maximum:
                    raise AssertionError(f"{model} {key}: too many exposed candidates")
                coverage = int(outcome["coverage"])
                rank_stable = outcome["rank_stable"]
                operational = int(outcome["operational_stable"])
                expected_operational = coverage * int(rank_stable or 0)
                if operational != expected_operational:
                    raise AssertionError(f"{model} {key}: operational event mismatch")
                if not coverage and rank_stable is not None:
                    raise AssertionError(f"{model} {key}: uncovered ranking is not null")
                if len(outcome["mapping"]) != len(tracked):
                    raise AssertionError(f"{model} {key}: mapping length mismatch")
                for index in outcome["missing_clean_indices"]:
                    if outcome["matched_scores"][index] is not None:
                        raise AssertionError(f"{model} {key}: missing score was imputed")
                failure_counts[outcome["primary_failure"]] += 1

            for profile_name in ("diagnostic_profile", "reference_profile"):
                profile = record[profile_name]
                theta_cov = float(profile["coverage"]["estimate"])
                conditional = profile["conditional_ranking"]
                theta_rank = 0.0 if conditional is None else float(conditional["estimate"])
                theta_op = float(profile["operational"]["estimate"])
                residual = abs((1.0 - theta_op) - (1.0 - theta_cov) - theta_cov * (1.0 - theta_rank))
                maximum_decomposition_residual = max(maximum_decomposition_residual, residual)

            max_row = summary[
                (summary["image_id"] == key[0])
                & (summary["ref_id"] == key[1])
                & (summary["diagnostic_budget"] == max(budgets))
            ].iloc[0]
            assert_close(
                float(max_row["diagnostic_operational"]),
                float(record["diagnostic_profile"]["operational"]["estimate"]),
                f"{model} {key}: diagnostic summary/trace mismatch",
            )
            assert_close(
                float(max_row["reference_operational"]),
                float(record["reference_profile"]["operational"]["estimate"]),
                f"{model} {key}: reference summary/trace mismatch",
            )

        audits[model] = {
            "samples": len(records),
            "eligible_samples": eligible_count,
            "clean_eligibility": eligible_count / len(records),
            "diagnostic_probes": diagnostic_probe_total,
            "reference_probes": reference_probe_total,
            "failure_counts_including_diagnostic_and_reference": dict(failure_counts),
            "maximum_risk_decomposition_residual": maximum_decomposition_residual,
            "duplicate_trace_records_from_resume": duplicate_trace_count,
        }

    shared_probe_pairs = 0
    left_model, right_model = list(config["models"])
    for key in expected_keys:
        left = traces_by_model[left_model][key]
        right = traces_by_model[right_model][key]
        if not left["clean_eligible"] or not right["clean_eligible"]:
            continue
        left_specs = [probe["spec"] for probe in left["probes"]]
        right_specs = [probe["spec"] for probe in right["probes"]]
        if left_specs != right_specs:
            raise AssertionError(f"shared probe registry mismatch at {key}")
        shared_probe_pairs += 1

    result = {
        "status": "passed",
        "manifest_samples": len(manifest),
        "unique_images": len({int(item["image_id"]) for item in manifest}),
        "budgets": budgets,
        "probe_families": families,
        "models": audits,
        "both_models_eligible_shared_probe_pairs": shared_probe_pairs,
    }
    output = args.result_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
