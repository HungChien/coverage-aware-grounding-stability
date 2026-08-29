from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("groundingdino", "yoloworld")


def load_reference_rows(result_root: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(result_root / model / "sample_budget_summary.csv")
    maximum_budget = int(frame["diagnostic_budget"].max())
    return frame.loc[frame["diagnostic_budget"] == maximum_budget].copy()


def attach_split(frame: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = pd.DataFrame(json.loads(manifest_path.read_text(encoding="utf-8")))
    keys = manifest[["image_id", "ref_id", "source_split"]].copy()
    # The completed analysis table contains one row per model and sample,
    # whereas the manifest contains one split label per sample.  Therefore
    # several model rows legitimately map to the same manifest key.
    joined = frame.merge(
        keys,
        on=["image_id", "ref_id"],
        how="left",
        validate="many_to_one",
    )
    if joined["source_split"].isna().any():
        raise ValueError("target summaries contain samples missing from the transfer manifest")
    return joined


def pooled_estimands(frame: pd.DataFrame) -> dict[str, float | int]:
    eligible = frame["clean_eligible"].astype(bool)
    eligible_frame = frame.loc[eligible]
    coverage = (
        float(eligible_frame["reference_coverage"].mean())
        if len(eligible_frame)
        else math.nan
    )
    operational = (
        float(eligible_frame["reference_operational"].mean())
        if len(eligible_frame)
        else math.nan
    )
    conditional = (
        operational / coverage if coverage and coverage > 0 else math.nan
    )
    return {
        "sample_count": int(len(frame)),
        "eligible_count": int(eligible.sum()),
        "clean_eligibility": float(eligible.mean()),
        "full_manifest_operational": float(frame["reference_operational"].mean()),
        "eligible_operational": operational,
        "coverage": coverage,
        "conditional_ranking": conditional,
        "conditional_minus_operational": conditional - operational,
    }


def bootstrap_mean_interval(
    values: np.ndarray, rng: np.random.Generator, repetitions: int
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    draws = rng.choice(values, size=(repetitions, values.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_delta_interval(
    target: np.ndarray,
    source: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> tuple[float, float]:
    target_draws = rng.choice(
        target, size=(repetitions, target.size), replace=True
    ).mean(axis=1)
    source_draws = rng.choice(
        source, size=(repetitions, source.size), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(target_draws - source_draws, [0.025, 0.975])
    return float(lower), float(upper)


def finite_probe_by_scope(frame: pd.DataFrame, scope: str) -> list[dict]:
    rows: list[dict] = []
    for budget, subset in frame.groupby("diagnostic_budget"):
        diagnostic = subset["diagnostic_operational"].to_numpy(float)
        reference = subset["reference_operational"].to_numpy(float)
        eligible = subset["clean_eligible"].to_numpy(int).astype(bool)
        eligible_diagnostic = diagnostic[eligible]
        eligible_reference = reference[eligible]
        correlation = (
            float(spearmanr(eligible_diagnostic, eligible_reference).statistic)
            if len(eligible_diagnostic) > 1
            else math.nan
        )
        rows.append(
            {
                "scope": scope,
                "diagnostic_budget": int(budget),
                "sample_count": int(len(subset)),
                "eligible_count": int(eligible.sum()),
                "full_manifest_bias": float((diagnostic - reference).mean()),
                "eligible_mae": (
                    float(np.abs(eligible_diagnostic - eligible_reference).mean())
                    if eligible.any()
                    else math.nan
                ),
                "eligible_spearman": correlation,
            }
        )
    return rows


def family_profile_transfer(
    source_root: Path, target_root: Path
) -> pd.DataFrame:
    source = pd.read_csv(source_root / "analysis" / "reference_family_risk.csv")
    target = pd.read_csv(target_root / "analysis" / "reference_family_risk.csv")
    rows = []
    for model in MODELS:
        left = source.loc[source["model"] == model, ["family", "risk_share"]]
        right = target.loc[target["model"] == model, ["family", "risk_share"]]
        joined = left.merge(right, on="family", suffixes=("_source", "_target"))
        source_vector = joined["risk_share_source"].to_numpy(float)
        target_vector = joined["risk_share_target"].to_numpy(float)
        denominator = np.linalg.norm(source_vector) * np.linalg.norm(target_vector)
        rows.append(
            {
                "model": model,
                "family_count": int(len(joined)),
                "spearman_risk_share": float(
                    spearmanr(source_vector, target_vector).statistic
                ),
                "cosine_risk_share": (
                    float(source_vector @ target_vector / denominator)
                    if denominator > 0
                    else math.nan
                ),
                "mean_absolute_risk_share_shift": float(
                    np.abs(source_vector - target_vector).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "results" / "operational_benchmark_v1",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "results" / "operational_transfer_refcocoplus_v1",
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=ROOT / "data_operational" / "refcocoplus_transfer1000" / "manifest.json",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    output = args.target_root / "analysis" / "transfer"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary_rows = []
    finite_rows = []
    delta_rows = []
    for model in MODELS:
        source = load_reference_rows(args.source_root, model)
        target = attach_split(
            load_reference_rows(args.target_root, model), args.target_manifest
        )
        scopes = {
            "source_refcoco": source,
            "target_refcocoplus_pooled": target,
            "target_refcocoplus_testA": target.loc[target["source_split"] == "testA"],
            "target_refcocoplus_testB": target.loc[target["source_split"] == "testB"],
        }
        for scope, subset in scopes.items():
            row = {"model": model, "scope": scope, **pooled_estimands(subset)}
            lower, upper = bootstrap_mean_interval(
                subset["reference_operational"].to_numpy(float),
                rng,
                args.bootstrap_repetitions,
            )
            row["full_manifest_lower_95"] = lower
            row["full_manifest_upper_95"] = upper
            summary_rows.append(row)
        target_values = target["reference_operational"].to_numpy(float)
        source_values = source["reference_operational"].to_numpy(float)
        lower, upper = bootstrap_delta_interval(
            target_values,
            source_values,
            rng,
            args.bootstrap_repetitions,
        )
        delta_rows.append(
            {
                "model": model,
                "comparison": "RefCOCO+ minus RefCOCO",
                "full_manifest_delta": float(target_values.mean() - source_values.mean()),
                "lower_95": lower,
                "upper_95": upper,
            }
        )

        complete = pd.read_csv(
            args.target_root / model / "sample_budget_summary.csv"
        )
        complete = attach_split(complete, args.target_manifest)
        finite_rows.extend(finite_probe_by_scope(complete, "pooled"))
        finite_rows.extend(
            finite_probe_by_scope(
                complete.loc[complete["source_split"] == "testA"], "testA"
            )
        )
        finite_rows.extend(
            finite_probe_by_scope(
                complete.loc[complete["source_split"] == "testB"], "testB"
            )
        )
        for row in finite_rows:
            if "model" not in row:
                row["model"] = model

    summary = pd.DataFrame(summary_rows)
    deltas = pd.DataFrame(delta_rows)
    finite = pd.DataFrame(finite_rows)
    families = family_profile_transfer(args.source_root, args.target_root)
    summary.to_csv(output / "transfer_estimands.csv", index=False)
    deltas.to_csv(output / "dataset_shift_bootstrap.csv", index=False)
    finite.to_csv(output / "transfer_finite_probe_metrics.csv", index=False)
    families.to_csv(output / "family_profile_transfer.csv", index=False)

    report = "\n\n".join(
        [
            "# Frozen RefCOCO+ Transfer Analysis",
            "All parameters, candidate contracts, probes, and estimators are inherited without target-data tuning.",
            "## Source and target estimands\n\n" + markdown_table(summary),
            "## Dataset-shift bootstrap\n\n" + markdown_table(deltas),
            "## Finite-probe transfer\n\n" + markdown_table(finite),
            "## Perturbation-family profile transfer\n\n" + markdown_table(families),
            "## Interpretation boundary\n\nThe benchmark estimates operational candidate-order stability, not semantic correctness. Family-profile agreement is descriptive evidence of diagnostic transfer and not a causal claim.",
        ]
    )
    (output / "transfer_report.md").write_text(report, encoding="utf-8")
    audit = {
        "status": "complete",
        "source_root": str(args.source_root),
        "target_root": str(args.target_root),
        "target_manifest": str(args.target_manifest),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "models": list(MODELS),
        "target_split_counts": {"testA": 500, "testB": 500},
    }
    (output / "transfer_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
