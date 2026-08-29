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


def manifest_fields(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.DataFrame(json.loads(manifest_path.read_text(encoding="utf-8")))
    required = ["image_id", "ref_id", "source_split", "query_word_count", "bbox_area"]
    missing = [column for column in required if column not in manifest]
    if missing:
        raise ValueError(f"Ref-L4 manifest is missing fields: {missing}")
    fields = manifest[required].copy()
    fields["query_length_stratum"] = pd.cut(
        fields["query_word_count"],
        bins=[-np.inf, 18, 29, np.inf],
        labels=["short_le18", "medium_19_29", "long_ge30"],
    ).astype(str)
    scale = np.sqrt(fields["bbox_area"].to_numpy(float))
    fields["target_scale_stratum"] = np.select(
        [scale < 32, scale <= 96],
        ["small_lt32", "medium_32_96"],
        default="large_gt96",
    )
    return fields


def attach_manifest(frame: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    joined = frame.merge(
        fields,
        on=["image_id", "ref_id"],
        how="left",
        validate="many_to_one",
    )
    if joined["source_split"].isna().any():
        raise ValueError("target summaries contain samples missing from the Ref-L4 manifest")
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
    conditional = operational / coverage if coverage and coverage > 0 else math.nan
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
    target_draws = rng.choice(target, size=(repetitions, target.size), replace=True).mean(axis=1)
    source_draws = rng.choice(source, size=(repetitions, source.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(target_draws - source_draws, [0.025, 0.975])
    return float(lower), float(upper)


def finite_probe_rows(frame: pd.DataFrame, model: str, scope: str) -> list[dict]:
    rows = []
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
                "model": model,
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


def family_transfer(
    baseline_name: str, baseline_root: Path, target_root: Path
) -> list[dict]:
    source = pd.read_csv(baseline_root / "analysis" / "reference_family_risk.csv")
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
                "baseline": baseline_name,
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
    return rows


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
        "--refcoco-root",
        type=Path,
        default=ROOT / "results" / "operational_benchmark_v1",
    )
    parser.add_argument(
        "--refcocoplus-root",
        type=Path,
        default=ROOT / "results" / "operational_transfer_refcocoplus_v1",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "results" / "operational_transfer_refl4_v1",
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=ROOT / "data_operational" / "refl4_transfer1000" / "manifest.json",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    output = args.target_root / "analysis" / "transfer"
    output.mkdir(parents=True, exist_ok=True)
    fields = manifest_fields(args.target_manifest)
    rng = np.random.default_rng(args.seed)
    estimand_rows: list[dict] = []
    finite_rows: list[dict] = []
    delta_rows: list[dict] = []
    stratum_rows: list[dict] = []

    for model in MODELS:
        refcoco = load_reference_rows(args.refcoco_root, model)
        refcocoplus = load_reference_rows(args.refcocoplus_root, model)
        target = attach_manifest(load_reference_rows(args.target_root, model), fields)
        scopes = {
            "source_refcoco": refcoco,
            "source_refcocoplus": refcocoplus,
            "target_refl4_pooled": target,
            "target_refl4_coco": target.loc[target["source_split"] == "coco"],
            "target_refl4_objects365": target.loc[
                target["source_split"] == "objects365"
            ],
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
            estimand_rows.append(row)

        target_values = target["reference_operational"].to_numpy(float)
        for baseline_name, baseline in (
            ("RefCOCO", refcoco),
            ("RefCOCO+", refcocoplus),
        ):
            baseline_values = baseline["reference_operational"].to_numpy(float)
            lower, upper = bootstrap_delta_interval(
                target_values,
                baseline_values,
                rng,
                args.bootstrap_repetitions,
            )
            delta_rows.append(
                {
                    "model": model,
                    "comparison": f"Ref-L4 minus {baseline_name}",
                    "full_manifest_delta": float(
                        target_values.mean() - baseline_values.mean()
                    ),
                    "lower_95": lower,
                    "upper_95": upper,
                }
            )

        complete = pd.read_csv(args.target_root / model / "sample_budget_summary.csv")
        complete = attach_manifest(complete, fields)
        finite_scopes = {
            "pooled": complete,
            "coco": complete.loc[complete["source_split"] == "coco"],
            "objects365": complete.loc[complete["source_split"] == "objects365"],
        }
        for scope, subset in finite_scopes.items():
            finite_rows.extend(finite_probe_rows(subset, model, scope))

        for dimension in (
            "source_split",
            "query_length_stratum",
            "target_scale_stratum",
        ):
            for level, subset in target.groupby(dimension, observed=True):
                stratum_rows.append(
                    {
                        "model": model,
                        "dimension": dimension,
                        "level": str(level),
                        **pooled_estimands(subset),
                    }
                )

    estimands = pd.DataFrame(estimand_rows)
    deltas = pd.DataFrame(delta_rows)
    finite = pd.DataFrame(finite_rows)
    strata = pd.DataFrame(stratum_rows)
    families = pd.DataFrame(
        family_transfer("RefCOCO", args.refcoco_root, args.target_root)
        + family_transfer("RefCOCO+", args.refcocoplus_root, args.target_root)
    )
    estimands.to_csv(output / "transfer_estimands.csv", index=False)
    deltas.to_csv(output / "dataset_shift_bootstrap.csv", index=False)
    finite.to_csv(output / "transfer_finite_probe_metrics.csv", index=False)
    strata.to_csv(output / "registered_stratum_estimands.csv", index=False)
    families.to_csv(output / "family_profile_transfer.csv", index=False)

    report = "\n\n".join(
        [
            "# Frozen Ref-L4 Transfer Analysis",
            "All candidate, probe, estimator, and analysis definitions were registered before Ref-L4 model inference.",
            "## Source and target estimands\n\n" + markdown_table(estimands),
            "## Dataset-shift bootstrap\n\n" + markdown_table(deltas),
            "## Finite-probe transfer\n\n" + markdown_table(finite),
            "## Registered Ref-L4 strata\n\n" + markdown_table(strata),
            "## Perturbation-family profile transfer\n\n" + markdown_table(families),
            "## Interpretation boundary\n\nThe benchmark estimates operational candidate-order stability under the registered visual probe distribution, not semantic correctness.  The source-stratified pooled result describes the fixed 600/400 manifest.  Family attribution is descriptive rather than causal.",
        ]
    )
    (output / "transfer_report.md").write_text(report, encoding="utf-8")
    audit = {
        "status": "complete",
        "refcoco_root": str(args.refcoco_root),
        "refcocoplus_root": str(args.refcocoplus_root),
        "target_root": str(args.target_root),
        "target_manifest": str(args.target_manifest),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "models": list(MODELS),
        "target_source_counts": {"coco": 600, "objects365": 400},
    }
    (output / "transfer_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
