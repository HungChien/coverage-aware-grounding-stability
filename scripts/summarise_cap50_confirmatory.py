from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "operational_benchmark_v1": ("RefCOCO", 500),
    "operational_transfer_refcocoplus_v1": ("RefCOCO+", 1000),
    "operational_transfer_refl4_v1": ("Ref-L4", 1000),
}
MODELS = ("groundingdino", "owlv2", "yoloworld")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "results" / "cap50_confirmatory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "cap50_confirmatory" / "summary",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    transitions = []
    for slug, (dataset, manifest_n) in DATASETS.items():
        for model in MODELS:
            path = args.input_root / slug / model / "cap50_probe_summary.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            selected = len(frame)
            old_stable = frame["old_operational_stable"].astype(int)
            replay20_stable = frame["ke20_operational_stable"].astype(int)
            replay20_label = frame["ke20_primary_failure"].astype(str)
            for exposure in (20, 30, 40, 50):
                new_stable = frame[f"ke{exposure}_operational_stable"].astype(int)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "exposure": exposure,
                        "selected_cap_hit_probes": selected,
                        "stable_selected_fraction": new_stable.mean(),
                        "net_stable_changes_vs_fresh_ke20": int(
                            (new_stable - replay20_stable).sum()
                        ),
                        "fail_to_stable_vs_fresh_ke20": int(
                            ((replay20_stable == 0) & (new_stable == 1)).sum()
                        ),
                        "stable_to_fail_vs_fresh_ke20": int(
                            ((replay20_stable == 1) & (new_stable == 0)).sum()
                        ),
                        "full_manifest_stability_delta_vs_fresh_ke20": float(
                            (new_stable - replay20_stable).sum() / (manifest_n * 80)
                        ),
                        "fresh_ke20_stability_agreement_with_old": float(
                            (replay20_stable == old_stable).mean()
                        ),
                        "fresh_ke20_label_agreement_with_old": float(
                            (replay20_label == frame["old_primary_failure"].astype(str)).mean()
                        ),
                        "clean_prefix_mismatch_count": int(
                            (frame["clean_tracked_prefix_same"].astype(int) == 0).sum()
                        ),
                    }
                )
            for old_label, group in frame.groupby("ke20_primary_failure"):
                for new_label, count in group["ke50_primary_failure"].value_counts().items():
                    transitions.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "fresh_ke20_label": old_label,
                            "ke50_label": new_label,
                            "count": int(count),
                        }
                    )
    summary = pd.DataFrame(rows)
    transition_frame = pd.DataFrame(transitions)
    summary_path = args.output / "cap50_stability_summary.csv"
    transition_path = args.output / "cap50_failure_transitions.csv"
    summary.to_csv(summary_path, index=False)
    transition_frame.to_csv(transition_path, index=False)
    audit = {
        "completed_dataset_model_groups": int(summary[["dataset", "model"]].drop_duplicates().shape[0]) if not summary.empty else 0,
        "expected_dataset_model_groups": 9,
        "all_clean_prefixes_match": bool(
            summary["clean_prefix_mismatch_count"].max() == 0
        ) if not summary.empty else False,
        "minimum_ke20_stability_reproduction_rate": float(
            summary["fresh_ke20_stability_agreement_with_old"].min()
        ) if not summary.empty else None,
        "minimum_ke20_label_reproduction_rate": float(
            summary["fresh_ke20_label_agreement_with_old"].min()
        ) if not summary.empty else None,
        "artifacts": {
            summary_path.name: sha256(summary_path),
            transition_path.name: sha256(transition_path),
        },
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
