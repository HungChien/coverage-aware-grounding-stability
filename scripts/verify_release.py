from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORE_PATHS = (
    "README.md",
    "REPRODUCIBILITY.md",
    "LOCAL_ASSET_MANIFEST.json",
    "pyproject.toml",
    "src/coverage_aware_grounding_stability/api.py",
    "src/coverage_aware_grounding_stability/operational_stability.py",
    "config/operational_benchmark_owlv2_control_v1.json",
    "config/operational_transfer_refcocoplus_owlv2_control_v1.json",
    "config/operational_transfer_refl4_owlv2_control_v1.json",
    "data_operational/refcoco_unseen500/manifest.json",
    "data_operational/refcocoplus_transfer1000/manifest.json",
    "data_operational/refl4_transfer1000/manifest.json",
    "results/operational_benchmark_v1/analysis/analysis_audit.json",
    "results/operational_transfer_refcocoplus_v1/analysis/analysis_audit.json",
    "results/operational_transfer_refl4_v1/analysis/analysis_audit.json",
    "results/complete_case_optimism_analysis/analysis_audit.json",
    "results/two_stage_sampling_analysis/analysis_audit.json",
    "results/reference_80_adequacy/analysis_audit.json",
    "results/output_contract_robustness/artifact_manifest.json",
    "results/reviewer_risk_controls/analysis_audit.json",
    "results/cap50_confirmatory/summary/analysis_audit.json",
)

MANIFESTS = (
    "data_operational/refcoco_unseen500/manifest.json",
    "data_operational/refcocoplus_transfer1000/manifest.json",
    "data_operational/refl4_transfer1000/manifest.json",
)

RUNS = (
    "operational_benchmark_v1",
    "operational_transfer_refcocoplus_v1",
    "operational_transfer_refl4_v1",
)

MODELS = ("groundingdino", "owlv2", "yoloworld")

PRIVATE_ROOTS = ("reports", "paper", "audit", "tmp", "private")
PRIVATE_SUFFIXES = (".docx", ".pdf", ".pptx")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(require_local_assets: bool) -> dict:
    missing_core = [path for path in CORE_PATHS if not (ROOT / path).is_file()]
    invalid_json = []
    for path in CORE_PATHS:
        candidate = ROOT / path
        if candidate.suffix == ".json" and candidate.is_file():
            try:
                json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                invalid_json.append(path)

    private_paths = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        relative = Path(_relative(path))
        if (
            relative.parts[0].lower() in PRIVATE_ROOTS
            or path.suffix.lower() in PRIVATE_SUFFIXES
        ):
            private_paths.append(_relative(path))

    missing_images = []
    trace_paths = []
    local_hash_mismatches = []
    dataset_hash_mismatches = []
    if require_local_assets:
        for manifest_path in MANIFESTS:
            rows = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
            for row in rows:
                image_path = ROOT / row["image_path"]
                if not image_path.is_file():
                    missing_images.append(_relative(image_path))
        trace_paths.extend(
            ROOT / "results" / run / model / "sample_traces.jsonl.gz"
            for run in RUNS
            for model in MODELS
        )
        trace_paths.extend(
            ROOT
            / "results"
            / "cap50_confirmatory"
            / run
            / model
            / "cap50_traces.jsonl.gz"
            for run in RUNS
            for model in MODELS
        )
    missing_traces = [_relative(path) for path in trace_paths if not path.is_file()]
    if require_local_assets and not missing_traces and not missing_images:
        local_assets = json.loads(
            (ROOT / "LOCAL_ASSET_MANIFEST.json").read_text(encoding="utf-8")
        )
        for record in local_assets["traces"]:
            path = ROOT / record["path"]
            if (
                path.stat().st_size != record["bytes"]
                or _sha256(path) != record["sha256"]
            ):
                local_hash_mismatches.append(record["path"])
        for record in local_assets["datasets"]:
            manifest = ROOT / record["manifest"]
            rows = json.loads(manifest.read_text(encoding="utf-8"))
            image_paths = sorted({ROOT / row["image_path"] for row in rows})
            tree = hashlib.sha256()
            total_bytes = 0
            for path in image_paths:
                size = path.stat().st_size
                total_bytes += size
                tree.update(
                    f"{_relative(path)}\0{size}\0{_sha256(path)}\n".encode()
                )
            observed = (
                _sha256(manifest),
                len(image_paths),
                total_bytes,
                tree.hexdigest(),
            )
            expected = (
                record["manifest_sha256"],
                record["image_count"],
                record["image_bytes"],
                record["image_tree_sha256"],
            )
            if observed != expected:
                dataset_hash_mismatches.append(record["manifest"])

    report = {
        "release": "1.1.0",
        "require_local_assets": require_local_assets,
        "missing_core": missing_core,
        "invalid_json": invalid_json,
        "private_paths": private_paths,
        "missing_images": missing_images,
        "missing_traces": missing_traces,
        "local_hash_mismatches": local_hash_mismatches,
        "dataset_hash_mismatches": dataset_hash_mismatches,
        "ok": not any(
            (
                missing_core,
                invalid_json,
                private_paths,
                missing_images,
                missing_traces,
                local_hash_mismatches,
                dataset_hash_mismatches,
            )
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the public release tree")
    parser.add_argument("--require-local-assets", action="store_true")
    args = parser.parse_args()
    report = verify(args.require_local_assets)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
