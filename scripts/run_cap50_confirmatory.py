from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from scripts.run_operational_benchmark import batched_probe_inference, make_adapter
from src.operational_stability import (
    OutputContract,
    evaluate_probe_outcome,
    raw_candidate_payload,
    select_spatially_distinct,
)
from src.random_probes import RandomProbe
from src.reliability import Candidate, iou_xyxy


def payload_to_candidates(payload: list[dict]) -> list[Candidate]:
    return [
        Candidate(
            tuple(float(value) for value in item["box"]),
            float(item["score"]),
            str(item.get("label", "")),
        )
        for item in payload
    ]


def latest_trace_records(path: Path) -> dict[tuple[int, int], dict]:
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[(int(record["image_id"]), int(record["ref_id"]))] = record
    return records


def probe_id(image_id: int, ref_id: int, split: str, spec: dict) -> str:
    payload = json.dumps(
        [image_id, ref_id, split, spec["family"], spec["severity"], spec["seed"]],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def same_tracked_prefix(
    stored: list[Candidate], recomputed: list[Candidate], score_tolerance: float = 1e-3
) -> bool:
    if len(stored) != len(recomputed):
        return False
    for left, right in zip(stored, recomputed):
        if abs(left.score - right.score) > score_tolerance:
            return False
        # CUDA kernels can move coordinates by a few hundredths of a pixel
        # across otherwise identical runs.  Spatial identity, not bitwise
        # equality, is the relevant prefix check.
        if iou_xyxy(left.box, right.box) < 0.999:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["groundingdino", "owlv2", "yoloworld"], required=True
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "cap50_confirmatory",
    )
    parser.add_argument("--raw-pool-size", type=int, default=50)
    parser.add_argument("--old-exposure", type=int, default=20)
    parser.add_argument("--exposures", type=int, nargs="+", default=[20, 30, 40, 50])
    parser.add_argument("--splits", nargs="+", default=["reference"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-probes", type=int)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result_root = args.result_root if args.result_root.is_absolute() else ROOT / args.result_root
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_key = {
        (int(row["image_id"]), int(row["ref_id"])): row for row in manifest
    }
    old_trace_path = result_root / args.model / "sample_traces.jsonl.gz"
    old_records = latest_trace_records(old_trace_path)

    tasks: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for key, record in old_records.items():
        if not int(record["clean_eligible"]):
            continue
        for probe in record.get("probes", []):
            if probe["split"] not in args.splits:
                continue
            if len(probe.get("candidates", [])) < args.old_exposure:
                continue
            pid = probe_id(key[0], key[1], probe["split"], probe["spec"])
            tasks[key].append({"probe_id": pid, "probe": probe})

    ordered = [(key, item) for key in sorted(tasks) for item in tasks[key]]
    if args.limit_probes is not None:
        selected_ids = {item["probe_id"] for _, item in ordered[: args.limit_probes]}
        tasks = {
            key: [item for item in values if item["probe_id"] in selected_ids]
            for key, values in tasks.items()
        }
        tasks = {key: values for key, values in tasks.items() if values}

    dataset_slug = result_root.name
    output_dir = args.output_root / dataset_slug / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "cap50_traces.jsonl.gz"
    summary_path = output_dir / "cap50_probe_summary.csv"
    progress_path = output_dir / "progress.json"
    metadata_path = output_dir / "run_metadata.json"

    if args.resume and summary_path.exists():
        frame = pd.read_csv(summary_path)
        completed = set(frame["probe_id"].astype(str))
    else:
        frame = pd.DataFrame()
        completed = set()
        if trace_path.exists():
            trace_path.unlink()

    remaining_count = sum(
        item["probe_id"] not in completed for values in tasks.values() for item in values
    )
    metadata = {
        "analysis": "cap50_targeted_confirmatory",
        "model": args.model,
        "config": str(config_path),
        "source_result_root": str(result_root),
        "source_trace": str(old_trace_path),
        "manifest": str(manifest_path),
        "selection_rule": (
            f"saved post-contract candidate count >= {args.old_exposure}; "
            f"splits={args.splits}"
        ),
        "raw_pool_size": args.raw_pool_size,
        "replay_exposures": args.exposures,
        "selected_pair_count": len(tasks),
        "selected_probe_count": sum(len(values) for values in tasks.values()),
        "remaining_probe_count_at_start": remaining_count,
        "started_unix": time.time(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if remaining_count == 0:
        print("No remaining cap-hit probes.")
        return

    adapter = make_adapter(args.model, config)
    default_contract = OutputContract(**config["output_contract"])
    contracts = {
        exposure: OutputContract(
            tracked_candidate_count=default_contract.tracked_candidate_count,
            exposed_candidate_count=exposure,
            duplicate_iou_threshold=default_contract.duplicate_iou_threshold,
            match_iou_threshold=default_contract.match_iou_threshold,
            birth_duplicate_iou_threshold=default_contract.birth_duplicate_iou_threshold,
        )
        for exposure in args.exposures
    }
    batch_size = int(config[f"{args.model}_batch_size"])
    rows = frame.to_dict(orient="records") if not frame.empty else []
    completed_this_run = 0
    start_time = time.time()

    for pair_index, key in enumerate(sorted(tasks), start=1):
        pending = [item for item in tasks[key] if item["probe_id"] not in completed]
        if not pending:
            continue
        source = manifest_by_key[key]
        image_path = Path(source["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image = Image.open(image_path).convert("RGB")
        query = str(source["query"])
        stored_clean = payload_to_candidates(old_records[key]["tracked_clean_candidates"])
        raw_clean_50 = list(adapter.predict(image, query, args.raw_pool_size))
        recomputed_clean = select_spatially_distinct(
            raw_clean_50,
            maximum=default_contract.tracked_candidate_count,
            duplicate_iou_threshold=default_contract.duplicate_iou_threshold,
        )
        clean_prefix_same = same_tracked_prefix(stored_clean, recomputed_clean)
        specs = [RandomProbe(**item["probe"]["spec"]) for item in pending]
        raw_outputs = batched_probe_inference(
            adapter, image, query, specs, args.raw_pool_size, batch_size
        )
        for item, spec, raw_output in zip(pending, specs, raw_outputs):
            replay = {}
            for exposure, contract in contracts.items():
                outcome, candidates = evaluate_probe_outcome(
                    stored_clean, raw_output, contract
                )
                replay[str(exposure)] = {
                    "outcome": outcome.to_dict(),
                    "candidates": raw_candidate_payload(candidates),
                }
            old = item["probe"]["outcome"]
            record = {
                "probe_id": item["probe_id"],
                "model": args.model,
                "image_id": key[0],
                "ref_id": key[1],
                "query": query,
                "split": item["probe"]["split"],
                "spec": spec.to_dict(),
                "stored_tracked_clean_candidates": raw_candidate_payload(stored_clean),
                "raw_clean_candidates_50": raw_candidate_payload(raw_clean_50),
                "clean_tracked_prefix_same": clean_prefix_same,
                "old_outcome": old,
                "old_post_contract_candidates": item["probe"].get("candidates", []),
                "raw_perturbed_candidates_50": raw_candidate_payload(raw_output),
                "replay": replay,
            }
            with gzip.open(trace_path, "at", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            row = {
                "probe_id": item["probe_id"],
                "model": args.model,
                "image_id": key[0],
                "ref_id": key[1],
                "split": item["probe"]["split"],
                "family": spec.family,
                "severity": spec.severity,
                "seed": spec.seed,
                "clean_tracked_prefix_same": int(clean_prefix_same),
                "raw_clean_count_50": len(raw_clean_50),
                "raw_perturbed_count_50": len(raw_output),
                "old_operational_stable": int(old["operational_stable"]),
                "old_primary_failure": old["primary_failure"],
            }
            for exposure in args.exposures:
                outcome = replay[str(exposure)]["outcome"]
                row[f"ke{exposure}_operational_stable"] = int(
                    outcome["operational_stable"]
                )
                row[f"ke{exposure}_primary_failure"] = outcome["primary_failure"]
                row[f"ke{exposure}_post_dedup_count"] = int(
                    outcome["perturbed_candidate_count"]
                )
            rows.append(row)
            completed.add(item["probe_id"])
            completed_this_run += 1
        atomic_csv(pd.DataFrame(rows), summary_path)
        elapsed = time.time() - start_time
        progress = {
            "model": args.model,
            "completed_probes_total": len(completed),
            "completed_probes_this_run": completed_this_run,
            "selected_probe_count": sum(len(values) for values in tasks.values()),
            "completed_pairs_this_run": pair_index,
            "selected_pair_count": len(tasks),
            "elapsed_seconds": elapsed,
            "seconds_per_probe": elapsed / max(completed_this_run, 1),
            "updated_unix": time.time(),
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(
            f"[{pair_index}/{len(tasks)}] {args.model} image={key[0]} "
            f"probes={completed_this_run}/{remaining_count} elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    metadata["finished_unix"] = time.time()
    metadata["elapsed_seconds"] = metadata["finished_unix"] - metadata["started_unix"]
    metadata["completed_probe_count"] = len(completed)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
