from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Benchmark inference uses frozen local checkpoints.  Prevent transient network
# availability from changing whether an otherwise reproducible run can start.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from coverage_aware_grounding_stability.adapters import (
    GroundingDINOAdapter,
    OWLv2Adapter,
    YOLOWorldAdapter,
)
from coverage_aware_grounding_stability.operational_stability import (
    OutputContract,
    evaluate_probe_outcome,
    flatten_profile,
    raw_candidate_payload,
    select_spatially_distinct,
    summarize_outcomes,
)
from coverage_aware_grounding_stability.random_probes import (
    apply_random_probe,
    sample_stratified_probes,
)
from coverage_aware_grounding_stability.candidates import Candidate, top1_correct


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def resolved_inference_thresholds(model: str, config: dict) -> dict[str, float]:
    """Resolve architecture-specific inference thresholds.

    Version-1 configurations used a shared ``box_threshold`` field.  The
    fallback is retained so the frozen runs remain exactly reproducible, while
    new configurations can state model-specific thresholds explicitly under
    ``inference_thresholds``.
    """

    per_model = config.get("inference_thresholds", {}).get(model, {})
    box_threshold = float(
        per_model.get("box_threshold", config.get("box_threshold", 0.05))
    )
    resolved = {"box_threshold": box_threshold}
    if model == "groundingdino":
        resolved["text_threshold"] = float(
            per_model.get("text_threshold", config.get("text_threshold", 0.05))
        )
    return resolved


def make_adapter(model: str, config: dict):
    model_name = config["models"][model]
    thresholds = resolved_inference_thresholds(model, config)
    if model == "groundingdino":
        return GroundingDINOAdapter(
            model_name,
            box_threshold=thresholds["box_threshold"],
            text_threshold=thresholds["text_threshold"],
        )
    if model == "owlv2":
        return OWLv2Adapter(
            model_name,
            box_threshold=thresholds["box_threshold"],
            revision=config.get("model_revisions", {}).get(model),
        )
    if model != "yoloworld":
        raise ValueError(f"unsupported grounding model: {model}")
    local_candidates = [
        ROOT / model_name,
        ROOT.parent / "week2_minimal_experiment" / model_name,
    ]
    local_model = next((path for path in local_candidates if path.exists()), None)
    return YOLOWorldAdapter(
        str(local_model or model_name),
        box_threshold=thresholds["box_threshold"],
    )


def sample_rng(base_seed: int, image_id: int, ref_id: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([base_seed, image_id, ref_id])
    )


def atomic_csv(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def completed_samples(summary_path: Path, budgets: list[int]) -> set[tuple[int, int]]:
    if not summary_path.exists():
        return set()
    frame = pd.read_csv(summary_path)
    expected = set(budgets)
    observed: dict[tuple[int, int], set[int]] = {}
    for row in frame.to_dict(orient="records"):
        key = (int(row["image_id"]), int(row["ref_id"]))
        observed.setdefault(key, set()).add(int(row["diagnostic_budget"]))
    return {key for key, values in observed.items() if values == expected}


def traced_samples(trace_path: Path) -> set[tuple[int, int]]:
    if not trace_path.exists():
        return set()
    keys = set()
    with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            keys.add((int(record["image_id"]), int(record["ref_id"])))
    return keys


def load_rows(
    summary_path: Path,
    resume: bool,
    completed: set[tuple[int, int]] | None = None,
) -> list[dict]:
    if not resume or not summary_path.exists():
        return []
    rows = pd.read_csv(summary_path).to_dict(orient="records")
    if completed is None:
        return rows
    return [
        row
        for row in rows
        if (int(row["image_id"]), int(row["ref_id"])) in completed
    ]


def batched_probe_inference(
    adapter,
    image: Image.Image,
    query: str,
    specs,
    top_k: int,
    batch_size: int,
) -> list[list[Candidate]]:
    outputs: list[list[Candidate]] = []
    for start in range(0, len(specs), batch_size):
        chunk = specs[start : start + batch_size]
        images = [apply_random_probe(image, spec) for spec in chunk]
        predictions = adapter.predict_batch(images, query, top_k)
        if len(predictions) != len(chunk):
            raise RuntimeError("adapter returned an unexpected batch length")
        outputs.extend([list(prediction) for prediction in predictions])
    return outputs


def ineligible_rows(
    model: str,
    source: dict,
    budgets: list[int],
    raw_clean: list[Candidate],
    clean_correct: int,
    clean_best_iou: float,
) -> list[dict]:
    return [
        {
            "model": model,
            "image_id": int(source["image_id"]),
            "ref_id": int(source["ref_id"]),
            "category_id": int(source["category_id"]),
            "query": source["query"],
            "clean_eligible": 0,
            "raw_clean_candidate_count": len(raw_clean),
            "tracked_clean_candidate_count": 0,
            "clean_correct": clean_correct,
            "clean_best_iou": clean_best_iou,
            "diagnostic_budget": budget,
            "diagnostic_probe_count": 0,
            "diagnostic_coverage": 0.0,
            "diagnostic_conditional_ranking": float("nan"),
            "diagnostic_operational": 0.0,
            "reference_probe_count": 0,
            "reference_coverage": 0.0,
            "reference_conditional_ranking": float("nan"),
            "reference_operational": 0.0,
            "full_manifest_operational": 0.0,
        }
        for budget in budgets
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "operational_benchmark_v1.json",
    )
    parser.add_argument(
        "--model",
        choices=["groundingdino", "yoloworld", "owlv2"],
        required=True,
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "operational_benchmark_v1",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        manifest = manifest[: args.limit]
    stop = len(manifest) if args.stop is None else min(args.stop, len(manifest))
    selected = manifest[args.start:stop]
    if not selected:
        raise ValueError("selected manifest range is empty")

    probe_config = json.loads(
        (ROOT / "config" / "random_probe_distribution.json").read_text(
            encoding="utf-8"
        )
    )
    families = probe_config["families"]
    contract = OutputContract(**config["output_contract"])
    raw_candidate_pool_size = int(
        config.get("raw_candidate_pool_size", contract.exposed_candidate_count)
    )
    if raw_candidate_pool_size < contract.exposed_candidate_count:
        raise ValueError(
            "raw_candidate_pool_size must be no smaller than "
            "output_contract.exposed_candidate_count"
        )
    diagnostic_per_family = int(config["diagnostic_probes_per_family"])
    reference_per_family = int(config["reference_probes_per_family"])
    diagnostic_count = diagnostic_per_family * len(families)
    reference_count = reference_per_family * len(families)
    budgets = [int(value) for value in config["reported_diagnostic_budgets"]]
    if budgets[-1] > diagnostic_count or any(
        budget % len(families) != 0 for budget in budgets
    ):
        raise ValueError("diagnostic budgets must be balanced family-block prefixes")
    confidence = float(config["confidence"])
    batch_size = int(config[f"{args.model}_batch_size"])

    output_dir = args.output_root / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sample_budget_summary.csv"
    trace_path = output_dir / "sample_traces.jsonl.gz"
    progress_path = output_dir / "progress.json"
    metadata_path = output_dir / "run_metadata.json"
    summary_done = completed_samples(summary_path, budgets) if args.resume else set()
    trace_done = traced_samples(trace_path) if args.resume else set()
    done = summary_done & trace_done
    rows = load_rows(summary_path, args.resume, done)
    if not args.resume:
        if trace_path.exists():
            trace_path.unlink()
        rows = []

    metadata = {
        **config,
        "model": args.model,
        "resolved_inference_thresholds": resolved_inference_thresholds(
            args.model, config
        ),
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "selected_range": [args.start, stop],
        "selected_count": len(selected),
        "diagnostic_probe_count": diagnostic_count,
        "reference_probe_count": reference_count,
        "raw_candidate_pool_size": raw_candidate_pool_size,
        "trace_candidate_schema": {
            "raw_clean_candidates": "pre-contract candidate pool",
            "probes.raw_candidates": "pre-contract perturbed candidate pool",
            "probes.candidates": "post-contract perturbed candidates",
        },
        "git_commit": git_commit(),
        "started_unix": time.time(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    adapter = make_adapter(args.model, config)
    start_time = time.time()
    for offset, source in enumerate(selected, start=1):
        key = (int(source["image_id"]), int(source["ref_id"]))
        if key in done:
            print(
                f"[{offset}/{len(selected)}] resume-skip image={key[0]} ref={key[1]}",
                flush=True,
            )
            continue
        image_path = Path(source["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image = Image.open(image_path).convert("RGB")
        query = str(source["query"])
        raw_clean = list(adapter.predict(image, query, raw_candidate_pool_size))
        clean_correct, clean_best_iou = top1_correct(
            raw_clean,
            [source["target_box"]],
            float(config["correct_iou_threshold"]),
        )
        clean = select_spatially_distinct(
            raw_clean,
            maximum=contract.tracked_candidate_count,
            duplicate_iou_threshold=contract.duplicate_iou_threshold,
        )

        if len(clean) < 2:
            sample_rows = ineligible_rows(
                args.model,
                source,
                budgets,
                raw_clean,
                clean_correct,
                clean_best_iou,
            )
            trace = {
                "model": args.model,
                "image_id": key[0],
                "ref_id": key[1],
                "query": query,
                "target_box": source["target_box"],
                "clean_eligible": 0,
                "raw_clean_candidates": raw_candidate_payload(raw_clean),
                "tracked_clean_candidates": [],
                "reason": "fewer_than_two_spatially_distinct_clean_candidates",
                "probes": [],
            }
        else:
            rng = sample_rng(int(config["base_seed"]), key[0], key[1])
            diagnostic_specs = sample_stratified_probes(
                rng, families, diagnostic_per_family
            )
            reference_specs = sample_stratified_probes(
                rng, families, reference_per_family
            )
            all_specs = diagnostic_specs + reference_specs
            raw_outputs = batched_probe_inference(
                adapter,
                image,
                query,
                all_specs,
                raw_candidate_pool_size,
                batch_size,
            )

            outcomes = []
            deduplicated_outputs = []
            for raw_output in raw_outputs:
                outcome, deduplicated = evaluate_probe_outcome(
                    clean, raw_output, contract
                )
                outcomes.append(outcome)
                deduplicated_outputs.append(deduplicated)

            diagnostic_outcomes = outcomes[:diagnostic_count]
            reference_outcomes = outcomes[diagnostic_count:]
            reference_profile = summarize_outcomes(
                reference_outcomes,
                [spec.family for spec in reference_specs],
                confidence=confidence,
            )
            sample_rows = []
            for budget in budgets:
                diagnostic_profile = summarize_outcomes(
                    diagnostic_outcomes[:budget],
                    [spec.family for spec in diagnostic_specs[:budget]],
                    confidence=confidence,
                )
                row = {
                    "model": args.model,
                    "image_id": key[0],
                    "ref_id": key[1],
                    "category_id": int(source["category_id"]),
                    "query": query,
                    "clean_eligible": 1,
                    "raw_clean_candidate_count": len(raw_clean),
                    "tracked_clean_candidate_count": len(clean),
                    "clean_correct": clean_correct,
                    "clean_best_iou": clean_best_iou,
                    "diagnostic_budget": budget,
                    **flatten_profile("diagnostic", diagnostic_profile),
                    **flatten_profile("reference", reference_profile),
                    "full_manifest_operational": diagnostic_profile.operational.estimate,
                }
                sample_rows.append(row)

            probe_records = []
            for index, (spec, outcome, raw_candidates, candidates) in enumerate(
                zip(all_specs, outcomes, raw_outputs, deduplicated_outputs)
            ):
                probe_records.append(
                    {
                        "split": (
                            "diagnostic" if index < diagnostic_count else "reference"
                        ),
                        "spec": spec.to_dict(),
                        "outcome": outcome.to_dict(),
                        "raw_candidates": raw_candidate_payload(raw_candidates),
                        "candidates": raw_candidate_payload(candidates),
                    }
                )
            trace = {
                "model": args.model,
                "image_id": key[0],
                "ref_id": key[1],
                "query": query,
                "target_box": source["target_box"],
                "category_id": int(source["category_id"]),
                "clean_eligible": 1,
                "clean_correct": clean_correct,
                "clean_best_iou": clean_best_iou,
                "raw_clean_candidates": raw_candidate_payload(raw_clean),
                "tracked_clean_candidates": raw_candidate_payload(clean),
                "diagnostic_profile": summarize_outcomes(
                    diagnostic_outcomes,
                    [spec.family for spec in diagnostic_specs],
                    confidence=confidence,
                ).to_dict(),
                "reference_profile": reference_profile.to_dict(),
                "probes": probe_records,
            }

        # Append the complete trace first. If execution stops before the atomic
        # summary update, resume recomputes the sample and the trace loader keeps
        # the last record. The reverse order could silently skip a sample whose
        # summary exists but whose trace was never written.
        with gzip.open(trace_path, "at", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")
        rows.extend(sample_rows)
        atomic_csv(rows, summary_path)

        completed = len(
            {
                (int(row["image_id"]), int(row["ref_id"])) for row in rows
            }
        )
        elapsed = time.time() - start_time
        progress = {
            "model": args.model,
            "completed": completed,
            "selected_count": len(selected),
            "last_image_id": key[0],
            "last_ref_id": key[1],
            "elapsed_seconds": elapsed,
            "seconds_per_completed_sample": elapsed / max(offset, 1),
            "updated_unix": time.time(),
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        final_row = sample_rows[-1]
        print(
            f"[{offset}/{len(selected)}] image={key[0]} eligible="
            f"{final_row['clean_eligible']} op="
            f"{final_row['diagnostic_operational']:.3f} ref="
            f"{final_row['reference_operational']:.3f} elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    metadata["finished_unix"] = time.time()
    metadata["elapsed_seconds"] = metadata["finished_unix"] - metadata["started_unix"]
    metadata["completed_samples"] = len(
        {(int(row["image_id"]), int(row["ref_id"])) for row in rows}
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
