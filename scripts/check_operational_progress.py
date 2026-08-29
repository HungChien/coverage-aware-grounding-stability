from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results" / "operational_benchmark_v1",
    )
    args = parser.parse_args()

    rows = []
    for model in ("groundingdino", "yoloworld"):
        progress_path = args.result_root / model / "progress.json"
        metadata_path = args.result_root / model / "run_metadata.json"
        if not progress_path.exists():
            rows.append({"model": model, "status": "not_started"})
            continue
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress["completed"])
        total = int(progress["selected_count"])
        seconds_per_sample = float(progress["seconds_per_completed_sample"])
        remaining = (total - completed) * seconds_per_sample
        status = "running_or_interrupted"
        if completed >= total:
            status = "complete"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "finished_unix" in metadata and completed >= total:
                status = "complete"
        rows.append(
            {
                "model": model,
                "status": status,
                "completed": completed,
                "total": total,
                "percent": round(100.0 * completed / total, 2),
                "elapsed": format_duration(float(progress["elapsed_seconds"])),
                "estimated_remaining": format_duration(remaining),
                "last_update_age_seconds": round(
                    time.time() - float(progress["updated_unix"]), 1
                ),
                "last_image_id": int(progress["last_image_id"]),
                "last_ref_id": int(progress["last_ref_id"]),
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
