from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "LOCAL_ASSET_MANIFEST.json"

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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def dataset_record(manifest_path: str) -> dict:
    manifest = ROOT / manifest_path
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    image_paths = sorted({ROOT / row["image_path"] for row in rows})
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images for {manifest_path}")
    tree = hashlib.sha256()
    total_bytes = 0
    for path in image_paths:
        relative = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        digest = file_sha256(path)
        total_bytes += size
        tree.update(f"{relative}\0{size}\0{digest}\n".encode())
    return {
        "manifest": manifest_path,
        "manifest_sha256": file_sha256(manifest),
        "image_count": len(image_paths),
        "image_bytes": total_bytes,
        "image_tree_sha256": tree.hexdigest(),
    }


def main() -> None:
    traces = [
        ROOT / "results" / run / model / "sample_traces.jsonl.gz"
        for run in RUNS
        for model in MODELS
    ]
    traces.extend(
        ROOT
        / "results"
        / "cap50_confirmatory"
        / run
        / model
        / "cap50_traces.jsonl.gz"
        for run in RUNS
        for model in MODELS
    )
    missing = [path for path in traces if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} local trace files")
    payload = {
        "schema_version": 1,
        "purpose": "Integrity record for Git-ignored local reproduction assets",
        "datasets": [dataset_record(path) for path in MANIFESTS],
        "traces": [file_record(path) for path in traces],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(payload['traces'])} trace hashes")


if __name__ == "__main__":
    main()

