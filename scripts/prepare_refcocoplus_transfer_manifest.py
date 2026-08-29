from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID_PATTERN = re.compile(r"COCO_[^_]+_(\d{12})(?:_\d+)?\.jpg$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_excluded_image_ids(paths: list[Path]) -> set[int]:
    excluded: set[int] = set()
    for path in paths:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        rows = json.loads(resolved.read_text(encoding="utf-8"))
        excluded.update(int(row["image_id"]) for row in rows)
    return excluded


def image_id_from_file_name(file_name: str) -> int:
    match = IMAGE_ID_PATTERN.search(file_name)
    if match is None:
        raise ValueError(f"cannot parse COCO image id from {file_name!r}")
    return int(match.group(1))


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid XYWH box: {box}")
    return [x, y, x + width, y + height]


def select_split(
    dataset_name: str,
    split: str,
    count: int,
    seed: int,
    excluded: set[int],
    selected_image_ids: set[int],
    image_dir: Path,
) -> list[dict]:
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=4096)
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for row in dataset:
        image_id = image_id_from_file_name(str(row["file_name"]))
        if image_id in excluded or image_id in selected_image_ids:
            continue
        answers = [str(value).strip() for value in row["answer"] if str(value).strip()]
        if not answers:
            continue
        query = answers[int(rng.integers(0, len(answers)))]
        image = row["image"].convert("RGB")
        target = image_dir / f"{image_id:012d}.jpg"
        image.save(target, quality=95)
        selected.append(
            {
                "dataset": "refcocoplus",
                "source_split": split,
                "image_id": image_id,
                "ref_id": int(row["question_id"]),
                "query": query,
                "target_box": xywh_to_xyxy(row["bbox"]),
                "category_id": -1,
                "image_path": str(target.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        selected_image_ids.add(image_id)
        if len(selected) % 50 == 0:
            print(f"{split}: prepared {len(selected)}/{count}", flush=True)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} eligible unique images were available in {split}; "
            f"requested {count}"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lmms-lab/RefCOCOplus")
    parser.add_argument("--count-per-split", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data_operational" / "refcocoplus_transfer1000",
    )
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.count_per_split <= 0:
        raise ValueError("count-per-split must be positive")

    excluded = load_excluded_image_ids(args.exclude_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected_image_ids: set[int] = set()
    rows: list[dict] = []
    for offset, split in enumerate(("testA", "testB")):
        rows.extend(
            select_split(
                args.dataset,
                split,
                args.count_per_split,
                args.seed + offset,
                excluded,
                selected_image_ids,
                image_dir,
            )
        )

    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    split_counts = {
        split: sum(row["source_split"] == split for row in rows)
        for split in ("testA", "testB")
    }
    metadata = {
        "source": args.dataset,
        "splits": split_counts,
        "count": len(rows),
        "unique_image_count": len({row["image_id"] for row in rows}),
        "excluded_image_count": len(excluded),
        "overlap_with_excluded": len({row["image_id"] for row in rows} & excluded),
        "cross_split_image_overlap": len(
            {
                row["image_id"]
                for row in rows
                if row["source_split"] == "testA"
            }
            & {
                row["image_id"]
                for row in rows
                if row["source_split"] == "testB"
            }
        ),
        "seed": args.seed,
        "selection_unit": "one_expression_per_unique_image",
        "bbox_source_format": "xywh",
        "manifest_bbox_format": "xyxy",
        "manifest_sha256": sha256(manifest),
        "excluded_manifests": [str(path) for path in args.exclude_manifest],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
