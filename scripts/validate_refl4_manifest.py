from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_image_ids(path: Path) -> set[int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["image_id"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data_operational" / "refl4_transfer1000" / "manifest.json",
    )
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(rows) != 1000:
        raise AssertionError(f"expected 1000 rows, found {len(rows)}")
    counts = Counter(row["source_split"] for row in rows)
    if counts != {"coco": 600, "objects365": 400}:
        raise AssertionError(f"unexpected source counts: {dict(counts)}")
    for key in ("image_id", "source_image_id", "ref_id"):
        if len({row[key] for row in rows}) != len(rows):
            raise AssertionError(f"{key} is not unique")

    excluded: set[int] = set()
    for path in args.exclude_manifest:
        resolved = path if path.is_absolute() else ROOT / path
        excluded.update(load_image_ids(resolved))
    selected_coco = {
        int(row["image_id"]) for row in rows if row["source_split"] == "coco"
    }
    overlap = selected_coco & excluded
    if overlap:
        raise AssertionError(f"selected COCO images overlap prior datasets: {len(overlap)}")

    for row in rows:
        image_path = ROOT / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        if (width, height) != (row["image_width"], row["image_height"]):
            raise AssertionError(f"dimension mismatch: {image_path}")
        x1, y1, x2, y2 = map(float, row["target_box"])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise AssertionError(f"invalid target box: {image_path} {row['target_box']}")
        if not str(row["query"]).strip():
            raise AssertionError(f"empty query: {row['ref_id']}")

    print(
        json.dumps(
            {
                "status": "passed",
                "sample_count": len(rows),
                "source_counts": dict(counts),
                "prior_coco_overlap": len(overlap),
                "all_images_present": True,
                "all_boxes_valid": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
