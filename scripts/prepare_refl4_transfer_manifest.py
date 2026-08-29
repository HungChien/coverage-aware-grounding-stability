from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COUNTS = {"coco": 600, "objects365": 400}
ID_PATTERN = re.compile(r"^(?:coco|o365)_(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_excluded_coco_ids(paths: list[Path]) -> set[int]:
    excluded: set[int] = set()
    for path in paths:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        rows = json.loads(resolved.read_text(encoding="utf-8"))
        excluded.update(int(row["image_id"]) for row in rows)
    return excluded


def source_name(raw_image_id: str) -> str:
    if raw_image_id.startswith("coco_"):
        return "coco"
    if raw_image_id.startswith("o365_"):
        return "objects365"
    raise ValueError(f"unsupported Ref-L4 image id: {raw_image_id!r}")


def numeric_image_id(raw_image_id: str) -> int:
    match = ID_PATTERN.match(raw_image_id)
    if match is None:
        raise ValueError(f"cannot parse Ref-L4 image id: {raw_image_id!r}")
    value = int(match.group(1))
    return value if raw_image_id.startswith("coco_") else 1_000_000_000 + value


def xywh_to_xyxy(box: np.ndarray | list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    if not np.isfinite([x, y, width, height]).all() or width <= 0 or height <= 0:
        raise ValueError(f"invalid XYWH box: {box}")
    return [x, y, x + width, y + height]


def build_image_index(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if path.name in index:
            raise ValueError(f"duplicate Ref-L4 image filename: {path.name}")
        index[path.name] = path
    if not index:
        raise RuntimeError(f"no images found below {image_root}")
    return index


def valid_box(box: list[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    tolerance = 1e-3
    return (
        x1 >= -tolerance
        and y1 >= -tolerance
        and x2 <= width + tolerance
        and y2 <= height + tolerance
        and x2 > x1
        and y2 > y1
    )


def choose_unique_images(
    frame: pd.DataFrame,
    source: str,
    count: int,
    rng: np.random.Generator,
    excluded_coco_ids: set[int],
    image_index: dict[str, Path],
    category_map: dict[str, int],
) -> list[dict]:
    raw_prefix = "coco_" if source == "coco" else "o365_"
    subset = frame.loc[frame["image_id"].astype(str).str.startswith(raw_prefix)].copy()
    row_order = rng.permutation(len(subset))
    subset = subset.iloc[row_order]
    chosen_images: set[str] = set()
    rows: list[dict] = []
    for row in subset.itertuples(index=False):
        raw_image_id = str(row.image_id)
        if raw_image_id in chosen_images:
            continue
        parsed_id = numeric_image_id(raw_image_id)
        if source == "coco" and parsed_id in excluded_coco_ids:
            continue
        query = str(row.caption).strip()
        if not query:
            continue
        image_path = image_index.get(str(row.file_name))
        if image_path is None:
            continue
        target_box = xywh_to_xyxy(row.bbox)
        width, height = int(row.width), int(row.height)
        if not valid_box(target_box, width, height):
            continue
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        if (actual_width, actual_height) != (width, height):
            raise ValueError(
                f"metadata/image size mismatch for {image_path}: "
                f"metadata={(width, height)} actual={(actual_width, actual_height)}"
            )
        original_category = str(row.ori_category_id)
        rows.append(
            {
                "dataset": "ref_l4",
                "source_split": source,
                "official_split": "test",
                "image_id": parsed_id,
                "source_image_id": raw_image_id,
                "ref_id": int(row.id),
                "query": query,
                "query_word_count": len(query.split()),
                "target_box": target_box,
                "bbox_area": float(row.bbox_area),
                "category_id": int(category_map[original_category]),
                "original_category_id": original_category,
                "is_rewrite": bool(row.is_rewrite),
                "image_width": width,
                "image_height": height,
                "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        chosen_images.add(raw_image_id)
        if len(rows) % 50 == 0:
            print(f"{source}: selected {len(rows)}/{count}", flush=True)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"selected {len(rows)} of the requested {count} {source} images")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        default=ROOT / "data_operational" / "ref_l4_official" / "ref-l4-test.parquet",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT / "data_operational" / "ref_l4_official",
    )
    parser.add_argument(
        "--image-archive",
        type=Path,
        default=ROOT / "data_operational" / "ref_l4_official" / "images.tar.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data_operational" / "refl4_transfer1000",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--coco-count", type=int, default=SOURCE_COUNTS["coco"])
    parser.add_argument(
        "--objects365-count", type=int, default=SOURCE_COUNTS["objects365"]
    )
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()

    if not args.image_archive.exists():
        raise FileNotFoundError(args.image_archive)
    excluded_coco_ids = load_excluded_coco_ids(args.exclude_manifest)
    frame = pd.read_parquet(args.parquet)
    if set(frame["split"].astype(str)) != {"test"}:
        raise ValueError("the Ref-L4 transfer manifest must be sampled from test only")
    category_values = sorted(frame["ori_category_id"].astype(str).unique())
    category_map = {value: index + 1 for index, value in enumerate(category_values)}
    image_index = build_image_index(args.image_root)
    rng = np.random.default_rng(args.seed)
    rows = []
    rows.extend(
        choose_unique_images(
            frame,
            "coco",
            args.coco_count,
            rng,
            excluded_coco_ids,
            image_index,
            category_map,
        )
    )
    rows.extend(
        choose_unique_images(
            frame,
            "objects365",
            args.objects365_count,
            rng,
            excluded_coco_ids,
            image_index,
            category_map,
        )
    )
    rows = [rows[index] for index in rng.permutation(len(rows))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    selected_coco_ids = {
        int(row["image_id"]) for row in rows if row["source_split"] == "coco"
    }
    word_counts = np.asarray([int(row["query_word_count"]) for row in rows])
    scale = np.sqrt(np.asarray([float(row["bbox_area"]) for row in rows]))
    metadata = {
        "source": "JierunChen/Ref-L4",
        "official_split": "test",
        "sampling_design": "deterministic_source_stratified_unique_image_sample",
        "source_counts": {
            source: sum(row["source_split"] == source for row in rows)
            for source in ("coco", "objects365")
        },
        "count": len(rows),
        "unique_image_count": len({int(row["image_id"]) for row in rows}),
        "unique_source_image_count": len({row["source_image_id"] for row in rows}),
        "unique_ref_count": len({int(row["ref_id"]) for row in rows}),
        "unique_category_count": len({int(row["category_id"]) for row in rows}),
        "excluded_coco_image_count": len(excluded_coco_ids),
        "coco_overlap_with_excluded": len(selected_coco_ids & excluded_coco_ids),
        "seed": args.seed,
        "bbox_source_format": "xywh",
        "manifest_bbox_format": "xyxy",
        "query_word_count": {
            "mean": float(word_counts.mean()),
            "minimum": int(word_counts.min()),
            "median": float(np.median(word_counts)),
            "maximum": int(word_counts.max()),
        },
        "sqrt_bbox_area": {
            "mean": float(scale.mean()),
            "minimum": float(scale.min()),
            "median": float(np.median(scale)),
            "maximum": float(scale.max()),
        },
        "rewrite_count": sum(bool(row["is_rewrite"]) for row in rows),
        "selection_unit": "one_expression_per_unique_image",
        "excluded_manifests": [str(path) for path in args.exclude_manifest],
        "parquet_sha256": sha256(args.parquet),
        "image_archive_bytes": args.image_archive.stat().st_size,
        "image_archive_sha256": sha256(args.image_archive),
        "manifest_sha256": sha256(manifest_path),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
