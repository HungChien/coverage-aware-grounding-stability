from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def xywh_to_xyxy(box) -> list[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def load_excluded_image_ids(paths: list[Path]) -> set[int]:
    excluded: set[int] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(int(row["image_id"]) for row in rows)
    return excluded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dddraxxx/refcoco-benchmark")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data_operational" / "refcoco_unseen500",
    )
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")

    excluded = load_excluded_image_ids(args.exclude_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    selected: list[dict] = []
    seen = set(excluded)
    for row in dataset:
        image_id = int(row["image_id"])
        sentence = str(row["sentence"]).strip()
        box = row["bbox"]
        if image_id in seen or not sentence or len(box) != 4:
            continue
        image = row["image"].convert("RGB")
        target = image_dir / f"{image_id:012d}.jpg"
        image.save(target, quality=95)
        selected.append(
            {
                "image_id": image_id,
                "ref_id": int(row["ref_id"]),
                "query": sentence,
                "target_box": xywh_to_xyxy(box),
                "category_id": int(row["category_id"]),
                "image_path": str(target.relative_to(ROOT)),
            }
        )
        seen.add(image_id)
        if len(selected) >= args.count:
            break
        if len(selected) % 50 == 0:
            print(f"prepared {len(selected)}/{args.count} unique image-query pairs")

    if len(selected) != args.count:
        raise RuntimeError(
            f"only {len(selected)} unseen unique images were available; "
            f"requested {args.count}"
        )

    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    metadata = {
        "source": args.dataset,
        "split": args.split,
        "count": len(selected),
        "unique_image_count": len({row["image_id"] for row in selected}),
        "excluded_image_count": len(excluded),
        "overlap_with_excluded": len(
            {row["image_id"] for row in selected} & excluded
        ),
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "manifest_sha256": sha256(manifest),
        "excluded_manifests": [str(path) for path in args.exclude_manifest],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
