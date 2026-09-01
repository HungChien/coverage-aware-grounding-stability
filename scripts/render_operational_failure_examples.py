from __future__ import annotations

import argparse
import gzip
import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverage_aware_grounding_stability.random_probes import RandomProbe, apply_random_probe


CAUSES = [
    "winner_missing",
    "competitor_missing",
    "threatening_birth",
    "ranking_reversal",
]
COLORS = ["#EF4444", "#2563EB", "#10B981", "#F59E0B", "#8B5CF6"]
CAUSE_LABELS = {
    "winner_missing": "Winner loss",
    "competitor_missing": "Competitor loss",
    "threatening_birth": "Candidate birth",
    "ranking_reversal": "Rank reversal",
}


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "calibrib.ttf", "segoeuib.ttf"] if bold else [
        "arial.ttf",
        "calibri.ttf",
        "segoeui.ttf",
    ]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_trace(path: Path) -> dict[tuple[int, int], dict]:
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[(int(record["image_id"]), int(record["ref_id"]))] = record
    return records


def draw_candidates(image: Image.Image, candidates: list[dict], prefix: str) -> Image.Image:
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(max(17, round(min(image.size) / 28)), bold=True)
    width = max(3, round(min(image.size) / 145))
    for index, candidate in enumerate(candidates[:5]):
        color = COLORS[index % len(COLORS)]
        box = tuple(float(value) for value in candidate["box"])
        draw.rectangle(box, outline=color, width=width)
        label = f"{prefix}{index + 1}  {candidate['score']:.3f}"
        text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        label_height = text_box[3] - text_box[1]
        label_width = text_box[2] - text_box[0]
        anchor_x = max(3, min(float(box[0]), image.width - label_width - 9))
        anchor_y = max(3, float(box[1]) - label_height - 11)
        draw.rounded_rectangle(
            (anchor_x - 4, anchor_y - 3, anchor_x + label_width + 5, anchor_y + label_height + 5),
            radius=4,
            fill="white",
            outline=color,
            width=2,
        )
        draw.text(
            (anchor_x, anchor_y),
            label,
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="white",
        )
    return image


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail((size[0] - 12, size[1] - 12), Image.Resampling.LANCZOS)
    position = ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2)
    canvas.paste(copy, position)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results" / "operational_benchmark_v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "operational_benchmark_v1.json",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = Path(config["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {
        (int(item["image_id"]), int(item["ref_id"])): item for item in manifest
    }
    output = args.result_root / "analysis" / "failure_examples"
    output.mkdir(parents=True, exist_ok=True)

    audit = {}
    for model in config["models"]:
        records = load_trace(args.result_root / model / "sample_traces.jsonl.gz")
        selected: dict[str, tuple[dict, dict]] = {}
        for key in sorted(records):
            record = records[key]
            if not record.get("clean_eligible"):
                continue
            for probe in record["probes"]:
                if probe["split"] != "reference":
                    continue
                cause = probe["outcome"]["primary_failure"]
                if cause in CAUSES and cause not in selected:
                    selected[cause] = (record, probe)
            if len(selected) == len(CAUSES):
                break

        panel_width, panel_height = 780, 540
        header_height = 150
        canvas = Image.new(
            "RGB",
            (panel_width * 2, (panel_height + header_height) * len(CAUSES)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        title_font = load_font(30, bold=True)
        meta_font = load_font(24)
        column_font = load_font(24, bold=True)
        audit[model] = {}
        for row, cause in enumerate(CAUSES):
            top = row * (panel_height + header_height)
            draw.rectangle(
                (0, top, canvas.width - 1, top + header_height - 1),
                fill="#F3F4F6",
                outline="#D1D5DB",
                width=2,
            )
            if cause not in selected:
                draw.text(
                    (24, top + 48),
                    f"{CAUSE_LABELS[cause]}: no observed example",
                    fill="black",
                    font=title_font,
                )
                audit[model][cause] = None
                continue
            record, probe = selected[cause]
            key = (int(record["image_id"]), int(record["ref_id"]))
            source = sources[key]
            image_path = Path(source["image_path"])
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            clean_image = Image.open(image_path).convert("RGB")
            spec = RandomProbe(**probe["spec"])
            perturbed_image = apply_random_probe(clean_image, spec)
            clean_view = draw_candidates(
                clean_image, record["tracked_clean_candidates"], "C"
            )
            perturbed_view = draw_candidates(
                perturbed_image, probe["candidates"], "P"
            )
            clean_panel = fit_panel(clean_view, (panel_width, panel_height))
            perturbed_panel = fit_panel(perturbed_view, (panel_width, panel_height))
            canvas.paste(clean_panel, (0, top + header_height))
            canvas.paste(perturbed_panel, (panel_width, top + header_height))
            query_lines = textwrap.wrap(str(record["query"]), width=62) or [""]
            query_text = "\n".join(query_lines[:2])
            draw.text((24, top + 14), CAUSE_LABELS[cause], fill="#111827", font=title_font)
            draw.multiline_text(
                (370, top + 18),
                f'Query: “{query_text}”',
                fill="#111827",
                font=meta_font,
                spacing=5,
            )
            draw.text(
                (24, top + 82),
                f"Probe: {spec.family}, severity {spec.severity:.3f}",
                fill="#4B5563",
                font=meta_font,
            )
            draw.text(
                (24, top + 116),
                "Clean: tracked candidates",
                fill="#374151",
                font=column_font,
            )
            draw.text(
                (panel_width + 24, top + 116),
                "Perturbed: exposed candidates",
                fill="#374151",
                font=column_font,
            )
            audit[model][cause] = {
                "image_id": key[0],
                "ref_id": key[1],
                "query": record["query"],
                "probe": probe["spec"],
            }
        canvas.save(
            output / f"{model}_failure_examples.png",
            dpi=(300, 300),
            optimize=True,
        )

    (output / "failure_example_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
