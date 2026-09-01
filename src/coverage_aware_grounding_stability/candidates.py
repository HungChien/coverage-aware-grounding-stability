from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class Candidate:
    box: tuple[float, float, float, float]
    score: float
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def top1_correct(
    candidates: Sequence[Candidate],
    ground_truth_boxes: Sequence[Sequence[float]],
    threshold: float = 0.5,
) -> tuple[int, float]:
    if not candidates or not ground_truth_boxes:
        return 0, 0.0
    top = max(candidates, key=lambda c: c.score)
    best_iou = max(iou_xyxy(top.box, gt) for gt in ground_truth_boxes)
    return int(best_iou >= threshold), float(best_iou)
