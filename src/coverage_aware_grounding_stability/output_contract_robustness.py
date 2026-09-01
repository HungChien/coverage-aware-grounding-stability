from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ContractSetting:
    tracked_candidate_count: int
    exposed_candidate_count: int
    match_iou_threshold: float
    birth_duplicate_iou_threshold: float
    duplicate_iou_threshold: float = 0.70

    @property
    def setting_id(self) -> str:
        return (
            f"kt{self.tracked_candidate_count}_ke{self.exposed_candidate_count}_"
            f"match{self.match_iou_threshold:.2f}_"
            f"birth{self.birth_duplicate_iou_threshold:.2f}_"
            f"dup{self.duplicate_iou_threshold:.2f}"
        )


def registered_settings(preregistration: dict) -> list[ContractSetting]:
    grid = preregistration["replayable_factorial_grid"]
    duplicate = float(grid["fixed_duplicate_iou_threshold"])
    return [
        ContractSetting(
            tracked_candidate_count=int(tracked),
            exposed_candidate_count=int(exposed),
            match_iou_threshold=float(match),
            birth_duplicate_iou_threshold=float(birth),
            duplicate_iou_threshold=duplicate,
        )
        for tracked, exposed, match, birth in product(
            grid["tracked_candidate_count"],
            grid["exposed_candidate_count"],
            grid["match_iou_threshold"],
            grid["birth_duplicate_iou_threshold"],
        )
    ]


def default_setting_index(
    settings: Iterable[ContractSetting], preregistration: dict
) -> int:
    default = preregistration["default_output_contract"]
    target = ContractSetting(
        tracked_candidate_count=int(default["tracked_candidate_count"]),
        exposed_candidate_count=int(default["exposed_candidate_count"]),
        match_iou_threshold=float(default["match_iou_threshold"]),
        birth_duplicate_iou_threshold=float(
            default["birth_duplicate_iou_threshold"]
        ),
        duplicate_iou_threshold=float(default["duplicate_iou_threshold"]),
    )
    settings = list(settings)
    if target not in settings:
        raise ValueError("the default output contract is absent from the grid")
    return settings.index(target)


def contract_envelope(values: np.ndarray, default_index: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not 0 <= default_index < values.size:
        raise IndexError("default_index is outside values")
    minimum_index = int(np.argmin(values))
    maximum_index = int(np.argmax(values))
    default = float(values[default_index])
    return {
        "default": default,
        "minimum": float(values[minimum_index]),
        "maximum": float(values[maximum_index]),
        "width": float(values[maximum_index] - values[minimum_index]),
        "maximum_absolute_departure": float(np.max(np.abs(values - default))),
        "minimum_index": minimum_index,
        "maximum_index": maximum_index,
    }


def finite_grid_ranking(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float | bool | int]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("model values must be aligned one-dimensional arrays")
    gaps = values_a - values_b
    worst_index = int(np.argmin(gaps))
    return {
        "minimum_same_contract_gap": float(gaps[worst_index]),
        "maximum_same_contract_gap": float(np.max(gaps)),
        "worst_setting_index": worst_index,
        "same_contract_ranking_invariant": bool(np.all(gaps > 0.0)),
        "model_a_minimum": float(np.min(values_a)),
        "model_b_maximum": float(np.max(values_b)),
        "strong_interval_gap": float(np.min(values_a) - np.max(values_b)),
        "strong_interval_separation": bool(np.min(values_a) > np.max(values_b)),
    }


def paired_bootstrap_minimum_gap(
    sample_values_a: np.ndarray,
    sample_values_b: np.ndarray,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap the worst same-contract model gap over a fixed finite grid.

    Arrays have shape ``(settings, paired samples)``.  The minimum is taken
    inside every repetition so the uncertainty calculation includes selection
    of the empirically worst registered contract.
    """

    sample_values_a = np.asarray(sample_values_a, dtype=float)
    sample_values_b = np.asarray(sample_values_b, dtype=float)
    if sample_values_a.shape != sample_values_b.shape:
        raise ValueError("paired sample matrices must have identical shapes")
    if sample_values_a.ndim != 2 or sample_values_a.shape[1] == 0:
        raise ValueError("sample matrices must have shape (settings, samples)")
    rng = np.random.default_rng(seed)
    sample_count = sample_values_a.shape[1]
    difference = sample_values_a - sample_values_b
    result = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        indices = rng.integers(0, sample_count, size=sample_count)
        result[repetition] = float(np.min(difference[:, indices].mean(axis=1)))
    return result
