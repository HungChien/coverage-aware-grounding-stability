from __future__ import annotations

import numpy as np
import pytest

from scripts.analyse_reference_80_adequacy import (
    clopper_pearson_interval,
    family_balanced_split_indices,
    paired_metrics,
)


def test_clopper_pearson_boundaries() -> None:
    lower_zero, upper_zero = clopper_pearson_interval(0, 80)
    lower_one, upper_one = clopper_pearson_interval(80, 80)
    assert lower_zero == 0.0
    assert upper_one == 1.0
    assert 0.0 < upper_zero < 0.1
    assert 0.9 < lower_one < 1.0


def test_family_balanced_split_is_disjoint_and_balanced() -> None:
    families = tuple(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    left, right = family_balanced_split_indices(families, np.random.default_rng(7))
    assert len(left) == len(right) == 6
    assert set(left).isdisjoint(set(right))
    assert set(left) | set(right) == set(range(12))
    for family in set(families):
        assert sum(families[index] == family for index in left) == 2
        assert sum(families[index] == family for index in right) == 2


def test_paired_metrics_have_expected_values() -> None:
    left = np.asarray([0.0, 0.5, 1.0])
    right = np.asarray([0.0, 1.0, 0.5])
    metrics = paired_metrics(left, right)
    assert metrics["absolute_model_mean_difference"] == pytest.approx(0.0)
    assert metrics["sample_mae"] == pytest.approx(1 / 3)
    assert metrics["sample_rmse"] == pytest.approx(np.sqrt(1 / 6))
