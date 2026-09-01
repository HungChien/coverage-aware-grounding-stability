from __future__ import annotations

import math

import numpy as np
import pytest

from coverage_aware_grounding_stability.two_stage_sampling import (
    crossover_probe_count,
    design_effect,
    effective_independent_trials,
    estimate_variance_components,
    intraclass_correlation,
    optimal_probe_count,
    probe_variance_share,
    probes_for_variance_share,
    required_unit_count,
    two_stage_hoeffding_radius,
    variance_of_mean,
)


def test_exact_variance_matches_design_effect_identity() -> None:
    theta = 0.6
    between = 0.08
    within = theta * (1.0 - theta) - between
    unit_count = 125
    probe_count = 10
    direct = variance_of_mean(between, within, unit_count, probe_count)
    rho = intraclass_correlation(between, within)
    via_design_effect = (
        theta
        * (1.0 - theta)
        / (unit_count * probe_count)
        * (1.0 + (probe_count - 1) * rho)
    )
    assert direct == pytest.approx(via_design_effect)
    assert design_effect(between, within, probe_count) == pytest.approx(1 + 9 * rho)


def test_method_of_moments_recovers_nonnegative_components() -> None:
    outcomes = np.asarray(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    estimate = estimate_variance_components(outcomes)
    assert estimate.theta_hat == pytest.approx(0.5)
    assert estimate.within >= 0
    assert estimate.between >= 0
    assert estimate.estimated_variance == pytest.approx(
        estimate.between / 4 + estimate.within / 16
    )


def test_probe_share_and_crossover() -> None:
    between = 0.04
    within = 0.16
    assert crossover_probe_count(between, within) == pytest.approx(4.0)
    assert probe_variance_share(between, within, 4) == pytest.approx(0.5)
    assert probes_for_variance_share(between, within, 0.10) == 36


def test_effective_sample_size_is_bounded_by_raw_probe_count() -> None:
    effective = effective_independent_trials(0.09, 0.11, 100, 80)
    assert 100 < effective < 8000


def test_optimal_probe_count_matches_continuous_solution() -> None:
    # sqrt(B*c_X/(A*c_U)) = sqrt(0.16*100/(0.04*1)) = 20.
    assert optimal_probe_count(0.04, 0.16, 100.0, 1.0, maximum=80) == 20


def test_required_sample_size_and_bound_are_well_formed() -> None:
    count = required_unit_count(0.04, 0.16, 10, 0.03)
    assert count > 0
    assert two_stage_hoeffding_radius(500, 20, 0.05) > 0


def test_zero_between_component_boundary() -> None:
    assert math.isinf(crossover_probe_count(0.0, 0.1))
    assert math.isinf(probes_for_variance_share(0.0, 0.1, 0.1))
