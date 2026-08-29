from __future__ import annotations

import pytest


def optimism_components(gamma: float, coverage: float, persistence: float):
    eligible_operational = coverage * persistence
    full_operational = gamma * eligible_operational
    coverage_optimism = persistence - eligible_operational
    eligibility_optimism = eligible_operational - full_operational
    total_optimism = persistence - full_operational
    return (
        coverage_optimism,
        eligibility_optimism,
        total_optimism,
        full_operational,
    )


def test_complete_case_optimism_identity() -> None:
    gamma = 0.781
    coverage = 0.778585147247119
    persistence = 0.8820046869218435
    coverage_gap, eligibility_gap, total_gap, full_operational = optimism_components(
        gamma, coverage, persistence
    )

    assert full_operational == pytest.approx(0.536325)
    assert coverage_gap + eligibility_gap == pytest.approx(total_gap)
    assert total_gap == pytest.approx(persistence * (1.0 - gamma * coverage))
    assert total_gap >= 0.0


@pytest.mark.parametrize(
    ("gamma", "coverage", "persistence"),
    [
        (1.0, 1.0, 0.9),
        (1.0, 0.8, 0.9),
        (0.7, 1.0, 0.9),
        (0.7, 0.8, 0.9),
        (0.0, 0.0, 0.0),
    ],
)
def test_optimism_is_non_negative(
    gamma: float, coverage: float, persistence: float
) -> None:
    coverage_gap, eligibility_gap, total_gap, _ = optimism_components(
        gamma, coverage, persistence
    )
    assert coverage_gap >= 0.0
    assert eligibility_gap >= 0.0
    assert total_gap >= 0.0
    assert total_gap == pytest.approx(coverage_gap + eligibility_gap)
