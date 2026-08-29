import pytest

from src.operational_stability import (
    OutputContract,
    aggregate_model_operational,
    evaluate_probe_outcome,
    exact_binomial_interval,
    family_risk_identity,
    summarize_outcomes,
)
from src.reliability import Candidate


def candidate(box, score):
    return Candidate(tuple(map(float, box)), float(score), "object")


CONTRACT = OutputContract(
    tracked_candidate_count=3,
    exposed_candidate_count=10,
    duplicate_iou_threshold=0.7,
    match_iou_threshold=0.2,
    birth_duplicate_iou_threshold=0.7,
)

CLEAN = [
    candidate((0, 0, 10, 10), 0.9),
    candidate((20, 20, 30, 30), 0.7),
    candidate((40, 40, 50, 50), 0.5),
]


def test_covered_stable_probe():
    outcome, _ = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((0, 0, 10, 10), 0.85),
            candidate((20, 20, 30, 30), 0.65),
            candidate((40, 40, 50, 50), 0.45),
        ],
        CONTRACT,
    )
    assert outcome.coverage == 1
    assert outcome.rank_stable == 1
    assert outcome.operational_stable == 1
    assert outcome.primary_failure == "stable"
    assert outcome.gaps == pytest.approx([0.20, 0.40])


def test_covered_reversal_has_disjoint_culprit():
    outcome, _ = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((0, 0, 10, 10), 0.55),
            candidate((20, 20, 30, 30), 0.72),
            candidate((40, 40, 50, 50), 0.80),
        ],
        CONTRACT,
    )
    assert outcome.coverage == 1
    assert outcome.rank_stable == 0
    assert outcome.operational_stable == 0
    assert outcome.primary_failure == "ranking_reversal"
    assert outcome.culprit_clean_index == 2


def test_missing_winner_is_coverage_failure_without_score_imputation():
    outcome, _ = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((20, 20, 30, 30), 0.72),
            candidate((40, 40, 50, 50), 0.50),
        ],
        CONTRACT,
    )
    assert outcome.coverage == 0
    assert outcome.rank_stable is None
    assert outcome.gaps is None
    assert outcome.primary_failure == "winner_missing"
    assert outcome.matched_scores[0] is None


def test_missing_competitor_is_coverage_failure():
    outcome, _ = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((0, 0, 10, 10), 0.80),
            candidate((20, 20, 30, 30), 0.60),
        ],
        CONTRACT,
    )
    assert outcome.coverage == 0
    assert outcome.primary_failure == "competitor_missing"


def test_spatially_novel_high_score_birth_is_coverage_failure():
    outcome, _ = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((0, 0, 10, 10), 0.80),
            candidate((20, 20, 30, 30), 0.60),
            candidate((40, 40, 50, 50), 0.50),
            candidate((70, 70, 80, 80), 0.81),
        ],
        CONTRACT,
    )
    assert outcome.coverage == 0
    assert outcome.primary_failure == "threatening_birth"
    assert outcome.threatening_birth_scores == pytest.approx([0.81])


def test_duplicate_high_score_box_is_suppressed_not_counted_as_birth():
    outcome, candidates = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((0, 0, 10, 10), 0.80),
            candidate((0.2, 0.2, 10.2, 10.2), 0.79),
            candidate((20, 20, 30, 30), 0.60),
            candidate((40, 40, 50, 50), 0.50),
        ],
        CONTRACT,
    )
    assert len(candidates) == 3
    assert outcome.coverage == 1
    assert outcome.operational_stable == 1


def test_exact_risk_and_family_decompositions():
    stable, _ = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((0, 0, 10, 10), 0.8),
            candidate((20, 20, 30, 30), 0.6),
            candidate((40, 40, 50, 50), 0.5),
        ],
        CONTRACT,
    )
    reversal, _ = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((0, 0, 10, 10), 0.5),
            candidate((20, 20, 30, 30), 0.7),
            candidate((40, 40, 50, 50), 0.4),
        ],
        CONTRACT,
    )
    missing, _ = evaluate_probe_outcome(
        CLEAN,
        [
            candidate((0, 0, 10, 10), 0.8),
            candidate((20, 20, 30, 30), 0.6),
        ],
        CONTRACT,
    )
    profile = summarize_outcomes(
        [stable, reversal, missing, stable],
        ["blur", "blur", "noise", "noise"],
    )
    assert profile.coverage.estimate == pytest.approx(0.75)
    assert profile.conditional_ranking.estimate == pytest.approx(2 / 3)
    assert profile.operational.estimate == pytest.approx(0.5)
    assert (
        profile.conditional_ranking.estimate - profile.operational.estimate
    ) == pytest.approx(
        profile.conditional_ranking.estimate * (1.0 - profile.coverage.estimate)
    )
    assert profile.coverage_risk == pytest.approx(0.25)
    assert profile.conditional_ranking_risk == pytest.approx(0.25)
    assert profile.risk_decomposition_error < 1e-12
    assert family_risk_identity(profile) < 1e-12
    assert sum(profile.family_risk_share.values()) == pytest.approx(1.0)
    assert sum(profile.primary_failure_shares.values()) == pytest.approx(1.0)


def test_exact_interval_and_full_manifest_aggregation():
    interval = exact_binomial_interval(8, 10, confidence=0.95)
    assert interval.lower < interval.estimate < interval.upper
    full, eligible, eligibility = aggregate_model_operational(
        [0.9, 0.8, 0.7], [1, 0, 1]
    )
    assert full == pytest.approx((0.9 + 0.0 + 0.7) / 3)
    assert eligible == pytest.approx(0.8)
    assert eligibility == pytest.approx(2 / 3)


def test_order_event_is_invariant_to_strictly_increasing_score_transform():
    output = [
        candidate((0, 0, 10, 10), 0.8),
        candidate((20, 20, 30, 30), 0.6),
        candidate((40, 40, 50, 50), 0.5),
    ]
    transformed = [candidate(item.box, item.score**3 + 4.0) for item in output]
    original_outcome, _ = evaluate_probe_outcome(CLEAN, output, CONTRACT)
    transformed_outcome, _ = evaluate_probe_outcome(CLEAN, transformed, CONTRACT)
    assert transformed_outcome.coverage == original_outcome.coverage
    assert transformed_outcome.rank_stable == original_outcome.rank_stable
    assert transformed_outcome.culprit_clean_index == original_outcome.culprit_clean_index
