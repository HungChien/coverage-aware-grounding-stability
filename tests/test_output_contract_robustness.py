import json
from pathlib import Path

import numpy as np

from scripts.analyse_output_contract_robustness import (
    associate_all,
    outcome_from_assignment,
)
from src.operational_stability import OutputContract, evaluate_probe_outcome
from src.output_contract_robustness import (
    contract_envelope,
    default_setting_index,
    finite_grid_ranking,
    paired_bootstrap_minimum_gap,
    registered_settings,
)
from src.reliability import Candidate


ROOT = Path(__file__).resolve().parents[1]


def candidate(box, score):
    return Candidate(box=tuple(box), score=float(score), label="object")


def test_preregistered_grid_contains_default_and_has_108_settings():
    path = ROOT / "config" / "output_contract_preregistration_v2.json"
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    settings = registered_settings(preregistration)
    index = default_setting_index(settings, preregistration)

    assert len(settings) == 108
    assert settings[index].tracked_candidate_count == 5
    assert settings[index].exposed_candidate_count == 20
    assert settings[index].match_iou_threshold == 0.15
    assert settings[index].birth_duplicate_iou_threshold == 0.70


def test_preregistration_covers_every_numeric_and_fixed_contract_rule():
    path = ROOT / "config" / "output_contract_preregistration_v2.json"
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    default = preregistration["default_output_contract"]

    for field in OutputContract.__dataclass_fields__:
        assert field in default
    for rule in (
        "minimum_clean_candidate_count",
        "strict_score_order",
        "association_rule",
        "missing_candidate_policy",
        "threatening_birth_policy",
        "failure_precedence",
    ):
        assert rule in default


def test_contract_envelope_reports_absolute_departure():
    summary = contract_envelope(np.asarray([0.72, 0.80, 0.76]), default_index=2)

    assert summary["minimum"] == 0.72
    assert summary["maximum"] == 0.80
    assert np.isclose(summary["width"], 0.08)
    assert np.isclose(summary["maximum_absolute_departure"], 0.04)


def test_finite_grid_ranking_checks_same_contract_and_strong_separation():
    result = finite_grid_ranking(
        np.asarray([0.81, 0.78, 0.84]),
        np.asarray([0.55, 0.60, 0.58]),
    )

    assert result["same_contract_ranking_invariant"]
    assert result["strong_interval_separation"]
    assert np.isclose(result["minimum_same_contract_gap"], 0.18)
    assert np.isclose(result["strong_interval_gap"], 0.18)


def test_paired_bootstrap_recomputes_minimum_inside_each_repetition():
    model_a = np.asarray([[1.0, 0.0, 1.0], [0.8, 0.8, 0.8]])
    model_b = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    values = paired_bootstrap_minimum_gap(model_a, model_b, 50, seed=9)

    assert values.shape == (50,)
    assert np.all(values >= 0.0)
    assert np.all(values <= 0.7 + 1e-12)


def test_optimized_replay_event_matches_frozen_evaluator():
    clean = [
        candidate([0, 0, 10, 10], 0.9),
        candidate([20, 0, 30, 10], 0.7),
        candidate([40, 0, 50, 10], 0.4),
    ]
    perturbed = [
        candidate([1, 0, 11, 10], 0.85),
        candidate([20, 1, 30, 11], 0.72),
        candidate([40, 0, 50, 10], 0.45),
        candidate([60, 0, 70, 10], 0.84),
    ]
    contract = OutputContract(
        tracked_candidate_count=3,
        exposed_candidate_count=4,
        duplicate_iou_threshold=0.70,
        match_iou_threshold=0.15,
        birth_duplicate_iou_threshold=0.70,
    )
    expected, frozen_candidates = evaluate_probe_outcome(clean, perturbed, contract)
    mapping, overlaps = associate_all(clean, frozen_candidates)
    coverage, operational, rank_stable, failure = outcome_from_assignment(
        clean,
        frozen_candidates,
        mapping,
        overlaps,
        contract.match_iou_threshold,
        contract.birth_duplicate_iou_threshold,
    )

    assert coverage == expected.coverage
    assert operational == expected.operational_stable
    assert rank_stable == expected.rank_stable
    assert failure == expected.primary_failure
