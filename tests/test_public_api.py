import json

from coverage_aware_grounding_stability import (
    BenchmarkEvaluator,
    Candidate,
    OutputContract,
    candidate_from_dict,
)
from coverage_aware_grounding_stability.cli import main


def test_evaluator_reports_stable_pair():
    evaluator = BenchmarkEvaluator(OutputContract(tracked_candidate_count=2))
    clean = [
        Candidate((0, 0, 10, 10), 0.9),
        Candidate((20, 0, 30, 10), 0.7),
    ]
    perturbed = [
        Candidate((1, 0, 11, 10), 0.85),
        Candidate((20, 1, 30, 11), 0.65),
    ]

    result = evaluator.evaluate(clean, perturbed)

    assert result.clean_eligible == 1
    assert result.outcome is not None
    assert result.outcome.operational_stable == 1
    assert result.outcome.primary_failure == "stable"


def test_evaluator_retains_clean_ineligibility():
    evaluator = BenchmarkEvaluator()
    result = evaluator.evaluate([Candidate((0, 0, 10, 10), 0.9)], [])

    assert result.clean_eligible == 0
    assert result.outcome is None
    assert result.ineligible_reason is not None


def test_candidate_parser_rejects_degenerate_box():
    try:
        candidate_from_dict({"box": [0, 0, 0, 2], "score": 0.5})
    except ValueError as error:
        assert "positive width" in str(error)
    else:
        raise AssertionError("degenerate candidate should be rejected")


def test_cli_evaluates_jsonl(tmp_path):
    input_path = tmp_path / "candidate_pairs.jsonl"
    output_path = tmp_path / "outcomes.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "demo",
                "clean_candidates": [
                    {"box": [0, 0, 10, 10], "score": 0.9},
                    {"box": [20, 0, 30, 10], "score": 0.7},
                ],
                "perturbed_candidates": [
                    {"box": [1, 0, 11, 10], "score": 0.85},
                    {"box": [20, 1, 30, 11], "score": 0.65},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(["evaluate", "--input", str(input_path), "--output", str(output_path)])
        == 0
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["id"] == "demo"
    assert result["outcome"]["operational_stable"] == 1
