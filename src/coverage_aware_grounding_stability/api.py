from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .operational_stability import (
    OutputContract,
    ProbeOutcome,
    StabilityProfile,
    evaluate_probe_outcome,
    select_spatially_distinct,
    summarize_outcomes,
)
from .candidates import Candidate


def candidate_from_dict(value: Mapping[str, Any]) -> Candidate:
    """Parse one model-agnostic candidate from a JSON-compatible mapping."""

    box = value.get("box")
    if (
        not isinstance(box, Sequence)
        or isinstance(box, (str, bytes))
        or len(box) != 4
    ):
        raise ValueError("candidate.box must contain four xyxy coordinates")
    coordinates = tuple(float(item) for item in box)
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise ValueError("candidate.box must have positive width and height")
    return Candidate(
        box=coordinates,
        score=float(value["score"]),
        label=str(value.get("label", "")),
    )


@dataclass(frozen=True)
class EvaluationResult:
    """Result of applying one output contract to one clean/perturbed pair."""

    clean_eligible: int
    tracked_clean_candidates: list[Candidate]
    exposed_perturbed_candidates: list[Candidate]
    outcome: ProbeOutcome | None
    ineligible_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_eligible": self.clean_eligible,
            "tracked_clean_candidates": [
                asdict(item) for item in self.tracked_clean_candidates
            ],
            "exposed_perturbed_candidates": [
                asdict(item) for item in self.exposed_perturbed_candidates
            ],
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
            "ineligible_reason": self.ineligible_reason,
        }


class BenchmarkEvaluator:
    """Model-agnostic evaluator for candidate lists.

    A caller only needs to supply score-ordered boxes from its own model. The
    evaluator applies duplicate suppression, clean eligibility, one-to-one
    association, candidate-birth checks, and strict winner-order evaluation.
    """

    def __init__(self, contract: OutputContract | None = None):
        self.contract = contract or OutputContract()

    def prepare_clean(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        return select_spatially_distinct(
            candidates,
            maximum=self.contract.tracked_candidate_count,
            duplicate_iou_threshold=self.contract.duplicate_iou_threshold,
        )

    def evaluate(
        self,
        clean_candidates: Sequence[Candidate],
        perturbed_candidates: Sequence[Candidate],
    ) -> EvaluationResult:
        tracked = self.prepare_clean(clean_candidates)
        if len(tracked) < 2:
            return EvaluationResult(
                clean_eligible=0,
                tracked_clean_candidates=tracked,
                exposed_perturbed_candidates=[],
                outcome=None,
                ineligible_reason="fewer_than_two_spatially_distinct_clean_candidates",
            )
        outcome, exposed = evaluate_probe_outcome(
            tracked,
            perturbed_candidates,
            self.contract,
        )
        return EvaluationResult(
            clean_eligible=1,
            tracked_clean_candidates=tracked,
            exposed_perturbed_candidates=exposed,
            outcome=outcome,
        )

    def summarize(
        self,
        clean_candidates: Sequence[Candidate],
        perturbed_candidate_sets: Sequence[Sequence[Candidate]],
        families: Sequence[str],
        confidence: float = 0.95,
    ) -> StabilityProfile | None:
        if len(perturbed_candidate_sets) != len(families):
            raise ValueError("perturbed_candidate_sets and families must align")
        evaluated = [
            self.evaluate(clean_candidates, perturbed)
            for perturbed in perturbed_candidate_sets
        ]
        if not evaluated or evaluated[0].clean_eligible == 0:
            return None
        outcomes = [item.outcome for item in evaluated]
        if any(outcome is None for outcome in outcomes):
            raise RuntimeError("eligible evaluations must produce outcomes")
        return summarize_outcomes(
            outcomes,  # type: ignore[arg-type]
            families,
            confidence=confidence,
        )
