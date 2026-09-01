"""Public API for coverage-aware candidate-order stability."""

from .adapters import GroundingAdapter
from .api import BenchmarkEvaluator, EvaluationResult, candidate_from_dict
from .operational_stability import (
    OutputContract,
    ProbeOutcome,
    StabilityProfile,
    evaluate_probe_outcome,
    summarize_outcomes,
)
from .candidates import Candidate

__all__ = [
    "BenchmarkEvaluator",
    "Candidate",
    "EvaluationResult",
    "GroundingAdapter",
    "OutputContract",
    "ProbeOutcome",
    "StabilityProfile",
    "candidate_from_dict",
    "evaluate_probe_outcome",
    "summarize_outcomes",
]

__version__ = "1.1.0"
