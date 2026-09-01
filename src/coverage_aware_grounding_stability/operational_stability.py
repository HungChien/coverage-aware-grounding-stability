from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta as beta_distribution

from .candidates import Candidate, iou_xyxy


@dataclass(frozen=True)
class OutputContract:
    """Frozen rules that make candidate-order events reproducible.

    ``tracked_candidate_count`` controls the clean candidate universe.  The
    perturbed output may expose more candidates so that unmatched, high-score
    candidate births are not silently ignored.
    """

    tracked_candidate_count: int = 5
    exposed_candidate_count: int = 20
    duplicate_iou_threshold: float = 0.70
    match_iou_threshold: float = 0.15
    birth_duplicate_iou_threshold: float = 0.70

    def __post_init__(self) -> None:
        if self.tracked_candidate_count < 2:
            raise ValueError("tracked_candidate_count must be at least two")
        if self.exposed_candidate_count < self.tracked_candidate_count:
            raise ValueError(
                "exposed_candidate_count must be no smaller than "
                "tracked_candidate_count"
            )
        for name, value in (
            ("duplicate_iou_threshold", self.duplicate_iou_threshold),
            ("match_iou_threshold", self.match_iou_threshold),
            ("birth_duplicate_iou_threshold", self.birth_duplicate_iou_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProbeOutcome:
    """One observable operational-stability event under one registered probe."""

    coverage: int
    rank_stable: int | None
    operational_stable: int
    primary_failure: str
    culprit_clean_index: int | None
    mapping: list[int | None]
    matched_ious: list[float | None]
    matched_scores: list[float | None]
    matched_boxes: list[list[float] | None]
    gaps: list[float] | None
    missing_clean_indices: list[int]
    threatening_birth_indices: list[int]
    threatening_birth_scores: list[float]
    perturbed_candidate_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BinomialEstimate:
    successes: int
    trials: int
    estimate: float
    lower: float
    upper: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StabilityProfile:
    clean_eligible: int
    probe_count: int
    coverage: BinomialEstimate
    conditional_ranking: BinomialEstimate | None
    operational: BinomialEstimate
    operational_risk: float
    coverage_risk: float
    conditional_ranking_risk: float
    risk_decomposition_error: float
    family_trials: dict[str, int]
    family_operational_stability: dict[str, float]
    family_risk_contribution: dict[str, float]
    family_risk_share: dict[str, float]
    primary_failure_counts: dict[str, int]
    primary_failure_shares: dict[str, float]
    culprit_counts: dict[str, int]
    culprit_shares_among_ranking_failures: dict[str, float]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["coverage"] = self.coverage.to_dict()
        result["operational"] = self.operational.to_dict()
        result["conditional_ranking"] = (
            None
            if self.conditional_ranking is None
            else self.conditional_ranking.to_dict()
        )
        return result


def select_spatially_distinct(
    candidates: Sequence[Candidate],
    maximum: int,
    duplicate_iou_threshold: float,
) -> list[Candidate]:
    """Return a score-ordered, spatially distinct candidate list."""

    selected: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(
            iou_xyxy(candidate.box, accepted.box) < duplicate_iou_threshold
            for accepted in selected
        ):
            selected.append(candidate)
        if len(selected) >= maximum:
            break
    return selected


def _associate_candidates(
    clean: Sequence[Candidate],
    perturbed: Sequence[Candidate],
    minimum_iou: float,
) -> tuple[list[int | None], list[float | None]]:
    """Perform deterministic one-to-one maximum-IoU association."""

    if not clean:
        return [], []
    if not perturbed:
        return [None] * len(clean), [None] * len(clean)

    overlaps = np.asarray(
        [
            [iou_xyxy(source.box, target.box) for target in perturbed]
            for source in clean
        ],
        dtype=float,
    )
    rows, cols = linear_sum_assignment(1.0 - overlaps)
    mapping: list[int | None] = [None] * len(clean)
    matched_ious: list[float | None] = [None] * len(clean)
    for row, col in zip(rows, cols):
        overlap = float(overlaps[row, col])
        if overlap >= minimum_iou:
            mapping[int(row)] = int(col)
            matched_ious[int(row)] = overlap
    return mapping, matched_ious


def evaluate_probe_outcome(
    clean: Sequence[Candidate],
    perturbed_raw: Sequence[Candidate],
    contract: OutputContract,
) -> tuple[ProbeOutcome, list[Candidate]]:
    """Evaluate the coverage-aware candidate-order event for one probe.

    Coverage requires every tracked clean candidate to be associated.  It also
    requires that no spatially novel, unmatched candidate has a score at least
    as large as the matched clean winner.  Candidate disappearance and an
    unresolved high-score birth are therefore never converted into arbitrary
    numeric scores.
    """

    if len(clean) < 2:
        raise ValueError("clean must expose at least two tracked candidates")

    perturbed = select_spatially_distinct(
        perturbed_raw,
        maximum=contract.exposed_candidate_count,
        duplicate_iou_threshold=contract.duplicate_iou_threshold,
    )
    mapping, matched_ious = _associate_candidates(
        clean, perturbed, contract.match_iou_threshold
    )
    missing = [index for index, target in enumerate(mapping) if target is None]
    used = {target for target in mapping if target is not None}

    matched_scores: list[float | None] = [
        None if target is None else float(perturbed[target].score)
        for target in mapping
    ]
    matched_boxes: list[list[float] | None] = [
        None
        if target is None
        else [float(value) for value in perturbed[target].box]
        for target in mapping
    ]

    threatening_births: list[int] = []
    birth_scores: list[float] = []
    if mapping[0] is not None:
        winner_score = float(perturbed[mapping[0]].score)
        matched_targets = [perturbed[target] for target in used]
        for index, candidate in enumerate(perturbed):
            if index in used or candidate.score < winner_score:
                continue
            is_spatially_novel = all(
                iou_xyxy(candidate.box, matched.box)
                < contract.birth_duplicate_iou_threshold
                for matched in matched_targets
            )
            if is_spatially_novel:
                threatening_births.append(index)
                birth_scores.append(float(candidate.score))

    coverage = int(not missing and not threatening_births)
    rank_stable: int | None = None
    operational_stable = 0
    culprit: int | None = None
    gaps: list[float] | None = None

    if coverage:
        score_array = np.asarray(matched_scores, dtype=float)
        gap_array = score_array[0] - score_array[1:]
        gaps = gap_array.tolist()
        rank_stable = int(np.all(gap_array > 0.0))
        operational_stable = rank_stable
        if not rank_stable:
            # A deterministic argmin makes culprit events mutually exclusive.
            culprit = int(np.argmin(gap_array) + 1)

    if coverage and rank_stable:
        primary_failure = "stable"
    elif 0 in missing:
        primary_failure = "winner_missing"
    elif missing:
        primary_failure = "competitor_missing"
    elif threatening_births:
        primary_failure = "threatening_birth"
    else:
        primary_failure = "ranking_reversal"

    return (
        ProbeOutcome(
            coverage=coverage,
            rank_stable=rank_stable,
            operational_stable=operational_stable,
            primary_failure=primary_failure,
            culprit_clean_index=culprit,
            mapping=mapping,
            matched_ious=matched_ious,
            matched_scores=matched_scores,
            matched_boxes=matched_boxes,
            gaps=gaps,
            missing_clean_indices=missing,
            threatening_birth_indices=threatening_births,
            threatening_birth_scores=birth_scores,
            perturbed_candidate_count=len(perturbed),
        ),
        perturbed,
    )


def exact_binomial_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> BinomialEstimate:
    """Two-sided Clopper-Pearson interval at a fixed probe budget."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(
            beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1)
        )
    )
    upper = (
        1.0
        if successes == trials
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0, successes + 1, trials - successes
            )
        )
    )
    return BinomialEstimate(
        successes=successes,
        trials=trials,
        estimate=float(successes / trials),
        lower=lower,
        upper=upper,
        confidence=confidence,
    )


def summarize_outcomes(
    outcomes: Sequence[ProbeOutcome],
    families: Sequence[str],
    confidence: float = 0.95,
    clean_eligible: int = 1,
) -> StabilityProfile:
    """Return the complete, auditable stability profile for one probe set."""

    if len(outcomes) != len(families) or not outcomes:
        raise ValueError("outcomes and families must be non-empty and aligned")

    n = len(outcomes)
    coverage_successes = int(sum(item.coverage for item in outcomes))
    operational_successes = int(sum(item.operational_stable for item in outcomes))
    ranking_successes = int(
        sum(item.rank_stable == 1 for item in outcomes if item.coverage)
    )
    coverage = exact_binomial_interval(coverage_successes, n, confidence)
    operational = exact_binomial_interval(operational_successes, n, confidence)
    conditional = (
        None
        if coverage_successes == 0
        else exact_binomial_interval(
            ranking_successes, coverage_successes, confidence
        )
    )

    theta_cov = coverage.estimate
    theta_rank = 0.0 if conditional is None else conditional.estimate
    theta_op = operational.estimate
    operational_risk = 1.0 - theta_op
    coverage_risk = 1.0 - theta_cov
    ranking_risk = theta_cov * (1.0 - theta_rank)

    family_trials: dict[str, int] = {}
    family_stable: dict[str, int] = {}
    for family, outcome in zip(families, outcomes):
        family_trials[family] = family_trials.get(family, 0) + 1
        family_stable[family] = (
            family_stable.get(family, 0) + outcome.operational_stable
        )
    family_theta = {
        family: family_stable[family] / trials
        for family, trials in family_trials.items()
    }
    family_pi = {family: trials / n for family, trials in family_trials.items()}
    family_contribution = {
        family: family_pi[family] * (1.0 - family_theta[family])
        for family in family_trials
    }
    family_share = {
        family: (
            0.0
            if operational_risk == 0.0
            else contribution / operational_risk
        )
        for family, contribution in family_contribution.items()
    }

    failure_counts: dict[str, int] = {}
    culprit_counts: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.primary_failure != "stable":
            failure_counts[outcome.primary_failure] = (
                failure_counts.get(outcome.primary_failure, 0) + 1
            )
        if outcome.primary_failure == "ranking_reversal":
            key = str(outcome.culprit_clean_index)
            culprit_counts[key] = culprit_counts.get(key, 0) + 1
    total_failures = sum(failure_counts.values())
    ranking_failures = sum(culprit_counts.values())

    return StabilityProfile(
        clean_eligible=int(clean_eligible),
        probe_count=n,
        coverage=coverage,
        conditional_ranking=conditional,
        operational=operational,
        operational_risk=operational_risk,
        coverage_risk=coverage_risk,
        conditional_ranking_risk=ranking_risk,
        risk_decomposition_error=float(
            abs(operational_risk - coverage_risk - ranking_risk)
        ),
        family_trials=family_trials,
        family_operational_stability=family_theta,
        family_risk_contribution=family_contribution,
        family_risk_share=family_share,
        primary_failure_counts=failure_counts,
        primary_failure_shares={
            key: (0.0 if total_failures == 0 else count / total_failures)
            for key, count in failure_counts.items()
        },
        culprit_counts=culprit_counts,
        culprit_shares_among_ranking_failures={
            key: (0.0 if ranking_failures == 0 else count / ranking_failures)
            for key, count in culprit_counts.items()
        },
    )


def flatten_profile(prefix: str, profile: StabilityProfile) -> dict[str, float | int]:
    """Flatten primary statistics for a row-oriented result table."""

    conditional = profile.conditional_ranking
    return {
        f"{prefix}_probe_count": profile.probe_count,
        f"{prefix}_coverage": profile.coverage.estimate,
        f"{prefix}_coverage_lower": profile.coverage.lower,
        f"{prefix}_coverage_upper": profile.coverage.upper,
        f"{prefix}_conditional_ranking": (
            float("nan") if conditional is None else conditional.estimate
        ),
        f"{prefix}_conditional_ranking_lower": (
            float("nan") if conditional is None else conditional.lower
        ),
        f"{prefix}_conditional_ranking_upper": (
            float("nan") if conditional is None else conditional.upper
        ),
        f"{prefix}_operational": profile.operational.estimate,
        f"{prefix}_operational_lower": profile.operational.lower,
        f"{prefix}_operational_upper": profile.operational.upper,
        f"{prefix}_coverage_risk": profile.coverage_risk,
        f"{prefix}_ranking_risk": profile.conditional_ranking_risk,
        f"{prefix}_risk_decomposition_error": profile.risk_decomposition_error,
    }


def raw_candidate_payload(candidates: Sequence[Candidate]) -> list[dict]:
    """Compact JSON-compatible representation used by audit traces."""

    return [
        {
            "box": [float(value) for value in candidate.box],
            "score": float(candidate.score),
            "label": candidate.label,
        }
        for candidate in candidates
    ]


def family_risk_identity(profile: StabilityProfile) -> float:
    """Numerical residual of the law-of-total-probability decomposition."""

    return float(
        abs(
            profile.operational_risk
            - sum(profile.family_risk_contribution.values())
        )
    )


def aggregate_model_operational(
    sample_operational: Sequence[float],
    clean_eligible: Sequence[int],
) -> tuple[float, float, float]:
    """Return full-manifest, eligible-only and clean-eligibility averages.

    Ineligible clean outputs are assigned zero operational stability in the
    full-manifest statistic.  The eligible-only statistic is reported
    separately and must never replace it silently.
    """

    operational = np.asarray(sample_operational, dtype=float)
    eligible = np.asarray(clean_eligible, dtype=bool)
    if operational.ndim != 1 or eligible.shape != operational.shape:
        raise ValueError("sample_operational and clean_eligible must align")
    full = np.where(eligible, operational, 0.0)
    eligible_mean = float(np.mean(operational[eligible])) if eligible.any() else 0.0
    return float(full.mean()), eligible_mean, float(eligible.mean())
