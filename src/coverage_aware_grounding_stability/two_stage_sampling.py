"""Two-stage sampling theory for model-level operational stability.

The sampling unit is an image-query pair. For each sampled unit, repeated
registered probes produce a binary full-manifest operational event. The module
contains estimators and design quantities used by the accompanying analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VarianceComponents:
    """Estimated between-unit and within-unit variance components."""

    theta_hat: float
    between_raw: float
    between: float
    within: float
    cluster_mean_variance: float
    probe_count: int
    unit_count: int

    @property
    def estimated_variance(self) -> float:
        """Estimated variance of the observed balanced-design grand mean."""

        return self.between / self.unit_count + self.within / (
            self.unit_count * self.probe_count
        )


def validate_binary_matrix(outcomes: np.ndarray) -> np.ndarray:
    """Return a validated two-dimensional binary outcome matrix."""

    matrix = np.asarray(outcomes, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("outcomes must be a two-dimensional matrix")
    if matrix.shape[0] < 2:
        raise ValueError("at least two sampling units are required")
    if matrix.shape[1] < 2:
        raise ValueError("at least two probes per unit are required")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("outcomes contain non-finite values")
    if not np.all((matrix == 0.0) | (matrix == 1.0)):
        raise ValueError("outcomes must be binary")
    return matrix


def estimate_variance_components(outcomes: np.ndarray) -> VarianceComponents:
    """Estimate the exact two-stage variance components for a balanced design.

    If ``W_nr`` is Bernoulli conditional on sampling unit ``X_n``, define

    ``A = Var_X(E[W | X])`` and ``B = E_X[Var(W | X)]``.

    The average within-unit sample variance is unbiased for ``B``. The sample
    variance of unit means estimates ``A + B/R``; subtracting ``B/R`` therefore
    gives an unbiased method-of-moments estimator of ``A``. Sampling noise can
    make the raw estimate negative, so the operational estimate is truncated at
    zero while both values are retained for auditability.
    """

    matrix = validate_binary_matrix(outcomes)
    unit_count, probe_count = matrix.shape
    unit_means = matrix.mean(axis=1)
    within_variances = matrix.var(axis=1, ddof=1)
    within = float(within_variances.mean())
    cluster_mean_variance = float(unit_means.var(ddof=1))
    between_raw = cluster_mean_variance - within / probe_count
    between = max(0.0, float(between_raw))
    return VarianceComponents(
        theta_hat=float(unit_means.mean()),
        between_raw=float(between_raw),
        between=between,
        within=within,
        cluster_mean_variance=cluster_mean_variance,
        probe_count=probe_count,
        unit_count=unit_count,
    )


def variance_of_mean(between: float, within: float, unit_count: int, probe_count: int) -> float:
    """Return ``A/N + B/(N R)`` for a balanced two-stage design."""

    if between < 0 or within < 0:
        raise ValueError("variance components must be non-negative")
    if unit_count <= 0 or probe_count <= 0:
        raise ValueError("unit_count and probe_count must be positive")
    return float(between / unit_count + within / (unit_count * probe_count))


def probe_variance_share(between: float, within: float, probe_count: int) -> float:
    """Fraction of cluster-mean variance caused by finite probes."""

    total = between + within / probe_count
    if total == 0:
        return 0.0
    return float((within / probe_count) / total)


def intraclass_correlation(between: float, within: float) -> float:
    """Return the within-unit intraclass correlation ``A / (A + B)``."""

    if between < 0 or within < 0:
        raise ValueError("variance components must be non-negative")
    total = between + within
    return float(between / total) if total > 0 else 0.0


def design_effect(between: float, within: float, probe_count: int) -> float:
    """Return the repeated-probe design effect ``1 + (R - 1) rho``."""

    if probe_count <= 0:
        raise ValueError("probe_count must be positive")
    rho = intraclass_correlation(between, within)
    return float(1.0 + (probe_count - 1) * rho)


def effective_independent_trials(
    between: float, within: float, unit_count: int, probe_count: int
) -> float:
    """Equivalent number of independent Bernoulli trials under clustering."""

    if unit_count <= 0:
        raise ValueError("unit_count must be positive")
    return float(unit_count * probe_count / design_effect(between, within, probe_count))


def crossover_probe_count(between: float, within: float) -> float:
    """Return ``B/A``, where image and finite-probe variance contributions match."""

    if between < 0 or within < 0:
        raise ValueError("variance components must be non-negative")
    if between == 0:
        return math.inf if within > 0 else 0.0
    return float(within / between)


def probes_for_variance_share(between: float, within: float, maximum_share: float) -> int:
    """Minimum integer R making the finite-probe share no larger than ``maximum_share``."""

    if not 0 < maximum_share < 1:
        raise ValueError("maximum_share must lie strictly between zero and one")
    if between < 0 or within < 0:
        raise ValueError("variance components must be non-negative")
    if within == 0:
        return 1
    if between == 0:
        return math.inf
    threshold = within * (1.0 - maximum_share) / (between * maximum_share)
    return max(1, int(math.ceil(threshold)))


def required_unit_count(
    between: float,
    within: float,
    probe_count: int,
    half_width: float,
    z_value: float = 1.959963984540054,
) -> int:
    """Normal-approximation sample size for a target confidence half-width."""

    if half_width <= 0:
        raise ValueError("half_width must be positive")
    variance_per_unit = variance_of_mean(between, within, 1, probe_count)
    return max(1, int(math.ceil(z_value**2 * variance_per_unit / half_width**2)))


def optimal_probe_count(
    between: float,
    within: float,
    unit_cost: float,
    probe_cost: float,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Cost-optimal balanced integer probe count.

    For budget ``C = N(c_X + c_U R)``, the continuous optimum is
    ``sqrt(B c_X / (A c_U))``. The returned integer minimises the exact design
    variance over the feasible neighbouring integers and optional boundaries.
    """

    if between < 0 or within < 0:
        raise ValueError("variance components must be non-negative")
    if unit_cost <= 0 or probe_cost <= 0:
        raise ValueError("costs must be positive")
    if minimum < 1 or (maximum is not None and maximum < minimum):
        raise ValueError("invalid probe-count bounds")
    if between == 0:
        candidate = maximum if maximum is not None else minimum
        return int(candidate)
    continuous = math.sqrt(within * unit_cost / (between * probe_cost))
    candidates = {minimum, max(minimum, int(math.floor(continuous))), max(minimum, int(math.ceil(continuous)))}
    if maximum is not None:
        candidates = {min(maximum, value) for value in candidates}
        candidates.add(maximum)

    def objective(probe_count: int) -> float:
        return (unit_cost + probe_cost * probe_count) * (
            between + within / probe_count
        )

    return int(min(candidates, key=lambda value: (objective(value), value)))


def two_stage_hoeffding_radius(unit_count: int, probe_count: int, delta: float) -> float:
    """Conservative two-stage confidence radius from a union of Hoeffding bounds."""

    if unit_count <= 0 or probe_count <= 0:
        raise ValueError("unit_count and probe_count must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    common = math.log(4.0 / delta) / 2.0
    return float(math.sqrt(common / unit_count) + math.sqrt(common / (unit_count * probe_count)))
