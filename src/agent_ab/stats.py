"""
Statistical significance tests for A/B experiments.

All implemented in pure Python — no scipy, no numpy.
"""

from __future__ import annotations
import math
from typing import Tuple


def _normal_cdf(z: float) -> float:
    """Approximation of the standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_quantile(p: float) -> float:
    """Rational approximation of the inverse normal CDF (Beasley-Springer-Moro)."""
    # Abramowitz & Stegun approximation, accurate to ~1e-4
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        sign = -1.0
        p_adj = p
    else:
        sign = 1.0
        p_adj = 1.0 - p
    t = math.sqrt(-2.0 * math.log(p_adj))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - numerator / denominator)


def proportion_z_test(
    n_a: int,
    wins_a: int,
    n_b: int,
    wins_b: int,
) -> Tuple[float, float]:
    """Two-proportion z-test.

    Args:
        n_a: Total observations for variant A.
        wins_a: Successes for variant A.
        n_b: Total observations for variant B.
        wins_b: Successes for variant B.

    Returns:
        (z_score, p_value) — two-tailed p-value.
    """
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a = wins_a / n_a
    p_b = wins_b / n_b
    p_pool = (wins_a + wins_b) / (n_a + n_b)
    denominator = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if denominator == 0.0:
        return 0.0, 1.0
    z = (p_a - p_b) / denominator
    # Two-tailed p-value
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return z, p_value


def chi_square_test(
    n_a: int,
    wins_a: int,
    n_b: int,
    wins_b: int,
) -> Tuple[float, float]:
    """2x2 chi-square test of independence.

    Returns:
        (chi2_stat, p_value)
    """
    losses_a = n_a - wins_a
    losses_b = n_b - wins_b
    total = n_a + n_b
    if total == 0:
        return 0.0, 1.0

    # Expected frequencies
    row_wins = wins_a + wins_b
    row_losses = losses_a + losses_b

    def _expected(row_total: int, col_total: int) -> float:
        return row_total * col_total / total

    e_wa = _expected(row_wins, n_a)
    e_wb = _expected(row_wins, n_b)
    e_la = _expected(row_losses, n_a)
    e_lb = _expected(row_losses, n_b)

    chi2 = 0.0
    for obs, exp in [(wins_a, e_wa), (wins_b, e_wb), (losses_a, e_la), (losses_b, e_lb)]:
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp

    # p-value via chi-square CDF approximation for df=1
    # Use the relationship: chi2(df=1) ~ Z^2, so p = 2*(1 - Phi(sqrt(chi2)))
    p_value = 2.0 * (1.0 - _normal_cdf(math.sqrt(chi2))) if chi2 > 0 else 1.0
    return chi2, p_value


def confidence_interval(
    n: int,
    wins: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Args:
        n: Total observations.
        wins: Successes.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    alpha = 1.0 - confidence
    z = abs(_normal_quantile(alpha / 2.0))
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def is_significant(
    n_a: int,
    wins_a: int,
    n_b: int,
    wins_b: int,
    alpha: float = 0.05,
    min_samples: int = 30,
) -> bool:
    """Return True if the difference between A and B is statistically significant.

    Requires at least min_samples observations per variant.
    """
    if n_a < min_samples or n_b < min_samples:
        return False
    _, p_value = proportion_z_test(n_a, wins_a, n_b, wins_b)
    return p_value < alpha
