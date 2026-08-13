"""
Descriptive statistics and confidence intervals for simulated data.

Important distinction (see also module docstring in monte_carlo.py):
the confidence interval computed here is uncertainty in the Monte
Carlo *estimator* of the mean (it shrinks as simulations -> infinity).
It is NOT a statement about the range the future price is likely to
fall in — that is instead read off the percentiles of the terminal
distribution itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy import stats

_Z_SCORES = {90: 1.6448536269514722, 95: 1.9599639845400545, 99: 2.5758293035489004}

_DEFAULT_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


@dataclass(frozen=True)
class ConfidenceInterval:
    level: float  # e.g. 0.95
    point_estimate: float
    lower: float
    upper: float
    standard_error: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.level:.0%} CI: [{self.lower:.4f}, {self.upper:.4f}] "
            f"(point estimate {self.point_estimate:.4f}, SE {self.standard_error:.6f})"
        )


@dataclass(frozen=True)
class DescriptiveStatistics:
    """Structured descriptive-statistics result for a 1-D sample."""

    n: int
    mean: float
    median: float
    variance: float
    std: float
    skewness: float
    kurtosis: float  # excess kurtosis (Normal -> 0)
    minimum: float
    maximum: float
    standard_error: float
    percentiles: Dict[int, float] = field(default_factory=dict)
    confidence_intervals: Dict[int, ConfidenceInterval] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display only
        lines = [
            "Descriptive Statistics",
            "───────────────────────",
            f"n:          {self.n}",
            f"Mean:       {self.mean:.6f}",
            f"Median:     {self.median:.6f}",
            f"Std Dev:    {self.std:.6f}",
            f"Variance:   {self.variance:.6f}",
            f"Skewness:   {self.skewness:.6f}",
            f"Kurtosis:   {self.kurtosis:.6f} (excess)",
            f"Min / Max:  {self.minimum:.6f} / {self.maximum:.6f}",
            f"Std Error:  {self.standard_error:.6f}",
            "Percentiles:",
        ]
        for p, v in sorted(self.percentiles.items()):
            lines.append(f"  {p:>3}%: {v:.6f}")
        return "\n".join(lines)


def standard_error(data: np.ndarray) -> float:
    """SE = s / sqrt(n), the standard error of the sample mean."""
    data = np.asarray(data, dtype=float)
    return float(np.std(data, ddof=1) / np.sqrt(data.size))


def confidence_interval(data: np.ndarray, level: float = 0.95) -> ConfidenceInterval:
    """Confidence interval for the *mean* of ``data`` at the given level.

    CI = x_bar +/- z * (s / sqrt(n))

    Supports any level in (0, 1); uses the exact normal quantile via
    scipy for levels outside the common 90/95/99 fast-path table.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    data = np.asarray(data, dtype=float)
    n = data.size
    mean = float(np.mean(data))
    se = standard_error(data)

    level_pct = round(level * 100)
    if level_pct in _Z_SCORES:
        z = _Z_SCORES[level_pct]
    else:
        z = float(stats.norm.ppf(1 - (1 - level) / 2))

    margin = z * se
    return ConfidenceInterval(
        level=level,
        point_estimate=mean,
        lower=mean - margin,
        upper=mean + margin,
        standard_error=se,
    )


def descriptive_statistics(
    data: np.ndarray,
    percentiles=_DEFAULT_PERCENTILES,
    ci_levels=(0.90, 0.95, 0.99),
) -> DescriptiveStatistics:
    """Full descriptive-statistics report for a 1-D sample (e.g. S_T)."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("data must be 1-D")
    if data.size < 2:
        raise ValueError("need at least 2 observations")

    pct_values = {int(p): float(np.percentile(data, p)) for p in percentiles}
    ci_values = {
        round(level * 100): confidence_interval(data, level) for level in ci_levels
    }

    return DescriptiveStatistics(
        n=int(data.size),
        mean=float(np.mean(data)),
        median=float(np.median(data)),
        variance=float(np.var(data, ddof=1)),
        std=float(np.std(data, ddof=1)),
        skewness=float(stats.skew(data)),
        kurtosis=float(stats.kurtosis(data)),  # Fisher (excess) kurtosis
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        standard_error=standard_error(data),
        percentiles=pct_values,
        confidence_intervals=ci_values,
    )
