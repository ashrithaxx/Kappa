"""
Value at Risk (VaR) and Expected Shortfall (ES).

Both are computed two ways from the same simulated P&L sample:

- **Historical / simulation-based**: read the relevant percentile (VaR)
  or tail mean (ES) directly off the empirical P&L distribution
  produced by ``portfolio_simulation.py``. Makes no distributional
  assumption beyond what the Monte Carlo engine already assumes (GBM);
  this is the primary method the platform uses.
- **Parametric (variance-covariance)**: closed-form Normal-distribution
  formulas, included as an independent analytical benchmark — the same
  role Black-Scholes played for Monte Carlo option pricing in Week 2.
  Only valid to the extent P&L is approximately Normal, which
  Monte-Carlo-simulated log-normal terminal prices are not exactly, so
  the two methods are expected to diverge, especially in the tail and
  at high confidence levels — the divergence itself is informative
  (fat tails / skew that the parametric method misses).

Sign convention: VaR and ES are reported as **positive numbers**
representing a loss (i.e. VaR = -1 * the alpha-quantile of P&L), the
standard risk-reporting convention ("99% 1-day VaR is $2.3M" means a
loss, not a gain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RiskMetricResult:
    confidence_level: float  # e.g. 0.95
    var: float  # positive = loss
    expected_shortfall: float  # positive = loss, ES >= VaR always
    method: str  # "historical" or "parametric"

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.confidence_level:.0%} VaR ({self.method}):  "
            f"${self.var:,.2f}\n"
            f"{self.confidence_level:.0%} ES  ({self.method}):  "
            f"${self.expected_shortfall:,.2f}"
        )


def historical_var(pnl: np.ndarray, confidence_level: float = 0.95) -> float:
    """Empirical VaR: the loss at the (1 - confidence_level) quantile of P&L.

    E.g. at 95% confidence, VaR is the loss such that only 5% of
    simulated outcomes are worse.
    """
    _validate_confidence(confidence_level)
    pnl = np.asarray(pnl, dtype=float)
    alpha = 1 - confidence_level
    quantile = np.percentile(pnl, alpha * 100)
    return float(-quantile)


def historical_expected_shortfall(
    pnl: np.ndarray, confidence_level: float = 0.95
) -> float:
    """Empirical ES (a.k.a. CVaR/Tail VaR): mean loss *beyond* the VaR threshold.

    Averages every simulated outcome worse than the VaR quantile, so it
    captures tail severity that a single quantile (VaR) does not —
    two portfolios can share the same VaR while one has a far worse
    average loss beyond it.
    """
    _validate_confidence(confidence_level)
    pnl = np.asarray(pnl, dtype=float)
    alpha = 1 - confidence_level
    threshold = np.percentile(pnl, alpha * 100)
    tail = pnl[pnl <= threshold]
    if tail.size == 0:
        # Degenerate case (alpha so small relative to sample size that
        # no observation falls at/below the exact quantile boundary);
        # fall back to the single worst observation.
        tail = np.array([np.min(pnl)])
    return float(-np.mean(tail))


def historical_risk_metrics(
    pnl: np.ndarray, confidence_level: float = 0.95
) -> RiskMetricResult:
    """Convenience wrapper bundling historical VaR and ES at one confidence level."""
    return RiskMetricResult(
        confidence_level=confidence_level,
        var=historical_var(pnl, confidence_level),
        expected_shortfall=historical_expected_shortfall(pnl, confidence_level),
        method="historical",
    )


def parametric_var(
    pnl: np.ndarray, confidence_level: float = 0.95
) -> float:
    """Variance-covariance VaR assuming P&L ~ Normal(mean, std).

    VaR = -(mean + z_alpha * std), z_alpha = Phi^-1(1 - confidence_level)
    (z_alpha is negative for confidence_level > 50%, making VaR positive
    for a typical portfolio with mean losses far from the tail).
    """
    _validate_confidence(confidence_level)
    pnl = np.asarray(pnl, dtype=float)
    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))
    z_alpha = stats.norm.ppf(1 - confidence_level)
    return float(-(mean + z_alpha * std))


def parametric_expected_shortfall(
    pnl: np.ndarray, confidence_level: float = 0.95
) -> float:
    """Closed-form Normal ES: mean + std * phi(z_alpha) / (1 - confidence_level).

    Uses the standard analytical tail-expectation formula for a Normal
    distribution, where ``phi`` is the standard normal PDF and
    ``z_alpha = Phi^-1(1 - confidence_level)``.
    """
    _validate_confidence(confidence_level)
    pnl = np.asarray(pnl, dtype=float)
    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))
    alpha = 1 - confidence_level
    z_alpha = stats.norm.ppf(alpha)
    tail_mean = mean - std * stats.norm.pdf(z_alpha) / alpha
    return float(-tail_mean)


def parametric_risk_metrics(
    pnl: np.ndarray, confidence_level: float = 0.95
) -> RiskMetricResult:
    """Convenience wrapper bundling parametric VaR and ES at one confidence level."""
    return RiskMetricResult(
        confidence_level=confidence_level,
        var=parametric_var(pnl, confidence_level),
        expected_shortfall=parametric_expected_shortfall(pnl, confidence_level),
        method="parametric",
    )


def _validate_confidence(confidence_level: float) -> None:
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )
