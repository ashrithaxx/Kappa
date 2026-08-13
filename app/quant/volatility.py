"""
Historical volatility and return estimation.

Two kinds of returns show up in quantitative finance and they are
**not** interchangeable:

Arithmetic return
    R_t = (S_t - S_{t-1}) / S_{t-1}
    Simple percentage change. Additive across assets in a portfolio at
    a single point in time (portfolio return = weighted sum of asset
    arithmetic returns). Use for portfolio-level aggregation.

Log return
    r_t = ln(S_t / S_{t-1})
    Time-additive: the log return over N periods is the sum of the
    per-period log returns. This additivity is exactly what GBM is
    built on (log-prices are the object that follows a driftless-plus-
    drift random walk), so log returns are what we use to estimate the
    mu/sigma that feed the simulator.

This module uses **log returns** for volatility/drift estimation
(consistent with GBM's assumptions) and exposes arithmetic returns
separately for completeness/aggregation use cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VolatilityReport:
    """Structured result of a historical volatility estimation."""

    trading_days_per_year: int
    n_observations: int

    mean_daily_log_return: float
    std_daily_log_return: float
    annualized_mean_log_return: float
    annualized_volatility: float

    min_log_return: float
    max_log_return: float

    mean_arithmetic_return: float
    std_arithmetic_return: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            "Historical Volatility Report\n"
            "─────────────────────────────\n"
            f"Observations:            {self.n_observations}\n"
            f"Trading days / year:     {self.trading_days_per_year}\n"
            f"Mean daily log return:   {self.mean_daily_log_return:.6f}\n"
            f"Std daily log return:    {self.std_daily_log_return:.6f}\n"
            f"Annualized mean return:  {self.annualized_mean_log_return:.4%}\n"
            f"Annualized volatility:   {self.annualized_volatility:.4%}\n"
            f"Min / Max log return:    {self.min_log_return:.6f} / "
            f"{self.max_log_return:.6f}\n"
        )


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Continuously compounded (log) returns: r_t = ln(S_t / S_{t-1})."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be a 1-D array/sequence")
    if prices.size < 2:
        raise ValueError("need at least 2 price observations")
    if np.any(prices <= 0):
        raise ValueError("all prices must be strictly positive")
    return np.diff(np.log(prices))


def compute_arithmetic_returns(prices: np.ndarray) -> np.ndarray:
    """Simple returns: R_t = (S_t - S_{t-1}) / S_{t-1}."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be a 1-D array/sequence")
    if prices.size < 2:
        raise ValueError("need at least 2 price observations")
    if np.any(prices <= 0):
        raise ValueError("all prices must be strictly positive")
    return np.diff(prices) / prices[:-1]


def historical_volatility(
    prices: np.ndarray, trading_days_per_year: int = 252
) -> VolatilityReport:
    """Estimate annualized volatility and drift from a price history.

    Parameters
    ----------
    prices:
        1-D array of historical prices, ordered oldest → newest.
    trading_days_per_year:
        Annualization factor. Configurable — do not assume equities'
        conventional 252 for every asset class (crypto, for instance,
        is often annualized with 365).

    Returns
    -------
    VolatilityReport
        Daily and annualized statistics, using sample (n-1 denominator)
        standard deviation.
    """
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be > 0")

    log_returns = compute_log_returns(prices)
    arith_returns = compute_arithmetic_returns(prices)

    mean_daily = float(np.mean(log_returns))
    # ddof=1 -> sample standard deviation, sqrt(1/(n-1) * sum((r - mean)^2))
    std_daily = float(np.std(log_returns, ddof=1))

    annualized_vol = std_daily * np.sqrt(trading_days_per_year)
    # Annualizing the mean of log returns by simple scaling is the
    # standard convention; note this is the drift of ln(S), not a
    # simple-return CAGR.
    annualized_mean = mean_daily * trading_days_per_year

    return VolatilityReport(
        trading_days_per_year=trading_days_per_year,
        n_observations=int(log_returns.size),
        mean_daily_log_return=mean_daily,
        std_daily_log_return=std_daily,
        annualized_mean_log_return=annualized_mean,
        annualized_volatility=annualized_vol,
        min_log_return=float(np.min(log_returns)),
        max_log_return=float(np.max(log_returns)),
        mean_arithmetic_return=float(np.mean(arith_returns)),
        std_arithmetic_return=float(np.std(arith_returns, ddof=1)),
    )
