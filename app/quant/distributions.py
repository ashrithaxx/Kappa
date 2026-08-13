"""
Theoretical properties of the GBM terminal-price distribution.

Under
    dS_t = mu S_t dt + sigma S_t dW_t

the log-price is exactly Normal:

    ln(S_T) ~ Normal( ln(S0) + (mu - 0.5*sigma^2) T,  sigma^2 * T )

so S_T is Lognormal. This module provides the closed-form moments so
simulated statistics can be checked against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TheoreticalGBMDistribution:
    """Closed-form moments of ln(S_T) and S_T under GBM."""

    log_price_mean: float  # E[ln(S_T)]
    log_price_variance: float  # Var[ln(S_T)] = sigma^2 * T
    log_price_std: float

    price_mean: float  # E[S_T] = S0 * exp(mu*T)
    price_variance: float  # Var[S_T]
    price_std: float

    def price_percentile(self, q: float) -> float:
        """Theoretical q-th percentile (0 < q < 100) of S_T."""
        if not 0 < q < 100:
            raise ValueError("q must be in (0, 100)")
        z = stats.norm.ppf(q / 100.0)
        log_price = self.log_price_mean + z * self.log_price_std
        return float(np.exp(log_price))


def theoretical_gbm_distribution(
    initial_price: float, drift: float, volatility: float, time_horizon: float
) -> TheoreticalGBMDistribution:
    """Closed-form GBM terminal distribution moments.

    E[S_T]   = S0 * exp(mu * T)
    Var[S_T] = S0^2 * exp(2*mu*T) * (exp(sigma^2 * T) - 1)
    """
    if initial_price <= 0:
        raise ValueError("initial_price must be > 0")
    if volatility < 0:
        raise ValueError("volatility must be >= 0")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be > 0")

    log_mean = np.log(initial_price) + (drift - 0.5 * volatility**2) * time_horizon
    log_var = volatility**2 * time_horizon

    price_mean = initial_price * np.exp(drift * time_horizon)
    price_var = (
        initial_price**2
        * np.exp(2 * drift * time_horizon)
        * (np.exp(volatility**2 * time_horizon) - 1)
    )

    return TheoreticalGBMDistribution(
        log_price_mean=float(log_mean),
        log_price_variance=float(log_var),
        log_price_std=float(np.sqrt(log_var)),
        price_mean=float(price_mean),
        price_variance=float(price_var),
        price_std=float(np.sqrt(price_var)),
    )


def percentage_error(simulated: float, theoretical: float) -> float:
    """(simulated - theoretical) / theoretical * 100.

    Returns NaN if theoretical is exactly zero (undefined relative error).
    """
    if theoretical == 0:
        return float("nan")
    return (simulated - theoretical) / theoretical * 100.0
