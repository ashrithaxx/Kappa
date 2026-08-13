"""
Monte Carlo vs. Black-Scholes comparison and pricing-error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from app.quant.derivatives.black_scholes import black_scholes_price
from app.quant.derivatives.monte_carlo_pricer import MonteCarloPriceResult


@dataclass(frozen=True)
class PricingComparison:
    """Structured Monte Carlo vs. Black-Scholes comparison for one option."""

    mc_price: float
    bs_price: float
    absolute_error: float
    signed_error: float
    percentage_error: float
    standard_error: float
    ci_level: float
    ci_lower: float
    ci_upper: float
    bs_inside_ci: bool

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"Monte Carlo Price:       ${self.mc_price:.4f}\n"
            f"Black-Scholes Price:     ${self.bs_price:.4f}\n"
            f"Absolute Error:          ${self.absolute_error:.4f}\n"
            f"Percentage Error:         {self.percentage_error:.4f}%\n"
            f"{self.ci_level:.0%} Monte Carlo CI:       "
            f"[${self.ci_lower:.4f}, ${self.ci_upper:.4f}]\n"
            f"Black-Scholes inside CI: {'YES' if self.bs_inside_ci else 'NO'}"
        )


def compare_to_black_scholes(
    mc_result: MonteCarloPriceResult, ci_level: float = 0.95
) -> PricingComparison:
    """Compare a Monte Carlo pricing result against the Black-Scholes price."""
    bs_price = black_scholes_price(mc_result.option)
    mc_price = mc_result.price

    signed_error = mc_price - bs_price
    absolute_error = abs(signed_error)
    percentage_error = (signed_error / bs_price * 100.0) if bs_price != 0 else float("nan")

    ci_key = round(ci_level * 100)
    ci = mc_result.confidence_intervals.get(ci_key)
    if ci is None:
        raise ValueError(
            f"No confidence interval was computed at level {ci_level:.0%}; "
            f"available levels: {sorted(mc_result.confidence_intervals)}"
        )

    bs_inside_ci = ci.lower <= bs_price <= ci.upper

    return PricingComparison(
        mc_price=mc_price,
        bs_price=bs_price,
        absolute_error=absolute_error,
        signed_error=signed_error,
        percentage_error=percentage_error,
        standard_error=mc_result.standard_error,
        ci_level=ci_level,
        ci_lower=ci.lower,
        ci_upper=ci.upper,
        bs_inside_ci=bool(bs_inside_ci),
    )


def root_mean_squared_error(
    mc_prices: Sequence[float], bs_prices: Sequence[float]
) -> float:
    """RMSE across a set of (Monte Carlo, Black-Scholes) price pairs.

    Useful when comparing several parameter configurations (e.g. a
    moneyness sweep or a volatility sweep) rather than a single option.
    """
    mc = np.asarray(mc_prices, dtype=float)
    bs = np.asarray(bs_prices, dtype=float)
    if mc.shape != bs.shape:
        raise ValueError("mc_prices and bs_prices must have the same shape")
    if mc.size == 0:
        raise ValueError("need at least one price pair")
    return float(np.sqrt(np.mean((mc - bs) ** 2)))
