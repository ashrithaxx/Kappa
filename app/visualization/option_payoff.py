"""
Payoff diagrams for European options (Section 19 of the spec).

Deliberately independent of the pricing engine: these plots take a
strike and a range of terminal prices, nothing else. Pricing (Monte
Carlo or Black-Scholes) never has to run for a payoff diagram to be
drawn — keeping the two concerns separate matches this module's
placement outside ``derivatives/`` in the architecture.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.quant.derivatives.payoffs import call_payoff, put_payoff


def _price_range(strike: float, spot: Optional[float], span: float) -> np.ndarray:
    center = spot if spot is not None else strike
    lo = max(0.0, center * (1 - span))
    hi = center * (1 + span)
    return np.linspace(lo, hi, 400)


def plot_call_payoff(
    strike: float,
    spot: Optional[float] = None,
    span: float = 1.0,
    ax: Optional[plt.Axes] = None,
    title: str = "European Call Payoff at Maturity",
) -> plt.Axes:
    """Plot max(S_T - K, 0) against S_T, marking strike and break-even."""
    s_t = _price_range(strike, spot, span)
    payoff = call_payoff(s_t, strike)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.plot(s_t, payoff, color="#4C72B0", linewidth=2, label="Payoff")
    ax.axvline(strike, color="black", linestyle="--", linewidth=1, label=f"Strike K={strike:g}")
    ax.fill_between(s_t, 0, payoff, where=(s_t > strike), alpha=0.15, color="#4C72B0")
    ax.axhline(0, color="grey", linewidth=0.8)

    if spot is not None:
        ax.axvline(spot, color="#C44E52", linestyle=":", linewidth=1.2, label=f"Spot S0={spot:g}")

    ax.set_title(title)
    ax.set_xlabel("Terminal Price (S_T)")
    ax.set_ylabel("Payoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_put_payoff(
    strike: float,
    spot: Optional[float] = None,
    span: float = 1.0,
    ax: Optional[plt.Axes] = None,
    title: str = "European Put Payoff at Maturity",
) -> plt.Axes:
    """Plot max(K - S_T, 0) against S_T, marking strike and break-even."""
    s_t = _price_range(strike, spot, span)
    payoff = put_payoff(s_t, strike)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.plot(s_t, payoff, color="#55A868", linewidth=2, label="Payoff")
    ax.axvline(strike, color="black", linestyle="--", linewidth=1, label=f"Strike K={strike:g}")
    ax.fill_between(s_t, 0, payoff, where=(s_t < strike), alpha=0.15, color="#55A868")
    ax.axhline(0, color="grey", linewidth=0.8)

    if spot is not None:
        ax.axvline(spot, color="#C44E52", linestyle=":", linewidth=1.2, label=f"Spot S0={spot:g}")

    ax.set_title(title)
    ax.set_xlabel("Terminal Price (S_T)")
    ax.set_ylabel("Payoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_payoff_distribution(
    raw_payoffs: np.ndarray,
    discounted_mean: float,
    ax: Optional[plt.Axes] = None,
    bins: int = 100,
    title: str = "Simulated Payoff Distribution",
) -> plt.Axes:
    """Histogram of raw (undiscounted) simulated payoffs.

    Highlights why this distribution is asymmetric: a large point mass
    sits exactly at zero (every OTM path), while ITM paths spread out
    over a long right tail — very different from the terminal *price*
    distribution's shape, since the payoff floors at zero.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.hist(raw_payoffs, bins=bins, color="#8172B2", alpha=0.7)
    mean_payoff = float(np.mean(raw_payoffs))
    ax.axvline(mean_payoff, color="black", linestyle="-", label=f"Mean payoff = {mean_payoff:.2f}")
    ax.axvline(
        discounted_mean,
        color="black",
        linestyle="--",
        label=f"Discounted mean (price) = {discounted_mean:.2f}",
    )

    zero_fraction = float(np.mean(raw_payoffs == 0))
    ax.set_title(f"{title}  ({zero_fraction:.1%} of paths expire worthless)")
    ax.set_xlabel("Payoff")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
