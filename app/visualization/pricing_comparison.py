"""
Comparison and sensitivity plots (Sections 20, 29.5/10/11 of the spec):
Monte Carlo vs. Black-Scholes bar/error comparison, terminal price
distribution with strike/spot markers, and pricing error across
moneyness and volatility grids.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_terminal_price_distribution(
    terminal_prices: np.ndarray,
    spot: float,
    strike: float,
    ax: Optional[plt.Axes] = None,
    bins: int = 100,
    title: str = "Risk-Neutral Terminal Price Distribution",
) -> plt.Axes:
    """Histogram of S_T under the risk-neutral measure, marking S0, K, and percentiles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.hist(terminal_prices, bins=bins, density=True, alpha=0.6, color="#4C72B0")
    ax.axvline(spot, color="#C44E52", linestyle=":", linewidth=1.5, label=f"S0 = {spot:g}")
    ax.axvline(strike, color="black", linestyle="--", linewidth=1.5, label=f"K = {strike:g}")

    for p in (5, 95):
        ax.axvline(
            np.percentile(terminal_prices, p), color="grey", linestyle=":", linewidth=1,
            label=f"{p}th percentile",
        )

    ax.set_title(title)
    ax.set_xlabel("Terminal Price (S_T)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_mc_vs_bs_comparison(
    labels: Sequence[str],
    mc_prices: Sequence[float],
    bs_prices: Sequence[float],
    mc_errors: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Monte Carlo vs. Black-Scholes Price",
) -> plt.Axes:
    """Grouped bar chart comparing MC and BS prices across named scenarios.

    ``mc_errors``, if given, are used as symmetric error bars (e.g. one
    standard error or a half-CI-width) on the Monte Carlo bars.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 6))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, mc_prices, width, yerr=mc_errors, capsize=4, label="Monte Carlo", color="#4C72B0")
    ax.bar(x + width / 2, bs_prices, width, label="Black-Scholes", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_error_across_moneyness(
    moneyness_values: Sequence[float],
    percentage_errors: Sequence[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Pricing Error Across Moneyness (S0 / K)",
) -> plt.Axes:
    """Line plot of MC-vs-BS percentage error against moneyness = S0/K."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.plot(moneyness_values, percentage_errors, "o-", color="#55A868")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="ATM (S0/K = 1)")
    ax.axhline(0.0, color="grey", linewidth=0.8)

    ax.set_xlabel("Moneyness (S0 / K)")
    ax.set_ylabel("Percentage Error (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_error_across_volatility(
    volatility_values: Sequence[float],
    percentage_errors: Sequence[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Pricing Error Across Volatility",
) -> plt.Axes:
    """Line plot of MC-vs-BS percentage error against sigma."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.plot([v * 100 for v in volatility_values], percentage_errors, "o-", color="#8172B2")
    ax.axhline(0.0, color="grey", linewidth=0.8)

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Percentage Error (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
