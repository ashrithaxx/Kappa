"""
Plotting utilities for the terminal price distribution (Section 13.B/C).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from app.quant.distributions import TheoreticalGBMDistribution
from app.quant.statistics import DescriptiveStatistics


def plot_terminal_histogram(
    terminal_prices: np.ndarray,
    stats_report: DescriptiveStatistics,
    ax: Optional[plt.Axes] = None,
    bins: int = 100,
    title: str = "Terminal Price Distribution",
) -> plt.Axes:
    """Histogram of S_T annotated with mean, median, and key percentiles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.hist(terminal_prices, bins=bins, density=True, alpha=0.6, color="#4C72B0")
    ax.axvline(stats_report.mean, color="black", linestyle="-", label="Mean")
    ax.axvline(stats_report.median, color="black", linestyle="--", label="Median")

    for p in (5, 95):
        if p in stats_report.percentiles:
            ax.axvline(
                stats_report.percentiles[p],
                color="grey",
                linestyle=":",
                label=f"{p}th percentile",
            )

    ax.set_title(title)
    ax.set_xlabel("Terminal Price (S_T)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_log_price_distribution(
    terminal_prices: np.ndarray,
    theoretical: TheoreticalGBMDistribution,
    ax: Optional[plt.Axes] = None,
    bins: int = 100,
    title: str = "Log-Price Distribution vs. Theoretical Normal",
) -> plt.Axes:
    """Overlay the empirical ln(S_T) histogram with the theoretical Normal PDF."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    log_prices = np.log(terminal_prices)
    ax.hist(
        log_prices,
        bins=bins,
        density=True,
        alpha=0.6,
        color="#55A868",
        label="Simulated ln(S_T)",
    )

    x = np.linspace(log_prices.min(), log_prices.max(), 500)
    pdf = scipy_stats.norm.pdf(
        x, loc=theoretical.log_price_mean, scale=theoretical.log_price_std
    )
    ax.plot(x, pdf, color="black", linewidth=2, label="Theoretical Normal")

    ax.set_title(title)
    ax.set_xlabel("ln(S_T)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
