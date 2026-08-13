"""
Plotting utilities for portfolio P&L, risk metrics, and stress tests.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from app.quant.portfolio.risk_metrics import RiskMetricResult
from app.quant.portfolio.stress_testing import StressTestResult


def plot_pnl_distribution(
    pnl: np.ndarray,
    risk: RiskMetricResult,
    ax: Optional[plt.Axes] = None,
    bins: int = 100,
    title: str = "Portfolio P&L Distribution",
) -> plt.Axes:
    """Histogram of simulated P&L with VaR and ES thresholds marked.

    VaR/ES are stored as positive loss numbers (see ``risk_metrics.py``
    docstring), so they are negated back onto the P&L axis here.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.hist(pnl, bins=bins, density=True, alpha=0.6, color="#4C72B0")
    ax.axvline(-risk.var, color="darkorange", linewidth=2, label=f"VaR ({risk.confidence_level:.0%})")
    ax.axvline(
        -risk.expected_shortfall,
        color="crimson",
        linewidth=2,
        linestyle="--",
        label=f"ES ({risk.confidence_level:.0%})",
    )
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_correlation_heatmap(
    correlation_matrix: np.ndarray,
    asset_names,
    ax: Optional[plt.Axes] = None,
    title: str = "Asset Correlation Matrix",
) -> plt.Axes:
    """Heatmap of the (target or realized) correlation matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(asset_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(asset_names, rotation=45, ha="right")
    ax.set_yticklabels(asset_names)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{correlation_matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
            )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_var_by_confidence(
    pnl: np.ndarray,
    confidence_levels=(0.90, 0.95, 0.99),
    ax: Optional[plt.Axes] = None,
    title: str = "VaR and ES Across Confidence Levels",
) -> plt.Axes:
    """Bar chart of historical VaR/ES at several confidence levels, side by side."""
    from app.quant.portfolio.risk_metrics import (
        historical_expected_shortfall,
        historical_var,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    var_values = [historical_var(pnl, c) for c in confidence_levels]
    es_values = [historical_expected_shortfall(pnl, c) for c in confidence_levels]

    x = np.arange(len(confidence_levels))
    width = 0.35
    ax.bar(x - width / 2, var_values, width, label="VaR", color="darkorange")
    ax.bar(x + width / 2, es_values, width, label="ES", color="crimson")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.0%}" for c in confidence_levels])
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Loss ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_stress_comparison(
    result: StressTestResult,
    ax: Optional[plt.Axes] = None,
    bins: int = 80,
) -> plt.Axes:
    """Overlay baseline vs. stressed terminal-value distributions for one scenario."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        result.baseline.terminal_value, bins=bins, density=True, alpha=0.5,
        color="#4C72B0", label="Baseline",
    )
    ax.hist(
        result.stressed.terminal_value, bins=bins, density=True, alpha=0.5,
        color="crimson", label="Stressed",
    )
    ax.axvline(result.baseline.initial_value, color="black", linewidth=1, linestyle=":", label="Initial Value")

    ax.set_title(f"Stress Test: {result.scenario.name}")
    ax.set_xlabel("Terminal Portfolio Value ($)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
