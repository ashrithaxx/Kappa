"""
Plotting utilities for Monte Carlo convergence studies (Section 13.E).
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.quant.monte_carlo import ConvergenceStudy


def plot_convergence(
    study: ConvergenceStudy,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel convergence plot: estimate-vs-M, and error-vs-M (log scale).

    Left panel shows the running mean estimate with its confidence
    band converging to the theoretical mean as M grows. Right panel
    shows absolute error on a log-log scale, where GBM/Monte Carlo
    theory predicts a straight line of slope -1/2 (the O(1/sqrt(M))
    convergence rate).
    """
    arrays = study.as_arrays()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    ax1.plot(arrays["simulations"], arrays["mean_estimate"], "o-", label="MC estimate")
    ax1.fill_between(
        arrays["simulations"],
        arrays["ci_lower"],
        arrays["ci_upper"],
        alpha=0.2,
        label="95% CI of estimator",
    )
    ax1.axhline(
        study.theoretical_mean, color="black", linestyle="--", label="Theoretical mean"
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Simulations (M)")
    ax1.set_ylabel("Estimated E[S_T]")
    ax1.set_title("Convergence of the Mean Estimate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(arrays["simulations"], arrays["absolute_error"], "o-", color="#C44E52")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Simulations (M)")
    ax2.set_ylabel("Absolute Error vs. Theory")
    ax2.set_title("Error Decay (theory: slope ~ -1/2, i.e. O(1/sqrt(M)))")
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    return fig, axes
