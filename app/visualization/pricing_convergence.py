"""
Convergence plots for Monte Carlo option pricing (Sections 13/29 of the
Week 2 spec) — Monte Carlo price vs. Black-Scholes reference line,
absolute error decay, and standard error decay, all against simulation
count on a log axis.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.quant.derivatives.convergence import ConvergenceRateEstimate, OptionConvergenceStudy


def plot_price_convergence(
    study: OptionConvergenceStudy,
    ax: plt.Axes = None,
    title: str = None,
) -> plt.Axes:
    """Monte Carlo price vs. M, with a Black-Scholes reference line and CI band."""
    arrays = study.as_arrays()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.plot(arrays["simulations"], arrays["mc_price"], "o-", color="#4C72B0", label="MC price")
    ax.fill_between(
        arrays["simulations"], arrays["ci_lower"], arrays["ci_upper"],
        alpha=0.2, color="#4C72B0", label="95% CI",
    )
    ax.axhline(study.bs_price, color="black", linestyle="--", label=f"Black-Scholes = {study.bs_price:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Simulations (M)")
    ax.set_ylabel("Option Price")
    ax.set_title(title or f"Convergence of {study.option.option_type.value.title()} Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_error_convergence(
    study: OptionConvergenceStudy,
    rate: ConvergenceRateEstimate = None,
    ax: plt.Axes = None,
    title: str = "Absolute Pricing Error vs. Simulations",
) -> plt.Axes:
    """Absolute pricing error vs. M on a log-log scale, with fitted rate line."""
    arrays = study.as_arrays()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    mask = arrays["absolute_error"] > 0
    ax.plot(
        arrays["simulations"][mask], arrays["absolute_error"][mask],
        "o", color="#C44E52", label="Observed absolute error",
    )

    if rate is not None:
        sims = arrays["simulations"][mask]
        fitted = np.exp(rate.intercept) * sims.astype(float) ** rate.slope
        ax.plot(
            sims, fitted, "--", color="black",
            label=f"Fitted slope = {rate.slope:.3f} (theory: -0.5)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Simulations (M)")
    ax.set_ylabel("Absolute Error vs. Black-Scholes")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    return ax


def plot_standard_error_convergence(
    study: OptionConvergenceStudy,
    ax: plt.Axes = None,
    title: str = "Standard Error vs. Simulation Count",
) -> plt.Axes:
    """Standard error vs. M on a log-log scale — the O(1/sqrt(M)) signature."""
    arrays = study.as_arrays()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.plot(arrays["simulations"], arrays["standard_error"], "o-", color="#DD8452")

    # Reference O(1/sqrt(M)) line anchored at the first point for visual comparison.
    m0, se0 = arrays["simulations"][0], arrays["standard_error"][0]
    reference = se0 * np.sqrt(m0 / arrays["simulations"].astype(float))
    ax.plot(arrays["simulations"], reference, "--", color="grey", label="O(1/√M) reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Simulations (M)")
    ax.set_ylabel("Standard Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    return ax


def plot_convergence_panel(
    study: OptionConvergenceStudy,
    rate: ConvergenceRateEstimate = None,
    figsize: Tuple[int, int] = (16, 5),
):
    """Three-panel convergence view: price, absolute error, standard error."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_price_convergence(study, ax=axes[0])
    plot_error_convergence(study, rate=rate, ax=axes[1])
    plot_standard_error_convergence(study, ax=axes[2])
    fig.tight_layout()
    return fig, axes
