"""
Plotting utilities for simulated GBM price paths.

Deliberately plots only a user-controlled *subset* of paths (Section
13.A of the spec) — plotting all M paths (e.g. 100,000+) makes for an
unreadable, slow-to-render figure. Matplotlib is used for static/PDF
output; a Plotly variant is provided for interactive exploration.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_paths(
    time_grid: np.ndarray,
    price_paths: np.ndarray,
    n_paths_to_plot: int = 50,
    seed: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Simulated GBM Price Paths",
) -> plt.Axes:
    """Plot a random subset of simulated price paths.

    Parameters
    ----------
    time_grid:
        Shape (steps + 1,).
    price_paths:
        Shape (steps + 1, simulations). Requires full-path mode output.
    n_paths_to_plot:
        How many of the ``simulations`` columns to draw. Clamped to
        the available number of paths.
    seed:
        Controls which subset of paths is chosen, for reproducible
        plots independent of the simulation's own seed.
    """
    if price_paths is None:
        raise ValueError(
            "price_paths is None — this simulation ran in 'terminal' mode, "
            "which does not retain full paths for plotting."
        )

    n_sims = price_paths.shape[1]
    n_to_plot = min(n_paths_to_plot, n_sims)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_sims, size=n_to_plot, replace=False)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_grid, price_paths[:, indices], linewidth=0.8, alpha=0.6)
    ax.set_title(f"{title} ({n_to_plot} of {n_sims:,} paths shown)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    return ax


def plot_paths_plotly(
    time_grid: np.ndarray,
    price_paths: np.ndarray,
    n_paths_to_plot: int = 50,
    seed: Optional[int] = None,
    title: str = "Simulated GBM Price Paths",
):
    """Interactive Plotly version of ``plot_paths``."""
    import plotly.graph_objects as go

    if price_paths is None:
        raise ValueError(
            "price_paths is None — this simulation ran in 'terminal' mode."
        )

    n_sims = price_paths.shape[1]
    n_to_plot = min(n_paths_to_plot, n_sims)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_sims, size=n_to_plot, replace=False)

    fig = go.Figure()
    for idx in indices:
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=price_paths[:, idx],
                mode="lines",
                line=dict(width=1),
                opacity=0.5,
                showlegend=False,
            )
        )
    fig.update_layout(
        title=f"{title} ({n_to_plot} of {n_sims:,} paths shown)",
        xaxis_title="Time (years)",
        yaxis_title="Price",
        template="plotly_white",
    )
    return fig
