"""
Geometric Brownian Motion (GBM) simulation engine.

SDE
    dS_t = mu * S_t dt + sigma * S_t dW_t

Exact discretization (not an Euler approximation — this is the exact
solution of the SDE evaluated at each grid point, so there is no
discretization bias in the drift/vol terms themselves, only Monte
Carlo sampling error):

    S_{t+dt} = S_t * exp[ (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z ]

    Z ~ N(0, 1), dt = T / N

Equivalently, log-prices follow a random walk with i.i.d. Normal
increments, which is what makes the vectorized cumulative-sum
implementation below exact rather than approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.utils.random_state import Seed, get_rng


@dataclass(frozen=True)
class GBMResult:
    """Container for a GBM simulation run.

    Attributes
    ----------
    time_grid:
        1-D array of shape (steps + 1,) — the simulation times, 0..T.
    price_paths:
        Full price matrix of shape (steps + 1, simulations), or
        ``None`` if the engine was run in "terminal" mode.
    terminal_prices:
        1-D array of shape (simulations,) — S_T for every path. Always
        populated, in both modes.
    """

    time_grid: np.ndarray
    price_paths: Optional[np.ndarray]
    terminal_prices: np.ndarray

    @property
    def shape(self) -> tuple:
        if self.price_paths is not None:
            return self.price_paths.shape
        return (self.time_grid.size, self.terminal_prices.size)


def simulate_gbm(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    steps: int,
    simulations: int,
    seed: Seed = None,
    mode: str = "full",
) -> GBMResult:
    """Simulate asset price paths under Geometric Brownian Motion.

    Fully vectorized: no per-path Python loops. Generates all
    ``steps * simulations`` standard normal draws in one call and
    builds every path simultaneously via a cumulative sum over the
    time axis.

    Parameters
    ----------
    initial_price:
        S0 > 0.
    drift:
        mu, annualized.
    volatility:
        sigma >= 0, annualized.
    time_horizon:
        T > 0, in years.
    steps:
        N > 0 time steps.
    simulations:
        M > 0 independent paths.
    seed:
        Passed to ``get_rng`` for reproducibility.
    mode:
        "full" to keep the entire price matrix, or "terminal" to keep
        only S_T (O(M) memory instead of O(N*M)) — use this for large
        M when path shapes/plots are not needed.

    Returns
    -------
    GBMResult
    """
    if initial_price <= 0:
        raise ValueError(f"initial_price must be > 0, got {initial_price}")
    if volatility < 0:
        raise ValueError(f"volatility must be >= 0, got {volatility}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if simulations <= 0:
        raise ValueError(f"simulations must be > 0, got {simulations}")
    if mode not in ("full", "terminal"):
        raise ValueError(f"mode must be 'full' or 'terminal', got {mode!r}")

    rng = get_rng(seed)
    dt = time_horizon / steps
    time_grid = np.linspace(0.0, time_horizon, steps + 1)

    if mode == "terminal":
        # Terminal-only shortcut: by the self-similarity of Brownian
        # motion, the sum of N i.i.d. N(0, dt) increments over the full
        # horizon T is *exactly* distributed as a single N(0, T) draw.
        # So S_T can be sampled directly from one standard normal per
        # path — O(simulations) time and memory, completely independent
        # of `steps`, with no loss of exactness relative to the
        # step-by-step construction (this is the same result the full
        # cumulative-sum path would give at t=T, just without paying to
        # materialize the intermediate steps).
        z = rng.standard_normal(size=simulations)
        terminal_log_return = (
            drift - 0.5 * volatility**2
        ) * time_horizon + volatility * np.sqrt(time_horizon) * z
        terminal_prices = initial_price * np.exp(terminal_log_return)
        return GBMResult(
            time_grid=time_grid,
            price_paths=None,
            terminal_prices=terminal_prices,
        )

    # Full-path mode: one draw per (step, path), vectorized cumulative
    # sum over the time axis builds every path simultaneously — no
    # per-path Python loop.
    drift_term = (drift - 0.5 * volatility**2) * dt
    vol_term = volatility * np.sqrt(dt)

    z = rng.standard_normal(size=(steps, simulations))
    log_increments = drift_term + vol_term * z  # shape (steps, simulations)
    cumulative_log_returns = np.cumsum(log_increments, axis=0)

    log_paths = np.vstack([np.zeros((1, simulations)), cumulative_log_returns])
    price_paths = initial_price * np.exp(log_paths)
    terminal_prices = price_paths[-1, :]
    return GBMResult(
        time_grid=time_grid,
        price_paths=price_paths,
        terminal_prices=terminal_prices,
    )
