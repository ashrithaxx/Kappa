"""
Correlated multi-asset Geometric Brownian Motion.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.portfolio_parameters import PortfolioParameters
from app.utils.random_state import Seed, get_rng


@dataclass(frozen=True)
class CorrelatedTerminalPrices:
    """Result of a correlated multi-asset terminal-price simulation.

    Attributes
    ----------
    terminal_prices:
        Array of shape (simulations, n_assets) — S_T for every asset,
        every path.
    asset_names:
        Names in the same column order as ``terminal_prices``.
    """

    terminal_prices: np.ndarray
    asset_names: list


def simulate_correlated_terminal_prices(
    portfolio: PortfolioParameters,
    simulations: int,
    seed: Seed = None,
) -> CorrelatedTerminalPrices:
    """Simulate correlated S_T for every asset in ``portfolio``.

    Method
    ------
    1. Draw independent standard normals Z of shape (simulations, n).
    2. Correlate them via the Cholesky factor L of the correlation
       matrix (Sigma = L L^T): Z_corr = Z @ L^T, so that
       Corr(Z_corr) = Sigma exactly in expectation.
    3. Apply each asset's own GBM terminal-price formula  to its column of Z_corr:

           S_T = S0 * exp[(mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z_corr]

    This reproduces each asset's own marginal GBM distribution exactly
    (Cholesky-correlating standard normals does not change their
    individual N(0,1) marginals) while inducing the target linear
    correlation between assets' log returns.
    """
    if simulations <= 0:
        raise ValueError(f"simulations (M) must be > 0, got {simulations}")

    n = portfolio.n_assets
    corr = np.asarray(portfolio.correlation_matrix, dtype=float)
    chol = np.linalg.cholesky(corr)

    rng = get_rng(seed)
    z = rng.standard_normal(size=(simulations, n))
    z_corr = z @ chol.T

    T = portfolio.time_horizon
    s0 = np.array([a.initial_price for a in portfolio.assets])
    mu = np.array([a.drift for a in portfolio.assets])
    sigma = np.array([a.volatility for a in portfolio.assets])

    log_return = (mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z_corr
    terminal_prices = s0 * np.exp(log_return)

    return CorrelatedTerminalPrices(
        terminal_prices=terminal_prices,
        asset_names=portfolio.asset_names,
    )


def realized_correlation(terminal_prices: np.ndarray) -> np.ndarray:
    """Sample correlation of simulated log returns, for validating the induced structure.

    Compares against ``portfolio.correlation_matrix`` in tests/notebooks
    — converges to the target as ``simulations -> infinity`` but will
    show finite-sample deviation at any fixed M
    """
    log_prices = np.log(terminal_prices)
    return np.corrcoef(log_prices, rowvar=False)
