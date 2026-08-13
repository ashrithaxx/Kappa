"""
Portfolio value and P&L simulation.

Converts per-asset simulated terminal prices (``correlated_gbm.py``)
into portfolio-level dollar value and P&L distributions — the input
every risk metric (VaR, Expected Shortfall) and stress test operates
on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.models.portfolio_parameters import PortfolioParameters
from app.quant.portfolio.correlated_gbm import (
    CorrelatedTerminalPrices,
    simulate_correlated_terminal_prices,
)
from app.utils.random_state import Seed


@dataclass(frozen=True)
class PortfolioSimulationResult:
    """Simulated portfolio outcomes across all Monte Carlo paths.

    Attributes
    ----------
    initial_value:
        Portfolio value at t=0 (``portfolio.portfolio_value``).
    terminal_value:
        Array of shape (simulations,) — total portfolio value at T for
        every path.
    pnl:
        terminal_value - initial_value, in currency units.
    pnl_pct:
        pnl / initial_value, as a fraction.
    asset_terminal_prices:
        The underlying (simulations, n_assets) price paths this
        portfolio result was built from, kept for attribution/stress
        analysis rather than re-simulating.
    """

    initial_value: float
    terminal_value: np.ndarray
    pnl: np.ndarray
    pnl_pct: np.ndarray
    asset_terminal_prices: CorrelatedTerminalPrices


def simulate_portfolio(
    portfolio: PortfolioParameters,
    simulations: int,
    seed: Seed = None,
    units: Optional[np.ndarray] = None,
) -> PortfolioSimulationResult:
    """Simulate portfolio value at T under correlated multi-asset GBM.

    Each asset's dollar allocation is converted into a fixed number of
    "units" at t=0 (``allocation / S0``), which is then revalued at the
    simulated S_T — i.e. a static buy-and-hold portfolio over the
    horizon, not a dynamically rebalanced one.

    ``units`` (shares held per asset) is normally derived from
    ``portfolio`` itself, but can be supplied explicitly so a scenario
    with a shocked spot price can be revalued using the *same* holdings
    as an unshocked baseline (see ``stress_testing.py``) — without this,
    a spot shock would be silently absorbed by re-sizing how many shares
    the same dollar allocation buys at the new price, leaving the
    simulated outcome unchanged.
    """
    correlated = simulate_correlated_terminal_prices(portfolio, simulations, seed)

    s0 = np.array([a.initial_price for a in portfolio.assets])
    if units is None:
        units = portfolio.dollar_allocations / s0  # shares held per asset

    terminal_value = correlated.terminal_prices @ units  # shape (simulations,)
    initial_value = float(np.sum(units * s0))

    pnl = terminal_value - initial_value
    pnl_pct = pnl / initial_value

    return PortfolioSimulationResult(
        initial_value=initial_value,
        terminal_value=terminal_value,
        pnl=pnl,
        pnl_pct=pnl_pct,
        asset_terminal_prices=correlated,
    )
