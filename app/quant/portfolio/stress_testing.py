"""
Stress testing and scenario analysis.

A ``Scenario`` is a set of shocks applied to a ``PortfolioParameters``
*before* simulation, producing a new, independent portfolio spec that
``portfolio_simulation.py`` and ``risk_metrics.py`` are then run on
exactly as normal — stress testing reuses the same simulation and risk
machinery rather than duplicating it, the same "layer on top, don't
restructure" pattern as Week 2 on Week 1.

Four scenario types match the spec:
    - Market decline:        an immediate shock to spot prices (S0).
    - Increased volatility:  a multiplicative shock to sigma.
    - Interest-rate change:  an additive shift to drift (mu) — a rate
      rise increases the opportunity cost of holding risky assets,
      modeled here as a lower expected drift. This is a simplification
      (real rate effects also flow through discounting, valuations,
      and sector-specific channels); it is not a full rates model.
    - Correlation change:    an additive shift to off-diagonal
      correlations (e.g. "correlations rise in a crisis"), clipped to
      stay in [-1, 1] and re-validated for positive semi-definiteness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from app.models.portfolio_parameters import AssetParameters, PortfolioParameters
from app.quant.portfolio.portfolio_simulation import (
    PortfolioSimulationResult,
    simulate_portfolio,
)
from app.quant.portfolio.risk_metrics import RiskMetricResult, historical_risk_metrics
from app.utils.random_state import Seed


@dataclass(frozen=True)
class Scenario:
    """A named set of shocks to apply to a portfolio's parameters.

    All fields are optional; unset (``None``) fields leave that part of
    the portfolio unchanged. Multiple shocks can be combined in one
    scenario (e.g. a crash typically pairs a price decline with a
    volatility spike and a correlation increase).
    """

    name: str
    price_shock_pct: Optional[float] = None  # e.g. -0.20 for a 20% decline
    volatility_multiplier: Optional[float] = None  # e.g. 1.5 for +50% vol
    drift_shift: Optional[float] = None  # e.g. -0.02 for a 200bp drift cut
    correlation_shift: Optional[float] = None  # e.g. +0.20 off-diagonal


def apply_scenario(
    portfolio: PortfolioParameters, scenario: Scenario
) -> PortfolioParameters:
    """Return a new ``PortfolioParameters`` with ``scenario``'s shocks applied.

    Does not mutate ``portfolio`` — all dataclasses here are frozen, so
    the baseline portfolio remains valid for a separate un-stressed run.
    """
    assets = list(portfolio.assets)

    if scenario.price_shock_pct is not None:
        assets = [
            replace(a, initial_price=a.initial_price * (1 + scenario.price_shock_pct))
            for a in assets
        ]
    if scenario.volatility_multiplier is not None:
        assets = [
            replace(a, volatility=a.volatility * scenario.volatility_multiplier)
            for a in assets
        ]
    if scenario.drift_shift is not None:
        assets = [replace(a, drift=a.drift + scenario.drift_shift) for a in assets]

    correlation_matrix = np.asarray(portfolio.correlation_matrix, dtype=float)
    if scenario.correlation_shift is not None:
        n = portfolio.n_assets
        shocked = correlation_matrix + scenario.correlation_shift * (
            1 - np.eye(n)
        )
        correlation_matrix = np.clip(shocked, -1.0, 1.0)
        np.fill_diagonal(correlation_matrix, 1.0)

    return PortfolioParameters(
        assets=assets,
        weights=portfolio.weights,
        correlation_matrix=correlation_matrix,
        portfolio_value=portfolio.portfolio_value,
        time_horizon=portfolio.time_horizon,
    )


@dataclass(frozen=True)
class StressTestResult:
    """Baseline vs. stressed comparison for one scenario."""

    scenario: Scenario
    baseline: PortfolioSimulationResult
    stressed: PortfolioSimulationResult
    baseline_risk: RiskMetricResult
    stressed_risk: RiskMetricResult

    @property
    def value_change(self) -> float:
        """Change in expected (mean simulated) portfolio value, stressed - baseline."""
        return float(
            np.mean(self.stressed.terminal_value)
            - np.mean(self.baseline.terminal_value)
        )

    @property
    def value_change_pct(self) -> float:
        return self.value_change / self.baseline.initial_value

    @property
    def var_change(self) -> float:
        """Change in VaR, stressed - baseline. Positive = risk increased."""
        return self.stressed_risk.var - self.baseline_risk.var

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"Stress Scenario: {self.scenario.name}\n"
            f"{'-' * (17 + len(self.scenario.name))}\n"
            f"Baseline Expected Value:  ${np.mean(self.baseline.terminal_value):,.2f}\n"
            f"Stressed Expected Value:  ${np.mean(self.stressed.terminal_value):,.2f}\n"
            f"Value Change:             ${self.value_change:,.2f} "
            f"({self.value_change_pct:+.2%})\n\n"
            f"Baseline {self.baseline_risk.confidence_level:.0%} VaR / ES:  "
            f"${self.baseline_risk.var:,.2f} / ${self.baseline_risk.expected_shortfall:,.2f}\n"
            f"Stressed {self.stressed_risk.confidence_level:.0%} VaR / ES:  "
            f"${self.stressed_risk.var:,.2f} / ${self.stressed_risk.expected_shortfall:,.2f}\n"
            f"VaR Change:                ${self.var_change:,.2f}"
        )


def run_stress_test(
    portfolio: PortfolioParameters,
    scenario: Scenario,
    simulations: int,
    confidence_level: float = 0.95,
    seed: Seed = None,
) -> StressTestResult:
    """Simulate ``portfolio`` under baseline and stressed conditions and compare risk.

    Both legs use the same ``seed`` (where an int is given) so that
    differences are attributable to the scenario's shocks rather than
    to unrelated Monte Carlo sampling noise — the same common-random-
    numbers discipline used for convergence studies in Week 2.

    The stressed leg reuses the *baseline's* share counts (see
    ``simulate_portfolio``'s ``units`` parameter) rather than resizing
    positions at the shocked price — a market-decline scenario must
    revalue the holdings you already have, not buy fresh ones cheaper
    with the same dollar budget, or the price shock has no effect on
    the outcome at all.
    """
    stressed_portfolio = apply_scenario(portfolio, scenario)

    baseline_s0 = np.array([a.initial_price for a in portfolio.assets])
    baseline_units = portfolio.dollar_allocations / baseline_s0

    baseline = simulate_portfolio(portfolio, simulations, seed, units=baseline_units)
    stressed = simulate_portfolio(
        stressed_portfolio, simulations, seed, units=baseline_units
    )

    baseline_risk = historical_risk_metrics(baseline.pnl, confidence_level)
    stressed_risk = historical_risk_metrics(stressed.pnl, confidence_level)

    return StressTestResult(
        scenario=scenario,
        baseline=baseline,
        stressed=stressed,
        baseline_risk=baseline_risk,
        stressed_risk=stressed_risk,
    )


# A small library of common, ready-to-use scenarios matching the
# project spec ("market decline", "increased volatility", "interest-
# rate changes", "changes in asset correlations"). Callers can use
# these directly or define their own ``Scenario`` instances.
STANDARD_SCENARIOS = {
    "market_decline_moderate": Scenario(
        name="Market Decline (-15%)", price_shock_pct=-0.15
    ),
    "market_decline_severe": Scenario(
        name="Market Crash (-30%, vol spike, correlations rise)",
        price_shock_pct=-0.30,
        volatility_multiplier=1.75,
        correlation_shift=0.25,
    ),
    "volatility_spike": Scenario(
        name="Volatility Spike (+50%)", volatility_multiplier=1.5
    ),
    "rate_hike": Scenario(
        name="Rate Hike (drift -200bp)", drift_shift=-0.02
    ),
    "correlation_breakdown": Scenario(
        name="Correlations Rise (+0.30)", correlation_shift=0.30
    ),
}
