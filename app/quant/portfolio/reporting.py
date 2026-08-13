"""
Structured portfolio risk report.

Mirrors the role of ``app/quant/derivatives/reporting.py`` in Week 2:
bundles a simulation, both VaR/ES methods, and a plain-text dashboard
into one call, rather than requiring callers to wire the pieces
together by hand each time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.portfolio_parameters import PortfolioParameters
from app.quant.portfolio.portfolio_simulation import (
    PortfolioSimulationResult,
    simulate_portfolio,
)
from app.quant.portfolio.risk_metrics import (
    RiskMetricResult,
    historical_risk_metrics,
    parametric_risk_metrics,
)
from app.utils.random_state import Seed


@dataclass(frozen=True)
class PortfolioRiskReport:
    portfolio: PortfolioParameters
    simulation: PortfolioSimulationResult
    historical_risk: RiskMetricResult
    parametric_risk: RiskMetricResult

    def __str__(self) -> str:  # pragma: no cover - display only
        sim = self.simulation
        lines = [
            "Portfolio Risk Report",
            "──────────────────────",
            "",
            "Holdings",
        ]
        for name, w, alloc in zip(
            self.portfolio.asset_names,
            self.portfolio.weights,
            self.portfolio.dollar_allocations,
        ):
            lines.append(f"  {name:<12} weight {w:>7.2%}   ${alloc:>14,.2f}")
        lines += [
            "",
            f"Portfolio Value (t=0):   ${sim.initial_value:,.2f}",
            f"Horizon:                  {self.portfolio.time_horizon:.2f} yr",
            f"Simulations:              {sim.terminal_value.size:,}",
            "",
            f"Expected Terminal Value: ${np.mean(sim.terminal_value):,.2f}",
            f"Expected P&L:            ${np.mean(sim.pnl):,.2f} "
            f"({np.mean(sim.pnl_pct):+.2%})",
            f"P&L Std Dev:             ${np.std(sim.pnl, ddof=1):,.2f}",
            "",
            "Risk Metrics",
            f"  {self.historical_risk.confidence_level:.0%} VaR (historical):  "
            f"${self.historical_risk.var:,.2f}",
            f"  {self.historical_risk.confidence_level:.0%} ES  (historical):  "
            f"${self.historical_risk.expected_shortfall:,.2f}",
            f"  {self.parametric_risk.confidence_level:.0%} VaR (parametric):  "
            f"${self.parametric_risk.var:,.2f}",
            f"  {self.parametric_risk.confidence_level:.0%} ES  (parametric):  "
            f"${self.parametric_risk.expected_shortfall:,.2f}",
        ]
        return "\n".join(lines)


def generate_portfolio_risk_report(
    portfolio: PortfolioParameters,
    simulations: int,
    confidence_level: float = 0.95,
    seed: Seed = None,
) -> PortfolioRiskReport:
    """Simulate ``portfolio`` and bundle value, P&L, and risk metrics into one report."""
    sim = simulate_portfolio(portfolio, simulations, seed)
    return PortfolioRiskReport(
        portfolio=portfolio,
        simulation=sim,
        historical_risk=historical_risk_metrics(sim.pnl, confidence_level),
        parametric_risk=parametric_risk_metrics(sim.pnl, confidence_level),
    )
