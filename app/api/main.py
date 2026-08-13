"""
Week 4 — FinTech Product: FastAPI service layer.

Thin HTTP wrapper around the Week 1-3 quant engine (app/quant, app/models).
No business logic lives here — every endpoint validates a request, calls
into the existing dataclasses/functions, and serializes the result.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.models.option_parameters import OptionParameters, OptionSimulationConfig, OptionType
from app.models.portfolio_parameters import AssetParameters, PortfolioParameters
from app.quant.gbm import simulate_gbm
from app.quant.derivatives.black_scholes import black_scholes_price
from app.quant.derivatives.monte_carlo_pricer import price_option_monte_carlo
from app.quant.derivatives.convergence import option_convergence_study
from app.quant.portfolio.portfolio_simulation import simulate_portfolio
from app.quant.portfolio.risk_metrics import historical_risk_metrics, parametric_risk_metrics
from app.quant.portfolio.stress_testing import Scenario, run_stress_test

app = FastAPI(title="Monte Carlo Risk & Derivatives Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class GBMRequest(BaseModel):
    initial_price: float = Field(gt=0)
    drift: float
    volatility: float = Field(ge=0)
    time_horizon: float = Field(gt=0)
    steps: int = Field(gt=0, le=1000)
    simulations: int = Field(gt=0, le=50000)
    seed: Optional[int] = None


class OptionRequest(BaseModel):
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    risk_free_rate: float
    volatility: float = Field(ge=0)
    maturity: float = Field(gt=0)
    option_type: OptionType
    simulations: int = Field(default=100_000, gt=0, le=1_000_000)
    seed: Optional[int] = None


class ConvergenceRequest(OptionRequest):
    sample_sizes: List[int] = Field(default=[1000, 5000, 10000, 50000, 100000, 500000])


class NamedOptionRequest(OptionRequest):
    label: str = "Option"


class OptionBatchRequest(BaseModel):
    instruments: List[NamedOptionRequest]


class AssetInput(BaseModel):
    name: str
    initial_price: float = Field(gt=0)
    drift: float
    volatility: float = Field(ge=0)
    weight: float


class PortfolioRequest(BaseModel):
    assets: List[AssetInput]
    correlation_matrix: List[List[float]]
    portfolio_value: float = Field(gt=0)
    time_horizon: float = Field(gt=0)
    simulations: int = Field(default=50_000, gt=0, le=1_000_000)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    confidence_levels: List[float] = Field(default=[0.90, 0.95, 0.99])
    seed: Optional[int] = None


class StressTestRequest(PortfolioRequest):
    scenario_name: str = "Custom Scenario"
    price_shock_pct: Optional[float] = None
    volatility_multiplier: Optional[float] = None
    drift_shift: Optional[float] = None
    correlation_shift: Optional[float] = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _build_portfolio(req: PortfolioRequest) -> PortfolioParameters:
    assets = [
        AssetParameters(name=a.name, initial_price=a.initial_price, drift=a.drift, volatility=a.volatility)
        for a in req.assets
    ]
    weights = [a.weight for a in req.assets]
    try:
        return PortfolioParameters(
            assets=assets,
            weights=weights,
            correlation_matrix=np.array(req.correlation_matrix, dtype=float),
            portfolio_value=req.portfolio_value,
            time_horizon=req.time_horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _histogram(data: np.ndarray, bins: int = 50) -> dict:
    counts, edges = np.histogram(data, bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/gbm/simulate")
def gbm_simulate(req: GBMRequest):
    result = simulate_gbm(
        initial_price=req.initial_price,
        drift=req.drift,
        volatility=req.volatility,
        time_horizon=req.time_horizon,
        steps=req.steps,
        simulations=req.simulations,
        seed=req.seed,
        mode="full",
    )
    n_display = min(req.simulations, 200)
    return {
        "time_grid": result.time_grid.tolist(),
        "sample_paths": result.price_paths[:, :n_display].T.tolist(),
        "terminal_prices_histogram": _histogram(result.terminal_prices),
        "terminal_stats": {
            "mean": float(np.mean(result.terminal_prices)),
            "std": float(np.std(result.terminal_prices)),
            "min": float(np.min(result.terminal_prices)),
            "max": float(np.max(result.terminal_prices)),
        },
    }


@app.post("/api/options/price")
def option_price(req: OptionRequest):
    return _price_one(req)


def _price_one(req: OptionRequest) -> dict:
    option = OptionParameters(
        spot=req.spot, strike=req.strike, risk_free_rate=req.risk_free_rate,
        volatility=req.volatility, maturity=req.maturity, option_type=req.option_type,
    )
    sim_config = OptionSimulationConfig(simulations=req.simulations, seed=req.seed)
    mc = price_option_monte_carlo(option, sim_config)
    bs = black_scholes_price(option)
    return {
        "monte_carlo_price": mc.price,
        "black_scholes_price": bs,
        "pricing_error": abs(mc.price - bs),
        "pricing_error_pct": abs(mc.price - bs) / bs if bs else None,
        "standard_error": mc.standard_error,
        "confidence_intervals": {
            str(level): {"lower": ci.lower, "upper": ci.upper}
            for level, ci in mc.confidence_intervals.items()
        },
        "n_simulations": mc.n_simulations,
        "payoff_histogram": _histogram(mc.raw_payoffs),
    }


@app.post("/api/options/price-batch")
def option_price_batch(req: OptionBatchRequest):
    """Price a book of existing derivatives (multiple named instruments) in one call."""
    results = []
    for inst in req.instruments:
        r = _price_one(inst)
        r["label"] = inst.label
        results.append(r)
    return {
        "instruments": results,
        "total_monte_carlo_value": sum(r["monte_carlo_price"] for r in results),
        "total_black_scholes_value": sum(r["black_scholes_price"] for r in results),
    }


@app.post("/api/options/convergence")
def option_convergence(req: ConvergenceRequest):
    option = OptionParameters(
        spot=req.spot, strike=req.strike, risk_free_rate=req.risk_free_rate,
        volatility=req.volatility, maturity=req.maturity, option_type=req.option_type,
    )
    study = option_convergence_study(option, simulation_counts=req.sample_sizes, seed=req.seed)
    return {
        "black_scholes_price": study.bs_price,
        "points": [
            {
                "n_simulations": p.simulations,
                "price": p.mc_price,
                "standard_error": p.standard_error,
                "error_vs_bs": p.absolute_error,
                "ci_lower": p.ci_lower,
                "ci_upper": p.ci_upper,
            }
            for p in study.points
        ],
    }


def _risk_contributions(sim, portfolio: PortfolioParameters, var_total: float) -> List[dict]:
    """Euler/component VaR allocation: contribution_i sums exactly to var_total.

    contribution_i = beta_i * VaR_total, where beta_i = Cov(pnl_i, pnl_p) / Var(pnl_p).
    """
    s0 = np.array([a.initial_price for a in portfolio.assets])
    units = portfolio.dollar_allocations / s0
    asset_pnl = sim.asset_terminal_prices.terminal_prices * units - (units * s0)  # (sims, n)
    port_var = np.var(sim.pnl)
    contributions = []
    for i, name in enumerate(sim.asset_terminal_prices.asset_names):
        beta = float(np.cov(asset_pnl[:, i], sim.pnl)[0, 1] / port_var) if port_var > 0 else 0.0
        contributions.append({
            "asset": name,
            "standalone_volatility": float(np.std(asset_pnl[:, i])),
            "var_contribution": beta * var_total,
            "var_contribution_pct": beta,
        })
    return contributions


@app.post("/api/portfolio/simulate")
def portfolio_simulate(req: PortfolioRequest):
    portfolio = _build_portfolio(req)
    sim = simulate_portfolio(portfolio, req.simulations, seed=req.seed)

    risk_by_level = {}
    for level in req.confidence_levels:
        hist = historical_risk_metrics(sim.pnl, level)
        param = parametric_risk_metrics(sim.pnl, level)
        risk_by_level[str(level)] = {
            "historical": {"var": hist.var, "expected_shortfall": hist.expected_shortfall},
            "parametric": {"var": param.var, "expected_shortfall": param.expected_shortfall},
        }
    primary = risk_by_level[str(req.confidence_level)]["historical"]

    return {
        "initial_value": sim.initial_value,
        "expected_terminal_value": float(np.mean(sim.terminal_value)),
        "portfolio_volatility": float(np.std(sim.pnl)),
        "pnl_histogram": _histogram(sim.pnl),
        "pnl_pct_stats": {
            "mean": float(np.mean(sim.pnl_pct)),
            "std": float(np.std(sim.pnl_pct)),
        },
        "risk_by_level": risk_by_level,
        "risk": {  # kept for backward compatibility with the primary confidence level
            "historical": primary,
            "parametric": risk_by_level[str(req.confidence_level)]["parametric"],
        },
        "risk_contributions": _risk_contributions(sim, portfolio, primary["var"]),
        "confidence_level": req.confidence_level,
    }


@app.post("/api/portfolio/stress-test")
def portfolio_stress_test(req: StressTestRequest):
    portfolio = _build_portfolio(req)
    scenario = Scenario(
        name=req.scenario_name,
        price_shock_pct=req.price_shock_pct,
        volatility_multiplier=req.volatility_multiplier,
        drift_shift=req.drift_shift,
        correlation_shift=req.correlation_shift,
    )
    try:
        result = run_stress_test(portfolio, scenario, req.simulations, req.confidence_level, seed=req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "scenario": scenario.name,
        "baseline": {
            "expected_value": float(np.mean(result.baseline.terminal_value)),
            "var": result.baseline_risk.var,
            "expected_shortfall": result.baseline_risk.expected_shortfall,
            "pnl_histogram": _histogram(result.baseline.pnl),
        },
        "stressed": {
            "expected_value": float(np.mean(result.stressed.terminal_value)),
            "var": result.stressed_risk.var,
            "expected_shortfall": result.stressed_risk.expected_shortfall,
            "pnl_histogram": _histogram(result.stressed.pnl),
        },
        "value_change": result.value_change,
        "value_change_pct": result.value_change_pct,
        "var_change": result.var_change,
    }


# Serve the dashboard frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
