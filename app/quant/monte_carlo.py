"""
Monte Carlo engine: orchestrates GBM simulation, statistics, theoretical
comparison, convergence studies, and sanity checks.

This is the main entry point future modules (option pricing, portfolio
risk) are expected to build on top of — it deliberately returns plain
structured objects (dataclasses / dicts of floats) rather than anything
UI-specific, so it stays independent of the visualization layer.

IMPORTANT DISTINCTION
----------------------
The Monte Carlo confidence interval on E[S_T] measures uncertainty in
*our estimate of the mean*, and shrinks toward zero as simulations -> infinity.
It is not the same as, and should never be presented as, the range the
future price is likely to occupy — that's what the percentiles of the
terminal distribution are for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.models.market_parameters import MarketParameters
from app.models.simulation_parameters import SimulationParameters
from app.quant.distributions import (
    TheoreticalGBMDistribution,
    percentage_error,
    theoretical_gbm_distribution,
)
from app.quant.gbm import GBMResult, simulate_gbm
from app.quant.statistics import DescriptiveStatistics, descriptive_statistics
from app.utils.random_state import Seed


@dataclass(frozen=True)
class SimulationOutput:
    """Everything produced by a single Monte Carlo GBM run."""

    market: MarketParameters
    sim_params: SimulationParameters
    gbm_result: GBMResult
    statistics: DescriptiveStatistics
    theoretical: TheoreticalGBMDistribution
    mean_percentage_error: float
    std_percentage_error: float


def run_simulation(
    market: MarketParameters, sim_params: SimulationParameters
) -> SimulationOutput:
    """Run one full GBM Monte Carlo simulation and analyze the output."""
    gbm_result = simulate_gbm(
        initial_price=market.initial_price,
        drift=market.drift,
        volatility=market.volatility,
        time_horizon=market.time_horizon,
        steps=sim_params.steps,
        simulations=sim_params.simulations,
        seed=sim_params.seed,
        mode=sim_params.mode,
    )

    stats_report = descriptive_statistics(gbm_result.terminal_prices)
    theory = theoretical_gbm_distribution(
        market.initial_price, market.drift, market.volatility, market.time_horizon
    )

    mean_err = percentage_error(stats_report.mean, theory.price_mean)
    std_err = percentage_error(stats_report.std, theory.price_std)

    return SimulationOutput(
        market=market,
        sim_params=sim_params,
        gbm_result=gbm_result,
        statistics=stats_report,
        theoretical=theory,
        mean_percentage_error=mean_err,
        std_percentage_error=std_err,
    )


@dataclass(frozen=True)
class ConvergencePoint:
    simulations: int
    mean_estimate: float
    standard_error: float
    absolute_error: float
    percentage_error: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class ConvergenceStudy:
    points: List[ConvergencePoint]
    theoretical_mean: float

    def as_arrays(self) -> Dict[str, np.ndarray]:
        """Convenience accessor for plotting: parallel arrays by field."""
        return {
            "simulations": np.array([p.simulations for p in self.points]),
            "mean_estimate": np.array([p.mean_estimate for p in self.points]),
            "standard_error": np.array([p.standard_error for p in self.points]),
            "absolute_error": np.array([p.absolute_error for p in self.points]),
            "percentage_error": np.array([p.percentage_error for p in self.points]),
            "ci_lower": np.array([p.ci_lower for p in self.points]),
            "ci_upper": np.array([p.ci_upper for p in self.points]),
        }


def convergence_study(
    market: MarketParameters,
    steps: int,
    simulation_counts: Sequence[int] = (
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
    ),
    seed: Seed = 42,
    confidence_level: float = 0.95,
) -> ConvergenceStudy:
    """Study how the Monte Carlo mean estimate converges as M grows.

    For each M in ``simulation_counts``, runs an independent
    terminal-only simulation (cheap — no path matrix stored) and
    records the estimate, its standard error, and its error against
    the closed-form theoretical mean. Demonstrates the classical
    O(1/sqrt(M)) Monte Carlo convergence rate.

    A fixed seed is reused as a *base* seed but each simulation count
    draws its own independent stream (seed offset by index) so runs
    are reproducible without being literally the same underlying
    random path re-truncated.
    """
    from app.quant.statistics import confidence_interval as _ci

    theory = theoretical_gbm_distribution(
        market.initial_price, market.drift, market.volatility, market.time_horizon
    )
    theoretical_mean = theory.price_mean

    points: List[ConvergencePoint] = []
    for i, m in enumerate(simulation_counts):
        run_seed: Optional[int] = None if seed is None else int(seed) + i
        result = simulate_gbm(
            initial_price=market.initial_price,
            drift=market.drift,
            volatility=market.volatility,
            time_horizon=market.time_horizon,
            steps=steps,
            simulations=m,
            seed=run_seed,
            mode="terminal",
        )
        terminal = result.terminal_prices
        mean_est = float(np.mean(terminal))
        ci = _ci(terminal, level=confidence_level)
        abs_err = abs(mean_est - theoretical_mean)
        pct_err = percentage_error(mean_est, theoretical_mean)

        points.append(
            ConvergencePoint(
                simulations=int(m),
                mean_estimate=mean_est,
                standard_error=ci.standard_error,
                absolute_error=abs_err,
                percentage_error=pct_err,
                ci_lower=ci.lower,
                ci_upper=ci.upper,
            )
        )

    return ConvergenceStudy(points=points, theoretical_mean=float(theoretical_mean))


# ---------------------------------------------------------------------------
# Sanity checks (section 15 of the spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SanityCheckResult:
    name: str
    passed: bool
    detail: str


def run_sanity_checks(
    market: MarketParameters,
    sim_params: SimulationParameters,
    tolerance: float = 0.05,
) -> List[SanityCheckResult]:
    """Run the standard GBM sanity checks and return pass/fail results.

    Checks
    ------
    1. Zero volatility -> S_T is deterministic: S0 * exp(mu*T).
    2. Zero drift -> E[S_T] ~= S0 (within Monte Carlo tolerance).
    3. General expected value -> E[S_T] ~= S0 * exp(mu*T).
    4. Reproducibility -> identical seed produces identical terminal prices.
    """
    results: List[SanityCheckResult] = []

    # 1. Zero volatility
    zero_vol_market = MarketParameters(
        initial_price=market.initial_price,
        drift=market.drift,
        volatility=0.0,
        time_horizon=market.time_horizon,
    )
    zero_vol_result = simulate_gbm(
        zero_vol_market.initial_price,
        zero_vol_market.drift,
        0.0,
        zero_vol_market.time_horizon,
        sim_params.steps,
        sim_params.simulations,
        seed=sim_params.seed,
        mode="terminal",
    )
    expected_deterministic = market.initial_price * np.exp(
        market.drift * market.time_horizon
    )
    max_dev = float(
        np.max(np.abs(zero_vol_result.terminal_prices - expected_deterministic))
    )
    results.append(
        SanityCheckResult(
            name="Zero volatility -> deterministic growth",
            passed=max_dev < 1e-8 * max(1.0, expected_deterministic),
            detail=f"max deviation from S0*exp(mu*T): {max_dev:.2e}",
        )
    )

    # 2. Zero drift
    zero_drift_result = simulate_gbm(
        market.initial_price,
        0.0,
        market.volatility,
        market.time_horizon,
        sim_params.steps,
        sim_params.simulations,
        seed=sim_params.seed,
        mode="terminal",
    )
    mean_est = float(np.mean(zero_drift_result.terminal_prices))
    rel_err = abs(mean_est - market.initial_price) / market.initial_price
    results.append(
        SanityCheckResult(
            name="Zero drift -> E[S_T] ~= S0",
            passed=rel_err < tolerance,
            detail=f"E[S_T]={mean_est:.4f}, S0={market.initial_price:.4f}, "
            f"relative error={rel_err:.2%}",
        )
    )

    # 3. General expected value
    full_result = simulate_gbm(
        market.initial_price,
        market.drift,
        market.volatility,
        market.time_horizon,
        sim_params.steps,
        sim_params.simulations,
        seed=sim_params.seed,
        mode="terminal",
    )
    theory = theoretical_gbm_distribution(
        market.initial_price, market.drift, market.volatility, market.time_horizon
    )
    sim_mean = float(np.mean(full_result.terminal_prices))
    rel_err_mean = abs(sim_mean - theory.price_mean) / theory.price_mean
    results.append(
        SanityCheckResult(
            name="E[S_T] ~= S0 * exp(mu*T)",
            passed=rel_err_mean < tolerance,
            detail=f"simulated={sim_mean:.4f}, theoretical={theory.price_mean:.4f}, "
            f"relative error={rel_err_mean:.2%}",
        )
    )

    # 4. Reproducibility
    rerun_result = simulate_gbm(
        market.initial_price,
        market.drift,
        market.volatility,
        market.time_horizon,
        sim_params.steps,
        sim_params.simulations,
        seed=sim_params.seed,
        mode="terminal",
    )
    identical = (
        sim_params.seed is not None
        and np.array_equal(full_result.terminal_prices, rerun_result.terminal_prices)
    )
    results.append(
        SanityCheckResult(
            name="Reproducibility via seed",
            passed=bool(identical),
            detail="identical terminal prices across reruns with same seed"
            if identical
            else "seed was None, or reruns diverged",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Validation report (section 17 of the spec)
# ---------------------------------------------------------------------------


def generate_validation_report(output: SimulationOutput) -> str:
    """Render a human-readable validation report, mirroring the spec format."""
    m = output.market
    sp = output.sim_params
    stats_r = output.statistics
    ci95 = stats_r.confidence_intervals.get(95)

    sanity = run_sanity_checks(m, sp)
    convergence_pass = all(c.passed for c in sanity if c.name.startswith("E["))
    reproducibility_pass = next(
        c.passed for c in sanity if c.name == "Reproducibility via seed"
    )
    gbm_check_pass = all(c.passed for c in sanity)

    lines = [
        "Monte Carlo Validation Report",
        "──────────────────────────────",
        "",
        f"Initial Price:        ${m.initial_price:,.2f}",
        f"Drift:                {m.drift:.2%}",
        f"Volatility:           {m.volatility:.2%}",
        f"Time Horizon:         {m.time_horizon} year(s)",
        f"Simulations:          {sp.simulations:,}",
        "",
        f"Theoretical Mean:     {output.theoretical.price_mean:,.2f}",
        f"Simulated Mean:       {stats_r.mean:,.2f}",
        f"Error:                {output.mean_percentage_error:.2f}%",
        "",
        f"Theoretical Std:      {output.theoretical.price_std:,.2f}",
        f"Simulated Std:        {stats_r.std:,.2f}",
        "",
    ]
    if ci95 is not None:
        lines += [
            "95% CI (of the mean estimator):",
            f"[{ci95.lower:,.2f}, {ci95.upper:,.2f}]",
            "",
        ]
    lines += [
        f"Convergence: {'PASS' if convergence_pass else 'FAIL'}",
        f"Reproducibility: {'PASS' if reproducibility_pass else 'FAIL'}",
        f"GBM Check: {'PASS' if gbm_check_pass else 'FAIL'}",
    ]
    return "\n".join(lines)
