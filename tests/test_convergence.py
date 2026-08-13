import numpy as np

from app.models.market_parameters import MarketParameters
from app.models.simulation_parameters import SimulationParameters
from app.quant.monte_carlo import (
    convergence_study,
    run_sanity_checks,
    run_simulation,
)


DEFAULT_MARKET = MarketParameters(
    initial_price=100.0, drift=0.08, volatility=0.20, time_horizon=1.0
)


def test_convergence_error_generally_decreases():
    study = convergence_study(
        DEFAULT_MARKET,
        steps=252,
        simulation_counts=(200, 2_000, 20_000, 200_000),
        seed=42,
    )
    errors = [p.absolute_error for p in study.points]
    # Not strictly monotonic (MC noise), but the last point should beat
    # the first by a wide margin, and overall trend should be downward.
    assert errors[-1] < errors[0]
    # Standard error itself IS guaranteed to shrink monotonically —
    # it only depends on sample std and count, not on MC noise in the mean.
    ses = [p.standard_error for p in study.points]
    assert all(ses[i] > ses[i + 1] for i in range(len(ses) - 1))


def test_convergence_rate_matches_inverse_sqrt_m():
    study = convergence_study(
        DEFAULT_MARKET,
        steps=100,
        simulation_counts=(1_000, 4_000, 16_000, 64_000),
        seed=1,
    )
    ses = np.array([p.standard_error for p in study.points])
    ms = np.array([p.simulations for p in study.points])
    # SE should scale as 1/sqrt(M): SE * sqrt(M) should be roughly constant.
    normalized = ses * np.sqrt(ms)
    relative_spread = (normalized.max() - normalized.min()) / normalized.mean()
    assert relative_spread < 0.1


def test_sanity_checks_all_pass_for_default_scenario():
    sim_params = SimulationParameters(steps=252, simulations=50_000, seed=42, mode="terminal")
    results = run_sanity_checks(DEFAULT_MARKET, sim_params, tolerance=0.05)
    failed = [r for r in results if not r.passed]
    assert not failed, f"Sanity checks failed: {failed}"


def test_run_simulation_end_to_end_default_scenario():
    sim_params = SimulationParameters(steps=252, simulations=20_000, seed=42, mode="terminal")
    output = run_simulation(DEFAULT_MARKET, sim_params)
    assert abs(output.mean_percentage_error) < 5.0
    assert output.statistics.n == 20_000
    assert output.gbm_result.terminal_prices.min() > 0
