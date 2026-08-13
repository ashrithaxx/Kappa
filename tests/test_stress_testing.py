import numpy as np
import pytest

from app.models.portfolio_parameters import AssetParameters, PortfolioParameters
from app.quant.portfolio.stress_testing import (
    STANDARD_SCENARIOS,
    Scenario,
    apply_scenario,
    run_stress_test,
)


def _portfolio():
    assets = [
        AssetParameters(name="A", initial_price=100, drift=0.08, volatility=0.2),
        AssetParameters(name="B", initial_price=50, drift=0.06, volatility=0.25),
    ]
    return PortfolioParameters(
        assets=assets,
        weights=[0.5, 0.5],
        correlation_matrix=np.array([[1.0, 0.2], [0.2, 1.0]]),
        portfolio_value=1_000_000,
        time_horizon=1.0,
    )


def test_price_shock_applies_to_all_assets():
    p = _portfolio()
    shocked = apply_scenario(p, Scenario(name="crash", price_shock_pct=-0.2))
    assert shocked.assets[0].initial_price == pytest.approx(80.0)
    assert shocked.assets[1].initial_price == pytest.approx(40.0)
    # Baseline is untouched (frozen dataclasses).
    assert p.assets[0].initial_price == 100


def test_volatility_multiplier_applies():
    p = _portfolio()
    shocked = apply_scenario(p, Scenario(name="vol", volatility_multiplier=2.0))
    assert shocked.assets[0].volatility == pytest.approx(0.4)
    assert shocked.assets[1].volatility == pytest.approx(0.5)


def test_drift_shift_applies():
    p = _portfolio()
    shocked = apply_scenario(p, Scenario(name="rates", drift_shift=-0.03))
    assert shocked.assets[0].drift == pytest.approx(0.05)
    assert shocked.assets[1].drift == pytest.approx(0.03)


def test_correlation_shift_applies_off_diagonal_only():
    p = _portfolio()
    shocked = apply_scenario(p, Scenario(name="corr", correlation_shift=0.5))
    assert shocked.correlation_matrix[0, 1] == pytest.approx(0.7)
    assert shocked.correlation_matrix[1, 0] == pytest.approx(0.7)
    assert shocked.correlation_matrix[0, 0] == pytest.approx(1.0)
    assert shocked.correlation_matrix[1, 1] == pytest.approx(1.0)


def test_correlation_shift_clips_to_valid_range():
    p = _portfolio()
    shocked = apply_scenario(p, Scenario(name="corr", correlation_shift=5.0))
    assert shocked.correlation_matrix[0, 1] <= 1.0


def test_combined_scenario_applies_all_shocks():
    p = _portfolio()
    scenario = Scenario(
        name="combo", price_shock_pct=-0.1, volatility_multiplier=1.5, drift_shift=-0.01
    )
    shocked = apply_scenario(p, scenario)
    assert shocked.assets[0].initial_price == pytest.approx(90.0)
    assert shocked.assets[0].volatility == pytest.approx(0.3)
    assert shocked.assets[0].drift == pytest.approx(0.07)


def test_market_decline_reduces_expected_value():
    result = run_stress_test(
        _portfolio(), Scenario(name="decline", price_shock_pct=-0.25),
        simulations=50_000, seed=1,
    )
    assert result.value_change < 0
    assert np.mean(result.stressed.terminal_value) < np.mean(result.baseline.terminal_value)


def test_volatility_spike_increases_var():
    result = run_stress_test(
        _portfolio(), Scenario(name="vol_spike", volatility_multiplier=3.0),
        simulations=100_000, seed=2,
    )
    assert result.stressed_risk.var > result.baseline_risk.var
    assert result.var_change > 0


def test_standard_scenarios_are_usable():
    p = _portfolio()
    for scenario in STANDARD_SCENARIOS.values():
        result = run_stress_test(p, scenario, simulations=5000, seed=1)
        assert result.scenario is scenario
        assert result.stressed.terminal_value.shape == (5000,)


def test_same_seed_isolates_scenario_effect():
    # With identical seeds, baseline vs stressed differences should be
    # attributable only to the scenario, not to different random draws
    # -- verified indirectly via a no-op scenario producing ~identical
    # (not just similarly-distributed) outcomes.
    result = run_stress_test(_portfolio(), Scenario(name="noop"), simulations=1000, seed=7)
    np.testing.assert_allclose(
        result.baseline.terminal_value, result.stressed.terminal_value
    )
