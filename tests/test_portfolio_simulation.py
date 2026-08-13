import numpy as np

from app.models.portfolio_parameters import AssetParameters, PortfolioParameters
from app.quant.portfolio.portfolio_simulation import simulate_portfolio


def _portfolio():
    assets = [
        AssetParameters(name="A", initial_price=100, drift=0.08, volatility=0.2),
        AssetParameters(name="B", initial_price=50, drift=0.05, volatility=0.3),
    ]
    return PortfolioParameters(
        assets=assets,
        weights=[0.6, 0.4],
        correlation_matrix=np.array([[1.0, 0.3], [0.3, 1.0]]),
        portfolio_value=1_000_000,
        time_horizon=1.0,
    )


def test_initial_value_matches_portfolio_value():
    result = simulate_portfolio(_portfolio(), simulations=1000, seed=1)
    assert result.initial_value == 1_000_000


def test_output_shapes():
    result = simulate_portfolio(_portfolio(), simulations=2000, seed=1)
    assert result.terminal_value.shape == (2000,)
    assert result.pnl.shape == (2000,)
    assert result.pnl_pct.shape == (2000,)


def test_pnl_consistent_with_terminal_value():
    result = simulate_portfolio(_portfolio(), simulations=500, seed=2)
    np.testing.assert_allclose(
        result.pnl, result.terminal_value - result.initial_value
    )
    np.testing.assert_allclose(result.pnl_pct, result.pnl / result.initial_value)


def test_zero_volatility_zero_correlation_is_deterministic():
    assets = [
        AssetParameters(name="A", initial_price=100, drift=0.1, volatility=0.0),
        AssetParameters(name="B", initial_price=100, drift=0.1, volatility=0.0),
    ]
    p = PortfolioParameters(
        assets=assets,
        weights=[0.5, 0.5],
        correlation_matrix=np.eye(2),
        portfolio_value=1_000_000,
        time_horizon=1.0,
    )
    result = simulate_portfolio(p, simulations=100, seed=1)
    expected = 1_000_000 * np.exp(0.1)
    np.testing.assert_allclose(result.terminal_value, expected, rtol=1e-10)


def test_terminal_value_always_positive():
    result = simulate_portfolio(_portfolio(), simulations=5000, seed=3)
    assert np.all(result.terminal_value > 0)


def test_reproducibility_with_seed():
    r1 = simulate_portfolio(_portfolio(), simulations=1000, seed=99)
    r2 = simulate_portfolio(_portfolio(), simulations=1000, seed=99)
    np.testing.assert_array_equal(r1.terminal_value, r2.terminal_value)


def test_expected_value_close_to_weighted_drift():
    result = simulate_portfolio(_portfolio(), simulations=300_000, seed=4)
    # E[S_T] = S0 * e^{mu T} per asset -> weighted sum should be close.
    expected = 0.6 * 1_000_000 * np.exp(0.08) + 0.4 * 1_000_000 * np.exp(0.05)
    actual = np.mean(result.terminal_value)
    assert abs(actual - expected) / expected < 0.01
