import numpy as np
import pytest

from app.models.portfolio_parameters import AssetParameters, PortfolioParameters
from app.quant.portfolio.correlated_gbm import (
    realized_correlation,
    simulate_correlated_terminal_prices,
)


def _portfolio(corr, n=3):
    assets = [
        AssetParameters(name=f"A{i}", initial_price=100, drift=0.08, volatility=0.25)
        for i in range(n)
    ]
    return PortfolioParameters(
        assets=assets,
        weights=[1 / n] * n,
        correlation_matrix=corr,
        portfolio_value=1_000_000,
        time_horizon=1.0,
    )


def test_output_shape():
    p = _portfolio(np.eye(3), n=3)
    result = simulate_correlated_terminal_prices(p, simulations=1000, seed=1)
    assert result.terminal_prices.shape == (1000, 3)
    assert result.asset_names == ["A0", "A1", "A2"]


def test_prices_always_positive():
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    p = _portfolio(corr, n=2)
    result = simulate_correlated_terminal_prices(p, simulations=5000, seed=2)
    assert np.all(result.terminal_prices > 0)


def test_reproducibility_with_seed():
    p = _portfolio(np.eye(2), n=2)
    r1 = simulate_correlated_terminal_prices(p, simulations=500, seed=42)
    r2 = simulate_correlated_terminal_prices(p, simulations=500, seed=42)
    np.testing.assert_array_equal(r1.terminal_prices, r2.terminal_prices)


def test_zero_correlation_gives_uncorrelated_returns():
    p = _portfolio(np.eye(3), n=3)
    result = simulate_correlated_terminal_prices(p, simulations=200_000, seed=5)
    realized = realized_correlation(result.terminal_prices)
    off_diag = realized[~np.eye(3, dtype=bool)]
    assert np.all(np.abs(off_diag) < 0.02)


def test_high_correlation_is_recovered():
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    p = _portfolio(corr, n=2)
    result = simulate_correlated_terminal_prices(p, simulations=200_000, seed=6)
    realized = realized_correlation(result.terminal_prices)
    assert abs(realized[0, 1] - 0.8) < 0.02


def test_negative_correlation_is_recovered():
    corr = np.array([[1.0, -0.7], [-0.7, 1.0]])
    p = _portfolio(corr, n=2)
    result = simulate_correlated_terminal_prices(p, simulations=200_000, seed=7)
    realized = realized_correlation(result.terminal_prices)
    assert abs(realized[0, 1] - (-0.7)) < 0.02


def test_marginal_distribution_matches_single_asset_gbm():
    # Each asset's own marginal should match its independent GBM formula,
    # regardless of the correlation structure imposed between assets.
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    p = _portfolio(corr, n=2)
    result = simulate_correlated_terminal_prices(p, simulations=200_000, seed=9)
    log_prices_0 = np.log(result.terminal_prices[:, 0])
    theoretical_mean = np.log(100) + (0.08 - 0.5 * 0.25**2) * 1.0
    theoretical_std = 0.25 * np.sqrt(1.0)
    assert abs(np.mean(log_prices_0) - theoretical_mean) < 0.01
    assert abs(np.std(log_prices_0, ddof=1) - theoretical_std) < 0.01


def test_rejects_nonpositive_simulations():
    p = _portfolio(np.eye(2), n=2)
    with pytest.raises(ValueError):
        simulate_correlated_terminal_prices(p, simulations=0, seed=1)
