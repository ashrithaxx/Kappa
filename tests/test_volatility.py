import numpy as np
import pytest

from app.quant.volatility import (
    compute_arithmetic_returns,
    compute_log_returns,
    historical_volatility,
)


def test_log_returns_known_values():
    prices = np.array([100.0, 110.0, 100.0])
    log_returns = compute_log_returns(prices)
    expected = np.array([np.log(110 / 100), np.log(100 / 110)])
    np.testing.assert_allclose(log_returns, expected)


def test_arithmetic_returns_known_values():
    prices = np.array([100.0, 110.0, 99.0])
    arith = compute_arithmetic_returns(prices)
    expected = np.array([0.10, -0.10])
    np.testing.assert_allclose(arith, expected)


def test_historical_volatility_annualization_scaling():
    rng = np.random.default_rng(0)
    daily_sigma = 0.01
    n = 5000
    log_rets = rng.normal(loc=0.0, scale=daily_sigma, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    prices = np.insert(prices, 0, 100.0)

    report = historical_volatility(prices, trading_days_per_year=252)
    # Annualized vol should be close to daily_sigma * sqrt(252)
    expected_annual = daily_sigma * np.sqrt(252)
    assert abs(report.annualized_volatility - expected_annual) / expected_annual < 0.1


def test_configurable_trading_days():
    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
    report_252 = historical_volatility(prices, trading_days_per_year=252)
    report_365 = historical_volatility(prices, trading_days_per_year=365)
    assert report_252.annualized_volatility != report_365.annualized_volatility
    ratio = report_365.annualized_volatility / report_252.annualized_volatility
    np.testing.assert_allclose(ratio, np.sqrt(365 / 252), rtol=1e-6)


def test_rejects_nonpositive_prices():
    with pytest.raises(ValueError):
        compute_log_returns(np.array([100.0, -1.0, 50.0]))


def test_rejects_too_few_observations():
    with pytest.raises(ValueError):
        historical_volatility(np.array([100.0]))
