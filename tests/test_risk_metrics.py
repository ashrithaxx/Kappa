import numpy as np
import pytest

from app.quant.portfolio.risk_metrics import (
    historical_expected_shortfall,
    historical_risk_metrics,
    historical_var,
    parametric_expected_shortfall,
    parametric_risk_metrics,
    parametric_var,
)


def test_historical_var_matches_percentile_definition():
    rng = np.random.default_rng(1)
    pnl = rng.normal(loc=0, scale=1000, size=100_000)
    var95 = historical_var(pnl, 0.95)
    expected = -np.percentile(pnl, 5)
    assert abs(var95 - expected) < 1e-6


def test_es_at_least_as_large_as_var():
    rng = np.random.default_rng(2)
    pnl = rng.standard_t(df=5, size=50_000) * 1000  # fat-tailed
    for level in (0.90, 0.95, 0.99):
        var = historical_var(pnl, level)
        es = historical_expected_shortfall(pnl, level)
        assert es >= var


def test_var_increases_with_confidence_level():
    rng = np.random.default_rng(3)
    pnl = rng.normal(0, 1000, 50_000)
    v90 = historical_var(pnl, 0.90)
    v95 = historical_var(pnl, 0.95)
    v99 = historical_var(pnl, 0.99)
    assert v90 < v95 < v99


def test_parametric_var_matches_normal_formula_for_normal_data():
    rng = np.random.default_rng(4)
    pnl = rng.normal(loc=100, scale=2000, size=500_000)
    var95 = parametric_var(pnl, 0.95)
    from scipy import stats

    expected = -(np.mean(pnl) + stats.norm.ppf(0.05) * np.std(pnl, ddof=1))
    assert abs(var95 - expected) / abs(expected) < 1e-6


def test_historical_and_parametric_agree_for_normal_data():
    # For Normal data, historical and parametric VaR should be close
    # (they estimate the same quantity two different ways).
    rng = np.random.default_rng(5)
    pnl = rng.normal(loc=0, scale=5000, size=500_000)
    hist = historical_var(pnl, 0.95)
    param = parametric_var(pnl, 0.95)
    assert abs(hist - param) / param < 0.03


def test_historical_risk_metrics_bundle():
    rng = np.random.default_rng(6)
    pnl = rng.normal(0, 1000, 10_000)
    result = historical_risk_metrics(pnl, 0.95)
    assert result.method == "historical"
    assert result.confidence_level == 0.95
    assert result.expected_shortfall >= result.var


def test_parametric_risk_metrics_bundle():
    rng = np.random.default_rng(7)
    pnl = rng.normal(0, 1000, 10_000)
    result = parametric_risk_metrics(pnl, 0.99)
    assert result.method == "parametric"
    assert result.expected_shortfall >= result.var


def test_rejects_invalid_confidence_level():
    pnl = np.random.default_rng(8).normal(0, 1, 100)
    with pytest.raises(ValueError):
        historical_var(pnl, 1.5)
    with pytest.raises(ValueError):
        parametric_var(pnl, 0.0)
