import numpy as np
import pytest

from app.quant.derivatives.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)


def test_benchmark_call_matches_known_value():
    # S0=100, K=100, r=5%, sigma=20%, T=1 -> Call ~= 10.45 (spec Section 24)
    price = black_scholes_call(100, 100, 0.05, 0.20, 1.0)
    assert price == pytest.approx(10.4506, abs=0.01)


def test_benchmark_put_matches_known_value():
    # Same benchmark -> Put ~= 5.57
    price = black_scholes_put(100, 100, 0.05, 0.20, 1.0)
    assert price == pytest.approx(5.5735, abs=0.01)


def test_put_call_parity_holds_exactly_for_black_scholes():
    call = black_scholes_call(105, 95, 0.03, 0.25, 0.75)
    put = black_scholes_put(105, 95, 0.03, 0.25, 0.75)
    lhs = call - put
    rhs = 105 - 95 * np.exp(-0.03 * 0.75)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_zero_volatility_call_matches_deterministic_limit():
    S0, K, r, T = 100.0, 90.0, 0.05, 1.0
    price = black_scholes_call(S0, K, r, 0.0, T)
    expected = max(S0 - K * np.exp(-r * T), 0.0)
    assert price == pytest.approx(expected, abs=1e-9)


def test_zero_volatility_put_matches_deterministic_limit():
    S0, K, r, T = 100.0, 110.0, 0.05, 1.0
    price = black_scholes_put(S0, K, r, 0.0, T)
    expected = max(K * np.exp(-r * T) - S0, 0.0)
    assert price == pytest.approx(expected, abs=1e-9)


def test_zero_volatility_no_division_by_zero():
    # Should not raise or produce nan/inf.
    call = black_scholes_call(100, 100, 0.05, 0.0, 1.0)
    put = black_scholes_put(100, 100, 0.05, 0.0, 1.0)
    assert np.isfinite(call)
    assert np.isfinite(put)


def test_near_zero_maturity_approaches_intrinsic_value():
    S0, K, r, sigma = 110.0, 100.0, 0.05, 0.2
    tiny_T = 1e-13
    call = black_scholes_call(S0, K, r, sigma, tiny_T)
    expected_intrinsic = max(S0 - K, 0.0)
    assert call == pytest.approx(expected_intrinsic, abs=0.05)


def test_deep_itm_call_approaches_intrinsic_minus_discounted_strike():
    # Deep ITM: extra time value is negligible relative to intrinsic value.
    S0, K, r, sigma, T = 500.0, 100.0, 0.05, 0.20, 1.0
    call = black_scholes_call(S0, K, r, sigma, T)
    lower_bound = max(S0 - K * np.exp(-r * T), 0.0)
    assert call >= lower_bound - 1e-9
    assert call == pytest.approx(lower_bound, rel=0.02)


def test_deep_otm_call_is_near_zero_but_positive():
    S0, K, r, sigma, T = 50.0, 500.0, 0.05, 0.20, 1.0
    call = black_scholes_call(S0, K, r, sigma, T)
    assert 0.0 <= call < 0.01


def test_deep_itm_put_approaches_discounted_strike_minus_spot():
    S0, K, r, sigma, T = 20.0, 500.0, 0.05, 0.20, 1.0
    put = black_scholes_put(S0, K, r, sigma, T)
    lower_bound = max(K * np.exp(-r * T) - S0, 0.0)
    assert put == pytest.approx(lower_bound, rel=0.02)


def test_deep_otm_put_is_near_zero_but_positive():
    S0, K, r, sigma, T = 500.0, 50.0, 0.05, 0.20, 1.0
    put = black_scholes_put(S0, K, r, sigma, T)
    assert 0.0 <= put < 0.01


def test_prices_are_always_nonnegative_across_grid():
    rng = np.random.default_rng(0)
    for _ in range(200):
        S0 = rng.uniform(1, 500)
        K = rng.uniform(1, 500)
        r = rng.uniform(-0.02, 0.15)
        sigma = rng.uniform(0.0, 1.5)
        T = rng.uniform(0.01, 5.0)
        assert black_scholes_call(S0, K, r, sigma, T) >= -1e-9
        assert black_scholes_put(S0, K, r, sigma, T) >= -1e-9


def test_call_upper_bound_is_spot():
    rng = np.random.default_rng(1)
    for _ in range(100):
        S0 = rng.uniform(1, 500)
        K = rng.uniform(1, 500)
        r = rng.uniform(0.0, 0.10)
        sigma = rng.uniform(0.01, 1.0)
        T = rng.uniform(0.01, 5.0)
        assert black_scholes_call(S0, K, r, sigma, T) <= S0 + 1e-9


def test_put_upper_bound_is_discounted_strike():
    rng = np.random.default_rng(2)
    for _ in range(100):
        S0 = rng.uniform(1, 500)
        K = rng.uniform(1, 500)
        r = rng.uniform(0.0, 0.10)
        sigma = rng.uniform(0.01, 1.0)
        T = rng.uniform(0.01, 5.0)
        assert black_scholes_put(S0, K, r, sigma, T) <= K * np.exp(-r * T) + 1e-9
