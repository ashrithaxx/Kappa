import numpy as np
import pytest

from app.models.option_parameters import OptionParameters, OptionSimulationConfig, OptionType
from app.quant.derivatives.black_scholes import black_scholes_call, black_scholes_put
from app.quant.derivatives.monte_carlo_pricer import (
    price_option_monte_carlo,
    simulate_risk_neutral_terminal_prices,
)

CALL = OptionParameters(100, 100, 0.05, 0.20, 1.0, OptionType.CALL)
PUT = OptionParameters(100, 100, 0.05, 0.20, 1.0, OptionType.PUT)


def test_risk_neutral_terminal_prices_use_r_not_mu():
    # If historical mu were mistakenly used instead of r, the mean of
    # S_T would drift toward S0*exp(mu*T) rather than S0*exp(r*T). Here
    # we don't even pass a mu — simulate_risk_neutral_terminal_prices
    # only accepts risk_free_rate — so this test mainly documents and
    # locks in the expected mean.
    terminal = simulate_risk_neutral_terminal_prices(
        spot=100, risk_free_rate=0.05, volatility=0.20, maturity=1.0,
        simulations=500_000, seed=1,
    )
    expected_mean = 100 * np.exp(0.05 * 1.0)
    rel_err = abs(terminal.mean() - expected_mean) / expected_mean
    assert rel_err < 0.01


def test_terminal_prices_always_positive():
    terminal = simulate_risk_neutral_terminal_prices(
        spot=50, risk_free_rate=-0.02, volatility=0.9, maturity=3.0,
        simulations=10_000, seed=2,
    )
    assert np.all(terminal > 0)


def test_mc_call_price_converges_toward_black_scholes():
    bs_price = black_scholes_call(100, 100, 0.05, 0.20, 1.0)
    sim_cfg = OptionSimulationConfig(simulations=500_000, seed=42)
    result = price_option_monte_carlo(CALL, sim_cfg)
    rel_err = abs(result.price - bs_price) / bs_price
    assert rel_err < 0.02


def test_mc_put_price_converges_toward_black_scholes():
    bs_price = black_scholes_put(100, 100, 0.05, 0.20, 1.0)
    sim_cfg = OptionSimulationConfig(simulations=500_000, seed=42)
    result = price_option_monte_carlo(PUT, sim_cfg)
    rel_err = abs(result.price - bs_price) / bs_price
    assert rel_err < 0.02


def test_confidence_interval_widens_with_higher_level():
    sim_cfg = OptionSimulationConfig(simulations=50_000, seed=7)
    result = price_option_monte_carlo(CALL, sim_cfg)
    ci90 = result.confidence_intervals[90]
    ci99 = result.confidence_intervals[99]
    assert (ci99.upper - ci99.lower) > (ci90.upper - ci90.lower)


def test_confidence_interval_narrows_with_more_simulations():
    small = price_option_monte_carlo(CALL, OptionSimulationConfig(simulations=1_000, seed=1))
    large = price_option_monte_carlo(CALL, OptionSimulationConfig(simulations=1_000_000, seed=1))
    assert large.standard_error < small.standard_error


def test_reproducibility_with_seed():
    cfg = OptionSimulationConfig(simulations=10_000, seed=99)
    r1 = price_option_monte_carlo(CALL, cfg)
    r2 = price_option_monte_carlo(CALL, cfg)
    assert r1.price == r2.price
    np.testing.assert_array_equal(r1.terminal_prices, r2.terminal_prices)


def test_prices_are_positive():
    cfg = OptionSimulationConfig(simulations=10_000, seed=3)
    call_result = price_option_monte_carlo(CALL, cfg)
    put_result = price_option_monte_carlo(PUT, cfg)
    assert call_result.price > 0
    assert put_result.price > 0


def test_discounting_is_applied_correctly():
    cfg = OptionSimulationConfig(simulations=200_000, seed=5)
    result = price_option_monte_carlo(CALL, cfg)
    discount_factor = np.exp(-CALL.risk_free_rate * CALL.maturity)
    # discounted_payoffs must equal raw_payoffs * e^{-rT} exactly.
    np.testing.assert_allclose(
        result.discounted_payoffs, result.raw_payoffs * discount_factor
    )
    # And the price is literally the mean of the discounted payoffs.
    assert result.price == pytest.approx(float(np.mean(result.discounted_payoffs)))


def test_deep_otm_call_price_near_zero():
    deep_otm = OptionParameters(50, 500, 0.05, 0.20, 1.0, OptionType.CALL)
    cfg = OptionSimulationConfig(simulations=200_000, seed=11)
    result = price_option_monte_carlo(deep_otm, cfg)
    assert 0.0 <= result.price < 0.5


def test_deep_itm_call_price_near_intrinsic():
    deep_itm = OptionParameters(500, 100, 0.05, 0.20, 1.0, OptionType.CALL)
    cfg = OptionSimulationConfig(simulations=200_000, seed=12)
    result = price_option_monte_carlo(deep_itm, cfg)
    discount_factor = np.exp(-0.05 * 1.0)
    lower_bound = 500 - 100 * discount_factor
    assert result.price == pytest.approx(lower_bound, rel=0.02)
