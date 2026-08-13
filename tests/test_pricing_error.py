import numpy as np
import pytest

from app.models.option_parameters import OptionParameters, OptionSimulationConfig, OptionType
from app.quant.derivatives.monte_carlo_pricer import price_option_monte_carlo
from app.quant.derivatives.pricing_error import (
    compare_to_black_scholes,
    root_mean_squared_error,
)
from app.quant.derivatives.reporting import generate_paired_pricing_report
from app.quant.derivatives.validation import (
    check_call_bounds,
    check_put_bounds,
    check_put_call_parity,
)

CALL = OptionParameters(100, 100, 0.05, 0.20, 1.0, OptionType.CALL)
PUT = OptionParameters(100, 100, 0.05, 0.20, 1.0, OptionType.PUT)


def test_comparison_reports_bs_inside_ci_for_well_converged_run():
    cfg = OptionSimulationConfig(simulations=500_000, seed=42)
    result = price_option_monte_carlo(CALL, cfg)
    comparison = compare_to_black_scholes(result, ci_level=0.95)
    assert comparison.bs_inside_ci


def test_comparison_error_signs_are_consistent():
    cfg = OptionSimulationConfig(simulations=100_000, seed=1)
    result = price_option_monte_carlo(CALL, cfg)
    comparison = compare_to_black_scholes(result)
    assert comparison.signed_error == pytest.approx(comparison.mc_price - comparison.bs_price)
    assert comparison.absolute_error == pytest.approx(abs(comparison.signed_error))


def test_rmse_across_multiple_configurations():
    mc_prices = [10.0, 5.5, 20.1]
    bs_prices = [10.4, 5.6, 20.0]
    rmse = root_mean_squared_error(mc_prices, bs_prices)
    expected = np.sqrt(np.mean((np.array(mc_prices) - np.array(bs_prices)) ** 2))
    assert rmse == pytest.approx(expected)


def test_rmse_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        root_mean_squared_error([1.0, 2.0], [1.0])


def test_put_call_parity_holds_for_black_scholes_prices():
    from app.quant.derivatives.black_scholes import black_scholes_call, black_scholes_put

    call = black_scholes_call(100, 100, 0.05, 0.20, 1.0)
    put = black_scholes_put(100, 100, 0.05, 0.20, 1.0)
    check = check_put_call_parity(call, put, 100, 100, 0.05, 1.0, tolerance=1e-6)
    assert check.passed


def test_put_call_parity_holds_for_monte_carlo_within_sampling_error():
    cfg = OptionSimulationConfig(simulations=500_000, seed=42)
    call_result = price_option_monte_carlo(CALL, cfg)
    put_result = price_option_monte_carlo(PUT, cfg)
    combined_se = call_result.standard_error + put_result.standard_error
    check = check_put_call_parity(
        call_result.price, put_result.price, 100, 100, 0.05, 1.0,
        tolerance=4 * combined_se,
    )
    assert check.passed


def test_paired_report_dashboard_validates_end_to_end():
    cfg = OptionSimulationConfig(simulations=200_000, seed=42)
    report = generate_paired_pricing_report(100, 100, 0.05, 0.20, 1.0, cfg)
    assert report.parity_check.passed
    assert report.call_report.bounds_check.passed
    assert report.put_report.bounds_check.passed


def test_call_lower_bound_violation_is_detected():
    # Deliberately price a call below its no-arbitrage lower bound.
    bad_price = -1.0
    check = check_call_bounds(bad_price, spot=100, strike=100, risk_free_rate=0.05, maturity=1.0)
    assert not check.passed
    assert check.violation_detail is not None


def test_call_upper_bound_violation_is_detected():
    bad_price = 1000.0  # far above S0
    check = check_call_bounds(bad_price, spot=100, strike=100, risk_free_rate=0.05, maturity=1.0)
    assert not check.passed


def test_put_bounds_pass_for_valid_black_scholes_price():
    from app.quant.derivatives.black_scholes import black_scholes_put

    price = black_scholes_put(100, 100, 0.05, 0.20, 1.0)
    check = check_put_bounds(price, spot=100, strike=100, risk_free_rate=0.05, maturity=1.0)
    assert check.passed


def test_call_bounds_pass_for_valid_black_scholes_price():
    from app.quant.derivatives.black_scholes import black_scholes_call

    price = black_scholes_call(100, 100, 0.05, 0.20, 1.0)
    check = check_call_bounds(price, spot=100, strike=100, risk_free_rate=0.05, maturity=1.0)
    assert check.passed
