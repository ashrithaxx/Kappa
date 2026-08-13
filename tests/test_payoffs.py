import numpy as np
import pytest

from app.models.option_parameters import OptionType
from app.quant.derivatives.payoffs import call_payoff, payoff, put_payoff


def test_call_payoff_itm():
    terminal = np.array([120.0, 150.0])
    result = call_payoff(terminal, strike=100.0)
    np.testing.assert_allclose(result, [20.0, 50.0])


def test_call_payoff_otm():
    terminal = np.array([80.0, 50.0])
    result = call_payoff(terminal, strike=100.0)
    np.testing.assert_allclose(result, [0.0, 0.0])


def test_call_payoff_atm():
    terminal = np.array([100.0])
    result = call_payoff(terminal, strike=100.0)
    np.testing.assert_allclose(result, [0.0])


def test_put_payoff_itm():
    terminal = np.array([80.0, 50.0])
    result = put_payoff(terminal, strike=100.0)
    np.testing.assert_allclose(result, [20.0, 50.0])


def test_put_payoff_otm():
    terminal = np.array([120.0, 150.0])
    result = put_payoff(terminal, strike=100.0)
    np.testing.assert_allclose(result, [0.0, 0.0])


def test_put_payoff_atm():
    terminal = np.array([100.0])
    result = put_payoff(terminal, strike=100.0)
    np.testing.assert_allclose(result, [0.0])


def test_payoffs_never_negative():
    rng = np.random.default_rng(0)
    terminal = rng.uniform(1, 300, size=10_000)
    assert np.all(call_payoff(terminal, 150.0) >= 0)
    assert np.all(put_payoff(terminal, 150.0) >= 0)


def test_payoff_dispatch_matches_direct_calls():
    terminal = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(
        payoff(terminal, 100.0, OptionType.CALL), call_payoff(terminal, 100.0)
    )
    np.testing.assert_allclose(
        payoff(terminal, 100.0, OptionType.PUT), put_payoff(terminal, 100.0)
    )


def test_payoff_dispatch_rejects_unknown_type():
    with pytest.raises(ValueError):
        payoff(np.array([100.0]), 100.0, "NOT_A_TYPE")


def test_payoffs_are_vectorized_not_looped():
    # Large array should compute instantly and match elementwise max().
    terminal = np.linspace(1, 1000, 1_000_000)
    result = call_payoff(terminal, 500.0)
    expected = np.maximum(terminal - 500.0, 0.0)
    np.testing.assert_array_equal(result, expected)
