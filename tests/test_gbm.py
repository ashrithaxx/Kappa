import numpy as np
import pytest

from app.quant.gbm import simulate_gbm


def test_output_shapes_full_mode():
    result = simulate_gbm(100, 0.08, 0.2, 1.0, steps=252, simulations=500, seed=1)
    assert result.time_grid.shape == (253,)
    assert result.price_paths.shape == (253, 500)
    assert result.terminal_prices.shape == (500,)


def test_output_shapes_terminal_mode():
    result = simulate_gbm(
        100, 0.08, 0.2, 1.0, steps=252, simulations=500, seed=1, mode="terminal"
    )
    assert result.price_paths is None
    assert result.terminal_prices.shape == (500,)
    assert result.time_grid.shape == (253,)


def test_prices_always_positive():
    result = simulate_gbm(50, -0.3, 0.9, 2.0, steps=100, simulations=2000, seed=7)
    assert np.all(result.price_paths > 0)
    assert np.all(result.terminal_prices > 0)


def test_initial_price_is_exact():
    result = simulate_gbm(123.45, 0.05, 0.15, 1.0, steps=10, simulations=50, seed=3)
    np.testing.assert_allclose(result.price_paths[0, :], 123.45)


def test_zero_volatility_is_deterministic():
    result = simulate_gbm(100, 0.1, 0.0, 1.0, steps=252, simulations=10, seed=1)
    expected = 100 * np.exp(0.1 * 1.0)
    np.testing.assert_allclose(result.terminal_prices, expected, rtol=1e-10)
    # All paths should be identical since there's no randomness at all
    for i in range(1, 10):
        np.testing.assert_allclose(
            result.price_paths[:, 0], result.price_paths[:, i], rtol=1e-10
        )


def test_reproducibility_with_seed():
    r1 = simulate_gbm(100, 0.08, 0.2, 1.0, steps=100, simulations=1000, seed=42)
    r2 = simulate_gbm(100, 0.08, 0.2, 1.0, steps=100, simulations=1000, seed=42)
    np.testing.assert_array_equal(r1.terminal_prices, r2.terminal_prices)
    np.testing.assert_array_equal(r1.price_paths, r2.price_paths)


def test_different_seeds_differ():
    r1 = simulate_gbm(100, 0.08, 0.2, 1.0, steps=100, simulations=1000, seed=1)
    r2 = simulate_gbm(100, 0.08, 0.2, 1.0, steps=100, simulations=1000, seed=2)
    assert not np.array_equal(r1.terminal_prices, r2.terminal_prices)


def test_full_and_terminal_modes_agree_statistically():
    # Terminal mode uses a direct one-draw-per-path shortcut (exploiting
    # Brownian self-similarity) rather than accumulating `steps` draws
    # per path, so it consumes the RNG stream differently and will not
    # produce bit-identical output to full mode even with the same seed.
    # What must hold is statistical equivalence: both modes simulate the
    # same terminal distribution, so large samples should have closely
    # matching mean/std.
    full = simulate_gbm(100, 0.08, 0.2, 1.0, steps=252, simulations=200_000, seed=5)
    term = simulate_gbm(
        100, 0.08, 0.2, 1.0, steps=252, simulations=200_000, seed=5, mode="terminal"
    )
    assert abs(full.terminal_prices.mean() - term.terminal_prices.mean()) / full.terminal_prices.mean() < 0.01
    assert abs(full.terminal_prices.std() - term.terminal_prices.std()) / full.terminal_prices.std() < 0.02


def test_terminal_mode_independent_of_steps():
    # Since terminal mode samples S_T directly, its distribution should
    # not depend on `steps` at all (unlike full mode, which has a tiny
    # amount of residual discretization variance in principle, though
    # here the discretization is exact too).
    r_coarse = simulate_gbm(
        100, 0.08, 0.2, 1.0, steps=10, simulations=100_000, seed=9, mode="terminal"
    )
    r_fine = simulate_gbm(
        100, 0.08, 0.2, 1.0, steps=5000, simulations=100_000, seed=9, mode="terminal"
    )
    np.testing.assert_array_equal(r_coarse.terminal_prices, r_fine.terminal_prices)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(initial_price=0, drift=0.1, volatility=0.2, time_horizon=1, steps=10, simulations=10),
        dict(initial_price=100, drift=0.1, volatility=-0.1, time_horizon=1, steps=10, simulations=10),
        dict(initial_price=100, drift=0.1, volatility=0.2, time_horizon=0, steps=10, simulations=10),
        dict(initial_price=100, drift=0.1, volatility=0.2, time_horizon=1, steps=0, simulations=10),
        dict(initial_price=100, drift=0.1, volatility=0.2, time_horizon=1, steps=10, simulations=0),
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        simulate_gbm(**kwargs)
