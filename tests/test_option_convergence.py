import numpy as np
import pytest

from app.models.option_parameters import OptionParameters, OptionType
from app.quant.derivatives.convergence import (
    estimate_convergence_rate,
    option_convergence_study,
)

CALL = OptionParameters(100, 100, 0.05, 0.20, 1.0, OptionType.CALL)


def test_standard_error_shrinks_monotonically_with_m():
    study = option_convergence_study(
        CALL, simulation_counts=(1_000, 10_000, 100_000, 1_000_000), seed=42
    )
    ses = [p.standard_error for p in study.points]
    assert all(ses[i] > ses[i + 1] for i in range(len(ses) - 1))


def test_standard_error_scales_as_inverse_sqrt_m():
    study = option_convergence_study(
        CALL, simulation_counts=(1_000, 4_000, 16_000, 64_000, 256_000), seed=1
    )
    arrays = study.as_arrays()
    normalized = arrays["standard_error"] * np.sqrt(arrays["simulations"])
    relative_spread = (normalized.max() - normalized.min()) / normalized.mean()
    assert relative_spread < 0.1


def test_absolute_error_generally_decreases_with_m():
    study = option_convergence_study(
        CALL, simulation_counts=(500, 5_000, 50_000, 500_000), seed=42
    )
    errors = [p.absolute_error for p in study.points]
    assert errors[-1] < errors[0]


def test_nested_common_random_numbers_are_genuine_prefixes():
    # A larger-M run's first m terminal prices must be byte-identical to
    # the smaller-M run's terminal prices — this is what "genuine nested
    # sampling" (Section 14 of the spec) means in practice, and it's what
    # makes the convergence curve monotonically comparable rather than
    # confounded by an unrelated random redraw at every M.
    small_study = option_convergence_study(CALL, simulation_counts=(1_000,), seed=7)
    large_study = option_convergence_study(CALL, simulation_counts=(1_000, 5_000), seed=7)
    # Reconstruct the underlying terminal-price arrays isn't directly
    # exposed on the study, so instead we check that the MC price at
    # M=1000 is identical whether it's the only point or the first of two.
    assert small_study.points[0].mc_price == large_study.points[0].mc_price
    assert small_study.points[0].standard_error == large_study.points[0].standard_error


def test_convergence_rate_close_to_theoretical():
    study = option_convergence_study(
        CALL,
        simulation_counts=(1_000, 5_000, 25_000, 125_000, 625_000),
        seed=42,
    )
    rate = estimate_convergence_rate(study)
    # Should be in the right ballpark of -0.5; finite-sample noise means
    # we allow a fairly wide band rather than an exact match.
    assert -1.2 < rate.slope < -0.1
    assert rate.r_squared > 0.5


def test_convergence_rate_rejects_insufficient_nonzero_error_points():
    # Directly construct a study with degenerate (all-zero) errors to
    # test the rate-estimation guard, rather than relying on floating
    # point noise from an actual zero-volatility simulation (which, in
    # practice, still produces tiny nonzero floating-point residuals
    # rather than exact zeros).
    from app.quant.derivatives.convergence import OptionConvergenceStudy, OptionConvergencePoint

    points = [
        OptionConvergencePoint(
            simulations=m, mc_price=10.0, bs_price=10.0, absolute_error=0.0,
            percentage_error=0.0, standard_error=0.01, ci_level=0.95,
            ci_lower=9.98, ci_upper=10.02, bs_inside_ci=True,
        )
        for m in (1_000, 10_000, 100_000)
    ]
    degenerate_study = OptionConvergenceStudy(points=points, bs_price=10.0, option=CALL)
    with pytest.raises(ValueError):
        estimate_convergence_rate(degenerate_study)


def test_to_table_contains_expected_columns():
    study = option_convergence_study(CALL, simulation_counts=(1_000, 10_000), seed=42)
    table = study.to_table()
    for col in ("Simulations", "MC Price", "BS Price", "Abs Error", "% Error", "Std Error", "BS in CI"):
        assert col in table


def test_bs_price_is_constant_across_the_study():
    study = option_convergence_study(CALL, simulation_counts=(1_000, 10_000, 100_000), seed=42)
    bs_prices = {p.bs_price for p in study.points}
    assert len(bs_prices) == 1
