"""
Monte Carlo convergence analysis for option pricing.

COMMON RANDOM NUMBERS — METHODOLOGY
-------------------------------------
A naive "use the same seed at every M" approach does NOT give nested
samples: seeding a fresh RNG with the same seed and then drawing M
values produces a *different* stream than drawing M' > M values with
that seed and taking a prefix, unless the generator and draw pattern
guarantee it (they don't, in general, and we make no such claim).
Re-seeding independently at every M instead means the comparison across
M is confounded by two things at once — more simulations, *and* an
unrelated random sample — which makes it harder to see the pure effect
of M on convergence.

This module avoids that confound honestly, by constructing genuine
nested samples: we draw ONE batch of `max(simulation_counts)` standard
normal variates up front, and every smaller M in the study uses the
first M draws from that same batch. Because the risk-neutral terminal
price for a European option is a deterministic function of a single
standard normal per path,

    S_T = S0 * exp[ (r - 0.5*sigma^2) T + sigma*sqrt(T) * Z ]

taking a prefix of the Z's is exactly equivalent to "the same
simulation, run with fewer paths" — a true nested subsample, not an
approximation. This is what makes the resulting convergence curve
monotonically comparable: each larger M's estimate literally contains
every smaller M's paths plus additional ones, so any improvement is
attributable to sample size alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from app.models.option_parameters import OptionParameters
from app.quant.derivatives.black_scholes import black_scholes_price
from app.quant.derivatives.payoffs import payoff
from app.quant.statistics import confidence_interval as _confidence_interval
from app.utils.random_state import Seed, get_rng


@dataclass(frozen=True)
class OptionConvergencePoint:
    simulations: int
    mc_price: float
    bs_price: float
    absolute_error: float
    percentage_error: float
    standard_error: float
    ci_level: float
    ci_lower: float
    ci_upper: float
    bs_inside_ci: bool


@dataclass(frozen=True)
class OptionConvergenceStudy:
    points: List[OptionConvergencePoint]
    bs_price: float
    option: OptionParameters

    def as_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "simulations": np.array([p.simulations for p in self.points]),
            "mc_price": np.array([p.mc_price for p in self.points]),
            "absolute_error": np.array([p.absolute_error for p in self.points]),
            "percentage_error": np.array([p.percentage_error for p in self.points]),
            "standard_error": np.array([p.standard_error for p in self.points]),
            "ci_lower": np.array([p.ci_lower for p in self.points]),
            "ci_upper": np.array([p.ci_upper for p in self.points]),
        }

    def to_table(self) -> str:
        """Render the convergence table (Section 30 of the spec)."""
        header = (
            f"{'Simulations':>12} | {'MC Price':>10} | {'BS Price':>10} | "
            f"{'Abs Error':>10} | {'% Error':>8} | {'Std Error':>10} | {'BS in CI':>8}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for p in self.points:
            lines.append(
                f"{p.simulations:>12,} | {p.mc_price:>10.4f} | {p.bs_price:>10.4f} | "
                f"{p.absolute_error:>10.4f} | {p.percentage_error:>7.3f}% | "
                f"{p.standard_error:>10.5f} | {'YES' if p.bs_inside_ci else 'NO':>8}"
            )
        return "\n".join(lines)


def option_convergence_study(
    option: OptionParameters,
    simulation_counts: Sequence[int] = (
        100,
        500,
        1_000,
        5_000,
        10_000,
        25_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
    ),
    seed: Seed = 42,
    ci_level: float = 0.95,
) -> OptionConvergenceStudy:
    """Study Monte Carlo option-price convergence as simulations increase.

    Uses genuine nested common random numbers (see module docstring):
    one batch of `max(simulation_counts)` normal draws is generated
    once, and every M uses a prefix of it.
    """
    simulation_counts = sorted(set(int(m) for m in simulation_counts))
    max_m = simulation_counts[-1]

    rng = get_rng(seed)
    z_full = rng.standard_normal(size=max_m)

    r, sigma, T, S0, K = (
        option.risk_free_rate,
        option.volatility,
        option.maturity,
        option.spot,
        option.strike,
    )
    log_return_full = (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z_full
    terminal_prices_full = S0 * np.exp(log_return_full)

    discount_factor = np.exp(-r * T)
    bs_price = black_scholes_price(option)

    points: List[OptionConvergencePoint] = []
    for m in simulation_counts:
        terminal_prices = terminal_prices_full[:m]
        raw_payoffs = payoff(terminal_prices, K, option.option_type)
        discounted_payoffs = discount_factor * raw_payoffs

        mc_price = float(np.mean(discounted_payoffs))
        ci = _confidence_interval(discounted_payoffs, level=ci_level)
        signed_err = mc_price - bs_price
        abs_err = abs(signed_err)
        # Signed percentage error, consistent with pricing_error.py's
        # convention (Section 10 of the spec): (MC - BS) / BS * 100.
        # This means the column can and will wobble around zero as M
        # grows rather than shrinking monotonically — that's expected
        # for a *signed* quantity under Monte Carlo noise; use
        # `absolute_error` (or `abs(percentage_error)`) to see the
        # magnitude trend instead.
        pct_err = (signed_err / bs_price * 100.0) if bs_price != 0 else float("nan")
        bs_inside_ci = ci.lower <= bs_price <= ci.upper

        points.append(
            OptionConvergencePoint(
                simulations=m,
                mc_price=mc_price,
                bs_price=bs_price,
                absolute_error=abs_err,
                percentage_error=pct_err,
                standard_error=ci.standard_error,
                ci_level=ci_level,
                ci_lower=ci.lower,
                ci_upper=ci.upper,
                bs_inside_ci=bool(bs_inside_ci),
            )
        )

    return OptionConvergenceStudy(points=points, bs_price=bs_price, option=option)


@dataclass(frozen=True)
class ConvergenceRateEstimate:
    """Empirical Monte Carlo convergence rate from a log-log regression.

    Fits  log(absolute_error) = a + b * log(simulations)  and reports
    the slope b, which theory predicts should be near -0.5 (the
    O(1/sqrt(M)) rate). Finite-sample b will not equal -0.5 exactly —
    see ``explanation`` for why.
    """

    slope: float
    intercept: float
    r_squared: float
    theoretical_slope: float = -0.5

    @property
    def explanation(self) -> str:
        return (
            "The theoretical -0.5 slope describes the asymptotic scaling of the "
            "*standard error*, E[|error|] ~ C/sqrt(M), as M -> infinity. At any "
            "finite M the observed absolute error is a single noisy realization, "
            "not its expectation — a lucky or unlucky draw at a particular M can "
            "make the empirical slope shallower or steeper than -0.5. The fit "
            "also mixes points near the theoretical asymptote (large M) with "
            "points still far from it (small M), and can be pulled around by any "
            "single point where the MC estimate happened to land unusually close "
            "to (or far from) the true Black-Scholes price by chance."
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"Empirical convergence slope: {self.slope:.4f} "
            f"(theoretical: {self.theoretical_slope:.4f}), R^2={self.r_squared:.4f}"
        )


def estimate_convergence_rate(study: OptionConvergenceStudy) -> ConvergenceRateEstimate:
    """Estimate the empirical Monte Carlo convergence rate via log-log regression."""
    arrays = study.as_arrays()
    sims = arrays["simulations"]
    errors = arrays["absolute_error"]

    mask = errors > 0  # log(0) is undefined; drop any exact-zero error points
    if mask.sum() < 2:
        raise ValueError(
            "Need at least 2 points with non-zero absolute error to fit a "
            "log-log convergence-rate regression."
        )

    log_m = np.log(sims[mask])
    log_err = np.log(errors[mask])

    slope, intercept, r_value, _p_value, _std_err = scipy_stats.linregress(log_m, log_err)

    return ConvergenceRateEstimate(
        slope=float(slope), intercept=float(intercept), r_squared=float(r_value**2)
    )
