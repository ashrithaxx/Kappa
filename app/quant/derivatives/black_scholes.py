"""
Black-Scholes analytical European option pricing, implemented from
first principles (no prebuilt option-pricing library).

    d1 = [ ln(S0/K) + (r + 0.5*sigma^2) T ] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S0 * N(d1) - K * e^{-rT} * N(d2)
    Put  = K * e^{-rT} * N(-d2) - S0 * N(-d1)

This is the exact analytical price under the same risk-neutral GBM
assumptions the Monte Carlo pricer simulates — it exists in this
platform as the ground-truth benchmark the simulator is checked
against, not as a competing "real" pricing method.

EDGE CASES
----------
sigma = 0:
    d1/d2 are undefined (division by zero) because there's no
    randomness to define a distribution's quantiles from. But the
    *economically correct* limit is well defined: under zero
    volatility, S_T is deterministic at S0*e^{rT}, so
        Call -> max(S0 - K e^{-rT}, 0)
        Put  -> max(K e^{-rT} - S0, 0)
    (equivalently, the discounted intrinsic value against the forward
    price). We implement this limit directly rather than dividing by
    zero.

T = 0 (at expiry):
    Similarly undefined via the direct formula (sigma*sqrt(T) = 0
    in the denominator). The correct limit is the undiscounted
    intrinsic value:
        Call -> max(S0 - K, 0)
        Put  -> max(K - S0, 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from app.models.option_parameters import OptionParameters, OptionType

_EPS = 1e-12


@dataclass(frozen=True)
class BlackScholesGreekInputs:
    """d1/d2 and other intermediates, exposed for a future Greeks module.

    Keeping these as a separate, reusable structure means Delta, Gamma,
    Vega, Theta, and Rho (all of which are expressed in terms of d1,
    d2, and N'(d1)) can be added later without recomputing or
    re-deriving these quantities.
    """

    d1: float
    d2: float
    discount_factor: float  # e^{-rT}
    n_d1: float  # N(d1)
    n_d2: float  # N(d2)
    pdf_d1: float  # standard normal density at d1, N'(d1)


def _compute_d1_d2(
    spot: float, strike: float, risk_free_rate: float, volatility: float, maturity: float
) -> BlackScholesGreekInputs:
    """Compute d1, d2 and related intermediates, handling the sigma=0 / T=0 limits."""
    discount_factor = float(np.exp(-risk_free_rate * maturity))

    if volatility <= _EPS or maturity <= _EPS:
        # No well-defined d1/d2 in the degenerate limit. Callers must
        # branch on this case for pricing (see black_scholes_price);
        # we return sentinel +/-inf-flavored values so any code that
        # *does* try to use N(d1)/N(d2) here fails loudly rather than
        # silently producing a wrong number.
        return BlackScholesGreekInputs(
            d1=float("nan"),
            d2=float("nan"),
            discount_factor=discount_factor,
            n_d1=float("nan"),
            n_d2=float("nan"),
            pdf_d1=float("nan"),
        )

    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return BlackScholesGreekInputs(
        d1=float(d1),
        d2=float(d2),
        discount_factor=discount_factor,
        n_d1=float(stats.norm.cdf(d1)),
        n_d2=float(stats.norm.cdf(d2)),
        pdf_d1=float(stats.norm.pdf(d1)),
    )


def _degenerate_intrinsic_value(
    spot: float, strike: float, risk_free_rate: float, maturity: float, is_call: bool
) -> float:
    """Correct limiting price when sigma=0 and/or T=0 (no BS formula division)."""
    if maturity <= _EPS:
        # At expiry, forward and spot coincide (no time for discounting
        # or growth) — undiscounted intrinsic value.
        intrinsic = (spot - strike) if is_call else (strike - spot)
        return max(intrinsic, 0.0)

    # sigma == 0, T > 0: S_T is deterministic at the forward price
    # S0 * e^{rT}. Price = e^{-rT} * payoff(forward) = discounted
    # intrinsic value against the forward, i.e. max(S0 - K e^{-rT}, 0)
    # for a call.
    discount_factor = np.exp(-risk_free_rate * maturity)
    if is_call:
        return max(spot - strike * discount_factor, 0.0)
    return max(strike * discount_factor - spot, 0.0)


def black_scholes_call(
    spot: float, strike: float, risk_free_rate: float, volatility: float, maturity: float
) -> float:
    """Black-Scholes European call price: S0*N(d1) - K*e^{-rT}*N(d2)."""
    if volatility <= _EPS or maturity <= _EPS:
        return _degenerate_intrinsic_value(spot, strike, risk_free_rate, maturity, is_call=True)

    g = _compute_d1_d2(spot, strike, risk_free_rate, volatility, maturity)
    return float(spot * g.n_d1 - strike * g.discount_factor * g.n_d2)


def black_scholes_put(
    spot: float, strike: float, risk_free_rate: float, volatility: float, maturity: float
) -> float:
    """Black-Scholes European put price: K*e^{-rT}*N(-d2) - S0*N(-d1)."""
    if volatility <= _EPS or maturity <= _EPS:
        return _degenerate_intrinsic_value(spot, strike, risk_free_rate, maturity, is_call=False)

    g = _compute_d1_d2(spot, strike, risk_free_rate, volatility, maturity)
    return float(strike * g.discount_factor * (1 - g.n_d2) - spot * (1 - g.n_d1))


def black_scholes_price(option: OptionParameters) -> float:
    """Dispatch to ``black_scholes_call``/``black_scholes_put`` for an ``OptionParameters``."""
    args = (option.spot, option.strike, option.risk_free_rate, option.volatility, option.maturity)
    if option.option_type == OptionType.CALL:
        return black_scholes_call(*args)
    if option.option_type == OptionType.PUT:
        return black_scholes_put(*args)
    raise ValueError(f"Unknown option_type: {option.option_type!r}")
