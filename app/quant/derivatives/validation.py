"""
Financial identity checks: put-call parity and no-arbitrage bounds.

Not present as a named file in the original architecture sketch, but
added as its own module (rather than folded into ``pricing_error.py``)
because these are structural, model-independent identities — they must
hold for Black-Scholes exactly and for Monte Carlo within sampling
error, and future option types (Asian, Barrier) will want their own
bound checks without inheriting vanilla put-call parity, which only
holds for European vanilla options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ParityCheck:
    """Put-call parity check: C - P = S0 - K*e^{-rT}."""

    call_price: float
    put_price: float
    spot: float
    strike: float
    risk_free_rate: float
    maturity: float
    lhs: float  # C - P
    rhs: float  # S0 - K*e^{-rT}
    parity_error: float  # lhs - rhs
    tolerance: float
    passed: bool

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"Put-Call Parity: {'PASS' if self.passed else 'FAIL'} "
            f"(C-P={self.lhs:.4f}, S0-Ke^-rT={self.rhs:.4f}, "
            f"error={self.parity_error:.4f}, tolerance={self.tolerance:.4f})"
        )


def check_put_call_parity(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    risk_free_rate: float,
    maturity: float,
    tolerance: float = 1e-6,
) -> ParityCheck:
    """Check C - P = S0 - K*e^{-rT}.

    ``tolerance`` should be set much looser for Monte Carlo prices
    than for Black-Scholes prices: Black-Scholes parity should hold to
    near machine precision (it's an algebraic identity of the closed
    form), while Monte Carlo parity holds only up to the combined
    sampling error of the two price estimates — pass a tolerance on
    the order of a few standard errors when checking simulated prices.
    """
    lhs = call_price - put_price
    rhs = spot - strike * np.exp(-risk_free_rate * maturity)
    parity_error = lhs - rhs
    return ParityCheck(
        call_price=call_price,
        put_price=put_price,
        spot=spot,
        strike=strike,
        risk_free_rate=risk_free_rate,
        maturity=maturity,
        lhs=float(lhs),
        rhs=float(rhs),
        parity_error=float(parity_error),
        tolerance=tolerance,
        passed=abs(parity_error) <= tolerance,
    )


@dataclass(frozen=True)
class BoundsCheck:
    """No-arbitrage bound check for a single option price."""

    option_type: str
    price: float
    lower_bound: float
    upper_bound: float
    passed: bool
    violation_detail: Optional[str]

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"No-Arbitrage Bounds ({self.option_type}): {'PASS' if self.passed else 'FAIL'}"


def check_call_bounds(
    price: float, spot: float, strike: float, risk_free_rate: float, maturity: float
) -> BoundsCheck:
    """max(S0 - K*e^{-rT}, 0) <= C <= S0."""
    lower = max(spot - strike * np.exp(-risk_free_rate * maturity), 0.0)
    upper = spot
    violation = None
    if price < lower - 1e-9:
        violation = f"price {price:.4f} below lower bound {lower:.4f}"
    elif price > upper + 1e-9:
        violation = f"price {price:.4f} above upper bound {upper:.4f}"
    return BoundsCheck(
        option_type="CALL",
        price=price,
        lower_bound=float(lower),
        upper_bound=float(upper),
        passed=violation is None,
        violation_detail=violation,
    )


def check_put_bounds(
    price: float, spot: float, strike: float, risk_free_rate: float, maturity: float
) -> BoundsCheck:
    """max(K*e^{-rT} - S0, 0) <= P <= K*e^{-rT}."""
    discounted_strike = strike * np.exp(-risk_free_rate * maturity)
    lower = max(discounted_strike - spot, 0.0)
    upper = discounted_strike
    violation = None
    if price < lower - 1e-9:
        violation = f"price {price:.4f} below lower bound {lower:.4f}"
    elif price > upper + 1e-9:
        violation = f"price {price:.4f} above upper bound {upper:.4f}"
    return BoundsCheck(
        option_type="PUT",
        price=price,
        lower_bound=float(lower),
        upper_bound=float(upper),
        passed=violation is None,
        violation_detail=violation,
    )
