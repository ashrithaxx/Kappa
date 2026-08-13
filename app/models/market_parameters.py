"""
Market parameter model.

Encapsulates the market-side inputs to the GBM engine (as opposed to
the purely numerical simulation controls, which live in
``simulation_parameters.py``). Keeping these separate means future
modules (option pricing, portfolio risk) can mix-and-match market data
with different simulation configurations without duplicating validation
logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MarketParameters:
    """Market-side inputs describing a single underlying asset.

    Attributes
    ----------
    initial_price:
        Current/initial asset price, S0. Must be strictly positive.
    drift:
        Annualized expected return, mu (e.g. 0.08 for 8%). May be
        negative or zero.
    volatility:
        Annualized volatility, sigma (e.g. 0.20 for 20%). Must be
        non-negative; 0 collapses GBM to deterministic growth.
    time_horizon:
        Simulation horizon in years, T. Must be strictly positive.
    risk_free_rate:
        Optional risk-free rate, r, reserved for future
        discounting/derivatives-pricing modules. Not used by the GBM
        engine itself (which simulates under the real-world measure
        using ``drift``), but validated and carried through so it is
        available when option pricing is added.
    """

    initial_price: float
    drift: float
    volatility: float
    time_horizon: float
    risk_free_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.initial_price <= 0:
            raise ValueError(
                f"initial_price (S0) must be > 0, got {self.initial_price}"
            )
        if self.volatility < 0:
            raise ValueError(
                f"volatility (sigma) must be >= 0, got {self.volatility}"
            )
        if self.time_horizon <= 0:
            raise ValueError(
                f"time_horizon (T) must be > 0, got {self.time_horizon}"
            )
        if self.risk_free_rate is not None and self.risk_free_rate < -1:
            # Sanity bound only — rates below -100% are not economically
            # meaningful. Kept loose since negative rates are real.
            raise ValueError(
                f"risk_free_rate looks implausible: {self.risk_free_rate}"
            )
