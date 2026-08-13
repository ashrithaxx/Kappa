"""
European option parameter model.

Deliberately separate from ``MarketParameters`` (Week 1): market
parameters describe the *physical-measure* dynamics of an asset (drift
mu estimated from history), whereas option pricing happens under the
*risk-neutral measure*, where the drift is the risk-free rate r, not
mu. Keeping these as distinct models makes that distinction structural
rather than a convention someone has to remember to apply correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass(frozen=True)
class OptionParameters:
    """Inputs for pricing a single European vanilla option.

    Attributes
    ----------
    spot:
        Current underlying price, S0. Must be > 0.
    strike:
        Strike price, K. Must be > 0.
    risk_free_rate:
        Continuously compounded risk-free rate, r. Used as the
        risk-neutral drift — NOT the historical/physical expected
        return. May be negative (real-world rates sometimes are) but
        must be economically plausible.
    volatility:
        Annualized volatility, sigma. Must be >= 0.
    maturity:
        Time to expiry in years, T. Must be > 0.
    option_type:
        ``OptionType.CALL`` or ``OptionType.PUT``.
    """

    spot: float
    strike: float
    risk_free_rate: float
    volatility: float
    maturity: float
    option_type: OptionType

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot (S0) must be > 0, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"strike (K) must be > 0, got {self.strike}")
        if self.volatility < 0:
            raise ValueError(f"volatility (sigma) must be >= 0, got {self.volatility}")
        if self.maturity <= 0:
            raise ValueError(f"maturity (T) must be > 0, got {self.maturity}")
        if self.risk_free_rate < -1:
            raise ValueError(
                f"risk_free_rate looks implausible: {self.risk_free_rate}"
            )
        if not isinstance(self.option_type, OptionType):
            raise ValueError(
                f"option_type must be an OptionType (CALL or PUT), got "
                f"{self.option_type!r}"
            )

    @property
    def moneyness(self) -> float:
        """S0 / K. >1 favors calls (spot above strike), <1 favors puts."""
        return self.spot / self.strike

    def classify_moneyness(self, atm_tolerance: float = 0.02) -> str:
        """Classify as DEEP_ITM / ITM / ATM / OTM / DEEP_OTM for this option's type.

        ``atm_tolerance`` is the relative distance from moneyness=1
        within which the option is considered at-the-money (default 2%).
        Deep ITM/OTM thresholds are set at a 20% relative move, a common
        rule-of-thumb boundary — not a precise financial definition.
        """
        m = self.moneyness
        if abs(m - 1.0) <= atm_tolerance:
            return "ATM"

        is_call = self.option_type == OptionType.CALL
        in_the_money = (m > 1.0) if is_call else (m < 1.0)
        deep = abs(m - 1.0) >= 0.20

        if in_the_money:
            return "DEEP_ITM" if deep else "ITM"
        return "DEEP_OTM" if deep else "OTM"


@dataclass(frozen=True)
class OptionSimulationConfig:
    """Numerical controls for option-pricing Monte Carlo runs.

    Distinct from Week 1's ``SimulationParameters`` because European
    vanilla options never need ``steps`` or ``mode`` — they are
    terminal-dependent, so pricing always uses the O(M) direct
    terminal-sampling shortcut. ``steps``/``mode`` reappear once
    path-dependent (Asian/Barrier) options are added.
    """

    simulations: int
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.simulations <= 0 or not isinstance(self.simulations, int):
            raise ValueError(
                f"simulations (M) must be a positive integer, got {self.simulations}"
            )
