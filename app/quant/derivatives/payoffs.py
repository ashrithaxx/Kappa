"""
European option payoff functions.

Kept deliberately separate from both the simulation engine and the
pricing engine (Section 19/23 of the spec): a payoff function only
needs terminal prices and a strike, and knows nothing about
simulation, discounting, or path-dependence. This separation is what
lets Asian/Barrier options later reuse the same
``call_payoff``/``put_payoff`` primitives while supplying a different
"terminal quantity" (e.g. the path average instead of S_T).
"""

from __future__ import annotations

import numpy as np

from app.models.option_parameters import OptionType


def call_payoff(terminal_prices: np.ndarray, strike: float) -> np.ndarray:
    """European call payoff: max(S_T - K, 0), vectorized."""
    terminal_prices = np.asarray(terminal_prices, dtype=float)
    return np.maximum(terminal_prices - strike, 0.0)


def put_payoff(terminal_prices: np.ndarray, strike: float) -> np.ndarray:
    """European put payoff: max(K - S_T, 0), vectorized."""
    terminal_prices = np.asarray(terminal_prices, dtype=float)
    return np.maximum(strike - terminal_prices, 0.0)


def payoff(
    terminal_prices: np.ndarray, strike: float, option_type: OptionType
) -> np.ndarray:
    """Dispatch to ``call_payoff`` or ``put_payoff`` based on ``option_type``."""
    if option_type == OptionType.CALL:
        return call_payoff(terminal_prices, strike)
    if option_type == OptionType.PUT:
        return put_payoff(terminal_prices, strike)
    raise ValueError(f"Unknown option_type: {option_type!r}")
