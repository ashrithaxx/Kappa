"""
Monte Carlo option pricing under the risk-neutral measure.

RISK-NEUTRAL VS. PHYSICAL MEASURE
----------------------------------
The GBM engine simulates the *physical* (real-world) measure,
where the drift is the historically-estimated expected return, mu.
Option pricing is different: no-arbitrage valuation requires
discounting expected payoffs *under the risk-neutral measure*, where
every asset's drift is set to the risk-free rate, r, regardless of its
real-world expected return. This is not a modeling choice — it's a
consequence of the fact that a replicating/hedging portfolio earns the
risk-free rate under absence of arbitrage. So:

    Physical measure:      dS_t = mu S_t dt + sigma S_t dW_t
    Risk-neutral measure:  dS_t =  r S_t dt + sigma S_t dW_t   (this module)

Everything else about the GBM SDE — including sigma and its exact
discretization — is unchanged. This module therefore reuses the GBM
engine's terminal-mode machinery, just called with
``drift=risk_free_rate`` instead of the physical mu.

PRICING EQUATION
-----------------
    V_0 = e^{-rT} * E^Q[ Payoff_T ]

Monte Carlo estimator (payoff_i is the payoff on the i-th simulated
path, discounted individually so a confidence interval can be built
from the discounted-payoff sample directly):

    X_i    = e^{-rT} * Payoff_i
    V_hat  = (1/M) * sum(X_i)
    SE     = std(X) / sqrt(M)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from app.models.option_parameters import OptionParameters, OptionSimulationConfig, OptionType
from app.quant.derivatives.payoffs import payoff
from app.quant.gbm import simulate_gbm
from app.quant.statistics import confidence_interval as _confidence_interval
from app.utils.random_state import Seed


def simulate_risk_neutral_terminal_prices(
    spot: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    simulations: int,
    seed: Seed = None,
) -> np.ndarray:
    """Sample S_T under the risk-neutral GBM measure (drift = r).

    Reuses the GBM engine in terminal-only mode — for
    European vanilla options only S_T matters, so full paths are never
    generated. This makes the pricer O(M) in both time and memory,
    independent of any notion of "steps" (there is no path to step
    through), and comfortably handles 1,000,000+ simulations.

    Note ``steps`` is passed as 1 purely because ``simulate_gbm``'s
    signature expects it; in terminal mode the value is mathematically
    irrelevant (terminal sampling uses Brownian self-similarity and does
    not depend on step count).
    """
    result = simulate_gbm(
        initial_price=spot,
        drift=risk_free_rate,
        volatility=volatility,
        time_horizon=maturity,
        steps=1,
        simulations=simulations,
        seed=seed,
        mode="terminal",
    )
    return result.terminal_prices


@dataclass(frozen=True)
class MonteCarloPriceResult:
    """Structured result of a single Monte Carlo option-pricing run.

    Attributes
    ----------
    price:
        The Monte Carlo option price estimate, V_hat.
    standard_error:
        SE of the price estimator, std(discounted payoffs) / sqrt(M).
    confidence_intervals:
        Keyed by confidence level in percent (90, 95, 99).
    raw_payoffs:
        Undiscounted payoff on every simulated path, shape (M,).
    discounted_payoffs:
        ``raw_payoffs * exp(-rT)``, shape (M,) — this is the sample
        the price estimator and its CI are computed from.
    terminal_prices:
        The simulated S_T for every path, shape (M,).
    n_simulations:
        M.
    option:
        The ``OptionParameters`` this result was priced from.
    """

    price: float
    standard_error: float
    confidence_intervals: Dict[int, "ConfidenceIntervalLike"]
    raw_payoffs: np.ndarray
    discounted_payoffs: np.ndarray
    terminal_prices: np.ndarray
    n_simulations: int
    option: OptionParameters

    def __str__(self) -> str:  # pragma: no cover - display only
        ci95 = self.confidence_intervals.get(95)
        lines = [
            f"Monte Carlo {self.option.option_type.value} Price: ${self.price:.4f}",
            f"Standard Error:                ${self.standard_error:.4f}",
        ]
        if ci95 is not None:
            lines.append(f"95% CI:                        [${ci95.lower:.4f}, ${ci95.upper:.4f}]")
        lines.append(f"Simulations:                   {self.n_simulations:,}")
        return "\n".join(lines)


# Type alias only for the docstring above; the real type is
# app.quant.statistics.ConfidenceInterval, imported lazily to avoid a
# hard dependency at type-annotation time.
ConfidenceIntervalLike = object


def price_option_monte_carlo(
    option: OptionParameters,
    sim_config: OptionSimulationConfig,
    ci_levels=(0.90, 0.95, 0.99),
) -> MonteCarloPriceResult:
    """Price a European vanilla option via risk-neutral Monte Carlo simulation.

    Computational flow (Section 25 of the spec):

        random normal draws -> risk-neutral S_T -> payoff -> discount -> mean
    """
    terminal_prices = simulate_risk_neutral_terminal_prices(
        spot=option.spot,
        risk_free_rate=option.risk_free_rate,
        volatility=option.volatility,
        maturity=option.maturity,
        simulations=sim_config.simulations,
        seed=sim_config.seed,
    )

    raw_payoffs = payoff(terminal_prices, option.strike, option.option_type)
    discount_factor = np.exp(-option.risk_free_rate * option.maturity)
    discounted_payoffs = discount_factor * raw_payoffs

    price = float(np.mean(discounted_payoffs))

    ci_results = {
        round(level * 100): _confidence_interval(discounted_payoffs, level=level)
        for level in ci_levels
    }
    # Standard error of the mean, taken directly off whichever CI we
    # computed (they all share the same SE; grab any one).
    standard_error = next(iter(ci_results.values())).standard_error

    return MonteCarloPriceResult(
        price=price,
        standard_error=standard_error,
        confidence_intervals=ci_results,
        raw_payoffs=raw_payoffs,
        discounted_payoffs=discounted_payoffs,
        terminal_prices=terminal_prices,
        n_simulations=sim_config.simulations,
        option=option,
    )
