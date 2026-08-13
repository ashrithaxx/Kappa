"""
Parameter sensitivity analysis.

Varies one option parameter at a time (spot, strike, volatility,
risk-free rate, or maturity) while holding the others fixed, and
records Black-Scholes price, Monte Carlo price, and pricing error at
each value. This is explicitly NOT a Greeks engine — Greeks measure
instantaneous local sensitivity (derivatives), while this sweeps
discrete parameter values and re-prices from scratch at each one. The
`BlackScholesGreekInputs` structure in ``black_scholes.py`` exists
precisely so a future Greeks module can compute derivatives directly
instead of finite-differencing across a sweep like this one does.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence

from app.models.option_parameters import OptionParameters, OptionSimulationConfig
from app.quant.derivatives.black_scholes import black_scholes_price
from app.quant.derivatives.monte_carlo_pricer import price_option_monte_carlo

_VALID_FIELDS = {"spot", "strike", "volatility", "risk_free_rate", "maturity"}


@dataclass(frozen=True)
class SensitivityPoint:
    parameter: str
    value: float
    bs_price: float
    mc_price: float
    absolute_error: float
    percentage_error: float


def sweep_parameter(
    base_option: OptionParameters,
    parameter: str,
    values: Sequence[float],
    sim_config: OptionSimulationConfig,
) -> List[SensitivityPoint]:
    """Re-price ``base_option`` across ``values`` of a single parameter.

    Parameters
    ----------
    parameter:
        One of "spot", "strike", "volatility", "risk_free_rate", "maturity".
    values:
        The values to substitute for that parameter, one option re-priced
        per value; all other fields of ``base_option`` are held fixed.
    """
    if parameter not in _VALID_FIELDS:
        raise ValueError(f"parameter must be one of {_VALID_FIELDS}, got {parameter!r}")

    points: List[SensitivityPoint] = []
    for v in values:
        option = replace(base_option, **{parameter: v})
        bs_price = black_scholes_price(option)
        mc_result = price_option_monte_carlo(option, sim_config)
        mc_price = mc_result.price

        signed_error = mc_price - bs_price
        pct_error = (signed_error / bs_price * 100.0) if bs_price != 0 else float("nan")

        points.append(
            SensitivityPoint(
                parameter=parameter,
                value=float(v),
                bs_price=bs_price,
                mc_price=mc_price,
                absolute_error=abs(signed_error),
                percentage_error=pct_error,
            )
        )
    return points
