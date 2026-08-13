"""
Structured pricing report combining Monte Carlo pricing, Black-Scholes,
comparison, and financial validation into a single object suitable for
future frontend integration (Section 21 of the spec).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.models.option_parameters import OptionParameters, OptionSimulationConfig, OptionType
from app.quant.derivatives.monte_carlo_pricer import (
    MonteCarloPriceResult,
    price_option_monte_carlo,
)
from app.quant.derivatives.pricing_error import PricingComparison, compare_to_black_scholes
from app.quant.derivatives.validation import (
    BoundsCheck,
    ParityCheck,
    check_call_bounds,
    check_put_bounds,
    check_put_call_parity,
)


@dataclass(frozen=True)
class OptionPricingReport:
    """Everything needed to display or further process one option's pricing."""

    option: OptionParameters
    mc_result: MonteCarloPriceResult
    comparison: PricingComparison
    bounds_check: BoundsCheck

    def __str__(self) -> str:  # pragma: no cover - display only
        o = self.option
        mc = self.mc_result
        cmp = self.comparison
        ci95 = mc.confidence_intervals.get(95)

        lines = [
            f"European {o.option_type.value.title()} — Pricing Report",
            "─" * 44,
            "",
            "Market Inputs",
            f"Spot:                  ${o.spot:>10.2f}",
            f"Strike:                ${o.strike:>10.2f}",
            f"Risk-Free Rate:        {o.risk_free_rate:>10.2%}",
            f"Volatility:            {o.volatility:>10.2%}",
            f"Maturity:              {o.maturity:>10.2f} yr",
            "",
            "Monte Carlo",
            f"Simulations:           {mc.n_simulations:>10,}",
            f"Price:                 ${mc.price:>10.4f}",
            f"Std Error:             ${mc.standard_error:>10.4f}",
        ]
        if ci95 is not None:
            lines.append(f"95% CI:              [${ci95.lower:.4f}, ${ci95.upper:.4f}]")
        lines += [
            "",
            "Black-Scholes",
            f"Price:                 ${cmp.bs_price:>10.4f}",
            "",
            "Comparison",
            f"Absolute Error:        ${cmp.absolute_error:>10.4f}",
            f"Percentage Error:      {cmp.percentage_error:>10.4f}%",
            f"BS Inside MC CI:       {'YES' if cmp.bs_inside_ci else 'NO'}",
            "",
            "Validation",
            f"No-Arbitrage Bounds:   {'PASS' if self.bounds_check.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


def generate_pricing_report(
    option: OptionParameters,
    sim_config: OptionSimulationConfig,
    ci_level: float = 0.95,
) -> OptionPricingReport:
    """Price ``option`` via Monte Carlo, compare to Black-Scholes, and validate."""
    mc_result = price_option_monte_carlo(option, sim_config)
    comparison = compare_to_black_scholes(mc_result, ci_level=ci_level)

    if option.option_type == OptionType.CALL:
        bounds_check = check_call_bounds(
            comparison.mc_price, option.spot, option.strike, option.risk_free_rate, option.maturity
        )
    else:
        bounds_check = check_put_bounds(
            comparison.mc_price, option.spot, option.strike, option.risk_free_rate, option.maturity
        )

    return OptionPricingReport(
        option=option,
        mc_result=mc_result,
        comparison=comparison,
        bounds_check=bounds_check,
    )


@dataclass(frozen=True)
class PairedOptionPricingReport:
    """Call + put pricing side by side, with put-call parity (Section 21's
    full dashboard needs both legs — a single ``OptionPricingReport`` only
    has one option type, so parity can't be checked from it alone).
    """

    call_report: OptionPricingReport
    put_report: OptionPricingReport
    parity_check: ParityCheck

    @property
    def convergence_passed(self) -> bool:
        """True if Black-Scholes fell inside the Monte Carlo CI for both legs.

        Used here as a practical stand-in for "did the simulation converge
        well enough at this M" — it is not a claim that M is at some
        universal convergence threshold, just that this run's estimate is
        statistically consistent with the analytical benchmark.
        """
        return self.call_report.comparison.bs_inside_ci and self.put_report.comparison.bs_inside_ci

    def __str__(self) -> str:  # pragma: no cover - display only
        o = self.call_report.option
        cr, pr = self.call_report, self.put_report
        lines = [
            f"European Option Pair — Pricing Dashboard (S0={o.spot}, K={o.strike})",
            "─" * 60,
            "",
            "Market Inputs",
            f"Spot:                  ${o.spot:>10.2f}",
            f"Strike:                ${o.strike:>10.2f}",
            f"Risk-Free Rate:        {o.risk_free_rate:>10.2%}",
            f"Volatility:            {o.volatility:>10.2%}",
            f"Maturity:              {o.maturity:>10.2f} yr",
            f"Simulations:           {cr.mc_result.n_simulations:>10,}",
            "",
            "Call",
            f"Monte Carlo Price:     ${cr.comparison.mc_price:>10.4f}",
            f"Black-Scholes Price:   ${cr.comparison.bs_price:>10.4f}",
            f"Percentage Error:      {cr.comparison.percentage_error:>10.4f}%",
            "",
            "Put",
            f"Monte Carlo Price:     ${pr.comparison.mc_price:>10.4f}",
            f"Black-Scholes Price:   ${pr.comparison.bs_price:>10.4f}",
            f"Percentage Error:      {pr.comparison.percentage_error:>10.4f}%",
            "",
            "Validation",
            f"Put-Call Parity:       {'PASS' if self.parity_check.passed else 'FAIL'}",
            f"No-Arbitrage Bounds:   "
            f"{'PASS' if (cr.bounds_check.passed and pr.bounds_check.passed) else 'FAIL'}",
            f"Convergence:           {'PASS' if self.convergence_passed else 'FAIL'}",
        ]
        return "\n".join(lines)


def generate_paired_pricing_report(
    spot: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    sim_config: OptionSimulationConfig,
    ci_level: float = 0.95,
    parity_tolerance: Optional[float] = None,
) -> PairedOptionPricingReport:
    """Price a call and a put on the same underlying/strike/maturity and
    validate put-call parity between them (Section 17/21 of the spec).

    ``parity_tolerance``: if not given, defaults to 3x the combined
    standard errors of the two Monte Carlo estimates — Monte Carlo parity
    only holds up to the *combined* sampling error of both legs, not to
    the near-zero tolerance appropriate for exact Black-Scholes parity.
    """
    call_option = OptionParameters(spot, strike, risk_free_rate, volatility, maturity, OptionType.CALL)
    put_option = OptionParameters(spot, strike, risk_free_rate, volatility, maturity, OptionType.PUT)

    call_report = generate_pricing_report(call_option, sim_config, ci_level=ci_level)
    put_report = generate_pricing_report(put_option, sim_config, ci_level=ci_level)

    if parity_tolerance is None:
        parity_tolerance = 3.0 * (
            call_report.mc_result.standard_error + put_report.mc_result.standard_error
        )

    parity_check = check_put_call_parity(
        call_price=call_report.comparison.mc_price,
        put_price=put_report.comparison.mc_price,
        spot=spot,
        strike=strike,
        risk_free_rate=risk_free_rate,
        maturity=maturity,
        tolerance=parity_tolerance,
    )

    return PairedOptionPricingReport(
        call_report=call_report, put_report=put_report, parity_check=parity_check
    )
