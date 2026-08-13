# Monte Carlo Risk & Derivatives Platform

This platform is being built in weekly modules on a shared Monte Carlo
foundation:

- **Week 1 — Quantitative Foundation**: Geometric Brownian Motion (GBM)
  simulation, historical volatility estimation, descriptive statistics,
  confidence intervals, convergence analysis, and visualization.
- **Week 2 — Derivatives Engine**: European option pricing via Monte
  Carlo, an independent Black-Scholes analytical benchmark, pricing-error
  and convergence analysis, put-call parity, no-arbitrage bounds, and
  moneyness/parameter sensitivity — all built on top of Week 1 rather
  than duplicating it.

- **Week 3 — Portfolio Risk Engine**: correlated multi-asset GBM
  (Cholesky-decomposed correlation), portfolio value/P&L simulation,
  Value at Risk and Expected Shortfall (historical and parametric),
  and scenario-based stress testing (market decline, volatility spike,
  rate change, correlation breakdown) — layered on Week 1's GBM engine
  the same way Week 2 was.

All three are architected so that path-dependent options (Asian,
Barrier) can be layered on top without restructuring the core.

## Project Structure

```text
monte_carlo_platform/
├── app/
│   ├── models/
│   │   ├── market_parameters.py       # S0, mu, sigma, T, r (validated)      [Week 1]
│   │   ├── simulation_parameters.py   # N, M, seed, mode (validated)         [Week 1]
│   │   └── option_parameters.py       # S0, K, r, sigma, T, type (validated) [Week 2]
│   ├── quant/
│   │   ├── volatility.py              # historical vol estimation           [Week 1]
│   │   ├── distributions.py           # theoretical GBM moments             [Week 1]
│   │   ├── gbm.py                     # vectorized GBM simulator            [Week 1]
│   │   ├── monte_carlo.py             # engine, convergence, sanity checks  [Week 1]
│   │   ├── statistics.py              # descriptive stats & CIs            [Week 1]
│   │   └── derivatives/                                                   # [Week 2]
│   │       ├── payoffs.py             # call/put payoff functions
│   │       ├── monte_carlo_pricer.py  # risk-neutral sim, discounted payoff pricer
│   │       ├── black_scholes.py       # analytical call/put, d1/d2, edge cases
│   │       ├── pricing_error.py       # MC vs BS error, RMSE
│   │       ├── convergence.py         # option price convergence + rate fit
│   │       ├── validation.py          # put-call parity, no-arbitrage bounds
│   │       ├── sensitivity.py         # parameter sweeps (spot/strike/vol/r/T)
│   │       └── reporting.py           # structured pricing report / dashboard
│   │   └── portfolio/                                                     # [Week 3]
│   │       ├── correlated_gbm.py      # Cholesky-correlated multi-asset GBM
│   │       ├── portfolio_simulation.py# portfolio value / P&L simulation
│   │       ├── risk_metrics.py        # VaR & Expected Shortfall (hist + parametric)
│   │       ├── stress_testing.py      # scenario shocks + baseline-vs-stressed compare
│   │       └── reporting.py           # structured portfolio risk report
│   ├── visualization/
│   │   ├── paths.py                   # path plots (matplotlib + plotly)     [Week 1]
│   │   ├── distributions.py           # terminal / log-price histograms     [Week 1]
│   │   ├── convergence.py             # convergence plots                   [Week 1]
│   │   ├── option_payoff.py           # call/put payoff diagrams            [Week 2]
│   │   ├── pricing_convergence.py     # option price/error convergence      [Week 2]
│   │   ├── pricing_comparison.py      # MC vs BS bars, moneyness/vol plots  [Week 2]
│   │   └── portfolio.py               # P&L histogram, VaR/ES, correlation heatmap, stress plots [Week 3]
│   └── utils/
│       └── random_state.py            # centralized, reproducible RNG
├── tests/                             # pytest suite (127 tests)
├── notebooks/
│   ├── week1_quant_foundations.ipynb  # Week 1 end-to-end walkthrough
│   ├── week2_derivatives_engine.ipynb # Week 2 end-to-end walkthrough
│   └── week3_risk_analytics.ipynb     # Week 3 end-to-end walkthrough
├── requirements.txt
└── README.md
```

Data flows strictly in one direction:

```
market/option params → quant (gbm, volatility, derivatives pricing, statistics) → visualization
```

Nothing in `app/quant` imports from `app/visualization`, so the
simulation and pricing engines can be reused headlessly (APIs, batch
jobs, future risk modules) without pulling in plotting dependencies.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```python
from app.models.market_parameters import MarketParameters
from app.models.simulation_parameters import SimulationParameters
from app.quant.monte_carlo import run_simulation, generate_validation_report

market = MarketParameters(
    initial_price=100, drift=0.08, volatility=0.20, time_horizon=1.0
)
sim_params = SimulationParameters(steps=252, simulations=100_000, seed=42)

output = run_simulation(market, sim_params)
print(generate_validation_report(output))
```

```text
Monte Carlo Validation Report
──────────────────────────────

Initial Price:        $100.00
Drift:                8.00%
Volatility:           20.00%
Time Horizon:         1.0 year(s)
Simulations:          100,000

Theoretical Mean:     108.33
Simulated Mean:       108.37
Error:                0.04%

Theoretical Std:      21.88
Simulated Std:        21.98

95% CI (of the mean estimator):
[108.24, 108.51]

Convergence: PASS
Reproducibility: PASS
GBM Check: PASS
```

### Quick Start — Week 2 (Derivatives Engine)

```python
from app.models.option_parameters import OptionSimulationConfig
from app.quant.derivatives.reporting import generate_paired_pricing_report

sim_config = OptionSimulationConfig(simulations=100_000, seed=42)
report = generate_paired_pricing_report(
    spot=100, strike=100, risk_free_rate=0.05, volatility=0.20, maturity=1.0,
    sim_config=sim_config,
)
print(report)
```

```text
European Option Pair — Pricing Dashboard (S0=100.0, K=100.0)
────────────────────────────────────────────────────────────

Market Inputs
Spot:                  $    100.00
Strike:                $    100.00
Risk-Free Rate:             5.00%
Volatility:                20.00%
Maturity:                    1.00 yr
Simulations:              100,000

Call
Monte Carlo Price:     $   10.4205
Black-Scholes Price:   $   10.4506
Percentage Error:         -0.2875%

Put
Monte Carlo Price:     $    5.6122
Black-Scholes Price:   $    5.5735
Percentage Error:          0.6940%

Validation
Put-Call Parity:       PASS
No-Arbitrage Bounds:   PASS
Convergence:           PASS
```

## The Math

### Geometric Brownian Motion

The engine simulates the exact solution of:

```
dS_t = μ S_t dt + σ S_t dW_t
```

evaluated at each grid point:

```
S_{t+Δt} = S_t · exp[ (μ − ½σ²)Δt + σ√Δt · Z ],   Z ~ N(0,1),  Δt = T/N
```

This is the **exact** discretization (not an Euler approximation) —
log-prices are a random walk with i.i.d. Normal increments, computed
in one vectorized cumulative sum over all paths simultaneously
(`app/quant/gbm.py`), with no per-path Python loops.

### Theoretical distribution

```
ln(S_T) ~ Normal( ln(S0) + (μ − ½σ²)T,  σ²T )

E[S_T]   = S0 · e^{μT}
Var[S_T] = S0² · e^{2μT} · (e^{σ²T} − 1)
```

Simulated moments are checked against these closed forms in every run
(`app/quant/distributions.py`, `run_sanity_checks`).

### Volatility estimation

Estimated from **log returns** `r_t = ln(S_t / S_{t-1})` (not
arithmetic returns), because log returns are exactly the quantity GBM
assumes is i.i.d. Normal and time-additive:

```
σ_daily  = sqrt( 1/(n−1) · Σ(r_t − r̄)² )
σ_annual = σ_daily · sqrt(trading_days_per_year)
```

`trading_days_per_year` is configurable (252 for equities, 365 for
crypto, etc.) — never hardcoded.

### Monte Carlo error and confidence intervals

The standard error of the mean estimator shrinks as:

```
SE = s / √M        (classical O(1/√M) Monte Carlo convergence rate)
CI = x̄ ± z · SE     (z = 1.645 / 1.960 / 2.576 for 90% / 95% / 99%)
```

**This CI is uncertainty in the estimator, not the market.** It tells
you how precisely you've measured `E[S_T]` given `M` simulations — it
shrinks toward zero as `M → ∞`. It is *not* a statement about the
range the future price will occupy. For that, use the percentiles of
the terminal distribution itself (`descriptive_statistics(...).percentiles`).

## Two Simulation Modes (Week 1 GBM Engine)

- **`mode="full"`** — stores the entire `(steps+1, simulations)` price
  matrix. Needed for path plots or path-dependent payoffs later.
- **`mode="terminal"`** — stores only `S_T`, i.e. `O(M)` memory instead
  of `O(N·M)`. Use this for convergence studies or when only the
  terminal distribution matters; it comfortably handles 1,000,000+
  simulations.

### Quick Start — Week 3 (Portfolio Risk Engine)

```python
import numpy as np
from app.models.portfolio_parameters import AssetParameters, PortfolioParameters
from app.quant.portfolio.reporting import generate_portfolio_risk_report
from app.quant.portfolio.stress_testing import STANDARD_SCENARIOS, run_stress_test

assets = [
    AssetParameters("TechCo", 180, 0.12, 0.32),
    AssetParameters("BankCorp", 95, 0.07, 0.24),
]
portfolio = PortfolioParameters(
    assets=assets, weights=[0.6, 0.4],
    correlation_matrix=np.array([[1.0, 0.35], [0.35, 1.0]]),
    portfolio_value=10_000_000, time_horizon=1.0,
)

report = generate_portfolio_risk_report(portfolio, simulations=200_000, seed=42)
print(report)

stress = run_stress_test(portfolio, STANDARD_SCENARIOS["market_decline_severe"], simulations=200_000, seed=42)
print(stress)
```

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

127 tests cover: return calculations, GBM correctness and positivity,
shape/reproducibility guarantees, statistics correctness against
NumPy/SciPy, confidence-interval coverage, Monte Carlo convergence
behavior (Week 1); payoff correctness, Black-Scholes benchmark values
and edge cases, Monte Carlo option pricing (discounting, reproducibility,
risk-neutral drift), pricing-error/RMSE, put-call parity, no-arbitrage
bounds, and option-pricing convergence rate (Week 2); portfolio
parameter validation (symmetry, PSD, weight/asset alignment), induced
correlation accuracy, portfolio P&L consistency, VaR/ES correctness
(quantile definition, ES >= VaR, monotonicity in confidence level,
historical-vs-parametric agreement on Normal data), and stress-test
scenario mechanics including the fixed-share-count invariant that
keeps a price shock from silently canceling out (Week 3).

## Model Limitations (GBM / Black-Scholes)

GBM — and, by extension, Black-Scholes, which is derived from it — is a
foundational model, not a complete market model. It assumes:

- **Constant volatility** — no volatility clustering or smile/skew.
- **No jumps** — price paths are continuous; no gap risk.
- **No regime shifts** — a single (μ or r, σ) pair for the whole horizon.
- **No liquidity effects** — infinite depth, no market impact, no
  bid/ask spread, no transaction costs.
- **Log-normal terminal prices** — real markets exhibit fatter tails
  (excess kurtosis) than GBM produces, and implied volatility in real
  option markets varies by strike and maturity (the "volatility smile"),
  which a single constant σ cannot reproduce.

These are known, standard simplifications; the sanity checks and tests
in this package validate that the *implementation* is correct given
these assumptions, not that the assumptions themselves match real
markets. **Black-Scholes matching Monte Carlo validates internal
consistency of this codebase — it is not evidence that either model
accurately forecasts real-world option prices.**

---

# Week 2 — Derivatives Engine

Builds a European option-pricing engine on top of the Week 1 GBM
foundation: risk-neutral Monte Carlo pricing, an independent
Black-Scholes analytical benchmark, and the validation machinery
(error analysis, convergence, put-call parity, no-arbitrage bounds,
sensitivity) needed to trust the numbers it produces.

## The Physical Measure vs. the Risk-Neutral Measure

Week 1's GBM engine simulates under the **physical measure**, using a
historical/expected drift `μ`:

```
dS_t = μ S_t dt + σ S_t dW_t          (physical measure — for forecasting)
```

**Option pricing requires a different drift.** No-arbitrage theory
shows a derivative's price, when it can be replicated by trading the
underlying and a risk-free bond, does not depend on investors' risk
preferences or the asset's real expected return — only on the
risk-free rate `r`. This is the **risk-neutral measure**:

```
dS_t = r S_t dt + σ S_t dW_t           (risk-neutral measure — for pricing)

S_T = S0 · exp[(r − ½σ²)T + σ√T · Z],   Z ~ N(0,1)
```

Every option-pricing function in `app/quant/derivatives/` takes `r`,
never `μ` — `OptionParameters` doesn't even have a `μ` field, making the
distinction structural rather than a convention to remember.

## Monte Carlo Option Pricing

The fundamental risk-neutral pricing identity:

```
V0 = e^{-rT} · E^Q[Payoff_T]
```

with Monte Carlo estimator:

```
V_hat = (1/M) · Σ e^{-rT} · Payoff_i
```

Because a European payoff depends only on `S_T`, the engine uses
**terminal-only simulation** (`simulate_risk_neutral_terminal_prices`,
reusing Week 1's terminal-mode GBM shortcut) — no full paths are
generated, so pricing scales to 1,000,000+ simulations at O(M) memory
and sub-second runtime. `price_option_monte_carlo` returns a structured
`MonteCarloPriceResult` carrying the price, raw and discounted payoff
samples, standard error, and 90/95/99% confidence intervals — never
just a bare number.

## Black-Scholes (Implemented From First Principles)

```
C = S0·N(d1) − K·e^{-rT}·N(d2)          P = K·e^{-rT}·N(−d2) − S0·N(−d1)

d1 = [ln(S0/K) + (r + ½σ²)T] / (σ√T)     d2 = d1 − σ√T
```

No option-pricing library is used — `black_scholes.py` computes `d1`,
`d2`, and `N(·)` (via `scipy.stats.norm.cdf`) directly. Edge cases are
handled as mathematical limits rather than left to produce NaN/inf:

- **σ = 0**: price collapses to the deterministic discounted intrinsic
  value (`max(S0 − Ke^{-rT}, 0)` for a call) instead of dividing by
  `σ√T = 0`.
- **T → 0**: price approaches intrinsic value (`max(S0 − K, 0)`).
- **Deep ITM / deep OTM**: verified against the analytical no-arbitrage
  bounds rather than just "looks reasonable."

## Pricing Error, Confidence Intervals, and Convergence

For every priced option, `compare_to_black_scholes` reports signed
error, absolute error, percentage error, and whether the Black-Scholes
price falls inside the Monte Carlo confidence interval — the same
estimator-vs-market distinction from Week 1 applies here: **the CI
quantifies uncertainty in the Monte Carlo estimate, not in the "true"
option value** (Black-Scholes has no CI because it's exact).

`option_convergence_study` reprices the same option at simulation
counts from 100 to 1,000,000+ using **common random numbers** — nested
sampling where a larger run's first `m` draws are byte-identical to a
smaller run's `m` draws — so the convergence curve isn't confounded by
unrelated redraws between points. `estimate_convergence_rate` fits
`log(error) = a + b·log(M)` and compares the empirical slope `b`
against the theoretical `-0.5`. In practice the fitted slope rarely
lands exactly on `-0.5`: standard error decreases smoothly and
monotonically as `1/√M`, but the *realized* pricing error at any single
`M` is one noisy draw from a random process, not an ensemble average,
so it does not have to decrease monotonically at every step — only on
average, and only asymptotically at the textbook rate.

## Put-Call Parity and No-Arbitrage Bounds

```
C − P = S0 − K·e^{-rT}
```

holds essentially exactly for Black-Scholes (floating-point only) but
only within a tolerance related to the **combined sampling error of
both legs** for Monte Carlo — two independent Monte Carlo estimates
won't cancel to zero, so `check_put_call_parity` is meant to be called
with a tolerance like `4 * (SE_call + SE_put)`, not near-zero.
No-arbitrage bounds (`C ≥ max(S0 − Ke^{-rT}, 0)`, `C ≤ S0`, and the put
equivalents) are checked explicitly and flagged rather than silently
accepted if violated.

## Moneyness and Parameter Sensitivity

`OptionParameters.classify_moneyness()` labels an option
DEEP_ITM/ITM/ATM/OTM/DEEP_OTM from `S0/K`. `sweep_parameter` reprices
an option across a grid of one parameter (spot, strike, volatility,
rate, or maturity) at a time — this is **explicitly not a Greeks
engine**: it re-prices from scratch at each discrete value rather than
computing an analytical derivative, though `black_scholes.py`'s
internal `d1`/`d2` are exactly the building blocks a future Greeks
module (Delta = `N(d1)` for a call, etc.) would need.

One practical finding from the moneyness sweep: **deep out-of-the-money
options show much larger *percentage* pricing error** than at-the-money
options at the same simulation count, even though their *absolute*
error is small — a deep OTM option's fair price is itself tiny, so the
same absolute Monte Carlo noise translates into a large relative error
(see `week2_derivatives_engine.ipynb`, Section 16).

## Reporting

`generate_pricing_report` produces a structured `OptionPricingReport`
(price, comparison, bounds check) for a single option.
`generate_paired_pricing_report` prices a call and a put on the same
underlying/strike/maturity together and adds the put-call parity
check that a single-leg report can't compute alone — its `__str__`
matches the dashboard format from the Week 2 spec.

## Architecture: Ready for Path-Dependent Options and Greeks

European payoffs depend only on `S_T`, so this module deliberately
never generates full paths. **Asian options** (payoff depends on the
path average `S̄`) and **Barrier options** (payoff depends on whether
`S_t` crosses a level during the path) will need Week 1's `mode="full"`
GBM output instead — that mode was preserved specifically so a future
module can request it without changing `simulate_gbm`'s API. Likewise,
`black_scholes.py` already exposes the `d1`/`d2` intermediate values a
Greeks module would differentiate analytically, rather than requiring
Week 3 to re-derive them.

## Model Validation vs. Real-World Forecasting

To repeat the point made above because it's easy to lose sight of:
**Monte Carlo matching Black-Scholes in this codebase demonstrates that
the implementation correctly solves the risk-neutral GBM pricing
problem — it is not evidence that GBM/Black-Scholes accurately predict
real market option prices.** Real markets exhibit volatility smiles,
jumps, and stochastic volatility, none of which this model captures.

## Roadmap (Future Modules, Not Yet Implemented)

- **Greeks**: Delta, Gamma, Vega, Theta, Rho — analytically from
  `black_scholes.py`'s existing `d1`/`d2`, and/or via pathwise or
  finite-difference estimators for the Monte Carlo pricer.
- **Asian and Barrier options**: path-dependent payoffs built on Week
  1's `mode="full"` GBM output rather than terminal-only simulation.

These are designed to sit on top of existing APIs (`simulate_gbm`,
`black_scholes.py`, `price_option_monte_carlo`) without modifying them.
Portfolio risk (correlated multi-asset GBM, VaR/ES, stress testing) is
implemented in Week 3, below.

---

# Week 3 — Portfolio Risk Engine

Builds correlated multi-asset simulation and portfolio-level risk
analytics on top of Week 1's GBM engine: each asset still follows its
own exact GBM terminal-price formula, but the standard normal draws
feeding different assets are correlated before being applied, and
portfolio value/P&L, VaR, Expected Shortfall, and scenario stress tests
are all computed from the resulting joint distribution.

## Correlated Multi-Asset GBM

`correlated_gbm.py` induces the target correlation via Cholesky
decomposition of the correlation matrix, `Sigma = L L^T`:

```
Z ~ N(0, I)  (n independent standard normals per path)
Z_corr = Z @ L^T                    →  Corr(Z_corr) = Sigma
S_T^(i) = S0^(i) * exp[(mu^(i) - 0.5*sigma^(i,2))*T + sigma^(i)*sqrt(T)*Z_corr^(i)]
```

Correlating the underlying standard normals — rather than correlating
prices directly — leaves each asset's own marginal GBM distribution
exactly as it would be simulated independently (Week 1); only the
*co-movement* between assets changes. `PortfolioParameters` validates
that the correlation matrix is symmetric, unit-diagonal, bounded in
[-1, 1], and positive semi-definite, since a non-PSD matrix has no real
Cholesky factor and is not a valid correlation structure.
`realized_correlation` lets a caller check the induced correlation
against the target — like simulated-vs-theoretical moment checks in
Week 1, this converges as simulations increase but shows finite-sample
deviation at any fixed count.

## Portfolio Value and P&L Simulation

`portfolio_simulation.py` converts each asset's dollar allocation
(`weight * portfolio_value`) into a fixed share count at `t=0`, then
revalues those shares at the simulated `S_T` — a static buy-and-hold
portfolio over the horizon, not a dynamically rebalanced one. This
produces `terminal_value`, `pnl = terminal_value - initial_value`, and
`pnl_pct`, all of shape `(simulations,)`, which every risk metric below
is computed from.

## Value at Risk and Expected Shortfall

Two independent methods, computed from the same simulated P&L sample —
the same "Monte Carlo vs. closed-form benchmark" relationship as
Monte Carlo pricing vs. Black-Scholes in Week 2:

- **Historical (simulation-based)**: VaR is the loss at the
  `(1 - confidence_level)` percentile of simulated P&L; ES is the mean
  loss *beyond* that threshold. Makes no distributional assumption
  beyond what the GBM engine already assumes.
- **Parametric (variance-covariance)**: closed-form Normal-distribution
  formulas —
  `VaR = -(mean + z_alpha * std)`,
  `ES = -(mean - std * phi(z_alpha) / alpha)`
  — included as an analytical cross-check, expected to diverge from the
  historical method since simulated portfolio P&L (a weighted sum of
  log-normal terms) is not exactly Normal, particularly at high
  confidence levels where tail shape matters most.

Both are reported as **positive numbers representing a loss** (the
standard risk-reporting convention: "95% VaR is $2.1M" means a loss).
`ES >= VaR` holds by construction at every confidence level tested,
since ES averages every outcome at or beyond the VaR threshold.

## Stress Testing and Scenario Analysis

`stress_testing.py`'s `Scenario` shocks a portfolio's spot prices,
volatility, drift, and/or correlation *before* simulation, reusing the
same simulation and risk code rather than a separate stress-specific
path — matching the "layer on top, don't restructure" pattern Week 2
used on Week 1. Four scenario types match the project spec (market
decline, increased volatility, interest-rate change, and correlation
change), with a small `STANDARD_SCENARIOS` library of ready-to-use
combinations (e.g. a severe crash bundles a price shock, a volatility
spike, and rising correlations, matching how real crises behave).

**A subtlety the implementation specifically guards against:** a
market-decline scenario must revalue the *same* share holdings at a
lower price. If share counts were instead re-derived from
`dollar_allocation / shocked_price` for the stressed leg (the naive
approach), the same dollar budget would simply buy proportionally more
shares at the cheaper price, and the price shock would cancel out of
the terminal-value formula entirely — silently producing a "zero-impact
market crash." `run_stress_test` avoids this by computing share counts
once, from the *baseline* portfolio, and reusing them for both the
baseline and stressed simulations; only the simulated price *paths*
differ between the two legs. Both legs also share the same random seed,
so differences are attributable to the scenario's shocks rather than
unrelated Monte Carlo sampling noise (the common-random-numbers
discipline from Week 2's convergence studies).

## Reporting

`generate_portfolio_risk_report` bundles a simulation and both VaR/ES
methods into one structured `PortfolioRiskReport` with a formatted
dashboard `__str__`, mirroring `generate_paired_pricing_report` from
Week 2.

## Model Limitations (Portfolio Layer)

Everything in Week 1's "GBM/Black-Scholes limitations" section applies
per-asset here too. Additionally:

- **Linear correlation only.** Cholesky-induced correlation captures
  linear co-movement; it does not model tail dependence (assets
  crashing together more than their correlation alone would predict) —
  `STANDARD_SCENARIOS["market_decline_severe"]` approximates this
  qualitatively by also shocking correlation upward, but that is a
  scenario choice, not something the base model produces endogenously.
- **Static buy-and-hold.** No rebalancing, no interim cash flows.
- **Terminal-only risk.** VaR/ES are computed on the terminal (`T`)
  P&L distribution, not on interim drawdown paths — Week 1's
  `mode="full"` output would be needed to extend this to path-based
  risk (e.g. maximum drawdown).
- **Stress scenarios are user-specified, not model-implied.** The
  platform does not (yet) estimate scenario shock sizes from historical
  crisis data; `STANDARD_SCENARIOS` are illustrative starting points.

## Architecture: Ready for Week 4

Every Week 3 function is a plain, headless Python call
(`simulate_portfolio`, `historical_risk_metrics`,
`run_stress_test`) with no plotting or web dependencies, so Week 4's
FastAPI dashboard can wrap these directly as endpoints without any
restructuring — the same reason Week 1/2's quant code was kept separate
from `app/visualization` from the start.

## Roadmap (Future Modules, Not Yet Implemented)

- **FastAPI dashboard (Week 4)**: expose Week 1–3 engines as API
  endpoints with interactive Plotly/JS charts.
- **Live market-data calibration**: estimate `AssetParameters.drift`/
  `volatility` and the correlation matrix from historical returns
  (yfinance) rather than assumed inputs, reusing Week 1's
  `volatility.py`.
- **Path-based risk**: maximum drawdown and interim VaR using
  `mode="full"` path output.
- **Greeks-aware portfolio risk** for portfolios holding options,
  building on Week 2's `black_scholes.py`.
