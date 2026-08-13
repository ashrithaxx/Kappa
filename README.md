# Monte Carlo Risk & Derivatives Platform

A production-grade quantitative platform for simulating asset price
dynamics, pricing derivatives, and quantifying portfolio risk —
all built on a single, reusable Monte Carlo core.

- **Quantitative Foundation** — Geometric Brownian Motion (GBM)
  simulation, historical volatility estimation, descriptive statistics,
  confidence intervals, convergence analysis, and visualization.
- **Derivatives Engine** — European option pricing via Monte Carlo,
  an independent Black-Scholes analytical benchmark, pricing-error and
  convergence analysis, put-call parity, no-arbitrage bounds, and
  moneyness/parameter sensitivity.
- **Portfolio Risk Engine** — correlated multi-asset GBM
  (Cholesky-decomposed correlation), portfolio value/P&L simulation,
  Value at Risk and Expected Shortfall (historical and parametric),
  and scenario-based stress testing (market decline, volatility spike,
  rate change, correlation breakdown).

Every layer is built directly on the GBM core rather than duplicating
it, and the architecture is designed so that path-dependent options
(Asian, Barrier) and a Greeks module can be added on top without
restructuring anything underneath.

## Project Structure

```text
monte_carlo_platform/
├── app/
│   ├── models/
│   │   ├── market_parameters.py       # S0, mu, sigma, T, r (validated)
│   │   ├── simulation_parameters.py   # N, M, seed, mode (validated)
│   │   ├── option_parameters.py       # S0, K, r, sigma, T, type (validated)
│   │   └── portfolio_parameters.py    # assets, weights, correlation matrix
│   ├── quant/
│   │   ├── volatility.py              # historical vol estimation
│   │   ├── distributions.py           # theoretical GBM moments
│   │   ├── gbm.py                     # vectorized GBM simulator
│   │   ├── monte_carlo.py             # engine, convergence, sanity checks
│   │   ├── statistics.py              # descriptive stats & CIs
│   │   ├── derivatives/
│   │   │   ├── payoffs.py             # call/put payoff functions
│   │   │   ├── monte_carlo_pricer.py  # risk-neutral sim, discounted payoff pricer
│   │   │   ├── black_scholes.py       # analytical call/put, d1/d2, edge cases
│   │   │   ├── pricing_error.py       # MC vs BS error, RMSE
│   │   │   ├── convergence.py         # option price convergence + rate fit
│   │   │   ├── validation.py          # put-call parity, no-arbitrage bounds
│   │   │   ├── sensitivity.py         # parameter sweeps (spot/strike/vol/r/T)
│   │   │   └── reporting.py           # structured pricing report / dashboard
│   │   └── portfolio/
│   │       ├── correlated_gbm.py      # Cholesky-correlated multi-asset GBM
│   │       ├── portfolio_simulation.py# portfolio value / P&L simulation
│   │       ├── risk_metrics.py        # VaR & Expected Shortfall (hist + parametric)
│   │       ├── stress_testing.py      # scenario shocks + baseline-vs-stressed compare
│   │       └── reporting.py           # structured portfolio risk report
│   ├── visualization/
│   │   ├── paths.py                   # path plots (matplotlib + plotly)
│   │   ├── distributions.py           # terminal / log-price histograms
│   │   ├── convergence.py             # convergence plots
│   │   ├── option_payoff.py           # call/put payoff diagrams
│   │   ├── pricing_convergence.py     # option price/error convergence
│   │   ├── pricing_comparison.py      # MC vs BS bars, moneyness/vol plots
│   │   └── portfolio.py               # P&L histogram, VaR/ES, correlation heatmap, stress plots
│   └── utils/
│       └── random_state.py            # centralized, reproducible RNG
├── tests/                             # pytest suite (127 tests)
├── notebooks/
│   ├── quant_foundations.ipynb        # GBM engine end-to-end walkthrough
│   ├── derivatives_engine.ipynb       # Derivatives pricing walkthrough
│   └── risk_analytics.ipynb           # Portfolio risk walkthrough
├── requirements.txt
└── README.md
```

Data flows strictly in one direction:

```
market/option/portfolio params → quant (gbm, volatility, pricing, risk) → visualization
```

Nothing in `app/quant` imports from `app/visualization`, so the
simulation, pricing, and risk engines can be reused headlessly (APIs,
batch jobs, dashboards) without pulling in plotting dependencies.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Simulate asset paths

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

### Price a European option

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

### Analyze portfolio risk

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

stress = run_stress_test(
    portfolio, STANDARD_SCENARIOS["market_decline_severe"],
    simulations=200_000, seed=42,
)
print(stress)
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

### Two simulation modes

- **`mode="full"`** — stores the entire `(steps+1, simulations)` price
  matrix. Needed for path plots or path-dependent payoffs.
- **`mode="terminal"`** — stores only `S_T`, i.e. `O(M)` memory instead
  of `O(N·M)`. Use this for convergence studies or when only the
  terminal distribution matters; it comfortably handles 1,000,000+
  simulations.

## Derivatives Pricing

### Physical measure vs. risk-neutral measure

Forecasting uses the **physical measure**, with a historical/expected
drift `μ`:

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
never `μ` — `OptionParameters` doesn't even have a `μ` field, making
the distinction structural rather than a convention to remember.

### Monte Carlo option pricing

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
reusing the GBM engine's terminal-mode shortcut) — no full paths are
generated, so pricing scales to 1,000,000+ simulations at O(M) memory
and sub-second runtime. `price_option_monte_carlo` returns a structured
`MonteCarloPriceResult` carrying the price, raw and discounted payoff
samples, standard error, and 90/95/99% confidence intervals — never
just a bare number.

### Black-Scholes, implemented from first principles

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

### Pricing error, confidence intervals, and convergence

For every priced option, `compare_to_black_scholes` reports signed
error, absolute error, percentage error, and whether the Black-Scholes
price falls inside the Monte Carlo confidence interval — the same
estimator-vs-market distinction from above applies: **the CI
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

### Put-call parity and no-arbitrage bounds

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

### Moneyness and parameter sensitivity

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
(see `notebooks/derivatives_engine.ipynb`).

### Reporting

`generate_pricing_report` produces a structured `OptionPricingReport`
(price, comparison, bounds check) for a single option.
`generate_paired_pricing_report` prices a call and a put on the same
underlying/strike/maturity together and adds the put-call parity
check that a single-leg report can't compute alone.

## Portfolio Risk

### Correlated multi-asset GBM

`correlated_gbm.py` induces the target correlation via Cholesky
decomposition of the correlation matrix, `Sigma = L L^T`:

```
Z ~ N(0, I)  (n independent standard normals per path)
Z_corr = Z @ L^T                    →  Corr(Z_corr) = Sigma
S_T^(i) = S0^(i) * exp[(mu^(i) - 0.5*sigma^(i,2))*T + sigma^(i)*sqrt(T)*Z_corr^(i)]
```

Correlating the underlying standard normals — rather than correlating
prices directly — leaves each asset's own marginal GBM distribution
exactly as it would be simulated independently; only the
*co-movement* between assets changes. `PortfolioParameters` validates
that the correlation matrix is symmetric, unit-diagonal, bounded in
[-1, 1], and positive semi-definite, since a non-PSD matrix has no real
Cholesky factor and is not a valid correlation structure.
`realized_correlation` lets a caller check the induced correlation
against the target — like simulated-vs-theoretical moment checks
elsewhere, this converges as simulations increase but shows
finite-sample deviation at any fixed count.

### Portfolio value and P&L simulation

`portfolio_simulation.py` converts each asset's dollar allocation
(`weight * portfolio_value`) into a fixed share count at `t=0`, then
revalues those shares at the simulated `S_T` — a static buy-and-hold
portfolio over the horizon, not a dynamically rebalanced one. This
produces `terminal_value`, `pnl = terminal_value - initial_value`, and
`pnl_pct`, all of shape `(simulations,)`, which every risk metric below
is computed from.

### Value at Risk and Expected Shortfall

Two independent methods, computed from the same simulated P&L sample —
the same "Monte Carlo vs. closed-form benchmark" relationship as
Monte Carlo pricing vs. Black-Scholes above:

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

### Stress testing and scenario analysis

`stress_testing.py`'s `Scenario` shocks a portfolio's spot prices,
volatility, drift, and/or correlation *before* simulation, reusing the
same simulation and risk code rather than a separate stress-specific
path. Four scenario types are supported (market decline, increased
volatility, interest-rate change, and correlation change), with a
small `STANDARD_SCENARIOS` library of ready-to-use combinations (e.g.
a severe crash bundles a price shock, a volatility spike, and rising
correlations, matching how real crises behave).

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
differ between the two legs. Both legs also share the same random
seed, so differences are attributable to the scenario's shocks rather
than unrelated Monte Carlo sampling noise (the same common-random-
numbers discipline used in the convergence studies above).

### Reporting

`generate_portfolio_risk_report` bundles a simulation and both VaR/ES
methods into one structured `PortfolioRiskReport` with a formatted
dashboard `__str__`, mirroring `generate_paired_pricing_report` for
derivatives.

## Architecture: Built to Extend

Every function across the platform is a plain, headless Python call
with no plotting or web dependencies — quant code is kept separate
from `app/visualization` throughout, so any layer can be reused in
APIs, batch jobs, or a future dashboard without restructuring.

- **Path-dependent options** (Asian, Barrier) and a **Greeks module**
  (Delta, Gamma, Vega, Theta, Rho) sit naturally on top of the existing
  GBM (`mode="full"`) and Black-Scholes (`d1`/`d2`) building blocks.
- **Live market-data calibration** can estimate `AssetParameters.drift`
  /`volatility` and the correlation matrix from historical returns,
  reusing the existing `volatility.py`.
- **Path-based risk** (maximum drawdown, interim VaR) can be built
  directly on `mode="full"` path output.
- **A FastAPI dashboard** can expose every engine here as an endpoint
  with interactive Plotly/JS charts, since nothing in `app/quant`
  depends on how it's called.

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
- **Linear correlation only** (portfolio layer) — Cholesky-induced
  correlation captures linear co-movement; it does not model tail
  dependence (assets crashing together more than their correlation
  alone would predict). `STANDARD_SCENARIOS["market_decline_severe"]`
  approximates this qualitatively by also shocking correlation upward,
  but that is a scenario choice, not something the base model produces
  endogenously.
- **Static buy-and-hold** (portfolio layer) — no rebalancing, no
  interim cash flows.
- **Terminal-only risk** (portfolio layer) — VaR/ES are computed on the
  terminal (`T`) P&L distribution, not on interim drawdown paths.
- **Stress scenarios are user-specified, not model-implied** — shock
  sizes aren't (yet) estimated from historical crisis data;
  `STANDARD_SCENARIOS` are illustrative starting points.

These are known, standard simplifications; the sanity checks and tests
in this package validate that the *implementation* is correct given
these assumptions, not that the assumptions themselves match real
markets. **Black-Scholes matching Monte Carlo validates internal
consistency of this codebase — it is not evidence that either model
accurately forecasts real-world option prices.**

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

127 tests cover: return calculations, GBM correctness and positivity,
shape/reproducibility guarantees, statistics correctness against
NumPy/SciPy, confidence-interval coverage, and Monte Carlo convergence
behavior; payoff correctness, Black-Scholes benchmark values and edge
cases, Monte Carlo option pricing (discounting, reproducibility,
risk-neutral drift), pricing-error/RMSE, put-call parity, no-arbitrage
bounds, and option-pricing convergence rate; portfolio parameter
validation (symmetry, PSD, weight/asset alignment), induced correlation
accuracy, portfolio P&L consistency, VaR/ES correctness (quantile
definition, ES >= VaR, monotonicity in confidence level,
historical-vs-parametric agreement on Normal data), and stress-test
scenario mechanics including the fixed-share-count invariant that
keeps a price shock from silently canceling out.
