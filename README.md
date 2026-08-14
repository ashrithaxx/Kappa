# Monte Carlo Risk & Derivatives Platform

A quantitative analytics platform for **Monte Carlo simulation, derivatives pricing, and portfolio risk analysis**.

## What it does

* **Monte Carlo Simulation** — GBM asset-price simulation, volatility estimation, statistical analysis, confidence intervals, and convergence checks.
* **Derivatives Pricing** — European call/put pricing using Monte Carlo and Black-Scholes, with pricing-error and no-arbitrage validation.
* **Portfolio Risk** — Correlated multi-asset simulation, portfolio P&L, VaR, Expected Shortfall, and stress testing.

The platform is built around a reusable quantitative engine, with simulation and risk logic separated from visualization.

## Project Structure

```text
monte_carlo_platform/
├── app/
│   ├── models/              # Input and configuration models
│   ├── quant/
│   │   ├── gbm.py           # GBM simulation
│   │   ├── derivatives/     # Option pricing
│   │   └── portfolio/       # Portfolio risk
│   ├── visualization/       # Charts and visualizations
│   └── utils/               # Utilities
├── notebooks/               # Quantitative walkthroughs
├── tests/                   # Automated tests
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Quick Start

Run a Monte Carlo simulation:

```python
from app.models.market_parameters import MarketParameters
from app.models.simulation_parameters import SimulationParameters
from app.quant.monte_carlo import run_simulation

market = MarketParameters(
    initial_price=100,
    drift=0.08,
    volatility=0.20,
    time_horizon=1.0
)

params = SimulationParameters(
    steps=252,
    simulations=100_000,
    seed=42
)

result = run_simulation(market, params)
```

The same engine supports **European option pricing** and **multi-asset portfolio risk analysis**.

## Validation

The project includes **127 automated tests** covering simulation correctness, reproducibility, convergence, option pricing, portfolio risk metrics, and stress testing.

```bash
PYTHONPATH=. pytest tests/ -v
```

## Tech Stack

**Python · NumPy · SciPy · Pandas · Matplotlib · Plotly · Pydantic · Pytest**
