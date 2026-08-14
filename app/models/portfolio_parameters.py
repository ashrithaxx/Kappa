"""
Multi-asset portfolio parameter model.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AssetParameters:
    """Market-side inputs for one asset inside a portfolio.

    Same fields as ``MarketParameters`` minus ``risk_free_rate`` (that
    is a portfolio-level, not asset-level, input here) plus a display
    ``name``.
    """

    name: str
    initial_price: float
    drift: float
    volatility: float

    def __post_init__(self) -> None:
        if self.initial_price <= 0:
            raise ValueError(
                f"initial_price (S0) for {self.name!r} must be > 0, "
                f"got {self.initial_price}"
            )
        if self.volatility < 0:
            raise ValueError(
                f"volatility (sigma) for {self.name!r} must be >= 0, "
                f"got {self.volatility}"
            )


@dataclass(frozen=True)
class PortfolioParameters:
    """Full specification of a multi-asset portfolio for simulation.

    Attributes
    ----------
    assets:
        Sequence of ``AssetParameters``, one per holding. Order defines
        the index used by ``weights`` and ``correlation_matrix``.
    weights:
        Portfolio weights, one per asset. Need not sum to 1 (e.g. cash
        drag or leverage) but are validated for shape only; economic
        interpretation is left to the caller.
    correlation_matrix:
        Symmetric (n_assets, n_assets) matrix with unit diagonal and
        entries in [-1, 1]. Validated for symmetry, diagonal, and
        positive semi-definiteness (required for the Cholesky
        decomposition used to induce correlation between assets).
    portfolio_value:
        Total capital allocated across the portfolio, in currency
        units. Used to convert weights into per-asset dollar exposure.
    time_horizon:
        Simulation horizon in years, T. Must be > 0.
    """

    assets: Sequence[AssetParameters]
    weights: Sequence[float]
    correlation_matrix: np.ndarray
    portfolio_value: float
    time_horizon: float

    def __post_init__(self) -> None:
        n = len(self.assets)
        if n < 1:
            raise ValueError("portfolio must contain at least one asset")
        if len(self.weights) != n:
            raise ValueError(
                f"weights length ({len(self.weights)}) must match "
                f"number of assets ({n})"
            )
        corr = np.asarray(self.correlation_matrix, dtype=float)
        if corr.shape != (n, n):
            raise ValueError(
                f"correlation_matrix must be ({n}, {n}), got {corr.shape}"
            )
        if not np.allclose(corr, corr.T, atol=1e-8):
            raise ValueError("correlation_matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValueError("correlation_matrix diagonal must be 1.0")
        if np.any(corr < -1 - 1e-8) or np.any(corr > 1 + 1e-8):
            raise ValueError("correlation_matrix entries must lie in [-1, 1]")
        eigenvalues = np.linalg.eigvalsh(corr)
        if np.min(eigenvalues) < -1e-8:
            raise ValueError(
                "correlation_matrix is not positive semi-definite "
                f"(smallest eigenvalue {np.min(eigenvalues):.6f}) — it is "
                "not a valid correlation structure and cannot be "
                "Cholesky-decomposed"
            )
        if self.portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be > 0, got {self.portfolio_value}"
            )
        if self.time_horizon <= 0:
            raise ValueError(f"time_horizon (T) must be > 0, got {self.time_horizon}")

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def dollar_allocations(self) -> np.ndarray:
        """Per-asset dollar exposure = weight * portfolio_value."""
        return np.asarray(self.weights, dtype=float) * self.portfolio_value

    @property
    def asset_names(self) -> list:
        return [a.name for a in self.assets]
