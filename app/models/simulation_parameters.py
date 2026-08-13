"""
Simulation-control parameter model (as opposed to market data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SimulationMode = Literal["full", "terminal"]


@dataclass(frozen=True)
class SimulationParameters:
    """Numerical controls for a Monte Carlo run.

    Attributes
    ----------
    steps:
        Number of time steps, N, dividing ``time_horizon`` into N
        intervals of size dt = T / N. Must be a positive integer.
    simulations:
        Number of independent simulated paths, M. Must be a positive
        integer.
    seed:
        Optional seed for reproducibility. ``None`` means
        non-reproducible (OS entropy).
    mode:
        - ``"full"``: store the entire (steps+1, simulations) price
          matrix. Needed for path plots, path-dependent payoffs, etc.
        - ``"terminal"``: sample only the terminal prices, S_T,
          directly (via Brownian self-similarity) at O(simulations)
          time and memory, completely independent of ``steps``.
          Preferred for large M — e.g. convergence studies up to
          1,000,000+ simulations — when only the terminal distribution
          is needed and intermediate path shape doesn't matter.
    """

    steps: int
    simulations: int
    seed: Optional[int] = None
    mode: SimulationMode = "full"

    def __post_init__(self) -> None:
        if self.steps <= 0 or not isinstance(self.steps, int):
            raise ValueError(f"steps (N) must be a positive integer, got {self.steps}")
        if self.simulations <= 0 or not isinstance(self.simulations, int):
            raise ValueError(
                f"simulations (M) must be a positive integer, got {self.simulations}"
            )
        if self.mode not in ("full", "terminal"):
            raise ValueError(f"mode must be 'full' or 'terminal', got {self.mode!r}")

    @property
    def dt_fraction_of(self) -> float:
        """Convenience: 1 / steps (multiply by T elsewhere to get dt)."""
        return 1.0 / self.steps
