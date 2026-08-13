"""
Centralized, reproducible random number generation.

The platform never touches the *global* NumPy random state. Every
simulation explicitly creates (or receives) its own
``numpy.random.Generator`` instance, so identical inputs always produce
identical outputs and multiple simulations can run concurrently without
interfering with one another.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]


def get_rng(seed: Seed = None) -> np.random.Generator:
    """Return a NumPy ``Generator`` for the given seed.

    Parameters
    ----------
    seed:
        - ``None``: a fresh, non-reproducible generator (entropy from the OS).
        - ``int``: a reproducible generator seeded deterministically.
        - ``np.random.SeedSequence``: passed straight through.
        - ``np.random.Generator``: returned unchanged, so callers can share
          state deliberately if they choose to.

    Returns
    -------
    numpy.random.Generator
        A PCG64-backed generator (NumPy's modern, recommended default).
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
