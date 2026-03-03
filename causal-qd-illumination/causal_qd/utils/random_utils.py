"""Random number generation utilities for reproducibility."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Set global random seeds for *numpy*, *random*, and (optionally) *torch*.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch  # type: ignore[import-untyped]

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a new :class:`numpy.random.Generator`.

    Parameters
    ----------
    seed:
        If *None* a non-deterministic generator is returned.
    """
    return np.random.default_rng(seed)
