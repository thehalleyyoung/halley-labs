"""Matrix utility functions for linear algebra operations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from causal_qd.types import DataMatrix


def symmetrize(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the symmetric part of a square matrix: ``(m + m.T) / 2``."""
    return (m + m.T) / 2.0


def is_symmetric(m: npt.NDArray[np.float64], *, atol: float = 1e-8) -> bool:
    """Check whether *m* is symmetric within absolute tolerance *atol*."""
    return bool(np.allclose(m, m.T, atol=atol))


def submatrix(
    m: npt.NDArray[np.float64],
    rows: Sequence[int],
    cols: Sequence[int],
) -> npt.NDArray[np.float64]:
    """Extract a sub-matrix by selecting specific rows and columns.

    Parameters
    ----------
    m:
        Input 2-D array.
    rows:
        Row indices to select.
    cols:
        Column indices to select.

    Returns
    -------
    ndarray
        Sub-matrix of shape ``(len(rows), len(cols))``.
    """
    return m[np.ix_(list(rows), list(cols))]


def scatter_matrix(data: DataMatrix) -> npt.NDArray[np.float64]:
    """Compute the scatter (un-normalised covariance) matrix of *data*.

    Parameters
    ----------
    data:
        ``N × p`` data matrix.

    Returns
    -------
    ndarray
        ``p × p`` scatter matrix ``(data - mean).T @ (data - mean)``.
    """
    centered = data - data.mean(axis=0, keepdims=True)
    return centered.T @ centered
