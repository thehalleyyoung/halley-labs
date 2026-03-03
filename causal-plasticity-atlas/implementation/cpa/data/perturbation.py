"""Data perturbation for robustness testing.

Applies controlled perturbations (noise injection, missing data,
outliers, permutation, scaling, mean shift, bootstrap) to
observational datasets to evaluate the robustness of causal
discovery methods.

Provides:

* :class:`PerturbationType` – enum of supported perturbation kinds.
* :class:`DataPerturbation` – main perturbation engine with
  per-sample and batch perturbation support.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# Enum
# ===================================================================


class PerturbationType(Enum):
    """Supported data perturbation types."""

    GAUSSIAN_NOISE = "gaussian_noise"
    MISSING_DATA = "missing_data"
    OUTLIERS = "outliers"
    PERMUTATION = "permutation"
    SCALING = "scaling"
    MEAN_SHIFT = "mean_shift"
    SUBSAMPLE = "subsample"
    BOOTSTRAP = "bootstrap"


# ===================================================================
# DataPerturbation
# ===================================================================


class DataPerturbation:
    """Apply perturbations to data arrays for robustness evaluation.

    Parameters
    ----------
    perturbation_type : PerturbationType or None
        Default perturbation type.  If ``None``, must be specified
        per call.
    severity : float
        Default perturbation strength in ``[0, 1]``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        perturbation_type: Optional[PerturbationType] = None,
        severity: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self._perturbation_type = perturbation_type
        self._severity = severity
        self._rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    # Primary dispatch
    # -----------------------------------------------------------------

    def perturb(
        self,
        data: NDArray,
        perturbation_type: Optional[PerturbationType] = None,
        severity: Optional[float] = None,
    ) -> NDArray:
        """Apply a perturbation to a data matrix.

        Parameters
        ----------
        data : NDArray, shape (n, p)
        perturbation_type : PerturbationType or None
            Overrides the instance default.
        severity : float or None
            Overrides the instance default.

        Returns
        -------
        NDArray
            Perturbed data matrix.
        """
        ptype = perturbation_type or self._perturbation_type
        sev = severity if severity is not None else self._severity

        if ptype is None:
            raise ValueError("perturbation_type must be specified")

        dispatch = {
            PerturbationType.GAUSSIAN_NOISE: self.add_noise,
            PerturbationType.MISSING_DATA: self.missing_values,
            PerturbationType.OUTLIERS: self.outliers,
            PerturbationType.PERMUTATION: self._permute_all,
            PerturbationType.SCALING: self._scale_all,
            PerturbationType.MEAN_SHIFT: self._shift_all,
            PerturbationType.SUBSAMPLE: self.subsample,
            PerturbationType.BOOTSTRAP: self.bootstrap_resample,
        }
        fn = dispatch.get(ptype)
        if fn is None:
            raise ValueError(f"Unsupported perturbation type: {ptype}")

        if ptype in (PerturbationType.SUBSAMPLE, PerturbationType.BOOTSTRAP):
            return fn(data, fraction=sev)
        return fn(data, sev)

    # -----------------------------------------------------------------
    # Individual perturbation methods
    # -----------------------------------------------------------------

    def add_noise(
        self,
        data: NDArray,
        noise_level: float = 0.1,
    ) -> NDArray:
        """Add independent Gaussian noise to each element.

        Parameters
        ----------
        data : NDArray, shape (n, p)
        noise_level : float
            Standard deviation of the noise *relative* to each
            column's standard deviation.

        Returns
        -------
        NDArray
        """
        data = np.array(data, dtype=np.float64, copy=True)
        col_std = np.std(data, axis=0)
        col_std[col_std < 1e-12] = 1.0
        noise = self._rng.normal(0, 1, size=data.shape) * col_std * noise_level
        return data + noise

    def missing_values(
        self,
        data: NDArray,
        fraction: float = 0.1,
    ) -> NDArray:
        """Introduce missing values (NaN) at random positions.

        Parameters
        ----------
        data : NDArray
        fraction : float
            Fraction of entries to make missing.

        Returns
        -------
        NDArray
            Copy of data with NaN at missing positions.
        """
        data = np.array(data, dtype=np.float64, copy=True)
        mask = self._rng.random(data.shape) < fraction
        data[mask] = np.nan
        return data

    def outliers(
        self,
        data: NDArray,
        fraction: float = 0.05,
        magnitude: float = 5.0,
    ) -> NDArray:
        """Add outliers to random positions.

        Outlier values are drawn from ``magnitude * std`` away from
        the column mean.

        Parameters
        ----------
        data : NDArray
        fraction : float
            Fraction of entries to corrupt.
        magnitude : float
            Number of standard deviations for outlier displacement.

        Returns
        -------
        NDArray
        """
        data = np.array(data, dtype=np.float64, copy=True)
        n, p = data.shape
        col_std = np.std(data, axis=0)
        col_std[col_std < 1e-12] = 1.0
        col_mean = np.mean(data, axis=0)

        n_outliers = max(1, int(n * p * fraction))
        rows = self._rng.integers(0, n, size=n_outliers)
        cols = self._rng.integers(0, p, size=n_outliers)
        signs = self._rng.choice([-1.0, 1.0], size=n_outliers)

        for idx in range(n_outliers):
            r, c = rows[idx], cols[idx]
            data[r, c] = col_mean[c] + signs[idx] * magnitude * col_std[c]

        return data

    def subsample(
        self,
        data: NDArray,
        fraction: float = 0.5,
    ) -> NDArray:
        """Random subsampling without replacement.

        Parameters
        ----------
        data : NDArray
        fraction : float
            Fraction of rows to keep.

        Returns
        -------
        NDArray
        """
        n = data.shape[0]
        k = max(1, int(n * fraction))
        indices = self._rng.choice(n, size=k, replace=False)
        return data[np.sort(indices)].copy()

    def permute_column(
        self,
        data: NDArray,
        column: int,
    ) -> NDArray:
        """Permute a single column to break dependence.

        Parameters
        ----------
        data : NDArray
        column : int
            Column index to permute.

        Returns
        -------
        NDArray
        """
        data = np.array(data, dtype=np.float64, copy=True)
        data[:, column] = self._rng.permutation(data[:, column])
        return data

    def shift_mean(
        self,
        data: NDArray,
        column: int,
        shift: float,
    ) -> NDArray:
        """Shift the mean of a column (distribution shift).

        Parameters
        ----------
        data : NDArray
        column : int
        shift : float
            Additive shift in units of column standard deviation.

        Returns
        -------
        NDArray
        """
        data = np.array(data, dtype=np.float64, copy=True)
        col_std = np.std(data[:, column])
        if col_std < 1e-12:
            col_std = 1.0
        data[:, column] += shift * col_std
        return data

    def scale_variance(
        self,
        data: NDArray,
        column: int,
        scale: float,
    ) -> NDArray:
        """Scale the variance of a column.

        Parameters
        ----------
        data : NDArray
        column : int
        scale : float
            Multiplicative scaling factor for the column's deviation
            from its mean.

        Returns
        -------
        NDArray
        """
        data = np.array(data, dtype=np.float64, copy=True)
        col_mean = np.mean(data[:, column])
        data[:, column] = col_mean + scale * (data[:, column] - col_mean)
        return data

    def bootstrap_resample(
        self,
        data: NDArray,
        fraction: float = 1.0,
    ) -> NDArray:
        """Bootstrap resample (sampling with replacement).

        Parameters
        ----------
        data : NDArray
        fraction : float
            Fraction of original size for the resample.

        Returns
        -------
        NDArray
        """
        n = data.shape[0]
        k = max(1, int(n * fraction))
        indices = self._rng.choice(n, size=k, replace=True)
        return data[indices].copy()

    # -----------------------------------------------------------------
    # SCM mechanism perturbation
    # -----------------------------------------------------------------

    def perturb_mechanism(
        self,
        scm: Any,
        node: int,
        severity: float = 0.1,
    ) -> None:
        """Perturb a specific mechanism inside an SCM in-place.

        Adds Gaussian noise to the regression coefficients of the
        given node's parents.

        Parameters
        ----------
        scm : StructuralCausalModel
        node : int
        severity : float
        """
        coefs = scm.regression_coefficients
        parents = np.where(coefs[:, node] != 0)[0]
        for pa in parents:
            coefs[pa, node] += self._rng.normal(0, severity * abs(coefs[pa, node]) + 0.01)

    # -----------------------------------------------------------------
    # Batch perturbation
    # -----------------------------------------------------------------

    def batch_perturb(
        self,
        data: NDArray,
        n_perturbations: int,
    ) -> List[NDArray]:
        """Generate multiple perturbed copies of the data.

        Parameters
        ----------
        data : NDArray
        n_perturbations : int

        Returns
        -------
        list of NDArray
        """
        return [self.perturb(data) for _ in range(n_perturbations)]

    # -----------------------------------------------------------------
    # Helpers for dispatch
    # -----------------------------------------------------------------

    def _permute_all(self, data: NDArray, fraction: float = 0.1) -> NDArray:
        """Permute a fraction of columns."""
        data = np.array(data, dtype=np.float64, copy=True)
        p = data.shape[1]
        n_cols = max(1, int(p * fraction))
        cols = self._rng.choice(p, size=n_cols, replace=False)
        for c in cols:
            data[:, c] = self._rng.permutation(data[:, c])
        return data

    def _scale_all(self, data: NDArray, scale: float = 0.1) -> NDArray:
        """Scale variance of all columns by ``1 + scale``."""
        data = np.array(data, dtype=np.float64, copy=True)
        means = np.mean(data, axis=0)
        data = means + (1.0 + scale) * (data - means)
        return data

    def _shift_all(self, data: NDArray, shift: float = 0.1) -> NDArray:
        """Shift mean of all columns."""
        data = np.array(data, dtype=np.float64, copy=True)
        stds = np.std(data, axis=0)
        stds[stds < 1e-12] = 1.0
        data += shift * stds
        return data
