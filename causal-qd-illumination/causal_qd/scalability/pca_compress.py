"""PCA-based descriptor dimensionality reduction.

Provides:
  - PCACompressor: project descriptors onto principal components
  - Incremental PCA for streaming settings
  - Variance-explained threshold for automatic dimension selection
  - Reconstruction error monitoring
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from causal_qd.types import BehavioralDescriptor

logger = logging.getLogger(__name__)


class PCACompressor:
    """Project behavioural descriptors onto their top principal components.

    Supports both batch PCA and incremental PCA for streaming settings.

    Parameters
    ----------
    n_components : int or None
        Number of PCA components to retain.  If *None*, use
        ``variance_threshold`` to determine automatically.
    variance_threshold : float
        Minimum cumulative explained variance ratio (default 0.95).
        Only used when ``n_components`` is *None*.
    """

    def __init__(
        self,
        n_components: Optional[int] = 2,
        variance_threshold: float = 0.95,
    ) -> None:
        self._n_components = n_components
        self._variance_threshold = variance_threshold
        # Fitted state
        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._singular_values: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None
        self._total_variance: float = 0.0
        self._n_samples_seen: int = 0
        # Incremental PCA state
        self._running_sum: Optional[np.ndarray] = None
        self._running_sq_sum: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Batch PCA
    # ------------------------------------------------------------------

    def fit(self, descriptors: List[BehavioralDescriptor]) -> None:
        """Fit PCA on a collection of descriptors.

        Parameters
        ----------
        descriptors :
            List of 1-D descriptor vectors (all same length).
        """
        X = np.array(descriptors, dtype=np.float64)
        n_samples, n_features = X.shape
        self._mean = X.mean(axis=0)
        centered = X - self._mean

        # Full SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Explained variance
        explained_var = (S ** 2) / (n_samples - 1)
        total_var = explained_var.sum()
        self._total_variance = float(total_var)

        # Determine number of components
        if self._n_components is not None:
            k = min(self._n_components, len(S))
        else:
            # Use variance threshold
            ratio_cumsum = np.cumsum(explained_var) / total_var
            k = int(np.searchsorted(ratio_cumsum, self._variance_threshold) + 1)
            k = min(k, len(S))

        self._components = Vt[:k]
        self._singular_values = S[:k]
        self._explained_variance = explained_var[:k]
        self._explained_variance_ratio = explained_var[:k] / max(total_var, 1e-12)
        self._n_samples_seen = n_samples

        logger.info(
            "PCA fit: %d -> %d components, explained variance ratio = %.4f",
            n_features, k, float(self._explained_variance_ratio.sum()),
        )

    def transform(self, descriptor: BehavioralDescriptor) -> BehavioralDescriptor:
        """Project a single descriptor into the low-dimensional space.

        Parameters
        ----------
        descriptor :
            Original high-dimensional descriptor.

        Returns
        -------
        BehavioralDescriptor
            Reduced descriptor.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._mean is None or self._components is None:
            raise RuntimeError("PCACompressor has not been fitted yet.")
        centered = np.asarray(descriptor, dtype=np.float64) - self._mean
        return (self._components @ centered).astype(np.float64)

    def transform_batch(
        self, descriptors: List[BehavioralDescriptor],
    ) -> List[BehavioralDescriptor]:
        """Transform multiple descriptors at once.

        Parameters
        ----------
        descriptors :
            List of descriptors.

        Returns
        -------
        list of BehavioralDescriptor
        """
        if self._mean is None or self._components is None:
            raise RuntimeError("PCACompressor has not been fitted yet.")
        X = np.array(descriptors, dtype=np.float64)
        centered = X - self._mean
        projected = centered @ self._components.T
        return [projected[i] for i in range(projected.shape[0])]

    def inverse_transform(self, reduced: BehavioralDescriptor) -> BehavioralDescriptor:
        """Reconstruct a descriptor from its reduced representation.

        Parameters
        ----------
        reduced :
            Low-dimensional descriptor.

        Returns
        -------
        BehavioralDescriptor
            Reconstructed high-dimensional descriptor.
        """
        if self._mean is None or self._components is None:
            raise RuntimeError("PCACompressor has not been fitted yet.")
        return (self._components.T @ reduced + self._mean).astype(np.float64)

    def fit_transform(
        self, descriptors: List[BehavioralDescriptor],
    ) -> List[BehavioralDescriptor]:
        """Fit and transform in one step.

        Parameters
        ----------
        descriptors :
            List of descriptors.

        Returns
        -------
        list of BehavioralDescriptor
        """
        self.fit(descriptors)
        return self.transform_batch(descriptors)

    # ------------------------------------------------------------------
    # Incremental PCA
    # ------------------------------------------------------------------

    def partial_fit(self, descriptors: List[BehavioralDescriptor]) -> None:
        """Incrementally update PCA with new descriptors.

        Uses the incremental SVD update to incorporate new data without
        re-processing all previous data.

        Parameters
        ----------
        descriptors :
            New batch of descriptors to incorporate.
        """
        X_new = np.array(descriptors, dtype=np.float64)
        n_new, d = X_new.shape

        if self._mean is None:
            # First batch: do full fit
            self.fit(descriptors)
            self._running_sum = X_new.sum(axis=0)
            self._running_sq_sum = (X_new ** 2).sum(axis=0)
            return

        # Update running statistics
        if self._running_sum is None:
            self._running_sum = self._mean * self._n_samples_seen
            self._running_sq_sum = np.zeros(d, dtype=np.float64)

        old_n = self._n_samples_seen
        new_n = old_n + n_new
        self._running_sum += X_new.sum(axis=0)
        new_mean = self._running_sum / new_n

        # Center new data with the overall mean
        centered_new = X_new - new_mean

        # If we have existing components, combine
        if self._components is not None:
            # Project centered data onto current components
            # and compute residuals
            centered_old_mean_diff = (self._mean - new_mean) * np.sqrt(old_n)
            augmented = np.vstack([centered_new, centered_old_mean_diff.reshape(1, -1)])

            # Update via SVD of the augmented matrix projected through
            # current components
            k = self._components.shape[0]
            # Simple approach: refit on projected + new
            old_projected = np.sqrt(
                np.maximum(self._explained_variance * (old_n - 1), 0)
            ).reshape(-1, 1) * self._components

            combined = np.vstack([old_projected, centered_new])
            _, S, Vt = np.linalg.svd(combined, full_matrices=False)

            n_comp = self._n_components if self._n_components is not None else k
            n_comp = min(n_comp, len(S))

            self._components = Vt[:n_comp]
            self._singular_values = S[:n_comp]
            self._explained_variance = (S[:n_comp] ** 2) / max(new_n - 1, 1)

        self._mean = new_mean
        self._n_samples_seen = new_n

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def reconstruction_error(self, descriptor: BehavioralDescriptor) -> float:
        """Compute reconstruction error for a single descriptor.

        Parameters
        ----------
        descriptor :
            Original descriptor.

        Returns
        -------
        float
            L2 reconstruction error.
        """
        reduced = self.transform(descriptor)
        reconstructed = self.inverse_transform(reduced)
        return float(np.linalg.norm(descriptor - reconstructed))

    def mean_reconstruction_error(
        self, descriptors: List[BehavioralDescriptor],
    ) -> float:
        """Mean reconstruction error over a batch of descriptors.

        Parameters
        ----------
        descriptors :
            List of descriptors.

        Returns
        -------
        float
        """
        errors = [self.reconstruction_error(d) for d in descriptors]
        return float(np.mean(errors)) if errors else 0.0

    @property
    def n_components_fitted(self) -> int:
        """Number of components in the fitted model."""
        if self._components is None:
            return 0
        return self._components.shape[0]

    @property
    def explained_variance_ratio_total(self) -> float:
        """Total explained variance ratio."""
        if self._explained_variance_ratio is None:
            return 0.0
        return float(self._explained_variance_ratio.sum())

    def summary(self) -> dict:
        """Return a summary of the fitted PCA model."""
        return {
            "n_components": self.n_components_fitted,
            "explained_variance_ratio": self.explained_variance_ratio_total,
            "n_samples_seen": self._n_samples_seen,
            "total_variance": self._total_variance,
        }
