"""Composite behavioral descriptor that concatenates, weights, normalizes,
and optionally PCA-reduces the outputs of multiple sub-descriptors.

This module provides :class:`CompositeDescriptor`, the primary mechanism for
building rich, multi-faceted behavioral descriptors in the CausalQD framework.
It supports per-dimension weighting, several normalization strategies (z-score,
min-max, quantile), variance-based feature selection, and SVD-based PCA
dimensionality reduction.

Example
-------
>>> from causal_qd.descriptors.composite import CompositeDescriptor
>>> comp = CompositeDescriptor(
...     descriptors=[desc_a, desc_b],
...     weights=[1.0, 2.0],
...     normalization="zscore",
...     pca_dim=3,
... )
>>> bd = comp.compute(dag, data)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.descriptors.descriptor_base import DescriptorComputer
from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix

_VALID_NORMALIZATIONS = ("none", "zscore", "minmax", "quantile")
_MIN_SAMPLES_FOR_STATS = 10


class CompositeDescriptor(DescriptorComputer):
    """Combine multiple :class:`DescriptorComputer` instances into a single
    behavioral descriptor with optional weighting, normalization, feature
    selection, and PCA dimensionality reduction.

    The processing pipeline executed by :meth:`compute` is:

    1. **Concatenation** – each sub-descriptor is computed and their outputs
       are concatenated into a single raw vector.
    2. **Weighting** – each dimension of the raw vector is multiplied by
       its corresponding weight.
    3. **Feature selection** – dimensions whose running variance is below
       *min_variance* are dropped.
    4. **Normalization** – the weighted vector is normalized according to
       the chosen strategy (``"none"``, ``"zscore"``, ``"minmax"``, or
       ``"quantile"``).
    5. **PCA reduction** – if *pca_dim* is set the normalized vector is
       projected onto its top principal components via truncated SVD.

    Parameters
    ----------
    descriptors:
        Ordered sequence of :class:`DescriptorComputer` instances whose
        outputs will be concatenated.
    weights:
        Per-component weights applied element-wise to each sub-descriptor's
        output **before** normalization.  When ``None`` every dimension
        receives weight 1.0.  The total length must equal the sum of all
        sub-descriptor dimensionalities.
    normalization:
        Normalization strategy applied after weighting.  One of
        ``"none"`` (identity), ``"zscore"`` (zero-mean, unit-variance),
        ``"minmax"`` (scale to ``[0, 1]``), or ``"quantile"``
        (rank-based uniform mapping).
    pca_dim:
        If not ``None``, reduce the final descriptor to this number of
        dimensions using SVD-based PCA.  Must be ≤ the raw descriptor
        dimensionality (after feature selection).
    min_variance:
        Features whose running variance falls below this threshold are
        dropped during :meth:`_select_features`.  Set to ``0.0``
        (the default) to keep all features.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        descriptors: Sequence[DescriptorComputer],
        weights: Sequence[float] | None = None,
        normalization: str = "none",
        pca_dim: int | None = None,
        min_variance: float = 0.0,
    ) -> None:
        if not descriptors:
            raise ValueError("At least one sub-descriptor is required.")
        self._descriptors: List[DescriptorComputer] = list(descriptors)

        # Raw (pre-selection / pre-PCA) dimensionality.
        raw_dim = sum(d.descriptor_dim for d in self._descriptors)

        # Weights --------------------------------------------------------
        if weights is not None:
            if len(weights) != raw_dim:
                raise ValueError(
                    f"Length of weights ({len(weights)}) must match the total "
                    f"raw descriptor dimensionality ({raw_dim})."
                )
            self._weights: npt.NDArray[np.float64] = np.asarray(
                weights, dtype=np.float64
            )
        else:
            self._weights = np.ones(raw_dim, dtype=np.float64)

        # Normalization ---------------------------------------------------
        if normalization not in _VALID_NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization '{normalization}'. "
                f"Choose from {_VALID_NORMALIZATIONS}."
            )
        self._normalization: str = normalization

        # PCA -------------------------------------------------------------
        if pca_dim is not None and pca_dim < 1:
            raise ValueError("pca_dim must be a positive integer or None.")
        self._pca_dim: int | None = pca_dim

        # Feature selection -----------------------------------------------
        self._min_variance: float = float(min_variance)

        # Running statistics (updated incrementally) ----------------------
        self._running_mean: npt.NDArray[np.float64] | None = None
        self._running_var: npt.NDArray[np.float64] | None = None
        self._running_min: npt.NDArray[np.float64] | None = None
        self._running_max: npt.NDArray[np.float64] | None = None
        self._count: int = 0

        # Quantile normalization stores a history buffer ------------------
        self._quantile_buffer: List[npt.NDArray[np.float64]] = []

        # PCA components (set by fit or lazily during compute) ------------
        self._pca_components: npt.NDArray[np.float64] | None = None
        self._pca_mean: npt.NDArray[np.float64] | None = None

        # Feature mask (set during fit or lazily) -------------------------
        self._feature_mask: npt.NDArray[np.bool_] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _raw_dim(self) -> int:
        """Total dimensionality before feature selection and PCA."""
        return sum(d.descriptor_dim for d in self._descriptors)

    @property
    def descriptor_dim(self) -> int:
        """Final descriptor dimensionality after feature selection and PCA.

        If PCA is enabled and the model has been fitted (or enough samples
        have been seen), returns *pca_dim*.  Otherwise returns the raw
        dimension minus any features removed by variance thresholding.
        """
        if self._pca_dim is not None and self._is_fitted:
            return self._pca_dim
        if self._feature_mask is not None:
            return int(self._feature_mask.sum())
        return self._raw_dim

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-dimension lower and upper bounds.

        Before fitting, bounds are derived from the concatenated
        sub-descriptor bounds (scaled by weights).  After fitting with
        z-score or min-max normalization the bounds reflect the transformed
        space.  PCA bounds default to ``[-inf, inf]``.
        """
        lows: List[npt.NDArray[np.float64]] = []
        highs: List[npt.NDArray[np.float64]] = []
        for d in self._descriptors:
            lo, hi = d.descriptor_bounds
            lows.append(lo)
            highs.append(hi)
        raw_lo = np.concatenate(lows) * self._weights
        raw_hi = np.concatenate(highs) * self._weights

        # Apply feature mask if available.
        if self._feature_mask is not None:
            raw_lo = raw_lo[self._feature_mask]
            raw_hi = raw_hi[self._feature_mask]

        if self._normalization == "zscore" and self._is_fitted:
            dim = raw_lo.shape[0]
            return (
                np.full(dim, -5.0, dtype=np.float64),
                np.full(dim, 5.0, dtype=np.float64),
            )
        if self._normalization == "minmax" and self._is_fitted:
            dim = raw_lo.shape[0]
            return (
                np.zeros(dim, dtype=np.float64),
                np.ones(dim, dtype=np.float64),
            )
        if self._normalization == "quantile" and self._is_fitted:
            dim = raw_lo.shape[0]
            return (
                np.zeros(dim, dtype=np.float64),
                np.ones(dim, dtype=np.float64),
            )

        if self._pca_dim is not None and self._pca_components is not None:
            return (
                np.full(self._pca_dim, -np.inf, dtype=np.float64),
                np.full(self._pca_dim, np.inf, dtype=np.float64),
            )

        return raw_lo, raw_hi

    @property
    def _is_fitted(self) -> bool:
        """Whether normalization statistics have been computed.

        Returns ``True`` when at least :data:`_MIN_SAMPLES_FOR_STATS`
        samples have been observed (either via :meth:`fit` or
        incrementally through :meth:`compute`).
        """
        return self._count >= _MIN_SAMPLES_FOR_STATS

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self, dag: AdjacencyMatrix, data: Optional[DataMatrix] = None
    ) -> BehavioralDescriptor:
        """Compute the composite behavioral descriptor for *dag*.

        Executes the full pipeline: concatenation → weighting → feature
        selection → normalization → PCA reduction.

        Running statistics used for normalization are updated each time
        this method is called so that the descriptor adapts over the
        course of a QD run.

        Parameters
        ----------
        dag:
            Adjacency matrix of the candidate DAG.
        data:
            Optional N × p observation matrix.

        Returns
        -------
        BehavioralDescriptor
            1-D ``float64`` array of length :pyattr:`descriptor_dim`.
        """
        # 1. Concatenate sub-descriptors.
        parts = [d.compute(dag, data) for d in self._descriptors]
        raw = np.concatenate(parts).astype(np.float64)

        # 2. Apply per-dimension weights.
        weighted = self._apply_weights(raw, self._weights)

        # 3. Update running statistics (before selection, on full vector).
        self._update_running_stats(weighted)

        # 4. Feature selection (drop low-variance dimensions).
        selected = self._select_features(weighted)

        # 5. Normalization.
        normalized = self._dispatch_normalization(selected)

        # 6. PCA reduction.
        reduced = self._apply_pca(normalized)

        return reduced.astype(np.float64)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        dags: list[AdjacencyMatrix],
        data: DataMatrix,
    ) -> None:
        """Pre-compute normalization statistics from a collection of DAGs.

        This is useful when you want stable normalization from the start
        of a QD run rather than relying on incremental updates.

        Parameters
        ----------
        dags:
            List of adjacency matrices to derive statistics from.
        data:
            Shared observation matrix passed to each sub-descriptor.
        """
        if not dags:
            raise ValueError("Need at least one DAG for fitting.")

        # Collect raw weighted vectors.
        vectors: List[npt.NDArray[np.float64]] = []
        for dag in dags:
            parts = [d.compute(dag, data) for d in self._descriptors]
            raw = np.concatenate(parts).astype(np.float64)
            weighted = self._apply_weights(raw, self._weights)
            vectors.append(weighted)

        mat = np.stack(vectors)  # (n_dags, raw_dim)

        # Bulk statistics.
        self._running_mean = mat.mean(axis=0)
        self._running_var = mat.var(axis=0)
        self._running_min = mat.min(axis=0)
        self._running_max = mat.max(axis=0)
        self._count = len(dags)

        # Quantile buffer.
        self._quantile_buffer = [v.copy() for v in vectors]

        # Feature mask from variance.
        self._feature_mask = self._compute_feature_mask(self._running_var)

        # PCA on the selected features.
        if self._pca_dim is not None:
            selected = mat[:, self._feature_mask]
            self._fit_pca(selected)

    # ------------------------------------------------------------------
    # Weighting
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_weights(
        raw: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply each dimension of *raw* by its corresponding weight.

        Parameters
        ----------
        raw:
            1-D descriptor vector.
        weights:
            1-D weight vector of the same length.

        Returns
        -------
        npt.NDArray[np.float64]
            Element-wise product ``raw * weights``.
        """
        return raw * weights

    # ------------------------------------------------------------------
    # Running statistics
    # ------------------------------------------------------------------

    def _update_running_stats(
        self, vec: npt.NDArray[np.float64]
    ) -> None:
        """Incrementally update running mean, variance, min, and max.

        Uses Welford's online algorithm for numerically stable variance
        estimation.

        Parameters
        ----------
        vec:
            Newly observed weighted descriptor vector.
        """
        self._count += 1

        if self._running_mean is None:
            self._running_mean = vec.copy()
            self._running_var = np.zeros_like(vec)
            self._running_min = vec.copy()
            self._running_max = vec.copy()
            return

        # Welford update.
        delta = vec - self._running_mean
        self._running_mean = self._running_mean + delta / self._count
        delta2 = vec - self._running_mean
        self._running_var = (
            self._running_var * (self._count - 1) + delta * delta2
        ) / self._count

        # Min / max.
        np.minimum(self._running_min, vec, out=self._running_min)
        np.maximum(self._running_max, vec, out=self._running_max)

        # Quantile buffer (cap at a reasonable size).
        if len(self._quantile_buffer) < 5000:
            self._quantile_buffer.append(vec.copy())

        # Recompute feature mask periodically.
        if self._count % _MIN_SAMPLES_FOR_STATS == 0 and self._running_var is not None:
            self._feature_mask = self._compute_feature_mask(self._running_var)

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def _compute_feature_mask(
        self, variance: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Return a boolean mask keeping features above *min_variance*.

        Parameters
        ----------
        variance:
            Per-dimension variance array.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean mask of length ``raw_dim``.
        """
        mask = variance >= self._min_variance
        # Always keep at least one feature.
        if not mask.any():
            mask[np.argmax(variance)] = True
        return mask

    def _select_features(
        self, raw: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Remove features whose variance is below *min_variance*.

        If no feature mask has been computed yet (fewer than
        :data:`_MIN_SAMPLES_FOR_STATS` observations), all features are
        retained.

        Parameters
        ----------
        raw:
            Weighted descriptor vector of length ``raw_dim``.

        Returns
        -------
        npt.NDArray[np.float64]
            Descriptor with low-variance features removed.
        """
        if self._feature_mask is None or not self._is_fitted:
            return raw
        return raw[self._feature_mask]

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _dispatch_normalization(
        self, raw: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Route to the configured normalization method.

        Before enough samples have been observed (< 10), the raw vector
        is returned unchanged regardless of the normalization setting.

        Parameters
        ----------
        raw:
            Descriptor vector (after weighting and feature selection).

        Returns
        -------
        npt.NDArray[np.float64]
            Normalized descriptor vector.
        """
        if not self._is_fitted:
            return raw

        if self._normalization == "none":
            return raw
        if self._normalization == "zscore":
            return self._normalize_zscore(raw)
        if self._normalization == "minmax":
            return self._normalize_minmax(raw)
        if self._normalization == "quantile":
            return self._normalize_quantile(raw)
        # Should not reach here due to __init__ validation.
        return raw  # pragma: no cover

    def _normalize_zscore(
        self, raw: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Z-score normalization using running mean and standard deviation.

        Each dimension is transformed as ``(x - mean) / std``.  Dimensions
        with near-zero standard deviation are left centred but un-scaled
        to avoid division-by-zero artefacts.

        Parameters
        ----------
        raw:
            Descriptor vector to normalize.

        Returns
        -------
        npt.NDArray[np.float64]
            Z-scored descriptor.
        """
        mean = self._running_mean
        std = np.sqrt(self._running_var)

        if self._feature_mask is not None and self._is_fitted:
            mean = mean[self._feature_mask]
            std = std[self._feature_mask]

        safe_std = np.where(std > 1e-12, std, 1.0)
        return (raw - mean) / safe_std

    def _normalize_minmax(
        self, raw: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Min-max normalization scaling each dimension to ``[0, 1]``.

        Uses the running minimum and maximum observed so far.  Dimensions
        with no observed range (min == max) are mapped to 0.5.

        Parameters
        ----------
        raw:
            Descriptor vector to normalize.

        Returns
        -------
        npt.NDArray[np.float64]
            Normalized descriptor in ``[0, 1]``.
        """
        lo = self._running_min
        hi = self._running_max

        if self._feature_mask is not None and self._is_fitted:
            lo = lo[self._feature_mask]
            hi = hi[self._feature_mask]

        rng = hi - lo
        safe_rng = np.where(rng > 1e-12, rng, 1.0)
        normalized = (raw - lo) / safe_rng
        # Clamp to [0, 1] to handle slight extrapolation.
        return np.clip(normalized, 0.0, 1.0)

    def _normalize_quantile(
        self, raw: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Quantile-based normalization mapping each dimension to ``[0, 1]``.

        For each dimension the value is replaced by the fraction of
        previously observed values that are less than or equal to it
        (i.e., its empirical CDF value).

        Parameters
        ----------
        raw:
            Descriptor vector to normalize.

        Returns
        -------
        npt.NDArray[np.float64]
            Quantile-normalized descriptor in ``[0, 1]``.
        """
        if not self._quantile_buffer:
            return raw

        history = np.stack(self._quantile_buffer)  # (n_samples, dim)

        if self._feature_mask is not None and self._is_fitted:
            history = history[:, self._feature_mask]

        n = history.shape[0]
        quantiles = np.empty_like(raw)
        for j in range(raw.shape[0]):
            quantiles[j] = np.searchsorted(
                np.sort(history[:, j]), raw[j], side="right"
            ) / n
        return np.clip(quantiles, 0.0, 1.0)

    # ------------------------------------------------------------------
    # PCA dimensionality reduction
    # ------------------------------------------------------------------

    def _fit_pca(
        self, mat: npt.NDArray[np.float64]
    ) -> None:
        """Compute PCA components from a matrix of descriptor vectors.

        Uses truncated SVD so that only the top *pca_dim* components are
        retained.

        Parameters
        ----------
        mat:
            (n_samples, n_features) array of descriptor vectors.
        """
        if self._pca_dim is None:
            return

        self._pca_mean = mat.mean(axis=0)
        centred = mat - self._pca_mean

        # Truncated SVD – we only need the top pca_dim right-singular vectors.
        effective_dim = min(self._pca_dim, centred.shape[1], centred.shape[0])
        _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
        self._pca_components = vt[:effective_dim]  # (pca_dim, n_features)

    def _apply_pca(
        self, raw: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Project *raw* onto the principal component subspace.

        If PCA has not yet been fitted (insufficient samples or
        :meth:`fit` not called), the vector is returned unchanged.
        When enough samples accumulate during incremental operation, PCA
        is fitted lazily from the quantile buffer.

        Parameters
        ----------
        raw:
            Normalized descriptor vector.

        Returns
        -------
        npt.NDArray[np.float64]
            Projected descriptor of length *pca_dim* (or the original
            vector if PCA is not active).
        """
        if self._pca_dim is None:
            return raw

        # Lazy PCA fitting once we have enough samples.
        if self._pca_components is None and self._is_fitted:
            if self._quantile_buffer:
                buf = np.stack(self._quantile_buffer)
                if self._feature_mask is not None:
                    buf = buf[:, self._feature_mask]
                self._fit_pca(buf)

        if self._pca_components is None or self._pca_mean is None:
            return raw

        centred = raw - self._pca_mean
        return centred @ self._pca_components.T  # (pca_dim,)
