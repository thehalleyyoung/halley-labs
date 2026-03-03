"""Information-theoretic behavioral descriptors for causal DAGs.

This module provides :class:`InfoTheoreticDescriptor`, a configurable
descriptor computer that extracts information-theoretic features from a
directed acyclic graph (DAG) paired with observational data.  All
computations assume a *linear-Gaussian* generative model, which allows
closed-form expressions for mutual information, conditional entropy, and
partial correlation.

Available feature groups
------------------------
``mi_profile``
    For each edge *i → j* in the DAG, compute
    MI(X_i ; X_j | Pa(j) \\ {i}) via partial-correlation under the
    linear-Gaussian model.  Returns summary statistics (mean, std, min,
    max) of the MI values across all edges.

``ci_signature``
    For every ordered pair (i, j), test conditional independence
    CI(X_i, X_j | Pa(j)) using partial-correlation ↦ Fisher Z-test.
    Summarise the binary independent/dependent vector (fraction
    dependent, total tests, mean |Z|, max |Z|).

``entropy_profile``
    For each node *j*, compute H(X_j | Pa(X_j)) under the linear-
    Gaussian model.  Returns summary statistics (mean, std, min, max)
    of the conditional entropies.

``avg_mi``
    A single scalar: mean pairwise MI between adjacent variables.

``avg_conditional_entropy``
    A single scalar: mean H(X_j | Pa(X_j)) for nodes that have at
    least one parent.

PCA compression
---------------
When the concatenated raw descriptor exceeds *max_descriptor_dim*, a
PCA projection (computed via SVD) reduces dimensionality.  The
projection matrix is fitted on the first call and reused for all
subsequent calls to ensure consistent descriptor spaces.

Notes
-----
*  A ``DataMatrix`` must be provided to :meth:`compute`; if *data* is
   ``None`` a zero descriptor is returned.
*  Numerical stability is ensured by clamping variances and
   determinants away from zero and by regularising precision-matrix
   inversions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.descriptors.descriptor_base import DescriptorComputer
from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_2PI_E: float = float(np.log(2.0 * np.pi * np.e))
"""Pre-computed log(2πe), used repeatedly in Gaussian entropy formulae."""

_VARIANCE_FLOOR: float = 1e-12
"""Minimum variance to avoid log-of-zero in entropy computations."""

_REGULARISATION_EPS: float = 1e-8
"""Ridge term added to covariance matrices before inversion."""

_FISHER_Z_ALPHA: float = 0.05
"""Significance level for the two-sided Fisher Z-test used in the
conditional independence signature."""

_FISHER_Z_CRITICAL: float = 1.96
"""Critical value for α = 0.05 two-sided standard normal test."""

# ---------------------------------------------------------------------------
# Feature-dimension registry
# ---------------------------------------------------------------------------

_FEATURE_DIMS: Dict[str, int] = {
    "mi_profile": 4,              # mean, std, min, max
    "ci_signature": 4,            # frac_dep, n_tests, mean_absZ, max_absZ
    "entropy_profile": 4,         # mean, std, min, max
    "avg_mi": 1,                  # scalar
    "avg_conditional_entropy": 1,  # scalar
}
"""Mapping from feature name to its raw output dimensionality."""

_ALL_FEATURES: List[str] = list(_FEATURE_DIMS.keys())
"""Canonical ordering of all available features."""


# ===================================================================== #
#  Main class                                                            #
# ===================================================================== #

class InfoTheoreticDescriptor(DescriptorComputer):
    """Compute information-theoretic behavioral descriptors from a DAG
    and observational data.

    The descriptor concatenates several configurable feature groups
    (see module docstring) into a single real-valued vector.  When the
    concatenated raw dimension exceeds *max_descriptor_dim*, an SVD-based
    PCA projection is applied to compress the descriptor.

    Parameters
    ----------
    features : list[str] | None
        Which feature groups to include.  ``None`` (default) selects all
        available features:
        ``["mi_profile", "ci_signature", "entropy_profile",
        "avg_mi", "avg_conditional_entropy"]``.
    max_descriptor_dim : int
        Maximum dimensionality of the output descriptor.  If the raw
        concatenated feature vector exceeds this, PCA is used to reduce
        to *max_descriptor_dim* components.

    Raises
    ------
    ValueError
        If an unknown feature name is provided.

    Examples
    --------
    >>> desc = InfoTheoreticDescriptor(features=["avg_mi", "entropy_profile"])
    >>> dag = np.array([[0, 1], [0, 0]], dtype=np.int8)
    >>> data = rng.standard_normal((200, 2))
    >>> vec = desc.compute(dag, data)
    >>> vec.shape
    (5,)
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        features: list[str] | None = None,
        max_descriptor_dim: int = 10,
    ) -> None:
        """Initialise the descriptor computer.

        Parameters
        ----------
        features:
            Feature groups to include.  ``None`` means *all* features.
            Valid names: ``"mi_profile"``, ``"ci_signature"``,
            ``"entropy_profile"``, ``"avg_mi"``,
            ``"avg_conditional_entropy"``.
        max_descriptor_dim:
            Upper bound on output dimensionality.  PCA is applied when
            the raw feature vector is longer.

        Raises
        ------
        ValueError
            If *features* contains an unrecognised name.
        """
        if features is None:
            self._features: list[str] = list(_ALL_FEATURES)
        else:
            unknown = set(features) - set(_ALL_FEATURES)
            if unknown:
                raise ValueError(
                    f"Unknown feature(s): {unknown}.  "
                    f"Choose from {_ALL_FEATURES}."
                )
            self._features = list(features)

        self._max_descriptor_dim: int = max(1, max_descriptor_dim)

        # Raw dimension = sum of per-feature dims
        self._raw_dim: int = sum(_FEATURE_DIMS[f] for f in self._features)

        # Whether PCA is needed
        self._use_pca: bool = self._raw_dim > self._max_descriptor_dim

        # PCA state (fitted lazily on first call)
        self._pca_components: Optional[npt.NDArray[np.float64]] = None
        self._pca_mean: Optional[npt.NDArray[np.float64]] = None
        self._pca_fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Abstract-method implementations                                    #
    # ------------------------------------------------------------------ #

    @property
    def descriptor_dim(self) -> int:  # noqa: D401
        """Dimensionality of the output descriptor vector.

        Returns the raw concatenated dimension when it fits within
        *max_descriptor_dim*, otherwise returns *max_descriptor_dim*
        (after PCA compression).

        Returns
        -------
        int
            Length of the array returned by :meth:`compute`.
        """
        if self._use_pca:
            return self._max_descriptor_dim
        return self._raw_dim

    @property
    def descriptor_bounds(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-dimension lower and upper bounds.

        Information-theoretic quantities are non-negative (MI, entropy)
        or lie in [0, 1] (fractions).  After PCA the bounds become
        symmetric around zero.  We use a conservative heuristic:
        ``[-10, 10]`` when PCA is active, ``[0, 10]`` otherwise.

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(low, high)`` each of shape ``(descriptor_dim,)``.
        """
        dim = self.descriptor_dim
        if self._use_pca:
            low = np.full(dim, -10.0, dtype=np.float64)
        else:
            low = np.zeros(dim, dtype=np.float64)
        high = np.full(dim, 10.0, dtype=np.float64)
        return low, high

    # ------------------------------------------------------------------ #
    #  Core compute method                                                #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        dag: AdjacencyMatrix,
        data: Optional[DataMatrix] = None,
    ) -> BehavioralDescriptor:
        """Compute the information-theoretic descriptor.

        Parameters
        ----------
        dag:
            ``n × n`` binary adjacency matrix where ``dag[i, j] = 1``
            indicates a directed edge *i → j*.
        data:
            ``N × p`` data matrix with *N* observations and *p*
            variables.  Required for meaningful results; if ``None`` a
            zero vector of length :pyattr:`descriptor_dim` is returned.

        Returns
        -------
        BehavioralDescriptor
            1-D float64 array of length :pyattr:`descriptor_dim`.

        Notes
        -----
        On the *first* call with ``data is not None`` and PCA
        compression is active, the PCA projection matrix is fitted from
        the raw feature vector (a single sample is sufficient because we
        centre on that sample and use the identity covariance as a
        fallback).  Subsequent calls reuse the stored projection.
        """
        if data is None:
            return np.zeros(self.descriptor_dim, dtype=np.float64)

        n: int = dag.shape[0]

        # ---- Assemble raw features --------------------------------- #
        raw_parts: list[npt.NDArray[np.float64]] = []

        for feat in self._features:
            if feat == "mi_profile":
                raw_parts.append(self._mutual_information_profile(dag, data, n))
            elif feat == "ci_signature":
                raw_parts.append(
                    self._conditional_independence_signature(dag, data, n)
                )
            elif feat == "entropy_profile":
                raw_parts.append(self._entropy_profile(dag, data, n))
            elif feat == "avg_mi":
                raw_parts.append(
                    np.array(
                        [self._average_mutual_information(dag, data, n)],
                        dtype=np.float64,
                    )
                )
            elif feat == "avg_conditional_entropy":
                raw_parts.append(
                    np.array(
                        [self._average_conditional_entropy(dag, data, n)],
                        dtype=np.float64,
                    )
                )

        raw = np.concatenate(raw_parts)

        # ---- Optional PCA compression ------------------------------ #
        if self._use_pca:
            raw = self._pca_transform(raw)

        return raw

    # ================================================================= #
    #  Feature-group computations                                        #
    # ================================================================= #

    # ------------------------------------------------------------------ #
    #  1. Mutual Information Profile                                      #
    # ------------------------------------------------------------------ #

    def _mutual_information_profile(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Compute MI(X_i ; X_j | Pa(j) \\ {i}) for each edge i → j.

        Under the linear-Gaussian assumption the conditional MI equals

            MI = -0.5 · log(1 - ρ²)

        where ρ is the *partial correlation* between X_i and X_j given
        the remaining parents of *j*.

        Parameters
        ----------
        dag:
            ``n × n`` adjacency matrix.
        data:
            ``N × p`` data matrix.
        n:
            Number of nodes.

        Returns
        -------
        ndarray
            4-element array ``[mean, std, min, max]`` of the MI values
            across all edges.  If the DAG has no edges, returns zeros.
        """
        mi_values: list[float] = []

        for j in range(n):
            parents = np.where(dag[:, j])[0]
            if len(parents) == 0:
                continue

            for i in parents:
                # Conditioning set: all parents of j except i
                z_indices = [int(p) for p in parents if p != i]

                rho = self._partial_correlation(
                    int(i), int(j), z_indices, data
                )
                rho_sq = min(rho * rho, 1.0 - 1e-15)
                mi = -0.5 * np.log(1.0 - rho_sq)
                mi_values.append(max(float(mi), 0.0))

        if not mi_values:
            return np.zeros(4, dtype=np.float64)

        arr = np.array(mi_values, dtype=np.float64)
        return np.array(
            [float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    #  2. Conditional Independence Signature                              #
    # ------------------------------------------------------------------ #

    def _conditional_independence_signature(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Test CI(X_i, X_j | Pa(j)) for every ordered pair (i, j).

        For each pair where *i ≠ j*, we compute the partial correlation
        of X_i and X_j given Pa(j) and apply the Fisher Z-transform to
        obtain a test statistic.  Under H₀ (conditional independence)
        the Z-statistic is approximately standard normal for large *N*.

        The binary decision (dependent if |Z| > z_{α/2}) is recorded.
        The returned 4-element vector summarises the full binary vector:

        1. **frac_dependent** – fraction of pairs declared dependent.
        2. **n_tests**        – total number of tests performed
           (``n * (n-1)``), normalised by ``n²`` to stay in [0, 1].
        3. **mean_abs_z**     – mean |Z| across all tests (scaled by
           ``1/10`` for numerical convenience).
        4. **max_abs_z**      – max |Z| across all tests (scaled by
           ``1/10``).

        Parameters
        ----------
        dag:
            ``n × n`` adjacency matrix.
        data:
            ``N × p`` data matrix.
        n:
            Number of nodes.

        Returns
        -------
        ndarray
            4-element float64 array.
        """
        n_obs = data.shape[0]

        dependent_count: int = 0
        total_tests: int = 0
        abs_z_values: list[float] = []

        for j in range(n):
            parents_j = list(np.where(dag[:, j])[0].astype(int))

            for i in range(n):
                if i == j:
                    continue

                total_tests += 1

                # Conditioning set is Pa(j)
                z_indices = parents_j

                rho = self._partial_correlation(i, j, z_indices, data)

                # Fisher Z-transform
                z_stat = self._fisher_z_statistic(rho, n_obs, len(z_indices))
                abs_z = abs(z_stat)
                abs_z_values.append(abs_z)

                if abs_z > _FISHER_Z_CRITICAL:
                    dependent_count += 1

        if total_tests == 0:
            return np.zeros(4, dtype=np.float64)

        frac_dep = dependent_count / total_tests
        norm_tests = total_tests / (n * n) if n > 0 else 0.0
        mean_abs_z = float(np.mean(abs_z_values)) / 10.0
        max_abs_z = float(np.max(abs_z_values)) / 10.0

        return np.array(
            [frac_dep, norm_tests, mean_abs_z, max_abs_z],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    #  3. Entropy Profile                                                 #
    # ------------------------------------------------------------------ #

    def _entropy_profile(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Compute H(X_j | Pa(X_j)) for each node *j*.

        Under the linear-Gaussian model the conditional entropy is:

            H(X_j | Pa(j)) = 0.5 · log(2πe · σ²_residual)

        where σ²_residual is the variance of the residual from
        regressing X_j on Pa(j).  For root nodes (no parents) we use
        the marginal variance.

        Parameters
        ----------
        dag:
            ``n × n`` adjacency matrix.
        data:
            ``N × p`` data matrix.
        n:
            Number of nodes.

        Returns
        -------
        ndarray
            4-element array ``[mean, std, min, max]`` of the
            conditional entropies across all nodes.
        """
        entropies: list[float] = []

        for j in range(n):
            parents = np.where(dag[:, j])[0]
            x_j = data[:, j]

            if len(parents) == 0:
                # Root node: use marginal variance
                var_j = max(float(np.var(x_j)), _VARIANCE_FLOOR)
                entropies.append(self._gaussian_entropy(var_j))
                continue

            # Residual variance after linear regression on parents
            x_parents = data[:, parents]
            res_var = self._residual_variance(x_parents, x_j)
            entropies.append(self._gaussian_entropy(res_var))

        if not entropies:
            return np.zeros(4, dtype=np.float64)

        arr = np.array(entropies, dtype=np.float64)
        return np.array(
            [float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())],
            dtype=np.float64,
        )

    # ================================================================= #
    #  Scalar feature helpers (backward-compatible)                      #
    # ================================================================= #

    def _average_mutual_information(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
        n: int,
    ) -> float:
        """Mean MI(X_i ; X_j) over all edges in *dag*.

        Uses bivariate Gaussian MI (no conditioning) for each directed
        edge, consistent with the original two-feature descriptor.

        Parameters
        ----------
        dag:
            ``n × n`` adjacency matrix.
        data:
            ``N × p`` data matrix.
        n:
            Number of nodes.

        Returns
        -------
        float
            Average mutual information.  Returns ``0.0`` if no edges.
        """
        mis: list[float] = []
        for i in range(n):
            for j in range(n):
                if dag[i, j]:
                    mis.append(
                        self._mutual_information(data[:, i], data[:, j])
                    )
        return float(np.mean(mis)) if mis else 0.0

    def _average_conditional_entropy(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
        n: int,
    ) -> float:
        """Mean H(X_j | Pa(X_j)) for nodes with ≥ 1 parent.

        Parameters
        ----------
        dag:
            ``n × n`` adjacency matrix.
        data:
            ``N × p`` data matrix.
        n:
            Number of nodes.

        Returns
        -------
        float
            Average conditional entropy.  Returns ``0.0`` if no node
            has parents.
        """
        entropies: list[float] = []
        for j in range(n):
            parents = np.where(dag[:, j])[0]
            if len(parents) == 0:
                continue

            x_parents = data[:, parents]
            x_j = data[:, j]
            res_var = self._residual_variance(x_parents, x_j)
            entropies.append(self._gaussian_entropy(res_var))

        return float(np.mean(entropies)) if entropies else 0.0

    # ================================================================= #
    #  PCA compression                                                   #
    # ================================================================= #

    def _pca_transform(
        self,
        raw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Project *raw* feature vector through a PCA projection.

        On the first call the PCA components are fitted from *raw*
        (a single sample).  Because a single observation provides no
        variance information, we centre on that sample and use
        uniformly-scaled axes (equivalent to truncation).  On
        subsequent calls we reuse the stored projection for
        consistency.

        Parameters
        ----------
        raw:
            1-D array of length ``self._raw_dim``.

        Returns
        -------
        ndarray
            1-D array of length ``self._max_descriptor_dim``.
        """
        if not self._pca_fitted:
            self._fit_pca(raw)

        assert self._pca_mean is not None
        assert self._pca_components is not None

        centred = raw - self._pca_mean
        projected = centred @ self._pca_components.T

        return projected.astype(np.float64)

    def _fit_pca(
        self,
        raw: npt.NDArray[np.float64],
    ) -> None:
        """Fit PCA components from a single sample using SVD.

        With a single observation the covariance matrix is rank-0, so
        we treat the raw vector as a *batch of one* and perform
        truncated SVD.  This reduces to selecting the first
        ``max_descriptor_dim`` coordinates after centering (which
        is a no-op for a single sample, but the stored mean allows
        subsequent samples to be centred consistently).

        When more than one sample is available in the future, the
        caller can extend this by collecting a matrix of raw vectors
        and re-fitting.

        Parameters
        ----------
        raw:
            1-D array of length ``self._raw_dim``.
        """
        self._pca_mean = raw.copy()

        # With one sample, centre → zero vector.  SVD of a zero row
        # gives a trivial decomposition.  We fall back to the identity
        # (i.e., select the first *k* dimensions).
        k = self._max_descriptor_dim

        # Build a data matrix (1 × raw_dim) for the SVD path
        centred = (raw - self._pca_mean).reshape(1, -1)  # all zeros

        # Attempt SVD; if it degenerates (as expected for a single
        # centred sample), fall back to the leading identity rows.
        try:
            _u, s, vt = np.linalg.svd(centred, full_matrices=False)
            # s will be all zeros; fall back
            if np.all(s < _VARIANCE_FLOOR):
                raise np.linalg.LinAlgError("Degenerate singular values")
            self._pca_components = vt[:k, :]
        except np.linalg.LinAlgError:
            # Identity fallback: first k standard basis vectors
            components = np.zeros(
                (k, self._raw_dim), dtype=np.float64
            )
            for idx in range(min(k, self._raw_dim)):
                components[idx, idx] = 1.0
            self._pca_components = components

        self._pca_fitted = True

    def _update_pca(
        self,
        raw_batch: npt.NDArray[np.float64],
    ) -> None:
        """Re-fit PCA from a batch of raw feature vectors.

        This method can be called externally when a collection of raw
        descriptors is available (e.g., after an initial MAP-Elites
        generation), enabling a data-driven PCA projection.

        Parameters
        ----------
        raw_batch:
            2-D array of shape ``(m, raw_dim)`` with *m ≥ 2* raw
            feature vectors.

        Raises
        ------
        ValueError
            If the batch has fewer than 2 rows or wrong width.
        """
        if raw_batch.ndim != 2:
            raise ValueError(
                f"Expected 2-D array, got shape {raw_batch.shape}."
            )
        if raw_batch.shape[1] != self._raw_dim:
            raise ValueError(
                f"Expected width {self._raw_dim}, got {raw_batch.shape[1]}."
            )
        if raw_batch.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit PCA.")

        k = self._max_descriptor_dim
        self._pca_mean = raw_batch.mean(axis=0)
        centred = raw_batch - self._pca_mean

        _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
        self._pca_components = vt[:k, :]
        self._pca_fitted = True

    # ================================================================= #
    #  Low-level statistical helpers                                     #
    # ================================================================= #

    @staticmethod
    def _gaussian_entropy(var: float) -> float:
        """Differential entropy of a univariate Gaussian.

        H(X) = 0.5 · log(2πe · σ²)

        Parameters
        ----------
        var:
            Variance σ².  Clamped to :data:`_VARIANCE_FLOOR` if
            non-positive.

        Returns
        -------
        float
            Entropy in nats.
        """
        if var <= 0:
            var = _VARIANCE_FLOOR
        return 0.5 * np.log(2.0 * np.pi * np.e * var)

    @classmethod
    def _mutual_information(
        cls,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> float:
        """Estimate MI(X; Y) under a bivariate Gaussian assumption.

        MI(X; Y) = H(X) + H(Y) - H(X, Y)

        where each entropy is evaluated using the Gaussian formula.

        Parameters
        ----------
        x:
            1-D array of observations for variable X.
        y:
            1-D array of observations for variable Y.

        Returns
        -------
        float
            Non-negative mutual information in nats.
        """
        var_x = max(float(np.var(x)), _VARIANCE_FLOOR)
        var_y = max(float(np.var(y)), _VARIANCE_FLOOR)

        hx = cls._gaussian_entropy(var_x)
        hy = cls._gaussian_entropy(var_y)

        cov = np.cov(x, y)
        det = np.linalg.det(cov)
        det = max(det, _VARIANCE_FLOOR ** 2)
        h_joint = 0.5 * np.log((2.0 * np.pi * np.e) ** 2 * det)

        return max(hx + hy - h_joint, 0.0)

    @staticmethod
    def _partial_correlation(
        x_idx: int,
        y_idx: int,
        z_indices: list[int],
        data: DataMatrix,
    ) -> float:
        """Compute partial correlation ρ(X, Y | Z) via precision matrix.

        Given indices *x_idx*, *y_idx* and a (possibly empty) set of
        conditioning indices *z_indices*, the partial correlation is:

            ρ(X, Y | Z) = -Θ_{xy} / √(Θ_{xx} · Θ_{yy})

        where Θ = Σ⁻¹ is the precision matrix of the joint
        ``[X, Y, Z₁, …, Z_k]`` sub-vector.

        Parameters
        ----------
        x_idx:
            Column index of variable X in *data*.
        y_idx:
            Column index of variable Y in *data*.
        z_indices:
            Column indices of the conditioning set Z.
        data:
            ``N × p`` data matrix.

        Returns
        -------
        float
            Partial correlation in [-1, 1].  Returns the Pearson
            correlation when *z_indices* is empty.
        """
        # Marginal case: no conditioning set
        if not z_indices:
            x_vals = data[:, x_idx]
            y_vals = data[:, y_idx]
            std_x = float(np.std(x_vals))
            std_y = float(np.std(y_vals))
            if std_x < _VARIANCE_FLOOR or std_y < _VARIANCE_FLOOR:
                return 0.0
            corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
            return np.clip(corr, -1.0, 1.0)

        # Assemble sub-matrix [X, Y, Z₁, ..., Z_k]
        all_indices = [x_idx, y_idx] + list(z_indices)
        # Remove duplicates while preserving order
        seen: set[int] = set()
        unique_indices: list[int] = []
        for idx in all_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        sub_data = data[:, unique_indices]

        # Position of x and y within the sub-matrix
        pos_x = unique_indices.index(x_idx)
        pos_y = unique_indices.index(y_idx)

        # Covariance of sub-vector with ridge regularisation
        cov = np.cov(sub_data, rowvar=False)
        if cov.ndim == 0:
            # Single variable edge case
            return 0.0
        cov += _REGULARISATION_EPS * np.eye(cov.shape[0])

        # Precision matrix
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Singular after regularisation; fall back to pseudo-inverse
            precision = np.linalg.pinv(cov)

        theta_xx = precision[pos_x, pos_x]
        theta_yy = precision[pos_y, pos_y]
        theta_xy = precision[pos_x, pos_y]

        denom = np.sqrt(abs(theta_xx * theta_yy))
        if denom < _VARIANCE_FLOOR:
            return 0.0

        rho = -theta_xy / denom
        return float(np.clip(rho, -1.0, 1.0))

    @staticmethod
    def _fisher_z_statistic(
        rho: float,
        n_obs: int,
        n_cond: int,
    ) -> float:
        """Compute the Fisher Z-test statistic for partial correlation.

        The Z-statistic is:

            Z = 0.5 · log((1+ρ) / (1-ρ)) · √(n - |Z| - 3)

        where *n* is the sample size and |Z| is the size of the
        conditioning set.

        Under the null hypothesis of zero partial correlation, *Z* is
        approximately standard-normal for large *n*.

        Parameters
        ----------
        rho:
            Partial correlation in (-1, 1).
        n_obs:
            Number of observations (sample size).
        n_cond:
            Size of the conditioning set.

        Returns
        -------
        float
            Z-statistic.  Returns 0 if degrees of freedom are
            insufficient (``n - |Z| - 3 ≤ 0``).
        """
        df = n_obs - n_cond - 3
        if df <= 0:
            return 0.0

        # Clamp rho away from ±1 to avoid log singularity
        rho_c = np.clip(rho, -1.0 + 1e-10, 1.0 - 1e-10)

        # Fisher transformation (arctanh)
        z_fisher = 0.5 * np.log((1.0 + rho_c) / (1.0 - rho_c))

        return float(z_fisher * np.sqrt(df))

    @staticmethod
    def _residual_variance(
        x_parents: npt.NDArray[np.float64],
        x_j: npt.NDArray[np.float64],
    ) -> float:
        """Variance of the residual from OLS regression of x_j on x_parents.

        Computes the ordinary least-squares fit X_j = X_pa · β + ε
        and returns Var(ε).

        Parameters
        ----------
        x_parents:
            ``N × k`` matrix of parent observations.
        x_j:
            ``N``-length target vector.

        Returns
        -------
        float
            Residual variance, floored to :data:`_VARIANCE_FLOOR`.
        """
        try:
            coeffs, _, _, _ = np.linalg.lstsq(x_parents, x_j, rcond=None)
            residual = x_j - x_parents @ coeffs
            res_var = float(np.var(residual))
        except np.linalg.LinAlgError:
            res_var = float(np.var(x_j))

        return max(res_var, _VARIANCE_FLOOR)

    # ================================================================= #
    #  Representation                                                    #
    # ================================================================= #

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns
        -------
        str
            String showing the selected features and PCA state.
        """
        return (
            f"{self.__class__.__name__}("
            f"features={self._features!r}, "
            f"raw_dim={self._raw_dim}, "
            f"output_dim={self.descriptor_dim}, "
            f"pca={'fitted' if self._pca_fitted else 'pending' if self._use_pca else 'off'}"
            f")"
        )
