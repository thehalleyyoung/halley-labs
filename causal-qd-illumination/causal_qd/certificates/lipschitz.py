"""Lipschitz continuity bounds on scoring functions.

Provides :class:`LipschitzBoundComputer` which computes upper bounds on
the sensitivity of BIC (or other decomposable) scores to data
perturbations, using matrix perturbation theory, finite differences,
and sample splitting.

Also provides perturbation analysis utilities: how much must the data
change to flip an edge decision, and the corresponding stability radius.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import DataMatrix

if TYPE_CHECKING:
    from causal_qd.types import AdjacencyMatrix


# ---------------------------------------------------------------------------
# LipschitzBound (original, spectral-norm based)
# ---------------------------------------------------------------------------

class LipschitzBound:
    """Spectral-norm Lipschitz bound on a score function.

    The Lipschitz constant *L* is estimated as ``σ_max(Σ)``, the
    largest singular value of the sample covariance.  The bound is::

        |score(G₁) − score(G₂)| ≤ L · d_H(G₁, G₂) / p²

    Parameters
    ----------
    regularisation : float
        Constant added to the diagonal for numerical stability.
    """

    def __init__(self, regularisation: float = 1e-6) -> None:
        self._regularisation = regularisation
        self._lipschitz_constant: float | None = None

    def estimate_constant(self, data: DataMatrix) -> float:
        """Estimate the Lipschitz constant from data.

        Parameters
        ----------
        data : DataMatrix
            Observed data matrix (N × p).

        Returns
        -------
        float
            Estimated Lipschitz constant ``σ_max(Σ)``.
        """
        cov = np.cov(data, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += self._regularisation * np.eye(cov.shape[0])
        singular_values = np.linalg.svd(cov, compute_uv=False)
        self._lipschitz_constant = float(singular_values[0])
        return self._lipschitz_constant

    def bound(
        self,
        dag1: AdjacencyMatrix,
        dag2: AdjacencyMatrix,
        data: DataMatrix,
    ) -> float:
        """Upper-bound the score difference between *dag1* and *dag2*.

        Parameters
        ----------
        dag1, dag2 : AdjacencyMatrix
        data : DataMatrix

        Returns
        -------
        float
            Upper bound on ``|score(dag1, data) − score(dag2, data)|``.
        """
        if dag1.shape != dag2.shape:
            raise ValueError(
                f"DAG shapes must match: {dag1.shape} vs {dag2.shape}"
            )
        if self._lipschitz_constant is None:
            self.estimate_constant(data)
        assert self._lipschitz_constant is not None

        p = dag1.shape[0]
        hamming = int(np.sum(dag1 != dag2))
        return self._lipschitz_constant * hamming / (p * p)

    def reset(self) -> None:
        """Clear the cached Lipschitz constant."""
        self._lipschitz_constant = None


# ---------------------------------------------------------------------------
# LipschitzBoundComputer (full implementation)
# ---------------------------------------------------------------------------

# Score function type: (adjacency, data) -> float
ScoreFunction = Callable[["AdjacencyMatrix", DataMatrix], float]


class LipschitzBoundComputer:
    """Compute Lipschitz bounds for BIC scores w.r.t. data perturbation.

    Provides three methods:
    1. **Spectral bound**: Based on the maximum singular value of the
       covariance matrix (matrix perturbation theory).
    2. **Empirical bound**: Estimated via finite differences with random
       data perturbations.
    3. **Sample-splitting bound**: Tighter bound using independent data
       splits.

    Parameters
    ----------
    score_fn : ScoreFunction
        Scoring function to bound.
    regularisation : float
        Regularisation for covariance estimation.
    n_perturbations : int
        Number of random perturbations for the empirical bound.
    perturbation_scale : float
        Scale of random perturbations (fraction of data std).
    """

    def __init__(
        self,
        score_fn: ScoreFunction,
        regularisation: float = 1e-6,
        n_perturbations: int = 50,
        perturbation_scale: float = 0.01,
    ) -> None:
        self._score_fn = score_fn
        self._reg = regularisation
        self._n_perturbations = n_perturbations
        self._perturbation_scale = perturbation_scale

    # ------------------------------------------------------------------
    # Spectral bound (matrix perturbation theory)
    # ------------------------------------------------------------------

    def spectral_bound(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
    ) -> float:
        """Compute spectral Lipschitz bound for BIC score (loose upper bound).

        Based on Weyl's theorem: if the covariance changes by ΔΣ,
        the eigenvalues change by at most ``‖ΔΣ‖₂``.

        For the BIC score ``−(N/2) log(σ²_res)``, the gradient w.r.t.
        the data is bounded by the spectral norm of the precision
        matrix times a scaling factor.

        .. note::
            This bound grows linearly with *N* and can be vacuously
            large.  Use :meth:`fisher_information_bound` for a tighter,
            *N*-independent alternative.

        Parameters
        ----------
        dag : AdjacencyMatrix
        data : DataMatrix

        Returns
        -------
        float
            Spectral Lipschitz constant.
        """
        n_obs, p = data.shape
        cov = np.cov(data, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += self._reg * np.eye(p)

        # Eigenvalues of the covariance
        eigvals = np.linalg.eigvalsh(cov)
        lambda_min = max(float(eigvals[0]), self._reg)
        lambda_max = float(eigvals[-1])

        # The BIC gradient w.r.t. data perturbation δX scales as:
        # ‖∇_X score‖ ≤ N * λ_max / λ_min^2  (for linear regression)
        # We normalise by p for per-variable scaling.
        L = n_obs * lambda_max / (lambda_min ** 2 * p)
        return float(L)

    # ------------------------------------------------------------------
    # Fisher information bound (N-independent)
    # ------------------------------------------------------------------

    def fisher_information_bound(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
    ) -> float:
        """Tighter Lipschitz bound based on Fisher information.

        For the BIC score -(N/2)log(σ²_res) - (k/2)log(N), the sensitivity
        to a single data perturbation of magnitude ε is bounded by:
        L_FI = √p / (2 * σ²_min)
        where σ²_min is the minimum residual variance across all nodes.
        This bound is O(1/σ²_min) and independent of N.
        """
        n_obs, p = data.shape
        residual_vars = []
        for j in range(p):
            parents = list(np.where(dag[:, j])[0])
            if len(parents) == 0:
                residual_vars.append(float(np.var(data[:, j], ddof=1)))
            else:
                X_pa = data[:, parents]
                y = data[:, j]
                # OLS residual variance
                coef, res, _, _ = np.linalg.lstsq(X_pa, y, rcond=None)
                residuals = y - X_pa @ coef
                residual_vars.append(float(np.var(residuals, ddof=1)))

        sigma2_min = max(min(residual_vars), self._reg)
        L = np.sqrt(p) / (2.0 * sigma2_min)
        return float(L)

    # ------------------------------------------------------------------
    # Empirical bound (finite differences)
    # ------------------------------------------------------------------

    def empirical_bound(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Estimate the Lipschitz constant via finite differences.

        Perturbs the data matrix by small random vectors and measures
        the maximum ratio ``|Δscore| / ‖Δdata‖``.

        Parameters
        ----------
        dag : AdjacencyMatrix
        data : DataMatrix
        rng : np.random.Generator, optional

        Returns
        -------
        float
            Empirical Lipschitz constant estimate.
        """
        if rng is None:
            rng = np.random.default_rng()

        base_score = self._score_fn(dag, data)
        stds = np.std(data, axis=0)
        stds[stds == 0] = 1.0

        max_ratio = 0.0
        for _ in range(self._n_perturbations):
            delta = rng.standard_normal(data.shape) * stds * self._perturbation_scale
            perturbed = data + delta
            new_score = self._score_fn(dag, perturbed)
            score_diff = abs(new_score - base_score)
            data_diff = float(np.linalg.norm(delta))
            if data_diff > 1e-15:
                ratio = score_diff / data_diff
                max_ratio = max(max_ratio, ratio)

        return float(max_ratio)

    # ------------------------------------------------------------------
    # Sample-splitting bound
    # ------------------------------------------------------------------

    def sample_splitting_bound(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
        n_splits: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Tighter Lipschitz bound via sample splitting.

        Splits data into two halves, computes the score on each, and
        bounds the sensitivity based on the variance across splits.

        Parameters
        ----------
        dag : AdjacencyMatrix
        data : DataMatrix
        n_splits : int
        rng : np.random.Generator, optional

        Returns
        -------
        float
        """
        if rng is None:
            rng = np.random.default_rng()

        n_obs = data.shape[0]
        half = n_obs // 2
        scores: List[float] = []

        for _ in range(n_splits):
            perm = rng.permutation(n_obs)
            split_a = data[perm[:half]]
            split_b = data[perm[half: 2 * half]]
            score_a = self._score_fn(dag, split_a)
            score_b = self._score_fn(dag, split_b)
            scores.append(abs(score_a - score_b))

        # The max difference across splits provides an empirical bound
        # on score sensitivity to data sub-sampling
        return float(np.max(scores))

    # ------------------------------------------------------------------
    # Per-edge Lipschitz
    # ------------------------------------------------------------------

    def per_edge_lipschitz(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[Tuple[int, int], float]:
        """Compute per-edge Lipschitz bounds.

        For each edge in the DAG, measures how much the local score
        for that edge's child changes under data perturbation.

        Parameters
        ----------
        dag : AdjacencyMatrix
        data : DataMatrix
        rng : np.random.Generator, optional

        Returns
        -------
        Dict[Tuple[int, int], float]
            ``{(source, target): lipschitz_constant}``.
        """
        if rng is None:
            rng = np.random.default_rng()

        edges = list(zip(*np.nonzero(dag)))
        result: Dict[Tuple[int, int], float] = {}

        for src, tgt in edges:
            # Score with edge
            score_with = self._score_fn(dag, data)
            # Score without edge
            dag_mod = dag.copy()
            dag_mod[src, tgt] = 0
            score_without = self._score_fn(dag_mod, data)
            base_delta = abs(score_with - score_without)

            max_ratio = 0.0
            stds = np.std(data, axis=0)
            stds[stds == 0] = 1.0
            for _ in range(min(self._n_perturbations, 20)):
                perturbation = (
                    rng.standard_normal(data.shape) * stds * self._perturbation_scale
                )
                pert_data = data + perturbation
                pert_with = self._score_fn(dag, pert_data)
                pert_without = self._score_fn(dag_mod, pert_data)
                pert_delta = abs(pert_with - pert_without)
                diff = abs(pert_delta - base_delta)
                norm = float(np.linalg.norm(perturbation))
                if norm > 1e-15:
                    max_ratio = max(max_ratio, diff / norm)

            result[(int(src), int(tgt))] = max_ratio

        return result


# ---------------------------------------------------------------------------
# Perturbation analysis
# ---------------------------------------------------------------------------

class PerturbationAnalyzer:
    """Analyse how much data perturbation is needed to flip edge decisions.

    Parameters
    ----------
    score_fn : ScoreFunction
        Scoring function used for evaluation.
    """

    def __init__(self, score_fn: ScoreFunction) -> None:
        self._score_fn = score_fn

    def edge_flip_radius(
        self,
        dag: "AdjacencyMatrix",
        source: int,
        target: int,
        data: DataMatrix,
        n_trials: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Estimate the minimum data perturbation needed to flip an edge.

        Tries perturbations of increasing magnitude and returns the
        smallest perturbation norm that reverses the edge decision.

        Parameters
        ----------
        dag : AdjacencyMatrix
        source, target : int
        data : DataMatrix
        n_trials : int
        rng : np.random.Generator, optional

        Returns
        -------
        float
            Estimated flip radius (``float('inf')`` if no flip found).
        """
        if rng is None:
            rng = np.random.default_rng()

        # Base decision: is including the edge better?
        dag_with = dag.copy()
        dag_with[source, target] = 1
        dag_without = dag.copy()
        dag_without[source, target] = 0

        base_with = self._score_fn(dag_with, data)
        base_without = self._score_fn(dag_without, data)
        base_includes = base_with > base_without

        stds = np.std(data, axis=0)
        stds[stds == 0] = 1.0
        min_flip_norm = float("inf")

        scales = np.logspace(-3, 0, n_trials)
        for scale in scales:
            perturbation = rng.standard_normal(data.shape) * stds * scale
            pert_data = data + perturbation
            pert_with = self._score_fn(dag_with, pert_data)
            pert_without = self._score_fn(dag_without, pert_data)
            pert_includes = pert_with > pert_without

            if pert_includes != base_includes:
                norm = float(np.linalg.norm(perturbation))
                min_flip_norm = min(min_flip_norm, norm)

        return min_flip_norm

    def stability_radius_all_edges(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
        n_trials: int = 30,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[Tuple[int, int], float]:
        """Compute stability radii for all edges in the DAG.

        Parameters
        ----------
        dag : AdjacencyMatrix
        data : DataMatrix
        n_trials : int
        rng : np.random.Generator, optional

        Returns
        -------
        Dict[Tuple[int, int], float]
        """
        edges = list(zip(*np.nonzero(dag)))
        result: Dict[Tuple[int, int], float] = {}
        for src, tgt in edges:
            result[(int(src), int(tgt))] = self.edge_flip_radius(
                dag, int(src), int(tgt), data, n_trials, rng
            )
        return result
