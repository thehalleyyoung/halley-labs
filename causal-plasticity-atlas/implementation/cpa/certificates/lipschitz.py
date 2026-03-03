"""Fisher Information Lipschitz Bounds.

Provides Lipschitz continuity certificates for causal mechanisms via
Fisher information.  The Fisher information matrix quantifies the
curvature of the log-likelihood surface around the estimated parameters;
its spectral norm yields a Lipschitz constant that bounds how much
the mechanism output can change under parameter perturbation.

Classes
-------
FisherInformationBound
    Compute Fisher information matrix, Lipschitz constants, and
    parameter sensitivity bounds for individual mechanisms.
MechanismStabilityBound
    Aggregate Fisher bounds into stability radii and certificates.

Theory reference: §5 of the CPA monograph (Fisher information metric
on the parameter manifold Theta_i^K).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sla


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FisherBound:
    """Result of a Fisher information analysis for one mechanism."""

    node: int
    parents: List[int]
    fisher_matrix: NDArray          # (p+1, p+1) Fisher information matrix
    lipschitz_constant: float       # operator norm of Fisher matrix
    eigenvalues: NDArray            # eigenvalues of Fisher matrix
    n_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StabilityCertificate:
    """Certificate issued from Fisher information bounds."""

    node: int
    stability_radius: float
    lipschitz_constant: float
    confidence_level: float
    worst_case_perturbation: float
    certified: bool
    tolerance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fisher Information Bound
# ---------------------------------------------------------------------------

class FisherInformationBound:
    """Fisher information bounds for mechanism parameters.

    Assumes a linear-Gaussian mechanism:
        X_j = sum_{k in pa(j)} beta_k * X_k + epsilon_j,
        epsilon_j ~ N(0, sigma_j^2)

    so the parameter vector theta_j = (beta_{pa(j)}, sigma_j^2).

    Parameters
    ----------
    adjacency : (p, p) adjacency matrix (adj[i, j] != 0 => i -> j)
    data : (n, p) observation matrix
    """

    def __init__(self, adjacency: NDArray, data: NDArray) -> None:
        self._adj = np.asarray(adjacency, dtype=np.float64)
        self._data = np.asarray(data, dtype=np.float64)
        self._n, self._p = self._data.shape
        if self._adj.shape != (self._p, self._p):
            raise ValueError(
                f"adjacency shape {self._adj.shape} != data variables ({self._p}, {self._p})"
            )
        self._cache: Dict[int, FisherBound] = {}

    # -- Public API --

    def compute_fisher_matrix(
        self, node: int, parents: Optional[List[int]] = None
    ) -> NDArray:
        """Compute the Fisher information matrix for the mechanism at *node*.

        For a Gaussian linear model the Fisher matrix has a closed form:
            I(theta) = (1/sigma^2) * X_pa^T X_pa   for regression coefs
        augmented with the variance entry 1/(2 sigma^4).

        Parameters
        ----------
        node : target variable index
        parents : parent indices (inferred from adjacency if None)

        Returns
        -------
        (d+1, d+1) Fisher information matrix where d = |parents|.
        The last row/column corresponds to the variance parameter.
        """
        if parents is None:
            parents = self._parents(node)
        parents = sorted(parents)

        y = self._data[:, node]
        if parents:
            X = self._data[:, parents]
        else:
            X = np.ones((self._n, 0))

        # Fit OLS
        beta, sigma2 = self._fit_ols(X, y)
        sigma2 = max(sigma2, 1e-12)
        d = len(parents)

        fisher = np.zeros((d + 1, d + 1), dtype=np.float64)

        # Block for regression coefficients: (1/sigma^2) * X^T X
        if d > 0:
            fisher[:d, :d] = (X.T @ X) / sigma2

        # Entry for variance parameter: n / (2 sigma^4)
        fisher[d, d] = self._n / (2.0 * sigma2 ** 2)

        # Cache result
        evals = np.linalg.eigvalsh(fisher)
        lip = float(np.max(np.abs(evals)))
        self._cache[node] = FisherBound(
            node=node,
            parents=parents,
            fisher_matrix=fisher,
            lipschitz_constant=lip,
            eigenvalues=evals,
            n_samples=self._n,
        )
        return fisher

    def _score_function(
        self,
        data: NDArray,
        node: int,
        parents: List[int],
        params: NDArray,
    ) -> NDArray:
        """Score function: gradient of log-likelihood w.r.t. params.

        For each observation i, returns nabla_theta log p(x_i | theta).

        Parameters
        ----------
        data : (n, p) observations
        node : target variable
        parents : parent indices
        params : (d+1,) — first d are regression coefs, last is sigma^2

        Returns
        -------
        (n, d+1) matrix of per-observation score vectors
        """
        n = data.shape[0]
        d = len(parents)
        beta = params[:d]
        sigma2 = max(float(params[d]), 1e-12)

        y = data[:, node]
        if d > 0:
            X = data[:, parents]
            residual = y - X @ beta
        else:
            X = np.empty((n, 0))
            residual = y.copy()

        scores = np.zeros((n, d + 1), dtype=np.float64)

        # Gradient w.r.t. beta: (1/sigma^2) * X_i * residual_i
        if d > 0:
            scores[:, :d] = (X * residual[:, np.newaxis]) / sigma2

        # Gradient w.r.t. sigma^2: -1/(2 sigma^2) + residual^2 / (2 sigma^4)
        scores[:, d] = -1.0 / (2.0 * sigma2) + residual ** 2 / (2.0 * sigma2 ** 2)

        return scores

    def _outer_product_gradient(self, scores: NDArray) -> NDArray:
        """Average outer product of score vectors.

        This provides an alternative (and numerically robust) estimator
        of the Fisher information matrix.

        Parameters
        ----------
        scores : (n, d+1) per-observation scores

        Returns
        -------
        (d+1, d+1) average outer product
        """
        n = scores.shape[0]
        return (scores.T @ scores) / n

    def lipschitz_constant(self, node: int) -> float:
        """Lipschitz constant for the mechanism at *node*.

        This is the operator norm (largest singular value) of the
        Fisher information matrix.
        """
        if node not in self._cache:
            self.compute_fisher_matrix(node)
        return self._cache[node].lipschitz_constant

    def _operator_norm(self, matrix: NDArray) -> float:
        """Largest singular value of *matrix*."""
        sv = np.linalg.svd(matrix, compute_uv=False)
        return float(sv[0]) if len(sv) > 0 else 0.0

    def parameter_sensitivity(
        self, node: int, perturbation: float
    ) -> float:
        """Upper bound on output change given parameter perturbation.

        By the Lipschitz bound:
            |f(theta + delta) - f(theta)| <= L * ||delta||

        Parameters
        ----------
        node : variable index
        perturbation : ||delta|| (L2 norm of parameter perturbation)

        Returns
        -------
        Upper bound on mechanism output change.
        """
        L = self.lipschitz_constant(node)
        return L * perturbation

    def get_bound(self, node: int) -> FisherBound:
        """Return the cached FisherBound for *node*."""
        if node not in self._cache:
            self.compute_fisher_matrix(node)
        return self._cache[node]

    # -- Helpers --

    def _parents(self, node: int) -> List[int]:
        """Return sorted parent indices of *node* from adjacency."""
        return sorted(int(i) for i in range(self._p) if self._adj[i, node] != 0)

    @staticmethod
    def _fit_ols(X: NDArray, y: NDArray) -> Tuple[NDArray, float]:
        """OLS fit returning (beta, residual_variance)."""
        n = len(y)
        if X.shape[1] == 0:
            return np.array([]), float(np.var(y, ddof=0))
        beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        dof = max(n - X.shape[1], 1)
        sigma2 = float(np.sum(residuals ** 2) / dof)
        return beta, sigma2


# ---------------------------------------------------------------------------
# Mechanism Stability Bound
# ---------------------------------------------------------------------------

class MechanismStabilityBound:
    """Aggregate Fisher information bounds into stability certificates.

    Parameters
    ----------
    fisher_bounds : FisherInformationBound
        Precomputed Fisher bounds for the model.
    """

    def __init__(self, fisher_bounds: FisherInformationBound) -> None:
        self._fb = fisher_bounds

    def worst_case_perturbation(self, node: int, epsilon: float) -> float:
        """Worst-case mechanism output change for an epsilon-ball perturbation.

        Parameters
        ----------
        node : variable index
        epsilon : radius of parameter perturbation ball

        Returns
        -------
        Upper bound on output change.
        """
        bound = self._fb.get_bound(node)
        return self._spectral_bound(bound.fisher_matrix, epsilon)

    def stability_radius(self, node: int, tolerance: float) -> float:
        """Radius within which mechanism stays within *tolerance*.

        Solves  L * r = tolerance  =>  r = tolerance / L.

        Parameters
        ----------
        node : variable index
        tolerance : maximum allowable output change

        Returns
        -------
        Maximum perturbation radius.
        """
        L = self._fb.lipschitz_constant(node)
        if L < 1e-15:
            return float("inf")
        return tolerance / L

    def _spectral_bound(self, fisher_matrix: NDArray, epsilon: float) -> float:
        """Spectral norm bound on output change.

        ||delta_output|| <= sqrt(lambda_max(F)) * epsilon
        where lambda_max is the largest eigenvalue of the Fisher matrix.
        """
        evals = np.linalg.eigvalsh(fisher_matrix)
        lambda_max = float(np.max(np.abs(evals)))
        return np.sqrt(max(lambda_max, 0.0)) * epsilon

    def certificate_from_fisher(
        self, node: int, confidence_level: float = 0.95, tolerance: float = 0.1
    ) -> StabilityCertificate:
        """Generate a stability certificate from Fisher bounds.

        The certificate is issued if the stability radius (at the given
        tolerance) exceeds a threshold derived from the asymptotic
        distribution of the MLE.

        Under regularity conditions the MLE has
            sqrt(n) * (theta_hat - theta_0) ~ N(0, I(theta)^{-1})
        so a (1-alpha) confidence region has radius ~ chi_d(alpha) / sqrt(n * lambda_min).

        Parameters
        ----------
        node : variable index
        confidence_level : desired confidence level (e.g. 0.95)
        tolerance : maximum allowable mechanism change

        Returns
        -------
        StabilityCertificate
        """
        from scipy.stats import chi2

        bound = self._fb.get_bound(node)
        d = bound.fisher_matrix.shape[0]
        n = bound.n_samples

        evals = bound.eigenvalues
        lambda_min = float(np.min(evals[evals > 1e-12])) if np.any(evals > 1e-12) else 1e-12

        # Radius of the confidence ellipsoid for the MLE
        chi2_val = chi2.ppf(confidence_level, df=d)
        mle_radius = np.sqrt(chi2_val / (n * lambda_min)) if n * lambda_min > 0 else float("inf")

        L = bound.lipschitz_constant
        stability_rad = self.stability_radius(node, tolerance)

        # Certified if the MLE uncertainty radius is within the stability radius
        certified = mle_radius < stability_rad

        worst_case = self.worst_case_perturbation(node, mle_radius)

        return StabilityCertificate(
            node=node,
            stability_radius=stability_rad,
            lipschitz_constant=L,
            confidence_level=confidence_level,
            worst_case_perturbation=worst_case,
            certified=certified,
            tolerance=tolerance,
            metadata={
                "mle_radius": mle_radius,
                "lambda_min": lambda_min,
                "lambda_max": float(np.max(evals)),
                "n_params": d,
                "n_samples": n,
            },
        )
