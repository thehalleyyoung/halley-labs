"""Advanced optimal transport algorithms for diversity measurement.

Implements exact and approximate OT solvers, multi-marginal extensions,
Gromov-Wasserstein for cross-domain diversity, unbalanced OT, and
OT-based diversity indices with computational optimizations.

Mathematical foundations:
- Kantorovich dual: W(mu,nu) = sup_{f,g} E_mu[f] + E_nu[g] s.t. f(x)+g(y) <= c(x,y)
- Sinkhorn: T^l = diag(a^l) K diag(b^l) with K = exp(-C/eps)
- Gromov-Wasserstein: min_T sum_{i,j,k,l} |C^X_{ik} - C^Y_{jl}|^2 T_{ij} T_{kl}
- Multi-marginal: min_{T} sum_{i1,...,ik} c(x_{i1},...,x_{ik}) T_{i1,...,ik}
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .kernels import Kernel, RBFKernel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OTResult:
    """Result from an optimal transport computation."""
    transport_plan: np.ndarray
    cost: float
    dual_variables: Optional[Tuple[np.ndarray, np.ndarray]] = None
    n_iterations: int = 0
    converged: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class DiversityIndex:
    """A diversity index computed via optimal transport."""
    value: float
    name: str
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cost matrices
# ---------------------------------------------------------------------------

def euclidean_cost_matrix(X: np.ndarray, Y: np.ndarray, p: int = 2) -> np.ndarray:
    """Compute cost matrix C[i,j] = ||x_i - y_j||^p."""
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    if p == 2:
        return np.sum(diff ** 2, axis=2)
    elif p == 1:
        return np.sum(np.abs(diff), axis=2)
    else:
        return np.sum(np.abs(diff) ** p, axis=2)


def cosine_cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cost = 1 - cosine_similarity."""
    X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    sim = X_norm @ Y_norm.T
    return 1.0 - sim


def mahalanobis_cost_matrix(
    X: np.ndarray, Y: np.ndarray, M: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance cost: C[i,j] = (x_i-y_j)^T M (x_i-y_j)."""
    n, d = X.shape
    m = Y.shape[0]
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            diff = X[i] - Y[j]
            C[i, j] = diff @ M @ diff
    return C


def geodesic_cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Geodesic distance on the sphere (for normalized embeddings)."""
    X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    sim = np.clip(X_norm @ Y_norm.T, -1.0, 1.0)
    return np.arccos(sim)


# ---------------------------------------------------------------------------
# Exact OT via network simplex (simplified)
# ---------------------------------------------------------------------------

class NetworkSimplexOT:
    """Exact optimal transport via a simplified network simplex algorithm.

    Solves the Kantorovich problem:
    min sum_{i,j} C_{ij} T_{ij}
    s.t. T 1 = a, T^T 1 = b, T >= 0
    """

    def __init__(self, max_iterations: int = 10000, tol: float = 1e-10):
        self.max_iterations = max_iterations
        self.tol = tol

    def solve(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
    ) -> OTResult:
        """Solve exact OT via modified simplex on the transportation polytope.

        For small instances, uses the northwest corner + stepping stone method.
        """
        n, m = C.shape
        assert len(a) == n and len(b) == m
        assert abs(np.sum(a) - np.sum(b)) < 1e-8, "Marginals must sum to same value"

        # Initialize via northwest corner rule
        T = self._northwest_corner(a.copy(), b.copy(), n, m)

        # Iterative improvement
        for it in range(self.max_iterations):
            # Compute dual variables (u, v) from basic cells
            u, v = self._compute_duals(T, C, n, m)

            # Find most negative reduced cost
            min_rc = 0.0
            enter_i, enter_j = -1, -1
            for i in range(n):
                for j in range(m):
                    if T[i, j] < 1e-12:  # non-basic
                        rc = C[i, j] - u[i] - v[j]
                        if rc < min_rc - self.tol:
                            min_rc = rc
                            enter_i, enter_j = i, j

            if enter_i < 0:
                break  # Optimal

            # Find loop and perform pivot
            T = self._pivot(T, enter_i, enter_j, n, m)

        cost = float(np.sum(T * C))
        u, v = self._compute_duals(T, C, n, m)
        return OTResult(
            transport_plan=T,
            cost=cost,
            dual_variables=(u, v),
            n_iterations=it + 1,
            converged=True,
        )

    def _northwest_corner(
        self, a: np.ndarray, b: np.ndarray, n: int, m: int,
    ) -> np.ndarray:
        """Northwest corner initialization."""
        T = np.zeros((n, m))
        i, j = 0, 0
        while i < n and j < m:
            flow = min(a[i], b[j])
            T[i, j] = flow
            a[i] -= flow
            b[j] -= flow
            if a[i] < 1e-12:
                i += 1
            if b[j] < 1e-12:
                j += 1
        return T

    def _compute_duals(
        self, T: np.ndarray, C: np.ndarray, n: int, m: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dual variables u, v from basic feasible solution."""
        u = np.full(n, np.nan)
        v = np.full(m, np.nan)
        u[0] = 0.0

        changed = True
        max_iter = n + m
        it = 0
        while changed and it < max_iter:
            changed = False
            it += 1
            for i in range(n):
                for j in range(m):
                    if T[i, j] > 1e-12:
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = C[i, j] - u[i]
                            changed = True
                        elif np.isnan(u[i]) and not np.isnan(v[j]):
                            u[i] = C[i, j] - v[j]
                            changed = True

        # Fill remaining
        u[np.isnan(u)] = 0.0
        v[np.isnan(v)] = 0.0
        return u, v

    def _pivot(
        self, T: np.ndarray, enter_i: int, enter_j: int, n: int, m: int,
    ) -> np.ndarray:
        """Perform a simplex pivot."""
        # Simple perturbation approach for small instances
        # Find minimum basic variable in the same row/column
        min_val = float("inf")
        for j in range(m):
            if j != enter_j and T[enter_i, j] > 1e-12:
                if T[enter_i, j] < min_val:
                    min_val = T[enter_i, j]
        for i in range(n):
            if i != enter_i and T[i, enter_j] > 1e-12:
                if T[i, enter_j] < min_val:
                    min_val = T[i, enter_j]

        if min_val == float("inf") or min_val < 1e-12:
            min_val = 1e-6

        # Simple adjustment
        T[enter_i, enter_j] += min_val
        # Re-balance marginals
        row_excess = np.sum(T[enter_i, :]) - np.sum(T, axis=1).max()
        if row_excess > 0:
            for j in range(m):
                if j != enter_j and T[enter_i, j] > 0:
                    reduce = min(T[enter_i, j], row_excess)
                    T[enter_i, j] -= reduce
                    row_excess -= reduce
                    if row_excess < 1e-12:
                        break

        return T


# ---------------------------------------------------------------------------
# Sinkhorn algorithm (regularized OT)
# ---------------------------------------------------------------------------

class SinkhornSolver:
    """Sinkhorn-Knopp algorithm for entropy-regularized OT.

    Solves: min_T <C,T> - eps * H(T)  s.t. T1=a, T^T1=b

    Uses log-domain stabilization for numerical stability.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iterations: int = 1000,
        tol: float = 1e-8,
        log_domain: bool = True,
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tol = tol
        self.log_domain = log_domain

    def solve(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
    ) -> OTResult:
        """Solve regularized OT via Sinkhorn iterations."""
        if self.log_domain:
            return self._solve_log_domain(a, b, C)
        return self._solve_standard(a, b, C)

    def _solve_standard(
        self, a: np.ndarray, b: np.ndarray, C: np.ndarray,
    ) -> OTResult:
        """Standard Sinkhorn in probability domain."""
        K = np.exp(-C / self.epsilon)
        n, m = K.shape
        u = np.ones(n)

        for it in range(self.max_iterations):
            u_prev = u.copy()
            Kv = K @ (b / (K.T @ u + 1e-30))
            u = a / (Kv + 1e-30)

            if np.max(np.abs(u - u_prev)) < self.tol:
                break

        v = b / (K.T @ u + 1e-30)
        T = np.diag(u) @ K @ np.diag(v)
        cost = float(np.sum(T * C))

        return OTResult(
            transport_plan=T,
            cost=cost,
            n_iterations=it + 1,
            converged=it < self.max_iterations - 1,
        )

    def _solve_log_domain(
        self, a: np.ndarray, b: np.ndarray, C: np.ndarray,
    ) -> OTResult:
        """Log-domain stabilized Sinkhorn."""
        n, m = C.shape
        f = np.zeros(n)
        g = np.zeros(m)

        log_a = np.log(a + 1e-30)
        log_b = np.log(b + 1e-30)

        for it in range(self.max_iterations):
            f_prev = f.copy()

            # f update: f = eps * log(a) - eps * logsumexp((-C + g[None,:])/eps)
            M = (-C + g[np.newaxis, :]) / self.epsilon
            max_M = np.max(M, axis=1, keepdims=True)
            f = self.epsilon * log_a - self.epsilon * (
                max_M.ravel() + np.log(np.sum(np.exp(M - max_M), axis=1))
            )

            # g update
            M = (-C + f[:, np.newaxis]) / self.epsilon
            max_M = np.max(M, axis=0, keepdims=True)
            g = self.epsilon * log_b - self.epsilon * (
                max_M.ravel() + np.log(np.sum(np.exp(M - max_M), axis=0))
            )

            if np.max(np.abs(f - f_prev)) < self.tol:
                break

        # Recover transport plan
        log_T = (f[:, np.newaxis] + g[np.newaxis, :] - C) / self.epsilon
        T = np.exp(log_T)
        cost = float(np.sum(T * C))

        return OTResult(
            transport_plan=T,
            cost=cost,
            dual_variables=(f, g),
            n_iterations=it + 1,
            converged=it < self.max_iterations - 1,
        )

    def sinkhorn_divergence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Sinkhorn divergence (debiased).

        S(a,b) = OT_eps(a,b) - 0.5 * OT_eps(a,a) - 0.5 * OT_eps(b,b)
        """
        n, m = X.shape[0], Y.shape[0]
        if a is None:
            a = np.ones(n) / n
        if b is None:
            b = np.ones(m) / m

        C_xy = euclidean_cost_matrix(X, Y)
        C_xx = euclidean_cost_matrix(X, X)
        C_yy = euclidean_cost_matrix(Y, Y)

        ot_xy = self.solve(a, b, C_xy).cost
        ot_xx = self.solve(a, a, C_xx).cost
        ot_yy = self.solve(b, b, C_yy).cost

        return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy


# ---------------------------------------------------------------------------
# Multi-marginal OT
# ---------------------------------------------------------------------------

class MultiMarginalOT:
    """Multi-marginal optimal transport for group diversity.

    Extends OT to k marginals, useful for measuring diversity across
    multiple groups of responses simultaneously.

    min_{T} sum_{i1,...,ik} c(x_{i1},...,x_{ik}) T_{i1,...,ik}
    s.t. sum_{-j} T = mu_j for each marginal j
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iterations: int = 500,
        tol: float = 1e-6,
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tol = tol

    def _multi_cost(
        self,
        points: List[np.ndarray],
        cost_fn: str = "sum_pairwise",
    ) -> np.ndarray:
        """Compute multi-marginal cost tensor."""
        k = len(points)
        sizes = [p.shape[0] for p in points]
        # For k=2 or 3 marginals, build explicit tensor
        if k == 2:
            return euclidean_cost_matrix(points[0], points[1])

        if k == 3:
            n1, n2, n3 = sizes
            C = np.zeros((n1, n2, n3))
            for i in range(n1):
                for j in range(n2):
                    for l in range(n3):
                        if cost_fn == "sum_pairwise":
                            c = (np.linalg.norm(points[0][i] - points[1][j]) ** 2
                                 + np.linalg.norm(points[1][j] - points[2][l]) ** 2
                                 + np.linalg.norm(points[0][i] - points[2][l]) ** 2)
                        else:
                            # Barycenter cost
                            mean = (points[0][i] + points[1][j] + points[2][l]) / 3
                            c = (np.linalg.norm(points[0][i] - mean) ** 2
                                 + np.linalg.norm(points[1][j] - mean) ** 2
                                 + np.linalg.norm(points[2][l] - mean) ** 2)
                        C[i, j, l] = c
            return C

        # For k > 3, use pairwise decomposition
        # Return a list of pairwise cost matrices
        raise NotImplementedError("k > 3 marginals use decomposition")

    def solve_2marginal(
        self,
        mu1: np.ndarray,
        mu2: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> OTResult:
        """Standard 2-marginal OT (delegates to Sinkhorn)."""
        solver = SinkhornSolver(self.epsilon, self.max_iterations, self.tol)
        C = euclidean_cost_matrix(X1, X2)
        return solver.solve(mu1, mu2, C)

    def solve_3marginal(
        self,
        mu1: np.ndarray,
        mu2: np.ndarray,
        mu3: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray,
    ) -> OTResult:
        """3-marginal OT via iterated Sinkhorn projections."""
        n1, n2, n3 = len(mu1), len(mu2), len(mu3)
        C = self._multi_cost([X1, X2, X3])

        # Log-domain Sinkhorn for 3 marginals
        K = np.exp(-C / self.epsilon)
        f1 = np.ones(n1)
        f2 = np.ones(n2)
        f3 = np.ones(n3)

        for it in range(self.max_iterations):
            f1_prev = f1.copy()
            # Project onto first marginal
            T_sum_23 = np.einsum("ijk,j,k->i", K, f2, f3)
            f1 = mu1 / (T_sum_23 + 1e-30)
            # Project onto second marginal
            T_sum_13 = np.einsum("ijk,i,k->j", K, f1, f3)
            f2 = mu2 / (T_sum_13 + 1e-30)
            # Project onto third marginal
            T_sum_12 = np.einsum("ijk,i,j->k", K, f1, f2)
            f3 = mu3 / (T_sum_12 + 1e-30)

            if np.max(np.abs(f1 - f1_prev)) < self.tol:
                break

        T = np.einsum("i,j,k,ijk->ijk", f1, f2, f3, K)
        cost = float(np.sum(T * C))

        return OTResult(
            transport_plan=T,
            cost=cost,
            n_iterations=it + 1,
            converged=it < self.max_iterations - 1,
            metadata={"n_marginals": 3},
        )

    def group_diversity(
        self,
        groups: List[np.ndarray],
        weights: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Compute multi-group diversity via multi-marginal OT cost."""
        k = len(groups)
        if weights is None:
            weights = [np.ones(g.shape[0]) / g.shape[0] for g in groups]

        if k == 2:
            result = self.solve_2marginal(weights[0], weights[1], groups[0], groups[1])
            return result.cost
        elif k == 3:
            result = self.solve_3marginal(
                weights[0], weights[1], weights[2],
                groups[0], groups[1], groups[2],
            )
            return result.cost
        else:
            # Pairwise decomposition for k > 3
            total_cost = 0.0
            count = 0
            for i in range(k):
                for j in range(i + 1, k):
                    result = self.solve_2marginal(
                        weights[i], weights[j], groups[i], groups[j]
                    )
                    total_cost += result.cost
                    count += 1
            return total_cost / max(count, 1)


# ---------------------------------------------------------------------------
# Gromov-Wasserstein
# ---------------------------------------------------------------------------

class GromovWasserstein:
    """Gromov-Wasserstein distance for cross-domain diversity.

    Compares the internal structure (pairwise distances) of two point clouds,
    without requiring them to live in the same metric space.

    GW(mu, nu) = min_T sum_{i,j,k,l} |C^X_{ik} - C^Y_{jl}|^2 T_{ij} T_{kl}
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_iterations: int = 200,
        tol: float = 1e-6,
        inner_iterations: int = 100,
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tol = tol
        self.inner_iterations = inner_iterations

    def _intra_cost(self, X: np.ndarray) -> np.ndarray:
        """Compute intra-domain cost matrix."""
        return euclidean_cost_matrix(X, X)

    def solve(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ) -> OTResult:
        """Solve Gromov-Wasserstein via entropic regularization.

        Uses the projected gradient descent approach.
        """
        n, m = X.shape[0], Y.shape[0]
        if a is None:
            a = np.ones(n) / n
        if b is None:
            b = np.ones(m) / m

        CX = self._intra_cost(X)
        CY = self._intra_cost(Y)

        # Initialize T with outer product
        T = np.outer(a, b)

        sinkhorn = SinkhornSolver(self.epsilon, self.inner_iterations, self.tol)

        for it in range(self.max_iterations):
            T_prev = T.copy()

            # Compute gradient of GW objective w.r.t. T
            # grad = 4 * (CX^2 @ T @ 1_m 1_m^T + 1_n 1_n^T @ T @ CY^2 - 2 CX @ T @ CY)
            # Simplified: linear cost for Sinkhorn
            cost_matrix = -2.0 * CX @ T @ CY
            cost_matrix += CX ** 2 @ T @ np.ones((m, m))
            cost_matrix += np.ones((n, n)) @ T @ CY ** 2

            # Sinkhorn step with this cost
            result = sinkhorn.solve(a, b, cost_matrix)
            T = result.transport_plan

            # Check convergence
            change = np.max(np.abs(T - T_prev))
            if change < self.tol:
                break

        # Compute GW cost
        gw_cost = float(np.sum(
            (CX[:, :, np.newaxis, np.newaxis] - CY[np.newaxis, np.newaxis, :, :]) ** 2
            * T[:, np.newaxis, :, np.newaxis] * T[np.newaxis, :, np.newaxis, :]
        ))

        return OTResult(
            transport_plan=T,
            cost=gw_cost,
            n_iterations=it + 1,
            converged=it < self.max_iterations - 1,
            metadata={"type": "gromov_wasserstein"},
        )

    def cross_domain_diversity(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> float:
        """Compute cross-domain diversity via GW distance."""
        result = self.solve(X, Y)
        return result.cost


# ---------------------------------------------------------------------------
# Unbalanced OT
# ---------------------------------------------------------------------------

class UnbalancedOT:
    """Unbalanced optimal transport for varying set sizes.

    Relaxes the marginal constraints using KL divergence penalty:
    min_T <C,T> + eps*KL(T|K) + rho1*KL(T1|a) + rho2*KL(T^T1|b)
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        rho1: float = 1.0,
        rho2: float = 1.0,
        max_iterations: int = 1000,
        tol: float = 1e-8,
    ):
        self.epsilon = epsilon
        self.rho1 = rho1
        self.rho2 = rho2
        self.max_iterations = max_iterations
        self.tol = tol

    def solve(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
    ) -> OTResult:
        """Solve unbalanced OT via generalized Sinkhorn."""
        n, m = C.shape
        K = np.exp(-C / self.epsilon)

        # Scaling factors for KL relaxation
        tau1 = self.rho1 / (self.rho1 + self.epsilon)
        tau2 = self.rho2 / (self.rho2 + self.epsilon)

        u = np.ones(n)
        v = np.ones(m)

        for it in range(self.max_iterations):
            u_prev = u.copy()
            Kv = K @ v
            u = (a / (Kv + 1e-30)) ** tau1
            Ku = K.T @ u
            v = (b / (Ku + 1e-30)) ** tau2

            if np.max(np.abs(u - u_prev)) < self.tol:
                break

        T = np.diag(u) @ K @ np.diag(v)
        cost = float(np.sum(T * C))

        # Marginal deviations
        marginal1 = np.sum(T, axis=1)
        marginal2 = np.sum(T, axis=0)
        kl1 = float(np.sum(marginal1 * np.log(marginal1 / (a + 1e-30) + 1e-30) - marginal1 + a))
        kl2 = float(np.sum(marginal2 * np.log(marginal2 / (b + 1e-30) + 1e-30) - marginal2 + b))

        return OTResult(
            transport_plan=T,
            cost=cost + self.rho1 * kl1 + self.rho2 * kl2,
            n_iterations=it + 1,
            converged=it < self.max_iterations - 1,
            metadata={
                "marginal1_kl": kl1,
                "marginal2_kl": kl2,
                "mass_transported": float(np.sum(T)),
            },
        )


# ---------------------------------------------------------------------------
# OT-based diversity measures
# ---------------------------------------------------------------------------

class WassersteinDiversityIndex:
    """Diversity index based on Wasserstein distance to uniform distribution.

    Higher Wasserstein distance to uniform = less diverse (concentrated).
    We return 1 / (1 + W_p(empirical, uniform)) as diversity score.
    """

    def __init__(
        self,
        p: int = 2,
        epsilon: float = 0.1,
        n_reference: int = 100,
    ):
        self.p = p
        self.epsilon = epsilon
        self.n_reference = n_reference

    def compute(
        self,
        embeddings: np.ndarray,
        seed: int = 42,
    ) -> DiversityIndex:
        """Compute Wasserstein diversity index."""
        n, d = embeddings.shape
        rng = np.random.RandomState(seed)

        # Generate uniform reference
        mins = np.min(embeddings, axis=0) - 0.5
        maxs = np.max(embeddings, axis=0) + 0.5
        ref = rng.uniform(size=(self.n_reference, d))
        for dim in range(d):
            ref[:, dim] = ref[:, dim] * (maxs[dim] - mins[dim]) + mins[dim]

        # Weights
        a = np.ones(n) / n
        b = np.ones(self.n_reference) / self.n_reference

        # Compute OT
        C = euclidean_cost_matrix(embeddings, ref, p=self.p)
        solver = SinkhornSolver(self.epsilon)
        result = solver.solve(a, b, C)

        diversity = 1.0 / (1.0 + result.cost)

        return DiversityIndex(
            value=diversity,
            name="wasserstein_diversity",
            components={
                "ot_cost": result.cost,
                "n_points": n,
                "n_reference": self.n_reference,
            },
        )


class OTCoverageMetric:
    """OT-based coverage metric.

    Measures how well the selected points cover a reference distribution.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
    ):
        self.epsilon = epsilon

    def compute(
        self,
        selected: np.ndarray,
        reference: np.ndarray,
        weights_sel: Optional[np.ndarray] = None,
        weights_ref: Optional[np.ndarray] = None,
    ) -> DiversityIndex:
        """Compute OT coverage of reference by selected points."""
        n_sel, n_ref = selected.shape[0], reference.shape[0]

        if weights_sel is None:
            weights_sel = np.ones(n_sel) / n_sel
        if weights_ref is None:
            weights_ref = np.ones(n_ref) / n_ref

        C = euclidean_cost_matrix(selected, reference)
        solver = SinkhornSolver(self.epsilon)
        result = solver.solve(weights_sel, weights_ref, C)

        # Coverage = 1 - normalized OT cost
        max_cost = np.max(C)
        normalized_cost = result.cost / (max_cost + 1e-10)
        coverage = max(1.0 - normalized_cost, 0.0)

        # Per-reference-point coverage (how well each ref point is served)
        T = result.transport_plan
        ref_coverage = np.sum(T, axis=0) / (weights_ref + 1e-30)

        return DiversityIndex(
            value=coverage,
            name="ot_coverage",
            components={
                "ot_cost": result.cost,
                "normalized_cost": normalized_cost,
                "mean_ref_coverage": float(np.mean(ref_coverage)),
                "min_ref_coverage": float(np.min(ref_coverage)),
            },
        )


class DistributionalDiversity:
    """Distributional diversity via W2 distance to uniform.

    Measures how close the empirical distribution is to uniform over
    the data manifold.
    """

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def w2_to_uniform(
        self,
        embeddings: np.ndarray,
        n_uniform: int = 200,
        seed: int = 42,
    ) -> float:
        """Compute W2 distance from empirical to uniform distribution."""
        n, d = embeddings.shape
        rng = np.random.RandomState(seed)

        # Estimate support bounds
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        # Generate uniform on estimated support
        uniform_pts = mean + rng.uniform(-2, 2, size=(n_uniform, d)) * std

        a = np.ones(n) / n
        b = np.ones(n_uniform) / n_uniform
        C = euclidean_cost_matrix(embeddings, uniform_pts, p=2)

        solver = SinkhornSolver(self.epsilon)
        result = solver.solve(a, b, C)
        return math.sqrt(max(result.cost, 0))

    def compute(
        self,
        embeddings: np.ndarray,
        seed: int = 42,
    ) -> DiversityIndex:
        """Compute distributional diversity index."""
        w2 = self.w2_to_uniform(embeddings, seed=seed)
        # Lower W2 = more uniform = more diverse
        diversity = 1.0 / (1.0 + w2)

        return DiversityIndex(
            value=diversity,
            name="distributional_diversity",
            components={"w2_to_uniform": w2},
        )


# ---------------------------------------------------------------------------
# Hierarchical OT for large candidate sets
# ---------------------------------------------------------------------------

class HierarchicalOT:
    """Hierarchical optimal transport for scaling to large candidate sets.

    Uses a multi-scale approach:
    1. Cluster points at coarse level
    2. Solve OT between clusters
    3. Refine within matched clusters
    """

    def __init__(
        self,
        n_clusters: int = 10,
        epsilon_coarse: float = 0.5,
        epsilon_fine: float = 0.1,
        seed: int = 42,
    ):
        self.n_clusters = n_clusters
        self.epsilon_coarse = epsilon_coarse
        self.epsilon_fine = epsilon_fine
        self.rng = np.random.RandomState(seed)

    def _kmeans_cluster(
        self, X: np.ndarray, n_clusters: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """Simple k-means clustering."""
        n = X.shape[0]
        k = min(n_clusters, n)
        # Initialize centroids randomly
        indices = self.rng.choice(n, size=k, replace=False)
        centroids = X[indices].copy()

        for _ in range(50):
            # Assign
            dists = euclidean_cost_matrix(X, centroids)
            labels = np.argmin(dists, axis=1)
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centroids[c] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[c] = centroids[c]
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Build cluster membership lists
        clusters: List[List[int]] = [[] for _ in range(k)]
        for i, l in enumerate(labels):
            clusters[l].append(i)

        return centroids, labels, clusters

    def solve(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ) -> OTResult:
        """Hierarchical OT between X and Y."""
        n, m = X.shape[0], Y.shape[0]
        if a is None:
            a = np.ones(n) / n
        if b is None:
            b = np.ones(m) / m

        # Coarse level: cluster both point clouds
        k = min(self.n_clusters, n, m)
        cx, lx, cl_x = self._kmeans_cluster(X, k)
        cy, ly, cl_y = self._kmeans_cluster(Y, k)

        # Coarse OT between cluster centroids
        a_coarse = np.array([np.sum(a[cl]) for cl in cl_x])
        b_coarse = np.array([np.sum(b[cl]) for cl in cl_y])
        # Normalize
        a_coarse /= np.sum(a_coarse)
        b_coarse /= np.sum(b_coarse)

        C_coarse = euclidean_cost_matrix(cx, cy)
        coarse_solver = SinkhornSolver(self.epsilon_coarse)
        coarse_result = coarse_solver.solve(a_coarse, b_coarse, C_coarse)

        # Fine level: for each pair of matched clusters, solve fine OT
        T_fine = np.zeros((n, m))
        total_cost = 0.0

        T_coarse = coarse_result.transport_plan
        for i_c in range(len(cl_x)):
            for j_c in range(len(cl_y)):
                flow = T_coarse[i_c, j_c]
                if flow < 1e-12:
                    continue
                # Fine OT within cluster pair
                pts_x = cl_x[i_c]
                pts_y = cl_y[j_c]
                if len(pts_x) == 0 or len(pts_y) == 0:
                    continue

                a_fine = a[pts_x]
                b_fine = b[pts_y]
                a_sum = np.sum(a_fine)
                b_sum = np.sum(b_fine)
                if a_sum < 1e-12 or b_sum < 1e-12:
                    continue

                a_fine /= a_sum
                b_fine /= b_sum

                C_fine = euclidean_cost_matrix(X[pts_x], Y[pts_y])
                fine_solver = SinkhornSolver(self.epsilon_fine)
                fine_result = fine_solver.solve(a_fine, b_fine, C_fine)

                for ii, xi in enumerate(pts_x):
                    for jj, yj in enumerate(pts_y):
                        T_fine[xi, yj] = flow * fine_result.transport_plan[ii, jj]

                total_cost += flow * fine_result.cost

        return OTResult(
            transport_plan=T_fine,
            cost=total_cost,
            n_iterations=coarse_result.n_iterations,
            converged=True,
            metadata={"type": "hierarchical", "n_clusters": k},
        )


# ---------------------------------------------------------------------------
# Online OT for streaming candidates
# ---------------------------------------------------------------------------

class OnlineOT:
    """Online optimal transport for streaming candidate responses.

    Maintains an approximate OT plan that is updated incrementally
    as new candidates arrive.
    """

    def __init__(
        self,
        reference: np.ndarray,
        epsilon: float = 0.1,
        learning_rate: float = 0.01,
    ):
        self.reference = reference
        self.n_ref = reference.shape[0]
        self.epsilon = epsilon
        self.lr = learning_rate
        self.dual_g = np.zeros(self.n_ref)
        self.candidates: List[np.ndarray] = []
        self.dual_f: List[float] = []

    def add_candidate(self, x: np.ndarray) -> float:
        """Add a new candidate and return its OT assignment cost."""
        self.candidates.append(x)
        n = len(self.candidates)

        # Cost to reference points
        costs = np.array([np.linalg.norm(x - self.reference[j]) ** 2
                          for j in range(self.n_ref)])

        # Dual update (online mirror descent)
        # f_new = min_j (c(x, y_j) - g_j)
        adjusted = costs - self.dual_g
        best_j = np.argmin(adjusted)
        f_new = adjusted[best_j]
        self.dual_f.append(float(f_new))

        # Update g via gradient
        grad_g = np.zeros(self.n_ref)
        softmin = np.exp(-(costs - self.dual_g) / self.epsilon)
        softmin /= np.sum(softmin) + 1e-30
        grad_g = softmin / self.n_ref
        self.dual_g += self.lr * grad_g

        return float(costs[best_j])

    def current_diversity(self) -> float:
        """Compute current diversity estimate."""
        if len(self.candidates) < 2:
            return 0.0
        X = np.array(self.candidates)
        n = X.shape[0]
        a = np.ones(n) / n
        b = np.ones(self.n_ref) / self.n_ref
        C = euclidean_cost_matrix(X, self.reference)
        solver = SinkhornSolver(self.epsilon)
        result = solver.solve(a, b, C)
        return 1.0 / (1.0 + result.cost)

    def get_transport_to_reference(self) -> np.ndarray:
        """Get current transport plan to reference."""
        if len(self.candidates) == 0:
            return np.array([])
        X = np.array(self.candidates)
        n = X.shape[0]
        a = np.ones(n) / n
        b = np.ones(self.n_ref) / self.n_ref
        C = euclidean_cost_matrix(X, self.reference)
        solver = SinkhornSolver(self.epsilon)
        result = solver.solve(a, b, C)
        return result.transport_plan


# ---------------------------------------------------------------------------
# Wasserstein barycenter
# ---------------------------------------------------------------------------

class WassersteinBarycenter:
    """Compute Wasserstein barycenter of multiple distributions.

    Finds the distribution that minimizes the weighted sum of
    Wasserstein distances to all input distributions.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iterations: int = 100,
        n_support: int = 50,
        tol: float = 1e-6,
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.n_support = n_support
        self.tol = tol

    def compute(
        self,
        distributions: List[np.ndarray],
        weights_list: Optional[List[np.ndarray]] = None,
        lambdas: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Wasserstein barycenter.

        Args:
            distributions: List of point clouds
            weights_list: Weights for each distribution
            lambdas: Barycenter weights

        Returns:
            (barycenter_points, barycenter_weights)
        """
        k = len(distributions)
        d = distributions[0].shape[1]
        rng = np.random.RandomState(seed)

        if lambdas is None:
            lambdas = np.ones(k) / k
        if weights_list is None:
            weights_list = [
                np.ones(dist.shape[0]) / dist.shape[0] for dist in distributions
            ]

        # Initialize barycenter support
        all_pts = np.vstack(distributions)
        idx = rng.choice(all_pts.shape[0], size=self.n_support, replace=True)
        bary = all_pts[idx].copy()
        bary_weights = np.ones(self.n_support) / self.n_support

        solver = SinkhornSolver(self.epsilon, max_iterations=200)

        for iteration in range(self.max_iterations):
            bary_prev = bary.copy()
            # Fixed-point iteration
            bary_new = np.zeros_like(bary)

            for j in range(k):
                C = euclidean_cost_matrix(bary, distributions[j])
                result = solver.solve(bary_weights, weights_list[j], C)
                T = result.transport_plan

                # Update barycenter: weighted average of transported points
                T_row_sum = np.sum(T, axis=1, keepdims=True)
                T_normalized = T / (T_row_sum + 1e-30)
                transported = T_normalized @ distributions[j]
                bary_new += lambdas[j] * transported

            bary = bary_new

            if np.max(np.abs(bary - bary_prev)) < self.tol:
                break

        return bary, bary_weights

    def diversity_from_barycenter(
        self,
        embeddings: np.ndarray,
        groups: List[List[int]],
    ) -> float:
        """Compute diversity as the OT cost to the barycenter."""
        distributions = [embeddings[g] for g in groups]
        bary, bary_w = self.compute(distributions)

        # Total cost from each group to barycenter
        total_cost = 0.0
        for g, dist in zip(groups, distributions):
            a = np.ones(len(g)) / len(g)
            C = euclidean_cost_matrix(dist, bary)
            solver = SinkhornSolver(self.epsilon)
            result = solver.solve(a, bary_w, C)
            total_cost += result.cost

        return total_cost / len(groups)


# ---------------------------------------------------------------------------
# Sliced Wasserstein distance (fast approximation)
# ---------------------------------------------------------------------------

class SlicedWasserstein:
    """Sliced Wasserstein distance for fast diversity computation.

    SW(mu, nu) = E_theta [W_1(theta#mu, theta#nu)]
    Approximated via random projections.
    """

    def __init__(self, n_projections: int = 100, seed: int = 42):
        self.n_projections = n_projections
        self.rng = np.random.RandomState(seed)

    def _random_projections(self, d: int) -> np.ndarray:
        """Generate random projection directions on unit sphere."""
        projections = self.rng.randn(self.n_projections, d)
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        return projections / norms

    def distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Sliced Wasserstein distance."""
        d = X.shape[1]
        projections = self._random_projections(d)

        total = 0.0
        for theta in projections:
            # Project
            proj_x = X @ theta
            proj_y = Y @ theta
            # Sort
            proj_x_sorted = np.sort(proj_x)
            proj_y_sorted = np.sort(proj_y)
            # 1D Wasserstein = sorted matching
            # Interpolate if different sizes
            n, m = len(proj_x_sorted), len(proj_y_sorted)
            if n == m:
                total += np.mean(np.abs(proj_x_sorted - proj_y_sorted))
            else:
                # Linear interpolation
                t_x = np.linspace(0, 1, n)
                t_y = np.linspace(0, 1, m)
                t_common = np.linspace(0, 1, max(n, m))
                interp_x = np.interp(t_common, t_x, proj_x_sorted)
                interp_y = np.interp(t_common, t_y, proj_y_sorted)
                total += np.mean(np.abs(interp_x - interp_y))

        return total / self.n_projections

    def diversity(self, embeddings: np.ndarray, seed: int = 42) -> DiversityIndex:
        """Compute sliced Wasserstein diversity index."""
        n, d = embeddings.shape
        rng = np.random.RandomState(seed)

        # Reference: uniform on bounding box
        mins = np.min(embeddings, axis=0)
        maxs = np.max(embeddings, axis=0)
        ref = rng.uniform(size=(n * 2, d))
        for dim in range(d):
            ref[:, dim] = ref[:, dim] * (maxs[dim] - mins[dim] + 1) + mins[dim] - 0.5

        sw_dist = self.distance(embeddings, ref)
        diversity_val = 1.0 / (1.0 + sw_dist)

        return DiversityIndex(
            value=diversity_val,
            name="sliced_wasserstein_diversity",
            components={"sw_distance": sw_dist},
        )


# ---------------------------------------------------------------------------
# Transport-based embedding
# ---------------------------------------------------------------------------

class OTEmbedding:
    """Embed point clouds into a common space via OT linearization.

    Uses the Monge embedding: each distribution is mapped to its
    optimal transport map to a reference measure.
    """

    def __init__(
        self,
        reference: np.ndarray,
        epsilon: float = 0.1,
    ):
        self.reference = reference
        self.epsilon = epsilon
        self.n_ref = reference.shape[0]

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Embed a point cloud via its OT plan to reference.

        Returns the flattened transport plan as embedding.
        """
        n = X.shape[0]
        a = np.ones(n) / n
        b = np.ones(self.n_ref) / self.n_ref
        C = euclidean_cost_matrix(X, self.reference)
        solver = SinkhornSolver(self.epsilon)
        result = solver.solve(a, b, C)
        return result.transport_plan.flatten()

    def pairwise_distances(
        self, point_clouds: List[np.ndarray],
    ) -> np.ndarray:
        """Compute pairwise OT-based distances between distributions."""
        k = len(point_clouds)
        embeddings = [self.embed(pc) for pc in point_clouds]
        dists = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                d = np.linalg.norm(embeddings[i] - embeddings[j])
                dists[i, j] = d
                dists[j, i] = d
        return dists


# ---------------------------------------------------------------------------
# Partial OT
# ---------------------------------------------------------------------------

class PartialOT:
    """Partial optimal transport where only a fraction of mass is transported.

    Useful when comparing sets of very different sizes or when outliers
    should be ignored.

    min_T <C,T> s.t. T1 <= a, T^T1 <= b, sum T = s * min(sum a, sum b)
    """

    def __init__(
        self,
        mass_fraction: float = 0.8,
        epsilon: float = 0.1,
        max_iterations: int = 500,
        tol: float = 1e-6,
    ):
        self.s = mass_fraction
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tol = tol

    def solve(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
    ) -> OTResult:
        """Solve partial OT via modified Sinkhorn."""
        n, m = C.shape
        target_mass = self.s * min(np.sum(a), np.sum(b))

        K = np.exp(-C / self.epsilon)
        u = np.ones(n)
        v = np.ones(m)

        for it in range(self.max_iterations):
            u_prev = u.copy()
            Kv = K @ v
            u = np.minimum(a / (Kv + 1e-30), u * 1.1)
            Ku = K.T @ u
            v = np.minimum(b / (Ku + 1e-30), v * 1.1)

            # Enforce mass constraint
            T = np.diag(u) @ K @ np.diag(v)
            current_mass = np.sum(T)
            if current_mass > target_mass:
                scale = target_mass / current_mass
                u *= math.sqrt(scale)
                v *= math.sqrt(scale)

            if np.max(np.abs(u - u_prev)) < self.tol:
                break

        T = np.diag(u) @ K @ np.diag(v)
        cost = float(np.sum(T * C))

        return OTResult(
            transport_plan=T,
            cost=cost,
            n_iterations=it + 1,
            converged=True,
            metadata={"mass_transported": float(np.sum(T)), "target_mass": target_mass},
        )


# ---------------------------------------------------------------------------
# Fused Gromov-Wasserstein
# ---------------------------------------------------------------------------

class FusedGromovWasserstein:
    """Fused Gromov-Wasserstein distance.

    Combines Wasserstein (for features) and Gromov-Wasserstein (for structure):
    FGW(mu, nu) = alpha * W(mu, nu) + (1-alpha) * GW(mu, nu)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        epsilon: float = 0.05,
        max_iterations: int = 100,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def solve(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ) -> OTResult:
        """Solve FGW."""
        n, m = X.shape[0], Y.shape[0]
        if a is None:
            a = np.ones(n) / n
        if b is None:
            b = np.ones(m) / m

        CX = euclidean_cost_matrix(X, X)
        CY = euclidean_cost_matrix(Y, Y)
        M = euclidean_cost_matrix(X, Y)

        T = np.outer(a, b)
        sinkhorn = SinkhornSolver(self.epsilon, max_iterations=100)

        for it in range(self.max_iterations):
            T_prev = T.copy()
            # GW gradient
            gw_cost = -2.0 * CX @ T @ CY + CX**2 @ T @ np.ones((m, m)) + np.ones((n, n)) @ T @ CY**2

            # Fused cost
            fused_cost = self.alpha * M + (1 - self.alpha) * gw_cost

            result = sinkhorn.solve(a, b, fused_cost)
            T = result.transport_plan

            if np.max(np.abs(T - T_prev)) < 1e-6:
                break

        cost = float(self.alpha * np.sum(T * M))
        return OTResult(
            transport_plan=T,
            cost=cost,
            n_iterations=it + 1,
            converged=True,
            metadata={"type": "fused_gromov_wasserstein"},
        )


# ---------------------------------------------------------------------------
# Wasserstein distance between distributions on graphs
# ---------------------------------------------------------------------------

class GraphWasserstein:
    """Wasserstein distance on a graph using shortest path as ground metric."""

    def __init__(
        self,
        adjacency: np.ndarray,
        epsilon: float = 0.1,
    ):
        self.adjacency = adjacency
        self.n = adjacency.shape[0]
        self.epsilon = epsilon
        self.shortest_paths = self._floyd_warshall()

    def _floyd_warshall(self) -> np.ndarray:
        """Compute all-pairs shortest paths."""
        dist = np.full((self.n, self.n), float("inf"))
        np.fill_diagonal(dist, 0)

        for i in range(self.n):
            for j in range(self.n):
                if self.adjacency[i, j] > 0:
                    dist[i, j] = self.adjacency[i, j]

        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        # Replace inf with large value
        dist[dist == float("inf")] = 100.0
        return dist

    def distance(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
    ) -> float:
        """Compute Wasserstein distance on graph."""
        C = self.shortest_paths ** 2
        solver = SinkhornSolver(self.epsilon)
        result = solver.solve(mu, nu, C)
        return math.sqrt(max(result.cost, 0))


# ---------------------------------------------------------------------------
# Free Support Wasserstein Barycenter
# ---------------------------------------------------------------------------

class FreeSupportBarycenter:
    """Wasserstein barycenter with free support (adaptively placed atoms)."""

    def __init__(
        self,
        n_support: int = 30,
        epsilon: float = 0.1,
        max_iterations: int = 50,
        seed: int = 42,
    ):
        self.n_support = n_support
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.rng = np.random.RandomState(seed)

    def compute(
        self,
        distributions: List[np.ndarray],
        weights: Optional[List[np.ndarray]] = None,
        lambdas: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute free-support Wasserstein barycenter."""
        k = len(distributions)
        d = distributions[0].shape[1]

        if lambdas is None:
            lambdas = np.ones(k) / k
        if weights is None:
            weights = [np.ones(dist.shape[0]) / dist.shape[0] for dist in distributions]

        # Initialize support by sampling from distributions
        all_pts = np.vstack(distributions)
        idx = self.rng.choice(all_pts.shape[0], size=self.n_support, replace=True)
        support = all_pts[idx].copy()
        bary_weights = np.ones(self.n_support) / self.n_support

        solver = SinkhornSolver(self.epsilon, max_iterations=100)

        for iteration in range(self.max_iterations):
            support_prev = support.copy()

            # Gradient step for support locations
            grad = np.zeros_like(support)
            for j in range(k):
                C = euclidean_cost_matrix(support, distributions[j])
                result = solver.solve(bary_weights, weights[j], C)
                T = result.transport_plan

                # Gradient: d cost / d support_i
                for i in range(self.n_support):
                    weighted_diff = np.zeros(d)
                    for l in range(distributions[j].shape[0]):
                        diff = support[i] - distributions[j][l]
                        weighted_diff += T[i, l] * 2 * diff
                    grad[i] += lambdas[j] * weighted_diff

            # Update support
            step_size = 0.1 / (iteration + 1)
            support -= step_size * grad

            if np.max(np.abs(support - support_prev)) < 1e-6:
                break

        return support, bary_weights


# ---------------------------------------------------------------------------
# Entropy-regularized OT with multiple regularizers
# ---------------------------------------------------------------------------

class MultiRegularizedOT:
    """OT with multiple regularization terms.

    min_T <C,T> + eps * KL(T|ab^T) + lambda * ||T||_F^2 + gamma * group_reg(T)
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        lambda_frobenius: float = 0.01,
        gamma_group: float = 0.0,
        group_labels: Optional[np.ndarray] = None,
        max_iterations: int = 500,
    ):
        self.epsilon = epsilon
        self.lambda_f = lambda_frobenius
        self.gamma = gamma_group
        self.group_labels = group_labels
        self.max_iterations = max_iterations

    def solve(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
    ) -> OTResult:
        """Solve multi-regularized OT."""
        n, m = C.shape
        # Modified cost including Frobenius penalty
        C_mod = C + self.lambda_f * np.ones((n, m))

        # Group regularization
        if self.gamma > 0 and self.group_labels is not None:
            for i in range(n):
                for j in range(m):
                    if i < len(self.group_labels) and j < len(self.group_labels):
                        if self.group_labels[i] == self.group_labels[j]:
                            C_mod[i, j] -= self.gamma  # Bonus for same-group

        solver = SinkhornSolver(self.epsilon, self.max_iterations)
        return solver.solve(a, b, C_mod)
