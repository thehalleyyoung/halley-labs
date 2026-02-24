"""Submodular optimization for diversity maximization.

Implements submodular maximization algorithms (lazy greedy, continuous greedy,
DPP sampling), submodularity verification, and constrained optimization
under matroid, knapsack, and graph constraints.

Mathematical foundations:
- Submodularity: f(A+e) - f(A) >= f(B+e) - f(B) for A subset B
- Lazy greedy: (1-1/e) approximation for monotone submodular under cardinality
- Continuous greedy: (1-1/e) for monotone under matroid constraint
- DPP: P(S) proportional to det(L_S)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .kernels import Kernel, RBFKernel
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubmodularResult:
    """Result from submodular optimization."""
    selected: List[int]
    value: float
    marginal_gains: List[float]
    n_evaluations: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class SubmodularityReport:
    """Report on submodularity properties of a function."""
    is_submodular: bool
    submodularity_ratio: float
    curvature: float
    n_tests: int
    violations: int
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Submodular functions for diversity
# ---------------------------------------------------------------------------

class SubmodularFunction(ABC):
    """Base class for submodular set functions."""

    @abstractmethod
    def evaluate(self, S: List[int]) -> float:
        """Evaluate f(S)."""
        ...

    def marginal_gain(self, S: List[int], e: int) -> float:
        """Compute marginal gain f(S + e) - f(S)."""
        if e in S:
            return 0.0
        return self.evaluate(S + [e]) - self.evaluate(S)


class LogDetDiversity(SubmodularFunction):
    """Log-determinant diversity: f(S) = log det(I + alpha * K_S)."""

    def __init__(
        self,
        embeddings: np.ndarray,
        alpha: float = 1.0,
        kernel: Optional[Kernel] = None,
    ):
        self.embeddings = embeddings
        self.alpha = alpha
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n = embeddings.shape[0]

    def evaluate(self, S: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        K_S = self.kernel.gram_matrix(self.embeddings[S])
        return log_det_safe(np.eye(len(S)) + self.alpha * K_S)


class FacilityLocation(SubmodularFunction):
    """Facility location: f(S) = sum_v max_{s in S} sim(v, s)."""

    def __init__(
        self,
        embeddings: np.ndarray,
        client_embeddings: Optional[np.ndarray] = None,
        kernel: Optional[Kernel] = None,
    ):
        self.embeddings = embeddings
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n = embeddings.shape[0]
        if client_embeddings is not None:
            self.clients = client_embeddings
        else:
            self.clients = embeddings  # Self-referencing

    def evaluate(self, S: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        total = 0.0
        for client in self.clients:
            max_sim = max(
                self.kernel.evaluate(client, self.embeddings[s]) for s in S
            )
            total += max_sim
        return total


class GraphCut(SubmodularFunction):
    """Graph cut function: f(S) = sum_{i in S, j not in S} w_{ij}."""

    def __init__(
        self,
        embeddings: np.ndarray,
        kernel: Optional[Kernel] = None,
    ):
        self.embeddings = embeddings
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n = embeddings.shape[0]
        # Precompute similarity matrix
        self.W = self.kernel.gram_matrix(embeddings)

    def evaluate(self, S: List[int]) -> float:
        if len(S) == 0 or len(S) == self.n:
            return 0.0
        complement = [i for i in range(self.n) if i not in S]
        return float(np.sum(self.W[np.ix_(S, complement)]))


class SaturatedCoverage(SubmodularFunction):
    """Saturated coverage: f(S) = sum_j min(sum_{i in S} w_{ij}, alpha_j)."""

    def __init__(
        self,
        embeddings: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        kernel: Optional[Kernel] = None,
    ):
        self.embeddings = embeddings
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n = embeddings.shape[0]
        self.W = self.kernel.gram_matrix(embeddings)
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = np.ones(self.n) * 2.0

    def evaluate(self, S: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        total = 0.0
        for j in range(self.n):
            coverage = sum(self.W[i, j] for i in S)
            total += min(coverage, self.thresholds[j])
        return total


class FeatureBasedDiversity(SubmodularFunction):
    """Feature-based diversity: f(S) = |union of features covered by S|.

    Each element covers a set of features (binary). Monotone submodular.
    """

    def __init__(
        self,
        feature_matrix: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ):
        # feature_matrix: (n, m) binary matrix
        self.features = feature_matrix
        self.n, self.m = feature_matrix.shape
        self.weights = weights if weights is not None else np.ones(self.m)

    def evaluate(self, S: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        covered = np.max(self.features[S], axis=0)
        return float(np.sum(covered * self.weights))


# ---------------------------------------------------------------------------
# Lazy Greedy (accelerated greedy)
# ---------------------------------------------------------------------------

class LazyGreedy:
    """Lazy greedy algorithm for monotone submodular maximization.

    Uses lazy evaluations to avoid recomputing marginal gains that
    haven't changed since the last iteration.

    Guarantee: (1 - 1/e) approximation under cardinality constraint.
    """

    def __init__(self, func: SubmodularFunction, n: int):
        self.func = func
        self.n = n

    def maximize(self, k: int) -> SubmodularResult:
        """Select k items to maximize f."""
        selected: List[int] = []
        marginals: List[float] = []
        n_evals = 0

        # Initialize upper bounds on marginal gains
        upper_bounds = [(float("inf"), i) for i in range(self.n)]
        # Use a priority-queue-like approach with sorted list
        import heapq
        heap = [(-ub, idx) for ub, idx in upper_bounds]
        heapq.heapify(heap)

        for _ in range(k):
            while True:
                neg_ub, idx = heapq.heappop(heap)
                if idx in selected:
                    continue
                # Compute actual marginal gain
                gain = self.func.marginal_gain(selected, idx)
                n_evals += 1

                # Check if it's still the best
                if len(heap) == 0 or gain >= -heap[0][0]:
                    selected.append(idx)
                    marginals.append(gain)
                    break
                else:
                    # Push back with updated bound
                    heapq.heappush(heap, (-gain, idx))

        value = self.func.evaluate(selected)
        return SubmodularResult(
            selected=selected,
            value=value,
            marginal_gains=marginals,
            n_evaluations=n_evals,
            metadata={"algorithm": "lazy_greedy"},
        )


class StandardGreedy:
    """Standard greedy algorithm (non-lazy) for comparison."""

    def __init__(self, func: SubmodularFunction, n: int):
        self.func = func
        self.n = n

    def maximize(self, k: int) -> SubmodularResult:
        """Select k items greedily."""
        selected: List[int] = []
        marginals: List[float] = []
        n_evals = 0

        for _ in range(k):
            best_gain = -float("inf")
            best_idx = -1

            for i in range(self.n):
                if i in selected:
                    continue
                gain = self.func.marginal_gain(selected, i)
                n_evals += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx < 0:
                break
            selected.append(best_idx)
            marginals.append(best_gain)

        value = self.func.evaluate(selected)
        return SubmodularResult(
            selected=selected,
            value=value,
            marginal_gains=marginals,
            n_evaluations=n_evals,
            metadata={"algorithm": "standard_greedy"},
        )


# ---------------------------------------------------------------------------
# Continuous Greedy + Pipage Rounding
# ---------------------------------------------------------------------------

class ContinuousGreedy:
    """Continuous greedy algorithm for submodular maximization.

    Works with the multilinear extension F(x) of f.
    Provides (1-1/e) approximation under matroid constraints.
    """

    def __init__(
        self,
        func: SubmodularFunction,
        n: int,
        n_samples: int = 50,
        step_size: float = 0.01,
        seed: int = 42,
    ):
        self.func = func
        self.n = n
        self.n_samples = n_samples
        self.step_size = step_size
        self.rng = np.random.RandomState(seed)

    def _multilinear_extension(self, x: np.ndarray) -> float:
        """Estimate multilinear extension F(x) via sampling.

        F(x) = E[f(S)] where S includes each element i independently with prob x_i.
        """
        total = 0.0
        for _ in range(self.n_samples):
            sample = (self.rng.random(self.n) < x).astype(bool)
            S = [i for i in range(self.n) if sample[i]]
            total += self.func.evaluate(S)
        return total / self.n_samples

    def _gradient_estimate(self, x: np.ndarray) -> np.ndarray:
        """Estimate gradient of multilinear extension."""
        grad = np.zeros(self.n)
        for i in range(self.n):
            # partial F / partial x_i = E[f(S+i) - f(S\i)] where S~x
            total = 0.0
            for _ in range(self.n_samples):
                sample = (self.rng.random(self.n) < x).astype(bool)
                S_without = [j for j in range(self.n) if sample[j] and j != i]
                S_with = S_without + [i]
                total += self.func.evaluate(S_with) - self.func.evaluate(S_without)
            grad[i] = total / self.n_samples
        return grad

    def maximize(
        self,
        k: int,
        n_steps: int = 100,
    ) -> SubmodularResult:
        """Run continuous greedy and round."""
        x = np.zeros(self.n)
        n_evals = 0

        for step in range(n_steps):
            grad = self._gradient_estimate(x)
            n_evals += self.n_samples * self.n * 2

            # Move in direction of gradient, maintaining x in [0,1]^n
            # and sum(x) <= k
            direction = np.zeros(self.n)
            # Greedy: set top-k coordinates of gradient to 1
            top_k = np.argsort(grad)[-k:]
            direction[top_k] = 1.0

            x = x + self.step_size * direction
            x = np.clip(x, 0, 1)
            # Project to sum <= k
            if np.sum(x) > k:
                x *= k / np.sum(x)

        # Pipage rounding
        selected = self._pipage_round(x, k)

        value = self.func.evaluate(selected)
        return SubmodularResult(
            selected=selected,
            value=value,
            marginal_gains=[],
            n_evaluations=n_evals,
            metadata={"algorithm": "continuous_greedy_pipage"},
        )

    def _pipage_round(self, x: np.ndarray, k: int) -> List[int]:
        """Pipage rounding: convert fractional solution to integral."""
        x = x.copy()
        max_iter = self.n * 10

        for _ in range(max_iter):
            # Find two fractional coordinates
            fractional = [i for i in range(self.n) if 1e-8 < x[i] < 1 - 1e-8]
            if len(fractional) < 2:
                break
            i, j = fractional[0], fractional[1]

            # Transfer mass to make one integral
            eps1 = min(x[i], 1 - x[j])
            eps2 = min(x[j], 1 - x[i])

            # Evaluate both directions
            x_plus = x.copy()
            x_plus[i] -= eps1
            x_plus[j] += eps1
            x_minus = x.copy()
            x_minus[i] += eps2
            x_minus[j] -= eps2

            f_plus = self._multilinear_extension(x_plus)
            f_minus = self._multilinear_extension(x_minus)

            x = x_plus if f_plus >= f_minus else x_minus

        # Final: select top-k by x value
        selected = np.argsort(x)[-k:]
        return sorted(selected.tolist())


# ---------------------------------------------------------------------------
# Determinantal Point Processes (DPP)
# ---------------------------------------------------------------------------

class DPPSampler:
    """Determinantal Point Process for diverse subset sampling.

    P(S) proportional to det(L_S) where L is the DPP kernel.
    DPPs naturally encode repulsiveness (diversity).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        quality_scores: Optional[np.ndarray] = None,
        kernel: Optional[Kernel] = None,
    ):
        self.embeddings = embeddings
        self.n = embeddings.shape[0]
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        if quality_scores is not None:
            self.quality = quality_scores
        else:
            self.quality = np.ones(self.n)
        self.L = self._build_L_kernel()

    def _build_L_kernel(self) -> np.ndarray:
        """Build L-ensemble kernel: L = B^T B where B_i = q_i * phi(x_i)."""
        K = self.kernel.gram_matrix(self.embeddings)
        q = self.quality
        L = np.outer(q, q) * K
        return L

    def _eigendecompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecompose L kernel."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.L)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvalues, eigenvectors

    def sample(self, seed: int = 42) -> List[int]:
        """Sample from DPP using spectral algorithm."""
        rng = np.random.RandomState(seed)
        eigenvalues, eigenvectors = self._eigendecompose()

        # Phase 1: Select eigenvectors
        selected_eigenvectors = []
        for i in range(len(eigenvalues)):
            prob = eigenvalues[i] / (eigenvalues[i] + 1.0)
            if rng.random() < prob:
                selected_eigenvectors.append(i)

        if len(selected_eigenvectors) == 0:
            return [rng.randint(self.n)]

        # Phase 2: Sample items from selected eigenspace
        V = eigenvectors[:, selected_eigenvectors].copy()
        k = len(selected_eigenvectors)
        selected: List[int] = []

        for _ in range(k):
            if V.shape[1] == 0:
                break
            # Compute probabilities
            probs = np.sum(V ** 2, axis=1)
            probs = np.maximum(probs, 0)
            p_sum = np.sum(probs)
            if p_sum < 1e-12:
                break
            probs /= p_sum

            # Sample item
            idx = int(rng.choice(self.n, p=probs))
            selected.append(idx)

            # Update V: project out selected item's component
            v_idx = V[idx, :]
            norm = np.linalg.norm(v_idx)
            if norm < 1e-12:
                break
            v_idx = v_idx / norm
            V = V - np.outer(V @ v_idx, v_idx)

            # Re-orthogonalize (numerical stability)
            if V.shape[1] > 1:
                try:
                    Q, R = np.linalg.qr(V)
                    # Keep non-zero columns
                    mask = np.abs(np.diag(R)) > 1e-10
                    V = Q[:, mask]
                except np.linalg.LinAlgError:
                    break

        return selected

    def sample_k(self, k: int, seed: int = 42) -> List[int]:
        """Sample exactly k items from k-DPP."""
        rng = np.random.RandomState(seed)
        eigenvalues, eigenvectors = self._eigendecompose()
        n = len(eigenvalues)

        # Compute elementary symmetric polynomials for k-DPP
        # e_k(lambda_1, ..., lambda_n) where lambda_i = eigenvalue_i
        E = self._elementary_symmetric_polynomials(eigenvalues, k)

        # Phase 1: Select exactly k eigenvectors
        selected_eigenvectors: List[int] = []
        remaining = k
        for i in range(n - 1, -1, -1):
            if remaining == 0:
                break
            if remaining > i + 1:
                continue
            # Probability of including eigenvector i
            if remaining <= 0:
                break
            prob = eigenvalues[i] * E[i][remaining - 1] / (E[i + 1][remaining] + 1e-30)
            prob = min(max(prob, 0), 1)
            if rng.random() < prob:
                selected_eigenvectors.append(i)
                remaining -= 1

        if len(selected_eigenvectors) < k:
            # Fallback: select highest eigenvalue eigenvectors
            top_k = np.argsort(eigenvalues)[-k:]
            selected_eigenvectors = top_k.tolist()

        # Phase 2: Sample items (same as standard DPP)
        V = eigenvectors[:, selected_eigenvectors].copy()
        selected: List[int] = []
        for _ in range(k):
            if V.shape[1] == 0:
                break
            probs = np.sum(V ** 2, axis=1)
            probs = np.maximum(probs, 0)
            p_sum = np.sum(probs)
            if p_sum < 1e-12:
                remaining_items = [i for i in range(self.n) if i not in selected]
                if remaining_items:
                    selected.append(rng.choice(remaining_items))
                break
            probs /= p_sum

            idx = int(rng.choice(self.n, p=probs))
            selected.append(idx)

            v_idx = V[idx, :]
            norm = np.linalg.norm(v_idx)
            if norm < 1e-12:
                break
            v_idx = v_idx / norm
            V = V - np.outer(V @ v_idx, v_idx)

        return selected[:k]

    @staticmethod
    def _elementary_symmetric_polynomials(
        lambdas: np.ndarray, k: int,
    ) -> List[List[float]]:
        """Compute elementary symmetric polynomials via DP.

        E[i][j] = e_j(lambda_1, ..., lambda_i)
        """
        n = len(lambdas)
        E: List[List[float]] = [[0.0] * (k + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            E[i][0] = 1.0
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                E[i][j] = E[i - 1][j] + lambdas[i - 1] * E[i - 1][j - 1]
        return E

    def log_probability(self, S: List[int]) -> float:
        """Compute log P(S) for a subset."""
        if len(S) == 0:
            return float(-np.log(np.linalg.det(self.L + np.eye(self.n))))
        L_S = self.L[np.ix_(S, S)]
        log_det_S = log_det_safe(L_S)
        log_det_LI = log_det_safe(self.L + np.eye(self.n))
        return log_det_S - log_det_LI

    def map_inference(self, k: int) -> SubmodularResult:
        """MAP inference: find most probable subset of size k.

        This is equivalent to maximizing log det(L_S) subject to |S| = k.
        """
        func = LogDetDPP(self.L)
        greedy = LazyGreedy(func, self.n)
        return greedy.maximize(k)


class LogDetDPP(SubmodularFunction):
    """Wrapper: log det(L_S) as submodular function."""

    def __init__(self, L: np.ndarray):
        self.L = L
        self.n = L.shape[0]

    def evaluate(self, S: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        L_S = self.L[np.ix_(S, S)]
        return log_det_safe(L_S)


# ---------------------------------------------------------------------------
# Submodularity Verification
# ---------------------------------------------------------------------------

class SubmodularityVerifier:
    """Verify submodularity properties of a set function."""

    def __init__(self, func: SubmodularFunction, n: int, seed: int = 42):
        self.func = func
        self.n = n
        self.rng = np.random.RandomState(seed)

    def verify_submodularity(
        self,
        n_tests: int = 1000,
    ) -> SubmodularityReport:
        """Test submodularity: f(A+e) - f(A) >= f(B+e) - f(B) for A ⊆ B."""
        violations = 0
        min_ratio = float("inf")

        for _ in range(n_tests):
            # Generate random A subset B
            B_size = self.rng.randint(1, min(self.n, 8))
            B = sorted(self.rng.choice(self.n, size=B_size, replace=False).tolist())
            A_size = self.rng.randint(0, len(B))
            A = sorted(self.rng.choice(B, size=A_size, replace=False).tolist()) if A_size > 0 else []

            # Pick element not in B
            complement = [i for i in range(self.n) if i not in B]
            if len(complement) == 0:
                continue
            e = int(self.rng.choice(complement))

            gain_A = self.func.marginal_gain(A, e)
            gain_B = self.func.marginal_gain(B, e)

            if gain_B > 1e-12:
                ratio = gain_A / gain_B
                min_ratio = min(min_ratio, ratio)

            if gain_A < gain_B - 1e-8:
                violations += 1

        sub_ratio = min_ratio if min_ratio < float("inf") else 1.0
        return SubmodularityReport(
            is_submodular=violations == 0,
            submodularity_ratio=sub_ratio,
            curvature=0.0,  # Computed separately
            n_tests=n_tests,
            violations=violations,
        )

    def estimate_curvature(
        self,
        n_tests: int = 500,
    ) -> float:
        """Estimate curvature kappa of submodular function.

        kappa = 1 - min_{e, S} f(S+e) - f(S) / f({e})
        """
        min_ratio = 1.0

        for _ in range(n_tests):
            e = self.rng.randint(self.n)
            f_e = self.func.evaluate([e])
            if f_e < 1e-12:
                continue

            S_size = self.rng.randint(1, min(self.n, 8))
            S = sorted(self.rng.choice(
                [i for i in range(self.n) if i != e],
                size=min(S_size, self.n - 1),
                replace=False,
            ).tolist())

            gain = self.func.marginal_gain(S, e)
            ratio = gain / f_e
            min_ratio = min(min_ratio, ratio)

        curvature = 1.0 - max(min_ratio, 0.0)
        return curvature

    def estimate_submodularity_ratio(
        self,
        n_tests: int = 500,
    ) -> float:
        """Estimate submodularity ratio gamma.

        gamma = min_{A subset B, e not in B} f(A+e)-f(A) / f(B+e)-f(B)
        For submodular functions, gamma >= 1. Below 1 = approximately submodular.
        """
        min_ratio = float("inf")

        for _ in range(n_tests):
            B_size = self.rng.randint(1, min(self.n, 6))
            B = sorted(self.rng.choice(self.n, size=B_size, replace=False).tolist())
            A_size = self.rng.randint(0, len(B))
            A = sorted(self.rng.choice(B, size=A_size, replace=False).tolist()) if A_size > 0 else []

            complement = [i for i in range(self.n) if i not in B]
            if len(complement) == 0:
                continue
            e = int(self.rng.choice(complement))

            gain_A = self.func.marginal_gain(A, e)
            gain_B = self.func.marginal_gain(B, e)

            if abs(gain_B) > 1e-12:
                ratio = gain_A / gain_B
                min_ratio = min(min_ratio, ratio)

        return min_ratio if min_ratio < float("inf") else 1.0

    def full_report(
        self,
        n_tests: int = 500,
    ) -> SubmodularityReport:
        """Generate comprehensive submodularity report."""
        report = self.verify_submodularity(n_tests)
        report.curvature = self.estimate_curvature(n_tests)
        report.submodularity_ratio = self.estimate_submodularity_ratio(n_tests)
        report.metadata = {
            "monotone": self._check_monotonicity(n_tests),
            "normalized": abs(self.func.evaluate([])) < 1e-10,
        }
        return report

    def _check_monotonicity(self, n_tests: int) -> bool:
        """Check if function is monotone: f(A) <= f(B) for A subset B."""
        for _ in range(n_tests):
            B_size = self.rng.randint(1, min(self.n, 8))
            B = sorted(self.rng.choice(self.n, size=B_size, replace=False).tolist())
            A_size = self.rng.randint(0, len(B))
            A = sorted(self.rng.choice(B, size=A_size, replace=False).tolist()) if A_size > 0 else []
            if self.func.evaluate(A) > self.func.evaluate(B) + 1e-8:
                return False
        return True


# ---------------------------------------------------------------------------
# Matroid Constraints
# ---------------------------------------------------------------------------

class Matroid(ABC):
    """Base matroid class."""

    @abstractmethod
    def is_independent(self, S: List[int]) -> bool:
        """Check if set S is independent."""
        ...

    @abstractmethod
    def rank(self, S: List[int]) -> int:
        """Rank of set S."""
        ...


class UniformMatroid(Matroid):
    """Uniform matroid: sets of size <= k are independent."""

    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k

    def is_independent(self, S: List[int]) -> bool:
        return len(S) <= self.k

    def rank(self, S: List[int]) -> int:
        return min(len(S), self.k)


class PartitionMatroid(Matroid):
    """Partition matroid: elements divided into groups, max b_i from group i."""

    def __init__(self, groups: Dict[int, List[int]], bounds: Dict[int, int]):
        self.groups = groups  # group_id -> list of elements
        self.bounds = bounds  # group_id -> max allowed from group
        self.element_to_group: Dict[int, int] = {}
        for gid, elements in groups.items():
            for e in elements:
                self.element_to_group[e] = gid

    def is_independent(self, S: List[int]) -> bool:
        group_counts: Dict[int, int] = {}
        for e in S:
            gid = self.element_to_group.get(e, -1)
            group_counts[gid] = group_counts.get(gid, 0) + 1
            if group_counts[gid] > self.bounds.get(gid, 0):
                return False
        return True

    def rank(self, S: List[int]) -> int:
        group_counts: Dict[int, int] = {}
        count = 0
        for e in sorted(S):
            gid = self.element_to_group.get(e, -1)
            gc = group_counts.get(gid, 0)
            if gc < self.bounds.get(gid, 0):
                group_counts[gid] = gc + 1
                count += 1
        return count


class GraphicMatroid(Matroid):
    """Graphic matroid: independent sets are forests (acyclic edge sets)."""

    def __init__(self, n_vertices: int, edges: List[Tuple[int, int]]):
        self.n_vertices = n_vertices
        self.edges = edges

    def is_independent(self, S: List[int]) -> bool:
        """Check if edge subset forms a forest (no cycles)."""
        # Use union-find
        parent = list(range(self.n_vertices))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        for idx in S:
            if idx >= len(self.edges):
                continue
            u, v = self.edges[idx]
            if not union(u, v):
                return False
        return True

    def rank(self, S: List[int]) -> int:
        parent = list(range(self.n_vertices))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        count = 0
        for idx in S:
            if idx >= len(self.edges):
                continue
            u, v = self.edges[idx]
            if union(u, v):
                count += 1
        return count


# ---------------------------------------------------------------------------
# Constrained Submodular Maximization
# ---------------------------------------------------------------------------

class MatroidConstrainedGreedy:
    """Greedy submodular maximization under matroid constraint.

    Provides 1/2 approximation for monotone submodular under general matroid.
    """

    def __init__(
        self,
        func: SubmodularFunction,
        matroid: Matroid,
        n: int,
    ):
        self.func = func
        self.matroid = matroid
        self.n = n

    def maximize(self) -> SubmodularResult:
        """Greedy maximization respecting matroid constraint."""
        selected: List[int] = []
        marginals: List[float] = []
        n_evals = 0

        while True:
            best_gain = -float("inf")
            best_idx = -1

            for i in range(self.n):
                if i in selected:
                    continue
                candidate = selected + [i]
                if not self.matroid.is_independent(candidate):
                    continue
                gain = self.func.marginal_gain(selected, i)
                n_evals += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx < 0 or best_gain <= 0:
                break
            selected.append(best_idx)
            marginals.append(best_gain)

        value = self.func.evaluate(selected)
        return SubmodularResult(
            selected=selected,
            value=value,
            marginal_gains=marginals,
            n_evaluations=n_evals,
            metadata={"algorithm": "matroid_greedy"},
        )


class KnapsackConstrainedGreedy:
    """Greedy submodular maximization under knapsack constraint.

    Budget B, costs c_i. Select S with sum c_i <= B to maximize f(S).
    """

    def __init__(
        self,
        func: SubmodularFunction,
        n: int,
        costs: np.ndarray,
        budget: float,
    ):
        self.func = func
        self.n = n
        self.costs = costs
        self.budget = budget

    def maximize(self) -> SubmodularResult:
        """Cost-effective greedy: select by gain/cost ratio."""
        selected: List[int] = []
        marginals: List[float] = []
        remaining_budget = self.budget
        n_evals = 0

        while remaining_budget > 0:
            best_ratio = -float("inf")
            best_idx = -1

            for i in range(self.n):
                if i in selected:
                    continue
                if self.costs[i] > remaining_budget:
                    continue
                gain = self.func.marginal_gain(selected, i)
                n_evals += 1
                ratio = gain / max(self.costs[i], 1e-12)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i

            if best_idx < 0:
                break
            selected.append(best_idx)
            marginals.append(self.func.marginal_gain(selected[:-1], best_idx))
            remaining_budget -= self.costs[best_idx]

        # Also try single best element (for approximation guarantee)
        best_single = -1
        best_single_val = -float("inf")
        for i in range(self.n):
            if self.costs[i] <= self.budget:
                val = self.func.evaluate([i])
                if val > best_single_val:
                    best_single_val = val
                    best_single = i

        greedy_val = self.func.evaluate(selected)
        if best_single >= 0 and best_single_val > greedy_val:
            return SubmodularResult(
                selected=[best_single],
                value=best_single_val,
                marginal_gains=[best_single_val],
                n_evaluations=n_evals,
                metadata={"algorithm": "knapsack_greedy_single"},
            )

        return SubmodularResult(
            selected=selected,
            value=greedy_val,
            marginal_gains=marginals,
            n_evaluations=n_evals,
            metadata={"algorithm": "knapsack_greedy"},
        )


class GraphConstrainedGreedy:
    """Greedy submodular maximization under graph constraints.

    Selected items must form an independent set in a conflict graph
    (no two adjacent items can be selected simultaneously).
    """

    def __init__(
        self,
        func: SubmodularFunction,
        n: int,
        adjacency: np.ndarray,
    ):
        self.func = func
        self.n = n
        self.adjacency = adjacency  # (n, n) binary adjacency matrix

    def _is_feasible(self, selected: List[int], candidate: int) -> bool:
        """Check if adding candidate maintains independent set."""
        for s in selected:
            if self.adjacency[s, candidate] > 0:
                return False
        return True

    def maximize(self, k: int) -> SubmodularResult:
        """Greedy maximization under graph independence constraint."""
        selected: List[int] = []
        marginals: List[float] = []
        n_evals = 0

        for _ in range(k):
            best_gain = -float("inf")
            best_idx = -1

            for i in range(self.n):
                if i in selected:
                    continue
                if not self._is_feasible(selected, i):
                    continue
                gain = self.func.marginal_gain(selected, i)
                n_evals += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx < 0:
                break
            selected.append(best_idx)
            marginals.append(best_gain)

        value = self.func.evaluate(selected)
        return SubmodularResult(
            selected=selected,
            value=value,
            marginal_gains=marginals,
            n_evaluations=n_evals,
            metadata={"algorithm": "graph_constrained_greedy"},
        )


# ---------------------------------------------------------------------------
# Comparison of submodular algorithms
# ---------------------------------------------------------------------------

class SubmodularComparison:
    """Compare different submodular optimization algorithms."""

    def __init__(
        self,
        embeddings: np.ndarray,
        kernel: Optional[Kernel] = None,
    ):
        self.embeddings = embeddings
        self.n = embeddings.shape[0]
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def compare(
        self,
        k: int,
        functions: Optional[List[Tuple[str, SubmodularFunction]]] = None,
    ) -> Dict[str, Dict[str, SubmodularResult]]:
        """Compare algorithms across different submodular functions."""
        if functions is None:
            functions = [
                ("logdet", LogDetDiversity(self.embeddings, kernel=self.kernel)),
                ("facility", FacilityLocation(self.embeddings, kernel=self.kernel)),
                ("graphcut", GraphCut(self.embeddings, kernel=self.kernel)),
                ("coverage", SaturatedCoverage(self.embeddings, kernel=self.kernel)),
            ]

        results: Dict[str, Dict[str, SubmodularResult]] = {}
        for func_name, func in functions:
            results[func_name] = {}

            # Standard greedy
            sg = StandardGreedy(func, self.n)
            results[func_name]["standard_greedy"] = sg.maximize(k)

            # Lazy greedy
            lg = LazyGreedy(func, self.n)
            results[func_name]["lazy_greedy"] = lg.maximize(k)

            # DPP (only for logdet-like)
            if func_name == "logdet":
                dpp = DPPSampler(self.embeddings, kernel=self.kernel)
                dpp_result = dpp.map_inference(k)
                results[func_name]["dpp_map"] = dpp_result

        return results


# ---------------------------------------------------------------------------
# Stochastic Greedy
# ---------------------------------------------------------------------------

class StochasticGreedy:
    """Stochastic greedy for submodular maximization.

    Instead of evaluating all remaining elements, samples a random subset
    of size n/k * log(1/epsilon) and picks the best from that sample.
    Runtime: O(n log(1/epsilon)) vs O(nk) for standard greedy.
    Provides (1-1/e-epsilon) approximation.
    """

    def __init__(
        self,
        func: SubmodularFunction,
        n: int,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        self.func = func
        self.n = n
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)

    def maximize(self, k: int) -> SubmodularResult:
        selected: List[int] = []
        marginals: List[float] = []
        n_evals = 0

        sample_size = max(1, int(self.n / k * math.log(1 / self.epsilon)))

        for _ in range(k):
            remaining = [i for i in range(self.n) if i not in selected]
            if not remaining:
                break
            sample_size_actual = min(sample_size, len(remaining))
            sample = self.rng.choice(remaining, size=sample_size_actual, replace=False).tolist()

            best_gain = -float("inf")
            best_idx = -1
            for idx in sample:
                gain = self.func.marginal_gain(selected, idx)
                n_evals += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                marginals.append(best_gain)

        value = self.func.evaluate(selected)
        return SubmodularResult(
            selected=selected,
            value=value,
            marginal_gains=marginals,
            n_evaluations=n_evals,
            metadata={"algorithm": "stochastic_greedy", "epsilon": self.epsilon},
        )


# ---------------------------------------------------------------------------
# Random Greedy (non-monotone)
# ---------------------------------------------------------------------------

class RandomGreedy:
    """Randomized greedy for non-monotone submodular maximization.

    Randomly decides whether to add the greedy element or skip it.
    Provides 1/e approximation for unconstrained non-monotone.
    """

    def __init__(
        self,
        func: SubmodularFunction,
        n: int,
        seed: int = 42,
    ):
        self.func = func
        self.n = n
        self.rng = np.random.RandomState(seed)

    def maximize(self, k: int) -> SubmodularResult:
        selected: List[int] = []
        marginals: List[float] = []
        n_evals = 0

        for _ in range(k):
            best_gain = -float("inf")
            best_idx = -1
            for i in range(self.n):
                if i in selected:
                    continue
                gain = self.func.marginal_gain(selected, i)
                n_evals += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx < 0 or best_gain <= 0:
                break
            # Randomly decide to add or skip
            if self.rng.random() < 0.5:
                selected.append(best_idx)
                marginals.append(best_gain)

        value = self.func.evaluate(selected)
        return SubmodularResult(
            selected=selected,
            value=value,
            marginal_gains=marginals,
            n_evaluations=n_evals,
            metadata={"algorithm": "random_greedy"},
        )


# ---------------------------------------------------------------------------
# Double Greedy (non-monotone)
# ---------------------------------------------------------------------------

class DoubleGreedy:
    """Double greedy for unconstrained non-monotone submodular maximization.

    Maintains two solutions X (growing) and Y (shrinking) simultaneously.
    Provides 1/2 approximation for unconstrained.
    """

    def __init__(
        self,
        func: SubmodularFunction,
        n: int,
    ):
        self.func = func
        self.n = n

    def maximize(self) -> SubmodularResult:
        X: List[int] = []
        Y = list(range(self.n))
        n_evals = 0

        for i in range(self.n):
            # Gain of adding i to X
            gain_add = self.func.marginal_gain(X, i)
            n_evals += 1
            # Gain of removing i from Y
            Y_without = [j for j in Y if j != i]
            gain_remove = self.func.evaluate(Y) - self.func.evaluate(Y_without)
            n_evals += 2

            if gain_add >= gain_remove:
                X.append(i)
            else:
                Y = Y_without

        # Return better of X and Y
        val_X = self.func.evaluate(X)
        val_Y = self.func.evaluate(Y)
        if val_X >= val_Y:
            return SubmodularResult(
                selected=X, value=val_X, marginal_gains=[], n_evaluations=n_evals,
                metadata={"algorithm": "double_greedy", "solution": "X"},
            )
        return SubmodularResult(
            selected=Y, value=val_Y, marginal_gains=[], n_evaluations=n_evals,
            metadata={"algorithm": "double_greedy", "solution": "Y"},
        )


# ---------------------------------------------------------------------------
# Streaming Submodular Maximization
# ---------------------------------------------------------------------------

class StreamingSubmodular:
    """Streaming algorithm for submodular maximization.

    Processes elements one at a time, maintaining a solution of size k.
    Uses the Sieve-Streaming algorithm.
    """

    def __init__(
        self,
        func: SubmodularFunction,
        k: int,
        epsilon: float = 0.1,
    ):
        self.func = func
        self.k = k
        self.epsilon = epsilon
        # Thresholds for sieving
        self.thresholds: List[float] = []
        self.sieves: Dict[float, List[int]] = {}

    def initialize(self, max_marginal: float) -> None:
        """Initialize sieves with geometric thresholds."""
        t = max_marginal
        while t > max_marginal * self.epsilon:
            self.thresholds.append(t)
            self.sieves[t] = []
            t *= (1 - self.epsilon)

    def process(self, element: int) -> None:
        """Process a new element."""
        for tau in self.thresholds:
            if len(self.sieves[tau]) >= self.k:
                continue
            gain = self.func.marginal_gain(self.sieves[tau], element)
            if gain >= tau / (2 * self.k):
                self.sieves[tau].append(element)

    def get_solution(self) -> SubmodularResult:
        """Return best solution among all sieves."""
        best_val = -float("inf")
        best_sieve: List[int] = []
        for tau, sieve in self.sieves.items():
            val = self.func.evaluate(sieve)
            if val > best_val:
                best_val = val
                best_sieve = sieve

        return SubmodularResult(
            selected=best_sieve,
            value=best_val,
            marginal_gains=[],
            n_evaluations=0,
            metadata={"algorithm": "streaming_sieve"},
        )

    def run(self, n_elements: int) -> SubmodularResult:
        """Process n_elements through the stream."""
        # Estimate max marginal
        max_mg = max(self.func.evaluate([i]) for i in range(min(n_elements, 10)))
        self.initialize(max_mg)

        for i in range(n_elements):
            self.process(i)

        return self.get_solution()


# ---------------------------------------------------------------------------
# Multilinear extension utilities
# ---------------------------------------------------------------------------

class MultilinearExtension:
    """Utilities for the multilinear extension of submodular functions."""

    def __init__(
        self,
        func: SubmodularFunction,
        n: int,
        n_samples: int = 100,
        seed: int = 42,
    ):
        self.func = func
        self.n = n
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate F(x) = E[f(S)] where S ~ Bernoulli(x)."""
        total = 0.0
        for _ in range(self.n_samples):
            S = [i for i in range(self.n) if self.rng.random() < x[i]]
            total += self.func.evaluate(S)
        return total / self.n_samples

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Estimate gradient of multilinear extension."""
        grad = np.zeros(self.n)
        for i in range(self.n):
            total = 0.0
            for _ in range(self.n_samples):
                S = [j for j in range(self.n) if j != i and self.rng.random() < x[j]]
                marginal = self.func.evaluate(S + [i]) - self.func.evaluate(S)
                total += marginal
            grad[i] = total / self.n_samples
        return grad

    def concavity_gap(self, x: np.ndarray) -> float:
        """Measure gap between F(x) and concave closure at x."""
        f_x = self.evaluate(x)
        # Upper bound: concave closure is at most sum of marginals
        grad = self.gradient(x)
        concave_upper = np.sum(grad * x)
        return max(concave_upper - f_x, 0.0)
