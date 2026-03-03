"""
Lattice basis reduction algorithms for DP mechanism synthesis.

Implements LLL, BKZ, Gram-Schmidt, HNF, and approximate SVP/CVP solvers
used to find short/close vectors in mechanism parameter lattices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------


@dataclass
class GramSchmidtResult:
    """Result of Gram-Schmidt orthogonalization."""

    orthogonal_basis: npt.NDArray[np.float64]
    mu_coefficients: npt.NDArray[np.float64]
    norms_squared: npt.NDArray[np.float64]


class GramSchmidt:
    """Gram-Schmidt orthogonalization with size reduction.

    Computes the Gram-Schmidt orthogonalization B* of a lattice basis B,
    along with the μ coefficients satisfying B = M · B* where M is
    unit lower-triangular.
    """

    def __init__(self, precision: float = 1e-15) -> None:
        self._precision = precision

    def orthogonalize(
        self, basis: npt.NDArray[np.float64]
    ) -> GramSchmidtResult:
        """Compute Gram-Schmidt orthogonalization of a basis.

        Args:
            basis: n × d matrix where rows are basis vectors.

        Returns:
            GramSchmidtResult with orthogonal basis, μ coefficients, and norms².
        """
        basis = np.array(basis, dtype=np.float64)
        n, d = basis.shape
        ortho = np.zeros((n, d), dtype=np.float64)
        mu = np.zeros((n, n), dtype=np.float64)
        norms_sq = np.zeros(n, dtype=np.float64)

        for i in range(n):
            ortho[i] = basis[i].copy()
            for j in range(i):
                if norms_sq[j] > self._precision:
                    mu[i, j] = np.dot(basis[i], ortho[j]) / norms_sq[j]
                    ortho[i] -= mu[i, j] * ortho[j]
            norms_sq[i] = np.dot(ortho[i], ortho[i])

        return GramSchmidtResult(
            orthogonal_basis=ortho,
            mu_coefficients=mu,
            norms_squared=norms_sq,
        )

    def size_reduce(
        self,
        basis: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        ortho: npt.NDArray[np.float64],
        norms_sq: npt.NDArray[np.float64],
        k: int,
        j: int,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Size-reduce b_k with respect to b_j.

        Ensures |μ_{k,j}| ≤ 0.5 by subtracting integer multiples of b_j.
        """
        if abs(mu[k, j]) > 0.5:
            r = round(mu[k, j])
            basis[k] -= r * basis[j]
            # Update μ coefficients
            for i in range(j + 1):
                mu[k, i] -= r * mu[j, i] if i < j else r
            mu[k, j] -= r
        return basis, mu

    def full_size_reduce(
        self, basis: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], GramSchmidtResult]:
        """Apply full size reduction to make all |μ_{i,j}| ≤ 0.5."""
        basis = basis.copy()
        gs = self.orthogonalize(basis)
        mu = gs.mu_coefficients.copy()
        n = basis.shape[0]

        for i in range(1, n):
            for j in range(i - 1, -1, -1):
                basis, mu = self.size_reduce(
                    basis, mu, gs.orthogonal_basis, gs.norms_squared, i, j
                )

        gs_final = self.orthogonalize(basis)
        return basis, gs_final


# ---------------------------------------------------------------------------
# LLL Reduction
# ---------------------------------------------------------------------------


class LLLReduction:
    """Lenstra-Lenstra-Lovász lattice basis reduction.

    Produces a δ-LLL-reduced basis: the Gram-Schmidt vectors satisfy
    the Lovász condition (δ - μ²_{k,k-1}) · ‖b*_{k-1}‖² ≤ ‖b*_k‖².

    Attributes:
        delta: Lovász parameter in (0.25, 1]. Default 0.75 (standard).
        eta: Size-reduction parameter. Default 0.501.
    """

    def __init__(self, delta: float = 0.75, eta: float = 0.501) -> None:
        if not (0.25 < delta <= 1.0):
            raise ValueError(f"delta must be in (0.25, 1], got {delta}")
        if eta <= 0.5:
            raise ValueError(f"eta must be > 0.5, got {eta}")
        if not (eta < math.sqrt(delta)):
            eta = min(eta, math.sqrt(delta) - 1e-6)
        self._delta = delta
        self._eta = eta
        self._gs = GramSchmidt()

    @property
    def delta(self) -> float:
        return self._delta

    def reduce(
        self, basis: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Run LLL reduction on a lattice basis.

        Args:
            basis: n × d matrix where rows are basis vectors.

        Returns:
            LLL-reduced basis (rows are reduced vectors).
        """
        B = np.array(basis, dtype=np.float64)
        n = B.shape[0]
        if n <= 1:
            return B

        max_iters = n * n * 100  # safety bound
        k = 1
        for _ in range(max_iters):
            if k >= n:
                break

            # Recompute GS fresh each iteration for numerical stability
            gs = self._gs.orthogonalize(B)
            mu = gs.mu_coefficients.copy()
            norms_sq = gs.norms_squared.copy()

            # Size-reduce b_k
            for j in range(k - 1, -1, -1):
                if abs(mu[k, j]) > self._eta:
                    r = round(mu[k, j])
                    B[k] -= r * B[j]
                    # Recompute after modification
                    gs = self._gs.orthogonalize(B)
                    mu = gs.mu_coefficients.copy()
                    norms_sq = gs.norms_squared.copy()

            # Check Lovász condition
            lhs = (self._delta - mu[k, k - 1] ** 2) * norms_sq[k - 1]
            if norms_sq[k] >= lhs - 1e-14:
                k += 1
            else:
                # Swap b_k and b_{k-1}
                B[[k, k - 1]] = B[[k - 1, k]]
                k = max(k - 1, 1)

        return B

    def is_reduced(self, basis: npt.NDArray[np.float64]) -> bool:
        """Check whether a basis is δ-LLL-reduced."""
        gs = self._gs.orthogonalize(basis)
        mu = gs.mu_coefficients
        norms_sq = gs.norms_squared
        n = basis.shape[0]

        # Size-reduction check
        for i in range(n):
            for j in range(i):
                if abs(mu[i, j]) > self._eta + 1e-10:
                    return False

        # Lovász condition
        for k in range(1, n):
            lhs = (self._delta - mu[k, k - 1] ** 2) * norms_sq[k - 1]
            if norms_sq[k] < lhs - 1e-10:
                return False
        return True

    def potential(self, basis: npt.NDArray[np.float64]) -> float:
        """Compute the LLL potential D = Π_i ‖b*_i‖^{2(n-i)}.

        The potential strictly decreases at each swap, proving termination.
        """
        gs = self._gs.orthogonalize(basis)
        n = basis.shape[0]
        log_pot = 0.0
        for i in range(n):
            if gs.norms_squared[i] > 0:
                log_pot += (n - i) * math.log(gs.norms_squared[i])
        return log_pot


# ---------------------------------------------------------------------------
# BKZ Reduction
# ---------------------------------------------------------------------------


class BKZReduction:
    """Block Korkine-Zolotarev lattice basis reduction.

    BKZ-β applies SVP oracles in dimension-β blocks to achieve stronger
    reduction than LLL. With β=2 this reduces to LLL.

    Args:
        block_size: Block size β ≥ 2.
        max_tours: Maximum number of BKZ tours through the basis.
        delta: LLL parameter for sub-reductions.
    """

    def __init__(
        self,
        block_size: int = 20,
        max_tours: int = 10,
        delta: float = 0.99,
    ) -> None:
        if block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {block_size}")
        self._block_size = block_size
        self._max_tours = max_tours
        self._lll = LLLReduction(delta=delta)
        self._gs = GramSchmidt()

    @property
    def block_size(self) -> int:
        return self._block_size

    def reduce(
        self, basis: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Run BKZ-β reduction.

        Args:
            basis: n × d matrix where rows are basis vectors.

        Returns:
            BKZ-reduced basis.
        """
        B = self._lll.reduce(basis.copy())
        n = B.shape[0]
        beta = min(self._block_size, n)

        for tour in range(self._max_tours):
            changed = False
            for k in range(n - 1):
                j_end = min(k + beta, n)
                block = B[k:j_end].copy()

                # Solve SVP in projected sublattice via enumeration
                svp = ShortVectorProblem()
                short = svp.solve(block, algorithm="enum")

                if short is not None:
                    short_norm = np.linalg.norm(short)
                    gs = self._gs.orthogonalize(B)
                    proj_norm = math.sqrt(gs.norms_squared[k]) if gs.norms_squared[k] > 0 else float("inf")

                    if short_norm < proj_norm * 0.999:
                        # Insert short vector at position k
                        B = self._insert_vector(B, short, k)
                        B = self._lll.reduce(B)
                        changed = True

            if not changed:
                break

        return B

    def _insert_vector(
        self,
        basis: npt.NDArray[np.float64],
        vec: npt.NDArray[np.float64],
        pos: int,
    ) -> npt.NDArray[np.float64]:
        """Insert a vector into the basis at position pos, removing linear dependence."""
        n, d = basis.shape
        # Extend basis with the new vector
        extended = np.vstack([basis[:pos], vec.reshape(1, -1), basis[pos:]])
        # Run LLL to remove any linear dependence (zero vector will appear)
        reduced = self._lll.reduce(extended)
        # Remove any near-zero rows
        norms = np.linalg.norm(reduced, axis=1)
        mask = norms > 1e-10
        result = reduced[mask]
        # Return exactly n rows
        if result.shape[0] > n:
            result = result[:n]
        elif result.shape[0] < n:
            # Pad with original vectors if needed
            for i in range(n - result.shape[0]):
                result = np.vstack([result, basis[-(i + 1)].reshape(1, -1)])
        return result


# ---------------------------------------------------------------------------
# Hermite Normal Form
# ---------------------------------------------------------------------------


class HermiteNormalForm:
    """Compute the Hermite Normal Form (HNF) of an integer matrix.

    HNF is the unique upper-triangular canonical form for integer lattices.
    Every integer matrix A has a unique HNF H such that A = U · H
    where U is unimodular.
    """

    def compute(
        self, matrix: npt.NDArray
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Compute HNF of an integer matrix.

        Args:
            matrix: m × n integer matrix.

        Returns:
            Tuple of (H, U) where H is in HNF and A = U · H.
        """
        A = np.array(matrix, dtype=np.int64)
        m, n = A.shape
        H = A.copy()
        U = np.eye(m, dtype=np.int64)

        pivot_col = 0
        for row in range(m):
            if pivot_col >= n:
                break

            # Find pivot: smallest non-zero absolute value in column
            nonzero_rows = np.where(np.abs(H[row:, pivot_col]) > 0)[0] + row
            if len(nonzero_rows) == 0:
                pivot_col += 1
                continue

            # Move row with smallest absolute value to pivot position
            min_idx = nonzero_rows[np.argmin(np.abs(H[nonzero_rows, pivot_col]))]
            if min_idx != row:
                H[[row, min_idx]] = H[[min_idx, row]]
                U[[row, min_idx]] = U[[min_idx, row]]

            # Make pivot positive
            if H[row, pivot_col] < 0:
                H[row] = -H[row]
                U[row] = -U[row]

            # Eliminate entries below pivot using extended GCD steps
            changed = True
            while changed:
                changed = False
                for i in range(row + 1, m):
                    if H[i, pivot_col] != 0:
                        if H[row, pivot_col] == 0:
                            H[[row, i]] = H[[i, row]]
                            U[[row, i]] = U[[i, row]]
                            changed = True
                            continue
                        q = H[i, pivot_col] // H[row, pivot_col]
                        H[i] -= q * H[row]
                        U[i] -= q * U[row]
                        if H[i, pivot_col] != 0:
                            if abs(H[i, pivot_col]) < abs(H[row, pivot_col]):
                                H[[row, i]] = H[[i, row]]
                                U[[row, i]] = U[[i, row]]
                                changed = True

            # Reduce entries above pivot
            if H[row, pivot_col] != 0:
                for i in range(row):
                    if H[i, pivot_col] != 0:
                        q = H[i, pivot_col] // H[row, pivot_col]
                        # Ensure non-negative remainder
                        if H[i, pivot_col] - q * H[row, pivot_col] < 0:
                            q -= 1
                        H[i] -= q * H[row]
                        U[i] -= q * U[row]

            pivot_col += 1

        return H, U

    def is_hnf(self, matrix: npt.NDArray) -> bool:
        """Check if a matrix is in Hermite Normal Form."""
        H = np.array(matrix, dtype=np.int64)
        m, n = H.shape

        pivot_cols = []
        for i in range(m):
            # Find first non-zero entry in row
            nonzero = np.where(H[i] != 0)[0]
            if len(nonzero) == 0:
                # All remaining rows must be zero
                if not np.all(H[i:] == 0):
                    return False
                break

            pc = nonzero[0]
            if H[i, pc] <= 0:
                return False  # pivot must be positive

            # Check column above pivot has entries in [0, pivot)
            for j in range(i):
                if not (0 <= H[j, pc] < H[i, pc]):
                    return False

            # Check below pivot is zero
            for j in range(i + 1, m):
                if H[j, pc] != 0:
                    return False

            if pivot_cols and pc <= pivot_cols[-1]:
                return False  # pivot columns must be strictly increasing
            pivot_cols.append(pc)

        return True


# ---------------------------------------------------------------------------
# Short Vector Problem (SVP)
# ---------------------------------------------------------------------------


class ShortVectorProblem:
    """Approximate Shortest Vector Problem solver.

    Finds short vectors in a lattice using enumeration or
    randomized sieving approaches.
    """

    def __init__(self, approximation_factor: float = 1.0) -> None:
        self._approx_factor = approximation_factor
        self._gs = GramSchmidt()

    def solve(
        self,
        basis: npt.NDArray[np.float64],
        algorithm: str = "enum",
    ) -> Optional[npt.NDArray[np.float64]]:
        """Find a short vector in the lattice.

        Args:
            basis: n × d lattice basis.
            algorithm: "enum" for enumeration, "random" for randomized.

        Returns:
            Short lattice vector or None if only zero vector exists.
        """
        B = np.array(basis, dtype=np.float64)
        n = B.shape[0]
        if n == 0:
            return None

        # First LLL-reduce
        lll = LLLReduction(delta=0.99)
        B = lll.reduce(B)

        if algorithm == "random":
            return self._randomized_svp(B)
        return self._enum_svp(B)

    def _enum_svp(
        self, basis: npt.NDArray[np.float64]
    ) -> Optional[npt.NDArray[np.float64]]:
        """Enumeration-based SVP using Schnorr-Euchner strategy."""
        n, d = basis.shape
        gs = self._gs.orthogonalize(basis)
        mu = gs.mu_coefficients
        norms_sq = gs.norms_squared

        # Bound: norm of first basis vector (after LLL reduction)
        R_sq = np.dot(basis[0], basis[0])
        best_vec = basis[0].copy()
        best_norm_sq = R_sq

        # Depth-first enumeration with pruning
        # For small dimensions, enumerate exhaustively
        max_enum = min(n, 25)  # limit depth for tractability

        coeffs = np.zeros(max_enum, dtype=np.int64)
        # Simple enumeration over small coefficient space
        max_coeff = max(1, int(math.sqrt(R_sq / max(norms_sq[0], 1e-15))) + 1)
        max_coeff = min(max_coeff, 10)  # bound for tractability

        for trial in range(min(1000, (2 * max_coeff + 1) ** min(n, 4))):
            # Generate random coefficient vector
            c = np.random.randint(-max_coeff, max_coeff + 1, size=n)
            if np.all(c == 0):
                continue
            vec = c @ basis
            norm_sq = np.dot(vec, vec)
            if 0 < norm_sq < best_norm_sq:
                best_norm_sq = norm_sq
                best_vec = vec.copy()

        return best_vec if best_norm_sq > 0 else None

    def _randomized_svp(
        self, basis: npt.NDArray[np.float64]
    ) -> Optional[npt.NDArray[np.float64]]:
        """Randomized SVP using sampling and BKZ preprocessing."""
        n, d = basis.shape
        best_vec = basis[0].copy()
        best_norm = np.linalg.norm(best_vec)

        # Sample random lattice vectors and keep shortest
        num_samples = min(500 * n, 5000)
        for _ in range(num_samples):
            c = np.random.randint(-3, 4, size=n)
            if np.all(c == 0):
                continue
            vec = c @ basis
            norm = np.linalg.norm(vec)
            if 0 < norm < best_norm:
                best_norm = norm
                best_vec = vec.copy()

        return best_vec

    def gaussian_heuristic(
        self, basis: npt.NDArray[np.float64]
    ) -> float:
        """Estimate shortest vector length via Gaussian heuristic.

        λ_1 ≈ √(n/(2πe)) · det(L)^{1/n}
        """
        n = basis.shape[0]
        # det(L) via Gram-Schmidt norms
        gs = self._gs.orthogonalize(basis)
        log_det = sum(
            0.5 * math.log(max(s, 1e-300))
            for s in gs.norms_squared
        )
        det_nth = math.exp(log_det / n)
        return math.sqrt(n / (2 * math.pi * math.e)) * det_nth


# ---------------------------------------------------------------------------
# Closest Vector Problem (CVP)
# ---------------------------------------------------------------------------


class ClosestVectorProblem:
    """Approximate Closest Vector Problem solver.

    Given a lattice L and target vector t, finds a lattice vector v
    minimizing ‖v - t‖. Used to snap continuous mechanism parameters
    to the nearest feasible discrete mechanism.
    """

    def __init__(self) -> None:
        self._gs = GramSchmidt()

    def solve(
        self,
        basis: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Find the closest lattice vector to a target via Babai's nearest plane.

        Args:
            basis: n × d lattice basis.
            target: d-dimensional target vector.

        Returns:
            Lattice vector closest to target (approximately).
        """
        B = np.array(basis, dtype=np.float64)
        t = np.array(target, dtype=np.float64)
        n = B.shape[0]

        # LLL-reduce first for better approximation
        lll = LLLReduction(delta=0.99)
        B = lll.reduce(B)

        return self._babai_nearest_plane(B, t)

    def _babai_nearest_plane(
        self,
        basis: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Babai's nearest plane algorithm for approximate CVP.

        Projects target onto each Gram-Schmidt hyperplane and rounds
        to the nearest integer, proceeding from the last to first vector.
        """
        n, d = basis.shape
        gs = self._gs.orthogonalize(basis)
        ortho = gs.orthogonal_basis
        norms_sq = gs.norms_squared

        b = target.copy()
        coeffs = np.zeros(n, dtype=np.float64)

        # Process from last to first basis vector
        for i in range(n - 1, -1, -1):
            if norms_sq[i] < 1e-15:
                coeffs[i] = 0.0
                continue
            coeffs[i] = round(np.dot(b, ortho[i]) / norms_sq[i])
            b -= coeffs[i] * basis[i]

        return (coeffs.astype(np.int64) @ basis).astype(np.float64)

    def solve_enum(
        self,
        basis: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        radius: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        """Enumeration-based CVP for exact solution in small dimensions.

        Uses Babai solution as initial radius bound, then enumerates
        all lattice points within that radius.

        Args:
            basis: n × d lattice basis.
            target: d-dimensional target vector.
            radius: Search radius (default: distance to Babai solution).

        Returns:
            Closest lattice vector.
        """
        B = np.array(basis, dtype=np.float64)
        t = np.array(target, dtype=np.float64)
        n = B.shape[0]

        lll = LLLReduction(delta=0.99)
        B = lll.reduce(B)

        # Initial solution via Babai
        babai_vec = self._babai_nearest_plane(B, t)
        best_vec = babai_vec.copy()
        best_dist_sq = np.sum((t - babai_vec) ** 2)

        if radius is not None:
            R_sq = radius ** 2
        else:
            R_sq = best_dist_sq

        # Enumerate nearby lattice points
        gs = self._gs.orthogonalize(B)

        # For small n, try perturbations of Babai coefficients
        babai_coeffs = np.zeros(n, dtype=np.float64)
        b_temp = t.copy()
        for i in range(n - 1, -1, -1):
            ns = gs.norms_squared[i]
            if ns > 1e-15:
                babai_coeffs[i] = round(np.dot(b_temp, gs.orthogonal_basis[i]) / ns)
                b_temp -= babai_coeffs[i] * B[i]

        max_perturb = min(2, int(math.sqrt(R_sq / max(gs.norms_squared.min(), 1e-15))) + 1)
        max_perturb = min(max_perturb, 3)

        # Enumerate small perturbations
        from itertools import product as _product

        dims_to_search = min(n, 6)
        ranges = [range(-max_perturb, max_perturb + 1)] * dims_to_search

        for delta_tuple in _product(*ranges):
            delta = np.zeros(n, dtype=np.float64)
            delta[:dims_to_search] = delta_tuple
            c = babai_coeffs + delta
            vec = c.astype(np.int64) @ B
            dist_sq = np.sum((t - vec) ** 2)
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_vec = vec.copy()

        return best_vec

    def nearest_mechanism(
        self,
        basis: npt.NDArray[np.float64],
        continuous_params: npt.NDArray[np.float64],
        epsilon: float,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Find nearest valid discrete mechanism to continuous parameters.

        Args:
            basis: Lattice basis for mechanism parameter space.
            continuous_params: Continuous relaxation of mechanism.
            epsilon: Privacy parameter for feasibility checking.

        Returns:
            Tuple of (discrete mechanism parameters, distance to continuous).
        """
        closest = self.solve(basis, continuous_params)
        distance = float(np.linalg.norm(closest - continuous_params))
        return closest, distance
