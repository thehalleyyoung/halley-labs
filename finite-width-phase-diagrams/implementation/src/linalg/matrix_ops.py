"""
Matrix operations for finite-width phase diagram computations.

Provides structured linear algebra primitives—Kronecker products, Sylvester
and Lyapunov solvers, low-rank updates, and matrix balancing—used throughout
the kernel and moment computations that characterise phase boundaries at
finite width.

Mathematical conventions
------------------------
* Matrices are stored as 2-D NumPy arrays (dtype ``float64`` or
  ``complex128``).
* ``vec(X)`` denotes column-major vectorisation of *X*.
* ⊗ denotes the Kronecker (tensor) product.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray


# ======================================================================
#  Kronecker product
# ======================================================================

@dataclass
class KroneckerProduct:
    """Lazy representation of A ⊗ B.

    Stores the two factors and exposes matrix-vector products, traces,
    determinants, and solves *without* materialising the full Kronecker
    product whenever possible.

    Parameters
    ----------
    A : NDArray
        Left factor, shape ``(m, m)``.
    B : NDArray
        Right factor, shape ``(n, n)``.
    """

    A: NDArray
    B: NDArray

    def __post_init__(self) -> None:
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        self.B = np.atleast_2d(np.asarray(self.B, dtype=float))
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError("A and B must be 2-D arrays.")

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the full Kronecker product ``(m*n, m*n)``."""
        return (self.A.shape[0] * self.B.shape[0],
                self.A.shape[1] * self.B.shape[1])

    # ------------------------------------------------------------------
    #  Dense materialisation
    # ------------------------------------------------------------------

    def to_dense(self) -> NDArray:
        """Materialise A ⊗ B as a dense array using ``np.kron``."""
        return np.kron(self.A, self.B)

    # ------------------------------------------------------------------
    #  Matrix-vector products
    # ------------------------------------------------------------------

    def matvec(self, x: NDArray) -> NDArray:
        """Compute (A ⊗ B) x without forming the Kronecker product.

        Uses the identity ``(A ⊗ B) vec(X) = vec(B X Aᵀ)`` where
        ``x = vec(X)`` with *X* of shape ``(n, m)`` (column-major).

        Parameters
        ----------
        x : NDArray, shape ``(m*n,)``

        Returns
        -------
        NDArray, shape ``(m*n,)``
        """
        m = self.A.shape[1]
        n = self.B.shape[1]
        X = x.reshape((n, m), order="F")
        Y = self.B @ X @ self.A.T
        return Y.reshape(-1, order="F")

    def rmatvec(self, x: NDArray) -> NDArray:
        """Compute xᵀ (A ⊗ B), i.e. (Aᵀ ⊗ Bᵀ) x.

        Uses ``(Aᵀ ⊗ Bᵀ) vec(X) = vec(Bᵀ X A)``.

        Parameters
        ----------
        x : NDArray, shape ``(m*n,)``

        Returns
        -------
        NDArray, shape ``(m*n,)``
        """
        m = self.A.shape[0]
        n = self.B.shape[0]
        X = x.reshape((n, m), order="F")
        Y = self.B.T @ X @ self.A
        return Y.reshape(-1, order="F")

    # ------------------------------------------------------------------
    #  Algebraic quantities
    # ------------------------------------------------------------------

    def trace(self) -> float:
        """``tr(A ⊗ B) = tr(A) · tr(B)``."""
        return float(np.trace(self.A) * np.trace(self.B))

    def det(self) -> float:
        """``det(A ⊗ B) = det(A)^n · det(B)^m``.

        Here *m* and *n* are the sizes of *A* and *B* respectively.
        """
        m = self.A.shape[0]
        n = self.B.shape[0]
        return float(np.linalg.det(self.A) ** n *
                      np.linalg.det(self.B) ** m)

    # ------------------------------------------------------------------
    #  Solve
    # ------------------------------------------------------------------

    def solve(self, b: NDArray) -> NDArray:
        """Solve ``(A ⊗ B) x = b`` via the factored form.

        Rewrites the system as ``B X Aᵀ = R`` where ``b = vec(R)`` and
        solves two smaller systems.

        Parameters
        ----------
        b : NDArray, shape ``(m*n,)``

        Returns
        -------
        NDArray, shape ``(m*n,)``
            Solution vector.

        Raises
        ------
        np.linalg.LinAlgError
            If either *A* or *B* is singular.
        """
        m = self.A.shape[0]
        n = self.B.shape[0]
        R = b.reshape((n, m), order="F")

        # Solve B @ Z = R  →  Z = B⁻¹ R
        try:
            Z = sla.solve(self.B, R)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Factor B is singular.")

        # Solve X Aᵀ = Z  →  A Xᵀ = Zᵀ  →  Xᵀ = A⁻¹ Zᵀ
        try:
            X_T = sla.solve(self.A, Z.T)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Factor A is singular.")

        return X_T.T.reshape(-1, order="F")

    # ------------------------------------------------------------------
    #  Eigenvalues
    # ------------------------------------------------------------------

    def eigenvalues(self) -> NDArray:
        """Eigenvalues of A ⊗ B.

        ``σ(A ⊗ B) = {λ_i(A) · μ_j(B)}`` for all pairs ``(i, j)``.

        Returns
        -------
        NDArray, shape ``(m*n,)``
            All eigenvalues (may be complex).
        """
        eig_A = np.linalg.eigvals(self.A)
        eig_B = np.linalg.eigvals(self.B)
        return np.outer(eig_A, eig_B).ravel()


# ======================================================================
#  Sylvester equation solver
# ======================================================================

@dataclass
class SylvesterSolver:
    """Solve the Sylvester equation AX + XB = C and variants.

    Wraps ``scipy.linalg.solve_sylvester`` with condition-number
    estimation and residual validation.

    Parameters
    ----------
    rtol : float
        Relative tolerance for residual checks.
    """

    rtol: float = 1e-8

    # ------------------------------------------------------------------
    #  Standard Sylvester
    # ------------------------------------------------------------------

    def solve(self, A: NDArray, B: NDArray, C: NDArray) -> NDArray:
        """Solve AX + XB = C.

        Parameters
        ----------
        A : NDArray, shape ``(m, m)``
        B : NDArray, shape ``(n, n)``
        C : NDArray, shape ``(m, n)``

        Returns
        -------
        X : NDArray, shape ``(m, n)``

        Warns
        -----
        UserWarning
            If the residual exceeds *rtol*.
        """
        X = sla.solve_sylvester(A, B, C)
        self._validate_solution(A, B, C, X)
        return X

    # ------------------------------------------------------------------
    #  Generalised Sylvester  AXB + CXD = E
    # ------------------------------------------------------------------

    def solve_generalized(
        self,
        A: NDArray,
        B: NDArray,
        C: NDArray,
        E: NDArray,
        F: NDArray,
    ) -> NDArray:
        """Solve the generalised Sylvester equation AXB + CXD = F.

        Reduces to a standard Sylvester equation when *B* and *D* are
        invertible:  A X + C X (D B⁻¹) = F B⁻¹  ⇒  A X + X' B' = C'
        with appropriate substitutions.

        Parameters
        ----------
        A, B, C, E, F : NDArray
            Coefficient matrices.  *A*, *C* are ``(m, m)``; *B*, *E*
            are ``(n, n)``; *F* is ``(m, n)``.

        Returns
        -------
        X : NDArray, shape ``(m, n)``
        """
        B_inv = stable_inverse(B)
        D_inv_B = E @ B_inv  # D B⁻¹  (here E plays the role of D)
        rhs = F @ B_inv

        # Now solve: A X + C X (D B⁻¹) = rhs
        # Rewrite as: A C⁻¹ (C X) + (C X)(D B⁻¹) = rhs
        # Let Y = C X, M = A C⁻¹  →  M Y + Y (D B⁻¹) = rhs
        try:
            C_inv = stable_inverse(C)
            M = A @ C_inv
            Y = sla.solve_sylvester(M, D_inv_B, rhs)
            X = C_inv @ Y
        except np.linalg.LinAlgError:
            # Fall back to dense Kronecker solve
            m, n = F.shape
            I_n = np.eye(n)
            I_m = np.eye(m)
            op = np.kron(B.T, A) + np.kron(E.T, C)
            X = sla.solve(op, F.reshape(-1, order="F"))
            X = X.reshape((m, n), order="F")

        return X

    # ------------------------------------------------------------------
    #  Condition number
    # ------------------------------------------------------------------

    def condition_number(self, A: NDArray, B: NDArray) -> float:
        """Estimate the condition number of the Sylvester operator.

        The operator ``T(X) = AX + XB`` can be written as the
        Kronecker sum ``I ⊗ A + Bᵀ ⊗ I``.  We estimate its 2-norm
        condition number via singular values.

        Parameters
        ----------
        A : NDArray, shape ``(m, m)``
        B : NDArray, shape ``(n, n)``

        Returns
        -------
        float
            Estimated condition number.
        """
        m = A.shape[0]
        n = B.shape[0]
        I_m = np.eye(m)
        I_n = np.eye(n)
        T = np.kron(I_n, A) + np.kron(B.T, I_m)
        sv = sla.svdvals(T)
        if sv[-1] < 1e-15:
            return np.inf
        return float(sv[0] / sv[-1])

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_solution(
        self,
        A: NDArray,
        B: NDArray,
        C: NDArray,
        X: NDArray,
    ) -> None:
        """Check ||AX + XB - C|| / ||C|| < rtol."""
        residual = A @ X + X @ B - C
        rel = sla.norm(residual) / max(sla.norm(C), 1e-30)
        if rel > self.rtol:
            warnings.warn(
                f"Sylvester residual is large: "
                f"||AX+XB-C||/||C|| = {rel:.3e} > rtol = {self.rtol:.1e}",
                stacklevel=3,
            )


# ======================================================================
#  Lyapunov equation solver
# ======================================================================

@dataclass
class LyapunovSolver:
    """Solve continuous and discrete Lyapunov equations.

    Continuous:  ``A X + X Aᵀ = -Q``
    Discrete  :  ``A X Aᵀ - X + Q = 0``

    Parameters
    ----------
    rtol : float
        Relative tolerance for residual validation.
    """

    rtol: float = 1e-8

    # ------------------------------------------------------------------
    #  Continuous
    # ------------------------------------------------------------------

    def solve_continuous(self, A: NDArray, Q: NDArray) -> NDArray:
        """Solve AX + XAᵀ = -Q (continuous Lyapunov).

        Parameters
        ----------
        A : NDArray, shape ``(n, n)``
        Q : NDArray, shape ``(n, n)``
            Typically symmetric positive-semidefinite.

        Returns
        -------
        X : NDArray, shape ``(n, n)``
        """
        X = sla.solve_continuous_lyapunov(A, -Q)
        self._validate_solution(A, Q, X, "continuous")
        return X

    # ------------------------------------------------------------------
    #  Discrete
    # ------------------------------------------------------------------

    def solve_discrete(self, A: NDArray, Q: NDArray) -> NDArray:
        """Solve AXAᵀ - X + Q = 0 (discrete Lyapunov).

        Parameters
        ----------
        A : NDArray, shape ``(n, n)``
        Q : NDArray, shape ``(n, n)``

        Returns
        -------
        X : NDArray, shape ``(n, n)``
        """
        X = sla.solve_discrete_lyapunov(A, Q)
        self._validate_solution(A, Q, X, "discrete")
        return X

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_solution(
        self,
        A: NDArray,
        Q: NDArray,
        X: NDArray,
        equation_type: Literal["continuous", "discrete"],
    ) -> None:
        """Check residual of the Lyapunov solution.

        Parameters
        ----------
        equation_type : ``"continuous"`` or ``"discrete"``
        """
        if equation_type == "continuous":
            residual = A @ X + X @ A.T + Q
        else:
            residual = A @ X @ A.T - X + Q

        rel = sla.norm(residual) / max(sla.norm(Q), 1e-30)
        if rel > self.rtol:
            warnings.warn(
                f"Lyapunov ({equation_type}) residual is large: "
                f"{rel:.3e} > rtol = {self.rtol:.1e}",
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    #  Stability check
    # ------------------------------------------------------------------

    @staticmethod
    def is_stable(
        A: NDArray,
        kind: Literal["continuous", "discrete"] = "continuous",
    ) -> bool:
        """Check if *A* is stable.

        For continuous-time systems, stability requires all eigenvalues
        to have strictly negative real parts.  For discrete-time, all
        eigenvalues must lie strictly inside the unit circle.

        Parameters
        ----------
        A : NDArray, shape ``(n, n)``
        kind : ``"continuous"`` or ``"discrete"``

        Returns
        -------
        bool
        """
        eigvals = np.linalg.eigvals(A)
        if kind == "continuous":
            return bool(np.all(eigvals.real < 0))
        return bool(np.all(np.abs(eigvals) < 1.0))


# ======================================================================
#  Low-rank updates
# ======================================================================

@dataclass
class LowRankUpdate:
    """Rank-one and rank-*k* updates to inverses and factorisations.

    Implements Sherman–Morrison, Woodbury, rank-one eigenvalue update,
    and Cholesky rank-one update.
    """

    # ------------------------------------------------------------------
    #  Sherman–Morrison
    # ------------------------------------------------------------------

    @staticmethod
    def sherman_morrison(
        A_inv: NDArray,
        u: NDArray,
        v: NDArray,
    ) -> NDArray:
        """Compute (A + u vᵀ)⁻¹ via the Sherman–Morrison formula.

        ``(A + u vᵀ)⁻¹ = A⁻¹ − (A⁻¹ u vᵀ A⁻¹) / (1 + vᵀ A⁻¹ u)``

        Parameters
        ----------
        A_inv : NDArray, shape ``(n, n)``
            Inverse of the original matrix *A*.
        u, v : NDArray, shape ``(n,)``
            Rank-one update vectors.

        Returns
        -------
        NDArray, shape ``(n, n)``
            Updated inverse.

        Raises
        ------
        ValueError
            If the denominator ``1 + vᵀ A⁻¹ u`` is near zero
            (update produces a singular matrix).
        """
        u = u.reshape(-1, 1)
        v = v.reshape(-1, 1)
        A_inv_u = A_inv @ u
        vT_A_inv = v.T @ A_inv
        denom = 1.0 + float(v.T @ A_inv_u)
        if abs(denom) < 1e-14:
            raise ValueError(
                "Sherman–Morrison denominator is near zero; "
                "the updated matrix is (near) singular."
            )
        return A_inv - (A_inv_u @ vT_A_inv) / denom

    # ------------------------------------------------------------------
    #  Woodbury
    # ------------------------------------------------------------------

    @staticmethod
    def woodbury(
        A_inv: NDArray,
        U: NDArray,
        C: NDArray,
        V: NDArray,
    ) -> NDArray:
        """Compute (A + U C V)⁻¹ via the Woodbury identity.

        ``(A + UCV)⁻¹ = A⁻¹ − A⁻¹ U (C⁻¹ + V A⁻¹ U)⁻¹ V A⁻¹``

        Parameters
        ----------
        A_inv : NDArray, shape ``(n, n)``
        U : NDArray, shape ``(n, k)``
        C : NDArray, shape ``(k, k)``
        V : NDArray, shape ``(k, n)``

        Returns
        -------
        NDArray, shape ``(n, n)``
        """
        A_inv_U = A_inv @ U
        V_A_inv = V @ A_inv
        C_inv = sla.inv(C)
        inner = C_inv + V @ A_inv_U
        inner_inv = sla.inv(inner)
        return A_inv - A_inv_U @ inner_inv @ V_A_inv

    # ------------------------------------------------------------------
    #  Rank-one eigenvalue update
    # ------------------------------------------------------------------

    def rank_one_eigenvalue_update(
        self,
        eigenvalues: NDArray,
        eigenvectors: NDArray,
        u: NDArray,
        rho: float,
    ) -> Tuple[NDArray, NDArray]:
        """Update eigendecomposition of ``A + ρ u uᵀ``.

        Given ``A = Q diag(d) Qᵀ``, the eigenvalues of ``A + ρ u uᵀ``
        are the roots of the secular equation

            ``f(σ) = 1 + ρ Σ_i z_i² / (d_i − σ) = 0``

        where ``z = Qᵀ u``.

        Parameters
        ----------
        eigenvalues : NDArray, shape ``(n,)``
        eigenvectors : NDArray, shape ``(n, n)``
        u : NDArray, shape ``(n,)``
        rho : float
            Scaling factor for the rank-one term.

        Returns
        -------
        new_eigenvalues : NDArray, shape ``(n,)``
        new_eigenvectors : NDArray, shape ``(n, n)``
        """
        d = eigenvalues.copy().astype(float)
        Q = eigenvectors.copy().astype(float)
        z = Q.T @ u

        n = len(d)
        new_eigs = np.empty(n, dtype=float)

        # Sort eigenvalues for bracket search
        idx = np.argsort(d)
        d_sorted = d[idx]
        z_sorted = z[idx]

        for k in range(n):
            # Bracket: eigenvalue k lies in (d_sorted[k], d_sorted[k+1])
            # for rho > 0 (or shifted for rho < 0).
            if abs(z_sorted[k]) < 1e-15:
                new_eigs[k] = d_sorted[k]
                continue

            if rho > 0:
                lo = d_sorted[k] + 1e-14
                hi = (d_sorted[k + 1] - 1e-14 if k < n - 1
                      else d_sorted[k] + abs(rho) * np.dot(z, z) + 1.0)
            else:
                hi = d_sorted[k] - 1e-14
                lo = (d_sorted[k - 1] + 1e-14 if k > 0
                      else d_sorted[k] - abs(rho) * np.dot(z, z) - 1.0)

            # Bisection to find root of secular equation
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fmid = self._secular_equation(d_sorted, z_sorted, rho, mid)
                if abs(fmid) < 1e-13 or (hi - lo) < 1e-15:
                    break
                if fmid > 0:
                    if rho > 0:
                        lo = mid
                    else:
                        hi = mid
                else:
                    if rho > 0:
                        hi = mid
                    else:
                        lo = mid
            new_eigs[k] = mid

        # Compute eigenvectors via the formula:
        # v_k = normalise( (D - σ_k I)⁻¹ z )
        new_vecs = np.empty((n, n), dtype=float)
        for k in range(n):
            diffs = d_sorted - new_eigs[k]
            diffs[np.abs(diffs) < 1e-15] = 1e-15
            w = z_sorted / diffs
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-30:
                new_vecs[:, k] = Q[:, idx[k]]
            else:
                new_vecs[:, k] = Q[:, idx] @ (w / norm_w)

        return new_eigs, new_vecs

    # ------------------------------------------------------------------
    #  Secular equation
    # ------------------------------------------------------------------

    @staticmethod
    def _secular_equation(
        d: NDArray,
        z: NDArray,
        rho: float,
        sigma: float,
    ) -> float:
        """Evaluate the secular equation.

        ``f(σ) = 1 + ρ Σ_i z_i² / (d_i − σ)``

        Parameters
        ----------
        d : NDArray
            Sorted eigenvalues of the original matrix.
        z : NDArray
            Transformed update vector ``z = Qᵀ u``.
        rho : float
            Rank-one scaling.
        sigma : float
            Point at which to evaluate *f*.

        Returns
        -------
        float
        """
        diffs = d - sigma
        # Guard against exact zeros
        diffs[np.abs(diffs) < 1e-30] = 1e-30
        return 1.0 + rho * np.sum(z ** 2 / diffs)

    # ------------------------------------------------------------------
    #  Cholesky rank-one update
    # ------------------------------------------------------------------

    @staticmethod
    def cholesky_rank_one_update(
        L: NDArray,
        x: NDArray,
        sign: Literal["+", "-"] = "+",
    ) -> NDArray:
        """Rank-one update/downdate of a Cholesky factor.

        Given ``A = L Lᵀ``, compute the Cholesky factor of
        ``A ± x xᵀ`` in *O(n²)* time.

        Parameters
        ----------
        L : NDArray, shape ``(n, n)``
            Lower-triangular Cholesky factor.
        x : NDArray, shape ``(n,)``
            Update vector.
        sign : ``"+"`` or ``"-"``
            ``"+"`` for update, ``"-"`` for downdate.

        Returns
        -------
        NDArray, shape ``(n, n)``
            Updated lower-triangular Cholesky factor.

        Raises
        ------
        ValueError
            If a downdate would make the matrix non-positive-definite.
        """
        n = L.shape[0]
        L_new = L.copy()
        w = x.copy().astype(float)
        s = 1.0 if sign == "+" else -1.0

        for k in range(n):
            r = np.sqrt(L_new[k, k] ** 2 + s * w[k] ** 2)
            if r < 1e-15:
                raise ValueError(
                    "Cholesky downdate failed: matrix is not "
                    "positive-definite after update."
                )
            c = r / L_new[k, k]
            ss = w[k] / L_new[k, k]
            L_new[k, k] = r
            if k + 1 < n:
                L_new[k + 1:, k] = (
                    L_new[k + 1:, k] + s * ss * w[k + 1:]
                ) / c
                w[k + 1:] = c * w[k + 1:] - ss * L_new[k + 1:, k]

        return L_new


# ======================================================================
#  Matrix balancing
# ======================================================================

@dataclass
class MatrixBalancer:
    """Iterative diagonal scaling to improve matrix conditioning.

    Aims to make row and column norms approximately equal, which
    improves the accuracy of eigenvalue and linear-system solvers.

    Parameters
    ----------
    max_iter : int
        Maximum number of balancing sweeps.
    tol : float
        Convergence tolerance on relative norm change.
    """

    max_iter: int = 50
    tol: float = 1e-6

    # ------------------------------------------------------------------
    #  Main balancing routine
    # ------------------------------------------------------------------

    def balance(self, A: NDArray) -> Tuple[NDArray, NDArray]:
        """Balance the matrix *A* via iterative diagonal scaling.

        Returns the balanced matrix ``D⁻¹ A D`` and the diagonal
        scaling matrix *D*.

        Parameters
        ----------
        A : NDArray, shape ``(n, n)``

        Returns
        -------
        A_balanced : NDArray, shape ``(n, n)``
        D : NDArray, shape ``(n, n)``
            Diagonal scaling matrix such that ``A_balanced = D⁻¹ A D``.
        """
        n = A.shape[0]
        D = np.ones(n, dtype=float)
        A_bal = A.copy().astype(float)

        for iteration in range(self.max_iter):
            A_prev = A_bal.copy()
            A_bal, D = self._balance_iteration(A_bal, D)
            change = sla.norm(A_bal - A_prev) / max(sla.norm(A_prev), 1e-30)
            if change < self.tol:
                break

        return A_bal, np.diag(D)

    # ------------------------------------------------------------------
    #  Single iteration
    # ------------------------------------------------------------------

    def _balance_iteration(
        self,
        A: NDArray,
        D: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Perform one sweep of Parlett–Reinsch balancing.

        For each row/column pair, scale so that the row norm ≈ column
        norm (using powers of 2 for numerical safety).

        Parameters
        ----------
        A : NDArray, shape ``(n, n)``
            Current (partially balanced) matrix.
        D : NDArray, shape ``(n,)``
            Current diagonal scaling entries.

        Returns
        -------
        A_new : NDArray
        D_new : NDArray
        """
        n = A.shape[0]
        A_new = A.copy()
        D_new = D.copy()

        for i in range(n):
            row_norm = np.sum(np.abs(A_new[i, :])) - np.abs(A_new[i, i])
            col_norm = np.sum(np.abs(A_new[:, i])) - np.abs(A_new[i, i])

            if row_norm < 1e-30 or col_norm < 1e-30:
                continue

            # Compute scaling factor as a power of 2
            f = 1.0
            target = np.sqrt(row_norm * col_norm)
            while col_norm < target / 2.0:
                col_norm *= 2.0
                row_norm /= 2.0
                f *= 2.0
            while col_norm > target * 2.0:
                col_norm /= 2.0
                row_norm *= 2.0
                f /= 2.0

            if abs(f - 1.0) > 1e-14:
                A_new[i, :] /= f
                A_new[:, i] *= f
                D_new[i] *= f

        return A_new, D_new

    # ------------------------------------------------------------------
    #  Optimal diagonal scaling (heuristic)
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_scaling(A: NDArray) -> NDArray:
        """Compute an optimal diagonal scaling using row/column norms.

        Returns a diagonal vector *d* such that ``diag(d)⁻¹ A diag(d)``
        has approximately equal row and column norms.

        Parameters
        ----------
        A : NDArray, shape ``(n, n)``

        Returns
        -------
        NDArray, shape ``(n,)``
        """
        n = A.shape[0]
        row_norms = np.array([sla.norm(A[i, :]) for i in range(n)])
        col_norms = np.array([sla.norm(A[:, j]) for j in range(n)])

        row_norms = np.maximum(row_norms, 1e-30)
        col_norms = np.maximum(col_norms, 1e-30)

        # Geometric mean balancing: d_i = (r_i / c_i)^{1/4}
        d = (row_norms / col_norms) ** 0.25
        return d

    # ------------------------------------------------------------------
    #  Condition number estimate
    # ------------------------------------------------------------------

    @staticmethod
    def condition_number_estimate(A: NDArray) -> float:
        """Estimate the 1-norm condition number of *A*.

        Uses the LU factorisation and ``scipy.linalg.norm`` for an
        *O(n²)* estimate (as opposed to computing the SVD).

        Parameters
        ----------
        A : NDArray, shape ``(n, n)``

        Returns
        -------
        float
            Estimated 1-norm condition number.
        """
        norm_A = sla.norm(A, 1)
        try:
            lu, piv = sla.lu_factor(A)
            norm_A_inv = 1.0 / sla.norm(sla.lu_solve((lu, piv), np.eye(A.shape[0])), 1)
            # Use a cheaper estimate: solve for a few columns
            n = A.shape[0]
            e = np.zeros(n)
            e[0] = 1.0
            x = sla.lu_solve((lu, piv), e)
            norm_A_inv_est = np.linalg.norm(x, 1)
            # Refine with a couple more columns
            for j in [n // 3, 2 * n // 3]:
                if j < n:
                    e = np.zeros(n)
                    e[j] = 1.0
                    x = sla.lu_solve((lu, piv), e)
                    norm_A_inv_est = max(norm_A_inv_est, np.linalg.norm(x, 1))
            return float(norm_A * norm_A_inv_est)
        except sla.LinAlgError:
            return np.inf


# ======================================================================
#  Helper functions
# ======================================================================

def matrix_logarithm(A: NDArray) -> NDArray:
    """Matrix logarithm with branch-cut handling.

    Uses the Schur decomposition to compute ``log(A)`` via the
    Schur–Parlett algorithm implemented in SciPy, with a fallback
    for matrices with eigenvalues on the negative real axis.

    Parameters
    ----------
    A : NDArray, shape ``(n, n)``

    Returns
    -------
    NDArray, shape ``(n, n)``
        The principal matrix logarithm.

    Raises
    ------
    ValueError
        If *A* is singular (has a zero eigenvalue).
    """
    eigvals = np.linalg.eigvals(A)
    if np.any(np.abs(eigvals) < 1e-15):
        raise ValueError(
            "Cannot compute the logarithm of a singular matrix."
        )

    # Check for eigenvalues on the negative real axis
    neg_real = (eigvals.real < 0) & (np.abs(eigvals.imag) < 1e-12)
    if np.any(neg_real):
        warnings.warn(
            "Matrix has eigenvalues on the negative real axis; "
            "the principal logarithm may have a large imaginary part.",
            stacklevel=2,
        )

    return sla.logm(A)


def matrix_square_root(A: NDArray) -> NDArray:
    """Matrix square root via Schur decomposition.

    Computes *S* such that ``S @ S = A``.  Uses the Schur form to
    compute the square root of the upper-triangular factor, then
    transforms back.

    Parameters
    ----------
    A : NDArray, shape ``(n, n)``

    Returns
    -------
    NDArray, shape ``(n, n)``
        A matrix *S* with ``S @ S ≈ A``.

    Raises
    ------
    ValueError
        If *A* has negative real eigenvalues (no real square root).
    """
    T, Z = sla.schur(A, output="complex")
    n = T.shape[0]

    # Compute square root of upper-triangular T column by column
    R = np.zeros_like(T)
    for j in range(n):
        eigval = T[j, j]
        if eigval.real < 0 and abs(eigval.imag) < 1e-12:
            raise ValueError(
                f"Matrix has a negative real eigenvalue ({eigval.real:.4e}); "
                "no real square root exists."
            )
        R[j, j] = np.sqrt(eigval)

    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            s = T[i, j]
            for k in range(i + 1, j):
                s -= R[i, k] * R[k, j]
            denom = R[i, i] + R[j, j]
            if abs(denom) < 1e-15:
                R[i, j] = 0.0
            else:
                R[i, j] = s / denom

    result = Z @ R @ Z.conj().T

    # Return real part if input was real and result is nearly real
    if np.isrealobj(A) and np.allclose(result.imag, 0, atol=1e-10):
        return result.real
    return result


def stable_inverse(A: NDArray, rcond: float = 1e-12) -> NDArray:
    """Pseudoinverse with condition-number checking.

    If the matrix is well-conditioned (reciprocal condition number
    above *rcond*), returns the standard inverse.  Otherwise falls
    back to the Moore–Penrose pseudoinverse with a warning.

    Parameters
    ----------
    A : NDArray, shape ``(n, n)``
    rcond : float
        Threshold on the reciprocal condition number.

    Returns
    -------
    NDArray, shape ``(n, n)``
    """
    sv = sla.svdvals(A)
    if sv[-1] < rcond * sv[0]:
        warnings.warn(
            f"Matrix is ill-conditioned (rcond ≈ {sv[-1]/sv[0]:.2e}); "
            "using pseudoinverse.",
            stacklevel=2,
        )
        return np.linalg.pinv(A, rcond=rcond)
    return sla.inv(A)
