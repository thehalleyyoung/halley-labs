"""Spectral methods for finite-width phase diagram analysis.

Provides eigendecomposition utilities, randomized SVD, matrix functions,
and pseudospectral computation for studying operator spectra arising in
neural network kernel theory.

Mathematical background
-----------------------
For a matrix A ∈ ℂ^{n×n}, the spectrum σ(A) = {λ : det(A - λI) = 0}
governs stability and convergence of iterative dynamics.  The ε-pseudo-
spectrum σ_ε(A) = {z ∈ ℂ : ‖(zI - A)^{-1}‖ ≥ 1/ε} captures transient
behaviour invisible to eigenvalues alone.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ======================================================================
#  Type aliases
# ======================================================================

RealArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
AnyArray = NDArray


# ======================================================================
#  Helper utilities
# ======================================================================

def _ensure_square(A: AnyArray) -> AnyArray:
    """Validate that *A* is a 2-D square array and return it."""
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Expected square matrix, got shape {A.shape}"
        )
    return A


def _is_hermitian(A: AnyArray, tol: float = 1e-10) -> bool:
    """Return True when ‖A - A^H‖_F / ‖A‖_F < *tol*."""
    norm_A = np.linalg.norm(A, "fro")
    if norm_A == 0.0:
        return True
    return float(np.linalg.norm(A - A.conj().T, "fro") / norm_A) < tol


# ======================================================================
#  SpectralDecomposition
# ======================================================================

@dataclass
class SpectralDecomposition:
    """Eigenvalue analysis with error bounds and convenience methods.

    Parameters
    ----------
    check_hermitian : bool
        When True, ``eigh_with_bounds`` verifies Hermitian symmetry before
        calling the specialised solver.
    error_tol : float
        Tolerance used when checking reconstruction error after
        eigendecomposition.
    """

    check_hermitian: bool = True
    error_tol: float = 1e-10

    # ------------------------------------------------------------------
    #  Hermitian eigendecomposition
    # ------------------------------------------------------------------

    def eigh_with_bounds(
        self, A: AnyArray
    ) -> Tuple[RealArray, AnyArray, float]:
        r"""Eigendecomposition of a Hermitian matrix with backward error.

        Computes eigenvalues λ and eigenvectors V of A such that
        A V = V diag(λ) and returns the relative backward error

        .. math::
            \eta = \frac{\|A - V \Lambda V^H\|_F}{\|A\|_F}.

        Parameters
        ----------
        A : array_like, shape (n, n)
            Hermitian (or real-symmetric) matrix.

        Returns
        -------
        eigenvalues : ndarray, shape (n,)
            Eigenvalues in ascending order.
        eigenvectors : ndarray, shape (n, n)
            Corresponding orthonormal eigenvectors as columns.
        backward_error : float
            Relative reconstruction error η.

        Raises
        ------
        ValueError
            If *A* is not square or fails the Hermitian check.
        """
        A = _ensure_square(A)
        n = A.shape[0]

        if self.check_hermitian and not _is_hermitian(A, self.error_tol):
            raise ValueError(
                "Matrix does not appear Hermitian within the "
                f"specified tolerance ({self.error_tol:.1e})."
            )

        eigenvalues, eigenvectors = la.eigh(A)

        # Backward error  η = ‖A - V Λ V^H‖_F / ‖A‖_F
        norm_A = np.linalg.norm(A, "fro")
        if norm_A == 0.0:
            backward_error = 0.0
        else:
            reconstructed = (
                eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
            )
            backward_error = float(
                np.linalg.norm(A - reconstructed, "fro") / norm_A
            )

        return eigenvalues, eigenvectors, backward_error

    # ------------------------------------------------------------------
    #  General eigendecomposition with sorting
    # ------------------------------------------------------------------

    def eig_sorted(
        self,
        A: AnyArray,
        sort: Literal[
            "real_descending",
            "real_ascending",
            "abs_descending",
            "abs_ascending",
        ] = "real_descending",
    ) -> Tuple[ComplexArray, ComplexArray]:
        r"""General eigendecomposition with configurable sort order.

        Parameters
        ----------
        A : array_like, shape (n, n)
            Square matrix.
        sort : str
            Sorting criterion for the returned eigenvalues:

            * ``'real_descending'``  – descending by Re(λ)
            * ``'real_ascending'``   – ascending by Re(λ)
            * ``'abs_descending'``   – descending by |λ|
            * ``'abs_ascending'``    – ascending by |λ|

        Returns
        -------
        eigenvalues : ndarray, shape (n,)
        eigenvectors : ndarray, shape (n, n)
        """
        A = _ensure_square(A)
        eigenvalues, eigenvectors = la.eig(A)

        if sort == "real_descending":
            idx = np.argsort(-eigenvalues.real)
        elif sort == "real_ascending":
            idx = np.argsort(eigenvalues.real)
        elif sort == "abs_descending":
            idx = np.argsort(-np.abs(eigenvalues))
        elif sort == "abs_ascending":
            idx = np.argsort(np.abs(eigenvalues))
        else:
            raise ValueError(f"Unknown sort order: {sort!r}")

        return eigenvalues[idx], eigenvectors[:, idx]

    # ------------------------------------------------------------------
    #  Truncated eigendecomposition
    # ------------------------------------------------------------------

    def truncated_eig(
        self,
        A: AnyArray,
        k: int,
        which: Literal["largest", "smallest"] = "largest",
    ) -> Tuple[ComplexArray, ComplexArray]:
        r"""Return the *k* largest or smallest eigenvalues (by magnitude).

        Parameters
        ----------
        A : array_like, shape (n, n)
        k : int
            Number of eigenvalues to keep.
        which : ``'largest'`` | ``'smallest'``

        Returns
        -------
        eigenvalues : ndarray, shape (k,)
        eigenvectors : ndarray, shape (n, k)
        """
        A = _ensure_square(A)
        n = A.shape[0]
        if k < 1 or k > n:
            raise ValueError(f"k={k} out of range [1, {n}]")

        sort = "abs_descending" if which == "largest" else "abs_ascending"
        vals, vecs = self.eig_sorted(A, sort=sort)
        return vals[:k], vecs[:, :k]

    # ------------------------------------------------------------------
    #  Scalar spectral quantities
    # ------------------------------------------------------------------

    def spectral_gap(self, A: AnyArray) -> float:
        r"""Gap between the two largest eigenvalues (by real part).

        .. math::
            \Delta = \operatorname{Re}(\lambda_1) -
                     \operatorname{Re}(\lambda_2)

        where λ₁ ≥ λ₂ ≥ … in real-part order.
        """
        A = _ensure_square(A)
        eigenvalues = la.eigvals(A)
        reals = np.sort(eigenvalues.real)[::-1]
        if len(reals) < 2:
            return float("inf")
        return float(reals[0] - reals[1])

    def spectral_abscissa(self, A: AnyArray) -> float:
        r"""Spectral abscissa: α(A) = max_i Re(λ_i).

        Controls asymptotic growth of exp(tA): ‖exp(tA)‖ ~ e^{α t}.
        """
        A = _ensure_square(A)
        eigenvalues = la.eigvals(A)
        return float(np.max(eigenvalues.real))

    def spectral_radius(self, A: AnyArray) -> float:
        r"""Spectral radius: ρ(A) = max_i |λ_i|."""
        A = _ensure_square(A)
        eigenvalues = la.eigvals(A)
        return float(np.max(np.abs(eigenvalues)))

    def condition_number_spectral(self, A: AnyArray) -> float:
        r"""Eigenvalue condition number: κ(A) = |λ_max| / |λ_min|.

        Uses eigenvalues sorted by absolute value.  Returns inf when the
        smallest eigenvalue is zero.
        """
        A = _ensure_square(A)
        eigenvalues = la.eigvals(A)
        abs_eigs = np.abs(eigenvalues)
        lam_min = float(np.min(abs_eigs))
        if lam_min == 0.0:
            return float("inf")
        return float(np.max(abs_eigs) / lam_min)

    def eigenvalue_condition_numbers(self, A: AnyArray) -> RealArray:
        r"""Per-eigenvalue condition numbers.

        For each eigenvalue λ_i the condition number is

        .. math::
            \kappa_i = \frac{1}{|v_i^H w_i|}

        where v_i and w_i are the corresponding left and right
        eigenvectors normalised to unit 2-norm.

        Returns
        -------
        kappa : ndarray, shape (n,)
            Condition numbers ordered consistently with the eigenvalues
            from ``scipy.linalg.eig``.
        """
        A = _ensure_square(A)
        eigenvalues, right_vecs = la.eig(A)
        _, left_vecs = la.eig(A.conj().T)

        # Match left eigenvectors to right ones via eigenvalue proximity
        n = A.shape[0]
        left_eigs = la.eigvals(A.conj().T).conj()

        # Pair left/right eigenvectors by matching eigenvalues
        kappa = np.empty(n, dtype=np.float64)
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            # Find the closest left eigenvalue to eigenvalues[i]
            dists = np.abs(left_eigs - eigenvalues[i])
            dists[used] = np.inf
            j = int(np.argmin(dists))
            used[j] = True

            v = right_vecs[:, i]
            w = left_vecs[:, j]
            dot = np.abs(np.dot(w.conj(), v))
            kappa[i] = 1.0 / dot if dot > 0.0 else float("inf")

        return kappa


# ======================================================================
#  RandomizedSVD
# ======================================================================

@dataclass
class RandomizedSVD:
    """Randomized singular value decomposition.

    Implements the Halko–Martinsson–Tropp (2011) algorithm for computing
    a rank-*k* approximation A ≈ U_k Σ_k V_k^T using randomised range
    finding with optional power iteration for improved accuracy.

    Parameters
    ----------
    random_state : int | np.random.Generator | None
        Seed or generator for reproducibility.
    """

    random_state: Optional[Union[int, np.random.Generator]] = None

    def __post_init__(self) -> None:
        if isinstance(self.random_state, np.random.Generator):
            self._rng = self.random_state
        else:
            self._rng = np.random.default_rng(self.random_state)

    # ------------------------------------------------------------------
    #  Randomised range finder (private)
    # ------------------------------------------------------------------

    def _random_projection(
        self,
        A: AnyArray,
        k: int,
        n_oversampling: int,
        n_power_iter: int,
    ) -> AnyArray:
        r"""Compute an approximate orthonormal basis Q for range(A).

        Algorithm 4.3 from Halko, Martinsson & Tropp (2011):

        1. Draw Gaussian random matrix Ω ∈ ℝ^{n×(k+p)}.
        2. Form Y = (A A^T)^q A Ω  via *q* power iterations.
        3. Orthonormalise Y → Q.

        Parameters
        ----------
        A : ndarray, shape (m, n)
        k : int
            Target rank.
        n_oversampling : int
            Extra columns *p* for oversampling.
        n_power_iter : int
            Number of power iterations *q*.

        Returns
        -------
        Q : ndarray, shape (m, k + n_oversampling)
        """
        m, n = A.shape
        l = k + n_oversampling  # total sketch width

        # Step 1 – random Gaussian sketch
        Omega = self._rng.standard_normal((n, l))
        if np.iscomplexobj(A):
            Omega = Omega + 1j * self._rng.standard_normal((n, l))

        Y = A @ Omega

        # Step 2 – power iterations for spectral decay enhancement
        for _ in range(n_power_iter):
            Y, _ = la.qr(Y, mode="economic")
            Z = A.conj().T @ Y
            Z, _ = la.qr(Z, mode="economic")
            Y = A @ Z

        # Step 3 – orthonormalise
        Q, _ = la.qr(Y, mode="economic")
        return Q

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        A: AnyArray,
        k: int,
        n_oversampling: int = 10,
        n_power_iter: int = 2,
    ) -> Tuple[AnyArray, RealArray, AnyArray]:
        r"""Randomised SVD: A ≈ U_k Σ_k V_k^T.

        Parameters
        ----------
        A : array_like, shape (m, n)
        k : int
            Target rank.
        n_oversampling : int
            Oversampling parameter (default 10).
        n_power_iter : int
            Power-iteration steps (default 2).

        Returns
        -------
        U : ndarray, shape (m, k)
        sigma : ndarray, shape (k,)
        Vt : ndarray, shape (k, n)
        """
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"Expected 2-D array, got ndim={A.ndim}")
        m, n = A.shape
        k = min(k, min(m, n))

        Q = self._random_projection(A, k, n_oversampling, n_power_iter)

        # Project A into the low-dimensional subspace
        B = Q.conj().T @ A  # (l, n)

        # Thin SVD of the small matrix B
        U_hat, sigma, Vt = la.svd(B, full_matrices=False)

        # Lift left singular vectors back to original space
        U = Q @ U_hat

        return U[:, :k], sigma[:k], Vt[:k, :]

    def truncated_svd(
        self, A: AnyArray, k: int
    ) -> Tuple[AnyArray, RealArray, AnyArray]:
        r"""Deterministic truncated SVD via full decomposition.

        More accurate than the randomised variant but O(mn min(m,n)).

        Returns
        -------
        U : ndarray, shape (m, k)
        sigma : ndarray, shape (k,)
        Vt : ndarray, shape (k, n)
        """
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"Expected 2-D array, got ndim={A.ndim}")
        m, n = A.shape
        k = min(k, min(m, n))

        U, sigma, Vt = la.svd(A, full_matrices=False)
        return U[:, :k], sigma[:k], Vt[:k, :]

    def optimal_rank(
        self, A: AnyArray, threshold: float = 0.99
    ) -> int:
        r"""Find the smallest rank *k* capturing ≥ *threshold* of total energy.

        Energy is measured as the fraction of squared Frobenius norm:

        .. math::
            \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}
            \geq \text{threshold}

        Parameters
        ----------
        A : array_like, shape (m, n)
        threshold : float
            Fraction of energy to retain (default 0.99).

        Returns
        -------
        k : int
        """
        A = np.asarray(A)
        sigma = la.svdvals(A)
        total_energy = float(np.sum(sigma ** 2))
        if total_energy == 0.0:
            return 0

        cumulative = np.cumsum(sigma ** 2) / total_energy
        indices = np.where(cumulative >= threshold)[0]
        if len(indices) == 0:
            return len(sigma)
        return int(indices[0]) + 1

    def low_rank_approximation(
        self, A: AnyArray, k: int
    ) -> AnyArray:
        r"""Best rank-*k* approximation in Frobenius / operator norm.

        By the Eckart–Young–Mirsky theorem the optimal approximation is
        A_k = U_k Σ_k V_k^T obtained from the truncated SVD.
        """
        U, sigma, Vt = self.truncated_svd(A, k)
        return U @ np.diag(sigma) @ Vt


# ======================================================================
#  MatrixFunction
# ======================================================================

@dataclass
class MatrixFunction:
    r"""Evaluate standard matrix functions with error checking.

    Wraps ``scipy.linalg`` routines and adds:

    * Input validation (square, finite entries).
    * Post-hoc consistency checks (e.g. expm(logm(A)) ≈ A).
    * A Schur-based generic ``funm`` with Parlett recurrence.

    Parameters
    ----------
    method : ``'schur'`` | ``'diag'``
        Strategy for ``funm``.  ``'schur'`` (default) uses the Schur
        decomposition; ``'diag'`` assumes A is diagonalisable.
    """

    method: Literal["schur", "diag"] = "schur"

    # ------------------------------------------------------------------
    #  Matrix exponential
    # ------------------------------------------------------------------

    def expm(self, A: AnyArray) -> AnyArray:
        r"""Matrix exponential via scaling-and-squaring Padé approximation.

        .. math::
            e^A = \sum_{k=0}^{\infty} \frac{A^k}{k!}

        Delegates to ``scipy.linalg.expm`` with additional input checks.
        """
        A = _ensure_square(A)
        if not np.all(np.isfinite(A)):
            raise ValueError("Matrix contains non-finite entries.")
        result = la.expm(A)
        return result

    # ------------------------------------------------------------------
    #  Matrix logarithm
    # ------------------------------------------------------------------

    def logm(self, A: AnyArray) -> AnyArray:
        r"""Principal matrix logarithm via Schur decomposition.

        .. math::
            \log A \; \text{s.t.} \; e^{\log A} = A

        The principal logarithm is defined when A has no eigenvalues on
        ℝ⁻ ∪ {0}.  A warning is issued when eigenvalues are detected
        near the branch cut.

        Returns
        -------
        logA : ndarray
        """
        A = _ensure_square(A)
        eigenvalues = la.eigvals(A)

        # Check for eigenvalues on or near the negative real axis / zero
        for lam in eigenvalues:
            if np.abs(lam) < 1e-14:
                warnings.warn(
                    "Matrix has near-zero eigenvalue; logm may be "
                    "inaccurate or undefined.",
                    stacklevel=2,
                )
                break
            if lam.real < 0 and np.abs(lam.imag) < 1e-14:
                warnings.warn(
                    "Eigenvalue near negative real axis detected; "
                    "principal logarithm may cross a branch cut.",
                    stacklevel=2,
                )
                break

        result = la.logm(A)
        return result

    # ------------------------------------------------------------------
    #  Matrix square root
    # ------------------------------------------------------------------

    def sqrtm(self, A: AnyArray, validate: bool = True) -> AnyArray:
        r"""Principal matrix square root via Schur decomposition.

        .. math::
            X = A^{1/2} \;\;\text{s.t.}\;\; X^2 = A

        Parameters
        ----------
        A : array_like, shape (n, n)
        validate : bool
            If True, checks that X² ≈ A and warns on large residual.

        Returns
        -------
        sqrtA : ndarray
        """
        A = _ensure_square(A)
        result, info = la.sqrtm(A, disp=False)

        if validate:
            residual = np.linalg.norm(result @ result - A, "fro")
            norm_A = np.linalg.norm(A, "fro")
            rel_err = residual / norm_A if norm_A > 0 else residual
            if rel_err > 1e-6:
                warnings.warn(
                    f"sqrtm validation: ‖X² - A‖/‖A‖ = {rel_err:.2e} "
                    f"exceeds 1e-6.",
                    stacklevel=2,
                )

        return result

    # ------------------------------------------------------------------
    #  Generic matrix function (Schur / diag)
    # ------------------------------------------------------------------

    def funm(self, A: AnyArray, f: Callable) -> AnyArray:
        r"""Evaluate a general scalar function *f* on the matrix *A*.

        When ``method='schur'`` (default) the computation proceeds via
        the Schur decomposition A = Q T Q^H followed by Parlett
        recurrence on T:

        .. math::
            f(A) = Q \, f(T) \, Q^H

        Parameters
        ----------
        A : array_like, shape (n, n)
        f : callable
            Scalar function applied element-wise to eigenvalues /
            Schur diagonal.

        Returns
        -------
        fA : ndarray, shape (n, n)
        """
        A = _ensure_square(A)

        if self.method == "schur":
            T, Q = la.schur(A, output="complex")
            F = self._schur_funm(T, f)
            return Q @ F @ Q.conj().T

        # Diagonalisation path
        eigenvalues, V = la.eig(A)
        try:
            V_inv = la.inv(V)
        except la.LinAlgError:
            raise la.LinAlgError(
                "Matrix is not diagonalisable; use method='schur'."
            )
        f_lambda = np.diag(np.array([f(lam) for lam in eigenvalues]))
        return V @ f_lambda @ V_inv

    # ------------------------------------------------------------------
    #  exp(tA) @ B  without forming exp(tA)
    # ------------------------------------------------------------------

    def expm_multiply(
        self, A: AnyArray, B: AnyArray, t: float = 1.0
    ) -> AnyArray:
        r"""Compute exp(tA) B without forming the full matrix exponential.

        Uses ``scipy.sparse.linalg.expm_multiply`` when available,
        otherwise falls back to explicit expm.

        Parameters
        ----------
        A : array_like, shape (n, n)
        B : array_like, shape (n,) or (n, p)
        t : float
            Scalar time parameter (default 1.0).

        Returns
        -------
        result : ndarray, shape matching B
        """
        A = _ensure_square(A)
        B = np.asarray(B)
        try:
            from scipy.sparse.linalg import expm_multiply as _expm_multiply
            return np.asarray(_expm_multiply(t * A, B))
        except ImportError:
            return la.expm(t * A) @ B

    # ------------------------------------------------------------------
    #  Parlett recurrence on upper-triangular Schur factor (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _schur_funm(T: ComplexArray, f: Callable) -> ComplexArray:
        r"""Apply scalar function *f* to upper-triangular *T* via Parlett.

        For an upper-triangular T with distinct diagonal entries the
        Parlett recurrence computes F = f(T) via:

        .. math::
            F_{ii} = f(T_{ii}), \quad
            F_{ij} = \frac{
                T_{ij}(F_{ii} - F_{jj})
                + \sum_{k=i+1}^{j-1}(F_{ik}T_{kj} - T_{ik}F_{kj})
            }{T_{ii} - T_{jj}}

        When diagonal entries are too close (|T_{ii} - T_{jj}| < δ) a
        divided-difference approximation is used to avoid division by
        near-zero.

        Parameters
        ----------
        T : ndarray, shape (n, n)
            Upper-triangular (complex Schur) factor.
        f : callable
            Scalar function.

        Returns
        -------
        F : ndarray, shape (n, n)
        """
        n = T.shape[0]
        F = np.zeros_like(T)

        # Diagonal entries
        for i in range(n):
            F[i, i] = f(T[i, i])

        # Super-diagonals via Parlett recurrence
        delta = 1e-14 * (np.linalg.norm(T, "fro") + 1.0)
        for diag_idx in range(1, n):
            for i in range(n - diag_idx):
                j = i + diag_idx
                denom = T[i, i] - T[j, j]

                s = T[i, j] * (F[i, i] - F[j, j])
                for k in range(i + 1, j):
                    s += F[i, k] * T[k, j] - T[i, k] * F[k, j]

                if np.abs(denom) > delta:
                    F[i, j] = s / denom
                else:
                    # Divided-difference fallback for clustered eigenvalues
                    h = max(np.abs(T[i, i]), np.abs(T[j, j])) * 1e-8
                    h = max(h, 1e-15)
                    mid = 0.5 * (T[i, i] + T[j, j])
                    dd = (f(mid + h) - f(mid - h)) / (2.0 * h)
                    F[i, j] = T[i, j] * dd

        return F

    # ------------------------------------------------------------------
    #  Validation helper (private)
    # ------------------------------------------------------------------

    def _validate_result(
        self,
        A: AnyArray,
        f_A: AnyArray,
        f_name: str,
    ) -> float:
        r"""Validate a matrix function result.

        For inverse-pair functions (exp/log, sqr/sqrt) checks that
        round-tripping gives back the original matrix.

        Parameters
        ----------
        A : ndarray
            Original matrix.
        f_A : ndarray
            Computed f(A).
        f_name : str
            Human-readable name of the function (used in warnings).

        Returns
        -------
        rel_error : float
            ‖A_reconstructed - A‖_F / ‖A‖_F  (inf if check is not
            applicable).
        """
        norm_A = np.linalg.norm(A, "fro")
        if norm_A == 0.0:
            return 0.0

        inverse_map = {
            "expm": ("logm", la.logm),
            "logm": ("expm", la.expm),
            "sqrtm": ("square", lambda X: X @ X),
        }

        if f_name not in inverse_map:
            return float("inf")

        inv_name, inv_func = inverse_map[f_name]
        try:
            reconstructed = inv_func(f_A)
        except Exception:
            warnings.warn(
                f"Could not validate {f_name} via {inv_name}.",
                stacklevel=2,
            )
            return float("inf")

        rel_error = float(
            np.linalg.norm(reconstructed - A, "fro") / norm_A
        )
        if rel_error > 1e-6:
            warnings.warn(
                f"{f_name} validation ({inv_name} round-trip): "
                f"relative error = {rel_error:.2e}.",
                stacklevel=2,
            )
        return rel_error


# ======================================================================
#  PseudoSpectrum
# ======================================================================

@dataclass
class PseudoSpectrum:
    r"""Computation of ε-pseudospectra.

    The ε-pseudospectrum of A ∈ ℂ^{n×n} is

    .. math::
        \sigma_\varepsilon(A) = \bigl\{z \in \mathbb{C} :
        \|(zI - A)^{-1}\| \geq 1/\varepsilon \bigr\}.

    Equivalently it is the set of eigenvalues of all matrices
    A + E with ‖E‖ ≤ ε.

    Parameters
    ----------
    grid_size : int
        Default number of grid points per axis when computing on a
        regular complex grid.
    """

    grid_size: int = 100

    # ------------------------------------------------------------------
    #  Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        A: AnyArray,
        epsilon_values: Sequence[float],
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[RealArray, RealArray, RealArray]:
        r"""Compute the resolvent norm on a grid and return contour data.

        Parameters
        ----------
        A : array_like, shape (n, n)
        epsilon_values : sequence of float
            ε levels at which to extract pseudospectral boundaries.
        region : (x_min, x_max, y_min, y_max) or None
            Complex-plane region.  When None a bounding box is chosen
            automatically from the eigenvalues of A.

        Returns
        -------
        X : ndarray, shape (grid_size, grid_size)
            Real parts of grid points.
        Y : ndarray, shape (grid_size, grid_size)
            Imaginary parts of grid points.
        sigma_min_grid : ndarray, shape (grid_size, grid_size)
            Minimum singular value σ_min(zI - A) at each grid point.
            The ε-pseudospectrum boundary is the contour where
            σ_min = ε.
        """
        A = _ensure_square(A)

        if region is None:
            eigenvalues = la.eigvals(A)
            margin = max(
                0.5,
                0.2 * (np.max(np.abs(eigenvalues)) + 1.0),
            )
            x_min = float(np.min(eigenvalues.real)) - margin
            x_max = float(np.max(eigenvalues.real)) + margin
            y_min = float(np.min(eigenvalues.imag)) - margin
            y_max = float(np.max(eigenvalues.imag)) + margin
        else:
            x_min, x_max, y_min, y_max = region

        X, Y, sigma_min_grid = self.compute_on_grid(
            A,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            n_grid=self.grid_size,
        )
        return X, Y, sigma_min_grid

    # ------------------------------------------------------------------
    #  Resolvent norm on a regular grid
    # ------------------------------------------------------------------

    def compute_on_grid(
        self,
        A: AnyArray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        n_grid: Optional[int] = None,
    ) -> Tuple[RealArray, RealArray, RealArray]:
        r"""Compute σ_min(zI - A) on a regular grid in the complex plane.

        The resolvent norm is ‖(zI - A)^{-1}‖₂ = 1 / σ_min(zI - A).

        Parameters
        ----------
        A : ndarray, shape (n, n)
        x_range : (float, float)
            Real-axis bounds.
        y_range : (float, float)
            Imaginary-axis bounds.
        n_grid : int or None
            Number of grid points per axis (defaults to ``self.grid_size``).

        Returns
        -------
        X, Y : ndarray, shape (n_grid, n_grid)
        sigma_min_grid : ndarray, shape (n_grid, n_grid)
        """
        A = _ensure_square(A)
        n = A.shape[0]
        if n_grid is None:
            n_grid = self.grid_size

        x_vals = np.linspace(x_range[0], x_range[1], n_grid)
        y_vals = np.linspace(y_range[0], y_range[1], n_grid)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Use Schur decomposition for efficiency: σ_min(zI - A) =
        # σ_min(zI - T) because unitary similarity preserves singular
        # values.
        T, _ = la.schur(A, output="complex")

        sigma_min_grid = np.empty_like(X)
        I_n = np.eye(n, dtype=T.dtype)

        for i in range(n_grid):
            for j in range(n_grid):
                z = complex(X[i, j], Y[i, j])
                M = z * I_n - T
                svals = la.svdvals(M)
                sigma_min_grid[i, j] = svals[-1]

        return X, Y, sigma_min_grid

    # ------------------------------------------------------------------
    #  Pseudospectral abscissa
    # ------------------------------------------------------------------

    def pseudospectral_abscissa(
        self, A: AnyArray, epsilon: float
    ) -> float:
        r"""Pseudospectral abscissa: α_ε(A) = max Re(z) for z ∈ σ_ε(A).

        Approximated by grid search; for high accuracy increase
        ``grid_size``.

        Parameters
        ----------
        A : array_like, shape (n, n)
        epsilon : float
            Perturbation level ε > 0.

        Returns
        -------
        alpha_eps : float
        """
        A = _ensure_square(A)
        eigenvalues = la.eigvals(A)
        margin = epsilon + 0.5
        x_min = float(np.min(eigenvalues.real)) - margin
        x_max = float(np.max(eigenvalues.real)) + margin
        y_min = float(np.min(eigenvalues.imag)) - margin
        y_max = float(np.max(eigenvalues.imag)) + margin

        X, Y, sigma_min = self.compute_on_grid(
            A,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )

        # Points inside the ε-pseudospectrum: σ_min(zI-A) ≤ ε
        mask = sigma_min <= epsilon
        if not np.any(mask):
            # Fall back to spectral abscissa
            return float(np.max(eigenvalues.real))

        return float(np.max(X[mask]))

    # ------------------------------------------------------------------
    #  Kreiss constant
    # ------------------------------------------------------------------

    def kreiss_constant(self, A: AnyArray) -> float:
        r"""Kreiss constant of *A*.

        .. math::
            K(A) = \sup_{|z| > 1}
            \|(zI - A)^{-1}\| \, (|z| - 1)

        The Kreiss matrix theorem bounds the power-iteration growth:
        K(A) ≤ sup_n ‖A^n‖ ≤ e n K(A).

        Approximated by sampling on radial lines outside the unit disc.
        """
        A = _ensure_square(A)
        n = A.shape[0]
        I_n = np.eye(n, dtype=complex)

        n_angles = max(4 * n, 64)
        n_radii = max(2 * n, 32)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii = np.linspace(1.01, 1.0 + 2.0 * SpectralDecomposition().spectral_radius(A), n_radii)

        kreiss = 0.0
        for r in radii:
            for theta in angles:
                z = r * np.exp(1j * theta)
                try:
                    resolvent = la.inv(z * I_n - A)
                    norm_res = np.linalg.norm(resolvent, 2)
                    val = norm_res * (r - 1.0)
                    kreiss = max(kreiss, val)
                except la.LinAlgError:
                    # zI - A is singular ⇒ z is an eigenvalue
                    kreiss = float("inf")
                    return kreiss

        return float(kreiss)
