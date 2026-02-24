"""
Eigenvalue tracking along ODE trajectories.

Tracks eigenvalues λ_i(γ) of parameter-dependent operators L(γ) as the
parameter γ varies along a path or ODE trajectory.  Handles eigenvalue
sorting via the Hungarian algorithm, zero-crossing detection with bisection
refinement, avoided-crossing detection, spectral sensitivity analysis, and
eigenvalue continuation.

Mathematical background
-----------------------
Given a one-parameter family of matrices L(γ) ∈ ℂ^{n×n}, the eigenvalues
λ_i(γ) trace smooth curves in the complex plane (away from multiplicities).
The sensitivity of a simple eigenvalue is

    dλ/dγ  =  vᴴ (∂L/∂γ) w  /  (vᴴ w)

where v, w are the left and right eigenvectors associated with λ.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d


# ======================================================================
#  Data containers
# ======================================================================

@dataclass
class SpectralPath:
    """Container for eigenvalue paths along a parameter sweep.

    Attributes
    ----------
    parameter_values : ndarray, shape (n_params,)
        The parameter values γ_0, γ_1, … at which eigenvalues were computed.
    eigenvalues : ndarray, shape (n_params, n_eigs)
        Tracked (and sorted) eigenvalues.  Entry ``eigenvalues[i, j]`` is the
        j-th eigenvalue at parameter value ``parameter_values[i]``.
    eigenvectors : list of ndarray, optional
        ``eigenvectors[i]`` is the matrix of right eigenvectors at step *i*.
        Each matrix has shape ``(n, n_eigs)`` where columns correspond to the
        eigenvalues in ``eigenvalues[i]``.
    spectral_gaps : ndarray, shape (n_params,)
        Gap between the two largest eigenvalues (by real part) at each step.
    spectral_abscissa : ndarray, shape (n_params,)
        Maximum real part of all eigenvalues at each step, i.e. α(L(γ)).
    multiplicities : list of list of int
        ``multiplicities[i]`` lists the multiplicity of each eigenvalue at
        step *i*.
    """

    parameter_values: NDArray[np.floating]
    eigenvalues: NDArray[np.complexfloating]
    eigenvectors: Optional[List[NDArray[np.complexfloating]]] = None
    spectral_gaps: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    spectral_abscissa: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    multiplicities: List[List[int]] = field(default_factory=list)


# ======================================================================

@dataclass
class ZeroCrossing:
    """Record of an eigenvalue real-part zero crossing.

    A zero crossing occurs when Re(λ_j(γ)) changes sign, which typically
    signals a stability transition (bifurcation).

    Attributes
    ----------
    parameter_value : float
        Nearest sampled parameter value to the crossing.
    eigenvalue_index : int
        Index *j* of the eigenvalue that crosses zero.
    crossing_type : str
        Either ``'positive_to_negative'`` or ``'negative_to_positive'``.
    eigenvalue_before : float
        Re(λ_j) just before the crossing.
    eigenvalue_after : float
        Re(λ_j) just after the crossing.
    eigenvector_at_crossing : ndarray or None
        Right eigenvector at the refined crossing location (if available).
    interpolated_parameter : float
        Parameter value obtained by bisection refinement.
    """

    parameter_value: float
    eigenvalue_index: int
    crossing_type: str
    eigenvalue_before: float
    eigenvalue_after: float
    eigenvector_at_crossing: Optional[NDArray[np.complexfloating]] = None
    interpolated_parameter: float = 0.0


# ======================================================================
#  Avoided crossing record
# ======================================================================

@dataclass
class AvoidedCrossing:
    """Record of an avoided crossing between two eigenvalue branches.

    At an avoided crossing the two eigenvalue curves approach each other
    closely (gap ≤ threshold) but do not actually cross.

    Attributes
    ----------
    parameter_value : float
        Parameter value at which the minimum gap occurs.
    eigenvalue_indices : Tuple[int, int]
        Pair of eigenvalue branch indices involved.
    minimum_gap : float
        Smallest distance between the two eigenvalues in the neighbourhood.
    """

    parameter_value: float
    eigenvalue_indices: Tuple[int, int]
    minimum_gap: float


# ======================================================================
#  Main tracker
# ======================================================================

class EigenvalueTracker:
    """Track eigenvalues of L(γ) as γ varies along a path.

    Parameters
    ----------
    n_eigenvalues : int or None
        Number of eigenvalues to track.  ``None`` means *all*.
    track_vectors : bool
        Whether to store eigenvectors along the path.
    sort_method : str
        Sorting criterion for the initial ordering of eigenvalues.
        One of ``'real_part'``, ``'magnitude'``, ``'imag_part'``.
    crossing_tol : float
        Tolerance used in bisection refinement of zero crossings.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_eigenvalues: Optional[int] = None,
        track_vectors: bool = True,
        sort_method: str = "real_part",
        crossing_tol: float = 1e-8,
    ) -> None:
        if sort_method not in ("real_part", "magnitude", "imag_part"):
            raise ValueError(
                f"Unknown sort_method '{sort_method}'; expected one of "
                "'real_part', 'magnitude', 'imag_part'."
            )
        self.n_eigenvalues = n_eigenvalues
        self.track_vectors = track_vectors
        self.sort_method = sort_method
        self.crossing_tol = crossing_tol

    # ==================================================================
    #  Public interface
    # ==================================================================

    def track(
        self,
        operator_fn: Callable[[float], NDArray],
        parameter_values: NDArray[np.floating],
    ) -> SpectralPath:
        """Track eigenvalues of L(γ) over a sequence of parameter values.

        Parameters
        ----------
        operator_fn : callable
            ``operator_fn(gamma)`` returns the matrix L(γ) as an ndarray.
        parameter_values : ndarray, shape (n_params,)
            Monotonic sequence of parameter values to sweep.

        Returns
        -------
        SpectralPath
            Collected eigenvalue data along the sweep.
        """
        parameter_values = np.asarray(parameter_values, dtype=float)
        n_params = len(parameter_values)

        # -- first evaluation to determine dimensions -----------------
        L0 = np.atleast_2d(operator_fn(parameter_values[0]))
        eigs0, vecs0 = self._compute_eigenvalues(L0, k=self.n_eigenvalues)
        n_eigs = len(eigs0)

        eigenvalue_array = np.empty((n_params, n_eigs), dtype=complex)
        eigenvalue_array[0] = eigs0

        eigenvector_list: Optional[List[NDArray]] = [] if self.track_vectors else None
        if self.track_vectors:
            eigenvector_list.append(vecs0)

        prev_eigs = eigs0
        prev_vecs = vecs0

        # -- sweep over remaining parameter values --------------------
        for i in range(1, n_params):
            L = np.atleast_2d(operator_fn(parameter_values[i]))
            eigs, vecs = self._compute_eigenvalues(L, k=self.n_eigenvalues)
            eigs, vecs = self._sort_eigenvalues(eigs, vecs, prev_eigs, prev_vecs)

            eigenvalue_array[i] = eigs
            if self.track_vectors:
                eigenvector_list.append(vecs)

            prev_eigs = eigs
            prev_vecs = vecs

        # -- derived quantities ----------------------------------------
        gaps = np.array([self.spectral_gap(eigenvalue_array[i]) for i in range(n_params)])
        abscissa = np.array([self.spectral_abscissa_val(eigenvalue_array[i]) for i in range(n_params)])
        mults = [self.multiplicity_detection(eigenvalue_array[i]) for i in range(n_params)]

        return SpectralPath(
            parameter_values=parameter_values,
            eigenvalues=eigenvalue_array,
            eigenvectors=eigenvector_list,
            spectral_gaps=gaps,
            spectral_abscissa=abscissa,
            multiplicities=mults,
        )

    # ------------------------------------------------------------------

    def track_along_trajectory(
        self,
        trajectory: NDArray[np.floating],
        operator_fn: Callable[[NDArray], NDArray],
    ) -> SpectralPath:
        """Track eigenvalues along an ODE trajectory.

        Parameters
        ----------
        trajectory : ndarray, shape (n_steps, state_dim)
            State vectors along an ODE trajectory.  Each row is treated as
            the parameter value for ``operator_fn``.
        operator_fn : callable
            ``operator_fn(state)`` returns the linearised operator (matrix)
            at the given state.  *state* is a 1-D array of length
            ``state_dim``.

        Returns
        -------
        SpectralPath
            Eigenvalue data with ``parameter_values`` set to the step
            indices ``[0, 1, …, n_steps - 1]``.
        """
        trajectory = np.atleast_2d(trajectory)
        n_steps = trajectory.shape[0]
        indices = np.arange(n_steps, dtype=float)

        def _op(idx: float) -> NDArray:
            i = int(round(idx))
            i = max(0, min(i, n_steps - 1))
            return np.atleast_2d(operator_fn(trajectory[i]))

        return self.track(_op, indices)

    # ==================================================================
    #  Eigendecomposition helpers
    # ==================================================================

    def _compute_eigenvalues(
        self,
        L: NDArray,
        k: Optional[int] = None,
    ) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
        """Compute eigenvalues (and optionally eigenvectors) of *L*.

        Parameters
        ----------
        L : ndarray, shape (n, n)
            Square matrix.
        k : int or None
            Number of eigenvalues to keep (largest by the current
            ``sort_method``).  ``None`` keeps all.

        Returns
        -------
        eigenvalues : ndarray, shape (m,)
        eigenvectors : ndarray, shape (n, m)
            Columns are the right eigenvectors.
        """
        L = np.asarray(L, dtype=complex)
        n = L.shape[0]

        eigenvalues, eigenvectors = np.linalg.eig(L)

        # Initial sort according to the chosen criterion
        order = self._sort_key(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        if k is not None and k < n:
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]

        return eigenvalues, eigenvectors

    def _sort_key(self, eigenvalues: NDArray) -> NDArray:
        """Return index array that sorts eigenvalues in descending order."""
        if self.sort_method == "real_part":
            return np.argsort(-eigenvalues.real)
        elif self.sort_method == "magnitude":
            return np.argsort(-np.abs(eigenvalues))
        elif self.sort_method == "imag_part":
            return np.argsort(-eigenvalues.imag)
        else:
            raise ValueError(f"Unknown sort_method '{self.sort_method}'.")

    # ==================================================================
    #  Eigenvalue matching (Hungarian algorithm)
    # ==================================================================

    def _sort_eigenvalues(
        self,
        eigenvalues: NDArray[np.complexfloating],
        eigenvectors: NDArray[np.complexfloating],
        prev_eigenvalues: NDArray[np.complexfloating],
        prev_eigenvectors: NDArray[np.complexfloating],
    ) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
        """Match current eigenvalues to previous ones to maintain continuity.

        Uses the Hungarian algorithm on a cost matrix that combines
        eigenvalue distance and eigenvector overlap:

            C_{ij} = |λ_i^{curr} - λ_j^{prev}|
                     + α (1 - |⟨v_i^{curr}, v_j^{prev}⟩|)

        where α is a weight that balances the two contributions.

        Parameters
        ----------
        eigenvalues, eigenvectors : current step data.
        prev_eigenvalues, prev_eigenvectors : previous step data.

        Returns
        -------
        sorted_eigenvalues : ndarray
        sorted_eigenvectors : ndarray
        """
        n_curr = len(eigenvalues)
        n_prev = len(prev_eigenvalues)
        n = min(n_curr, n_prev)

        # -- build cost matrix -----------------------------------------
        eig_dist = np.abs(
            eigenvalues[:n, np.newaxis] - prev_eigenvalues[np.newaxis, :n]
        )

        # Eigenvector overlap term (only if vectors available)
        alpha = 0.0
        vec_cost = np.zeros((n, n))
        if eigenvectors is not None and prev_eigenvectors is not None:
            # Normalise columns for overlap computation
            curr_norms = np.linalg.norm(eigenvectors[:, :n], axis=0, keepdims=True)
            prev_norms = np.linalg.norm(prev_eigenvectors[:, :n], axis=0, keepdims=True)
            curr_normed = eigenvectors[:, :n] / np.where(curr_norms > 0, curr_norms, 1.0)
            prev_normed = prev_eigenvectors[:, :n] / np.where(prev_norms > 0, prev_norms, 1.0)

            overlap = np.abs(curr_normed.conj().T @ prev_normed)  # (n, n)
            vec_cost = 1.0 - overlap

            # Scale α so that eigenvector term is comparable to eigenvalue term
            eig_scale = np.max(eig_dist) if np.max(eig_dist) > 0 else 1.0
            alpha = eig_scale

        cost = eig_dist + alpha * vec_cost

        # -- solve assignment ------------------------------------------
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build permutation for full arrays
        perm = np.arange(n_curr)
        # col_ind tells us: new position row_ind[k] should get the value
        # that was at col_ind[k] in the *previous* ordering – but here
        # row_ind indexes *current* eigenvalues and col_ind indexes
        # *previous* eigenvalues.  We want to reorder current eigenvalues
        # so that index j of the result best matches index j of prev.
        reorder = np.empty(n, dtype=int)
        reorder[col_ind] = row_ind
        perm[:n] = reorder

        sorted_eigenvalues = eigenvalues[perm]
        sorted_eigenvectors = eigenvectors[:, perm] if eigenvectors is not None else eigenvectors

        return sorted_eigenvalues, sorted_eigenvectors

    # ==================================================================
    #  Zero-crossing detection
    # ==================================================================

    def _detect_zero_crossings(
        self,
        parameter_values: NDArray[np.floating],
        eigenvalue_paths: NDArray[np.complexfloating],
    ) -> List[ZeroCrossing]:
        """Find intervals where Re(λ_j(γ)) changes sign.

        Parameters
        ----------
        parameter_values : ndarray, shape (n_params,)
        eigenvalue_paths : ndarray, shape (n_params, n_eigs)

        Returns
        -------
        crossings : list of ZeroCrossing
        """
        n_params, n_eigs = eigenvalue_paths.shape
        crossings: List[ZeroCrossing] = []

        real_parts = eigenvalue_paths.real  # (n_params, n_eigs)

        for j in range(n_eigs):
            for i in range(n_params - 1):
                r_before = real_parts[i, j]
                r_after = real_parts[i + 1, j]

                if r_before * r_after < 0.0:
                    # Sign change detected
                    if r_before > 0 and r_after < 0:
                        ctype = "positive_to_negative"
                    else:
                        ctype = "negative_to_positive"

                    # Linear interpolation for initial estimate
                    gamma_cross = parameter_values[i] - r_before * (
                        parameter_values[i + 1] - parameter_values[i]
                    ) / (r_after - r_before)

                    crossings.append(
                        ZeroCrossing(
                            parameter_value=float(parameter_values[i]),
                            eigenvalue_index=j,
                            crossing_type=ctype,
                            eigenvalue_before=float(r_before),
                            eigenvalue_after=float(r_after),
                            eigenvector_at_crossing=None,
                            interpolated_parameter=float(gamma_cross),
                        )
                    )

        return crossings

    # ------------------------------------------------------------------

    def detect_crossings(
        self,
        operator_fn: Callable[[float], NDArray],
        spectral_path: SpectralPath,
        refine: bool = True,
    ) -> List[ZeroCrossing]:
        """Detect and optionally refine zero crossings of Re(λ).

        Parameters
        ----------
        operator_fn : callable
            Operator function L(γ).
        spectral_path : SpectralPath
            Previously computed spectral path.
        refine : bool
            If ``True``, refine each crossing location with bisection.

        Returns
        -------
        list of ZeroCrossing
        """
        crossings = self._detect_zero_crossings(
            spectral_path.parameter_values, spectral_path.eigenvalues
        )

        if refine:
            refined: List[ZeroCrossing] = []
            for c in crossings:
                idx = np.searchsorted(spectral_path.parameter_values, c.parameter_value)
                idx = min(idx, len(spectral_path.parameter_values) - 2)
                gamma_a = spectral_path.parameter_values[idx]
                gamma_b = spectral_path.parameter_values[idx + 1]

                gamma_star, eig_vec = self._refine_crossing(
                    operator_fn, gamma_a, gamma_b, c.eigenvalue_index, self.crossing_tol
                )
                refined.append(
                    ZeroCrossing(
                        parameter_value=c.parameter_value,
                        eigenvalue_index=c.eigenvalue_index,
                        crossing_type=c.crossing_type,
                        eigenvalue_before=c.eigenvalue_before,
                        eigenvalue_after=c.eigenvalue_after,
                        eigenvector_at_crossing=eig_vec,
                        interpolated_parameter=float(gamma_star),
                    )
                )
            return refined

        return crossings

    # ------------------------------------------------------------------

    def _refine_crossing(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_before: float,
        gamma_after: float,
        eig_idx: int,
        tol: float,
    ) -> Tuple[float, Optional[NDArray]]:
        """Bisection refinement of a zero crossing of Re(λ_{eig_idx}).

        Parameters
        ----------
        operator_fn : callable
            Returns L(γ).
        gamma_before, gamma_after : float
            Bracket containing the sign change.
        eig_idx : int
            Index of the eigenvalue branch.
        tol : float
            Desired accuracy in γ.

        Returns
        -------
        gamma_star : float
            Refined crossing parameter value.
        eigenvector : ndarray or None
            Right eigenvector at ``gamma_star`` (if ``track_vectors`` is True).
        """
        a, b = float(gamma_before), float(gamma_after)

        # Evaluate real part at endpoints
        def _real_part(gamma: float) -> Tuple[float, Optional[NDArray]]:
            L = np.atleast_2d(operator_fn(gamma))
            eigs, vecs = self._compute_eigenvalues(L, k=self.n_eigenvalues)
            # We need the eig_idx-th eigenvalue after sorting
            idx = min(eig_idx, len(eigs) - 1)
            vec = vecs[:, idx] if self.track_vectors else None
            return float(eigs[idx].real), vec

        fa, _ = _real_part(a)
        fb, _ = _real_part(b)

        # Ensure bracket is valid; if not, return midpoint
        if fa * fb > 0:
            mid = 0.5 * (a + b)
            _, vec_mid = _real_part(mid)
            return mid, vec_mid

        max_iter = 60
        vec_mid: Optional[NDArray] = None

        for _ in range(max_iter):
            mid = 0.5 * (a + b)
            if (b - a) < tol:
                break
            fm, vec_mid = _real_part(mid)
            if fm == 0.0:
                break
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm

        return 0.5 * (a + b), vec_mid

    # ==================================================================
    #  Avoided-crossing detection
    # ==================================================================

    def _detect_avoided_crossings(
        self,
        eigenvalue_paths: NDArray[np.complexfloating],
        parameter_values: NDArray[np.floating],
        gap_threshold: float = 1e-3,
    ) -> List[AvoidedCrossing]:
        """Detect avoided crossings between eigenvalue branches.

        An avoided crossing is identified when the distance between two
        eigenvalue branches dips below ``gap_threshold`` at a local
        minimum of the pairwise distance.

        Parameters
        ----------
        eigenvalue_paths : ndarray, shape (n_params, n_eigs)
        parameter_values : ndarray, shape (n_params,)
        gap_threshold : float
            Maximum gap to classify as an avoided crossing.

        Returns
        -------
        list of AvoidedCrossing
        """
        n_params, n_eigs = eigenvalue_paths.shape
        avoided: List[AvoidedCrossing] = []

        for j in range(n_eigs):
            for k in range(j + 1, n_eigs):
                gaps = np.abs(eigenvalue_paths[:, j] - eigenvalue_paths[:, k])

                # Look for local minima in the gap profile
                for i in range(1, n_params - 1):
                    if gaps[i] < gaps[i - 1] and gaps[i] < gaps[i + 1]:
                        if gaps[i] < gap_threshold:
                            avoided.append(
                                AvoidedCrossing(
                                    parameter_value=float(parameter_values[i]),
                                    eigenvalue_indices=(j, k),
                                    minimum_gap=float(gaps[i]),
                                )
                            )

        return avoided

    def detect_avoided_crossings(
        self,
        spectral_path: SpectralPath,
        gap_threshold: float = 1e-3,
    ) -> List[AvoidedCrossing]:
        """Public wrapper for avoided-crossing detection.

        Parameters
        ----------
        spectral_path : SpectralPath
        gap_threshold : float

        Returns
        -------
        list of AvoidedCrossing
        """
        return self._detect_avoided_crossings(
            spectral_path.eigenvalues,
            spectral_path.parameter_values,
            gap_threshold,
        )

    # ==================================================================
    #  Spectral quantities
    # ==================================================================

    @staticmethod
    def spectral_gap(eigenvalues: NDArray[np.complexfloating]) -> float:
        """Gap between the largest and second-largest eigenvalue (real parts).

        Parameters
        ----------
        eigenvalues : ndarray, shape (n,)

        Returns
        -------
        float
            Non-negative gap  Δ = Re(λ₁) - Re(λ₂)  where λ₁ ≥ λ₂ ≥ …
        """
        reals = np.sort(eigenvalues.real)[::-1]
        if len(reals) < 2:
            return float("inf")
        return float(reals[0] - reals[1])

    @staticmethod
    def spectral_abscissa_val(eigenvalues: NDArray[np.complexfloating]) -> float:
        """Maximum real part of eigenvalues,  α(L) = max_i Re(λ_i).

        Parameters
        ----------
        eigenvalues : ndarray, shape (n,)

        Returns
        -------
        float
        """
        return float(np.max(eigenvalues.real))

    # Keep a convenience alias matching the specification name
    spectral_abscissa = spectral_abscissa_val

    @staticmethod
    def spectral_radius(eigenvalues: NDArray[np.complexfloating]) -> float:
        """Spectral radius  ρ(L) = max_i |λ_i|.

        Parameters
        ----------
        eigenvalues : ndarray, shape (n,)

        Returns
        -------
        float
        """
        return float(np.max(np.abs(eigenvalues)))

    # ==================================================================
    #  Jacobian & sensitivity
    # ==================================================================

    @staticmethod
    def compute_jacobian(
        operator_fn: Callable[[float], NDArray],
        gamma: float,
        eps: float = 1e-6,
    ) -> NDArray:
        """Compute ∂L/∂γ via centred finite differences.

        Parameters
        ----------
        operator_fn : callable
            Returns L(γ) as an ndarray.
        gamma : float
            Parameter value at which to evaluate the derivative.
        eps : float
            Step size for finite differences.

        Returns
        -------
        dL_dgamma : ndarray, same shape as L(γ)
            Approximate Jacobian matrix ∂L/∂γ.
        """
        L_plus = np.atleast_2d(operator_fn(gamma + eps))
        L_minus = np.atleast_2d(operator_fn(gamma - eps))
        return (L_plus - L_minus) / (2.0 * eps)

    @staticmethod
    def eigenvalue_sensitivity(
        L: NDArray,
        dL_dgamma: NDArray,
        eigenvalue_idx: int,
    ) -> complex:
        """Compute dλ/dγ for a simple eigenvalue.

        Uses the formula

            dλ/dγ  =  vᴴ (∂L/∂γ) w  /  (vᴴ w)

        where w is the right eigenvector and v is the left eigenvector
        (i.e. right eigenvector of Lᴴ) associated with λ.

        Parameters
        ----------
        L : ndarray, shape (n, n)
            Operator matrix.
        dL_dgamma : ndarray, shape (n, n)
            Derivative of the operator with respect to γ.
        eigenvalue_idx : int
            Index of the eigenvalue (after sorting by descending real part).

        Returns
        -------
        complex
            Sensitivity  dλ/dγ.

        Warns
        -----
        RuntimeWarning
            If the denominator vᴴ w is nearly zero (defective eigenvalue).
        """
        L = np.asarray(L, dtype=complex)
        dL_dgamma = np.asarray(dL_dgamma, dtype=complex)

        # Right eigenvectors
        eigvals, right_vecs = np.linalg.eig(L)
        order = np.argsort(-eigvals.real)
        eigvals = eigvals[order]
        right_vecs = right_vecs[:, order]

        # Left eigenvectors (right eigenvectors of L^H)
        eigvals_H, left_vecs = np.linalg.eig(L.conj().T)
        order_H = np.argsort(-eigvals_H.real)
        left_vecs = left_vecs[:, order_H]

        # Match left eigenvector to the same eigenvalue via overlap
        idx = min(eigenvalue_idx, len(eigvals) - 1)
        w = right_vecs[:, idx]

        # Find best-matching left eigenvector
        overlaps = np.abs(left_vecs.conj().T @ w)
        best_left = np.argmax(overlaps)
        v = left_vecs[:, best_left]

        denom = np.dot(v.conj(), w)
        if np.abs(denom) < 1e-14:
            warnings.warn(
                "Denominator vᴴw ≈ 0; eigenvalue may be defective.",
                RuntimeWarning,
                stacklevel=2,
            )
            return complex(np.nan)

        numer = np.dot(v.conj(), dL_dgamma @ w)
        return complex(numer / denom)

    # ==================================================================
    #  Multiplicity detection
    # ==================================================================

    @staticmethod
    def multiplicity_detection(
        eigenvalues: NDArray[np.complexfloating],
        tol: float = 1e-6,
    ) -> List[int]:
        """Detect eigenvalue multiplicities by clustering.

        Eigenvalues closer than ``tol`` in absolute value are considered
        identical.

        Parameters
        ----------
        eigenvalues : ndarray, shape (n,)
        tol : float
            Tolerance for considering two eigenvalues equal.

        Returns
        -------
        multiplicities : list of int, length n
            ``multiplicities[j]`` is the multiplicity of the j-th
            eigenvalue.
        """
        n = len(eigenvalues)
        visited = np.zeros(n, dtype=bool)
        multiplicities = np.ones(n, dtype=int)

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            for j in range(i + 1, n):
                if not visited[j] and np.abs(eigenvalues[i] - eigenvalues[j]) < tol:
                    cluster.append(j)
                    visited[j] = True
            mult = len(cluster)
            for idx in cluster:
                multiplicities[idx] = mult
            visited[i] = True

        return multiplicities.tolist()

    # ==================================================================
    #  Eigenvalue continuation
    # ==================================================================

    def continuation_step(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma: float,
        eigenvalue: complex,
        eigenvector: NDArray[np.complexfloating],
        dgamma: float,
    ) -> Tuple[complex, NDArray[np.complexfloating], float]:
        """Predict the next eigenvalue using first-order continuation.

        Given λ(γ) and dλ/dγ, the predicted eigenvalue at γ + dγ is

            λ_pred  =  λ(γ)  +  (dλ/dγ) · dγ

        A correction step re-solves the eigenvalue problem at γ + dγ and
        picks the eigenvalue closest to the prediction.

        Parameters
        ----------
        operator_fn : callable
            Returns L(γ).
        gamma : float
            Current parameter value.
        eigenvalue : complex
            Current eigenvalue λ(γ).
        eigenvector : ndarray, shape (n,)
            Current right eigenvector.
        dgamma : float
            Step size in the parameter.

        Returns
        -------
        new_eigenvalue : complex
            Corrected eigenvalue at γ + dγ.
        new_eigenvector : ndarray
            Corresponding right eigenvector.
        new_gamma : float
            The new parameter value γ + dγ.
        """
        L = np.atleast_2d(operator_fn(gamma))
        dL = self.compute_jacobian(operator_fn, gamma)

        # Determine eigenvalue index by proximity
        eigs, _ = self._compute_eigenvalues(L, k=self.n_eigenvalues)
        eig_idx = int(np.argmin(np.abs(eigs - eigenvalue)))

        sensitivity = self.eigenvalue_sensitivity(L, dL, eig_idx)
        predicted = eigenvalue + sensitivity * dgamma

        # Correction: solve at new γ and pick closest eigenvalue
        new_gamma = gamma + dgamma
        L_new = np.atleast_2d(operator_fn(new_gamma))
        new_eigs, new_vecs = self._compute_eigenvalues(L_new, k=self.n_eigenvalues)

        best = int(np.argmin(np.abs(new_eigs - predicted)))
        new_eigenvalue = new_eigs[best]
        new_eigenvector = new_vecs[:, best]

        return new_eigenvalue, new_eigenvector, new_gamma

    # ==================================================================
    #  Convenience: full analysis pipeline
    # ==================================================================

    def full_analysis(
        self,
        operator_fn: Callable[[float], NDArray],
        parameter_values: NDArray[np.floating],
        gap_threshold: float = 1e-3,
    ) -> dict:
        """Run tracking, crossing detection, and avoided-crossing detection.

        Parameters
        ----------
        operator_fn : callable
        parameter_values : ndarray
        gap_threshold : float

        Returns
        -------
        dict
            Keys: ``'spectral_path'``, ``'zero_crossings'``,
            ``'avoided_crossings'``.
        """
        path = self.track(operator_fn, parameter_values)
        crossings = self.detect_crossings(operator_fn, path, refine=True)
        avoided = self.detect_avoided_crossings(path, gap_threshold)

        return {
            "spectral_path": path,
            "zero_crossings": crossings,
            "avoided_crossings": avoided,
        }


# ======================================================================
#  Module-level convenience functions
# ======================================================================

def track_eigenvalues(
    operator_fn: Callable[[float], NDArray],
    parameter_values: NDArray[np.floating],
    n_eigenvalues: Optional[int] = None,
    sort_method: str = "real_part",
) -> SpectralPath:
    """Convenience wrapper: track eigenvalues over a parameter sweep.

    Parameters
    ----------
    operator_fn : callable
        ``operator_fn(gamma)`` → L(γ).
    parameter_values : ndarray
    n_eigenvalues : int or None
    sort_method : str

    Returns
    -------
    SpectralPath
    """
    tracker = EigenvalueTracker(
        n_eigenvalues=n_eigenvalues,
        track_vectors=True,
        sort_method=sort_method,
    )
    return tracker.track(operator_fn, parameter_values)


def find_bifurcations(
    operator_fn: Callable[[float], NDArray],
    parameter_values: NDArray[np.floating],
    n_eigenvalues: Optional[int] = None,
    crossing_tol: float = 1e-10,
) -> List[ZeroCrossing]:
    """Convenience wrapper: find stability transitions (bifurcations).

    A bifurcation occurs where the real part of an eigenvalue crosses zero,
    indicating a change in the stability of the equilibrium.

    Parameters
    ----------
    operator_fn : callable
    parameter_values : ndarray
    n_eigenvalues : int or None
    crossing_tol : float

    Returns
    -------
    list of ZeroCrossing
    """
    tracker = EigenvalueTracker(
        n_eigenvalues=n_eigenvalues,
        track_vectors=True,
        crossing_tol=crossing_tol,
    )
    path = tracker.track(operator_fn, parameter_values)
    return tracker.detect_crossings(operator_fn, path, refine=True)


def interpolate_spectral_path(
    spectral_path: SpectralPath,
    new_parameter_values: NDArray[np.floating],
    kind: str = "cubic",
) -> NDArray[np.complexfloating]:
    """Interpolate eigenvalue paths to a finer parameter grid.

    Parameters
    ----------
    spectral_path : SpectralPath
    new_parameter_values : ndarray
        Finer grid of parameter values.
    kind : str
        Interpolation kind passed to :class:`scipy.interpolate.interp1d`.

    Returns
    -------
    ndarray, shape (len(new_parameter_values), n_eigs)
        Interpolated eigenvalues (real and imaginary parts interpolated
        separately).
    """
    n_eigs = spectral_path.eigenvalues.shape[1]
    result = np.empty((len(new_parameter_values), n_eigs), dtype=complex)

    for j in range(n_eigs):
        real_interp = interp1d(
            spectral_path.parameter_values,
            spectral_path.eigenvalues[:, j].real,
            kind=kind,
            fill_value="extrapolate",
        )
        imag_interp = interp1d(
            spectral_path.parameter_values,
            spectral_path.eigenvalues[:, j].imag,
            kind=kind,
            fill_value="extrapolate",
        )
        result[:, j] = real_interp(new_parameter_values) + 1j * imag_interp(new_parameter_values)

    return result
