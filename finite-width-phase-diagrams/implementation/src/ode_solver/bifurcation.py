"""Bifurcation detection and analysis for parameter-dependent operators.

Detects, classifies, and analyses bifurcations of one-parameter families
of operators L(γ) arising from finite-width kernel ODEs.  Supports the
four standard codimension-1 bifurcation types (transcritical, saddle-node,
pitchfork, Hopf) with normal-form computation, direction determination,
and two-parameter continuation for codimension-2 analysis.

Mathematical background
-----------------------
A bifurcation of the equilibrium x* of  dx/dt = F(x, γ)  occurs at
parameter value γ = γ* when the Jacobian L(γ*) = D_x F(x*, γ*) has an
eigenvalue λ with Re(λ) = 0.  The *type* of bifurcation is determined by
the structure of the normal form on the centre manifold.

References
----------
* Kuznetsov, *Elements of Applied Bifurcation Theory*, Springer (2004).
* Seydel, *Practical Bifurcation and Stability Analysis*, Springer (2010).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg


# ======================================================================
#  Enumerations
# ======================================================================

class BifurcationType(Enum):
    """Classification of codimension-1 local bifurcations."""

    TRANSCRITICAL = auto()
    SADDLE_NODE = auto()
    PITCHFORK = auto()
    HOPF = auto()
    UNKNOWN = auto()


# ======================================================================
#  Data containers
# ======================================================================

@dataclass
class NormalForm:
    """Normal-form description of a bifurcation.

    Attributes
    ----------
    bifurcation_type : BifurcationType
        The classified bifurcation type.
    coefficients : dict[str, float]
        Named coefficients of the normal form (e.g. ``{'a': 1.2, 'b': -0.3}``).
    normal_form_equation : str
        Human-readable description of the normal form ODE,
        e.g. ``"dx/dt = a*γ*x + b*x²"``.
    direction : str
        One of ``'supercritical'``, ``'subcritical'``, or ``'degenerate'``.
    center_manifold_dim : int
        Dimension of the centre manifold at the bifurcation.
    """

    bifurcation_type: BifurcationType
    coefficients: Dict[str, float]
    normal_form_equation: str
    direction: str
    center_manifold_dim: int


@dataclass
class BifurcationPoint:
    """Record of a detected bifurcation point.

    Attributes
    ----------
    parameter_value : float
        Critical parameter value γ* at which the bifurcation occurs.
    eigenvalue : complex
        Critical eigenvalue (Re ≈ 0) at γ*.
    eigenvector : ndarray
        Right eigenvector associated with the critical eigenvalue.
    bifurcation_type : BifurcationType
        Classified bifurcation type.
    normal_form : NormalForm or None
        Normal-form coefficients (``None`` if not yet computed).
    test_function_value : float
        Value of the relevant test function at the bifurcation.
    stability_before : str
        Stability label (``'stable'``, ``'unstable'``, ``'center'``)
        for γ slightly less than γ*.
    stability_after : str
        Stability label for γ slightly greater than γ*.
    codimension : int
        Codimension of the bifurcation (typically 1).
    """

    parameter_value: float
    eigenvalue: complex
    eigenvector: NDArray[np.complexfloating]
    bifurcation_type: BifurcationType
    normal_form: Optional[NormalForm] = None
    test_function_value: float = 0.0
    stability_before: str = "unknown"
    stability_after: str = "unknown"
    codimension: int = 1


# ======================================================================
#  Bifurcation detector
# ======================================================================

class BifurcationDetector:
    """Detect and classify bifurcations of L(γ) along a parameter sweep.

    Parameters
    ----------
    tol : float
        Tolerance for considering an eigenvalue real part as zero.
    max_bisection_iter : int
        Maximum number of bisection iterations for refining the
        bifurcation parameter value.
    test_function_tol : float
        Tolerance on test functions for classification.
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_bisection_iter: int = 50,
        test_function_tol: float = 1e-6,
    ) -> None:
        self.tol = tol
        self.max_bisection_iter = max_bisection_iter
        self.test_function_tol = test_function_tol

    # ------------------------------------------------------------------
    #  Main detection entry points
    # ------------------------------------------------------------------

    def detect(
        self,
        operator_fn: Callable[[float], NDArray],
        parameter_range: Tuple[float, float],
        n_points: int = 100,
    ) -> List[BifurcationPoint]:
        """Scan a parameter range for bifurcations.

        Evaluates L(γ) at *n_points* equally-spaced values of γ, tracks
        the sign of the spectral abscissa, and refines each detected
        sign change via bisection.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
            Returns an (n, n) matrix for each scalar parameter γ.
        parameter_range : (float, float)
            Interval [γ_min, γ_max] to scan.
        n_points : int
            Number of sampling points.

        Returns
        -------
        list of BifurcationPoint
        """
        gammas = np.linspace(parameter_range[0], parameter_range[1], n_points)
        eigenvalue_data: List[NDArray] = []
        for g in gammas:
            L = np.atleast_2d(np.asarray(operator_fn(g), dtype=complex))
            eigenvalue_data.append(np.linalg.eigvals(L))

        bifurcations: List[BifurcationPoint] = []

        for i in range(len(gammas) - 1):
            re_max_left = np.max(eigenvalue_data[i].real)
            re_max_right = np.max(eigenvalue_data[i + 1].real)

            # Sign change in spectral abscissa ⇒ potential bifurcation
            if re_max_left * re_max_right < 0 or abs(re_max_left) < self.tol:
                gamma_bif = self._refine_bifurcation_point(
                    operator_fn,
                    gammas[i],
                    gammas[i + 1],
                    lambda g: np.max(np.linalg.eigvals(
                        np.atleast_2d(np.asarray(operator_fn(g), dtype=complex))
                    ).real),
                )

                eig_val, eig_vec = self._compute_critical_eigenvector(
                    operator_fn, gamma_bif,
                )

                stab_before = self.stability_analysis(operator_fn, gammas[i])
                stab_after = self.stability_analysis(operator_fn, gammas[i + 1])

                bif_type = self.classify(
                    operator_fn, gamma_bif, eig_val, eig_vec,
                )

                nf: Optional[NormalForm] = None
                try:
                    nf = self.compute_normal_form(
                        operator_fn, gamma_bif, bif_type, eig_vec,
                    )
                except Exception:  # noqa: BLE001
                    pass

                bifurcations.append(BifurcationPoint(
                    parameter_value=gamma_bif,
                    eigenvalue=eig_val,
                    eigenvector=eig_vec,
                    bifurcation_type=bif_type,
                    normal_form=nf,
                    test_function_value=abs(eig_val.real),
                    stability_before=stab_before,
                    stability_after=stab_after,
                    codimension=1,
                ))

        return bifurcations

    def detect_from_spectral_path(
        self,
        spectral_path: Any,
        operator_fn: Callable[[float], NDArray],
    ) -> List[BifurcationPoint]:
        """Detect bifurcations from a pre-computed :class:`SpectralPath`.

        Parameters
        ----------
        spectral_path : SpectralPath
            Object with ``parameter_values`` (1-D) and ``eigenvalues``
            (2-D, shape ``(n_params, n_eigs)``) attributes, plus optional
            ``eigenvectors`` list.
        operator_fn : callable  γ ↦ L(γ)
            Operator function used for classification and normal-form
            computation.

        Returns
        -------
        list of BifurcationPoint
        """
        gammas = np.asarray(spectral_path.parameter_values)
        eigs = np.asarray(spectral_path.eigenvalues)
        n_params, n_eigs = eigs.shape
        has_vecs = (
            spectral_path.eigenvectors is not None
            and len(spectral_path.eigenvectors) == n_params
        )

        bifurcations: List[BifurcationPoint] = []

        for j in range(n_eigs):
            for i in range(n_params - 1):
                re_left = eigs[i, j].real
                re_right = eigs[i + 1, j].real

                if re_left * re_right < 0:
                    # Bisection refinement using eigenvalue j
                    def _test_fn(g: float, _j: int = j) -> float:
                        L = np.atleast_2d(np.asarray(operator_fn(g), dtype=complex))
                        ev = np.linalg.eigvals(L)
                        ev_sorted = ev[np.argsort(np.abs(ev.real))]
                        idx = min(_j, len(ev_sorted) - 1)
                        return ev_sorted[idx].real

                    gamma_bif = self._refine_bifurcation_point(
                        operator_fn, gammas[i], gammas[i + 1], _test_fn,
                    )

                    if has_vecs:
                        eig_vec = spectral_path.eigenvectors[i][:, j]
                    else:
                        _, eig_vec = self._compute_critical_eigenvector(
                            operator_fn, gamma_bif,
                        )

                    eig_val_crit, _ = self._compute_critical_eigenvector(
                        operator_fn, gamma_bif,
                    )

                    stab_before = self.stability_analysis(operator_fn, gammas[i])
                    stab_after = self.stability_analysis(
                        operator_fn, gammas[i + 1],
                    )

                    bif_type = self.classify(
                        operator_fn, gamma_bif, eig_val_crit, eig_vec,
                    )

                    nf: Optional[NormalForm] = None
                    try:
                        nf = self.compute_normal_form(
                            operator_fn, gamma_bif, bif_type, eig_vec,
                        )
                    except Exception:  # noqa: BLE001
                        pass

                    bifurcations.append(BifurcationPoint(
                        parameter_value=gamma_bif,
                        eigenvalue=eig_val_crit,
                        eigenvector=eig_vec,
                        bifurcation_type=bif_type,
                        normal_form=nf,
                        test_function_value=abs(eig_val_crit.real),
                        stability_before=stab_before,
                        stability_after=stab_after,
                        codimension=1,
                    ))

        return bifurcations

    # ------------------------------------------------------------------
    #  Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_bif: float,
        eigenvalue: complex,
        eigenvector: NDArray,
    ) -> BifurcationType:
        """Classify the bifurcation type at a detected point.

        Uses a hierarchy of test functions evaluated at γ*:

        1. Hopf  — pair of purely imaginary eigenvalues.
        2. Pitchfork — symmetry-based test.
        3. Transcritical — determinant-based test τ.
        4. Saddle-node — det(L(γ*)) ≈ 0.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma_bif : float
            Bifurcation parameter value.
        eigenvalue : complex
            Critical eigenvalue at γ*.
        eigenvector : ndarray
            Associated right eigenvector.

        Returns
        -------
        BifurcationType
        """
        # Hopf: purely imaginary pair
        hopf_val = self._test_function_hopf(operator_fn, gamma_bif)
        if hopf_val < self.test_function_tol:
            return BifurcationType.HOPF

        # For real eigenvalue bifurcations
        pf_val = self._test_function_pitchfork(
            operator_fn, gamma_bif, eigenvector,
        )
        if pf_val < self.test_function_tol:
            return BifurcationType.PITCHFORK

        tc_val = self._test_function_transcritical(
            operator_fn, gamma_bif, eigenvalue, eigenvector,
        )
        if tc_val < self.test_function_tol:
            return BifurcationType.TRANSCRITICAL

        sn_val = self._test_function_saddle_node(operator_fn, gamma_bif)
        if sn_val < self.test_function_tol:
            return BifurcationType.SADDLE_NODE

        return BifurcationType.UNKNOWN

    # ------------------------------------------------------------------
    #  Test functions
    # ------------------------------------------------------------------

    def _test_function_transcritical(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma: float,
        eigenvalue: complex,
        eigenvector: NDArray,
    ) -> float:
        """Test function τ for transcritical bifurcation.

        For a transcritical bifurcation the trivial branch x ≡ 0 exists
        for all γ, so the augmented system

            ⎡ L(γ)   ∂L/∂γ · φ ⎤
            ⎣  φᵀ       0      ⎦

        is singular at γ* (where φ is the null eigenvector).  The test
        function is the absolute value of the determinant of this
        bordered matrix.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma : float
        eigenvalue : complex  (unused, kept for API symmetry)
        eigenvector : ndarray

        Returns
        -------
        float
            |det(B)|; small values indicate transcritical.
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma), dtype=complex))
        n = L.shape[0]
        phi = np.asarray(eigenvector, dtype=complex).ravel()[:n]

        eps = max(self.tol, 1e-7)
        L_plus = np.atleast_2d(np.asarray(operator_fn(gamma + eps), dtype=complex))
        dL_dgamma = (L_plus - L) / eps

        dL_phi = dL_dgamma @ phi

        # Build bordered matrix  (n+1) × (n+1)
        B = np.zeros((n + 1, n + 1), dtype=complex)
        B[:n, :n] = L
        B[:n, n] = dL_phi
        B[n, :n] = phi.conj()
        B[n, n] = 0.0

        return float(abs(np.linalg.det(B)))

    def _test_function_saddle_node(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma: float,
    ) -> float:
        """Test function for saddle-node bifurcation: |det L(γ)|.

        A saddle-node (fold) bifurcation occurs when a simple zero
        eigenvalue appears, i.e. det L(γ*) = 0.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma : float

        Returns
        -------
        float
            |det L(γ)|.
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma), dtype=complex))
        return float(abs(np.linalg.det(L)))

    def _test_function_hopf(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma: float,
    ) -> float:
        """Test function for Hopf bifurcation.

        A Hopf bifurcation requires a pair of complex-conjugate
        eigenvalues λ = ±iω crossing the imaginary axis.  The test
        function returns the minimum |Re(λ)| over all eigenvalues λ
        that have |Im(λ)| > threshold.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma : float

        Returns
        -------
        float
            Minimum |Re(λ)| among eigenvalues with non-trivial
            imaginary part.  Returns ``inf`` if no such pair exists.
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma), dtype=complex))
        eigs = np.linalg.eigvals(L)

        imag_threshold = 1e-6
        candidates = eigs[np.abs(eigs.imag) > imag_threshold]
        if len(candidates) == 0:
            return float("inf")

        # Check for conjugate pairing
        for c in candidates:
            partner_dists = np.abs(candidates - c.conjugate())
            if np.min(partner_dists) < self.tol * 10:
                return float(abs(c.real))

        return float("inf")

    def _test_function_pitchfork(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma: float,
        eigenvector: NDArray,
    ) -> float:
        """Test function for pitchfork bifurcation.

        A pitchfork bifurcation typically arises when the system has a
        Z₂ symmetry  F(-x, γ) = -F(x, γ).  We test this by checking
        that the quadratic coefficient in the projection onto the
        critical eigenvector vanishes.

        Specifically we estimate ⟨φ, D²F · (φ, φ)⟩ via a second-order
        finite difference applied to L(γ) · φ and check whether it is
        small.  If the quadratic term vanishes the leading nonlinearity
        is cubic, signalling a pitchfork.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma : float
        eigenvector : ndarray

        Returns
        -------
        float
            Approximate |⟨φ, D²F · (φ, φ)⟩|; small values indicate
            pitchfork.
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma), dtype=complex))
        n = L.shape[0]
        phi = np.asarray(eigenvector, dtype=complex).ravel()[:n]
        phi = phi / (np.linalg.norm(phi) + 1e-30)

        eps = max(self.tol, 1e-6)

        # Approximate second directional derivative of operator action
        # D²(Lφ) ≈ [L(γ)(φ+εφ) + L(γ)(φ-εφ) - 2L(γ)φ] / ε²
        # For a linear operator this vanishes; the test relies on
        # parameter perturbation to capture curvature:
        #   ⟨φ, [L(γ+ε) + L(γ-ε) - 2L(γ)] φ⟩ / ε²
        L_plus = np.atleast_2d(np.asarray(operator_fn(gamma + eps), dtype=complex))
        L_minus = np.atleast_2d(np.asarray(operator_fn(gamma - eps), dtype=complex))

        d2L = (L_plus + L_minus - 2.0 * L) / (eps * eps)
        quadratic_coeff = abs(float(np.real(phi.conj() @ d2L @ phi)))

        # Also check the first-order projection for symmetry breaking
        dL = (L_plus - L_minus) / (2.0 * eps)
        linear_coeff = abs(float(np.real(phi.conj() @ dL @ phi)))

        # Pitchfork if quadratic projection is small and linear is
        # non-degenerate
        if linear_coeff > self.test_function_tol:
            return quadratic_coeff
        return float("inf")

    # ------------------------------------------------------------------
    #  Normal-form computation
    # ------------------------------------------------------------------

    def compute_normal_form(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_bif: float,
        bif_type: BifurcationType,
        eigenvector: NDArray,
    ) -> NormalForm:
        """Compute normal-form coefficients near a bifurcation.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma_bif : float
            Bifurcation parameter value.
        bif_type : BifurcationType
        eigenvector : ndarray

        Returns
        -------
        NormalForm
        """
        if bif_type == BifurcationType.TRANSCRITICAL:
            return self._normal_form_transcritical(
                operator_fn, gamma_bif, eigenvector,
            )
        if bif_type == BifurcationType.SADDLE_NODE:
            return self._normal_form_saddle_node(
                operator_fn, gamma_bif, eigenvector,
            )
        if bif_type == BifurcationType.PITCHFORK:
            return self._normal_form_pitchfork(
                operator_fn, gamma_bif, eigenvector,
            )
        if bif_type == BifurcationType.HOPF:
            return self._normal_form_hopf(operator_fn, gamma_bif, eigenvector)

        # UNKNOWN — return stub
        return NormalForm(
            bifurcation_type=bif_type,
            coefficients={},
            normal_form_equation="unknown",
            direction="degenerate",
            center_manifold_dim=1,
        )

    # ---- transcritical: dx/dt = a·γ·x + b·x² -----------------------

    def _normal_form_transcritical(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_bif: float,
        eigenvector: NDArray,
    ) -> NormalForm:
        """Normal form for transcritical bifurcation.

        On the centre manifold the reduced ODE is

            dx/dt = a · μ · x + b · x²

        where μ = γ − γ* is the unfolding parameter.

        Coefficients are estimated via finite-difference Taylor
        expansion of the projected operator action.
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma_bif), dtype=complex))
        n = L.shape[0]
        phi = np.asarray(eigenvector, dtype=complex).ravel()[:n]
        phi = phi / (np.linalg.norm(phi) + 1e-30)

        eps = max(self.tol, 1e-7)
        L_plus = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif + eps), dtype=complex,
        ))
        L_minus = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif - eps), dtype=complex,
        ))

        # a ≈ ⟨φ, ∂L/∂γ · φ⟩
        dL = (L_plus - L_minus) / (2.0 * eps)
        a = float(np.real(phi.conj() @ dL @ phi))

        # b ≈ ½ ⟨φ, ∂²L/∂γ² · φ⟩  (proxy for nonlinear term)
        d2L = (L_plus + L_minus - 2.0 * L) / (eps * eps)
        b = 0.5 * float(np.real(phi.conj() @ d2L @ phi))

        coeffs = {"a": a, "b": b}
        direction = self._determine_direction(coeffs, BifurcationType.TRANSCRITICAL)

        return NormalForm(
            bifurcation_type=BifurcationType.TRANSCRITICAL,
            coefficients=coeffs,
            normal_form_equation="dx/dt = a*γ*x + b*x²",
            direction=direction,
            center_manifold_dim=1,
        )

    # ---- saddle-node: dx/dt = a·γ + b·x² ----------------------------

    def _normal_form_saddle_node(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_bif: float,
        eigenvector: NDArray,
    ) -> NormalForm:
        """Normal form for saddle-node (fold) bifurcation.

        On the centre manifold:

            dx/dt = a · μ + b · x²
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma_bif), dtype=complex))
        n = L.shape[0]
        phi = np.asarray(eigenvector, dtype=complex).ravel()[:n]
        phi = phi / (np.linalg.norm(phi) + 1e-30)

        eps = max(self.tol, 1e-7)
        L_plus = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif + eps), dtype=complex,
        ))
        L_minus = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif - eps), dtype=complex,
        ))

        # a = ⟨ψ, ∂F/∂γ⟩ ≈ ⟨φ, ∂L/∂γ · φ⟩  (first-order parameter
        #     derivative projected onto critical mode)
        dL = (L_plus - L_minus) / (2.0 * eps)
        a = float(np.real(phi.conj() @ dL @ phi))

        # b ≈ ½ ⟨φ, D²F(φ, φ)⟩ estimated from parameter curvature
        d2L = (L_plus + L_minus - 2.0 * L) / (eps * eps)
        b = 0.5 * float(np.real(phi.conj() @ d2L @ phi))

        # Fallback: if b ≈ 0 use trace-based estimate
        if abs(b) < 1e-14:
            b = float(np.real(np.trace(d2L))) / (2.0 * n)

        coeffs = {"a": a, "b": b}
        direction = self._determine_direction(coeffs, BifurcationType.SADDLE_NODE)

        return NormalForm(
            bifurcation_type=BifurcationType.SADDLE_NODE,
            coefficients=coeffs,
            normal_form_equation="dx/dt = a*γ + b*x²",
            direction=direction,
            center_manifold_dim=1,
        )

    # ---- pitchfork: dx/dt = a·γ·x + b·x³ ----------------------------

    def _normal_form_pitchfork(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_bif: float,
        eigenvector: NDArray,
    ) -> NormalForm:
        """Normal form for pitchfork bifurcation.

        On the centre manifold:

            dx/dt = a · μ · x + b · x³
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma_bif), dtype=complex))
        n = L.shape[0]
        phi = np.asarray(eigenvector, dtype=complex).ravel()[:n]
        phi = phi / (np.linalg.norm(phi) + 1e-30)

        eps = max(self.tol, 1e-7)
        L_plus = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif + eps), dtype=complex,
        ))
        L_minus = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif - eps), dtype=complex,
        ))

        # a = ⟨φ, ∂L/∂γ · φ⟩
        dL = (L_plus - L_minus) / (2.0 * eps)
        a = float(np.real(phi.conj() @ dL @ phi))

        # Cubic coefficient from third-order finite difference
        eps3 = eps * 5.0
        L_p2 = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif + 2 * eps3), dtype=complex,
        ))
        L_p1 = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif + eps3), dtype=complex,
        ))
        L_m1 = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif - eps3), dtype=complex,
        ))
        L_m2 = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif - 2 * eps3), dtype=complex,
        ))

        # Third derivative approximation  ∂³L/∂γ³
        d3L = (L_p2 - 2 * L_p1 + 2 * L_m1 - L_m2) / (2 * eps3 ** 3)
        b = float(np.real(phi.conj() @ d3L @ phi)) / 6.0

        coeffs = {"a": a, "b": b}
        direction = self._determine_direction(coeffs, BifurcationType.PITCHFORK)

        return NormalForm(
            bifurcation_type=BifurcationType.PITCHFORK,
            coefficients=coeffs,
            normal_form_equation="dx/dt = a*γ*x + b*x³",
            direction=direction,
            center_manifold_dim=1,
        )

    # ---- Hopf: dz/dt = (μ + iω)z + c₁|z|²z --------------------------

    def _normal_form_hopf(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_bif: float,
        eigenvector: NDArray,
    ) -> NormalForm:
        """Normal form for Hopf bifurcation.

        On the centre manifold (complex amplitude z):

            dz/dt = (α'(0)·μ + i·ω) z + c₁ |z|² z

        where α'(0) is the crossing speed of Re(λ) through zero, ω is
        the frequency Im(λ) at criticality, and c₁ is the first
        Lyapunov coefficient.
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma_bif), dtype=complex))
        n = L.shape[0]
        eigs = np.linalg.eigvals(L)

        # Find the pair with smallest |Re| and non-trivial Im
        imag_mask = np.abs(eigs.imag) > 1e-8
        if not np.any(imag_mask):
            warnings.warn(
                "No imaginary eigenvalue pair found for Hopf normal form.",
                stacklevel=2,
            )
            return NormalForm(
                bifurcation_type=BifurcationType.HOPF,
                coefficients={"omega": 0.0, "alpha_prime": 0.0, "c1": 0.0},
                normal_form_equation="dz/dt = (α'μ + iω)z + c₁|z|²z",
                direction="degenerate",
                center_manifold_dim=2,
            )

        candidates = eigs[imag_mask]
        idx_best = int(np.argmin(np.abs(candidates.real)))
        lam_crit = candidates[idx_best]
        omega = float(lam_crit.imag)

        # Crossing speed  α'(0) = d Re(λ) / dγ  via finite difference
        eps = max(self.tol, 1e-7)
        L_plus = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif + eps), dtype=complex,
        ))
        eigs_plus = np.linalg.eigvals(L_plus)
        # Match by closest to lam_crit
        dists = np.abs(eigs_plus - lam_crit)
        lam_plus = eigs_plus[int(np.argmin(dists))]
        alpha_prime = float((lam_plus.real - lam_crit.real) / eps)

        # First Lyapunov coefficient estimate via Hassard formula
        # approximation: c₁ ≈ Re(⟨p, D²F(q,q̄)⟩) / (2ω) where p, q
        # are left/right eigenvectors.  We approximate via parameter
        # curvature.
        L_minus = np.atleast_2d(np.asarray(
            operator_fn(gamma_bif - eps), dtype=complex,
        ))
        d2L = (L_plus + L_minus - 2.0 * L) / (eps * eps)

        phi = np.asarray(eigenvector, dtype=complex).ravel()[:n]
        phi = phi / (np.linalg.norm(phi) + 1e-30)

        c1 = float(np.real(phi.conj() @ d2L @ phi)) / (2.0 * abs(omega) + 1e-30)

        coeffs = {"omega": omega, "alpha_prime": alpha_prime, "c1": c1}
        direction = self._determine_direction(coeffs, BifurcationType.HOPF)

        return NormalForm(
            bifurcation_type=BifurcationType.HOPF,
            coefficients=coeffs,
            normal_form_equation="dz/dt = (α'μ + iω)z + c₁|z|²z",
            direction=direction,
            center_manifold_dim=2,
        )

    # ------------------------------------------------------------------
    #  Direction determination
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_direction(
        coefficients: Dict[str, float],
        bif_type: BifurcationType,
    ) -> str:
        """Determine supercritical vs subcritical from normal-form signs.

        Parameters
        ----------
        coefficients : dict[str, float]
        bif_type : BifurcationType

        Returns
        -------
        str
            ``'supercritical'``, ``'subcritical'``, or ``'degenerate'``.
        """
        if bif_type in (
            BifurcationType.TRANSCRITICAL,
            BifurcationType.SADDLE_NODE,
            BifurcationType.PITCHFORK,
        ):
            b = coefficients.get("b", 0.0)
            if abs(b) < 1e-14:
                return "degenerate"
            # For pitchfork/transcritical: b < 0 ⇒ supercritical
            return "supercritical" if b < 0 else "subcritical"

        if bif_type == BifurcationType.HOPF:
            c1 = coefficients.get("c1", 0.0)
            if abs(c1) < 1e-14:
                return "degenerate"
            # c₁ < 0 ⇒ supercritical Hopf
            return "supercritical" if c1 < 0 else "subcritical"

        return "degenerate"

    # ------------------------------------------------------------------
    #  Bisection refinement
    # ------------------------------------------------------------------

    def _refine_bifurcation_point(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_left: float,
        gamma_right: float,
        test_fn: Callable[[float], float],
    ) -> float:
        """Refine bifurcation location by bisection on a test function.

        Finds γ* ∈ [γ_left, γ_right] such that ``test_fn(γ*) ≈ 0``
        using the bisection method (requires a sign change of
        ``test_fn`` across the interval).

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
            Kept for sub-class overrides; not used directly here.
        gamma_left, gamma_right : float
            Bracket enclosing the bifurcation.
        test_fn : callable  γ ↦ float
            Scalar test function whose sign change indicates the
            bifurcation.

        Returns
        -------
        float
            Refined parameter value.
        """
        f_left = test_fn(gamma_left)
        f_right = test_fn(gamma_right)

        # If no sign change, return midpoint as best estimate
        if f_left * f_right > 0:
            return 0.5 * (gamma_left + gamma_right)

        for _ in range(self.max_bisection_iter):
            gamma_mid = 0.5 * (gamma_left + gamma_right)
            f_mid = test_fn(gamma_mid)

            if abs(f_mid) < self.tol:
                return gamma_mid

            if f_left * f_mid < 0:
                gamma_right = gamma_mid
                f_right = f_mid
            else:
                gamma_left = gamma_mid
                f_left = f_mid

            if abs(gamma_right - gamma_left) < self.tol:
                break

        return 0.5 * (gamma_left + gamma_right)

    # ------------------------------------------------------------------
    #  Critical eigenvector computation
    # ------------------------------------------------------------------

    def _compute_critical_eigenvector(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_bif: float,
    ) -> Tuple[complex, NDArray]:
        """Eigenvector corresponding to the critical eigenvalue at γ*.

        The critical eigenvalue is the one with the smallest |Re(λ)|.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma_bif : float

        Returns
        -------
        eigenvalue : complex
            The critical eigenvalue.
        eigenvector : ndarray
            Right eigenvector (column of the eigenvector matrix).
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma_bif), dtype=complex))
        eigenvalues, eigenvectors = np.linalg.eig(L)

        # Critical = smallest |Re(λ)|
        idx = int(np.argmin(np.abs(eigenvalues.real)))
        return eigenvalues[idx], eigenvectors[:, idx]

    # ------------------------------------------------------------------
    #  Stability analysis
    # ------------------------------------------------------------------

    def stability_analysis(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma: float,
    ) -> str:
        """Determine stability from eigenvalues of L(γ).

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma : float

        Returns
        -------
        str
            ``'stable'`` if all Re(λ) < 0,
            ``'unstable'`` if any Re(λ) > 0,
            ``'center'`` if max |Re(λ)| < tol.
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma), dtype=complex))
        eigs = np.linalg.eigvals(L)
        re_parts = eigs.real
        max_re = float(np.max(re_parts))

        if abs(max_re) < self.tol:
            return "center"
        if max_re > 0:
            return "unstable"
        return "stable"

    # ------------------------------------------------------------------
    #  Bifurcation diagram data
    # ------------------------------------------------------------------

    def bifurcation_diagram_data(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_range: Tuple[float, float],
        n_points: int = 200,
        state_dim: int = 1,
    ) -> Dict[str, NDArray]:
        """Compute data for a bifurcation diagram.

        For each parameter value γ, computes the steady-state branches
        (equilibria) and their stability.  For a linear operator L(γ)
        the trivial equilibrium x = 0 always exists; non-trivial
        branches are estimated from the spectral structure.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma_range : (float, float)
        n_points : int
        state_dim : int
            Dimension of the state variable x used for branch
            amplitude estimation.

        Returns
        -------
        dict with keys:
            ``'gammas'``          — parameter values, shape (n_points,)
            ``'trivial_branch'``  — zeros, shape (n_points,)
            ``'branch_plus'``     — upper branch amplitudes
            ``'branch_minus'``    — lower branch amplitudes
            ``'stability'``       — list of stability labels
            ``'spectral_abscissa'`` — max Re(λ) at each γ
            ``'eigenvalues'``     — full eigenvalue array (n_points, n)
        """
        gammas = np.linspace(gamma_range[0], gamma_range[1], n_points)
        trivial = np.zeros(n_points)
        branch_plus = np.full(n_points, np.nan)
        branch_minus = np.full(n_points, np.nan)
        stability_labels: List[str] = []
        spectral_abscissa = np.zeros(n_points)
        all_eigs: List[NDArray] = []

        for i, g in enumerate(gammas):
            L = np.atleast_2d(np.asarray(operator_fn(g), dtype=complex))
            eigs = np.linalg.eigvals(L)
            all_eigs.append(eigs)

            re_max = float(np.max(eigs.real))
            spectral_abscissa[i] = re_max

            if abs(re_max) < self.tol:
                stability_labels.append("center")
            elif re_max > 0:
                stability_labels.append("unstable")
            else:
                stability_labels.append("stable")

            # Estimate non-trivial branch amplitude from unstable
            # eigenvalues: for a simple pitchfork / transcritical,
            # amplitude ∝ sqrt(|μ|) or |μ| where μ = γ − γ*.
            unstable_eigs = eigs[eigs.real > self.tol]
            if len(unstable_eigs) > 0:
                # Rough amplitude from dominant unstable eigenvalue
                dominant = unstable_eigs[int(np.argmax(unstable_eigs.real))]
                amplitude = float(np.sqrt(abs(dominant.real)))
                branch_plus[i] = amplitude
                branch_minus[i] = -amplitude

        # Pad eigenvalue array to uniform shape
        max_n = max(len(e) for e in all_eigs)
        eig_array = np.full((n_points, max_n), np.nan, dtype=complex)
        for i, e in enumerate(all_eigs):
            eig_array[i, : len(e)] = e

        return {
            "gammas": gammas,
            "trivial_branch": trivial,
            "branch_plus": branch_plus,
            "branch_minus": branch_minus,
            "stability": stability_labels,
            "spectral_abscissa": spectral_abscissa,
            "eigenvalues": eig_array,
        }


# ======================================================================
#  Codimension analyser
# ======================================================================

class CodimensionAnalyzer:
    """Analyse codimension of detected bifurcation points.

    Provides checks for codimension-1 non-degeneracy conditions and
    detection of codimension-2 (degenerate) bifurcations, as well as
    two-parameter continuation.

    Parameters
    ----------
    tol : float
        Tolerance for degeneracy checks.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self.tol = tol

    # ------------------------------------------------------------------
    #  Codimension-1 verification
    # ------------------------------------------------------------------

    def codimension_1_check(self, bif_point: BifurcationPoint) -> bool:
        """Verify codimension-1 non-degeneracy conditions.

        For each bifurcation type the relevant non-degeneracy condition
        is checked using the normal-form coefficients (if available).

        Parameters
        ----------
        bif_point : BifurcationPoint

        Returns
        -------
        bool
            ``True`` if all codimension-1 conditions are satisfied.
        """
        if bif_point.normal_form is None:
            warnings.warn(
                "Normal form not computed; cannot verify codimension-1.",
                stacklevel=2,
            )
            return False

        nf = bif_point.normal_form
        bt = nf.bifurcation_type
        coeffs = nf.coefficients

        if bt == BifurcationType.SADDLE_NODE:
            # Non-degeneracy: a ≠ 0  and  b ≠ 0
            return (
                abs(coeffs.get("a", 0.0)) > self.tol
                and abs(coeffs.get("b", 0.0)) > self.tol
            )

        if bt == BifurcationType.TRANSCRITICAL:
            # Non-degeneracy: a ≠ 0  and  b ≠ 0
            return (
                abs(coeffs.get("a", 0.0)) > self.tol
                and abs(coeffs.get("b", 0.0)) > self.tol
            )

        if bt == BifurcationType.PITCHFORK:
            # Non-degeneracy: a ≠ 0  and  b ≠ 0
            return (
                abs(coeffs.get("a", 0.0)) > self.tol
                and abs(coeffs.get("b", 0.0)) > self.tol
            )

        if bt == BifurcationType.HOPF:
            # Non-degeneracy: α'(0) ≠ 0  and  c₁ ≠ 0
            return (
                abs(coeffs.get("alpha_prime", 0.0)) > self.tol
                and abs(coeffs.get("c1", 0.0)) > self.tol
            )

        return False

    # ------------------------------------------------------------------
    #  Degenerate (codimension-2) detection
    # ------------------------------------------------------------------

    def degenerate_check(
        self,
        operator_fn: Callable[[float], NDArray],
        gamma_bif: float,
        bif_type: BifurcationType,
    ) -> Dict[str, Any]:
        """Check for degenerate (codimension-2) bifurcation conditions.

        Returns a dictionary with degeneracy indicators for the given
        bifurcation type.

        Parameters
        ----------
        operator_fn : callable  γ ↦ L(γ)
        gamma_bif : float
        bif_type : BifurcationType

        Returns
        -------
        dict
            Keys include ``'is_degenerate'`` (bool), ``'codimension'``
            (int), and type-specific degeneracy information.
        """
        L = np.atleast_2d(np.asarray(operator_fn(gamma_bif), dtype=complex))
        n = L.shape[0]
        eigs = np.linalg.eigvals(L)

        result: Dict[str, Any] = {
            "is_degenerate": False,
            "codimension": 1,
            "details": {},
        }

        # Check multiplicity of the zero eigenvalue
        zero_eigs = eigs[np.abs(eigs.real) < self.tol * 100]
        multiplicity = len(zero_eigs)

        if multiplicity >= 2:
            result["is_degenerate"] = True
            result["codimension"] = 2
            result["details"]["zero_eigenvalue_multiplicity"] = multiplicity

        if bif_type == BifurcationType.SADDLE_NODE:
            # Cusp (codim-2): the quadratic coefficient b vanishes
            detector = BifurcationDetector(tol=self.tol)
            crit_val, crit_vec = detector._compute_critical_eigenvector(
                operator_fn, gamma_bif,
            )
            nf = detector._normal_form_saddle_node(
                operator_fn, gamma_bif, crit_vec,
            )
            if abs(nf.coefficients.get("b", 1.0)) < self.tol:
                result["is_degenerate"] = True
                result["codimension"] = 2
                result["details"]["cusp"] = True

        elif bif_type == BifurcationType.HOPF:
            # Bautin (codim-2): first Lyapunov coefficient c₁ = 0
            detector = BifurcationDetector(tol=self.tol)
            crit_val, crit_vec = detector._compute_critical_eigenvector(
                operator_fn, gamma_bif,
            )
            nf = detector._normal_form_hopf(
                operator_fn, gamma_bif, crit_vec,
            )
            if abs(nf.coefficients.get("c1", 1.0)) < self.tol:
                result["is_degenerate"] = True
                result["codimension"] = 2
                result["details"]["bautin"] = True

        elif bif_type == BifurcationType.PITCHFORK:
            # Degenerate pitchfork: cubic coefficient b = 0
            detector = BifurcationDetector(tol=self.tol)
            crit_val, crit_vec = detector._compute_critical_eigenvector(
                operator_fn, gamma_bif,
            )
            nf = detector._normal_form_pitchfork(
                operator_fn, gamma_bif, crit_vec,
            )
            if abs(nf.coefficients.get("b", 1.0)) < self.tol:
                result["is_degenerate"] = True
                result["codimension"] = 2
                result["details"]["degenerate_pitchfork"] = True

        elif bif_type == BifurcationType.TRANSCRITICAL:
            # Degenerate transcritical: quadratic coefficient b = 0
            detector = BifurcationDetector(tol=self.tol)
            crit_val, crit_vec = detector._compute_critical_eigenvector(
                operator_fn, gamma_bif,
            )
            nf = detector._normal_form_transcritical(
                operator_fn, gamma_bif, crit_vec,
            )
            if abs(nf.coefficients.get("b", 1.0)) < self.tol:
                result["is_degenerate"] = True
                result["codimension"] = 2
                result["details"]["degenerate_transcritical"] = True

        # Check for Bogdanov–Takens: double zero eigenvalue
        if multiplicity >= 2:
            # Check algebraic vs geometric multiplicity
            rank_deficiency = n - int(np.linalg.matrix_rank(
                L, tol=self.tol * 100,
            ))
            if rank_deficiency >= 2:
                result["details"]["bogdanov_takens"] = True
                result["codimension"] = max(result["codimension"], 2)

        return result

    # ------------------------------------------------------------------
    #  Two-parameter continuation
    # ------------------------------------------------------------------

    def two_parameter_continuation(
        self,
        operator_fn_2param: Callable[[float, float], NDArray],
        gamma1_bif: float,
        gamma2_range: Tuple[float, float],
        n_points: int = 100,
    ) -> Dict[str, NDArray]:
        """Continue a bifurcation point in a second parameter.

        Given a two-parameter family L(γ₁, γ₂), traces the curve
        of bifurcation points in the (γ₁, γ₂) plane starting from
        a known bifurcation at (γ₁*, γ₂_start).

        Uses a simple prediction-correction strategy: for each new γ₂
        value, bisects in γ₁ to locate the bifurcation.

        Parameters
        ----------
        operator_fn_2param : callable  (γ₁, γ₂) ↦ L(γ₁, γ₂)
        gamma1_bif : float
            Known bifurcation value in the first parameter.
        gamma2_range : (float, float)
            Range of the second parameter to sweep.
        n_points : int
            Number of continuation steps.

        Returns
        -------
        dict with keys:
            ``'gamma1_curve'`` — array of γ₁ bifurcation values
            ``'gamma2_curve'`` — array of γ₂ values
            ``'eigenvalues'``  — critical eigenvalue at each point
            ``'types'``        — list of BifurcationType at each point
        """
        gamma2_vals = np.linspace(gamma2_range[0], gamma2_range[1], n_points)
        gamma1_curve = np.full(n_points, np.nan)
        crit_eigs = np.full(n_points, np.nan, dtype=complex)
        bif_types: List[BifurcationType] = []

        detector = BifurcationDetector(
            tol=self.tol, max_bisection_iter=50,
        )

        current_gamma1 = gamma1_bif
        search_width = abs(gamma1_bif) * 0.5 + 1.0

        for i, g2 in enumerate(gamma2_vals):
            # Build single-parameter operator at fixed γ₂
            def _op(g1: float, _g2: float = g2) -> NDArray:
                return operator_fn_2param(g1, _g2)

            # Search in neighbourhood of current prediction
            g1_lo = current_gamma1 - search_width
            g1_hi = current_gamma1 + search_width

            bifs = detector.detect(_op, (g1_lo, g1_hi), n_points=60)

            if bifs:
                # Pick the closest to current prediction
                dists = [abs(b.parameter_value - current_gamma1) for b in bifs]
                best = bifs[int(np.argmin(dists))]
                gamma1_curve[i] = best.parameter_value
                crit_eigs[i] = best.eigenvalue
                bif_types.append(best.bifurcation_type)
                current_gamma1 = best.parameter_value
            else:
                gamma1_curve[i] = np.nan
                crit_eigs[i] = np.nan
                bif_types.append(BifurcationType.UNKNOWN)

        return {
            "gamma1_curve": gamma1_curve,
            "gamma2_curve": gamma2_vals,
            "eigenvalues": crit_eigs,
            "types": bif_types,
        }
