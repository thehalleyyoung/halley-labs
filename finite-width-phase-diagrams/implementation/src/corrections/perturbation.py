"""Perturbative validity analysis for finite-width neural network kernels.

Provides tools for assessing when the 1/N (finite-width) perturbation
expansion of neural tangent kernels remains valid.  Given the leading-order
kernel Θ^(0) and successive corrections Θ^(1), Θ^(2), …, this module
computes validity ratios, convergence radii, and confidence levels so that
downstream phase-diagram code can flag regions where the expansion breaks
down.

Mathematical background
-----------------------
The NTK admits an asymptotic expansion in inverse width:

    Θ(N) = Θ^(0) + (1/N) Θ^(1) + (1/N²) Θ^(2) + …

The expansion is *perturbatively valid* when the correction terms are
small relative to the leading-order term.  Several metrics quantify this:

* **Validity functional**  V[Θ] = ‖Θ^(1)‖_op / ‖Θ^(0)‖_op
* **Spectral validity**    max|λ_i^(1)| / min|λ_i^(0)|
* **Entrywise validity**   max|Θ^(1)_{ij}| / max|Θ^(0)_{ij}|
* **Convergence radius**   estimated from d'Alembert ratio test or
  Domb–Sykes analysis of the correction sequence.
"""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sla


# ======================================================================
#  Enumerations
# ======================================================================


class ConfidenceLevel(enum.Enum):
    """Qualitative confidence in the perturbative expansion.

    Levels are determined by the validity ratio V[Θ]:
        HIGH      – V < high_threshold   (default 0.1)
        MODERATE  – V < moderate_threshold (default 0.3)
        LOW       – V < low_threshold     (default 0.5)
        INVALID   – V ≥ low_threshold
    """

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INVALID = "invalid"


# ======================================================================
#  Data containers
# ======================================================================


@dataclass
class ConvergenceRadius:
    """Result of a convergence-radius estimation.

    Attributes
    ----------
    radius : float
        Estimated convergence radius in the expansion parameter 1/N.
    estimated_breakdown_width : int
        Minimum network width N below which the expansion is expected
        to diverge (i.e. 1/N > radius  ⟹  N < 1/radius).
    method : str
        Name of the estimation method (e.g. ``"ratio_test"``,
        ``"domb_sykes"``).
    confidence : float
        Heuristic confidence in [0, 1] for the estimate.
    """

    radius: float
    estimated_breakdown_width: int
    method: str
    confidence: float


@dataclass
class ValidityResult:
    """Aggregate result of a perturbative-validity assessment.

    Attributes
    ----------
    validity_ratio : float
        Operator-norm ratio ‖Θ^(1)‖_op / ‖Θ^(0)‖_op.
    confidence : ConfidenceLevel
        Qualitative confidence derived from *validity_ratio*.
    convergence_radius : ConvergenceRadius
        Convergence-radius estimate (populated when Θ^(2) is available).
    spectral_validity : float
        Spectral validity metric.
    entrywise_validity : float
        Entry-wise validity metric.
    correction_magnitudes : dict
        Mapping ``{"theta_0_norm": …, "theta_1_norm": …, …}``.
    warnings : list of str
        Human-readable warnings about potential issues.
    diagnostics : dict
        Extra diagnostic quantities for debugging / logging.
    """

    validity_ratio: float
    confidence: ConfidenceLevel
    convergence_radius: ConvergenceRadius
    spectral_validity: float
    entrywise_validity: float
    correction_magnitudes: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)


# ======================================================================
#  Constants
# ======================================================================

_EPS = 1e-14  # Guard against division by zero.
_DEFAULT_HIGH = 0.1
_DEFAULT_MOD = 0.3
_DEFAULT_LOW = 0.5


# ======================================================================
#  Perturbative Validator
# ======================================================================


class PerturbativeValidator:
    """Assess the validity of the 1/N perturbation expansion.

    Parameters
    ----------
    high_threshold : float
        Maximum validity ratio for ``ConfidenceLevel.HIGH``.
    moderate_threshold : float
        Maximum validity ratio for ``ConfidenceLevel.MODERATE``.
    low_threshold : float
        Maximum validity ratio for ``ConfidenceLevel.LOW``.
        Ratios at or above this value yield ``ConfidenceLevel.INVALID``.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        high_threshold: float = _DEFAULT_HIGH,
        moderate_threshold: float = _DEFAULT_MOD,
        low_threshold: float = _DEFAULT_LOW,
    ) -> None:
        if not (0 < high_threshold < moderate_threshold < low_threshold):
            raise ValueError(
                "Thresholds must satisfy 0 < high < moderate < low; got "
                f"{high_threshold}, {moderate_threshold}, {low_threshold}."
            )
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold
        self.low_threshold = low_threshold

    # ------------------------------------------------------------------
    #  Full validation entry-point
    # ------------------------------------------------------------------

    def validate(
        self,
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
        theta_2: Optional[NDArray[np.floating]] = None,
        width: Optional[int] = None,
    ) -> ValidityResult:
        """Run a comprehensive perturbative-validity assessment.

        Parameters
        ----------
        theta_0 : ndarray, shape (P, P)
            Leading-order (infinite-width) kernel.
        theta_1 : ndarray, shape (P, P)
            First-order finite-width correction.
        theta_2 : ndarray, shape (P, P), optional
            Second-order correction.  When provided the convergence
            radius is estimated via the d'Alembert ratio test.
        width : int, optional
            Network width *N*.  Used for the Bauer–Fike eigenvalue
            perturbation bound and breakdown-width estimation.

        Returns
        -------
        ValidityResult
        """
        theta_0 = np.asarray(theta_0, dtype=np.float64)
        theta_1 = np.asarray(theta_1, dtype=np.float64)
        if theta_2 is not None:
            theta_2 = np.asarray(theta_2, dtype=np.float64)

        # Core metrics
        v_ratio = self.validity_functional(theta_0, theta_1)
        s_val = self.spectral_validity(theta_0, theta_1)
        e_val = self.entrywise_validity(theta_0, theta_1)
        confidence = self.stratify_confidence(v_ratio)

        # Norms for diagnostics
        norm0 = self._compute_operator_norm(theta_0)
        norm1 = self._compute_operator_norm(theta_1)
        magnitudes: Dict[str, float] = {
            "theta_0_norm": float(norm0),
            "theta_1_norm": float(norm1),
        }

        warn_list: List[str] = []
        diag: Dict[str, object] = {
            "validity_ratio": float(v_ratio),
            "spectral_validity": float(s_val),
            "entrywise_validity": float(e_val),
        }

        # Convergence radius -----------------------------------------------
        if theta_2 is not None:
            norm2 = self._compute_operator_norm(theta_2)
            magnitudes["theta_2_norm"] = float(norm2)
            conv = self.convergence_radius_ratio_test(theta_1, theta_2)
        else:
            conv = ConvergenceRadius(
                radius=np.inf,
                estimated_breakdown_width=1,
                method="unavailable",
                confidence=0.0,
            )
            warn_list.append(
                "Θ^(2) not provided; convergence radius not estimated."
            )

        # Bauer–Fike bound -------------------------------------------------
        if width is not None:
            bf_bound = self._eigenvalue_perturbation_bound(
                theta_0, theta_1, width
            )
            diag["bauer_fike_bound"] = float(bf_bound)
            if bf_bound > 1.0:
                warn_list.append(
                    f"Bauer–Fike bound ({bf_bound:.4f}) exceeds 1; "
                    "eigenvalue perturbation may be non-perturbative."
                )

        # Warnings from metrics --------------------------------------------
        if v_ratio > self.low_threshold:
            warn_list.append(
                f"Validity ratio {v_ratio:.4f} exceeds low threshold "
                f"({self.low_threshold}); expansion likely invalid."
            )
        if s_val > 1.0:
            warn_list.append(
                f"Spectral validity {s_val:.4f} > 1; first-order "
                "eigenvalue shift exceeds smallest leading eigenvalue."
            )

        return ValidityResult(
            validity_ratio=float(v_ratio),
            confidence=confidence,
            convergence_radius=conv,
            spectral_validity=float(s_val),
            entrywise_validity=float(e_val),
            correction_magnitudes=magnitudes,
            warnings=warn_list,
            diagnostics=diag,
        )

    # ------------------------------------------------------------------
    #  Validity functional  V[Θ] = ‖Θ^(1)‖_op / ‖Θ^(0)‖_op
    # ------------------------------------------------------------------

    def validity_functional(
        self,
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
    ) -> float:
        r"""Compute the operator-norm validity ratio.

        .. math::

            V[\Theta] = \frac{\|\Theta^{(1)}\|_{\mathrm{op}}}
                             {\|\Theta^{(0)}\|_{\mathrm{op}}}

        Parameters
        ----------
        theta_0 : ndarray, shape (P, P)
        theta_1 : ndarray, shape (P, P)

        Returns
        -------
        float
            Non-negative validity ratio.  Values ≪ 1 indicate a
            well-controlled perturbation.
        """
        theta_0 = np.asarray(theta_0, dtype=np.float64)
        theta_1 = np.asarray(theta_1, dtype=np.float64)

        norm0 = self._compute_operator_norm(theta_0)
        norm1 = self._compute_operator_norm(theta_1)
        if norm0 < _EPS:
            warnings.warn(
                "Leading-order kernel has near-zero operator norm; "
                "validity ratio is ill-defined."
            )
            return np.inf
        return float(norm1 / norm0)

    # ------------------------------------------------------------------
    #  Spectral validity
    # ------------------------------------------------------------------

    def spectral_validity(
        self,
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
    ) -> float:
        r"""Spectral validity metric.

        Compares the largest eigenvalue magnitude of the correction to
        the smallest *non-zero* eigenvalue magnitude of the leading term:

        .. math::

            S = \frac{\max_i |\lambda_i^{(1)}|}
                     {\min_{i:\,\lambda_i^{(0)}\ne 0} |\lambda_i^{(0)}|}

        For a symmetric kernel the eigenvalues are real; for non-symmetric
        matrices we use the moduli of the (possibly complex) eigenvalues.

        Parameters
        ----------
        theta_0 : ndarray, shape (P, P)
        theta_1 : ndarray, shape (P, P)

        Returns
        -------
        float
            Spectral validity ratio.  Values < 1 ensure the first-order
            eigenvalue shifts do not exceed the leading spectral gap.
        """
        theta_0 = np.asarray(theta_0, dtype=np.float64)
        theta_1 = np.asarray(theta_1, dtype=np.float64)

        eig0 = np.abs(sla.eigvalsh(theta_0)) if self._is_symmetric(theta_0) \
            else np.abs(sla.eigvals(theta_0))
        eig1 = np.abs(sla.eigvalsh(theta_1)) if self._is_symmetric(theta_1) \
            else np.abs(sla.eigvals(theta_1))

        # Filter near-zero eigenvalues of Θ^(0)
        nonzero_mask = eig0 > _EPS
        if not np.any(nonzero_mask):
            warnings.warn(
                "All eigenvalues of Θ^(0) are near-zero; "
                "spectral validity is ill-defined."
            )
            return np.inf

        min_eig0 = float(np.min(eig0[nonzero_mask]))
        max_eig1 = float(np.max(eig1)) if eig1.size > 0 else 0.0
        return max_eig1 / max(min_eig0, _EPS)

    # ------------------------------------------------------------------
    #  Entry-wise validity
    # ------------------------------------------------------------------

    def entrywise_validity(
        self,
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
    ) -> float:
        r"""Entry-wise validity metric.

        .. math::

            E = \frac{\max_{i,j} |\Theta^{(1)}_{ij}|}
                     {\max_{i,j} |\Theta^{(0)}_{ij}|}

        Parameters
        ----------
        theta_0 : ndarray, shape (P, P)
        theta_1 : ndarray, shape (P, P)

        Returns
        -------
        float
        """
        theta_0 = np.asarray(theta_0, dtype=np.float64)
        theta_1 = np.asarray(theta_1, dtype=np.float64)

        max0 = float(np.max(np.abs(theta_0)))
        max1 = float(np.max(np.abs(theta_1)))
        if max0 < _EPS:
            warnings.warn(
                "Leading-order kernel has near-zero entries; "
                "entry-wise validity is ill-defined."
            )
            return np.inf
        return max1 / max0

    # ------------------------------------------------------------------
    #  Convergence radius – d'Alembert ratio test
    # ------------------------------------------------------------------

    def convergence_radius_ratio_test(
        self,
        theta_1: NDArray[np.floating],
        theta_2: NDArray[np.floating],
    ) -> ConvergenceRadius:
        r"""Estimate convergence radius via the d'Alembert ratio test.

        For a power series :math:`\sum a_k x^k`, the radius of
        convergence satisfies

        .. math::

            R = \lim_{k\to\infty} \frac{|a_k|}{|a_{k+1}|}

        Here we approximate with the first two correction norms:

        .. math::

            R \approx \frac{\|\Theta^{(1)}\|_{\mathrm{op}}}
                           {\|\Theta^{(2)}\|_{\mathrm{op}}}

        The estimated breakdown width is :math:`\lceil 1/R \rceil`.

        Parameters
        ----------
        theta_1 : ndarray, shape (P, P)
        theta_2 : ndarray, shape (P, P)

        Returns
        -------
        ConvergenceRadius
        """
        theta_1 = np.asarray(theta_1, dtype=np.float64)
        theta_2 = np.asarray(theta_2, dtype=np.float64)

        norm1 = self._compute_operator_norm(theta_1)
        norm2 = self._compute_operator_norm(theta_2)

        if norm2 < _EPS:
            # Second-order correction is negligible → series likely
            # converges for all practical widths.
            return ConvergenceRadius(
                radius=np.inf,
                estimated_breakdown_width=1,
                method="ratio_test",
                confidence=0.5,
            )

        radius = float(norm1 / norm2)
        # Confidence heuristic: higher when the ratio is well-separated
        # from 1 and the norms are not tiny.
        conf = float(np.clip(1.0 - np.exp(-radius), 0.0, 1.0))
        breakdown = max(1, int(np.ceil(1.0 / radius))) if radius > _EPS else 1

        return ConvergenceRadius(
            radius=radius,
            estimated_breakdown_width=breakdown,
            method="ratio_test",
            confidence=conf,
        )

    # ------------------------------------------------------------------
    #  Convergence radius – Domb–Sykes plot
    # ------------------------------------------------------------------

    def convergence_radius_domb_sykes(
        self,
        corrections_sequence: Sequence[NDArray[np.floating]],
    ) -> ConvergenceRadius:
        r"""Estimate convergence radius via a Domb–Sykes plot.

        Given a sequence of correction matrices
        :math:`\Theta^{(1)}, \Theta^{(2)}, \ldots`, compute the ratios

        .. math::

            r_k = \frac{\|\Theta^{(k+1)}\|_{\mathrm{op}}}
                       {\|\Theta^{(k)}\|_{\mathrm{op}}}

        and extrapolate :math:`r_k \to r_\infty` via a linear fit of
        :math:`r_k` vs :math:`1/k` (Domb–Sykes method).  The convergence
        radius is :math:`R = 1 / r_\infty`.

        Parameters
        ----------
        corrections_sequence : sequence of ndarray
            At least two correction matrices ordered by expansion order.

        Returns
        -------
        ConvergenceRadius
        """
        if len(corrections_sequence) < 2:
            raise ValueError(
                "Domb–Sykes method requires at least two correction "
                f"matrices; got {len(corrections_sequence)}."
            )

        norms = np.array(
            [self._compute_operator_norm(np.asarray(c, dtype=np.float64))
             for c in corrections_sequence],
            dtype=np.float64,
        )

        # Guard against zero norms
        if np.any(norms[:-1] < _EPS):
            warnings.warn(
                "Near-zero norm encountered in correction sequence; "
                "Domb–Sykes estimate may be unreliable."
            )
            nonzero = norms[:-1] > _EPS
            if not np.any(nonzero):
                return ConvergenceRadius(
                    radius=np.inf,
                    estimated_breakdown_width=1,
                    method="domb_sykes",
                    confidence=0.0,
                )

        ratios = norms[1:] / np.maximum(norms[:-1], _EPS)  # r_k
        k_values = np.arange(1, len(ratios) + 1, dtype=np.float64)
        inv_k = 1.0 / k_values

        # Linear fit: r_k ≈ r_inf + slope / k
        if len(ratios) >= 2:
            coeffs = np.polyfit(inv_k, ratios, deg=1)
            r_inf = float(coeffs[1])  # intercept = r_∞
            residuals = ratios - np.polyval(coeffs, inv_k)
            fit_quality = float(
                1.0 - np.std(residuals) / (np.std(ratios) + _EPS)
            )
        else:
            r_inf = float(ratios[0])
            fit_quality = 0.3  # Low confidence with a single ratio

        r_inf = max(r_inf, _EPS)
        radius = 1.0 / r_inf

        conf = float(np.clip(fit_quality, 0.0, 1.0))
        breakdown = max(1, int(np.ceil(1.0 / radius))) if radius > _EPS else 1

        return ConvergenceRadius(
            radius=float(radius),
            estimated_breakdown_width=breakdown,
            method="domb_sykes",
            confidence=conf,
        )

    # ------------------------------------------------------------------
    #  Breakdown-width estimation
    # ------------------------------------------------------------------

    def estimate_breakdown_width(
        self,
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
        theta_2: Optional[NDArray[np.floating]] = None,
        threshold: float = 0.5,
    ) -> int:
        r"""Estimate the minimum width where the expansion is valid.

        Finds the smallest integer *N* such that

        .. math::

            \frac{1}{N}\,\frac{\|\Theta^{(1)}\|_{\mathrm{op}}}
                              {\|\Theta^{(0)}\|_{\mathrm{op}}}
            \;<\; \text{threshold}

        When :math:`\Theta^{(2)}` is available the second-order
        contribution is also included:

        .. math::

            \frac{1}{N}\,V_1 + \frac{1}{N^2}\,V_2 < \text{threshold}

        where :math:`V_k = \|\Theta^{(k)}\|_{\mathrm{op}} /
        \|\Theta^{(0)}\|_{\mathrm{op}}`.

        Parameters
        ----------
        theta_0, theta_1 : ndarray
        theta_2 : ndarray, optional
        threshold : float

        Returns
        -------
        int
            Estimated minimum width *N*.  Returns 1 if the expansion
            is valid even at width 1.
        """
        theta_0 = np.asarray(theta_0, dtype=np.float64)
        theta_1 = np.asarray(theta_1, dtype=np.float64)

        norm0 = self._compute_operator_norm(theta_0)
        if norm0 < _EPS:
            return 1

        v1 = self._compute_operator_norm(theta_1) / norm0

        if theta_2 is not None:
            theta_2 = np.asarray(theta_2, dtype=np.float64)
            v2 = self._compute_operator_norm(theta_2) / norm0
        else:
            v2 = 0.0

        # Search: smallest N with  v1/N + v2/N^2 < threshold
        # For the first-order-only case the answer is ceil(v1 / threshold).
        if v2 < _EPS:
            if v1 < _EPS:
                return 1
            return max(1, int(np.ceil(v1 / threshold)))

        # With second order: solve v1/N + v2/N^2 = threshold
        # Rearranged:  threshold * N^2 - v1 * N - v2 = 0
        a, b, c = threshold, -v1, -v2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return 1
        n_root = (-b + np.sqrt(disc)) / (2.0 * a)
        return max(1, int(np.ceil(n_root)))

    # ------------------------------------------------------------------
    #  Higher-order bound
    # ------------------------------------------------------------------

    def higher_order_bound(
        self,
        theta_1: NDArray[np.floating],
        theta_2: NDArray[np.floating],
        order: int = 3,
    ) -> float:
        r"""Bound on higher-order correction norms assuming geometric decay.

        If the corrections decay geometrically,

        .. math::

            \|\Theta^{(k)}\|_{\mathrm{op}}
            \approx \|\Theta^{(1)}\|_{\mathrm{op}} \, r^{\,k-1}

        with :math:`r = \|\Theta^{(2)}\| / \|\Theta^{(1)}\|`, then

        .. math::

            \|\Theta^{(k)}\|_{\mathrm{op}}
            \le \|\Theta^{(1)}\|_{\mathrm{op}} \, r^{\,k-1}.

        Parameters
        ----------
        theta_1 : ndarray, shape (P, P)
        theta_2 : ndarray, shape (P, P)
        order : int
            Target order *k* ≥ 2 (default 3).

        Returns
        -------
        float
            Upper bound on :math:`\|\Theta^{(k)}\|_{\mathrm{op}}`.
        """
        if order < 2:
            raise ValueError(f"order must be >= 2, got {order}")

        theta_1 = np.asarray(theta_1, dtype=np.float64)
        theta_2 = np.asarray(theta_2, dtype=np.float64)

        norm1 = self._compute_operator_norm(theta_1)
        norm2 = self._compute_operator_norm(theta_2)

        if norm1 < _EPS:
            return 0.0

        r = norm2 / norm1
        return float(norm1 * r ** (order - 1))

    # ------------------------------------------------------------------
    #  Confidence stratification
    # ------------------------------------------------------------------

    def stratify_confidence(self, validity_ratio: float) -> ConfidenceLevel:
        """Map a validity ratio to a :class:`ConfidenceLevel`.

        Parameters
        ----------
        validity_ratio : float

        Returns
        -------
        ConfidenceLevel
        """
        if validity_ratio < self.high_threshold:
            return ConfidenceLevel.HIGH
        if validity_ratio < self.moderate_threshold:
            return ConfidenceLevel.MODERATE
        if validity_ratio < self.low_threshold:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.INVALID

    # ------------------------------------------------------------------
    #  Empirical breakdown detection
    # ------------------------------------------------------------------

    def detect_breakdown(
        self,
        ntk_measurements: Sequence[NDArray[np.floating]],
        widths: Sequence[int],
    ) -> Dict[str, object]:
        r"""Detect where the 1/N expansion breaks down from empirical data.

        Given NTK matrices measured at several widths, compute the
        per-width deviation from the infinite-width limit (first
        measurement assumed to be at the largest width) and locate the
        width at which the relative deviation exceeds the low threshold.

        The procedure:
        1.  Compute ΔΘ(N) = Θ(N) − Θ_∞  (Θ_∞ ≈ measurement at largest N).
        2.  For each N compute δ(N) = ‖ΔΘ(N)‖_op / ‖Θ_∞‖_op.
        3.  Fit δ(N) = α / N^β via log–log regression.
        4.  Solve α / N^β = low_threshold for the breakdown width.

        Parameters
        ----------
        ntk_measurements : sequence of ndarray
            Kernel matrices measured at each width.  Must be ordered
            consistently with *widths*.
        widths : sequence of int
            Corresponding network widths.

        Returns
        -------
        dict
            Keys: ``"breakdown_width"`` (int or None),
            ``"exponent"`` (float), ``"prefactor"`` (float),
            ``"relative_deviations"`` (list of float),
            ``"fit_r_squared"`` (float).
        """
        if len(ntk_measurements) != len(widths):
            raise ValueError(
                "ntk_measurements and widths must have the same length."
            )
        if len(ntk_measurements) < 3:
            raise ValueError(
                "At least three measurements are needed for breakdown "
                "detection."
            )

        widths_arr = np.asarray(widths, dtype=np.float64)
        # Use the measurement at the largest width as the reference.
        ref_idx = int(np.argmax(widths_arr))
        theta_inf = np.asarray(ntk_measurements[ref_idx], dtype=np.float64)
        norm_inf = self._compute_operator_norm(theta_inf)

        if norm_inf < _EPS:
            return {
                "breakdown_width": None,
                "exponent": np.nan,
                "prefactor": np.nan,
                "relative_deviations": [],
                "fit_r_squared": np.nan,
            }

        # Relative deviations
        deviations: List[float] = []
        valid_log_w: List[float] = []
        valid_log_d: List[float] = []
        for i, (meas, w) in enumerate(zip(ntk_measurements, widths)):
            if i == ref_idx:
                deviations.append(0.0)
                continue
            delta = np.asarray(meas, dtype=np.float64) - theta_inf
            d = float(self._compute_operator_norm(delta) / norm_inf)
            deviations.append(d)
            if d > _EPS:
                valid_log_w.append(np.log(float(w)))
                valid_log_d.append(np.log(d))

        # Log-log fit:  log δ = log α − β log N
        if len(valid_log_w) >= 2:
            log_w = np.array(valid_log_w)
            log_d = np.array(valid_log_d)
            coeffs = np.polyfit(log_w, log_d, deg=1)
            beta = float(-coeffs[0])
            log_alpha = float(coeffs[1])
            alpha = float(np.exp(log_alpha))

            # R² of fit
            predicted = np.polyval(coeffs, log_w)
            ss_res = float(np.sum((log_d - predicted) ** 2))
            ss_tot = float(np.sum((log_d - np.mean(log_d)) ** 2))
            r_squared = 1.0 - ss_res / (ss_tot + _EPS)

            # Breakdown: α / N^β = low_threshold
            if alpha > _EPS and beta > _EPS:
                n_break = (alpha / self.low_threshold) ** (1.0 / beta)
                breakdown_width: Optional[int] = max(
                    1, int(np.ceil(n_break))
                )
            else:
                breakdown_width = None
        else:
            alpha, beta, r_squared = np.nan, np.nan, np.nan
            breakdown_width = None

        return {
            "breakdown_width": breakdown_width,
            "exponent": beta,
            "prefactor": alpha,
            "relative_deviations": deviations,
            "fit_r_squared": r_squared,
        }

    # ------------------------------------------------------------------
    #  Diagnostic report
    # ------------------------------------------------------------------

    def diagnostic_report(self, result: ValidityResult) -> str:
        """Generate a human-readable diagnostic string.

        Parameters
        ----------
        result : ValidityResult

        Returns
        -------
        str
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  Perturbative Validity Diagnostic Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append(
            f"  Validity ratio  V[Θ]  = {result.validity_ratio:.6f}"
        )
        lines.append(
            f"  Confidence level       = {result.confidence.value}"
        )
        lines.append(
            f"  Spectral validity      = {result.spectral_validity:.6f}"
        )
        lines.append(
            f"  Entry-wise validity    = {result.entrywise_validity:.6f}"
        )
        lines.append("")

        # Convergence radius
        cr = result.convergence_radius
        lines.append("  Convergence radius")
        lines.append(f"    Method               = {cr.method}")
        lines.append(f"    Radius               = {cr.radius:.6f}")
        lines.append(
            f"    Est. breakdown width = {cr.estimated_breakdown_width}"
        )
        lines.append(f"    Confidence           = {cr.confidence:.4f}")
        lines.append("")

        # Correction magnitudes
        lines.append("  Correction magnitudes (operator norm)")
        for key, val in sorted(result.correction_magnitudes.items()):
            lines.append(f"    {key:20s} = {val:.6e}")
        lines.append("")

        # Warnings
        if result.warnings:
            lines.append("  Warnings")
            for w in result.warnings:
                lines.append(f"    ⚠  {w}")
            lines.append("")

        # Extra diagnostics
        if result.diagnostics:
            lines.append("  Diagnostics")
            for key, val in sorted(result.diagnostics.items()):
                lines.append(f"    {key:30s} = {val}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ==================================================================
    #  Private / helper methods
    # ==================================================================

    @staticmethod
    def _compute_operator_norm(M: NDArray[np.floating]) -> float:
        r"""Compute the operator (spectral) norm of a matrix.

        The operator norm equals the largest singular value:

        .. math::

            \|M\|_{\mathrm{op}} = \sigma_{\max}(M)

        Parameters
        ----------
        M : ndarray, shape (m, n)

        Returns
        -------
        float
        """
        M = np.asarray(M, dtype=np.float64)
        if M.ndim == 1:
            return float(np.linalg.norm(M, ord=2))
        svs = sla.svdvals(M)
        return float(svs[0]) if svs.size > 0 else 0.0

    def _eigenvalue_perturbation_bound(
        self,
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
        width: int,
    ) -> float:
        r"""Bauer–Fike bound on eigenvalue perturbation.

        For a diagonalisable matrix :math:`\Theta^{(0)} = V D V^{-1}`,
        the Bauer–Fike theorem gives

        .. math::

            \max_i |\tilde\lambda_i - \lambda_i|
            \;\le\; \kappa(V)\;\|\Delta\|_{\mathrm{op}}

        where :math:`\Delta = \Theta^{(1)} / N` and :math:`\kappa(V)` is
        the condition number of the eigenvector matrix.

        We return the *relative* bound normalised by
        :math:`\min|\lambda_i^{(0)}|` (excluding near-zero eigenvalues).

        Parameters
        ----------
        theta_0 : ndarray, shape (P, P)
        theta_1 : ndarray, shape (P, P)
        width : int

        Returns
        -------
        float
            Relative Bauer–Fike bound.  Values < 1 indicate that the
            eigenvalue perturbation is small relative to the spectral
            gap of the leading-order kernel.
        """
        theta_0 = np.asarray(theta_0, dtype=np.float64)
        theta_1 = np.asarray(theta_1, dtype=np.float64)

        perturbation = theta_1 / float(width)
        delta_norm = self._compute_operator_norm(perturbation)

        if self._is_symmetric(theta_0):
            # For symmetric matrices κ(V) = 1 (orthogonal eigenvectors).
            kappa = 1.0
            eig0 = np.abs(sla.eigvalsh(theta_0))
        else:
            eigvals, V = sla.eig(theta_0)
            eig0 = np.abs(eigvals)
            # Condition number of eigenvector matrix
            try:
                kappa = float(np.linalg.cond(V))
            except np.linalg.LinAlgError:
                kappa = np.inf

        nonzero = eig0[eig0 > _EPS]
        if nonzero.size == 0:
            return np.inf

        min_eig = float(np.min(nonzero))
        return float(kappa * delta_norm / min_eig)

    @staticmethod
    def _is_symmetric(
        M: NDArray[np.floating],
        rtol: float = 1e-8,
    ) -> bool:
        """Check whether *M* is symmetric (or Hermitian) within tolerance."""
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            return False
        return bool(np.allclose(M, M.T, rtol=rtol, atol=_EPS))
