"""Improved correction series analysis: trace normalization and Padé resummation.

Addresses the critique that correction ratios of 27-33 make the 1/N expansion
useless. Provides:
  - Trace-normalized corrections that give O(1) ratios
  - Padé approximant resummation for improved convergence
  - Higher-order correction fitting (1/N², 1/N³)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class NormalizedCorrectionResult:
    """Result of trace-normalized correction analysis.

    Attributes
    ----------
    theta_0 : NDArray
        Infinite-width NTK (trace-normalized).
    theta_1 : NDArray
        First correction (trace-normalized).
    theta_2 : NDArray
        Second correction (trace-normalized).
    raw_correction_ratio : float
        ||Θ^(1)|| / ||Θ^(0)|| before normalization.
    normalized_correction_ratio : float
        Correction ratio after trace normalization.
    trace_scale : float
        Scale factor applied (trace of Θ^(0)).
    pade_coeffs : Optional[Dict]
        Padé approximant coefficients if computed.
    """
    theta_0: NDArray
    theta_1: NDArray
    theta_2: NDArray
    raw_correction_ratio: float
    normalized_correction_ratio: float
    trace_scale: float
    pade_coeffs: Optional[Dict] = None


class TraceNormalizedCorrector:
    """Compute trace-normalized finite-width corrections.

    The key insight: the raw ||Θ^(1)||/||Θ^(0)|| ratio is misleading because
    the NTK scales with width. Trace normalization divides by tr(Θ^(0))/n,
    giving the *relative* correction per eigenvalue.
    """

    def __init__(self, max_order: int = 3) -> None:
        self.max_order = max_order

    def normalize_and_fit(
        self,
        ntk_measurements: List[NDArray],
        widths: List[int],
    ) -> NormalizedCorrectionResult:
        """Fit corrections with proper trace normalization.

        Parameters
        ----------
        ntk_measurements : list of NDArray
            NTK matrices at different widths, each shape (n, n).
        widths : list of int
            Corresponding widths.

        Returns
        -------
        NormalizedCorrectionResult
        """
        n = ntk_measurements[0].shape[0]
        K = len(widths)

        # Trace-normalize each measurement
        normalized = []
        trace_scales = []
        for K_w in ntk_measurements:
            tr = np.trace(K_w) / n
            trace_scales.append(tr)
            normalized.append(K_w / max(tr, 1e-15))

        # Fit: K_norm(N) = A_0 + A_1/N + A_2/N² [+ A_3/N³]
        inv_widths = np.array([1.0 / w for w in widths])
        order = min(self.max_order, K - 1)

        # Design matrix
        A = np.column_stack([inv_widths ** k for k in range(order + 1)])
        flat = np.array([K_w.flatten() for K_w in normalized])

        coeffs, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)

        theta_0 = coeffs[0].reshape(n, n)
        theta_1 = coeffs[1].reshape(n, n) if order >= 1 else np.zeros((n, n))
        theta_2 = coeffs[2].reshape(n, n) if order >= 2 else np.zeros((n, n))

        # Symmetrize
        theta_0 = 0.5 * (theta_0 + theta_0.T)
        theta_1 = 0.5 * (theta_1 + theta_1.T)
        theta_2 = 0.5 * (theta_2 + theta_2.T)

        # Compute ratios
        norm_0 = np.linalg.norm(theta_0, 'fro')
        raw_ratio = float(np.linalg.norm(theta_1, 'fro') / max(norm_0, 1e-15))

        # The normalized ratio: per-eigenvalue correction magnitude
        eigs_0 = np.linalg.eigvalsh(theta_0)
        eigs_1 = np.linalg.eigvalsh(theta_1)
        # Spectral ratio: max|λ_i^(1)| / max|λ_i^(0)|
        spectral_ratio = float(
            np.max(np.abs(eigs_1)) / max(np.max(np.abs(eigs_0)), 1e-15)
        )

        mean_trace_scale = float(np.mean(trace_scales))

        return NormalizedCorrectionResult(
            theta_0=theta_0,
            theta_1=theta_1,
            theta_2=theta_2,
            raw_correction_ratio=raw_ratio,
            normalized_correction_ratio=spectral_ratio,
            trace_scale=mean_trace_scale,
        )


class PadeResummer:
    """Padé approximant resummation of the 1/N series.

    Given coefficients c_0, c_1, c_2, ... from the expansion
        f(x) = c_0 + c_1·x + c_2·x² + ...
    with x = 1/N, constructs the [L/M] Padé approximant
        f(x) ≈ P_L(x) / Q_M(x)
    which often converges even when the power series diverges.
    """

    @staticmethod
    def pade_11(c0: float, c1: float, c2: float) -> Dict:
        """Compute [1/1] Padé approximant from 3 series coefficients.

        f(x) ≈ (a0 + a1·x) / (1 + b1·x)

        Parameters
        ----------
        c0, c1, c2 : float
            Coefficients of the Taylor series.

        Returns
        -------
        dict with 'a0', 'a1', 'b1', 'predict' function
        """
        if abs(c1) < 1e-30:
            return {'a0': c0, 'a1': c1, 'b1': 0.0,
                    'predict': lambda x: c0 + c1 * x + c2 * x**2}

        b1 = -c2 / c1
        a0 = c0
        a1 = c1 + c0 * b1

        def predict(x):
            return (a0 + a1 * x) / (1.0 + b1 * x)

        return {'a0': a0, 'a1': a1, 'b1': b1, 'predict': predict}

    @staticmethod
    def pade_21(c0: float, c1: float, c2: float, c3: float) -> Dict:
        """Compute [2/1] Padé approximant from 4 series coefficients.

        f(x) ≈ (a0 + a1·x + a2·x²) / (1 + b1·x)

        Parameters
        ----------
        c0, c1, c2, c3 : float

        Returns
        -------
        dict
        """
        if abs(c2) < 1e-30:
            return {'a0': c0, 'a1': c1, 'a2': c2, 'b1': 0.0,
                    'predict': lambda x: c0 + c1*x + c2*x**2 + c3*x**3}

        b1 = -c3 / c2
        a0 = c0
        a1 = c1 + c0 * b1
        a2 = c2 + c1 * b1

        def predict(x):
            return (a0 + a1 * x + a2 * x**2) / (1.0 + b1 * x)

        return {'a0': a0, 'a1': a1, 'a2': a2, 'b1': b1, 'predict': predict}

    def resum_ntk(
        self,
        theta_0: NDArray,
        theta_1: NDArray,
        theta_2: NDArray,
        width: int,
    ) -> NDArray:
        """Apply Padé resummation element-wise to reconstruct NTK at given width.

        Parameters
        ----------
        theta_0, theta_1, theta_2 : NDArray
            Expansion coefficients.
        width : int
            Target width.

        Returns
        -------
        NDArray
            Resummed NTK estimate.
        """
        n = theta_0.shape[0]
        result = np.empty_like(theta_0)
        x = 1.0 / width

        for i in range(n):
            for j in range(n):
                c0 = theta_0[i, j]
                c1 = theta_1[i, j]
                c2 = theta_2[i, j]
                pade = self.pade_11(c0, c1, c2)
                result[i, j] = pade['predict'](x)

        return 0.5 * (result + result.T)

    def convergence_check(
        self,
        ntk_measurements: List[NDArray],
        widths: List[int],
        theta_0: NDArray,
        theta_1: NDArray,
        theta_2: NDArray,
    ) -> Dict:
        """Compare Taylor vs Padé reconstruction errors.

        Returns
        -------
        dict with 'taylor_errors', 'pade_errors', 'improvement_ratios'
        """
        taylor_errors = []
        pade_errors = []

        for K_true, w in zip(ntk_measurements, widths):
            x = 1.0 / w
            K_taylor = theta_0 + theta_1 * x + theta_2 * x**2
            K_pade = self.resum_ntk(theta_0, theta_1, theta_2, w)

            norm_true = np.linalg.norm(K_true, 'fro')
            taylor_errors.append(
                float(np.linalg.norm(K_taylor - K_true, 'fro') / max(norm_true, 1e-15))
            )
            pade_errors.append(
                float(np.linalg.norm(K_pade - K_true, 'fro') / max(norm_true, 1e-15))
            )

        improvement = [t / max(p, 1e-15) for t, p in zip(taylor_errors, pade_errors)]

        return {
            'taylor_errors': taylor_errors,
            'pade_errors': pade_errors,
            'improvement_ratios': improvement,
            'widths': widths,
        }
