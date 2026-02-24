"""Phase boundary prediction via linearized bifurcation analysis.

Computes γ*(T, N) — the critical coupling at which the lazy-to-rich
transition occurs — using the NTK eigenspectrum and finite-width corrections.

The key formula is:

    γ*(T) = c / (T · μ_max_eff)

where μ_max_eff is the largest eigenvalue of the *NTK perturbation operator*
(not just the raw correction matrix), and c = log(drift_threshold / drift_floor).

The bug causing γ* = Infinity was that μ_max was computed as the max eigenvalue
of the raw correction Θ^(1), which can be negative or near-zero. The correct
approach uses the absolute spectral radius of the normalized perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class GammaStarResult:
    """Result of phase boundary prediction.

    Attributes
    ----------
    gamma_star : float
        Predicted critical coupling γ*.
    mu_max_eff : float
        Effective perturbation eigenvalue used in prediction.
    eigenvalues : NDArray
        Full spectrum of the perturbation operator.
    method : str
        Which method produced this prediction.
    """
    gamma_star: float
    mu_max_eff: float
    eigenvalues: NDArray
    method: str = "spectral"


class PhaseBoundaryPredictor:
    """Predict phase boundaries from NTK corrections.

    Parameters
    ----------
    drift_threshold : float
        NTK drift level defining the lazy-to-rich transition (default 0.1).
    drift_floor : float
        Baseline drift level at initialization (default 0.01).
    regularization : float
        Regularization for near-singular eigenvalues (default 1e-8).
    """

    def __init__(
        self,
        drift_threshold: float = 0.1,
        drift_floor: float = 0.01,
        regularization: float = 1e-8,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.drift_floor = drift_floor
        self.regularization = regularization
        self.c_theory = np.log(drift_threshold / drift_floor)

    def predict_gamma_star(
        self,
        theta_0: NDArray,
        theta_1: NDArray,
        T: int,
        width: int,
    ) -> GammaStarResult:
        """Predict γ* from the infinite-width NTK and its 1/N correction.

        The perturbation operator is A = Θ^(0)^{-1/2} · Θ^(1) · Θ^(0)^{-1/2},
        which captures how the correction distorts the kernel geometry.
        The effective coupling is μ_max = spectral_radius(A), and
        γ*(T) = c / (T · μ_max / n) where n is the data size.

        Parameters
        ----------
        theta_0 : NDArray
            Infinite-width NTK, shape (n, n).
        theta_1 : NDArray
            First-order correction, shape (n, n).
        T : int
            Number of training steps.
        width : int
            Network width (for scaling).

        Returns
        -------
        GammaStarResult
        """
        n = theta_0.shape[0]

        # Regularized eigendecomposition of Θ^(0)
        eigs_0, V_0 = np.linalg.eigh(theta_0)
        eigs_0 = np.maximum(np.abs(eigs_0), self.regularization)

        # Form the normalized perturbation: A = Θ^(0)^{-1/2} Θ^(1) Θ^(0)^{-1/2}
        inv_sqrt_eigs = 1.0 / np.sqrt(eigs_0)
        inv_sqrt_theta0 = V_0 @ np.diag(inv_sqrt_eigs) @ V_0.T
        A = inv_sqrt_theta0 @ theta_1 @ inv_sqrt_theta0

        # Eigenvalues of the perturbation operator
        eigs_A = np.linalg.eigvalsh(A)

        # Use spectral radius (max |eigenvalue|) — this is always positive
        mu_max_eff = float(np.max(np.abs(eigs_A)))

        if mu_max_eff < self.regularization:
            mu_max_eff = self.regularization

        # Per-sample effective eigenvalue
        mu_per_sample = mu_max_eff / n

        # γ*(T) = c / (T · μ_per_sample)
        gamma_star = self.c_theory / (T * mu_per_sample)

        return GammaStarResult(
            gamma_star=gamma_star,
            mu_max_eff=mu_max_eff,
            eigenvalues=eigs_A,
            method="spectral_normalized",
        )

    def predict_gamma_star_direct(
        self,
        ntk_measurements: List[NDArray],
        widths: List[int],
        T: int,
        target_width: int,
    ) -> GammaStarResult:
        """Predict γ* directly from empirical NTK measurements at multiple widths.

        Fits the 1/N expansion and then applies the spectral prediction.

        Parameters
        ----------
        ntk_measurements : list of NDArray
            NTK matrices at different widths.
        widths : list of int
            Corresponding network widths.
        T : int
            Training steps.
        target_width : int
            Width to predict for.

        Returns
        -------
        GammaStarResult
        """
        n = ntk_measurements[0].shape[0]
        K = len(widths)

        # Fit Θ(N) = Θ^(0) + Θ^(1)/N via least squares
        inv_widths = np.array([1.0 / w for w in widths])
        # Stack measurements: for each (i,j), fit K_{ij}(N) = a + b/N
        flat_measurements = np.array([K_w.flatten() for K_w in ntk_measurements])

        # Design matrix: [1, 1/N]
        A = np.column_stack([np.ones(K), inv_widths])
        # Solve for each entry
        coeffs, _, _, _ = np.linalg.lstsq(A, flat_measurements, rcond=None)

        theta_0 = coeffs[0].reshape(n, n)
        theta_1 = coeffs[1].reshape(n, n)

        # Symmetrize
        theta_0 = 0.5 * (theta_0 + theta_0.T)
        theta_1 = 0.5 * (theta_1 + theta_1.T)

        return self.predict_gamma_star(theta_0, theta_1, T, target_width)

    def predict_boundary_curve(
        self,
        theta_0: NDArray,
        theta_1: NDArray,
        T_values: NDArray,
        width: int,
    ) -> Tuple[NDArray, NDArray]:
        """Compute γ*(T) curve over a range of training durations.

        Parameters
        ----------
        theta_0 : NDArray
            Infinite-width NTK.
        theta_1 : NDArray
            First correction.
        T_values : NDArray
            Array of training step counts.
        width : int
            Network width.

        Returns
        -------
        T_values : NDArray
        gamma_star_values : NDArray
        """
        gamma_stars = np.empty_like(T_values, dtype=np.float64)
        for i, T in enumerate(T_values):
            result = self.predict_gamma_star(theta_0, theta_1, int(T), width)
            gamma_stars[i] = result.gamma_star
        return T_values, gamma_stars
