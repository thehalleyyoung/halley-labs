"""Architecture-dependent universality classes for phase transitions.

Computes critical exponents for different architectures (MLP, CNN, ResNet)
and performs finite-size scaling analysis to determine whether they belong
to the same universality class.

The key idea: near the lazy-to-rich phase boundary, the order parameter
(NTK drift) scales as:

    Ψ(γ, N) = N^{β/ν} · F((γ - γ*) · N^{1/ν})

where β, ν are critical exponents and F is a universal scaling function.
Different universality classes have different (β, ν).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, curve_fit


@dataclass
class CriticalExponents:
    """Critical exponents for a phase transition.

    Attributes
    ----------
    nu : float
        Correlation length exponent (controls boundary sharpness with N).
    beta : float
        Order parameter exponent (Ψ ~ |γ - γ*|^β).
    gamma_star : float
        Critical coupling.
    fit_quality : float
        R² or chi² of the scaling collapse.
    architecture : str
        Architecture name.
    """
    nu: float
    beta: float
    gamma_star: float
    fit_quality: float
    architecture: str = "unknown"


@dataclass
class UniversalityResult:
    """Result of universality class analysis.

    Attributes
    ----------
    exponents : dict
        Architecture -> CriticalExponents.
    same_class : bool
        Whether all architectures belong to the same universality class.
    class_distance : float
        Distance between exponent vectors (0 = same class).
    scaling_functions : dict
        Architecture -> (x_scaled, y_scaled) for the scaling collapse.
    """
    exponents: Dict[str, CriticalExponents] = field(default_factory=dict)
    same_class: bool = False
    class_distance: float = float('inf')
    scaling_functions: Dict[str, Tuple[NDArray, NDArray]] = field(default_factory=dict)


class UniversalityAnalyzer:
    """Analyze universality classes of neural network phase transitions.

    Parameters
    ----------
    exponent_tolerance : float
        Maximum relative difference in exponents to declare same class.
    """

    def __init__(self, exponent_tolerance: float = 0.2) -> None:
        self.exponent_tolerance = exponent_tolerance

    def extract_critical_exponents(
        self,
        gammas: NDArray,
        order_params: Dict[int, NDArray],
        architecture: str = "unknown",
    ) -> CriticalExponents:
        """Extract critical exponents from multi-width order parameter data.

        Parameters
        ----------
        gammas : NDArray
            Array of coupling values γ.
        order_params : dict
            {width: array_of_order_parameter_values} for each width.
        architecture : str
            Architecture label.

        Returns
        -------
        CriticalExponents
        """
        widths = sorted(order_params.keys())

        # Step 1: Estimate γ* from the crossing point of curves at different N
        gamma_star = self._find_crossing_point(gammas, order_params)

        # Step 2: Extract ν from the width-dependence of the boundary sharpness
        nu = self._extract_nu(gammas, order_params, gamma_star)

        # Step 3: Extract β from the order parameter scaling at γ*
        beta = self._extract_beta(gammas, order_params, gamma_star)

        # Step 4: Compute scaling collapse quality
        quality = self._scaling_collapse_quality(
            gammas, order_params, gamma_star, nu, beta
        )

        return CriticalExponents(
            nu=nu,
            beta=beta,
            gamma_star=gamma_star,
            fit_quality=quality,
            architecture=architecture,
        )

    def compare_universality_classes(
        self,
        exponents_list: List[CriticalExponents],
    ) -> UniversalityResult:
        """Compare critical exponents across architectures.

        Parameters
        ----------
        exponents_list : list of CriticalExponents

        Returns
        -------
        UniversalityResult
        """
        result = UniversalityResult()
        for exp in exponents_list:
            result.exponents[exp.architecture] = exp

        if len(exponents_list) < 2:
            result.same_class = True
            result.class_distance = 0.0
            return result

        # Compute pairwise distances in (ν, β) space
        vectors = np.array([[e.nu, e.beta] for e in exponents_list])
        max_dist = 0.0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                rel_dist = np.linalg.norm(vectors[i] - vectors[j]) / (
                    np.linalg.norm(vectors[i]) + 1e-10
                )
                max_dist = max(max_dist, rel_dist)

        result.class_distance = max_dist
        result.same_class = max_dist < self.exponent_tolerance

        return result

    def finite_size_scaling_collapse(
        self,
        gammas: NDArray,
        order_params: Dict[int, NDArray],
        gamma_star: float,
        nu: float,
        beta: float,
    ) -> Tuple[NDArray, NDArray]:
        """Perform scaling collapse: plot N^{β/ν}·Ψ vs (γ-γ*)·N^{1/ν}.

        Parameters
        ----------
        gammas : NDArray
        order_params : dict {width: NDArray}
        gamma_star, nu, beta : float

        Returns
        -------
        x_collapsed, y_collapsed : NDArray
        """
        x_all = []
        y_all = []

        for N, psi in order_params.items():
            x_scaled = (gammas - gamma_star) * N ** (1.0 / max(nu, 0.01))
            y_scaled = psi * N ** (beta / max(nu, 0.01))
            x_all.append(x_scaled)
            y_all.append(y_scaled)

        return np.concatenate(x_all), np.concatenate(y_all)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _find_crossing_point(
        self,
        gammas: NDArray,
        order_params: Dict[int, NDArray],
    ) -> float:
        """Find γ* as the crossing point of order parameter curves."""
        widths = sorted(order_params.keys())
        if len(widths) < 2:
            # Single width: find inflection point
            psi = order_params[widths[0]]
            grad = np.gradient(psi, gammas)
            idx = np.argmax(np.abs(grad))
            return float(gammas[idx])

        # Find where curves at different widths cross
        crossings = []
        for i in range(len(widths) - 1):
            psi_lo = order_params[widths[i]]
            psi_hi = order_params[widths[i + 1]]
            diff = psi_lo - psi_hi
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            for sc in sign_changes:
                # Linear interpolation
                t = abs(diff[sc]) / (abs(diff[sc]) + abs(diff[sc + 1]) + 1e-15)
                gamma_cross = gammas[sc] + t * (gammas[sc + 1] - gammas[sc])
                crossings.append(gamma_cross)

        if crossings:
            return float(np.median(crossings))

        # Fallback: maximum gradient location
        psi = order_params[widths[-1]]
        grad = np.gradient(psi, gammas)
        return float(gammas[np.argmax(np.abs(grad))])

    def _extract_nu(
        self,
        gammas: NDArray,
        order_params: Dict[int, NDArray],
        gamma_star: float,
    ) -> float:
        """Extract ν from width-dependence of transition sharpness.

        Near γ*, the derivative dΨ/dγ|_{γ*} scales as N^{1/ν}.
        """
        widths = sorted(order_params.keys())
        slopes = []

        for N in widths:
            psi = order_params[N]
            grad = np.gradient(psi, gammas)
            # Interpolate gradient at γ*
            idx = np.searchsorted(gammas, gamma_star)
            idx = np.clip(idx, 1, len(gammas) - 2)
            slope_at_star = float(grad[idx])
            slopes.append((N, abs(slope_at_star)))

        if len(slopes) < 2 or all(s[1] < 1e-15 for s in slopes):
            return 1.0  # default

        log_N = np.log([s[0] for s in slopes if s[1] > 1e-15])
        log_slope = np.log([s[1] for s in slopes if s[1] > 1e-15])

        if len(log_N) < 2:
            return 1.0

        # Fit: log(slope) = (1/ν) * log(N) + const
        coeffs = np.polyfit(log_N, log_slope, 1)
        inv_nu = coeffs[0]

        if abs(inv_nu) < 0.01:
            return 1.0

        return float(1.0 / max(abs(inv_nu), 0.01))

    def _extract_beta(
        self,
        gammas: NDArray,
        order_params: Dict[int, NDArray],
        gamma_star: float,
    ) -> float:
        """Extract β from how the order parameter at γ* scales with N.

        Ψ(γ*, N) ~ N^{-β/ν}
        """
        widths = sorted(order_params.keys())
        vals = []

        for N in widths:
            psi = order_params[N]
            idx = np.searchsorted(gammas, gamma_star)
            idx = np.clip(idx, 0, len(psi) - 1)
            vals.append((N, abs(float(psi[idx]))))

        if len(vals) < 2 or all(v[1] < 1e-15 for v in vals):
            return 0.5

        log_N = np.log([v[0] for v in vals if v[1] > 1e-15])
        log_psi = np.log([v[1] for v in vals if v[1] > 1e-15])

        if len(log_N) < 2:
            return 0.5

        coeffs = np.polyfit(log_N, log_psi, 1)
        # slope = -β/ν, but we return β directly
        return float(max(abs(coeffs[0]), 0.01))

    def _scaling_collapse_quality(
        self,
        gammas: NDArray,
        order_params: Dict[int, NDArray],
        gamma_star: float,
        nu: float,
        beta: float,
    ) -> float:
        """Measure quality of scaling collapse via variance reduction.

        Returns R²-like metric: 1 - Var(residuals) / Var(data).
        """
        x_all, y_all = self.finite_size_scaling_collapse(
            gammas, order_params, gamma_star, nu, beta
        )

        if len(x_all) < 5:
            return 0.0

        # Sort and bin
        sort_idx = np.argsort(x_all)
        x_sorted = x_all[sort_idx]
        y_sorted = y_all[sort_idx]

        # Total variance
        total_var = np.var(y_sorted)
        if total_var < 1e-15:
            return 1.0

        # Binned variance (residual after collapsing)
        n_bins = min(20, len(x_sorted) // 3)
        if n_bins < 2:
            return 0.0

        bins = np.linspace(x_sorted[0], x_sorted[-1], n_bins + 1)
        residual_var = 0.0
        count = 0
        for i in range(n_bins):
            mask = (x_sorted >= bins[i]) & (x_sorted < bins[i + 1])
            if np.sum(mask) > 1:
                residual_var += np.var(y_sorted[mask]) * np.sum(mask)
                count += np.sum(mask)

        if count < 2:
            return 0.0

        residual_var /= count
        return float(1.0 - residual_var / total_var)

    def optimize_exponents(
        self,
        gammas: NDArray,
        order_params: Dict[int, NDArray],
        architecture: str = "unknown",
    ) -> CriticalExponents:
        """Optimize (γ*, ν, β) jointly for best scaling collapse.

        Parameters
        ----------
        gammas, order_params, architecture

        Returns
        -------
        CriticalExponents
        """
        # Initial estimates
        init = self.extract_critical_exponents(gammas, order_params, architecture)

        def objective(params):
            gs, nu, beta = params
            if nu < 0.01 or beta < 0.01:
                return 1e6
            quality = self._scaling_collapse_quality(
                gammas, order_params, gs, nu, beta
            )
            return -quality  # minimize negative quality

        result = minimize(
            objective,
            x0=[init.gamma_star, init.nu, init.beta],
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6},
        )

        if result.success:
            gs, nu, beta = result.x
            quality = -result.fun
        else:
            gs, nu, beta = init.gamma_star, init.nu, init.beta
            quality = init.fit_quality

        return CriticalExponents(
            nu=max(nu, 0.01),
            beta=max(beta, 0.01),
            gamma_star=gs,
            fit_quality=quality,
            architecture=architecture,
        )
