"""
usability_oracle.analysis.sensitivity — Sensitivity analysis framework.

Implements Sobol variance-based sensitivity indices (first-order, total-order),
Morris elementary effects screening, and Fourier Amplitude Sensitivity Test
(FAST) for quantifying how oracle outputs respond to parameter perturbations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SobolIndices:
    """Sobol sensitivity indices for all parameters.

    Attributes:
        first_order: S_i — fraction of variance attributable to parameter i alone.
        total_order: S_Ti — fraction of variance attributable to parameter i
            including all interactions.
        second_order: S_ij — pairwise interaction indices (optional).
        confidence_intervals: Bootstrap CIs for first-order indices.
        n_samples: Number of model evaluations used.
    """
    first_order: np.ndarray
    total_order: np.ndarray
    second_order: Optional[np.ndarray] = None
    confidence_intervals: Optional[np.ndarray] = None
    n_samples: int = 0
    parameter_names: list[str] = field(default_factory=list)

    def most_influential(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k most influential parameters by total-order index."""
        indices = np.argsort(-self.total_order)[:top_k]
        names = self.parameter_names or [f"x{i}" for i in range(len(self.total_order))]
        return [(names[i], float(self.total_order[i])) for i in indices]

    def summary(self) -> str:
        lines = [f"Sobol Sensitivity Analysis ({self.n_samples} evaluations)"]
        names = self.parameter_names or [f"x{i}" for i in range(len(self.first_order))]
        for i, name in enumerate(names):
            lines.append(
                f"  {name}: S1={self.first_order[i]:.4f}, ST={self.total_order[i]:.4f}"
            )
        return "\n".join(lines)


@dataclass
class MorrisResult:
    """Results from Morris elementary effects screening.

    Attributes:
        mu_star: Mean of absolute elementary effects (importance).
        sigma: Standard deviation of elementary effects (interaction/nonlinearity).
        mu: Mean of elementary effects (direction of influence).
        parameter_names: Names of the input parameters.
    """
    mu_star: np.ndarray
    sigma: np.ndarray
    mu: np.ndarray
    parameter_names: list[str] = field(default_factory=list)
    n_trajectories: int = 0

    def classify_parameters(
        self,
        mu_star_threshold: float = 0.1,
        sigma_ratio_threshold: float = 0.5,
    ) -> dict[str, list[str]]:
        """Classify parameters into negligible, linear, and nonlinear/interactive."""
        names = self.parameter_names or [f"x{i}" for i in range(len(self.mu_star))]
        negligible, linear, nonlinear = [], [], []
        max_mu = np.max(self.mu_star) if np.max(self.mu_star) > 0 else 1.0
        for i, name in enumerate(names):
            normed = self.mu_star[i] / max_mu
            if normed < mu_star_threshold:
                negligible.append(name)
            elif self.sigma[i] / max(self.mu_star[i], 1e-10) < sigma_ratio_threshold:
                linear.append(name)
            else:
                nonlinear.append(name)
        return {"negligible": negligible, "linear": linear, "nonlinear": nonlinear}


@dataclass
class FASTResult:
    """Results from Fourier Amplitude Sensitivity Test."""
    first_order: np.ndarray
    total_order: np.ndarray
    parameter_names: list[str] = field(default_factory=list)
    n_samples: int = 0


# ---------------------------------------------------------------------------
# Saltelli sampling for Sobol
# ---------------------------------------------------------------------------

def _saltelli_sample(
    n_samples: int,
    n_params: int,
    bounds: list[tuple[float, float]],
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Saltelli's sampling scheme for Sobol analysis.

    Returns matrices A and B, each of shape (n_samples, n_params),
    drawn from quasi-random sequences within the given bounds.
    """
    A = np.zeros((n_samples, n_params))
    B = np.zeros((n_samples, n_params))
    for j in range(n_params):
        lo, hi = bounds[j]
        A[:, j] = rng.uniform(lo, hi, n_samples)
        B[:, j] = rng.uniform(lo, hi, n_samples)
    return A, B


def _build_ab_matrices(
    A: np.ndarray,
    B: np.ndarray,
) -> list[np.ndarray]:
    """Build AB_i matrices for Sobol index estimation.

    AB_i is a copy of A with column i replaced by column i from B.
    """
    n_params = A.shape[1]
    matrices = []
    for i in range(n_params):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        matrices.append(AB_i)
    return matrices


# ---------------------------------------------------------------------------
# Morris trajectory generation
# ---------------------------------------------------------------------------

def _morris_trajectories(
    n_trajectories: int,
    n_params: int,
    n_levels: int,
    bounds: list[tuple[float, float]],
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    """Generate Morris trajectories (one-at-a-time design).

    Each trajectory has (n_params + 1) points where consecutive points
    differ in exactly one coordinate by delta = n_levels / (2*(n_levels-1)).
    """
    delta = n_levels / (2.0 * (n_levels - 1))
    trajectories = []

    for _ in range(n_trajectories):
        # Random base point on the level grid
        base = np.zeros(n_params)
        for j in range(n_params):
            lo, hi = bounds[j]
            level = rng.randint(0, n_levels)
            base[j] = lo + (hi - lo) * level / n_levels

        # Random permutation of parameter indices
        perm = rng.permutation(n_params)
        trajectory = np.zeros((n_params + 1, n_params))
        trajectory[0, :] = base.copy()

        for step, j in enumerate(perm):
            trajectory[step + 1, :] = trajectory[step, :].copy()
            lo, hi = bounds[j]
            perturbation = delta * (hi - lo)
            direction = rng.choice([-1, 1])
            new_val = trajectory[step + 1, j] + direction * perturbation
            # Reflect if out of bounds
            if new_val > hi:
                new_val = trajectory[step + 1, j] - perturbation
            if new_val < lo:
                new_val = lo + perturbation
            trajectory[step + 1, j] = np.clip(new_val, lo, hi)

        trajectories.append(trajectory)

    return trajectories


# ---------------------------------------------------------------------------
# FAST frequency assignment
# ---------------------------------------------------------------------------

def _fast_frequencies(n_params: int, max_freq: int = 64) -> np.ndarray:
    """Assign integer frequencies for FAST analysis.

    The i-th parameter gets a distinct frequency omega_i so that
    their Fourier spectra do not alias.
    """
    if n_params == 1:
        return np.array([1])
    omega = np.zeros(n_params, dtype=int)
    omega[0] = max_freq
    # Remaining frequencies are chosen to avoid aliasing
    for i in range(1, n_params):
        omega[i] = max(1, max_freq // (2 * i))
    return omega


def _fast_sample(
    n_samples: int,
    n_params: int,
    bounds: list[tuple[float, float]],
    omega: np.ndarray,
) -> np.ndarray:
    """Generate the FAST sampling matrix.

    Uses a sinusoidal search curve parametrized by s in [-pi, pi].
    """
    s = np.linspace(-math.pi, math.pi, n_samples, endpoint=False)
    X = np.zeros((n_samples, n_params))
    for j in range(n_params):
        lo, hi = bounds[j]
        # Transform via inverse CDF of uniform: x = lo + (hi-lo) * (arcsin(sin(omega_j*s))/pi + 0.5)
        X[:, j] = lo + (hi - lo) * (np.arcsin(np.sin(omega[j] * s)) / math.pi + 0.5)
    return X


# ---------------------------------------------------------------------------
# SensitivityAnalyzer
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """Multi-method sensitivity analysis for the usability oracle.

    Supports Sobol variance-based decomposition, Morris elementary effects,
    and Fourier Amplitude Sensitivity Test (FAST).

    Parameters:
        model_fn: Callable (parameter_vector -> scalar output).
        parameter_names: Names of the input parameters.
        bounds: List of (lower, upper) bounds for each parameter.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], float],
        parameter_names: list[str],
        bounds: list[tuple[float, float]],
        seed: int = 42,
    ) -> None:
        if len(parameter_names) != len(bounds):
            raise ValueError("parameter_names and bounds must have the same length")
        self._model_fn = model_fn
        self._names = parameter_names
        self._bounds = bounds
        self._n_params = len(parameter_names)
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Sobol indices (Saltelli 2002)
    # ------------------------------------------------------------------

    def sobol(
        self,
        n_samples: int = 1024,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
    ) -> SobolIndices:
        """Compute first-order and total-order Sobol indices.

        Uses Saltelli's estimator which requires N*(2k+2) model evaluations
        where k is the number of parameters and N is n_samples.
        """
        A, B = _saltelli_sample(n_samples, self._n_params, self._bounds, self._rng)
        AB_matrices = _build_ab_matrices(A, B)

        # Evaluate model on all sample matrices
        y_A = np.array([self._model_fn(A[i, :]) for i in range(n_samples)])
        y_B = np.array([self._model_fn(B[i, :]) for i in range(n_samples)])
        y_AB = []
        for mat in AB_matrices:
            y_AB.append(np.array([self._model_fn(mat[i, :]) for i in range(n_samples)]))

        total_evals = n_samples * (2 + self._n_params)

        # Variance of combined output
        y_all = np.concatenate([y_A, y_B])
        var_total = np.var(y_all)
        if var_total < 1e-15:
            return SobolIndices(
                first_order=np.zeros(self._n_params),
                total_order=np.zeros(self._n_params),
                n_samples=total_evals,
                parameter_names=self._names,
            )

        # First-order: S_i = V_i / V(Y) using Jansen estimator
        S1 = np.zeros(self._n_params)
        ST = np.zeros(self._n_params)
        for i in range(self._n_params):
            # First-order: Saltelli 2010 estimator
            S1[i] = float(np.mean(y_B * (y_AB[i] - y_A)) / var_total)
            # Total-order
            diff = y_A - y_AB[i]
            ST[i] = float(np.mean(diff ** 2) / (2.0 * var_total))

        S1 = np.clip(S1, 0.0, 1.0)
        ST = np.clip(ST, 0.0, 1.0)

        # Bootstrap confidence intervals for S1
        ci = None
        if n_bootstrap > 0:
            ci = np.zeros((self._n_params, 2))
            for i in range(self._n_params):
                boot_s1 = []
                for _ in range(n_bootstrap):
                    idx = self._rng.choice(n_samples, size=n_samples, replace=True)
                    v = np.var(np.concatenate([y_A[idx], y_B[idx]]))
                    if v > 1e-15:
                        s = float(np.mean(y_B[idx] * (y_AB[i][idx] - y_A[idx])) / v)
                        boot_s1.append(np.clip(s, 0.0, 1.0))
                if boot_s1:
                    alpha = (1 - confidence) / 2
                    ci[i, 0] = float(np.percentile(boot_s1, 100 * alpha))
                    ci[i, 1] = float(np.percentile(boot_s1, 100 * (1 - alpha)))

        # Second-order indices (pairwise)
        S2 = None
        if self._n_params <= 10:
            S2 = np.zeros((self._n_params, self._n_params))
            for i in range(self._n_params):
                for j in range(i + 1, self._n_params):
                    S2[i, j] = max(0.0, ST[i] + ST[j] - S1[i] - S1[j])
                    S2[j, i] = S2[i, j]

        return SobolIndices(
            first_order=S1,
            total_order=ST,
            second_order=S2,
            confidence_intervals=ci,
            n_samples=total_evals,
            parameter_names=self._names,
        )

    # ------------------------------------------------------------------
    # Morris screening (Morris 1991)
    # ------------------------------------------------------------------

    def morris(
        self,
        n_trajectories: int = 20,
        n_levels: int = 4,
    ) -> MorrisResult:
        """Compute Morris elementary effects for parameter screening.

        Efficient screening method requiring r*(k+1) evaluations where
        r is the number of trajectories and k is the number of parameters.
        """
        trajectories = _morris_trajectories(
            n_trajectories, self._n_params, n_levels, self._bounds, self._rng,
        )

        elementary_effects: list[list[float]] = [[] for _ in range(self._n_params)]

        for traj in trajectories:
            # Evaluate model at each point in the trajectory
            y_vals = np.array([self._model_fn(traj[step, :]) for step in range(traj.shape[0])])

            # Determine which parameter changed at each step
            for step in range(1, traj.shape[0]):
                diff = traj[step, :] - traj[step - 1, :]
                changed = np.nonzero(np.abs(diff) > 1e-12)[0]
                if len(changed) == 1:
                    j = changed[0]
                    delta_x = diff[j]
                    if abs(delta_x) > 1e-12:
                        ee = (y_vals[step] - y_vals[step - 1]) / delta_x
                        elementary_effects[j].append(ee)

        # Compute statistics
        mu = np.zeros(self._n_params)
        mu_star = np.zeros(self._n_params)
        sigma = np.zeros(self._n_params)

        for j in range(self._n_params):
            if elementary_effects[j]:
                ee = np.array(elementary_effects[j])
                mu[j] = float(np.mean(ee))
                mu_star[j] = float(np.mean(np.abs(ee)))
                sigma[j] = float(np.std(ee, ddof=1)) if len(ee) > 1 else 0.0

        return MorrisResult(
            mu_star=mu_star,
            sigma=sigma,
            mu=mu,
            parameter_names=self._names,
            n_trajectories=n_trajectories,
        )

    # ------------------------------------------------------------------
    # FAST (Cukier 1973, Saltelli 1999)
    # ------------------------------------------------------------------

    def fast(self, n_samples: int = 1024) -> FASTResult:
        """Compute first-order and total-order indices using FAST.

        Uses Fourier decomposition of the model output along a
        sinusoidal search curve in parameter space.
        """
        # Ensure n_samples is odd for symmetry
        if n_samples % 2 == 0:
            n_samples += 1

        omega = _fast_frequencies(self._n_params)
        X = _fast_sample(n_samples, self._n_params, self._bounds, omega)

        # Evaluate model
        y = np.array([self._model_fn(X[i, :]) for i in range(n_samples)])
        var_y = np.var(y)

        S1 = np.zeros(self._n_params)
        ST = np.zeros(self._n_params)

        if var_y < 1e-15:
            return FASTResult(
                first_order=S1, total_order=ST,
                parameter_names=self._names, n_samples=n_samples,
            )

        # Compute Fourier coefficients
        fft_y = np.fft.fft(y)
        power = np.abs(fft_y) ** 2 / n_samples

        for j in range(self._n_params):
            # First-order: sum power at harmonics of omega_j
            harm_power = 0.0
            for p in range(1, n_samples // (2 * max(omega[j], 1)) + 1):
                freq_idx = p * omega[j]
                if freq_idx < n_samples // 2:
                    harm_power += power[freq_idx]
            S1[j] = float(harm_power / (var_y * n_samples)) * 2.0

            # Total-order: complementary power
            comp_power = 0.0
            for k in range(self._n_params):
                if k == j:
                    continue
                for p in range(1, n_samples // (2 * max(omega[k], 1)) + 1):
                    freq_idx = p * omega[k]
                    if freq_idx < n_samples // 2:
                        comp_power += power[freq_idx]
            ST[j] = 1.0 - float(comp_power / (var_y * n_samples)) * 2.0

        S1 = np.clip(S1, 0.0, 1.0)
        ST = np.clip(ST, 0.0, 1.0)

        return FASTResult(
            first_order=S1, total_order=ST,
            parameter_names=self._names, n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Parameter importance ranking
    # ------------------------------------------------------------------

    def rank_parameters(
        self,
        method: str = "sobol",
        n_samples: int = 512,
    ) -> list[tuple[str, float]]:
        """Rank parameters by importance using the specified method.

        Parameters:
            method: One of "sobol", "morris", or "fast".
            n_samples: Number of base samples.

        Returns:
            List of (parameter_name, importance_score) sorted descending.
        """
        if method == "sobol":
            result = self.sobol(n_samples=n_samples, n_bootstrap=0)
            scores = result.total_order
        elif method == "morris":
            result = self.morris(n_trajectories=max(10, n_samples // (self._n_params + 1)))
            scores = result.mu_star
        elif method == "fast":
            result = self.fast(n_samples=n_samples)
            scores = result.total_order
        else:
            raise ValueError(f"Unknown method: {method}")

        names = self._names
        ranked = sorted(zip(names, scores.tolist()), key=lambda x: -x[1])
        return ranked

    # ------------------------------------------------------------------
    # One-at-a-time local sensitivity
    # ------------------------------------------------------------------

    def local_sensitivity(
        self,
        base_point: np.ndarray,
        delta: float = 0.01,
    ) -> dict[str, float]:
        """Compute local (derivative-based) sensitivity at a given point.

        Uses central finite differences: dF/dx_i ≈ (F(x+h) - F(x-h)) / (2h).
        """
        base = np.asarray(base_point, dtype=float)
        if len(base) != self._n_params:
            raise ValueError("base_point length must match number of parameters")

        y_base = self._model_fn(base)
        sensitivities: dict[str, float] = {}

        for j in range(self._n_params):
            lo, hi = self._bounds[j]
            h = delta * (hi - lo)
            if h < 1e-12:
                sensitivities[self._names[j]] = 0.0
                continue

            x_plus = base.copy()
            x_minus = base.copy()
            x_plus[j] = min(base[j] + h, hi)
            x_minus[j] = max(base[j] - h, lo)

            y_plus = self._model_fn(x_plus)
            y_minus = self._model_fn(x_minus)

            actual_h = x_plus[j] - x_minus[j]
            if abs(actual_h) > 1e-12:
                deriv = (y_plus - y_minus) / actual_h
                # Normalised sensitivity: (x/y) * dy/dx
                if abs(y_base) > 1e-12:
                    sensitivities[self._names[j]] = deriv * base[j] / y_base
                else:
                    sensitivities[self._names[j]] = deriv
            else:
                sensitivities[self._names[j]] = 0.0

        return sensitivities

    # ------------------------------------------------------------------
    # Scatter plots data generation
    # ------------------------------------------------------------------

    def scatter_data(
        self,
        n_samples: int = 500,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Generate scatter plot data (x_i vs y) for each parameter.

        Useful for visual inspection of parameter-output relationships.
        """
        X = np.zeros((n_samples, self._n_params))
        for j in range(self._n_params):
            lo, hi = self._bounds[j]
            X[:, j] = self._rng.uniform(lo, hi, n_samples)

        y = np.array([self._model_fn(X[i, :]) for i in range(n_samples)])

        data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for j in range(self._n_params):
            data[self._names[j]] = (X[:, j], y)
        return data
