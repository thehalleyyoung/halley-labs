"""Global sensitivity analysis (Sobol indices, Morris screening).

Provides variance-based (Sobol) and elementary-effects (Morris)
global sensitivity analyses to quantify how pipeline hyper-parameters
affect output quality metrics.

* :class:`SobolAnalyzer` – first-order and total-order Sobol indices
  via Saltelli's sampling scheme with bootstrap confidence intervals.
* :class:`MorrisScreening` – elementary-effects screening with
  mu*, sigma statistics for parameter ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# Dataclass
# ===================================================================


@dataclass
class GlobalSensitivityResult:
    """Container for global sensitivity analysis results.

    Attributes
    ----------
    parameter_names : list of str
        Names of the analysed parameters.
    first_order : NDArray
        First-order sensitivity indices.
    total_order : NDArray
        Total-order sensitivity indices.
    interactions : NDArray or None
        Second-order interaction indices (Sobol only).
    first_order_ci : NDArray or None
        Bootstrap confidence intervals for first-order indices.
    total_order_ci : NDArray or None
        Bootstrap confidence intervals for total-order indices.
    mu_star : NDArray or None
        Modified mean of absolute elementary effects (Morris only).
    sigma : NDArray or None
        Standard deviation of elementary effects (Morris only).
    """

    parameter_names: List[str]
    first_order: NDArray
    total_order: NDArray
    interactions: Optional[NDArray] = None
    first_order_ci: Optional[NDArray] = None
    total_order_ci: Optional[NDArray] = None
    mu_star: Optional[NDArray] = None
    sigma: Optional[NDArray] = None


# ===================================================================
# SobolAnalyzer
# ===================================================================


class SobolAnalyzer:
    """Variance-based Sobol sensitivity analysis.

    Uses Saltelli's extension of the Sobol' sequence to compute
    first-order and total-order sensitivity indices.

    Parameters
    ----------
    model_fn : callable
        Function mapping a parameter vector (1-D array of length *d*)
        to a scalar output.
    param_bounds : list of (float, float)
        Lower and upper bounds for each parameter.
    n_samples : int
        Number of base samples (total model evaluations = n*(2d+2)).
    param_names : list of str or None
        Names for each parameter.
    """

    def __init__(
        self,
        model_fn: Callable[..., float],
        param_bounds: List[Tuple[float, float]],
        n_samples: int = 1024,
        param_names: Optional[List[str]] = None,
    ) -> None:
        self._model_fn = model_fn
        self._bounds = list(param_bounds)
        self._n_samples = n_samples
        self._d = len(param_bounds)
        self._names = param_names or [f"p{i}" for i in range(self._d)]
        self._result: Optional[GlobalSensitivityResult] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def analyze(self, n_bootstrap: int = 100) -> GlobalSensitivityResult:
        """Run the Sobol analysis and return results.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap resamples for confidence intervals.

        Returns
        -------
        GlobalSensitivityResult
        """
        A, B, AB_list = self._saltelli_sampling(
            self._n_samples, self._d, self._bounds
        )

        Y_A = self._evaluate_batch(A)
        Y_B = self._evaluate_batch(B)
        Y_AB = np.column_stack(
            [self._evaluate_batch(ab) for ab in AB_list]
        )

        S1 = self._first_order_indices(Y_A, Y_B, Y_AB)
        ST = self._total_order_indices(Y_A, Y_B, Y_AB)

        S1_ci = self._confidence_intervals(Y_A, Y_B, Y_AB, "first", n_bootstrap)
        ST_ci = self._confidence_intervals(Y_A, Y_B, Y_AB, "total", n_bootstrap)

        self._result = GlobalSensitivityResult(
            parameter_names=self._names,
            first_order=S1,
            total_order=ST,
            interactions=None,
            first_order_ci=S1_ci,
            total_order_ci=ST_ci,
        )
        return self._result

    def first_order_indices(self) -> NDArray:
        """Return first-order Sobol indices."""
        if self._result is None:
            self.analyze()
        return self._result.first_order  # type: ignore[union-attr]

    def total_order_indices(self) -> NDArray:
        """Return total-order Sobol indices."""
        if self._result is None:
            self.analyze()
        return self._result.total_order  # type: ignore[union-attr]

    # -----------------------------------------------------------------
    # Saltelli sampling
    # -----------------------------------------------------------------

    @staticmethod
    def _saltelli_sampling(
        n: int,
        d: int,
        bounds: List[Tuple[float, float]],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[NDArray, NDArray, List[NDArray]]:
        """Generate Saltelli sample matrices.

        Produces two independent base matrices A and B of shape
        (n, d) and d cross-matrices AB_i where column i of A is
        replaced with column i of B.

        Parameters
        ----------
        n : int
            Number of base samples.
        d : int
            Number of parameters.
        bounds : list of (float, float)
            Parameter bounds.
        rng : Generator or None
            Random generator.

        Returns
        -------
        (A, B, AB_list) : tuple
        """
        rng = rng or np.random.default_rng()
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        A_unit = rng.random((n, d))
        B_unit = rng.random((n, d))

        A = lo + A_unit * (hi - lo)
        B = lo + B_unit * (hi - lo)

        AB_list: List[NDArray] = []
        for i in range(d):
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            AB_list.append(AB_i)

        return A, B, AB_list

    def _evaluate_batch(self, X: NDArray) -> NDArray:
        """Evaluate the model function on a batch of inputs."""
        return np.array([self._model_fn(x) for x in X], dtype=np.float64)

    # -----------------------------------------------------------------
    # Index computation
    # -----------------------------------------------------------------

    @staticmethod
    def _first_order_indices(
        Y_A: NDArray, Y_B: NDArray, Y_AB: NDArray
    ) -> NDArray:
        """Compute first-order Sobol indices.

        S_i = V[E[Y | X_i]] / V[Y]
            ≈ (1/n) Σ Y_B * (Y_AB_i - Y_A) / V[Y]

        Parameters
        ----------
        Y_A : NDArray, shape (n,)
        Y_B : NDArray, shape (n,)
        Y_AB : NDArray, shape (n, d)

        Returns
        -------
        NDArray, shape (d,)
        """
        n = len(Y_A)
        d = Y_AB.shape[1]
        f0 = np.mean(np.concatenate([Y_A, Y_B]))
        var_y = np.var(np.concatenate([Y_A, Y_B]))
        if var_y < 1e-15:
            return np.zeros(d, dtype=np.float64)

        S1 = np.empty(d, dtype=np.float64)
        for i in range(d):
            S1[i] = np.mean(Y_B * (Y_AB[:, i] - Y_A)) / var_y

        return S1

    @staticmethod
    def _total_order_indices(
        Y_A: NDArray, Y_B: NDArray, Y_AB: NDArray
    ) -> NDArray:
        """Compute total-order Sobol indices.

        ST_i = 1 - V[E[Y | X_{~i}]] / V[Y]
             ≈ (1/2n) Σ (Y_A - Y_AB_i)^2 / V[Y]

        Parameters
        ----------
        Y_A : NDArray, shape (n,)
        Y_B : NDArray, shape (n,)
        Y_AB : NDArray, shape (n, d)

        Returns
        -------
        NDArray, shape (d,)
        """
        n = len(Y_A)
        d = Y_AB.shape[1]
        var_y = np.var(np.concatenate([Y_A, Y_B]))
        if var_y < 1e-15:
            return np.zeros(d, dtype=np.float64)

        ST = np.empty(d, dtype=np.float64)
        for i in range(d):
            ST[i] = 0.5 * np.mean((Y_A - Y_AB[:, i]) ** 2) / var_y

        return ST

    def _confidence_intervals(
        self,
        Y_A: NDArray,
        Y_B: NDArray,
        Y_AB: NDArray,
        order: str,
        n_bootstrap: int,
    ) -> NDArray:
        """Bootstrap confidence intervals for Sobol indices.

        Parameters
        ----------
        order : str
            ``"first"`` or ``"total"``.
        n_bootstrap : int

        Returns
        -------
        NDArray, shape (d, 2) – lower and upper 95% CI.
        """
        n = len(Y_A)
        d = Y_AB.shape[1]
        rng = np.random.default_rng(42)
        boot_indices = np.empty((n_bootstrap, d), dtype=np.float64)

        compute = (
            self._first_order_indices
            if order == "first"
            else self._total_order_indices
        )

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_indices[b] = compute(Y_A[idx], Y_B[idx], Y_AB[idx])

        ci = np.column_stack(
            [
                np.percentile(boot_indices, 2.5, axis=0),
                np.percentile(boot_indices, 97.5, axis=0),
            ]
        )
        return ci


# ===================================================================
# MorrisScreening
# ===================================================================


class MorrisScreening:
    """Elementary-effects (Morris) screening method.

    Generates random one-at-a-time (OAT) trajectories through the
    parameter space and computes elementary effects for each parameter.

    Parameters
    ----------
    model_fn : callable
        Function mapping a parameter vector to a scalar output.
    param_bounds : list of (float, float)
        Lower and upper bounds for each parameter.
    n_trajectories : int
        Number of Morris trajectories.
    n_levels : int
        Number of discretisation levels per parameter.
    param_names : list of str or None
        Names for each parameter.
    """

    def __init__(
        self,
        model_fn: Callable[..., float],
        param_bounds: List[Tuple[float, float]],
        n_trajectories: int = 10,
        n_levels: int = 4,
        param_names: Optional[List[str]] = None,
    ) -> None:
        self._model_fn = model_fn
        self._bounds = list(param_bounds)
        self._n_traj = n_trajectories
        self._n_levels = n_levels
        self._d = len(param_bounds)
        self._names = param_names or [f"p{i}" for i in range(self._d)]
        self._effects: Optional[NDArray] = None
        self._result: Optional[GlobalSensitivityResult] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def analyze(self) -> GlobalSensitivityResult:
        """Run the Morris screening and return results.

        Returns
        -------
        GlobalSensitivityResult
        """
        trajectories = self._generate_trajectories(
            self._d, self._n_traj, self._n_levels
        )
        effects = self._elementary_effects(self._model_fn, trajectories)
        self._effects = effects

        mu_star = self._mu_star(effects)
        sigma = self._sigma(effects)

        normaliser = np.sum(mu_star)
        if normaliser < 1e-15:
            normaliser = 1.0
        first_order = mu_star / normaliser
        total_order = (mu_star + sigma) / (normaliser + np.sum(sigma))

        self._result = GlobalSensitivityResult(
            parameter_names=self._names,
            first_order=first_order,
            total_order=total_order,
            mu_star=mu_star,
            sigma=sigma,
        )
        return self._result

    def elementary_effects(self) -> NDArray:
        """Return elementary effects matrix.

        Returns
        -------
        NDArray
            Matrix of shape ``(n_trajectories, n_params)``.
        """
        if self._effects is None:
            self.analyze()
        return self._effects  # type: ignore[return-value]

    # -----------------------------------------------------------------
    # Trajectory generation
    # -----------------------------------------------------------------

    def _generate_trajectories(
        self,
        d: int,
        n_traj: int,
        n_levels: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[NDArray]:
        """Generate Morris trajectories in the unit hypercube.

        Each trajectory is a ``(d+1, d)`` matrix where consecutive
        rows differ in exactly one coordinate by ``delta = p / (2*(p-1))``
        with ``p = n_levels``.

        Parameters
        ----------
        d : int
            Number of parameters.
        n_traj : int
            Number of trajectories.
        n_levels : int
            Number of levels.
        rng : Generator or None

        Returns
        -------
        list of NDArray, each shape ``(d+1, d)``
        """
        rng = rng or np.random.default_rng()
        delta = n_levels / (2.0 * (n_levels - 1)) if n_levels > 1 else 0.5
        lo = np.array([b[0] for b in self._bounds], dtype=np.float64)
        hi = np.array([b[1] for b in self._bounds], dtype=np.float64)

        trajectories: List[NDArray] = []
        for _ in range(n_traj):
            levels = np.arange(n_levels, dtype=np.float64) / max(n_levels - 1, 1)
            x0 = np.array(
                [rng.choice(levels[: n_levels // 2 + 1]) for _ in range(d)],
                dtype=np.float64,
            )

            order = rng.permutation(d)
            traj = np.empty((d + 1, d), dtype=np.float64)
            traj[0] = x0.copy()
            current = x0.copy()
            for step, idx in enumerate(order):
                sign = rng.choice([-1.0, 1.0])
                new_val = current[idx] + sign * delta
                if new_val > 1.0:
                    new_val = current[idx] - delta
                if new_val < 0.0:
                    new_val = current[idx] + delta
                new_val = np.clip(new_val, 0.0, 1.0)
                current[idx] = new_val
                traj[step + 1] = current.copy()

            physical = lo + traj * (hi - lo)
            trajectories.append(physical)

        return trajectories

    def _elementary_effects(
        self,
        model_fn: Callable[..., float],
        trajectories: List[NDArray],
    ) -> NDArray:
        """Compute elementary effects from trajectories.

        Parameters
        ----------
        model_fn : callable
        trajectories : list of NDArray, each shape (d+1, d)

        Returns
        -------
        NDArray, shape (n_traj, d)
        """
        d = self._d
        lo = np.array([b[0] for b in self._bounds], dtype=np.float64)
        hi = np.array([b[1] for b in self._bounds], dtype=np.float64)
        ranges = hi - lo
        ranges[ranges < 1e-12] = 1.0

        effects = np.zeros((len(trajectories), d), dtype=np.float64)

        for t, traj in enumerate(trajectories):
            y_vals = np.array(
                [model_fn(traj[i]) for i in range(d + 1)], dtype=np.float64
            )
            for step in range(d):
                diff = traj[step + 1] - traj[step]
                changed = np.argmax(np.abs(diff))
                delta_physical = diff[changed]
                if abs(delta_physical) < 1e-15:
                    effects[t, changed] = 0.0
                else:
                    effects[t, changed] = (
                        (y_vals[step + 1] - y_vals[step]) / delta_physical * ranges[changed]
                    )

        return effects

    @staticmethod
    def _mu_star(effects: NDArray) -> NDArray:
        """Modified mean of absolute elementary effects.

        Parameters
        ----------
        effects : NDArray, shape (r, d)

        Returns
        -------
        NDArray, shape (d,)
        """
        return np.mean(np.abs(effects), axis=0)

    @staticmethod
    def _sigma(effects: NDArray) -> NDArray:
        """Standard deviation of elementary effects.

        Parameters
        ----------
        effects : NDArray, shape (r, d)

        Returns
        -------
        NDArray, shape (d,)
        """
        return np.std(effects, axis=0, ddof=1)
