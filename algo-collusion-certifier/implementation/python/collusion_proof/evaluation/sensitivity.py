"""Sensitivity analysis for CollusionProof.

Implements one-at-a-time (OAT), Latin Hypercube Sampling (LHS),
Sobol-style variance decomposition, Morris elementary effects screening,
and tornado diagram data generation.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple


class SensitivityAnalyzer:
    """Perform sensitivity analysis on detection parameters.

    Parameters
    ----------
    base_params : dict mapping parameter names to their default (centre) values.
    """

    def __init__(self, base_params: Dict[str, float]) -> None:
        self.base_params = dict(base_params)

    # ------------------------------------------------------------------
    # One-At-a-Time (OAT) sweep
    # ------------------------------------------------------------------

    def one_at_a_time(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objective: Callable[..., float],
        n_points: int = 20,
    ) -> Dict[str, Any]:
        """Vary each parameter independently while holding others at base.

        Parameters
        ----------
        param_ranges : {name: (lo, hi)} for each parameter to sweep.
        objective : callable(**params) -> float.
        n_points : number of evaluation points per parameter.

        Returns
        -------
        Dict with keys ``param_name`` → dict with ``values``, ``outputs``,
        ``sensitivity`` (range of output / range of parameter).
        """
        results: Dict[str, Any] = {}

        for name, (lo, hi) in param_ranges.items():
            grid = np.linspace(lo, hi, n_points)
            outputs = np.empty(n_points)
            for i, val in enumerate(grid):
                params = dict(self.base_params)
                params[name] = float(val)
                outputs[i] = objective(**params)

            output_range = float(np.max(outputs) - np.min(outputs))
            param_range = hi - lo
            sensitivity = output_range / param_range if param_range > 0 else 0.0

            results[name] = {
                "values": grid.tolist(),
                "outputs": outputs.tolist(),
                "sensitivity": sensitivity,
                "output_range": output_range,
                "base_value": self.base_params.get(name, 0.0),
                "base_output": float(objective(**self.base_params)),
            }

        return results

    # ------------------------------------------------------------------
    # Latin Hypercube Sampling
    # ------------------------------------------------------------------

    def latin_hypercube(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objective: Callable[..., float],
        n_samples: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Space-filling Latin Hypercube sample of the parameter space.

        Returns
        -------
        Dict with ``samples`` (n_samples × n_params array), ``outputs``,
        ``correlations`` (Spearman rank correlation of each param with output).
        """
        rng = np.random.RandomState(seed)
        names = sorted(param_ranges.keys())
        n_params = len(names)

        # Generate LHS: for each dimension, draw a random permutation
        # then perturb within each stratum.
        lhs = np.zeros((n_samples, n_params))
        for j in range(n_params):
            perm = rng.permutation(n_samples)
            uniform = (perm + rng.uniform(size=n_samples)) / n_samples
            lo, hi = param_ranges[names[j]]
            lhs[:, j] = lo + (hi - lo) * uniform

        outputs = np.empty(n_samples)
        for i in range(n_samples):
            params = dict(self.base_params)
            for j, name in enumerate(names):
                params[name] = float(lhs[i, j])
            outputs[i] = objective(**params)

        # Spearman rank correlations
        correlations: Dict[str, float] = {}
        for j, name in enumerate(names):
            rank_x = _rank_array(lhs[:, j])
            rank_y = _rank_array(outputs)
            corr = _pearson(rank_x, rank_y)
            correlations[name] = corr

        return {
            "param_names": names,
            "samples": lhs.tolist(),
            "outputs": outputs.tolist(),
            "correlations": correlations,
            "n_samples": n_samples,
        }

    # ------------------------------------------------------------------
    # Sobol-like variance decomposition
    # ------------------------------------------------------------------

    def sobol_indices(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objective: Callable[..., float],
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Estimate first-order and total-effect Sobol sensitivity indices.

        Uses the Saltelli (2010) sampling scheme with a Sobol-like
        quasi-random design approximated via LHS.

        Returns
        -------
        Dict with ``first_order`` and ``total_effect`` dicts mapping param
        names to their index values.
        """
        rng = np.random.RandomState(seed)
        names = sorted(param_ranges.keys())
        d = len(names)

        def _sample_matrix(n: int) -> np.ndarray:
            mat = np.zeros((n, d))
            for j in range(d):
                lo, hi = param_ranges[names[j]]
                mat[:, j] = lo + (hi - lo) * rng.uniform(size=n)
            return mat

        A = _sample_matrix(n_samples)
        B = _sample_matrix(n_samples)

        def _evaluate(mat: np.ndarray) -> np.ndarray:
            out = np.empty(mat.shape[0])
            for i in range(mat.shape[0]):
                params = dict(self.base_params)
                for j, name in enumerate(names):
                    params[name] = float(mat[i, j])
                out[i] = objective(**params)
            return out

        f_A = _evaluate(A)
        f_B = _evaluate(B)
        total_var = float(np.var(np.concatenate([f_A, f_B])))
        if total_var < 1e-30:
            # Output is essentially constant
            first_order = {name: 0.0 for name in names}
            total_effect = {name: 0.0 for name in names}
            return {
                "first_order": first_order,
                "total_effect": total_effect,
                "total_variance": total_var,
            }

        first_order: Dict[str, float] = {}
        total_effect: Dict[str, float] = {}

        for j, name in enumerate(names):
            # A_B^(j): take A but replace column j with B's column j
            AB_j = A.copy()
            AB_j[:, j] = B[:, j]
            f_AB_j = _evaluate(AB_j)

            # First-order: V_j ≈ (1/N) Σ f_B * (f_AB_j - f_A)
            si = float(np.mean(f_B * (f_AB_j - f_A)) / total_var)
            first_order[name] = float(np.clip(si, 0.0, 1.0))

            # Total effect: ST_j ≈ (1/2N) Σ (f_A - f_AB_j)^2
            st = float(np.mean((f_A - f_AB_j) ** 2) / (2.0 * total_var))
            total_effect[name] = float(np.clip(st, 0.0, 1.0))

        return {
            "first_order": first_order,
            "total_effect": total_effect,
            "total_variance": total_var,
        }

    # ------------------------------------------------------------------
    # Morris elementary effects screening
    # ------------------------------------------------------------------

    def morris_screening(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objective: Callable[..., float],
        n_trajectories: int = 10,
        n_levels: int = 4,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Morris (1991) elementary effects screening method.

        Parameters
        ----------
        n_trajectories : number of random trajectories through the grid.
        n_levels : number of levels for the discretisation grid.

        Returns
        -------
        Dict mapping param names to ``mu_star`` (mean absolute EE) and
        ``sigma`` (std of EE), plus the raw elementary effects.
        """
        rng = np.random.RandomState(seed)
        names = sorted(param_ranges.keys())
        d = len(names)
        delta = n_levels / (2 * (n_levels - 1)) if n_levels > 1 else 0.5

        elementary_effects: Dict[str, List[float]] = {n: [] for n in names}

        for _ in range(n_trajectories):
            # Start from a random base point on the grid
            x0 = np.zeros(d)
            for j in range(d):
                lo, hi = param_ranges[names[j]]
                level = rng.randint(0, n_levels)
                x0[j] = lo + (hi - lo) * level / max(n_levels - 1, 1)

            # Evaluate base point
            params_0 = dict(self.base_params)
            for j, name in enumerate(names):
                params_0[name] = float(x0[j])
            f_base = objective(**params_0)

            # Perturb one parameter at a time in a random order
            order = rng.permutation(d)
            x_current = x0.copy()
            f_current = f_base

            for j in order:
                lo, hi = param_ranges[names[j]]
                step = delta * (hi - lo)
                direction = 1 if rng.random() < 0.5 else -1
                x_new = x_current.copy()
                x_new[j] = np.clip(x_current[j] + direction * step, lo, hi)

                params_new = dict(self.base_params)
                for k, name in enumerate(names):
                    params_new[name] = float(x_new[k])
                f_new = objective(**params_new)

                actual_delta = x_new[j] - x_current[j]
                if abs(actual_delta) > 1e-30:
                    ee = (f_new - f_current) / actual_delta
                else:
                    ee = 0.0
                elementary_effects[names[j]].append(ee)

                x_current = x_new
                f_current = f_new

        # Summarise
        result: Dict[str, Any] = {}
        for name in names:
            ees = np.asarray(elementary_effects[name])
            result[name] = {
                "mu_star": float(np.mean(np.abs(ees))) if len(ees) > 0 else 0.0,
                "mu": float(np.mean(ees)) if len(ees) > 0 else 0.0,
                "sigma": float(np.std(ees, ddof=1)) if len(ees) > 1 else 0.0,
                "n_effects": len(ees),
                "elementary_effects": ees.tolist(),
            }

        return result

    # ------------------------------------------------------------------
    # Tornado diagram data
    # ------------------------------------------------------------------

    def tornado_diagram_data(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objective: Callable[..., float],
    ) -> Dict[str, Tuple[float, float]]:
        """Generate tornado diagram data.

        For each parameter, evaluate the objective at the low and high
        ends of its range while holding all others at base.

        Returns
        -------
        Dict mapping parameter name to ``(output_at_low, output_at_high)``.
        """
        tornado: Dict[str, Tuple[float, float]] = {}

        for name, (lo, hi) in param_ranges.items():
            params_lo = dict(self.base_params)
            params_lo[name] = lo
            out_lo = objective(**params_lo)

            params_hi = dict(self.base_params)
            params_hi[name] = hi
            out_hi = objective(**params_hi)

            tornado[name] = (float(out_lo), float(out_hi))

        return tornado


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_array(x: np.ndarray) -> np.ndarray:
    """Assign ranks to array values (average-rank for ties)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Handle ties with average ranking
    for i in range(n):
        tied = np.where(x == x[i])[0]
        if len(tied) > 1:
            avg_rank = np.mean(ranks[tied])
            ranks[tied] = avg_rank
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between two arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    if sx < 1e-30 or sy < 1e-30:
        return 0.0
    return float(np.sum((x - mx) * (y - my)) / ((n - 1) * sx * sy))
