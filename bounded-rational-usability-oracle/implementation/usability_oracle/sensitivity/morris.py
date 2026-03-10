"""
usability_oracle.sensitivity.morris — Morris elementary effects screening.

Implements the Morris one-at-a-time (OAT) screening method for identifying
influential, non-linear, and interacting parameters in the cognitive cost
model with optimised trajectory design.

References
----------
Morris, M. D. (1991). Factorial sampling plans for preliminary computational
    experiments. Technometrics, 33(2), 161–174.
Campolongo, F., Cariboni, J., & Saltelli, A. (2007). An effective screening
    design for sensitivity analysis of large models. Environmental Modelling
    & Software, 22(10), 1509–1518.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.sensitivity.types import (
    MorrisResult,
    ParameterRange,
    SensitivityConfig,
    SensitivityResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# Parameter classification
# ═══════════════════════════════════════════════════════════════════════════


class ParameterEffect(enum.Enum):
    """Classification of a parameter's effect on model output."""

    NEGLIGIBLE = "negligible"
    LINEAR = "linear"
    NONLINEAR_OR_INTERACTING = "nonlinear_or_interacting"


def classify_parameter(result: MorrisResult, mu_star_threshold: float = 0.1) -> ParameterEffect:
    """Classify a parameter based on Morris screening results.

    Decision rules (Campolongo et al., 2007):
    - μ* < threshold → negligible
    - μ* >= threshold and σ/μ* < 0.5 → linear (additive) effect
    - μ* >= threshold and σ/μ* >= 0.5 → nonlinear or interaction effect

    Parameters
    ----------
    result : MorrisResult
        Morris screening result for one parameter.
    mu_star_threshold : float
        Absolute threshold for μ* below which the parameter is negligible.

    Returns
    -------
    ParameterEffect
        Classification of the parameter's influence.
    """
    if result.mu_star < mu_star_threshold:
        return ParameterEffect.NEGLIGIBLE
    if result.sigma_over_mu_star < 0.5:
        return ParameterEffect.LINEAR
    return ParameterEffect.NONLINEAR_OR_INTERACTING


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory generation
# ═══════════════════════════════════════════════════════════════════════════


def _generate_trajectory(
    k: int,
    p: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate one Morris trajectory in the unit hypercube.

    A trajectory is a (k+1) × k matrix where each successive row differs
    from its predecessor in exactly one coordinate by ±Δ, where Δ = p/(2(p-1)).

    Parameters
    ----------
    k : int
        Number of parameters.
    p : int
        Number of grid levels (typically 4 or 6).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    NDArray[np.float64]
        Trajectory matrix of shape ``(k+1, k)``.
    """
    delta = p / (2.0 * (p - 1))

    # Base point on the grid: each coordinate is a random grid level from
    # the lower half so that +delta stays in [0, 1].
    grid_values = np.arange(0, p) / (p - 1)
    lower_half = grid_values[grid_values + delta <= 1.0]
    if len(lower_half) == 0:
        lower_half = np.array([0.0])
    base = rng.choice(lower_half, size=k)

    # Build the orientation matrix B*
    B_star = np.zeros((k + 1, k), dtype=np.float64)
    B_star[0, :] = base

    # Random permutation of parameter indices
    perm = rng.permutation(k)
    # Random sign for each perturbation
    signs = rng.choice([-1.0, 1.0], size=k)

    for step in range(k):
        B_star[step + 1, :] = B_star[step, :]
        idx = perm[step]
        new_val = B_star[step + 1, idx] + signs[step] * delta
        # Clamp to [0, 1]
        new_val = np.clip(new_val, 0.0, 1.0)
        # If clamping didn't change, flip direction
        if new_val == B_star[step, idx]:
            new_val = B_star[step + 1, idx] - signs[step] * delta
            new_val = np.clip(new_val, 0.0, 1.0)
        B_star[step + 1, idx] = new_val

    return B_star


def _trajectory_distance(t1: NDArray[np.float64], t2: NDArray[np.float64]) -> float:
    """Sum of pairwise Euclidean distances between trajectory points."""
    dist = 0.0
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            dist += float(np.linalg.norm(t1[i] - t2[j]))
    return dist


def optimized_trajectories(
    k: int,
    n_trajectories: int,
    p: int = 4,
    n_candidates: int = 0,
    seed: int = 42,
) -> List[NDArray[np.float64]]:
    """Generate optimised Morris trajectories maximising spread.

    Generates a larger candidate set and selects the subset with maximum
    pairwise distance (greedy algorithm from Campolongo et al., 2007).

    Parameters
    ----------
    k : int
        Number of parameters.
    n_trajectories : int
        Desired number of trajectories.
    p : int
        Grid levels.
    n_candidates : int
        Size of candidate pool (default: max(4 * n_trajectories, 50)).
    seed : int
        Random seed.

    Returns
    -------
    List[NDArray[np.float64]]
        Selected trajectories, each of shape ``(k+1, k)``.
    """
    rng = np.random.default_rng(seed)
    if n_candidates <= 0:
        n_candidates = max(4 * n_trajectories, 50)

    # Generate candidate pool
    candidates = [_generate_trajectory(k, p, rng) for _ in range(n_candidates)]

    if n_candidates <= n_trajectories:
        return candidates[:n_trajectories]

    # Greedy selection: start with the pair that has maximum distance
    n_cand = len(candidates)
    dist_matrix = np.zeros((n_cand, n_cand), dtype=np.float64)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            d = _trajectory_distance(candidates[i], candidates[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Pick the pair with maximum distance
    flat_idx = int(np.argmax(dist_matrix))
    best_i, best_j = divmod(flat_idx, n_cand)
    selected_idx = [best_i, best_j]

    # Greedily add trajectories
    while len(selected_idx) < n_trajectories:
        best_score = -1.0
        best_candidate = -1
        for c in range(n_cand):
            if c in selected_idx:
                continue
            min_dist = min(dist_matrix[c, s] for s in selected_idx)
            if min_dist > best_score:
                best_score = min_dist
                best_candidate = c
        if best_candidate < 0:
            break
        selected_idx.append(best_candidate)

    return [candidates[i] for i in selected_idx]


# ═══════════════════════════════════════════════════════════════════════════
# Elementary effect computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_elementary_effects(
    model_fn: Callable[..., float],
    trajectory: NDArray[np.float64],
    parameters: Sequence[ParameterRange],
) -> Dict[str, float]:
    """Compute elementary effects from a single Morris trajectory.

    For each step in the trajectory, identify which parameter changed
    and compute EE_i = (f(x') - f(x)) / Δ_i (scaled).

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function accepting keyword arguments.
    trajectory : NDArray[np.float64]
        Unit-hypercube trajectory of shape ``(k+1, k)``.
    parameters : Sequence[ParameterRange]
        Parameter specifications.

    Returns
    -------
    Dict[str, float]
        Parameter name → elementary effect for this trajectory.
    """
    k = len(parameters)
    param_names = [p.name for p in parameters]
    lows = np.array([p.interval.low for p in parameters], dtype=np.float64)
    highs = np.array([p.interval.high for p in parameters], dtype=np.float64)
    widths = highs - lows

    # Scale trajectory to parameter space
    scaled = lows + trajectory * widths

    # Evaluate model at each trajectory point
    outputs = np.empty(k + 1, dtype=np.float64)
    for row in range(k + 1):
        kwargs = {name: float(scaled[row, j]) for j, name in enumerate(param_names)}
        outputs[row] = model_fn(**kwargs)

    # Identify changed parameter and compute EE for each step
    effects: Dict[str, float] = {}
    for step in range(k):
        diff = trajectory[step + 1] - trajectory[step]
        changed = np.nonzero(diff)[0]
        if len(changed) == 0:
            continue
        idx = int(changed[0])
        delta_unit = diff[idx]
        if abs(delta_unit) < 1e-15:
            continue
        # EE normalised by parameter range
        ee = (outputs[step + 1] - outputs[step]) / (delta_unit * widths[idx])
        effects[param_names[idx]] = float(ee) * widths[idx]

    return effects


# ═══════════════════════════════════════════════════════════════════════════
# MorrisAnalyzer — main entry point
# ═══════════════════════════════════════════════════════════════════════════


class MorrisAnalyzer:
    """Morris elementary effects screening method.

    Identifies influential, linear, and nonlinear/interacting parameters
    using optimised trajectory design.

    Parameters
    ----------
    p : int
        Number of grid levels (default 4).
    n_candidates_factor : int
        Candidate pool multiplier for trajectory optimisation.
    """

    def __init__(self, p: int = 4, n_candidates_factor: int = 4) -> None:
        self._p = p
        self._n_candidates_factor = n_candidates_factor

    def screening(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        n_trajectories: int = 10,
        *,
        seed: int = 42,
    ) -> List[MorrisResult]:
        """Run Morris elementary-effects screening.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameter specifications.
        n_trajectories : int
            Number of Morris trajectories.
        seed : int
            Random seed.

        Returns
        -------
        List[MorrisResult]
            Morris results for each parameter.
        """
        k = len(parameters)
        trajectories = optimized_trajectories(
            k, n_trajectories, p=self._p,
            n_candidates=self._n_candidates_factor * n_trajectories,
            seed=seed,
        )

        # Collect elementary effects per parameter
        all_effects: Dict[str, List[float]] = {p.name: [] for p in parameters}

        for traj in trajectories:
            ee = compute_elementary_effects(model_fn, traj, parameters)
            for name, effect in ee.items():
                all_effects[name].append(effect)

        results: List[MorrisResult] = []
        for p_range in parameters:
            effects = all_effects[p_range.name]
            if not effects:
                results.append(MorrisResult(
                    parameter_name=p_range.name,
                    mu_star=0.0, mu=0.0, sigma=0.0,
                    n_trajectories=n_trajectories,
                    elementary_effects=(),
                ))
                continue

            arr = np.array(effects, dtype=np.float64)
            mu = float(np.mean(arr))
            mu_star = float(np.mean(np.abs(arr)))
            sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

            results.append(MorrisResult(
                parameter_name=p_range.name,
                mu_star=mu_star,
                mu=mu,
                sigma=sigma,
                n_trajectories=n_trajectories,
                elementary_effects=tuple(effects),
            ))

        return results

    def classify_all(
        self,
        results: Sequence[MorrisResult],
        mu_star_threshold: Optional[float] = None,
    ) -> Dict[str, ParameterEffect]:
        """Classify all parameters from Morris screening results.

        Parameters
        ----------
        results : Sequence[MorrisResult]
            Morris results for each parameter.
        mu_star_threshold : float, optional
            Custom threshold for negligible classification.
            Defaults to 10% of the maximum μ*.

        Returns
        -------
        Dict[str, ParameterEffect]
            Parameter name → effect classification.
        """
        if not results:
            return {}

        if mu_star_threshold is None:
            max_mu_star = max(r.mu_star for r in results)
            mu_star_threshold = 0.1 * max_mu_star if max_mu_star > 0 else 0.01

        return {
            r.parameter_name: classify_parameter(r, mu_star_threshold)
            for r in results
        }

    def analyze(
        self,
        model_fn: Callable[..., float],
        config: SensitivityConfig,
    ) -> SensitivityResult:
        """Run Morris analysis using a SensitivityConfig.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        config : SensitivityConfig
            Analysis configuration.

        Returns
        -------
        SensitivityResult
            Aggregate result with Morris screening results.
        """
        morris_results = self.screening(
            model_fn,
            config.parameters,
            n_trajectories=config.n_samples,
            seed=config.seed,
        )

        classifications = self.classify_all(morris_results)
        k = config.n_parameters
        n_evals = config.n_samples * (k + 1)

        return SensitivityResult(
            config=config,
            output_name=config.output_names[0] if config.output_names else "",
            morris_results=tuple(morris_results),
            n_evaluations=n_evals,
            metadata={
                "method": "morris",
                "grid_levels": self._p,
                "classifications": {
                    name: eff.value for name, eff in classifications.items()
                },
            },
        )


__all__ = [
    "MorrisAnalyzer",
    "ParameterEffect",
    "classify_parameter",
    "compute_elementary_effects",
    "optimized_trajectories",
]
