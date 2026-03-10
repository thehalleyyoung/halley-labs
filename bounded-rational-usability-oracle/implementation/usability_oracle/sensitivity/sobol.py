"""
usability_oracle.sensitivity.sobol — Sobol' variance-based sensitivity indices.

Implements Saltelli's sampling scheme for efficient estimation of first-order,
total-order, and second-order Sobol' sensitivity indices with bootstrap
confidence intervals and convergence monitoring.

References
----------
Saltelli, A. (2002). Making best use of model evaluations to compute
    sensitivity indices. Computer Physics Communications, 145(2), 280–297.
Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical
    models and their Monte Carlo estimates. Mathematics and Computers in
    Simulation, 55(1-3), 271–280.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.core.types import Interval
from usability_oracle.sensitivity.types import (
    ParameterRange,
    SensitivityConfig,
    SensitivityResult,
    SobolIndices,
)


# ═══════════════════════════════════════════════════════════════════════════
# Sobol' quasi-random sequence (direction numbers)
# ═══════════════════════════════════════════════════════════════════════════


def sobol_sequence(n_points: int, n_dims: int, seed: int = 0) -> NDArray[np.float64]:
    """Generate Sobol' low-discrepancy quasi-random sequence in [0, 1)^d.

    Uses the Gray-code based generation with Joe-Kuo direction numbers.
    Falls back to ``scipy.stats.qmc.Sobol`` when available.

    Parameters
    ----------
    n_points : int
        Number of points (rounded up to next power of 2 internally).
    n_dims : int
        Number of dimensions.
    seed : int
        Random seed for scrambling.

    Returns
    -------
    NDArray[np.float64]
        Array of shape ``(n_points, n_dims)`` with values in [0, 1).
    """
    try:
        from scipy.stats.qmc import Sobol as ScipySobol

        sampler = ScipySobol(d=n_dims, scramble=True, seed=seed)
        # Sobol requires power-of-2 sample count
        m = int(math.ceil(math.log2(max(n_points, 2))))
        points = sampler.random_base2(m)
        return points[:n_points]
    except ImportError:
        pass

    # Fallback: use scrambled Halton-like construction
    rng = np.random.default_rng(seed)
    points = np.zeros((n_points, n_dims), dtype=np.float64)
    for d in range(n_dims):
        base = _prime(d + 1)
        for i in range(n_points):
            points[i, d] = _radical_inverse(i, base)
    # Owen scrambling approximation
    shift = rng.random(n_dims)
    points = (points + shift[np.newaxis, :]) % 1.0
    return points


def _prime(n: int) -> int:
    """Return the n-th prime number (1-indexed, small n)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    if n <= len(primes):
        return primes[n - 1]
    # Extend via simple sieve for larger n
    candidate = primes[-1] + 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 2
    return primes[n - 1]


def _radical_inverse(i: int, base: int) -> float:
    """Van der Corput radical inverse in given base."""
    result = 0.0
    denom = 1.0
    n = i + 1  # avoid 0
    while n > 0:
        denom *= base
        n, remainder = divmod(n, base)
        result += remainder / denom
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Saltelli sampling scheme
# ═══════════════════════════════════════════════════════════════════════════


def saltelli_sample(
    n_samples: int,
    parameters: Sequence[ParameterRange],
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[NDArray[np.float64]]]:
    """Generate Saltelli's sampling matrices A, B, and A_B^(i).

    Total model evaluations = N * (2k + 2) where k = number of parameters.

    Parameters
    ----------
    n_samples : int
        Base sample count N.
    parameters : Sequence[ParameterRange]
        Parameter range specifications.
    seed : int
        Random seed.

    Returns
    -------
    A : NDArray
        Base sample matrix A, shape ``(N, k)``.
    B : NDArray
        Complementary sample matrix B, shape ``(N, k)``.
    AB_list : list of NDArray
        Cross-matrices A_B^(i) for each parameter, each shape ``(N, k)``.
    """
    k = len(parameters)
    # Generate 2k columns of quasi-random numbers
    raw = sobol_sequence(n_samples, 2 * k, seed=seed)

    A_unit = raw[:, :k]
    B_unit = raw[:, k:]

    # Scale to parameter ranges
    lows = np.array([p.interval.low for p in parameters], dtype=np.float64)
    highs = np.array([p.interval.high for p in parameters], dtype=np.float64)

    A = lows + A_unit * (highs - lows)
    B = lows + B_unit * (highs - lows)

    # Cross-matrices: A_B^(i) = A with i-th column replaced by B's
    AB_list: List[NDArray[np.float64]] = []
    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        AB_list.append(AB_i)

    return A, B, AB_list


def _evaluate_matrix(
    model_fn: Callable[..., float],
    matrix: NDArray[np.float64],
    param_names: Sequence[str],
) -> NDArray[np.float64]:
    """Evaluate model_fn on each row of a parameter matrix.

    Parameters
    ----------
    model_fn : Callable
        Model function accepting keyword arguments.
    matrix : NDArray
        Parameter matrix of shape ``(N, k)``.
    param_names : Sequence[str]
        Parameter names corresponding to columns.

    Returns
    -------
    NDArray[np.float64]
        Model outputs of shape ``(N,)``.
    """
    n = matrix.shape[0]
    outputs = np.empty(n, dtype=np.float64)
    for i in range(n):
        kwargs = {name: float(matrix[i, j]) for j, name in enumerate(param_names)}
        outputs[i] = model_fn(**kwargs)
    return outputs


# ═══════════════════════════════════════════════════════════════════════════
# Sobol' index computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_first_order(
    y_A: NDArray[np.float64],
    y_B: NDArray[np.float64],
    y_AB_i: NDArray[np.float64],
) -> float:
    """Compute first-order Sobol' index S_i using Jansen (1999) estimator.

    S_i = (Var[Y] - 0.5 * mean((y_B - y_AB_i)^2)) / Var[Y]

    Parameters
    ----------
    y_A : NDArray
        Model outputs for matrix A.
    y_B : NDArray
        Model outputs for matrix B.
    y_AB_i : NDArray
        Model outputs for cross-matrix A_B^(i).

    Returns
    -------
    float
        First-order Sobol' index S_i.
    """
    n = len(y_A)
    total_var = np.var(np.concatenate([y_A, y_B]), ddof=1)
    if total_var < 1e-30:
        return 0.0

    # Saltelli 2010 estimator
    Vi = np.mean(y_B * (y_AB_i - y_A))
    return float(Vi / total_var)


def compute_total_order(
    y_A: NDArray[np.float64],
    y_B: NDArray[np.float64],
    y_AB_i: NDArray[np.float64],
) -> float:
    """Compute total-order Sobol' index S_Ti using Jansen (1999) estimator.

    S_Ti = 0.5 * mean((y_A - y_AB_i)^2) / Var[Y]

    Parameters
    ----------
    y_A : NDArray
        Model outputs for matrix A.
    y_B : NDArray
        Model outputs for matrix B.
    y_AB_i : NDArray
        Model outputs for cross-matrix A_B^(i).

    Returns
    -------
    float
        Total-order Sobol' index S_Ti.
    """
    total_var = np.var(np.concatenate([y_A, y_B]), ddof=1)
    if total_var < 1e-30:
        return 0.0

    VTi = 0.5 * np.mean((y_A - y_AB_i) ** 2)
    return float(VTi / total_var)


def compute_second_order(
    y_A: NDArray[np.float64],
    y_B: NDArray[np.float64],
    y_AB_i: NDArray[np.float64],
    y_AB_j: NDArray[np.float64],
    S_i: float,
    S_j: float,
) -> float:
    """Compute second-order Sobol' interaction index S_ij.

    S_ij = V_ij / Var[Y] - S_i - S_j

    Uses the estimator from Saltelli (2002).

    Parameters
    ----------
    y_A, y_B : NDArray
        Base and complementary outputs.
    y_AB_i, y_AB_j : NDArray
        Cross-matrix outputs for parameters i and j.
    S_i, S_j : float
        First-order indices for parameters i and j.

    Returns
    -------
    float
        Second-order interaction index S_ij.
    """
    total_var = np.var(np.concatenate([y_A, y_B]), ddof=1)
    if total_var < 1e-30:
        return 0.0

    Vij = np.mean(y_AB_i * y_AB_j) - np.mean(y_A) ** 2
    Sij_raw = Vij / total_var - S_i - S_j
    return float(Sij_raw)


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════════


def bootstrap_sobol_ci(
    y_A: NDArray[np.float64],
    y_B: NDArray[np.float64],
    y_AB_i: NDArray[np.float64],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[Interval, Interval]:
    """Bootstrap confidence intervals for first-order and total-order indices.

    Parameters
    ----------
    y_A, y_B, y_AB_i : NDArray
        Model outputs from Saltelli scheme.
    confidence_level : float
        Confidence level (e.g. 0.95).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    Tuple[Interval, Interval]
        (CI for S_i, CI for S_Ti).
    """
    rng = np.random.default_rng(seed)
    n = len(y_A)
    alpha = 1.0 - confidence_level

    first_orders = np.empty(n_bootstrap, dtype=np.float64)
    total_orders = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        first_orders[b] = compute_first_order(y_A[idx], y_B[idx], y_AB_i[idx])
        total_orders[b] = compute_total_order(y_A[idx], y_B[idx], y_AB_i[idx])

    fo_low, fo_high = float(np.percentile(first_orders, 100 * alpha / 2)), \
                      float(np.percentile(first_orders, 100 * (1 - alpha / 2)))
    to_low, to_high = float(np.percentile(total_orders, 100 * alpha / 2)), \
                      float(np.percentile(total_orders, 100 * (1 - alpha / 2)))

    return Interval(fo_low, fo_high), Interval(to_low, to_high)


# ═══════════════════════════════════════════════════════════════════════════
# Convergence monitoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ConvergenceRecord:
    """Track convergence of Sobol' indices across sample sizes."""

    sample_sizes: List[int] = field(default_factory=list)
    first_order_history: Dict[str, List[float]] = field(default_factory=dict)
    total_order_history: Dict[str, List[float]] = field(default_factory=dict)

    def is_converged(self, tolerance: float = 0.02) -> bool:
        """Check if all indices have stabilised within tolerance.

        Compares the last two recorded values for each index.
        """
        for name, history in self.total_order_history.items():
            if len(history) < 2:
                return False
            if abs(history[-1] - history[-2]) > tolerance:
                return False
        return True

    def relative_change(self, param_name: str) -> float:
        """Relative change in total-order index for the last step."""
        history = self.total_order_history.get(param_name, [])
        if len(history) < 2:
            return float("inf")
        prev = history[-2]
        if abs(prev) < 1e-15:
            return abs(history[-1] - prev)
        return abs((history[-1] - prev) / prev)


def monitor_convergence(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    sample_schedule: Sequence[int] = (64, 128, 256, 512, 1024),
    seed: int = 42,
) -> ConvergenceRecord:
    """Compute Sobol' indices at increasing sample sizes to track convergence.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    sample_schedule : Sequence[int]
        Increasing sample sizes.
    seed : int
        Random seed.

    Returns
    -------
    ConvergenceRecord
        History of indices at each sample size.
    """
    record = ConvergenceRecord()
    param_names = [p.name for p in parameters]

    for n_samples in sample_schedule:
        A, B, AB_list = saltelli_sample(n_samples, parameters, seed=seed)
        y_A = _evaluate_matrix(model_fn, A, param_names)
        y_B = _evaluate_matrix(model_fn, B, param_names)

        record.sample_sizes.append(n_samples)

        for i, p in enumerate(parameters):
            y_AB_i = _evaluate_matrix(model_fn, AB_list[i], param_names)
            si = compute_first_order(y_A, y_B, y_AB_i)
            sti = compute_total_order(y_A, y_B, y_AB_i)

            record.first_order_history.setdefault(p.name, []).append(si)
            record.total_order_history.setdefault(p.name, []).append(sti)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# Parameter importance ranking
# ═══════════════════════════════════════════════════════════════════════════


def rank_parameters_by_total_order(
    indices: Sequence[SobolIndices],
) -> List[Tuple[str, float]]:
    """Rank parameters by total-order Sobol' index (descending).

    Parameters
    ----------
    indices : Sequence[SobolIndices]
        Sobol' results for each parameter.

    Returns
    -------
    List[Tuple[str, float]]
        (parameter_name, S_Ti) pairs sorted by decreasing S_Ti.
    """
    ranked = [(idx.parameter_name, idx.total_order) for idx in indices]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def identify_interactions(
    indices: Sequence[SobolIndices],
    threshold: float = 0.01,
) -> List[Tuple[str, str, float]]:
    """Identify significant second-order interactions.

    Parameters
    ----------
    indices : Sequence[SobolIndices]
        Sobol' results including second-order indices.
    threshold : float
        Minimum |S_ij| to report.

    Returns
    -------
    List[Tuple[str, str, float]]
        (param_i, param_j, S_ij) triples with |S_ij| >= threshold.
    """
    interactions: List[Tuple[str, str, float]] = []
    seen: set[Tuple[str, str]] = set()
    for idx in indices:
        for other_name, sij in idx.second_order.items():
            pair = tuple(sorted([idx.parameter_name, other_name]))
            if pair not in seen and abs(sij) >= threshold:
                seen.add(pair)  # type: ignore[arg-type]
                interactions.append((pair[0], pair[1], sij))
    interactions.sort(key=lambda x: abs(x[2]), reverse=True)
    return interactions


# ═══════════════════════════════════════════════════════════════════════════
# SobolAnalyzer — main entry point
# ═══════════════════════════════════════════════════════════════════════════


class SobolAnalyzer:
    """Sobol' variance-based global sensitivity analysis.

    Implements the ``GlobalSensitivity`` protocol using Saltelli's
    sampling scheme with bootstrap confidence intervals.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples for CIs.
    compute_second_order : bool
        Whether to compute pairwise interaction indices.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        compute_second_order_flag: bool = True,
    ) -> None:
        self._n_bootstrap = n_bootstrap
        self._compute_second_order = compute_second_order_flag

    def compute_sobol(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        n_samples: int = 1024,
        *,
        seed: int = 42,
        confidence_level: float = 0.95,
    ) -> List[SobolIndices]:
        """Compute Sobol' indices for all parameters.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function with keyword arguments for each parameter.
        parameters : Sequence[ParameterRange]
            Parameter specifications.
        n_samples : int
            Base sample count N.
        seed : int
            Random seed.
        confidence_level : float
            Bootstrap confidence level.

        Returns
        -------
        List[SobolIndices]
            Sobol' indices for each parameter.
        """
        k = len(parameters)
        param_names = [p.name for p in parameters]

        A, B, AB_list = saltelli_sample(n_samples, parameters, seed=seed)
        y_A = _evaluate_matrix(model_fn, A, param_names)
        y_B = _evaluate_matrix(model_fn, B, param_names)

        # Evaluate cross-matrices
        y_AB: List[NDArray[np.float64]] = []
        for i in range(k):
            y_AB.append(_evaluate_matrix(model_fn, AB_list[i], param_names))

        # First-order and total-order indices
        first_orders = [compute_first_order(y_A, y_B, y_AB[i]) for i in range(k)]
        total_orders = [compute_total_order(y_A, y_B, y_AB[i]) for i in range(k)]

        # Bootstrap CIs
        cis = [
            bootstrap_sobol_ci(
                y_A, y_B, y_AB[i],
                confidence_level=confidence_level,
                n_bootstrap=self._n_bootstrap,
                seed=seed + i,
            )
            for i in range(k)
        ]

        # Second-order interactions
        second_orders: List[Dict[str, float]] = [{} for _ in range(k)]
        if self._compute_second_order and k > 1:
            for i in range(k):
                for j in range(i + 1, k):
                    sij = compute_second_order(
                        y_A, y_B, y_AB[i], y_AB[j],
                        first_orders[i], first_orders[j],
                    )
                    second_orders[i][param_names[j]] = sij
                    second_orders[j][param_names[i]] = sij

        results: List[SobolIndices] = []
        for i in range(k):
            results.append(SobolIndices(
                parameter_name=param_names[i],
                first_order=first_orders[i],
                total_order=total_orders[i],
                first_order_ci=cis[i][0],
                total_order_ci=cis[i][1],
                second_order=second_orders[i],
            ))
        return results

    def total_variance(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        n_samples: int = 1024,
        *,
        seed: int = 42,
    ) -> float:
        """Estimate total output variance via Monte Carlo.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameter specifications.
        n_samples : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        float
            Estimated total variance.
        """
        param_names = [p.name for p in parameters]
        samples = sobol_sequence(n_samples, len(parameters), seed=seed)
        lows = np.array([p.interval.low for p in parameters])
        highs = np.array([p.interval.high for p in parameters])
        scaled = lows + samples * (highs - lows)
        outputs = _evaluate_matrix(model_fn, scaled, param_names)
        return float(np.var(outputs, ddof=1))

    def analyze(
        self,
        model_fn: Callable[..., float],
        config: SensitivityConfig,
    ) -> SensitivityResult:
        """Run full Sobol' sensitivity analysis.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        config : SensitivityConfig
            Analysis configuration.

        Returns
        -------
        SensitivityResult
            Aggregate result with Sobol' indices.
        """
        indices = self.compute_sobol(
            model_fn,
            config.parameters,
            n_samples=config.n_samples,
            seed=config.seed,
            confidence_level=config.confidence_level,
        )
        var = self.total_variance(
            model_fn, config.parameters,
            n_samples=config.n_samples, seed=config.seed,
        )
        # Estimate mean
        param_names = [p.name for p in config.parameters]
        samples = sobol_sequence(config.n_samples, config.n_parameters, seed=config.seed)
        lows = np.array([p.interval.low for p in config.parameters])
        highs = np.array([p.interval.high for p in config.parameters])
        scaled = lows + samples * (highs - lows)
        outputs = _evaluate_matrix(model_fn, scaled, param_names)
        mean_out = float(np.mean(outputs))

        k = config.n_parameters
        n_evals = config.n_samples * (2 * k + 2)

        return SensitivityResult(
            config=config,
            output_name=config.output_names[0] if config.output_names else "",
            sobol_indices=tuple(indices),
            total_variance=var,
            mean_output=mean_out,
            n_evaluations=n_evals,
            metadata={
                "method": "sobol",
                "n_bootstrap": self._n_bootstrap,
                "compute_second_order": self._compute_second_order,
            },
        )


__all__ = [
    "SobolAnalyzer",
    "bootstrap_sobol_ci",
    "compute_first_order",
    "compute_second_order",
    "compute_total_order",
    "identify_interactions",
    "monitor_convergence",
    "rank_parameters_by_total_order",
    "saltelli_sample",
    "sobol_sequence",
    "ConvergenceRecord",
]
