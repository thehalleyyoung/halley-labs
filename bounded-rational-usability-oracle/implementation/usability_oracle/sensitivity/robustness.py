"""
usability_oracle.sensitivity.robustness — Robustness analysis.

Computes parameter robustness regions, worst-case (minimax) analysis,
probabilistic robustness certificates, vertex analysis, and scenario-based
robustness assessment for cognitive model usability verdicts.

References
----------
Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009). Robust Optimization.
    Princeton University Press.
Calafiore, G. C., & Campi, M. C. (2006). The scenario approach to robust
    control design. IEEE Transactions on Automatic Control, 51(5), 742–753.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as iterproduct
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.core.types import Interval
from usability_oracle.sensitivity.types import (
    ParameterRange,
    SensitivityConfig,
    SensitivityResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class RobustnessRegion:
    """Region of parameter space where a usability verdict is stable.

    Attributes
    ----------
    parameter_radii : Mapping[str, float]
        Maximum perturbation radius for each parameter around nominal.
    nominal_verdict : float
        Model output at the nominal parameter values.
    region_volume : float
        Approximate hyper-volume of the robustness region.
    is_bounded : bool
        Whether the region is bounded (i.e. verdict changes within range).
    """

    parameter_radii: Mapping[str, float]
    nominal_verdict: float
    region_volume: float = 0.0
    is_bounded: bool = True


@dataclass(frozen=True, slots=True)
class RobustnessCertificate:
    """Certificate quantifying verdict robustness under parameter uncertainty.

    Attributes
    ----------
    probability_correct : float
        Estimated probability that the verdict holds under parameter uncertainty.
    worst_case_output : float
        Worst-case model output over the uncertainty set.
    best_case_output : float
        Best-case model output over the uncertainty set.
    critical_parameters : tuple[str, ...]
        Parameters most responsible for robustness degradation.
    n_scenarios : int
        Number of scenarios evaluated.
    """

    probability_correct: float
    worst_case_output: float
    best_case_output: float
    critical_parameters: Tuple[str, ...] = ()
    n_scenarios: int = 0


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Result of a single robustness scenario evaluation.

    Attributes
    ----------
    parameter_values : Mapping[str, float]
        Parameter values for this scenario.
    output : float
        Model output.
    verdict_matches_nominal : bool
        Whether the verdict matches the nominal verdict.
    """

    parameter_values: Mapping[str, float]
    output: float
    verdict_matches_nominal: bool


# ═══════════════════════════════════════════════════════════════════════════
# Robustness region computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_robustness_region(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    verdict_threshold: float = 0.0,
    n_bisection: int = 20,
) -> RobustnessRegion:
    """Find the largest hyperrectangular region around nominal where verdict holds.

    For each parameter, performs bisection search to find the maximum
    perturbation radius such that the model output stays on the same
    side of verdict_threshold.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications with nominal values.
    verdict_threshold : float
        Threshold for verdict (output >= threshold ↔ "usable").
    n_bisection : int
        Number of bisection steps per parameter.

    Returns
    -------
    RobustnessRegion
        The computed robustness region.
    """
    nominal = {p.name: p.nominal for p in parameters}
    f0 = model_fn(**nominal)
    verdict_positive = f0 >= verdict_threshold

    radii: Dict[str, float] = {}

    for p in parameters:
        max_radius = p.range_width / 2.0
        # Search in both positive and negative directions
        min_safe_radius = max_radius
        for direction in [1.0, -1.0]:
            lo, hi = 0.0, max_radius
            for _ in range(n_bisection):
                mid = (lo + hi) / 2.0
                test = dict(nominal)
                test[p.name] = np.clip(
                    nominal[p.name] + direction * mid,
                    p.interval.low, p.interval.high,
                )
                f_test = model_fn(**test)
                if (f_test >= verdict_threshold) == verdict_positive:
                    lo = mid  # safe at this radius
                else:
                    hi = mid  # verdict flipped
            min_safe_radius = min(min_safe_radius, lo)
        radii[p.name] = min_safe_radius

    volume = 1.0
    for r in radii.values():
        volume *= 2.0 * r  # diameter in each dimension

    is_bounded = any(r < p.range_width / 2.0 - 1e-10 for r, p in zip(radii.values(), parameters))

    return RobustnessRegion(
        parameter_radii=radii,
        nominal_verdict=f0,
        region_volume=volume,
        is_bounded=is_bounded,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Worst-case (minimax) analysis
# ═══════════════════════════════════════════════════════════════════════════


def worst_case_analysis(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    n_samples: int = 1000,
    seed: int = 42,
    minimize: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Find worst-case output over the parameter uncertainty set.

    Uses random multi-start with local refinement (coordinate descent).

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications (uncertainty set = parameter ranges).
    n_samples : int
        Number of random starting points.
    seed : int
        Random seed.
    minimize : bool
        If True, find minimum (worst-case for "lower is worse").
        If False, find maximum.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (worst_case_output, worst_case_parameters).
    """
    rng = np.random.default_rng(seed)
    k = len(parameters)
    param_names = [p.name for p in parameters]
    lows = np.array([p.interval.low for p in parameters])
    highs = np.array([p.interval.high for p in parameters])

    best_val = float("inf") if minimize else float("-inf")
    best_params: Dict[str, float] = {}

    # Random sampling phase
    samples = lows + rng.random((n_samples, k)) * (highs - lows)
    for i in range(n_samples):
        kwargs = {name: float(samples[i, j]) for j, name in enumerate(param_names)}
        val = model_fn(**kwargs)
        if (minimize and val < best_val) or (not minimize and val > best_val):
            best_val = val
            best_params = dict(kwargs)

    # Coordinate descent refinement from best point
    current = np.array([best_params[name] for name in param_names])
    n_refine = 5
    for _ in range(n_refine):
        for j in range(k):
            # Search along dimension j
            test_vals = np.linspace(lows[j], highs[j], 20)
            local_best = current[j]
            local_best_val = best_val
            for tv in test_vals:
                trial = current.copy()
                trial[j] = tv
                kwargs = {name: float(trial[jj]) for jj, name in enumerate(param_names)}
                fv = model_fn(**kwargs)
                if (minimize and fv < local_best_val) or (not minimize and fv > local_best_val):
                    local_best_val = fv
                    local_best = tv
            current[j] = local_best
            best_val = local_best_val

    best_params = {name: float(current[j]) for j, name in enumerate(param_names)}
    return best_val, best_params


# ═══════════════════════════════════════════════════════════════════════════
# Probabilistic robustness
# ═══════════════════════════════════════════════════════════════════════════


def probabilistic_robustness(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    verdict_threshold: float = 0.0,
    n_samples: int = 10000,
    seed: int = 42,
) -> float:
    """Estimate probability that the verdict holds under parameter uncertainty.

    P(f(θ) ≥ threshold) estimated via Monte Carlo sampling from the
    parameter distributions.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    verdict_threshold : float
        Verdict threshold.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int
        Random seed.

    Returns
    -------
    float
        Estimated probability ∈ [0, 1].
    """
    rng = np.random.default_rng(seed)
    k = len(parameters)
    param_names = [p.name for p in parameters]

    nominal = {p.name: p.nominal for p in parameters}
    f0 = model_fn(**nominal)
    verdict_positive = f0 >= verdict_threshold

    count_match = 0
    for _ in range(n_samples):
        kwargs: Dict[str, float] = {}
        for j, p in enumerate(parameters):
            lo, hi = p.interval.low, p.interval.high
            if p.distribution == "normal":
                mu = (lo + hi) / 2.0
                sigma = (hi - lo) / 4.0
                kwargs[p.name] = float(np.clip(rng.normal(mu, sigma), lo, hi))
            elif p.distribution == "lognormal":
                mu_log = np.log(max((lo + hi) / 2.0, 1e-30))
                kwargs[p.name] = float(np.clip(rng.lognormal(mu_log, 0.5), lo, hi))
            else:
                kwargs[p.name] = float(rng.uniform(lo, hi))

        val = model_fn(**kwargs)
        if (val >= verdict_threshold) == verdict_positive:
            count_match += 1

    return count_match / n_samples


# ═══════════════════════════════════════════════════════════════════════════
# Vertex analysis for hyperrectangular uncertainty
# ═══════════════════════════════════════════════════════════════════════════


def vertex_analysis(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    verdict_threshold: float = 0.0,
) -> Tuple[List[ScenarioResult], bool]:
    """Evaluate model at all vertices of the hyperrectangular parameter space.

    For k parameters there are 2^k vertices. Only practical for k ≤ ~15.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    verdict_threshold : float
        Verdict threshold.

    Returns
    -------
    Tuple[List[ScenarioResult], bool]
        (scenario_results, all_consistent) where all_consistent is True
        if the verdict is the same at all vertices.
    """
    k = len(parameters)
    if k > 20:
        raise ValueError(f"Vertex analysis impractical for k={k} > 20")

    nominal = {p.name: p.nominal for p in parameters}
    f0 = model_fn(**nominal)
    verdict_positive = f0 >= verdict_threshold

    bounds = [(p.interval.low, p.interval.high) for p in parameters]
    param_names = [p.name for p in parameters]

    results: List[ScenarioResult] = []
    all_consistent = True

    for vertex in iterproduct(*bounds):
        kwargs = {name: float(v) for name, v in zip(param_names, vertex)}
        output = model_fn(**kwargs)
        matches = (output >= verdict_threshold) == verdict_positive
        if not matches:
            all_consistent = False
        results.append(ScenarioResult(
            parameter_values=kwargs,
            output=output,
            verdict_matches_nominal=matches,
        ))

    return results, all_consistent


# ═══════════════════════════════════════════════════════════════════════════
# Scenario-based robustness assessment
# ═══════════════════════════════════════════════════════════════════════════


def scenario_robustness(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    verdict_threshold: float = 0.0,
    n_scenarios: int = 1000,
    seed: int = 42,
) -> List[ScenarioResult]:
    """Evaluate model on random scenarios from the parameter space.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    verdict_threshold : float
        Verdict threshold.
    n_scenarios : int
        Number of scenarios.
    seed : int
        Random seed.

    Returns
    -------
    List[ScenarioResult]
        Scenario evaluation results.
    """
    rng = np.random.default_rng(seed)
    param_names = [p.name for p in parameters]

    nominal = {p.name: p.nominal for p in parameters}
    f0 = model_fn(**nominal)
    verdict_positive = f0 >= verdict_threshold

    results: List[ScenarioResult] = []
    for _ in range(n_scenarios):
        kwargs: Dict[str, float] = {}
        for p in parameters:
            kwargs[p.name] = float(rng.uniform(p.interval.low, p.interval.high))

        output = model_fn(**kwargs)
        matches = (output >= verdict_threshold) == verdict_positive
        results.append(ScenarioResult(
            parameter_values=kwargs,
            output=output,
            verdict_matches_nominal=matches,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Critical parameter identification
# ═══════════════════════════════════════════════════════════════════════════


def identify_critical_parameters(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    verdict_threshold: float = 0.0,
    n_samples: int = 500,
    seed: int = 42,
) -> List[Tuple[str, float]]:
    """Identify parameters that most affect verdict robustness.

    For each parameter, estimates the probability of verdict change when
    only that parameter is varied (others held at nominal).

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    verdict_threshold : float
        Verdict threshold.
    n_samples : int
        Samples per parameter.
    seed : int
        Random seed.

    Returns
    -------
    List[Tuple[str, float]]
        (parameter_name, flip_probability) sorted by decreasing flip probability.
    """
    rng = np.random.default_rng(seed)
    nominal = {p.name: p.nominal for p in parameters}
    f0 = model_fn(**nominal)
    verdict_positive = f0 >= verdict_threshold

    criticality: List[Tuple[str, float]] = []

    for p in parameters:
        flip_count = 0
        values = rng.uniform(p.interval.low, p.interval.high, n_samples)
        for val in values:
            test = dict(nominal)
            test[p.name] = float(val)
            output = model_fn(**test)
            if (output >= verdict_threshold) != verdict_positive:
                flip_count += 1
        criticality.append((p.name, flip_count / n_samples))

    criticality.sort(key=lambda x: x[1], reverse=True)
    return criticality


# ═══════════════════════════════════════════════════════════════════════════
# RobustnessAnalyzer — main entry point
# ═══════════════════════════════════════════════════════════════════════════


class RobustnessAnalyzer:
    """Robustness analysis for usability verdicts under parameter uncertainty.

    Parameters
    ----------
    verdict_threshold : float
        Threshold separating "usable" from "not usable".
    n_scenarios : int
        Default scenario count for probabilistic analysis.
    """

    def __init__(
        self,
        verdict_threshold: float = 0.0,
        n_scenarios: int = 1000,
    ) -> None:
        self._threshold = verdict_threshold
        self._n_scenarios = n_scenarios

    def compute_certificate(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        seed: int = 42,
    ) -> RobustnessCertificate:
        """Compute a robustness certificate for the current verdict.

        Combines probabilistic robustness, worst/best-case analysis,
        and critical parameter identification.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameter specifications.
        seed : int
            Random seed.

        Returns
        -------
        RobustnessCertificate
            Robustness certificate.
        """
        prob = probabilistic_robustness(
            model_fn, parameters,
            verdict_threshold=self._threshold,
            n_samples=self._n_scenarios,
            seed=seed,
        )
        worst, _ = worst_case_analysis(
            model_fn, parameters,
            n_samples=self._n_scenarios // 2,
            seed=seed, minimize=True,
        )
        best, _ = worst_case_analysis(
            model_fn, parameters,
            n_samples=self._n_scenarios // 2,
            seed=seed, minimize=False,
        )
        critical = identify_critical_parameters(
            model_fn, parameters,
            verdict_threshold=self._threshold,
            n_samples=max(100, self._n_scenarios // 10),
            seed=seed,
        )
        # Top critical parameters with flip probability > 5%
        critical_names = tuple(name for name, prob_flip in critical if prob_flip > 0.05)

        return RobustnessCertificate(
            probability_correct=prob,
            worst_case_output=worst,
            best_case_output=best,
            critical_parameters=critical_names,
            n_scenarios=self._n_scenarios,
        )

    def analyze(
        self,
        model_fn: Callable[..., float],
        config: SensitivityConfig,
    ) -> SensitivityResult:
        """Run robustness analysis using a SensitivityConfig.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        config : SensitivityConfig
            Analysis configuration.

        Returns
        -------
        SensitivityResult
            Aggregate result with robustness metadata.
        """
        cert = self.compute_certificate(model_fn, config.parameters, seed=config.seed)
        region = compute_robustness_region(
            model_fn, config.parameters,
            verdict_threshold=self._threshold,
        )

        return SensitivityResult(
            config=config,
            output_name=config.output_names[0] if config.output_names else "",
            mean_output=region.nominal_verdict,
            n_evaluations=self._n_scenarios * 3,
            metadata={
                "method": "robustness",
                "probability_correct": cert.probability_correct,
                "worst_case_output": cert.worst_case_output,
                "best_case_output": cert.best_case_output,
                "critical_parameters": cert.critical_parameters,
                "region_volume": region.region_volume,
                "parameter_radii": dict(region.parameter_radii),
            },
        )


__all__ = [
    "RobustnessAnalyzer",
    "RobustnessCertificate",
    "RobustnessRegion",
    "ScenarioResult",
    "compute_robustness_region",
    "identify_critical_parameters",
    "probabilistic_robustness",
    "scenario_robustness",
    "vertex_analysis",
    "worst_case_analysis",
]
