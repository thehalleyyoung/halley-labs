"""
usability_oracle.sensitivity.variance — Variance-based decomposition.

ANOVA-style decomposition of model output variance into first-order,
higher-order, and total contributions with functional ANOVA and
correlation ratio estimation.

References
----------
Sobol', I. M. (1993). Sensitivity estimates for nonlinear mathematical
    models. Mathematical Modelling and Computational Experiment, 1, 407–414.
Hoeffding, W. (1948). A class of statistics with asymptotically normal
    distribution. Annals of Mathematical Statistics, 19(3), 293–325.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.sensitivity.types import (
    ParameterRange,
    SensitivityConfig,
    SensitivityResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class VarianceComponent:
    """Single component in ANOVA-style variance decomposition.

    Attributes
    ----------
    parameter_names : tuple[str, ...]
        Parameters in this component (length 1 = first-order, 2 = interaction, etc.).
    variance : float
        Absolute variance contribution.
    fraction : float
        Fraction of total variance explained.
    """

    parameter_names: Tuple[str, ...]
    variance: float
    fraction: float

    @property
    def order(self) -> int:
        """Interaction order (1 = main effect, 2 = pairwise, etc.)."""
        return len(self.parameter_names)


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    """Full ANOVA-style variance decomposition.

    Attributes
    ----------
    total_variance : float
        Total output variance.
    mean_output : float
        Grand mean f₀ = E[Y].
    components : tuple[VarianceComponent, ...]
        Variance components from first-order through interactions.
    residual_fraction : float
        Unexplained variance fraction (numerical error / higher-order).
    n_evaluations : int
        Total model evaluations.
    """

    total_variance: float
    mean_output: float
    components: Tuple[VarianceComponent, ...] = ()
    residual_fraction: float = 0.0
    n_evaluations: int = 0

    @property
    def first_order_components(self) -> Tuple[VarianceComponent, ...]:
        return tuple(c for c in self.components if c.order == 1)

    @property
    def interaction_components(self) -> Tuple[VarianceComponent, ...]:
        return tuple(c for c in self.components if c.order > 1)

    @property
    def sum_of_fractions(self) -> float:
        return sum(c.fraction for c in self.components)


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo variance estimation
# ═══════════════════════════════════════════════════════════════════════════


def _sample_parameters(
    parameters: Sequence[ParameterRange],
    n_samples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample parameter values according to their distributions.

    Parameters
    ----------
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    n_samples : int
        Number of samples.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    NDArray[np.float64]
        Sample matrix of shape ``(n_samples, k)``.
    """
    k = len(parameters)
    samples = np.empty((n_samples, k), dtype=np.float64)

    for j, p in enumerate(parameters):
        lo, hi = p.interval.low, p.interval.high
        if p.distribution == "normal":
            mu = (lo + hi) / 2.0
            sigma = (hi - lo) / 4.0  # ~95% within range
            samples[:, j] = np.clip(rng.normal(mu, sigma, n_samples), lo, hi)
        elif p.distribution == "lognormal":
            mu_log = np.log(max((lo + hi) / 2.0, 1e-30))
            sigma_log = 0.5
            samples[:, j] = np.clip(rng.lognormal(mu_log, sigma_log, n_samples), lo, hi)
        else:  # uniform
            samples[:, j] = rng.uniform(lo, hi, n_samples)

    return samples


def _evaluate_samples(
    model_fn: Callable[..., float],
    samples: NDArray[np.float64],
    param_names: Sequence[str],
) -> NDArray[np.float64]:
    """Evaluate model on each sample row."""
    n = samples.shape[0]
    outputs = np.empty(n, dtype=np.float64)
    for i in range(n):
        kwargs = {name: float(samples[i, j]) for j, name in enumerate(param_names)}
        outputs[i] = model_fn(**kwargs)
    return outputs


def total_variance(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    n_samples: int = 4096,
    seed: int = 42,
) -> Tuple[float, float]:
    """Estimate total output variance and mean via Monte Carlo.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    n_samples : int
        Sample count.
    seed : int
        Random seed.

    Returns
    -------
    Tuple[float, float]
        (total_variance, mean_output).
    """
    rng = np.random.default_rng(seed)
    param_names = [p.name for p in parameters]
    samples = _sample_parameters(parameters, n_samples, rng)
    outputs = _evaluate_samples(model_fn, samples, param_names)
    return float(np.var(outputs, ddof=1)), float(np.mean(outputs))


# ═══════════════════════════════════════════════════════════════════════════
# First-order variance contributions (conditional variance approach)
# ═══════════════════════════════════════════════════════════════════════════


def first_order_variance(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    param_index: int,
    n_outer: int = 128,
    n_inner: int = 64,
    seed: int = 42,
) -> float:
    """Estimate V_i = Var_Xi[E_{X~i}[Y | X_i]] via double-loop Monte Carlo.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    param_index : int
        Index of the parameter.
    n_outer : int
        Outer loop samples (for X_i).
    n_inner : int
        Inner loop samples (for X_{~i} | X_i).
    seed : int
        Random seed.

    Returns
    -------
    float
        First-order variance contribution V_i.
    """
    rng = np.random.default_rng(seed)
    k = len(parameters)
    param_names = [p.name for p in parameters]

    conditional_means = np.empty(n_outer, dtype=np.float64)

    for outer in range(n_outer):
        # Fix parameter i
        p_i = parameters[param_index]
        lo, hi = p_i.interval.low, p_i.interval.high
        xi_val = rng.uniform(lo, hi)

        # Sample all other parameters n_inner times
        inner_samples = _sample_parameters(parameters, n_inner, rng)
        inner_samples[:, param_index] = xi_val

        outputs = _evaluate_samples(model_fn, inner_samples, param_names)
        conditional_means[outer] = np.mean(outputs)

    return float(np.var(conditional_means, ddof=1))


# ═══════════════════════════════════════════════════════════════════════════
# Higher-order interaction variance
# ═══════════════════════════════════════════════════════════════════════════


def interaction_variance(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    param_indices: Tuple[int, ...],
    first_order_variances: Mapping[int, float],
    n_outer: int = 128,
    n_inner: int = 64,
    seed: int = 42,
) -> float:
    """Estimate interaction variance V_{ij...} via the inclusion-exclusion principle.

    V_{ij} = Var_{Xi,Xj}[E[Y|Xi,Xj]] - V_i - V_j

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    param_indices : Tuple[int, ...]
        Indices of interacting parameters.
    first_order_variances : Mapping[int, float]
        Pre-computed first-order variances.
    n_outer : int
        Outer loop samples.
    n_inner : int
        Inner loop samples.
    seed : int
        Random seed.

    Returns
    -------
    float
        Interaction variance contribution.
    """
    rng = np.random.default_rng(seed)
    k = len(parameters)
    param_names = [p.name for p in parameters]

    conditional_means = np.empty(n_outer, dtype=np.float64)

    for outer in range(n_outer):
        # Fix the interacting parameters
        inner_samples = _sample_parameters(parameters, n_inner, rng)
        for idx in param_indices:
            p_j = parameters[idx]
            val = rng.uniform(p_j.interval.low, p_j.interval.high)
            inner_samples[:, idx] = val

        outputs = _evaluate_samples(model_fn, inner_samples, param_names)
        conditional_means[outer] = np.mean(outputs)

    joint_var = float(np.var(conditional_means, ddof=1))
    # Subtract lower-order contributions
    for idx in param_indices:
        joint_var -= first_order_variances.get(idx, 0.0)
    return max(joint_var, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Correlation ratio η²
# ═══════════════════════════════════════════════════════════════════════════


def correlation_ratio(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    param_index: int,
    n_bins: int = 20,
    n_samples: int = 4096,
    seed: int = 42,
) -> float:
    """Estimate correlation ratio η² for a single parameter.

    η² = Var[E[Y|X_i]] / Var[Y]  (equivalent to first-order Sobol' index
    for independent inputs).

    Uses binning of X_i to estimate conditional expectations.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    param_index : int
        Index of the parameter.
    n_bins : int
        Number of bins for conditional averaging.
    n_samples : int
        Total samples.
    seed : int
        Random seed.

    Returns
    -------
    float
        Correlation ratio η² ∈ [0, 1].
    """
    rng = np.random.default_rng(seed)
    param_names = [p.name for p in parameters]
    samples = _sample_parameters(parameters, n_samples, rng)
    outputs = _evaluate_samples(model_fn, samples, param_names)

    xi = samples[:, param_index]
    total_var = np.var(outputs, ddof=1)
    if total_var < 1e-30:
        return 0.0

    # Bin X_i and compute conditional means
    bin_edges = np.linspace(xi.min(), xi.max() + 1e-15, n_bins + 1)
    bin_indices = np.digitize(xi, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    grand_mean = np.mean(outputs)
    between_var = 0.0
    total_count = 0
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        bin_mean = np.mean(outputs[mask])
        between_var += count * (bin_mean - grand_mean) ** 2
        total_count += count

    if total_count <= 1:
        return 0.0
    between_var /= (total_count - 1)
    return float(np.clip(between_var / total_var, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Functional ANOVA decomposition
# ═══════════════════════════════════════════════════════════════════════════


def functional_anova(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    max_order: int = 2,
    n_outer: int = 128,
    n_inner: int = 64,
    seed: int = 42,
) -> VarianceDecomposition:
    """Functional ANOVA decomposition up to a given interaction order.

    Decomposes Var[Y] = Σ V_i + Σ V_{ij} + ... using Hoeffding's
    functional decomposition estimated via Monte Carlo.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    max_order : int
        Maximum interaction order (1 = main effects only, 2 = pairwise, etc.).
    n_outer : int
        Outer loop samples.
    n_inner : int
        Inner loop samples.
    seed : int
        Random seed.

    Returns
    -------
    VarianceDecomposition
        Full decomposition with components up to max_order.
    """
    from itertools import combinations

    k = len(parameters)
    param_names = [p.name for p in parameters]
    total_var, mean_out = total_variance(model_fn, parameters, n_outer * n_inner, seed)
    n_evals = n_outer * n_inner

    components: List[VarianceComponent] = []

    # First-order
    fo_variances: Dict[int, float] = {}
    for i in range(k):
        vi = first_order_variance(
            model_fn, parameters, i,
            n_outer=n_outer, n_inner=n_inner, seed=seed + i,
        )
        fo_variances[i] = vi
        frac = vi / total_var if total_var > 1e-30 else 0.0
        components.append(VarianceComponent(
            parameter_names=(param_names[i],),
            variance=vi,
            fraction=frac,
        ))
        n_evals += n_outer * n_inner

    # Higher-order interactions
    if max_order >= 2 and k >= 2:
        for order in range(2, min(max_order, k) + 1):
            for combo in combinations(range(k), order):
                iv = interaction_variance(
                    model_fn, parameters, combo, fo_variances,
                    n_outer=n_outer, n_inner=n_inner,
                    seed=seed + sum(combo) * 100,
                )
                frac = iv / total_var if total_var > 1e-30 else 0.0
                names = tuple(param_names[c] for c in combo)
                components.append(VarianceComponent(
                    parameter_names=names,
                    variance=iv,
                    fraction=frac,
                ))
                n_evals += n_outer * n_inner

    explained = sum(c.fraction for c in components)
    residual = max(1.0 - explained, 0.0)

    return VarianceDecomposition(
        total_variance=total_var,
        mean_output=mean_out,
        components=tuple(components),
        residual_fraction=residual,
        n_evaluations=n_evals,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Importance measures
# ═══════════════════════════════════════════════════════════════════════════


def variance_importance_ranking(
    decomposition: VarianceDecomposition,
) -> List[Tuple[str, float]]:
    """Rank parameters by first-order variance fraction.

    Parameters
    ----------
    decomposition : VarianceDecomposition
        Completed decomposition.

    Returns
    -------
    List[Tuple[str, float]]
        (parameter_name, fraction) sorted by decreasing fraction.
    """
    first_order = [
        (c.parameter_names[0], c.fraction) for c in decomposition.first_order_components
    ]
    first_order.sort(key=lambda x: x[1], reverse=True)
    return first_order


# ═══════════════════════════════════════════════════════════════════════════
# VarianceAnalyzer — main entry point
# ═══════════════════════════════════════════════════════════════════════════


class VarianceAnalyzer:
    """Variance-based decomposition for cognitive model parameters.

    Parameters
    ----------
    max_order : int
        Maximum interaction order for ANOVA decomposition.
    n_outer : int
        Outer loop samples.
    n_inner : int
        Inner loop samples.
    """

    def __init__(
        self,
        max_order: int = 2,
        n_outer: int = 128,
        n_inner: int = 64,
    ) -> None:
        self._max_order = max_order
        self._n_outer = n_outer
        self._n_inner = n_inner

    def decompose(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        seed: int = 42,
    ) -> VarianceDecomposition:
        """Run functional ANOVA decomposition.

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
        VarianceDecomposition
            Complete decomposition.
        """
        return functional_anova(
            model_fn, parameters,
            max_order=self._max_order,
            n_outer=self._n_outer,
            n_inner=self._n_inner,
            seed=seed,
        )

    def analyze(
        self,
        model_fn: Callable[..., float],
        config: SensitivityConfig,
    ) -> SensitivityResult:
        """Run variance decomposition via SensitivityConfig.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        config : SensitivityConfig
            Analysis configuration.

        Returns
        -------
        SensitivityResult
            Aggregate result.
        """
        decomp = self.decompose(model_fn, config.parameters, seed=config.seed)
        ranking = variance_importance_ranking(decomp)

        return SensitivityResult(
            config=config,
            output_name=config.output_names[0] if config.output_names else "",
            total_variance=decomp.total_variance,
            mean_output=decomp.mean_output,
            n_evaluations=decomp.n_evaluations,
            metadata={
                "method": "variance_decomposition",
                "max_order": self._max_order,
                "components": [
                    {
                        "parameters": c.parameter_names,
                        "variance": c.variance,
                        "fraction": c.fraction,
                    }
                    for c in decomp.components
                ],
                "residual_fraction": decomp.residual_fraction,
                "ranking": ranking,
            },
        )


__all__ = [
    "VarianceAnalyzer",
    "VarianceComponent",
    "VarianceDecomposition",
    "correlation_ratio",
    "first_order_variance",
    "functional_anova",
    "interaction_variance",
    "total_variance",
    "variance_importance_ranking",
]
