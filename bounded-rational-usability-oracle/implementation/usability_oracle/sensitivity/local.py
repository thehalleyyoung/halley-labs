"""
usability_oracle.sensitivity.local — Local sensitivity analysis.

Finite-difference computation of partial derivatives, Jacobians, normalised
sensitivity coefficients, elasticities, and condition numbers for parameter
identifiability of cognitive model parameters.

References
----------
Saltelli, A., Ratto, M., Andres, T., et al. (2008). Global Sensitivity
    Analysis: The Primer. Wiley.
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
# Finite difference methods
# ═══════════════════════════════════════════════════════════════════════════


def forward_difference(
    model_fn: Callable[..., float],
    nominal: Dict[str, float],
    param_name: str,
    step: float,
) -> float:
    """Forward-difference approximation of ∂f/∂θ_i.

    ∂f/∂θ_i ≈ (f(θ + h·e_i) - f(θ)) / h

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    nominal : Dict[str, float]
        Nominal parameter values.
    param_name : str
        Parameter to differentiate with respect to.
    step : float
        Perturbation step size h.

    Returns
    -------
    float
        Approximate partial derivative.
    """
    f0 = model_fn(**nominal)
    perturbed = dict(nominal)
    perturbed[param_name] = nominal[param_name] + step
    f1 = model_fn(**perturbed)
    return (f1 - f0) / step


def central_difference(
    model_fn: Callable[..., float],
    nominal: Dict[str, float],
    param_name: str,
    step: float,
) -> float:
    """Central-difference approximation of ∂f/∂θ_i (O(h²) accuracy).

    ∂f/∂θ_i ≈ (f(θ + h·e_i) - f(θ - h·e_i)) / (2h)

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    nominal : Dict[str, float]
        Nominal parameter values.
    param_name : str
        Parameter to differentiate.
    step : float
        Half-step size h.

    Returns
    -------
    float
        Approximate partial derivative.
    """
    fwd = dict(nominal)
    bwd = dict(nominal)
    fwd[param_name] = nominal[param_name] + step
    bwd[param_name] = nominal[param_name] - step
    return (model_fn(**fwd) - model_fn(**bwd)) / (2.0 * step)


def complex_step(
    model_fn: Callable[..., complex],
    nominal: Dict[str, float],
    param_name: str,
    step: float = 1e-30,
) -> float:
    """Complex-step differentiation (machine-precision accuracy, no cancellation).

    ∂f/∂θ_i = Im[f(θ + ih·e_i)] / h

    Requires model_fn to accept and propagate complex numbers.

    Parameters
    ----------
    model_fn : Callable
        Model function that supports complex arithmetic.
    nominal : Dict[str, float]
        Nominal parameter values.
    param_name : str
        Parameter to differentiate.
    step : float
        Imaginary perturbation (can be very small, e.g. 1e-30).

    Returns
    -------
    float
        Partial derivative at machine precision.
    """
    perturbed: Dict[str, complex] = {k: complex(v) for k, v in nominal.items()}
    perturbed[param_name] = complex(nominal[param_name], step)
    result = model_fn(**perturbed)
    return float(result.imag) / step  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity matrix (Jacobian)
# ═══════════════════════════════════════════════════════════════════════════


def compute_jacobian(
    model_fns: Sequence[Callable[..., float]],
    parameters: Sequence[ParameterRange],
    *,
    step_fraction: float = 0.01,
    method: str = "central",
) -> NDArray[np.float64]:
    """Compute the sensitivity matrix (Jacobian) J_{ij} = ∂f_i/∂θ_j.

    Parameters
    ----------
    model_fns : Sequence[Callable[..., float]]
        Output functions (one per row of the Jacobian).
    parameters : Sequence[ParameterRange]
        Parameter specifications with nominal values.
    step_fraction : float
        Step size as fraction of parameter range.
    method : str
        ``"forward"`` or ``"central"`` difference.

    Returns
    -------
    NDArray[np.float64]
        Jacobian matrix of shape ``(n_outputs, n_params)``.
    """
    n_out = len(model_fns)
    n_param = len(parameters)
    J = np.zeros((n_out, n_param), dtype=np.float64)

    nominal = {p.name: p.nominal for p in parameters}
    diff_fn = central_difference if method == "central" else forward_difference

    for j, p in enumerate(parameters):
        step = step_fraction * p.range_width
        if step < 1e-15:
            step = 1e-8  # fallback for zero-width ranges
        for i, fn in enumerate(model_fns):
            J[i, j] = diff_fn(fn, nominal, p.name, step)

    return J


def compute_gradient(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    *,
    step_fraction: float = 0.01,
    method: str = "central",
) -> Dict[str, float]:
    """Compute local sensitivity (gradient) at nominal parameter values.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameters with nominal values.
    step_fraction : float
        Perturbation step as fraction of parameter range.
    method : str
        ``"forward"``, ``"central"``, or ``"complex"``.

    Returns
    -------
    Dict[str, float]
        Parameter name → ∂f/∂θ_i.
    """
    nominal = {p.name: p.nominal for p in parameters}
    gradient: Dict[str, float] = {}

    for p in parameters:
        step = step_fraction * p.range_width
        if step < 1e-15:
            step = 1e-8

        if method == "complex":
            gradient[p.name] = complex_step(model_fn, nominal, p.name, step=1e-30)  # type: ignore[arg-type]
        elif method == "central":
            gradient[p.name] = central_difference(model_fn, nominal, p.name, step)
        else:
            gradient[p.name] = forward_difference(model_fn, nominal, p.name, step)

    return gradient


# ═══════════════════════════════════════════════════════════════════════════
# Normalised sensitivity coefficients
# ═══════════════════════════════════════════════════════════════════════════


def normalized_sensitivity(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    *,
    step_fraction: float = 0.01,
) -> Dict[str, float]:
    """Normalised sensitivity: S_i = (∂f/∂θ_i) × (range_i / f₀).

    Scales the partial derivative by the parameter range and the baseline
    output so that sensitivities are dimensionless and comparable.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    step_fraction : float
        Perturbation fraction.

    Returns
    -------
    Dict[str, float]
        Parameter name → normalised sensitivity coefficient.
    """
    nominal = {p.name: p.nominal for p in parameters}
    f0 = model_fn(**nominal)
    grad = compute_gradient(model_fn, parameters, step_fraction=step_fraction)

    result: Dict[str, float] = {}
    for p in parameters:
        if abs(f0) < 1e-30:
            result[p.name] = 0.0
        else:
            result[p.name] = grad[p.name] * p.range_width / abs(f0)
    return result


def elasticity(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    *,
    step_fraction: float = 0.01,
) -> Dict[str, float]:
    """Elasticity: E_i = (θ_i / f₀) × (∂f/∂θ_i).

    Dimensionless measure: percentage change in output per percentage
    change in input parameter.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    step_fraction : float
        Perturbation fraction.

    Returns
    -------
    Dict[str, float]
        Parameter name → elasticity.
    """
    nominal = {p.name: p.nominal for p in parameters}
    f0 = model_fn(**nominal)
    grad = compute_gradient(model_fn, parameters, step_fraction=step_fraction)

    result: Dict[str, float] = {}
    for p in parameters:
        if abs(f0) < 1e-30:
            result[p.name] = 0.0
        else:
            result[p.name] = (p.nominal / f0) * grad[p.name]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Condition number for parameter identifiability
# ═══════════════════════════════════════════════════════════════════════════


def condition_number(
    model_fns: Sequence[Callable[..., float]],
    parameters: Sequence[ParameterRange],
    *,
    step_fraction: float = 0.01,
) -> float:
    """Condition number of the sensitivity matrix for parameter identifiability.

    A high condition number indicates that some parameter combinations are
    poorly identifiable from the given outputs.

    κ(J) = σ_max / σ_min  (ratio of largest to smallest singular value)

    Parameters
    ----------
    model_fns : Sequence[Callable[..., float]]
        Output functions.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    step_fraction : float
        Perturbation fraction.

    Returns
    -------
    float
        Condition number κ(J). Returns inf if the matrix is rank-deficient.
    """
    J = compute_jacobian(model_fns, parameters, step_fraction=step_fraction)
    sv = np.linalg.svd(J, compute_uv=False)
    if sv[-1] < 1e-30:
        return float("inf")
    return float(sv[0] / sv[-1])


def collinearity_index(
    model_fns: Sequence[Callable[..., float]],
    parameters: Sequence[ParameterRange],
    subset_indices: Optional[Sequence[int]] = None,
    *,
    step_fraction: float = 0.01,
) -> float:
    """Collinearity index for a subset of parameters.

    γ_K = 1 / σ_min(J_K)  where J_K is the Jacobian restricted to the
    parameter subset K.  High γ indicates near-collinearity.

    Parameters
    ----------
    model_fns : Sequence[Callable[..., float]]
        Output functions.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    subset_indices : Sequence[int], optional
        Indices of parameters to include. Defaults to all.
    step_fraction : float
        Perturbation fraction.

    Returns
    -------
    float
        Collinearity index.
    """
    J = compute_jacobian(model_fns, parameters, step_fraction=step_fraction)
    if subset_indices is not None:
        J = J[:, list(subset_indices)]

    sv = np.linalg.svd(J, compute_uv=False)
    if sv[-1] < 1e-30:
        return float("inf")
    return float(1.0 / sv[-1])


# ═══════════════════════════════════════════════════════════════════════════
# LocalSensitivityAnalyzer — main entry point
# ═══════════════════════════════════════════════════════════════════════════


class LocalSensitivityAnalyzer:
    """Local sensitivity analysis via finite differences.

    Implements the ``LocalSensitivity`` protocol.

    Parameters
    ----------
    method : str
        Difference method: ``"forward"``, ``"central"``, ``"complex"``.
    step_fraction : float
        Default perturbation step as fraction of parameter range.
    """

    def __init__(
        self,
        method: str = "central",
        step_fraction: float = 0.01,
    ) -> None:
        self._method = method
        self._step_fraction = step_fraction

    def compute_gradient(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        *,
        step_fraction: float = 0.0,
    ) -> Dict[str, float]:
        """Compute ∂f/∂θ_i at nominal values.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameters with nominal values.
        step_fraction : float
            Override step fraction (0 → use default).

        Returns
        -------
        Dict[str, float]
            Parameter name → partial derivative.
        """
        sf = step_fraction if step_fraction > 0 else self._step_fraction
        return compute_gradient(
            model_fn, parameters,
            step_fraction=sf, method=self._method,
        )

    def elasticity(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
    ) -> Dict[str, float]:
        """Compute elasticity E_i = (θ_i / f₀) · ∂f/∂θ_i.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameters with nominal values.

        Returns
        -------
        Dict[str, float]
            Parameter name → elasticity.
        """
        return elasticity(
            model_fn, parameters,
            step_fraction=self._step_fraction,
        )

    def analyze(
        self,
        model_fn: Callable[..., float],
        config: SensitivityConfig,
    ) -> SensitivityResult:
        """Run local sensitivity analysis.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        config : SensitivityConfig
            Analysis configuration.

        Returns
        -------
        SensitivityResult
            Result with gradient and elasticity in metadata.
        """
        grad = self.compute_gradient(model_fn, config.parameters)
        elast = self.elasticity(model_fn, config.parameters)
        norm_sens = normalized_sensitivity(
            model_fn, config.parameters,
            step_fraction=self._step_fraction,
        )

        nominal = {p.name: p.nominal for p in config.parameters}
        f0 = model_fn(**nominal)

        # Number of evaluations: 2 per param for central, 1+k for forward
        if self._method == "central":
            n_evals = 2 * config.n_parameters
        else:
            n_evals = 1 + config.n_parameters

        return SensitivityResult(
            config=config,
            output_name=config.output_names[0] if config.output_names else "",
            mean_output=f0,
            n_evaluations=n_evals,
            metadata={
                "method": "local",
                "difference_method": self._method,
                "gradient": dict(grad),
                "elasticity": dict(elast),
                "normalized_sensitivity": dict(norm_sens),
            },
        )


__all__ = [
    "LocalSensitivityAnalyzer",
    "central_difference",
    "collinearity_index",
    "complex_step",
    "compute_gradient",
    "compute_jacobian",
    "condition_number",
    "elasticity",
    "forward_difference",
    "normalized_sensitivity",
]
