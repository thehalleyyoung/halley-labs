"""
usability_oracle.sensitivity.protocols — Parametric sensitivity analysis protocols.

Structural interfaces for Sobol' (global), Morris (screening), and
local sensitivity analysis of cognitive model parameters.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from usability_oracle.sensitivity.types import (
        MorrisResult,
        ParameterRange,
        SensitivityConfig,
        SensitivityResult,
        SobolIndices,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — unified entry point
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class SensitivityAnalyzer(Protocol):
    """Unified parametric sensitivity analysis for cognitive model parameters.

    Dispatches to global (Sobol'), screening (Morris), or local methods
    based on configuration.
    """

    def analyze(
        self,
        model_fn: Callable[..., float],
        config: SensitivityConfig,
    ) -> SensitivityResult:
        """Run sensitivity analysis on a model function.

        Parameters
        ----------
        model_fn : Callable[..., float]
            The model function mapping parameter dict → scalar output.
            Signature: ``model_fn(**{param_name: value}) -> float``.
        config : SensitivityConfig
            Analysis configuration (parameters, method, samples).

        Returns
        -------
        SensitivityResult
            Aggregate sensitivity analysis result.
        """
        ...

    def rank_parameters(
        self,
        result: SensitivityResult,
    ) -> Sequence[str]:
        """Rank parameters by influence (most influential first).

        Parameters
        ----------
        result : SensitivityResult
            Completed analysis result.

        Returns
        -------
        Sequence[str]
            Parameter names ordered by decreasing influence.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# GlobalSensitivity — variance-based global sensitivity (Sobol')
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class GlobalSensitivity(Protocol):
    """Compute Sobol' variance-based sensitivity indices.

    Uses Saltelli's sampling scheme to estimate first-order, total-order,
    and second-order Sobol' indices with bootstrap confidence intervals.
    """

    def compute_sobol(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        n_samples: int = 1024,
        *,
        seed: int = 42,
        confidence_level: float = 0.95,
    ) -> Sequence[SobolIndices]:
        """Compute Sobol' indices for all parameters.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameter ranges.
        n_samples : int
            Number of Saltelli samples (total evaluations ≈ n*(2k+2)).
        seed : int
            RNG seed.
        confidence_level : float
            Bootstrap confidence level.

        Returns
        -------
        Sequence[SobolIndices]
            Sobol' indices for each parameter.
        """
        ...

    def total_variance(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        n_samples: int = 1024,
        *,
        seed: int = 42,
    ) -> float:
        """Estimate total output variance.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameter ranges.
        n_samples : int
            Number of samples.
        seed : int
            RNG seed.

        Returns
        -------
        float
            Estimated output variance.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# LocalSensitivity — gradient / OAT-based local sensitivity
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class LocalSensitivity(Protocol):
    """Local sensitivity analysis via finite differences or OAT perturbation.

    Evaluates the model at the nominal parameter values and perturbs
    each parameter one at a time.
    """

    def compute_gradient(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        *,
        step_fraction: float = 0.01,
    ) -> Mapping[str, float]:
        """Compute local sensitivity (partial derivatives) at nominal values.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameters with nominal values.
        step_fraction : float
            Perturbation step as a fraction of parameter range.

        Returns
        -------
        Mapping[str, float]
            Parameter name → partial derivative ∂f/∂xᵢ.
        """
        ...

    def elasticity(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
    ) -> Mapping[str, float]:
        """Compute elasticity (normalised sensitivity) at nominal values.

        Elasticity = (xᵢ / f) · (∂f/∂xᵢ) — dimensionless measure of
        the percentage change in output per percentage change in input.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameters with nominal values.

        Returns
        -------
        Mapping[str, float]
            Parameter name → elasticity.
        """
        ...

    def morris_screening(
        self,
        model_fn: Callable[..., float],
        parameters: Sequence[ParameterRange],
        n_trajectories: int = 10,
        *,
        seed: int = 42,
    ) -> Sequence[MorrisResult]:
        """Run Morris elementary-effects screening.

        Parameters
        ----------
        model_fn : Callable[..., float]
            Model function.
        parameters : Sequence[ParameterRange]
            Parameter ranges.
        n_trajectories : int
            Number of Morris trajectories.
        seed : int
            RNG seed.

        Returns
        -------
        Sequence[MorrisResult]
            Morris screening results for each parameter.
        """
        ...


__all__ = [
    "GlobalSensitivity",
    "LocalSensitivity",
    "SensitivityAnalyzer",
]
