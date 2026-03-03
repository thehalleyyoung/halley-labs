"""
Configuration management for DP-Forge.

Provides a single :class:`DPForgeConfig` that aggregates all sub-configs
(numerical, synthesis, sampling) and supports:

- Programmatic construction with validated defaults.
- Environment variable overrides (``DPFORGE_*`` prefix).
- Solver auto-detection: probes for HiGHS → GLPK → MOSEK → SCS → SciPy.
- Serialisation to / from dict for persistence.

Usage::

    from dp_forge.config import get_config

    cfg = get_config()                 # defaults + env overrides
    cfg = get_config(max_iter=100)     # override max_iter
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

from dp_forge.exceptions import ConfigurationError
from dp_forge.types import (
    NumericalConfig,
    SamplingConfig,
    SamplingMethod,
    SolverBackend,
    SynthesisConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default values (aligned with approach.json)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    # Synthesis loop
    "max_iter": 50,
    "tol": 1e-8,
    "warm_start": True,
    "solver": SolverBackend.AUTO,
    "verbose": 1,
    "symmetry_detection": True,
    # Numerical
    "solver_tol": 1e-8,
    "dp_tol": 1e-6,
    "eta_min_scale": 1e-10,
    "max_condition_number": 1e12,
    # Sampling
    "sampling_method": SamplingMethod.ALIAS,
    "sampling_seed": None,
}

# Env-var name → (config key, converter)
_ENV_MAP: Dict[str, tuple[str, type]] = {
    "DPFORGE_MAX_ITER": ("max_iter", int),
    "DPFORGE_TOL": ("tol", float),
    "DPFORGE_WARM_START": ("warm_start", lambda v: v.lower() in ("1", "true", "yes")),
    "DPFORGE_SOLVER": ("solver", lambda v: SolverBackend(v.lower())),
    "DPFORGE_VERBOSE": ("verbose", int),
    "DPFORGE_SOLVER_TOL": ("solver_tol", float),
    "DPFORGE_DP_TOL": ("dp_tol", float),
    "DPFORGE_ETA_MIN_SCALE": ("eta_min_scale", float),
    "DPFORGE_MAX_CONDITION": ("max_condition_number", float),
    "DPFORGE_SYMMETRY": ("symmetry_detection", lambda v: v.lower() in ("1", "true", "yes")),
    "DPFORGE_SEED": ("sampling_seed", lambda v: None if v.lower() == "none" else int(v)),
}


# ---------------------------------------------------------------------------
# Solver auto-detection
# ---------------------------------------------------------------------------


def _probe_solver(name: str) -> bool:
    """Return True if *name* solver is importable and minimally functional."""
    try:
        if name == "highs":
            from scipy.optimize import linprog  # noqa: F401

            # HiGHS is bundled with SciPy ≥ 1.9
            return True
        elif name == "glpk":
            import glpk  # noqa: F401

            return True
        elif name == "mosek":
            import mosek  # noqa: F401

            return True
        elif name == "scs":
            import scs  # noqa: F401

            return True
        elif name == "scipy":
            from scipy.optimize import linprog  # noqa: F401

            return True
    except ImportError:
        return False
    return False


def detect_solver() -> SolverBackend:
    """Detect the best available solver in preference order.

    Order: HiGHS > GLPK > MOSEK > SCS > SciPy (fallback).

    Returns:
        The :class:`SolverBackend` for the best available solver.

    Raises:
        ConfigurationError: If no solver can be found at all.
    """
    preference = [
        ("highs", SolverBackend.HIGHS),
        ("glpk", SolverBackend.GLPK),
        ("mosek", SolverBackend.MOSEK),
        ("scs", SolverBackend.SCS),
        ("scipy", SolverBackend.SCIPY),
    ]
    for name, backend in preference:
        if _probe_solver(name):
            logger.debug("Auto-detected solver: %s", backend.name)
            return backend
    raise ConfigurationError(
        "No LP/SDP solver found. Install scipy (HiGHS), python-glpk, mosek, or scs.",
        parameter="solver",
        constraint="at least one solver must be importable",
    )


# ---------------------------------------------------------------------------
# Resolve solver backend
# ---------------------------------------------------------------------------


def _resolve_solver(backend: SolverBackend) -> SolverBackend:
    """Resolve AUTO to a concrete solver, validate explicit choices."""
    if backend == SolverBackend.AUTO:
        return detect_solver()
    name = backend.value
    if not _probe_solver(name):
        raise ConfigurationError(
            f"Solver {backend.name} requested but not importable",
            parameter="solver",
            value=backend.name,
            constraint=f"pip install the {name} package",
        )
    return backend


# ---------------------------------------------------------------------------
# DPForgeConfig
# ---------------------------------------------------------------------------


@dataclass
class DPForgeConfig:
    """Top-level configuration for the DP-Forge pipeline.

    Aggregates :class:`SynthesisConfig`, :class:`NumericalConfig`, and
    :class:`SamplingConfig` into one validated object.  Typically obtained
    via :func:`get_config` which applies environment-variable overrides.

    Attributes:
        synthesis: CEGIS loop parameters.
        numerical: Numerical precision parameters.
        sampling: Sampling parameters.
        resolved_solver: The concrete solver backend after AUTO resolution.
    """

    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    resolved_solver: Optional[SolverBackend] = None

    def __post_init__(self) -> None:
        # Wire sub-configs into synthesis config
        self.synthesis.numerical = self.numerical
        self.synthesis.sampling = self.sampling
        # Resolve solver
        if self.resolved_solver is None:
            try:
                self.resolved_solver = _resolve_solver(self.synthesis.solver)
            except ConfigurationError:
                self.resolved_solver = self.synthesis.solver

    def validate(self, epsilon: Optional[float] = None) -> None:
        """Run full validation, optionally checking ε-dependent invariants.

        Raises:
            ConfigurationError: On any invalid parameter.
        """
        if epsilon is not None:
            if not self.numerical.validate_dp_tol(epsilon):
                required = math.exp(epsilon) * self.numerical.solver_tol
                raise ConfigurationError(
                    f"Invariant I4 violated: dp_tol ({self.numerical.dp_tol:.2e}) "
                    f"< exp(ε)·solver_tol ({required:.2e}). "
                    f"Increase dp_tol or decrease solver_tol.",
                    parameter="dp_tol",
                    value=self.numerical.dp_tol,
                    constraint=f"dp_tol >= exp({epsilon}) * solver_tol",
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "synthesis": {
                "max_iter": self.synthesis.max_iter,
                "tol": self.synthesis.tol,
                "warm_start": self.synthesis.warm_start,
                "solver": self.synthesis.solver.value,
                "verbose": self.synthesis.verbose,
                "symmetry_detection": self.synthesis.symmetry_detection,
                "eta_min": self.synthesis.eta_min,
            },
            "numerical": {
                "solver_tol": self.numerical.solver_tol,
                "dp_tol": self.numerical.dp_tol,
                "eta_min_scale": self.numerical.eta_min_scale,
                "max_condition_number": self.numerical.max_condition_number,
            },
            "sampling": {
                "method": self.sampling.method.name,
                "seed": self.sampling.seed,
            },
            "resolved_solver": self.resolved_solver.value if self.resolved_solver else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DPForgeConfig:
        """Deserialise from a plain dict."""
        syn_d = d.get("synthesis", {})
        num_d = d.get("numerical", {})
        sam_d = d.get("sampling", {})

        solver_val = syn_d.get("solver", "auto")
        solver = SolverBackend(solver_val) if isinstance(solver_val, str) else SolverBackend.AUTO

        numerical = NumericalConfig(
            solver_tol=num_d.get("solver_tol", _DEFAULTS["solver_tol"]),
            dp_tol=num_d.get("dp_tol", _DEFAULTS["dp_tol"]),
            eta_min_scale=num_d.get("eta_min_scale", _DEFAULTS["eta_min_scale"]),
            max_condition_number=num_d.get("max_condition_number", _DEFAULTS["max_condition_number"]),
        )
        sampling = SamplingConfig(
            method=SamplingMethod[sam_d.get("method", "ALIAS")],
            seed=sam_d.get("seed"),
        )
        synthesis = SynthesisConfig(
            max_iter=syn_d.get("max_iter", _DEFAULTS["max_iter"]),
            tol=syn_d.get("tol", _DEFAULTS["tol"]),
            warm_start=syn_d.get("warm_start", _DEFAULTS["warm_start"]),
            solver=solver,
            verbose=syn_d.get("verbose", _DEFAULTS["verbose"]),
            symmetry_detection=syn_d.get("symmetry_detection", _DEFAULTS["symmetry_detection"]),
            eta_min=syn_d.get("eta_min"),
            numerical=numerical,
            sampling=sampling,
        )
        return cls(synthesis=synthesis, numerical=numerical, sampling=sampling)

    def __repr__(self) -> str:
        solver = self.resolved_solver.name if self.resolved_solver else "UNRESOLVED"
        return (
            f"DPForgeConfig(solver={solver}, max_iter={self.synthesis.max_iter}, "
            f"tol={self.synthesis.tol:.0e})"
        )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _apply_env_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Read DPFORGE_* environment variables and merge into *overrides*."""
    for env_var, (key, converter) in _ENV_MAP.items():
        val = os.environ.get(env_var)
        if val is not None and key not in overrides:
            try:
                overrides[key] = converter(val)
                logger.debug("Env override: %s=%s → %s=%r", env_var, val, key, overrides[key])
            except (ValueError, KeyError) as exc:
                raise ConfigurationError(
                    f"Invalid value for environment variable {env_var}: {val!r}",
                    parameter=env_var,
                    value=val,
                    constraint=str(exc),
                ) from exc
    return overrides


def get_config(**overrides: Any) -> DPForgeConfig:
    """Create a :class:`DPForgeConfig` with defaults, env overrides, and kwargs.

    Keyword arguments take highest precedence, then environment variables,
    then built-in defaults.

    Args:
        **overrides: Any parameter name from SynthesisConfig, NumericalConfig,
            or SamplingConfig (flat namespace).

    Returns:
        A fully validated DPForgeConfig.

    Raises:
        ConfigurationError: On invalid parameters or missing solvers.
    """
    merged = dict(_DEFAULTS)
    merged = _apply_env_overrides(merged)
    merged.update(overrides)

    numerical = NumericalConfig(
        solver_tol=merged["solver_tol"],
        dp_tol=merged["dp_tol"],
        eta_min_scale=merged["eta_min_scale"],
        max_condition_number=merged["max_condition_number"],
    )
    sampling = SamplingConfig(
        method=merged["sampling_method"],
        seed=merged["sampling_seed"],
    )
    solver = merged["solver"]
    if isinstance(solver, str):
        solver = SolverBackend(solver)

    synthesis = SynthesisConfig(
        max_iter=merged["max_iter"],
        tol=merged["tol"],
        warm_start=merged["warm_start"],
        solver=solver,
        verbose=merged["verbose"],
        symmetry_detection=merged["symmetry_detection"],
        numerical=numerical,
        sampling=sampling,
    )
    return DPForgeConfig(synthesis=synthesis, numerical=numerical, sampling=sampling)
