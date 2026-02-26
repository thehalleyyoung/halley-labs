"""
Verification trace: records all data needed for independent certificate checking.

A VerificationTrace captures per-step states, errors, and bounds so that
an independent verifier can re-derive error certificates without re-running
the full computation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Record of a single integration/checking step."""
    step_index: int
    time: float
    truncation_error: float
    clamping_error: float
    bond_dims: list[int]
    total_probability: float
    negativity_mass: float = 0.0
    integration_error: float = 0.0
    method: str = ""

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "time": self.time,
            "truncation_error": self.truncation_error,
            "clamping_error": self.clamping_error,
            "bond_dims": self.bond_dims,
            "total_probability": self.total_probability,
            "negativity_mass": self.negativity_mass,
            "integration_error": self.integration_error,
            "method": self.method,
        }


@dataclass
class FSPBoundRecord:
    """Record of FSP (Finite State Projection) truncation bounds."""
    physical_dims: list[int]
    fsp_error_bound: float
    truncated_mass: float = 0.0

    def to_dict(self) -> dict:
        return {
            "physical_dims": self.physical_dims,
            "fsp_error_bound": self.fsp_error_bound,
            "truncated_mass": self.truncated_mass,
        }


@dataclass
class CSLCheckRecord:
    """Record of a CSL property check."""
    formula_str: str
    probability_lower: float
    probability_upper: float
    verdict: str  # "true", "false", "indeterminate"
    total_certified_error: float
    fixpoint_iterations: int = 0
    converged: bool = True
    spectral_gap_estimate: float = 0.0
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {
            "formula_str": self.formula_str,
            "probability_lower": self.probability_lower,
            "probability_upper": self.probability_upper,
            "verdict": self.verdict,
            "total_certified_error": self.total_certified_error,
            "fixpoint_iterations": self.fixpoint_iterations,
            "converged": self.converged,
            "spectral_gap_estimate": self.spectral_gap_estimate,
            "fallback_used": self.fallback_used,
        }


@dataclass
class ClampingProofRecord:
    """Record of per-step clamping proof data for independent verification."""
    step_index: int
    iteration_data: list[dict] = field(default_factory=list)
    final_negativity: float = 0.0
    bound_verified: bool = True

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "iteration_data": self.iteration_data,
            "final_negativity": self.final_negativity,
            "bound_verified": self.bound_verified,
        }


@dataclass
class VerificationTrace:
    """
    Complete verification trace for independent auditing.

    Contains all data needed to re-derive error certificates:
    - Per-step truncation and clamping errors
    - Bond dimension history
    - FSP bounds
    - CSL check results
    - Configuration parameters

    The trace is designed so that an independent verifier can:
    1. Check per-step error bounds are consistent with bond dims
    2. Verify clamping errors satisfy Proposition 1 (≤ 2·ε_trunc)
    3. Validate error composition via triangle inequality
    4. Confirm probability conservation
    5. Audit FSP truncation bounds
    """
    model_name: str = ""
    num_species: int = 0
    physical_dims: list[int] = field(default_factory=list)
    time_horizon: float = 0.0
    max_bond_dim: int = 0
    truncation_tolerance: float = 0.0
    integration_method: str = ""

    steps: list[StepRecord] = field(default_factory=list)
    fsp_bounds: Optional[FSPBoundRecord] = None
    csl_checks: list[CSLCheckRecord] = field(default_factory=list)
    clamping_proofs: list[ClampingProofRecord] = field(default_factory=list)

    # Aggregate error tracking
    total_truncation_error: float = 0.0
    total_clamping_error: float = 0.0
    total_fsp_error: float = 0.0
    total_certified_error: float = 0.0

    def record_step(
        self,
        step_index: int,
        time: float,
        truncation_error: float,
        clamping_error: float,
        bond_dims: list[int],
        total_probability: float,
        negativity_mass: float = 0.0,
        integration_error: float = 0.0,
        method: str = "",
    ) -> None:
        """Record a single computation step."""
        record = StepRecord(
            step_index=step_index,
            time=time,
            truncation_error=truncation_error,
            clamping_error=clamping_error,
            bond_dims=bond_dims,
            total_probability=total_probability,
            negativity_mass=negativity_mass,
            integration_error=integration_error,
            method=method,
        )
        self.steps.append(record)
        self.total_truncation_error += truncation_error
        self.total_clamping_error += clamping_error

    def record_fsp_bounds(
        self,
        physical_dims: list[int],
        fsp_error_bound: float,
        truncated_mass: float = 0.0,
    ) -> None:
        """Record FSP truncation bounds."""
        self.fsp_bounds = FSPBoundRecord(
            physical_dims=physical_dims,
            fsp_error_bound=fsp_error_bound,
            truncated_mass=truncated_mass,
        )
        self.total_fsp_error = fsp_error_bound

    def record_clamping_proof(
        self,
        step_index: int,
        iteration_data: list[dict],
        final_negativity: float,
        bound_verified: bool,
    ) -> None:
        """Record per-step clamping proof data for independent verification."""
        record = ClampingProofRecord(
            step_index=step_index,
            iteration_data=iteration_data,
            final_negativity=final_negativity,
            bound_verified=bound_verified,
        )
        self.clamping_proofs.append(record)

    def record_csl_check(
        self,
        formula_str: str,
        probability_lower: float,
        probability_upper: float,
        verdict: str,
        total_certified_error: float,
        fixpoint_iterations: int = 0,
        converged: bool = True,
        spectral_gap_estimate: float = 0.0,
        fallback_used: bool = False,
    ) -> None:
        """Record a CSL property check result."""
        record = CSLCheckRecord(
            formula_str=formula_str,
            probability_lower=probability_lower,
            probability_upper=probability_upper,
            verdict=verdict,
            total_certified_error=total_certified_error,
            fixpoint_iterations=fixpoint_iterations,
            converged=converged,
            spectral_gap_estimate=spectral_gap_estimate,
            fallback_used=fallback_used,
        )
        self.csl_checks.append(record)

    def finalize(self) -> None:
        """Compute aggregate error bounds."""
        self.total_truncation_error = sum(s.truncation_error for s in self.steps)
        self.total_clamping_error = sum(s.clamping_error for s in self.steps)
        clamping_bound = min(
            self.total_clamping_error,
            2 * self.total_truncation_error,
        )
        self.total_certified_error = (
            self.total_truncation_error
            + clamping_bound
            + self.total_fsp_error
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        self.finalize()
        return {
            "model_name": self.model_name,
            "num_species": self.num_species,
            "physical_dims": self.physical_dims,
            "time_horizon": self.time_horizon,
            "max_bond_dim": self.max_bond_dim,
            "truncation_tolerance": self.truncation_tolerance,
            "integration_method": self.integration_method,
            "num_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "fsp_bounds": self.fsp_bounds.to_dict() if self.fsp_bounds else None,
            "csl_checks": [c.to_dict() for c in self.csl_checks],
            "clamping_proofs": [p.to_dict() for p in self.clamping_proofs],
            "total_truncation_error": self.total_truncation_error,
            "total_clamping_error": self.total_clamping_error,
            "total_fsp_error": self.total_fsp_error,
            "total_certified_error": self.total_certified_error,
        }

    def to_json(self, path: str) -> None:
        """Export trace to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Verification trace written to {path}")

    @classmethod
    def from_json(cls, path: str) -> "VerificationTrace":
        """Load trace from JSON file."""
        with open(path) as f:
            data = json.load(f)

        trace = cls(
            model_name=data.get("model_name", ""),
            num_species=data.get("num_species", 0),
            physical_dims=data.get("physical_dims", []),
            time_horizon=data.get("time_horizon", 0.0),
            max_bond_dim=data.get("max_bond_dim", 0),
            truncation_tolerance=data.get("truncation_tolerance", 0.0),
            integration_method=data.get("integration_method", ""),
        )

        for s in data.get("steps", []):
            trace.steps.append(StepRecord(**s))

        if data.get("fsp_bounds"):
            trace.fsp_bounds = FSPBoundRecord(**data["fsp_bounds"])

        for c in data.get("csl_checks", []):
            trace.csl_checks.append(CSLCheckRecord(**c))

        for p in data.get("clamping_proofs", []):
            trace.clamping_proofs.append(ClampingProofRecord(**p))

        trace.finalize()
        return trace
