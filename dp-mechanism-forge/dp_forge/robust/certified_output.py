"""
Certified mechanism wrapper with numerical certificates and error bounds.

A :class:`CertifiedMechanism` bundles a synthesised DP mechanism with a
numerical certificate proving that the mechanism satisfies (ε+ε_ν, δ+δ_ν)-DP
despite solver imprecision.  The certificate records the solver tolerance,
inflation margins, interval-verification results, and perturbation bounds
so that the guarantee can be independently verified.

Key class:
    - :class:`CertifiedMechanism` — Mechanism + certificate.
    - :class:`NumericalCertificate` — Certificate data.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.robust.interval_arithmetic import interval_verify_dp

logger = logging.getLogger(__name__)


@dataclass
class NumericalCertificate:
    """Certificate of numerical robustness for a DP mechanism.

    Records all parameters and analysis results needed to verify that
    the mechanism's privacy guarantee is sound despite floating-point
    solver imprecision.

    Attributes:
        solver_tolerance: Solver feasibility tolerance ν used during synthesis.
        epsilon_target: Target privacy parameter ε.
        delta_target: Target privacy parameter δ.
        epsilon_margin: Inflation margin consumed for ε.
        delta_margin: Inflation margin consumed for δ.
        epsilon_effective: Guaranteed effective ε = epsilon_target + epsilon_margin.
        delta_effective: Guaranteed effective δ = delta_target + delta_margin.
        interval_verified: Whether interval arithmetic verification passed.
        condition_number: Estimated LP basis condition number.
        max_constraint_violation: Maximum constraint violation in the solution.
        perturbation_epsilon_bound: Upper bound on Δε from perturbation analysis.
        perturbation_delta_bound: Upper bound on Δδ from perturbation analysis.
        synthesis_time: Wall-clock time for robust synthesis (seconds).
        cegis_iterations: Number of CEGIS iterations.
        timestamp: Unix timestamp when the certificate was created.
    """

    solver_tolerance: float
    epsilon_target: float
    delta_target: float
    epsilon_margin: float
    delta_margin: float
    epsilon_effective: float
    delta_effective: float
    interval_verified: bool
    condition_number: float
    max_constraint_violation: float
    perturbation_epsilon_bound: float
    perturbation_delta_bound: float
    synthesis_time: float
    cegis_iterations: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the certificate to a plain dictionary."""
        return {
            "solver_tolerance": self.solver_tolerance,
            "epsilon_target": self.epsilon_target,
            "delta_target": self.delta_target,
            "epsilon_margin": self.epsilon_margin,
            "delta_margin": self.delta_margin,
            "epsilon_effective": self.epsilon_effective,
            "delta_effective": self.delta_effective,
            "interval_verified": self.interval_verified,
            "condition_number": self.condition_number,
            "max_constraint_violation": self.max_constraint_violation,
            "perturbation_epsilon_bound": self.perturbation_epsilon_bound,
            "perturbation_delta_bound": self.perturbation_delta_bound,
            "synthesis_time": self.synthesis_time,
            "cegis_iterations": self.cegis_iterations,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NumericalCertificate:
        """Deserialise from a dictionary."""
        return cls(
            solver_tolerance=float(d["solver_tolerance"]),
            epsilon_target=float(d["epsilon_target"]),
            delta_target=float(d["delta_target"]),
            epsilon_margin=float(d["epsilon_margin"]),
            delta_margin=float(d["delta_margin"]),
            epsilon_effective=float(d["epsilon_effective"]),
            delta_effective=float(d["delta_effective"]),
            interval_verified=bool(d["interval_verified"]),
            condition_number=float(d["condition_number"]),
            max_constraint_violation=float(d["max_constraint_violation"]),
            perturbation_epsilon_bound=float(d["perturbation_epsilon_bound"]),
            perturbation_delta_bound=float(d["perturbation_delta_bound"]),
            synthesis_time=float(d["synthesis_time"]),
            cegis_iterations=int(d["cegis_iterations"]),
            timestamp=float(d.get("timestamp", 0.0)),
        )

    def __repr__(self) -> str:
        iv = "✓" if self.interval_verified else "✗"
        return (
            f"NumericalCertificate(ε_eff={self.epsilon_effective:.6f}, "
            f"δ_eff={self.delta_effective:.2e}, "
            f"interval={iv}, κ={self.condition_number:.2e})"
        )


class CertifiedMechanism:
    """A DP mechanism with a numerical robustness certificate.

    Wraps the n×k probability table with a :class:`NumericalCertificate`
    that records the solver tolerance, inflation margins, interval
    verification status, and perturbation bounds.  The certificate
    guarantees (ε_effective, δ_effective)-DP.

    Attributes:
        mechanism: Probability table p[i][j] = Pr[M(x_i) = y_j], shape (n, k).
        y_grid: Output discretisation grid, shape (k,).
        certificate: Numerical robustness certificate.
        edges: Adjacent database pairs used for DP verification.
        metadata: Additional metadata from synthesis.

    Example::

        cm = CertifiedMechanism(mechanism=p, y_grid=y, certificate=cert, ...)
        assert cm.verify_certificate()
        cm.to_json("mechanism.json")
        cm2 = CertifiedMechanism.from_json("mechanism.json")
    """

    def __init__(
        self,
        mechanism: npt.NDArray[np.float64],
        y_grid: npt.NDArray[np.float64],
        certificate: NumericalCertificate,
        edges: List[Tuple[int, int]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mechanism = np.asarray(mechanism, dtype=np.float64)
        self.y_grid = np.asarray(y_grid, dtype=np.float64)
        self.certificate = certificate
        self.edges = edges
        self.metadata = metadata or {}

        if self.mechanism.ndim != 2:
            raise ValueError(
                f"mechanism must be 2-D (n×k), got shape {self.mechanism.shape}"
            )
        if len(self.y_grid) != self.mechanism.shape[1]:
            raise ValueError(
                f"y_grid length {len(self.y_grid)} != mechanism columns "
                f"{self.mechanism.shape[1]}"
            )

    @property
    def n(self) -> int:
        """Number of database inputs."""
        return self.mechanism.shape[0]

    @property
    def k(self) -> int:
        """Number of output bins."""
        return self.mechanism.shape[1]

    @property
    def epsilon_effective(self) -> float:
        """Guaranteed effective privacy parameter ε."""
        return self.certificate.epsilon_effective

    @property
    def delta_effective(self) -> float:
        """Guaranteed effective privacy parameter δ."""
        return self.certificate.delta_effective

    def verify_certificate(self, strict: bool = False) -> bool:
        """Re-verify the numerical certificate against the mechanism.

        Checks:
        1. Mechanism rows sum to ~1 (valid probability distributions).
        2. All probabilities are non-negative.
        3. Interval arithmetic verification at (ε_effective, δ_effective).
        4. Certificate consistency (margins are non-negative, etc.).

        Args:
            strict: If True, require interval verification to pass.
                If False, only check structural validity.

        Returns:
            True if the certificate is valid.
        """
        cert = self.certificate

        # Check structural validity
        if cert.epsilon_margin < 0 or cert.delta_margin < 0:
            logger.warning("Certificate has negative margins")
            return False

        if cert.epsilon_effective < cert.epsilon_target:
            logger.warning("epsilon_effective < epsilon_target")
            return False

        if cert.delta_effective < cert.delta_target:
            logger.warning("delta_effective < delta_target")
            return False

        # Validate mechanism structure
        p = self.mechanism
        if np.any(p < -1e-12):
            logger.warning("Mechanism contains negative probabilities")
            return False

        row_sums = p.sum(axis=1)
        if float(np.max(np.abs(row_sums - 1.0))) > 1e-6:
            logger.warning("Mechanism rows do not sum to 1")
            return False

        if strict:
            # Interval verification: use solver_tol as the uncertainty on each p[i][j]
            nu = cert.solver_tolerance
            p_lo = np.maximum(p - nu, 0.0)
            p_hi = p + nu

            valid, violation = interval_verify_dp(
                p_lo, p_hi,
                cert.epsilon_effective,
                self.edges,
                delta=cert.delta_effective,
            )
            if not valid:
                logger.warning(
                    "Interval verification failed at effective parameters: %s",
                    violation,
                )
                return False

        return True

    def to_json(self, path: str) -> None:
        """Serialise the certified mechanism to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "mechanism": self.mechanism.tolist(),
            "y_grid": self.y_grid.tolist(),
            "certificate": self.certificate.to_dict(),
            "edges": self.edges,
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Wrote certified mechanism to %s", path)

    @classmethod
    def from_json(cls, path: str) -> CertifiedMechanism:
        """Load a certified mechanism from a JSON file.

        Args:
            path: Input file path.

        Returns:
            CertifiedMechanism instance.
        """
        with open(path, "r") as f:
            data = json.load(f)

        mechanism = np.array(data["mechanism"], dtype=np.float64)
        y_grid = np.array(data["y_grid"], dtype=np.float64)
        certificate = NumericalCertificate.from_dict(data["certificate"])
        edges = [tuple(e) for e in data["edges"]]
        metadata = data.get("metadata", {})

        return cls(
            mechanism=mechanism,
            y_grid=y_grid,
            certificate=certificate,
            edges=edges,
            metadata=metadata,
        )

    def summary(self) -> str:
        """Human-readable summary of the certified mechanism."""
        cert = self.certificate
        iv = "PASSED" if cert.interval_verified else "FAILED"
        lines = [
            f"CertifiedMechanism (n={self.n}, k={self.k})",
            f"  Target:    (ε={cert.epsilon_target:.6f}, δ={cert.delta_target:.2e})",
            f"  Effective: (ε={cert.epsilon_effective:.6f}, δ={cert.delta_effective:.2e})",
            f"  Margins:   ε_margin={cert.epsilon_margin:.2e}, δ_margin={cert.delta_margin:.2e}",
            f"  Solver tolerance: {cert.solver_tolerance:.2e}",
            f"  Condition number: {cert.condition_number:.2e}",
            f"  Max constraint violation: {cert.max_constraint_violation:.2e}",
            f"  Interval verified: {iv}",
            f"  CEGIS iterations: {cert.cegis_iterations}",
            f"  Synthesis time: {cert.synthesis_time:.3f}s",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        iv = "✓" if self.certificate.interval_verified else "✗"
        return (
            f"CertifiedMechanism(n={self.n}, k={self.k}, "
            f"ε_eff={self.epsilon_effective:.4f}, "
            f"δ_eff={self.delta_effective:.2e}, interval={iv})"
        )
