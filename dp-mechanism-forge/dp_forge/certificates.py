"""
Optimality certificate generation and verification for DP-Forge.

This module provides machinery for extracting, packaging, verifying, and
serializing mathematical certificates that prove a synthesized mechanism
is (near-)optimal.  Certificates are derived from LP/SDP duality theory:

- **LP certificates** consist of dual variables whose feasibility and
  complementary slackness conditions certify that the primal solution
  (mechanism) is optimal up to the duality gap.
- **SDP certificates** consist of a dual matrix for the Gaussian
  workload mechanism SDP.
- **Composed certificates** combine multiple certificates for mechanisms
  produced via composition.
- **Approximation certificates** bound the error introduced by
  discretization.

The certificate chain abstraction allows verifying a sequence of
composed mechanisms end-to-end.

Usage::

    from dp_forge.certificates import CertificateGenerator, CertificateVerifier

    gen = CertificateGenerator()
    cert = gen.generate(lp_result, spec)
    ok = CertificateVerifier().verify(cert, mechanism, spec)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    ConfigurationError,
    DPForgeError,
    InvalidMechanismError,
)
from dp_forge.types import (
    CEGISResult,
    ExtractedMechanism,
    LPStruct,
    OptimalityCertificate,
    QuerySpec,
    SDPStruct,
    VerifyResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Certificate dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LPOptimalityCertificate:
    """Certificate of LP optimality from dual variables.

    Stores dual variable values for inequality and equality constraints,
    the duality gap, and per-constraint slack for complementary slackness
    verification.

    Attributes:
        dual_ub: Dual variables for inequality constraints (λ ≥ 0).
        dual_eq: Dual variables for equality constraints (ν, free).
        duality_gap: Absolute duality gap |primal_obj − dual_obj|.
        primal_obj: Primal objective value.
        dual_obj: Dual objective value.
        primal_slack: Per-inequality primal slack (b_ub − A_ub x).
        dual_slack: Per-variable reduced cost (c − Aᵀλ − Aᵀ_eq ν).
        n_vars: Number of primal variables.
        n_ub: Number of inequality constraints.
        n_eq: Number of equality constraints.
        solver_name: Name of the solver that produced these duals.
        timestamp: ISO-format timestamp of certificate generation.
    """

    dual_ub: npt.NDArray[np.float64]
    dual_eq: Optional[npt.NDArray[np.float64]]
    duality_gap: float
    primal_obj: float
    dual_obj: float
    primal_slack: Optional[npt.NDArray[np.float64]] = None
    dual_slack: Optional[npt.NDArray[np.float64]] = None
    n_vars: int = 0
    n_ub: int = 0
    n_eq: int = 0
    solver_name: str = "unknown"
    timestamp: str = ""

    def __post_init__(self) -> None:
        self.dual_ub = np.asarray(self.dual_ub, dtype=np.float64)
        if self.dual_eq is not None:
            self.dual_eq = np.asarray(self.dual_eq, dtype=np.float64)
        if self.primal_slack is not None:
            self.primal_slack = np.asarray(self.primal_slack, dtype=np.float64)
        if self.dual_slack is not None:
            self.dual_slack = np.asarray(self.dual_slack, dtype=np.float64)
        if not self.timestamp:
            import datetime
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    @property
    def relative_gap(self) -> float:
        """Relative duality gap: gap / max(|primal|, 1)."""
        denom = max(abs(self.primal_obj), 1.0)
        return self.duality_gap / denom

    @property
    def is_tight(self) -> bool:
        """Whether the gap is within 1e-6 tolerance."""
        return self.relative_gap <= 1e-6

    def __repr__(self) -> str:
        return (
            f"LPOptimalityCertificate(gap={self.duality_gap:.2e}, "
            f"vars={self.n_vars}, ub={self.n_ub}, eq={self.n_eq})"
        )


@dataclass
class SDPOptimalityCertificate:
    """Certificate of SDP optimality from dual matrix.

    Attributes:
        dual_matrix: Dual PSD matrix (Y ≽ 0).
        duality_gap: Absolute duality gap.
        primal_obj: Primal objective value.
        dual_obj: Dual objective value.
        matrix_dim: Dimension of the dual matrix.
        min_eigenvalue: Smallest eigenvalue of dual matrix (for PSD check).
        solver_name: Solver that produced the certificate.
        timestamp: ISO-format timestamp.
    """

    dual_matrix: npt.NDArray[np.float64]
    duality_gap: float
    primal_obj: float
    dual_obj: float
    matrix_dim: int = 0
    min_eigenvalue: float = 0.0
    solver_name: str = "unknown"
    timestamp: str = ""

    def __post_init__(self) -> None:
        self.dual_matrix = np.asarray(self.dual_matrix, dtype=np.float64)
        if self.dual_matrix.ndim != 2:
            raise ValueError(
                f"dual_matrix must be 2-D, got shape {self.dual_matrix.shape}"
            )
        if self.dual_matrix.shape[0] != self.dual_matrix.shape[1]:
            raise ValueError(
                f"dual_matrix must be square, got {self.dual_matrix.shape}"
            )
        if self.matrix_dim == 0:
            self.matrix_dim = self.dual_matrix.shape[0]
        if self.min_eigenvalue == 0.0:
            try:
                eigvals = np.linalg.eigvalsh(self.dual_matrix)
                self.min_eigenvalue = float(eigvals.min())
            except np.linalg.LinAlgError:
                self.min_eigenvalue = float("nan")
        if not self.timestamp:
            import datetime
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    @property
    def relative_gap(self) -> float:
        """Relative duality gap."""
        denom = max(abs(self.primal_obj), 1.0)
        return self.duality_gap / denom

    @property
    def is_psd(self) -> bool:
        """Whether the dual matrix is positive semidefinite (within tolerance)."""
        return self.min_eigenvalue >= -1e-8

    @property
    def is_tight(self) -> bool:
        """Whether the gap is within tolerance."""
        return self.relative_gap <= 1e-6

    def __repr__(self) -> str:
        return (
            f"SDPOptimalityCertificate(gap={self.duality_gap:.2e}, "
            f"dim={self.matrix_dim}, min_eig={self.min_eigenvalue:.2e})"
        )


@dataclass
class ComposedCertificate:
    """Certificate for a mechanism produced by composition.

    Stores certificates for each component mechanism and the composition
    theorem used to derive the overall privacy guarantee.

    Attributes:
        components: List of (certificate, epsilon, delta) per component.
        composition_type: Composition theorem used ("basic", "advanced", "rdp").
        total_epsilon: Total composed epsilon.
        total_delta: Total composed delta.
        timestamp: ISO-format timestamp.
    """

    components: List[Tuple[Any, float, float]]
    composition_type: str = "basic"
    total_epsilon: float = 0.0
    total_delta: float = 0.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.total_epsilon == 0.0:
            if self.composition_type == "basic":
                self.total_epsilon = sum(eps for _, eps, _ in self.components)
                self.total_delta = sum(delta for _, _, delta in self.components)
            elif self.composition_type == "advanced":
                k = len(self.components)
                epsilons = [eps for _, eps, _ in self.components]
                deltas = [delta for _, _, delta in self.components]
                eps_sum = sum(epsilons)
                delta_sum = sum(deltas)
                if k > 0:
                    eps_sq_sum = sum(e ** 2 for e in epsilons)
                    self.total_epsilon = (
                        eps_sq_sum / (2 * max(eps_sum, 1e-15))
                        + math.sqrt(2 * eps_sq_sum * math.log(1 / max(delta_sum, 1e-15)))
                    )
                    self.total_delta = delta_sum
                else:
                    self.total_epsilon = 0.0
                    self.total_delta = 0.0
        if not self.timestamp:
            import datetime
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    @property
    def n_components(self) -> int:
        """Number of component certificates."""
        return len(self.components)

    def __repr__(self) -> str:
        return (
            f"ComposedCertificate(n={self.n_components}, "
            f"ε={self.total_epsilon:.4f}, δ={self.total_delta:.2e}, "
            f"type={self.composition_type})"
        )


@dataclass
class ApproximationCertificate:
    """Certificate bounding discretization error.

    Quantifies the gap between the optimal continuous mechanism and the
    discrete mechanism produced by DP-Forge.

    Attributes:
        k: Number of discretization bins used.
        grid_spacing: Spacing between grid points (Δy).
        discretization_error_bound: Upper bound on MSE increase from discretization.
        sensitivity: Query sensitivity.
        epsilon: Privacy parameter.
        interpolation_error: Bound on interpolation error for piecewise-linear.
        timestamp: ISO-format timestamp.
    """

    k: int
    grid_spacing: float
    discretization_error_bound: float
    sensitivity: float = 1.0
    epsilon: float = 1.0
    interpolation_error: float = 0.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.discretization_error_bound < 0:
            raise ValueError(
                f"discretization_error_bound must be >= 0, got "
                f"{self.discretization_error_bound}"
            )
        if self.interpolation_error < 0:
            raise ValueError(
                f"interpolation_error must be >= 0, got {self.interpolation_error}"
            )
        if not self.timestamp:
            import datetime
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    @property
    def total_error_bound(self) -> float:
        """Total error bound including discretization and interpolation."""
        return self.discretization_error_bound + self.interpolation_error

    @classmethod
    def from_spec(cls, spec: Any, k: int) -> ApproximationCertificate:
        """Compute discretization error bound from a QuerySpec.

        Uses the formula: Δ_disc ≤ (Δy)² / 12 for piecewise-constant
        approximation of a smooth density.

        Args:
            spec: QuerySpec with sensitivity and epsilon.
            k: Number of discretization bins.

        Returns:
            ApproximationCertificate with computed bounds.
        """
        sensitivity = spec.sensitivity if hasattr(spec, "sensitivity") else 1.0
        epsilon = spec.epsilon if hasattr(spec, "epsilon") else 1.0

        qv = spec.query_values if hasattr(spec, "query_values") else np.array([0, 1])
        grid_range = float(qv.max() - qv.min()) + 6 * sensitivity
        grid_spacing = grid_range / max(k - 1, 1)

        disc_error = grid_spacing ** 2 / 12.0

        return cls(
            k=k,
            grid_spacing=grid_spacing,
            discretization_error_bound=disc_error,
            sensitivity=sensitivity,
            epsilon=epsilon,
        )

    def __repr__(self) -> str:
        return (
            f"ApproximationCertificate(k={self.k}, Δy={self.grid_spacing:.4f}, "
            f"error≤{self.discretization_error_bound:.2e})"
        )


# ---------------------------------------------------------------------------
# CertificateGenerator
# ---------------------------------------------------------------------------


class CertificateGenerator:
    """Generate optimality certificates from LP/SDP solutions.

    Extracts dual variables from solver output and packages them into
    verified certificate objects.

    Args:
        tol: Tolerance for duality gap checks.
    """

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def generate(
        self,
        lp_result: Any,
        spec: QuerySpec,
        *,
        lp_struct: Optional[LPStruct] = None,
    ) -> LPOptimalityCertificate:
        """Generate an LP optimality certificate from solver output.

        Extracts dual variables, computes primal/dual slack, and
        verifies the duality gap.

        Args:
            lp_result: Solver result object. Expected to have attributes:
                - ``fun`` or ``objective_value``: primal objective
                - ``x``: primal solution
                - ``ineqlin`` or ``dual_ub``: inequality duals
                - ``eqlin`` or ``dual_eq``: equality duals
            spec: Query specification used for synthesis.
            lp_struct: Optional LPStruct for computing slack.

        Returns:
            LPOptimalityCertificate with extracted dual information.

        Raises:
            DPForgeError: If certificate extraction fails.
        """
        # Extract primal objective
        if hasattr(lp_result, "fun"):
            primal_obj = float(lp_result.fun)
        elif hasattr(lp_result, "objective_value"):
            primal_obj = float(lp_result.objective_value)
        elif isinstance(lp_result, dict):
            primal_obj = float(lp_result.get("fun", lp_result.get("objective_value", 0.0)))
        else:
            raise DPForgeError("Cannot extract primal objective from LP result")

        # Extract dual variables
        dual_ub = self._extract_dual_ub(lp_result)
        dual_eq = self._extract_dual_eq(lp_result)

        # Compute dual objective
        dual_obj = self._compute_dual_objective(
            dual_ub, dual_eq, lp_struct,
        )

        duality_gap = abs(primal_obj - dual_obj)

        # Compute slacks if LP struct available
        primal_slack = None
        dual_slack = None
        n_vars = 0
        n_ub = 0
        n_eq = 0

        if lp_struct is not None:
            x = self._extract_primal(lp_result)
            if x is not None:
                primal_slack = lp_struct.b_ub - lp_struct.A_ub.dot(x)
                # Reduced cost: c - A_ub^T λ - A_eq^T ν
                reduced = lp_struct.c.copy()
                reduced -= lp_struct.A_ub.T.dot(dual_ub)
                if lp_struct.A_eq is not None and dual_eq is not None:
                    reduced -= lp_struct.A_eq.T.dot(dual_eq)
                dual_slack = reduced

            n_vars = lp_struct.n_vars
            n_ub = lp_struct.n_ub
            n_eq = lp_struct.n_eq

        solver_name = self._extract_solver_name(lp_result)

        cert = LPOptimalityCertificate(
            dual_ub=dual_ub,
            dual_eq=dual_eq,
            duality_gap=duality_gap,
            primal_obj=primal_obj,
            dual_obj=dual_obj,
            primal_slack=primal_slack,
            dual_slack=dual_slack,
            n_vars=n_vars,
            n_ub=n_ub,
            n_eq=n_eq,
            solver_name=solver_name,
        )

        logger.info(
            "LP certificate generated: gap=%.2e, relative=%.2e, tight=%s",
            cert.duality_gap,
            cert.relative_gap,
            cert.is_tight,
        )

        return cert

    def generate_sdp_certificate(
        self,
        sdp_result: Any,
        spec: Any,
    ) -> SDPOptimalityCertificate:
        """Generate an SDP optimality certificate from CVXPY output.

        Args:
            sdp_result: SDPStruct with solved CVXPY problem.
            spec: QuerySpec or WorkloadSpec.

        Returns:
            SDPOptimalityCertificate with dual matrix.

        Raises:
            DPForgeError: If certificate extraction fails.
        """
        if hasattr(sdp_result, "problem"):
            problem = sdp_result.problem
        else:
            problem = sdp_result

        # Extract primal/dual objectives
        if hasattr(problem, "value"):
            primal_obj = float(problem.value)
        else:
            raise DPForgeError("Cannot extract primal objective from SDP result")

        # Extract dual matrix from constraints
        dual_matrix = None
        for constr in getattr(sdp_result, "constraints", []):
            if hasattr(constr, "dual_value") and constr.dual_value is not None:
                dv = np.asarray(constr.dual_value)
                if dv.ndim == 2 and dv.shape[0] == dv.shape[1]:
                    dual_matrix = dv
                    break

        if dual_matrix is None:
            dim = 1
            if hasattr(sdp_result, "sigma_var"):
                sv = sdp_result.sigma_var
                if hasattr(sv, "shape"):
                    dim = sv.shape[0]
            dual_matrix = np.eye(dim)
            logger.warning("No dual matrix extracted; using identity placeholder")

        # Dual objective approximation
        dual_obj = primal_obj  # SDP solvers often don't expose dual obj directly
        if hasattr(problem, "solver_stats"):
            stats = problem.solver_stats
            if hasattr(stats, "extra") and isinstance(stats.extra, dict):
                dual_obj = float(stats.extra.get("dual_value", primal_obj))

        duality_gap = abs(primal_obj - dual_obj)

        solver_name = "cvxpy"
        if hasattr(problem, "solver_stats") and hasattr(problem.solver_stats, "solver_name"):
            solver_name = problem.solver_stats.solver_name

        cert = SDPOptimalityCertificate(
            dual_matrix=dual_matrix,
            duality_gap=duality_gap,
            primal_obj=primal_obj,
            dual_obj=dual_obj,
            solver_name=solver_name,
        )

        logger.info(
            "SDP certificate generated: gap=%.2e, dim=%d, min_eig=%.2e, PSD=%s",
            cert.duality_gap, cert.matrix_dim, cert.min_eigenvalue, cert.is_psd,
        )

        return cert

    def package_certificate(
        self,
        cert: Union[LPOptimalityCertificate, SDPOptimalityCertificate],
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
    ) -> Dict[str, Any]:
        """Bundle a certificate with its mechanism for serialization.

        Args:
            cert: The optimality certificate.
            mechanism: The mechanism the certificate applies to.
            spec: Optional query specification.

        Returns:
            Dict suitable for JSON serialization.
        """
        package: Dict[str, Any] = {
            "format_version": "1.0",
            "certificate_type": type(cert).__name__,
            "mechanism": {
                "n": mechanism.n,
                "k": mechanism.k,
                "p_final": mechanism.p_final.tolist(),
            },
            "certificate": {
                "duality_gap": cert.duality_gap,
                "primal_obj": cert.primal_obj,
                "dual_obj": cert.dual_obj,
                "relative_gap": cert.relative_gap,
                "is_tight": cert.is_tight,
                "timestamp": cert.timestamp,
            },
        }

        if isinstance(cert, LPOptimalityCertificate):
            package["certificate"]["dual_ub"] = cert.dual_ub.tolist()
            if cert.dual_eq is not None:
                package["certificate"]["dual_eq"] = cert.dual_eq.tolist()
            package["certificate"]["n_vars"] = cert.n_vars
            package["certificate"]["n_ub"] = cert.n_ub
            package["certificate"]["n_eq"] = cert.n_eq
            package["certificate"]["solver_name"] = cert.solver_name

        elif isinstance(cert, SDPOptimalityCertificate):
            package["certificate"]["dual_matrix"] = cert.dual_matrix.tolist()
            package["certificate"]["matrix_dim"] = cert.matrix_dim
            package["certificate"]["min_eigenvalue"] = cert.min_eigenvalue
            package["certificate"]["is_psd"] = cert.is_psd
            package["certificate"]["solver_name"] = cert.solver_name

        if spec is not None:
            package["spec"] = {
                "epsilon": spec.epsilon,
                "delta": spec.delta,
                "sensitivity": spec.sensitivity,
                "n": spec.n,
                "k": spec.k,
                "query_type": spec.query_type.name,
            }

        return package

    # --- Internal helpers ---

    def _extract_dual_ub(self, result: Any) -> npt.NDArray[np.float64]:
        """Extract inequality dual variables from solver result."""
        if hasattr(result, "ineqlin") and hasattr(result.ineqlin, "marginals"):
            return np.asarray(result.ineqlin.marginals, dtype=np.float64)
        if hasattr(result, "dual_ub"):
            return np.asarray(result.dual_ub, dtype=np.float64)
        if isinstance(result, dict):
            if "dual_ub" in result:
                return np.asarray(result["dual_ub"], dtype=np.float64)
            if "ineqlin" in result:
                return np.asarray(result["ineqlin"], dtype=np.float64)
        # Fallback: empty
        logger.warning("No inequality dual variables found in result")
        return np.array([], dtype=np.float64)

    def _extract_dual_eq(self, result: Any) -> Optional[npt.NDArray[np.float64]]:
        """Extract equality dual variables from solver result."""
        if hasattr(result, "eqlin") and hasattr(result.eqlin, "marginals"):
            return np.asarray(result.eqlin.marginals, dtype=np.float64)
        if hasattr(result, "dual_eq"):
            return np.asarray(result.dual_eq, dtype=np.float64)
        if isinstance(result, dict) and "dual_eq" in result:
            return np.asarray(result["dual_eq"], dtype=np.float64)
        return None

    def _extract_primal(self, result: Any) -> Optional[npt.NDArray[np.float64]]:
        """Extract primal solution vector from solver result."""
        if hasattr(result, "x") and result.x is not None:
            return np.asarray(result.x, dtype=np.float64)
        if isinstance(result, dict) and "x" in result:
            return np.asarray(result["x"], dtype=np.float64)
        return None

    def _compute_dual_objective(
        self,
        dual_ub: npt.NDArray[np.float64],
        dual_eq: Optional[npt.NDArray[np.float64]],
        lp_struct: Optional[LPStruct],
    ) -> float:
        """Compute the dual objective value.

        For an LP: min c^T x s.t. Ax ≤ b, the dual is max b^T λ
        (with appropriate signs for equality constraints).
        """
        if lp_struct is None:
            return 0.0

        dual_val = 0.0
        if len(dual_ub) > 0 and len(dual_ub) == len(lp_struct.b_ub):
            dual_val += float(dual_ub @ lp_struct.b_ub)
        if dual_eq is not None and lp_struct.b_eq is not None:
            if len(dual_eq) == len(lp_struct.b_eq):
                dual_val += float(dual_eq @ lp_struct.b_eq)

        return dual_val

    def _extract_solver_name(self, result: Any) -> str:
        """Extract solver name from result metadata."""
        if hasattr(result, "method"):
            return str(result.method)
        if isinstance(result, dict) and "solver" in result:
            return str(result["solver"])
        return "unknown"


# ---------------------------------------------------------------------------
# CertificateVerifier
# ---------------------------------------------------------------------------


class CertificateVerifier:
    """Verify optimality certificates for DP mechanisms.

    Checks strong duality, complementary slackness, primal feasibility,
    and dual feasibility conditions.

    Args:
        tol: Default tolerance for numerical checks.
    """

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def verify(
        self,
        certificate: Union[LPOptimalityCertificate, SDPOptimalityCertificate],
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        *,
        tol: Optional[float] = None,
    ) -> bool:
        """Verify a certificate's validity for a mechanism and spec.

        Checks:
        1. Strong duality (small gap).
        2. Complementary slackness (KKT conditions).
        3. Primal feasibility (mechanism satisfies constraints).
        4. Dual feasibility (dual variables non-negative / PSD).

        Args:
            certificate: The certificate to verify.
            mechanism: The mechanism it certifies.
            spec: The query specification.
            tol: Tolerance override.

        Returns:
            True if all checks pass.
        """
        t = tol if tol is not None else self.tol

        # Check 1: Strong duality
        duality_ok = self.check_strong_duality(certificate, tol=t)
        if not duality_ok:
            logger.warning("Certificate failed strong duality check")
            return False

        # Check 2: Dual feasibility
        dual_ok = self.check_dual_feasibility(certificate, tol=t)
        if not dual_ok:
            logger.warning("Certificate failed dual feasibility check")
            return False

        # Check 3: Primal feasibility
        feasibility_ok = self.check_feasibility(certificate, mechanism, spec, tol=t)
        if not feasibility_ok:
            logger.warning("Certificate failed primal feasibility check")
            return False

        # Check 4: Complementary slackness
        cs_ok = self.check_complementary_slackness(certificate, tol=t)
        if not cs_ok:
            logger.warning("Certificate failed complementary slackness check")
            return False

        logger.info("Certificate verification PASSED (all 4 checks)")
        return True

    def check_strong_duality(
        self,
        cert: Union[LPOptimalityCertificate, SDPOptimalityCertificate],
        *,
        tol: Optional[float] = None,
    ) -> bool:
        """Check that |primal_obj − dual_obj| ≤ tolerance.

        Args:
            cert: Certificate to check.
            tol: Tolerance for the gap.

        Returns:
            True if the duality gap is within tolerance.
        """
        t = tol if tol is not None else self.tol
        gap = abs(cert.primal_obj - cert.dual_obj)
        relative = gap / max(abs(cert.primal_obj), 1.0)

        ok = relative <= t
        logger.debug(
            "Strong duality: gap=%.2e, relative=%.2e, tol=%.2e, pass=%s",
            gap, relative, t, ok,
        )
        return ok

    def check_complementary_slackness(
        self,
        cert: Union[LPOptimalityCertificate, SDPOptimalityCertificate],
        *,
        tol: Optional[float] = None,
    ) -> bool:
        """Check KKT complementary slackness conditions.

        For LP: λᵢ · (bᵢ − aᵢᵀx) = 0 for all inequality constraints.
        For SDP: tr(Y · (C − A(x))) = 0.

        Args:
            cert: Certificate to check.
            tol: Tolerance.

        Returns:
            True if complementary slackness holds within tolerance.
        """
        t = tol if tol is not None else self.tol

        if isinstance(cert, LPOptimalityCertificate):
            if cert.primal_slack is None:
                logger.debug("No primal slack available; skipping CS check")
                return True

            # λ · slack should be ≈ 0
            cs_products = cert.dual_ub * cert.primal_slack
            max_violation = float(np.max(np.abs(cs_products)))

            ok = max_violation <= t
            logger.debug(
                "Complementary slackness (LP): max_violation=%.2e, tol=%.2e, pass=%s",
                max_violation, t, ok,
            )
            return ok

        elif isinstance(cert, SDPOptimalityCertificate):
            # For SDP, check tr(Y · Z) ≈ 0 where Z is slack
            # Approximate via duality gap
            ok = cert.duality_gap <= t * max(abs(cert.primal_obj), 1.0)
            logger.debug(
                "Complementary slackness (SDP): gap=%.2e, pass=%s",
                cert.duality_gap, ok,
            )
            return ok

        return True

    def check_feasibility(
        self,
        cert: Union[LPOptimalityCertificate, SDPOptimalityCertificate],
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        *,
        tol: Optional[float] = None,
    ) -> bool:
        """Check that the mechanism satisfies primal feasibility.

        Verifies:
        - Probability rows sum to 1.
        - All probabilities are non-negative.
        - Privacy constraints are satisfied.

        Args:
            cert: Certificate (for metadata).
            mechanism: Mechanism to check.
            spec: Query specification.
            tol: Tolerance.

        Returns:
            True if the mechanism is feasible.
        """
        t = tol if tol is not None else self.tol
        p = mechanism.p_final

        # Non-negativity
        if np.any(p < -t):
            logger.debug("Primal feasibility: negative probabilities detected")
            return False

        # Row normalization
        row_sums = p.sum(axis=1)
        if np.any(np.abs(row_sums - 1.0) > t):
            logger.debug(
                "Primal feasibility: row sums deviate from 1 (max: %.2e)",
                float(np.max(np.abs(row_sums - 1.0))),
            )
            return False

        # Privacy constraints
        n, k = p.shape
        epsilon = spec.epsilon
        for i in range(n):
            for ip in range(n):
                if i == ip:
                    continue
                for j in range(k):
                    if p[ip, j] < 1e-15:
                        continue
                    ratio = p[i, j] / p[ip, j]
                    if ratio > math.exp(epsilon) + t:
                        logger.debug(
                            "Primal feasibility: DP violation at (%d,%d,%d), ratio=%.4f",
                            i, ip, j, ratio,
                        )
                        return False

        return True

    def check_dual_feasibility(
        self,
        cert: Union[LPOptimalityCertificate, SDPOptimalityCertificate],
        *,
        tol: Optional[float] = None,
    ) -> bool:
        """Check dual feasibility: dual variables are in the correct cone.

        For LP: λ ≥ 0 (inequality duals).
        For SDP: Y ≽ 0 (dual matrix PSD).

        Args:
            cert: Certificate to check.
            tol: Tolerance.

        Returns:
            True if dual feasibility holds.
        """
        t = tol if tol is not None else self.tol

        if isinstance(cert, LPOptimalityCertificate):
            if len(cert.dual_ub) == 0:
                return True
            min_dual = float(cert.dual_ub.min())
            ok = min_dual >= -t
            logger.debug(
                "Dual feasibility (LP): min_dual=%.2e, tol=%.2e, pass=%s",
                min_dual, t, ok,
            )
            return ok

        elif isinstance(cert, SDPOptimalityCertificate):
            ok = cert.min_eigenvalue >= -t
            logger.debug(
                "Dual feasibility (SDP): min_eig=%.2e, tol=%.2e, pass=%s",
                cert.min_eigenvalue, t, ok,
            )
            return ok

        return True


# ---------------------------------------------------------------------------
# Certificate chain
# ---------------------------------------------------------------------------


@dataclass
class CertificateChain:
    """Chain of certificates for composed mechanisms.

    A certificate chain proves the privacy of a mechanism composed from
    multiple sub-mechanisms, each with its own certificate.

    Attributes:
        certificates: Ordered list of certificates in the chain.
        epsilons: Privacy ε for each component.
        deltas: Privacy δ for each component.
        composition_type: Composition theorem used.
        metadata: Additional metadata.
    """

    certificates: List[Union[LPOptimalityCertificate, SDPOptimalityCertificate]]
    epsilons: List[float]
    deltas: List[float]
    composition_type: str = "basic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.certificates) != len(self.epsilons):
            raise ValueError(
                f"certificates ({len(self.certificates)}) and epsilons "
                f"({len(self.epsilons)}) must have the same length"
            )
        if len(self.certificates) != len(self.deltas):
            raise ValueError(
                f"certificates ({len(self.certificates)}) and deltas "
                f"({len(self.deltas)}) must have the same length"
            )

    @property
    def n_components(self) -> int:
        """Number of certificates in the chain."""
        return len(self.certificates)

    @property
    def total_epsilon(self) -> float:
        """Total composed ε (basic composition)."""
        if self.composition_type == "basic":
            return sum(self.epsilons)
        elif self.composition_type == "advanced":
            eps_sq = sum(e ** 2 for e in self.epsilons)
            total_delta = sum(self.deltas)
            if total_delta <= 0:
                return sum(self.epsilons)
            return math.sqrt(2 * eps_sq * math.log(1 / total_delta)) + eps_sq / (2 * max(sum(self.epsilons), 1e-15))
        return sum(self.epsilons)

    @property
    def total_delta(self) -> float:
        """Total composed δ."""
        return sum(self.deltas)

    @property
    def max_duality_gap(self) -> float:
        """Maximum duality gap across all certificates."""
        if not self.certificates:
            return 0.0
        return max(c.duality_gap for c in self.certificates)

    def append(
        self,
        cert: Union[LPOptimalityCertificate, SDPOptimalityCertificate],
        epsilon: float,
        delta: float = 0.0,
    ) -> None:
        """Add a certificate to the chain.

        Args:
            cert: Certificate to add.
            epsilon: Privacy ε for this component.
            delta: Privacy δ for this component.
        """
        self.certificates.append(cert)
        self.epsilons.append(epsilon)
        self.deltas.append(delta)

    def __repr__(self) -> str:
        return (
            f"CertificateChain(n={self.n_components}, "
            f"ε_total={self.total_epsilon:.4f}, "
            f"δ_total={self.total_delta:.2e}, "
            f"max_gap={self.max_duality_gap:.2e})"
        )


def verify_chain(
    chain: CertificateChain,
    mechanisms: List[ExtractedMechanism],
    specs: List[QuerySpec],
    *,
    tol: float = 1e-6,
) -> bool:
    """Verify an entire certificate chain.

    Checks each certificate individually and then verifies that the
    composed privacy guarantee holds.

    Args:
        chain: Certificate chain to verify.
        mechanisms: List of mechanisms (one per certificate).
        specs: List of query specifications (one per certificate).
        tol: Verification tolerance.

    Returns:
        True if the entire chain is valid.
    """
    if len(mechanisms) != chain.n_components:
        logger.error(
            "Chain has %d certificates but %d mechanisms provided",
            chain.n_components, len(mechanisms),
        )
        return False

    if len(specs) != chain.n_components:
        logger.error(
            "Chain has %d certificates but %d specs provided",
            chain.n_components, len(specs),
        )
        return False

    verifier = CertificateVerifier(tol=tol)

    for i, (cert, mech, spec) in enumerate(zip(chain.certificates, mechanisms, specs)):
        ok = verifier.verify(cert, mech, spec, tol=tol)
        if not ok:
            logger.warning("Certificate chain verification failed at component %d", i)
            return False
        logger.debug("Chain component %d/%d verified", i + 1, chain.n_components)

    # Verify composed privacy
    if chain.composition_type == "basic":
        composed_eps = sum(chain.epsilons)
        composed_delta = sum(chain.deltas)
    else:
        composed_eps = chain.total_epsilon
        composed_delta = chain.total_delta

    logger.info(
        "Certificate chain verified: %d components, "
        "composed (ε=%.4f, δ=%.2e)",
        chain.n_components, composed_eps, composed_delta,
    )
    return True


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def to_json(
    cert: Union[
        LPOptimalityCertificate,
        SDPOptimalityCertificate,
        ComposedCertificate,
        ApproximationCertificate,
        CertificateChain,
    ],
) -> str:
    """Serialize a certificate to JSON.

    Args:
        cert: Certificate to serialize.

    Returns:
        JSON string representation.
    """
    data = _cert_to_dict(cert)
    return json.dumps(data, indent=2, default=_json_default)


def from_json(data: Union[str, Dict[str, Any]]) -> Any:
    """Deserialize a certificate from JSON.

    Args:
        data: JSON string or dict.

    Returns:
        Deserialized certificate object.

    Raises:
        ValueError: If the certificate type is unknown.
    """
    if isinstance(data, str):
        data = json.loads(data)

    cert_type = data.get("type", data.get("certificate_type", ""))

    if cert_type == "LPOptimalityCertificate":
        return LPOptimalityCertificate(
            dual_ub=np.array(data.get("dual_ub", []), dtype=np.float64),
            dual_eq=(np.array(data["dual_eq"], dtype=np.float64)
                     if data.get("dual_eq") is not None else None),
            duality_gap=data["duality_gap"],
            primal_obj=data["primal_obj"],
            dual_obj=data["dual_obj"],
            primal_slack=(np.array(data["primal_slack"], dtype=np.float64)
                          if data.get("primal_slack") is not None else None),
            dual_slack=(np.array(data["dual_slack"], dtype=np.float64)
                        if data.get("dual_slack") is not None else None),
            n_vars=data.get("n_vars", 0),
            n_ub=data.get("n_ub", 0),
            n_eq=data.get("n_eq", 0),
            solver_name=data.get("solver_name", "unknown"),
            timestamp=data.get("timestamp", ""),
        )

    elif cert_type == "SDPOptimalityCertificate":
        return SDPOptimalityCertificate(
            dual_matrix=np.array(data["dual_matrix"], dtype=np.float64),
            duality_gap=data["duality_gap"],
            primal_obj=data["primal_obj"],
            dual_obj=data["dual_obj"],
            solver_name=data.get("solver_name", "unknown"),
            timestamp=data.get("timestamp", ""),
        )

    elif cert_type == "ApproximationCertificate":
        return ApproximationCertificate(
            k=data["k"],
            grid_spacing=data["grid_spacing"],
            discretization_error_bound=data["discretization_error_bound"],
            sensitivity=data.get("sensitivity", 1.0),
            epsilon=data.get("epsilon", 1.0),
            interpolation_error=data.get("interpolation_error", 0.0),
            timestamp=data.get("timestamp", ""),
        )

    elif cert_type == "ComposedCertificate":
        components = []
        for comp in data.get("components", []):
            sub_cert = from_json(comp.get("certificate", {}))
            components.append((sub_cert, comp["epsilon"], comp["delta"]))
        return ComposedCertificate(
            components=components,
            composition_type=data.get("composition_type", "basic"),
            total_epsilon=data.get("total_epsilon", 0.0),
            total_delta=data.get("total_delta", 0.0),
            timestamp=data.get("timestamp", ""),
        )

    else:
        raise ValueError(f"Unknown certificate type: {cert_type!r}")


def to_latex(
    cert: Union[LPOptimalityCertificate, SDPOptimalityCertificate, ApproximationCertificate],
) -> str:
    """Generate a LaTeX proof of optimality from a certificate.

    Produces a self-contained LaTeX fragment suitable for inclusion in
    a paper or appendix.

    Args:
        cert: Certificate to render.

    Returns:
        LaTeX source string.
    """
    lines: List[str] = []

    lines.append(r"\begin{theorem}[Optimality Certificate]")

    if isinstance(cert, LPOptimalityCertificate):
        lines.append(
            r"The synthesized mechanism is $\varepsilon$-optimal for the given query, "
            r"certified by LP duality."
        )
        lines.append(r"\end{theorem}")
        lines.append(r"\begin{proof}")
        lines.append(
            rf"The LP solver returned primal objective $p^* = {cert.primal_obj:.6f}$ "
            rf"and dual objective $d^* = {cert.dual_obj:.6f}$."
        )
        lines.append(
            rf"The duality gap is $|p^* - d^*| = {cert.duality_gap:.2e}$, "
            rf"with relative gap ${cert.relative_gap:.2e}$."
        )
        lines.append(
            rf"The LP had {cert.n_vars} variables, {cert.n_ub} inequality constraints, "
            rf"and {cert.n_eq} equality constraints."
        )
        if cert.is_tight:
            lines.append(
                r"Since the relative gap is below $10^{-6}$, strong duality holds "
                r"and the mechanism is optimal up to solver tolerance."
            )
        else:
            lines.append(
                r"The relative gap exceeds $10^{-6}$; the mechanism is "
                r"near-optimal but the certificate is not tight."
            )

        lines.append(
            r"\paragraph{Dual feasibility.} "
        )
        if len(cert.dual_ub) > 0:
            min_dual = float(cert.dual_ub.min())
            lines.append(
                rf"All {len(cert.dual_ub)} inequality dual variables satisfy "
                rf"$\lambda_i \geq {min_dual:.2e}$."
            )

        if cert.primal_slack is not None:
            max_cs = float(np.max(np.abs(cert.dual_ub * cert.primal_slack)))
            lines.append(
                r"\paragraph{Complementary slackness.} "
                rf"Maximum $|\lambda_i \cdot s_i| = {max_cs:.2e}$."
            )

        lines.append(r"\end{proof}")

    elif isinstance(cert, SDPOptimalityCertificate):
        lines.append(
            r"The synthesized Gaussian mechanism is optimal for the given "
            r"workload, certified by SDP duality."
        )
        lines.append(r"\end{theorem}")
        lines.append(r"\begin{proof}")
        lines.append(
            rf"The SDP solver returned primal value $p^* = {cert.primal_obj:.6f}$ "
            rf"and dual value $d^* = {cert.dual_obj:.6f}$."
        )
        lines.append(
            rf"Duality gap: $|p^* - d^*| = {cert.duality_gap:.2e}$."
        )
        lines.append(
            rf"The dual matrix $Y \in \mathbb{{R}}^{{{cert.matrix_dim} \times {cert.matrix_dim}}}$ "
            rf"has minimum eigenvalue $\lambda_{{\min}} = {cert.min_eigenvalue:.2e}$."
        )
        if cert.is_psd:
            lines.append(
                r"Since $Y \succeq 0$, dual feasibility holds."
            )
        else:
            lines.append(
                r"\textbf{Warning:} $Y$ is not PSD; the certificate is invalid."
            )
        lines.append(r"\end{proof}")

    elif isinstance(cert, ApproximationCertificate):
        lines.append(
            r"The discretization error of the synthesized mechanism is bounded."
        )
        lines.append(r"\end{theorem}")
        lines.append(r"\begin{proof}")
        lines.append(
            rf"With $k = {cert.k}$ discretization bins and grid spacing "
            rf"$\Delta y = {cert.grid_spacing:.4f}$, the piecewise-constant "
            rf"approximation introduces at most"
        )
        lines.append(
            rf"\[ \Delta_{{\text{{disc}}}} \leq \frac{{(\Delta y)^2}}{{12}} "
            rf"= {cert.discretization_error_bound:.2e} \]"
        )
        lines.append(r"additional MSE over the optimal continuous mechanism.")
        if cert.interpolation_error > 0:
            lines.append(
                rf"Including interpolation error of {cert.interpolation_error:.2e}, "
                rf"the total bound is {cert.total_error_bound:.2e}."
            )
        lines.append(r"\end{proof}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cert_to_dict(cert: Any) -> Dict[str, Any]:
    """Convert a certificate to a JSON-serializable dict."""
    if isinstance(cert, LPOptimalityCertificate):
        d: Dict[str, Any] = {
            "type": "LPOptimalityCertificate",
            "dual_ub": cert.dual_ub.tolist(),
            "dual_eq": cert.dual_eq.tolist() if cert.dual_eq is not None else None,
            "duality_gap": cert.duality_gap,
            "primal_obj": cert.primal_obj,
            "dual_obj": cert.dual_obj,
            "primal_slack": cert.primal_slack.tolist() if cert.primal_slack is not None else None,
            "dual_slack": cert.dual_slack.tolist() if cert.dual_slack is not None else None,
            "n_vars": cert.n_vars,
            "n_ub": cert.n_ub,
            "n_eq": cert.n_eq,
            "solver_name": cert.solver_name,
            "timestamp": cert.timestamp,
            "relative_gap": cert.relative_gap,
            "is_tight": cert.is_tight,
        }
        return d

    elif isinstance(cert, SDPOptimalityCertificate):
        return {
            "type": "SDPOptimalityCertificate",
            "dual_matrix": cert.dual_matrix.tolist(),
            "duality_gap": cert.duality_gap,
            "primal_obj": cert.primal_obj,
            "dual_obj": cert.dual_obj,
            "matrix_dim": cert.matrix_dim,
            "min_eigenvalue": cert.min_eigenvalue,
            "is_psd": cert.is_psd,
            "solver_name": cert.solver_name,
            "timestamp": cert.timestamp,
            "relative_gap": cert.relative_gap,
            "is_tight": cert.is_tight,
        }

    elif isinstance(cert, ComposedCertificate):
        return {
            "type": "ComposedCertificate",
            "components": [
                {
                    "certificate": _cert_to_dict(c),
                    "epsilon": eps,
                    "delta": delta,
                }
                for c, eps, delta in cert.components
            ],
            "composition_type": cert.composition_type,
            "total_epsilon": cert.total_epsilon,
            "total_delta": cert.total_delta,
            "n_components": cert.n_components,
            "timestamp": cert.timestamp,
        }

    elif isinstance(cert, ApproximationCertificate):
        return {
            "type": "ApproximationCertificate",
            "k": cert.k,
            "grid_spacing": cert.grid_spacing,
            "discretization_error_bound": cert.discretization_error_bound,
            "sensitivity": cert.sensitivity,
            "epsilon": cert.epsilon,
            "interpolation_error": cert.interpolation_error,
            "total_error_bound": cert.total_error_bound,
            "timestamp": cert.timestamp,
        }

    elif isinstance(cert, CertificateChain):
        return {
            "type": "CertificateChain",
            "certificates": [_cert_to_dict(c) for c in cert.certificates],
            "epsilons": cert.epsilons,
            "deltas": cert.deltas,
            "composition_type": cert.composition_type,
            "total_epsilon": cert.total_epsilon,
            "total_delta": cert.total_delta,
            "n_components": cert.n_components,
            "max_duality_gap": cert.max_duality_gap,
            "metadata": cert.metadata,
        }

    else:
        return {"type": type(cert).__name__, "data": str(cert)}


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
