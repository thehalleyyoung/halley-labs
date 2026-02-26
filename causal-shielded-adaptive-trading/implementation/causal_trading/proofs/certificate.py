"""
Formal certificate generation for Causal-Shielded Adaptive Trading.

Provides a unified Certificate class that aggregates guarantees from
the causal identification, shield soundness, and composition modules
into a single JSON-serializable, verifiable, and human-readable
certificate format.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Assumption:
    """A single assumption in a certificate."""
    id: str
    description: str
    category: str = "general"
    verified: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "verified": self.verified,
        }


@dataclass
class Guarantee:
    """A single guarantee (bound) provided by the certificate."""
    id: str
    description: str
    bound_type: str
    value: float
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "bound_type": self.bound_type,
            "value": self.value,
            "confidence": self.confidence,
            "parameters": self.parameters,
        }


@dataclass
class CertificateParameters:
    """All numeric parameters entering the certificate."""
    kl_divergence: float = 0.0
    n_samples: int = 0
    delta: float = 0.05
    epsilon: float = 0.05
    eps1: float = 0.0
    eps2: float = 0.0
    complexity_term: float = 0.0
    empirical_risk: float = 0.0
    composed_bound: float = 0.0
    n_environments: int = 0
    n_invariant_features: int = 0
    n_variant_features: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "kl_divergence": self.kl_divergence,
            "n_samples": self.n_samples,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "eps1": self.eps1,
            "eps2": self.eps2,
            "complexity_term": self.complexity_term,
            "empirical_risk": self.empirical_risk,
            "composed_bound": self.composed_bound,
            "n_environments": self.n_environments,
            "n_invariant_features": self.n_invariant_features,
            "n_variant_features": self.n_variant_features,
        }
        d.update(self.extra)
        return d


class Certificate:
    """
    Unified formal certificate for the Causal-Shielded Adaptive
    Trading system.

    Aggregates assumptions, guarantees, parameters, and bounds from
    the causal identification, shield soundness, and composition
    modules. Supports JSON serialization, LaTeX rendering,
    verification, and aggregation.

    Parameters
    ----------
    name : str
        Human-readable name for the certificate.
    version : str
        Version string.
    """

    def __init__(
        self,
        name: str = "CausalShieldedTradingCertificate",
        version: str = "1.0",
    ) -> None:
        self.name = name
        self.version = version

        self.assumptions: List[Assumption] = []
        self.guarantees: List[Guarantee] = []
        self.parameters = CertificateParameters()
        self.bounds: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
        self.timestamp: Optional[str] = None
        self._verified: Optional[bool] = None
        self._verification_log: List[str] = []

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def add_assumption(
        self,
        id: str,
        description: str,
        category: str = "general",
        verified: Optional[bool] = None,
    ) -> None:
        """Add an assumption to the certificate."""
        self.assumptions.append(Assumption(
            id=id,
            description=description,
            category=category,
            verified=verified,
        ))

    def add_guarantee(
        self,
        id: str,
        description: str,
        bound_type: str,
        value: float,
        confidence: float = 0.95,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a guarantee (bound) to the certificate."""
        self.guarantees.append(Guarantee(
            id=id,
            description=description,
            bound_type=bound_type,
            value=value,
            confidence=confidence,
            parameters=parameters or {},
        ))
        self.bounds[id] = value

    def set_parameters(self, params: CertificateParameters) -> None:
        """Set the certificate parameters."""
        self.parameters = params

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the certificate."""
        self.metadata[key] = value

    def generate(
        self,
        causal_result: Optional[Dict[str, Any]] = None,
        shield_result: Optional[Dict[str, Any]] = None,
        composition_result: Optional[Dict[str, Any]] = None,
    ) -> Certificate:
        """
        Auto-populate the certificate from module results.

        Parameters
        ----------
        causal_result : dict, optional
            From CausalResult.to_dict().
        shield_result : dict, optional
            From ShieldResult.to_dict().
        composition_result : dict, optional
            From ComposedCertificate.to_dict().

        Returns
        -------
        self : Certificate
        """
        import datetime
        self.timestamp = datetime.datetime.utcnow().isoformat()

        if causal_result:
            self._populate_from_causal(causal_result)
        if shield_result:
            self._populate_from_shield(shield_result)
        if composition_result:
            self._populate_from_composition(composition_result)

        # Add standard assumptions
        self.add_assumption(
            "iid",
            "Data samples are drawn i.i.d. from the environment distribution",
            category="statistical",
            verified=None,
        )
        self.add_assumption(
            "prior_fixed",
            "Prior distribution is fixed before observing data",
            category="pac-bayes",
            verified=None,
        )
        self.add_assumption(
            "bounded_loss",
            "Loss function is bounded in [0, 1]",
            category="pac-bayes",
            verified=None,
        )

        return self

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #

    def verify(self) -> bool:
        """
        Verify internal consistency of the certificate.

        Checks:
        1. All guarantees have valid bound values.
        2. eps1 + eps2 <= 1 (otherwise vacuous).
        3. Complexity term is consistent with KL and n.
        4. Composed bound matches eps1 + eps2.

        Returns
        -------
        valid : bool
        """
        self._verification_log = []
        valid = True

        # Check guarantee values
        for g in self.guarantees:
            if not (0.0 <= g.value <= 1.0):
                self._verification_log.append(
                    f"Guarantee '{g.id}' value {g.value:.6f} out of [0, 1]"
                )
                valid = False

        # Check eps budget
        eps_total = self.parameters.eps1 + self.parameters.eps2
        if eps_total > 1.0:
            self._verification_log.append(
                f"eps1 + eps2 = {eps_total:.6f} > 1.0 (vacuous)"
            )
            valid = False

        # Check complexity term consistency
        if self.parameters.n_samples > 0 and self.parameters.kl_divergence >= 0:
            n = self.parameters.n_samples
            kl = self.parameters.kl_divergence
            eps = self.parameters.epsilon
            expected_complexity = math.sqrt(
                (kl + math.log(2.0 * math.sqrt(n) / max(eps, 1e-12)))
                / (2.0 * n)
            )
            actual = self.parameters.complexity_term
            if actual > 0 and abs(actual - expected_complexity) / max(actual, 1e-12) > 0.1:
                self._verification_log.append(
                    f"Complexity term mismatch: stated={actual:.6f}, "
                    f"computed={expected_complexity:.6f}"
                )
                valid = False

        # Check composed bound
        expected_composed = max(0.0, 1.0 - eps_total)
        actual_composed = self.parameters.composed_bound
        if actual_composed > 0 and abs(actual_composed - expected_composed) > 0.01:
            self._verification_log.append(
                f"Composed bound mismatch: stated={actual_composed:.6f}, "
                f"computed={expected_composed:.6f}"
            )
            valid = False

        self._verified = valid
        if valid:
            self._verification_log.append("Certificate verified successfully")
        else:
            logger.warning(
                "Certificate verification failed: %s",
                "; ".join(self._verification_log),
            )
        return valid

    def get_verification_log(self) -> List[str]:
        """Return the verification log from the last verify() call."""
        return list(self._verification_log)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the certificate to a dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "timestamp": self.timestamp,
            "verified": self._verified,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "guarantees": [g.to_dict() for g in self.guarantees],
            "parameters": self.parameters.to_dict(),
            "bounds": self.bounds,
            "metadata": self.metadata,
            "verification_log": self._verification_log,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the certificate to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    @classmethod
    def from_json(cls, json_str: str) -> Certificate:
        """Deserialize a certificate from JSON."""
        data = json.loads(json_str)
        cert = cls(name=data.get("name", ""), version=data.get("version", "1.0"))
        cert.timestamp = data.get("timestamp")
        cert._verified = data.get("verified")
        cert.bounds = data.get("bounds", {})
        cert.metadata = data.get("metadata", {})
        cert._verification_log = data.get("verification_log", [])

        for a in data.get("assumptions", []):
            cert.assumptions.append(Assumption(**a))
        for g in data.get("guarantees", []):
            cert.guarantees.append(Guarantee(**g))

        params_data = data.get("parameters", {})
        extra = {}
        known_fields = set(CertificateParameters.__dataclass_fields__.keys())
        for k, v in params_data.items():
            if k not in known_fields:
                extra[k] = v
        filtered = {k: v for k, v in params_data.items() if k in known_fields and k != "extra"}
        cert.parameters = CertificateParameters(**filtered, extra=extra)

        return cert

    def to_latex(self) -> str:
        """
        Render the certificate as a LaTeX fragment suitable for
        inclusion in a paper appendix.
        """
        lines = [
            r"\begin{tcolorbox}[title=Safety Certificate]",
            r"\textbf{" + _latex_escape(self.name) + r"}\\",
            r"Version: " + _latex_escape(self.version) + r"\\",
        ]
        if self.timestamp:
            lines.append(r"Timestamp: " + _latex_escape(self.timestamp) + r"\\")
        lines.append(r"")

        # Assumptions
        lines.append(r"\textbf{Assumptions:}")
        lines.append(r"\begin{enumerate}")
        for a in self.assumptions:
            status = ""
            if a.verified is True:
                status = r" \textcolor{green}{[\checkmark]}"
            elif a.verified is False:
                status = r" \textcolor{red}{[$\times$]}"
            lines.append(
                r"  \item " + _latex_escape(a.description) + status
            )
        lines.append(r"\end{enumerate}")
        lines.append(r"")

        # Guarantees
        lines.append(r"\textbf{Guarantees:}")
        lines.append(r"\begin{itemize}")
        for g in self.guarantees:
            lines.append(
                r"  \item \textbf{" + _latex_escape(g.id) + r"}: "
                + _latex_escape(g.description)
                + r" ($\leq " + f"{g.value:.6f}" + r"$)"
            )
        lines.append(r"\end{itemize}")
        lines.append(r"")

        # Parameters
        lines.append(r"\textbf{Key Parameters:}")
        lines.append(r"\begin{align*}")
        p = self.parameters
        lines.append(
            r"  \mathrm{KL}(\rho \| \pi) &= "
            + f"{p.kl_divergence:.6f}" + r" \\"
        )
        lines.append(r"  n &= " + str(p.n_samples) + r" \\")
        lines.append(r"  \delta &= " + f"{p.delta:.4f}" + r" \\")
        lines.append(
            r"  \varepsilon_1 &= " + f"{p.eps1:.6f}"
            + r",\quad \varepsilon_2 = " + f"{p.eps2:.6f}" + r" \\"
        )
        lines.append(
            r"  P(\text{correct}) &\geq "
            + f"{p.composed_bound:.6f}"
        )
        lines.append(r"\end{align*}")
        lines.append(r"")

        # Verification
        if self._verified is not None:
            status_str = (
                r"\textcolor{green}{VERIFIED}"
                if self._verified
                else r"\textcolor{red}{FAILED}"
            )
            lines.append(r"\textbf{Verification Status}: " + status_str)

        lines.append(r"\end{tcolorbox}")
        return "\n".join(lines)

    def render(self) -> str:
        """
        Render the certificate as a human-readable text block.

        Returns
        -------
        text : str
        """
        sep = "=" * 60
        lines = [
            sep,
            f"  SAFETY CERTIFICATE: {self.name}",
            f"  Version: {self.version}",
        ]
        if self.timestamp:
            lines.append(f"  Timestamp: {self.timestamp}")
        lines.append(sep)

        # Verification
        if self._verified is not None:
            status = "VERIFIED ✓" if self._verified else "FAILED ✗"
            lines.append(f"\n  Status: {status}\n")

        # Assumptions
        lines.append("  ASSUMPTIONS:")
        for i, a in enumerate(self.assumptions, 1):
            check = ""
            if a.verified is True:
                check = " [✓]"
            elif a.verified is False:
                check = " [✗]"
            lines.append(f"    {i}. [{a.category}] {a.description}{check}")

        # Guarantees
        lines.append("\n  GUARANTEES:")
        for g in self.guarantees:
            lines.append(
                f"    • {g.id}: {g.description}\n"
                f"      Bound: {g.value:.6f} (confidence: {g.confidence:.4f})"
            )

        # Parameters
        lines.append("\n  PARAMETERS:")
        p = self.parameters
        lines.append(f"    KL(rho || pi)   = {p.kl_divergence:.6f}")
        lines.append(f"    n               = {p.n_samples}")
        lines.append(f"    delta           = {p.delta:.4f}")
        lines.append(f"    eps1 (causal)   = {p.eps1:.6f}")
        lines.append(f"    eps2 (shield)   = {p.eps2:.6f}")
        lines.append(f"    complexity      = {p.complexity_term:.6f}")
        lines.append(f"    empirical risk  = {p.empirical_risk:.6f}")
        lines.append(f"    composed bound  = {p.composed_bound:.6f}")

        if p.extra:
            lines.append("    Extra:")
            for k, v in p.extra.items():
                lines.append(f"      {k} = {v}")

        # Verification log
        if self._verification_log:
            lines.append("\n  VERIFICATION LOG:")
            for msg in self._verification_log:
                lines.append(f"    - {msg}")

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    @staticmethod
    def aggregate(certificates: List[Certificate]) -> Certificate:
        """
        Aggregate multiple certificates into a single combined certificate.

        Takes the worst-case (loosest) bound across certificates for
        each guarantee, and the union of all assumptions.

        Parameters
        ----------
        certificates : list of Certificate
            Certificates to aggregate.

        Returns
        -------
        combined : Certificate
        """
        if not certificates:
            return Certificate(name="EmptyAggregate")

        import datetime
        combined = Certificate(
            name="AggregatedCertificate",
            version="1.0",
        )
        combined.timestamp = datetime.datetime.utcnow().isoformat()

        # Union of assumptions (deduplicated by id)
        seen_assumptions: set = set()
        for cert in certificates:
            for a in cert.assumptions:
                if a.id not in seen_assumptions:
                    combined.assumptions.append(a)
                    seen_assumptions.add(a.id)

        # Worst-case guarantees (grouped by id)
        guarantee_groups: Dict[str, List[Guarantee]] = {}
        for cert in certificates:
            for g in cert.guarantees:
                if g.id not in guarantee_groups:
                    guarantee_groups[g.id] = []
                guarantee_groups[g.id].append(g)

        for gid, group in guarantee_groups.items():
            worst = max(group, key=lambda g: g.value)
            combined.guarantees.append(Guarantee(
                id=gid,
                description=worst.description + " (worst-case aggregate)",
                bound_type=worst.bound_type,
                value=worst.value,
                confidence=min(g.confidence for g in group),
                parameters=worst.parameters,
            ))
            combined.bounds[gid] = worst.value

        # Aggregate parameters: worst case
        combined.parameters = CertificateParameters(
            kl_divergence=max(c.parameters.kl_divergence for c in certificates),
            n_samples=min(c.parameters.n_samples for c in certificates),
            delta=max(c.parameters.delta for c in certificates),
            epsilon=max(c.parameters.epsilon for c in certificates),
            eps1=max(c.parameters.eps1 for c in certificates),
            eps2=max(c.parameters.eps2 for c in certificates),
            complexity_term=max(c.parameters.complexity_term for c in certificates),
            empirical_risk=max(c.parameters.empirical_risk for c in certificates),
            composed_bound=min(c.parameters.composed_bound for c in certificates),
            n_environments=min(c.parameters.n_environments for c in certificates),
            n_invariant_features=min(
                c.parameters.n_invariant_features for c in certificates
            ),
            n_variant_features=max(
                c.parameters.n_variant_features for c in certificates
            ),
        )

        combined.add_metadata("n_source_certificates", len(certificates))
        combined.add_metadata(
            "source_names",
            [c.name for c in certificates],
        )

        return combined

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _populate_from_causal(self, result: Dict[str, Any]) -> None:
        """Populate certificate from causal identification results."""
        eps1 = result.get("eps1", 0.0)
        self.parameters.eps1 = eps1
        self.parameters.n_environments = result.get("n_environments", 0)
        self.parameters.n_invariant_features = len(
            result.get("invariant_features", [])
        )
        self.parameters.n_variant_features = len(
            result.get("variant_features", [])
        )

        self.add_guarantee(
            id="causal_identification",
            description=(
                f"Causal invariant features correctly identified with "
                f"probability >= {1 - eps1:.6f}"
            ),
            bound_type="union_bound",
            value=eps1,
            confidence=1 - eps1,
            parameters={
                "n_environments": result.get("n_environments", 0),
                "n_invariant": self.parameters.n_invariant_features,
            },
        )

    def _populate_from_shield(self, result: Dict[str, Any]) -> None:
        """Populate certificate from shield soundness results."""
        eps2 = result.get("eps2", 0.0)
        self.parameters.eps2 = eps2
        self.parameters.kl_divergence = result.get("kl_divergence", 0.0)
        self.parameters.complexity_term = result.get("complexity_term", 0.0)
        self.parameters.delta = result.get("delta", 0.05)
        self.parameters.n_samples = result.get("n_samples", 0)

        self.add_guarantee(
            id="shield_soundness",
            description=(
                f"Safety property holds under shielded policy with "
                f"probability >= {1 - eps2:.6f}"
            ),
            bound_type="pac_bayes",
            value=eps2,
            confidence=1 - eps2,
            parameters={
                "kl_divergence": self.parameters.kl_divergence,
                "n_samples": self.parameters.n_samples,
            },
        )

    def _populate_from_composition(self, result: Dict[str, Any]) -> None:
        """Populate certificate from composition checker results."""
        composed = result.get("composed_bound", 0.0)
        self.parameters.composed_bound = composed

        self.add_guarantee(
            id="composed_safety",
            description=(
                f"Full system correct with probability >= {composed:.6f}"
            ),
            bound_type="union_bound_composition",
            value=1.0 - composed,
            confidence=composed,
            parameters={
                "eps1": result.get("eps1", 0.0),
                "eps2": result.get("eps2", 0.0),
            },
        )

        if result.get("warnings"):
            self.add_metadata("composition_warnings", result["warnings"])


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
