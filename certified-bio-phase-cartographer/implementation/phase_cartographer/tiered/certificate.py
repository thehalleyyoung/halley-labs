"""
Unified certificate format for all verification tiers.

A CertifiedCell is the atomic unit of the certified phase atlas.
Every cell contains:
  - The parameter box it covers
  - The regime classification (derived via formal inference rules)
  - Certification evidence from one or more tiers
  - Metadata for provenance tracking

The regime label is NOT arbitrary — it is derived from certified mathematical
facts via explicit inference rules (see RegimeInferenceRules).
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any


class VerificationTier(Enum):
    """Verification tier with increasing assurance."""
    TIER1_IA = "tier1_interval_arithmetic"
    TIER2_DREAL = "tier2_dreal_delta_complete"
    TIER3_Z3 = "tier3_z3_exact"


class RegimeType(Enum):
    """Formally defined regime types with inference rules."""
    MONOSTABLE = "monostable"
    BISTABLE = "bistable"
    MULTISTABLE = "multistable"
    OSCILLATORY = "oscillatory"
    EXCITABLE = "excitable"
    INCONCLUSIVE = "inconclusive"


class StabilityType(Enum):
    STABLE_NODE = "stable_node"
    STABLE_FOCUS = "stable_focus"
    STABLE_SPIRAL = "stable_spiral"
    UNSTABLE_NODE = "unstable_node"
    UNSTABLE_FOCUS = "unstable_focus"
    SADDLE = "saddle"
    CENTER = "center"
    DEGENERATE = "degenerate"
    UNKNOWN = "unknown"


@dataclass
class EquilibriumCertificate:
    """Certificate for a single equilibrium within a parameter box."""
    state_enclosure: List[Tuple[float, float]]
    stability: StabilityType
    eigenvalue_real_parts: List[Tuple[float, float]]
    krawczyk_contraction: float
    krawczyk_iterations: int
    delta_bound: Optional[Dict[str, float]] = None

    def is_stable(self) -> bool:
        return self.stability in (
            StabilityType.STABLE_NODE,
            StabilityType.STABLE_FOCUS,
            StabilityType.STABLE_SPIRAL,
        )

    def to_dict(self) -> dict:
        return {
            "state_enclosure": list(self.state_enclosure),
            "stability": self.stability.value,
            "eigenvalue_real_parts": list(self.eigenvalue_real_parts),
            "krawczyk_contraction": self.krawczyk_contraction,
            "krawczyk_iterations": self.krawczyk_iterations,
            "delta_bound": self.delta_bound,
        }


@dataclass
class CertifiedCell:
    """
    Unified certificate for a single parameter-space cell.

    This is the atomic unit of the certified phase atlas, used uniformly
    across all three verification tiers.
    """
    parameter_box: List[Tuple[float, float]]
    model_name: str
    n_states: int
    n_params: int
    equilibria: List[EquilibriumCertificate]
    regime: RegimeType
    tier: VerificationTier
    depth: int = 0  # refinement depth
    certification_time_s: float = 0.0
    minicheck_passed: Optional[bool] = None

    def volume(self) -> float:
        """Volume of the parameter box (only varying dimensions)."""
        v = 1.0
        for lo, hi in self.parameter_box:
            w = hi - lo
            if w > 0:
                v *= w
        return v

    def n_stable_equilibria(self) -> int:
        return sum(1 for eq in self.equilibria if eq.is_stable())

    def fingerprint(self) -> str:
        """SHA-256 fingerprint for certificate integrity."""
        content = json.dumps({
            "model": {
                "name": self.model_name,
                "n_states": self.n_states,
                "n_params": self.n_params,
            },
            "parameter_box": list(self.parameter_box),
            "equilibria": [eq.to_dict() for eq in self.equilibria],
            "regime_label": self.regime.value,
            "tier": self.tier.value,
            "depth": self.depth,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "model": {
                "name": self.model_name,
                "n_states": self.n_states,
                "n_params": self.n_params,
            },
            "parameter_box": list(self.parameter_box),
            "equilibria": [eq.to_dict() for eq in self.equilibria],
            "regime_label": self.regime.value,
            "tier": self.tier.value,
            "depth": self.depth,
            "certification_time_s": self.certification_time_s,
            "minicheck_passed": self.minicheck_passed,
            "fingerprint": self.fingerprint() if self.minicheck_passed else None,
        }

    def to_minicheck_format(self) -> dict:
        """Convert to the format expected by minicheck.verify_certificate."""
        return {
            "model": {
                "name": self.model_name,
                "n_states": self.n_states,
                "n_params": self.n_params,
            },
            "parameter_box": list(self.parameter_box),
            "equilibria": [eq.to_dict() for eq in self.equilibria],
            "regime_label": self.regime.value,
            "coverage_fraction": 1.0,
        }


class RegimeInferenceRules:
    """
    Formal inference rules mapping certified mathematical facts to regime labels.

    These rules replace ad-hoc procedural classification with logical derivation:

    Rule MONO: If exactly 1 stable equilibrium is certified in cell,
               and all other equilibria (if any) are unstable → MONOSTABLE.

    Rule BI:   If ≥ 2 stable equilibria are certified in cell → BISTABLE.

    Rule MULTI: If ≥ 3 stable equilibria are certified → MULTISTABLE.

    Rule OSC:  If a certified periodic orbit exists AND
               all equilibria are unstable → OSCILLATORY.

    Rule EXC:  If exactly 1 stable equilibrium AND
               a nearby saddle exists → EXCITABLE.

    Rule INC:  Otherwise → INCONCLUSIVE.
    """

    @staticmethod
    def infer(equilibria: List[EquilibriumCertificate],
              has_periodic_orbit: bool = False) -> RegimeType:
        """Apply inference rules to derive regime type."""
        n_stable = sum(1 for eq in equilibria if eq.is_stable())
        n_saddle = sum(1 for eq in equilibria
                       if eq.stability == StabilityType.SADDLE)
        n_total = len(equilibria)

        if n_stable >= 3:
            return RegimeType.MULTISTABLE
        if n_stable >= 2:
            return RegimeType.BISTABLE
        if has_periodic_orbit and n_stable == 0:
            return RegimeType.OSCILLATORY
        if n_stable == 1 and n_saddle >= 1:
            return RegimeType.EXCITABLE
        if n_stable == 1:
            return RegimeType.MONOSTABLE
        if has_periodic_orbit:
            return RegimeType.OSCILLATORY
        return RegimeType.INCONCLUSIVE

    @staticmethod
    def validate(cell: CertifiedCell) -> Tuple[bool, str]:
        """
        Validate that the regime label is derivable from the certified facts.

        Returns (valid, reason).
        """
        inferred = RegimeInferenceRules.infer(cell.equilibria)
        if inferred == cell.regime:
            return True, "Regime label matches inference rules"
        # Allow INCONCLUSIVE to be compatible with any label
        if cell.regime == RegimeType.INCONCLUSIVE:
            return True, "INCONCLUSIVE is always valid"
        return False, (f"Inferred {inferred.value} but labeled {cell.regime.value}; "
                       f"{cell.n_stable_equilibria()} stable equilibria found")
