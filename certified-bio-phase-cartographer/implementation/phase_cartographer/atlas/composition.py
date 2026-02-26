"""
Formal certificate composition calculus for phase atlas construction.

Provides rules for composing CertifiedCells into atlas-level certificates
with provable soundness preservation.

Theorem B2 (Certificate Composition Soundness):
  If cells C_1, ..., C_k cover a region R with:
    (i)   each C_i individually certified (Krawczyk-verified, minicheck-passed),
    (ii)  cells are pairwise disjoint (non-overlapping parameter boxes),
    (iii) union(C_i) covers R up to measure-zero boundaries,
  then the atlas certificate for R is sound: every regime claim in R
  is backed by a cell-level certificate.

Certificate composition rules:
  MERGE:  Two adjacent cells with identical regime can be merged if the
          combined box is also certifiable.
  SPLIT:  A certified cell can be split into subcells; each subcell inherits
          the parent's certification if Krawczyk still verifies on the subbox.
  BOUNDARY: Adjacent cells with different regimes imply a bifurcation
          boundary exists between them.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json

from ..tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)


@dataclass
class RegimeDerivation:
    """
    Proof tree recording logical inference from certified facts to regime label.
    
    Each derivation records:
      - The certified mathematical facts (premises)
      - The inference rule applied
      - The derived regime label (conclusion)
    """
    premises: List[str]
    rule_name: str
    conclusion: str
    n_stable: int = 0
    n_unstable: int = 0
    n_saddle: int = 0
    has_periodic_orbit: bool = False
    
    def to_dict(self) -> dict:
        return {
            "premises": self.premises,
            "rule": self.rule_name,
            "conclusion": self.conclusion,
            "counts": {
                "stable": self.n_stable,
                "unstable": self.n_unstable,
                "saddle": self.n_saddle,
            },
            "has_periodic_orbit": self.has_periodic_orbit,
        }
    
    @staticmethod
    def derive(cell: 'CertifiedCell') -> 'RegimeDerivation':
        """Construct a regime derivation proof tree from a certified cell."""
        premises = []
        n_stable = 0
        n_unstable = 0
        n_saddle = 0
        
        for i, eq in enumerate(cell.equilibria):
            stability = eq.stability.value if hasattr(eq.stability, 'value') else str(eq.stability)
            premises.append(
                f"Equilibrium {i}: Krawczyk-verified in {_format_box(eq.state_enclosure)}, "
                f"stability={stability}, contraction={eq.krawczyk_contraction:.4f}"
            )
            if eq.is_stable():
                n_stable += 1
            elif eq.stability == StabilityType.SADDLE:
                n_saddle += 1
            else:
                n_unstable += 1
        
        # Determine which rule applies
        if n_stable >= 3:
            rule = "MULTI"
            conclusion = RegimeType.MULTISTABLE.value
        elif n_stable >= 2:
            rule = "BI"
            conclusion = RegimeType.BISTABLE.value
        elif n_stable == 1 and n_saddle >= 1:
            rule = "EXC"
            conclusion = RegimeType.EXCITABLE.value
        elif n_stable == 1:
            rule = "MONO"
            conclusion = RegimeType.MONOSTABLE.value
        else:
            rule = "INC"
            conclusion = RegimeType.INCONCLUSIVE.value
        
        return RegimeDerivation(
            premises=premises,
            rule_name=rule,
            conclusion=conclusion,
            n_stable=n_stable,
            n_unstable=n_unstable,
            n_saddle=n_saddle,
        )


@dataclass
class CertificateProofObject:
    """
    Complete proof object for a certified cell, wrapping all evidence.
    
    This is the unified certificate format across all verification tiers,
    containing:
      - Cell data (parameter box, equilibria, regime)
      - Regime derivation proof tree
      - Verification evidence per tier
      - Certificate fingerprint for integrity
    """
    cell: CertifiedCell
    derivation: RegimeDerivation
    tier_evidence: Dict[str, dict] = field(default_factory=dict)
    composition_parent: Optional[str] = None  # fingerprint of parent if split
    
    def to_dict(self) -> dict:
        return {
            "cell": self.cell.to_dict(),
            "derivation": self.derivation.to_dict(),
            "tier_evidence": self.tier_evidence,
            "composition_parent": self.composition_parent,
        }
    
    @staticmethod
    def from_cell(cell: CertifiedCell) -> 'CertificateProofObject':
        """Construct proof object from a certified cell."""
        derivation = RegimeDerivation.derive(cell)
        evidence = {
            "tier1_minicheck": {
                "passed": cell.minicheck_passed,
                "method": "independent_interval_arithmetic_recomputation",
            }
        }
        return CertificateProofObject(
            cell=cell,
            derivation=derivation,
            tier_evidence=evidence,
        )


@dataclass
class CompositionResult:
    """Result of a certificate composition operation."""
    valid: bool
    cells: List[CertifiedCell]
    coverage_fraction: float
    boundary_cells: List[Tuple[int, int]]  # pairs of adjacent cells with different regimes
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "n_cells": len(self.cells),
            "coverage_fraction": self.coverage_fraction,
            "n_boundaries": len(self.boundary_cells),
            "errors": self.errors,
        }


def boxes_adjacent(box1: List[Tuple[float, float]],
                   box2: List[Tuple[float, float]],
                   tol: float = 1e-10) -> bool:
    """Check if two parameter boxes share a face (are adjacent)."""
    if len(box1) != len(box2):
        return False
    
    n_shared = 0
    n_touching = 0
    for (lo1, hi1), (lo2, hi2) in zip(box1, box2):
        if abs(hi1 - lo2) < tol or abs(hi2 - lo1) < tol:
            n_touching += 1
        # Check overlap in this dimension
        if hi1 + tol >= lo2 and hi2 + tol >= lo1:
            n_shared += 1
    
    # Adjacent = touching in exactly 1 dimension, overlapping in all others
    return n_touching == 1 and n_shared == len(box1)


def boxes_disjoint(box1: List[Tuple[float, float]],
                   box2: List[Tuple[float, float]],
                   tol: float = 1e-10) -> bool:
    """Check if two parameter boxes are disjoint (no interior overlap)."""
    for (lo1, hi1), (lo2, hi2) in zip(box1, box2):
        if hi1 <= lo2 + tol or hi2 <= lo1 + tol:
            return True
    return False


def verify_atlas_composition(cells: List[CertifiedCell],
                             domain: List[Tuple[float, float]]) -> CompositionResult:
    """
    Verify that a collection of certified cells forms a valid atlas.
    
    Checks:
    1. All cells are individually certified (minicheck passed)
    2. Cells are pairwise disjoint
    3. Coverage fraction is computed correctly
    4. Boundary cells (adjacent cells with different regimes) are identified
    """
    errors = []
    
    # Check individual certification
    for i, cell in enumerate(cells):
        if not cell.minicheck_passed:
            errors.append(f"Cell {i}: minicheck not passed")
        
        # Verify regime derivation consistency
        valid, reason = RegimeInferenceRules.validate(cell)
        if not valid:
            errors.append(f"Cell {i}: {reason}")
    
    # Check pairwise disjointness
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if not boxes_disjoint(cells[i].parameter_box, cells[j].parameter_box):
                errors.append(f"Cells {i} and {j} overlap")
    
    # Compute coverage
    domain_vol = 1.0
    for lo, hi in domain:
        w = hi - lo
        if w > 0:
            domain_vol *= w
    
    certified_vol = sum(c.volume() for c in cells)
    coverage = min(1.0, certified_vol / domain_vol) if domain_vol > 0 else 0.0
    
    # Find boundary cells (adjacent with different regimes)
    boundaries = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if (cells[i].regime != cells[j].regime and
                boxes_adjacent(cells[i].parameter_box, cells[j].parameter_box)):
                boundaries.append((i, j))
    
    return CompositionResult(
        valid=len(errors) == 0,
        cells=cells,
        coverage_fraction=coverage,
        boundary_cells=boundaries,
        errors=errors,
    )


def _format_box(box) -> str:
    """Format a parameter box for display."""
    if isinstance(box, list):
        return " × ".join(f"[{lo:.4f}, {hi:.4f}]" for lo, hi in box)
    return str(box)
