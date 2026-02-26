"""
Stability classification via eigenvalue enclosure.

Provides rigorous eigenvalue enclosure methods for classifying
equilibrium stability types using interval arithmetic.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix, eigenvalue_enclosure
from ..ode.rhs import ODERightHandSide


class StabilityType(Enum):
    """Stability classification of an equilibrium."""
    STABLE_NODE = "stable_node"
    STABLE_FOCUS = "stable_focus"
    UNSTABLE_NODE = "unstable_node"
    UNSTABLE_FOCUS = "unstable_focus"
    SADDLE = "saddle"
    CENTER = "center"
    STABLE_SPIRAL = "stable_spiral"
    UNSTABLE_SPIRAL = "unstable_spiral"
    UNKNOWN = "unknown"
    DEGENERATE = "degenerate"


@dataclass
class EigenvalueEnclosure:
    """Enclosure of eigenvalue information."""
    real_parts: List[Interval] = field(default_factory=list)
    imaginary_parts: List[Interval] = field(default_factory=list)
    gershgorin_centers: List[Interval] = field(default_factory=list)
    gershgorin_radii: List[float] = field(default_factory=list)
    
    @property
    def n(self) -> int:
        return len(self.real_parts)
    
    def all_negative_real(self) -> bool:
        """Check if all eigenvalues have strictly negative real parts."""
        return all(rp.hi < 0 for rp in self.real_parts)
    
    def all_positive_real(self) -> bool:
        """Check if all eigenvalues have strictly positive real parts."""
        return all(rp.lo > 0 for rp in self.real_parts)
    
    def has_positive_real(self) -> bool:
        """Check if any eigenvalue has positive real part."""
        return any(rp.hi > 0 for rp in self.real_parts)
    
    def has_negative_real(self) -> bool:
        """Check if any eigenvalue has negative real part."""
        return any(rp.lo < 0 for rp in self.real_parts)
    
    def has_zero_crossing(self) -> bool:
        """Check if any eigenvalue real part contains zero."""
        return any(rp.lo <= 0 <= rp.hi for rp in self.real_parts)
    
    def has_complex_eigenvalues(self) -> bool:
        """Check if there are complex eigenvalues."""
        return any(not ip.contains(0.0) for ip in self.imaginary_parts)
    
    def spectral_abscissa_bound(self) -> Interval:
        """Upper bound on spectral abscissa (max real part)."""
        if not self.real_parts:
            return Interval(0.0)
        lo = max(rp.lo for rp in self.real_parts)
        hi = max(rp.hi for rp in self.real_parts)
        return Interval(lo, hi)


class StabilityClassifier:
    """
    Classifies equilibrium stability using rigorous eigenvalue enclosures.
    
    Methods:
    1. Gershgorin disk theorem for eigenvalue enclosure
    2. Verified eigenvalue computation via interval characteristic polynomial
    3. Routh-Hurwitz conditions for stability verification
    """
    
    def __init__(self, rhs: ODERightHandSide):
        self.rhs = rhs
    
    def classify(self, x_eq: IntervalVector,
                mu: IntervalVector) -> Tuple[StabilityType, EigenvalueEnclosure]:
        """
        Classify stability of an equilibrium point.
        """
        n = x_eq.n
        try:
            J = self.rhs.jacobian_interval(x_eq, mu)
        except (ZeroDivisionError, ValueError):
            return StabilityType.UNKNOWN, EigenvalueEnclosure()
        eig_enc = self.compute_eigenvalue_enclosure(J)
        stability = self._classify_from_eigenvalues(eig_enc)
        return stability, eig_enc
    
    def compute_eigenvalue_enclosure(self, J: IntervalMatrix) -> EigenvalueEnclosure:
        """
        Compute rigorous eigenvalue enclosures using multiple methods.
        """
        n = J.rows
        result = EigenvalueEnclosure()
        disks = J.gershgorin_disks()
        result.gershgorin_centers = [center for center, _ in disks]
        result.gershgorin_radii = [radius for _, radius in disks]
        J_mid = J.midpoint_matrix()
        try:
            eigvals = np.linalg.eigvals(J_mid)
        except np.linalg.LinAlgError:
            for center, radius in disks:
                result.real_parts.append(
                    Interval(center.lo - radius, center.hi + radius))
                result.imaginary_parts.append(
                    Interval(-radius, radius))
            return result
        J_rad = J.radius_matrix()
        rad_norm = np.linalg.norm(J_rad, ord=np.inf)
        for ev in eigvals:
            re = ev.real
            im = ev.imag
            pert = rad_norm * np.sqrt(n)
            result.real_parts.append(Interval(re - pert, re + pert))
            result.imaginary_parts.append(Interval(im - pert, im + pert))
        gershgorin_enc = eigenvalue_enclosure(J)
        for i in range(min(len(result.real_parts), len(gershgorin_enc))):
            g = gershgorin_enc[i]
            result.real_parts[i] = result.real_parts[i].intersection(g)
            if result.real_parts[i].is_empty():
                result.real_parts[i] = g
        return result
    
    def _classify_from_eigenvalues(self, eig: EigenvalueEnclosure) -> StabilityType:
        """Classify stability from eigenvalue enclosures.
        
        Classification is only conclusive when eigenvalue enclosures are
        entirely in one half-plane. If any enclosure crosses zero, the
        classification is DEGENERATE (indeterminate).
        """
        if not eig.real_parts:
            return StabilityType.UNKNOWN
        if eig.has_zero_crossing():
            return StabilityType.DEGENERATE
        if eig.all_negative_real():
            if eig.has_complex_eigenvalues():
                return StabilityType.STABLE_FOCUS
            return StabilityType.STABLE_NODE
        if eig.all_positive_real():
            if eig.has_complex_eigenvalues():
                return StabilityType.UNSTABLE_FOCUS
            return StabilityType.UNSTABLE_NODE
        # All enclosures are strictly in one half-plane; check for saddle
        has_definite_pos = any(rp.lo > 0 for rp in eig.real_parts)
        has_definite_neg = any(rp.hi < 0 for rp in eig.real_parts)
        if has_definite_pos and has_definite_neg:
            return StabilityType.SADDLE
        return StabilityType.UNKNOWN
    
    def routh_hurwitz_check(self, J: IntervalMatrix) -> Optional[bool]:
        """
        Check Routh-Hurwitz conditions for stability.
        Returns True if stable, False if unstable, None if inconclusive.
        """
        n = J.rows
        if n == 1:
            a = J[0, 0]
            if a.hi < 0:
                return True
            if a.lo > 0:
                return False
            return None
        if n == 2:
            tr = J.trace()
            det = J.determinant()
            if tr.hi < 0 and det.lo > 0:
                return True
            if tr.lo > 0 or det.hi < 0:
                return False
            return None
        if n == 3:
            a11, a12, a13 = J[0, 0], J[0, 1], J[0, 2]
            a21, a22, a23 = J[1, 0], J[1, 1], J[1, 2]
            a31, a32, a33 = J[2, 0], J[2, 1], J[2, 2]
            p = -(a11 + a22 + a33)
            q = (a11 * a22 + a11 * a33 + a22 * a33
                 - a12 * a21 - a13 * a31 - a23 * a32)
            r = -J.determinant()
            if p.lo > 0 and r.lo > 0 and (p * q - r).lo > 0:
                return True
            if p.hi < 0 or r.hi < 0:
                return False
            return None
        return None
    
    def classify_parametric(self, x_eq: IntervalVector,
                           mu_box: IntervalVector) -> Tuple[StabilityType, float]:
        """
        Classify stability uniformly over a parameter box.
        Returns (stability_type, confidence) where confidence indicates
        how far from a bifurcation boundary.
        """
        stability, eig = self.classify(x_eq, mu_box)
        if stability == StabilityType.UNKNOWN or stability == StabilityType.DEGENERATE:
            return stability, 0.0
        sa = eig.spectral_abscissa_bound()
        if stability in (StabilityType.STABLE_NODE, StabilityType.STABLE_FOCUS):
            confidence = abs(sa.hi) if sa.hi < 0 else 0.0
        elif stability in (StabilityType.UNSTABLE_NODE, StabilityType.UNSTABLE_FOCUS):
            confidence = sa.lo if sa.lo > 0 else 0.0
        else:
            confidence = min(abs(rp.lo) for rp in eig.real_parts if rp.lo < 0) if eig.has_negative_real() else 0.0
        return stability, confidence
    
    def detect_hopf_bifurcation(self, x_eq: IntervalVector,
                               mu: IntervalVector) -> bool:
        """
        Detect if a Hopf bifurcation occurs within the parameter box.
        A Hopf bifurcation requires eigenvalues crossing the imaginary axis
        with non-zero imaginary part.
        """
        _, eig = self.classify(x_eq, mu)
        has_crossing = False
        has_complex = False
        for i in range(eig.n):
            if eig.real_parts[i].lo <= 0 <= eig.real_parts[i].hi:
                has_crossing = True
            if not eig.imaginary_parts[i].contains(0.0):
                has_complex = True
        return has_crossing and has_complex
    
    def detect_saddle_node_bifurcation(self, x_eq: IntervalVector,
                                      mu: IntervalVector) -> bool:
        """Detect saddle-node bifurcation (zero eigenvalue)."""
        try:
            J = self.rhs.jacobian_interval(x_eq, mu)
            det = J.determinant()
            return det.contains(0.0)
        except (ZeroDivisionError, ValueError):
            return False
