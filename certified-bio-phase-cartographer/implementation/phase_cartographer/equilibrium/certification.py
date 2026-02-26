"""
Equilibrium certification combining Krawczyk and stability analysis.

Provides unified certification pipeline that:
1. Finds candidate equilibria via numerical methods
2. Certifies existence/uniqueness via Krawczyk operator
3. Classifies stability via eigenvalue enclosure
4. Generates certification records
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from ..ode.rhs import ODERightHandSide
from .krawczyk import KrawczykOperator, KrawczykResult, KrawczykStatus
from .newton import IntervalNewton, ParametricNewton, NewtonResult
from .stability import StabilityClassifier, StabilityType, EigenvalueEnclosure


class CertificationLevel(Enum):
    """Level of certification achieved."""
    FULL = "full"
    EXISTENCE_ONLY = "existence_only"
    NUMERICAL = "numerical"
    FAILED = "failed"


@dataclass
class EquilibriumCertificate:
    """
    Certificate for a single equilibrium point.
    """
    state_enclosure: Optional[IntervalVector] = None
    parameter_box: Optional[IntervalVector] = None
    stability: StabilityType = StabilityType.UNKNOWN
    eigenvalue_enclosure: Optional[EigenvalueEnclosure] = None
    krawczyk_result: Optional[KrawczykResult] = None
    certification_level: CertificationLevel = CertificationLevel.FAILED
    timestamp: float = 0.0
    computation_time: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_certified(self) -> bool:
        return self.certification_level in (
            CertificationLevel.FULL, CertificationLevel.EXISTENCE_ONLY)
    
    @property
    def is_stable(self) -> bool:
        return self.stability in (
            StabilityType.STABLE_NODE, StabilityType.STABLE_FOCUS,
            StabilityType.STABLE_SPIRAL)
    
    @property
    def is_unstable(self) -> bool:
        return self.stability in (
            StabilityType.UNSTABLE_NODE, StabilityType.UNSTABLE_FOCUS,
            StabilityType.UNSTABLE_SPIRAL, StabilityType.SADDLE)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            'certification_level': self.certification_level.value,
            'stability': self.stability.value,
            'timestamp': self.timestamp,
            'computation_time': self.computation_time,
        }
        if self.state_enclosure is not None:
            result['state_enclosure'] = [
                (c.lo, c.hi) for c in self.state_enclosure.components
            ]
        if self.parameter_box is not None:
            result['parameter_box'] = [
                (c.lo, c.hi) for c in self.parameter_box.components
            ]
        if self.eigenvalue_enclosure is not None:
            result['eigenvalue_real_parts'] = [
                (rp.lo, rp.hi) for rp in self.eigenvalue_enclosure.real_parts
            ]
        result.update(self.metadata)
        return result
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'EquilibriumCertificate':
        """Deserialize from dictionary."""
        cert = cls()
        cert.certification_level = CertificationLevel(d.get('certification_level', 'failed'))
        cert.stability = StabilityType(d.get('stability', 'unknown'))
        cert.timestamp = d.get('timestamp', 0.0)
        cert.computation_time = d.get('computation_time', 0.0)
        if 'state_enclosure' in d:
            cert.state_enclosure = IntervalVector([
                Interval(lo, hi) for lo, hi in d['state_enclosure']
            ])
        if 'parameter_box' in d:
            cert.parameter_box = IntervalVector([
                Interval(lo, hi) for lo, hi in d['parameter_box']
            ])
        return cert


@dataclass
class RegimeCertificate:
    """Certificate for a regime classification within a parameter box."""
    parameter_box: IntervalVector
    n_equilibria: int
    equilibrium_certificates: List[EquilibriumCertificate]
    regime_label: str = ""
    certified: bool = False
    coverage_fraction: float = 0.0
    
    @property
    def n_stable(self) -> int:
        return sum(1 for c in self.equilibrium_certificates if c.is_stable)
    
    @property
    def n_unstable(self) -> int:
        return sum(1 for c in self.equilibrium_certificates if c.is_unstable)
    
    @property
    def is_bistable(self) -> bool:
        return self.n_stable >= 2
    
    @property
    def is_monostable(self) -> bool:
        return self.n_stable == 1
    
    def auto_label(self) -> str:
        """Generate automatic regime label."""
        if self.n_stable == 0:
            return "no_stable_eq"
        if self.n_stable == 1:
            stability = self.equilibrium_certificates[0].stability
            return f"monostable_{stability.value}"
        if self.n_stable == 2:
            return "bistable"
        return f"multistable_{self.n_stable}"


class EquilibriumCertifier:
    """
    Unified equilibrium certification pipeline.
    
    Workflow:
    1. Find candidate equilibria numerically
    2. Verify each via Krawczyk operator
    3. Classify stability via eigenvalue enclosure
    4. Count and label the regime
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 krawczyk_max_iter: int = 20,
                 newton_max_iter: int = 30,
                 search_depth: int = 10,
                 tolerance: float = 1e-10):
        self.rhs = rhs
        self.krawczyk = KrawczykOperator(rhs, max_iter=krawczyk_max_iter)
        self.newton = IntervalNewton(rhs, max_iter=newton_max_iter, tol=tolerance)
        self.parametric_newton = ParametricNewton(rhs, max_iter=newton_max_iter)
        self.stability = StabilityClassifier(rhs)
        self.search_depth = search_depth
        self.tolerance = tolerance
    
    def certify_equilibrium(self, x_guess: np.ndarray,
                           mu: IntervalVector,
                           search_radius: float = 0.5) -> EquilibriumCertificate:
        """
        Certify a single equilibrium near x_guess.
        """
        t_start = time.time()
        cert = EquilibriumCertificate()
        cert.parameter_box = mu
        cert.timestamp = t_start
        n = self.rhs.n_states
        X = IntervalVector.from_midpoint_radius(
            x_guess, np.full(n, search_radius))
        kr = self.krawczyk.verify(X, mu)
        if kr.verified and kr.enclosure is not None:
            cert.state_enclosure = kr.enclosure
            cert.krawczyk_result = kr
            cert.certification_level = CertificationLevel.EXISTENCE_ONLY
            stab, eig = self.stability.classify(kr.enclosure, mu)
            cert.stability = stab
            cert.eigenvalue_enclosure = eig
            if stab != StabilityType.UNKNOWN:
                cert.certification_level = CertificationLevel.FULL
        else:
            nr = self.newton.solve(X, mu)
            if nr.converged and nr.enclosure is not None:
                cert.state_enclosure = nr.enclosure
                cert.certification_level = CertificationLevel.NUMERICAL
                stab, eig = self.stability.classify(nr.enclosure, mu)
                cert.stability = stab
                cert.eigenvalue_enclosure = eig
            else:
                cert.certification_level = CertificationLevel.FAILED
        cert.computation_time = time.time() - t_start
        return cert
    
    def certify_all_equilibria(self, state_domain: IntervalVector,
                              mu: IntervalVector) -> List[EquilibriumCertificate]:
        """Find and certify all equilibria within a state domain."""
        t_start = time.time()
        kr_results = self.krawczyk.find_equilibria(state_domain, mu,
                                                  max_depth=self.search_depth)
        certificates = []
        for kr in kr_results:
            if kr.verified and kr.enclosure is not None:
                cert = EquilibriumCertificate()
                cert.parameter_box = mu
                cert.state_enclosure = kr.enclosure
                cert.krawczyk_result = kr
                cert.certification_level = CertificationLevel.EXISTENCE_ONLY
                stab, eig = self.stability.classify(kr.enclosure, mu)
                cert.stability = stab
                cert.eigenvalue_enclosure = eig
                if stab != StabilityType.UNKNOWN:
                    cert.certification_level = CertificationLevel.FULL
                cert.timestamp = time.time()
                cert.computation_time = time.time() - t_start
                certificates.append(cert)
        return certificates
    
    def certify_regime(self, state_domain: IntervalVector,
                      mu_box: IntervalVector) -> RegimeCertificate:
        """
        Certify the dynamical regime for a parameter box.
        """
        eq_certs = self.certify_all_equilibria(state_domain, mu_box)
        regime = RegimeCertificate(
            parameter_box=mu_box,
            n_equilibria=len(eq_certs),
            equilibrium_certificates=eq_certs,
            certified=all(c.is_certified for c in eq_certs) if eq_certs else False
        )
        regime.regime_label = regime.auto_label()
        if eq_certs:
            certified_count = sum(1 for c in eq_certs if c.is_certified)
            regime.coverage_fraction = certified_count / len(eq_certs)
        return regime
    
    def certify_parametric(self, state_domain: IntervalVector,
                          mu_box: IntervalVector,
                          equilibrium_guesses: Optional[List[np.ndarray]] = None) -> RegimeCertificate:
        """
        Certify regime uniformly over parameter box.
        """
        if equilibrium_guesses is None:
            mu_mid = IntervalVector([Interval(mu_box[i].mid) for i in range(mu_box.n)])
            eq_certs = self.certify_all_equilibria(state_domain, mu_mid)
            equilibrium_guesses = [
                c.state_enclosure.midpoint()
                for c in eq_certs
                if c.state_enclosure is not None
            ]
        certificates = []
        for guess in equilibrium_guesses:
            cert = self.certify_equilibrium(guess, mu_box, search_radius=1.0)
            certificates.append(cert)
        regime = RegimeCertificate(
            parameter_box=mu_box,
            n_equilibria=len(certificates),
            equilibrium_certificates=certificates,
            certified=all(c.is_certified for c in certificates) if certificates else False
        )
        regime.regime_label = regime.auto_label()
        return regime
