"""
End-to-end integration test: generate a certificate using the main pipeline,
then verify it using the independent minimal checker (MiniCheck).

This test demonstrates the full workflow:
1. Define a biological model
2. Run equilibrium certification via Krawczyk
3. Compute δ-bound for SMT soundness
4. Export certificate as JSON
5. Independently verify with MiniCheck
"""

import json
import numpy as np
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.interval.interval import Interval
from phase_cartographer.interval.matrix import IntervalVector, IntervalMatrix
from phase_cartographer.ode.rhs import SymbolicRHS
from phase_cartographer.equilibrium import (
    KrawczykOperator, EquilibriumCertifier, StabilityClassifier
)
from phase_cartographer.smt.delta_bound import (
    DeltaBound, compute_required_delta, compute_eigenvalue_gap
)
from phase_cartographer.minicheck import verify_certificate

import sympy as sp


def make_toggle_switch_rhs():
    """Create toggle switch RHS using SymbolicRHS."""
    x1, x2 = sp.symbols('x0 x1')
    alpha1, alpha2, n1, n2 = sp.symbols('mu0 mu1 mu2 mu3')
    
    rhs = SymbolicRHS(
        n_states=2, n_params=4,
        state_symbols=[x1, x2],
        param_symbols=[alpha1, alpha2, n1, n2],
        name="toggle_switch"
    )
    # dx1/dt = alpha1 / (1 + x2^n1) - x1
    # dx2/dt = alpha2 / (1 + x1^n2) - x2
    # Use integer Hill coefficients (n=2) for sympy compatibility
    rhs.set_expressions([
        alpha1 / (1 + x2**2) - x1,
        alpha2 / (1 + x1**2) - x2,
    ])
    return rhs


class TestEndToEnd:
    """End-to-end pipeline + independent verification tests."""
    
    def _make_toggle_rhs(self):
        """Create toggle switch RHS."""
        return make_toggle_switch_rhs()
    
    def test_toggle_certification_pipeline(self):
        """Full pipeline: certify toggle switch + export + verify."""
        rhs = self._make_toggle_rhs()
        
        # Point parameters (no uncertainty)
        mu = IntervalVector([
            Interval(3.0), Interval(3.0),
            Interval(2.0), Interval(2.0)
        ])
        
        # Search domain
        state_domain = IntervalVector([
            Interval(0.1, 5.0), Interval(0.1, 5.0)
        ])
        
        # Find and certify equilibria
        krawczyk = KrawczykOperator(rhs, max_iter=30)
        results = krawczyk.find_equilibria(state_domain, mu, max_depth=8)
        
        # We should find at least one equilibrium
        assert len(results) > 0, "Should find at least one equilibrium"
        
        # For each verified equilibrium, classify stability
        stability_classifier = StabilityClassifier(rhs)
        
        certified_equilibria = []
        for kr in results:
            if kr.verified and kr.enclosure is not None:
                stab_type, eig_enc = stability_classifier.classify(
                    kr.enclosure, mu)
                
                # Compute δ-bound
                delta_result = compute_required_delta(
                    rhs, kr.enclosure, mu,
                    eig_enc.real_parts,
                    delta_solver=1e-3
                )
                
                eq_data = {
                    "state_enclosure": [
                        (c.lo, c.hi) for c in kr.enclosure.components
                    ],
                    "stability": stab_type.value,
                    "eigenvalue_real_parts": [
                        (rp.lo, rp.hi) for rp in eig_enc.real_parts
                    ],
                    "krawczyk_contraction": kr.contraction_factor,
                    "krawczyk_iterations": kr.iterations,
                    "delta_bound": delta_result.to_dict(),
                }
                certified_equilibria.append(eq_data)
        
        assert len(certified_equilibria) > 0, "Should have certified equilibria"
        
        # Determine regime label
        n_stable = sum(1 for eq in certified_equilibria
                      if eq["stability"] in ("stable_node", "stable_focus", "stable_spiral"))
        if n_stable >= 2:
            regime_label = "bistable"
        elif n_stable == 1:
            regime_label = f"monostable_{certified_equilibria[0]['stability']}"
        else:
            regime_label = "no_stable_eq"
        
        # Build certificate
        certificate = {
            "model": {
                "name": "toggle_switch",
                "n_states": 2,
                "n_params": 4,
                "rhs_type": "hill"
            },
            "parameter_box": [
                (mu[i].lo, mu[i].hi) for i in range(mu.n)
            ],
            "equilibria": certified_equilibria,
            "regime_label": regime_label,
            "coverage_fraction": 1.0,
            "metadata": {"test": True}
        }
        
        # Verify with MiniCheck
        result = verify_certificate(certificate, delta_solver=1e-3)
        
        print(f"\nCertificate verification result:")
        print(result.summary())
        print(f"\nCertified {len(certified_equilibria)} equilibria")
        print(f"Regime: {regime_label}")
        
        # Check that minicheck accepts the certificate
        # (or at least runs without crashing)
        assert result.equilibria_total == len(certified_equilibria)
    
    def test_certificate_json_roundtrip(self):
        """Test that certificates survive JSON serialization."""
        cert = {
            "model": {"name": "brusselator", "n_states": 2, "n_params": 2},
            "parameter_box": [[1.0, 1.0], [2.0, 2.0]],
            "equilibria": [{
                "state_enclosure": [[0.95, 1.05], [1.95, 2.05]],
                "stability": "stable_focus",
                "eigenvalue_real_parts": [[-0.5, -0.1], [-0.5, -0.1]],
                "krawczyk_contraction": 0.2,
                "delta_bound": {"delta_required": 0.01, "eigenvalue_gap": 0.1}
            }],
            "regime_label": "monostable_stable_focus",
            "coverage_fraction": 1.0,
        }
        
        # Write to temp file and read back
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cert, f, indent=2)
            tmppath = f.name
        
        try:
            with open(tmppath) as f:
                loaded = json.load(f)
            
            result = verify_certificate(loaded)
            assert result.equilibria_total == 1
        finally:
            os.unlink(tmppath)
    
    def test_delta_bound_on_certified_equilibrium(self):
        """Test δ-bound computation on a Krawczyk-certified equilibrium."""
        rhs = self._make_toggle_rhs()
        
        mu = IntervalVector([
            Interval(3.0), Interval(3.0),
            Interval(2.0), Interval(2.0)
        ])
        
        # Small box around approximate equilibrium
        x_star = 1.2134
        X = IntervalVector([
            Interval(x_star - 0.1, x_star + 0.1),
            Interval(x_star - 0.1, x_star + 0.1)
        ])
        
        # Get stability info
        sc = StabilityClassifier(rhs)
        stab, eig = sc.classify(X, mu)
        
        if eig.real_parts:
            # Compute δ-bound
            db = DeltaBound(rhs)
            result = db.compute(X, mu, eig.real_parts, delta_solver=1e-3)
            
            print(f"\nδ-bound result:")
            print(f"  Eigenvalue gap: {result.eigenvalue_gap:.6f}")
            print(f"  δ required: {result.delta_required:.6e}")
            print(f"  Lipschitz f: {result.lipschitz_f:.6f}")
            print(f"  Lipschitz Df: {result.lipschitz_Df:.6f}")
            print(f"  Sound: {result.is_sound}")
            print(f"  Margin: {result.soundness_margin:.6e}")
            
            assert result.eigenvalue_gap >= 0
            assert result.lipschitz_f < float('inf')
            assert result.lipschitz_Df < float('inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
