"""
Tests for certificate composition and regime derivation.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_cartographer.tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)
from phase_cartographer.atlas.composition import (
    RegimeDerivation, CertificateProofObject,
    CompositionResult, boxes_adjacent, boxes_disjoint,
    verify_atlas_composition,
)


def _make_cell(box, regime, stability, minicheck=True):
    """Helper to create a CertifiedCell."""
    eq = EquilibriumCertificate(
        state_enclosure=[(0.5, 1.5), (0.5, 1.5)],
        stability=stability,
        eigenvalue_real_parts=[(-2.0, -0.5), (-2.0, -0.5)],
        krawczyk_contraction=0.3,
        krawczyk_iterations=5,
    )
    cell = CertifiedCell(
        parameter_box=box,
        model_name="test",
        n_states=2,
        n_params=2,
        equilibria=[eq],
        regime=regime,
        tier=VerificationTier.TIER1_IA,
        minicheck_passed=minicheck,
    )
    return cell


class TestBoxGeometry:
    def test_adjacent_boxes(self):
        box1 = [(0.0, 1.0), (0.0, 1.0)]
        box2 = [(1.0, 2.0), (0.0, 1.0)]
        assert boxes_adjacent(box1, box2)

    def test_non_adjacent_boxes(self):
        box1 = [(0.0, 1.0), (0.0, 1.0)]
        box2 = [(2.0, 3.0), (0.0, 1.0)]
        assert not boxes_adjacent(box1, box2)

    def test_disjoint_boxes(self):
        box1 = [(0.0, 1.0), (0.0, 1.0)]
        box2 = [(2.0, 3.0), (0.0, 1.0)]
        assert boxes_disjoint(box1, box2)

    def test_overlapping_boxes(self):
        box1 = [(0.0, 1.5), (0.0, 1.0)]
        box2 = [(1.0, 2.0), (0.0, 1.0)]
        assert not boxes_disjoint(box1, box2)


class TestRegimeDerivation:
    def test_monostable_derivation(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        deriv = RegimeDerivation.derive(cell)
        assert deriv.rule_name == "MONO"
        assert deriv.conclusion == "monostable"
        assert deriv.n_stable == 1

    def test_bistable_derivation(self):
        eq1 = EquilibriumCertificate(
            state_enclosure=[(0.5, 1.0), (0.5, 1.0)],
            stability=StabilityType.STABLE_NODE,
            eigenvalue_real_parts=[(-2.0, -0.5)],
            krawczyk_contraction=0.3,
            krawczyk_iterations=5,
        )
        eq2 = EquilibriumCertificate(
            state_enclosure=[(2.0, 3.0), (2.0, 3.0)],
            stability=StabilityType.STABLE_FOCUS,
            eigenvalue_real_parts=[(-1.0, -0.2)],
            krawczyk_contraction=0.4,
            krawczyk_iterations=6,
        )
        cell = CertifiedCell(
            parameter_box=[(0.0, 1.0), (0.0, 1.0)],
            model_name="test",
            n_states=2, n_params=2,
            equilibria=[eq1, eq2],
            regime=RegimeType.BISTABLE,
            tier=VerificationTier.TIER1_IA,
            minicheck_passed=True,
        )
        deriv = RegimeDerivation.derive(cell)
        assert deriv.rule_name == "BI"
        assert deriv.n_stable == 2

    def test_derivation_to_dict(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        deriv = RegimeDerivation.derive(cell)
        d = deriv.to_dict()
        assert "premises" in d
        assert "rule" in d
        assert d["rule"] == "MONO"


class TestCertificateProofObject:
    def test_from_cell(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        proof = CertificateProofObject.from_cell(cell)
        assert proof.derivation.rule_name == "MONO"
        assert proof.tier_evidence["tier1_minicheck"]["passed"]

    def test_to_dict(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        proof = CertificateProofObject.from_cell(cell)
        d = proof.to_dict()
        assert "cell" in d
        assert "derivation" in d


class TestAtlasComposition:
    def test_valid_atlas(self):
        cell1 = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        cell2 = _make_cell(
            [(1.0, 2.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        domain = [(0.0, 2.0), (0.0, 1.0)]
        result = verify_atlas_composition([cell1, cell2], domain)
        assert result.valid
        assert abs(result.coverage_fraction - 1.0) < 1e-10

    def test_overlapping_cells_detected(self):
        cell1 = _make_cell(
            [(0.0, 1.5), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        cell2 = _make_cell(
            [(1.0, 2.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        domain = [(0.0, 2.0), (0.0, 1.0)]
        result = verify_atlas_composition([cell1, cell2], domain)
        assert not result.valid

    def test_boundary_detection(self):
        cell1 = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
        )
        cell2 = _make_cell(
            [(1.0, 2.0), (0.0, 1.0)],
            RegimeType.BISTABLE,
            StabilityType.STABLE_NODE,
        )
        domain = [(0.0, 2.0), (0.0, 1.0)]
        result = verify_atlas_composition([cell1, cell2], domain)
        assert len(result.boundary_cells) == 1

    def test_uncertified_cell_flagged(self):
        cell = _make_cell(
            [(0.0, 1.0), (0.0, 1.0)],
            RegimeType.MONOSTABLE,
            StabilityType.STABLE_NODE,
            minicheck=False,
        )
        domain = [(0.0, 1.0), (0.0, 1.0)]
        result = verify_atlas_composition([cell], domain)
        assert not result.valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
