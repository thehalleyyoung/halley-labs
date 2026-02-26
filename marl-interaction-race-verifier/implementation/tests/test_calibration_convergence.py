"""Tests for marace.race.calibration_convergence — fixed-point convergence."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.race.calibration_convergence import (
    BanachFixedPointTheorem,
    ContractionCondition,
    AdaptiveCalibration,
    FixedPointCertificate,
)


# ======================================================================
# BanachFixedPointTheorem
# ======================================================================

class TestBanachFixedPointTheorem:
    """Test Banach fixed-point iteration."""

    def test_contraction_map_half(self):
        """f(x) = x/2 is a contraction with fixed point 0."""
        phi = lambda x: x / 2.0
        bfpt = BanachFixedPointTheorem(phi, interval=(0.0, 1.0))
        cert = bfpt.iterate(eps0=1.0)
        assert abs(cert.fixed_point) < 1e-6
        assert cert.iterations_used > 0

    def test_contraction_detected(self):
        """f(x) = x/2 has contraction constant q = 0.5."""
        phi = lambda x: x / 2.0
        bfpt = BanachFixedPointTheorem(phi, interval=(0.0, 1.0))
        is_contr, q = bfpt.verify_contraction()
        assert is_contr
        assert q < 1.0
        assert q == pytest.approx(0.5, abs=0.1)

    def test_affine_contraction(self):
        """f(x) = 0.3x + 0.2 has fixed point 2/7 ≈ 0.2857."""
        phi = lambda x: 0.3 * x + 0.2
        bfpt = BanachFixedPointTheorem(phi, interval=(0.0, 1.0))
        cert = bfpt.iterate(eps0=0.5)
        expected = 0.2 / 0.7  # 2/7
        assert cert.fixed_point == pytest.approx(expected, abs=1e-5)

    def test_non_contraction_detected(self):
        """f(x) = 2x has q >= 1, not a contraction."""
        phi = lambda x: 2.0 * x
        bfpt = BanachFixedPointTheorem(phi, interval=(0.0, 1.0))
        is_contr, q = bfpt.verify_contraction()
        assert not is_contr
        assert q >= 1.0

    def test_contraction_constant_property(self):
        phi = lambda x: x / 3.0
        bfpt = BanachFixedPointTheorem(phi, interval=(0.0, 1.0))
        bfpt.verify_contraction()
        assert bfpt.contraction_constant is not None
        assert bfpt.contraction_constant < 1.0

    def test_certificate_verify(self):
        """Certificate should verify against the iteration map."""
        phi = lambda x: x / 2.0
        bfpt = BanachFixedPointTheorem(phi, interval=(0.0, 1.0))
        cert = bfpt.iterate(eps0=0.8)
        assert cert.verify(phi)

    def test_certificate_to_dict(self):
        phi = lambda x: x / 2.0
        bfpt = BanachFixedPointTheorem(phi, interval=(0.0, 1.0))
        cert = bfpt.iterate(eps0=0.5)
        d = cert.to_dict()
        assert "fixed_point" in d
        assert "contraction_constant" in d

    @pytest.mark.parametrize("q", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_various_contraction_rates(self, q):
        """Various contraction constants should converge."""
        phi = lambda x: q * x
        bfpt = BanachFixedPointTheorem(phi, interval=(0.0, 1.0))
        cert = bfpt.iterate(eps0=1.0)
        assert abs(cert.fixed_point) < 1e-4


# ======================================================================
# ContractionCondition
# ======================================================================

class TestContractionCondition:
    """Test contraction condition diagnosis."""

    def test_contraction_q_less_than_1(self):
        """For f(x) = x/2, the contraction constant should be ~0.5."""
        margin_fn = lambda x: x / 2.0
        cc = ContractionCondition(margin_fn, lipschitz_policy=1.0,
                                  interval=(0.0, 1.0))
        cc.compute()
        q = cc.contraction_constant
        assert q < 1.0

    def test_non_contraction_q_geq_1(self):
        """For f(x) = 2x, the contraction constant should be >= 1."""
        margin_fn = lambda x: 2.0 * x
        cc = ContractionCondition(margin_fn, lipschitz_policy=1.0,
                                  interval=(0.0, 1.0))
        cc.compute()
        q = cc.contraction_constant
        assert q >= 1.0

    def test_is_contraction_property(self):
        margin_fn = lambda x: x / 3.0
        cc = ContractionCondition(margin_fn, lipschitz_policy=1.0,
                                  interval=(0.0, 1.0))
        cc.compute()
        assert cc.is_contraction

    def test_diagnose_output(self):
        margin_fn = lambda x: x / 2.0
        cc = ContractionCondition(margin_fn, lipschitz_policy=1.0,
                                  interval=(0.0, 1.0))
        cc.compute()
        diag = cc.diagnose()
        assert isinstance(diag, dict)


# ======================================================================
# AdaptiveCalibration
# ======================================================================

class TestAdaptiveCalibration:
    """Test adaptive calibration with damping."""

    def test_convergence_with_damping(self):
        """Damped iteration should converge for a contractive map."""
        phi = lambda x: 0.4 * x + 0.1
        ac = AdaptiveCalibration(phi, interval=(0.0, 1.0))
        cert = ac.calibrate(eps0=0.5)
        expected = 0.1 / 0.6
        assert cert.fixed_point == pytest.approx(expected, abs=1e-4)

    def test_optimal_damping(self):
        """Optimal damping α* = 2/(1+q)."""
        alpha = AdaptiveCalibration.optimal_damping(q=0.5)
        assert alpha == pytest.approx(2.0 / 1.5, abs=1e-6)

    def test_optimal_damping_zero_q(self):
        alpha = AdaptiveCalibration.optimal_damping(q=0.0)
        assert alpha == 1.0

    def test_damped_contraction_constant(self):
        """q_α = max(|1-α(1-q)|, |1-α(1+q)|)."""
        q_alpha = AdaptiveCalibration.damped_contraction_constant(
            q=0.5, alpha=1.0
        )
        # |1-1*(1-0.5)| = 0.5, |1-1*(1+0.5)| = 0.5
        assert q_alpha == pytest.approx(0.5, abs=1e-6)

    def test_damped_contraction_constant_optimal(self):
        """With optimal damping α=2/(1+q), verify the formula."""
        q = 3.0
        alpha = AdaptiveCalibration.optimal_damping(q)
        # α = 2/(1+3) = 0.5
        assert alpha == pytest.approx(0.5, abs=1e-6)
        q_alpha = AdaptiveCalibration.damped_contraction_constant(q, alpha)
        # q_α = max(|1-α(1-q)|, |1-α(1+q)|) = max(|1+1|, |1-2|) = max(2,1) = 2
        expected = max(abs(1.0 - alpha * (1.0 - q)),
                       abs(1.0 - alpha * (1.0 + q)))
        assert q_alpha == pytest.approx(expected, abs=1e-6)


# ======================================================================
# FixedPointCertificate
# ======================================================================

class TestFixedPointCertificate:
    """Test FixedPointCertificate dataclass."""

    def test_a_priori_bound_finite(self):
        cert = FixedPointCertificate(
            fixed_point=0.0, contraction_constant=0.5,
            iterations_used=10, error_bound=0.001,
        )
        bound = cert.a_priori_bound
        assert np.isfinite(bound)
        assert bound >= 0

    def test_a_priori_bound_infinite_for_non_contraction(self):
        cert = FixedPointCertificate(
            contraction_constant=1.5, iterations_used=10, error_bound=0.1,
        )
        assert cert.a_priori_bound == float("inf")

    def test_verify_correct_fixed_point(self):
        cert = FixedPointCertificate(fixed_point=0.0, tolerance=1e-6)
        assert cert.verify(lambda x: x / 2.0)

    def test_verify_wrong_fixed_point(self):
        cert = FixedPointCertificate(fixed_point=0.5, tolerance=1e-6)
        assert not cert.verify(lambda x: x / 2.0)
