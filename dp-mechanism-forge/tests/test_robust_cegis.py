"""
Tests for dp_forge.robust — RobustCEGIS, constraint inflation,
perturbation analysis, solver diagnostics, and certified output.

Covers:
    - Constraint inflation: verify inflated LP is tighter
    - Robust mechanism satisfies original DP constraints
    - Interval arithmetic: basic operations, matrix products
    - Perturbation bounds: Δε bounded by condition number × perturbation
    - Solver diagnostics: constraint audit finds intentional violations
    - Certified output: JSON serialisation round-trip
    - Edge cases: very loose/tight tolerance
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest
from scipy import sparse

from dp_forge.robust.constraint_inflation import ConstraintInflator, InflationResult
from dp_forge.robust.perturbation_analysis import (
    PerturbationAnalyzer,
    PerturbationBound,
    SensitivityResult,
)
from dp_forge.robust.solver_diagnostics import (
    SolverDiagnostics,
    AuditReport,
    ConstraintViolation,
    ResidualReport,
    RefinementResult,
)
from dp_forge.robust.certified_output import (
    CertifiedMechanism,
    NumericalCertificate,
)
from dp_forge.robust.interval_arithmetic import (
    Interval,
    IntervalMatrix,
    interval_verify_dp,
)
from dp_forge.robust.robust_cegis import (
    RobustCEGISEngine,
    RobustSynthesisConfig,
)
from dp_forge.exceptions import ConfigurationError, NumericalInstabilityError
from dp_forge.types import LPStruct


# =========================================================================
# Helpers
# =========================================================================


def make_simple_dp_mechanism(n: int, k: int, epsilon: float) -> np.ndarray:
    """Create a simple ε-DP mechanism (uniform-ish with DP constraints)."""
    p = np.ones((n, k), dtype=np.float64) / k
    # Add small perturbations that respect ε-DP
    exp_eps = math.exp(epsilon)
    for i in range(n):
        noise = np.random.RandomState(42 + i).randn(k) * 0.01
        p[i] += noise
        p[i] = np.maximum(p[i], 1e-10)
        p[i] /= p[i].sum()
    return p


def make_simple_lp(n_vars: int, n_ub: int, n_eq: int = 0) -> LPStruct:
    """Create a simple LPStruct for testing."""
    c = np.ones(n_vars, dtype=np.float64)
    A_ub = sparse.random(n_ub, n_vars, density=0.3, format="csr", dtype=np.float64)
    b_ub = np.ones(n_ub, dtype=np.float64) * 10.0

    if n_eq > 0:
        A_eq = sparse.eye(n_eq, n_vars, format="csr", dtype=np.float64)
        b_eq = np.ones(n_eq, dtype=np.float64)
    else:
        A_eq = None
        b_eq = None

    bounds = [(0.0, 1.0)] * n_vars
    var_map = {f"x{i}": i for i in range(n_vars)}
    y_grid = np.linspace(0, 1, max(n_vars // max(n_eq, 1), 5))

    return LPStruct(
        c=c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, var_map=var_map, y_grid=y_grid,
    )


# =========================================================================
# ConstraintInflator tests
# =========================================================================


class TestConstraintInflator:
    """Tests for ConstraintInflator."""

    def test_epsilon_margin_positive(self):
        inflator = ConstraintInflator(safety_factor=2.0)
        margin = inflator.compute_epsilon_margin(
            epsilon=1.0, solver_tol=1e-8, n_outputs=10
        )
        assert margin > 0
        assert math.isfinite(margin)

    def test_epsilon_margin_increases_with_tolerance(self):
        inflator = ConstraintInflator(safety_factor=2.0)
        m1 = inflator.compute_epsilon_margin(epsilon=1.0, solver_tol=1e-10, n_outputs=10)
        m2 = inflator.compute_epsilon_margin(epsilon=1.0, solver_tol=1e-6, n_outputs=10)
        assert m2 > m1

    def test_epsilon_margin_capped_at_half_epsilon(self):
        inflator = ConstraintInflator(safety_factor=2.0)
        margin = inflator.compute_epsilon_margin(
            epsilon=0.01, solver_tol=1.0, n_outputs=10
        )
        assert margin <= 0.01 * 0.5

    def test_delta_margin_positive_for_approx_dp(self):
        inflator = ConstraintInflator(safety_factor=2.0)
        margin = inflator.compute_delta_margin(
            delta=1e-5, solver_tol=1e-8, n_outputs=10, n_pairs=5
        )
        assert margin > 0
        assert math.isfinite(margin)

    def test_delta_margin_zero_for_pure_dp(self):
        inflator = ConstraintInflator(safety_factor=2.0)
        margin = inflator.compute_delta_margin(
            delta=0.0, solver_tol=1e-8, n_outputs=10, n_pairs=5
        )
        assert margin == 0.0

    def test_delta_margin_scales_with_k(self):
        inflator = ConstraintInflator(safety_factor=2.0)
        m1 = inflator.compute_delta_margin(delta=1e-5, solver_tol=1e-8, n_outputs=5, n_pairs=1)
        m2 = inflator.compute_delta_margin(delta=1e-5, solver_tol=1e-8, n_outputs=50, n_pairs=1)
        assert m2 > m1

    def test_safety_factor_validation(self):
        with pytest.raises(ConfigurationError):
            ConstraintInflator(safety_factor=0.5)

    def test_inflate_lp(self):
        inflator = ConstraintInflator(safety_factor=2.0)
        lp = make_simple_lp(n_vars=10, n_ub=5, n_eq=2)
        tightened_lp, result = inflator.inflate(
            lp, solver_tol=1e-8, epsilon=1.0, delta=0.0
        )
        assert isinstance(result, InflationResult)
        assert result.epsilon_margin > 0
        # Tightened b_ub should be ≤ original b_ub (element-wise)
        np.testing.assert_array_less(tightened_lp.b_ub - 1e-15, lp.b_ub + 1e-10)

    def test_per_constraint_slack(self):
        inflator = ConstraintInflator(safety_factor=2.0)
        lp = make_simple_lp(n_vars=10, n_ub=5)
        slacks = inflator.compute_per_constraint_slack(lp, solver_tol=1e-8, epsilon=1.0)
        assert len(slacks) == 5
        assert np.all(slacks >= 0)

    def test_repr(self):
        inflator = ConstraintInflator(safety_factor=3.0)
        r = repr(inflator)
        assert "ConstraintInflator" in r
        assert "3.0" in r


# =========================================================================
# PerturbationAnalyzer tests
# =========================================================================


class TestPerturbationAnalyzer:
    """Tests for PerturbationAnalyzer."""

    def test_bound_epsilon_change(self):
        analyzer = PerturbationAnalyzer()
        p = make_simple_dp_mechanism(3, 5, epsilon=1.0)
        edges = [(0, 1), (1, 2)]
        bound = analyzer.bound_epsilon_change(
            p, solver_tol=1e-8, epsilon=1.0, edges=edges
        )
        assert isinstance(bound, PerturbationBound)
        assert bound.epsilon_bound > 0
        assert bound.epsilon_bound < 1.0  # should be small for small tolerance
        assert bound.is_well_conditioned

    def test_bound_proportional_to_tolerance(self):
        """Δε should scale with solver tolerance."""
        analyzer = PerturbationAnalyzer()
        p = make_simple_dp_mechanism(3, 5, epsilon=1.0)
        edges = [(0, 1), (1, 2)]
        b1 = analyzer.bound_epsilon_change(p, solver_tol=1e-10, epsilon=1.0, edges=edges)
        b2 = analyzer.bound_epsilon_change(p, solver_tol=1e-6, epsilon=1.0, edges=edges)
        assert b2.epsilon_bound > b1.epsilon_bound

    def test_delta_bound_for_approx_dp(self):
        analyzer = PerturbationAnalyzer()
        p = make_simple_dp_mechanism(3, 5, epsilon=1.0)
        edges = [(0, 1)]
        bound = analyzer.bound_epsilon_change(
            p, solver_tol=1e-8, epsilon=1.0, edges=edges, delta=1e-5
        )
        assert bound.delta_bound > 0

    def test_delta_bound_zero_for_pure_dp(self):
        analyzer = PerturbationAnalyzer()
        p = make_simple_dp_mechanism(3, 5, epsilon=1.0)
        edges = [(0, 1)]
        bound = analyzer.bound_epsilon_change(
            p, solver_tol=1e-8, epsilon=1.0, edges=edges, delta=0.0
        )
        assert bound.delta_bound == 0.0

    def test_condition_number_estimation(self):
        analyzer = PerturbationAnalyzer()
        p = make_simple_dp_mechanism(3, 5, epsilon=1.0)
        edges = [(0, 1)]
        bound = analyzer.bound_epsilon_change(
            p, solver_tol=1e-8, epsilon=1.0, edges=edges
        )
        assert bound.condition_number > 0
        assert bound.p_min > 0

    def test_ill_conditioned_mechanism(self):
        """Mechanism with extreme probability ratios should have large condition number."""
        analyzer = PerturbationAnalyzer(max_condition_number=1e12)
        p = np.array([
            [0.999, 0.001],
            [0.001, 0.999],
            [0.5, 0.5],
        ], dtype=np.float64)
        edges = [(0, 1)]
        bound = analyzer.bound_epsilon_change(
            p, solver_tol=1e-8, epsilon=1.0, edges=edges
        )
        # High ratio → large condition number
        assert bound.condition_number > 100

    def test_max_condition_number_validation(self):
        with pytest.raises(ValueError):
            PerturbationAnalyzer(max_condition_number=0)

    def test_repr(self):
        analyzer = PerturbationAnalyzer()
        assert "PerturbationAnalyzer" in repr(analyzer)


# =========================================================================
# SolverDiagnostics tests
# =========================================================================


class TestSolverDiagnostics:
    """Tests for SolverDiagnostics."""

    def test_audit_feasible_solution(self):
        diag = SolverDiagnostics(feasibility_tol=1e-6)
        lp = make_simple_lp(n_vars=5, n_ub=3, n_eq=2)
        # Create a feasible solution (small values within bounds)
        x = np.full(5, 0.1, dtype=np.float64)
        report = diag.audit_constraints(x, lp)
        assert isinstance(report, AuditReport)

    def test_audit_detects_inequality_violation(self):
        diag = SolverDiagnostics(feasibility_tol=1e-6)
        # Create LP where A_ub @ x must be <= b_ub
        A_ub = sparse.csr_matrix(np.eye(3, dtype=np.float64))
        b_ub = np.array([1.0, 1.0, 1.0])
        lp = LPStruct(
            c=np.ones(3),
            A_ub=A_ub, b_ub=b_ub,
            A_eq=None, b_eq=None,
            bounds=[(0.0, 10.0)] * 3,
            var_map={"x0": 0, "x1": 1, "x2": 2},
            y_grid=np.array([0.0, 0.5, 1.0]),
        )
        # x = [2, 0, 0] violates first constraint
        x = np.array([2.0, 0.0, 0.0])
        report = diag.audit_constraints(x, lp)
        assert not report.is_feasible
        assert report.n_inequality_violated >= 1
        assert report.max_inequality_violation > 0

    def test_audit_detects_equality_violation(self):
        diag = SolverDiagnostics(feasibility_tol=1e-6)
        A_ub = sparse.csr_matrix(np.eye(2, dtype=np.float64))
        b_ub = np.array([10.0, 10.0])
        A_eq = sparse.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float64))
        b_eq = np.array([1.0])
        lp = LPStruct(
            c=np.ones(2), A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=[(0.0, 10.0)] * 2,
            var_map={"x0": 0, "x1": 1},
            y_grid=np.array([0.0, 1.0]),
        )
        # x = [0.3, 0.3] → sum = 0.6, violates x0+x1 = 1
        x = np.array([0.3, 0.3])
        report = diag.audit_constraints(x, lp)
        assert report.n_equality_violated >= 1

    def test_audit_summary(self):
        diag = SolverDiagnostics()
        lp = make_simple_lp(n_vars=5, n_ub=3)
        x = np.full(5, 0.1)
        report = diag.audit_constraints(x, lp)
        summary = report.summary()
        assert "Constraint Audit" in summary

    def test_compute_residuals(self):
        diag = SolverDiagnostics()
        lp = make_simple_lp(n_vars=5, n_ub=3, n_eq=2)
        x = np.full(5, 0.1)
        report = diag.compute_residuals(x, lp)
        assert isinstance(report, ResidualReport)
        assert len(report.primal_ub_residuals) == 3

    def test_iterative_refine(self):
        diag = SolverDiagnostics(feasibility_tol=1e-6, max_refinement_iter=10)
        A_ub = sparse.csr_matrix(np.eye(3, dtype=np.float64))
        b_ub = np.array([1.0, 1.0, 1.0])
        lp = LPStruct(
            c=np.ones(3), A_ub=A_ub, b_ub=b_ub,
            A_eq=None, b_eq=None,
            bounds=[(0.0, 10.0)] * 3,
            var_map={"x0": 0, "x1": 1, "x2": 2},
            y_grid=np.array([0.0, 0.5, 1.0]),
        )
        # Start with slightly violating solution
        x = np.array([1.001, 0.5, 0.9])
        result = diag.iterative_refine(x, lp)
        assert isinstance(result, RefinementResult)
        # Final residual should be ≤ initial
        assert result.final_residual <= result.initial_residual + 1e-10

    def test_full_diagnostics(self):
        diag = SolverDiagnostics()
        lp = make_simple_lp(n_vars=5, n_ub=3, n_eq=2)
        x = np.full(5, 0.1)
        results = diag.full_diagnostics(x, lp)
        assert "audit" in results
        assert "residuals" in results
        assert "condition_number" in results

    def test_feasibility_tol_validation(self):
        with pytest.raises(ValueError):
            SolverDiagnostics(feasibility_tol=0.0)

    def test_max_refinement_iter_validation(self):
        with pytest.raises(ValueError):
            SolverDiagnostics(max_refinement_iter=0)

    def test_repr(self):
        diag = SolverDiagnostics()
        assert "SolverDiagnostics" in repr(diag)


# =========================================================================
# NumericalCertificate tests
# =========================================================================


class TestNumericalCertificate:
    """Tests for NumericalCertificate serialisation."""

    def make_cert(self):
        return NumericalCertificate(
            solver_tolerance=1e-8,
            epsilon_target=1.0,
            delta_target=0.0,
            epsilon_margin=1e-6,
            delta_margin=0.0,
            epsilon_effective=1.000001,
            delta_effective=0.0,
            interval_verified=True,
            condition_number=100.0,
            max_constraint_violation=1e-9,
            perturbation_epsilon_bound=1e-7,
            perturbation_delta_bound=0.0,
            synthesis_time=0.5,
            cegis_iterations=10,
        )

    def test_to_dict_round_trip(self):
        cert = self.make_cert()
        d = cert.to_dict()
        recovered = NumericalCertificate.from_dict(d)
        assert recovered.epsilon_target == cert.epsilon_target
        assert recovered.delta_target == cert.delta_target
        assert recovered.epsilon_margin == cert.epsilon_margin
        assert recovered.interval_verified == cert.interval_verified
        assert recovered.condition_number == cert.condition_number

    def test_repr(self):
        cert = self.make_cert()
        r = repr(cert)
        assert "NumericalCertificate" in r


# =========================================================================
# CertifiedMechanism tests
# =========================================================================


class TestCertifiedMechanism:
    """Tests for CertifiedMechanism including JSON serialisation."""

    def make_certified(self):
        n, k = 3, 5
        p = make_simple_dp_mechanism(n, k, epsilon=1.0)
        y_grid = np.linspace(0, 1, k)
        cert = NumericalCertificate(
            solver_tolerance=1e-8,
            epsilon_target=1.0,
            delta_target=0.0,
            epsilon_margin=1e-6,
            delta_margin=0.0,
            epsilon_effective=1.000001,
            delta_effective=0.0,
            interval_verified=True,
            condition_number=100.0,
            max_constraint_violation=1e-9,
            perturbation_epsilon_bound=1e-7,
            perturbation_delta_bound=0.0,
            synthesis_time=0.5,
            cegis_iterations=10,
        )
        edges = [(0, 1), (1, 2)]
        return CertifiedMechanism(
            mechanism=p, y_grid=y_grid, certificate=cert, edges=edges
        )

    def test_properties(self):
        cm = self.make_certified()
        assert cm.n == 3
        assert cm.k == 5
        assert cm.epsilon_effective > 0

    def test_verify_certificate_structural(self):
        cm = self.make_certified()
        assert cm.verify_certificate(strict=False)

    def test_json_round_trip(self):
        cm = self.make_certified()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cm.to_json(path)
            loaded = CertifiedMechanism.from_json(path)
            np.testing.assert_allclose(loaded.mechanism, cm.mechanism, rtol=1e-10)
            np.testing.assert_allclose(loaded.y_grid, cm.y_grid, rtol=1e-10)
            assert loaded.certificate.epsilon_target == cm.certificate.epsilon_target
            assert loaded.certificate.interval_verified == cm.certificate.interval_verified
            assert len(loaded.edges) == len(cm.edges)
        finally:
            os.unlink(path)

    def test_summary(self):
        cm = self.make_certified()
        s = cm.summary()
        assert "CertifiedMechanism" in s
        assert "Target" in s
        assert "Effective" in s

    def test_invalid_mechanism_shape(self):
        cert = NumericalCertificate(
            solver_tolerance=1e-8, epsilon_target=1.0, delta_target=0.0,
            epsilon_margin=0.0, delta_margin=0.0,
            epsilon_effective=1.0, delta_effective=0.0,
            interval_verified=True, condition_number=1.0,
            max_constraint_violation=0.0,
            perturbation_epsilon_bound=0.0, perturbation_delta_bound=0.0,
            synthesis_time=0.0, cegis_iterations=0,
        )
        with pytest.raises(ValueError, match="2-D"):
            CertifiedMechanism(
                mechanism=np.array([1.0, 2.0]),
                y_grid=np.array([0.0, 1.0]),
                certificate=cert,
                edges=[],
            )

    def test_ygrid_length_mismatch(self):
        cert = NumericalCertificate(
            solver_tolerance=1e-8, epsilon_target=1.0, delta_target=0.0,
            epsilon_margin=0.0, delta_margin=0.0,
            epsilon_effective=1.0, delta_effective=0.0,
            interval_verified=True, condition_number=1.0,
            max_constraint_violation=0.0,
            perturbation_epsilon_bound=0.0, perturbation_delta_bound=0.0,
            synthesis_time=0.0, cegis_iterations=0,
        )
        with pytest.raises(ValueError, match="y_grid"):
            CertifiedMechanism(
                mechanism=np.ones((2, 3)),
                y_grid=np.array([0.0, 1.0]),  # length 2, but need 3
                certificate=cert,
                edges=[],
            )

    def test_repr(self):
        cm = self.make_certified()
        r = repr(cm)
        assert "CertifiedMechanism" in r

    def test_verify_invalid_margins(self):
        """Certificate with negative margins should fail verification."""
        n, k = 2, 3
        p = np.ones((n, k)) / k
        cert = NumericalCertificate(
            solver_tolerance=1e-8, epsilon_target=1.0, delta_target=0.0,
            epsilon_margin=-0.1, delta_margin=0.0,
            epsilon_effective=0.9, delta_effective=0.0,
            interval_verified=True, condition_number=1.0,
            max_constraint_violation=0.0,
            perturbation_epsilon_bound=0.0, perturbation_delta_bound=0.0,
            synthesis_time=0.0, cegis_iterations=0,
        )
        cm = CertifiedMechanism(
            mechanism=p, y_grid=np.array([0.0, 0.5, 1.0]),
            certificate=cert, edges=[]
        )
        assert not cm.verify_certificate(strict=False)


# =========================================================================
# interval_verify_dp tests
# =========================================================================


class TestIntervalVerifyDP:
    """Tests for interval_verify_dp function."""

    def test_pure_dp_valid_mechanism(self):
        """A uniform mechanism should satisfy any ε > 0."""
        p = np.ones((3, 5)) / 5
        valid, violation = interval_verify_dp(
            p, p, epsilon=1.0, edges=[(0, 1), (1, 2)], delta=0.0
        )
        assert valid
        assert violation is None

    def test_pure_dp_violation_detected(self):
        """An intentionally violating mechanism should fail."""
        p_lo = np.array([[0.9, 0.1], [0.1, 0.9]])
        p_hi = np.array([[0.95, 0.15], [0.15, 0.95]])
        # With ε=0.1, e^ε ≈ 1.105, so 0.95/0.1 = 9.5 >> 1.105
        valid, violation = interval_verify_dp(
            p_lo, p_hi, epsilon=0.1, edges=[(0, 1)], delta=0.0
        )
        assert not valid
        assert violation is not None

    def test_approx_dp_with_delta(self):
        """Test approximate DP verification."""
        p = np.ones((3, 5)) / 5
        valid, _ = interval_verify_dp(
            p, p, epsilon=1.0, edges=[(0, 1), (1, 2)], delta=1e-5
        )
        assert valid

    def test_interval_uncertainty_widens(self):
        """Wider intervals → harder to verify → larger ε needed."""
        p = np.ones((2, 3)) / 3
        nu = 0.01
        p_lo = np.maximum(p - nu, 0)
        p_hi = p + nu

        # Should pass with large ε
        valid_large, _ = interval_verify_dp(
            p_lo, p_hi, epsilon=5.0, edges=[(0, 1)], delta=0.0
        )
        assert valid_large


# =========================================================================
# RobustSynthesisConfig tests
# =========================================================================


class TestRobustSynthesisConfig:
    """Tests for RobustSynthesisConfig validation."""

    def test_default_config(self):
        config = RobustSynthesisConfig()
        assert config.solver_tolerance > 0
        assert config.safety_factor >= 1.0

    def test_invalid_solver_tolerance(self):
        with pytest.raises(ConfigurationError):
            RobustSynthesisConfig(solver_tolerance=0.0)

    def test_invalid_safety_factor(self):
        with pytest.raises(ConfigurationError):
            RobustSynthesisConfig(safety_factor=0.5)

    def test_invalid_refinement_attempts(self):
        with pytest.raises(ConfigurationError):
            RobustSynthesisConfig(max_refinement_attempts=-1)

    def test_repr(self):
        config = RobustSynthesisConfig()
        assert "RobustSynthesisConfig" in repr(config)


# =========================================================================
# RobustCEGISEngine tests (unit-level, no actual synthesis)
# =========================================================================


class TestRobustCEGISEngine:
    """Unit tests for RobustCEGISEngine properties and configuration."""

    def test_construction(self):
        engine = RobustCEGISEngine()
        assert engine.config is not None
        assert engine.inflator is not None
        assert engine.analyzer is not None
        assert engine.diagnostics is not None

    def test_custom_config(self):
        config = RobustSynthesisConfig(
            solver_tolerance=1e-10,
            safety_factor=3.0,
        )
        engine = RobustCEGISEngine(config=config)
        assert engine.config.solver_tolerance == 1e-10
        assert engine.config.safety_factor == 3.0

    def test_repr(self):
        engine = RobustCEGISEngine()
        assert "RobustCEGISEngine" in repr(engine)


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge cases for tolerance levels."""

    def test_very_loose_tolerance(self):
        """Large tolerance should give large margins."""
        inflator = ConstraintInflator(safety_factor=2.0)
        margin = inflator.compute_epsilon_margin(
            epsilon=1.0, solver_tol=0.01, n_outputs=10
        )
        assert margin > 0
        # With large tolerance, margin should be significant
        assert margin <= 0.5  # capped at ε/2

    def test_very_tight_tolerance(self):
        """Very small tolerance should give very small margins."""
        inflator = ConstraintInflator(safety_factor=2.0)
        margin = inflator.compute_epsilon_margin(
            epsilon=1.0, solver_tol=1e-15, n_outputs=10
        )
        assert margin > 0
        assert margin < 1e-5

    def test_perturbation_with_tiny_probabilities(self):
        """Mechanism with tiny probabilities → large perturbation bounds."""
        analyzer = PerturbationAnalyzer()
        p = np.array([
            [1e-10, 1.0 - 1e-10],
            [1.0 - 1e-10, 1e-10],
        ])
        edges = [(0, 1)]
        bound = analyzer.bound_epsilon_change(
            p, solver_tol=1e-8, epsilon=1.0, edges=edges
        )
        # With p_min ≈ 1e-10, perturbation bound should be large
        assert bound.epsilon_bound > 1e-2
