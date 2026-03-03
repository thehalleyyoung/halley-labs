"""
Comprehensive tests for dp_forge.dual_certificates module.

Tests cover dual certificate extraction from solved LPs, complementary
slackness verification, duality gap analysis, gap tracking, and
human-readable optimality proof construction.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest
from scipy import sparse

from dp_forge.dual_certificates import (
    DualCertificateExtractor,
    DualCertificate,
    ComplementarySlacknessReport,
    OptimalityProof,
    GapTracker,
    GapRecord,
)
from dp_forge.types import LPStruct, OptimalityCertificate
from dp_forge.exceptions import NumericalInstabilityError


# =========================================================================
# Fixtures
# =========================================================================


def _make_simple_lp_struct():
    """Build a small LPStruct for testing.

    min x0 + x1
    s.t. x0 + x1 <= 2   (inequality)
         x0 + x1  = 1    (equality, simplex-like)
         x0, x1 >= 0
    """
    c = np.array([1.0, 1.0])
    A_ub = sparse.csr_matrix(np.array([[1.0, 1.0]]))
    b_ub = np.array([2.0])
    A_eq = sparse.csr_matrix(np.array([[1.0, 1.0]]))
    b_eq = np.array([1.0])
    bounds = [(0.0, None), (0.0, None)]
    var_map = {(0, 0): 0, (0, 1): 1}
    y_grid = np.array([0.0, 1.0])
    return LPStruct(
        c=c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, var_map=var_map, y_grid=y_grid,
    )


class _MockSolverResult:
    """Mock scipy.optimize.linprog result."""

    def __init__(self, x, fun, ineqlin_marginals, eqlin_marginals, success=True):
        self.x = np.asarray(x, dtype=np.float64)
        self.fun = float(fun)
        self.success = success
        self.status = 0
        self.message = "Optimization terminated successfully."
        # Dual variables (scipy uses .ineqlin.marginals, .eqlin.marginals)
        self.ineqlin = type("obj", (object,), {"marginals": np.asarray(ineqlin_marginals, dtype=np.float64)})()
        self.eqlin = type("obj", (object,), {"marginals": np.asarray(eqlin_marginals, dtype=np.float64)})()


@pytest.fixture
def lp_struct():
    return _make_simple_lp_struct()


@pytest.fixture
def mock_optimal():
    """Mock optimal LP result: x = (0.5, 0.5), obj = 1.0."""
    return _MockSolverResult(
        x=[0.5, 0.5],
        fun=1.0,
        ineqlin_marginals=[0.0],  # Inactive inequality (slack = 1.0)
        eqlin_marginals=[1.0],    # Equality dual
    )


@pytest.fixture
def extractor():
    return DualCertificateExtractor(tol=1e-6)


# =========================================================================
# Section 1: Dual Certificate Extraction
# =========================================================================


class TestDualCertificateExtraction:
    """Tests for extracting dual certificates from solver output."""

    def test_extract_returns_certificate(self, extractor, mock_optimal, lp_struct):
        """extract() returns a DualCertificate."""
        cert = extractor.extract(mock_optimal, lp_struct)
        assert isinstance(cert, DualCertificate)

    def test_primal_solution_shape(self, extractor, mock_optimal, lp_struct):
        """Primal solution has correct dimension."""
        cert = extractor.extract(mock_optimal, lp_struct)
        assert len(cert.primal_solution) == lp_struct.n_vars

    def test_primal_objective(self, extractor, mock_optimal, lp_struct):
        """Primal objective matches solver output."""
        cert = extractor.extract(mock_optimal, lp_struct)
        assert abs(cert.primal_obj - 1.0) < 1e-10

    def test_dual_ub_shape(self, extractor, mock_optimal, lp_struct):
        """Inequality dual variables have correct shape."""
        cert = extractor.extract(mock_optimal, lp_struct)
        assert len(cert.dual_ub) == lp_struct.n_ub

    def test_dual_eq_shape(self, extractor, mock_optimal, lp_struct):
        """Equality dual variables have correct shape."""
        cert = extractor.extract(mock_optimal, lp_struct)
        assert cert.dual_eq is not None
        assert len(cert.dual_eq) == lp_struct.n_eq

    def test_duality_gap_small(self, extractor, mock_optimal, lp_struct):
        """Duality gap is small for well-solved LP."""
        cert = extractor.extract(mock_optimal, lp_struct)
        assert cert.duality_gap >= -1e-6
        # For a well-solved LP the gap should be very small
        assert cert.duality_gap < 1.0

    def test_solver_name(self, extractor, mock_optimal, lp_struct):
        """Solver name is recorded."""
        cert = extractor.extract(mock_optimal, lp_struct, solver_name="TestSolver")
        assert cert.solver_name == "TestSolver"

    def test_n_vars_recorded(self, extractor, mock_optimal, lp_struct):
        """Number of variables is recorded."""
        cert = extractor.extract(mock_optimal, lp_struct)
        assert cert.n_vars == 2

    def test_timestamp_set(self, extractor, mock_optimal, lp_struct):
        """Timestamp is set on extraction."""
        cert = extractor.extract(mock_optimal, lp_struct)
        assert cert.timestamp is not None


# =========================================================================
# Section 2: Complementary Slackness
# =========================================================================


class TestComplementarySlackness:
    """Tests for complementary slackness verification."""

    def test_inactive_constraint_zero_dual(self, extractor, mock_optimal, lp_struct):
        """Inactive constraint has zero dual variable."""
        cert = extractor.extract(mock_optimal, lp_struct)
        report = extractor.verify_complementary_slackness(cert, lp_struct)
        assert isinstance(report, ComplementarySlacknessReport)
        # Inequality constraint is inactive (slack=1), so dual should be 0
        assert report.max_violation < 1e-4

    def test_slackness_report_fields(self, extractor, mock_optimal, lp_struct):
        """ComplementarySlacknessReport has expected fields."""
        cert = extractor.extract(mock_optimal, lp_struct)
        report = extractor.verify_complementary_slackness(cert, lp_struct)
        assert hasattr(report, "max_violation")
        assert hasattr(report, "mean_violation")
        assert hasattr(report, "satisfied")
        assert hasattr(report, "tolerance")

    def test_satisfied_for_optimal(self, extractor, mock_optimal, lp_struct):
        """Complementary slackness satisfied for optimal solution."""
        cert = extractor.extract(mock_optimal, lp_struct)
        report = extractor.verify_complementary_slackness(cert, lp_struct)
        assert report.satisfied


# =========================================================================
# Section 3: Duality Gap Analysis
# =========================================================================


class TestDualityGap:
    """Tests for duality gap computation."""

    def test_gap_dict(self, extractor, mock_optimal, lp_struct):
        """duality_gap() returns dict with expected keys."""
        cert = extractor.extract(mock_optimal, lp_struct)
        gap_info = extractor.duality_gap(cert)
        assert "absolute_gap" in gap_info
        assert "relative_gap" in gap_info

    def test_gap_nonneg(self, extractor, mock_optimal, lp_struct):
        """Duality gap is non-negative for feasible LP."""
        cert = extractor.extract(mock_optimal, lp_struct)
        gap_info = extractor.duality_gap(cert)
        assert gap_info["absolute_gap"] >= -1e-6

    def test_relative_gap_finite(self, extractor, mock_optimal, lp_struct):
        """Relative gap is finite."""
        cert = extractor.extract(mock_optimal, lp_struct)
        gap_info = extractor.duality_gap(cert)
        assert math.isfinite(gap_info["relative_gap"])


# =========================================================================
# Section 4: Gap Tracker
# =========================================================================


class TestGapTracker:
    """Tests for gap tracking across iterations."""

    def test_record_and_history(self):
        """Records are stored and retrievable."""
        tracker = GapTracker(target_gap=1e-6)
        tracker.record(0, absolute_gap=1.0, relative_gap=0.5, primal_obj=2.0, dual_obj=1.0)
        tracker.record(1, absolute_gap=0.5, relative_gap=0.25, primal_obj=1.5, dual_obj=1.0)
        assert tracker.n_records == 2
        assert abs(tracker.latest_gap - 0.5) < 1e-10

    def test_convergence_detection(self):
        """Tracker detects convergence when gap < target."""
        tracker = GapTracker(target_gap=0.1)
        tracker.record(0, absolute_gap=0.05)
        assert tracker.converged

    def test_not_converged(self):
        """Tracker is not converged when gap > target."""
        tracker = GapTracker(target_gap=0.01)
        tracker.record(0, absolute_gap=1.0, relative_gap=0.5, primal_obj=2.0, dual_obj=1.0)
        assert not tracker.converged

    def test_convergence_rate_estimation(self):
        """Convergence rate is estimated from history."""
        tracker = GapTracker(target_gap=1e-10)
        for i in range(20):
            gap = 1.0 / (i + 1)
            tracker.record(i, absolute_gap=gap, relative_gap=gap / 2)
        rate = tracker.convergence_rate()
        if rate is not None:
            assert rate > 0

    def test_predicted_iterations(self):
        """Predicted iterations remaining returns reasonable value."""
        tracker = GapTracker(target_gap=0.01)
        for i in range(10):
            gap = 1.0 / (i + 1)
            tracker.record(i, absolute_gap=gap)
        remaining = tracker.predicted_iterations_remaining()
        if remaining is not None:
            assert remaining >= 0

    def test_gap_decrease_per_iteration(self):
        """gap_decrease_per_iteration returns positive value for decreasing gaps."""
        tracker = GapTracker(target_gap=1e-10)
        for i in range(5):
            tracker.record(i, absolute_gap=1.0 - i * 0.1)
        decrease = tracker.gap_decrease_per_iteration()
        if decrease is not None:
            assert decrease > 0

    def test_stall_detection(self):
        """is_stalled detects no improvement."""
        tracker = GapTracker(target_gap=1e-10)
        for i in range(10):
            tracker.record(i, absolute_gap=1.0)
        assert tracker.is_stalled(window=5, min_improvement=1e-8)

    def test_not_stalled(self):
        """is_stalled returns False when gap is decreasing."""
        tracker = GapTracker(target_gap=1e-10)
        for i in range(10):
            tracker.record(i, absolute_gap=10.0 / (i + 1))
        assert not tracker.is_stalled(window=5, min_improvement=1e-8)

    def test_gaps_array(self):
        """gaps property returns array of recorded gaps."""
        tracker = GapTracker(target_gap=1e-6)
        for i in range(5):
            tracker.record(i, absolute_gap=float(5 - i))
        gaps = tracker.gaps
        assert len(gaps) == 5
        assert gaps[0] == 5.0
        assert gaps[-1] == 1.0

    def test_summary(self):
        """summary() returns non-empty string."""
        tracker = GapTracker(target_gap=1e-6)
        tracker.record(0, absolute_gap=1.0)
        s = tracker.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_record_from_certificate(self, extractor, mock_optimal, lp_struct):
        """record_from_certificate works with DualCertificate."""
        cert = extractor.extract(mock_optimal, lp_struct)
        tracker = GapTracker(target_gap=1e-6)
        tracker.record_from_certificate(0, cert, n_constraints=1)
        assert tracker.n_records == 1


# =========================================================================
# Section 5: Optimality Proof Construction
# =========================================================================


class TestOptimalityProof:
    """Tests for human-readable optimality proof construction."""

    def test_construct_proof(self, extractor, mock_optimal, lp_struct):
        """construct_proof returns OptimalityProof."""
        cert = extractor.extract(mock_optimal, lp_struct)
        proof = extractor.construct_proof(cert, lp_struct)
        assert isinstance(proof, OptimalityProof)

    def test_proof_has_summary(self, extractor, mock_optimal, lp_struct):
        """Proof contains a summary string."""
        cert = extractor.extract(mock_optimal, lp_struct)
        proof = extractor.construct_proof(cert, lp_struct)
        assert isinstance(proof.summary, str)
        assert len(proof.summary) > 0

    def test_proof_has_conclusion(self, extractor, mock_optimal, lp_struct):
        """Proof contains a conclusion."""
        cert = extractor.extract(mock_optimal, lp_struct)
        proof = extractor.construct_proof(cert, lp_struct)
        assert isinstance(proof.conclusion, str)
        assert len(proof.conclusion) > 0

    def test_proof_references_certificate(self, extractor, mock_optimal, lp_struct):
        """Proof references the certificate."""
        cert = extractor.extract(mock_optimal, lp_struct)
        proof = extractor.construct_proof(cert, lp_struct)
        assert proof.certificate is cert

    def test_proof_feasibility_fields(self, extractor, mock_optimal, lp_struct):
        """Proof has primal and dual feasibility fields."""
        cert = extractor.extract(mock_optimal, lp_struct)
        proof = extractor.construct_proof(cert, lp_struct)
        assert hasattr(proof, "primal_feasibility")
        assert hasattr(proof, "dual_feasibility")
        assert hasattr(proof, "strong_duality")
        assert hasattr(proof, "complementary_slackness")


# =========================================================================
# Section 6: Conversion to OptimalityCertificate
# =========================================================================


class TestOptimalityCertificateConversion:
    """Tests for converting DualCertificate to OptimalityCertificate."""

    def test_to_optimality_certificate(self, extractor, mock_optimal, lp_struct):
        """to_optimality_certificate returns valid OptimalityCertificate."""
        cert = extractor.extract(mock_optimal, lp_struct)
        opt_cert = extractor.to_optimality_certificate(cert)
        assert isinstance(opt_cert, OptimalityCertificate)
        assert math.isfinite(opt_cert.primal_obj)
        assert math.isfinite(opt_cert.dual_obj)
        assert opt_cert.duality_gap >= -1e-6
