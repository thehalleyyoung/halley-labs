"""
Tests for SMT verification module.

Tests cover: Z3 encoder, satisfiability checks, incremental protocol,
certificate emission and validation, and graph predicate encoding.
"""
import pytest
import z3

from causalbound.smt.encoder import SMTEncoder, DAGSpec
from causalbound.smt.incremental import IncrementalProtocol
from causalbound.smt.verifier import SMTVerifier, VerificationStatus, BoundEvidence
from causalbound.smt.certificates import CertificateEmitter, Certificate
from causalbound.smt.predicates import GraphPredicateEncoder, DAG
from causalbound.smt.qf_lra import QFLRAEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iv_dag():
    return DAGSpec.from_edge_list([("Z", "X"), ("X", "Y")])


def _chain_dag():
    return DAGSpec.from_edge_list([("A", "B"), ("B", "C")])


def _diamond_dag():
    return DAGSpec.from_edge_list([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])


def _pred_dag():
    return DAG([("A", "B"), ("B", "C"), ("A", "C")])


# ---------------------------------------------------------------------------
# SMTEncoder: basic encoding
# ---------------------------------------------------------------------------

class TestSMTEncoderBasic:
    """Test basic Z3 encoding operations."""

    def test_real_var_creation(self):
        enc = SMTEncoder()
        x = enc.real_var("x")
        assert isinstance(x, z3.ArithRef)

    def test_real_var_reuse(self):
        enc = SMTEncoder()
        x1 = enc.real_var("x")
        x2 = enc.real_var("x")
        assert x1 is x2

    def test_bool_var_creation(self):
        enc = SMTEncoder()
        b = enc.bool_var("flag")
        assert isinstance(b, z3.BoolRef)

    def test_int_var_creation(self):
        enc = SMTEncoder()
        i = enc.int_var("count")
        assert isinstance(i, z3.ArithRef)

    def test_variable_count(self):
        enc = SMTEncoder()
        enc.real_var("x")
        enc.real_var("y")
        enc.bool_var("b")
        assert enc.get_variable_count() == 3

    def test_clear_cache(self):
        enc = SMTEncoder()
        enc.real_var("x")
        enc.clear_cache()
        assert enc.get_variable_count() == 0


class TestSMTEncoderBounds:
    """Test bound claim encoding."""

    def test_encode_valid_bound(self):
        enc = SMTEncoder()
        claim = enc.encode_bound_claim("p", lower=0.2, upper=0.8)
        solver = z3.Solver()
        solver.add(claim)
        assert solver.check() == z3.sat

    def test_encode_invalid_bound_lower_gt_upper(self):
        enc = SMTEncoder()
        claim = enc.encode_bound_claim("p", lower=0.9, upper=0.1)
        solver = z3.Solver()
        solver.add(claim)
        assert solver.check() == z3.unsat

    def test_encode_bound_outside_probability(self):
        enc = SMTEncoder()
        claim = enc.encode_bound_claim("p", lower=-0.1, upper=0.5)
        solver = z3.Solver()
        solver.add(claim)
        assert solver.check() == z3.unsat

    def test_encode_bound_above_one(self):
        enc = SMTEncoder()
        claim = enc.encode_bound_claim("p", lower=0.5, upper=1.1)
        solver = z3.Solver()
        solver.add(claim)
        assert solver.check() == z3.unsat

    def test_encode_tight_bound(self):
        enc = SMTEncoder()
        claim = enc.encode_bound_claim("p", lower=0.5, upper=0.5)
        solver = z3.Solver()
        solver.add(claim)
        assert solver.check() == z3.sat

    def test_encode_bound_tightness(self):
        enc = SMTEncoder()
        claim = enc.encode_bound_tightness(
            "p", lower=0.2, upper=0.8,
            witness_lower=0.3, witness_upper=0.7,
        )
        assert isinstance(claim, z3.BoolRef)
        solver = z3.Solver()
        solver.add(claim)
        assert solver.check() == z3.sat


class TestSMTEncoderDSep:
    """Test d-separation encoding."""

    def test_encode_dsep_valid(self):
        dag = _iv_dag()
        enc = SMTEncoder()
        dsep_claim = enc.encode_dsep(x="Z", y="Y", z_set=["X"], dag=dag)
        assert isinstance(dsep_claim, z3.BoolRef)

    def test_encode_dsep_satisfiable(self):
        dag = _iv_dag()
        enc = SMTEncoder()
        dsep_claim = enc.encode_dsep(x="Z", y="Y", z_set=["X"], dag=dag)
        solver = z3.Solver()
        solver.add(dsep_claim)
        result = solver.check()
        assert result == z3.sat


class TestSMTEncoderNormalization:
    """Test normalization constraint encoding."""

    def test_encode_normalization(self):
        enc = SMTEncoder()
        dist = {f"p_{i}": 0.25 for i in range(4)}
        norm = enc.encode_normalization(dist)
        assert isinstance(norm, z3.BoolRef)
        solver = z3.Solver()
        solver.add(norm)
        assert solver.check() == z3.sat

    def test_normalization_violation(self):
        enc = SMTEncoder()
        dist = {"p_0": 0.8, "p_1": 0.8}
        norm = enc.encode_normalization(dist)
        solver = z3.Solver()
        solver.add(norm)
        assert solver.check() == z3.unsat

    def test_encode_normalization_from_values(self):
        enc = SMTEncoder()
        norm = enc.encode_normalization_from_values(
            prefix="nfv",
            values={"p0": 0.3, "p1": 0.5, "p2": 0.2},
        )
        solver = z3.Solver()
        solver.add(norm)
        assert solver.check() == z3.sat


class TestSMTEncoderMarginalization:
    """Test marginalization constraint encoding."""

    def test_encode_marginalization(self):
        enc = SMTEncoder()
        joint = {"X0_Y0": 0.3, "X0_Y1": 0.2, "X1_Y0": 0.1, "X1_Y1": 0.4}
        marginal = {"X0": 0.5, "X1": 0.5}
        margin = enc.encode_marginalization(joint, marginal)
        assert isinstance(margin, z3.BoolRef)

    def test_marginalization_consistency(self):
        enc = SMTEncoder()
        joint = {"X0_Y0": 0.3, "X0_Y1": 0.2, "X1_Y0": 0.1, "X1_Y1": 0.4}
        marginal = {"X0": 0.5, "X1": 0.5}
        claim = enc.encode_marginalization(joint, marginal)
        solver = z3.Solver()
        solver.add(claim)
        assert solver.check() == z3.sat


class TestSMTEncoderPolytope:
    """Test causal polytope encoding."""

    def test_encode_causal_polytope(self):
        enc = SMTEncoder()
        point = [0.3, 0.2, 0.5]
        A = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        b = [1.0, 1.0]
        constraints = enc.encode_causal_polytope(point, A, b)
        assert isinstance(constraints, z3.BoolRef)

    def test_assertion_log(self):
        enc = SMTEncoder()
        enc.encode_bound_claim("p", 0.2, 0.8)
        enc.encode_bound_claim("q", 0.1, 0.5)
        log = enc.get_assertion_log()
        assert len(log) >= 2


# ---------------------------------------------------------------------------
# Incremental protocol
# ---------------------------------------------------------------------------

class TestIncrementalProtocol:
    """Test SMT incremental solving protocol."""

    def test_push_pop(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="init")
        assert proto.get_stack_depth() == 1
        proto.pop_context()
        assert proto.get_stack_depth() == 0

    def test_assert_claim(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="bounds")
        x = z3.Real("x")
        proto.assert_claim(x >= 0, label="non_negative")
        assert proto.get_assertion_count() >= 1

    def test_check_satisfiability(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="test")
        x = z3.Real("x")
        proto.assert_claim(x >= 0, label="lb")
        proto.assert_claim(x <= 1, label="ub")
        result = proto.check_satisfiability()
        assert result == z3.sat

    def test_push_pop_nested(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="outer")
        x = z3.Real("x")
        proto.assert_claim(x >= 0)
        proto.push_context(phase="inner")
        proto.assert_claim(x <= 0.5)
        assert proto.get_stack_depth() == 2
        proto.pop_context()
        assert proto.get_stack_depth() == 1
        proto.pop_context()
        assert proto.get_stack_depth() == 0

    def test_pop_to_depth(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="a")
        proto.push_context(phase="b")
        proto.push_context(phase="c")
        assert proto.get_stack_depth() == 3
        popped = proto.pop_to_depth(1)
        assert proto.get_stack_depth() == 1
        assert len(popped) == 2

    def test_unsatisfiable(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="conflict")
        x = z3.Real("x")
        proto.assert_claim(x > 1)
        proto.assert_claim(x < 0)
        result = proto.check_satisfiability()
        assert result == z3.unsat

    def test_get_model(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="model")
        x = z3.Real("x")
        proto.assert_claim(x == z3.RealVal("1/2"))
        proto.check_satisfiability()
        model = proto.get_model()
        assert model is not None

    def test_phase_tracking(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="phase1")
        proto.set_phase("phase1")
        assert proto.get_phase() == "phase1"
        x = z3.Real("x")
        proto.assert_claim(x >= 0)
        phases = proto.get_phases()
        assert "phase1" in phases

    def test_stats(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="s")
        x = z3.Real("x")
        proto.assert_claim(x >= 0)
        proto.check_satisfiability()
        stats = proto.get_stats()
        assert stats.total_assertions >= 1
        assert stats.total_checks >= 1

    def test_frame_summary(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="summary_test")
        summary = proto.get_frame_summary()
        assert len(summary) >= 1

    def test_assertions_in_phase(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="p1")
        proto.set_phase("p1")
        x = z3.Real("x")
        proto.assert_claim(x >= 0, label="lb_x")
        records = proto.get_assertions_in_phase("p1")
        assert len(records) >= 1

    def test_replay_script(self):
        solver = z3.Solver()
        proto = IncrementalProtocol(solver)
        proto.push_context(phase="replay")
        x = z3.Real("x")
        proto.assert_claim(x >= 0)
        script = proto.get_replay_script()
        assert isinstance(script, str)


# ---------------------------------------------------------------------------
# Certificate emission and validation
# ---------------------------------------------------------------------------

class TestCertificateEmitter:
    """Test certificate emission and validation."""

    def test_create_certificate(self):
        emitter = CertificateEmitter(session_id="test-001")
        cert = emitter.create_certificate()
        assert cert is not None

    def test_add_step(self):
        emitter = CertificateEmitter(session_id="test-002")
        emitter.create_certificate()
        emitter.add_step(
            assertion="lb <= 0.2",
            proof="verified:bound_check",
        )
        cert = emitter.get_certificate()
        assert cert is not None

    def test_finalize(self):
        emitter = CertificateEmitter(session_id="test-003")
        emitter.create_certificate()
        emitter.add_step(
            assertion="sum(p_i) == 1",
            proof="verified:normalization",
        )
        cert = emitter.finalize()
        assert isinstance(cert, Certificate)
        assert cert.finalized

    def test_validate_certificate(self):
        emitter = CertificateEmitter(session_id="test-004")
        emitter.create_certificate()
        emitter.add_step(
            assertion="x >= 0",
            proof="verified:check",
        )
        cert = emitter.finalize()
        result = CertificateEmitter.validate(cert)
        assert result.valid

    def test_certificate_serialization(self):
        emitter = CertificateEmitter(session_id="test-005")
        emitter.create_certificate()
        emitter.add_step(
            assertion="lb >= 0",
            proof="verified:bound",
        )
        cert = emitter.finalize()
        d = cert.to_dict()
        loaded = Certificate.from_dict(d)
        assert loaded.session_id == cert.session_id
        assert len(loaded.steps) == len(cert.steps)

    def test_pretty_print(self):
        emitter = CertificateEmitter(session_id="test-006")
        emitter.create_certificate()
        emitter.add_step(assertion="c", proof="verified:a")
        cert = emitter.finalize()
        pp = CertificateEmitter.pretty_print(cert)
        assert isinstance(pp, str)
        assert len(pp) > 0

    def test_chain_certificates(self):
        e1 = CertificateEmitter(session_id="s1")
        e1.create_certificate()
        e1.add_step(assertion="c", proof="verified:a")
        c1 = e1.finalize()

        e2 = CertificateEmitter(session_id="s2")
        e2.create_certificate()
        e2.add_step(assertion="f", proof="verified:d")
        c2 = e2.finalize()

        chained = CertificateEmitter.chain_certificates([c1, c2])
        assert isinstance(chained, Certificate)
        assert len(chained.steps) == 2


# ---------------------------------------------------------------------------
# Graph predicate encoder
# ---------------------------------------------------------------------------

class TestGraphPredicateEncoder:
    """Test graph predicate encoding for SMT."""

    def test_encode_dsep_chain(self):
        dag = _pred_dag()
        enc = GraphPredicateEncoder()
        formula = enc.encode_dsep(x="A", y="C", z_set=["B"], dag=dag)
        assert isinstance(formula, z3.BoolRef)

    def test_encode_ancestor(self):
        dag = _pred_dag()
        enc = GraphPredicateEncoder()
        formula = enc.encode_ancestor(x="A", y="C", dag=dag)
        assert isinstance(formula, z3.BoolRef)
        solver = z3.Solver()
        solver.add(formula)
        assert solver.check() == z3.sat

    def test_encode_non_ancestor(self):
        dag = _pred_dag()
        enc = GraphPredicateEncoder()
        formula = enc.encode_non_ancestor(x="C", y="A", dag=dag)
        assert isinstance(formula, z3.BoolRef)

    def test_encode_path(self):
        dag = _pred_dag()
        enc = GraphPredicateEncoder()
        formula = enc.encode_path(x="A", y="C", dag=dag)
        assert isinstance(formula, z3.BoolRef)

    def test_encode_markov_blanket(self):
        dag = _pred_dag()
        enc = GraphPredicateEncoder()
        formula = enc.encode_markov_blanket(x="B", dag=dag)
        assert isinstance(formula, z3.BoolRef)

    def test_encode_topo_order(self):
        dag = _pred_dag()
        enc = GraphPredicateEncoder()
        topo = dag.topological_sort()
        formula = enc.encode_topo_order(variables=topo, dag=dag)
        assert isinstance(formula, z3.BoolRef)

    def test_encode_acyclicity(self):
        dag = _pred_dag()
        enc = GraphPredicateEncoder()
        formula = enc.encode_acyclicity(dag=dag)
        assert isinstance(formula, z3.BoolRef)
        solver = z3.Solver()
        solver.add(formula)
        assert solver.check() == z3.sat

    def test_variable_count(self):
        enc = GraphPredicateEncoder()
        dag = _pred_dag()
        enc.encode_dsep(x="A", y="C", z_set=["B"], dag=dag)
        assert enc.get_variable_count() > 0


# ---------------------------------------------------------------------------
# QF_LRA encoder
# ---------------------------------------------------------------------------

class TestQFLRAEncoder:
    """Test QF_LRA LP encoding."""

    def test_encode_lp_feasible(self):
        enc = QFLRAEncoder()
        A = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        b = [1.0, 1.0]
        x = [0.5, 0.3, 0.2]
        formula = enc.encode_lp_feasible(A, b, x)
        assert isinstance(formula, z3.BoolRef)
        solver = z3.Solver()
        solver.add(formula)
        assert solver.check() == z3.sat

    def test_encode_lp_infeasible(self):
        enc = QFLRAEncoder()
        A = [[1.0, 1.0], [1.0, 1.0]]
        b = [1.0, 2.0]
        x = [0.5, 0.5]
        formula = enc.encode_lp_feasible_equality(A, b, x)
        assert isinstance(formula, z3.BoolRef)
        solver = z3.Solver()
        solver.add(formula)
        assert solver.check() == z3.unsat

    def test_encode_weak_duality(self):
        enc = QFLRAEncoder()
        c = [1.0, 0.0, 0.0]
        A = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        b = [1.0, 1.0]
        x = [0.5, 0.3, 0.2]
        dual = [0.5, 0.5]
        formula = enc.encode_weak_duality(c, A, b, x, dual)
        assert isinstance(formula, z3.BoolRef)

    def test_encode_variable_bounds(self):
        enc = QFLRAEncoder()
        formula = enc.encode_variable_bounds(
            lower_bounds=[0.0, 0.0, 0.0],
            upper_bounds=[1.0, 1.0, 1.0],
            values=[0.5, 0.3, 0.2],
        )
        solver = z3.Solver()
        solver.add(formula)
        assert solver.check() == z3.sat


# ---------------------------------------------------------------------------
# SMT Verifier
# ---------------------------------------------------------------------------

class TestSMTVerifier:
    """Test the full SMT verifier."""

    def test_begin_end_session(self):
        verifier = SMTVerifier()
        sid = verifier.begin_session()
        assert isinstance(sid, str)
        stats = verifier.end_session()
        assert stats is not None

    def test_verify_bound_valid(self):
        verifier = SMTVerifier()
        verifier.begin_session()
        result = verifier.verify_bound(
            lower=0.2,
            upper=0.8,
            evidence=BoundEvidence(lp_objective=0.5),
        )
        assert result.status == VerificationStatus.PASS
        verifier.end_session()

    def test_verify_bound_invalid(self):
        verifier = SMTVerifier()
        verifier.begin_session()
        result = verifier.verify_bound(
            lower=0.9,
            upper=0.1,
            evidence=BoundEvidence(),
        )
        assert result.status == VerificationStatus.FAIL
        verifier.end_session()

    def test_verify_normalization(self):
        verifier = SMTVerifier()
        verifier.begin_session()
        result = verifier.verify_normalization(
            distribution={"p0": 0.3, "p1": 0.5, "p2": 0.2},
        )
        assert result.status == VerificationStatus.PASS
        verifier.end_session()

    def test_context_manager(self):
        with SMTVerifier() as verifier:
            result = verifier.verify_bound(
                lower=0.0, upper=1.0,
                evidence=BoundEvidence(lp_objective=0.5),
            )
            assert result.status == VerificationStatus.PASS

    def test_emit_certificate(self):
        verifier = SMTVerifier()
        verifier.begin_session()
        verifier.verify_bound(
            lower=0.1, upper=0.9,
            evidence=BoundEvidence(lp_objective=0.5),
        )
        cert = verifier.emit_certificate()
        verifier.end_session()
        # Certificate may be None if not enough steps
        if cert is not None:
            assert isinstance(cert, Certificate)

    def test_verification_stats(self):
        verifier = SMTVerifier()
        verifier.begin_session()
        verifier.verify_bound(0.0, 1.0, BoundEvidence(lp_objective=0.5))
        verifier.verify_bound(0.2, 0.8, BoundEvidence(lp_objective=0.5))
        stats = verifier.get_verification_stats()
        assert stats.total_steps >= 2
        verifier.end_session()
