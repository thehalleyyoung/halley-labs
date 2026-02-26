"""Tests for marace.reporting.tcb_analysis — TCB analysis, soundness argument,
Alethe adapter, and independent checker."""

import json
import os
import tempfile
import textwrap

import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.reporting.tcb_analysis import (
    AletheCertificateAdapter,
    IndependentChecker,
    SoundnessArgument,
    StandaloneCheckResult,
    TCBAnalyzer,
    TCBComponent,
    TCBReport,
    TrustLevel,
    _count_loc,
    _classify_file,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_minimal_certificate(verdict: str = "SAFE") -> dict:
    """Build a minimal certificate dict for testing."""
    center = np.array([1.0, 2.0, 1.0, 2.0])
    gens = np.eye(4) * 0.5
    return {
        "version": "1.0",
        "certificate_id": "test-cert-001",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "verdict": verdict,
        "environment": {
            "id": "test-env",
            "state_dimension": 4,
            "action_dimensions": [2, 2],
            "num_agents": 2,
            "transition_hash": "abc",
        },
        "policies": [
            {"id": "pi_0", "hash": "h0", "architecture": "mlp"},
        ],
        "specification": {
            "grammar_tree": None,
            "temporal_formula": "always safe",
            "unsafe_predicate": None,
        },
        "abstract_fixpoint": {
            "domain": "zonotope",
            "iterations": 10,
            "widening_points": [5],
            "fixpoint_state": {
                "center": center.tolist(),
                "generators": gens.tolist(),
                "constraints": None,
            },
            "convergence_proof": {
                "ascending_chain": [],
                "widening_certificate": {},
                "narrowing_steps": [],
            },
        },
        "inductive_invariant": {
            "invariant_zonotope": {
                "center": center.tolist(),
                "generators": gens.tolist(),
            },
            "initial_zonotope": {
                "center": center.tolist(),
                "generators": (gens * 0.1).tolist(),
            },
            "post_zonotope": {
                "center": center.tolist(),
                "generators": gens.tolist(),
            },
        },
        "hb_consistency": {
            "hb_graph": {
                "nodes": [
                    {"id": "e0", "agent": "a0", "timestep": 0},
                    {"id": "e1", "agent": "a1", "timestep": 1},
                ],
                "edges": [
                    {"from": "e0", "to": "e1", "source": "program_order"},
                ],
            },
            "topological_order": ["e0", "e1"],
            "topological_order_exists": True,
            "cycle_freedom_proof": "DFS",
            "dfs_timestamps": {},
        },
        "composition_certificate": None,
        "race_witnesses": None,
        "hash": "",
    }


def _make_temp_py_tree():
    """Create a temporary directory with a small Python project."""
    tmpdir = tempfile.mkdtemp()

    # A file with some real code
    core = os.path.join(tmpdir, "abstract")
    os.makedirs(core)
    with open(os.path.join(core, "zonotope.py"), "w") as f:
        f.write(textwrap.dedent("""\
            import numpy as np

            class Zonotope:
                def __init__(self, center, generators):
                    self.center = center
                    self.generators = generators

                def contains(self, point):
                    return True
        """))

    # A reporting file (untrusted)
    rep = os.path.join(tmpdir, "reporting")
    os.makedirs(rep)
    with open(os.path.join(rep, "viz.py"), "w") as f:
        f.write("# just a comment\nprint('hello')\n")

    # A spec file (tested)
    spec = os.path.join(tmpdir, "spec")
    os.makedirs(spec)
    with open(os.path.join(spec, "parser.py"), "w") as f:
        f.write("def parse(s):\n    return s.split()\n")

    return tmpdir


# ======================================================================
# Test _count_loc
# ======================================================================

class TestCountLoc:
    def test_counts_code_lines(self, tmp_path):
        p = tmp_path / "sample.py"
        p.write_text("x = 1\ny = 2\n# comment\n\nz = 3\n")
        assert _count_loc(str(p)) == 3

    def test_skips_docstrings(self, tmp_path):
        p = tmp_path / "sample.py"
        p.write_text('x = 1\n"""\nlong docstring\n"""\ny = 2\n')
        assert _count_loc(str(p)) == 2

    def test_nonexistent_file(self):
        assert _count_loc("/nonexistent/file.py") == 0


# ======================================================================
# Test TCBComponent
# ======================================================================

class TestTCBComponent:
    def test_to_dict(self):
        c = TCBComponent(
            name="core.zonotope",
            path="core/zonotope.py",
            loc=100,
            trust_level=TrustLevel.TESTED,
            trust_justification="Has tests",
            dependencies=["numpy"],
            role="abstract domain",
        )
        d = c.to_dict()
        assert d["name"] == "core.zonotope"
        assert d["trust_level"] == "tested"
        assert d["loc"] == 100

    def test_categorization_abstract(self):
        level, role = _classify_file("abstract/zonotope.py")
        assert level == TrustLevel.TESTED
        assert "abstract" in role

    def test_categorization_reporting(self):
        level, role = _classify_file("reporting/viz.py")
        assert level == TrustLevel.UNTRUSTED

    def test_categorization_unknown(self):
        level, role = _classify_file("random/stuff.py")
        assert level == TrustLevel.UNTRUSTED


# ======================================================================
# Test TCBAnalyzer
# ======================================================================

class TestTCBAnalyzer:
    def test_analyze_codebase(self):
        tmpdir = _make_temp_py_tree()
        analyzer = TCBAnalyzer()
        components = analyzer.analyze_codebase(tmpdir)
        assert len(components) >= 3
        names = [c.name for c in components]
        # Should find abstract, reporting, spec files
        assert any("abstract" in n for n in names)
        assert any("spec" in n for n in names)

    def test_compute_tcb_size(self):
        tmpdir = _make_temp_py_tree()
        analyzer = TCBAnalyzer()
        analyzer.analyze_codebase(tmpdir)
        tcb_size = analyzer.compute_tcb_size()
        # abstract and spec are TESTED (in TCB), reporting is UNTRUSTED
        assert tcb_size > 0

    def test_generate_tcb_report(self):
        tmpdir = _make_temp_py_tree()
        analyzer = TCBAnalyzer()
        analyzer.analyze_codebase(tmpdir)
        report = analyzer.generate_tcb_report()
        assert isinstance(report, TCBReport)
        assert report.total_loc > 0
        assert report.tcb_loc <= report.total_loc
        assert 0.0 <= report.tcb_fraction <= 1.0
        assert len(report.trust_summary) > 0

    def test_identify_critical_path(self):
        tmpdir = _make_temp_py_tree()
        analyzer = TCBAnalyzer()
        analyzer.analyze_codebase(tmpdir)
        path = analyzer.identify_critical_path()
        assert isinstance(path, list)
        # abstract and spec components should be on the critical path
        assert any("abstract" in p for p in path)

    def test_check_dependency_trust(self):
        tmpdir = _make_temp_py_tree()
        analyzer = TCBAnalyzer()
        analyzer.analyze_codebase(tmpdir)
        warnings = analyzer.check_dependency_trust()
        # numpy is allowed, so no warnings for it
        assert isinstance(warnings, list)

    def test_report_to_dict(self):
        tmpdir = _make_temp_py_tree()
        analyzer = TCBAnalyzer()
        analyzer.analyze_codebase(tmpdir)
        report = analyzer.generate_tcb_report()
        d = report.to_dict()
        assert "total_loc" in d
        assert "tcb_fraction" in d
        assert "critical_path" in d
        json.dumps(d)  # must be JSON-serializable


# ======================================================================
# Test SoundnessArgument
# ======================================================================

class TestSoundnessArgument:
    def test_verify_chain_passes(self):
        sa = SoundnessArgument()
        valid, issues = sa.verify_chain()
        assert valid, f"Chain should be valid but got issues: {issues}"

    def test_generate_narrative(self):
        sa = SoundnessArgument()
        text = sa.generate_narrative()
        assert "MARACE Soundness Argument" in text
        assert "spec_parser" in text
        assert "fixpoint_engine" in text
        assert "PASS" in text

    def test_chain_has_five_links(self):
        sa = SoundnessArgument()
        assert len(sa._CHAIN) == 5


# ======================================================================
# Test AletheCertificateAdapter
# ======================================================================

class TestAletheCertificateAdapter:
    def test_convert_produces_output(self):
        cert = _make_minimal_certificate()
        adapter = AletheCertificateAdapter()
        text = adapter.convert(cert)
        assert len(text) > 0
        assert "Alethe" in text or "MARACE" in text

    def test_output_has_assume_and_step(self):
        cert = _make_minimal_certificate()
        adapter = AletheCertificateAdapter()
        text = adapter.convert(cert)
        assert "(assume" in text
        assert "(step" in text
        assert ":rule" in text

    def test_output_has_verdict(self):
        cert = _make_minimal_certificate("SAFE")
        adapter = AletheCertificateAdapter()
        text = adapter.convert(cert)
        assert "verdict SAFE" in text

    def test_parse_roundtrip(self):
        cert = _make_minimal_certificate()
        adapter = AletheCertificateAdapter()
        text = adapter.convert(cert)
        parsed = adapter.parse(text)
        assert len(parsed) > 0
        types = {s["type"] for s in parsed}
        assert "assume" in types
        assert "step" in types

    def test_parse_assume_line(self):
        adapter = AletheCertificateAdapter()
        result = adapter.parse("(assume t1 (some clause))")
        assert len(result) == 1
        assert result[0]["type"] == "assume"
        assert result[0]["id"] == "t1"

    def test_parse_step_with_rule(self):
        adapter = AletheCertificateAdapter()
        result = adapter.parse("(step t2 (verdict SAFE) :rule resolution)")
        assert len(result) == 1
        assert result[0]["rule"] == "resolution"

    def test_convert_with_composition(self):
        cert = _make_minimal_certificate()
        cert["composition_certificate"] = {
            "groups": [{"group_id": "g0", "agent_ids": ["a0"]}],
            "contracts": [],
            "discharge_proofs": [
                {
                    "assumption_id": "asm_0",
                    "discharged_by": "g1",
                    "lp_dual_witness": [0.5, 0.3],
                }
            ],
        }
        adapter = AletheCertificateAdapter()
        text = adapter.convert(cert)
        assert "composition" in text.lower() or "discharge" in text.lower()


# ======================================================================
# Test IndependentChecker
# ======================================================================

class TestIndependentChecker:
    def test_check_valid_certificate(self):
        cert = _make_minimal_certificate()
        checker = IndependentChecker()
        result = checker.check(cert)
        assert result.overall_passed
        assert "zonotope_containment" in result.obligations
        assert "hb_acyclicity" in result.obligations

    def test_zonotope_containment_pass(self):
        cert = _make_minimal_certificate()
        checker = IndependentChecker()
        result = checker.check(cert)
        ok, msg = result.obligations["zonotope_containment"]
        assert ok, msg

    def test_zonotope_containment_fail(self):
        cert = _make_minimal_certificate()
        # Make init zonotope larger than invariant
        cert["inductive_invariant"]["initial_zonotope"]["generators"] = (
            (np.eye(4) * 10.0).tolist()
        )
        checker = IndependentChecker()
        result = checker.check(cert)
        ok, _ = result.obligations["zonotope_containment"]
        assert not ok

    def test_hb_acyclicity_pass(self):
        cert = _make_minimal_certificate()
        checker = IndependentChecker()
        result = checker.check(cert)
        ok, msg = result.obligations["hb_acyclicity"]
        assert ok, msg

    def test_hb_acyclicity_fail(self):
        cert = _make_minimal_certificate()
        # Reverse topological order to make it invalid
        cert["hb_consistency"]["topological_order"] = ["e1", "e0"]
        checker = IndependentChecker()
        result = checker.check(cert)
        ok, _ = result.obligations["hb_acyclicity"]
        assert not ok

    def test_lp_dual_feasibility_pass(self):
        cert = _make_minimal_certificate()
        cert["composition_certificate"] = {
            "discharge_proofs": [
                {"lp_dual_witness": [0.5, 0.3, 0.0]},
            ]
        }
        checker = IndependentChecker()
        result = checker.check(cert)
        ok, msg = result.obligations["lp_dual_feasibility"]
        assert ok, msg

    def test_lp_dual_feasibility_fail(self):
        cert = _make_minimal_certificate()
        cert["composition_certificate"] = {
            "discharge_proofs": [
                {"lp_dual_witness": [0.5, -1.0, 0.0]},
            ]
        }
        checker = IndependentChecker()
        result = checker.check(cert)
        ok, _ = result.obligations["lp_dual_feasibility"]
        assert not ok

    def test_fixpoint_convergence_pass(self):
        cert = _make_minimal_certificate()
        checker = IndependentChecker()
        result = checker.check(cert)
        ok, msg = result.obligations["fixpoint_convergence"]
        assert ok, msg

    def test_result_summary(self):
        cert = _make_minimal_certificate()
        checker = IndependentChecker()
        result = checker.check(cert)
        summary = result.summary()
        assert "PASS" in summary

    def test_fixpoint_with_chain(self):
        cert = _make_minimal_certificate()
        center = [1.0, 2.0, 1.0, 2.0]
        cert["abstract_fixpoint"]["convergence_proof"]["ascending_chain"] = [
            {"center": center, "generators": (np.eye(4) * 0.3).tolist()},
            {"center": center, "generators": (np.eye(4) * 0.5).tolist()},
        ]
        checker = IndependentChecker()
        result = checker.check(cert)
        ok, msg = result.obligations["fixpoint_convergence"]
        assert ok, msg
