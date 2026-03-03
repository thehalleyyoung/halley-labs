"""
Comprehensive tests for dp_forge.codegen — code generation module.

Tests cover CodegenMetadata, PythonCodeGenerator, CppCodeGenerator,
RustCodeGenerator, NumpyCodeGenerator, DocumentationGenerator,
CodeGenerator facade, validate_generated_code, round-trip execution,
edge cases, probability preservation, and cross-generator consistency.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from dp_forge.codegen import (
    CodeGenerator,
    CodegenMetadata,
    CppCodeGenerator,
    DocumentationGenerator,
    NumpyCodeGenerator,
    PythonCodeGenerator,
    RustCodeGenerator,
    _build_metadata,
    _compute_certificate_hash,
    _compute_mechanism_hash,
    _estimate_table_size,
    _format_float,
    _format_probability_row_cpp,
    _format_probability_row_python,
    _format_probability_row_rust,
    _format_probability_table_cpp,
    _format_probability_table_python,
    _format_probability_table_rust,
    validate_generated_code,
)
from dp_forge.types import (
    ExtractedMechanism,
    LossFunction,
    MechanismFamily,
    OptimalityCertificate,
    QuerySpec,
    QueryType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_prob_table(n: int, k: int, epsilon: float = 1.0) -> np.ndarray:
    """Build a valid DP probability table with bounded likelihood ratios."""
    table = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        mode = (k // 2 + i) % k
        raw = np.array(
            [math.exp(-epsilon * abs(j - mode) / k) for j in range(k)],
            dtype=np.float64,
        )
        table[i] = raw / raw.sum()
    return table


@pytest.fixture
def small_mechanism() -> ExtractedMechanism:
    p = _make_prob_table(3, 5, epsilon=1.0)
    return ExtractedMechanism(p_final=p, metadata={"mechanism_family": "PIECEWISE_CONST"})


@pytest.fixture
def small_mechanism_with_cert() -> ExtractedMechanism:
    p = _make_prob_table(3, 5, epsilon=1.0)
    cert = OptimalityCertificate(
        dual_vars=None, duality_gap=1e-9, primal_obj=0.42, dual_obj=0.42 - 1e-9,
    )
    return ExtractedMechanism(
        p_final=p, optimality_certificate=cert,
        metadata={"mechanism_family": MechanismFamily.PIECEWISE_CONST},
    )


@pytest.fixture
def small_spec() -> QuerySpec:
    return QuerySpec(
        query_values=np.array([0.0, 1.0, 2.0]), domain="test",
        sensitivity=1.0, epsilon=1.0, delta=0.0, k=5,
        loss_fn=LossFunction.L2, query_type=QueryType.COUNTING,
    )


@pytest.fixture
def uniform_mechanism() -> ExtractedMechanism:
    row = np.ones(4, dtype=np.float64) / 4
    return ExtractedMechanism(p_final=np.tile(row, (2, 1)))


@pytest.fixture
def single_row_mechanism() -> ExtractedMechanism:
    return ExtractedMechanism(p_final=np.array([[0.1, 0.2, 0.3, 0.4]]))


@pytest.fixture
def larger_mechanism() -> ExtractedMechanism:
    p = _make_prob_table(10, 20, epsilon=0.5)
    return ExtractedMechanism(p_final=p, metadata={"y_grid": list(np.linspace(-1, 1, 20))})

# ---------------------------------------------------------------------------
# CodegenMetadata
# ---------------------------------------------------------------------------


class TestCodegenMetadata:
    def test_basic_construction(self):
        m = CodegenMetadata(epsilon=1.0, delta=0.0, k=10, n=3)
        assert m.epsilon == 1.0 and m.k == 10 and m.query_type == "CUSTOM"

    def test_auto_timestamp(self):
        m = CodegenMetadata(epsilon=1.0, delta=0.0, k=10, n=3)
        assert m.generation_timestamp and "T" in m.generation_timestamp

    def test_to_dict_all_keys(self):
        d = CodegenMetadata(epsilon=0.5, delta=1e-5, k=100, n=5).to_dict()
        expected = {
            "epsilon", "delta", "k", "n", "query_type", "loss_function",
            "sensitivity", "objective_value", "certificate_hash",
            "dp_forge_version", "generation_timestamp", "mechanism_family",
            "mechanism_hash",
        }
        assert set(d.keys()) == expected

    def test_to_json_roundtrip(self):
        m = CodegenMetadata(epsilon=1.0, delta=0.0, k=10, n=3, mechanism_hash="abc")
        assert json.loads(m.to_json())["mechanism_hash"] == "abc"

    def test_custom_fields(self):
        m = CodegenMetadata(
            epsilon=2.0, delta=0.01, k=50, n=10, query_type="HISTOGRAM",
            loss_function="L1", sensitivity=2.0, mechanism_family="PIECEWISE_LINEAR",
        )
        assert m.query_type == "HISTOGRAM" and m.mechanism_family == "PIECEWISE_LINEAR"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_format_float_precision(self):
        r = _format_float(1.0 / 3.0)
        assert len(r) > 10

    def test_mechanism_hash_deterministic(self):
        p = np.array([[0.5, 0.5]], dtype=np.float64)
        h = _compute_mechanism_hash(p)
        assert len(h) == 64 and _compute_mechanism_hash(p) == h

    def test_mechanism_hash_differs(self):
        assert _compute_mechanism_hash(np.array([[0.5, 0.5]])) != \
               _compute_mechanism_hash(np.array([[0.3, 0.7]]))

    def test_certificate_hash_none(self):
        assert _compute_certificate_hash(None) == ""

    def test_certificate_hash_valid(self):
        cert = OptimalityCertificate(
            dual_vars=None, duality_gap=1e-9, primal_obj=1.0, dual_obj=1.0 - 1e-9,
        )
        assert len(_compute_certificate_hash(cert)) == 64

    def test_estimate_table_size(self):
        s = _estimate_table_size(10, 100)
        assert s["entries"] == 1000 and s["bytes_raw"] == 8000

    def test_build_metadata_no_spec(self, small_mechanism):
        m = _build_metadata(small_mechanism)
        assert m.k == 5 and m.n == 3 and m.mechanism_hash != ""

    def test_build_metadata_with_spec(self, small_mechanism, small_spec):
        m = _build_metadata(small_mechanism, small_spec)
        assert m.epsilon == 1.0 and m.query_type == "COUNTING"

    def test_build_metadata_with_cert(self, small_mechanism_with_cert):
        m = _build_metadata(small_mechanism_with_cert)
        assert m.objective_value == pytest.approx(0.42) and m.certificate_hash != ""

# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


class TestTableFormatting:
    def test_python_row(self):
        r = _format_probability_row_python(np.array([0.1, 0.2, 0.7]))
        assert r.startswith("[") and r.endswith("]")

    def test_python_table(self):
        r = _format_probability_table_python(np.array([[0.5, 0.5], [0.3, 0.7]]))
        assert "PROBABILITY_TABLE" in r

    def test_cpp_row(self):
        r = _format_probability_row_cpp(np.array([0.1, 0.9]))
        assert r.startswith("{") and r.endswith("}")

    def test_cpp_table(self):
        r = _format_probability_table_cpp(np.array([[0.5, 0.5], [0.3, 0.7]]))
        assert "static constexpr double" in r and "[2][2]" in r

    def test_rust_row(self):
        r = _format_probability_row_rust(np.array([0.25, 0.75]))
        assert "_f64" in r and r.startswith("[")

    def test_rust_table(self):
        r = _format_probability_table_rust(np.array([[0.5, 0.5], [0.3, 0.7]]))
        assert "static PROBABILITY_TABLE" in r and "[[f64; 2]; 2]" in r

# ---------------------------------------------------------------------------
# PythonCodeGenerator
# ---------------------------------------------------------------------------


class TestPythonCodeGenerator:
    def test_valid_syntax(self, small_mechanism):
        ast.parse(PythonCodeGenerator().generate(small_mechanism))

    def test_valid_syntax_with_spec(self, small_mechanism, small_spec):
        code = PythonCodeGenerator().generate(small_mechanism, small_spec)
        ast.parse(code)
        assert "COUNTING" in code

    def test_metadata_embedded(self, small_mechanism, small_spec):
        code = PythonCodeGenerator().generate(small_mechanism, small_spec)
        for s in ("EPSILON = ", "DELTA = ", "N_INPUTS = 3", "K_OUTPUTS = 5",
                   "SENSITIVITY = ", "MECHANISM_HASH = "):
            assert s in code

    def test_samplers_present(self, small_mechanism):
        code = PythonCodeGenerator().generate(small_mechanism)
        assert "class AliasSampler" in code and "class CDFSampler" in code

    def test_privacy_and_tests(self, small_mechanism):
        code = PythonCodeGenerator().generate(small_mechanism)
        assert "def assert_dp_holds" in code and "def run_self_tests" in code

    def test_main_block(self, small_mechanism):
        code = PythonCodeGenerator().generate(small_mechanism)
        assert 'if __name__ == "__main__"' in code

    def test_write(self, small_mechanism, tmp_path):
        gen = PythonCodeGenerator()
        code = gen.generate(small_mechanism)
        out = tmp_path / "nested" / "sampler.py"
        gen.write(code, out)
        assert out.read_text() == code

    def test_y_grid_present(self, larger_mechanism):
        assert "Y_GRID = [" in PythonCodeGenerator().generate(larger_mechanism)

    def test_y_grid_none(self, small_mechanism):
        assert "Y_GRID = None" in PythonCodeGenerator().generate(small_mechanism)

    def test_certificate_info(self, small_mechanism_with_cert, small_spec):
        code = PythonCodeGenerator().generate(small_mechanism_with_cert, small_spec)
        ast.parse(code)
        assert "CERTIFICATE_HASH" in code

    def test_single_row(self, single_row_mechanism):
        code = PythonCodeGenerator().generate(single_row_mechanism)
        ast.parse(code)
        assert "N_INPUTS = 1" in code

# ---------------------------------------------------------------------------
# CppCodeGenerator
# ---------------------------------------------------------------------------


class TestCppCodeGenerator:
    def test_file_keys(self, small_mechanism):
        f = CppCodeGenerator().generate(small_mechanism)
        assert set(f.keys()) == {"dp_sampler.h", "dp_sampler.cpp", "CMakeLists.txt"}

    def test_header_guards(self, small_mechanism):
        h = CppCodeGenerator().generate(small_mechanism)["dp_sampler.h"]
        assert "#ifndef DP_SAMPLER_H" in h and "#endif" in h

    def test_metadata_constants(self, small_mechanism, small_spec):
        h = CppCodeGenerator().generate(small_mechanism, small_spec)["dp_sampler.h"]
        assert "constexpr double EPSILON" in h and "N_INPUTS = 3" in h

    def test_source_table(self, small_mechanism):
        s = CppCodeGenerator().generate(small_mechanism)["dp_sampler.cpp"]
        assert "static constexpr double PROB_TABLE" in s

    def test_source_functions(self, small_mechanism):
        s = CppCodeGenerator().generate(small_mechanism)["dp_sampler.cpp"]
        assert "AliasSampler::build" in s and "assert_dp_holds" in s and "run_tests" in s

    def test_cmake(self, small_mechanism):
        c = CppCodeGenerator().generate(small_mechanism)["CMakeLists.txt"]
        assert "cmake_minimum_required" in c and "CXX_STANDARD 17" in c

    def test_write(self, small_mechanism, tmp_path):
        gen = CppCodeGenerator()
        gen.write(gen.generate(small_mechanism), tmp_path)
        assert (tmp_path / "dp_sampler.h").exists()
        assert (tmp_path / "dp_sampler.cpp").exists()

# ---------------------------------------------------------------------------
# RustCodeGenerator
# ---------------------------------------------------------------------------


class TestRustCodeGenerator:
    def test_file_keys(self, small_mechanism):
        f = RustCodeGenerator().generate(small_mechanism)
        assert {"src/lib.rs", "src/main.rs", "Cargo.toml"} == set(f.keys())

    def test_lib_constants(self, small_mechanism, small_spec):
        lib = RustCodeGenerator().generate(small_mechanism, small_spec)["src/lib.rs"]
        assert "pub const EPSILON: f64" in lib and "N_INPUTS: usize = 3" in lib

    def test_lib_structs(self, small_mechanism):
        lib = RustCodeGenerator().generate(small_mechanism)["src/lib.rs"]
        assert "pub struct AliasSampler" in lib and "pub struct DPMechanismSampler" in lib

    def test_lib_tests(self, small_mechanism):
        lib = RustCodeGenerator().generate(small_mechanism)["src/lib.rs"]
        assert "#[cfg(test)]" in lib and "#[test]" in lib

    def test_cargo_toml(self, small_mechanism):
        c = RustCodeGenerator().generate(small_mechanism)["Cargo.toml"]
        assert 'name = "dp-sampler"' in c and 'edition = "2021"' in c

    def test_write(self, small_mechanism, tmp_path):
        gen = RustCodeGenerator()
        gen.write(gen.generate(small_mechanism), tmp_path)
        assert (tmp_path / "src" / "lib.rs").exists()
        assert (tmp_path / "Cargo.toml").exists()

# ---------------------------------------------------------------------------
# NumpyCodeGenerator
# ---------------------------------------------------------------------------


class TestNumpyCodeGenerator:
    def test_contains_expected_functions(self, small_mechanism):
        code = NumpyCodeGenerator().generate(small_mechanism)
        assert "def dp_sample(" in code and "def dp_sample_batch(" in code

    def test_uses_numpy(self, small_mechanism):
        code = NumpyCodeGenerator().generate(small_mechanism)
        assert "np.cumsum" in code and "np.searchsorted" in code

    def test_contains_table(self, small_mechanism):
        code = NumpyCodeGenerator().generate(small_mechanism)
        assert "PROBABILITY_TABLE" in code

# ---------------------------------------------------------------------------
# DocumentationGenerator
# ---------------------------------------------------------------------------


class TestDocumentationGenerator:
    def test_markdown(self, small_mechanism, small_spec):
        md = DocumentationGenerator().generate_markdown(small_mechanism, small_spec)
        assert "# DP Mechanism Documentation" in md and "## Privacy Parameters" in md

    def test_markdown_with_cert(self, small_mechanism_with_cert):
        md = DocumentationGenerator().generate_markdown(small_mechanism_with_cert)
        assert "## Optimality Certificate" in md

    def test_latex(self, small_mechanism, small_spec):
        tex = DocumentationGenerator().generate_latex(small_mechanism, small_spec)
        assert r"\begin{table}" in tex and r"\varepsilon" in tex

    def test_latex_small_table(self, small_mechanism):
        assert "Probability Table" in DocumentationGenerator().generate_latex(small_mechanism)

    def test_latex_no_table_for_large(self, larger_mechanism):
        assert "Probability Table" not in DocumentationGenerator().generate_latex(larger_mechanism)

# ---------------------------------------------------------------------------
# validate_generated_code
# ---------------------------------------------------------------------------


class TestValidateGeneratedCode:
    def test_valid_mechanism(self, small_mechanism, small_spec):
        r = validate_generated_code(small_mechanism, small_spec)
        assert r["valid"] is True
        assert r["checks"]["row_sums"]["passed"]
        assert r["checks"]["non_negative"]["passed"]
        assert r["checks"]["dp_constraint"]["passed"]

    def test_without_spec(self, small_mechanism):
        r = validate_generated_code(small_mechanism)
        assert r["valid"] is True and "note" in r["checks"]["dp_constraint"]

    def test_table_size(self, small_mechanism):
        r = validate_generated_code(small_mechanism)
        assert r["checks"]["table_size"]["entries"] == 15

    def test_uniform(self, uniform_mechanism):
        assert validate_generated_code(uniform_mechanism)["valid"] is True

# ---------------------------------------------------------------------------
# CodeGenerator facade
# ---------------------------------------------------------------------------


class TestCodeGeneratorFacade:
    def test_generate_python(self, small_mechanism, small_spec):
        ast.parse(CodeGenerator().generate_python(small_mechanism, small_spec))

    def test_generate_python_file(self, small_mechanism, tmp_path):
        out = tmp_path / "s.py"
        code = CodeGenerator().generate_python(small_mechanism, output_path=out)
        assert out.read_text() == code

    def test_generate_cpp(self, small_mechanism):
        assert "dp_sampler.h" in CodeGenerator().generate_cpp(small_mechanism)

    def test_generate_rust(self, small_mechanism):
        assert "src/lib.rs" in CodeGenerator().generate_rust(small_mechanism)

    def test_generate_numpy(self, small_mechanism):
        code = CodeGenerator().generate_numpy(small_mechanism)
        assert "def dp_sample(" in code

    def test_generate_docs_markdown(self, small_mechanism):
        assert "# DP Mechanism" in CodeGenerator().generate_documentation(small_mechanism)

    def test_generate_docs_latex(self, small_mechanism):
        assert r"\begin{table}" in CodeGenerator().generate_documentation(
            small_mechanism, format="latex")

    def test_generate_docs_bad_format(self, small_mechanism):
        with pytest.raises(ValueError, match="Unknown format"):
            CodeGenerator().generate_documentation(small_mechanism, format="html")

    def test_generate_all_keys(self, small_mechanism, small_spec):
        r = CodeGenerator().generate_all(small_mechanism, small_spec)
        for key in ("python", "numpy", "cpp", "rust", "markdown", "latex", "validation"):
            assert key in r

    def test_generate_all_to_dir(self, small_mechanism, small_spec, tmp_path):
        CodeGenerator().generate_all(small_mechanism, small_spec, output_dir=tmp_path)
        assert (tmp_path / "python" / "dp_sampler.py").exists()
        assert (tmp_path / "cpp" / "dp_sampler.h").exists()
        assert (tmp_path / "rust" / "src" / "lib.rs").exists()
        assert (tmp_path / "docs" / "mechanism.md").exists()

    def test_generate_all_valid(self, small_mechanism):
        assert CodeGenerator().generate_all(small_mechanism)["validation"]["valid"]

# ---------------------------------------------------------------------------
# Round-trip: generate → execute → verify
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _exec(self, mechanism, spec=None):
        code = PythonCodeGenerator().generate(mechanism, spec)
        ns: dict = {}
        exec(compile(code, "<generated>", "exec"), ns)
        return ns

    def test_table_matches(self, small_mechanism):
        ns = self._exec(small_mechanism)
        for i in range(3):
            for j in range(5):
                assert abs(ns["PROBABILITY_TABLE"][i][j] - small_mechanism.p_final[i, j]) < 1e-12

    def test_metadata_values(self, small_mechanism, small_spec):
        ns = self._exec(small_mechanism, small_spec)
        assert ns["EPSILON"] == pytest.approx(1.0) and ns["N_INPUTS"] == 3

    def test_alias_sampler_range(self, small_mechanism):
        ns = self._exec(small_mechanism)
        samples = ns["AliasSampler"](ns["PROBABILITY_TABLE"][0]).sample_n(100)
        assert all(0 <= s < 5 for s in samples)

    def test_cdf_sampler_range(self, small_mechanism):
        ns = self._exec(small_mechanism)
        samples = ns["CDFSampler"](ns["PROBABILITY_TABLE"][0]).sample_n(100)
        assert all(0 <= s < 5 for s in samples)

    def test_sample_mechanism_alias(self, small_mechanism):
        ns = self._exec(small_mechanism)
        r = ns["sample_mechanism"](0, 50, "alias")
        assert len(r) == 50 and all(0 <= x < 5 for x in r)

    def test_sample_mechanism_cdf(self, small_mechanism):
        ns = self._exec(small_mechanism)
        assert len(ns["sample_mechanism"](1, 50, "cdf")) == 50

    def test_dp_assertion_passes(self, small_mechanism, small_spec):
        assert self._exec(small_mechanism, small_spec)["assert_dp_holds"]() is True

    def test_row_sums(self, small_mechanism):
        ns = self._exec(small_mechanism)
        for row in ns["PROBABILITY_TABLE"]:
            assert abs(sum(row) - 1.0) < 1e-8

    def test_get_output_value_no_grid(self, small_mechanism):
        ns = self._exec(small_mechanism)
        assert ns["get_output_value"](2) == 2.0

    def test_get_output_value_with_grid(self, larger_mechanism):
        ns = self._exec(larger_mechanism)
        assert ns["Y_GRID"] is not None
        val = ns["get_output_value"](0)
        assert isinstance(val, (int, float))

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row_all(self, single_row_mechanism):
        r = CodeGenerator().generate_all(single_row_mechanism)
        ast.parse(r["python"])
        assert "N_INPUTS = 1" in r["cpp"]["dp_sampler.h"]
        assert "N_INPUTS: usize = 1" in r["rust"]["src/lib.rs"]

    def test_two_bin_mechanism(self):
        p = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float64)
        code = PythonCodeGenerator().generate(ExtractedMechanism(p_final=p))
        ast.parse(code)
        ns: dict = {}
        exec(compile(code, "<generated>", "exec"), ns)
        assert ns["K_OUTPUTS"] == 2

    def test_larger_mechanism(self, larger_mechanism):
        r = CodeGenerator().generate_all(larger_mechanism)
        ast.parse(r["python"])
        assert "def dp_sample(" in r["numpy"]

    def test_near_zero_probs(self):
        p = np.array([[1e-15, 1.0 - 1e-15], [1.0 - 1e-15, 1e-15]], dtype=np.float64)
        ast.parse(PythonCodeGenerator().generate(ExtractedMechanism(p_final=p)))

    def test_y_grid_metadata(self):
        p = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
        mech = ExtractedMechanism(p_final=p, metadata={"y_grid": [-1.0, 1.0]})
        code = PythonCodeGenerator().generate(mech)
        assert "Y_GRID = [" in code
        ns: dict = {}
        exec(compile(code, "<generated>", "exec"), ns)
        assert len(ns["Y_GRID"]) == 2

# ---------------------------------------------------------------------------
# Probability preservation
# ---------------------------------------------------------------------------


class TestProbabilityPreservation:
    def test_python_full_precision(self):
        p = np.array([[1.0 / 3.0, 2.0 / 3.0]], dtype=np.float64)
        p[0] /= p[0].sum()
        code = PythonCodeGenerator().generate(ExtractedMechanism(p_final=p))
        ns: dict = {}
        exec(compile(code, "<generated>", "exec"), ns)
        for j in range(2):
            assert abs(ns["PROBABILITY_TABLE"][0][j] - p[0, j]) < 1e-14

    def test_cpp_precision(self):
        v = 1.0 / 7.0
        p = np.array([[v, 1.0 - v]], dtype=np.float64)
        src = CppCodeGenerator().generate(ExtractedMechanism(p_final=p))["dp_sampler.cpp"]
        assert f"{v:.17g}" in src

    def test_rust_f64_suffix(self):
        p = np.array([[0.123456789012345, 1.0 - 0.123456789012345]], dtype=np.float64)
        assert "_f64" in RustCodeGenerator().generate(ExtractedMechanism(p_final=p))["src/lib.rs"]

# ---------------------------------------------------------------------------
# Cross-generator consistency
# ---------------------------------------------------------------------------


class TestCrossGeneratorConsistency:
    def test_same_hash_across_generators(self, small_mechanism, small_spec):
        py_code = PythonCodeGenerator().generate(small_mechanism, small_spec)
        ns: dict = {}
        exec(compile(py_code, "<generated>", "exec"), ns)
        py_hash = ns["MECHANISM_HASH"]

        cpp = CppCodeGenerator().generate(small_mechanism, small_spec)
        rust = RustCodeGenerator().generate(small_mechanism, small_spec)
        assert py_hash[:16] in cpp["dp_sampler.h"]
        assert py_hash in rust["src/lib.rs"]

    def test_same_dimensions(self, small_mechanism):
        ns: dict = {}
        exec(compile(PythonCodeGenerator().generate(small_mechanism), "<gen>", "exec"), ns)
        cpp_h = CppCodeGenerator().generate(small_mechanism)["dp_sampler.h"]
        assert ns["N_INPUTS"] == 3 and "N_INPUTS = 3" in cpp_h
        assert ns["K_OUTPUTS"] == 5 and "K_OUTPUTS = 5" in cpp_h

# ---------------------------------------------------------------------------
# _estimate_table_size
# ---------------------------------------------------------------------------


class TestEstimateTableSize:
    def test_small(self):
        s = _estimate_table_size(2, 3)
        assert s["entries"] == 6 and s["bytes_raw"] == 48

    def test_large(self):
        s = _estimate_table_size(100, 1000)
        assert s["entries"] == 100_000 and s["bytes_raw"] == 800_000

    def test_alias_double(self):
        s = _estimate_table_size(5, 10)
        assert s["bytes_alias"] == s["bytes_raw"] * 2
