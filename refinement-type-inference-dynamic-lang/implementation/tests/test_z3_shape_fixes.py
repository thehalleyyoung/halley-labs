"""Tests for Z3 symbolic shape verification fixes.

Covers:
  1. Symbolic matmul dimension mismatch detection (Issue 1)
  2. Broadcasting constraint verification with Z3 (Issue 2)
  3. Interprocedural shape propagation across function calls (Issue 3)
"""
import pytest

from src.tensor_shapes import (
    analyze_shapes, TensorShape, ShapeDim, ShapeErrorKind,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ── Issue 1: Symbolic matmul dimension mismatch detection ────────────────────

class TestSymbolicMatmulMismatch:
    """The Z3 symbolic path should now report errors for symbolic dimensions."""

    @pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
    def test_symbolic_matmul_mismatch_detected(self):
        """When symbolic dims can mismatch, Z3 should report an error."""
        source = """\
import torch

def f(n, m, k1, k2):
    a = torch.randn(n, k1)
    b = torch.randn(k2, m)
    return a @ b
"""
        result = analyze_shapes(source)
        # With symbolic dims k1 and k2 (unconstrained), Z3 can find k1 != k2
        matmul_errs = [
            e for e in result.errors
            if e.kind == ShapeErrorKind.MATMUL_INCOMPAT
        ]
        assert len(matmul_errs) >= 1, (
            "Z3 should detect possible matmul mismatch for symbolic dims"
        )

    @pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
    def test_concrete_matmul_mismatch_still_detected(self):
        """Concrete dimension mismatches should still be detected."""
        source = """\
import torch

x = torch.randn(3, 4)
y = torch.randn(5, 6)
z = x @ y
"""
        result = analyze_shapes(source)
        matmul_errs = [
            e for e in result.errors
            if e.kind == ShapeErrorKind.MATMUL_INCOMPAT
        ]
        assert len(matmul_errs) >= 1

    @pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
    def test_matching_symbolic_dims_no_false_positive(self):
        """When the same symbolic dim is used, no error should be reported."""
        source = """\
import torch

def f(n, m, k):
    a = torch.randn(n, k)
    b = torch.randn(k, m)
    return a @ b
"""
        result = analyze_shapes(source)
        matmul_errs = [
            e for e in result.errors
            if e.kind == ShapeErrorKind.MATMUL_INCOMPAT
        ]
        # Same symbolic name 'k' used for both dims — should not report error
        assert len(matmul_errs) == 0, (
            "Same symbolic dim name should not trigger false positive"
        )


# ── Issue 2: Broadcasting constraint verification with Z3 ───────────────────

class TestBroadcastZ3Encoding:
    """SHAPE_COMPAT encoding should now produce real Z3 constraints."""

    @pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
    def test_matmul_compat_encoding_uses_shape_dims(self):
        """Matmul SHAPE_COMPAT should encode shape_dim(var1,-1)==shape_dim(var2,-2)."""
        from src._experimental.refinement_lattice import (
            Z3Encoder, Pred, PredOp,
        )
        enc = Z3Encoder()
        pred = Pred(PredOp.SHAPE_COMPAT, ("A", "B", "matmul"))
        formula = enc.encode(pred)

        # The formula should reference shape_A_-1 and shape_B_-2
        s = z3.Solver()
        # Set shape_A[-1] = 4, shape_B[-2] = 5 → formula should be unsat
        s.add(enc.shape_dim_var("A", -1) == 4)
        s.add(enc.shape_dim_var("B", -2) == 5)
        s.add(formula)
        assert s.check() == z3.unsat, (
            "Matmul compat should be unsat when inner dims differ"
        )

        # Set both to 4 → should be sat
        s2 = z3.Solver()
        s2.add(enc.shape_dim_var("A", -1) == 4)
        s2.add(enc.shape_dim_var("B", -2) == 4)
        s2.add(formula)
        assert s2.check() == z3.sat

    @pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
    def test_broadcast_encoding_not_trivially_true(self):
        """Broadcasting encoding should reject incompatible non-broadcastable dims."""
        from src._experimental.refinement_lattice import (
            Z3Encoder, Pred, PredOp,
        )
        enc = Z3Encoder()
        pred = Pred(PredOp.SHAPE_COMPAT, ("X", "Y", "add"))
        formula = enc.encode(pred)

        # d1=3, d2=4 — not equal, neither is 1 → should be unsat
        s = z3.Solver()
        s.add(enc.shape_dim_var("X", -1) == 3)
        s.add(enc.shape_dim_var("Y", -1) == 4)
        s.add(formula)
        assert s.check() == z3.unsat, (
            "Broadcast should reject dims 3 vs 4"
        )

    @pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
    def test_broadcast_encoding_allows_dim_one(self):
        """Broadcasting should allow dim=1 to be broadcast against any dim."""
        from src._experimental.refinement_lattice import (
            Z3Encoder, Pred, PredOp,
        )
        enc = Z3Encoder()
        pred = Pred(PredOp.SHAPE_COMPAT, ("X", "Y", "add"))
        formula = enc.encode(pred)

        s = z3.Solver()
        # dim -1: X=1 (broadcastable), Y=5
        s.add(enc.shape_dim_var("X", -1) == 1)
        s.add(enc.shape_dim_var("Y", -1) == 5)
        # Fill remaining broadcast dims with equal values to satisfy constraints
        for i in range(2, 5):
            s.add(enc.shape_dim_var("X", -i) == 1)
            s.add(enc.shape_dim_var("Y", -i) == 1)
        s.add(formula)
        assert s.check() == z3.sat, (
            "Broadcast should allow dim=1 against any dim"
        )


# ── Issue 3: Interprocedural shape propagation ──────────────────────────────

class TestInterproceduralShapePropagation:
    """Shape contracts should propagate across function boundaries."""

    def test_infer_return_shape_from_constructor(self):
        """A function returning torch.randn(3, 4) should have shape (3, 4)."""
        from src.interprocedural import InterproceduralShapeAnalyzer
        source = """\
import torch

def make_tensor():
    return torch.randn(3, 4)
"""
        analyzer = InterproceduralShapeAnalyzer()
        contracts = analyzer.analyze_source(source)
        assert "make_tensor" in contracts
        assert contracts["make_tensor"].return_shape == (3, 4)

    def test_propagate_shape_through_call(self):
        """Return shape should propagate from callee to caller."""
        from src.interprocedural import InterproceduralShapeAnalyzer
        source = """\
import torch

def make_tensor():
    return torch.randn(3, 4)

def use_tensor():
    x = make_tensor()
    return x
"""
        analyzer = InterproceduralShapeAnalyzer()
        contracts = analyzer.analyze_source(source)
        assert "make_tensor" in contracts
        assert contracts["make_tensor"].return_shape == (3, 4)

    def test_nn_module_forward_annotated(self):
        """nn.Module.forward() should be marked as is_nn_module_forward."""
        from src.interprocedural import InterproceduralShapeAnalyzer
        source = """\
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.randn(3, 4)
"""
        analyzer = InterproceduralShapeAnalyzer()
        contracts = analyzer.analyze_source(source)
        forward_key = "MyModel.forward"
        assert forward_key in contracts
        assert contracts[forward_key].is_nn_module_forward is True
        assert contracts[forward_key].return_shape == (3, 4)

    def test_matmul_shape_inference_in_return(self):
        """Matmul return shape should be inferred from operands."""
        from src.interprocedural import InterproceduralShapeAnalyzer
        source = """\
import torch

def matmul_fn():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    return a @ b
"""
        analyzer = InterproceduralShapeAnalyzer()
        contracts = analyzer.analyze_source(source)
        assert "matmul_fn" in contracts
        assert contracts["matmul_fn"].return_shape == (3, 5)

    def test_shape_contract_has_params(self):
        """Shape contracts should list parameter names."""
        from src.interprocedural import InterproceduralShapeAnalyzer
        source = """\
import torch

def fn(x, y, z):
    return torch.randn(2, 3)
"""
        analyzer = InterproceduralShapeAnalyzer()
        contracts = analyzer.analyze_source(source)
        assert "fn" in contracts
        assert "x" in contracts["fn"].param_shapes
        assert "y" in contracts["fn"].param_shapes
        assert "z" in contracts["fn"].param_shapes
