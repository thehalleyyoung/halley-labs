"""
Tests for reverse-mode automatic differentiation tape.

Covers ComputationGraph construction, backward pass gradient correctness,
checkpointed backward, VJP computation, TracedVar operations, and complex
computation graph topologies (diamond, fan-out).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dp_forge.autodiff import OpType
from dp_forge.autodiff.tape import (
    BackwardPass,
    CheckpointedBackward,
    ComputationGraph,
    TapeEntry,
    TracedVar,
    build_graph_from_function,
    reverse_gradient,
    reverse_hessian,
    traced_max,
    traced_sum,
    vjp_from_graph,
)


# Finite-difference helper
def _finite_diff_grad(fn, x, h=1e-7):
    """Central finite-difference gradient for scalar functions."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (fn(xp) - fn(xm)) / (2 * h)
    return grad


# ===================================================================
# TapeEntry tests
# ===================================================================


class TestTapeEntry:
    """Tests for TapeEntry validation."""

    def test_valid_entry(self):
        e = TapeEntry(
            node_id=0, op=OpType.ADD,
            parents=(1, 2), value=3.0,
            local_grads=(1.0, 1.0),
        )
        assert e.node_id == 0

    def test_mismatched_parents_grads_raises(self):
        with pytest.raises(ValueError, match="parents length"):
            TapeEntry(
                node_id=0, op=OpType.ADD,
                parents=(1, 2), value=3.0,
                local_grads=(1.0,),
            )

    def test_leaf_entry(self):
        e = TapeEntry(
            node_id=0, op=OpType.ADD,
            parents=(), value=2.0,
            local_grads=(), name="input",
        )
        assert e.name == "input"
        assert len(e.parents) == 0


# ===================================================================
# ComputationGraph construction
# ===================================================================


class TestComputationGraph:
    """Tests for computation graph construction."""

    def test_create_input(self):
        g = ComputationGraph()
        nid = g.create_input(2.0, name="x")
        assert g.size == 1
        assert g.input_ids == [nid]
        assert g.get_value(nid) == 2.0

    def test_add_operation(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.add(a, b)
        assert g.get_value(c) == 5.0
        assert g.size == 3

    def test_mul_operation(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.mul(a, b)
        assert g.get_value(c) == 6.0

    def test_div_operation(self):
        g = ComputationGraph()
        a = g.create_input(6.0)
        b = g.create_input(3.0)
        c = g.div(a, b)
        assert g.get_value(c) == pytest.approx(2.0)

    def test_div_by_zero_raises(self):
        g = ComputationGraph()
        a = g.create_input(1.0)
        b = g.create_input(0.0)
        with pytest.raises(ZeroDivisionError):
            g.div(a, b)

    def test_neg_operation(self):
        g = ComputationGraph()
        a = g.create_input(3.0)
        c = g.neg(a)
        assert g.get_value(c) == -3.0

    def test_log_operation(self):
        g = ComputationGraph()
        a = g.create_input(math.e)
        c = g.log(a)
        assert g.get_value(c) == pytest.approx(1.0)

    def test_log_nonpositive_raises(self):
        g = ComputationGraph()
        a = g.create_input(-1.0)
        with pytest.raises(ValueError):
            g.log(a)

    def test_exp_operation(self):
        g = ComputationGraph()
        a = g.create_input(1.0)
        c = g.exp(a)
        assert g.get_value(c) == pytest.approx(math.e)

    def test_abs_operation(self):
        g = ComputationGraph()
        a = g.create_input(-3.0)
        c = g.abs_op(a)
        assert g.get_value(c) == 3.0

    def test_max_operation(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(5.0)
        c = g.max_op(a, b)
        assert g.get_value(c) == 5.0

    def test_sum_operation(self):
        g = ComputationGraph()
        ids = [g.create_input(float(i)) for i in range(1, 4)]
        c = g.sum_op(ids)
        assert g.get_value(c) == 6.0

    def test_pow_operation(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.pow(a, b)
        assert g.get_value(c) == pytest.approx(8.0)

    def test_checkpointing(self):
        g = ComputationGraph()
        g.create_input(1.0)
        g.set_checkpoint("cp1")
        g.create_input(2.0)
        assert g.get_checkpoint("cp1") == 1

    def test_topological_order(self):
        """Tape entries should be in topological order (parents before children)."""
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.add(a, b)
        tape = g.tape
        ids = [e.node_id for e in tape]
        assert ids.index(a) < ids.index(c)
        assert ids.index(b) < ids.index(c)


# ===================================================================
# Backward pass gradient correctness
# ===================================================================


class TestBackwardPass:
    """Tests for reverse-mode gradient computation."""

    def test_add_gradient(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.add(a, b)
        bp = BackwardPass(g)
        grads = bp.compute_gradients(c)
        assert grads[a] == pytest.approx(1.0)
        assert grads[b] == pytest.approx(1.0)

    def test_mul_gradient(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.mul(a, b)
        bp = BackwardPass(g)
        grads = bp.compute_gradients(c)
        assert grads[a] == pytest.approx(3.0)  # d(ab)/da = b
        assert grads[b] == pytest.approx(2.0)  # d(ab)/db = a

    def test_div_gradient(self):
        g = ComputationGraph()
        a = g.create_input(6.0)
        b = g.create_input(3.0)
        c = g.div(a, b)
        bp = BackwardPass(g)
        grads = bp.compute_gradients(c)
        assert grads[a] == pytest.approx(1.0 / 3.0)
        assert grads[b] == pytest.approx(-6.0 / 9.0)

    def test_log_gradient(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        c = g.log(a)
        bp = BackwardPass(g)
        grads = bp.compute_gradients(c)
        assert grads[a] == pytest.approx(0.5)

    def test_exp_gradient(self):
        g = ComputationGraph()
        a = g.create_input(1.0)
        c = g.exp(a)
        bp = BackwardPass(g)
        grads = bp.compute_gradients(c)
        assert grads[a] == pytest.approx(math.e)

    def test_neg_gradient(self):
        g = ComputationGraph()
        a = g.create_input(3.0)
        c = g.neg(a)
        bp = BackwardPass(g)
        grads = bp.compute_gradients(c)
        assert grads[a] == pytest.approx(-1.0)

    def test_chain_gradient(self):
        """f = (a + b) * a => df/da = a+b + a = 2a+b, df/db = a"""
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        s = g.add(a, b)
        c = g.mul(s, a)
        bp = BackwardPass(g)
        grads = bp.compute_gradients(c)
        # f = (a+b)*a = a^2 + ab, df/da = 2a + b = 7
        assert grads[a] == pytest.approx(7.0)
        assert grads[b] == pytest.approx(2.0)

    def test_gradient_array(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.add(a, b)
        bp = BackwardPass(g)
        grad_arr = bp.gradient_array(c)
        np.testing.assert_allclose(grad_arr, [1.0, 1.0])

    def test_gradient_wrt_specific_nodes(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.add(a, b)
        bp = BackwardPass(g)
        grads = bp.compute_gradients(c, wrt=[a])
        assert a in grads
        assert b not in grads


# ===================================================================
# VJP tests
# ===================================================================


class TestVJP:
    """Tests for vector-Jacobian products."""

    def test_vjp_identity(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.add(a, b)
        v = np.array([1.0])
        vjp = vjp_from_graph(g, c, v)
        np.testing.assert_allclose(vjp, [1.0, 1.0])

    def test_vjp_scaled(self):
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.create_input(3.0)
        c = g.add(a, b)
        v = np.array([2.0])
        vjp = vjp_from_graph(g, c, v)
        np.testing.assert_allclose(vjp, [2.0, 2.0])

    def test_vjp_against_finite_diff(self):
        """VJP should agree with finite-difference gradient."""
        def fn(x):
            return x[0] ** 2 + x[0] * x[1]
        x = np.array([2.0, 3.0])
        graph, output_id = build_graph_from_function(fn, x)
        v = np.array([1.0])
        vjp = vjp_from_graph(graph, output_id, v)
        fd_grad = _finite_diff_grad(fn, x)
        np.testing.assert_allclose(vjp, fd_grad, rtol=1e-4)


# ===================================================================
# CheckpointedBackward tests
# ===================================================================


class TestCheckpointedBackward:
    """Tests for checkpointed gradient computation."""

    def test_checkpointed_matches_non_checkpointed(self):
        def build_fn(x):
            g = ComputationGraph()
            inputs = [g.create_input(float(xi)) for xi in x]
            s = g.add(inputs[0], inputs[1])
            return g, s

        x = np.array([2.0, 3.0])
        cb = CheckpointedBackward(segment_size=1)
        grad_cp = cb.compute(build_fn, x)

        graph, out = build_fn(x)
        bp = BackwardPass(graph)
        grad_no_cp = bp.gradient_array(out)

        np.testing.assert_allclose(grad_cp, grad_no_cp)

    def test_checkpointed_complex_graph(self):
        def build_fn(x):
            g = ComputationGraph()
            a = g.create_input(float(x[0]))
            b = g.create_input(float(x[1]))
            c = g.mul(a, b)
            d = g.add(c, a)
            return g, d

        x = np.array([3.0, 4.0])
        cb = CheckpointedBackward(segment_size=2)
        grad = cb.compute(build_fn, x)
        # f = a*b + a = a*(b+1), df/da = b+1 = 5, df/db = a = 3
        assert grad[0] == pytest.approx(5.0)
        assert grad[1] == pytest.approx(3.0)


# ===================================================================
# Complex graph topologies
# ===================================================================


class TestComplexGraphs:
    """Tests for diamond and fan-out graph patterns."""

    def test_diamond_graph(self):
        """Diamond: a -> b, a -> c, b+c -> d."""
        g = ComputationGraph()
        a = g.create_input(3.0)
        b = g.add(a, g.create_input(1.0))  # b = a + 1 = 4
        c = g.mul(a, g.create_input(2.0))  # c = a * 2 = 6
        d = g.add(b, c)  # d = b + c = 10
        bp = BackwardPass(g)
        grads = bp.compute_gradients(d, wrt=[a])
        # dd/da = db/da + dc/da = 1 + 2 = 3
        assert grads[a] == pytest.approx(3.0)

    def test_fan_out_graph(self):
        """Fan-out: one input used in multiple operations."""
        g = ComputationGraph()
        a = g.create_input(2.0)
        b = g.mul(a, a)  # b = a^2 = 4
        c = g.add(a, a)  # c = 2a = 4
        d = g.add(b, c)  # d = a^2 + 2a = 8
        bp = BackwardPass(g)
        grads = bp.compute_gradients(d, wrt=[a])
        # dd/da = 2a + 2 = 6
        assert grads[a] == pytest.approx(6.0)

    def test_sequential_chain(self):
        """Long sequential chain: a -> b -> c -> d."""
        g = ComputationGraph()
        a = g.create_input(1.0)
        two = g.create_input(2.0)
        b = g.mul(a, two)    # 2
        c = g.mul(b, two)    # 4
        d = g.mul(c, two)    # 8
        bp = BackwardPass(g)
        grads = bp.compute_gradients(d, wrt=[a])
        # d = 8*a, dd/da = 8
        assert grads[a] == pytest.approx(8.0)


# ===================================================================
# TracedVar tests
# ===================================================================


class TestTracedVar:
    """Tests for TracedVar operations."""

    def test_basic_arithmetic(self):
        g = ComputationGraph()
        x = TracedVar.input(g, 2.0, "x")
        y = TracedVar.input(g, 3.0, "y")
        z = x + y
        assert z.value == pytest.approx(5.0)

    def test_subtraction(self):
        g = ComputationGraph()
        x = TracedVar.input(g, 5.0, "x")
        y = TracedVar.input(g, 3.0, "y")
        z = x - y
        assert z.value == pytest.approx(2.0)

    def test_multiplication(self):
        g = ComputationGraph()
        x = TracedVar.input(g, 2.0, "x")
        y = TracedVar.input(g, 3.0, "y")
        z = x * y
        assert z.value == pytest.approx(6.0)

    def test_division(self):
        g = ComputationGraph()
        x = TracedVar.input(g, 6.0, "x")
        y = TracedVar.input(g, 3.0, "y")
        z = x / y
        assert z.value == pytest.approx(2.0)

    def test_scalar_operations(self):
        g = ComputationGraph()
        x = TracedVar.input(g, 3.0, "x")
        z = x + 2.0
        assert z.value == pytest.approx(5.0)
        z2 = 2.0 + x
        assert z2.value == pytest.approx(5.0)
        z3 = x * 4.0
        assert z3.value == pytest.approx(12.0)
        z4 = 4.0 * x
        assert z4.value == pytest.approx(12.0)

    def test_rsub(self):
        g = ComputationGraph()
        x = TracedVar.input(g, 2.0, "x")
        z = 5.0 - x
        assert z.value == pytest.approx(3.0)

    def test_negation(self):
        g = ComputationGraph()
        x = TracedVar.input(g, 3.0, "x")
        z = -x
        assert z.value == pytest.approx(-3.0)

    def test_log_exp(self):
        g = ComputationGraph()
        x = TracedVar.input(g, math.e, "x")
        z = x.log()
        assert z.value == pytest.approx(1.0)
        w = TracedVar.input(g, 1.0, "w")
        e = w.exp()
        assert e.value == pytest.approx(math.e)

    def test_abs(self):
        g = ComputationGraph()
        x = TracedVar.input(g, -3.0, "x")
        z = x.abs()
        assert z.value == pytest.approx(3.0)

    def test_traced_sum(self):
        g = ComputationGraph()
        vars = [TracedVar.input(g, float(i), f"x{i}") for i in range(1, 4)]
        s = traced_sum(vars)
        assert s.value == pytest.approx(6.0)

    def test_traced_sum_empty_raises(self):
        with pytest.raises(ValueError):
            traced_sum([])

    def test_traced_max(self):
        g = ComputationGraph()
        x = TracedVar.input(g, 2.0, "x")
        y = TracedVar.input(g, 5.0, "y")
        m = traced_max(x, y)
        assert m.value == pytest.approx(5.0)

    def test_traced_var_gradient(self):
        """TracedVar gradient through backward pass."""
        g = ComputationGraph()
        x = TracedVar.input(g, 2.0, "x")
        y = TracedVar.input(g, 3.0, "y")
        z = x * y + x
        bp = BackwardPass(g)
        grads = bp.compute_gradients(z.node_id)
        # z = xy + x = x(y+1), dz/dx = y+1 = 4
        assert grads[x.node_id] == pytest.approx(4.0)
        assert grads[y.node_id] == pytest.approx(2.0)


# ===================================================================
# reverse_gradient / reverse_hessian
# ===================================================================


class TestReverseGradient:
    """Tests for the full reverse-mode gradient API."""

    def test_linear_function(self):
        def fn(x):
            return 2.0 * x[0] + 3.0 * x[1]
        x = np.array([1.0, 1.0])
        val, grad = reverse_gradient(fn, x)
        assert val == pytest.approx(5.0)
        np.testing.assert_allclose(grad, [2.0, 3.0], rtol=1e-4)

    def test_quadratic_function(self):
        def fn(x):
            return x[0] ** 2 + x[1] ** 2
        x = np.array([3.0, 4.0])
        val, grad = reverse_gradient(fn, x)
        assert val == pytest.approx(25.0)
        np.testing.assert_allclose(grad, [6.0, 8.0], rtol=1e-4)

    def test_reverse_hessian_quadratic(self):
        def fn(x):
            return x[0] ** 2 + x[1] ** 2
        x = np.array([1.0, 1.0])
        H = reverse_hessian(fn, x)
        np.testing.assert_allclose(H, np.diag([2.0, 2.0]), atol=0.1)

    @pytest.mark.parametrize("a,b", [(1.0, 2.0), (3.0, 4.0), (0.5, 0.5)])
    def test_gradient_matches_finite_diff(self, a, b):
        def fn(x):
            return x[0] * x[1] + x[0]
        x = np.array([a, b])
        _, grad = reverse_gradient(fn, x)
        fd = _finite_diff_grad(fn, x)
        np.testing.assert_allclose(grad, fd, rtol=1e-3)
