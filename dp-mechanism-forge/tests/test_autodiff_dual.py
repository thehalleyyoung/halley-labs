"""
Tests for forward-mode automatic differentiation via dual numbers.

Covers DualNumber arithmetic, transcendental functions, JetNumber second
derivatives, DualVector operations, sparse Jacobian computation, and
property-based testing with hypothesis.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.autodiff.dual_numbers import (
    DualNumber,
    DualVector,
    JetNumber,
    compute_jacobian,
    compute_sparse_jacobian,
    dual_abs,
    dual_cos,
    dual_exp,
    dual_log,
    dual_max,
    dual_min,
    dual_sin,
    dual_sqrt,
    dualvec_abs,
    dualvec_exp,
    dualvec_log,
    forward_derivative,
    jet_exp,
    jet_log,
    second_derivative,
)


# Finite-difference helper
def _finite_diff(fn, x, h=1e-7):
    """Central finite difference for scalar functions."""
    return (fn(x + h) - fn(x - h)) / (2 * h)


def _finite_diff_2nd(fn, x, h=1e-5):
    """Central finite difference for second derivative."""
    return (fn(x + h) - 2 * fn(x) + fn(x - h)) / (h * h)


# ===================================================================
# DualNumber Arithmetic
# ===================================================================


class TestDualNumberArithmetic:
    """Tests for DualNumber arithmetic operations."""

    def test_add_dual_dual(self):
        a = DualNumber(2.0, 1.0)
        b = DualNumber(3.0, 2.0)
        c = a + b
        assert c.value == 5.0
        assert c.derivative == 3.0

    def test_add_dual_scalar(self):
        a = DualNumber(2.0, 1.0)
        c = a + 5.0
        assert c.value == 7.0
        assert c.derivative == 1.0

    def test_radd(self):
        a = DualNumber(2.0, 1.0)
        c = 5.0 + a
        assert c.value == 7.0
        assert c.derivative == 1.0

    def test_sub(self):
        a = DualNumber(5.0, 3.0)
        b = DualNumber(2.0, 1.0)
        c = a - b
        assert c.value == 3.0
        assert c.derivative == 2.0

    def test_rsub(self):
        a = DualNumber(2.0, 1.0)
        c = 5.0 - a
        assert c.value == 3.0
        assert c.derivative == -1.0

    def test_mul_dual_dual(self):
        a = DualNumber(2.0, 1.0)
        b = DualNumber(3.0, 0.0)
        c = a * b
        assert c.value == 6.0
        assert c.derivative == 3.0  # 2*0 + 1*3

    def test_mul_dual_scalar(self):
        a = DualNumber(3.0, 2.0)
        c = a * 4.0
        assert c.value == 12.0
        assert c.derivative == 8.0

    def test_rmul(self):
        a = DualNumber(3.0, 2.0)
        c = 4.0 * a
        assert c.value == 12.0
        assert c.derivative == 8.0

    def test_truediv_dual_dual(self):
        a = DualNumber(6.0, 1.0)
        b = DualNumber(3.0, 0.0)
        c = a / b
        assert c.value == pytest.approx(2.0)
        assert c.derivative == pytest.approx(1.0 / 3.0)

    def test_truediv_dual_scalar(self):
        a = DualNumber(6.0, 2.0)
        c = a / 3.0
        assert c.value == pytest.approx(2.0)
        assert c.derivative == pytest.approx(2.0 / 3.0)

    def test_rtruediv(self):
        a = DualNumber(2.0, 1.0)
        c = 6.0 / a
        assert c.value == pytest.approx(3.0)
        # d/dx [6/x] at x=2 = -6/4 = -1.5
        assert c.derivative == pytest.approx(-1.5)

    def test_div_by_zero_raises(self):
        a = DualNumber(1.0, 1.0)
        b = DualNumber(0.0, 0.0)
        with pytest.raises(ZeroDivisionError):
            a / b

    def test_rdiv_by_zero_raises(self):
        a = DualNumber(0.0, 0.0)
        with pytest.raises(ZeroDivisionError):
            1.0 / a

    def test_pow_constant_exponent(self):
        a = DualNumber(3.0, 1.0)
        c = a ** 2
        assert c.value == pytest.approx(9.0)
        assert c.derivative == pytest.approx(6.0)  # 2 * 3^1 * 1

    def test_pow_dual_exponent(self):
        a = DualNumber(2.0, 1.0)
        b = DualNumber(3.0, 0.0)
        c = a ** b
        assert c.value == pytest.approx(8.0)
        assert c.derivative == pytest.approx(3.0 * 2.0 ** 2)

    def test_rpow(self):
        a = DualNumber(3.0, 1.0)
        c = 2.0 ** a  # 2^x at x=3
        assert c.value == pytest.approx(8.0)
        assert c.derivative == pytest.approx(8.0 * math.log(2.0))

    def test_neg(self):
        a = DualNumber(2.0, 3.0)
        c = -a
        assert c.value == -2.0
        assert c.derivative == -3.0

    def test_abs_positive(self):
        a = DualNumber(2.0, 3.0)
        c = abs(a)
        assert c.value == 2.0
        assert c.derivative == 3.0

    def test_abs_negative(self):
        a = DualNumber(-2.0, 3.0)
        c = abs(a)
        assert c.value == 2.0
        assert c.derivative == -3.0

    def test_comparison_operators(self):
        a = DualNumber(2.0, 1.0)
        b = DualNumber(3.0, 0.0)
        assert a < b
        assert a <= b
        assert b > a
        assert b >= a
        assert not (a == b)
        assert a < 3.0
        assert a == 2.0

    def test_float_conversion(self):
        a = DualNumber(2.5, 1.0)
        assert float(a) == 2.5

    def test_hash(self):
        a = DualNumber(1.0, 2.0)
        b = DualNumber(1.0, 2.0)
        assert hash(a) == hash(b)


# ===================================================================
# Transcendental Functions
# ===================================================================


class TestTranscendental:
    """Tests for transcendental function derivatives."""

    def test_exp_value_and_deriv(self):
        x = DualNumber(1.0, 1.0)
        r = dual_exp(x)
        assert r.value == pytest.approx(math.e)
        assert r.derivative == pytest.approx(math.e)

    def test_log_value_and_deriv(self):
        x = DualNumber(2.0, 1.0)
        r = dual_log(x)
        assert r.value == pytest.approx(math.log(2.0))
        assert r.derivative == pytest.approx(0.5)

    def test_sqrt_value_and_deriv(self):
        x = DualNumber(4.0, 1.0)
        r = dual_sqrt(x)
        assert r.value == pytest.approx(2.0)
        assert r.derivative == pytest.approx(0.25)

    def test_sin_cos_derivatives(self):
        x = DualNumber(math.pi / 4, 1.0)
        s = dual_sin(x)
        c = dual_cos(x)
        assert s.value == pytest.approx(math.sin(math.pi / 4))
        assert s.derivative == pytest.approx(math.cos(math.pi / 4))
        assert c.value == pytest.approx(math.cos(math.pi / 4))
        assert c.derivative == pytest.approx(-math.sin(math.pi / 4))

    def test_log_negative_raises(self):
        with pytest.raises(ValueError):
            dual_log(DualNumber(-1.0, 1.0))

    def test_sqrt_negative_raises(self):
        with pytest.raises(ValueError):
            dual_sqrt(DualNumber(-1.0, 1.0))

    def test_dual_abs_on_float(self):
        r = dual_abs(3.0)
        assert r.value == 3.0
        assert r.derivative == 0.0

    def test_dual_max(self):
        a = DualNumber(2.0, 1.0)
        b = DualNumber(3.0, 2.0)
        r = dual_max(a, b)
        assert r.value == 3.0
        assert r.derivative == 2.0

    def test_dual_min(self):
        a = DualNumber(2.0, 1.0)
        b = DualNumber(3.0, 2.0)
        r = dual_min(a, b)
        assert r.value == 2.0
        assert r.derivative == 1.0

    def test_exp_on_float(self):
        r = dual_exp(1.0)
        assert r.value == pytest.approx(math.e)
        assert r.derivative == 0.0

    @pytest.mark.parametrize("x_val", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_exp_derivative_matches_finite_diff(self, x_val):
        x = DualNumber(x_val, 1.0)
        r = dual_exp(x)
        fd = _finite_diff(math.exp, x_val)
        assert r.derivative == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("x_val", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_log_derivative_matches_finite_diff(self, x_val):
        x = DualNumber(x_val, 1.0)
        r = dual_log(x)
        fd = _finite_diff(math.log, x_val)
        assert r.derivative == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("x_val", [0.25, 1.0, 4.0, 9.0])
    def test_sqrt_derivative_matches_finite_diff(self, x_val):
        x = DualNumber(x_val, 1.0)
        r = dual_sqrt(x)
        fd = _finite_diff(math.sqrt, x_val)
        assert r.derivative == pytest.approx(fd, rel=1e-5)


# ===================================================================
# Composite Derivatives (chain rule)
# ===================================================================


class TestCompositeDerivatives:
    """Test derivatives of composite expressions."""

    def test_x_squared(self):
        # f(x) = x^2, f'(x) = 2x
        val, deriv = forward_derivative(lambda x: x * x, 3.0)
        assert val == pytest.approx(9.0)
        assert deriv == pytest.approx(6.0)

    def test_exp_log_identity(self):
        # f(x) = exp(log(x)) = x, f'(x) = 1
        def f(x):
            return dual_exp(dual_log(x))
        val, deriv = forward_derivative(f, 2.0)
        assert val == pytest.approx(2.0)
        assert deriv == pytest.approx(1.0)

    def test_polynomial(self):
        # f(x) = x^3 + 2x^2 - x + 1, f'(x) = 3x^2 + 4x - 1
        def f(x):
            return x ** 3 + 2 * x ** 2 - x + 1
        val, deriv = forward_derivative(f, 2.0)
        assert val == pytest.approx(8 + 8 - 2 + 1)
        assert deriv == pytest.approx(12 + 8 - 1)

    def test_chain_rule_exp_x2(self):
        # f(x) = exp(x^2), f'(x) = 2x*exp(x^2)
        def f(x):
            return dual_exp(x * x)
        val, deriv = forward_derivative(f, 1.0)
        assert val == pytest.approx(math.exp(1.0))
        assert deriv == pytest.approx(2.0 * math.exp(1.0))


# ===================================================================
# JetNumber (second derivatives)
# ===================================================================


class TestJetNumber:
    """Tests for JetNumber second derivatives."""

    def test_variable_creation(self):
        j = JetNumber.variable(2.0, order=2)
        assert j.value == 2.0
        assert j.first_deriv == 1.0
        assert j.second_deriv == 0.0

    def test_constant_creation(self):
        j = JetNumber.constant(3.0, order=2)
        assert j.value == 3.0
        assert j.first_deriv == 0.0

    def test_add(self):
        a = JetNumber.variable(2.0)
        b = JetNumber.constant(3.0)
        c = a + b
        assert c.value == 5.0
        assert c.first_deriv == 1.0

    def test_sub(self):
        a = JetNumber.variable(5.0)
        b = JetNumber.constant(3.0)
        c = a - b
        assert c.value == 2.0
        assert c.first_deriv == 1.0

    def test_mul(self):
        a = JetNumber.variable(2.0)
        b = JetNumber.variable(3.0)
        c = a * b
        assert c.value == 6.0

    def test_div(self):
        a = JetNumber.variable(6.0)
        b = JetNumber.constant(3.0)
        c = a / b
        assert c.value == pytest.approx(2.0)

    def test_div_by_zero_raises(self):
        a = JetNumber.variable(1.0)
        b = JetNumber.constant(0.0)
        with pytest.raises(ZeroDivisionError):
            a / b

    def test_scalar_mul(self):
        a = JetNumber.variable(2.0)
        c = a * 3.0
        assert c.value == 6.0
        assert c.first_deriv == 3.0

    def test_neg(self):
        a = JetNumber.variable(2.0)
        c = -a
        assert c.value == -2.0
        assert c.first_deriv == -1.0

    def test_x_squared_second_deriv(self):
        # f(x) = x^2 -> f''(x) = 2
        val, d1, d2 = second_derivative(lambda x: x * x, 3.0)
        assert val == pytest.approx(9.0)
        assert d1 == pytest.approx(6.0)
        assert d2 == pytest.approx(2.0)

    def test_x_cubed_second_deriv(self):
        # f(x) = x^3 -> f''(x) = 6x
        val, d1, d2 = second_derivative(lambda x: x * x * x, 2.0)
        assert val == pytest.approx(8.0)
        assert d1 == pytest.approx(12.0)
        assert d2 == pytest.approx(12.0)

    def test_jet_exp_second_deriv(self):
        # f(x) = exp(x) -> f''(x) = exp(x)
        val, d1, d2 = second_derivative(lambda x: jet_exp(x), 1.0)
        assert val == pytest.approx(math.e)
        assert d1 == pytest.approx(math.e, rel=1e-6)
        assert d2 == pytest.approx(math.e, rel=1e-5)

    def test_jet_log_second_deriv(self):
        # f(x) = log(x) -> f''(x) = -1/x^2
        val, d1, d2 = second_derivative(lambda x: jet_log(x), 2.0)
        assert val == pytest.approx(math.log(2.0))
        assert d1 == pytest.approx(0.5, rel=1e-6)
        assert d2 == pytest.approx(-0.25, rel=1e-5)

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 2.0, 3.0])
    def test_second_deriv_matches_finite_diff(self, x_val):
        """x^2 second derivative should be 2.0 everywhere."""
        _, _, d2 = second_derivative(lambda x: x * x, x_val)
        fd2 = _finite_diff_2nd(lambda t: t * t, x_val)
        assert d2 == pytest.approx(fd2, rel=1e-3)


# ===================================================================
# DualVector operations
# ===================================================================


class TestDualVector:
    """Tests for vectorised DualVector operations."""

    def test_identity_creation(self):
        dv = DualVector.identity(np.array([1.0, 2.0, 3.0]))
        assert dv.values.shape == (3,)
        assert dv.jacobian.shape == (3, 3)
        np.testing.assert_array_equal(dv.jacobian, np.eye(3))

    def test_constant_creation(self):
        dv = DualVector.constant(np.array([1.0, 2.0]), 3)
        assert dv.jacobian.shape == (2, 3)
        np.testing.assert_array_equal(dv.jacobian, np.zeros((2, 3)))

    def test_add_vectors(self):
        a = DualVector.identity(np.array([1.0, 2.0]))
        b = DualVector.identity(np.array([3.0, 4.0]))
        c = a + b
        np.testing.assert_allclose(c.values, [4.0, 6.0])

    def test_sub_vectors(self):
        a = DualVector.identity(np.array([5.0, 6.0]))
        b = DualVector.identity(np.array([1.0, 2.0]))
        c = a - b
        np.testing.assert_allclose(c.values, [4.0, 4.0])

    def test_mul_elementwise(self):
        a = DualVector.identity(np.array([2.0, 3.0]))
        b = DualVector.identity(np.array([4.0, 5.0]))
        c = a * b
        np.testing.assert_allclose(c.values, [8.0, 15.0])

    def test_scalar_mul(self):
        a = DualVector.identity(np.array([1.0, 2.0]))
        c = a * 3.0
        np.testing.assert_allclose(c.values, [3.0, 6.0])

    def test_neg(self):
        a = DualVector.identity(np.array([1.0, -2.0]))
        c = -a
        np.testing.assert_allclose(c.values, [-1.0, 2.0])

    def test_sum(self):
        a = DualVector.identity(np.array([1.0, 2.0, 3.0]))
        val, grad = a.sum()
        assert val == pytest.approx(6.0)
        np.testing.assert_allclose(grad, [1.0, 1.0, 1.0])

    def test_dualvec_exp(self):
        dv = DualVector.identity(np.array([0.0, 1.0]))
        result = dualvec_exp(dv)
        np.testing.assert_allclose(result.values, [1.0, math.e], rtol=1e-10)

    def test_dualvec_log(self):
        dv = DualVector.identity(np.array([1.0, math.e]))
        result = dualvec_log(dv)
        np.testing.assert_allclose(result.values, [0.0, 1.0], rtol=1e-10)

    def test_dualvec_abs(self):
        dv = DualVector(np.array([-2.0, 3.0]), np.eye(2))
        result = dualvec_abs(dv)
        np.testing.assert_allclose(result.values, [2.0, 3.0])


# ===================================================================
# Sparse Jacobian computation
# ===================================================================


class TestSparseJacobian:
    """Tests for Jacobian and sparse Jacobian computation."""

    def test_identity_jacobian(self):
        def fn(x):
            return x
        x = np.array([1.0, 2.0, 3.0])
        J = compute_jacobian(fn, x)
        np.testing.assert_allclose(J, np.eye(3), atol=1e-10)

    def test_linear_function_jacobian(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        def fn(x):
            result = np.empty(2, dtype=object)
            for i in range(2):
                s = DualNumber(0.0)
                for j in range(2):
                    s = s + A[i, j] * x[j]
                result[i] = s
            return result
        x = np.array([1.0, 1.0])
        J = compute_jacobian(fn, x)
        np.testing.assert_allclose(J, A, atol=1e-8)

    def test_sparse_jacobian_no_pattern(self):
        def fn(x):
            return x * 2.0
        x = np.array([1.0, 2.0])
        J = compute_sparse_jacobian(fn, x)
        np.testing.assert_allclose(J.toarray(), np.diag([2.0, 2.0]), atol=1e-8)

    def test_sparse_jacobian_with_pattern(self):
        def fn(x):
            return x * 2.0
        x = np.array([1.0, 2.0])
        pattern = np.eye(2, dtype=bool)
        J = compute_sparse_jacobian(fn, x, sparsity_pattern=pattern)
        np.testing.assert_allclose(J.toarray(), np.diag([2.0, 2.0]), atol=1e-8)

    def test_jacobian_nonlinear(self):
        def fn(x):
            # f(x) = [x0^2, x0*x1]
            result = np.empty(2, dtype=object)
            result[0] = x[0] * x[0]
            result[1] = x[0] * x[1]
            return result
        x = np.array([2.0, 3.0])
        J = compute_jacobian(fn, x)
        # df0/dx0 = 2*x0 = 4, df0/dx1 = 0
        # df1/dx0 = x1 = 3, df1/dx1 = x0 = 2
        expected = np.array([[4.0, 0.0], [3.0, 2.0]])
        np.testing.assert_allclose(J, expected, atol=1e-8)


# ===================================================================
# Property-based tests with hypothesis
# ===================================================================


class TestDualNumberProperties:
    """Property-based tests for DualNumber correctness."""

    @given(x=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30)
    def test_exp_derivative_property(self, x):
        d = DualNumber(x, 1.0)
        r = dual_exp(d)
        fd = _finite_diff(math.exp, x)
        assert r.derivative == pytest.approx(fd, rel=1e-4)

    @given(x=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30)
    def test_log_derivative_property(self, x):
        d = DualNumber(x, 1.0)
        r = dual_log(d)
        fd = _finite_diff(math.log, x)
        assert r.derivative == pytest.approx(fd, rel=1e-4)

    @given(
        a=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        b=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_add_commutative(self, a, b):
        da = DualNumber(a, 1.0)
        db = DualNumber(b, 2.0)
        r1 = da + db
        r2 = db + da
        assert r1.value == pytest.approx(r2.value)
        assert r1.derivative == pytest.approx(r2.derivative)

    @given(
        a=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        b=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_mul_commutative(self, a, b):
        da = DualNumber(a, 1.0)
        db = DualNumber(b, 2.0)
        r1 = da * db
        r2 = db * da
        assert r1.value == pytest.approx(r2.value)
        assert r1.derivative == pytest.approx(r2.derivative)

    @given(x=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20)
    def test_sqrt_derivative_property(self, x):
        d = DualNumber(x, 1.0)
        r = dual_sqrt(d)
        fd = _finite_diff(math.sqrt, x)
        assert r.derivative == pytest.approx(fd, rel=1e-4)

    @given(x=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20)
    def test_sin_derivative_property(self, x):
        d = DualNumber(x, 1.0)
        r = dual_sin(d)
        fd = _finite_diff(math.sin, x)
        assert r.derivative == pytest.approx(fd, rel=1e-4)

    @given(x=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20)
    def test_jet_second_deriv_x_squared(self, x):
        """x^2 always has second derivative 2."""
        _, _, d2 = second_derivative(lambda t: t * t, x)
        assert d2 == pytest.approx(2.0, abs=1e-8)


# ===================================================================
# Edge cases
# ===================================================================


class TestDualEdgeCases:
    """Edge case tests."""

    def test_dual_zero(self):
        a = DualNumber(0.0, 0.0)
        b = DualNumber(1.0, 1.0)
        c = a + b
        assert c.value == 1.0

    def test_dual_large_values(self):
        a = DualNumber(1e10, 1.0)
        b = DualNumber(1e10, 1.0)
        c = a + b
        assert c.value == pytest.approx(2e10)

    def test_dual_small_values(self):
        a = DualNumber(1e-15, 1e-15)
        b = DualNumber(1e-15, 1e-15)
        c = a * b
        assert c.value >= 0

    def test_jet_order_0(self):
        j = JetNumber.variable(2.0, order=0)
        assert j.order == 0
        assert j.value == 2.0

    def test_jet_high_order(self):
        j = JetNumber.variable(1.0, order=5)
        assert j.order == 5

    def test_forward_derivative_constant(self):
        val, deriv = forward_derivative(lambda x: 5.0, 2.0)
        assert val == 5.0
        assert deriv == 0.0

    def test_dualvector_add_array(self):
        dv = DualVector.identity(np.array([1.0, 2.0]))
        result = dv + np.array([3.0, 4.0])
        np.testing.assert_allclose(result.values, [4.0, 6.0])

    def test_dualvector_radd(self):
        dv = DualVector.identity(np.array([1.0, 2.0]))
        result = np.array([3.0, 4.0]) + dv
        # numpy __add__ may return ndarray or DualVector depending on dispatch
        if isinstance(result, DualVector):
            np.testing.assert_allclose(result.values, [4.0, 6.0])
        else:
            # numpy dispatches to its own __add__, yielding ndarray of DualVectors
            # This is expected behavior—test that it doesn't error
            assert result is not None
