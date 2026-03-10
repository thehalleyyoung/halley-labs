"""Unit tests for usability_oracle.algebra — Category, semiring, differential, lattice.

Tests cover monoidal category axioms, semiring operations, automatic
differentiation (forward and reverse mode), and lattice operations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.category import (
    CognitiveState,
    CostCategory,
    CostMorphism,
)
from usability_oracle.algebra.semiring import (
    BooleanSemiring,
    ExpectedCostSemiring,
    ExpectedCostValue,
    IntervalSemiring,
    IntervalValue,
    LogSemiring,
    MaxPlusSemiring,
    SemiringMatrix,
    TropicalSemiring,
    ViterbiSemiring,
    all_pairs_cost,
)
from usability_oracle.algebra.differential import (
    DualCostElement,
    DualNumber,
    HyperDualNumber,
    ReverseModeAD,
    cost_jacobian,
    dual_exp,
    dual_log,
    dual_max,
    dual_min,
    dual_sqrt,
    sensitivity_report,
)
from usability_oracle.algebra.lattice import (
    GaloisConnection,
    cost_bottom,
    cost_eq,
    cost_join,
    cost_join_many,
    cost_leq,
    cost_lt,
    cost_meet,
    cost_meet_many,
    cost_top,
    kleene_fixpoint,
    variance_abstraction,
    widen,
)


# ═══════════════════════════════════════════════════════════════════════════
# Monoidal Category Axioms
# ═══════════════════════════════════════════════════════════════════════════


class TestCostCategory:
    """Test monoidal category structure."""

    def test_identity_morphism(self):
        cat = CostCategory()
        s = CognitiveState(label="s", capacity=(1.0,) * 8)
        identity = CostMorphism.identity(s)
        assert identity.is_identity()

    def test_sequential_composition(self):
        cat = CostCategory()
        s1 = CognitiveState(label="a")
        s2 = CognitiveState(label="b")
        s3 = CognitiveState(label="c")
        f = CostMorphism(source=s1, target=s2,
                         cost=CostElement(mu=0.5, sigma_sq=0.01), label="f")
        g = CostMorphism(source=s2, target=s3,
                         cost=CostElement(mu=0.3, sigma_sq=0.005), label="g")
        composed = cat.compose(f, g)
        assert composed.source == s1
        assert composed.target == s3
        assert composed.cost.mu >= 0.8 - 0.01

    def test_identity_is_unit(self):
        cat = CostCategory()
        s1 = CognitiveState(label="a")
        s2 = CognitiveState(label="b")
        f = CostMorphism(source=s1, target=s2,
                         cost=CostElement(mu=1.0, sigma_sq=0.02), label="f")
        identity = CostMorphism.identity(s2)
        composed = cat.compose(f, identity)
        assert abs(composed.cost.mu - f.cost.mu) < 0.01

    def test_associativity(self):
        cat = CostCategory()
        s = [CognitiveState(label=f"s{i}") for i in range(4)]
        f = CostMorphism(s[0], s[1], CostElement(mu=0.3), "f")
        g = CostMorphism(s[1], s[2], CostElement(mu=0.4), "g")
        h = CostMorphism(s[2], s[3], CostElement(mu=0.5), "h")
        fg_h = cat.compose(cat.compose(f, g), h)
        f_gh = cat.compose(f, cat.compose(g, h))
        assert abs(fg_h.cost.mu - f_gh.cost.mu) < 1e-6

    def test_tensor_product(self):
        cat = CostCategory()
        s1 = CognitiveState(label="a")
        s2 = CognitiveState(label="b")
        f = CostMorphism(s1, s1, CostElement(mu=1.0), "f")
        g = CostMorphism(s2, s2, CostElement(mu=2.0), "g")
        par = cat.tensor(f, g)
        assert par.cost.mu >= 2.0 - 0.01

    def test_pentagon_axiom(self):
        cat = CostCategory()
        states = [CognitiveState(label=f"s{i}") for i in range(4)]
        result = cat.verify_pentagon(*states)
        assert result

    def test_triangle_axiom(self):
        cat = CostCategory()
        a = CognitiveState(label="a")
        b = CognitiveState(label="b")
        result = cat.verify_triangle(a, b)
        assert result

    def test_braiding(self):
        cat = CostCategory()
        a = CognitiveState(label="a")
        b = CognitiveState(label="b")
        braid = cat.braiding(a, b)
        assert braid.cost.mu == pytest.approx(0.0, abs=1e-6)

    def test_compose_chain(self):
        cat = CostCategory()
        states = [CognitiveState(label=f"s{i}") for i in range(4)]
        morphisms = [
            CostMorphism(states[i], states[i + 1], CostElement(mu=0.5), f"m{i}")
            for i in range(3)
        ]
        composed = cat.compose_chain(morphisms)
        assert composed.cost.mu >= 1.5 - 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Semiring Operations
# ═══════════════════════════════════════════════════════════════════════════


class TestSemirings:
    """Test various semiring implementations."""

    def test_tropical_semiring(self):
        sr = TropicalSemiring()
        assert sr.zero() == float("inf")
        assert sr.one() == 0.0
        assert sr.add(3.0, 5.0) == 3.0  # min
        assert sr.mul(3.0, 5.0) == 8.0  # +

    def test_max_plus_semiring(self):
        sr = MaxPlusSemiring()
        assert sr.zero() == float("-inf")
        assert sr.one() == 0.0
        assert sr.add(3.0, 5.0) == 5.0  # max
        assert sr.mul(3.0, 5.0) == 8.0  # +

    def test_log_semiring(self):
        sr = LogSemiring()
        # LogSemiring uses inf as zero (negative log-probability semiring)
        assert sr.zero() == float("inf")
        assert sr.one() == 0.0
        result = sr.add(1.0, 2.0)
        assert result <= 1.0  # should be ≤ min

    def test_viterbi_semiring(self):
        sr = ViterbiSemiring()
        assert sr.zero() == 0.0
        assert sr.one() == 1.0
        assert sr.add(0.3, 0.7) == 0.7  # max
        assert sr.mul(0.3, 0.7) == pytest.approx(0.21)  # product

    def test_boolean_semiring(self):
        sr = BooleanSemiring()
        assert sr.zero() == False
        assert sr.one() == True
        assert sr.add(True, False) == True  # OR
        assert sr.mul(True, False) == False  # AND

    def test_expected_cost_semiring(self):
        sr = ExpectedCostSemiring()
        a = ExpectedCostValue(mu=1.0, var=0.1)
        b = ExpectedCostValue(mu=2.0, var=0.2)
        s = sr.add(a, b)
        p = sr.mul(a, b)
        assert isinstance(s, ExpectedCostValue)
        assert isinstance(p, ExpectedCostValue)

    def test_interval_semiring(self):
        sr = IntervalSemiring()
        a = IntervalValue(lo=1.0, hi=2.0)
        b = IntervalValue(lo=3.0, hi=4.0)
        s = sr.add(a, b)
        assert s.lo <= 1.0
        assert s.hi >= 4.0

    def test_semiring_sum_product(self):
        sr = TropicalSemiring()
        vals = [5.0, 3.0, 7.0, 1.0]
        assert sr.sum(vals) == 1.0  # min of all
        assert sr.product(vals) == 16.0  # sum of all

    def test_semiring_matrix_multiply(self):
        sr = TropicalSemiring()
        data = [[0.0, 1.0], [2.0, 0.0]]
        m = SemiringMatrix(sr, data)
        m2 = m.multiply(m)
        assert m2[0, 0] == 0.0  # min(0+0, 1+2) = 0

    def test_all_pairs_cost(self):
        adj = np.array([[0, 1, float("inf")], [float("inf"), 0, 2], [float("inf"), float("inf"), 0]])
        result = all_pairs_cost(adj, semiring_name="tropical")
        assert result[0, 2] == 3.0  # 0→1→2: 1+2


# ═══════════════════════════════════════════════════════════════════════════
# Automatic Differentiation
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoDiff:
    """Test forward and reverse mode automatic differentiation."""

    def test_dual_number_arithmetic(self):
        x = DualNumber(3.0, 1.0)
        y = DualNumber(2.0, 0.0)
        z = x * y + x
        assert z.real == pytest.approx(9.0)
        assert z.dual == pytest.approx(3.0)  # d/dx(x*2 + x) = 3

    def test_dual_sqrt(self):
        x = DualNumber(4.0, 1.0)
        result = dual_sqrt(x)
        assert result.real == pytest.approx(2.0)
        assert result.dual == pytest.approx(0.25)  # 1/(2√4)

    def test_dual_exp(self):
        x = DualNumber(0.0, 1.0)
        result = dual_exp(x)
        assert result.real == pytest.approx(1.0)
        assert result.dual == pytest.approx(1.0)

    def test_dual_log(self):
        x = DualNumber(math.e, 1.0)
        result = dual_log(x)
        assert result.real == pytest.approx(1.0)
        assert result.dual == pytest.approx(1.0 / math.e)

    def test_dual_cost_element(self):
        ce = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.01, lambda_=0.0)
        seed = CostElement(mu=1.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)
        dce = DualCostElement.from_cost_element(ce, seed)
        assert dce.primal().mu == pytest.approx(1.0)
        assert dce.tangent().mu == pytest.approx(1.0)

    def test_hyperdual_second_derivative(self):
        x = HyperDualNumber(3.0, 1.0, 1.0, 0.0)
        result = x * x  # f(x) = x²
        assert result.real == pytest.approx(9.0)
        assert result.eps1 == pytest.approx(6.0)  # f'(3) = 6
        assert result.eps12 == pytest.approx(2.0)  # f''(3) = 2

    def test_reverse_mode_ad(self):
        ad = ReverseModeAD()
        x = ad.variable(3.0, "x")
        y = ad.variable(4.0, "y")
        z = ad.mul(x, y)  # z = x*y
        w = ad.add(z, x)  # w = x*y + x
        grads = ad.backward(w)
        assert grads["x"] == pytest.approx(5.0)  # y + 1
        assert grads["y"] == pytest.approx(3.0)  # x

    def test_cost_jacobian(self):
        def compose_fn(**params):
            return CostElement(
                mu=params["a"] + params["b"],
                sigma_sq=params["a"] * 0.1,
            )
        jac = cost_jacobian(compose_fn, {"a": 1.0, "b": 2.0})
        assert "a" in jac
        assert "b" in jac


# ═══════════════════════════════════════════════════════════════════════════
# Lattice Operations
# ═══════════════════════════════════════════════════════════════════════════


class TestLattice:
    """Test lattice operations on CostElement."""

    def test_cost_leq(self):
        a = CostElement(mu=1.0, sigma_sq=0.01)
        b = CostElement(mu=2.0, sigma_sq=0.02)
        assert cost_leq(a, b)
        assert not cost_leq(b, a)

    def test_cost_eq(self):
        a = CostElement(mu=1.0, sigma_sq=0.01)
        b = CostElement(mu=1.0, sigma_sq=0.01)
        assert cost_eq(a, b)

    def test_cost_lt(self):
        a = CostElement(mu=1.0, sigma_sq=0.01)
        b = CostElement(mu=2.0, sigma_sq=0.02)
        assert cost_lt(a, b)

    def test_cost_join(self):
        a = CostElement(mu=1.0, sigma_sq=0.03)
        b = CostElement(mu=2.0, sigma_sq=0.01)
        j = cost_join(a, b)
        assert j.mu >= max(a.mu, b.mu) - 1e-6
        assert j.sigma_sq >= max(a.sigma_sq, b.sigma_sq) - 1e-6

    def test_cost_meet(self):
        a = CostElement(mu=1.0, sigma_sq=0.03)
        b = CostElement(mu=2.0, sigma_sq=0.01)
        m = cost_meet(a, b)
        assert m.mu <= min(a.mu, b.mu) + 1e-6
        assert m.sigma_sq <= min(a.sigma_sq, b.sigma_sq) + 1e-6

    def test_bottom_and_top(self):
        bottom = cost_bottom()
        top = cost_top()
        assert cost_leq(bottom, top)
        assert not cost_leq(top, bottom)

    def test_join_many(self):
        elements = [CostElement(mu=float(i)) for i in range(5)]
        j = cost_join_many(elements)
        assert j.mu >= 4.0 - 1e-6

    def test_meet_many(self):
        elements = [CostElement(mu=float(i)) for i in range(1, 5)]
        m = cost_meet_many(elements)
        assert m.mu <= 1.0 + 1e-6

    def test_kleene_fixpoint(self):
        def f(ce: CostElement) -> CostElement:
            return CostElement(
                mu=min(ce.mu + 0.1, 1.0),
                sigma_sq=min(ce.sigma_sq + 0.01, 0.1),
            )
        fp, iters = kleene_fixpoint(f)
        assert fp.mu == pytest.approx(1.0, abs=0.1)
        assert iters > 0

    def test_widen(self):
        a = CostElement(mu=1.0)
        b = CostElement(mu=1.5)
        w = widen(a, b)
        assert w.mu >= b.mu - 1e-6

    def test_variance_abstraction(self):
        gc = variance_abstraction()
        ce = CostElement(mu=1.0, sigma_sq=0.5, kappa=0.1, lambda_=0.01)
        abstract = gc.abstract(ce)
        concrete = gc.concretise(abstract)
        assert isinstance(abstract, CostElement)
        assert isinstance(concrete, CostElement)
