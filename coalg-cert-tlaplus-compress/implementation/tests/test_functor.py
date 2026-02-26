"""Comprehensive tests for the coacert.functor module."""

import pytest
from coacert.functor.polynomial import (
    ConstantFunctor,
    PowersetFunctor,
    ExponentialFunctor,
    ProductFunctor,
    CoproductFunctor,
    CoproductValue,
    FairnessFunctor,
    FairnessValue,
    CompositeFunctor,
    NaturalTransformation,
    KripkeFairnessFunctor,
)
from coacert.functor.coalgebra import (
    FCoalgebra,
    QuotientCoalgebra,
    SubCoalgebra,
    ProductCoalgebra,
    CoalgebraState,
)
from coacert.functor.stutter import (
    StutterPath,
    StutterMonad,
    StutterEquivalenceClass,
    are_stutter_equivalent,
)
from coacert.functor.tfair_coherence import (
    TFairCoherenceChecker,
    CoherenceWitness,
    CoherenceViolation,
)
from coacert.functor.morphism import (
    CoalgebraMorphism,
    MorphismComposition,
)
from coacert.functor.behavioral_equiv import (
    BehavioralEquivalence,
    PartitionRefinement,
    EquivalenceClass,
)


# ── helpers ───────────────────────────────────────────────────────────────

def _simple_coalgebra():
    """Two-state coalgebra: s0 -a-> s1, s1 -a-> s0, s1 -a-> s1."""
    c = FCoalgebra(name="simple", actions={"a"})
    c.add_state("s0", propositions={"p"}, is_initial=True)
    c.add_state("s1", propositions={"q"})
    c.add_transition("s0", "a", "s1")
    c.add_transition("s1", "a", "s0")
    c.add_transition("s1", "a", "s1")
    return c


def _diamond_coalgebra():
    """Diamond: s0 -> {s1, s2}, s1 -> s3, s2 -> s3."""
    c = FCoalgebra(name="diamond", actions={"a"})
    c.add_state("s0", propositions={"p"}, is_initial=True)
    c.add_state("s1", propositions={"q"})
    c.add_state("s2", propositions={"q"})
    c.add_state("s3", propositions={"r"})
    c.add_transition("s0", "a", "s1")
    c.add_transition("s0", "a", "s2")
    c.add_transition("s1", "a", "s3")
    c.add_transition("s2", "a", "s3")
    return c


def _deadlock_coalgebra():
    """s0 -a-> s1, s1 has no transitions (deadlock)."""
    c = FCoalgebra(name="deadlock", actions={"a"})
    c.add_state("s0", propositions={"p"}, is_initial=True)
    c.add_state("s1", propositions={"q"})
    c.add_transition("s0", "a", "s1")
    return c


# ═════════════════════════════════════════════════════════════════════════
# 1. TestFCoalgebra
# ═════════════════════════════════════════════════════════════════════════

class TestFCoalgebra:

    def test_construction(self):
        c = _simple_coalgebra()
        assert c.state_count == 2
        assert c.states == frozenset({"s0", "s1"})
        assert c.name == "simple"

    def test_initial_states(self):
        c = _simple_coalgebra()
        assert c.initial_states == frozenset({"s0"})

    def test_successors(self):
        c = _simple_coalgebra()
        assert c.successors("s0", "a") == frozenset({"s1"})
        assert c.successors("s1", "a") == frozenset({"s0", "s1"})

    def test_all_successors(self):
        c = _simple_coalgebra()
        assert c.all_successors("s1") == frozenset({"s0", "s1"})

    def test_predecessors(self):
        c = _simple_coalgebra()
        preds = c.predecessors("s1")
        assert "s0" in preds.get("a", set())
        assert "s1" in preds.get("a", set())

    def test_apply_functor(self):
        c = _simple_coalgebra()
        fv = c.apply_functor("s0")
        assert fv.propositions == frozenset({"p"})
        assert fv.successor_set("a") == frozenset({"s1"})

    def test_labels(self):
        c = _simple_coalgebra()
        fv0 = c.apply_functor("s0")
        fv1 = c.apply_functor("s1")
        assert fv0.propositions == frozenset({"p"})
        assert fv1.propositions == frozenset({"q"})

    def test_deadlock_detection(self):
        c = _deadlock_coalgebra()
        assert c.deadlock_states() == frozenset({"s1"})

    def test_no_deadlocks(self):
        c = _simple_coalgebra()
        assert c.deadlock_states() == frozenset()

    def test_reachability(self):
        c = _diamond_coalgebra()
        reachable = c.reachable_states({"s0"})
        assert reachable == frozenset({"s0", "s1", "s2", "s3"})

    def test_reachability_partial(self):
        c = _diamond_coalgebra()
        reachable = c.reachable_states({"s1"})
        assert reachable == frozenset({"s1", "s3"})

    def test_is_deterministic_false(self):
        c = _simple_coalgebra()
        assert c.is_deterministic() is False

    def test_is_deterministic_true(self):
        c = _deadlock_coalgebra()
        assert c.is_deterministic() is True

    def test_structure_map(self):
        c = _simple_coalgebra()
        sm = c.structure_map()
        assert set(sm.keys()) == {"s0", "s1"}

    def test_add_fairness_constraint(self):
        c = _simple_coalgebra()
        idx = c.add_fairness_constraint({"s0"}, {"s1"})
        assert idx == 0
        assert len(c.fairness_constraints) == 1

    def test_restrict_to(self):
        c = _diamond_coalgebra()
        sub = c.restrict_to(frozenset({"s0", "s1"}))
        assert sub.states == frozenset({"s0", "s1"})
        # s1's successors restricted to subset
        assert sub.successors("s1", "a") <= frozenset({"s0", "s1"})


# ═════════════════════════════════════════════════════════════════════════
# 2. TestPowersetFunctor
# ═════════════════════════════════════════════════════════════════════════

class TestPowersetFunctor:
    def setup_method(self):
        self.pf = PowersetFunctor()

    def test_apply_small(self):
        carrier = frozenset({"a", "b"})
        ps = self.pf.apply(carrier)
        assert len(ps) == 4  # 2^2
        assert frozenset() in ps
        assert frozenset({"a"}) in ps
        assert frozenset({"a", "b"}) in ps

    def test_apply_empty(self):
        ps = self.pf.apply(frozenset())
        assert ps == frozenset({frozenset()})

    def test_fmap_identity(self):
        subset = frozenset({"a", "b"})
        source = frozenset({"a", "b", "c"})
        identity = {"a": "a", "b": "b", "c": "c"}
        assert self.pf.fmap(identity, subset, source) == subset

    def test_fmap_collapse(self):
        subset = frozenset({"a", "b"})
        source = frozenset({"a", "b"})
        f = {"a": "x", "b": "x"}
        result = self.pf.fmap(f, subset, source)
        assert result == frozenset({"x"})

    def test_fmap_disjoint(self):
        subset = frozenset({"a"})
        source = frozenset({"a", "b"})
        f = {"a": "x", "b": "y"}
        assert self.pf.fmap(f, subset, source) == frozenset({"x"})

    def test_fmap_type_error(self):
        with pytest.raises(TypeError):
            self.pf.fmap({}, "not_a_set", frozenset())

    def test_signature(self):
        assert "P(X)" in self.pf.signature()

    def test_contains(self):
        carrier = frozenset({"a", "b", "c"})
        assert self.pf.contains(frozenset({"a"}), carrier) is True
        assert self.pf.contains(frozenset({"d"}), carrier) is False

    def test_image(self):
        f = {"a": "x", "b": "y"}
        subsets = frozenset({frozenset({"a"}), frozenset({"b"})})
        result = self.pf.image(f, subsets)
        assert frozenset({"x"}) in result
        assert frozenset({"y"}) in result

    def test_preimage(self):
        f = {"a": "x", "b": "x", "c": "y"}
        source = frozenset({"a", "b", "c"})
        pre = self.pf.preimage(f, frozenset({"x"}), source)
        assert pre == frozenset({"a", "b"})

    @pytest.mark.parametrize("size", [0, 1, 3, 5])
    def test_powerset_size(self, size):
        carrier = frozenset(str(i) for i in range(size))
        ps = self.pf.apply(carrier)
        assert len(ps) == 2 ** size


# ═════════════════════════════════════════════════════════════════════════
# 3. TestExponentialFunctor
# ═════════════════════════════════════════════════════════════════════════

class TestExponentialFunctor:
    def setup_method(self):
        self.ef = ExponentialFunctor(frozenset({"a", "b"}))

    def test_apply(self):
        result = self.ef.apply(frozenset({"s0", "s1"}))
        assert isinstance(result, str)  # description

    def test_fmap_identity(self):
        val = {"a": frozenset({"s0"}), "b": frozenset({"s1"})}
        source = frozenset({"s0", "s1"})
        identity = {"s0": "s0", "s1": "s1"}
        assert self.ef.fmap(identity, val, source) == val

    def test_fmap_remap(self):
        val = {"a": frozenset({"s0"}), "b": frozenset({"s1"})}
        source = frozenset({"s0", "s1"})
        f = {"s0": "t0", "s1": "t1"}
        result = self.ef.fmap(f, val, source)
        assert result["a"] == frozenset({"t0"})
        assert result["b"] == frozenset({"t1"})

    def test_fmap_collapse_successors(self):
        val = {"a": frozenset({"s0", "s1"}), "b": frozenset()}
        source = frozenset({"s0", "s1"})
        f = {"s0": "t", "s1": "t"}
        result = self.ef.fmap(f, val, source)
        assert result["a"] == frozenset({"t"})

    def test_is_total(self):
        val_total = {"a": frozenset({"s0"}), "b": frozenset({"s1"})}
        val_partial = {"a": frozenset({"s0"})}
        assert self.ef.is_total(val_total) is True
        assert self.ef.is_total(val_partial) is False


# ═════════════════════════════════════════════════════════════════════════
# 4. TestProductFunctor
# ═════════════════════════════════════════════════════════════════════════

class TestProductFunctor:
    def setup_method(self):
        self.left = ConstantFunctor(frozenset({"p", "q"}), label="AP")
        self.right = PowersetFunctor()
        self.pf = ProductFunctor(self.left, self.right)

    def test_apply(self):
        carrier = frozenset({"s0"})
        result = self.pf.apply(carrier)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == frozenset({"p", "q"})  # constant part

    def test_fmap(self):
        source = frozenset({"s0", "s1"})
        f = {"s0": "t0", "s1": "t1"}
        val = (frozenset({"p"}), frozenset({"s0"}))
        result = self.pf.fmap(f, val, source)
        assert result[0] == frozenset({"p"})  # constant unchanged
        assert result[1] == frozenset({"t0"})  # powerset mapped

    def test_project_left(self):
        val = ("left_val", "right_val")
        assert self.pf.project_left(val) == "left_val"

    def test_project_right(self):
        val = ("left_val", "right_val")
        assert self.pf.project_right(val) == "right_val"

    def test_pair(self):
        assert self.pf.pair("a", "b") == ("a", "b")

    def test_signature(self):
        sig = self.pf.signature()
        assert "×" in sig


# ═════════════════════════════════════════════════════════════════════════
# 5. TestCoproductFunctor
# ═════════════════════════════════════════════════════════════════════════

class TestCoproductFunctor:
    def setup_method(self):
        self.left = ConstantFunctor(frozenset({"a"}))
        self.right = PowersetFunctor()
        self.cf = CoproductFunctor(self.left, self.right)

    def test_inject_left(self):
        cv = self.cf.inject_left("val")
        assert cv.tag == "left"
        assert cv.value == "val"

    def test_inject_right(self):
        cv = self.cf.inject_right("val")
        assert cv.tag == "right"
        assert cv.value == "val"

    def test_fmap_left(self):
        source = frozenset({"s0"})
        f = {"s0": "t0"}
        cv = self.cf.inject_left(frozenset({"a"}))
        result = self.cf.fmap(f, cv, source)
        assert result.tag == "left"
        assert result.value == frozenset({"a"})  # constant

    def test_fmap_right(self):
        source = frozenset({"s0"})
        f = {"s0": "t0"}
        cv = self.cf.inject_right(frozenset({"s0"}))
        result = self.cf.fmap(f, cv, source)
        assert result.tag == "right"
        assert result.value == frozenset({"t0"})

    def test_case_left(self):
        cv = self.cf.inject_left(42)
        result = self.cf.case(cv, lambda x: x + 1, lambda x: x - 1)
        assert result == 43

    def test_case_right(self):
        cv = self.cf.inject_right(42)
        result = self.cf.case(cv, lambda x: x + 1, lambda x: x - 1)
        assert result == 41

    def test_signature(self):
        assert "+" in self.cf.signature()

    def test_coproduct_value_frozen(self):
        cv = CoproductValue("left", 10)
        with pytest.raises(AttributeError):
            cv.tag = "right"


# ═════════════════════════════════════════════════════════════════════════
# 6. TestFairnessFunctor
# ═════════════════════════════════════════════════════════════════════════

class TestFairnessFunctor:
    def setup_method(self):
        self.ff = FairnessFunctor(num_pairs=2)

    def test_apply(self):
        result = self.ff.apply(frozenset({"s0", "s1"}))
        assert isinstance(result, str)
        assert "2" in result

    def test_empty_value(self):
        ev = self.ff.empty_value()
        assert ev.pair_count() == 2
        assert ev.b_set(0) == frozenset()
        assert ev.g_set(0) == frozenset()

    def test_from_sets(self):
        fv = self.ff.from_sets([
            ({"s0"}, {"s1"}),
            ({"s0", "s1"}, {"s1"}),
        ])
        assert fv.pair_count() == 2
        assert fv.b_set(0) == frozenset({"s0"})
        assert fv.g_set(1) == frozenset({"s1"})

    def test_from_sets_wrong_count(self):
        with pytest.raises(ValueError):
            self.ff.from_sets([({"s0"}, {"s1"})])

    def test_fmap(self):
        fv = self.ff.from_sets([
            ({"s0"}, {"s1"}),
            ({"s0", "s1"}, set()),
        ])
        f = {"s0": "t0", "s1": "t1"}
        source = frozenset({"s0", "s1"})
        result = self.ff.fmap(f, fv, source)
        assert result.b_set(0) == frozenset({"t0"})
        assert result.g_set(0) == frozenset({"t1"})

    def test_preserves_acceptance(self):
        fv = self.ff.from_sets([
            ({"s0"}, {"s1"}),
            (set(), set()),
        ])
        f = {"s0": "t0", "s1": "t1"}
        source = frozenset({"s0", "s1"})
        assert self.ff.preserves_acceptance(f, fv, source) is True

    def test_merge(self):
        fv1 = self.ff.from_sets([({"a"}, {"b"}), (set(), set())])
        fv2 = self.ff.from_sets([({"c"}, set()), ({"d"}, {"e"})])
        merged = self.ff.merge(fv1, fv2)
        assert merged.b_set(0) == frozenset({"a", "c"})
        assert merged.g_set(0) == frozenset({"b"})
        assert merged.b_set(1) == frozenset({"d"})

    @pytest.mark.parametrize("n", [0, 1, 3])
    def test_empty_value_pair_count(self, n):
        ff = FairnessFunctor(num_pairs=n)
        assert ff.empty_value().pair_count() == n


# ═════════════════════════════════════════════════════════════════════════
# 7. TestCompositeFunctor
# ═════════════════════════════════════════════════════════════════════════

class TestCompositeFunctor:
    def test_apply_powerset_of_powerset(self):
        inner = PowersetFunctor(label="P1")
        outer = PowersetFunctor(label="P2")
        comp = CompositeFunctor(outer, inner)
        carrier = frozenset({"a"})
        inner_result = inner.apply(carrier)
        assert isinstance(inner_result, frozenset)
        result = comp.apply(carrier)
        assert isinstance(result, frozenset)

    def test_fmap_composite(self):
        inner = PowersetFunctor()
        outer = PowersetFunctor()
        comp = CompositeFunctor(outer, inner)
        source = frozenset({"a", "b"})
        f = {"a": "x", "b": "y"}
        # value is a subset (inner powerset value)
        val = frozenset({"a"})
        result = comp.fmap(f, val, source)
        assert frozenset({"x"}) == result  # P(P(f)) on a subset

    def test_signature(self):
        inner = PowersetFunctor()
        outer = ConstantFunctor(frozenset({"c"}))
        comp = CompositeFunctor(outer, inner)
        assert "∘" in comp.signature()


# ═════════════════════════════════════════════════════════════════════════
# 8. TestStutterMonad
# ═════════════════════════════════════════════════════════════════════════

class TestStutterMonad:
    def _load_simple_monad(self):
        """s0 and s1 share label {p} and mutually reach each other + s2.
        This makes s0 and s1 stutter-equivalent."""
        monad = StutterMonad()
        states = {"s0", "s1", "s2"}
        labels = {
            "s0": frozenset({"p"}),
            "s1": frozenset({"p"}),
            "s2": frozenset({"q"}),
        }
        transitions = {
            "s0": {"a": {"s1", "s2"}},
            "s1": {"a": {"s0", "s2"}},
            "s2": {"a": {"s2"}},
        }
        monad.load_system(states, labels, transitions, {"a"})
        return monad

    def test_unit(self):
        monad = self._load_simple_monad()
        # s0 and s1 have same label so may be merged
        eta_s0 = monad.unit("s0")
        eta_s1 = monad.unit("s1")
        assert eta_s0 == eta_s1  # same equivalence class

    def test_unit_distinct_labels(self):
        monad = self._load_simple_monad()
        eta_s0 = monad.unit("s0")
        eta_s2 = monad.unit("s2")
        assert eta_s0 != eta_s2

    def test_left_unit_law(self):
        monad = self._load_simple_monad()
        ok, violations = monad.verify_left_unit_law()
        assert ok, f"Left unit law failed: {violations}"

    def test_right_unit_law(self):
        monad = self._load_simple_monad()
        ok, violations = monad.verify_right_unit_law()
        assert ok, f"Right unit law failed: {violations}"

    def test_associativity_law(self):
        monad = self._load_simple_monad()
        ok, violations = monad.verify_associativity()
        assert ok, f"Associativity failed: {violations}"

    def test_all_laws(self):
        monad = self._load_simple_monad()
        ok, results = monad.verify_all_laws()
        assert ok, f"Monad laws failed: {results}"

    def test_stutter_equivalence_classes(self):
        monad = self._load_simple_monad()
        classes = monad.compute_stutter_equivalence_classes()
        assert len(classes) >= 1
        all_members = set()
        for cls in classes:
            all_members |= cls.members
        assert all_members == {"s0", "s1", "s2"}

    def test_stutter_trace_equivalent(self):
        monad = self._load_simple_monad()
        assert monad.are_stutter_trace_equivalent("s0", "s1") is True
        assert monad.are_stutter_trace_equivalent("s0", "s2") is False


# ═════════════════════════════════════════════════════════════════════════
# 9. TestStutterPath
# ═════════════════════════════════════════════════════════════════════════

class TestStutterPath:
    def test_length(self):
        p = StutterPath(states=("a", "b", "c"))
        assert p.length == 3

    def test_empty(self):
        p = StutterPath(states=())
        assert p.is_empty is True
        assert p.length == 0

    def test_stutter_free(self):
        p = StutterPath(states=("a", "b", "c"))
        assert p.is_stutter_free() is True

    def test_stutter_count(self):
        p = StutterPath(states=("a", "a", "b", "b", "b"))
        assert p.stutter_count() == 3

    def test_stutter_free_core(self):
        p = StutterPath(states=("a", "a", "b", "b", "c"))
        core = p.stutter_free_core()
        assert core.states == ("a", "b", "c")

    def test_stutter_free_core_idempotent(self):
        p = StutterPath(states=("a", "b", "c"))
        assert p.stutter_free_core() == p

    def test_blocks(self):
        p = StutterPath(states=("a", "a", "b", "c", "c", "c"))
        blocks = p.blocks()
        assert blocks == [("a", 2), ("b", 1), ("c", 3)]

    def test_stutter_equivalence(self):
        p1 = StutterPath(states=("a", "b", "c"))
        p2 = StutterPath(states=("a", "a", "b", "c", "c"))
        assert are_stutter_equivalent(p1, p2) is True

    def test_stutter_non_equivalence(self):
        p1 = StutterPath(states=("a", "b"))
        p2 = StutterPath(states=("a", "c"))
        assert are_stutter_equivalent(p1, p2) is False

    def test_extend_at(self):
        p = StutterPath(states=("a", "b"))
        extended = p.extend_at(0, 2)
        assert extended.states == ("a", "a", "a", "b")

    def test_contract_at(self):
        p = StutterPath(states=("a", "a", "b"))
        contracted = p.contract_at(1)
        assert contracted.states == ("a", "b")

    def test_contract_non_stutter(self):
        p = StutterPath(states=("a", "b", "c"))
        # contracting at non-stutter position returns same path
        result = p.contract_at(1)
        assert result == p

    def test_concatenation(self):
        p1 = StutterPath(states=("a", "b"))
        p2 = StutterPath(states=("c", "d"))
        combined = p1 + p2
        assert combined.states == ("a", "b", "c", "d")

    @pytest.mark.parametrize(
        "states,expected_count",
        [
            (("a",), 0),
            (("a", "a"), 1),
            (("a", "b", "a"), 0),
            (("a", "a", "a"), 2),
        ],
    )
    def test_stutter_count_parametrized(self, states, expected_count):
        p = StutterPath(states=states)
        assert p.stutter_count() == expected_count


# ═════════════════════════════════════════════════════════════════════════
# 10. TestTFairCoherence
# ═════════════════════════════════════════════════════════════════════════

class TestTFairCoherence:
    def _build_coherent_system(self):
        """System where fairness sets align with stutter classes.
        s0 and s1 are stutter-equivalent (same label, symmetric transitions)."""
        monad = StutterMonad()
        states = {"s0", "s1", "s2"}
        labels = {
            "s0": frozenset({"p"}),
            "s1": frozenset({"p"}),
            "s2": frozenset({"q"}),
        }
        transitions = {
            "s0": {"a": {"s1", "s2"}},
            "s1": {"a": {"s0", "s2"}},
            "s2": {"a": {"s2"}},
        }
        monad.load_system(states, labels, transitions, {"a"})
        # B = {s0, s1} (full stutter class), G = {s2} (full stutter class)
        fairness_pairs = [({"s0", "s1"}, {"s2"})]
        return monad, fairness_pairs

    def _build_incoherent_system(self):
        """System where fairness sets split a stutter class.
        s0 and s1 are stutter-equivalent but only s0 is in B."""
        monad = StutterMonad()
        states = {"s0", "s1", "s2"}
        labels = {
            "s0": frozenset({"p"}),
            "s1": frozenset({"p"}),
            "s2": frozenset({"q"}),
        }
        transitions = {
            "s0": {"a": {"s1", "s2"}},
            "s1": {"a": {"s0", "s2"}},
            "s2": {"a": {"s2"}},
        }
        monad.load_system(states, labels, transitions, {"a"})
        # B = {s0} splits the {s0, s1} stutter class
        fairness_pairs = [({"s0"}, {"s2"})]
        return monad, fairness_pairs

    def test_coherent_system(self):
        monad, fairness_pairs = self._build_coherent_system()
        checker = TFairCoherenceChecker()
        result = checker.check_coherence(
            coalgebra=None,
            stutter_monad=monad,
            fairness_pairs=fairness_pairs,
        )
        assert result.is_coherent is True
        assert len(result.violations) == 0

    def test_incoherent_system(self):
        monad, fairness_pairs = self._build_incoherent_system()
        checker = TFairCoherenceChecker()
        result = checker.check_coherence(
            coalgebra=None,
            stutter_monad=monad,
            fairness_pairs=fairness_pairs,
        )
        assert result.is_coherent is False
        assert len(result.violations) > 0

    def test_no_fairness_pairs_vacuous(self):
        monad = StutterMonad()
        monad.load_system({"s0"}, {"s0": frozenset()}, {}, set())
        checker = TFairCoherenceChecker()
        result = checker.check_coherence(
            coalgebra=None,
            stutter_monad=monad,
            fairness_pairs=[],
        )
        assert result.is_coherent is True

    def test_violation_has_details(self):
        monad, fairness_pairs = self._build_incoherent_system()
        checker = TFairCoherenceChecker()
        result = checker.check_coherence(
            coalgebra=None,
            stutter_monad=monad,
            fairness_pairs=fairness_pairs,
        )
        for v in result.violations:
            assert v.component in ("B", "G")
            assert v.state_1_member != v.state_2_member


# ═════════════════════════════════════════════════════════════════════════
# 11. TestCoalgebraMorphism
# ═════════════════════════════════════════════════════════════════════════

class TestCoalgebraMorphism:
    def _make_morphism(self):
        return CoalgebraMorphism(
            source_name="A",
            target_name="B",
            mapping={"s0": "t0", "s1": "t1", "s2": "t1"},
        )

    def test_apply(self):
        m = self._make_morphism()
        assert m.apply("s0") == "t0"
        assert m.apply("s2") == "t1"

    def test_domain(self):
        m = self._make_morphism()
        assert m.domain() == frozenset({"s0", "s1", "s2"})

    def test_codomain(self):
        m = self._make_morphism()
        assert m.codomain() == frozenset({"t0", "t1"})

    def test_image(self):
        m = self._make_morphism()
        assert m.image() == frozenset({"t0", "t1"})

    def test_injective_false(self):
        m = self._make_morphism()
        assert m.is_injective() is False

    def test_injective_true(self):
        m = CoalgebraMorphism("A", "B", {"s0": "t0", "s1": "t1"})
        assert m.is_injective() is True

    def test_surjective(self):
        m = self._make_morphism()
        assert m.is_surjective(frozenset({"t0", "t1"})) is True
        assert m.is_surjective(frozenset({"t0", "t1", "t2"})) is False

    def test_kernel(self):
        m = self._make_morphism()
        kernel = m.kernel()
        sizes = sorted(len(k) for k in kernel)
        assert sizes == [1, 2]  # {s0} and {s1, s2}

    def test_fibers(self):
        m = self._make_morphism()
        fibs = m.fibers()
        assert fibs["t0"] == frozenset({"s0"})
        assert fibs["t1"] == frozenset({"s1", "s2"})

    def test_restrict(self):
        m = self._make_morphism()
        r = m.restrict(frozenset({"s0", "s1"}))
        assert r.domain() == frozenset({"s0", "s1"})
        assert "s2" not in r.mapping

    def test_composition(self):
        f = CoalgebraMorphism("A", "B", {"s0": "t0", "s1": "t1"})
        g = CoalgebraMorphism("B", "C", {"t0": "u0", "t1": "u1"})
        composed = MorphismComposition.compose(f, g)
        assert composed.apply("s0") == "u0"
        assert composed.apply("s1") == "u1"
        assert composed.source_name == "A"
        assert composed.target_name == "C"

    def test_identity(self):
        states = frozenset({"s0", "s1"})
        ident = MorphismComposition.identity("A", states)
        assert ident.apply("s0") == "s0"
        assert ident.apply("s1") == "s1"
        assert ident.is_injective() is True


# ═════════════════════════════════════════════════════════════════════════
# 12. TestBehavioralEquivalence
# ═════════════════════════════════════════════════════════════════════════

class TestBehavioralEquivalence:
    def test_partition_refinement_initialize(self):
        pr = PartitionRefinement()
        pr.initialize([{"s0", "s1"}, {"s2"}])
        assert pr.num_blocks == 2
        assert pr.are_equivalent("s0", "s1") is True
        assert pr.are_equivalent("s0", "s2") is False

    def test_partition_refinement_refine_by_signature(self):
        pr = PartitionRefinement()
        pr.initialize([{"s0", "s1", "s2"}])
        # Signature distinguishes s2 from s0, s1
        sig_fn = lambda s, stb: "A" if s in ("s0", "s1") else "B"
        changed = pr.refine_by_signature(sig_fn)
        assert changed is True
        assert pr.num_blocks == 2
        assert pr.are_equivalent("s0", "s1") is True
        assert pr.are_equivalent("s0", "s2") is False

    def test_partition_refinement_no_change(self):
        pr = PartitionRefinement()
        pr.initialize([{"s0"}, {"s1"}])
        sig_fn = lambda s, stb: stb.get(s, -1)
        changed = pr.refine_by_signature(sig_fn)
        assert changed is False

    def test_equivalence_class_contains(self):
        ec = EquivalenceClass(
            representative="s0",
            members=frozenset({"s0", "s1"}),
            propositions=frozenset({"p"}),
        )
        assert ec.contains("s0") is True
        assert ec.contains("s2") is False

    def test_equivalence_class_size(self):
        ec = EquivalenceClass(
            representative="s0",
            members=frozenset({"s0", "s1", "s2"}),
        )
        assert ec.size() == 3

    def test_equivalence_class_singleton(self):
        ec_single = EquivalenceClass(representative="s0", members=frozenset({"s0"}))
        ec_multi = EquivalenceClass(representative="s0", members=frozenset({"s0", "s1"}))
        assert ec_single.is_singleton() is True
        assert ec_multi.is_singleton() is False

    def test_partition_to_equivalence_classes(self):
        pr = PartitionRefinement()
        pr.initialize([{"s0", "s1"}, {"s2"}])
        classes = pr.to_equivalence_classes()
        assert len(classes) == 2

    @pytest.mark.parametrize(
        "s1,s2,expected",
        [("s0", "s1", True), ("s0", "s2", False)],
    )
    def test_partition_are_equivalent(self, s1, s2, expected):
        pr = PartitionRefinement()
        pr.initialize([{"s0", "s1"}, {"s2"}])
        assert pr.are_equivalent(s1, s2) is expected


# ═════════════════════════════════════════════════════════════════════════
# 13. TestQuotientCoalgebra
# ═════════════════════════════════════════════════════════════════════════

class TestQuotientCoalgebra:
    def test_build_quotient(self):
        c = _diamond_coalgebra()
        partition = [
            frozenset({"s0"}),
            frozenset({"s1", "s2"}),
            frozenset({"s3"}),
        ]
        quot, proj = QuotientCoalgebra.build(c, partition)
        assert quot.state_count == 3
        # s1 and s2 merged
        assert proj["s1"] == proj["s2"]
        assert proj["s0"] != proj["s1"]

    def test_quotient_preserves_initial(self):
        c = _diamond_coalgebra()
        partition = [
            frozenset({"s0"}),
            frozenset({"s1", "s2"}),
            frozenset({"s3"}),
        ]
        quot, _ = QuotientCoalgebra.build(c, partition)
        assert "s0" in quot.initial_states

    def test_quotient_transitions(self):
        c = _diamond_coalgebra()
        partition = [
            frozenset({"s0"}),
            frozenset({"s1", "s2"}),
            frozenset({"s3"}),
        ]
        quot, proj = QuotientCoalgebra.build(c, partition)
        rep_12 = proj["s1"]
        # s0 should reach the merged block
        succs = quot.successors("s0", "a")
        assert rep_12 in succs

    def test_quotient_invalid_partition(self):
        c = _simple_coalgebra()
        partition = [frozenset({"s0"})]  # missing s1
        with pytest.raises(ValueError):
            QuotientCoalgebra.build(c, partition)

    def test_quotient_morphism_valid(self):
        c = _diamond_coalgebra()
        partition = [
            frozenset({"s0"}),
            frozenset({"s1", "s2"}),
            frozenset({"s3"}),
        ]
        quot, proj = QuotientCoalgebra.build(c, partition)
        result = QuotientCoalgebra.verify_quotient_morphism(c, quot, proj)
        assert result.is_morphism is True

    def test_sub_coalgebra_extract(self):
        c = _diamond_coalgebra()
        sub = SubCoalgebra.extract(c, {"s1"})
        assert "s1" in sub.states
        assert "s3" in sub.states
        assert "s0" not in sub.states

    def test_sub_coalgebra_check(self):
        c = _diamond_coalgebra()
        sub = SubCoalgebra.extract(c, {"s1"})
        assert SubCoalgebra.is_sub_coalgebra(sub, c) is True

    def test_product_coalgebra(self):
        c1 = FCoalgebra(name="c1", actions={"a"})
        c1.add_state("x", propositions={"p"}, is_initial=True)
        c1.add_transition("x", "a", "x")

        c2 = FCoalgebra(name="c2", actions={"a"})
        c2.add_state("y", propositions={"q"}, is_initial=True)
        c2.add_transition("y", "a", "y")

        prod, pair_map = ProductCoalgebra.build(c1, c2)
        assert prod.state_count >= 1
        assert len(pair_map) == prod.state_count


# ═════════════════════════════════════════════════════════════════════════
# Additional: KripkeFairnessFunctor
# ═════════════════════════════════════════════════════════════════════════

class TestKripkeFairnessFunctor:
    def setup_method(self):
        self.kf = KripkeFairnessFunctor(
            atomic_propositions=frozenset({"p", "q"}),
            actions=frozenset({"a"}),
            num_fairness_pairs=1,
        )

    def test_make_value(self):
        fair = FairnessValue(pairs=((frozenset({"s0"}), frozenset({"s1"})),))
        val = self.kf.make_value(
            frozenset({"p"}),
            {"a": frozenset({"s1"})},
            fair,
        )
        assert val[0] == frozenset({"p"})

    def test_decompose(self):
        fair = FairnessValue(pairs=((frozenset(), frozenset()),))
        val = (frozenset({"p"}), {"a": frozenset({"s0"})}, fair)
        props, succ, f = self.kf.decompose(val)
        assert props == frozenset({"p"})
        assert "a" in succ

    def test_fmap_preserves_props(self):
        fair = FairnessValue(pairs=((frozenset({"s0"}), frozenset()),))
        val = (frozenset({"p"}), {"a": frozenset({"s0"})}, fair)
        f = {"s0": "t0"}
        source = frozenset({"s0"})
        result = self.kf.fmap(f, val, source)
        assert result[0] == frozenset({"p"})
        assert result[1]["a"] == frozenset({"t0"})

    def test_check_value_well_formed(self):
        fair = FairnessValue(pairs=((frozenset({"s0"}), frozenset({"s1"})),))
        val = (frozenset({"p"}), {"a": frozenset({"s1"})}, fair)
        errors = self.kf.check_value_well_formed(val, frozenset({"s0", "s1"}))
        assert errors == []

    def test_check_value_ill_formed(self):
        fair = FairnessValue(pairs=((frozenset({"s0"}), frozenset({"s_bad"})),))
        val = (frozenset({"p"}), {"a": frozenset({"s1"})}, fair)
        errors = self.kf.check_value_well_formed(val, frozenset({"s0", "s1"}))
        assert len(errors) > 0  # s_bad not in carrier


# ═════════════════════════════════════════════════════════════════════════
# Additional: NaturalTransformation
# ═════════════════════════════════════════════════════════════════════════

class TestNaturalTransformation:
    def test_apply_generic(self):
        src = PowersetFunctor()
        tgt = PowersetFunctor()
        nt = NaturalTransformation(src, tgt, name="id")
        nt.set_generic_component(lambda v: v)  # identity
        result = nt.apply("any_carrier", frozenset({"a"}))
        assert result == frozenset({"a"})

    def test_compose(self):
        src = PowersetFunctor()
        mid = PowersetFunctor()
        tgt = PowersetFunctor()
        alpha = NaturalTransformation(src, mid, name="α")
        alpha.set_generic_component(lambda v: v)
        beta = NaturalTransformation(mid, tgt, name="β")
        beta.set_generic_component(lambda v: v)
        composed = alpha.compose(beta)
        assert "β" in composed.name and "α" in composed.name

    def test_apply_specific_component(self):
        src = PowersetFunctor()
        tgt = ConstantFunctor(frozenset({"c"}))
        nt = NaturalTransformation(
            src, tgt,
            components={"X": lambda v: frozenset({"c"})},
        )
        result = nt.apply("X", frozenset({"a", "b"}))
        assert result == frozenset({"c"})

    def test_apply_missing_component(self):
        src = PowersetFunctor()
        tgt = PowersetFunctor()
        nt = NaturalTransformation(src, tgt)
        with pytest.raises(KeyError):
            nt.apply("missing", frozenset())


# ═════════════════════════════════════════════════════════════════════════
# ConstantFunctor
# ═════════════════════════════════════════════════════════════════════════

class TestConstantFunctor:
    def test_apply_ignores_carrier(self):
        cf = ConstantFunctor(frozenset({"a", "b"}))
        assert cf.apply(frozenset({"x"})) == frozenset({"a", "b"})
        assert cf.apply(frozenset()) == frozenset({"a", "b"})

    def test_fmap_identity_on_values(self):
        cf = ConstantFunctor(frozenset({"a"}))
        val = frozenset({"a"})
        result = cf.fmap({"x": "y"}, val, frozenset({"x"}))
        assert result == val  # constant: unchanged

    def test_element_count(self):
        cf = ConstantFunctor(frozenset({"a", "b", "c"}))
        assert cf.element_count() == 3
