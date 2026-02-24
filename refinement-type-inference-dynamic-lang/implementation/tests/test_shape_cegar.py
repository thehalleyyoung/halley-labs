"""
Tests for the Shape Contract Discovery (counterexample-guided, CEGAR-style)
module.

Covers:
  - Basic contract discovery loop convergence
  - Predicate discovery from counterexamples
  - Integration with the constraint verifier
  - Cases where contract discovery finds real bugs vs spurious warnings
  - Trace-back engine
  - Predicate deduplication and refinement
  - Contract inference
"""

from __future__ import annotations

import pytest

from src.shape_cegar import (
    ShapeCEGARLoop,
    ShapeCEGARResult,
    CEGARStatus,
    ShapePredicate,
    PredicateKind,
    InferredContract,
    CounterexampleClassification,
    AnalysedCounterexample,
    CounterexampleAnalyser,
    TraceBackEngine,
    ShapeRefinement,
    PredicateSet,
    Z3CounterexampleExtractor,
    IterationRecord,
    run_shape_cegar,
    verify_and_discover,
    infer_contracts,
)
from src.tensor_shapes import TensorShape, ShapeDim, ShapeEnv
from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    extract_computation_graph,
    verify_model,
    OpKind,
    LayerKind,
    LayerDef,
    SafetyViolation,
    CounterexampleTrace,
    ModelState,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures: source-code snippets
# ═══════════════════════════════════════════════════════════════════════════════

SIMPLE_LINEAR = """\
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""

TWO_LAYER_MLP = """\
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

SHAPE_MISMATCH_MODEL = """\
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

TRANSFORMER_BLOCK = """\
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

CONV_MODEL = """\
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Basic CEGAR loop convergence
# ═══════════════════════════════════════════════════════════════════════════════

class TestCEGARConvergence:
    """Test that the CEGAR loop converges correctly."""

    def test_safe_model_converges(self):
        """A safe model with concrete shapes converges in one iteration."""
        result = run_shape_cegar(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        assert result.is_safe
        assert result.final_status == CEGARStatus.SAFE
        assert result.iterations <= 2

    def test_safe_mlp_converges(self):
        """A two-layer MLP with correct shapes converges."""
        result = run_shape_cegar(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
        )
        assert result.is_safe
        assert result.final_status == CEGARStatus.SAFE

    def test_symbolic_input_discovers_predicate(self):
        """A symbolic input dimension should trigger predicate discovery."""
        result = run_shape_cegar(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", "features")},
        )
        # Should discover that features == 10
        assert result.is_safe or result.final_status == CEGARStatus.SAFE
        if result.discovered_predicates:
            found_dim_eq = any(
                p.kind == PredicateKind.DIM_EQ and p.value == 10
                for p in result.discovered_predicates
            )
            assert found_dim_eq, (
                f"Expected DIM_EQ predicate with value=10, got: "
                f"{[p.pretty() for p in result.discovered_predicates]}"
            )

    def test_max_iterations_respected(self):
        """CEGAR should stop after max_iterations."""
        result = run_shape_cegar(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", "features")},
            max_iterations=1,
        )
        assert result.iterations <= 1

    def test_empty_graph_safe(self):
        """An empty model (no forward steps) should be trivially safe."""
        source = """\
import torch.nn as nn

class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
"""
        result = run_shape_cegar(source, input_shapes={"x": ("batch", 10)})
        assert result.is_safe


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Predicate discovery from counterexamples
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredicateDiscovery:
    """Test that the CEGAR loop discovers correct predicates."""

    def test_discovers_linear_in_features(self):
        """Should discover input.shape[-1] == in_features for nn.Linear."""
        result = run_shape_cegar(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", "d_in")},
        )
        for p in result.discovered_predicates:
            if p.kind == PredicateKind.DIM_EQ and p.tensor == "x":
                assert p.value == 10
                break

    def test_discovers_mlp_chain(self):
        """Should discover input.shape[-1] == 784 for the first layer."""
        result = run_shape_cegar(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", "d_in")},
        )
        for p in result.discovered_predicates:
            if p.kind == PredicateKind.DIM_EQ and p.tensor == "x":
                assert p.value == 784
                break

    def test_predicate_pretty_print(self):
        """Test human-readable predicate representation."""
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768)
        assert p.pretty() == "x.shape[-1] == 768"

        p2 = ShapePredicate(PredicateKind.DIM_GE, "x", axis=0, value=1)
        assert p2.pretty() == "x.shape[0] >= 1"

        p3 = ShapePredicate(PredicateKind.DIM_DIVISIBLE, "x", axis=1, divisor=8)
        assert p3.pretty() == "x.shape[1] % 8 == 0"

        p4 = ShapePredicate(
            PredicateKind.DIM_MATCH, "x", axis=-1,
            match_tensor="w", match_axis=0,
        )
        assert "x.shape[-1] == w.shape[0]" in p4.pretty()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Integration with model checker
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelCheckerIntegration:
    """Test integration between shape contract discovery and the constraint verifier."""

    def test_uses_constraint_verifier(self):
        """The contract discovery loop should internally use ConstraintVerifier."""
        loop = ShapeCEGARLoop(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        result = loop.run()
        assert result.verification_result is not None

    def test_graph_extraction(self):
        """Should successfully extract computation graph."""
        graph = extract_computation_graph(SIMPLE_LINEAR)
        assert graph.class_name == "SimpleModel"
        assert len(graph.layers) >= 1

    def test_verify_and_discover_api(self):
        """Test the convenience API."""
        safe, preds, contracts = verify_and_discover(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        assert safe

    def test_result_contains_graph_info(self):
        """Result should contain the verification result from the checker."""
        result = run_shape_cegar(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
        )
        assert result.verification_result is not None

    def test_parse_error_handled(self):
        """Invalid source code should produce PARSE_ERROR status."""
        result = run_shape_cegar("this is not valid python!!!", input_shapes={})
        assert result.final_status == CEGARStatus.PARSE_ERROR

    def test_no_module_handled(self):
        """Source without nn.Module should produce PARSE_ERROR status."""
        result = run_shape_cegar(
            "x = 1\ny = 2\n",
            input_shapes={},
        )
        assert result.final_status == CEGARStatus.PARSE_ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Real bugs vs spurious warnings
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealBugsVsSpurious:
    """Test that CEGAR correctly distinguishes real bugs from spurious
    warnings.
    """

    def test_real_shape_mismatch_detected(self):
        """A model with fc1.out=20, fc2.in=50 has a real shape bug."""
        result = run_shape_cegar(
            SHAPE_MISMATCH_MODEL,
            input_shapes={"x": ("batch", 10)},
        )
        assert result.has_real_bugs
        assert result.final_status == CEGARStatus.REAL_BUG_FOUND
        assert len(result.real_bugs) > 0

    def test_correct_model_no_false_positives(self):
        """A correctly-shaped model should not produce false positives."""
        result = run_shape_cegar(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
        )
        assert result.is_safe
        assert len(result.real_bugs) == 0

    def test_symbolic_input_not_false_positive(self):
        """Symbolic inputs should lead to predicate discovery, not false
        bug reports.
        """
        result = run_shape_cegar(
            TRANSFORMER_BLOCK,
            input_shapes={"x": ("batch", "seq_len", "d_model")},
        )
        # Should either be safe (with predicates) or find the correct
        # constraint: d_model == 768
        if not result.is_safe:
            assert result.final_status != CEGARStatus.REAL_BUG_FOUND or (
                len(result.real_bugs) == 0
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Trace-back engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceBack:
    """Test the trace-back engine for following data flow."""

    def test_trace_direct_input(self):
        """Tracing from a step that directly uses an input."""
        graph = ComputationGraph(class_name="Test")
        graph.input_names = ["x"]
        graph.layers = {
            "fc": LayerDef(
                attr_name="fc",
                kind=LayerKind.LINEAR,
                in_features=10,
                out_features=5,
            ),
        }
        graph.steps = [
            ComputationStep(
                op=OpKind.LAYER_CALL,
                inputs=["x"],
                output="y",
                layer_ref="fc",
            ),
        ]

        tracer = TraceBackEngine(graph)
        inputs = tracer.trace_to_inputs(
            0, SafetyViolation(
                kind="shape_incompatible", step_index=0,
                step=graph.steps[0], message="test",
            ),
        )
        assert "x" in inputs

    def test_trace_through_activation(self):
        """Tracing should pass through shape-preserving activations."""
        graph = ComputationGraph(class_name="Test")
        graph.input_names = ["x"]
        graph.layers = {
            "fc": LayerDef(
                attr_name="fc",
                kind=LayerKind.LINEAR,
                in_features=10,
                out_features=5,
            ),
        }
        graph.steps = [
            ComputationStep(
                op=OpKind.ACTIVATION,
                inputs=["x"],
                output="x_relu",
            ),
            ComputationStep(
                op=OpKind.LAYER_CALL,
                inputs=["x_relu"],
                output="y",
                layer_ref="fc",
            ),
        ]

        tracer = TraceBackEngine(graph)
        inputs = tracer.trace_to_inputs(
            1, SafetyViolation(
                kind="shape_incompatible", step_index=1,
                step=graph.steps[1], message="test",
            ),
        )
        assert "x" in inputs

    def test_trace_multi_step_chain(self):
        """Tracing through a chain of operations."""
        graph = ComputationGraph(class_name="Test")
        graph.input_names = ["x"]
        graph.steps = [
            ComputationStep(op=OpKind.ACTIVATION, inputs=["x"], output="a"),
            ComputationStep(op=OpKind.DROPOUT, inputs=["a"], output="b"),
            ComputationStep(op=OpKind.CONTIGUOUS, inputs=["b"], output="c"),
        ]

        tracer = TraceBackEngine(graph)
        inputs = tracer.trace_to_inputs(
            2, SafetyViolation(
                kind="test", step_index=2,
                step=graph.steps[2], message="test",
            ),
        )
        assert "x" in inputs

    def test_find_constraint_origin_linear(self):
        """find_constraint_origin should identify the layer's in_features."""
        graph = ComputationGraph(class_name="Test")
        graph.input_names = ["x"]
        graph.layers = {
            "fc": LayerDef(
                attr_name="fc",
                kind=LayerKind.LINEAR,
                in_features=768,
                out_features=10,
            ),
        }
        graph.steps = [
            ComputationStep(
                op=OpKind.LAYER_CALL,
                inputs=["x"],
                output="y",
                layer_ref="fc",
            ),
        ]

        shape_env = {
            "x": TensorShape.from_tuple(("batch", "d")),
        }

        tracer = TraceBackEngine(graph)
        origins = tracer.find_constraint_origin(
            0,
            SafetyViolation(
                kind="shape_incompatible", step_index=0,
                step=graph.steps[0], message="test",
            ),
            shape_env,
        )
        assert any(expected == 768 for _, _, expected in origins)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Predicate set management
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredicateSet:
    """Test the PredicateSet deduplication and subsumption."""

    def test_deduplication(self):
        """Duplicate predicates should not be added twice."""
        ps = PredicateSet()
        p1 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
        p2 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
        assert ps.add(p1) is True
        assert ps.add(p2) is False
        assert len(ps) == 1

    def test_different_predicates_kept(self):
        """Different predicates should all be kept."""
        ps = PredicateSet()
        p1 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
        p2 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        ps.add(p1)
        ps.add(p2)
        assert len(ps) == 2

    def test_dim_eq_subsumes_dim_ge(self):
        """DIM_EQ should subsume DIM_GE on the same tensor and axis."""
        ps = PredicateSet()
        p_ge = ShapePredicate(PredicateKind.DIM_GE, "x", axis=-1, value=5)
        p_eq = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
        ps.add(p_ge)
        assert len(ps) == 1
        ps.add(p_eq)
        assert len(ps) == 1  # GE was subsumed
        assert ps.predicates[0].kind == PredicateKind.DIM_EQ

    def test_add_all_returns_count(self):
        """add_all should return the count of newly added predicates."""
        ps = PredicateSet()
        p1 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
        p2 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        p3 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
        count = ps.add_all([p1, p2, p3])
        assert count == 2

    def test_contains(self):
        """contains should check for predicate membership."""
        ps = PredicateSet()
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
        assert not ps.contains(p)
        ps.add(p)
        assert ps.contains(p)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Shape refinement
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeRefinement:
    """Test the shape environment refinement logic."""

    def test_apply_dim_eq_updates_input_shapes(self):
        """A DIM_EQ predicate should replace the symbolic dim in input_shapes."""
        input_shapes = {"x": ("batch", "features")}
        shape_env = {
            "x": TensorShape((ShapeDim("batch"), ShapeDim("features"))),
        }
        predicates = [
            ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10),
        ]

        new_inputs, new_env = ShapeRefinement.apply_predicates(
            input_shapes, shape_env, predicates,
        )
        assert new_inputs["x"][-1] == 10
        assert new_env["x"].dims[-1].value == 10

    def test_apply_preserves_other_dims(self):
        """Refinement should not touch unrelated dimensions."""
        input_shapes = {"x": ("batch", "features")}
        shape_env = {
            "x": TensorShape((ShapeDim("batch"), ShapeDim("features"))),
        }
        predicates = [
            ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10),
        ]

        new_inputs, new_env = ShapeRefinement.apply_predicates(
            input_shapes, shape_env, predicates,
        )
        assert new_inputs["x"][0] == "batch"
        assert new_env["x"].dims[0].value == "batch"

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_feasibility_check_sat(self):
        """A satisfiable set of predicates should pass feasibility check."""
        preds = [
            ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32),
            ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=10),
        ]
        assert ShapeRefinement.check_feasibility(preds) is True

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_predicates_to_z3(self):
        """predicates_to_z3 should produce Z3 constraints."""
        preds = [
            ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32),
        ]
        constraints = ShapeRefinement.predicates_to_z3(preds)
        assert len(constraints) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Contract inference
# ═══════════════════════════════════════════════════════════════════════════════

class TestContractInference:
    """Test the contract inference from discovered predicates."""

    def test_contracts_grouped_by_parameter(self):
        """Contracts should group predicates by parameter name."""
        graph = ComputationGraph(class_name="MyModel")
        predicates = [
            ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10),
            ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32),
            ShapePredicate(PredicateKind.DIM_EQ, "y", axis=-1, value=5),
        ]
        contracts = infer_contracts(graph, predicates)
        assert len(contracts) == 2
        x_contract = next(c for c in contracts if c.parameter == "x")
        assert len(x_contract.predicates) == 2

    def test_contract_pretty_print(self):
        """Test contract string representation."""
        contract = InferredContract(
            function_name="Model.forward",
            parameter="x",
            predicates=[
                ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768),
            ],
        )
        s = contract.pretty()
        assert "x.shape[-1] == 768" in s
        assert "Model.forward" in s

    def test_end_to_end_contract_discovery(self):
        """Full CEGAR should produce contracts for discovered predicates."""
        result = run_shape_cegar(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", "d")},
        )
        if result.discovered_predicates:
            assert len(result.contracts_inferred) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Z3 counterexample extraction
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
class TestZ3Extraction:
    """Test Z3-based counterexample extraction."""

    def test_find_violating_assignment(self):
        """Should find a concrete assignment that violates a constraint."""
        # Constraint: x == 10.  Negated: x != 10.  Should find x != 10.
        x = z3.Int("x")
        constraints = [x == 10]
        result = Z3CounterexampleExtractor.find_violating_assignment(
            constraints, ["x"],
        )
        assert result is not None
        assert result["x"] != 10

    def test_unsat_returns_none(self):
        """When constraints are tautologically true (x > 0 given x > 0),
        no violation should exist.
        """
        x = z3.Int("x")
        # "x > 0" is already enforced as a positive-dim constraint
        constraints = [x > 0]
        result = Z3CounterexampleExtractor.find_violating_assignment(
            constraints, ["x"],
        )
        # x > 0 is enforced AND Not(x > 0) means x <= 0, which contradicts
        # the positive-dim constraint (x > 0) → UNSAT
        assert result is None

    def test_check_predicate_eliminates_cex(self):
        """A predicate should be verified as eliminating a counterexample."""
        shape_env = {
            "x": TensorShape((ShapeDim("batch"), ShapeDim("d"))),
        }
        pred = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
        cex_dims = {"d": 5}  # Counterexample has d=5, but we want d=10

        eliminates = Z3CounterexampleExtractor.check_predicate_eliminates_cex(
            pred, cex_dims, shape_env,
        )
        assert eliminates is True


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Result type tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeCEGARResult:
    """Test the ShapeCEGARResult data structure."""

    def test_is_safe_property(self):
        result = ShapeCEGARResult(final_status=CEGARStatus.SAFE)
        assert result.is_safe is True

        result2 = ShapeCEGARResult(final_status=CEGARStatus.REAL_BUG_FOUND)
        assert result2.is_safe is False

    def test_has_real_bugs_property(self):
        result = ShapeCEGARResult(final_status=CEGARStatus.REAL_BUG_FOUND)
        assert result.has_real_bugs is True

    def test_summary_format(self):
        result = ShapeCEGARResult(
            discovered_predicates=[
                ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10),
            ],
            iterations=3,
            final_status=CEGARStatus.SAFE,
            total_time_ms=42.0,
        )
        s = result.summary()
        assert "SAFE" in s
        assert "3" in s
        assert "1 predicates" in s

    def test_iteration_log(self):
        """Iteration log should be populated."""
        result = run_shape_cegar(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        assert isinstance(result.iteration_log, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Counterexample analyser unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCounterexampleAnalyser:
    """Test the counterexample analyser in isolation."""

    def _make_linear_graph(self, in_features=10, out_features=5):
        """Helper: build a minimal computation graph with one Linear layer."""
        graph = ComputationGraph(class_name="Test")
        graph.input_names = ["x"]
        graph.output_names = ["y"]
        graph.layers = {
            "fc": LayerDef(
                attr_name="fc",
                kind=LayerKind.LINEAR,
                in_features=in_features,
                out_features=out_features,
            ),
        }
        graph.steps = [
            ComputationStep(
                op=OpKind.LAYER_CALL,
                inputs=["x"],
                output="y",
                layer_ref="fc",
            ),
        ]
        return graph

    def test_classify_spurious_symbolic(self):
        """A violation due to unconstrained symbolic dim is spurious."""
        graph = self._make_linear_graph(10, 5)
        shape_env = {
            "x": TensorShape((ShapeDim("batch"), ShapeDim("d"))),
        }
        analyser = CounterexampleAnalyser(
            graph, shape_env, {"x": ("batch", "d")},
        )

        cex = CounterexampleTrace(
            model_name="Test",
            violations=[
                SafetyViolation(
                    kind="shape_incompatible",
                    step_index=0,
                    step=graph.steps[0],
                    message="Linear in_features mismatch",
                    tensor_a="x",
                    shape_a=TensorShape((ShapeDim("batch"), ShapeDim("d"))),
                ),
            ],
            concrete_dims={"d": 7},
        )

        analysed = analyser.analyse(cex)
        assert len(analysed) == 1
        # Should be spurious (d is symbolic and can be constrained to 10)
        assert analysed[0].classification in (
            CounterexampleClassification.SPURIOUS,
            CounterexampleClassification.UNKNOWN,
        )

    def test_classify_real_bug_concrete(self):
        """A violation with concrete mismatched dims is a real bug."""
        graph = ComputationGraph(class_name="Test")
        graph.input_names = ["x"]
        graph.layers = {
            "fc1": LayerDef(
                attr_name="fc1", kind=LayerKind.LINEAR,
                in_features=10, out_features=20,
            ),
            "fc2": LayerDef(
                attr_name="fc2", kind=LayerKind.LINEAR,
                in_features=50, out_features=5,
            ),
        }
        graph.steps = [
            ComputationStep(
                op=OpKind.LAYER_CALL, inputs=["x"],
                output="h", layer_ref="fc1",
            ),
            ComputationStep(
                op=OpKind.LAYER_CALL, inputs=["h"],
                output="y", layer_ref="fc2",
            ),
        ]
        shape_env = {
            "x": TensorShape((ShapeDim("batch"), ShapeDim(10))),
            "h": TensorShape((ShapeDim("batch"), ShapeDim(20))),
        }
        analyser = CounterexampleAnalyser(
            graph, shape_env, {"x": ("batch", 10)},
        )

        cex = CounterexampleTrace(
            model_name="Test",
            violations=[
                SafetyViolation(
                    kind="shape_incompatible",
                    step_index=1,
                    step=graph.steps[1],
                    message="Linear expects 50 but got 20",
                    tensor_a="h",
                    shape_a=TensorShape((ShapeDim("batch"), ShapeDim(20))),
                ),
            ],
        )

        analysed = analyser.analyse(cex)
        assert len(analysed) == 1
        assert analysed[0].is_real_bug()
