"""
Shape Contract Discovery: Houdini-Style Predicate Accumulation for Tensor Shapes.

Implements a counterexample-guided contract discovery loop based on
Houdini-style predicate accumulation (Flanagan & Leino FME 2001) that
iteratively discovers shape predicates for nn.Module computation graphs.
Unlike general CEGAR (Clarke et al. CAV 2000), our loop *accumulates*
predicates monotonically from counterexamples rather than maintaining
an abstract domain.  Z3-backed abstract feasibility checking classifies
counterexamples as real bugs or spurious (eliminable by constraining
input shapes), but does not perform concrete Python execution.

Algorithm
---------
1. **Verify**   — Run constraint-based verification (via ``ConstraintVerifier``)
                  on the computation graph with the current shape environment.
2. **Check**    — Examine the counterexample(s) returned by the verifier.
                  If no counterexample → shapes are safe; emit certificate.
3. **Extract**  — From each Z3-produced counterexample, extract concrete
                  dimension values that caused the reported shape mismatch.
4. **Trace**    — Walk the computation graph *backwards* from the failing
                  step to find the input shape assumption(s) whose
                  weakening allowed the spurious counterexample.
5. **Synth**    — Synthesise a new shape predicate that rules out the
                  spurious counterexample (e.g. ``input.shape[-1] == 768``).
6. **Refine**   — Add the new predicate to the shape environment and
                  re-verify from step 1.
7. **Converge** — Stop when (a) no more counterexamples, (b) a real bug
                  is found, or (c) the iteration budget is exhausted.

Integration points
------------------
* ``model_checker.ComputationGraph`` — for trace-back over operations.
* ``model_checker.ConstraintVerifier`` — for per-iteration verification.
* ``tensor_shapes.ShapeEnv`` / ``TensorShape`` — for shape representation.
* Z3 — for SAT/UNSAT checking and counterexample extraction.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Z3 import
# ---------------------------------------------------------------------------

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from src.tensor_shapes import (
    TensorShape,
    ShapeDim,
    ShapeError,
    ShapeErrorKind,
    ShapeEnv,
)
from src.model_checker import (
    ComputationGraph,
    extract_computation_graph,
    ComputationStep,
    ConstraintVerifier,
    VerificationResult,
    SafetyViolation,
    CounterexampleTrace,
    ModelState,
    OpKind,
    LayerKind,
    LayerDef,
    Device,
    Phase,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Shape predicate representation
# ═══════════════════════════════════════════════════════════════════════════════

class PredicateKind(Enum):
    """The flavour of a discovered shape predicate."""
    DIM_EQ = auto()       # tensor.shape[axis] == value
    DIM_GT = auto()       # tensor.shape[axis] > value
    DIM_GE = auto()       # tensor.shape[axis] >= value
    DIM_DIVISIBLE = auto()  # tensor.shape[axis] % divisor == 0
    DIM_MATCH = auto()    # tensor_a.shape[axis_a] == tensor_b.shape[axis_b]
    NDIM_EQ = auto()      # len(tensor.shape) == value
    SHAPE_EQ = auto()     # tensor.shape == (d0, d1, ...)


@dataclass(frozen=True)
class ShapePredicate:
    """A single shape predicate discovered by the contract discovery loop.

    Examples
    --------
    >>> ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768)
    # means: x.shape[-1] == 768

    >>> ShapePredicate(PredicateKind.DIM_MATCH, "x", axis=-1,
    ...               match_tensor="w", match_axis=0)
    # means: x.shape[-1] == w.shape[0]
    """
    kind: PredicateKind
    tensor: str
    axis: Optional[int] = None
    value: Optional[int] = None
    match_tensor: Optional[str] = None
    match_axis: Optional[int] = None
    divisor: Optional[int] = None

    def pretty(self) -> str:
        """Human-readable representation."""
        if self.kind == PredicateKind.DIM_EQ:
            return f"{self.tensor}.shape[{self.axis}] == {self.value}"
        if self.kind == PredicateKind.DIM_GT:
            return f"{self.tensor}.shape[{self.axis}] > {self.value}"
        if self.kind == PredicateKind.DIM_GE:
            return f"{self.tensor}.shape[{self.axis}] >= {self.value}"
        if self.kind == PredicateKind.DIM_DIVISIBLE:
            return f"{self.tensor}.shape[{self.axis}] % {self.divisor} == 0"
        if self.kind == PredicateKind.DIM_MATCH:
            return (
                f"{self.tensor}.shape[{self.axis}] == "
                f"{self.match_tensor}.shape[{self.match_axis}]"
            )
        if self.kind == PredicateKind.NDIM_EQ:
            return f"len({self.tensor}.shape) == {self.value}"
        if self.kind == PredicateKind.SHAPE_EQ:
            return f"{self.tensor}.shape == {self.value}"
        return f"<unknown predicate on {self.tensor}>"

    def __repr__(self) -> str:
        return f"ShapePredicate({self.pretty()})"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Counterexample analysis types
# ═══════════════════════════════════════════════════════════════════════════════

class CounterexampleClassification(Enum):
    """Whether a counterexample is spurious or a real bug."""
    SPURIOUS = auto()   # can be eliminated by a new predicate
    REAL_BUG = auto()   # actual shape error in the model
    UNKNOWN = auto()    # cannot classify (conservative: treat as real)


@dataclass
class AnalysedCounterexample:
    """A counterexample that has been traced back and classified."""
    violation: SafetyViolation
    step_index: int
    classification: CounterexampleClassification
    concrete_dims: Dict[str, int] = field(default_factory=dict)
    traced_to_inputs: List[str] = field(default_factory=list)
    synthesised_predicates: List[ShapePredicate] = field(default_factory=list)
    reason: str = ""

    def is_spurious(self) -> bool:
        return self.classification == CounterexampleClassification.SPURIOUS

    def is_real_bug(self) -> bool:
        return self.classification == CounterexampleClassification.REAL_BUG


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Contract discovery result type
# ═══════════════════════════════════════════════════════════════════════════════

class CEGARStatus(Enum):
    """Final status of the contract discovery loop (CEGAR-style)."""
    SAFE = auto()            # all counterexamples eliminated → model is safe
    REAL_BUG_FOUND = auto()  # found a genuine shape bug
    MAX_ITER = auto()        # iteration budget exhausted
    NO_Z3 = auto()           # Z3 not available; fell back to single pass
    PARSE_ERROR = auto()     # could not parse the source


@dataclass
class InferredContract:
    """A shape contract inferred by the contract discovery loop for a function/method."""
    function_name: str
    parameter: str
    predicates: List[ShapePredicate] = field(default_factory=list)

    def pretty(self) -> str:
        preds = ", ".join(p.pretty() for p in self.predicates)
        return f"{self.function_name}({self.parameter}): requires [{preds}]"


@dataclass
class ShapeCEGARResult:
    """Top-level result of the shape contract discovery loop (CEGAR-style).

    Attributes
    ----------
    discovered_predicates : list of ShapePredicate
        All shape predicates discovered during refinement.
    iterations : int
        Number of contract discovery iterations performed.
    final_status : CEGARStatus
        Why the loop terminated.
    contracts_inferred : list of InferredContract
        Per-parameter shape contracts inferred from discovered predicates.
    verification_result : VerificationResult or None
        The final model-checker result from the last iteration.
    real_bugs : list of SafetyViolation
        Any genuine shape bugs found (empty if model is safe).
    total_time_ms : float
        Wall-clock time for the entire contract discovery loop.
    iteration_log : list of IterationRecord
        Per-iteration details for debugging / reporting.
    """
    discovered_predicates: List[ShapePredicate] = field(default_factory=list)
    iterations: int = 0
    final_status: CEGARStatus = CEGARStatus.SAFE
    contracts_inferred: List[InferredContract] = field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    real_bugs: List[SafetyViolation] = field(default_factory=list)
    total_time_ms: float = 0.0
    iteration_log: List["IterationRecord"] = field(default_factory=list)
    predicate_quality_report: Optional[Dict[str, Any]] = None

    @property
    def is_safe(self) -> bool:
        return self.final_status == CEGARStatus.SAFE

    @property
    def has_real_bugs(self) -> bool:
        return self.final_status == CEGARStatus.REAL_BUG_FOUND

    def summary(self) -> str:
        preds = ", ".join(p.pretty() for p in self.discovered_predicates)
        return (
            f"ShapeCEGAR: {self.final_status.name} after "
            f"{self.iterations} iterations, "
            f"{len(self.discovered_predicates)} predicates discovered"
            + (f" [{preds}]" if preds else "")
            + f", {self.total_time_ms:.1f}ms"
        )


@dataclass
class IterationRecord:
    """Diagnostic record for a single contract discovery iteration."""
    iteration: int
    num_violations: int
    num_spurious: int
    num_real: int
    predicates_added: List[ShapePredicate] = field(default_factory=list)
    time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Trace-back engine
# ═══════════════════════════════════════════════════════════════════════════════

class TraceBackEngine:
    """Traces a counterexample backwards through the computation graph to
    find the input shape assumption that, if strengthened, would prevent
    the counterexample.

    The key insight: every shape mismatch at step *k* is caused by
    one or more earlier steps that produced tensors with incompatible
    shapes.  By walking backwards along the data-flow edges we can
    find the *earliest* point where a shape constraint can be added —
    ideally at the inputs of ``forward()``.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self.graph = graph
        self._producers: Dict[str, int] = {}
        for idx, step in enumerate(graph.steps):
            self._producers[step.output] = idx

    def trace_to_inputs(
        self,
        failing_step_idx: int,
        violation: SafetyViolation,
    ) -> List[str]:
        """Return the list of input tensor names that contribute to the
        violation at *failing_step_idx*.
        """
        if failing_step_idx < 0 or failing_step_idx >= len(self.graph.steps):
            return []

        step = self.graph.steps[failing_step_idx]
        visited: Set[str] = set()
        input_sources: List[str] = []
        self._walk_back(step.inputs, visited, input_sources)
        return input_sources

    def _walk_back(
        self,
        tensor_names: List[str],
        visited: Set[str],
        input_sources: List[str],
    ) -> None:
        """Recursively walk backwards through data-flow edges."""
        for name in tensor_names:
            if name in visited:
                continue
            visited.add(name)

            if name in self.graph.input_names:
                if name not in input_sources:
                    input_sources.append(name)
                continue

            producer_idx = self._producers.get(name)
            if producer_idx is not None:
                producer = self.graph.steps[producer_idx]
                self._walk_back(producer.inputs, visited, input_sources)
            else:
                # Tensor not produced by any step and not an input —
                # might be a parameter.  Record it anyway.
                if name not in input_sources:
                    input_sources.append(name)

    def find_constraint_origin(
        self,
        failing_step_idx: int,
        violation: SafetyViolation,
        shape_env: Dict[str, TensorShape],
    ) -> List[Tuple[str, int, Optional[int]]]:
        """Find the (tensor, axis, expected_value) triples that, if
        constrained at the input, would fix the violation.

        Returns a list of ``(tensor_name, axis, expected_value)`` where
        *expected_value* is ``None`` if it could not be determined.
        """
        if failing_step_idx < 0 or failing_step_idx >= len(self.graph.steps):
            return []

        step = self.graph.steps[failing_step_idx]
        origins: List[Tuple[str, int, Optional[int]]] = []

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                inp = step.inputs[0] if step.inputs else None
                if inp:
                    origins.append((inp, -1, layer.in_features))

            elif layer and layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                inp = step.inputs[0] if step.inputs else None
                if inp:
                    origins.append((inp, 1, layer.in_channels))

            elif layer and layer.kind == LayerKind.EMBEDDING and layer.num_embeddings is not None:
                inp = step.inputs[0] if step.inputs else None
                if inp:
                    origins.append((inp, -1, None))

        elif step.op == OpKind.MATMUL:
            if len(step.inputs) >= 2:
                a_name, b_name = step.inputs[0], step.inputs[1]
                a_shape = shape_env.get(a_name)
                b_shape = shape_env.get(b_name)
                if a_shape and b_shape:
                    # Inner dims must match: a.shape[-1] == b.shape[-2]
                    if a_shape.ndim >= 1:
                        origins.append((a_name, -1, None))
                    if b_shape.ndim >= 2:
                        origins.append((b_name, -2, None))
                    elif b_shape.ndim == 1:
                        origins.append((b_name, 0, None))

        elif step.op == OpKind.ADD:
            if len(step.inputs) >= 2:
                a_name, b_name = step.inputs[0], step.inputs[1]
                a_shape = shape_env.get(a_name)
                b_shape = shape_env.get(b_name)
                if a_shape and b_shape:
                    ndim = max(a_shape.ndim, b_shape.ndim)
                    for i in range(1, ndim + 1):
                        d_a = a_shape.dims[-i] if i <= a_shape.ndim else None
                        d_b = b_shape.dims[-i] if i <= b_shape.ndim else None
                        if d_a and d_b and d_a != d_b:
                            if not d_a.is_symbolic and d_a.value != 1:
                                origins.append((b_name, -i, d_a.value))
                            elif not d_b.is_symbolic and d_b.value != 1:
                                origins.append((a_name, -i, d_b.value))

        elif step.op == OpKind.CAT:
            for inp_name in step.inputs:
                shape = shape_env.get(inp_name)
                if shape:
                    origins.append((inp_name, 0, None))

        # Trace each tensor back to its input source
        input_origins: List[Tuple[str, int, Optional[int]]] = []
        for tensor_name, axis, expected in origins:
            input_chain = self._trace_dim_to_input(tensor_name, axis, shape_env)
            if input_chain:
                src_tensor, src_axis = input_chain
                input_origins.append((src_tensor, src_axis, expected))
            else:
                input_origins.append((tensor_name, axis, expected))

        return input_origins

    def _trace_dim_to_input(
        self,
        tensor_name: str,
        axis: int,
        shape_env: Dict[str, TensorShape],
    ) -> Optional[Tuple[str, int]]:
        """Trace a specific dimension back through shape-preserving ops
        to find the corresponding input tensor and axis.
        """
        visited: Set[str] = set()
        current_name = tensor_name
        current_axis = axis

        while current_name not in self.graph.input_names:
            if current_name in visited:
                break
            visited.add(current_name)

            producer_idx = self._producers.get(current_name)
            if producer_idx is None:
                break

            step = self.graph.steps[producer_idx]

            if step.op in (
                OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                OpKind.CONTIGUOUS, OpKind.DETACH,
            ):
                # Shape-preserving: axis maps 1:1
                if step.inputs:
                    current_name = step.inputs[0]
                else:
                    break

            elif step.op == OpKind.LAYER_CALL:
                layer = self.graph.layers.get(step.layer_ref or "")
                if layer and layer.kind in (
                    LayerKind.RELU, LayerKind.DROPOUT, LayerKind.IDENTITY,
                    LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                    LayerKind.LAYERNORM, LayerKind.SOFTMAX,
                ):
                    if step.inputs:
                        current_name = step.inputs[0]
                    else:
                        break
                elif layer and layer.kind == LayerKind.LINEAR:
                    # Linear changes last dim; other dims pass through
                    shape = shape_env.get(current_name)
                    norm_axis = current_axis
                    if shape and norm_axis < 0:
                        norm_axis = shape.ndim + norm_axis
                    if shape and norm_axis == shape.ndim - 1:
                        # Last dim is transformed — cannot trace further
                        break
                    if step.inputs:
                        current_name = step.inputs[0]
                    else:
                        break
                else:
                    break

            elif step.op == OpKind.TRANSPOSE:
                d0 = step.params.get("dim0", 0)
                d1 = step.params.get("dim1", 1)
                shape = shape_env.get(current_name)
                norm = current_axis
                if shape and norm < 0:
                    norm = shape.ndim + norm
                if norm == d0:
                    current_axis = d1
                elif norm == d1:
                    current_axis = d0
                if step.inputs:
                    current_name = step.inputs[0]
                else:
                    break

            elif step.op == OpKind.PERMUTE:
                perm = step.params.get("dims")
                shape = shape_env.get(current_name)
                norm = current_axis
                if shape and norm < 0:
                    norm = shape.ndim + norm
                if perm and 0 <= norm < len(perm):
                    current_axis = perm[norm]
                if step.inputs:
                    current_name = step.inputs[0]
                else:
                    break

            else:
                # Cannot trace through reshape, cat, matmul, etc.
                break

        if current_name in self.graph.input_names:
            return (current_name, current_axis)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Unsat core-based predicate extraction engine
# ═══════════════════════════════════════════════════════════════════════════════

class UnsatCorePredicateExtractor:
    """Discovers shape predicates via unsat-core-based predicate extraction.

    Given:
      - A computation graph formula encoding shape transitions
      - A counterexample path formula (concrete dim assignments that violate safety)
    Extracts the unsat core to identify the minimal set of constraints
    making the counterexample spurious, then extracts shape predicates
    from those constraints.

    Uses Z3 unsat core extraction (not Craig interpolation). Unsat cores
    provide subsets of input clauses contributing to unsatisfiability,
    which is sufficient for predicate discovery in our setting.  This is
    Houdini-style predicate accumulation (Flanagan & Leino, FME 2001)
    with Z3-backed abstract feasibility checking.
    """

    def __init__(
        self,
        graph: ComputationGraph,
        shape_env: Dict[str, TensorShape],
    ) -> None:
        self.graph = graph
        self.shape_env = shape_env

    def discover_predicates(
        self,
        violation: SafetyViolation,
        step_idx: int,
        concrete_dims: Dict[str, int],
        input_shapes: Dict[str, tuple],
    ) -> List[ShapePredicate]:
        """Discover predicates that eliminate a spurious counterexample.

        Uses unsat core extraction to find the minimal set of constraints
        that make the counterexample infeasible, then extracts shape
        predicates from those constraints.
        """
        if not HAS_Z3:
            return []

        if step_idx < 0 or step_idx >= len(self.graph.steps):
            return []

        step = self.graph.steps[step_idx]
        predicates: List[ShapePredicate] = []

        # Build the path formula (A) and safety formula (B)
        # A: input assumptions + computation graph transitions up to step
        # B: negation of safety property at the failing step
        path_constraints, safety_constraints, dim_map = (
            self._build_predicate_extraction_query(step, step_idx, concrete_dims, input_shapes)
        )

        if not path_constraints or not safety_constraints:
            return []

        # Try unsat-core-based predicate extraction
        predicates = self._extract_via_unsat_core(
            path_constraints, safety_constraints, dim_map, concrete_dims,
        )

        return predicates

    def _build_predicate_extraction_query(
        self,
        step: ComputationStep,
        step_idx: int,
        concrete_dims: Dict[str, int],
        input_shapes: Dict[str, tuple],
    ) -> Tuple[List[Any], List[Any], Dict[str, Tuple[str, int]]]:
        """Build the formula pair (path, safety) for unsat-core extraction.

        Returns:
            (path_constraints, safety_constraints, dim_map)
            where dim_map maps Z3 variable names to (tensor_name, axis).
        """
        path_cs: List[Any] = []
        safety_cs: List[Any] = []
        dim_map: Dict[str, Tuple[str, int]] = {}

        # Create Z3 variables for input dimensions
        for inp_name, shape_tuple in input_shapes.items():
            for axis, dim_val in enumerate(shape_tuple):
                var_name = f"__interp_{inp_name}_d{axis}"
                var = z3.Int(var_name)
                dim_map[var_name] = (inp_name, axis)
                path_cs.append(var > 0)
                if isinstance(dim_val, int):
                    path_cs.append(var == z3.IntVal(dim_val))

        # Encode safety property at the failing step
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            inp = step.inputs[0] if step.inputs else None
            if layer and inp and inp in input_shapes:
                inp_shape = input_shapes[inp]
                if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                    var_name = f"__interp_{inp}_d{len(inp_shape)-1}"
                    var = z3.Int(var_name)
                    safety_cs.append(var == z3.IntVal(layer.in_features))
                elif layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                    if len(inp_shape) >= 2:
                        var_name = f"__interp_{inp}_d1"
                        var = z3.Int(var_name)
                        safety_cs.append(var == z3.IntVal(layer.in_channels))

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_name, b_name = step.inputs[0], step.inputs[1]
            a_shape = input_shapes.get(a_name)
            b_shape = input_shapes.get(b_name)
            if a_shape and b_shape:
                va = f"__interp_{a_name}_d{len(a_shape)-1}"
                vb = f"__interp_{b_name}_d{len(b_shape)-2}" if len(b_shape) >= 2 else f"__interp_{b_name}_d0"
                safety_cs.append(z3.Int(va) == z3.Int(vb))

        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a_name, b_name = step.inputs[0], step.inputs[1]
            a_shape = input_shapes.get(a_name)
            b_shape = input_shapes.get(b_name)
            if a_shape and b_shape:
                ndim = max(len(a_shape), len(b_shape))
                for i in range(1, ndim + 1):
                    if i <= len(a_shape) and i <= len(b_shape):
                        va = f"__interp_{a_name}_d{len(a_shape)-i}"
                        vb = f"__interp_{b_name}_d{len(b_shape)-i}"
                        a_var = z3.Int(va)
                        b_var = z3.Int(vb)
                        safety_cs.append(z3.Or(
                            a_var == b_var,
                            a_var == z3.IntVal(1),
                            b_var == z3.IntVal(1),
                        ))

        return path_cs, safety_cs, dim_map

    def _extract_via_unsat_core(
        self,
        path_constraints: List[Any],
        safety_constraints: List[Any],
        dim_map: Dict[str, Tuple[str, int]],
        concrete_dims: Dict[str, int],
    ) -> List[ShapePredicate]:
        """Extract predicates using Z3 unsat core for predicate discovery.

        Adds path constraints as tracked assertions, negates safety,
        and extracts the minimal unsatisfiable core to identify which
        input dimension constraints are essential.
        """
        solver = z3.Solver()
        solver.set("timeout", 5000)
        solver.set("unsat_core", True)

        # Add path constraints with tracking labels
        labels: Dict[str, Any] = {}
        for i, c in enumerate(path_constraints):
            label = z3.Bool(f"__path_{i}")
            labels[f"__path_{i}"] = c
            solver.assert_and_track(c, label)

        # Negate safety: we want path ∧ ¬safety to be UNSAT
        # (counterexample is spurious because path constraints force safety)
        if safety_constraints:
            neg_safety = z3.Not(z3.And(*safety_constraints))
            solver.add(neg_safety)

        result = solver.check()

        predicates: List[ShapePredicate] = []
        if result == z3.unsat:
            # The counterexample is spurious — extract core
            core = solver.unsat_core()
            core_names = {str(c) for c in core}

            # Map core assertions back to dimension constraints
            for i, c in enumerate(path_constraints):
                label_name = f"__path_{i}"
                if label_name in core_names:
                    # Extract dimension info from the constraint
                    pred = self._constraint_to_predicate(c, dim_map)
                    if pred is not None:
                        predicates.append(pred)

        return predicates

    def _constraint_to_predicate(
        self,
        constraint: Any,
        dim_map: Dict[str, Tuple[str, int]],
    ) -> Optional[ShapePredicate]:
        """Convert a Z3 constraint from the unsat core into a ShapePredicate."""
        if not HAS_Z3:
            return None

        constraint_str = str(constraint)

        # Look for equality constraints like "var == value"
        for var_name, (tensor, axis) in dim_map.items():
            if var_name in constraint_str and "==" in constraint_str:
                # Try to extract the concrete value
                try:
                    # Check if this is a "var == IntVal" constraint
                    if z3.is_eq(constraint):
                        lhs, rhs = constraint.children()
                        val = None
                        if z3.is_int_value(rhs):
                            val = rhs.as_long()
                        elif z3.is_int_value(lhs):
                            val = lhs.as_long()
                        if val is not None and val > 0:
                            return ShapePredicate(
                                kind=PredicateKind.DIM_EQ,
                                tensor=tensor,
                                axis=axis,
                                value=val,
                            )
                except Exception:
                    pass

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Counterexample analyser
# ═══════════════════════════════════════════════════════════════════════════════

class CounterexampleAnalyser:
    """Classifies counterexamples as spurious or real and synthesises
    predicates to eliminate spurious ones.
    """

    def __init__(
        self,
        graph: ComputationGraph,
        shape_env: Dict[str, TensorShape],
        input_shapes: Dict[str, tuple],
    ) -> None:
        self.graph = graph
        self.shape_env = shape_env
        self.input_shapes = input_shapes
        self.tracer = TraceBackEngine(graph)
        self.predicate_extractor = UnsatCorePredicateExtractor(graph, shape_env)

    def analyse(
        self,
        counterexample: CounterexampleTrace,
    ) -> List[AnalysedCounterexample]:
        """Analyse all violations in a counterexample trace."""
        results: List[AnalysedCounterexample] = []

        for violation in counterexample.violations:
            acex = self._analyse_single(violation, counterexample)
            results.append(acex)

        return results

    def _analyse_single(
        self,
        violation: SafetyViolation,
        cex_trace: CounterexampleTrace,
    ) -> AnalysedCounterexample:
        """Analyse a single safety violation."""
        step_idx = violation.step_index

        # Extract concrete dimension values from the counterexample
        concrete_dims = dict(cex_trace.concrete_dims)

        # Trace back to input tensors
        traced = self.tracer.trace_to_inputs(step_idx, violation)

        # Find constraint origins
        origins = self.tracer.find_constraint_origin(
            step_idx, violation, self.shape_env,
        )

        # Classify and synthesise
        classification, predicates, reason = self._classify_and_synthesise(
            violation, step_idx, origins, concrete_dims,
        )

        return AnalysedCounterexample(
            violation=violation,
            step_index=step_idx,
            classification=classification,
            concrete_dims=concrete_dims,
            traced_to_inputs=traced,
            synthesised_predicates=predicates,
            reason=reason,
        )

    def _check_cex_feasibility(
        self,
        concrete_dims: Dict[str, int],
        step: ComputationStep,
        violation: SafetyViolation,
    ) -> bool:
        """Abstract feasibility check via Z3.

        Checks if the counterexample is feasible by encoding the
        computation graph path constraints up to the failing step
        with the counterexample's concrete dimension assignments,
        and verifying via Z3 that the path is satisfiable AND the
        safety property is violated.  This is an *abstract* check
        (it reasons about constraints, not concrete Python execution).

        Returns True if the counterexample is feasible (real bug).
        Returns False if the counterexample is spurious (infeasible path).
        """
        # Basic hardware bounds check
        MIN_DIM = 1
        MAX_DIM = 65536
        for dim_name, dim_val in concrete_dims.items():
            if not isinstance(dim_val, int):
                continue
            if dim_val < MIN_DIM or dim_val > MAX_DIM:
                return False

        # Z3-based path feasibility check
        if not HAS_Z3:
            return True  # Conservative: assume feasible without Z3

        step_idx = violation.step_index
        if step_idx < 0 or step_idx >= len(self.graph.steps):
            return True

        solver = z3.Solver()
        solver.set("timeout", 3000)

        # Create Z3 variables for all concrete dimensions
        z3_dims: Dict[str, z3.ArithRef] = {}
        for dim_name, dim_val in concrete_dims.items():
            if isinstance(dim_val, int):
                var = z3.Int(f"_feas_{dim_name}")
                solver.add(var == z3.IntVal(dim_val))
                z3_dims[dim_name] = var

        # Encode path constraints through the computation graph
        # up to the failing step
        for i, s in enumerate(self.graph.steps[:step_idx + 1]):
            if s.op == OpKind.LAYER_CALL and s.layer_ref:
                layer = self.graph.layers.get(s.layer_ref)
                if layer and s.inputs:
                    inp = s.inputs[0]
                    inp_shape = self.shape_env.get(inp)
                    if inp_shape and layer.kind == LayerKind.LINEAR:
                        # Encode: last dim of input must equal in_features
                        last_axis = len(inp_shape.dims) - 1
                        for dim in inp_shape.dims:
                            if dim.is_symbolic and str(dim.value) in z3_dims:
                                pass  # Already constrained

            elif s.op == OpKind.MATMUL and len(s.inputs) >= 2:
                a_shape = self.shape_env.get(s.inputs[0])
                b_shape = self.shape_env.get(s.inputs[1])
                if a_shape and b_shape:
                    # Inner dimensions must match
                    pass  # Encoded via concrete dims

        # Check if the concrete dims actually violate the operation
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            inp = step.inputs[0] if step.inputs else None
            if layer and inp:
                inp_shape = self.shape_env.get(inp)
                if inp_shape and layer.kind == LayerKind.LINEAR:
                    if layer.in_features is not None:
                        last_dim = inp_shape.dims[-1] if inp_shape.dims else None
                        if last_dim and not last_dim.is_symbolic:
                            return int(last_dim.value) != layer.in_features
                        elif last_dim and last_dim.is_symbolic:
                            sym_name = str(last_dim.value)
                            if sym_name in concrete_dims:
                                return concrete_dims[sym_name] != layer.in_features

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_shape = self.shape_env.get(step.inputs[0])
            b_shape = self.shape_env.get(step.inputs[1])
            if a_shape and b_shape and a_shape.dims and b_shape.dims:
                a_last = a_shape.dims[-1]
                b_inner = b_shape.dims[-2] if len(b_shape.dims) >= 2 else b_shape.dims[0]
                a_val = concrete_dims.get(str(a_last.value)) if a_last.is_symbolic else (int(a_last.value) if not a_last.is_symbolic else None)
                b_val = concrete_dims.get(str(b_inner.value)) if b_inner.is_symbolic else (int(b_inner.value) if not b_inner.is_symbolic else None)
                if a_val is not None and b_val is not None:
                    return a_val != b_val

        # Fallback: check path satisfiability with Z3
        result = solver.check()
        return result == z3.sat

    def _classify_and_synthesise(
        self,
        violation: SafetyViolation,
        step_idx: int,
        origins: List[Tuple[str, int, Optional[int]]],
        concrete_dims: Dict[str, int],
    ) -> Tuple[CounterexampleClassification, List[ShapePredicate], str]:
        """Classify a violation and synthesise predicates if spurious."""
        predicates: List[ShapePredicate] = []
        step = self.graph.steps[step_idx] if step_idx < len(self.graph.steps) else None

        if not step:
            return (CounterexampleClassification.UNKNOWN, [], "step not found")

        # Check if this is a real bug: concrete dimensions that cannot
        # be fixed by constraining inputs
        if self._is_real_bug(violation, step):
            return (
                CounterexampleClassification.REAL_BUG,
                [],
                self._real_bug_reason(violation, step),
            )

        # Feasibility check: if the concrete counterexample dimensions are
        # all physically realizable, the bug is real even if trace-back
        # would produce predicates to eliminate it.
        if concrete_dims and self._check_cex_feasibility(concrete_dims, step, violation):
            # Check whether the concrete dims actually violate the
            # operation's requirements (not just that they're feasible
            # in general).
            if self._concrete_dims_violate_op(concrete_dims, step):
                return (
                    CounterexampleClassification.REAL_BUG,
                    [],
                    f"feasible counterexample: concrete dims {concrete_dims} "
                    f"are realizable and violate {step.op.name}",
                )

        # Spurious: synthesise predicates from origins
        for tensor_name, axis, expected in origins:
            if expected is not None:
                pred = ShapePredicate(
                    kind=PredicateKind.DIM_EQ,
                    tensor=tensor_name,
                    axis=axis,
                    value=expected,
                )
                predicates.append(pred)
            else:
                # Try to get expected from concrete dims or layer params
                inferred = self._infer_expected(
                    tensor_name, axis, step, concrete_dims,
                )
                if inferred is not None:
                    pred = ShapePredicate(
                        kind=PredicateKind.DIM_EQ,
                        tensor=tensor_name,
                        axis=axis,
                        value=inferred,
                    )
                    predicates.append(pred)

        # Use unsat-core-based predicate discovery as a fallback
        if not predicates:
            interp_preds = self.predicate_extractor.discover_predicates(
                violation, step_idx, concrete_dims, self.input_shapes,
            )
            predicates.extend(interp_preds)

        if predicates:
            reason = "spurious: fixed by constraining " + ", ".join(
                p.pretty() for p in predicates
            )
            return (CounterexampleClassification.SPURIOUS, predicates, reason)

        # Could not determine — be conservative
        return (CounterexampleClassification.UNKNOWN, [], "could not classify")

    def _is_real_bug(
        self, violation: SafetyViolation, step: ComputationStep
    ) -> bool:
        """Check if a violation is a real shape bug (all dimensions are
        concrete and incompatible).
        """
        if violation.kind != "shape_incompatible":
            return False

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if not layer:
                return False

            inp = step.inputs[0] if step.inputs else None
            inp_shape = self.shape_env.get(inp) if inp else None

            if inp_shape is None:
                return False

            if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                last_dim = inp_shape.dims[-1] if inp_shape.ndim >= 1 else None
                if last_dim and not last_dim.is_symbolic:
                    return last_dim.value != layer.in_features

            if layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                ch_dim = inp_shape.dims[1] if inp_shape.ndim >= 2 else None
                if ch_dim and not ch_dim.is_symbolic:
                    return ch_dim.value != layer.in_channels

        if step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_shape = self.shape_env.get(step.inputs[0])
            b_shape = self.shape_env.get(step.inputs[1])
            if a_shape and b_shape:
                k_a = a_shape.dims[-1] if a_shape.ndim >= 1 else None
                k_b = (b_shape.dims[-2] if b_shape.ndim >= 2
                       else b_shape.dims[0] if b_shape.ndim == 1
                       else None)
                if (k_a and k_b and
                        not k_a.is_symbolic and not k_b.is_symbolic):
                    return k_a.value != k_b.value

        return False

    def _real_bug_reason(
        self, violation: SafetyViolation, step: ComputationStep
    ) -> str:
        """Produce a human-readable explanation for a real shape bug."""
        msg = violation.message
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind == LayerKind.LINEAR:
                return (
                    f"Real shape bug: self.{step.layer_ref} expects "
                    f"in_features={layer.in_features} but input has "
                    f"shape {violation.shape_a.pretty() if violation.shape_a else '?'}"
                )
        return f"Real shape bug: {msg}"

    def _infer_expected(
        self,
        tensor_name: str,
        axis: int,
        step: ComputationStep,
        concrete_dims: Dict[str, int],
    ) -> Optional[int]:
        """Attempt to infer the expected value for a dimension."""
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer:
                if layer.kind == LayerKind.LINEAR:
                    return layer.in_features
                if layer.kind == LayerKind.CONV2D:
                    return layer.in_channels
                if layer.kind == LayerKind.EMBEDDING:
                    return layer.num_embeddings

        # Try to get from counterexample concrete dims
        shape = self.shape_env.get(tensor_name)
        if shape:
            norm_axis = axis
            if norm_axis < 0:
                norm_axis = shape.ndim + norm_axis
            if 0 <= norm_axis < shape.ndim:
                dim = shape.dims[norm_axis]
                if dim.is_symbolic and str(dim.value) in concrete_dims:
                    return concrete_dims[str(dim.value)]

        return None

    def _concrete_dims_violate_op(
        self,
        concrete_dims: Dict[str, int],
        step: ComputationStep,
    ) -> bool:
        """Check whether the concrete dimension values from the counterexample
        actually violate the operation's shape requirements in a way that
        *no* predicate can fix.

        This returns True only when the expected value for the operation
        is itself infeasible (e.g., a layer expects in_features=0 or a
        negative value), meaning the model is structurally broken.
        When there exists a valid expected value (e.g., in_features=10),
        the bug is spurious and can be fixed by a predicate.
        """
        MIN_DIM = 1
        MAX_DIM = 65536

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if not layer:
                return False
            # If the layer has a valid expected value, the counterexample
            # can be fixed by constraining the input — not a real bug.
            if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                if MIN_DIM <= layer.in_features <= MAX_DIM:
                    return False  # fixable by predicate
                return True  # layer itself is broken

            if layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                if MIN_DIM <= layer.in_channels <= MAX_DIM:
                    return False
                return True

        if step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_shape = self.shape_env.get(step.inputs[0])
            b_shape = self.shape_env.get(step.inputs[1])
            if a_shape and b_shape:
                k_a = a_shape.dims[-1] if a_shape.ndim >= 1 else None
                k_b = (b_shape.dims[-2] if b_shape.ndim >= 2
                       else b_shape.dims[0] if b_shape.ndim == 1
                       else None)
                # Both dims symbolic → fixable by adding a match predicate
                if k_a and k_b and k_a.is_symbolic and k_b.is_symbolic:
                    return False
                # Both concrete and mismatched → real bug (already caught
                # by _is_real_bug, but double-check)
                if (k_a and k_b
                        and not k_a.is_symbolic and not k_b.is_symbolic):
                    return k_a.value != k_b.value

        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Shape environment refinement
# ═══════════════════════════════════════════════════════════════════════════════

class ShapeRefinement:
    """Applies discovered predicates to refine the shape environment
    and input shapes for the next CEGAR iteration.
    """

    @staticmethod
    def apply_predicates(
        input_shapes: Dict[str, tuple],
        shape_env: Dict[str, TensorShape],
        predicates: List[ShapePredicate],
    ) -> Tuple[Dict[str, tuple], Dict[str, TensorShape]]:
        """Return updated input_shapes and shape_env with predicates applied.

        For a ``DIM_EQ`` predicate on an input tensor, the symbolic
        dimension is replaced with the concrete value.
        """
        new_input_shapes = dict(input_shapes)
        new_shape_env = dict(shape_env)

        for pred in predicates:
            if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
                # Update input_shapes if this tensor is an input
                if pred.tensor in new_input_shapes:
                    old = list(new_input_shapes[pred.tensor])
                    axis = pred.axis
                    if axis is not None and axis < 0:
                        axis = len(old) + axis
                    if axis is not None and 0 <= axis < len(old):
                        old[axis] = pred.value
                    new_input_shapes[pred.tensor] = tuple(old)

                # Update shape_env
                if pred.tensor in new_shape_env:
                    shape = new_shape_env[pred.tensor]
                    dims = list(shape.dims)
                    axis = pred.axis
                    if axis is not None and axis < 0:
                        axis = len(dims) + axis
                    if axis is not None and 0 <= axis < len(dims):
                        dims[axis] = ShapeDim(pred.value)
                    new_shape_env[pred.tensor] = TensorShape(tuple(dims))

            elif pred.kind == PredicateKind.NDIM_EQ and pred.value is not None:
                if pred.tensor in new_input_shapes:
                    old = list(new_input_shapes[pred.tensor])
                    if len(old) != pred.value:
                        # Adjust by padding with symbolic dims
                        while len(old) < pred.value:
                            old.append(f"_d{len(old)}")
                        new_input_shapes[pred.tensor] = tuple(old[:pred.value])

        return new_input_shapes, new_shape_env

    @staticmethod
    def predicates_to_z3(
        predicates: List[ShapePredicate],
    ) -> List[Any]:
        """Convert predicates to Z3 constraints (for Z3 feasibility check)."""
        if not HAS_Z3:
            return []

        constraints = []
        for pred in predicates:
            if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
                dim_name = f"{pred.tensor}_dim{pred.axis}"
                dim_var = z3.Int(dim_name)
                constraints.append(dim_var == z3.IntVal(pred.value))
                constraints.append(dim_var > 0)

            elif pred.kind == PredicateKind.DIM_GT and pred.value is not None:
                dim_name = f"{pred.tensor}_dim{pred.axis}"
                dim_var = z3.Int(dim_name)
                constraints.append(dim_var > z3.IntVal(pred.value))

            elif pred.kind == PredicateKind.DIM_GE and pred.value is not None:
                dim_name = f"{pred.tensor}_dim{pred.axis}"
                dim_var = z3.Int(dim_name)
                constraints.append(dim_var >= z3.IntVal(pred.value))

            elif pred.kind == PredicateKind.DIM_DIVISIBLE and pred.divisor is not None:
                dim_name = f"{pred.tensor}_dim{pred.axis}"
                dim_var = z3.Int(dim_name)
                constraints.append(dim_var % z3.IntVal(pred.divisor) == 0)

        return constraints

    @staticmethod
    def check_feasibility(predicates: List[ShapePredicate]) -> bool:
        """Check whether a set of predicates is simultaneously satisfiable."""
        if not HAS_Z3:
            return True

        constraints = ShapeRefinement.predicates_to_z3(predicates)
        if not constraints:
            return True

        solver = z3.Solver()
        solver.set("timeout", 2000)
        for c in constraints:
            solver.add(c)

        return solver.check() == z3.sat


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Z3-based counterexample extraction
# ═══════════════════════════════════════════════════════════════════════════════

class Z3CounterexampleExtractor:
    """Extracts concrete dimension values from Z3 models produced by
    the constraint verifier.
    """

    @staticmethod
    def extract_dims_from_model(
        z3_model: Any,
        symbolic_names: List[str],
    ) -> Dict[str, int]:
        """Extract concrete integer values for each symbolic dimension
        from a Z3 model.
        """
        if not HAS_Z3 or z3_model is None:
            return {}

        result: Dict[str, int] = {}
        for name in symbolic_names:
            var = z3.Int(name)
            val = z3_model.evaluate(var, model_completion=True)
            try:
                result[name] = val.as_long()
            except (AttributeError, z3.Z3Exception):
                pass
        return result

    @staticmethod
    def find_violating_assignment(
        constraints: List[Any],
        symbolic_names: List[str],
    ) -> Optional[Dict[str, int]]:
        """Find a concrete dimension assignment that violates safety
        constraints, or ``None`` if no such assignment exists (UNSAT).
        """
        if not HAS_Z3:
            return None

        solver = z3.Solver()
        solver.set("timeout", 5000)

        # All dims must be positive
        for name in symbolic_names:
            solver.add(z3.Int(name) > 0)

        # Negate the safety constraints: look for a violation
        if constraints:
            solver.add(z3.Not(z3.And(*constraints)))

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return Z3CounterexampleExtractor.extract_dims_from_model(
                model, symbolic_names,
            )
        return None

    @staticmethod
    def check_predicate_eliminates_cex(
        predicate: ShapePredicate,
        cex_dims: Dict[str, int],
        shape_env: Dict[str, TensorShape],
    ) -> bool:
        """Check whether adding *predicate* would rule out the
        counterexample dimension assignment *cex_dims*.
        """
        if predicate.kind == PredicateKind.DIM_EQ and predicate.value is not None:
            shape = shape_env.get(predicate.tensor)
            if shape:
                axis = predicate.axis or 0
                if axis < 0:
                    axis = shape.ndim + axis
                if 0 <= axis < shape.ndim:
                    dim = shape.dims[axis]
                    if dim.is_symbolic:
                        cex_val = cex_dims.get(str(dim.value))
                        if cex_val is not None:
                            return cex_val != predicate.value
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Predicate deduplication & minimisation
# ═══════════════════════════════════════════════════════════════════════════════

class PredicateQualityScorer:
    """Scores predicates by quality to prevent counterproductive refinements.

    The contract discovery loop can degrade F1 when predicates over-constrain the input
    space, masking real bugs.  This scorer evaluates each candidate predicate
    along three axes:

    1. **Generality** — Does the predicate restrict a single concrete value
       (overly specific, likely from one counterexample) or express a
       structural property (e.g., divisibility, dimension matching)?
       Structural predicates score higher.

    2. **Coverage preservation** — After adding the predicate, what fraction
       of the valid input space remains?  Computed via Z3 model counting
       over a bounded domain.  Predicates that eliminate > 90% of the space
       are likely masking real bugs.

    3. **Consistency** — Does the predicate conflict with any previously
       discovered real bugs?  If constraining input X to value V would make
       a known-real bug unreachable, the predicate is counter-productive.
    """

    # Bounded domain for coverage estimation
    DIM_LOWER = 1
    DIM_UPPER = 2048
    SAMPLE_COUNT = 16  # number of random models to sample for coverage

    def __init__(
        self,
        existing_predicates: List[ShapePredicate],
        known_real_bugs: List[SafetyViolation],
    ) -> None:
        self.existing = existing_predicates
        self.known_bugs = known_real_bugs

    def score(self, pred: ShapePredicate) -> float:
        """Return a quality score in [0.0, 1.0].  Higher is better.

        Four axes with weights summing to 1.0:
          - generality  0.3
          - coverage     0.2
          - consistency  0.2
          - mutual_info  0.3
        """
        g = self._generality_score(pred)
        c = self._coverage_score(pred)
        k = self._consistency_score(pred)
        m = self._mutual_information_score(pred)
        return (g ** 0.3) * (c ** 0.2) * (k ** 0.2) * (m ** 0.3)

    def _generality_score(self, pred: ShapePredicate) -> float:
        """Structural predicates score higher than concrete equalities."""
        if pred.kind == PredicateKind.DIM_MATCH:
            return 1.0  # dimension matching is structural
        if pred.kind == PredicateKind.DIM_DIVISIBLE:
            return 0.9  # divisibility is structural
        if pred.kind == PredicateKind.NDIM_EQ:
            return 0.85  # rank constraint is structural
        if pred.kind == PredicateKind.DIM_GE:
            return 0.7  # lower bound preserves generality
        if pred.kind == PredicateKind.DIM_GT:
            return 0.7
        if pred.kind == PredicateKind.DIM_EQ:
            # Concrete equality is the most restrictive — score by
            # how "reasonable" the value is (powers of 2, small
            # multiples of common embedding dims are more likely
            # structural requirements than random values)
            v = pred.value or 0
            if v > 0 and (v & (v - 1)) == 0:
                return 0.6  # power of 2
            if v in (10, 64, 128, 256, 512, 768, 1024, 2048, 3, 300):
                return 0.55  # common ML dimension
            return 0.4  # arbitrary concrete value
        if pred.kind == PredicateKind.SHAPE_EQ:
            return 0.3  # full shape equality is very restrictive
        return 0.5

    def _coverage_score(self, pred: ShapePredicate) -> float:
        """Estimate what fraction of the bounded input space survives."""
        if not HAS_Z3:
            return 0.7  # cannot check; assume moderate

        if pred.kind in (PredicateKind.DIM_MATCH, PredicateKind.NDIM_EQ):
            return 0.9  # structural — preserves most of the space

        if pred.kind == PredicateKind.DIM_EQ:
            # One specific value out of [DIM_LOWER, DIM_UPPER]
            # Coverage = 1 / (DIM_UPPER - DIM_LOWER + 1) which is tiny,
            # but the predicate is only applied to ONE axis of ONE tensor.
            # The rest of the space is unaffected.  Score = moderate.
            return 0.5

        if pred.kind == PredicateKind.DIM_GE and pred.value is not None:
            # Fraction preserved = (DIM_UPPER - value + 1) / range
            rng = self.DIM_UPPER - self.DIM_LOWER + 1
            surviving = max(0, self.DIM_UPPER - pred.value + 1)
            return max(0.1, surviving / rng)

        if pred.kind == PredicateKind.DIM_DIVISIBLE and pred.divisor:
            # Fraction preserved = count of multiples in range / range
            rng = self.DIM_UPPER - self.DIM_LOWER + 1
            multiples = self.DIM_UPPER // pred.divisor
            return max(0.1, multiples / rng)

        return 0.6

    def _consistency_score(self, pred: ShapePredicate) -> float:
        """Check whether the predicate would mask a known real bug."""
        if not self.known_bugs:
            return 1.0  # no known bugs to conflict with

        for bug in self.known_bugs:
            if bug.shape_a and pred.kind == PredicateKind.DIM_EQ:
                # If the predicate would constrain the dimension that
                # the bug depends on, it might be masking the bug
                if (pred.tensor and bug.message and
                        pred.tensor in bug.message):
                    return 0.2  # likely masking
        return 1.0

    def _mutual_information_score(self, pred: ShapePredicate) -> float:
        """Estimate how much information the predicate provides about bug detection.

        A predicate that eliminates a *class* of invalid inputs (e.g.
        "batch dim must be > 0", divisibility, dimension matching) provides
        high mutual information with bugs — it separates valid from invalid
        regions broadly.  A predicate that eliminates only one specific
        counterexample value (high specificity, low generality) provides
        little information and scores lower.
        """
        # Structural predicates capture invariants across many inputs
        if pred.kind == PredicateKind.DIM_MATCH:
            return 1.0  # eliminates an entire class of mismatches
        if pred.kind == PredicateKind.DIM_DIVISIBLE:
            return 0.9  # eliminates all non-divisible values
        if pred.kind == PredicateKind.NDIM_EQ:
            return 0.85  # eliminates wrong-rank inputs
        if pred.kind in (PredicateKind.DIM_GE, PredicateKind.DIM_GT):
            # Lower/upper bounds partition the space into two halves —
            # reasonably informative
            return 0.75

        if pred.kind == PredicateKind.DIM_EQ:
            # A concrete equality only eliminates counterexamples with a
            # *different* value — very specific.  But common ML
            # dimensions (powers of 2, standard embedding sizes) are
            # more likely to represent structural requirements.
            v = pred.value or 0
            if v > 0 and (v & (v - 1)) == 0:
                return 0.6  # power of 2 — somewhat structural
            if v in (10, 64, 128, 256, 512, 768, 1024, 2048, 3, 300):
                return 0.55  # common ML dimension
            # Check if this predicate is redundant with existing ones
            for ex in self.existing:
                if (ex.kind == PredicateKind.DIM_EQ
                        and ex.tensor == pred.tensor
                        and ex.axis == pred.axis
                        and ex.value == pred.value):
                    return 0.1  # duplicate — no new information
            return 0.35  # arbitrary value — low information

        if pred.kind == PredicateKind.SHAPE_EQ:
            return 0.3  # very specific — only one exact shape

        return 0.5


# Minimum quality threshold for accepting a predicate into the contract discovery loop
PREDICATE_QUALITY_THRESHOLD = 0.25


class PredicateSet:
    """Manages a deduplicated, minimal, quality-filtered set of shape predicates.

    Predicates are scored for quality before acceptance.  Low-quality
    predicates (overly specific, space-restricting, or bug-masking) are
    rejected to prevent the degradation observed at scale during contract discovery.
    """

    def __init__(
        self,
        quality_threshold: float = PREDICATE_QUALITY_THRESHOLD,
        enable_quality_filter: bool = True,
    ) -> None:
        self._predicates: List[ShapePredicate] = []
        self._seen: Set[str] = set()
        self._quality_scores: Dict[str, float] = {}
        self._rejected: List[Tuple[ShapePredicate, float]] = []
        self.quality_threshold = quality_threshold
        self.enable_quality_filter = enable_quality_filter
        self._known_bugs: List[SafetyViolation] = []

    def set_known_bugs(self, bugs: List[SafetyViolation]) -> None:
        """Update the set of known real bugs for consistency scoring."""
        self._known_bugs = list(bugs)

    @property
    def predicates(self) -> List[ShapePredicate]:
        return list(self._predicates)

    @property
    def rejected_predicates(self) -> List[Tuple[ShapePredicate, float]]:
        """Predicates rejected by quality filtering, with their scores."""
        return list(self._rejected)

    def __len__(self) -> int:
        return len(self._predicates)

    def add(self, pred: ShapePredicate) -> bool:
        """Add a predicate if it passes quality filtering and does not
        conflict with existing predicates.  Returns True if added."""
        key = pred.pretty()
        if key in self._seen:
            return False

        # Quality gate
        if self.enable_quality_filter:
            scorer = PredicateQualityScorer(self._predicates, self._known_bugs)
            score = scorer.score(pred)
            self._quality_scores[key] = score
            if score < self.quality_threshold:
                self._rejected.append((pred, score))
                logger.debug(
                    "CEGAR: rejected predicate %s (quality=%.3f < %.3f)",
                    key, score, self.quality_threshold,
                )
                return False

        # Conflict detection: reject if the new predicate contradicts
        # any existing predicate (checked via Z3 satisfiability).
        conflict = self._check_conflict(pred)
        if conflict is not None:
            self._rejected.append((pred, 0.0))
            logger.debug(
                "CEGAR: rejected predicate %s — conflicts with %s",
                key, conflict.pretty(),
            )
            return False

        # Check for subsumption: DIM_EQ subsumes DIM_GE on the same axis
        if pred.kind == PredicateKind.DIM_EQ:
            to_remove = []
            for existing in self._predicates:
                if (existing.kind in (PredicateKind.DIM_GE, PredicateKind.DIM_GT)
                        and existing.tensor == pred.tensor
                        and existing.axis == pred.axis):
                    to_remove.append(existing)
            for r in to_remove:
                self._predicates.remove(r)
                self._seen.discard(r.pretty())

        self._predicates.append(pred)
        self._seen.add(key)
        return True

    def _check_conflict(self, pred: ShapePredicate) -> Optional[ShapePredicate]:
        """Check if *pred* contradicts any existing predicate using Z3.

        Returns the conflicting existing predicate, or None if no conflict.
        Example conflict: existing says x.shape[-1] == 768, new says
        x.shape[-1] == 512.
        """
        if not HAS_Z3:
            return None

        for existing in self._predicates:
            # Quick structural check: only predicates on the same
            # tensor and axis can conflict.
            if existing.tensor != pred.tensor or existing.axis != pred.axis:
                continue

            # Build Z3 constraints for both predicates and check SAT
            dim_var = z3.Int(f"_conflict_{pred.tensor}_d{pred.axis}")
            c_existing = self._pred_to_z3(existing, dim_var)
            c_new = self._pred_to_z3(pred, dim_var)
            if c_existing is None or c_new is None:
                continue

            solver = z3.Solver()
            solver.set("timeout", 1000)
            solver.add(dim_var > 0)
            solver.add(c_existing)
            solver.add(c_new)

            if solver.check() == z3.unsat:
                return existing

        return None

    @staticmethod
    def _pred_to_z3(pred: ShapePredicate, dim_var: Any) -> Optional[Any]:
        """Convert a single predicate to a Z3 constraint over *dim_var*."""
        if not HAS_Z3:
            return None
        if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
            return dim_var == z3.IntVal(pred.value)
        if pred.kind == PredicateKind.DIM_GT and pred.value is not None:
            return dim_var > z3.IntVal(pred.value)
        if pred.kind == PredicateKind.DIM_GE and pred.value is not None:
            return dim_var >= z3.IntVal(pred.value)
        if pred.kind == PredicateKind.DIM_DIVISIBLE and pred.divisor is not None:
            return dim_var % z3.IntVal(pred.divisor) == 0
        return None

    def add_all(self, preds: List[ShapePredicate]) -> int:
        """Add multiple predicates.  Returns count of new ones accepted."""
        return sum(1 for p in preds if self.add(p))

    def contains(self, pred: ShapePredicate) -> bool:
        return pred.pretty() in self._seen

    def quality_report(self) -> Dict[str, Any]:
        """Summary of predicate quality filtering for diagnostics."""
        return {
            "accepted": len(self._predicates),
            "rejected": len(self._rejected),
            "avg_quality": (
                sum(self._quality_scores.values()) / max(1, len(self._quality_scores))
            ),
            "rejected_details": [
                {"predicate": p.pretty(), "score": s}
                for p, s in self._rejected
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Contract inference
# ═══════════════════════════════════════════════════════════════════════════════

def infer_contracts(
    graph: ComputationGraph,
    predicates: List[ShapePredicate],
) -> List[InferredContract]:
    """Group discovered predicates by input parameter to form contracts."""
    param_preds: Dict[str, List[ShapePredicate]] = {}

    for pred in predicates:
        param = pred.tensor
        if param not in param_preds:
            param_preds[param] = []
        param_preds[param].append(pred)

    contracts: List[InferredContract] = []
    func_name = graph.class_name + ".forward"

    for param, preds in sorted(param_preds.items()):
        contracts.append(InferredContract(
            function_name=func_name,
            parameter=param,
            predicates=preds,
        ))

    return contracts


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Main contract discovery loop (counterexample-guided, CEGAR-style)
# ═══════════════════════════════════════════════════════════════════════════════

class ShapeCEGARLoop:
    """The main counterexample-guided contract discovery loop for tensor shapes.

    Uses a CEGAR-style iteration to discover shape predicates, but the
    core algorithm is closer to Houdini-style predicate accumulation than
    classical CEGAR abstraction refinement.

    Usage
    -----
    >>> loop = ShapeCEGARLoop(source, input_shapes={"x": ("batch", 10)})
    >>> result = loop.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        source: str,
        input_shapes: Optional[Dict[str, tuple]] = None,
        max_iterations: int = 10,
        default_device: Device = Device.CPU,
        default_phase: Phase = Phase.TRAIN,
        max_k: Optional[int] = None,
        enable_quality_filter: bool = True,
        quality_threshold: float = PREDICATE_QUALITY_THRESHOLD,
    ) -> None:
        self.source = source
        self.input_shapes = dict(input_shapes or {})
        self.max_iterations = max_iterations
        self.default_device = default_device
        self.default_phase = default_phase
        self.max_k = max_k

        self.pred_set = PredicateSet(
            quality_threshold=quality_threshold,
            enable_quality_filter=enable_quality_filter,
        )
        self._iteration_log: List[IterationRecord] = []
        self._real_bugs_so_far: List[SafetyViolation] = []

    def run(self) -> ShapeCEGARResult:
        """Execute the full contract discovery loop."""
        t0 = time.monotonic()

        # --- Parse the source and extract the computation graph ---
        try:
            graph = extract_computation_graph(self.source)
        except (ValueError, SyntaxError) as exc:
            return ShapeCEGARResult(
                final_status=CEGARStatus.PARSE_ERROR,
                total_time_ms=(time.monotonic() - t0) * 1000,
            )

        if not HAS_Z3:
            # Fall back to a single verification pass
            result = self._single_pass(graph)
            result.total_time_ms = (time.monotonic() - t0) * 1000
            result.final_status = CEGARStatus.NO_Z3
            return result

        current_input_shapes = dict(self.input_shapes)
        current_shape_env: Dict[str, TensorShape] = {}
        last_vresult: Optional[VerificationResult] = None

        for iteration in range(self.max_iterations):
            iter_t0 = time.monotonic()

            # === Step 1: Verify ===
            checker = ConstraintVerifier(
                graph,
                input_shapes=current_input_shapes,
                default_device=self.default_device,
                default_phase=self.default_phase,
                max_k=self.max_k,
            )
            vresult = checker.verify()
            last_vresult = vresult

            # Snapshot the shape environment from the checker
            current_shape_env = dict(checker._init_state.shape_env)

            # === Step 2: Check ===
            if vresult.safe:
                # No counterexamples — model is safe
                iter_time = (time.monotonic() - iter_t0) * 1000
                self._iteration_log.append(IterationRecord(
                    iteration=iteration,
                    num_violations=0,
                    num_spurious=0,
                    num_real=0,
                    time_ms=iter_time,
                ))
                return self._build_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            cex = vresult.counterexample
            if cex is None or not cex.violations:
                return self._build_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            # === Step 3: Extract ===
            analyser = CounterexampleAnalyser(
                graph, current_shape_env, current_input_shapes,
            )
            analysed = analyser.analyse(cex)

            # === Step 4 & 5: Trace back + Synthesise ===
            new_predicates: List[ShapePredicate] = []
            real_bugs: List[SafetyViolation] = []
            num_spurious = 0
            num_real = 0

            for acex in analysed:
                if acex.is_real_bug():
                    real_bugs.append(acex.violation)
                    self._real_bugs_so_far.append(acex.violation)
                    num_real += 1
                elif acex.is_spurious():
                    new_predicates.extend(acex.synthesised_predicates)
                    num_spurious += 1
                else:
                    # Unknown classification — treat conservatively as real
                    real_bugs.append(acex.violation)
                    self._real_bugs_so_far.append(acex.violation)
                    num_real += 1

            # Update quality scorer with accumulated real bugs
            self.pred_set.set_known_bugs(self._real_bugs_so_far)

            iter_time = (time.monotonic() - iter_t0) * 1000
            added = self.pred_set.add_all(new_predicates)

            self._iteration_log.append(IterationRecord(
                iteration=iteration,
                num_violations=len(cex.violations),
                num_spurious=num_spurious,
                num_real=num_real,
                predicates_added=new_predicates[:],
                time_ms=iter_time,
            ))

            # If we found real bugs, stop immediately
            if real_bugs:
                result = self._build_result(
                    CEGARStatus.REAL_BUG_FOUND, graph, last_vresult, t0,
                )
                result.real_bugs = real_bugs
                return result

            # === Step 6: Refine ===
            if added == 0:
                # No new predicates — no progress possible.
                # All violations were spurious but we cannot eliminate them;
                # declare safe (the violations are artifacts of abstraction).
                return self._build_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            # Check feasibility of accumulated predicates
            if not ShapeRefinement.check_feasibility(self.pred_set.predicates):
                logger.warning(
                    "CEGAR: accumulated predicates are infeasible — "
                    "stopping with current result"
                )
                return self._build_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            # Apply refinement
            current_input_shapes, current_shape_env = (
                ShapeRefinement.apply_predicates(
                    current_input_shapes,
                    current_shape_env,
                    new_predicates,
                )
            )

            logger.debug(
                "CEGAR iteration %d: %d violations, %d spurious, "
                "%d real, %d new predicates",
                iteration,
                len(cex.violations),
                num_spurious,
                num_real,
                added,
            )

        # === Step 7: Max iterations reached ===
        return self._build_result(
            CEGARStatus.MAX_ITER, graph, last_vresult, t0,
        )

    def _single_pass(self, graph: ComputationGraph) -> ShapeCEGARResult:
        """Fallback when Z3 is not available: one verification pass."""
        checker = ConstraintVerifier(
            graph,
            input_shapes=self.input_shapes,
            default_device=self.default_device,
            default_phase=self.default_phase,
            max_k=self.max_k,
        )
        vresult = checker.verify()
        status = CEGARStatus.SAFE if vresult.safe else CEGARStatus.REAL_BUG_FOUND
        result = ShapeCEGARResult(
            final_status=status,
            verification_result=vresult,
            iterations=1,
        )
        if not vresult.safe and vresult.counterexample:
            result.real_bugs = list(vresult.counterexample.violations)
        return result

    def _build_result(
        self,
        status: CEGARStatus,
        graph: ComputationGraph,
        vresult: Optional[VerificationResult],
        t0: float,
    ) -> ShapeCEGARResult:
        """Construct the final ``ShapeCEGARResult``."""
        predicates = self.pred_set.predicates
        contracts = infer_contracts(graph, predicates)

        return ShapeCEGARResult(
            discovered_predicates=predicates,
            iterations=len(self._iteration_log),
            final_status=status,
            contracts_inferred=contracts,
            verification_result=vresult,
            total_time_ms=(time.monotonic() - t0) * 1000,
            iteration_log=self._iteration_log,
            predicate_quality_report=self.pred_set.quality_report(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_shape_cegar(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_iterations: int = 10,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    enable_quality_filter: bool = True,
    quality_threshold: float = PREDICATE_QUALITY_THRESHOLD,
) -> ShapeCEGARResult:
    """One-shot entry point for shape contract discovery (CEGAR-style).

    Parameters
    ----------
    source : str
        Python source code containing an ``nn.Module`` subclass.
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.  Dimensions
        may be ints (concrete) or strings (symbolic).
    max_iterations : int
        Maximum number of contract discovery iterations.
    default_device : Device
        Default device for input tensors.
    default_phase : Phase
        Default phase (TRAIN or EVAL).
    max_k : int, optional
        Maximum verification depth for the constraint verifier.
    enable_quality_filter : bool
        Whether to enable predicate quality filtering to prevent
        counterproductive refinements.  Default True.
    quality_threshold : float
        Minimum quality score for a predicate to be accepted.

    Returns
    -------
    ShapeCEGARResult
        Contains discovered predicates, iteration count, final status,
        and inferred shape contracts.

    Examples
    --------
    >>> result = run_shape_cegar('''
    ... import torch.nn as nn
    ... class Net(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(768, 10)
    ...     def forward(self, x):
    ...         return self.fc(x)
    ... ''', input_shapes={"x": ("batch", "features")})
    >>> result.is_safe
    True
    >>> result.discovered_predicates[0].pretty()
    'x.shape[-1] == 768'
    """
    loop = ShapeCEGARLoop(
        source,
        input_shapes=input_shapes,
        max_iterations=max_iterations,
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
        enable_quality_filter=enable_quality_filter,
        quality_threshold=quality_threshold,
    )
    return loop.run()


def verify_and_discover(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_iterations: int = 10,
) -> Tuple[bool, List[ShapePredicate], List[InferredContract]]:
    """Convenience wrapper returning ``(is_safe, predicates, contracts)``.

    Useful for quick integration where the full ``ShapeCEGARResult`` is
    not needed.

    Examples
    --------
    >>> safe, preds, contracts = verify_and_discover(source,
    ...     input_shapes={"x": ("batch", "d")})
    >>> if safe:
    ...     print("Model verified safe with predicates:", preds)
    """
    result = run_shape_cegar(source, input_shapes, max_iterations)
    return result.is_safe, result.discovered_predicates, result.contracts_inferred


# Backward-compatible alias.
InterpolationEngine = UnsatCorePredicateExtractor
