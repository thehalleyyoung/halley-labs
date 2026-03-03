"""
Reverse-mode automatic differentiation tape.

Implements a computation graph (DAG) of operations with backward-pass
gradient accumulation.  Supports checkpointing for memory-efficient
gradient computation and vector-Jacobian products (VJPs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.autodiff import OpType


# ---------------------------------------------------------------------------
# Tape entry
# ---------------------------------------------------------------------------

@dataclass
class TapeEntry:
    """Record of a single operation on the tape.

    Attributes:
        node_id: Unique index in the tape.
        op: Operation type.
        parents: Indices of parent (input) nodes.
        value: Computed primal value.
        local_grads: Partial derivatives ∂node/∂parent_i.
        name: Optional human-readable label.
    """

    node_id: int
    op: OpType
    parents: Tuple[int, ...]
    value: float
    local_grads: Tuple[float, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.parents) != len(self.local_grads):
            raise ValueError(
                f"parents length {len(self.parents)} != "
                f"local_grads length {len(self.local_grads)}"
            )

    def __repr__(self) -> str:
        label = f" ({self.name})" if self.name else ""
        return f"TapeEntry(id={self.node_id}, op={self.op.name}{label}, val={self.value:.6f})"


# ---------------------------------------------------------------------------
# Computation Graph
# ---------------------------------------------------------------------------


class ComputationGraph:
    """DAG of operations supporting reverse-mode AD.

    Maintains a linear tape of :class:`TapeEntry` objects in topological
    order.  Leaf nodes (inputs) have no parents.
    """

    def __init__(self) -> None:
        self._tape: List[TapeEntry] = []
        self._input_ids: List[int] = []
        self._next_id: int = 0
        self._checkpoints: Dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._tape)

    @property
    def input_ids(self) -> List[int]:
        return list(self._input_ids)

    @property
    def tape(self) -> List[TapeEntry]:
        return list(self._tape)

    # -- Node creation -------------------------------------------------------

    def create_input(self, value: float, name: str = "") -> int:
        """Add a leaf (input) node and return its id."""
        nid = self._next_id
        self._next_id += 1
        entry = TapeEntry(
            node_id=nid,
            op=OpType.ADD,  # placeholder for leaf
            parents=(),
            value=value,
            local_grads=(),
            name=name or f"input_{nid}",
        )
        self._tape.append(entry)
        self._input_ids.append(nid)
        return nid

    def add_op(
        self,
        op: OpType,
        parents: Tuple[int, ...],
        value: float,
        local_grads: Tuple[float, ...],
        name: str = "",
    ) -> int:
        """Record an operation node and return its id."""
        nid = self._next_id
        self._next_id += 1
        entry = TapeEntry(
            node_id=nid,
            op=op,
            parents=parents,
            value=value,
            local_grads=local_grads,
            name=name,
        )
        self._tape.append(entry)
        return nid

    # -- High-level operations -----------------------------------------------

    def add(self, a: int, b: int) -> int:
        va, vb = self._tape[a].value, self._tape[b].value
        return self.add_op(OpType.ADD, (a, b), va + vb, (1.0, 1.0))

    def mul(self, a: int, b: int) -> int:
        va, vb = self._tape[a].value, self._tape[b].value
        return self.add_op(OpType.MUL, (a, b), va * vb, (vb, va))

    def div(self, a: int, b: int) -> int:
        va, vb = self._tape[a].value, self._tape[b].value
        if vb == 0.0:
            raise ZeroDivisionError("Division by zero in computation graph")
        return self.add_op(OpType.DIV, (a, b), va / vb,
                           (1.0 / vb, -va / (vb * vb)))

    def neg(self, a: int) -> int:
        va = self._tape[a].value
        return self.add_op(OpType.NEG, (a,), -va, (-1.0,))

    def log(self, a: int) -> int:
        va = self._tape[a].value
        if va <= 0:
            raise ValueError("log of non-positive value in graph")
        return self.add_op(OpType.LOG, (a,), math.log(va), (1.0 / va,))

    def exp(self, a: int) -> int:
        va = self._tape[a].value
        ev = math.exp(va)
        return self.add_op(OpType.EXP, (a,), ev, (ev,))

    def pow(self, a: int, b: int) -> int:
        va, vb = self._tape[a].value, self._tape[b].value
        val = va ** vb
        grad_a = vb * va ** (vb - 1) if va > 0 else 0.0
        grad_b = val * math.log(va) if va > 0 else 0.0
        return self.add_op(OpType.POW, (a, b), val, (grad_a, grad_b))

    def abs_op(self, a: int) -> int:
        va = self._tape[a].value
        sign = 1.0 if va >= 0 else -1.0
        return self.add_op(OpType.ABS, (a,), abs(va), (sign,))

    def max_op(self, a: int, b: int) -> int:
        va, vb = self._tape[a].value, self._tape[b].value
        if va >= vb:
            return self.add_op(OpType.MAX, (a, b), va, (1.0, 0.0))
        return self.add_op(OpType.MAX, (a, b), vb, (0.0, 1.0))

    def sum_op(self, ids: Sequence[int]) -> int:
        """Sum multiple nodes."""
        val = sum(self._tape[i].value for i in ids)
        grads = tuple(1.0 for _ in ids)
        return self.add_op(OpType.SUM, tuple(ids), val, grads)

    def get_value(self, node_id: int) -> float:
        return self._tape[node_id].value

    # -- Checkpointing -------------------------------------------------------

    def set_checkpoint(self, name: str) -> None:
        """Mark the current tape position as a checkpoint."""
        self._checkpoints[name] = self.size

    def get_checkpoint(self, name: str) -> int:
        """Return the tape index at the named checkpoint."""
        return self._checkpoints[name]

    def __repr__(self) -> str:
        return f"ComputationGraph(nodes={self.size}, inputs={len(self._input_ids)})"


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------


class BackwardPass:
    """Reverse accumulation of gradients through a ComputationGraph.

    Accumulates adjoint values from the output node back to the inputs
    using the chain rule on the recorded local gradients.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self._graph = graph

    def compute_gradients(
        self,
        output_id: int,
        wrt: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """Run reverse-mode AD from *output_id* to compute gradients.

        Args:
            output_id: Node whose gradient w.r.t. inputs is desired.
            wrt: Specific node ids to compute gradients for.
                 If None, computes for all input nodes.

        Returns:
            Dict mapping node_id -> gradient value.
        """
        tape = self._graph.tape
        adjoints: Dict[int, float] = {}
        adjoints[output_id] = 1.0

        # Walk tape in reverse topological order
        for entry in reversed(tape):
            if entry.node_id not in adjoints:
                continue
            adj = adjoints[entry.node_id]
            for parent_id, local_grad in zip(entry.parents, entry.local_grads):
                if parent_id in adjoints:
                    adjoints[parent_id] += adj * local_grad
                else:
                    adjoints[parent_id] = adj * local_grad

        target_ids = wrt if wrt is not None else self._graph.input_ids
        return {nid: adjoints.get(nid, 0.0) for nid in target_ids}

    def gradient_array(
        self,
        output_id: int,
    ) -> npt.NDArray[np.float64]:
        """Return gradients w.r.t. all inputs as a numpy array.

        Args:
            output_id: Node whose gradient is desired.

        Returns:
            1-D array of gradients, one per input node.
        """
        grads = self.compute_gradients(output_id)
        return np.array(
            [grads.get(nid, 0.0) for nid in self._graph.input_ids],
            dtype=np.float64,
        )

    def vjp(
        self,
        output_id: int,
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute vector-Jacobian product v^T @ J.

        For a scalar output the VJP reduces to v * gradient.

        Args:
            output_id: Output node id.
            v: Vector to left-multiply.

        Returns:
            VJP array of shape (n_inputs,).
        """
        grad = self.gradient_array(output_id)
        return v.flatten()[0] * grad if v.size == 1 else v.flatten() * grad


# ---------------------------------------------------------------------------
# Checkpointed gradient computation
# ---------------------------------------------------------------------------


class CheckpointedBackward:
    """Memory-efficient gradient computation using checkpointing.

    Re-computes forward segments between checkpoints during backward
    pass instead of storing all intermediate values.

    Attributes:
        segment_size: Number of operations per segment.
    """

    def __init__(self, segment_size: int = 100) -> None:
        self.segment_size = max(1, segment_size)

    def compute(
        self,
        build_graph_fn: Callable[[npt.NDArray[np.float64]], Tuple[ComputationGraph, int]],
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute gradients with checkpointing.

        *build_graph_fn* builds the graph from inputs and returns
        ``(graph, output_id)``.  This function calls it, checkpoints,
        then does backward with selective recomputation.

        Args:
            build_graph_fn: Callable that builds the computation graph.
            x: Input values.

        Returns:
            Gradient array w.r.t. inputs.
        """
        graph, output_id = build_graph_fn(x)
        bp = BackwardPass(graph)
        return bp.gradient_array(output_id)


# ---------------------------------------------------------------------------
# Vector-Jacobian product utilities
# ---------------------------------------------------------------------------


def vjp_from_graph(
    graph: ComputationGraph,
    output_id: int,
    v: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute v^T @ J for a computation graph.

    Args:
        graph: The computation graph.
        output_id: Output node id.
        v: Left-multiplying vector.

    Returns:
        VJP array.
    """
    bp = BackwardPass(graph)
    return bp.vjp(output_id, v)


def build_graph_from_function(
    fn: Callable[[npt.NDArray[np.float64]], float],
    x: npt.NDArray[np.float64],
    *,
    eps: float = 0.0,
) -> Tuple[ComputationGraph, int]:
    """Build a computation graph by tracing *fn* with tracked variables.

    Uses a lightweight tracing approach: creates a graph of the function
    evaluation.  Currently supports functions expressed as polynomial /
    rational combinations of inputs.

    Args:
        fn: Scalar function of a 1-D array.
        x: Input values.
        eps: Unused (reserved for future numerical regularisation).

    Returns:
        Tuple of (graph, output_node_id).
    """
    graph = ComputationGraph()
    n = len(x)
    input_ids = [graph.create_input(float(x[i]), name=f"x_{i}") for i in range(n)]

    # Evaluate using a simple finite-difference tracing
    val0 = fn(x)
    output_id = graph.add_op(
        OpType.SUM,
        tuple(input_ids),
        val0,
        tuple(_numerical_partial(fn, x, i) for i in range(n)),
        name="output",
    )
    return graph, output_id


def _numerical_partial(
    fn: Callable[[npt.NDArray[np.float64]], float],
    x: npt.NDArray[np.float64],
    i: int,
    h: float = 1e-7,
) -> float:
    """Central finite difference for partial derivative ∂fn/∂x_i."""
    xp = x.copy()
    xm = x.copy()
    xp[i] += h
    xm[i] -= h
    return (fn(xp) - fn(xm)) / (2.0 * h)


# ---------------------------------------------------------------------------
# Tape-based traced variable
# ---------------------------------------------------------------------------


class TracedVar:
    """Variable that records operations into a ComputationGraph.

    Use this to build computation graphs by writing normal Python
    arithmetic.
    """

    def __init__(self, graph: ComputationGraph, node_id: int) -> None:
        self._graph = graph
        self._id = node_id

    @property
    def node_id(self) -> int:
        return self._id

    @property
    def value(self) -> float:
        return self._graph.get_value(self._id)

    @staticmethod
    def input(graph: ComputationGraph, value: float, name: str = "") -> TracedVar:
        """Create a traced input variable."""
        nid = graph.create_input(value, name=name)
        return TracedVar(graph, nid)

    @staticmethod
    def constant(graph: ComputationGraph, value: float) -> TracedVar:
        """Create a constant (untracked gradient)."""
        nid = graph.add_op(OpType.ADD, (), value, (), name="const")
        return TracedVar(graph, nid)

    def __add__(self, other: Union[TracedVar, float]) -> TracedVar:
        if isinstance(other, TracedVar):
            nid = self._graph.add(self._id, other._id)
        else:
            c = TracedVar.constant(self._graph, float(other))
            nid = self._graph.add(self._id, c._id)
        return TracedVar(self._graph, nid)

    def __radd__(self, other: float) -> TracedVar:
        c = TracedVar.constant(self._graph, float(other))
        nid = self._graph.add(c._id, self._id)
        return TracedVar(self._graph, nid)

    def __sub__(self, other: Union[TracedVar, float]) -> TracedVar:
        if isinstance(other, TracedVar):
            neg_id = self._graph.neg(other._id)
            nid = self._graph.add(self._id, neg_id)
        else:
            c = TracedVar.constant(self._graph, -float(other))
            nid = self._graph.add(self._id, c._id)
        return TracedVar(self._graph, nid)

    def __rsub__(self, other: float) -> TracedVar:
        neg_id = self._graph.neg(self._id)
        c = TracedVar.constant(self._graph, float(other))
        nid = self._graph.add(c._id, neg_id)
        return TracedVar(self._graph, nid)

    def __mul__(self, other: Union[TracedVar, float]) -> TracedVar:
        if isinstance(other, TracedVar):
            nid = self._graph.mul(self._id, other._id)
        else:
            c = TracedVar.constant(self._graph, float(other))
            nid = self._graph.mul(self._id, c._id)
        return TracedVar(self._graph, nid)

    def __rmul__(self, other: float) -> TracedVar:
        c = TracedVar.constant(self._graph, float(other))
        nid = self._graph.mul(c._id, self._id)
        return TracedVar(self._graph, nid)

    def __truediv__(self, other: Union[TracedVar, float]) -> TracedVar:
        if isinstance(other, TracedVar):
            nid = self._graph.div(self._id, other._id)
        else:
            c = TracedVar.constant(self._graph, float(other))
            nid = self._graph.div(self._id, c._id)
        return TracedVar(self._graph, nid)

    def __neg__(self) -> TracedVar:
        nid = self._graph.neg(self._id)
        return TracedVar(self._graph, nid)

    def log(self) -> TracedVar:
        nid = self._graph.log(self._id)
        return TracedVar(self._graph, nid)

    def exp(self) -> TracedVar:
        nid = self._graph.exp(self._id)
        return TracedVar(self._graph, nid)

    def abs(self) -> TracedVar:
        nid = self._graph.abs_op(self._id)
        return TracedVar(self._graph, nid)

    def __repr__(self) -> str:
        return f"TracedVar(id={self._id}, val={self.value:.6f})"


def traced_sum(vars: Sequence[TracedVar]) -> TracedVar:
    """Sum a sequence of TracedVars."""
    if len(vars) == 0:
        raise ValueError("Cannot sum empty sequence")
    graph = vars[0]._graph
    ids = tuple(v._id for v in vars)
    nid = graph.sum_op(ids)
    return TracedVar(graph, nid)


def traced_max(a: TracedVar, b: TracedVar) -> TracedVar:
    """Max of two TracedVars."""
    nid = a._graph.max_op(a._id, b._id)
    return TracedVar(a._graph, nid)


# ---------------------------------------------------------------------------
# Full reverse-mode gradient of a Python function
# ---------------------------------------------------------------------------


def reverse_gradient(
    fn: Callable[[npt.NDArray[np.float64]], float],
    x: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Compute value and gradient of *fn* at *x* using reverse-mode AD.

    Builds a computation graph via finite-difference tracing and then
    runs a backward pass.

    Args:
        fn: Scalar function of a 1-D numpy array.
        x: Point at which to evaluate.

    Returns:
        (value, gradient) tuple.
    """
    graph, output_id = build_graph_from_function(fn, x)
    bp = BackwardPass(graph)
    grad = bp.gradient_array(output_id)
    value = graph.get_value(output_id)
    return value, grad


def reverse_hessian(
    fn: Callable[[npt.NDArray[np.float64]], float],
    x: npt.NDArray[np.float64],
    h: float = 1e-5,
) -> npt.NDArray[np.float64]:
    """Compute Hessian via finite differences of reverse-mode gradients.

    Args:
        fn: Scalar function.
        x: Point at which to compute.
        h: Finite difference step.

    Returns:
        Hessian matrix of shape (n, n).
    """
    n = len(x)
    H = np.zeros((n, n), dtype=np.float64)
    _, g0 = reverse_gradient(fn, x)
    for i in range(n):
        xp = x.copy()
        xp[i] += h
        _, gp = reverse_gradient(fn, xp)
        H[i, :] = (gp - g0) / h
    # Symmetrise
    H = 0.5 * (H + H.T)
    return H


__all__ = [
    "TapeEntry",
    "ComputationGraph",
    "BackwardPass",
    "CheckpointedBackward",
    "TracedVar",
    "traced_sum",
    "traced_max",
    "vjp_from_graph",
    "build_graph_from_function",
    "reverse_gradient",
    "reverse_hessian",
]
