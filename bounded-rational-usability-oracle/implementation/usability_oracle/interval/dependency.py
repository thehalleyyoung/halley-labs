"""Dependency tracking for interval computations.

Tracks variable dependencies through a computation graph, enabling:

* **Sensitivity analysis** — compute ∂output/∂variable using interval
  extensions of the partial derivatives.
* **Correlation tracking** — identify induced correlations between
  variables introduced by shared sub-expressions.
* **Constraint propagation** — use hull-consistency (HC4) and
  box-consistency (interval Newton) contractors to tighten variable
  domains given constraints.

The :class:`DependencyTracker` maintains a DAG of operations where
each node records its operation type, input variables, and output
interval.  This graph is traversed to compute sensitivities and to
drive constraint-propagation algorithms.

References
----------
Jaulin, L., Kieffer, M., Didrit, O., & Walter, É. (2001).
    *Applied Interval Analysis*. Springer.
Benhamou, F., Goualard, F., Granvilliers, L., & Puget, J.-F. (1999).
    Revising hull and box consistency. *ICLP*, 230–244.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from usability_oracle.interval.interval import Interval


# ---------------------------------------------------------------------------
# Operation types
# ---------------------------------------------------------------------------

@unique
class OpType(Enum):
    """Types of operations in the computation graph."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"
    NEG = "neg"
    ABS = "abs"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Computation graph node
# ---------------------------------------------------------------------------

@dataclass
class ComputationNode:
    """A node in the dependency computation graph.

    Attributes
    ----------
    node_id : str
        Unique identifier.
    op : OpType
        Operation that produced this node.
    inputs : list[str]
        IDs of input nodes.
    output : Interval
        Result interval.
    metadata : dict
        Arbitrary metadata (e.g. constant exponent for POW).
    """

    node_id: str
    op: OpType
    inputs: list[str] = field(default_factory=list)
    output: Interval = field(default_factory=lambda: Interval(0.0, 0.0))
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# DependencyTracker
# ═══════════════════════════════════════════════════════════════════════════

class DependencyTracker:
    """Track variable dependencies through interval computations.

    Usage::

        tracker = DependencyTracker()
        tracker.register_variable("x", Interval(1.0, 2.0))
        tracker.register_variable("y", Interval(3.0, 4.0))
        z = tracker.record_operation(OpType.ADD, ["x", "y"],
                                     Interval(4.0, 6.0), "z")
        sens = tracker.compute_sensitivity("z", "x")
    """

    def __init__(self) -> None:
        self._variables: Dict[str, Interval] = {}
        self._nodes: Dict[str, ComputationNode] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Variable registration
    # ------------------------------------------------------------------

    def register_variable(self, name: str, interval: Interval) -> str:
        """Register a named input variable with its domain.

        Parameters
        ----------
        name : str
            Variable name (must be unique).
        interval : Interval
            Domain of the variable.

        Returns
        -------
        str
            The variable's node ID (same as *name*).

        Raises
        ------
        ValueError
            If *name* is already registered.
        """
        if name in self._variables:
            raise ValueError(f"Variable '{name}' is already registered.")
        self._variables[name] = interval
        node = ComputationNode(
            node_id=name,
            op=OpType.CUSTOM,
            inputs=[],
            output=interval,
            metadata={"is_variable": True},
        )
        self._nodes[name] = node
        return name

    # ------------------------------------------------------------------
    # Operation recording
    # ------------------------------------------------------------------

    def record_operation(
        self,
        op: OpType,
        inputs: list[str],
        output: Interval,
        node_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Record a computation-graph edge.

        Parameters
        ----------
        op : OpType
            The operation performed.
        inputs : list[str]
            Node IDs of the inputs.
        output : Interval
            Result interval.
        node_id : str, optional
            Explicit node ID.  Auto-generated if omitted.
        metadata : dict, optional
            Extra metadata (e.g. exponent for POW).

        Returns
        -------
        str
            The new node's ID.

        Raises
        ------
        KeyError
            If any input ID is unknown.
        """
        for inp in inputs:
            if inp not in self._nodes:
                raise KeyError(f"Unknown input node '{inp}'.")

        if node_id is None:
            self._counter += 1
            node_id = f"_op_{self._counter}"

        node = ComputationNode(
            node_id=node_id,
            op=op,
            inputs=list(inputs),
            output=output,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_variable(self, name: str) -> Interval:
        """Return the domain of a registered variable."""
        return self._variables[name]

    def get_node(self, node_id: str) -> ComputationNode:
        """Return a computation node by ID."""
        return self._nodes[node_id]

    @property
    def variables(self) -> Dict[str, Interval]:
        """All registered variables and their domains."""
        return dict(self._variables)

    @property
    def nodes(self) -> Dict[str, ComputationNode]:
        """All computation-graph nodes."""
        return dict(self._nodes)

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def compute_sensitivity(
        self, output: str, variable: str
    ) -> Interval:
        """Compute ∂output/∂variable using interval extension.

        Uses reverse-mode automatic differentiation over the recorded
        computation graph to propagate interval-valued partial
        derivatives.

        Parameters
        ----------
        output : str
            Node ID of the output quantity.
        variable : str
            Name of the input variable.

        Returns
        -------
        Interval
            Interval enclosure of the partial derivative.

        Raises
        ------
        KeyError
            If *output* or *variable* is unknown.
        """
        if output not in self._nodes:
            raise KeyError(f"Unknown output node '{output}'.")
        if variable not in self._variables:
            raise KeyError(f"Unknown variable '{variable}'.")

        # Reverse-mode: adjoint[output] = [1, 1]
        adjoints: Dict[str, Interval] = {output: Interval(1.0, 1.0)}

        # Topological order (reverse)
        order = self._topological_sort()
        order.reverse()

        for nid in order:
            if nid not in adjoints:
                continue
            adj = adjoints[nid]
            node = self._nodes[nid]

            if node.metadata.get("is_variable"):
                continue

            # Propagate adjoint to inputs
            local_derivs = self._local_derivatives(node)
            for inp_id, local_deriv in zip(node.inputs, local_derivs):
                contribution = adj * local_deriv
                if inp_id in adjoints:
                    adjoints[inp_id] = adjoints[inp_id] + contribution
                else:
                    adjoints[inp_id] = contribution

        return adjoints.get(variable, Interval(0.0, 0.0))

    def _local_derivatives(
        self, node: ComputationNode
    ) -> list[Interval]:
        """Compute interval-valued local partial derivatives for a node."""
        op = node.op
        inputs = [self._nodes[i].output for i in node.inputs]

        if op == OpType.ADD:
            return [Interval(1.0, 1.0), Interval(1.0, 1.0)]

        elif op == OpType.SUB:
            return [Interval(1.0, 1.0), Interval(-1.0, -1.0)]

        elif op == OpType.MUL:
            # ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            return [inputs[1], inputs[0]]

        elif op == OpType.DIV:
            # ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
            b = inputs[1]
            b_sq = b * b
            inv_b = Interval(1.0, 1.0) / b
            neg_a_over_b2 = -(inputs[0] / b_sq)
            return [inv_b, neg_a_over_b2]

        elif op == OpType.NEG:
            return [Interval(-1.0, -1.0)]

        elif op == OpType.EXP:
            # ∂exp(a)/∂a = exp(a)
            return [node.output]

        elif op == OpType.LOG:
            # ∂ln(a)/∂a = 1/a
            inv_a = Interval(1.0, 1.0) / inputs[0]
            return [inv_a]

        elif op == OpType.SQRT:
            # ∂√a/∂a = 1/(2√a)
            two_sqrt = Interval(2.0, 2.0) * node.output
            if two_sqrt.low <= 0.0:
                return [Interval(0.0, math.inf)]
            return [Interval(1.0, 1.0) / two_sqrt]

        elif op == OpType.POW:
            n = node.metadata.get("exponent", 2)
            # ∂(a^n)/∂a = n * a^(n-1)
            if n == 0:
                return [Interval(0.0, 0.0)]
            a = inputs[0]
            deriv = Interval.from_value(float(n)) * (a ** (n - 1))
            return [deriv]

        else:
            # Unknown / custom: conservative [−∞, +∞]
            return [Interval(-math.inf, math.inf)] * len(inputs)

    def _topological_sort(self) -> list[str]:
        """Return node IDs in topological order."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            node = self._nodes[nid]
            for inp in node.inputs:
                visit(inp)
            order.append(nid)

        for nid in self._nodes:
            visit(nid)
        return order

    # ------------------------------------------------------------------
    # Correlation tracking
    # ------------------------------------------------------------------

    def compute_correlation(self, var_a: str, var_b: str) -> bool:
        """Check whether two variables are correlated through the graph.

        Two variables are correlated if they share a common descendant
        in the computation graph (i.e. they appear in the same
        expression).

        Parameters
        ----------
        var_a, var_b : str
            Variable names.

        Returns
        -------
        bool
            True if the variables share a common descendant.
        """
        if var_a not in self._variables or var_b not in self._variables:
            return False

        desc_a = self._descendants(var_a)
        desc_b = self._descendants(var_b)
        return bool(desc_a & desc_b)

    def _descendants(self, node_id: str) -> set[str]:
        """Return set of all descendant node IDs."""
        result: set[str] = set()
        for nid, node in self._nodes.items():
            if node_id in node.inputs:
                result.add(nid)
                result |= self._descendants(nid)
        return result

    # ------------------------------------------------------------------
    # Ancestors (for constraint propagation)
    # ------------------------------------------------------------------

    def _ancestors(self, node_id: str) -> set[str]:
        """Return set of all ancestor node IDs (inputs, recursively)."""
        result: set[str] = set()
        node = self._nodes.get(node_id)
        if node is None:
            return result
        for inp in node.inputs:
            result.add(inp)
            result |= self._ancestors(inp)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Constraint propagation
# ═══════════════════════════════════════════════════════════════════════════

def prune_domains(
    constraints: Sequence[Callable[..., Interval]],
    domains: Dict[str, Interval],
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> Dict[str, Interval]:
    """Constraint propagation to tighten variable domains.

    Repeatedly applies each constraint function as a contractor until
    a fixed point is reached or *max_iterations* is exhausted.

    Each constraint callable should accept the current domains dict
    and return a tightened domain dict (or the same dict if no
    tightening is possible).

    Parameters
    ----------
    constraints : Sequence[Callable]
        Contractor functions.
    domains : Dict[str, Interval]
        Initial variable domains.
    max_iterations : int
        Maximum propagation rounds.
    tolerance : float
        Convergence threshold on total width reduction.

    Returns
    -------
    Dict[str, Interval]
        Tightened domains.
    """
    current = dict(domains)
    for _ in range(max_iterations):
        prev_width = sum(iv.width for iv in current.values())
        for constraint in constraints:
            current = constraint(current)
        new_width = sum(iv.width for iv in current.values())
        if prev_width - new_width < tolerance:
            break
    return current


def hull_consistency(
    f: Callable[..., Interval],
    variables: list[str],
    domains: Dict[str, Interval],
    target: Interval,
) -> Dict[str, Interval]:
    """HC4 hull-consistency contractor.

    For the constraint f(x₁, …, xₙ) ∈ target, tighten each variable
    domain by evaluating the natural interval extension with all other
    variables at their current domains and intersecting the result with
    the target.

    This is a simplified one-pass HC4 algorithm suitable for single
    constraints.

    Parameters
    ----------
    f : Callable
        Function from Interval arguments to Interval result.
    variables : list[str]
        Variable names in the order expected by *f*.
    domains : Dict[str, Interval]
        Current variable domains.
    target : Interval
        Desired range for the constraint output.

    Returns
    -------
    Dict[str, Interval]
        Tightened domains (a copy of the input with modified entries).
    """
    result = dict(domains)
    args = [result[v] for v in variables]
    output = f(*args)

    # Forward evaluation
    contracted_output = output.intersection(target)
    if contracted_output is None:
        # Constraint infeasible — return empty domains
        return {v: Interval(0.0, 0.0) for v in result}

    # Backward pass: for each variable, try to tighten
    for i, var in enumerate(variables):
        original = result[var]
        # Simple strategy: bisect and check containment
        tightened = _backward_tighten(f, variables, result, i, target)
        if tightened is not None:
            result[var] = tightened

    return result


def _backward_tighten(
    f: Callable[..., Interval],
    variables: list[str],
    domains: Dict[str, Interval],
    var_index: int,
    target: Interval,
    n_bisections: int = 8,
) -> Optional[Interval]:
    """Tighten one variable domain via bisection probing."""
    var = variables[var_index]
    dom = domains[var]

    # Try narrowing the lower bound
    lo, hi = dom.low, dom.high
    if lo == hi:
        return dom

    best_lo, best_hi = lo, hi

    # Probe lower bound
    for _ in range(n_bisections):
        mid = (best_lo + hi) / 2.0
        test_domains = dict(domains)
        test_domains[var] = Interval(best_lo, mid)
        args = [test_domains[v] for v in variables]
        try:
            output = f(*args)
            if output.intersection(target) is not None:
                break
            best_lo = mid
        except (ValueError, ZeroDivisionError):
            best_lo = mid

    # Probe upper bound
    for _ in range(n_bisections):
        mid = (lo + best_hi) / 2.0
        test_domains = dict(domains)
        test_domains[var] = Interval(mid, best_hi)
        args = [test_domains[v] for v in variables]
        try:
            output = f(*args)
            if output.intersection(target) is not None:
                break
            best_hi = mid
        except (ValueError, ZeroDivisionError):
            best_hi = mid

    if best_lo > best_hi:
        return None
    return Interval(best_lo, best_hi)


def box_consistency(
    f: Callable[[Interval], Interval],
    f_deriv: Callable[[Interval], Interval],
    domain: Interval,
    target: Interval,
    max_iterations: int = 20,
    tolerance: float = 1e-10,
) -> Optional[Interval]:
    """Box-consistency contractor using interval Newton iteration.

    Applies the interval Newton method to the constraint
    f(x) ∈ target, which is equivalent to finding zeros of
    g(x) = f(x) − mid(target).

    Parameters
    ----------
    f : Callable[[Interval], Interval]
        Univariate interval function.
    f_deriv : Callable[[Interval], Interval]
        Interval extension of the derivative of *f*.
    domain : Interval
        Current domain of the variable.
    target : Interval
        Desired range for f(x).
    max_iterations : int
        Maximum Newton steps.
    tolerance : float
        Convergence threshold on domain width reduction.

    Returns
    -------
    Optional[Interval]
        Tightened domain, or None if the constraint is infeasible.
    """
    target_mid = target.midpoint
    current = domain

    for _ in range(max_iterations):
        mid = current.midpoint
        try:
            f_mid = f(Interval.from_value(mid))
        except (ValueError, ZeroDivisionError):
            return current

        g_mid_lo = f_mid.low - target_mid
        g_mid_hi = f_mid.high - target_mid

        deriv = f_deriv(current)
        if deriv.includes_zero():
            return current

        # Newton step: x_new = mid − g(mid) / g'(domain)
        try:
            step_lo = g_mid_lo / deriv.high if deriv.high != 0.0 else -math.inf
            step_hi = g_mid_hi / deriv.low if deriv.low != 0.0 else math.inf
        except ZeroDivisionError:
            return current

        newton_lo = mid - step_hi
        newton_hi = mid - step_lo

        if newton_lo > newton_hi:
            newton_lo, newton_hi = newton_hi, newton_lo

        newton_interval = Interval(newton_lo, newton_hi)
        contracted = current.intersection(newton_interval)
        if contracted is None:
            return None

        if current.width - contracted.width < tolerance:
            return contracted
        current = contracted

    return current
