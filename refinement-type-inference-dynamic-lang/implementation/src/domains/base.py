"""
Abstract domain base classes.

Each domain forms a Galois connection (α, γ) with the concrete powerset domain.
Provides the foundational abstractions for lattice-based abstract interpretation:
fixed-point solvers, widening/narrowing strategies, worklist algorithms, and
general monotone dataflow frameworks.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------
T = TypeVar("T")
V = TypeVar("V", bound="AbstractValue")
S = TypeVar("S")
K = TypeVar("K", bound=Hashable)
C = TypeVar("C")  # Concrete type
A = TypeVar("A")  # Abstract type


# ===================================================================
# AbstractValue – base for all abstract values
# ===================================================================

class AbstractValue(ABC):
    """Base class for all abstract values in any domain."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    def is_bottom(self) -> bool:
        """Return True if this value represents the bottom element."""
        return False

    def is_top(self) -> bool:
        """Return True if this value represents the top element."""
        return False


# ===================================================================
# JoinSemiLattice / MeetSemiLattice – partial lattice interfaces
# ===================================================================

class JoinSemiLattice(ABC, Generic[T]):
    """A join-semilattice: partially ordered set with least upper bound."""

    @abstractmethod
    def join(self, a: T, b: T) -> T:
        """Return the least upper bound of *a* and *b*."""
        ...

    @abstractmethod
    def leq(self, a: T, b: T) -> bool:
        """Return True if *a* ⊑ *b*."""
        ...

    @abstractmethod
    def bottom(self) -> T:
        ...


class MeetSemiLattice(ABC, Generic[T]):
    """A meet-semilattice: partially ordered set with greatest lower bound."""

    @abstractmethod
    def meet(self, a: T, b: T) -> T:
        """Return the greatest lower bound of *a* and *b*."""
        ...

    @abstractmethod
    def leq(self, a: T, b: T) -> bool:
        ...

    @abstractmethod
    def top(self) -> T:
        ...


# ===================================================================
# Lattice – full lattice interface
# ===================================================================

class Lattice(JoinSemiLattice[T], MeetSemiLattice[T], ABC):
    """Complete lattice with top, bottom, join, and meet."""

    @abstractmethod
    def top(self) -> T:
        ...

    @abstractmethod
    def bottom(self) -> T:
        ...

    @abstractmethod
    def join(self, a: T, b: T) -> T:
        ...

    @abstractmethod
    def meet(self, a: T, b: T) -> T:
        ...

    @abstractmethod
    def leq(self, a: T, b: T) -> bool:
        ...


# ===================================================================
# AbstractDomain – core domain ABC
# ===================================================================

class AbstractDomain(Lattice[V], ABC, Generic[V]):
    """
    Full abstract domain forming a Galois connection (α, γ) with the
    concrete powerset domain.
    """

    # -- lattice operations --------------------------------------------------

    @abstractmethod
    def top(self) -> V:
        """Return ⊤ (most imprecise element)."""
        ...

    @abstractmethod
    def bottom(self) -> V:
        """Return ⊥ (unreachable / empty set)."""
        ...

    @abstractmethod
    def join(self, a: V, b: V) -> V:
        """Least upper bound (over-approximating union)."""
        ...

    @abstractmethod
    def meet(self, a: V, b: V) -> V:
        """Greatest lower bound (intersection)."""
        ...

    @abstractmethod
    def leq(self, a: V, b: V) -> bool:
        """Partial order: a ⊑ b."""
        ...

    # -- widening / narrowing ------------------------------------------------

    def widen(self, a: V, b: V) -> V:
        """Widening operator — guarantees convergence.

        Default: return join(a, b).  Subclasses should override with a proper
        widening that jumps to stable bounds.
        """
        return self.join(a, b)

    def narrow(self, a: V, b: V) -> V:
        """Narrowing operator — recovers precision after widening.

        Default: return meet(a, b).
        """
        return self.meet(a, b)

    # -- predicates ----------------------------------------------------------

    def is_top(self, a: V) -> bool:
        return self.leq(self.top(), a)

    def is_bottom(self, a: V) -> bool:
        return self.leq(a, self.bottom())

    # -- Galois connection ---------------------------------------------------

    @abstractmethod
    def abstract(self, concrete: Any) -> V:
        """Abstraction function α: concrete value → abstract value."""
        ...

    @abstractmethod
    def concretize(self, abstract_val: V) -> Any:
        """Concretization function γ: abstract value → set of concrete values.

        May return a symbolic representation when the concrete set is infinite.
        """
        ...


# ===================================================================
# GaloisConnection – pairs (α, γ)
# ===================================================================

@dataclass(frozen=True)
class GaloisConnection(Generic[C, A]):
    """
    A Galois connection between a concrete domain C and an abstract domain A.

    Consists of:
      α : C → A   (abstraction)
      γ : A → C   (concretization)

    Soundness: ∀c∈C, a∈A.  α(c) ⊑ a  ⟺  c ⊆ γ(a)
    """

    alpha: Callable[[C], A]
    gamma: Callable[[A], C]

    _leq_abstract: Optional[Callable[[A, A], bool]] = None
    _subset_concrete: Optional[Callable[[C, C], bool]] = None

    def abstraction(self, concrete: C) -> A:
        return self.alpha(concrete)

    def concretization(self, abstract_val: A) -> C:
        return self.gamma(abstract_val)

    def check_soundness(self, concrete: C, abstract_val: A) -> bool:
        """Verify the Galois connection soundness property for a pair (c, a).

        Returns True iff α(c) ⊑ a  ⟺  c ⊆ γ(a).
        Requires _leq_abstract and _subset_concrete to be provided.
        """
        if self._leq_abstract is None or self._subset_concrete is None:
            raise ValueError(
                "Soundness check requires _leq_abstract and _subset_concrete"
            )
        alpha_c = self.alpha(concrete)
        gamma_a = self.gamma(abstract_val)
        lhs = self._leq_abstract(alpha_c, abstract_val)
        rhs = self._subset_concrete(concrete, gamma_a)
        return lhs == rhs

    def compose(
        self, other: "GaloisConnection[A, T]"
    ) -> "GaloisConnection[C, T]":
        """Compose two Galois connections: (α₂∘α₁, γ₁∘γ₂)."""
        return GaloisConnection(
            alpha=lambda c: other.alpha(self.alpha(c)),
            gamma=lambda a: self.gamma(other.gamma(a)),
        )


# ===================================================================
# AbstractTransformer – transforms abstract states
# ===================================================================

class AbstractTransformer(ABC, Generic[V]):
    """Transforms abstract states through program operations."""

    @abstractmethod
    def assign(
        self, state: "AbstractState[V]", var: str, expr: Any
    ) -> "AbstractState[V]":
        """Abstract assignment: var := expr."""
        ...

    @abstractmethod
    def guard(
        self, state: "AbstractState[V]", condition: Any, branch: bool
    ) -> "AbstractState[V]":
        """Apply a guard condition on the given branch (True / False)."""
        ...

    @abstractmethod
    def call(
        self,
        state: "AbstractState[V]",
        func: str,
        args: List[Any],
        result_var: Optional[str] = None,
    ) -> "AbstractState[V]":
        """Model a function call."""
        ...

    def forget(self, state: "AbstractState[V]", var: str) -> "AbstractState[V]":
        """Remove information about *var* (set to ⊤)."""
        new_map = dict(state.env)
        new_map.pop(var, None)
        return AbstractState(env=new_map, domain=state.domain)

    def project(
        self, state: "AbstractState[V]", variables: Iterable[str]
    ) -> "AbstractState[V]":
        """Project the state onto *variables*, forgetting everything else."""
        keep = set(variables)
        new_map = {k: v for k, v in state.env.items() if k in keep}
        return AbstractState(env=new_map, domain=state.domain)


# ===================================================================
# AbstractState – mapping from variables to abstract values
# ===================================================================

@dataclass
class AbstractState(Generic[V]):
    """Maps program variables to abstract values."""

    env: Dict[str, V] = field(default_factory=dict)
    domain: Optional[AbstractDomain[V]] = None
    _is_bottom: bool = False

    # -- convenience accessors -----------------------------------------------

    def get(self, var: str) -> Optional[V]:
        return self.env.get(var)

    def set(self, var: str, value: V) -> "AbstractState[V]":
        new_env = dict(self.env)
        new_env[var] = value
        return AbstractState(env=new_env, domain=self.domain, _is_bottom=False)

    def remove(self, var: str) -> "AbstractState[V]":
        new_env = dict(self.env)
        new_env.pop(var, None)
        return AbstractState(env=new_env, domain=self.domain, _is_bottom=self._is_bottom)

    def variables(self) -> Set[str]:
        return set(self.env.keys())

    def items(self) -> Iterable[Tuple[str, V]]:
        return self.env.items()

    # -- lattice operations on states ----------------------------------------

    @classmethod
    def bottom_state(cls, domain: AbstractDomain[V]) -> "AbstractState[V]":
        return cls(env={}, domain=domain, _is_bottom=True)

    @classmethod
    def top_state(
        cls, domain: AbstractDomain[V], variables: Iterable[str]
    ) -> "AbstractState[V]":
        env = {v: domain.top() for v in variables}
        return cls(env=env, domain=domain, _is_bottom=False)

    @property
    def is_bottom(self) -> bool:
        if self._is_bottom:
            return True
        if self.domain is not None:
            return any(self.domain.is_bottom(v) for v in self.env.values())
        return False

    def join(self, other: "AbstractState[V]") -> "AbstractState[V]":
        if self._is_bottom:
            return other
        if other._is_bottom:
            return self
        assert self.domain is not None
        all_vars = self.variables() | other.variables()
        new_env: Dict[str, V] = {}
        for var in all_vars:
            a = self.env.get(var)
            b = other.env.get(var)
            if a is None:
                new_env[var] = b if b is not None else self.domain.top()
            elif b is None:
                new_env[var] = a
            else:
                new_env[var] = self.domain.join(a, b)
        return AbstractState(env=new_env, domain=self.domain)

    def meet(self, other: "AbstractState[V]") -> "AbstractState[V]":
        if self._is_bottom or other._is_bottom:
            return AbstractState.bottom_state(self.domain or other.domain)
        assert self.domain is not None
        common = self.variables() & other.variables()
        new_env: Dict[str, V] = {}
        for var in common:
            new_env[var] = self.domain.meet(self.env[var], other.env[var])
        return AbstractState(env=new_env, domain=self.domain)

    def widen(self, other: "AbstractState[V]") -> "AbstractState[V]":
        if self._is_bottom:
            return other
        if other._is_bottom:
            return self
        assert self.domain is not None
        all_vars = self.variables() | other.variables()
        new_env: Dict[str, V] = {}
        for var in all_vars:
            a = self.env.get(var)
            b = other.env.get(var)
            if a is None:
                new_env[var] = b if b is not None else self.domain.top()
            elif b is None:
                new_env[var] = a
            else:
                new_env[var] = self.domain.widen(a, b)
        return AbstractState(env=new_env, domain=self.domain)

    def narrow(self, other: "AbstractState[V]") -> "AbstractState[V]":
        if self._is_bottom:
            return self
        if other._is_bottom:
            return other
        assert self.domain is not None
        all_vars = self.variables() & other.variables()
        new_env: Dict[str, V] = {}
        for var in all_vars:
            new_env[var] = self.domain.narrow(self.env[var], other.env[var])
        return AbstractState(env=new_env, domain=self.domain)

    def leq(self, other: "AbstractState[V]") -> bool:
        if self._is_bottom:
            return True
        if other._is_bottom:
            return False
        assert self.domain is not None
        for var in self.variables():
            a = self.env[var]
            b = other.env.get(var)
            if b is None:
                continue
            if not self.domain.leq(a, b):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractState):
            return NotImplemented
        if self._is_bottom and other._is_bottom:
            return True
        return self.env == other.env and self._is_bottom == other._is_bottom

    def __repr__(self) -> str:
        if self._is_bottom:
            return "AbstractState(⊥)"
        entries = ", ".join(f"{k}: {v}" for k, v in sorted(self.env.items()))
        return f"AbstractState({{{entries}}})"


# ===================================================================
# TransferFunction – abstract transfer for IR nodes
# ===================================================================

class IRNode:
    """Lightweight IR node stub for transfer functions."""

    def __init__(
        self,
        kind: str,
        *,
        target: Optional[str] = None,
        expr: Any = None,
        condition: Any = None,
        func: Optional[str] = None,
        args: Optional[List[Any]] = None,
        successors: Optional[List[int]] = None,
        label: Optional[int] = None,
    ):
        self.kind = kind
        self.target = target
        self.expr = expr
        self.condition = condition
        self.func = func
        self.args = args or []
        self.successors = successors or []
        self.label = label

    def __repr__(self) -> str:
        return f"IRNode({self.kind}, target={self.target}, label={self.label})"


class TransferFunction(ABC, Generic[V]):
    """Abstract transfer function mapping IR nodes to state transformations."""

    @abstractmethod
    def transfer(
        self, node: IRNode, state: AbstractState[V]
    ) -> AbstractState[V]:
        """Apply the transfer function for *node* to *state*."""
        ...

    def transfer_edge(
        self,
        src: IRNode,
        dst: IRNode,
        state: AbstractState[V],
        edge_label: Optional[Any] = None,
    ) -> AbstractState[V]:
        """Transfer along a CFG edge, possibly narrowing for branches."""
        return self.transfer(src, state)


class SimpleTransferFunction(TransferFunction[V]):
    """A transfer function that delegates to an AbstractTransformer."""

    def __init__(self, transformer: AbstractTransformer[V]):
        self.transformer = transformer

    def transfer(
        self, node: IRNode, state: AbstractState[V]
    ) -> AbstractState[V]:
        if node.kind == "assign" and node.target is not None:
            return self.transformer.assign(state, node.target, node.expr)
        elif node.kind == "guard":
            return self.transformer.guard(state, node.condition, True)
        elif node.kind == "call":
            return self.transformer.call(
                state, node.func or "", node.args, node.target
            )
        elif node.kind == "noop":
            return state
        return state


# ===================================================================
# WideningStrategy – configurable widening
# ===================================================================

class WideningStrategy(ABC, Generic[V]):
    """Configurable widening strategy."""

    @abstractmethod
    def should_widen(self, node_id: int, iteration: int) -> bool:
        """Return True if widening should be applied at this point."""
        ...

    @abstractmethod
    def apply(
        self, domain: AbstractDomain[V], old: V, new: V, iteration: int
    ) -> V:
        """Apply the widening, returning the widened value."""
        ...


class ThresholdWidening(WideningStrategy[V]):
    """Widening with user-supplied thresholds.

    Instead of jumping all the way to ±∞, the widened value steps through the
    threshold list in order.
    """

    def __init__(
        self,
        thresholds: List[Any],
        delay: int = 0,
        widen_func: Optional[Callable[[AbstractDomain[V], V, V, List[Any]], V]] = None,
    ):
        self.thresholds = sorted(set(thresholds))
        self.delay = delay
        self._widen_func = widen_func

    def should_widen(self, node_id: int, iteration: int) -> bool:
        return iteration >= self.delay

    def apply(
        self, domain: AbstractDomain[V], old: V, new: V, iteration: int
    ) -> V:
        if self._widen_func is not None:
            return self._widen_func(domain, old, new, self.thresholds)
        return domain.widen(old, new)


class DelayedWidening(WideningStrategy[V]):
    """Delay widening for the first *k* iterations, using join instead."""

    def __init__(self, delay: int = 3):
        self.delay = delay

    def should_widen(self, node_id: int, iteration: int) -> bool:
        return iteration >= self.delay

    def apply(
        self, domain: AbstractDomain[V], old: V, new: V, iteration: int
    ) -> V:
        if iteration < self.delay:
            return domain.join(old, new)
        return domain.widen(old, new)


class CombinedWidening(WideningStrategy[V]):
    """Combines delayed widening with threshold widening."""

    def __init__(self, delay: int = 3, thresholds: Optional[List[Any]] = None):
        self.delay = delay
        self.thresholds = sorted(set(thresholds)) if thresholds else []

    def should_widen(self, node_id: int, iteration: int) -> bool:
        return iteration >= self.delay

    def apply(
        self, domain: AbstractDomain[V], old: V, new: V, iteration: int
    ) -> V:
        if iteration < self.delay:
            return domain.join(old, new)
        return domain.widen(old, new)


# ===================================================================
# NarrowingIterator
# ===================================================================

class NarrowingIterator(Generic[V]):
    """Narrowing pass executed after widening reaches a fixed point."""

    def __init__(
        self,
        domain: AbstractDomain[V],
        max_iterations: int = 10,
    ):
        self.domain = domain
        self.max_iterations = max_iterations

    def narrow_state(
        self, old: AbstractState[V], new: AbstractState[V]
    ) -> AbstractState[V]:
        return old.narrow(new)

    def iterate(
        self,
        initial: Dict[int, AbstractState[V]],
        transfer: Callable[[int, AbstractState[V]], AbstractState[V]],
        cfg_succs: Dict[int, List[int]],
    ) -> Dict[int, AbstractState[V]]:
        """Run narrowing iterations until stable or max reached."""
        current = dict(initial)
        for _it in range(self.max_iterations):
            changed = False
            for node_id in sorted(current.keys()):
                state = current[node_id]
                transferred = transfer(node_id, state)
                for succ in cfg_succs.get(node_id, []):
                    old_succ = current.get(succ)
                    if old_succ is None:
                        continue
                    narrowed = self.narrow_state(old_succ, transferred)
                    if not narrowed.leq(old_succ) or not old_succ.leq(narrowed):
                        current[succ] = narrowed
                        changed = True
            if not changed:
                break
        return current


# ===================================================================
# FixedPointSolver – Kleene iteration
# ===================================================================

class FixedPointSolver(Generic[V]):
    """Kleene iteration with configurable widening/narrowing."""

    def __init__(
        self,
        domain: AbstractDomain[V],
        widening_strategy: Optional[WideningStrategy[V]] = None,
        max_iterations: int = 1000,
        narrowing_iterations: int = 5,
    ):
        self.domain = domain
        self.widening = widening_strategy or DelayedWidening(delay=3)
        self.max_iterations = max_iterations
        self.narrowing_iterations = narrowing_iterations

    def solve(
        self,
        entry: int,
        entry_state: AbstractState[V],
        cfg_succs: Dict[int, List[int]],
        cfg_preds: Dict[int, List[int]],
        transfer: Callable[[int, AbstractState[V]], AbstractState[V]],
        widen_points: Optional[Set[int]] = None,
    ) -> Dict[int, AbstractState[V]]:
        """
        Compute the least fixed point of the abstract transfer function
        via Kleene iteration with widening, followed by narrowing.
        """
        if widen_points is None:
            widen_points = self._compute_widen_points(cfg_succs, cfg_preds)

        states: Dict[int, AbstractState[V]] = {}
        bottom = AbstractState.bottom_state(self.domain)
        all_nodes = set(cfg_succs.keys()) | {
            s for succs in cfg_succs.values() for s in succs
        }
        for n in all_nodes:
            states[n] = bottom
        states[entry] = entry_state

        iteration_count: Dict[int, int] = defaultdict(int)
        worklist: List[int] = [entry]
        visited: Set[int] = set()

        for _global_iter in range(self.max_iterations):
            if not worklist:
                break
            node_id = worklist.pop(0)
            visited.add(node_id)

            old_state = states[node_id]
            new_state = transfer(node_id, old_state)

            for succ in cfg_succs.get(node_id, []):
                old_succ = states[succ]
                iteration_count[succ] += 1
                it = iteration_count[succ]

                if succ in widen_points and self.widening.should_widen(succ, it):
                    merged = AbstractState(
                        env={}, domain=self.domain
                    )
                    all_vars = old_succ.variables() | new_state.variables()
                    new_env: Dict[str, V] = {}
                    for var in all_vars:
                        a = old_succ.env.get(var)
                        b = new_state.env.get(var)
                        if a is None:
                            new_env[var] = b if b is not None else self.domain.top()
                        elif b is None:
                            new_env[var] = a
                        else:
                            new_env[var] = self.widening.apply(
                                self.domain, a, b, it
                            )
                    merged = AbstractState(env=new_env, domain=self.domain)
                else:
                    merged = old_succ.join(new_state)

                if not merged.leq(old_succ) or not old_succ.leq(merged):
                    states[succ] = merged
                    if succ not in worklist:
                        worklist.append(succ)

        # Narrowing phase
        if self.narrowing_iterations > 0:
            narrower = NarrowingIterator(
                self.domain, max_iterations=self.narrowing_iterations
            )
            states = narrower.iterate(states, transfer, cfg_succs)

        return states

    @staticmethod
    def _compute_widen_points(
        succs: Dict[int, List[int]], preds: Dict[int, List[int]]
    ) -> Set[int]:
        """Heuristic: widen at loop heads (nodes with a back-edge)."""
        visited: Set[int] = set()
        widen_pts: Set[int] = set()

        def dfs(n: int, path: Set[int]) -> None:
            if n in path:
                widen_pts.add(n)
                return
            if n in visited:
                return
            visited.add(n)
            path.add(n)
            for s in succs.get(n, []):
                dfs(s, path)
            path.discard(n)

        for start in succs:
            dfs(start, set())
        return widen_pts


# ===================================================================
# MonotoneFramework – general monotone dataflow framework
# ===================================================================

class MonotoneFramework(ABC, Generic[V]):
    """General monotone dataflow analysis framework.

    Subclasses provide the lattice, flow graph, transfer functions, and
    initial/boundary values. The framework computes the MFP (maximal fixed
    point) solution.
    """

    @abstractmethod
    def nodes(self) -> List[int]:
        ...

    @abstractmethod
    def entry_nodes(self) -> List[int]:
        ...

    @abstractmethod
    def successors(self, node: int) -> List[int]:
        ...

    @abstractmethod
    def predecessors(self, node: int) -> List[int]:
        ...

    @abstractmethod
    def initial_value(self) -> AbstractState[V]:
        ...

    @abstractmethod
    def boundary_value(self) -> AbstractState[V]:
        ...

    @abstractmethod
    def transfer(self, node: int, state: AbstractState[V]) -> AbstractState[V]:
        ...

    @abstractmethod
    def domain(self) -> AbstractDomain[V]:
        ...

    def is_forward(self) -> bool:
        return True


# ===================================================================
# WorklistSolver – efficient worklist-based fixed point
# ===================================================================

class WorklistSolver(Generic[V]):
    """Worklist-based fixed point solver with RPO ordering."""

    def __init__(self, framework: MonotoneFramework[V]):
        self.framework = framework

    def compute_rpo(self) -> List[int]:
        """Compute reverse post-order of the CFG."""
        all_nodes = self.framework.nodes()
        if not all_nodes:
            return []

        visited: Set[int] = set()
        post_order: List[int] = []

        def dfs(n: int) -> None:
            if n in visited:
                return
            visited.add(n)
            for s in self.framework.successors(n):
                dfs(s)
            post_order.append(n)

        for entry in self.framework.entry_nodes():
            dfs(entry)
        for n in all_nodes:
            if n not in visited:
                dfs(n)

        post_order.reverse()
        return post_order

    def solve(
        self,
        widening_strategy: Optional[WideningStrategy[V]] = None,
        max_iterations: int = 1000,
    ) -> Dict[int, AbstractState[V]]:
        """Solve the monotone framework using a worklist algorithm."""
        dom = self.framework.domain()
        rpo = self.compute_rpo()
        rpo_index = {n: i for i, n in enumerate(rpo)}

        bottom = AbstractState.bottom_state(dom)
        result: Dict[int, AbstractState[V]] = {}
        for n in rpo:
            result[n] = bottom

        for entry in self.framework.entry_nodes():
            result[entry] = self.framework.boundary_value()

        worklist: List[int] = list(rpo)
        iteration_count: Dict[int, int] = defaultdict(int)

        total_iterations = 0
        while worklist and total_iterations < max_iterations:
            total_iterations += 1
            worklist.sort(key=lambda n: rpo_index.get(n, len(rpo)))
            node = worklist.pop(0)

            if self.framework.is_forward():
                preds = self.framework.predecessors(node)
                if preds:
                    incoming = result.get(preds[0], bottom)
                    for p in preds[1:]:
                        incoming = incoming.join(result.get(p, bottom))
                else:
                    incoming = result[node]
            else:
                succs = self.framework.successors(node)
                if succs:
                    incoming = result.get(succs[0], bottom)
                    for s in succs[1:]:
                        incoming = incoming.join(result.get(s, bottom))
                else:
                    incoming = result[node]

            new_state = self.framework.transfer(node, incoming)
            iteration_count[node] += 1

            if widening_strategy is not None and widening_strategy.should_widen(
                node, iteration_count[node]
            ):
                old = result[node]
                widened_env: Dict[str, V] = {}
                all_vars = old.variables() | new_state.variables()
                for var in all_vars:
                    a = old.env.get(var)
                    b = new_state.env.get(var)
                    if a is None:
                        widened_env[var] = b if b is not None else dom.top()
                    elif b is None:
                        widened_env[var] = a
                    else:
                        widened_env[var] = widening_strategy.apply(
                            dom, a, b, iteration_count[node]
                        )
                new_state = AbstractState(env=widened_env, domain=dom)

            if not new_state.leq(result[node]) or not result[node].leq(new_state):
                result[node] = new_state
                if self.framework.is_forward():
                    for s in self.framework.successors(node):
                        if s not in worklist:
                            worklist.append(s)
                else:
                    for p in self.framework.predecessors(node):
                        if p not in worklist:
                            worklist.append(p)

        return result


# ===================================================================
# LiftedDomain – adds explicit ⊥ to a domain
# ===================================================================

class _LiftedBottom(AbstractValue):
    """Explicit bottom element for LiftedDomain."""

    _instance: Optional["_LiftedBottom"] = None

    def __new__(cls) -> "_LiftedBottom":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _LiftedBottom)

    def __hash__(self) -> int:
        return hash("_LiftedBottom")

    def __repr__(self) -> str:
        return "⊥"

    def is_bottom(self) -> bool:
        return True


LIFTED_BOTTOM = _LiftedBottom()


class LiftedDomain(AbstractDomain[V]):
    """Adds an explicit ⊥ element below an existing domain.

    Useful when the inner domain does not naturally have a bottom element.
    """

    def __init__(self, inner: AbstractDomain[V]):
        self.inner = inner

    def top(self) -> V:
        return self.inner.top()

    def bottom(self) -> V:
        return LIFTED_BOTTOM  # type: ignore[return-value]

    def join(self, a: V, b: V) -> V:
        if a is LIFTED_BOTTOM:
            return b
        if b is LIFTED_BOTTOM:
            return a
        return self.inner.join(a, b)

    def meet(self, a: V, b: V) -> V:
        if a is LIFTED_BOTTOM or b is LIFTED_BOTTOM:
            return LIFTED_BOTTOM  # type: ignore[return-value]
        return self.inner.meet(a, b)

    def leq(self, a: V, b: V) -> bool:
        if a is LIFTED_BOTTOM:
            return True
        if b is LIFTED_BOTTOM:
            return False
        return self.inner.leq(a, b)

    def widen(self, a: V, b: V) -> V:
        if a is LIFTED_BOTTOM:
            return b
        if b is LIFTED_BOTTOM:
            return a
        return self.inner.widen(a, b)

    def narrow(self, a: V, b: V) -> V:
        if a is LIFTED_BOTTOM:
            return LIFTED_BOTTOM  # type: ignore[return-value]
        if b is LIFTED_BOTTOM:
            return LIFTED_BOTTOM  # type: ignore[return-value]
        return self.inner.narrow(a, b)

    def is_bottom(self, a: V) -> bool:
        return a is LIFTED_BOTTOM or self.inner.is_bottom(a)

    def abstract(self, concrete: Any) -> V:
        return self.inner.abstract(concrete)

    def concretize(self, abstract_val: V) -> Any:
        if abstract_val is LIFTED_BOTTOM:
            return set()
        return self.inner.concretize(abstract_val)


# ===================================================================
# FlatDomain – flat lattice (⊥ < all elements < ⊤)
# ===================================================================

class FlatElement(AbstractValue, Generic[T]):
    """Element in a flat lattice."""

    class Kind(Enum):
        BOTTOM = auto()
        VALUE = auto()
        TOP = auto()

    def __init__(self, kind: "FlatElement.Kind", value: Optional[T] = None):
        self._kind = kind
        self._value = value

    @classmethod
    def make_bottom(cls) -> "FlatElement[T]":
        return cls(cls.Kind.BOTTOM)

    @classmethod
    def make_top(cls) -> "FlatElement[T]":
        return cls(cls.Kind.TOP)

    @classmethod
    def make_value(cls, v: T) -> "FlatElement[T]":
        return cls(cls.Kind.VALUE, v)

    @property
    def kind(self) -> "FlatElement.Kind":
        return self._kind

    @property
    def value(self) -> Optional[T]:
        return self._value

    def is_bottom(self) -> bool:
        return self._kind == self.Kind.BOTTOM

    def is_top(self) -> bool:
        return self._kind == self.Kind.TOP

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlatElement):
            return NotImplemented
        if self._kind != other._kind:
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        return hash((self._kind, self._value))

    def __repr__(self) -> str:
        if self._kind == self.Kind.BOTTOM:
            return "⊥"
        if self._kind == self.Kind.TOP:
            return "⊤"
        return f"Flat({self._value!r})"


class FlatDomain(AbstractDomain["FlatElement[T]"], Generic[T]):
    """Flat lattice: ⊥ < every element < ⊤."""

    def top(self) -> FlatElement[T]:
        return FlatElement.make_top()

    def bottom(self) -> FlatElement[T]:
        return FlatElement.make_bottom()

    def join(self, a: FlatElement[T], b: FlatElement[T]) -> FlatElement[T]:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        if a == b:
            return a
        return self.top()

    def meet(self, a: FlatElement[T], b: FlatElement[T]) -> FlatElement[T]:
        if a.is_top():
            return b
        if b.is_top():
            return a
        if a == b:
            return a
        return self.bottom()

    def leq(self, a: FlatElement[T], b: FlatElement[T]) -> bool:
        if a.is_bottom():
            return True
        if b.is_top():
            return True
        return a == b

    def abstract(self, concrete: Any) -> FlatElement[T]:
        if concrete is None:
            return self.bottom()
        return FlatElement.make_value(concrete)

    def concretize(self, abstract_val: FlatElement[T]) -> Any:
        if abstract_val.is_bottom():
            return set()
        if abstract_val.is_top():
            return None  # represents the full set
        return {abstract_val.value}


# ===================================================================
# PowersetDomain – powerset lattice with subset ordering
# ===================================================================

class PowersetValue(AbstractValue, Generic[T]):
    """A value in the powerset lattice."""

    def __init__(
        self,
        elements: Optional[FrozenSet[T]] = None,
        *,
        is_top: bool = False,
    ):
        self._elements = elements if elements is not None else frozenset()
        self._is_top = is_top

    @property
    def elements(self) -> FrozenSet[T]:
        return self._elements

    def is_top(self) -> bool:
        return self._is_top

    def is_bottom(self) -> bool:
        return not self._is_top and len(self._elements) == 0

    def __contains__(self, item: T) -> bool:
        if self._is_top:
            return True
        return item in self._elements

    def __iter__(self) -> Iterator[T]:
        return iter(self._elements)

    def __len__(self) -> int:
        if self._is_top:
            raise ValueError("Cannot take len of ⊤ powerset")
        return len(self._elements)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PowersetValue):
            return NotImplemented
        if self._is_top and other._is_top:
            return True
        if self._is_top or other._is_top:
            return False
        return self._elements == other._elements

    def __hash__(self) -> int:
        return hash((self._is_top, self._elements))

    def __repr__(self) -> str:
        if self._is_top:
            return "Powerset(⊤)"
        if not self._elements:
            return "Powerset(⊥)"
        elems = ", ".join(repr(e) for e in sorted(self._elements, key=repr))
        return f"Powerset({{{elems}}})"


class PowersetDomain(AbstractDomain["PowersetValue[T]"], Generic[T]):
    """Powerset lattice ⟨𝒫(U), ⊆⟩."""

    def __init__(self, universe: Optional[FrozenSet[T]] = None):
        self.universe = universe

    def top(self) -> PowersetValue[T]:
        if self.universe is not None:
            return PowersetValue(self.universe)
        return PowersetValue(is_top=True)

    def bottom(self) -> PowersetValue[T]:
        return PowersetValue(frozenset())

    def join(self, a: PowersetValue[T], b: PowersetValue[T]) -> PowersetValue[T]:
        if a.is_top() or b.is_top():
            return self.top()
        return PowersetValue(a.elements | b.elements)

    def meet(self, a: PowersetValue[T], b: PowersetValue[T]) -> PowersetValue[T]:
        if a.is_top():
            return b
        if b.is_top():
            return a
        return PowersetValue(a.elements & b.elements)

    def leq(self, a: PowersetValue[T], b: PowersetValue[T]) -> bool:
        if b.is_top():
            return True
        if a.is_top():
            return False
        return a.elements <= b.elements

    def abstract(self, concrete: Any) -> PowersetValue[T]:
        if isinstance(concrete, (set, frozenset)):
            return PowersetValue(frozenset(concrete))
        return PowersetValue(frozenset({concrete}))

    def concretize(self, abstract_val: PowersetValue[T]) -> Any:
        if abstract_val.is_top():
            if self.universe is not None:
                return set(self.universe)
            return None
        return set(abstract_val.elements)


# ===================================================================
# MapDomain – domain of maps from keys to abstract values
# ===================================================================

@dataclass
class MapValue(AbstractValue, Generic[K, V]):
    """A map from keys to abstract values."""

    mapping: Dict[K, V] = field(default_factory=dict)
    default: Optional[V] = None
    _is_top: bool = False
    _is_bottom: bool = False

    def is_top(self) -> bool:
        return self._is_top

    def is_bottom(self) -> bool:
        return self._is_bottom

    def get(self, key: K) -> Optional[V]:
        return self.mapping.get(key, self.default)

    def set(self, key: K, value: V) -> "MapValue[K, V]":
        new_map = dict(self.mapping)
        new_map[key] = value
        return MapValue(mapping=new_map, default=self.default)

    def keys(self) -> Set[K]:
        return set(self.mapping.keys())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MapValue):
            return NotImplemented
        if self._is_top and other._is_top:
            return True
        if self._is_bottom and other._is_bottom:
            return True
        return (
            self.mapping == other.mapping
            and self.default == other.default
            and self._is_top == other._is_top
            and self._is_bottom == other._is_bottom
        )

    def __hash__(self) -> int:
        items = tuple(sorted(self.mapping.items(), key=lambda kv: repr(kv[0])))
        return hash((items, self.default, self._is_top, self._is_bottom))

    def __repr__(self) -> str:
        if self._is_top:
            return "MapValue(⊤)"
        if self._is_bottom:
            return "MapValue(⊥)"
        entries = ", ".join(
            f"{k!r}: {v!r}" for k, v in sorted(self.mapping.items(), key=lambda kv: repr(kv[0]))
        )
        return f"MapValue({{{entries}}})"


class MapDomain(AbstractDomain["MapValue[K, V]"], Generic[K, V]):
    """Domain of maps from keys to abstract values."""

    def __init__(self, value_domain: AbstractDomain[V]):
        self.value_domain = value_domain

    def top(self) -> MapValue[K, V]:
        return MapValue(_is_top=True, default=self.value_domain.top())

    def bottom(self) -> MapValue[K, V]:
        return MapValue(_is_bottom=True)

    def join(self, a: MapValue[K, V], b: MapValue[K, V]) -> MapValue[K, V]:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        if a.is_top() or b.is_top():
            return self.top()

        all_keys = a.keys() | b.keys()
        new_map: Dict[K, V] = {}
        for k in all_keys:
            va = a.mapping.get(k, a.default)
            vb = b.mapping.get(k, b.default)
            if va is not None and vb is not None:
                new_map[k] = self.value_domain.join(va, vb)
            elif va is not None:
                new_map[k] = va
            elif vb is not None:
                new_map[k] = vb
        return MapValue(mapping=new_map)

    def meet(self, a: MapValue[K, V], b: MapValue[K, V]) -> MapValue[K, V]:
        if a.is_top():
            return b
        if b.is_top():
            return a
        if a.is_bottom() or b.is_bottom():
            return self.bottom()

        common_keys = a.keys() & b.keys()
        new_map: Dict[K, V] = {}
        for k in common_keys:
            va = a.mapping[k]
            vb = b.mapping[k]
            new_map[k] = self.value_domain.meet(va, vb)
        return MapValue(mapping=new_map)

    def leq(self, a: MapValue[K, V], b: MapValue[K, V]) -> bool:
        if a.is_bottom():
            return True
        if b.is_top():
            return True
        if b.is_bottom() or a.is_top():
            return False

        for k in a.keys():
            va = a.mapping[k]
            vb = b.mapping.get(k, b.default)
            if vb is None:
                return False
            if not self.value_domain.leq(va, vb):
                return False
        return True

    def widen(self, a: MapValue[K, V], b: MapValue[K, V]) -> MapValue[K, V]:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        if a.is_top() or b.is_top():
            return self.top()

        all_keys = a.keys() | b.keys()
        new_map: Dict[K, V] = {}
        for k in all_keys:
            va = a.mapping.get(k)
            vb = b.mapping.get(k)
            if va is not None and vb is not None:
                new_map[k] = self.value_domain.widen(va, vb)
            elif va is not None:
                new_map[k] = va
            elif vb is not None:
                new_map[k] = vb
        return MapValue(mapping=new_map)

    def narrow(self, a: MapValue[K, V], b: MapValue[K, V]) -> MapValue[K, V]:
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        if a.is_top():
            return b
        if b.is_top():
            return a

        common_keys = a.keys() & b.keys()
        new_map: Dict[K, V] = {}
        for k in common_keys:
            new_map[k] = self.value_domain.narrow(a.mapping[k], b.mapping[k])
        return MapValue(mapping=new_map)

    def abstract(self, concrete: Any) -> MapValue[K, V]:
        if isinstance(concrete, dict):
            m = {k: self.value_domain.abstract(v) for k, v in concrete.items()}
            return MapValue(mapping=m)
        return self.top()

    def concretize(self, abstract_val: MapValue[K, V]) -> Any:
        if abstract_val.is_bottom():
            return {}
        if abstract_val.is_top():
            return None
        return {
            k: self.value_domain.concretize(v)
            for k, v in abstract_val.mapping.items()
        }
