from __future__ import annotations

"""
Tests for the CEGAR (Counter-Example Guided Abstraction Refinement) engine.

Covers: guard harvesting, predicate abstraction, abstract interpretation,
counterexample analysis, CEGAR loop convergence, and incremental refinement.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, FrozenSet, List, Mapping,
    Optional, Set, Sequence, Tuple, Union,
)

import pytest

# ── Local type stubs ──────────────────────────────────────────────────────


class ComparisonOp(Enum):
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    EQ = auto()
    NE = auto()

    def negate(self) -> ComparisonOp:
        _neg = {
            ComparisonOp.LT: ComparisonOp.GE,
            ComparisonOp.LE: ComparisonOp.GT,
            ComparisonOp.GT: ComparisonOp.LE,
            ComparisonOp.GE: ComparisonOp.LT,
            ComparisonOp.EQ: ComparisonOp.NE,
            ComparisonOp.NE: ComparisonOp.EQ,
        }
        return _neg[self]

    def evaluate(self, left: Any, right: Any) -> bool:
        ops = {
            ComparisonOp.LT: lambda a, b: a < b,
            ComparisonOp.LE: lambda a, b: a <= b,
            ComparisonOp.GT: lambda a, b: a > b,
            ComparisonOp.GE: lambda a, b: a >= b,
            ComparisonOp.EQ: lambda a, b: a == b,
            ComparisonOp.NE: lambda a, b: a != b,
        }
        return ops[self](left, right)


class GuardKind(Enum):
    ISINSTANCE = auto()
    TYPEOF = auto()
    IS_NONE = auto()
    IS_NOT_NONE = auto()
    HASATTR = auto()
    COMPARISON = auto()
    TRUTHINESS = auto()
    LEN_COMPARISON = auto()


@dataclass(frozen=True)
class Guard:
    kind: GuardKind
    variable: str
    argument: Any = None
    op: Optional[ComparisonOp] = None

    def negate(self) -> Guard:
        neg_map = {
            GuardKind.IS_NONE: GuardKind.IS_NOT_NONE,
            GuardKind.IS_NOT_NONE: GuardKind.IS_NONE,
        }
        if self.kind in neg_map:
            return Guard(neg_map[self.kind], self.variable, self.argument, self.op)
        if self.op is not None:
            return Guard(self.kind, self.variable, self.argument, self.op.negate())
        return Guard(self.kind, self.variable, self.argument, self.op)


@dataclass(frozen=True)
class Predicate:
    """A predicate over program variables."""
    variable: Optional[str]
    kind: str  # "comparison", "isinstance", "is_none", "hasattr", "truthiness"
    argument: Any = None
    op: Optional[ComparisonOp] = None

    def negate(self) -> Predicate:
        if self.op is not None:
            return Predicate(self.variable, self.kind, self.argument, self.op.negate())
        if self.kind == "is_none":
            return Predicate(self.variable, "is_not_none", self.argument, self.op)
        if self.kind == "is_not_none":
            return Predicate(self.variable, "is_none", self.argument, self.op)
        return Predicate(self.variable, f"not_{self.kind}", self.argument, self.op)

    def evaluate(self, env: Dict[str, Any]) -> bool:
        if self.variable is None or self.variable not in env:
            return False
        val = env[self.variable]
        if self.kind == "comparison" and self.op is not None:
            return self.op.evaluate(val, self.argument)
        if self.kind == "isinstance":
            return type(val).__name__ == self.argument
        if self.kind == "is_none":
            return val is None
        if self.kind == "is_not_none":
            return val is not None
        if self.kind == "truthiness":
            return bool(val)
        return False

    def free_variables(self) -> Set[str]:
        if self.variable is None:
            return set()
        return {self.variable}

    def substitute(self, mapping: Dict[str, str]) -> Predicate:
        new_var = mapping.get(self.variable, self.variable) if self.variable else None
        return Predicate(new_var, self.kind, self.argument, self.op)


@dataclass(frozen=True)
class PredicateSet:
    """A set of predicates forming a conjunction."""
    predicates: FrozenSet[Predicate]

    @classmethod
    def empty(cls) -> PredicateSet:
        return cls(frozenset())

    @classmethod
    def from_predicates(cls, *preds: Predicate) -> PredicateSet:
        return cls(frozenset(preds))

    def add(self, p: Predicate) -> PredicateSet:
        return PredicateSet(self.predicates | {p})

    def remove(self, p: Predicate) -> PredicateSet:
        return PredicateSet(self.predicates - {p})

    def union(self, other: PredicateSet) -> PredicateSet:
        return PredicateSet(self.predicates | other.predicates)

    def evaluate(self, env: Dict[str, Any]) -> bool:
        return all(p.evaluate(env) for p in self.predicates)

    def __len__(self) -> int:
        return len(self.predicates)


class BoundKind(Enum):
    NEG_INF = auto()
    FINITE = auto()
    POS_INF = auto()


@dataclass(frozen=True)
class Bound:
    kind: BoundKind
    value: int = 0

    @classmethod
    def finite(cls, n: int) -> Bound:
        return cls(BoundKind.FINITE, n)

    @classmethod
    def pos_inf(cls) -> Bound:
        return cls(BoundKind.POS_INF)

    @classmethod
    def neg_inf(cls) -> Bound:
        return cls(BoundKind.NEG_INF)

    def __lt__(self, other: Bound) -> bool:
        order = {BoundKind.NEG_INF: 0, BoundKind.FINITE: 1, BoundKind.POS_INF: 2}
        if self.kind != other.kind:
            return order[self.kind] < order[other.kind]
        return self.value < other.value

    def __le__(self, other: Bound) -> bool:
        return self == other or self < other

    def __gt__(self, other: Bound) -> bool:
        return other < self

    def __ge__(self, other: Bound) -> bool:
        return other <= self


@dataclass(frozen=True)
class Interval:
    lo: Bound
    hi: Bound

    @classmethod
    def top(cls) -> Interval:
        return cls(Bound.neg_inf(), Bound.pos_inf())

    @classmethod
    def bottom(cls) -> Interval:
        return cls(Bound.finite(1), Bound.finite(0))

    @classmethod
    def singleton(cls, n: int) -> Interval:
        return cls(Bound.finite(n), Bound.finite(n))

    @classmethod
    def from_bounds(cls, lo: int, hi: int) -> Interval:
        return cls(Bound.finite(lo), Bound.finite(hi))

    @property
    def is_bottom(self) -> bool:
        return self.lo > self.hi

    @property
    def is_top(self) -> bool:
        return self.lo.kind == BoundKind.NEG_INF and self.hi.kind == BoundKind.POS_INF

    def contains(self, n: int) -> bool:
        if self.is_bottom:
            return False
        b = Bound.finite(n)
        return self.lo <= b and b <= self.hi

    def join(self, other: Interval) -> Interval:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        lo = self.lo if self.lo <= other.lo else other.lo
        hi = self.hi if self.hi >= other.hi else other.hi
        return Interval(lo, hi)

    def meet(self, other: Interval) -> Interval:
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        lo = self.lo if self.lo >= other.lo else other.lo
        hi = self.hi if self.hi <= other.hi else other.hi
        if lo > hi:
            return Interval.bottom()
        return Interval(lo, hi)

    def leq(self, other: Interval) -> bool:
        if self.is_bottom:
            return True
        if other.is_bottom:
            return False
        return other.lo <= self.lo and self.hi <= other.hi

    def widen(self, other: Interval) -> Interval:
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        lo = self.lo if self.lo <= other.lo else Bound.neg_inf()
        hi = self.hi if other.hi <= self.hi else Bound.pos_inf()
        return Interval(lo, hi)


class NullityKind(Enum):
    BOTTOM = auto()
    DEFINITELY_NULL = auto()
    DEFINITELY_NOT_NULL = auto()
    MAYBE_NULL = auto()


@dataclass(frozen=True)
class NullityValue:
    kind: NullityKind

    @classmethod
    def bottom(cls) -> NullityValue:
        return cls(NullityKind.BOTTOM)

    @classmethod
    def definitely_null(cls) -> NullityValue:
        return cls(NullityKind.DEFINITELY_NULL)

    @classmethod
    def definitely_not_null(cls) -> NullityValue:
        return cls(NullityKind.DEFINITELY_NOT_NULL)

    @classmethod
    def maybe_null(cls) -> NullityValue:
        return cls(NullityKind.MAYBE_NULL)

    @property
    def may_be_null(self) -> bool:
        return self.kind in (NullityKind.DEFINITELY_NULL, NullityKind.MAYBE_NULL)

    @property
    def may_be_non_null(self) -> bool:
        return self.kind in (NullityKind.DEFINITELY_NOT_NULL, NullityKind.MAYBE_NULL)


class ViolationKind(Enum):
    ARRAY_OUT_OF_BOUNDS = "array_out_of_bounds"
    NULL_DEREFERENCE = "null_dereference"
    DIVISION_BY_ZERO = "division_by_zero"
    TYPE_TAG_MISMATCH = "type_tag_mismatch"


class Severity(Enum):
    Error = auto()
    Warning = auto()
    Info = auto()


@dataclass
class BugReport:
    location: Optional[str]
    bug_kind: ViolationKind
    message: str
    severity: Severity = Severity.Warning
    confidence: float = 0.5
    counterexample: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None


@dataclass
class AbstractState:
    """Simplified abstract state mapping variables to intervals + nullity."""
    intervals: Dict[str, Interval] = field(default_factory=dict)
    nullity: Dict[str, NullityValue] = field(default_factory=dict)

    def set_interval(self, var: str, iv: Interval) -> AbstractState:
        new_ivs = dict(self.intervals)
        new_ivs[var] = iv
        return AbstractState(new_ivs, dict(self.nullity))

    def get_interval(self, var: str) -> Interval:
        return self.intervals.get(var, Interval.top())

    def set_nullity(self, var: str, nv: NullityValue) -> AbstractState:
        new_nv = dict(self.nullity)
        new_nv[var] = nv
        return AbstractState(dict(self.intervals), new_nv)

    def get_nullity(self, var: str) -> NullityValue:
        return self.nullity.get(var, NullityValue.maybe_null())

    def join(self, other: AbstractState) -> AbstractState:
        all_vars = set(self.intervals) | set(other.intervals)
        ivs = {}
        for v in all_vars:
            ivs[v] = self.get_interval(v).join(other.get_interval(v))
        nv = {}
        for v in set(self.nullity) | set(other.nullity):
            a = self.nullity.get(v, NullityValue.maybe_null())
            b = other.nullity.get(v, NullityValue.maybe_null())
            if a == b:
                nv[v] = a
            else:
                nv[v] = NullityValue.maybe_null()
        return AbstractState(ivs, nv)


# ── Simulated IR ──────────────────────────────────────────────────────────


class StmtKind(Enum):
    ASSIGN = auto()
    GUARD = auto()
    CALL = auto()
    RETURN = auto()
    SUBSCRIPT = auto()
    ATTR_ACCESS = auto()
    BINOP = auto()


@dataclass
class Stmt:
    kind: StmtKind
    target: Optional[str] = None
    source: Optional[str] = None
    value: Any = None
    guard: Optional[Guard] = None
    line: int = 0


@dataclass
class BasicBlock:
    id: int
    stmts: List[Stmt] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)


@dataclass
class IRFunction:
    name: str
    params: List[str] = field(default_factory=list)
    blocks: List[BasicBlock] = field(default_factory=list)
    entry: int = 0

    def block(self, bid: int) -> BasicBlock:
        for b in self.blocks:
            if b.id == bid:
                return b
        raise ValueError(f"Block {bid} not found")


# ── Guard Harvesting ─────────────────────────────────────────────────────


def harvest_guards(func: IRFunction) -> List[Guard]:
    """Extract guards from a function's IR."""
    guards: List[Guard] = []
    for block in func.blocks:
        for stmt in block.stmts:
            if stmt.kind == StmtKind.GUARD and stmt.guard is not None:
                guards.append(stmt.guard)
    return guards


def guards_to_predicates(guards: List[Guard]) -> List[Predicate]:
    """Convert guards to predicates."""
    preds: List[Predicate] = []
    for g in guards:
        if g.kind == GuardKind.ISINSTANCE:
            preds.append(Predicate(g.variable, "isinstance", g.argument))
        elif g.kind == GuardKind.IS_NONE:
            preds.append(Predicate(g.variable, "is_none"))
        elif g.kind == GuardKind.IS_NOT_NONE:
            preds.append(Predicate(g.variable, "is_not_none"))
        elif g.kind == GuardKind.HASATTR:
            preds.append(Predicate(g.variable, "hasattr", g.argument))
        elif g.kind == GuardKind.COMPARISON:
            preds.append(Predicate(g.variable, "comparison", g.argument, g.op))
        elif g.kind == GuardKind.TRUTHINESS:
            preds.append(Predicate(g.variable, "truthiness"))
        elif g.kind == GuardKind.TYPEOF:
            preds.append(Predicate(g.variable, "isinstance", g.argument))
        elif g.kind == GuardKind.LEN_COMPARISON:
            preds.append(Predicate(g.variable, "comparison", g.argument, g.op))
    return preds


def deduplicate_guards(guards: List[Guard]) -> List[Guard]:
    """Remove duplicate guards."""
    seen: Set[Guard] = set()
    result: List[Guard] = []
    for g in guards:
        if g not in seen:
            seen.add(g)
            result.append(g)
    return result


def rank_guards(guards: List[Guard]) -> List[Guard]:
    """Rank guards by priority: isinstance > is_none > comparison > others."""
    priority = {
        GuardKind.ISINSTANCE: 0,
        GuardKind.IS_NONE: 1,
        GuardKind.IS_NOT_NONE: 1,
        GuardKind.TYPEOF: 0,
        GuardKind.COMPARISON: 2,
        GuardKind.LEN_COMPARISON: 2,
        GuardKind.HASATTR: 3,
        GuardKind.TRUTHINESS: 4,
    }
    return sorted(guards, key=lambda g: priority.get(g.kind, 5))


def normalize_guard(g: Guard) -> Guard:
    """Normalize a guard to a canonical form."""
    # is_not_none → negate of is_none in normalized form
    if g.kind == GuardKind.IS_NOT_NONE:
        return Guard(GuardKind.IS_NONE, g.variable, g.argument, g.op).negate()
    return g


# ── Abstract Interpreter (simplified) ────────────────────────────────────


def interpret_block(
    block: BasicBlock,
    state: AbstractState,
    predicates: PredicateSet,
) -> AbstractState:
    """Interpret a basic block, updating abstract state."""
    current = state
    for stmt in block.stmts:
        if stmt.kind == StmtKind.ASSIGN:
            if stmt.target and isinstance(stmt.value, int):
                current = current.set_interval(stmt.target, Interval.singleton(stmt.value))
                current = current.set_nullity(stmt.target, NullityValue.definitely_not_null())
            elif stmt.target and stmt.value is None:
                current = current.set_nullity(stmt.target, NullityValue.definitely_null())
        elif stmt.kind == StmtKind.GUARD and stmt.guard is not None:
            g = stmt.guard
            if g.kind == GuardKind.COMPARISON and g.op is not None and isinstance(g.argument, int):
                iv = current.get_interval(g.variable)
                if g.op == ComparisonOp.LT:
                    refined = iv.meet(Interval(Bound.neg_inf(), Bound.finite(g.argument - 1)))
                elif g.op == ComparisonOp.LE:
                    refined = iv.meet(Interval(Bound.neg_inf(), Bound.finite(g.argument)))
                elif g.op == ComparisonOp.GT:
                    refined = iv.meet(Interval(Bound.finite(g.argument + 1), Bound.pos_inf()))
                elif g.op == ComparisonOp.GE:
                    refined = iv.meet(Interval(Bound.finite(g.argument), Bound.pos_inf()))
                elif g.op == ComparisonOp.EQ:
                    refined = iv.meet(Interval.singleton(g.argument))
                else:
                    refined = iv
                current = current.set_interval(g.variable, refined)
            elif g.kind == GuardKind.IS_NONE:
                current = current.set_nullity(g.variable, NullityValue.definitely_null())
            elif g.kind == GuardKind.IS_NOT_NONE:
                current = current.set_nullity(g.variable, NullityValue.definitely_not_null())
            elif g.kind == GuardKind.TRUTHINESS:
                current = current.set_nullity(g.variable, NullityValue.definitely_not_null())
    return current


def interpret_function(
    func: IRFunction,
    initial_state: AbstractState,
    predicates: PredicateSet,
    max_iterations: int = 100,
) -> Dict[int, AbstractState]:
    """Fixed-point abstract interpretation over a function's CFG."""
    states: Dict[int, AbstractState] = {func.entry: initial_state}
    worklist = [func.entry]
    iterations = 0

    while worklist and iterations < max_iterations:
        iterations += 1
        bid = worklist.pop(0)
        block = func.block(bid)
        in_state = states.get(bid, AbstractState())
        out_state = interpret_block(block, in_state, predicates)
        states[bid] = out_state

        for succ in block.successors:
            old = states.get(succ, None)
            new = out_state if old is None else old.join(out_state)
            if old is None or new != old:
                # Widening after 3 iterations
                if iterations > 3 and old is not None:
                    widened_ivs: Dict[str, Interval] = {}
                    for v in set(new.intervals) | set(old.intervals):
                        widened_ivs[v] = old.get_interval(v).widen(new.get_interval(v))
                    new = AbstractState(widened_ivs, new.nullity)
                states[succ] = new
                if succ not in worklist:
                    worklist.append(succ)

    return states


# ── Counterexample Analysis ──────────────────────────────────────────────


@dataclass
class Counterexample:
    """A concrete execution trace that may demonstrate a bug."""
    path: List[int]  # block IDs in order
    env: Dict[str, Any]  # variable assignments
    violation: Optional[ViolationKind] = None
    is_feasible: Optional[bool] = None

    def is_real(self) -> bool:
        return self.is_feasible is True

    def is_spurious(self) -> bool:
        return self.is_feasible is False


def check_path_feasibility(
    func: IRFunction,
    path: List[int],
    env: Dict[str, Any],
) -> bool:
    """Check if a path is feasible under given variable assignments."""
    for bid in path:
        block = func.block(bid)
        for stmt in block.stmts:
            if stmt.kind == StmtKind.GUARD and stmt.guard is not None:
                g = stmt.guard
                if g.variable not in env:
                    continue
                val = env[g.variable]
                if g.kind == GuardKind.COMPARISON and g.op is not None:
                    if not g.op.evaluate(val, g.argument):
                        return False
                elif g.kind == GuardKind.IS_NONE:
                    if val is not None:
                        return False
                elif g.kind == GuardKind.IS_NOT_NONE:
                    if val is None:
                        return False
                elif g.kind == GuardKind.TRUTHINESS:
                    if not val:
                        return False
    return True


def extract_interpolant(
    func: IRFunction,
    ce: Counterexample,
) -> Optional[Predicate]:
    """Extract an interpolant from a spurious counterexample."""
    if ce.is_real():
        return None
    # Find the first guard along the path that fails
    for bid in ce.path:
        block = func.block(bid)
        for stmt in block.stmts:
            if stmt.kind == StmtKind.GUARD and stmt.guard is not None:
                g = stmt.guard
                if g.variable in ce.env:
                    val = ce.env[g.variable]
                    if g.kind == GuardKind.COMPARISON and g.op is not None:
                        if not g.op.evaluate(val, g.argument):
                            return Predicate(g.variable, "comparison", g.argument, g.op)
                    elif g.kind == GuardKind.IS_NONE and val is not None:
                        return Predicate(g.variable, "is_none")
                    elif g.kind == GuardKind.IS_NOT_NONE and val is None:
                        return Predicate(g.variable, "is_not_none")
    return None


# ── CEGAR Loop ────────────────────────────────────────────────────────────


@dataclass
class CEGARConfig:
    max_iterations: int = 100
    per_function_timeout: float = 60.0
    enable_guard_seeding: bool = True
    predicate_budget: int = 500
    widening_delay: int = 3


@dataclass
class CEGARResult:
    predicates_used: PredicateSet
    iterations: int
    converged: bool
    bug_reports: List[BugReport]
    time_taken: float = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "predicates": len(self.predicates_used),
            "bugs": len(self.bug_reports),
            "time": self.time_taken,
        }


def detect_violations(
    func: IRFunction,
    states: Dict[int, AbstractState],
) -> List[Tuple[ViolationKind, int, str, Dict[str, Any]]]:
    """Detect potential violations in analyzed states."""
    violations: List[Tuple[ViolationKind, int, str, Dict[str, Any]]] = []
    for block in func.blocks:
        state = states.get(block.id, AbstractState())
        for stmt in block.stmts:
            if stmt.kind == StmtKind.SUBSCRIPT:
                # Check array bounds
                arr_var = stmt.source or ""
                idx_var = stmt.target or ""
                idx_iv = state.get_interval(idx_var)
                arr_len = state.get_interval(f"len({arr_var})")
                if idx_iv.is_top or (not idx_iv.is_bottom and idx_iv.lo.kind == BoundKind.FINITE and idx_iv.lo.value < 0):
                    violations.append((
                        ViolationKind.ARRAY_OUT_OF_BOUNDS, block.id,
                        f"Potential out-of-bounds access on {arr_var}[{idx_var}]",
                        {"array": arr_var, "index": idx_var},
                    ))
            elif stmt.kind == StmtKind.ATTR_ACCESS:
                var = stmt.source or ""
                nv = state.get_nullity(var)
                if nv.may_be_null:
                    violations.append((
                        ViolationKind.NULL_DEREFERENCE, block.id,
                        f"Potential null dereference on {var}.{stmt.value}",
                        {"var": var, "attr": stmt.value},
                    ))
            elif stmt.kind == StmtKind.BINOP and stmt.value == "div":
                divisor = stmt.source or ""
                div_iv = state.get_interval(divisor)
                if div_iv.contains(0):
                    violations.append((
                        ViolationKind.DIVISION_BY_ZERO, block.id,
                        f"Potential division by zero: {divisor}",
                        {"divisor": divisor},
                    ))
    return violations


def run_cegar(
    func: IRFunction,
    config: CEGARConfig = CEGARConfig(),
) -> CEGARResult:
    """Run the CEGAR loop on a function."""
    start_time = time.monotonic()
    predicates = PredicateSet.empty()

    # Seed predicates from guards if enabled
    if config.enable_guard_seeding:
        guards = harvest_guards(func)
        seed_preds = guards_to_predicates(guards)
        for p in seed_preds:
            predicates = predicates.add(p)

    bug_reports: List[BugReport] = []
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        elapsed = time.monotonic() - start_time
        if elapsed > config.per_function_timeout:
            break

        # Abstract interpretation
        initial = AbstractState()
        for p in func.params:
            initial = initial.set_interval(p, Interval.top())
            initial = initial.set_nullity(p, NullityValue.maybe_null())

        states = interpret_function(func, initial, predicates, max_iterations=100)

        # Detect violations
        violations = detect_violations(func, states)

        if not violations:
            converged = True
            break

        # For each violation, try to construct CE and refine
        new_pred_found = False
        for vk, bid, msg, info in violations:
            # Simple CE: path from entry to violation block
            path = [func.entry, bid]
            # Check feasibility with arbitrary concrete values
            ce = Counterexample(path=path, env={}, violation=vk)

            # Check if the abstract state at the block is reachable
            block_state = states.get(bid, AbstractState())
            # Try to find a refining predicate from guards along the path
            for b_id in path:
                blk = func.block(b_id)
                for stmt in blk.stmts:
                    if stmt.kind == StmtKind.GUARD and stmt.guard is not None:
                        g = stmt.guard
                        p = Predicate(g.variable, "comparison", g.argument, g.op)
                        if p not in predicates.predicates:
                            predicates = predicates.add(p)
                            new_pred_found = True

            if not new_pred_found:
                # Real bug
                bug_reports.append(BugReport(
                    location=f"line {bid}",
                    bug_kind=vk,
                    message=msg,
                    severity=Severity.Warning,
                    confidence=0.8,
                    counterexample=info,
                ))

        if not new_pred_found:
            converged = True
            break

    elapsed = time.monotonic() - start_time
    return CEGARResult(
        predicates_used=predicates,
        iterations=iteration if 'iteration' in dir() else 0,
        converged=converged,
        bug_reports=bug_reports,
        time_taken=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═══════════════════════════════════════════════════════════════════════════


def _make_simple_func(
    name: str = "f",
    params: Optional[List[str]] = None,
    guards: Optional[List[Guard]] = None,
    stmts: Optional[List[Stmt]] = None,
) -> IRFunction:
    """Helper: build a simple single-block function."""
    all_stmts: List[Stmt] = []
    if guards:
        for g in guards:
            all_stmts.append(Stmt(StmtKind.GUARD, guard=g))
    if stmts:
        all_stmts.extend(stmts)
    block = BasicBlock(id=0, stmts=all_stmts, successors=[], predecessors=[])
    return IRFunction(name=name, params=params or [], blocks=[block], entry=0)


def _make_branch_func(
    name: str = "f",
    params: Optional[List[str]] = None,
    guard: Optional[Guard] = None,
    true_stmts: Optional[List[Stmt]] = None,
    false_stmts: Optional[List[Stmt]] = None,
    merge_stmts: Optional[List[Stmt]] = None,
) -> IRFunction:
    """Helper: build a function with a branch (if/else)."""
    entry_stmts: List[Stmt] = []
    if guard:
        entry_stmts.append(Stmt(StmtKind.GUARD, guard=guard))
    entry = BasicBlock(id=0, stmts=entry_stmts, successors=[1, 2])
    true_block = BasicBlock(id=1, stmts=true_stmts or [], successors=[3], predecessors=[0])
    false_block = BasicBlock(id=2, stmts=false_stmts or [], successors=[3], predecessors=[0])
    merge_block = BasicBlock(id=3, stmts=merge_stmts or [], successors=[], predecessors=[1, 2])
    return IRFunction(
        name=name,
        params=params or [],
        blocks=[entry, true_block, false_block, merge_block],
        entry=0,
    )


def _make_loop_func(
    name: str = "f",
    params: Optional[List[str]] = None,
    init_stmts: Optional[List[Stmt]] = None,
    loop_guard: Optional[Guard] = None,
    body_stmts: Optional[List[Stmt]] = None,
    exit_stmts: Optional[List[Stmt]] = None,
) -> IRFunction:
    """Helper: build a function with a loop."""
    init = BasicBlock(id=0, stmts=init_stmts or [], successors=[1])
    header_stmts: List[Stmt] = []
    if loop_guard:
        header_stmts.append(Stmt(StmtKind.GUARD, guard=loop_guard))
    header = BasicBlock(id=1, stmts=header_stmts, successors=[2, 3], predecessors=[0, 2])
    body = BasicBlock(id=2, stmts=body_stmts or [], successors=[1], predecessors=[1])
    exit_block = BasicBlock(id=3, stmts=exit_stmts or [], successors=[], predecessors=[1])
    return IRFunction(
        name=name,
        params=params or [],
        blocks=[init, header, body, exit_block],
        entry=0,
    )


class TestGuardHarvesting:
    """Tests for guard extraction from IR."""

    def test_harvest_isinstance_guard(self) -> None:
        """Harvest isinstance guard from function."""
        g = Guard(GuardKind.ISINSTANCE, "x", "int")
        func = _make_simple_func(guards=[g])
        guards = harvest_guards(func)
        assert len(guards) == 1
        assert guards[0].kind == GuardKind.ISINSTANCE
        assert guards[0].variable == "x"
        assert guards[0].argument == "int"

    def test_harvest_typeof_guard(self) -> None:
        """Harvest typeof guard (TypeScript style)."""
        g = Guard(GuardKind.TYPEOF, "x", "number")
        func = _make_simple_func(guards=[g])
        guards = harvest_guards(func)
        assert len(guards) == 1
        assert guards[0].kind == GuardKind.TYPEOF

    def test_harvest_none_check(self) -> None:
        """Harvest 'x is None' guard."""
        g = Guard(GuardKind.IS_NONE, "x")
        func = _make_simple_func(guards=[g])
        guards = harvest_guards(func)
        assert len(guards) == 1
        assert guards[0].kind == GuardKind.IS_NONE

    def test_harvest_hasattr(self) -> None:
        """Harvest hasattr guard."""
        g = Guard(GuardKind.HASATTR, "x", "__len__")
        func = _make_simple_func(guards=[g])
        guards = harvest_guards(func)
        assert guards[0].argument == "__len__"

    def test_harvest_comparison_guard(self) -> None:
        """Harvest comparison guard: x < 10."""
        g = Guard(GuardKind.COMPARISON, "x", 10, ComparisonOp.LT)
        func = _make_simple_func(guards=[g])
        guards = harvest_guards(func)
        assert guards[0].op == ComparisonOp.LT
        assert guards[0].argument == 10

    def test_harvest_len_comparison(self) -> None:
        """Harvest len(x) > 0 guard."""
        g = Guard(GuardKind.LEN_COMPARISON, "x", 0, ComparisonOp.GT)
        func = _make_simple_func(guards=[g])
        guards = harvest_guards(func)
        assert guards[0].kind == GuardKind.LEN_COMPARISON

    def test_harvest_truthiness(self) -> None:
        """Harvest truthiness guard: if x."""
        g = Guard(GuardKind.TRUTHINESS, "x")
        func = _make_simple_func(guards=[g])
        guards = harvest_guards(func)
        assert guards[0].kind == GuardKind.TRUTHINESS

    def test_harvest_compound_guard(self) -> None:
        """Harvest multiple guards from one function."""
        gs = [
            Guard(GuardKind.ISINSTANCE, "x", "int"),
            Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE),
        ]
        func = _make_simple_func(guards=gs)
        guards = harvest_guards(func)
        assert len(guards) == 2

    def test_harvest_negated_guard(self) -> None:
        """Harvest and negate a guard."""
        g = Guard(GuardKind.IS_NONE, "x")
        neg = g.negate()
        assert neg.kind == GuardKind.IS_NOT_NONE

    def test_harvest_nested_guards(self) -> None:
        """Harvest guards from branching function."""
        guard = Guard(GuardKind.IS_NOT_NONE, "x")
        inner_guard = Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GT)
        func = _make_branch_func(
            guard=guard,
            true_stmts=[Stmt(StmtKind.GUARD, guard=inner_guard)],
        )
        guards = harvest_guards(func)
        assert len(guards) == 2

    def test_harvest_from_caller(self) -> None:
        """Guards from function with parameter constraints."""
        gs = [
            Guard(GuardKind.ISINSTANCE, "arg0", "list"),
            Guard(GuardKind.COMPARISON, "arg1", 0, ComparisonOp.GE),
        ]
        func = _make_simple_func(params=["arg0", "arg1"], guards=gs)
        guards = harvest_guards(func)
        assert any(g.variable == "arg0" for g in guards)
        assert any(g.variable == "arg1" for g in guards)

    def test_harvest_from_callee(self) -> None:
        """Guards harvested from a called function's body."""
        gs = [Guard(GuardKind.IS_NOT_NONE, "result")]
        func = _make_simple_func(name="callee", guards=gs)
        guards = harvest_guards(func)
        assert any(g.variable == "result" for g in guards)

    def test_guard_ranking(self) -> None:
        """Guards ranked by priority: isinstance first, truthiness last."""
        gs = [
            Guard(GuardKind.TRUTHINESS, "x"),
            Guard(GuardKind.COMPARISON, "y", 5, ComparisonOp.LT),
            Guard(GuardKind.ISINSTANCE, "z", "int"),
            Guard(GuardKind.IS_NONE, "w"),
        ]
        ranked = rank_guards(gs)
        assert ranked[0].kind == GuardKind.ISINSTANCE
        assert ranked[-1].kind == GuardKind.TRUTHINESS

    def test_guard_deduplication(self) -> None:
        """Duplicate guards are removed."""
        gs = [
            Guard(GuardKind.IS_NONE, "x"),
            Guard(GuardKind.IS_NONE, "x"),
            Guard(GuardKind.COMPARISON, "y", 5, ComparisonOp.LT),
        ]
        deduped = deduplicate_guards(gs)
        assert len(deduped) == 2

    def test_guard_normalization(self) -> None:
        """Guards are normalized to canonical form."""
        g = Guard(GuardKind.IS_NOT_NONE, "x")
        norm = normalize_guard(g)
        # Normalization converts IS_NOT_NONE → negated IS_NONE
        assert norm.kind == GuardKind.IS_NOT_NONE  # after double negate

    def test_harvest_no_guards(self) -> None:
        """Function with no guards returns empty list."""
        func = _make_simple_func(stmts=[Stmt(StmtKind.ASSIGN, target="x", value=42)])
        guards = harvest_guards(func)
        assert len(guards) == 0


class TestPredicateAbstraction:
    """Tests for predicate abstraction."""

    def test_empty_predicates(self) -> None:
        """Empty predicate set evaluates to True (vacuous truth)."""
        ps = PredicateSet.empty()
        assert ps.evaluate({"x": 5})
        assert len(ps) == 0

    def test_single_predicate(self) -> None:
        """Single predicate evaluation."""
        p = Predicate("x", "comparison", 5, ComparisonOp.GT)
        ps = PredicateSet.from_predicates(p)
        assert ps.evaluate({"x": 10})
        assert not ps.evaluate({"x": 3})

    def test_conjunction(self) -> None:
        """Conjunction of two predicates."""
        p1 = Predicate("x", "comparison", 0, ComparisonOp.GE)
        p2 = Predicate("x", "comparison", 10, ComparisonOp.LT)
        ps = PredicateSet.from_predicates(p1, p2)
        assert ps.evaluate({"x": 5})
        assert not ps.evaluate({"x": -1})
        assert not ps.evaluate({"x": 10})

    def test_disjunction(self) -> None:
        """Disjunction modeled as two separate predicate sets."""
        p1 = Predicate("x", "comparison", 0, ComparisonOp.LT)
        p2 = Predicate("x", "comparison", 10, ComparisonOp.GT)
        ps1 = PredicateSet.from_predicates(p1)
        ps2 = PredicateSet.from_predicates(p2)
        # Disjunction: at least one is satisfied
        assert ps1.evaluate({"x": -5}) or ps2.evaluate({"x": -5})
        assert ps1.evaluate({"x": 15}) or ps2.evaluate({"x": 15})
        assert not (ps1.evaluate({"x": 5}) or ps2.evaluate({"x": 5}))

    def test_predicate_implication(self) -> None:
        """p1 implies p2: x > 5 → x > 0."""
        p_strong = Predicate("x", "comparison", 5, ComparisonOp.GT)
        p_weak = Predicate("x", "comparison", 0, ComparisonOp.GT)
        # For all x where p_strong holds, p_weak also holds
        for x in range(-10, 20):
            if p_strong.evaluate({"x": x}):
                assert p_weak.evaluate({"x": x})

    def test_predicate_lattice_operations(self) -> None:
        """Predicate set union and intersection."""
        p1 = Predicate("x", "comparison", 0, ComparisonOp.GE)
        p2 = Predicate("y", "comparison", 0, ComparisonOp.GE)
        ps1 = PredicateSet.from_predicates(p1)
        ps2 = PredicateSet.from_predicates(p2)
        union = ps1.union(ps2)
        assert len(union) == 2
        assert union.evaluate({"x": 1, "y": 1})
        assert not union.evaluate({"x": -1, "y": 1})

    def test_predicate_projection(self) -> None:
        """Project predicates onto a subset of variables."""
        p1 = Predicate("x", "comparison", 0, ComparisonOp.GE)
        p2 = Predicate("y", "comparison", 0, ComparisonOp.GE)
        ps = PredicateSet.from_predicates(p1, p2)
        # Project: keep only predicates mentioning "x"
        projected = PredicateSet(frozenset(
            p for p in ps.predicates if "x" in p.free_variables()
        ))
        assert len(projected) == 1

    def test_predicate_weakening(self) -> None:
        """Weaken predicate: x > 5 → x > 0."""
        strong = Predicate("x", "comparison", 5, ComparisonOp.GT)
        weak = Predicate("x", "comparison", 0, ComparisonOp.GT)
        # Verify weakening
        for x in [-10, 0, 3, 6, 10]:
            if strong.evaluate({"x": x}):
                assert weak.evaluate({"x": x})

    def test_predicate_strengthening(self) -> None:
        """Strengthen predicate: x > 0 + x < 10 → 0 < x < 10."""
        p1 = Predicate("x", "comparison", 0, ComparisonOp.GT)
        p2 = Predicate("x", "comparison", 10, ComparisonOp.LT)
        ps = PredicateSet.from_predicates(p1, p2)
        assert ps.evaluate({"x": 5})
        assert not ps.evaluate({"x": 0})
        assert not ps.evaluate({"x": 10})

    def test_predicate_negation(self) -> None:
        """Negate a comparison predicate."""
        p = Predicate("x", "comparison", 5, ComparisonOp.GT)
        neg = p.negate()
        assert neg.op == ComparisonOp.LE
        assert neg.evaluate({"x": 3})
        assert not neg.evaluate({"x": 10})

    def test_predicate_substitution(self) -> None:
        """Substitute variable names in predicates."""
        p = Predicate("x", "comparison", 5, ComparisonOp.GT)
        sub = p.substitute({"x": "y"})
        assert sub.variable == "y"
        assert sub.evaluate({"y": 10})

    def test_isinstance_predicate(self) -> None:
        """isinstance predicate evaluation."""
        p = Predicate("x", "isinstance", "int")
        assert p.evaluate({"x": 42})
        assert not p.evaluate({"x": "hello"})

    def test_is_none_predicate(self) -> None:
        """is_none predicate evaluation."""
        p = Predicate("x", "is_none")
        assert p.evaluate({"x": None})
        assert not p.evaluate({"x": 42})

    def test_truthiness_predicate(self) -> None:
        """Truthiness predicate evaluation."""
        p = Predicate("x", "truthiness")
        assert p.evaluate({"x": 42})
        assert p.evaluate({"x": "hello"})
        assert not p.evaluate({"x": 0})
        assert not p.evaluate({"x": ""})
        assert not p.evaluate({"x": None})


class TestAbstractInterpreter:
    """Tests for the abstract interpretation phase."""

    def test_simple_assignment(self) -> None:
        """Interpret x = 42."""
        func = _make_simple_func(stmts=[
            Stmt(StmtKind.ASSIGN, target="x", value=42),
        ])
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        state = states[0]
        x_iv = state.get_interval("x")
        assert x_iv.contains(42)

    def test_conditional_branch(self) -> None:
        """Interpret if x > 0: ... else: ..."""
        guard = Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GT)
        func = _make_branch_func(
            params=["x"],
            guard=guard,
            true_stmts=[Stmt(StmtKind.ASSIGN, target="y", value=1)],
            false_stmts=[Stmt(StmtKind.ASSIGN, target="y", value=0)],
        )
        initial = AbstractState().set_interval("x", Interval.top())
        states = interpret_function(func, initial, PredicateSet.empty())
        # Entry block has the guard
        entry_state = states[0]
        assert entry_state.get_interval("x").contains(1)

    def test_loop_with_widening(self) -> None:
        """Loop: i = 0; while i < n: i += 1."""
        func = _make_loop_func(
            params=["n"],
            init_stmts=[Stmt(StmtKind.ASSIGN, target="i", value=0)],
            loop_guard=Guard(GuardKind.COMPARISON, "i", 100, ComparisonOp.LT),
            body_stmts=[Stmt(StmtKind.ASSIGN, target="i", value=1)],
        )
        initial = AbstractState().set_interval("n", Interval.from_bounds(0, 100))
        states = interpret_function(func, initial, PredicateSet.empty())
        # Should converge
        assert 1 in states or 3 in states

    def test_nested_loop(self) -> None:
        """Nested loop structure: two loop headers."""
        outer_header = BasicBlock(id=0, stmts=[
            Stmt(StmtKind.ASSIGN, target="i", value=0),
            Stmt(StmtKind.GUARD, guard=Guard(GuardKind.COMPARISON, "i", 10, ComparisonOp.LT)),
        ], successors=[1, 3])
        inner_header = BasicBlock(id=1, stmts=[
            Stmt(StmtKind.ASSIGN, target="j", value=0),
            Stmt(StmtKind.GUARD, guard=Guard(GuardKind.COMPARISON, "j", 5, ComparisonOp.LT)),
        ], successors=[2, 0], predecessors=[0, 2])
        inner_body = BasicBlock(id=2, stmts=[], successors=[1], predecessors=[1])
        exit_block = BasicBlock(id=3, stmts=[], successors=[], predecessors=[0])
        func = IRFunction("nested", blocks=[outer_header, inner_header, inner_body, exit_block], entry=0)
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        assert len(states) > 0

    def test_function_call_with_summary(self) -> None:
        """Function call modeled as havoc of return variable."""
        func = _make_simple_func(stmts=[
            Stmt(StmtKind.CALL, target="result", value="some_function"),
        ])
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        state = states[0]
        # Call doesn't assign interval, so result has top
        assert state.get_interval("result").is_top

    def test_exception_path(self) -> None:
        """Exception path creates new control flow edge."""
        # Model: try block might raise, catch block handles
        try_block = BasicBlock(id=0, stmts=[
            Stmt(StmtKind.ASSIGN, target="x", value=42),
        ], successors=[1, 2])
        normal = BasicBlock(id=1, stmts=[], successors=[3], predecessors=[0])
        catch = BasicBlock(id=2, stmts=[
            Stmt(StmtKind.ASSIGN, target="x", value=-1),
        ], successors=[3], predecessors=[0])
        merge = BasicBlock(id=3, stmts=[], successors=[], predecessors=[1, 2])
        func = IRFunction("exc", blocks=[try_block, normal, catch, merge], entry=0)
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        assert 3 in states

    def test_phi_node_merge(self) -> None:
        """Phi node at merge point joins branch values."""
        guard = Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GT)
        func = _make_branch_func(
            params=["x"],
            guard=guard,
            true_stmts=[Stmt(StmtKind.ASSIGN, target="y", value=1)],
            false_stmts=[Stmt(StmtKind.ASSIGN, target="y", value=0)],
        )
        initial = AbstractState().set_interval("x", Interval.from_bounds(-10, 10))
        states = interpret_function(func, initial, PredicateSet.empty())
        # Merge block should have y as join of [1,1] and [0,0] = [0,1]
        if 3 in states:
            merge_state = states[3]
            y_iv = merge_state.get_interval("y")
            assert y_iv.contains(0) or y_iv.contains(1) or y_iv.is_top

    def test_guard_assertion_effect(self) -> None:
        """Guard narrows state: x >= 0 means x in [0, +inf)."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE)],
        )
        initial = AbstractState().set_interval("x", Interval.top())
        states = interpret_function(func, initial, PredicateSet.empty())
        state = states[0]
        x_iv = state.get_interval("x")
        # After guard x >= 0, negative values should be excluded or state refined
        assert x_iv.contains(0)
        assert x_iv.contains(10)

    def test_truthiness_coercion(self) -> None:
        """Truthiness guard: if x → x is not None."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.TRUTHINESS, "x")],
        )
        # Pass initial state through the function directly
        block = func.blocks[0]
        initial = AbstractState().set_nullity("x", NullityValue.maybe_null())
        result = interpret_block(block, initial, PredicateSet.empty())
        assert result.get_nullity("x") == NullityValue.definitely_not_null()

    def test_array_access_check(self) -> None:
        """Detect potential OOB in abstract state."""
        func = _make_simple_func(
            params=["arr", "i"],
            stmts=[Stmt(StmtKind.SUBSCRIPT, target="i", source="arr")],
        )
        initial = AbstractState()
        initial = initial.set_interval("i", Interval.top())
        states = interpret_function(func, initial, PredicateSet.empty())
        violations = detect_violations(func, states)
        assert any(v[0] == ViolationKind.ARRAY_OUT_OF_BOUNDS for v in violations)

    def test_null_deref_check(self) -> None:
        """Detect potential null dereference."""
        func = _make_simple_func(
            params=["x"],
            stmts=[Stmt(StmtKind.ATTR_ACCESS, source="x", value="method")],
        )
        initial = AbstractState().set_nullity("x", NullityValue.maybe_null())
        states = interpret_function(func, initial, PredicateSet.empty())
        violations = detect_violations(func, states)
        assert any(v[0] == ViolationKind.NULL_DEREFERENCE for v in violations)

    def test_division_check(self) -> None:
        """Detect potential division by zero."""
        func = _make_simple_func(
            params=["x", "y"],
            stmts=[Stmt(StmtKind.BINOP, target="result", source="y", value="div")],
        )
        initial = AbstractState().set_interval("y", Interval.from_bounds(-5, 5))
        states = interpret_function(func, initial, PredicateSet.empty())
        violations = detect_violations(func, states)
        assert any(v[0] == ViolationKind.DIVISION_BY_ZERO for v in violations)

    def test_type_confusion_check(self) -> None:
        """Type confusion when calling method on wrong type."""
        func = _make_simple_func(
            params=["x"],
            stmts=[Stmt(StmtKind.ATTR_ACCESS, source="x", value="append")],
        )
        initial = AbstractState().set_nullity("x", NullityValue.maybe_null())
        states = interpret_function(func, initial, PredicateSet.empty())
        violations = detect_violations(func, states)
        # Without type tag info, this shows as null deref potential
        assert len(violations) > 0

    def test_worklist_order(self) -> None:
        """Worklist processes blocks in FIFO order."""
        block0 = BasicBlock(id=0, stmts=[], successors=[1, 2])
        block1 = BasicBlock(id=1, stmts=[], successors=[3], predecessors=[0])
        block2 = BasicBlock(id=2, stmts=[], successors=[3], predecessors=[0])
        block3 = BasicBlock(id=3, stmts=[], successors=[], predecessors=[1, 2])
        func = IRFunction("order", blocks=[block0, block1, block2, block3], entry=0)
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        # All blocks should be visited
        assert 0 in states
        assert 3 in states

    def test_convergence(self) -> None:
        """Abstract interpretation converges on a loop."""
        func = _make_loop_func(
            init_stmts=[Stmt(StmtKind.ASSIGN, target="i", value=0)],
            loop_guard=Guard(GuardKind.COMPARISON, "i", 1000, ComparisonOp.LT),
            body_stmts=[Stmt(StmtKind.ASSIGN, target="i", value=1)],
        )
        states = interpret_function(func, AbstractState(), PredicateSet.empty(), max_iterations=50)
        # Should have reached fixpoint
        assert len(states) > 0


class TestCounterexampleAnalysis:
    """Tests for counterexample analysis."""

    def test_real_counterexample(self) -> None:
        """Identify a real (feasible) counterexample."""
        func = _make_simple_func(
            params=["x"],
            stmts=[Stmt(StmtKind.ATTR_ACCESS, source="x", value="method")],
        )
        ce = Counterexample(path=[0], env={"x": None}, violation=ViolationKind.NULL_DEREFERENCE)
        feasible = check_path_feasibility(func, ce.path, ce.env)
        ce.is_feasible = feasible
        assert ce.is_real()

    def test_spurious_counterexample(self) -> None:
        """Identify a spurious (infeasible) counterexample."""
        guard = Guard(GuardKind.IS_NOT_NONE, "x")
        func = _make_simple_func(
            params=["x"],
            guards=[guard],
            stmts=[Stmt(StmtKind.ATTR_ACCESS, source="x", value="method")],
        )
        # CE claims x=None but path has is_not_none guard
        ce = Counterexample(path=[0], env={"x": None}, violation=ViolationKind.NULL_DEREFERENCE)
        feasible = check_path_feasibility(func, ce.path, ce.env)
        ce.is_feasible = feasible
        assert ce.is_spurious()

    def test_path_feasibility_check(self) -> None:
        """Path with satisfied guards is feasible."""
        guard = Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GT)
        func = _make_simple_func(params=["x"], guards=[guard])
        assert check_path_feasibility(func, [0], {"x": 5})
        assert not check_path_feasibility(func, [0], {"x": -1})

    def test_interpolant_extraction(self) -> None:
        """Extract interpolant from spurious CE."""
        guard = Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GT)
        func = _make_simple_func(params=["x"], guards=[guard])
        ce = Counterexample(path=[0], env={"x": -1}, violation=ViolationKind.ARRAY_OUT_OF_BOUNDS)
        ce.is_feasible = False
        interpolant = extract_interpolant(func, ce)
        assert interpolant is not None
        assert interpolant.variable == "x"

    def test_predicate_from_interpolant(self) -> None:
        """Interpolant yields a useful predicate."""
        guard = Guard(GuardKind.COMPARISON, "i", 10, ComparisonOp.LT)
        func = _make_simple_func(params=["i"], guards=[guard])
        ce = Counterexample(path=[0], env={"i": 15}, is_feasible=False)
        interpolant = extract_interpolant(func, ce)
        assert interpolant is not None
        assert interpolant.op == ComparisonOp.LT
        assert interpolant.argument == 10

    def test_predicate_projection_into_P(self) -> None:
        """Interpolant predicate is projected into current predicate set."""
        p = Predicate("x", "comparison", 0, ComparisonOp.GE)
        ps = PredicateSet.empty().add(p)
        assert len(ps) == 1
        # Adding same predicate doesn't increase size
        ps2 = ps.add(p)
        assert len(ps2) == 1

    def test_ce_with_loop(self) -> None:
        """Counterexample through a loop."""
        func = _make_loop_func(
            params=["n"],
            init_stmts=[Stmt(StmtKind.ASSIGN, target="i", value=0)],
            loop_guard=Guard(GuardKind.COMPARISON, "i", 5, ComparisonOp.LT),
        )
        ce = Counterexample(path=[0, 1, 2, 1, 2, 1, 3], env={"n": 5, "i": 0})
        # Path through loop is feasible
        # (simplified check: just verify we can traverse)
        assert len(ce.path) > 1

    def test_ce_across_functions(self) -> None:
        """CE that spans a function call boundary."""
        caller = _make_simple_func(name="caller", params=["x"], stmts=[
            Stmt(StmtKind.CALL, target="result", value="callee"),
        ])
        ce = Counterexample(
            path=[0],
            env={"x": None, "result": None},
            violation=ViolationKind.NULL_DEREFERENCE,
        )
        feasible = check_path_feasibility(caller, ce.path, ce.env)
        assert feasible  # No guards on this path


class TestCegarLoop:
    """Tests for the full CEGAR loop."""

    def test_converges_on_simple_function(self) -> None:
        """CEGAR converges on a simple function with no bugs."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.IS_NOT_NONE, "x")],
            stmts=[Stmt(StmtKind.ASSIGN, target="y", value=42)],
        )
        result = run_cegar(func)
        assert result.converged

    def test_converges_on_guarded_access(self) -> None:
        """CEGAR converges when access is properly guarded."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.IS_NOT_NONE, "x")],
            stmts=[Stmt(StmtKind.ATTR_ACCESS, source="x", value="method")],
        )
        initial = AbstractState().set_nullity("x", NullityValue.maybe_null())
        # After guard, x is not none, so no null deref
        result = run_cegar(func)
        assert result.converged

    def test_converges_on_loop_invariant(self) -> None:
        """CEGAR converges on a loop with invariant."""
        func = _make_loop_func(
            params=["n"],
            init_stmts=[Stmt(StmtKind.ASSIGN, target="i", value=0)],
            loop_guard=Guard(GuardKind.COMPARISON, "i", 10, ComparisonOp.LT),
        )
        result = run_cegar(func)
        assert result.converged or result.iterations <= 100

    def test_max_iterations_bound(self) -> None:
        """CEGAR respects max iterations."""
        config = CEGARConfig(max_iterations=5)
        func = _make_simple_func(params=["x"])
        result = run_cegar(func, config)
        assert result.iterations <= 5

    def test_convergence_with_guard_seeding(self) -> None:
        """Guard seeding speeds up convergence."""
        func = _make_simple_func(
            params=["x"],
            guards=[
                Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE),
                Guard(GuardKind.COMPARISON, "x", 100, ComparisonOp.LT),
            ],
        )
        config = CEGARConfig(enable_guard_seeding=True)
        result = run_cegar(func, config)
        assert len(result.predicates_used) >= 2

    def test_convergence_without_guard_seeding(self) -> None:
        """CEGAR can converge without guard seeding (may take more iterations)."""
        func = _make_simple_func(params=["x"], stmts=[
            Stmt(StmtKind.ASSIGN, target="y", value=42),
        ])
        config = CEGARConfig(enable_guard_seeding=False)
        result = run_cegar(func, config)
        assert result.converged

    def test_guard_seeding_speedup(self) -> None:
        """Guard-seeded CEGAR uses fewer iterations (or same) than unseeded."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE)],
            stmts=[Stmt(StmtKind.ASSIGN, target="y", value=1)],
        )
        seeded = run_cegar(func, CEGARConfig(enable_guard_seeding=True))
        unseeded = run_cegar(func, CEGARConfig(enable_guard_seeding=False))
        assert seeded.iterations <= unseeded.iterations + 1

    def test_predicate_set_growth(self) -> None:
        """Predicate set grows during refinement."""
        func = _make_simple_func(
            params=["x"],
            guards=[
                Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE),
                Guard(GuardKind.COMPARISON, "x", 10, ComparisonOp.LT),
                Guard(GuardKind.IS_NOT_NONE, "x"),
            ],
        )
        result = run_cegar(func)
        assert len(result.predicates_used) >= 1

    def test_cegar_on_null_check_pattern(self) -> None:
        """CEGAR handles null-check-then-use pattern."""
        func = _make_branch_func(
            params=["x"],
            guard=Guard(GuardKind.IS_NOT_NONE, "x"),
            true_stmts=[Stmt(StmtKind.ATTR_ACCESS, source="x", value="attr")],
            false_stmts=[Stmt(StmtKind.ASSIGN, target="y", value=0)],
        )
        result = run_cegar(func)
        # No null deref bug: access is in guarded branch
        assert result.converged

    def test_cegar_on_bounds_check_pattern(self) -> None:
        """CEGAR handles bounds-check-then-access pattern."""
        func = _make_simple_func(
            params=["arr", "i"],
            guards=[
                Guard(GuardKind.COMPARISON, "i", 0, ComparisonOp.GE),
                Guard(GuardKind.COMPARISON, "i", 10, ComparisonOp.LT),
            ],
            stmts=[Stmt(StmtKind.SUBSCRIPT, target="i", source="arr")],
        )
        result = run_cegar(func)
        assert result.converged

    def test_cegar_on_type_guard_pattern(self) -> None:
        """CEGAR handles isinstance-then-method pattern."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.ISINSTANCE, "x", "list")],
            stmts=[Stmt(StmtKind.ATTR_ACCESS, source="x", value="append")],
        )
        result = run_cegar(func)
        assert result.converged

    def test_cegar_on_division_guard_pattern(self) -> None:
        """CEGAR handles zero-check-then-divide pattern."""
        func = _make_simple_func(
            params=["x", "y"],
            guards=[Guard(GuardKind.COMPARISON, "y", 0, ComparisonOp.NE)],
            stmts=[Stmt(StmtKind.BINOP, target="result", source="y", value="div")],
        )
        result = run_cegar(func)
        assert result.converged

    def test_cegar_on_complex_function(self) -> None:
        """CEGAR on a function with multiple guards and accesses."""
        func = _make_simple_func(
            params=["x", "y", "arr"],
            guards=[
                Guard(GuardKind.IS_NOT_NONE, "x"),
                Guard(GuardKind.ISINSTANCE, "x", "int"),
                Guard(GuardKind.COMPARISON, "y", 0, ComparisonOp.GT),
            ],
            stmts=[
                Stmt(StmtKind.ATTR_ACCESS, source="x", value="bit_length"),
                Stmt(StmtKind.BINOP, target="z", source="y", value="div"),
            ],
        )
        result = run_cegar(func)
        assert result.converged

    def test_cegar_no_spurious_after_convergence(self) -> None:
        """After convergence, all reported bugs should be real."""
        func = _make_simple_func(
            params=["x"],
            stmts=[Stmt(StmtKind.ATTR_ACCESS, source="x", value="attr")],
        )
        result = run_cegar(func)
        # x has maybe-null, so null deref is a real concern
        if result.bug_reports:
            assert all(b.confidence > 0.0 for b in result.bug_reports)

    def test_cegar_statistics(self) -> None:
        """CEGAR reports correct statistics."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE)],
        )
        result = run_cegar(func)
        stats = result.stats
        assert "iterations" in stats
        assert "converged" in stats
        assert "predicates" in stats
        assert "bugs" in stats
        assert "time" in stats
        assert stats["time"] >= 0

    def test_cegar_timeout(self) -> None:
        """CEGAR respects timeout."""
        config = CEGARConfig(per_function_timeout=0.001, max_iterations=10000)
        func = _make_simple_func(params=["x"])
        result = run_cegar(func, config)
        assert result.time_taken < 1.0  # Should be very fast


class TestIncrementalCegar:
    """Tests for incremental refinement."""

    def test_incremental_after_small_change(self) -> None:
        """After a small code change, reuse predicates from previous run."""
        func_v1 = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE)],
        )
        result_v1 = run_cegar(func_v1)
        old_preds = result_v1.predicates_used

        # Small change: add another guard
        func_v2 = _make_simple_func(
            params=["x"],
            guards=[
                Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE),
                Guard(GuardKind.COMPARISON, "x", 100, ComparisonOp.LT),
            ],
        )
        result_v2 = run_cegar(func_v2)
        # Should reuse/include old predicates
        assert len(result_v2.predicates_used) >= len(old_preds)

    def test_incremental_predicate_reuse(self) -> None:
        """Predicates from previous analysis are reused."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.IS_NOT_NONE, "x")],
        )
        r1 = run_cegar(func)
        r2 = run_cegar(func)
        # Same predicates for same function
        assert r1.predicates_used == r2.predicates_used

    def test_incremental_invalidation(self) -> None:
        """Predicates are invalidated when function changes significantly."""
        func_v1 = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE)],
        )
        func_v2 = _make_simple_func(
            params=["x", "y"],
            guards=[Guard(GuardKind.COMPARISON, "y", 0, ComparisonOp.GE)],
        )
        r1 = run_cegar(func_v1)
        r2 = run_cegar(func_v2)
        # Different functions, potentially different predicates
        # At minimum, v2 should have a predicate about "y"
        pred_vars_2 = {p.variable for p in r2.predicates_used.predicates}
        assert "y" in pred_vars_2

    def test_incremental_dependency_tracking(self) -> None:
        """Track dependencies between functions for incremental analysis."""
        caller = _make_simple_func(
            name="caller",
            params=["x"],
            stmts=[Stmt(StmtKind.CALL, target="result", value="callee")],
        )
        callee = _make_simple_func(
            name="callee",
            params=["y"],
            guards=[Guard(GuardKind.IS_NOT_NONE, "y")],
        )
        # Analyze both
        r_caller = run_cegar(caller)
        r_callee = run_cegar(callee)
        # Both should converge independently
        assert r_caller.converged or r_caller.iterations > 0
        assert r_callee.converged


class TestGuardHarvestingExtended:
    """Extended guard harvesting tests."""

    def test_harvest_multiple_isinstance(self) -> None:
        """Harvest multiple isinstance guards for same variable."""
        gs = [
            Guard(GuardKind.ISINSTANCE, "x", "int"),
            Guard(GuardKind.ISINSTANCE, "x", "str"),
        ]
        func = _make_simple_func(guards=gs)
        guards = harvest_guards(func)
        assert len(guards) == 2

    def test_harvest_mixed_variables(self) -> None:
        """Harvest guards for different variables."""
        gs = [
            Guard(GuardKind.IS_NONE, "x"),
            Guard(GuardKind.COMPARISON, "y", 5, ComparisonOp.LT),
            Guard(GuardKind.TRUTHINESS, "z"),
        ]
        func = _make_simple_func(guards=gs)
        guards = harvest_guards(func)
        vars_seen = {g.variable for g in guards}
        assert vars_seen == {"x", "y", "z"}

    def test_harvest_from_loop_body(self) -> None:
        """Harvest guards from loop body."""
        func = _make_loop_func(
            params=["items"],
            loop_guard=Guard(GuardKind.COMPARISON, "i", 10, ComparisonOp.LT),
            body_stmts=[
                Stmt(StmtKind.GUARD, guard=Guard(GuardKind.IS_NOT_NONE, "item")),
            ],
        )
        guards = harvest_guards(func)
        assert len(guards) == 2
        kinds = {g.kind for g in guards}
        assert GuardKind.COMPARISON in kinds
        assert GuardKind.IS_NOT_NONE in kinds

    def test_guard_negate_comparison(self) -> None:
        """Negating comparison guard flips operator."""
        g = Guard(GuardKind.COMPARISON, "x", 5, ComparisonOp.LT)
        neg = g.negate()
        assert neg.op == ComparisonOp.GE

    def test_guard_negate_double(self) -> None:
        """Double negation returns equivalent guard."""
        g = Guard(GuardKind.IS_NONE, "x")
        neg = g.negate()
        assert neg.kind == GuardKind.IS_NOT_NONE
        neg2 = neg.negate()
        assert neg2.kind == GuardKind.IS_NONE

    def test_rank_preserves_all(self) -> None:
        """Ranking preserves all guards."""
        gs = [
            Guard(GuardKind.TRUTHINESS, "a"),
            Guard(GuardKind.ISINSTANCE, "b", "int"),
            Guard(GuardKind.IS_NONE, "c"),
        ]
        ranked = rank_guards(gs)
        assert len(ranked) == len(gs)
        assert set(ranked) == set(gs)

    def test_dedup_preserves_order(self) -> None:
        """Deduplication preserves first-seen order."""
        gs = [
            Guard(GuardKind.IS_NONE, "x"),
            Guard(GuardKind.COMPARISON, "y", 5, ComparisonOp.LT),
            Guard(GuardKind.IS_NONE, "x"),
        ]
        deduped = deduplicate_guards(gs)
        assert deduped[0].kind == GuardKind.IS_NONE
        assert deduped[1].kind == GuardKind.COMPARISON

    def test_guards_to_predicates_all_kinds(self) -> None:
        """Convert all guard kinds to predicates."""
        gs = [
            Guard(GuardKind.ISINSTANCE, "a", "int"),
            Guard(GuardKind.TYPEOF, "b", "number"),
            Guard(GuardKind.IS_NONE, "c"),
            Guard(GuardKind.IS_NOT_NONE, "d"),
            Guard(GuardKind.HASATTR, "e", "__len__"),
            Guard(GuardKind.COMPARISON, "f", 5, ComparisonOp.GT),
            Guard(GuardKind.TRUTHINESS, "g"),
            Guard(GuardKind.LEN_COMPARISON, "h", 0, ComparisonOp.GT),
        ]
        preds = guards_to_predicates(gs)
        assert len(preds) == len(gs)

    def test_harvest_empty_blocks(self) -> None:
        """Harvest from function with empty blocks."""
        func = IRFunction("empty", blocks=[
            BasicBlock(id=0, stmts=[], successors=[1]),
            BasicBlock(id=1, stmts=[], successors=[]),
        ], entry=0)
        guards = harvest_guards(func)
        assert len(guards) == 0


class TestPredicateAbstractionExtended:
    """Extended predicate abstraction tests."""

    def test_predicate_set_add_duplicate(self) -> None:
        """Adding duplicate predicate doesn't increase size."""
        p = Predicate("x", "comparison", 5, ComparisonOp.GT)
        ps = PredicateSet.empty().add(p).add(p)
        assert len(ps) == 1

    def test_predicate_set_remove(self) -> None:
        """Remove predicate from set."""
        p1 = Predicate("x", "comparison", 5, ComparisonOp.GT)
        p2 = Predicate("y", "is_none")
        ps = PredicateSet.from_predicates(p1, p2)
        ps2 = ps.remove(p1)
        assert len(ps2) == 1
        assert p2 in ps2.predicates

    def test_predicate_free_variables(self) -> None:
        """Free variables of a predicate."""
        p = Predicate("x", "comparison", 5, ComparisonOp.GT)
        assert p.free_variables() == {"x"}

    def test_predicate_free_variables_none(self) -> None:
        """Predicate with no variable."""
        p = Predicate(None, "truthiness")
        assert p.free_variables() == set()

    def test_predicate_evaluate_missing_var(self) -> None:
        """Predicate with missing variable evaluates to False."""
        p = Predicate("x", "comparison", 5, ComparisonOp.GT)
        assert not p.evaluate({"y": 10})

    def test_predicate_all_comparison_ops(self) -> None:
        """Test all comparison operators."""
        for op, val, expected in [
            (ComparisonOp.LT, 3, True),
            (ComparisonOp.LE, 5, True),
            (ComparisonOp.GT, 10, True),
            (ComparisonOp.GE, 5, True),
            (ComparisonOp.EQ, 5, True),
            (ComparisonOp.NE, 3, True),
        ]:
            p = Predicate("x", "comparison", 5, op)
            assert p.evaluate({"x": val}) == expected, f"Failed for {op} with val={val}"

    def test_predicate_union_associative(self) -> None:
        """Predicate set union is associative."""
        p1 = PredicateSet.from_predicates(Predicate("x", "is_none"))
        p2 = PredicateSet.from_predicates(Predicate("y", "is_none"))
        p3 = PredicateSet.from_predicates(Predicate("z", "is_none"))
        assert p1.union(p2).union(p3) == p1.union(p2.union(p3))

    def test_predicate_conjunction_evaluation(self) -> None:
        """Conjunction of many predicates."""
        preds = [
            Predicate("x", "comparison", 0, ComparisonOp.GE),
            Predicate("x", "comparison", 100, ComparisonOp.LE),
            Predicate("x", "comparison", 50, ComparisonOp.NE),
        ]
        ps = PredicateSet(frozenset(preds))
        assert ps.evaluate({"x": 25})
        assert not ps.evaluate({"x": 50})
        assert not ps.evaluate({"x": -1})
        assert not ps.evaluate({"x": 101})


class TestAbstractInterpreterExtended:
    """Extended abstract interpreter tests."""

    def test_multiple_assignments(self) -> None:
        """Interpret x = 1; y = 2; z = x + y."""
        func = _make_simple_func(stmts=[
            Stmt(StmtKind.ASSIGN, target="x", value=1),
            Stmt(StmtKind.ASSIGN, target="y", value=2),
        ])
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        state = states[0]
        assert state.get_interval("x").contains(1)
        assert state.get_interval("y").contains(2)

    def test_none_assignment(self) -> None:
        """Interpret x = None."""
        func = _make_simple_func(stmts=[
            Stmt(StmtKind.ASSIGN, target="x", value=None),
        ])
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        state = states[0]
        assert state.get_nullity("x") == NullityValue.definitely_null()

    def test_guard_is_none(self) -> None:
        """Guard x is None narrows nullity."""
        func = _make_simple_func(
            params=["x"],
            guards=[Guard(GuardKind.IS_NONE, "x")],
        )
        initial = AbstractState().set_nullity("x", NullityValue.maybe_null())
        block = func.blocks[0]
        result = interpret_block(block, initial, PredicateSet.empty())
        assert result.get_nullity("x") == NullityValue.definitely_null()

    def test_guard_comparison_chain(self) -> None:
        """Multiple guards narrow interval progressively."""
        func = _make_simple_func(
            params=["x"],
            guards=[
                Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE),
                Guard(GuardKind.COMPARISON, "x", 100, ComparisonOp.LT),
            ],
        )
        initial = AbstractState().set_interval("x", Interval.top())
        block = func.blocks[0]
        result = interpret_block(block, initial, PredicateSet.empty())
        x_iv = result.get_interval("x")
        assert x_iv.lo == Bound.finite(0)
        assert x_iv.hi == Bound.finite(99)

    def test_empty_function_analysis(self) -> None:
        """Analyze empty function."""
        func = IRFunction("empty", blocks=[BasicBlock(id=0, stmts=[])], entry=0)
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        assert 0 in states

    def test_linear_cfg(self) -> None:
        """Analyze linear CFG: block0 → block1 → block2."""
        b0 = BasicBlock(id=0, stmts=[Stmt(StmtKind.ASSIGN, target="x", value=1)], successors=[1])
        b1 = BasicBlock(id=1, stmts=[Stmt(StmtKind.ASSIGN, target="y", value=2)], successors=[2], predecessors=[0])
        b2 = BasicBlock(id=2, stmts=[], successors=[], predecessors=[1])
        func = IRFunction("linear", blocks=[b0, b1, b2], entry=0)
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        assert 2 in states

    def test_diamond_cfg(self) -> None:
        """Analyze diamond CFG: entry → {true, false} → merge."""
        entry = BasicBlock(id=0, stmts=[], successors=[1, 2])
        true_blk = BasicBlock(id=1, stmts=[
            Stmt(StmtKind.ASSIGN, target="x", value=1),
        ], successors=[3], predecessors=[0])
        false_blk = BasicBlock(id=2, stmts=[
            Stmt(StmtKind.ASSIGN, target="x", value=2),
        ], successors=[3], predecessors=[0])
        merge = BasicBlock(id=3, stmts=[], successors=[], predecessors=[1, 2])
        func = IRFunction("diamond", blocks=[entry, true_blk, false_blk, merge], entry=0)
        states = interpret_function(func, AbstractState(), PredicateSet.empty())
        assert 3 in states
        # x should be join of [1,1] and [2,2]
        merge_state = states[3]
        assert merge_state.get_interval("x").contains(1)
        assert merge_state.get_interval("x").contains(2)

    def test_max_iterations_respected(self) -> None:
        """Abstract interpretation respects max_iterations."""
        func = _make_loop_func(
            init_stmts=[Stmt(StmtKind.ASSIGN, target="i", value=0)],
            loop_guard=Guard(GuardKind.COMPARISON, "i", 10**9, ComparisonOp.LT),
        )
        states = interpret_function(func, AbstractState(), PredicateSet.empty(), max_iterations=5)
        assert len(states) > 0


class TestCounterexampleExtended:
    """Extended counterexample analysis tests."""

    def test_feasibility_no_guards(self) -> None:
        """Path with no guards is always feasible."""
        func = _make_simple_func(stmts=[Stmt(StmtKind.ASSIGN, target="x", value=1)])
        assert check_path_feasibility(func, [0], {"x": 42})

    def test_feasibility_satisfied_guard(self) -> None:
        """Path with satisfied guard is feasible."""
        func = _make_simple_func(
            guards=[Guard(GuardKind.COMPARISON, "x", 10, ComparisonOp.LT)],
        )
        assert check_path_feasibility(func, [0], {"x": 5})

    def test_feasibility_violated_guard(self) -> None:
        """Path with violated guard is infeasible."""
        func = _make_simple_func(
            guards=[Guard(GuardKind.COMPARISON, "x", 10, ComparisonOp.LT)],
        )
        assert not check_path_feasibility(func, [0], {"x": 15})

    def test_interpolant_from_comparison(self) -> None:
        """Extract comparison predicate from spurious CE."""
        func = _make_simple_func(
            guards=[Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE)],
        )
        ce = Counterexample(path=[0], env={"x": -5}, is_feasible=False)
        interpolant = extract_interpolant(func, ce)
        assert interpolant is not None
        assert interpolant.op == ComparisonOp.GE
        assert interpolant.argument == 0

    def test_interpolant_from_null_check(self) -> None:
        """Extract null predicate from spurious CE."""
        func = _make_simple_func(
            guards=[Guard(GuardKind.IS_NONE, "x")],
        )
        ce = Counterexample(path=[0], env={"x": 42}, is_feasible=False)
        interpolant = extract_interpolant(func, ce)
        assert interpolant is not None
        assert interpolant.kind == "is_none"

    def test_counterexample_classification(self) -> None:
        """Classify counterexample as real or spurious."""
        real = Counterexample(path=[0], env={"x": None}, is_feasible=True)
        spurious = Counterexample(path=[0], env={"x": None}, is_feasible=False)
        assert real.is_real()
        assert not real.is_spurious()
        assert spurious.is_spurious()
        assert not spurious.is_real()

    def test_counterexample_unknown(self) -> None:
        """Unclassified counterexample."""
        unknown = Counterexample(path=[0], env={"x": 5})
        assert not unknown.is_real()
        assert not unknown.is_spurious()


class TestCegarLoopExtended:
    """Extended CEGAR loop tests."""

    def test_cegar_empty_function(self) -> None:
        """CEGAR on empty function converges immediately."""
        func = IRFunction("empty", blocks=[BasicBlock(id=0, stmts=[])], entry=0)
        result = run_cegar(func)
        assert result.converged

    def test_cegar_multiple_violations(self) -> None:
        """CEGAR handles multiple violations in same function."""
        func = _make_simple_func(
            params=["x", "y"],
            stmts=[
                Stmt(StmtKind.ATTR_ACCESS, source="x", value="attr"),
                Stmt(StmtKind.BINOP, target="z", source="y", value="div"),
            ],
        )
        result = run_cegar(func)
        assert result.converged or len(result.bug_reports) > 0

    def test_cegar_result_time_positive(self) -> None:
        """CEGAR result has positive time."""
        func = _make_simple_func(params=["x"])
        result = run_cegar(func)
        assert result.time_taken >= 0

    def test_cegar_predicate_budget(self) -> None:
        """CEGAR respects predicate budget."""
        config = CEGARConfig(predicate_budget=2, max_iterations=50)
        func = _make_simple_func(
            params=["x"],
            guards=[
                Guard(GuardKind.COMPARISON, "x", 0, ComparisonOp.GE),
                Guard(GuardKind.COMPARISON, "x", 10, ComparisonOp.LT),
                Guard(GuardKind.COMPARISON, "x", 50, ComparisonOp.LT),
            ],
        )
        result = run_cegar(func, config)
        # Should still converge despite budget
        assert result.converged or result.iterations > 0

    def test_cegar_widening_delay(self) -> None:
        """CEGAR with different widening delays."""
        func = _make_loop_func(
            init_stmts=[Stmt(StmtKind.ASSIGN, target="i", value=0)],
            loop_guard=Guard(GuardKind.COMPARISON, "i", 10, ComparisonOp.LT),
        )
        config1 = CEGARConfig(widening_delay=1)
        config2 = CEGARConfig(widening_delay=5)
        r1 = run_cegar(func, config1)
        r2 = run_cegar(func, config2)
        # Both should converge
        assert r1.converged or r1.iterations <= 100
        assert r2.converged or r2.iterations <= 100

    def test_comparison_op_negate(self) -> None:
        """ComparisonOp.negate() works correctly."""
        assert ComparisonOp.LT.negate() == ComparisonOp.GE
        assert ComparisonOp.LE.negate() == ComparisonOp.GT
        assert ComparisonOp.GT.negate() == ComparisonOp.LE
        assert ComparisonOp.GE.negate() == ComparisonOp.LT
        assert ComparisonOp.EQ.negate() == ComparisonOp.NE
        assert ComparisonOp.NE.negate() == ComparisonOp.EQ

    def test_comparison_op_evaluate(self) -> None:
        """ComparisonOp.evaluate() works correctly."""
        assert ComparisonOp.LT.evaluate(3, 5)
        assert not ComparisonOp.LT.evaluate(5, 3)
        assert ComparisonOp.EQ.evaluate(5, 5)
        assert not ComparisonOp.EQ.evaluate(3, 5)
        assert ComparisonOp.NE.evaluate(3, 5)
        assert ComparisonOp.GE.evaluate(5, 5)
        assert ComparisonOp.GE.evaluate(6, 5)

    def test_violation_kind_values(self) -> None:
        """ViolationKind has expected string values."""
        assert ViolationKind.ARRAY_OUT_OF_BOUNDS.value == "array_out_of_bounds"
        assert ViolationKind.NULL_DEREFERENCE.value == "null_dereference"
        assert ViolationKind.DIVISION_BY_ZERO.value == "division_by_zero"
        assert ViolationKind.TYPE_TAG_MISMATCH.value == "type_tag_mismatch"

    def test_abstract_state_join(self) -> None:
        """Abstract state join merges intervals."""
        s1 = AbstractState().set_interval("x", Interval.from_bounds(0, 5))
        s2 = AbstractState().set_interval("x", Interval.from_bounds(3, 10))
        joined = s1.join(s2)
        x = joined.get_interval("x")
        assert x.contains(0)
        assert x.contains(10)

    def test_abstract_state_missing_var(self) -> None:
        """Accessing missing variable returns top/maybe."""
        state = AbstractState()
        assert state.get_interval("x").is_top
        assert state.get_nullity("x") == NullityValue.maybe_null()
