"""
Predicate management for CEGAR abstraction refinement.

Manages predicates used to define abstract states, with biology-specific
templates (Hill thresholds, steady-state, phase, ratio predicates) and
efficient redundancy elimination via SMT entailment checking.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from bioprover.encoding.expression import (
    Const,
    Div,
    ExprNode,
    Ge,
    Gt,
    Le,
    Lt,
    Eq,
    Var,
    And,
    Or,
    Not,
    Neg,
    Add,
    Mul,
    HillAct,
    HillRep,
    Interval,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------


class PredicateOrigin(Enum):
    """How a predicate was created."""

    INITIAL = auto()
    HILL_THRESHOLD = auto()
    NULLCLINE = auto()
    EIGENSPACE = auto()
    INTERPOLATION = auto()
    SIMULATION_GUIDED = auto()
    MONOTONICITY = auto()
    TIME_SCALE = auto()
    AI_HEURISTIC = auto()
    USER_SUPPLIED = auto()


@dataclass(frozen=True)
class Predicate:
    """A Boolean predicate over continuous state variables.

    Wraps an ``ExprNode`` that evaluates to a Boolean (typically a
    comparison like ``x > K``).  Predicates carry metadata about their
    origin and the CEGAR iteration in which they were introduced.
    """

    expr: ExprNode
    name: str = ""
    origin: PredicateOrigin = PredicateOrigin.INITIAL
    iteration_added: int = 0
    info: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def variables(self) -> FrozenSet[str]:
        """Free variables appearing in the predicate expression."""
        return self.expr.free_vars()

    @property
    def digest(self) -> str:
        """Short content hash for deduplication."""
        return hashlib.sha256(self.expr.pretty().encode()).hexdigest()[:12]

    def negation(self) -> Predicate:
        """Return a predicate representing ¬self."""
        return Predicate(
            expr=Not(self.expr),
            name=f"¬{self.name}" if self.name else "",
            origin=self.origin,
            iteration_added=self.iteration_added,
        )

    def evaluate(self, valuation: Dict[str, float]) -> Optional[bool]:
        """Evaluate predicate under a concrete valuation.

        Returns ``None`` when the expression cannot be evaluated (e.g.
        missing variables).
        """
        try:
            return _eval_bool(self.expr, valuation)
        except Exception:
            return None

    def substitute(self, mapping: Dict[str, ExprNode]) -> Predicate:
        """Return a new predicate with variables substituted."""
        return Predicate(
            expr=self.expr.substitute(mapping),
            name=self.name,
            origin=self.origin,
            iteration_added=self.iteration_added,
            info=dict(self.info),
        )

    def __repr__(self) -> str:
        label = self.name or self.expr.pretty()
        return f"Predicate({label})"


# ---------------------------------------------------------------------------
# Predicate set
# ---------------------------------------------------------------------------


class PredicateSet:
    """Collection of predicates with deduplication and entailment checks.

    Maintains an indexed set of predicates for fast lookup by variable,
    origin, and content hash.  Supports optional SMT-based redundancy
    removal when a solver is provided.
    """

    def __init__(self, solver: Optional[Any] = None) -> None:
        self._predicates: Dict[str, Predicate] = {}  # digest -> Predicate
        self._by_variable: Dict[str, Set[str]] = {}  # var -> set of digests
        self._by_origin: Dict[PredicateOrigin, Set[str]] = {}
        self._solver = solver
        self._entailment_cache: Dict[Tuple[str, str], Optional[bool]] = {}
        self._stats = PredicateStatistics()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._predicates)

    def __iter__(self) -> Iterator[Predicate]:
        return iter(self._predicates.values())

    def __contains__(self, pred: Predicate) -> bool:
        return pred.digest in self._predicates

    @property
    def predicates(self) -> List[Predicate]:
        """All predicates in insertion order."""
        return list(self._predicates.values())

    @property
    def statistics(self) -> "PredicateStatistics":
        return self._stats

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, pred: Predicate) -> bool:
        """Add predicate. Returns ``True`` if it was new."""
        digest = pred.digest
        if digest in self._predicates:
            self._stats.duplicates_skipped += 1
            return False

        self._predicates[digest] = pred
        for v in pred.variables:
            self._by_variable.setdefault(v, set()).add(digest)
        self._by_origin.setdefault(pred.origin, set()).add(digest)
        self._stats.total_added += 1
        return True

    def add_all(self, preds: Sequence[Predicate]) -> int:
        """Add multiple predicates. Returns count of new ones."""
        return sum(1 for p in preds if self.add(p))

    def remove(self, pred: Predicate) -> bool:
        """Remove predicate by digest. Returns ``True`` if it existed."""
        digest = pred.digest
        if digest not in self._predicates:
            return False
        removed = self._predicates.pop(digest)
        for v in removed.variables:
            s = self._by_variable.get(v)
            if s:
                s.discard(digest)
        origin_set = self._by_origin.get(removed.origin)
        if origin_set:
            origin_set.discard(digest)
        self._stats.total_removed += 1
        return True

    def clear(self) -> None:
        """Remove all predicates."""
        self._predicates.clear()
        self._by_variable.clear()
        self._by_origin.clear()
        self._entailment_cache.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def by_variable(self, var_name: str) -> List[Predicate]:
        """Return predicates mentioning *var_name*."""
        digests = self._by_variable.get(var_name, set())
        return [self._predicates[d] for d in digests if d in self._predicates]

    def by_origin(self, origin: PredicateOrigin) -> List[Predicate]:
        """Return predicates of a given origin."""
        digests = self._by_origin.get(origin, set())
        return [self._predicates[d] for d in digests if d in self._predicates]

    def by_iteration(self, iteration: int) -> List[Predicate]:
        """Return predicates introduced at a given CEGAR iteration."""
        return [p for p in self._predicates.values() if p.iteration_added == iteration]

    # ------------------------------------------------------------------
    # Entailment / independence
    # ------------------------------------------------------------------

    def entails(self, p: Predicate, q: Predicate) -> Optional[bool]:
        """Check if *p* logically entails *q* via SMT.

        Returns ``True`` if p ⊨ q, ``False`` if not, ``None`` on timeout.
        Requires a solver to be configured.
        """
        cache_key = (p.digest, q.digest)
        if cache_key in self._entailment_cache:
            return self._entailment_cache[cache_key]

        if self._solver is None:
            return None

        self._stats.entailment_queries += 1
        t0 = time.monotonic()

        try:
            self._solver.push()
            # p ∧ ¬q  is UNSAT iff p ⊨ q
            self._solver.assert_formula(p.expr)
            self._solver.assert_formula(Not(q.expr))
            result = self._solver.check_sat()
            self._solver.pop()

            answer = result.is_unsat if result.is_definite else None
        except Exception:
            answer = None
        finally:
            self._stats.entailment_time += time.monotonic() - t0

        self._entailment_cache[cache_key] = answer
        return answer

    def are_independent(self, p: Predicate, q: Predicate) -> Optional[bool]:
        """Check if *p* and *q* are logically independent.

        Independent means neither entails the other nor its negation.
        """
        fwd = self.entails(p, q)
        if fwd is True:
            return False
        bwd = self.entails(q, p)
        if bwd is True:
            return False
        neg_fwd = self.entails(p, q.negation())
        if neg_fwd is True:
            return False
        neg_bwd = self.entails(q, p.negation())
        if neg_bwd is True:
            return False
        if any(r is None for r in (fwd, bwd, neg_fwd, neg_bwd)):
            return None
        return True

    def remove_redundant(self) -> int:
        """Remove predicates entailed by others. Returns count removed."""
        if self._solver is None:
            return 0

        preds = list(self._predicates.values())
        to_remove: Set[str] = set()

        for i, p in enumerate(preds):
            if p.digest in to_remove:
                continue
            for j, q in enumerate(preds):
                if i == j or q.digest in to_remove:
                    continue
                if self.entails(p, q) is True:
                    to_remove.add(q.digest)
                    logger.debug("Predicate %s is redundant (entailed by %s)", q, p)

        for d in to_remove:
            p = self._predicates.get(d)
            if p:
                self.remove(p)

        self._stats.redundant_removed += len(to_remove)
        return len(to_remove)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_by_information_gain(
        self,
        abstract_states: Sequence[Any],
        top_k: int = 10,
    ) -> List[Tuple[Predicate, float]]:
        """Rank predicates by expected information gain.

        A predicate that splits many abstract states roughly in half has
        high information gain.  *abstract_states* should support an
        ``evaluate`` method compatible with predicate evaluation.

        Returns at most *top_k* (predicate, score) pairs, sorted
        descending by score.
        """
        scored: List[Tuple[Predicate, float]] = []

        for pred in self._predicates.values():
            true_count = 0
            false_count = 0
            unknown_count = 0

            for state in abstract_states:
                val = _evaluate_predicate_on_state(pred, state)
                if val is True:
                    true_count += 1
                elif val is False:
                    false_count += 1
                else:
                    unknown_count += 1

            total = true_count + false_count
            if total == 0:
                continue

            # Entropy-based score: max when split is 50/50
            p = true_count / total
            if p == 0.0 or p == 1.0:
                entropy = 0.0
            else:
                import math

                entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

            scored.append((pred, entropy))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "predicates": [
                {
                    "expr": p.expr.pretty(),
                    "name": p.name,
                    "origin": p.origin.name,
                    "iteration": p.iteration_added,
                    "digest": p.digest,
                }
                for p in self._predicates.values()
            ],
            "statistics": self._stats.to_dict(),
        }


# ---------------------------------------------------------------------------
# Predicate templates
# ---------------------------------------------------------------------------


class PredicateTemplate(ABC):
    """Factory for creating predicates from biological model structure."""

    @abstractmethod
    def generate(
        self,
        species_names: List[str],
        parameters: Dict[str, Any],
        iteration: int,
    ) -> List[Predicate]:
        """Generate predicates from the template."""
        ...


class HillThresholdTemplate(PredicateTemplate):
    """Generate predicates at Hill function thresholds.

    For each Hill function ``H(x, K, n)`` in the model, generate
    ``x > K`` and optionally ``x > K/2`` and ``x > 2K``.
    """

    def __init__(self, include_half_threshold: bool = True) -> None:
        self._include_half = include_half_threshold

    def generate(
        self,
        species_names: List[str],
        parameters: Dict[str, Any],
        iteration: int,
    ) -> List[Predicate]:
        preds: List[Predicate] = []
        hill_params = parameters.get("hill_functions", [])

        for hf in hill_params:
            species = hf.get("species", "")
            k_val = hf.get("K", 1.0)
            if species not in species_names:
                continue

            x = Var(species)
            k = Const(k_val)

            preds.append(
                Predicate(
                    expr=Gt(x, k),
                    name=f"{species} > K({k_val})",
                    origin=PredicateOrigin.HILL_THRESHOLD,
                    iteration_added=iteration,
                    info={"K": k_val, "species": species},
                )
            )

            if self._include_half:
                half_k = Const(k_val / 2.0)
                preds.append(
                    Predicate(
                        expr=Gt(x, half_k),
                        name=f"{species} > K/2({k_val / 2.0})",
                        origin=PredicateOrigin.HILL_THRESHOLD,
                        iteration_added=iteration,
                        info={"K": k_val / 2.0, "species": species},
                    )
                )
                double_k = Const(2.0 * k_val)
                preds.append(
                    Predicate(
                        expr=Gt(x, double_k),
                        name=f"{species} > 2K({2.0 * k_val})",
                        origin=PredicateOrigin.HILL_THRESHOLD,
                        iteration_added=iteration,
                        info={"K": 2.0 * k_val, "species": species},
                    )
                )

        return preds


class SteadyStateTemplate(PredicateTemplate):
    """Generate predicates for proximity to steady states.

    Given a list of known or computed steady states, create predicates
    ``|x_i - ss_i| < epsilon`` for each species.
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        self._epsilon = epsilon

    def generate(
        self,
        species_names: List[str],
        parameters: Dict[str, Any],
        iteration: int,
    ) -> List[Predicate]:
        preds: List[Predicate] = []
        steady_states: List[Dict[str, float]] = parameters.get("steady_states", [])

        for ss_idx, ss in enumerate(steady_states):
            for sp in species_names:
                if sp not in ss:
                    continue
                val = ss[sp]
                x = Var(sp)
                lo = Const(val - self._epsilon)
                hi = Const(val + self._epsilon)
                preds.append(
                    Predicate(
                        expr=And(Ge(x, lo), Le(x, hi)),
                        name=f"{sp}≈ss{ss_idx}({val:.3g})",
                        origin=PredicateOrigin.NULLCLINE,
                        iteration_added=iteration,
                        info={"steady_state_index": ss_idx, "value": val},
                    )
                )

        return preds


class PhaseTemplate(PredicateTemplate):
    """Generate predicates for oscillation phase regions.

    Divides concentration range into N equidistant segments to
    distinguish oscillation phases.
    """

    def __init__(self, num_phases: int = 4) -> None:
        self._num_phases = num_phases

    def generate(
        self,
        species_names: List[str],
        parameters: Dict[str, Any],
        iteration: int,
    ) -> List[Predicate]:
        preds: List[Predicate] = []
        bounds: Dict[str, Tuple[float, float]] = parameters.get("bounds", {})

        for sp in species_names:
            if sp not in bounds:
                continue
            lo, hi = bounds[sp]
            step = (hi - lo) / self._num_phases

            for i in range(1, self._num_phases):
                threshold = lo + i * step
                x = Var(sp)
                preds.append(
                    Predicate(
                        expr=Gt(x, Const(threshold)),
                        name=f"{sp}_phase>{i}/{self._num_phases}",
                        origin=PredicateOrigin.INITIAL,
                        iteration_added=iteration,
                        info={"phase_index": i, "threshold": threshold},
                    )
                )

        return preds


class RatioTemplate(PredicateTemplate):
    """Generate ratio predicates ``x_i / x_j > threshold``.

    Useful for capturing relative concentrations in multi-species
    systems (e.g. toggle switches where A/B ratio determines state).
    """

    def __init__(self, thresholds: Sequence[float] = (0.5, 1.0, 2.0)) -> None:
        self._thresholds = list(thresholds)

    def generate(
        self,
        species_names: List[str],
        parameters: Dict[str, Any],
        iteration: int,
    ) -> List[Predicate]:
        preds: List[Predicate] = []
        pairs: List[Tuple[str, str]] = parameters.get(
            "ratio_pairs",
            [(a, b) for i, a in enumerate(species_names) for b in species_names[i + 1:]],
        )

        for sp_a, sp_b in pairs:
            if sp_a not in species_names or sp_b not in species_names:
                continue
            x_a = Var(sp_a)
            x_b = Var(sp_b)
            for thr in self._thresholds:
                # x_a / x_b > thr  ⟺  x_a > thr * x_b  (when x_b > 0)
                preds.append(
                    Predicate(
                        expr=Gt(x_a, Mul(Const(thr), x_b)),
                        name=f"{sp_a}/{sp_b}>{thr}",
                        origin=PredicateOrigin.INITIAL,
                        iteration_added=iteration,
                        info={"species_a": sp_a, "species_b": sp_b, "threshold": thr},
                    )
                )

        return preds


class NullclineTemplate(PredicateTemplate):
    """Generate predicates for nullcline crossings.

    For each species i with RHS f_i(x), generate the predicate
    f_i(x) > 0 (production exceeds degradation).
    """

    def generate(
        self,
        species_names: List[str],
        parameters: Dict[str, Any],
        iteration: int,
    ) -> List[Predicate]:
        preds: List[Predicate] = []
        rhs_exprs: Dict[str, ExprNode] = parameters.get("rhs_expressions", {})

        for sp in species_names:
            rhs = rhs_exprs.get(sp)
            if rhs is None:
                continue
            preds.append(
                Predicate(
                    expr=Gt(rhs, Const(0.0)),
                    name=f"d{sp}/dt>0",
                    origin=PredicateOrigin.NULLCLINE,
                    iteration_added=iteration,
                    info={"species": sp},
                )
            )

        return preds


# ---------------------------------------------------------------------------
# Predicate cache
# ---------------------------------------------------------------------------


class PredicateCache:
    """Cross-iteration cache for predicate evaluations.

    Stores Boolean valuation of each predicate at sampled concrete
    states to avoid repeated SMT queries.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Optional[bool]]] = {}
        self._hits: int = 0
        self._misses: int = 0

    def lookup(
        self,
        pred: Predicate,
        state_key: str,
    ) -> Optional[bool]:
        """Lookup cached evaluation. Returns ``None`` on miss."""
        entry = self._cache.get(pred.digest)
        if entry is not None and state_key in entry:
            self._hits += 1
            return entry[state_key]
        self._misses += 1
        return None

    def store(
        self,
        pred: Predicate,
        state_key: str,
        value: bool,
    ) -> None:
        """Store a predicate evaluation result."""
        self._cache.setdefault(pred.digest, {})[state_key] = value

    def invalidate(self, pred: Predicate) -> None:
        """Remove all cached entries for a predicate."""
        self._cache.pop(pred.digest, None)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return sum(len(v) for v in self._cache.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class PredicateStatistics:
    """Tracks predicate management statistics."""

    total_added: int = 0
    total_removed: int = 0
    duplicates_skipped: int = 0
    redundant_removed: int = 0
    entailment_queries: int = 0
    entailment_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_added": self.total_added,
            "total_removed": self.total_removed,
            "duplicates_skipped": self.duplicates_skipped,
            "redundant_removed": self.redundant_removed,
            "entailment_queries": self.entailment_queries,
            "entailment_time_s": round(self.entailment_time, 3),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_bool(expr: ExprNode, val: Dict[str, float]) -> Optional[bool]:
    """Recursively evaluate a Boolean expression under a valuation."""
    v = _eval_numeric(expr, val)
    if isinstance(v, bool):
        return v
    return None


def _eval_numeric(expr: ExprNode, val: Dict[str, float]) -> Any:
    """Best-effort numeric evaluation of an ExprNode."""
    import math

    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        if expr.name in val:
            return val[expr.name]
        raise KeyError(expr.name)

    kids = expr.children()

    if isinstance(expr, Add):
        return _eval_numeric(kids[0], val) + _eval_numeric(kids[1], val)
    if isinstance(expr, Mul):
        return _eval_numeric(kids[0], val) * _eval_numeric(kids[1], val)
    if isinstance(expr, Div):
        denom = _eval_numeric(kids[1], val)
        if denom == 0:
            return float("inf")
        return _eval_numeric(kids[0], val) / denom
    if isinstance(expr, Gt):
        return _eval_numeric(kids[0], val) > _eval_numeric(kids[1], val)
    if isinstance(expr, Ge):
        return _eval_numeric(kids[0], val) >= _eval_numeric(kids[1], val)
    if isinstance(expr, Lt):
        return _eval_numeric(kids[0], val) < _eval_numeric(kids[1], val)
    if isinstance(expr, Le):
        return _eval_numeric(kids[0], val) <= _eval_numeric(kids[1], val)
    if isinstance(expr, Eq):
        return _eval_numeric(kids[0], val) == _eval_numeric(kids[1], val)
    if isinstance(expr, And):
        return all(_eval_numeric(c, val) for c in kids)
    if isinstance(expr, Or):
        return any(_eval_numeric(c, val) for c in kids)
    if isinstance(expr, Not):
        return not _eval_numeric(kids[0], val)
    if isinstance(expr, Neg):
        return -_eval_numeric(kids[0], val)
    raise TypeError(f"Cannot evaluate expression type {type(expr).__name__}")


def _evaluate_predicate_on_state(
    pred: Predicate,
    state: Any,
) -> Optional[bool]:
    """Evaluate a predicate on an abstract state.

    Tries ``state.contains_point`` at interval midpoints.  Returns
    ``None`` when evaluation is inconclusive.
    """
    try:
        midpoint = state.midpoint() if hasattr(state, "midpoint") else {}
        return pred.evaluate(midpoint)
    except Exception:
        return None
