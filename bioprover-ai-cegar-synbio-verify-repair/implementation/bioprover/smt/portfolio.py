"""Portfolio solver for BioProver.

Runs multiple SMT solver configurations in parallel and returns the
first definitive result, with adaptive strategy selection using a
UCB1 multi-armed bandit, formula feature extraction, and timeout
budget allocation.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bioprover.smt.solver_base import (
    AbstractSMTSolver,
    Model,
    SMTResult,
    SolverStatistics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

class StrategyKind(Enum):
    """Solver strategy identifiers."""

    Z3_DEFAULT = auto()
    Z3_NLSAT = auto()
    Z3_QFNRA = auto()
    DREAL_FINE = auto()
    DREAL_COARSE = auto()
    DREAL_MEDIUM = auto()


@dataclass
class StrategyConfig:
    """Configuration for a single solver strategy."""

    kind: StrategyKind
    name: str
    factory: Callable[[], AbstractSMTSolver]
    weight: float = 1.0
    enabled: bool = True

    def __repr__(self) -> str:
        return f"StrategyConfig({self.name}, w={self.weight:.2f})"


# ---------------------------------------------------------------------------
# Formula feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FormulaFeatures:
    """Lightweight structural features of an SMT formula.

    Used by the strategy selector to pick the best solving approach.
    """

    num_variables: int = 0
    num_constraints: int = 0
    max_degree: int = 1
    has_transcendental: bool = False
    transcendental_count: int = 0
    num_boolean_connectives: int = 0
    num_quantifiers: int = 0
    estimated_difficulty: float = 0.0

    def compute_difficulty(self) -> float:
        """Heuristic difficulty score in ``[0, 1]``."""
        score = 0.0
        score += min(self.num_variables / 50.0, 1.0) * 0.2
        score += min(self.num_constraints / 100.0, 1.0) * 0.15
        score += min(self.max_degree / 10.0, 1.0) * 0.25
        if self.has_transcendental:
            score += 0.2
        score += min(self.transcendental_count / 20.0, 1.0) * 0.1
        score += min(self.num_quantifiers / 5.0, 1.0) * 0.1
        self.estimated_difficulty = min(score, 1.0)
        return self.estimated_difficulty


def extract_formula_features(expr: Any) -> FormulaFeatures:
    """Extract structural features from an expression.

    Supports Z3 expressions and dict-based expression trees.
    """
    features = FormulaFeatures()
    _vars: set[str] = set()
    _extract_features_recurse(expr, features, _vars, set())
    features.num_variables = len(_vars)
    features.compute_difficulty()
    return features


def _extract_features_recurse(
    expr: Any,
    features: FormulaFeatures,
    variables: set[str],
    visited: set[int],
) -> None:
    eid = id(expr)
    if eid in visited:
        return
    visited.add(eid)

    if isinstance(expr, str):
        variables.add(expr)
        return

    if isinstance(expr, (int, float)):
        return

    if isinstance(expr, dict):
        op = expr.get("op", "")
        args = expr.get("args", [])
        features.num_constraints += 1

        if op in ("sin", "cos", "exp", "log", "tan", "asin", "acos", "atan"):
            features.has_transcendental = True
            features.transcendental_count += 1
        if op in ("^", "pow") and len(args) >= 2:
            if isinstance(args[1], (int, float)):
                features.max_degree = max(features.max_degree, int(args[1]))
        if op in ("and", "or", "not", "=>"):
            features.num_boolean_connectives += 1
        if op in ("forall", "exists"):
            features.num_quantifiers += 1

        for a in args:
            _extract_features_recurse(a, features, variables, visited)
        return

    # Z3 expressions.
    try:
        import z3 as _z3

        if isinstance(expr, _z3.ExprRef):
            if _z3.is_const(expr) and expr.decl().kind() == _z3.Z3_OP_UNINTERPRETED:
                variables.add(str(expr))
                return
            features.num_constraints += 1
            for child in expr.children():
                _extract_features_recurse(child, features, variables, visited)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Strategy performance history (UCB1)
# ---------------------------------------------------------------------------

@dataclass
class _StrategyRecord:
    """Per-strategy statistics for UCB1 selection."""

    total_reward: float = 0.0
    num_plays: int = 0
    total_time: float = 0.0
    successes: int = 0
    failures: int = 0

    @property
    def avg_reward(self) -> float:
        if self.num_plays == 0:
            return 0.0
        return self.total_reward / self.num_plays


class StrategySelector:
    """UCB1 multi-armed bandit for adaptive strategy selection.

    Each strategy is an arm.  Rewards are based on whether the strategy
    produced a definitive result and how quickly.
    """

    def __init__(self, exploration: float = 1.41) -> None:
        self._exploration = exploration
        self._records: Dict[StrategyKind, _StrategyRecord] = {}
        self._total_plays: int = 0

    def _ensure_record(self, kind: StrategyKind) -> _StrategyRecord:
        if kind not in self._records:
            self._records[kind] = _StrategyRecord()
        return self._records[kind]

    def select(self, candidates: List[StrategyConfig]) -> List[StrategyConfig]:
        """Rank *candidates* by UCB1 score and return sorted list."""
        if self._total_plays == 0:
            return list(candidates)

        scored: List[Tuple[float, StrategyConfig]] = []
        for sc in candidates:
            rec = self._ensure_record(sc.kind)
            if rec.num_plays == 0:
                ucb = float("inf")
            else:
                exploit = rec.avg_reward
                explore = self._exploration * math.sqrt(
                    math.log(self._total_plays) / rec.num_plays
                )
                ucb = exploit + explore
            scored.append((ucb, sc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [sc for _, sc in scored]

    def record_outcome(
        self,
        kind: StrategyKind,
        result: SMTResult,
        elapsed: float,
        timeout: float,
    ) -> None:
        """Record the outcome of running a strategy."""
        rec = self._ensure_record(kind)
        rec.num_plays += 1
        rec.total_time += elapsed
        self._total_plays += 1

        if result.is_definite:
            # Reward: 1 minus fraction of timeout used (faster = higher).
            reward = max(0.0, 1.0 - elapsed / timeout)
            rec.total_reward += reward
            rec.successes += 1
        else:
            rec.failures += 1

    def statistics(self) -> Dict[str, Any]:
        return {
            kind.name: {
                "plays": rec.num_plays,
                "avg_reward": round(rec.avg_reward, 4),
                "successes": rec.successes,
                "failures": rec.failures,
            }
            for kind, rec in self._records.items()
        }


# ---------------------------------------------------------------------------
# Timeout budget allocation
# ---------------------------------------------------------------------------

def allocate_budgets(
    strategies: List[StrategyConfig],
    total_timeout: float,
    min_budget: float = 1.0,
) -> Dict[StrategyKind, float]:
    """Allocate timeout budgets across strategies proportional to weight."""
    total_weight = sum(s.weight for s in strategies if s.enabled)
    if total_weight <= 0:
        equal = total_timeout / max(len(strategies), 1)
        return {s.kind: equal for s in strategies}

    budgets: Dict[StrategyKind, float] = {}
    for s in strategies:
        if s.enabled:
            raw = (s.weight / total_weight) * total_timeout
            budgets[s.kind] = max(raw, min_budget)
    return budgets


# ---------------------------------------------------------------------------
# PortfolioSolver
# ---------------------------------------------------------------------------

class ConflictResolutionPolicy(Enum):
    """Policy for resolving conflicts between portfolio solvers.

    When dReal returns δ-SAT and Z3 returns UNSAT (or vice versa),
    these policies determine which result to trust.
    """
    PREFER_UNSAT = auto()       # UNSAT wins: conservative, may miss solutions
    PREFER_SAT = auto()         # SAT wins: optimistic, may include δ-artifacts
    DELTA_AWARE = auto()        # Use δ to decide: if δ < threshold, trust SAT
    CONSENSUS_REQUIRED = auto() # Require agreement, else UNKNOWN


@dataclass
class ConflictRecord:
    """Records a conflict between solver strategies."""
    strategy_a: str
    result_a: str
    strategy_b: str
    result_b: str
    resolution: str
    delta_used: Optional[float] = None
    timestamp: float = 0.0


class PortfolioSolver(AbstractSMTSolver):
    """Run multiple solver strategies in parallel, return first result.

    Parameters
    ----------
    strategies:
        List of :class:`StrategyConfig` to run.  If ``None``, uses
        default strategies (requires ``z3`` to be installed).
    max_workers:
        Maximum number of parallel solver threads.
    total_timeout:
        Total timeout budget across all strategies.
    adaptive:
        If ``True``, use UCB1 to adaptively order strategies.
    conflict_policy:
        Policy for resolving conflicts between solvers.
    conflict_delta_threshold:
        When using DELTA_AWARE policy, δ values below this threshold
        make δ-SAT results trustworthy.
    """

    def __init__(
        self,
        strategies: Optional[List[StrategyConfig]] = None,
        max_workers: int = 4,
        total_timeout: float = 120.0,
        adaptive: bool = True,
        conflict_policy: ConflictResolutionPolicy = ConflictResolutionPolicy.DELTA_AWARE,
        conflict_delta_threshold: float = 1e-3,
    ) -> None:
        super().__init__(name="portfolio")
        self._strategies = strategies or self._default_strategies()
        self._max_workers = max_workers
        self._default_timeout = total_timeout
        self._adaptive = adaptive
        self._selector = StrategySelector()
        self._conflict_policy = conflict_policy
        self._conflict_delta_threshold = conflict_delta_threshold
        self._conflicts: List[ConflictRecord] = []

        self._formulas: List[Any] = []
        self._last_result: Optional[SMTResult] = None
        self._last_model: Optional[Model] = None
        self._context_stack: List[List[Any]] = []

    # -- default strategies -------------------------------------------------

    @staticmethod
    def _default_strategies() -> List[StrategyConfig]:
        from bioprover.smt.z3_interface import TacticPreset, Z3Solver

        strategies = [
            StrategyConfig(
                kind=StrategyKind.Z3_DEFAULT,
                name="z3-default",
                factory=lambda: Z3Solver(tactic=TacticPreset.DEFAULT),
                weight=1.0,
            ),
            StrategyConfig(
                kind=StrategyKind.Z3_NLSAT,
                name="z3-nlsat",
                factory=lambda: Z3Solver(tactic=TacticPreset.NLSAT),
                weight=1.2,
            ),
            StrategyConfig(
                kind=StrategyKind.Z3_QFNRA,
                name="z3-qfnra",
                factory=lambda: Z3Solver(tactic=TacticPreset.QFNRA),
                weight=1.1,
            ),
        ]

        # Add dReal strategies only if available.
        try:
            from bioprover.smt.dreal_interface import DRealSolver

            strategies.extend([
                StrategyConfig(
                    kind=StrategyKind.DREAL_FINE,
                    name="dreal-fine",
                    factory=lambda: DRealSolver(delta=1e-4, use_icp_fallback=True),
                    weight=0.8,
                ),
                StrategyConfig(
                    kind=StrategyKind.DREAL_COARSE,
                    name="dreal-coarse",
                    factory=lambda: DRealSolver(delta=1e-2, use_icp_fallback=True),
                    weight=0.9,
                ),
                StrategyConfig(
                    kind=StrategyKind.DREAL_MEDIUM,
                    name="dreal-medium",
                    factory=lambda: DRealSolver(delta=1e-3, use_icp_fallback=True),
                    weight=1.0,
                ),
            ])
        except Exception:
            pass

        return strategies

    # -- formula recommendation ---------------------------------------------

    def recommend_strategy(self, expr: Any) -> StrategyConfig:
        """Recommend a strategy based on formula features."""
        features = extract_formula_features(expr)
        enabled = [s for s in self._strategies if s.enabled]

        # Simple heuristic: transcendentals → dReal, polynomial → Z3.
        if features.has_transcendental:
            dreal = [s for s in enabled if "dreal" in s.name]
            if dreal:
                return dreal[0]

        if features.max_degree <= 2 and not features.has_transcendental:
            z3_default = [s for s in enabled if s.kind == StrategyKind.Z3_DEFAULT]
            if z3_default:
                return z3_default[0]

        if features.max_degree > 2:
            nlsat = [s for s in enabled if s.kind == StrategyKind.Z3_NLSAT]
            if nlsat:
                return nlsat[0]

        return enabled[0] if enabled else self._strategies[0]

    # -- AbstractSMTSolver implementation -----------------------------------

    def assert_formula(self, expr: Any) -> None:
        self._formulas.append(expr)

    def check_sat(self, timeout: Optional[float] = None) -> SMTResult:
        effective_timeout = timeout or self._default_timeout
        enabled = [s for s in self._strategies if s.enabled]

        if self._adaptive:
            enabled = self._selector.select(enabled)

        budgets = allocate_budgets(enabled, effective_timeout)

        result, model, winning_strategy = self._race(enabled, budgets)
        self._last_result = result
        self._last_model = model

        self.stats.record_query(result, 0.0)
        return result

    def get_model(self) -> Model:
        if self._last_model is None:
            raise RuntimeError("No model available")
        return self._last_model

    def push(self) -> None:
        self._context_stack.append(list(self._formulas))
        self.stats.record_push()

    def pop(self) -> None:
        if not self._context_stack:
            raise RuntimeError("Cannot pop: context stack is empty")
        self._formulas = self._context_stack.pop()
        self.stats.record_pop()

    def check_sat_assuming(
        self,
        assumptions: Sequence[Any],
        timeout: Optional[float] = None,
    ) -> SMTResult:
        self.push()
        try:
            for a in assumptions:
                self.assert_formula(a)
            return self.check_sat(timeout)
        finally:
            self.pop()

    def get_unsat_core(self) -> List[Any]:
        return []

    def reset(self) -> None:
        self._formulas.clear()
        self._context_stack.clear()
        self._last_result = None
        self._last_model = None

    # -- parallel race ------------------------------------------------------

    def _race(
        self,
        strategies: List[StrategyConfig],
        budgets: Dict[StrategyKind, float],
    ) -> Tuple[SMTResult, Optional[Model], Optional[StrategyConfig]]:
        """Run strategies sequentially with early termination and conflict resolution.

        When multiple solvers disagree (e.g., dReal δ-SAT vs Z3 UNSAT), the
        conflict resolution policy determines the final result.
        """
        if not strategies:
            return SMTResult.UNKNOWN, None, None

        results: List[Tuple[StrategyConfig, SMTResult, Optional[Model], float]] = []

        for sc in strategies:
            budget = budgets.get(sc.kind, self._default_timeout)
            try:
                result, model, elapsed = self._run_strategy(sc, budget)
            except Exception as exc:
                logger.warning("Strategy %s raised: %s", sc.name, exc)
                result = SMTResult.UNKNOWN
                model = None
                elapsed = 0.0

            if self._adaptive:
                self._selector.record_outcome(sc.kind, result, elapsed, budget)

            results.append((sc, result, model, elapsed))

            if result.is_definite:
                # Check for conflicts with previous results
                conflict = self._check_conflicts(results)
                if conflict is not None:
                    self._conflicts.append(conflict)
                    resolved_result, resolved_model, resolved_sc = (
                        self._resolve_conflict(results)
                    )
                    logger.warning(
                        "Portfolio conflict: %s(%s) vs %s(%s) → resolved as %s",
                        conflict.strategy_a, conflict.result_a,
                        conflict.strategy_b, conflict.result_b,
                        conflict.resolution,
                    )
                    return resolved_result, resolved_model, resolved_sc

                logger.info("Portfolio winner: %s → %s", sc.name, result)
                return result, model, sc

        # No definitive result – return best guess.
        for sc, res, mdl, _ in results:
            if res == SMTResult.DELTA_SAT:
                return res, mdl, sc

        return SMTResult.UNKNOWN, None, None

    def _check_conflicts(
        self,
        results: List[Tuple[StrategyConfig, SMTResult, Optional[Model], float]],
    ) -> Optional[ConflictRecord]:
        """Check if the latest result conflicts with a previous one."""
        if len(results) < 2:
            return None

        latest_sc, latest_res, _, _ = results[-1]
        for prev_sc, prev_res, _, _ in results[:-1]:
            if prev_res == SMTResult.UNKNOWN:
                continue
            # Conflict: one SAT/δ-SAT, the other UNSAT
            is_conflict = (
                (latest_res.is_sat and prev_res == SMTResult.UNSAT) or
                (latest_res == SMTResult.UNSAT and prev_res.is_sat) or
                (latest_res == SMTResult.DELTA_SAT and prev_res == SMTResult.UNSAT) or
                (latest_res == SMTResult.UNSAT and prev_res == SMTResult.DELTA_SAT)
            )
            if is_conflict:
                return ConflictRecord(
                    strategy_a=prev_sc.name,
                    result_a=str(prev_res),
                    strategy_b=latest_sc.name,
                    result_b=str(latest_res),
                    resolution="pending",
                    delta_used=self._get_strategy_delta(latest_sc) or self._get_strategy_delta(prev_sc),
                    timestamp=time.time(),
                )
        return None

    def _resolve_conflict(
        self,
        results: List[Tuple[StrategyConfig, SMTResult, Optional[Model], float]],
    ) -> Tuple[SMTResult, Optional[Model], Optional[StrategyConfig]]:
        """Resolve a conflict between solver strategies using the configured policy."""
        unsat_entries = [(sc, r, m) for sc, r, m, _ in results if r == SMTResult.UNSAT]
        sat_entries = [(sc, r, m) for sc, r, m, _ in results
                       if r.is_sat or r == SMTResult.DELTA_SAT]

        policy = self._conflict_policy

        if policy == ConflictResolutionPolicy.PREFER_UNSAT:
            if unsat_entries:
                sc, r, m = unsat_entries[0]
                self._conflicts[-1].resolution = f"PREFER_UNSAT→{r}"
                return r, m, sc

        elif policy == ConflictResolutionPolicy.PREFER_SAT:
            if sat_entries:
                sc, r, m = sat_entries[0]
                self._conflicts[-1].resolution = f"PREFER_SAT→{r}"
                return r, m, sc

        elif policy == ConflictResolutionPolicy.DELTA_AWARE:
            # Trust SAT only if δ is small enough
            delta = self._conflicts[-1].delta_used if self._conflicts else None
            if delta is not None and delta <= self._conflict_delta_threshold:
                if sat_entries:
                    sc, r, m = sat_entries[0]
                    self._conflicts[-1].resolution = (
                        f"DELTA_AWARE(δ={delta:.2e}≤{self._conflict_delta_threshold:.2e})→{r}"
                    )
                    return r, m, sc
            # δ too large or unknown: prefer UNSAT (conservative)
            if unsat_entries:
                sc, r, m = unsat_entries[0]
                self._conflicts[-1].resolution = (
                    f"DELTA_AWARE(δ={delta}>{self._conflict_delta_threshold:.2e})→UNSAT"
                )
                return r, m, sc

        elif policy == ConflictResolutionPolicy.CONSENSUS_REQUIRED:
            self._conflicts[-1].resolution = "CONSENSUS_REQUIRED→UNKNOWN"
            return SMTResult.UNKNOWN, None, None

        # Fallback
        if self._conflicts:
            self._conflicts[-1].resolution = "fallback→UNKNOWN"
        return SMTResult.UNKNOWN, None, None

    @staticmethod
    def _get_strategy_delta(sc: StrategyConfig) -> Optional[float]:
        """Extract the δ parameter from a strategy config if it's a dReal strategy."""
        if "dreal" in sc.name.lower():
            if "fine" in sc.name:
                return 1e-4
            elif "coarse" in sc.name:
                return 1e-2
            elif "medium" in sc.name:
                return 1e-3
        return None

    def _run_strategy(
        self,
        sc: StrategyConfig,
        budget: float,
    ) -> Tuple[SMTResult, Optional[Model], float]:
        """Run a single strategy with the given budget."""
        t0 = time.perf_counter()
        solver = sc.factory()
        for f in self._formulas:
            solver.assert_formula(f)
        result = solver.check_sat(budget)
        elapsed = time.perf_counter() - t0

        model = None
        if result.is_sat:
            try:
                model = solver.get_model()
            except Exception:
                pass

        return result, model, elapsed

    # -- statistics ---------------------------------------------------------

    def strategy_statistics(self) -> Dict[str, Any]:
        stats = self._selector.statistics()
        stats["conflicts"] = [
            {
                "strategy_a": c.strategy_a,
                "result_a": c.result_a,
                "strategy_b": c.strategy_b,
                "result_b": c.result_b,
                "resolution": c.resolution,
            }
            for c in self._conflicts
        ]
        stats["conflict_policy"] = self._conflict_policy.name
        return stats
