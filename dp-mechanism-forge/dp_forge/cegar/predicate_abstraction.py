"""
Predicate abstraction for CEGAR-based DP verification.

Implements automatic predicate discovery, Cartesian and Boolean predicate
abstraction, predicate evaluation, and refinement based on spurious
counterexamples.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    AbstractDomainType,
    AbstractValue,
    AdjacencyRelation,
    Formula,
    Predicate,
    PrivacyBudget,
)
from dp_forge.cegar import (
    AbstractCounterexample,
    AbstractState,
    AbstractionType,
    RefinementResult,
    RefinementStrategy,
)


# ---------------------------------------------------------------------------
# Predicate evaluation
# ---------------------------------------------------------------------------


class PredicateEvaluator:
    """Evaluates predicates over concrete mechanism states.

    Supports linear arithmetic predicates of the form a·x ≤ b,
    ratio predicates for privacy bounds, and Boolean combinations.
    """

    def __init__(self, tolerance: float = 1e-9) -> None:
        """Initialize evaluator.

        Args:
            tolerance: Numerical tolerance for floating-point comparisons.
        """
        self._tolerance = tolerance

    def evaluate(
        self,
        predicate: Predicate,
        state: npt.NDArray[np.float64],
    ) -> bool:
        """Evaluate a predicate at a concrete state.

        Parses the predicate formula and evaluates it over the state vector.

        Args:
            predicate: The predicate to evaluate.
            state: Concrete state vector.

        Returns:
            True if the predicate holds at the given state.
        """
        formula = predicate.formula
        metadata = formula.metadata

        if "type" in metadata:
            pred_type = metadata["type"]
            if pred_type == "linear_le":
                # a·x <= b
                coeffs = np.asarray(metadata["coeffs"], dtype=np.float64)
                rhs = float(metadata["rhs"])
                return bool(np.dot(coeffs, state) <= rhs + self._tolerance)
            elif pred_type == "ratio_le":
                # x[i] / x[j] <= bound
                i, j = int(metadata["i"]), int(metadata["j"])
                bound = float(metadata["bound"])
                denom = max(abs(state[j]), 1e-300)
                return bool(state[i] / denom <= bound + self._tolerance)
            elif pred_type == "log_ratio_le":
                # ln(x[i] / x[j]) <= bound
                i, j = int(metadata["i"]), int(metadata["j"])
                bound = float(metadata["bound"])
                eps = 1e-300
                ratio = np.log(max(state[i], eps) / max(state[j], eps))
                return bool(ratio <= bound + self._tolerance)
            elif pred_type == "ge_zero":
                idx = int(metadata["index"])
                return bool(state[idx] >= -self._tolerance)
            elif pred_type == "le_one":
                idx = int(metadata["index"])
                return bool(state[idx] <= 1.0 + self._tolerance)

        # Default: try parsing simple expression
        return self._evaluate_expression(formula.expr, state)

    def _evaluate_expression(
        self,
        expr: str,
        state: npt.NDArray[np.float64],
    ) -> bool:
        """Evaluate a simple expression string over a state vector.

        Supports expressions like 'x0 <= 0.5', 'x1 - x0 >= 0'.

        Args:
            expr: Expression string.
            state: Concrete state vector.

        Returns:
            Evaluation result.
        """
        # Create variable bindings
        env: Dict[str, float] = {}
        for i in range(len(state)):
            env[f"x{i}"] = float(state[i])
            env[f"x_{i}"] = float(state[i])

        try:
            # Simple safe evaluation for linear predicates
            if "<=" in expr:
                parts = expr.split("<=")
                lhs = self._eval_linear(parts[0].strip(), env)
                rhs = self._eval_linear(parts[1].strip(), env)
                return bool(lhs <= rhs + self._tolerance)
            elif ">=" in expr:
                parts = expr.split(">=")
                lhs = self._eval_linear(parts[0].strip(), env)
                rhs = self._eval_linear(parts[1].strip(), env)
                return bool(lhs >= rhs - self._tolerance)
            elif ">" in expr:
                parts = expr.split(">")
                lhs = self._eval_linear(parts[0].strip(), env)
                rhs = self._eval_linear(parts[1].strip(), env)
                return bool(lhs > rhs - self._tolerance)
            elif "<" in expr:
                parts = expr.split("<")
                lhs = self._eval_linear(parts[0].strip(), env)
                rhs = self._eval_linear(parts[1].strip(), env)
                return bool(lhs < rhs + self._tolerance)
        except (ValueError, KeyError):
            pass
        return True  # Conservative: assume predicate holds if unparseable

    def _eval_linear(self, term: str, env: Dict[str, float]) -> float:
        """Evaluate a linear arithmetic term.

        Args:
            term: String like '2*x0 + x1' or '0.5'.
            env: Variable bindings.

        Returns:
            Numeric value.
        """
        term = term.strip()
        if term in env:
            return env[term]
        try:
            return float(term)
        except ValueError:
            pass
        # Handle simple addition/subtraction
        result = 0.0
        current_sign = 1.0
        for token in term.replace("-", "+-").split("+"):
            token = token.strip()
            if not token:
                continue
            if token.startswith("-"):
                current_sign = -1.0
                token = token[1:].strip()
            else:
                current_sign = 1.0
            if "*" in token:
                parts = token.split("*")
                coeff = float(parts[0].strip())
                var = parts[1].strip()
                result += current_sign * coeff * env.get(var, 0.0)
            elif token in env:
                result += current_sign * env[token]
            else:
                result += current_sign * float(token)
        return result

    def evaluate_all(
        self,
        predicates: Sequence[Predicate],
        state: npt.NDArray[np.float64],
    ) -> FrozenSet[str]:
        """Evaluate all predicates and return the set of satisfied ones.

        Args:
            predicates: List of predicates.
            state: Concrete state vector.

        Returns:
            Frozenset of names of satisfied predicates.
        """
        satisfied: Set[str] = set()
        for pred in predicates:
            if self.evaluate(pred, state):
                satisfied.add(pred.name)
        return frozenset(satisfied)


# ---------------------------------------------------------------------------
# Predicate discovery
# ---------------------------------------------------------------------------


class PredicateDiscovery:
    """Automatic predicate generation from counterexamples.

    Discovers predicates useful for CEGAR refinement by analyzing
    counterexample traces and mechanism structure.
    """

    def __init__(
        self,
        evaluator: Optional[PredicateEvaluator] = None,
    ) -> None:
        """Initialize predicate discovery.

        Args:
            evaluator: Predicate evaluator instance.
        """
        self._evaluator = evaluator or PredicateEvaluator()
        self._discovered: List[Predicate] = []
        self._predicate_counter = 0

    @property
    def discovered_predicates(self) -> List[Predicate]:
        """All predicates discovered so far."""
        return list(self._discovered)

    def _next_id(self) -> int:
        """Generate a unique predicate ID."""
        self._predicate_counter += 1
        return self._predicate_counter

    def discover_from_counterexample(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> List[Predicate]:
        """Discover predicates from a spurious counterexample.

        Analyzes the counterexample to generate predicates that
        distinguish the spurious trace from genuine violations.

        Args:
            counterexample: The spurious counterexample trace.
            mechanism: Mechanism probability matrix.
            budget: Target privacy budget.

        Returns:
            List of newly discovered predicates.
        """
        new_preds: List[Predicate] = []

        i, ip = counterexample.violating_pair
        n, k = mechanism.shape

        # Generate ratio predicates for the violating pair
        for j in range(k):
            p_i = mechanism[i, j]
            p_ip = mechanism[ip, j]
            if p_ip > 1e-300:
                ratio = p_i / p_ip
                if ratio > 0:
                    log_ratio = float(np.log(max(ratio, 1e-300)))
                    pred = self._make_log_ratio_predicate(i, ip, j, log_ratio, budget.epsilon)
                    if pred is not None:
                        new_preds.append(pred)

        # Generate bound predicates for probabilities
        for j in range(min(k, 20)):
            for idx in [i, ip]:
                val = mechanism[idx, j]
                if 0 < val < 1:
                    pred = self._make_bound_predicate(idx * k + j, val)
                    new_preds.append(pred)

        # Deduplicate against existing predicates
        existing_names = {p.name for p in self._discovered}
        unique_new = [p for p in new_preds if p.name not in existing_names]
        self._discovered.extend(unique_new)
        return unique_new

    def discover_from_mechanism(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
        budget: PrivacyBudget,
        max_predicates: int = 50,
    ) -> List[Predicate]:
        """Discover initial predicates from mechanism structure.

        Generates predicates based on probability ratios, bounds,
        and structural properties of the mechanism.

        Args:
            mechanism: Mechanism probability matrix.
            adjacency: Adjacency relation.
            budget: Privacy budget.
            max_predicates: Maximum number of predicates to generate.

        Returns:
            List of discovered predicates.
        """
        preds: List[Predicate] = []
        n, k = mechanism.shape

        # Probability bound predicates
        for idx in range(min(n, 10)):
            for j in range(min(k, 10)):
                val = mechanism[idx, j]
                if 0 < val < 1:
                    pred = self._make_bound_predicate(idx * k + j, val)
                    preds.append(pred)
                    if len(preds) >= max_predicates:
                        break
            if len(preds) >= max_predicates:
                break

        # Privacy ratio predicates for adjacent pairs
        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges = all_edges + [(j_idx, i_idx) for i_idx, j_idx in adjacency.edges]

        for (i, ip) in all_edges:
            if len(preds) >= max_predicates:
                break
            for j in range(min(k, 10)):
                if len(preds) >= max_predicates:
                    break
                p_i = mechanism[i, j]
                p_ip = mechanism[ip, j]
                if p_ip > 1e-300 and p_i > 1e-300:
                    log_ratio = float(np.log(p_i / p_ip))
                    pred = self._make_log_ratio_predicate(i, ip, j, log_ratio, budget.epsilon)
                    if pred is not None:
                        preds.append(pred)

        # Threshold predicates at epsilon boundary
        eps_pred = Predicate(
            name=f"eps_bound_{self._next_id()}",
            formula=Formula(
                expr=f"privacy_loss <= {budget.epsilon}",
                variables=frozenset({"privacy_loss"}),
                formula_type="linear_arithmetic",
                metadata={"type": "linear_le", "coeffs": [1.0], "rhs": budget.epsilon},
            ),
            is_atomic=True,
        )
        preds.append(eps_pred)

        existing_names = {p.name for p in self._discovered}
        unique = [p for p in preds if p.name not in existing_names][:max_predicates]
        self._discovered.extend(unique)
        return unique

    def discover_interpolation_predicates(
        self,
        pre_state: npt.NDArray[np.float64],
        post_state: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> List[Predicate]:
        """Discover predicates via Craig interpolation.

        Computes linear interpolants between pre and post states
        that separate feasible from infeasible regions.

        Args:
            pre_state: Pre-condition state vector.
            post_state: Post-condition state vector.
            budget: Privacy budget for context.

        Returns:
            List of interpolation-derived predicates.
        """
        preds: List[Predicate] = []
        diff = post_state - pre_state
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            return preds

        # Hyperplane separating pre and post
        normal = diff / norm
        midpoint = (pre_state + post_state) / 2.0
        rhs = float(np.dot(normal, midpoint))

        pid = self._next_id()
        pred = Predicate(
            name=f"interpolant_{pid}",
            formula=Formula(
                expr=f"interpolant_{pid}",
                variables=frozenset({f"x{i}" for i in range(len(pre_state))}),
                formula_type="linear_arithmetic",
                metadata={
                    "type": "linear_le",
                    "coeffs": normal.tolist(),
                    "rhs": rhs,
                },
            ),
            is_atomic=True,
        )
        preds.append(pred)

        # Also add per-dimension predicates at the midpoint
        for i in range(min(len(pre_state), 5)):
            if abs(diff[i]) > 1e-12:
                mid_val = (pre_state[i] + post_state[i]) / 2.0
                pid2 = self._next_id()
                coeffs = np.zeros(len(pre_state), dtype=np.float64)
                coeffs[i] = 1.0
                pred2 = Predicate(
                    name=f"interp_dim{i}_{pid2}",
                    formula=Formula(
                        expr=f"x{i} <= {mid_val:.6f}",
                        variables=frozenset({f"x{i}"}),
                        formula_type="linear_arithmetic",
                        metadata={"type": "linear_le", "coeffs": coeffs.tolist(), "rhs": mid_val},
                    ),
                    is_atomic=True,
                )
                preds.append(pred2)

        existing_names = {p.name for p in self._discovered}
        unique = [p for p in preds if p.name not in existing_names]
        self._discovered.extend(unique)
        return unique

    def _make_log_ratio_predicate(
        self,
        i: int,
        ip: int,
        j: int,
        log_ratio: float,
        epsilon: float,
    ) -> Optional[Predicate]:
        """Create a predicate for log-ratio bounds.

        Args:
            i: First row index.
            ip: Second row index.
            j: Column index.
            log_ratio: Observed log-ratio value.
            epsilon: Privacy budget epsilon.

        Returns:
            Predicate or None if not useful.
        """
        pid = self._next_id()
        name = f"log_ratio_{i}_{ip}_{j}_{pid}"
        threshold = min(abs(log_ratio), epsilon)
        return Predicate(
            name=name,
            formula=Formula(
                expr=f"ln(P[{i},{j}]/P[{ip},{j}]) <= {threshold:.6f}",
                variables=frozenset({f"P_{i}_{j}", f"P_{ip}_{j}"}),
                formula_type="linear_arithmetic",
                metadata={"type": "log_ratio_le", "i": i, "j": ip, "bound": threshold},
            ),
            is_atomic=True,
        )

    def _make_bound_predicate(self, flat_idx: int, value: float) -> Predicate:
        """Create a bound predicate P[idx] <= value.

        Args:
            flat_idx: Flattened index into the mechanism matrix.
            value: Bound value.

        Returns:
            Bound predicate.
        """
        pid = self._next_id()
        name = f"bound_{flat_idx}_{pid}"
        coeffs = [0.0] * (flat_idx + 1)
        coeffs[flat_idx] = 1.0
        return Predicate(
            name=name,
            formula=Formula(
                expr=f"x{flat_idx} <= {value:.6f}",
                variables=frozenset({f"x{flat_idx}"}),
                formula_type="linear_arithmetic",
                metadata={"type": "linear_le", "coeffs": coeffs, "rhs": value},
            ),
            is_atomic=True,
        )


# ---------------------------------------------------------------------------
# Cartesian predicate abstraction
# ---------------------------------------------------------------------------


class CartesianAbstraction:
    """Cartesian predicate abstraction.

    Each predicate is tracked independently: the abstract state is the
    Cartesian product of individual predicate truth values. This is less
    precise than Boolean abstraction but much cheaper (linear vs exponential).
    """

    def __init__(
        self,
        predicates: Optional[List[Predicate]] = None,
        evaluator: Optional[PredicateEvaluator] = None,
    ) -> None:
        """Initialize Cartesian abstraction.

        Args:
            predicates: Initial set of predicates.
            evaluator: Predicate evaluator.
        """
        self._predicates: List[Predicate] = list(predicates or [])
        self._evaluator = evaluator or PredicateEvaluator()
        self._state_cache: Dict[FrozenSet[str], int] = {}
        self._next_state_id = 0

    @property
    def predicates(self) -> List[Predicate]:
        """Current set of predicates."""
        return list(self._predicates)

    @property
    def num_predicates(self) -> int:
        """Number of predicates in the abstraction."""
        return len(self._predicates)

    @property
    def num_abstract_states(self) -> int:
        """Number of distinct abstract states created."""
        return len(self._state_cache)

    def add_predicate(self, predicate: Predicate) -> int:
        """Add a predicate and return the number of states that would be split.

        Args:
            predicate: Predicate to add.

        Returns:
            Number of abstract states potentially split.
        """
        if any(p.name == predicate.name for p in self._predicates):
            return 0
        self._predicates.append(predicate)
        # Invalidate cache - all states may need re-evaluation
        old_count = len(self._state_cache)
        self._state_cache.clear()
        return old_count  # Upper bound on splits

    def abstract_state(
        self,
        concrete: npt.NDArray[np.float64],
        concrete_index: int = 0,
    ) -> AbstractState:
        """Map a concrete state to its Cartesian abstract state.

        Evaluates each predicate independently and creates an abstract
        state from the conjunction of results.

        Args:
            concrete: Concrete state vector.
            concrete_index: Index of the concrete state.

        Returns:
            AbstractState with satisfied predicates.
        """
        satisfied = self._evaluator.evaluate_all(self._predicates, concrete)

        if satisfied not in self._state_cache:
            self._state_cache[satisfied] = self._next_state_id
            self._next_state_id += 1

        state_id = self._state_cache[satisfied]
        return AbstractState(
            state_id=state_id,
            predicates=satisfied,
            concrete_states=frozenset({concrete_index}),
        )

    def check_satisfiability(
        self,
        predicates_pos: FrozenSet[str],
        predicates_neg: FrozenSet[str],
        mechanism: npt.NDArray[np.float64],
    ) -> Optional[npt.NDArray[np.float64]]:
        """Check if a predicate combination is satisfiable.

        Searches for a concrete state in the mechanism that satisfies
        all positive predicates and violates all negative ones.

        Args:
            predicates_pos: Predicates that must hold.
            predicates_neg: Predicates that must not hold.
            mechanism: Mechanism matrix to search over.

        Returns:
            A satisfying concrete state, or None if unsatisfiable.
        """
        mechanism = np.atleast_2d(mechanism)
        for i in range(mechanism.shape[0]):
            state = mechanism[i]
            sat = self._evaluator.evaluate_all(self._predicates, state)
            if predicates_pos.issubset(sat) and not predicates_neg.intersection(sat):
                return state.copy()
        return None

    def refine_from_counterexample(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        new_predicates: List[Predicate],
    ) -> RefinementResult:
        """Refine the abstraction by adding new predicates.

        Args:
            counterexample: The spurious counterexample.
            mechanism: Mechanism matrix.
            new_predicates: Predicates to add.

        Returns:
            RefinementResult with refinement details.
        """
        total_splits = 0
        for pred in new_predicates:
            splits = self.add_predicate(pred)
            total_splits += splits

        return RefinementResult(
            new_predicates=new_predicates,
            states_split=total_splits,
            refinement_type=RefinementStrategy.INTERPOLATION,
        )


# ---------------------------------------------------------------------------
# Boolean predicate abstraction
# ---------------------------------------------------------------------------


class BooleanAbstraction:
    """Full Boolean predicate abstraction.

    Tracks all Boolean combinations of predicates. More precise than
    Cartesian but exponential in the number of predicates.
    Only feasible for small predicate sets (≤ 15-20 predicates).
    """

    def __init__(
        self,
        predicates: Optional[List[Predicate]] = None,
        evaluator: Optional[PredicateEvaluator] = None,
        max_predicates: int = 15,
    ) -> None:
        """Initialize Boolean abstraction.

        Args:
            predicates: Initial set of predicates.
            evaluator: Predicate evaluator.
            max_predicates: Maximum number of predicates before falling back.
        """
        self._predicates: List[Predicate] = list(predicates or [])
        self._evaluator = evaluator or PredicateEvaluator()
        self._max_predicates = max_predicates
        self._truth_table: Dict[Tuple[bool, ...], Set[int]] = {}
        self._state_map: Dict[Tuple[bool, ...], int] = {}
        self._next_state_id = 0

    @property
    def predicates(self) -> List[Predicate]:
        """Current predicates."""
        return list(self._predicates)

    @property
    def num_predicates(self) -> int:
        """Number of predicates."""
        return len(self._predicates)

    @property
    def num_abstract_states(self) -> int:
        """Number of distinct abstract states."""
        return len(self._state_map)

    def add_predicate(self, predicate: Predicate) -> int:
        """Add a predicate to the Boolean abstraction.

        Args:
            predicate: Predicate to add.

        Returns:
            Number of states potentially split.

        Raises:
            ValueError: If adding would exceed max_predicates.
        """
        if any(p.name == predicate.name for p in self._predicates):
            return 0
        if len(self._predicates) >= self._max_predicates:
            raise ValueError(
                f"Cannot add predicate: at limit of {self._max_predicates}"
            )
        self._predicates.append(predicate)
        old_count = len(self._state_map)
        self._truth_table.clear()
        self._state_map.clear()
        return old_count

    def abstract_state(
        self,
        concrete: npt.NDArray[np.float64],
        concrete_index: int = 0,
    ) -> AbstractState:
        """Map concrete state to Boolean abstract state.

        Evaluates all predicates and uses the full truth assignment
        as the abstract state identifier.

        Args:
            concrete: Concrete state vector.
            concrete_index: Index of the concrete state.

        Returns:
            AbstractState with truth assignment.
        """
        truth = tuple(
            self._evaluator.evaluate(p, concrete) for p in self._predicates
        )

        if truth not in self._state_map:
            self._state_map[truth] = self._next_state_id
            self._next_state_id += 1
            self._truth_table[truth] = set()

        self._truth_table[truth].add(concrete_index)
        state_id = self._state_map[truth]

        satisfied = frozenset(
            p.name for p, val in zip(self._predicates, truth) if val
        )

        return AbstractState(
            state_id=state_id,
            predicates=satisfied,
            concrete_states=frozenset(self._truth_table[truth]),
        )

    def enumerate_reachable_states(
        self,
        mechanism: npt.NDArray[np.float64],
    ) -> List[AbstractState]:
        """Enumerate all reachable abstract states from a mechanism.

        Args:
            mechanism: Mechanism probability matrix.

        Returns:
            List of all distinct abstract states.
        """
        states: List[AbstractState] = []
        seen: Set[int] = set()

        mechanism = np.atleast_2d(mechanism)
        for i in range(mechanism.shape[0]):
            state = self.abstract_state(mechanism[i], concrete_index=i)
            if state.state_id not in seen:
                seen.add(state.state_id)
                states.append(state)

        return states

    def check_implication(
        self,
        from_preds: FrozenSet[str],
        target_pred: str,
        mechanism: npt.NDArray[np.float64],
    ) -> bool:
        """Check if a set of predicates implies a target predicate.

        Checks whether every concrete state satisfying from_preds
        also satisfies target_pred.

        Args:
            from_preds: Set of premise predicate names.
            target_pred: Conclusion predicate name.
            mechanism: Mechanism to check over.

        Returns:
            True if the implication holds.
        """
        mechanism = np.atleast_2d(mechanism)
        for i in range(mechanism.shape[0]):
            state = mechanism[i]
            sat = self._evaluator.evaluate_all(self._predicates, state)
            if from_preds.issubset(sat):
                if target_pred not in sat:
                    return False
        return True

    def refine_from_counterexample(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        new_predicates: List[Predicate],
    ) -> RefinementResult:
        """Refine by adding predicates and rebuilding the truth table.

        Args:
            counterexample: Spurious counterexample.
            mechanism: Mechanism matrix.
            new_predicates: Predicates to add.

        Returns:
            RefinementResult with details.
        """
        total_splits = 0
        added = []
        for pred in new_predicates:
            if len(self._predicates) < self._max_predicates:
                splits = self.add_predicate(pred)
                total_splits += splits
                added.append(pred)

        # Rebuild truth table
        mechanism = np.atleast_2d(mechanism)
        for i in range(mechanism.shape[0]):
            self.abstract_state(mechanism[i], concrete_index=i)

        return RefinementResult(
            new_predicates=added,
            states_split=total_splits,
            refinement_type=RefinementStrategy.INTERPOLATION,
        )


# ---------------------------------------------------------------------------
# Unified predicate abstraction manager
# ---------------------------------------------------------------------------


class PredicateAbstractionManager:
    """Unified manager for predicate abstraction in CEGAR.

    Implements the AbstractionManager protocol from __init__.py,
    supporting both Cartesian and Boolean abstractions.
    """

    def __init__(
        self,
        abstraction_type: AbstractionType = AbstractionType.CARTESIAN,
        max_predicates: int = 500,
        evaluator: Optional[PredicateEvaluator] = None,
    ) -> None:
        """Initialize the predicate abstraction manager.

        Args:
            abstraction_type: Type of predicate abstraction.
            max_predicates: Maximum number of predicates.
            evaluator: Predicate evaluator.
        """
        self._evaluator = evaluator or PredicateEvaluator()
        self._abstraction_type = abstraction_type

        if abstraction_type == AbstractionType.CARTESIAN:
            self._backend: Any = CartesianAbstraction(evaluator=self._evaluator)
        else:
            self._backend = BooleanAbstraction(
                evaluator=self._evaluator,
                max_predicates=min(max_predicates, 15),
            )

        self._mechanism: Optional[npt.NDArray[np.float64]] = None
        self._adjacency: Optional[AdjacencyRelation] = None
        self._discovery = PredicateDiscovery(evaluator=self._evaluator)

    def initialize(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
    ) -> None:
        """Initialize the abstraction from a concrete mechanism.

        Args:
            mechanism: The n × k probability table.
            adjacency: Adjacency relation.
        """
        self._mechanism = np.asarray(mechanism, dtype=np.float64)
        self._adjacency = adjacency

    def get_abstract_state(self, concrete_index: int) -> AbstractState:
        """Map a concrete state to its abstract state.

        Args:
            concrete_index: Row index in the mechanism matrix.

        Returns:
            Abstract state for the given row.
        """
        if self._mechanism is None:
            raise RuntimeError("Manager not initialized — call initialize() first")
        row = self._mechanism[concrete_index]
        return self._backend.abstract_state(row, concrete_index=concrete_index)

    def refine(self, counterexample: AbstractCounterexample) -> RefinementResult:
        """Refine the abstraction to eliminate a spurious counterexample.

        Args:
            counterexample: The spurious counterexample to eliminate.

        Returns:
            RefinementResult with details of the refinement.
        """
        if self._mechanism is None:
            raise RuntimeError("Manager not initialized")

        budget = PrivacyBudget(epsilon=counterexample.violation_magnitude)
        new_preds = self._discovery.discover_from_counterexample(
            counterexample, self._mechanism, budget,
        )

        return self._backend.refine_from_counterexample(
            counterexample, self._mechanism, new_preds,
        )

    def add_predicate(self, predicate: Predicate) -> int:
        """Add a predicate and return the number of states split.

        Args:
            predicate: Predicate to add.

        Returns:
            Number of abstract states split.
        """
        return self._backend.add_predicate(predicate)

    @property
    def num_abstract_states(self) -> int:
        """Current number of abstract states."""
        return self._backend.num_abstract_states

    @property
    def predicates(self) -> List[Predicate]:
        """Current set of predicates."""
        return self._backend.predicates
