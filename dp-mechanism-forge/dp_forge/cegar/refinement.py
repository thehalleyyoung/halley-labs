"""
Refinement engine for CEGAR-based DP verification.

Implements lazy abstraction with refinement trees, Craig interpolation-based
refinement, counterexample analysis, refinement strategies, and convergence
acceleration via widening with thresholds.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    AbstractDomainType,
    AbstractValue,
    AdjacencyRelation,
    Formula,
    InterpolantType,
    Predicate,
    PrivacyBudget,
)
from dp_forge.cegar import (
    AbstractCounterexample,
    AbstractState,
    AbstractionType,
    CEGARConfig,
    RefinementResult,
    RefinementStrategy as RefinementStrategyEnum,
)
from dp_forge.cegar.abstraction import (
    AbstractDomain,
    GaloisConnection,
    IntervalAbstraction,
    PolyhedralAbstraction,
    PrivacyLossAbstraction,
)
from dp_forge.cegar.predicate_abstraction import (
    PredicateDiscovery,
    PredicateEvaluator,
)


# ---------------------------------------------------------------------------
# Lazy abstraction tree
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """Node in a lazy abstraction refinement tree.

    Attributes:
        node_id: Unique identifier.
        predicates: Predicates associated with this node.
        parent: Parent node ID or None for root.
        children: List of child node IDs.
        abstract_value: Optional abstract value at this node.
        is_covered: Whether this node is covered by another node.
        depth: Depth in the tree (root = 0).
    """
    node_id: int
    predicates: List[Predicate] = field(default_factory=list)
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)
    abstract_value: Optional[AbstractValue] = None
    is_covered: bool = False
    depth: int = 0


class LazyAbstractionTree:
    """Lazy abstraction with refinement tree (LART).

    Maintains a tree of abstract states that is refined on-the-fly.
    Only nodes along counterexample paths are refined, keeping the
    abstraction coarse elsewhere for efficiency.
    """

    def __init__(self, domain: AbstractDomain) -> None:
        """Initialize the lazy abstraction tree.

        Args:
            domain: Abstract domain for computing abstract values.
        """
        self._domain = domain
        self._nodes: Dict[int, TreeNode] = {}
        self._next_id = 0
        self._root: Optional[int] = None

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the tree."""
        return len(self._nodes)

    @property
    def root(self) -> Optional[TreeNode]:
        """Root node of the tree."""
        if self._root is None:
            return None
        return self._nodes.get(self._root)

    def initialize(self, initial_value: AbstractValue) -> int:
        """Create the root node with initial abstract value.

        Args:
            initial_value: Initial abstract value for the root.

        Returns:
            Node ID of the root.
        """
        root_id = self._alloc_id()
        self._nodes[root_id] = TreeNode(
            node_id=root_id,
            abstract_value=initial_value,
            depth=0,
        )
        self._root = root_id
        return root_id

    def _alloc_id(self) -> int:
        """Allocate a new unique node ID."""
        nid = self._next_id
        self._next_id += 1
        return nid

    def get_node(self, node_id: int) -> Optional[TreeNode]:
        """Get a tree node by ID.

        Args:
            node_id: Node identifier.

        Returns:
            TreeNode or None if not found.
        """
        return self._nodes.get(node_id)

    def split_node(
        self,
        node_id: int,
        predicate: Predicate,
        true_value: AbstractValue,
        false_value: AbstractValue,
    ) -> Tuple[int, int]:
        """Split a node into two children based on a predicate.

        Creates two child nodes: one where the predicate holds and one
        where it does not. The parent node is no longer a leaf.

        Args:
            node_id: ID of the node to split.
            predicate: Predicate to split on.
            true_value: Abstract value for the true branch.
            false_value: Abstract value for the false branch.

        Returns:
            (true_child_id, false_child_id).
        """
        parent = self._nodes[node_id]

        true_id = self._alloc_id()
        false_id = self._alloc_id()

        true_preds = list(parent.predicates) + [predicate]
        false_preds = list(parent.predicates)

        self._nodes[true_id] = TreeNode(
            node_id=true_id,
            predicates=true_preds,
            parent=node_id,
            abstract_value=true_value,
            depth=parent.depth + 1,
        )
        self._nodes[false_id] = TreeNode(
            node_id=false_id,
            predicates=false_preds,
            parent=node_id,
            abstract_value=false_value,
            depth=parent.depth + 1,
        )

        parent.children.extend([true_id, false_id])
        return true_id, false_id

    def refine_along_path(
        self,
        path: List[int],
        predicates: List[Predicate],
        domain: AbstractDomain,
    ) -> int:
        """Refine nodes along a counterexample path.

        Adds predicates at each node along the path, splitting nodes
        that need refinement.

        Args:
            path: List of node IDs forming the counterexample path.
            predicates: Predicates to add at each node.
            domain: Abstract domain for computing new abstract values.

        Returns:
            Number of nodes split.
        """
        splits = 0
        for node_id in path:
            node = self._nodes.get(node_id)
            if node is None or node.abstract_value is None:
                continue

            for pred in predicates:
                if any(p.name == pred.name for p in node.predicates):
                    continue
                # Split this node
                true_val = copy.deepcopy(node.abstract_value)
                false_val = copy.deepcopy(node.abstract_value)
                self.split_node(node_id, pred, true_val, false_val)
                splits += 1
                break  # One split per node per refinement

        return splits

    def find_covering_node(
        self,
        node_id: int,
    ) -> Optional[int]:
        """Find a node that covers (subsumes) the given node.

        A node v covers node u if they have the same predicates and
        the abstract value of v contains that of u.

        Args:
            node_id: Node to find covering for.

        Returns:
            ID of covering node, or None.
        """
        target = self._nodes.get(node_id)
        if target is None or target.abstract_value is None:
            return None

        target_preds = frozenset(p.name for p in target.predicates)

        for nid, node in self._nodes.items():
            if nid == node_id or node.is_covered:
                continue
            if node.abstract_value is None:
                continue
            node_preds = frozenset(p.name for p in node.predicates)
            if node_preds == target_preds:
                if self._domain.leq(target.abstract_value, node.abstract_value):
                    return nid
        return None

    def get_leaf_nodes(self) -> List[TreeNode]:
        """Get all leaf nodes (nodes with no children).

        Returns:
            List of leaf TreeNodes.
        """
        return [
            node for node in self._nodes.values()
            if not node.children and not node.is_covered
        ]

    def get_path_to_root(self, node_id: int) -> List[int]:
        """Get the path from a node to the root.

        Args:
            node_id: Starting node.

        Returns:
            List of node IDs from node to root.
        """
        path = []
        current = node_id
        while current is not None:
            path.append(current)
            node = self._nodes.get(current)
            if node is None:
                break
            current = node.parent
        return path


# ---------------------------------------------------------------------------
# Craig interpolation refiner
# ---------------------------------------------------------------------------


class CraigInterpolationRefiner:
    """Refinement via Craig interpolation.

    Computes interpolants between consecutive steps of a counterexample
    trace to derive predicates that eliminate spurious counterexamples.
    """

    def __init__(
        self,
        interpolant_type: InterpolantType = InterpolantType.LINEAR_ARITHMETIC,
        tolerance: float = 1e-9,
    ) -> None:
        """Initialize the interpolation refiner.

        Args:
            interpolant_type: Type of interpolants to compute.
            tolerance: Numerical tolerance.
        """
        self._interpolant_type = interpolant_type
        self._tolerance = tolerance
        self._discovery = PredicateDiscovery()

    def compute_interpolants(
        self,
        trace: List[npt.NDArray[np.float64]],
        mechanism: npt.NDArray[np.float64],
    ) -> List[Formula]:
        """Compute Craig interpolants along a counterexample trace.

        For trace states s_0, s_1, ..., s_n, computes interpolants
        I_1, ..., I_{n-1} such that:
        - s_0 ∧ T_{0,1} → I_1
        - I_k ∧ T_{k,k+1} → I_{k+1}
        - I_{n-1} ∧ T_{n-1,n} → ¬(property)

        Args:
            trace: List of concrete states along the trace.
            mechanism: Mechanism matrix.

        Returns:
            List of interpolant formulas.
        """
        interpolants: List[Formula] = []

        for k in range(len(trace) - 1):
            pre = trace[k]
            post = trace[k + 1]

            # Compute separating hyperplane
            diff = post - pre
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                # Degenerate case: states are identical
                interpolants.append(Formula(
                    expr="true",
                    variables=frozenset(),
                    formula_type="boolean",
                ))
                continue

            normal = diff / norm
            midpoint = (pre + post) / 2.0
            rhs = float(np.dot(normal, midpoint))

            # Build formula string
            terms = []
            variables = set()
            for i, coeff in enumerate(normal):
                if abs(coeff) > 1e-12:
                    terms.append(f"{coeff:.6f}*x{i}")
                    variables.add(f"x{i}")

            expr = " + ".join(terms) + f" <= {rhs:.6f}" if terms else "true"

            interpolants.append(Formula(
                expr=expr,
                variables=frozenset(variables),
                formula_type="linear_arithmetic",
                metadata={
                    "type": "linear_le",
                    "coeffs": normal.tolist(),
                    "rhs": rhs,
                    "step": k,
                },
            ))

        return interpolants

    def refine_from_interpolants(
        self,
        interpolants: List[Formula],
    ) -> List[Predicate]:
        """Convert interpolants to predicates for abstraction refinement.

        Args:
            interpolants: List of interpolant formulas.

        Returns:
            List of predicates derived from the interpolants.
        """
        predicates: List[Predicate] = []
        for idx, formula in enumerate(interpolants):
            if formula.expr == "true":
                continue
            pred = Predicate(
                name=f"craig_interp_{idx}_{id(formula) % 10000}",
                formula=formula,
                is_atomic=True,
            )
            predicates.append(pred)
        return predicates

    def refine(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
    ) -> RefinementResult:
        """Perform refinement using Craig interpolation.

        Args:
            counterexample: Spurious counterexample to eliminate.
            mechanism: Mechanism matrix.

        Returns:
            RefinementResult with interpolation-derived predicates.
        """
        # Extract concrete states from the trace
        trace: List[npt.NDArray[np.float64]] = []
        for state in counterexample.trace:
            if state.abstract_value is not None:
                trace.append(state.abstract_value.lower)
            else:
                # Use mechanism row if available
                if state.concrete_states:
                    idx = min(state.concrete_states)
                    if idx < mechanism.shape[0]:
                        trace.append(mechanism[idx])

        if len(trace) < 2:
            # Fallback: use violating pair rows
            i, ip = counterexample.violating_pair
            if i < mechanism.shape[0] and ip < mechanism.shape[0]:
                trace = [mechanism[i], mechanism[ip]]

        interpolants = self.compute_interpolants(trace, mechanism)
        predicates = self.refine_from_interpolants(interpolants)

        return RefinementResult(
            new_predicates=predicates,
            states_split=len(predicates),
            refinement_type=RefinementStrategyEnum.INTERPOLATION,
            interpolant=interpolants[0] if interpolants else None,
        )


# ---------------------------------------------------------------------------
# Counterexample analysis
# ---------------------------------------------------------------------------


class CounterexampleAnalysis:
    """Analyze counterexamples to classify them as spurious or genuine.

    Checks feasibility of abstract counterexample traces in the concrete
    mechanism by verifying that the privacy violation actually exists.
    """

    def __init__(self, tolerance: float = 1e-9) -> None:
        """Initialize counterexample analyzer.

        Args:
            tolerance: Numerical tolerance for feasibility checks.
        """
        self._tolerance = tolerance

    def check_feasibility(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
    ) -> bool:
        """Check if an abstract counterexample is feasible in the concrete model.

        Verifies that the claimed privacy violation exists in the actual
        mechanism probability table.

        Args:
            counterexample: Abstract counterexample to check.
            mechanism: Concrete mechanism matrix.

        Returns:
            True if the counterexample is feasible (genuine).
        """
        i, ip = counterexample.violating_pair
        n, k = mechanism.shape

        if i >= n or ip >= n:
            return False

        # Check if the actual privacy loss exceeds the claimed violation
        max_log_ratio = 0.0
        for j in range(k):
            p_i = mechanism[i, j]
            p_ip = mechanism[ip, j]
            if p_ip > 1e-300 and p_i > 1e-300:
                ratio = abs(np.log(p_i / p_ip))
                max_log_ratio = max(max_log_ratio, ratio)

        # Feasible if actual max ratio is close to or exceeds the claimed magnitude
        return max_log_ratio >= counterexample.violation_magnitude - self._tolerance

    def classify_counterexample(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> Tuple[bool, float]:
        """Classify a counterexample and compute the actual violation.

        Args:
            counterexample: Abstract counterexample.
            mechanism: Mechanism matrix.
            budget: Privacy budget.

        Returns:
            (is_genuine, actual_violation_magnitude).
        """
        i, ip = counterexample.violating_pair
        n, k = mechanism.shape

        if i >= n or ip >= n:
            return False, 0.0

        max_log_ratio = 0.0
        for j in range(k):
            p_i = mechanism[i, j]
            p_ip = mechanism[ip, j]
            if p_ip > 1e-300 and p_i > 1e-300:
                ratio = abs(np.log(p_i / p_ip))
                max_log_ratio = max(max_log_ratio, ratio)

        is_genuine = max_log_ratio > budget.epsilon + self._tolerance
        return is_genuine, max_log_ratio

    def extract_concrete_witness(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
    ) -> Optional[Tuple[int, int, int, float]]:
        """Extract a concrete witness (i, i', j, violation) from the counterexample.

        Finds the specific output j that maximizes the privacy loss.

        Args:
            counterexample: Abstract counterexample.
            mechanism: Mechanism matrix.

        Returns:
            (i, i', j_worst, violation_magnitude) or None if no violation.
        """
        i, ip = counterexample.violating_pair
        n, k = mechanism.shape

        if i >= n or ip >= n:
            return None

        best_j = 0
        best_ratio = 0.0

        for j in range(k):
            p_i = mechanism[i, j]
            p_ip = mechanism[ip, j]
            if p_ip > 1e-300 and p_i > 1e-300:
                ratio = abs(np.log(p_i / p_ip))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_j = j

        if best_ratio > self._tolerance:
            return (i, ip, best_j, best_ratio)
        return None

    def analyze_spuriousness_cause(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
    ) -> Dict[str, Any]:
        """Analyze why a counterexample is spurious.

        Provides diagnostic information about the abstraction imprecision
        that led to the spurious counterexample.

        Args:
            counterexample: Spurious counterexample.
            mechanism: Mechanism matrix.

        Returns:
            Dictionary with diagnostic information.
        """
        i, ip = counterexample.violating_pair
        n, k = mechanism.shape

        info: Dict[str, Any] = {
            "violating_pair": (i, ip),
            "claimed_magnitude": counterexample.violation_magnitude,
        }

        if i < n and ip < n:
            actual_ratios = []
            for j in range(k):
                p_i = mechanism[i, j]
                p_ip = mechanism[ip, j]
                if p_ip > 1e-300 and p_i > 1e-300:
                    actual_ratios.append(abs(np.log(p_i / p_ip)))
                else:
                    actual_ratios.append(0.0)

            info["actual_max_ratio"] = max(actual_ratios) if actual_ratios else 0.0
            info["actual_ratios"] = actual_ratios
            info["gap"] = counterexample.violation_magnitude - info["actual_max_ratio"]
            info["num_outputs"] = k

        return info


# ---------------------------------------------------------------------------
# Refinement strategy selector
# ---------------------------------------------------------------------------


class RefinementStrategySelector:
    """Selects and applies the appropriate refinement strategy.

    Supports interpolation, weakest precondition, strongest postcondition,
    impact-based, and lazy refinement strategies.
    """

    def __init__(
        self,
        config: Optional[CEGARConfig] = None,
    ) -> None:
        """Initialize the strategy selector.

        Args:
            config: CEGAR configuration.
        """
        self._config = config or CEGARConfig()
        self._interpolation_refiner = CraigInterpolationRefiner(
            interpolant_type=self._config.interpolant_type,
        )
        self._discovery = PredicateDiscovery()
        self._refinement_history: List[RefinementResult] = []

    @property
    def refinement_count(self) -> int:
        """Number of refinements performed."""
        return len(self._refinement_history)

    def select_strategy(
        self,
        counterexample: AbstractCounterexample,
        iteration: int,
    ) -> RefinementStrategyEnum:
        """Select the best refinement strategy for the current iteration.

        May adapt the strategy based on the iteration count and
        counterexample characteristics.

        Args:
            counterexample: Current counterexample.
            iteration: Current CEGAR iteration number.

        Returns:
            Selected refinement strategy.
        """
        configured = self._config.refinement_strategy

        # Adaptive strategy selection
        if iteration > 20 and configured == RefinementStrategyEnum.INTERPOLATION:
            # Switch to impact after many interpolation rounds
            return RefinementStrategyEnum.IMPACT
        if counterexample.violation_magnitude < 1e-6:
            # Small violations benefit from weakest precondition
            return RefinementStrategyEnum.WEAKEST_PRECONDITION

        return configured

    def refine(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        strategy: Optional[RefinementStrategyEnum] = None,
    ) -> RefinementResult:
        """Apply the selected refinement strategy.

        Args:
            counterexample: Spurious counterexample to eliminate.
            mechanism: Mechanism matrix.
            budget: Privacy budget.
            strategy: Optional strategy override.

        Returns:
            RefinementResult with new predicates.
        """
        if strategy is None:
            strategy = self.select_strategy(
                counterexample, len(self._refinement_history),
            )

        if strategy == RefinementStrategyEnum.INTERPOLATION:
            result = self._refine_interpolation(counterexample, mechanism)
        elif strategy == RefinementStrategyEnum.WEAKEST_PRECONDITION:
            result = self._refine_wp(counterexample, mechanism, budget)
        elif strategy == RefinementStrategyEnum.STRONGEST_POSTCONDITION:
            result = self._refine_sp(counterexample, mechanism, budget)
        elif strategy == RefinementStrategyEnum.IMPACT:
            result = self._refine_impact(counterexample, mechanism, budget)
        elif strategy == RefinementStrategyEnum.LAZY:
            result = self._refine_lazy(counterexample, mechanism, budget)
        else:
            result = self._refine_interpolation(counterexample, mechanism)

        self._refinement_history.append(result)
        return result

    def _refine_interpolation(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
    ) -> RefinementResult:
        """Refinement via Craig interpolation."""
        return self._interpolation_refiner.refine(counterexample, mechanism)

    def _refine_wp(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> RefinementResult:
        """Refinement via weakest precondition computation.

        Computes the weakest precondition of the property violation
        and derives predicates from it.
        """
        i, ip = counterexample.violating_pair
        n, k = mechanism.shape
        preds: List[Predicate] = []

        # WP of "max_j ln(P[i,j]/P[i',j]) > epsilon" is a conjunction
        # of bounds on individual ratios
        for j in range(min(k, 20)):
            p_i = mechanism[min(i, n-1), j]
            p_ip = mechanism[min(ip, n-1), j]
            if p_ip > 1e-300 and p_i > 1e-300:
                log_ratio = np.log(p_i / p_ip)
                if abs(log_ratio) > 1e-12:
                    coeffs = np.zeros(k, dtype=np.float64)
                    coeffs[j] = 1.0
                    pred = Predicate(
                        name=f"wp_{i}_{ip}_{j}_{id(counterexample) % 10000}",
                        formula=Formula(
                            expr=f"P[{i},{j}]/P[{ip},{j}] <= exp({budget.epsilon})",
                            variables=frozenset({f"P_{i}_{j}", f"P_{ip}_{j}"}),
                            formula_type="linear_arithmetic",
                            metadata={
                                "type": "log_ratio_le",
                                "i": i,
                                "j": ip,
                                "bound": budget.epsilon,
                            },
                        ),
                        is_atomic=True,
                    )
                    preds.append(pred)

        return RefinementResult(
            new_predicates=preds,
            states_split=len(preds),
            refinement_type=RefinementStrategyEnum.WEAKEST_PRECONDITION,
        )

    def _refine_sp(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> RefinementResult:
        """Refinement via strongest postcondition computation."""
        i, ip = counterexample.violating_pair
        n, k = mechanism.shape
        preds: List[Predicate] = []

        # SP: propagate concrete bounds forward
        row_i = mechanism[min(i, n-1)]
        row_ip = mechanism[min(ip, n-1)]

        for j in range(min(k, 20)):
            if row_i[j] > 1e-300 and row_ip[j] > 1e-300:
                actual_ratio = np.log(row_i[j] / row_ip[j])
                # Create predicate at actual value
                pred = Predicate(
                    name=f"sp_{i}_{ip}_{j}_{id(counterexample) % 10000}",
                    formula=Formula(
                        expr=f"log_ratio_{i}_{ip}_{j} <= {actual_ratio:.8f}",
                        variables=frozenset({f"log_ratio_{i}_{ip}_{j}"}),
                        formula_type="linear_arithmetic",
                        metadata={
                            "type": "log_ratio_le",
                            "i": i,
                            "j": ip,
                            "bound": actual_ratio,
                        },
                    ),
                    is_atomic=True,
                )
                preds.append(pred)

        return RefinementResult(
            new_predicates=preds,
            states_split=len(preds),
            refinement_type=RefinementStrategyEnum.STRONGEST_POSTCONDITION,
        )

    def _refine_impact(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> RefinementResult:
        """Impact-based refinement: find predicates with maximum impact.

        Selects predicates that split the most abstract states or
        eliminate the most counterexamples.
        """
        # Combine interpolation and WP predicates, rank by impact
        interp_result = self._refine_interpolation(counterexample, mechanism)
        wp_result = self._refine_wp(counterexample, mechanism, budget)

        all_preds = interp_result.new_predicates + wp_result.new_predicates

        # Score predicates by how much they distinguish rows
        scored: List[Tuple[float, Predicate]] = []
        evaluator = PredicateEvaluator()
        for pred in all_preds:
            true_count = 0
            false_count = 0
            for idx in range(mechanism.shape[0]):
                if evaluator.evaluate(pred, mechanism[idx]):
                    true_count += 1
                else:
                    false_count += 1
            # Best predicates split roughly evenly
            balance = min(true_count, false_count) / max(true_count + false_count, 1)
            scored.append((balance, pred))

        scored.sort(key=lambda x: -x[0])
        best_preds = [pred for _, pred in scored[:10]]

        return RefinementResult(
            new_predicates=best_preds,
            states_split=len(best_preds),
            refinement_type=RefinementStrategyEnum.IMPACT,
        )

    def _refine_lazy(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> RefinementResult:
        """Lazy refinement: minimal predicate addition.

        Adds only the single most impactful predicate.
        """
        impact_result = self._refine_impact(counterexample, mechanism, budget)
        # Take only the best predicate
        best = impact_result.new_predicates[:1] if impact_result.new_predicates else []

        return RefinementResult(
            new_predicates=best,
            states_split=len(best),
            refinement_type=RefinementStrategyEnum.LAZY,
        )


# ---------------------------------------------------------------------------
# Convergence acceleration
# ---------------------------------------------------------------------------


class ConvergenceAccelerator:
    """Convergence acceleration for the CEGAR loop.

    Uses widening with thresholds and extrapolation to speed up
    convergence of the abstraction refinement loop.
    """

    def __init__(
        self,
        domain: AbstractDomain,
        thresholds: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Initialize convergence accelerator.

        Args:
            domain: Abstract domain.
            thresholds: Threshold values for widening. If None, uses
                default thresholds {0, 0.1, 0.5, 1, 2, 5, 10, 100}.
        """
        self._domain = domain
        self._thresholds = thresholds if thresholds is not None else np.array(
            [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0],
            dtype=np.float64,
        )
        self._history: List[AbstractValue] = []
        self._iteration = 0

    def should_widen(self, iteration: int, stabilized: bool = False) -> bool:
        """Determine whether to apply widening at this iteration.

        Args:
            iteration: Current iteration number.
            stabilized: Whether the fixpoint iteration has stabilized.

        Returns:
            True if widening should be applied.
        """
        if stabilized:
            return False
        # Apply widening after a delay to allow initial precision
        return iteration >= 3

    def accelerate(
        self,
        previous: AbstractValue,
        current: AbstractValue,
        iteration: int,
    ) -> AbstractValue:
        """Apply convergence acceleration.

        Uses widening with thresholds instead of standard widening
        to achieve convergence while maintaining precision.

        Args:
            previous: Abstract value from previous iteration.
            current: Abstract value from current iteration.
            iteration: Current iteration number.

        Returns:
            Accelerated abstract value.
        """
        self._iteration = iteration

        if not self.should_widen(iteration):
            return self._domain.join(previous, current)

        # Use widening with thresholds via the interval domain
        if isinstance(self._domain, IntervalAbstraction):
            return self._domain.widen_with_thresholds(
                previous, current, self._thresholds,
            )
        else:
            return self._domain.widen(previous, current)

    def check_convergence(
        self,
        previous: AbstractValue,
        current: AbstractValue,
    ) -> bool:
        """Check if the fixpoint iteration has converged.

        Convergence is detected when current ⊆ previous (the abstract
        value is no longer growing).

        Args:
            previous: Previous abstract value.
            current: Current abstract value.

        Returns:
            True if converged.
        """
        return self._domain.leq(current, previous)

    def narrow_result(
        self,
        widened: AbstractValue,
        precise: AbstractValue,
        max_iterations: int = 5,
    ) -> AbstractValue:
        """Apply narrowing iterations to improve precision after widening.

        Args:
            widened: Result from widened fixpoint.
            precise: More precise value from a descending iteration.
            max_iterations: Maximum narrowing iterations.

        Returns:
            Narrowed abstract value.
        """
        result = widened
        for _ in range(max_iterations):
            narrowed = self._domain.narrow(result, precise)
            if self._domain.leq(result, narrowed) and self._domain.leq(narrowed, result):
                break  # No change
            result = narrowed
        return result


# ---------------------------------------------------------------------------
# Integrated refinement engine
# ---------------------------------------------------------------------------


class RefinementEngine:
    """Integrated refinement engine for CEGAR.

    Coordinates counterexample analysis, strategy selection, interpolation,
    and convergence acceleration.
    """

    def __init__(
        self,
        config: Optional[CEGARConfig] = None,
        domain: Optional[AbstractDomain] = None,
    ) -> None:
        """Initialize the refinement engine.

        Args:
            config: CEGAR configuration.
            domain: Abstract domain (defaults to IntervalAbstraction).
        """
        self._config = config or CEGARConfig()
        self._domain = domain or IntervalAbstraction()
        self._analyzer = CounterexampleAnalysis()
        self._strategy = RefinementStrategySelector(config=self._config)
        self._accelerator = ConvergenceAccelerator(self._domain)
        self._tree: Optional[LazyAbstractionTree] = None
        self._total_predicates: List[Predicate] = []

    @property
    def refinement_count(self) -> int:
        """Total number of refinements performed."""
        return self._strategy.refinement_count

    @property
    def predicates(self) -> List[Predicate]:
        """All predicates discovered during refinement."""
        return list(self._total_predicates)

    def initialize_tree(self, initial_value: AbstractValue) -> None:
        """Initialize the lazy abstraction tree.

        Args:
            initial_value: Initial abstract value for root.
        """
        self._tree = LazyAbstractionTree(self._domain)
        self._tree.initialize(initial_value)

    def analyze_and_refine(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
    ) -> Tuple[bool, Optional[RefinementResult]]:
        """Analyze a counterexample and refine if spurious.

        This is the main entry point for the refinement engine.
        First classifies the counterexample, then refines if spurious.

        Args:
            counterexample: Abstract counterexample from verifier.
            mechanism: Mechanism matrix.
            budget: Privacy budget.

        Returns:
            (is_genuine, refinement_result_or_none).
        """
        # Check feasibility
        is_genuine, actual_mag = self._analyzer.classify_counterexample(
            counterexample, mechanism, budget,
        )

        if is_genuine:
            counterexample.is_spurious = False
            return True, None

        # Spurious — refine
        counterexample.is_spurious = True
        result = self._strategy.refine(counterexample, mechanism, budget)
        self._total_predicates.extend(result.new_predicates)

        # Update tree if using lazy abstraction
        if self._tree is not None and result.new_predicates:
            path = [s.state_id for s in counterexample.trace]
            valid_path = [p for p in path if self._tree.get_node(p) is not None]
            if valid_path:
                self._tree.refine_along_path(
                    valid_path, result.new_predicates, self._domain,
                )

        return False, result
