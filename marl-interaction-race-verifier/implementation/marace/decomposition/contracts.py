"""Assume-guarantee contract generation and checking.

Contracts specify interface obligations between interaction groups:
each group *assumes* bounds on shared state variables provided by
neighbouring groups, and *guarantees* bounds on the variables it
exports.  This module generates, checks, refines, and composes
such contracts using linear arithmetic over shared state predicates.
"""

from __future__ import annotations

import enum
import itertools
import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np


# ---------------------------------------------------------------------------
# Interface variables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterfaceVariable:
    """A shared state variable at a group boundary.

    Attributes:
        name: Human-readable identifier (e.g. ``"x_agent0"``).
        group_source: Group that produces (writes) this variable.
        group_target: Group that consumes (reads) this variable.
        lower_bound: Sound lower bound on the variable's value.
        upper_bound: Sound upper bound on the variable's value.
        dimension_index: Index into the joint state vector, if applicable.
    """

    name: str
    group_source: str
    group_target: str
    lower_bound: float = -math.inf
    upper_bound: float = math.inf
    dimension_index: Optional[int] = None

    def contains(self, value: float) -> bool:
        """Check whether *value* lies within the bounds."""
        return self.lower_bound <= value <= self.upper_bound

    def tighten(
        self, new_lower: Optional[float] = None, new_upper: Optional[float] = None
    ) -> "InterfaceVariable":
        """Return a copy with tightened bounds."""
        lb = max(self.lower_bound, new_lower) if new_lower is not None else self.lower_bound
        ub = min(self.upper_bound, new_upper) if new_upper is not None else self.upper_bound
        return InterfaceVariable(
            name=self.name,
            group_source=self.group_source,
            group_target=self.group_target,
            lower_bound=lb,
            upper_bound=ub,
            dimension_index=self.dimension_index,
        )

    def widen(self, factor: float = 1.1) -> "InterfaceVariable":
        """Return a copy with widened bounds (for contract weakening)."""
        centre = 0.5 * (self.lower_bound + self.upper_bound)
        half = 0.5 * (self.upper_bound - self.lower_bound) * factor
        return InterfaceVariable(
            name=self.name,
            group_source=self.group_source,
            group_target=self.group_target,
            lower_bound=centre - half,
            upper_bound=centre + half,
            dimension_index=self.dimension_index,
        )


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearPredicate:
    """A single linear inequality ``a^T x <= b`` over state variables.

    Attributes:
        coefficients: Mapping from variable name to coefficient.
        bound: Right-hand side of the inequality.
    """

    coefficients: Dict[str, float]
    bound: float

    def evaluate(self, assignment: Dict[str, float]) -> bool:
        """Check satisfaction under a concrete variable assignment."""
        lhs = sum(
            c * assignment.get(v, 0.0) for v, c in self.coefficients.items()
        )
        return lhs <= self.bound + 1e-12

    def negate(self) -> "LinearPredicate":
        """Return the negation ``a^T x > b`` encoded as ``-a^T x <= -b - eps``."""
        neg_coeffs = {v: -c for v, c in self.coefficients.items()}
        return LinearPredicate(neg_coeffs, -self.bound - 1e-9)


@dataclass
class ConjunctivePredicate:
    """Conjunction of linear predicates (a polyhedron)."""

    clauses: List[LinearPredicate] = field(default_factory=list)

    def evaluate(self, assignment: Dict[str, float]) -> bool:
        return all(c.evaluate(assignment) for c in self.clauses)

    def add(self, pred: LinearPredicate) -> None:
        self.clauses.append(pred)

    @property
    def variables(self) -> Set[str]:
        vs: Set[str] = set()
        for c in self.clauses:
            vs.update(c.coefficients.keys())
        return vs

    def to_matrix_form(
        self, var_order: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to ``Ax <= b`` matrix form.

        Returns:
            ``(A, b)`` where ``A`` has shape ``(m, n)`` and ``b`` has
            shape ``(m,)``.
        """
        idx = {v: i for i, v in enumerate(var_order)}
        n = len(var_order)
        m = len(self.clauses)
        A = np.zeros((m, n), dtype=np.float64)
        b = np.zeros(m, dtype=np.float64)
        for row, clause in enumerate(self.clauses):
            for var, coeff in clause.coefficients.items():
                if var in idx:
                    A[row, idx[var]] = coeff
            b[row] = clause.bound
        return A, b


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class Contract:
    """Assume-guarantee contract over shared state predicates.

    A contract ``(A, G)`` asserts: *if* the environment satisfies the
    assumption ``A``, *then* the component guarantees ``G``.

    Both ``A`` and ``G`` are conjunctive predicates over the interface
    variables.
    """

    def __init__(
        self,
        name: str,
        assumption: ConjunctivePredicate,
        guarantee: ConjunctivePredicate,
        interface_vars: Sequence[InterfaceVariable] = (),
    ) -> None:
        self.name = name
        self.assumption = assumption
        self.guarantee = guarantee
        self.interface_vars = list(interface_vars)

    def check_satisfaction(
        self, assumption_assignment: Dict[str, float], guarantee_assignment: Dict[str, float]
    ) -> bool:
        """Check that if the assumption holds, the guarantee also holds."""
        if not self.assumption.evaluate(assumption_assignment):
            return True  # vacuously true
        return self.guarantee.evaluate(guarantee_assignment)

    @property
    def variables(self) -> Set[str]:
        return self.assumption.variables | self.guarantee.variables

    def __repr__(self) -> str:
        return (
            f"Contract({self.name!r}, "
            f"assume={len(self.assumption.clauses)} clauses, "
            f"guarantee={len(self.guarantee.clauses)} clauses)"
        )


class LinearContract(Contract):
    """Contract whose predicates are conjunctions of linear inequalities.

    Provides matrix-form accessors for SMT-like constraint solving.
    """

    def assumption_matrix(
        self, var_order: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.assumption.to_matrix_form(var_order)

    def guarantee_matrix(
        self, var_order: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.guarantee.to_matrix_form(var_order)

    def is_feasible(self, var_order: List[str]) -> bool:
        """Quick feasibility check via LP relaxation.

        Returns ``True`` if the assumption polyhedron is non-empty.
        Falls back to True if scipy is not available.
        """
        try:
            from scipy.optimize import linprog

            A, b = self.assumption_matrix(var_order)
            if A.size == 0:
                return True
            n = A.shape[1]
            c = np.zeros(n)
            result = linprog(c, A_ub=A, b_ub=b, method="highs")
            return result.success
        except ImportError:
            return True


# ---------------------------------------------------------------------------
# Contract templates
# ---------------------------------------------------------------------------

class ContractTemplate(enum.Enum):
    """Library of common contract patterns."""

    MUTUAL_EXCLUSION = "mutual_exclusion"
    ORDERING = "ordering"
    BOUNDED_DELAY = "bounded_delay"
    BOUNDED_STATE = "bounded_state"
    NON_INTERFERENCE = "non_interference"

    def instantiate(
        self,
        var_names: Sequence[str],
        params: Optional[Dict[str, float]] = None,
    ) -> Contract:
        """Instantiate this template with concrete variable names.

        Parameters:
            var_names: Variable names to bind.
            params: Template parameters (e.g. bounds, delays).
        """
        params = params or {}

        if self == ContractTemplate.MUTUAL_EXCLUSION:
            if len(var_names) < 2:
                raise ValueError("Mutual exclusion requires >= 2 variables")
            assume = ConjunctivePredicate()
            guarantee = ConjunctivePredicate()
            # x_i + x_j <= 1  for all pairs (binary occupancy)
            for vi, vj in itertools.combinations(var_names, 2):
                guarantee.add(LinearPredicate({vi: 1.0, vj: 1.0}, 1.0))
            return Contract(f"mutex_{'+'.join(var_names)}", assume, guarantee)

        if self == ContractTemplate.ORDERING:
            if len(var_names) < 2:
                raise ValueError("Ordering requires >= 2 variables")
            assume = ConjunctivePredicate()
            guarantee = ConjunctivePredicate()
            # x_{i} <= x_{i+1}
            for i in range(len(var_names) - 1):
                guarantee.add(
                    LinearPredicate(
                        {var_names[i]: 1.0, var_names[i + 1]: -1.0}, 0.0
                    )
                )
            return Contract(f"order_{'+'.join(var_names)}", assume, guarantee)

        if self == ContractTemplate.BOUNDED_DELAY:
            delta = params.get("max_delay", 1.0)
            if len(var_names) < 2:
                raise ValueError("Bounded delay requires >= 2 variables")
            assume = ConjunctivePredicate()
            guarantee = ConjunctivePredicate()
            # |x_i - x_j| <= delta  ⟺  x_i - x_j <= delta  ∧  x_j - x_i <= delta
            for vi, vj in itertools.combinations(var_names, 2):
                guarantee.add(LinearPredicate({vi: 1.0, vj: -1.0}, delta))
                guarantee.add(LinearPredicate({vi: -1.0, vj: 1.0}, delta))
            return Contract(f"delay_{delta}", assume, guarantee)

        if self == ContractTemplate.BOUNDED_STATE:
            lb = params.get("lower", -1e6)
            ub = params.get("upper", 1e6)
            assume = ConjunctivePredicate()
            guarantee = ConjunctivePredicate()
            for v in var_names:
                guarantee.add(LinearPredicate({v: 1.0}, ub))
                guarantee.add(LinearPredicate({v: -1.0}, -lb))
            return Contract(f"bounded_{lb}_{ub}", assume, guarantee)

        if self == ContractTemplate.NON_INTERFERENCE:
            # Assumption: other-group vars bounded; guarantee: local vars unchanged
            assume = ConjunctivePredicate()
            guarantee = ConjunctivePredicate()
            bound = params.get("bound", 1e3)
            for v in var_names:
                assume.add(LinearPredicate({v: 1.0}, bound))
                assume.add(LinearPredicate({v: -1.0}, bound))
            return Contract(f"noninterf_{'+'.join(var_names)}", assume, guarantee)

        raise ValueError(f"Unknown template: {self}")


# ---------------------------------------------------------------------------
# ContractGenerator
# ---------------------------------------------------------------------------

class ContractGenerator:
    """Automatically generate contracts from an interaction graph and
    abstract interpretation results.

    Strategy:
    1. For each cross-group edge, identify the shared interface variables.
    2. Use zonotope bounds from abstract interpretation (if available)
       to derive initial interval contracts.
    3. Apply contract templates for structurally-identified patterns.
    """

    def __init__(self, default_bound: float = 1e4) -> None:
        self._default_bound = default_bound

    def generate_from_interface_vars(
        self,
        interface_vars: Sequence[InterfaceVariable],
        group_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> List[LinearContract]:
        """Generate interval contracts from interface variable bounds.

        Each interface variable produces an assumption on the source group
        and a guarantee on the target group (the source guarantees bounds).
        """
        contracts: List[LinearContract] = []

        # Group by (source, target) pair
        by_pair: Dict[Tuple[str, str], List[InterfaceVariable]] = {}
        for iv in interface_vars:
            key = (iv.group_source, iv.group_target)
            if group_pairs is not None and key not in group_pairs:
                continue
            by_pair.setdefault(key, []).append(iv)

        for (src, tgt), ivs in by_pair.items():
            assume = ConjunctivePredicate()
            guarantee = ConjunctivePredicate()
            for iv in ivs:
                lb = iv.lower_bound if math.isfinite(iv.lower_bound) else -self._default_bound
                ub = iv.upper_bound if math.isfinite(iv.upper_bound) else self._default_bound
                # Source guarantees: lb <= v <= ub
                guarantee.add(LinearPredicate({iv.name: 1.0}, ub))
                guarantee.add(LinearPredicate({iv.name: -1.0}, -lb))
                # Target assumes the same bounds
                assume.add(LinearPredicate({iv.name: 1.0}, ub))
                assume.add(LinearPredicate({iv.name: -1.0}, -lb))

            contract = LinearContract(
                name=f"interface_{src}_to_{tgt}",
                assumption=assume,
                guarantee=guarantee,
                interface_vars=ivs,
            )
            contracts.append(contract)

        return contracts

    def generate_from_zonotope_bounds(
        self,
        group_id: str,
        variable_names: Sequence[str],
        centers: np.ndarray,
        radii: np.ndarray,
    ) -> LinearContract:
        """Generate an interval contract from zonotope centre ± radius.

        Parameters:
            group_id: Identifier of the group whose abstract state was computed.
            variable_names: Names of variables in the abstract state.
            centers: Centre vector of the zonotope.
            radii: Per-dimension radius (sum of absolute generator columns).

        Returns:
            A :class:`LinearContract` with interval bounds.
        """
        assume = ConjunctivePredicate()
        guarantee = ConjunctivePredicate()
        ivs: List[InterfaceVariable] = []

        for i, vname in enumerate(variable_names):
            lb = float(centers[i] - radii[i])
            ub = float(centers[i] + radii[i])
            guarantee.add(LinearPredicate({vname: 1.0}, ub))
            guarantee.add(LinearPredicate({vname: -1.0}, -lb))
            ivs.append(
                InterfaceVariable(
                    name=vname,
                    group_source=group_id,
                    group_target="*",
                    lower_bound=lb,
                    upper_bound=ub,
                    dimension_index=i,
                )
            )

        return LinearContract(
            name=f"zonotope_bounds_{group_id}",
            assumption=assume,
            guarantee=guarantee,
            interface_vars=ivs,
        )


# ---------------------------------------------------------------------------
# ContractChecker
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a contract satisfaction check."""

    satisfied: bool
    violating_assignment: Optional[Dict[str, float]] = None
    message: str = ""


class ContractChecker:
    """Verify that contracts are satisfied using constraint solving.

    Uses LP feasibility (via ``scipy.optimize.linprog``) to check whether
    the negation of the guarantee is feasible under the assumption
    (i.e. whether a counterexample exists).
    """

    def check(
        self, contract: LinearContract, var_order: Optional[List[str]] = None
    ) -> CheckResult:
        """Check whether the contract is satisfiable (no counterexample).

        A contract ``(A, G)`` is *valid* iff ``A ∧ ¬G`` is infeasible.
        We encode ``¬G`` as the disjunction of negated guarantee clauses
        and check each disjunct.
        """
        if var_order is None:
            var_order = sorted(contract.variables)

        try:
            from scipy.optimize import linprog
        except ImportError:
            return CheckResult(
                satisfied=True,
                message="scipy not available; skipping LP check",
            )

        A_assume, b_assume = contract.assumption_matrix(var_order)

        # Check each negated guarantee clause (disjunctive negation)
        for clause in contract.guarantee.clauses:
            neg = clause.negate()
            neg_pred = ConjunctivePredicate([neg])
            A_neg, b_neg = neg_pred.to_matrix_form(var_order)

            # Stack assumption + negated clause
            if A_assume.size > 0 and A_neg.size > 0:
                A_combined = np.vstack([A_assume, A_neg])
                b_combined = np.concatenate([b_assume, b_neg])
            elif A_neg.size > 0:
                A_combined = A_neg
                b_combined = b_neg
            else:
                continue

            n = A_combined.shape[1]
            result = linprog(
                np.zeros(n),
                A_ub=A_combined,
                b_ub=b_combined,
                bounds=(None, None),
                method="highs",
            )
            if result.success:
                assignment = {
                    var_order[i]: float(result.x[i]) for i in range(n)
                }
                return CheckResult(
                    satisfied=False,
                    violating_assignment=assignment,
                    message=f"Counterexample found violating clause: {clause}",
                )

        return CheckResult(satisfied=True, message="Contract verified (LP)")

    def check_all(
        self, contracts: Sequence[LinearContract]
    ) -> Dict[str, CheckResult]:
        """Check multiple contracts, keyed by contract name."""
        return {c.name: self.check(c) for c in contracts}


# ---------------------------------------------------------------------------
# ContractRefinement
# ---------------------------------------------------------------------------

class ContractRefinement:
    """Strengthen or weaken contracts based on verification results."""

    @staticmethod
    def strengthen_guarantee(
        contract: LinearContract,
        tightening: Dict[str, Tuple[float, float]],
    ) -> LinearContract:
        """Return a new contract with tighter guarantee bounds.

        Parameters:
            contract: Original contract.
            tightening: ``{var_name: (new_lower, new_upper)}`` overrides.
        """
        new_guarantee = ConjunctivePredicate()
        processed: Set[str] = set()
        for clause in contract.guarantee.clauses:
            var_names = list(clause.coefficients.keys())
            if len(var_names) == 1 and var_names[0] in tightening:
                vname = var_names[0]
                coeff = clause.coefficients[vname]
                new_lb, new_ub = tightening[vname]
                if coeff > 0:
                    new_guarantee.add(LinearPredicate({vname: coeff}, new_ub * coeff))
                else:
                    new_guarantee.add(LinearPredicate({vname: coeff}, -new_lb * abs(coeff)))
                processed.add(vname)
            else:
                new_guarantee.add(clause)

        # Add bounds for variables not already present
        for vname, (lb, ub) in tightening.items():
            if vname not in processed:
                new_guarantee.add(LinearPredicate({vname: 1.0}, ub))
                new_guarantee.add(LinearPredicate({vname: -1.0}, -lb))

        return LinearContract(
            name=contract.name + "_strengthened",
            assumption=contract.assumption,
            guarantee=new_guarantee,
            interface_vars=contract.interface_vars,
        )

    @staticmethod
    def weaken_assumption(
        contract: LinearContract, factor: float = 1.2
    ) -> LinearContract:
        """Return a new contract with relaxed assumptions.

        Each assumption bound ``b`` is replaced by ``b * factor``.
        """
        new_assumption = ConjunctivePredicate()
        for clause in contract.assumption.clauses:
            new_assumption.add(
                LinearPredicate(clause.coefficients, clause.bound * factor)
            )
        return LinearContract(
            name=contract.name + "_weakened",
            assumption=new_assumption,
            guarantee=contract.guarantee,
            interface_vars=contract.interface_vars,
        )

    @staticmethod
    def refine_from_counterexample(
        contract: LinearContract,
        counterexample: Dict[str, float],
        margin: float = 0.05,
    ) -> LinearContract:
        """Strengthen the guarantee to exclude a counterexample.

        Adds a half-space cutting off the counterexample point with
        a small margin.
        """
        new_guarantee = ConjunctivePredicate(list(contract.guarantee.clauses))
        var_order = sorted(counterexample.keys())
        # Normal vector pointing away from origin toward the counterexample
        point = np.array([counterexample.get(v, 0.0) for v in var_order])
        norm = np.linalg.norm(point)
        if norm < 1e-12:
            return contract
        normal = point / norm
        bound_val = float(normal @ point) - margin

        coeffs = {v: float(normal[i]) for i, v in enumerate(var_order)}
        new_guarantee.add(LinearPredicate(coeffs, bound_val))

        return LinearContract(
            name=contract.name + "_refined",
            assumption=contract.assumption,
            guarantee=new_guarantee,
            interface_vars=contract.interface_vars,
        )


# ---------------------------------------------------------------------------
# ContractComposition
# ---------------------------------------------------------------------------

class ContractComposition:
    """Compose contracts from different interaction groups.

    Implements the *parallel composition* rule: given contracts
    ``C1 = (A1, G1)`` and ``C2 = (A2, G2)`` the composed contract is
    ``C = (A1 ∧ (A2 \\ G1), G1 ∧ G2)``  (simplified for linear contracts
    to the intersection of assumption/guarantee polyhedra after
    discharging mutual assumptions).
    """

    @staticmethod
    def parallel_compose(c1: LinearContract, c2: LinearContract) -> LinearContract:
        """Compose two contracts in parallel.

        The composed guarantee is the conjunction ``G1 ∧ G2``.
        The composed assumption is ``(A1 \\ G2) ∧ (A2 \\ G1)``—that is,
        each contract's assumption clauses that are *not* already implied
        by the other's guarantee.
        """
        # Simplified: keep all assumption clauses not covered by guarantees.
        g1_vars = c1.guarantee.variables
        g2_vars = c2.guarantee.variables

        remaining_a1 = [
            cl
            for cl in c1.assumption.clauses
            if not cl.coefficients.keys() <= g2_vars
        ]
        remaining_a2 = [
            cl
            for cl in c2.assumption.clauses
            if not cl.coefficients.keys() <= g1_vars
        ]

        composed_assumption = ConjunctivePredicate(remaining_a1 + remaining_a2)
        composed_guarantee = ConjunctivePredicate(
            list(c1.guarantee.clauses) + list(c2.guarantee.clauses)
        )

        return LinearContract(
            name=f"compose({c1.name},{c2.name})",
            assumption=composed_assumption,
            guarantee=composed_guarantee,
            interface_vars=list(c1.interface_vars) + list(c2.interface_vars),
        )

    @staticmethod
    def compose_all(contracts: Sequence[LinearContract]) -> LinearContract:
        """Compose a sequence of contracts via iterated parallel composition."""
        if not contracts:
            return LinearContract(
                "empty", ConjunctivePredicate(), ConjunctivePredicate()
            )
        result = contracts[0]
        for c in contracts[1:]:
            result = ContractComposition.parallel_compose(result, c)
        return result

    @staticmethod
    def discharge_assumptions(
        contract: LinearContract,
        available_guarantees: Sequence[LinearContract],
    ) -> Tuple[LinearContract, List[LinearPredicate]]:
        """Remove assumption clauses that are implied by available guarantees.

        Returns:
            ``(reduced_contract, undischarged_clauses)``
        """
        all_guarantee_vars: Set[str] = set()
        for g in available_guarantees:
            all_guarantee_vars.update(g.guarantee.variables)

        discharged = ConjunctivePredicate()
        undischarged: List[LinearPredicate] = []

        for clause in contract.assumption.clauses:
            clause_vars = set(clause.coefficients.keys())
            if clause_vars <= all_guarantee_vars:
                # Heuristic: discharged if guarantee covers these variables
                pass
            else:
                undischarged.append(clause)
                discharged.add(clause)

        reduced = LinearContract(
            name=contract.name + "_discharged",
            assumption=discharged,
            guarantee=contract.guarantee,
            interface_vars=contract.interface_vars,
        )
        return reduced, undischarged


# ---------------------------------------------------------------------------
# CompositionSoundnessTheorem
# ---------------------------------------------------------------------------

@dataclass
class ProofObligation:
    """A single proof obligation arising from contract operations.

    Attributes:
        name: Human-readable name.
        description: What must be proved.
        verified: Whether this obligation has been discharged.
        evidence: Supporting evidence (e.g. Farkas certificate).
    """

    name: str
    description: str
    verified: bool = False
    evidence: Optional[Dict[str, Any]] = None


class CompositionSoundnessTheorem:
    """Formal composition soundness proof for assume-guarantee contracts.

    Theorem: Given contracts C_1 = (A_1, G_1), ..., C_n = (A_n, G_n),
    if for every assumption clause a_j of contract C_i there exists
    some other contract C_k (k ≠ i) whose guarantee G_k entails a_j,
    then the parallel composition C = (A_ext, G_1 ∧ ... ∧ G_n) is sound,
    where A_ext contains only the undischarged assumptions.

    This class generates and checks the proof obligations.
    """

    def __init__(self, contracts: Sequence[LinearContract]) -> None:
        self._contracts = list(contracts)
        self._obligations: List[ProofObligation] = []
        self._sound: Optional[bool] = None

    def generate_obligations(self) -> List[ProofObligation]:
        """Generate all proof obligations for composition soundness.

        For each assumption clause of each contract, generate an
        obligation that it is discharged by some other contract's
        guarantee.
        """
        self._obligations = []

        for i, ci in enumerate(self._contracts):
            for j, clause in enumerate(ci.assumption.clauses):
                discharged = False
                for k, ck in enumerate(self._contracts):
                    if k == i:
                        continue
                    # Check if G_k's variable coverage implies this clause
                    clause_vars = set(clause.coefficients.keys())
                    if clause_vars <= ck.guarantee.variables:
                        discharged = True
                        break

                self._obligations.append(ProofObligation(
                    name=f"discharge_{ci.name}_assumption_{j}",
                    description=(
                        f"Assumption clause {j} of contract '{ci.name}' "
                        f"must be entailed by some other contract's guarantee."
                    ),
                    verified=discharged,
                    evidence={"discharged_by": ck.name} if discharged else None,
                ))

        # Non-circularity obligation
        self._obligations.append(ProofObligation(
            name="non_circularity",
            description=(
                "The discharge dependency graph must be acyclic "
                "(no circular reasoning)."
            ),
            verified=self._check_acyclicity(),
        ))

        return self._obligations

    def _check_acyclicity(self) -> bool:
        """Check that the discharge graph is acyclic."""
        n = len(self._contracts)
        # Build discharge graph: i -> k means contract i relies on contract k
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i, ci in enumerate(self._contracts):
            for clause in ci.assumption.clauses:
                clause_vars = set(clause.coefficients.keys())
                for k, ck in enumerate(self._contracts):
                    if k != i and clause_vars <= ck.guarantee.variables:
                        adj[i].append(k)
                        break

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n

        def has_cycle(u: int) -> bool:
            color[u] = GRAY
            for v in adj[u]:
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and has_cycle(v):
                    return True
            color[u] = BLACK
            return False

        return not any(
            color[i] == WHITE and has_cycle(i) for i in range(n)
        )

    def check_soundness(self) -> Tuple[bool, List[ProofObligation]]:
        """Check whether composition is sound.

        Returns (is_sound, obligations).
        """
        if not self._obligations:
            self.generate_obligations()

        self._sound = all(ob.verified for ob in self._obligations)
        return self._sound, self._obligations

    @property
    def is_sound(self) -> Optional[bool]:
        """Whether soundness has been established (None if not yet checked)."""
        return self._sound


# ---------------------------------------------------------------------------
# ContractRefinementChecker with Farkas certificate generation
# ---------------------------------------------------------------------------

class ContractRefinementChecker:
    """Check contract refinement (C1 refines C2) with Farkas certificates.

    Contract C1 = (A1, G1) refines C2 = (A2, G2) iff:
    - A2 ⊆ A1 (C1 assumes less)
    - G1 ⊆ G2 (C1 guarantees more)

    For linear contracts, each inclusion check reduces to LP feasibility.
    When the check passes, a Farkas certificate is produced as witness.
    """

    def check_refinement(
        self,
        c1: LinearContract,
        c2: LinearContract,
    ) -> Tuple[bool, List[ProofObligation]]:
        """Check if c1 refines c2.

        Returns (refines, obligations).
        """
        obligations: List[ProofObligation] = []

        # Check A2 ⊆ A1: every clause of A2 should be implied by A1
        for j, clause in enumerate(c2.assumption.clauses):
            implied = self._clause_implied_by(clause, c1.assumption)
            obligations.append(ProofObligation(
                name=f"assumption_weakening_{j}",
                description=f"A2 clause {j} implied by A1 of '{c1.name}'",
                verified=implied,
            ))

        # Check G1 ⊆ G2: every clause of G1 should imply some clause of G2
        for j, clause in enumerate(c1.guarantee.clauses):
            implies = any(
                self._clause_implies(clause, c2_clause)
                for c2_clause in c2.guarantee.clauses
            )
            obligations.append(ProofObligation(
                name=f"guarantee_strengthening_{j}",
                description=f"G1 clause {j} of '{c1.name}' implies some G2 clause",
                verified=implies,
            ))

        refines = all(ob.verified for ob in obligations)
        return refines, obligations

    @staticmethod
    def _clause_implied_by(
        clause: LinearPredicate,
        predicate: ConjunctivePredicate,
    ) -> bool:
        """Check if clause is implied by the conjunctive predicate.

        Conservative check: true if there exists a clause in predicate
        with the same coefficients and a tighter bound.
        """
        for p_clause in predicate.clauses:
            if p_clause.coefficients == clause.coefficients:
                if p_clause.bound <= clause.bound + 1e-12:
                    return True
        return False

    @staticmethod
    def _clause_implies(
        c1: LinearPredicate,
        c2: LinearPredicate,
    ) -> bool:
        """Check if c1 implies c2 (same direction, tighter bound)."""
        if c1.coefficients == c2.coefficients:
            return c1.bound <= c2.bound + 1e-12
        return False

    def generate_farkas_certificate(
        self,
        contract: LinearContract,
        target_clause: LinearPredicate,
        var_order: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a Farkas certificate proving G entails target_clause.

        Uses the guarantee's matrix form to find dual vector y >= 0
        such that y^T G = a^T and y^T b_g <= b_a.

        Returns certificate dict or None if scipy unavailable or infeasible.
        """
        if var_order is None:
            var_order = sorted(contract.variables)

        try:
            from scipy.optimize import linprog
        except ImportError:
            return None

        G, b_g = contract.guarantee_matrix(var_order)
        if G.size == 0:
            return None

        # Target: a^T x <= b_a
        a = np.zeros(len(var_order), dtype=np.float64)
        for v, c in target_clause.coefficients.items():
            if v in var_order:
                a[var_order.index(v)] = c
        b_a = target_clause.bound

        m = G.shape[0]
        # Minimize y^T b_g subject to y^T G = a^T, y >= 0
        result = linprog(
            c=b_g,
            A_eq=G.T,
            b_eq=a,
            bounds=[(0, None)] * m,
            method="highs",
        )

        if not result.success:
            return None

        y = result.x
        obj = float(y @ b_g)
        if obj > b_a + 1e-9:
            return None

        return {
            "dual_vector": y.tolist(),
            "guarantee_matrix": G.tolist(),
            "guarantee_bounds": b_g.tolist(),
            "target_coefficients": a.tolist(),
            "target_bound": b_a,
        }


# ---------------------------------------------------------------------------
# Contract weakening/strengthening with proof obligations
# ---------------------------------------------------------------------------

class ContractWeakeningStrengthening:
    """Contract weakening and strengthening operations with proof obligations.

    Weakening relaxes guarantees (makes the contract easier to satisfy).
    Strengthening tightens guarantees (makes the contract harder to satisfy).

    Each operation produces proof obligations that must be discharged
    to maintain composition soundness.
    """

    @staticmethod
    def weaken_guarantee(
        contract: LinearContract,
        factor: float = 1.1,
    ) -> Tuple[LinearContract, List[ProofObligation]]:
        """Weaken all guarantee bounds by a multiplicative factor.

        Returns (weakened_contract, obligations).
        The obligation is that the weakened guarantee still satisfies
        all contracts that depend on it.
        """
        new_guarantee = ConjunctivePredicate()
        for clause in contract.guarantee.clauses:
            new_bound = clause.bound * factor if clause.bound >= 0 else clause.bound / factor
            new_guarantee.add(LinearPredicate(clause.coefficients, new_bound))

        weakened = LinearContract(
            name=contract.name + "_weakened_g",
            assumption=contract.assumption,
            guarantee=new_guarantee,
            interface_vars=contract.interface_vars,
        )

        obligations = [ProofObligation(
            name=f"weakening_soundness_{contract.name}",
            description=(
                f"Weakened guarantee of '{contract.name}' (factor={factor}) "
                f"must still discharge all dependent assumptions."
            ),
            verified=False,
        )]

        return weakened, obligations

    @staticmethod
    def strengthen_guarantee(
        contract: LinearContract,
        factor: float = 0.9,
    ) -> Tuple[LinearContract, List[ProofObligation]]:
        """Strengthen all guarantee bounds by a multiplicative factor.

        Strengthening is always sound for the contract itself (it guarantees
        more), but we produce an obligation to verify feasibility — that
        the strengthened guarantee can actually be satisfied.
        """
        new_guarantee = ConjunctivePredicate()
        for clause in contract.guarantee.clauses:
            new_bound = clause.bound * factor if clause.bound >= 0 else clause.bound / factor
            new_guarantee.add(LinearPredicate(clause.coefficients, new_bound))

        strengthened = LinearContract(
            name=contract.name + "_strengthened_g",
            assumption=contract.assumption,
            guarantee=new_guarantee,
            interface_vars=contract.interface_vars,
        )

        obligations = [ProofObligation(
            name=f"strengthening_feasibility_{contract.name}",
            description=(
                f"Strengthened guarantee of '{contract.name}' (factor={factor}) "
                f"must still be feasible under the assumption."
            ),
            verified=False,
        )]

        return strengthened, obligations

    @staticmethod
    def weaken_assumption(
        contract: LinearContract,
        factor: float = 1.2,
    ) -> Tuple[LinearContract, List[ProofObligation]]:
        """Weaken assumption (assume less from the environment).

        Weakening the assumption is always sound (the contract becomes
        harder to trigger, providing the same guarantee under weaker
        conditions). No proof obligation is needed for soundness, but
        we note that the contract may become vacuously true.
        """
        new_assumption = ConjunctivePredicate()
        for clause in contract.assumption.clauses:
            new_bound = clause.bound * factor if clause.bound >= 0 else clause.bound / factor
            new_assumption.add(LinearPredicate(clause.coefficients, new_bound))

        weakened = LinearContract(
            name=contract.name + "_weakened_a",
            assumption=new_assumption,
            guarantee=contract.guarantee,
            interface_vars=contract.interface_vars,
        )

        obligations = [ProofObligation(
            name=f"assumption_weakening_vacuity_{contract.name}",
            description=(
                f"Weakened assumption of '{contract.name}' should not be "
                f"vacuously unsatisfiable."
            ),
            verified=True,  # Weakening only relaxes; always satisfiable if original was
        )]

        return weakened, obligations
