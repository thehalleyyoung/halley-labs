"""
Tinelli-Zarba Theory Combination for Finite-Domain Theories.

Implements the Tinelli-Zarba (JAR 2005) extension of Nelson-Oppen theory
combination for non-stably-infinite sorts.  Standard Nelson-Oppen requires
every sort to be stably-infinite (i.e., have infinitely many elements in
every model).  The broadcast and stride theories operate over Dim ⊆ ℤ_≥1
which is stably-infinite, so they combine classically.  However:

  - T_device has 5 elements: {CPU, CUDA_0, CUDA_1, CUDA_2, CUDA_3}
  - T_phase  has 2 elements: {TRAIN, EVAL}  (Bool: True/False)

These finite domains violate the Nelson-Oppen precondition.  The
Tinelli-Zarba method restores completeness by enumerating *arrangements*
— all possible equivalence classes over the shared variables — and
checking that at least one arrangement is consistent across all theories.

For k shared variables over a domain of size n, the number of
arrangements is bounded by the Stirling number S(k, min(k, n)).
With typical small k (≤ 4 shared device vars, ≤ 2 shared phase vars)
this is tractable.

Algorithm
---------
1. Collect shared variables: variables that appear in more than one
   theory's constraint set.
2. For each finite-domain sort, enumerate all arrangements of the
   shared variables of that sort (all partitions into equivalence
   classes, with at most n classes for an n-element domain).
3. For each arrangement, assert the corresponding equalities and
   disequalities in *each* theory's solver (via push/pop).
4. Check satisfiability in each solver under the arrangement.
5. If any arrangement is consistent in all solvers simultaneously,
   the combined theory is satisfiable.
6. If no arrangement works, the combination is unsatisfiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import product as iter_product
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Arrangement enumeration
# ═══════════════════════════════════════════════════════════════════════════


def _enumerate_partitions(
    n: int, max_classes: int
) -> List[List[int]]:
    """Enumerate all partitions of n elements into at most max_classes classes.

    Returns a list of assignments where assignment[i] is the class
    index (0-based) for element i.  Each assignment uses class indices
    in canonical order (first element is always class 0, second element
    is class 0 or 1, etc.) to avoid counting equivalent partitions.

    This generates *restricted growth strings* — the standard method
    for enumerating set partitions without duplicates.

    Args:
        n: Number of elements to partition.
        max_classes: Maximum number of equivalence classes allowed
                     (domain cardinality for finite sorts).

    Returns:
        List of partition assignments.
    """
    if n == 0:
        return [[]]

    results: List[List[int]] = []

    def _backtrack(pos: int, assignment: List[int], next_class: int) -> None:
        if pos == n:
            results.append(list(assignment))
            return
        # Element `pos` can go into any existing class [0, next_class)
        # or open a new class (if we haven't hit max_classes).
        for c in range(min(next_class + 1, max_classes)):
            assignment.append(c)
            _backtrack(
                pos + 1,
                assignment,
                max(next_class, c + 1),
            )
            assignment.pop()

    _backtrack(0, [], 0)
    return results


def _partition_to_equalities_disequalities(
    variables: List, assignment: List[int]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Convert a partition assignment to equality/disequality index pairs.

    Args:
        variables: The variable list (used only for length).
        assignment: Class assignment for each variable.

    Returns:
        (equalities, disequalities): pairs of variable indices.
    """
    n = len(variables)
    equalities: List[Tuple[int, int]] = []
    disequalities: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if assignment[i] == assignment[j]:
                equalities.append((i, j))
            else:
                disequalities.append((i, j))
    return equalities, disequalities


# ═══════════════════════════════════════════════════════════════════════════
# 2. TheorySolver — lightweight wrapper for a Z3 solver with metadata
# ═══════════════════════════════════════════════════════════════════════════


class DomainKind(Enum):
    """Classification of a theory's sort for combination purposes."""

    STABLY_INFINITE = "stably_infinite"
    FINITE = "finite"


@dataclass
class TheorySolver:
    """A theory solver participating in theory combination.

    Attributes:
        name: Human-readable theory name (e.g. "broadcast", "device").
        solver: The Z3 Solver instance for this theory.
        domain_kind: Whether the theory's sort is finite or stably-infinite.
        domain_size: For finite domains, the number of elements.
        shared_vars: Z3 variables shared with other theories.
    """

    name: str
    solver: "z3.Solver"
    domain_kind: DomainKind
    domain_size: Optional[int] = None
    shared_vars: List = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.domain_kind == DomainKind.FINITE:
            if self.domain_size is None or self.domain_size < 1:
                raise ValueError(
                    f"Theory '{self.name}': finite domain requires "
                    f"domain_size >= 1, got {self.domain_size}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 3. CombinationResult
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CombinationResult:
    """Result of theory combination consistency check.

    Attributes:
        is_consistent: True if a consistent arrangement exists.
        satisfying_arrangement: The arrangement that worked (if any).
            Maps variable index pairs to equality/disequality.
        inconsistencies: List of (theory_name, arrangement) pairs that
            were individually inconsistent.
        total_arrangements_checked: How many arrangements were tried.
    """

    is_consistent: bool
    satisfying_arrangement: Optional[Dict[str, List[int]]] = None
    inconsistencies: List[Tuple[str, List[int]]] = field(
        default_factory=list
    )
    total_arrangements_checked: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. TheoryCombination — main combination engine
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    class TheoryCombination:
        """Tinelli-Zarba theory combination for mixed finite/infinite domains.

        Usage::

            combo = TheoryCombination()
            combo.add_theory(TheorySolver(
                name="broadcast",
                solver=broadcast_solver,
                domain_kind=DomainKind.STABLY_INFINITE,
                shared_vars=[dim_x, dim_y],
            ))
            combo.add_theory(TheorySolver(
                name="device",
                solver=device_solver,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[dev_a, dev_b],
            ))
            result = combo.check_combination()
        """

        def __init__(self) -> None:
            self._theories: List[TheorySolver] = []

        def add_theory(self, theory: TheorySolver) -> None:
            """Register a theory solver for combination."""
            self._theories.append(theory)

        @property
        def theories(self) -> List[TheorySolver]:
            return list(self._theories)

        def _get_finite_theories(self) -> List[TheorySolver]:
            """Get all finite-domain theories."""
            return [
                t
                for t in self._theories
                if t.domain_kind == DomainKind.FINITE
            ]

        def _check_solver_with_arrangement(
            self,
            solver: "z3.Solver",
            variables: List["z3.ExprRef"],
            equalities: List[Tuple[int, int]],
            disequalities: List[Tuple[int, int]],
        ) -> bool:
            """Check if a solver is SAT under a given arrangement.

            Uses push/pop to temporarily assert equalities and
            disequalities, then checks satisfiability.
            """
            solver.push()
            try:
                for i, j in equalities:
                    solver.add(variables[i] == variables[j])
                for i, j in disequalities:
                    solver.add(variables[i] != variables[j])
                result = solver.check()
                return result == z3.sat
            finally:
                solver.pop()

        def check_combination(self) -> CombinationResult:
            """Run Tinelli-Zarba arrangement enumeration.

            For each finite-domain sort, enumerates all possible
            arrangements of the shared variables of that sort and
            checks whether all theories agree on at least one.

            The key correctness property: shared variables get a SINGLE
            arrangement that is checked against ALL theories simultaneously.
            This ensures theories cannot disagree on equalities.

            For stably-infinite theories, classical Nelson-Oppen
            applies: we only need equality propagation (which Z3
            handles internally via congruence closure).

            Returns:
                CombinationResult with consistency verdict.
            """
            if not self._theories:
                return CombinationResult(is_consistent=True)

            # Phase 1: check each theory individually
            for theory in self._theories:
                if theory.solver.check() != z3.sat:
                    return CombinationResult(
                        is_consistent=False,
                        inconsistencies=[
                            (theory.name, [])
                        ],
                    )

            finite_theories = self._get_finite_theories()

            # If no finite theories, classical Nelson-Oppen suffices.
            # Z3 handles this internally — just verify all SAT.
            if not finite_theories:
                return CombinationResult(
                    is_consistent=True,
                    total_arrangements_checked=0,
                )

            # Phase 2: Tinelli-Zarba for finite-domain theories.
            # Collect unique shared variables per sort, enumerate
            # arrangements once per sort, and check ALL theories
            # against the same arrangement.

            sort_groups = self._collect_shared_var_sort_groups(
                finite_theories
            )
            all_arrangements = self._enumerate_sort_arrangements(
                sort_groups
            )

            total_checked = 0
            for arrangement in all_arrangements:
                total_checked += 1
                all_consistent = True

                for theory in self._theories:
                    # Determine which shared vars this theory uses
                    theory_var_ids = {
                        v.get_id() for v in theory.shared_vars
                    }
                    equalities: List[Tuple[int, int]] = []
                    disequalities: List[Tuple[int, int]] = []
                    relevant_vars: List[z3.ExprRef] = []

                    for sort_key, (variables, assignment) in (
                        arrangement.items()
                    ):
                        # Map from sort-level indices to theory-level vars
                        var_index_map: Dict[int, int] = {}
                        for sort_idx, var in enumerate(variables):
                            if var.get_id() in theory_var_ids:
                                local_idx = len(relevant_vars)
                                relevant_vars.append(var)
                                var_index_map[sort_idx] = local_idx

                        if len(var_index_map) < 2:
                            continue

                        # Apply the sort-level arrangement to this
                        # theory's subset of the shared variables
                        sort_indices = sorted(var_index_map.keys())
                        for ii in range(len(sort_indices)):
                            for jj in range(ii + 1, len(sort_indices)):
                                si = sort_indices[ii]
                                sj = sort_indices[jj]
                                li = var_index_map[si]
                                lj = var_index_map[sj]
                                if assignment[si] == assignment[sj]:
                                    equalities.append((li, lj))
                                else:
                                    disequalities.append((li, lj))

                    if not relevant_vars:
                        # No shared vars — just check base SAT
                        if theory.solver.check() != z3.sat:
                            all_consistent = False
                            break
                        continue

                    if not self._check_solver_with_arrangement(
                        theory.solver,
                        relevant_vars,
                        equalities,
                        disequalities,
                    ):
                        all_consistent = False
                        break

                if all_consistent:
                    return CombinationResult(
                        is_consistent=True,
                        satisfying_arrangement={
                            k: v[1]
                            for k, v in arrangement.items()
                        },
                        total_arrangements_checked=total_checked,
                    )

            return CombinationResult(
                is_consistent=False,
                total_arrangements_checked=total_checked,
            )

        def _collect_shared_var_sort_groups(
            self, finite_theories: List[TheorySolver]
        ) -> Dict[str, Tuple[List["z3.ExprRef"], int]]:
            """Group shared variables by sort (theory name is proxy for sort).

            Returns a dict mapping sort_key -> (unique_vars, domain_size).
            Variables that appear in multiple theories of the same sort
            are deduplicated.
            """
            # Collect unique vars across all theories per sort, keyed
            # by domain_size as a sort proxy.
            sort_map: Dict[
                int, Tuple[List["z3.ExprRef"], Set[int]]
            ] = {}

            for theory in finite_theories:
                ds = theory.domain_size
                assert ds is not None
                if ds not in sort_map:
                    sort_map[ds] = ([], set())
                var_list, seen_ids = sort_map[ds]
                for v in theory.shared_vars:
                    vid = v.get_id()
                    if vid not in seen_ids:
                        seen_ids.add(vid)
                        var_list.append(v)

            return {
                f"sort_{ds}": (vars_list, ds)
                for ds, (vars_list, _) in sort_map.items()
                if len(vars_list) > 0
            }

        def _enumerate_sort_arrangements(
            self,
            sort_groups: Dict[str, Tuple[List["z3.ExprRef"], int]],
        ) -> List[Dict[str, Tuple[List["z3.ExprRef"], List[int]]]]:
            """Enumerate cross-product of arrangements across sorts.

            Returns list of dicts mapping sort_key -> (variables, assignment).
            """
            if not sort_groups:
                return [{}]

            per_sort: List[
                Tuple[str, List["z3.ExprRef"], List[List[int]]]
            ] = []
            for sort_key, (variables, domain_size) in sort_groups.items():
                n_vars = len(variables)
                if n_vars == 0:
                    continue
                partitions = _enumerate_partitions(n_vars, domain_size)
                per_sort.append((sort_key, variables, partitions))

            if not per_sort:
                return [{}]

            # Cross-product across sorts
            keys = [key for key, _, _ in per_sort]
            var_lists = [vs for _, vs, _ in per_sort]
            partition_lists = [ps for _, _, ps in per_sort]

            results = []
            for combo in iter_product(*partition_lists):
                entry = {}
                for key, vs, assign in zip(keys, var_lists, combo):
                    entry[key] = (vs, list(assign))
                results.append(entry)
            return results

        def verify_theory_combination_consistency(
            self,
        ) -> CombinationResult:
            """Verify theory combination consistency.

            Convenience method that runs the full Tinelli-Zarba check
            and logs results. Call after all constraints are registered
            but before relying on the combined result.

            Returns:
                CombinationResult with full diagnostic information.
            """
            result = self.check_combination()

            if result.is_consistent:
                logger.info(
                    "Theory combination consistent "
                    "(checked %d arrangements)",
                    result.total_arrangements_checked,
                )
                if result.satisfying_arrangement:
                    logger.debug(
                        "Satisfying arrangement: %s",
                        result.satisfying_arrangement,
                    )
            else:
                logger.warning(
                    "Theory combination INCONSISTENT after checking "
                    "%d arrangements",
                    result.total_arrangements_checked,
                )
                for name, assign in result.inconsistencies:
                    logger.warning(
                        "  Theory '%s' individually unsat "
                        "(assignment: %s)",
                        name,
                        assign,
                    )

            return result

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Multi-sort combination for the full system
    # ═══════════════════════════════════════════════════════════════════════

    class TensorTheoryCombination(TheoryCombination):
        """Specialized combination for the tensor analysis theories.

        Provides factory methods to set up the standard four-theory
        combination: broadcast (Dim, ∞), stride (Dim, ∞), device
        (Device, 5), phase (Bool, 2).

        Usage::

            combo = TensorTheoryCombination()
            combo.add_broadcast_theory(solver, shared_dim_vars)
            combo.add_stride_theory(solver, shared_dim_vars)
            combo.add_device_theory(solver, shared_dev_vars)
            combo.add_phase_theory(solver, shared_phase_vars)
            result = combo.verify_theory_combination_consistency()
        """

        def add_broadcast_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the broadcast theory (stably-infinite, Dim sort)."""
            self.add_theory(
                TheorySolver(
                    name="broadcast",
                    solver=solver,
                    domain_kind=DomainKind.STABLY_INFINITE,
                    shared_vars=shared_vars or [],
                )
            )

        def add_stride_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the stride theory (stably-infinite, Dim sort)."""
            self.add_theory(
                TheorySolver(
                    name="stride",
                    solver=solver,
                    domain_kind=DomainKind.STABLY_INFINITE,
                    shared_vars=shared_vars or [],
                )
            )

        def add_device_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the device theory (finite, 5-element Device sort)."""
            self.add_theory(
                TheorySolver(
                    name="device",
                    solver=solver,
                    domain_kind=DomainKind.FINITE,
                    domain_size=5,
                    shared_vars=shared_vars or [],
                )
            )

        def add_phase_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the phase theory (finite, 2-element Bool sort)."""
            self.add_theory(
                TheorySolver(
                    name="phase",
                    solver=solver,
                    domain_kind=DomainKind.FINITE,
                    domain_size=2,
                    shared_vars=shared_vars or [],
                )
            )
