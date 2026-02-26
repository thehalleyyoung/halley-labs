"""Assume-guarantee compositional verification.

Verifies whole-system properties from per-group verification results
via formal composition rules.  Handles circular dependencies, generates
soundness proof artefacts, and supports adaptive decomposition when
contracts cannot be discharged.
"""

from __future__ import annotations

import enum
import itertools
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

from marace.decomposition.contracts import (
    CheckResult,
    ContractChecker,
    ContractComposition,
    ContractRefinement,
    ConjunctivePredicate,
    InterfaceVariable,
    LinearContract,
    LinearPredicate,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class VerificationStatus(enum.Enum):
    """Outcome of a verification attempt."""
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class GroupVerificationResult:
    """Per-group verification result."""
    group_id: str
    status: VerificationStatus
    contract: LinearContract
    counterexample: Optional[Dict[str, float]] = None
    iterations: int = 0
    message: str = ""


@dataclass
class CompositionResult:
    """System-level verification result with proof certificate.

    Attributes:
        status: Overall verification outcome.
        group_results: Per-group results that were composed.
        proof: Soundness proof artefact (if verified).
        undischarged: Assumption clauses that could not be discharged.
        message: Human-readable summary.
    """
    status: VerificationStatus
    group_results: List[GroupVerificationResult] = field(default_factory=list)
    proof: Optional["SoundnessProof"] = None
    undischarged: List[LinearPredicate] = field(default_factory=list)
    message: str = ""

    @property
    def is_verified(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    @property
    def is_refuted(self) -> bool:
        return self.status == VerificationStatus.REFUTED


# ---------------------------------------------------------------------------
# Composition rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionRule:
    """Formal rule for composing per-group results.

    Each rule specifies which groups it covers and the logical
    justification for the composition step.

    Attributes:
        name: Rule identifier (e.g. ``"parallel_AG"``).
        premise_groups: Groups whose results are used as premises.
        conclusion: Description of the system-level property established.
        requires_no_circular: Whether the rule requires acyclic dependencies.
    """

    name: str
    premise_groups: FrozenSet[str]
    conclusion: str
    requires_no_circular: bool = True

    def applicable(
        self, group_results: Dict[str, GroupVerificationResult]
    ) -> bool:
        """Check whether all premise groups have been verified."""
        for gid in self.premise_groups:
            if gid not in group_results:
                return False
            if group_results[gid].status != VerificationStatus.VERIFIED:
                return False
        return True


# ---------------------------------------------------------------------------
# Soundness proof artefact
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    """A single step in the composition proof."""
    step_id: int
    rule: CompositionRule
    group_results_used: List[str]
    contracts_discharged: List[str]
    justification: str


@dataclass
class SoundnessProof:
    """Proof certificate certifying composition soundness.

    The proof is a sequence of steps, each applying a composition rule
    to discharge some assumptions.  The proof is *complete* when all
    external assumptions have been discharged.
    """

    steps: List[ProofStep] = field(default_factory=list)
    remaining_assumptions: List[LinearPredicate] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.remaining_assumptions) == 0

    def add_step(self, step: ProofStep) -> None:
        self.steps.append(step)

    def summary(self) -> str:
        lines = [f"Soundness proof with {len(self.steps)} step(s):"]
        for s in self.steps:
            lines.append(
                f"  Step {s.step_id}: {s.rule.name} "
                f"using groups {s.group_results_used}"
            )
        if self.remaining_assumptions:
            lines.append(
                f"  WARNING: {len(self.remaining_assumptions)} "
                "assumption(s) not discharged"
            )
        else:
            lines.append("  All assumptions discharged — proof complete.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Circular dependency resolution
# ---------------------------------------------------------------------------

class CircularDependencyResolver:
    """Handle circular assume-guarantee dependencies.

    When group A assumes a property that group B guarantees, and group B
    assumes a property that group A guarantees, a *circular dependency*
    arises.  We resolve it via iterative fixed-point: start with weak
    assumptions, verify each group, strengthen guarantees, and repeat
    until convergence or a bound is reached.
    """

    def __init__(self, max_iterations: int = 20, convergence_tol: float = 1e-6) -> None:
        self._max_iter = max_iterations
        self._tol = convergence_tol

    def detect_cycles(
        self,
        dependency_graph: Dict[str, Set[str]],
    ) -> List[List[str]]:
        """Find all simple cycles in the dependency graph.

        Parameters:
            dependency_graph: ``{group_id: set of groups it depends on}``.

        Returns:
            List of cycles, each a list of group IDs.
        """
        import networkx as nx

        G = nx.DiGraph()
        for node, deps in dependency_graph.items():
            for d in deps:
                G.add_edge(node, d)
        return [list(c) for c in nx.simple_cycles(G)]

    def resolve(
        self,
        contracts: Dict[str, LinearContract],
        dependency_graph: Dict[str, Set[str]],
        verify_group: Callable[[str, LinearContract], GroupVerificationResult],
    ) -> Tuple[Dict[str, GroupVerificationResult], bool]:
        """Iteratively resolve circular dependencies.

        Parameters:
            contracts: Current contracts keyed by group ID.
            dependency_graph: ``{group_id: set of groups it depends on}``.
            verify_group: Callable that verifies a single group given its
                          contract and returns the result.

        Returns:
            ``(results, converged)`` — per-group results and whether the
            iterative process converged.
        """
        results: Dict[str, GroupVerificationResult] = {}
        current_contracts = dict(contracts)
        groups = list(contracts.keys())

        for iteration in range(self._max_iter):
            all_verified = True
            changed = False

            for gid in groups:
                result = verify_group(gid, current_contracts[gid])
                results[gid] = result

                if result.status != VerificationStatus.VERIFIED:
                    all_verified = False
                    # Try weakening the assumption
                    weakened = ContractRefinement.weaken_assumption(
                        current_contracts[gid], factor=1.1
                    )
                    if weakened.name != current_contracts[gid].name:
                        changed = True
                    current_contracts[gid] = weakened

            if all_verified:
                return results, True

            if not changed:
                break

        return results, False


# ---------------------------------------------------------------------------
# Group merging
# ---------------------------------------------------------------------------

class GroupMergingStrategy:
    """Decide when and how to merge interaction groups.

    When contracts between two groups cannot be discharged, merging them
    into a single group eliminates the interface and allows monolithic
    verification of the combined group.
    """

    def __init__(self, max_group_size: int = 6) -> None:
        self._max_group_size = max_group_size

    def should_merge(
        self,
        group_a: str,
        group_b: str,
        group_sizes: Dict[str, int],
        undischarged_count: int,
    ) -> bool:
        """Heuristic: merge if combined size is tractable and there
        are undischarged obligations.
        """
        combined = group_sizes.get(group_a, 1) + group_sizes.get(group_b, 1)
        return undischarged_count > 0 and combined <= self._max_group_size

    def select_merge_candidates(
        self,
        group_results: Dict[str, GroupVerificationResult],
        contracts: Dict[Tuple[str, str], LinearContract],
        group_sizes: Dict[str, int],
    ) -> List[Tuple[str, str]]:
        """Select pairs of groups that should be merged.

        Returns pairs sorted by combined size (smallest first).
        """
        checker = ContractChecker()
        candidates: List[Tuple[str, str, int]] = []

        for (ga, gb), contract in contracts.items():
            result = checker.check(contract)
            if not result.satisfied:
                combined = group_sizes.get(ga, 1) + group_sizes.get(gb, 1)
                if combined <= self._max_group_size:
                    candidates.append((ga, gb, combined))

        candidates.sort(key=lambda x: x[2])
        return [(a, b) for a, b, _ in candidates]

    @staticmethod
    def merge_groups(
        partition: Dict[str, FrozenSet[str]],
        group_a: str,
        group_b: str,
    ) -> Dict[str, FrozenSet[str]]:
        """Merge two groups in the partition.

        The merged group takes the name ``"{group_a}+{group_b}"``.
        """
        merged_agents = partition[group_a] | partition[group_b]
        new_partition = {
            k: v for k, v in partition.items() if k not in (group_a, group_b)
        }
        new_partition[f"{group_a}+{group_b}"] = merged_agents
        return new_partition


# ---------------------------------------------------------------------------
# Assume-Guarantee verifier
# ---------------------------------------------------------------------------

class AssumeGuaranteeVerifier:
    """Verify whole-system properties from per-group verification results.

    Orchestrates the full assume-guarantee workflow:
    1. Verify each group under its contract.
    2. Check that all inter-group contracts are discharged.
    3. Handle circular dependencies if present.
    4. Produce a composition result with proof certificate.
    """

    def __init__(
        self,
        max_circular_iterations: int = 20,
        max_group_size: int = 6,
    ) -> None:
        self._checker = ContractChecker()
        self._resolver = CircularDependencyResolver(max_circular_iterations)
        self._merger = GroupMergingStrategy(max_group_size)

    def verify(
        self,
        group_contracts: Dict[str, LinearContract],
        inter_group_contracts: Dict[Tuple[str, str], LinearContract],
        verify_group_fn: Callable[[str, LinearContract], GroupVerificationResult],
        dependency_graph: Optional[Dict[str, Set[str]]] = None,
    ) -> CompositionResult:
        """Run the assume-guarantee verification pipeline.

        Parameters:
            group_contracts: Per-group contracts ``{group_id: contract}``.
            inter_group_contracts: Contracts on interfaces ``{(src, tgt): contract}``.
            verify_group_fn: Callable that verifies a single group.
            dependency_graph: Dependency structure between groups.

        Returns:
            :class:`CompositionResult` with proof certificate.
        """
        # Step 1: Verify each group
        group_results: Dict[str, GroupVerificationResult] = {}
        for gid, contract in group_contracts.items():
            group_results[gid] = verify_group_fn(gid, contract)

        # Step 2: Check inter-group contracts
        undischarged: List[LinearPredicate] = []
        contract_check_results: Dict[str, CheckResult] = {}
        for (src, tgt), contract in inter_group_contracts.items():
            check = self._checker.check(contract)
            contract_check_results[f"{src}->{tgt}"] = check
            if not check.satisfied:
                undischarged.extend(contract.assumption.clauses)

        # Step 3: Handle circular dependencies
        if dependency_graph is not None:
            cycles = self._resolver.detect_cycles(dependency_graph)
            if cycles:
                group_results, converged = self._resolver.resolve(
                    group_contracts, dependency_graph, verify_group_fn
                )
                if not converged:
                    return CompositionResult(
                        status=VerificationStatus.UNKNOWN,
                        group_results=list(group_results.values()),
                        undischarged=undischarged,
                        message=f"Circular dependencies not resolved after "
                                f"max iterations. Cycles: {cycles}",
                    )

        # Step 4: Build proof
        all_verified = all(
            r.status == VerificationStatus.VERIFIED
            for r in group_results.values()
        )
        all_contracts_satisfied = all(
            cr.satisfied for cr in contract_check_results.values()
        )

        proof = SoundnessProof(remaining_assumptions=undischarged)

        if all_verified and all_contracts_satisfied:
            proof.add_step(
                ProofStep(
                    step_id=0,
                    rule=CompositionRule(
                        name="parallel_AG",
                        premise_groups=frozenset(group_contracts.keys()),
                        conclusion="System race-free under A/G composition",
                    ),
                    group_results_used=list(group_contracts.keys()),
                    contracts_discharged=list(contract_check_results.keys()),
                    justification=(
                        "All groups verified under contracts; "
                        "all inter-group contracts satisfied."
                    ),
                )
            )
            return CompositionResult(
                status=VerificationStatus.VERIFIED,
                group_results=list(group_results.values()),
                proof=proof,
                message="System verified via assume-guarantee composition.",
            )

        # Some groups failed or contracts not discharged
        any_refuted = any(
            r.status == VerificationStatus.REFUTED
            for r in group_results.values()
        )

        return CompositionResult(
            status=VerificationStatus.REFUTED if any_refuted else VerificationStatus.UNKNOWN,
            group_results=list(group_results.values()),
            proof=proof,
            undischarged=undischarged,
            message=self._build_failure_message(group_results, contract_check_results),
        )

    @staticmethod
    def _build_failure_message(
        group_results: Dict[str, GroupVerificationResult],
        contract_checks: Dict[str, CheckResult],
    ) -> str:
        parts: List[str] = []
        for gid, r in group_results.items():
            if r.status != VerificationStatus.VERIFIED:
                parts.append(f"Group {gid}: {r.status.value} — {r.message}")
        for cname, cr in contract_checks.items():
            if not cr.satisfied:
                parts.append(f"Contract {cname}: unsatisfied — {cr.message}")
        return "; ".join(parts) if parts else "Unknown failure"


# ---------------------------------------------------------------------------
# Adaptive decomposition
# ---------------------------------------------------------------------------

class AdaptiveDecomposition:
    """Iteratively refine decomposition based on verification results.

    If verification fails due to undischarged contracts, the decomposition
    is refined by:
    1. Merging groups that cannot discharge their interface contracts.
    2. Re-generating contracts for the new partition.
    3. Re-verifying.

    The process terminates when either all contracts are discharged
    (success), the partition collapses to a single group (monolithic
    fallback), or a maximum number of refinement rounds is reached.
    """

    def __init__(
        self,
        max_refinements: int = 10,
        max_group_size: int = 6,
    ) -> None:
        self._max_refinements = max_refinements
        self._merger = GroupMergingStrategy(max_group_size)
        self._verifier = AssumeGuaranteeVerifier(max_group_size=max_group_size)

    def run(
        self,
        initial_partition: Dict[str, FrozenSet[str]],
        contract_generator_fn: Callable[
            [Dict[str, FrozenSet[str]]],
            Tuple[Dict[str, LinearContract], Dict[Tuple[str, str], LinearContract]],
        ],
        verify_group_fn: Callable[[str, LinearContract], GroupVerificationResult],
    ) -> Tuple[CompositionResult, Dict[str, FrozenSet[str]]]:
        """Run the adaptive decomposition loop.

        Parameters:
            initial_partition: ``{group_id: frozenset of agent_ids}``.
            contract_generator_fn: Given a partition, generates per-group
                and inter-group contracts.
            verify_group_fn: Verifies a single group under a contract.

        Returns:
            ``(result, final_partition)``
        """
        partition = dict(initial_partition)

        for round_idx in range(self._max_refinements):
            # Generate contracts for current partition
            group_contracts, inter_group_contracts = contract_generator_fn(partition)

            # Build dependency graph from inter-group contracts
            dep_graph: Dict[str, Set[str]] = {gid: set() for gid in partition}
            for (src, tgt) in inter_group_contracts:
                dep_graph.setdefault(tgt, set()).add(src)

            # Verify
            result = self._verifier.verify(
                group_contracts,
                inter_group_contracts,
                verify_group_fn,
                dep_graph,
            )

            if result.is_verified:
                return result, partition

            # Identify merge candidates
            group_sizes = {gid: len(agents) for gid, agents in partition.items()}
            merge_pairs = self._merger.select_merge_candidates(
                {r.group_id: r for r in result.group_results},
                inter_group_contracts,
                group_sizes,
            )

            if not merge_pairs:
                # No more merges possible — return best effort
                return result, partition

            # Merge the first candidate pair
            ga, gb = merge_pairs[0]
            partition = GroupMergingStrategy.merge_groups(partition, ga, gb)

            # If collapsed to single group, done
            if len(partition) <= 1:
                group_contracts, inter_group_contracts = contract_generator_fn(partition)
                result = self._verifier.verify(
                    group_contracts,
                    inter_group_contracts,
                    verify_group_fn,
                )
                return result, partition

        # Exhausted refinement budget
        group_contracts, inter_group_contracts = contract_generator_fn(partition)
        result = self._verifier.verify(
            group_contracts, inter_group_contracts, verify_group_fn
        )
        return result, partition
