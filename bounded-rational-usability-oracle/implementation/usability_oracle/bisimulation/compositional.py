"""
usability_oracle.bisimulation.compositional — Compositional bisimulation.

Implements bisimulation up-to techniques and compositional verification
methods that allow bisimulation proofs to be constructed modularly from
sub-components.

Key capabilities:
  - Bisimulation up to context
  - Bisimulation up to bisimilarity
  - Compositional verification of parallel composition
  - Congruence checking
  - Modular state-space reduction
  - Application: bisimulation of composed UI components

References
----------
- Sangiorgi, D. (1998). On the bisimulation proof method. *MSCS*.
- Pous, D. & Sangiorgi, D. (2011). Enhancements of the bisimulation
  proof method. *Advanced Topics in Bisimulation and Coinduction*.
- Milner, R. (1989). *Communication and Concurrency*.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bisimulation.models import (
    BisimulationResult,
    Partition,
)
from usability_oracle.mdp.models import MDP, Action, State, Transition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bisimulation up-to techniques
# ---------------------------------------------------------------------------

@dataclass
class BisimulationUpTo:
    """Bisimulation up-to techniques for accelerated convergence.

    Instead of requiring R to be a bisimulation directly, we check that
    R is a "bisimulation up to f" for some closure function f.  This
    allows smaller candidate relations to be used while still proving
    the same equivalences.

    Supported techniques:
      - up to bisimilarity: R ⊆ ~R~ where ~ is already-known bisimilarity
      - up to context: R ⊆ C[R] where C is a compatible context closure
      - up to union: R ∪ R⁻¹ is checked

    Parameters
    ----------
    technique : str
        One of ``"bisimilarity"``, ``"context"``, ``"union"``.
    max_iterations : int
        Maximum refinement iterations.
    tolerance : float
        Numerical tolerance for probability comparison.
    """

    technique: str = "bisimilarity"
    max_iterations: int = 500
    tolerance: float = 1e-10

    def verify(
        self,
        mdp: MDP,
        candidate: list[tuple[str, str]],
    ) -> bool:
        """Verify that a candidate relation is a bisimulation up-to.

        Parameters
        ----------
        mdp : MDP
        candidate : list[tuple[str, str]]
            Pairs (s₁, s₂) in the candidate relation.

        Returns
        -------
        bool
            True if the candidate is a valid bisimulation up-to.
        """
        state_ids = sorted(mdp.states.keys())
        state_idx = {sid: i for i, sid in enumerate(state_ids)}
        n = len(state_ids)

        # Build candidate relation matrix
        R = np.zeros((n, n), dtype=bool)
        for s1, s2 in candidate:
            i, j = state_idx.get(s1), state_idx.get(s2)
            if i is not None and j is not None:
                R[i, j] = True
                R[j, i] = True  # symmetric

        if self.technique == "bisimilarity":
            return self._verify_up_to_bisimilarity(mdp, R, state_ids, state_idx)
        elif self.technique == "context":
            return self._verify_up_to_context(mdp, R, state_ids, state_idx)
        elif self.technique == "union":
            return self._verify_up_to_union(mdp, R, state_ids, state_idx)
        else:
            raise ValueError(f"Unknown technique: {self.technique!r}")

    def _verify_up_to_bisimilarity(
        self,
        mdp: MDP,
        R: np.ndarray,
        state_ids: list[str],
        state_idx: dict[str, int],
    ) -> bool:
        """Verify R is a bisimulation up to bisimilarity.

        Check: for every (s₁, s₂) ∈ R and action a, the successor
        distributions match up to the closure ~R~ where ~ is the
        reflexive-transitive closure of R.
        """
        n = len(state_ids)
        # Compute transitive closure of R (the "bisimilarity" closure)
        closure = self._transitive_closure(R)

        for i in range(n):
            for j in range(n):
                if not R[i, j] or i == j:
                    continue

                s1, s2 = state_ids[i], state_ids[j]
                actions_1 = set(mdp.get_actions(s1))
                actions_2 = set(mdp.get_actions(s2))

                if actions_1 != actions_2:
                    return False

                for aid in actions_1:
                    if not self._transitions_match_up_to(
                        s1, s2, aid, mdp, closure, state_ids, state_idx,
                    ):
                        return False
        return True

    def _verify_up_to_context(
        self,
        mdp: MDP,
        R: np.ndarray,
        state_ids: list[str],
        state_idx: dict[str, int],
    ) -> bool:
        """Verify R is a bisimulation up to context.

        The context closure extends R by allowing state pairs that are
        related by a shared transition structure.
        """
        n = len(state_ids)
        # Context closure: if (s1, s2) ∈ R and s1→t1, s2→t2 with same
        # action, then (t1, t2) is in the closure
        closure = R.copy()

        for _ in range(self.max_iterations):
            old_closure = closure.copy()
            for i in range(n):
                for j in range(n):
                    if not closure[i, j]:
                        continue
                    s1, s2 = state_ids[i], state_ids[j]
                    for aid in mdp.get_actions(s1):
                        for t1, p1, _ in mdp.get_transitions(s1, aid):
                            for t2, p2, _ in mdp.get_transitions(s2, aid):
                                if abs(p1 - p2) < self.tolerance:
                                    ti = state_idx.get(t1)
                                    tj = state_idx.get(t2)
                                    if ti is not None and tj is not None:
                                        closure[ti, tj] = True
                                        closure[tj, ti] = True
            if np.array_equal(closure, old_closure):
                break

        # Now verify R is a bisimulation up to this closure
        return self._verify_up_to_bisimilarity(mdp, R, state_ids, state_idx)

    def _verify_up_to_union(
        self,
        mdp: MDP,
        R: np.ndarray,
        state_ids: list[str],
        state_idx: dict[str, int],
    ) -> bool:
        """Verify R ∪ R⁻¹ is a bisimulation up to bisimilarity."""
        R_union = R | R.T
        return self._verify_up_to_bisimilarity(
            mdp, R_union, state_ids, state_idx,
        )

    def _transitions_match_up_to(
        self,
        s1: str,
        s2: str,
        action: str,
        mdp: MDP,
        closure: np.ndarray,
        state_ids: list[str],
        state_idx: dict[str, int],
    ) -> bool:
        """Check if transitions from s1 and s2 match up to the closure."""
        trans_1 = mdp.get_transitions(s1, action)
        trans_2 = mdp.get_transitions(s2, action)

        for target1, prob1, _ in trans_1:
            if prob1 < 1e-12:
                continue
            i1 = state_idx.get(target1)
            if i1 is None:
                continue

            matched = False
            for target2, prob2, _ in trans_2:
                i2 = state_idx.get(target2)
                if i2 is None:
                    continue
                if abs(prob1 - prob2) < self.tolerance and closure[i1, i2]:
                    matched = True
                    break

            if not matched:
                return False

        return True

    @staticmethod
    def _transitive_closure(R: np.ndarray) -> np.ndarray:
        """Compute the reflexive-transitive closure via matrix powers."""
        n = R.shape[0]
        closure = R.copy().astype(bool)
        np.fill_diagonal(closure, True)

        # Warshall's algorithm
        for k in range(n):
            for i in range(n):
                if closure[i, k]:
                    closure[i] |= closure[k]

        return closure


# ---------------------------------------------------------------------------
# Congruence checking
# ---------------------------------------------------------------------------

@dataclass
class CongruenceChecker:
    """Check whether a bisimulation relation is a congruence.

    A bisimulation ~ is a congruence for an operator op if:
        s₁ ~ s₂  implies  op(s₁, C) ~ op(s₂, C)
    for every context C.

    For MDP bisimulation, this means the equivalence is preserved
    under parallel composition and other structural operators.

    Parameters
    ----------
    tolerance : float
        Numerical tolerance.
    """

    tolerance: float = 1e-10

    def check_parallel_congruence(
        self,
        partition: Partition,
        mdp: MDP,
    ) -> bool:
        """Check if the partition is a congruence for parallel composition.

        Verifies that replacing any state by its block representative
        preserves the transition structure.

        Parameters
        ----------
        partition : Partition
        mdp : MDP

        Returns
        -------
        bool
            True if the partition is a congruence.
        """
        for block in partition.blocks:
            if len(block) <= 1:
                continue

            states = sorted(block)
            ref = states[0]

            for state in states[1:]:
                if not self._equivalent_transitions(ref, state, partition, mdp):
                    logger.debug(
                        "Congruence violation: %s ≢ %s", ref, state,
                    )
                    return False

        return True

    def _equivalent_transitions(
        self,
        s1: str,
        s2: str,
        partition: Partition,
        mdp: MDP,
    ) -> bool:
        """Check if two states have equivalent abstract transitions."""
        actions_1 = set(mdp.get_actions(s1))
        actions_2 = set(mdp.get_actions(s2))

        if actions_1 != actions_2:
            return False

        for aid in actions_1:
            sig_1 = self._abstract_signature(s1, aid, partition, mdp)
            sig_2 = self._abstract_signature(s2, aid, partition, mdp)

            for key in set(sig_1.keys()) | set(sig_2.keys()):
                if abs(sig_1.get(key, 0.0) - sig_2.get(key, 0.0)) > self.tolerance:
                    return False

        return True

    def _abstract_signature(
        self,
        state: str,
        action: str,
        partition: Partition,
        mdp: MDP,
    ) -> dict[int, float]:
        """Compute abstract transition signature: block → probability."""
        sig: dict[int, float] = {}
        for target, prob, _ in mdp.get_transitions(state, action):
            bi = partition.state_to_block.get(target, -1)
            sig[bi] = sig.get(bi, 0.0) + prob
        return sig


# ---------------------------------------------------------------------------
# Parallel composition of MDPs
# ---------------------------------------------------------------------------

@dataclass
class ParallelComposition:
    """Construct the parallel composition of two MDPs.

    M₁ ‖ M₂ has states S₁ × S₂ and transitions that interleave
    or synchronise on shared actions.

    Parameters
    ----------
    sync_actions : set[str] or None
        Actions on which the two MDPs synchronise. If None, all shared
        action names synchronise.
    max_states : int
        Maximum number of product states (to prevent explosion).
    """

    sync_actions: Optional[set[str]] = None
    max_states: int = 10000

    def compose(self, mdp1: MDP, mdp2: MDP) -> MDP:
        """Construct M₁ ‖ M₂.

        Parameters
        ----------
        mdp1, mdp2 : MDP

        Returns
        -------
        MDP
            The parallel composition.

        Raises
        ------
        ValueError
            If the product state space exceeds ``max_states``.
        """
        states_1 = sorted(mdp1.states.keys())
        states_2 = sorted(mdp2.states.keys())

        if len(states_1) * len(states_2) > self.max_states:
            raise ValueError(
                f"Product state space ({len(states_1)}×{len(states_2)}) "
                f"exceeds max_states={self.max_states}"
            )

        shared = set(mdp1.actions.keys()) & set(mdp2.actions.keys())
        sync = self.sync_actions if self.sync_actions is not None else shared

        # Product states
        product_states: dict[str, State] = {}
        for s1 in states_1:
            for s2 in states_2:
                pid = f"{s1}||{s2}"
                st1 = mdp1.states[s1]
                st2 = mdp2.states[s2]

                features = {}
                for k, v in st1.features.items():
                    features[f"m1_{k}"] = v
                for k, v in st2.features.items():
                    features[f"m2_{k}"] = v

                product_states[pid] = State(
                    state_id=pid,
                    features=features,
                    label=f"{st1.label}||{st2.label}",
                    is_terminal=st1.is_terminal or st2.is_terminal,
                    is_goal=st1.is_goal and st2.is_goal,
                )

        # Product transitions
        transitions: list[Transition] = []

        for s1 in states_1:
            for s2 in states_2:
                source = f"{s1}||{s2}"

                # Interleaving: actions of M₁ not in sync
                for aid in mdp1.get_actions(s1):
                    if aid in sync:
                        continue
                    for t1, p1, c1 in mdp1.get_transitions(s1, aid):
                        target = f"{t1}||{s2}"
                        if target in product_states:
                            transitions.append(Transition(
                                source=source, action=f"m1_{aid}",
                                target=target, probability=p1, cost=c1,
                            ))

                # Interleaving: actions of M₂ not in sync
                for aid in mdp2.get_actions(s2):
                    if aid in sync:
                        continue
                    for t2, p2, c2 in mdp2.get_transitions(s2, aid):
                        target = f"{s1}||{t2}"
                        if target in product_states:
                            transitions.append(Transition(
                                source=source, action=f"m2_{aid}",
                                target=target, probability=p2, cost=c2,
                            ))

                # Synchronisation
                for aid in sync:
                    if aid not in mdp1.get_actions(s1):
                        continue
                    if aid not in mdp2.get_actions(s2):
                        continue
                    for t1, p1, c1 in mdp1.get_transitions(s1, aid):
                        for t2, p2, c2 in mdp2.get_transitions(s2, aid):
                            target = f"{t1}||{t2}"
                            if target in product_states:
                                transitions.append(Transition(
                                    source=source, action=f"sync_{aid}",
                                    target=target,
                                    probability=p1 * p2,
                                    cost=c1 + c2,
                                ))

        # Merge actions
        all_actions: dict[str, Action] = {}
        for aid, act in mdp1.actions.items():
            if aid not in sync:
                all_actions[f"m1_{aid}"] = Action(
                    action_id=f"m1_{aid}", action_type=act.action_type,
                )
        for aid, act in mdp2.actions.items():
            if aid not in sync:
                all_actions[f"m2_{aid}"] = Action(
                    action_id=f"m2_{aid}", action_type=act.action_type,
                )
        for aid in sync:
            act = mdp1.actions.get(aid) or mdp2.actions.get(aid)
            atype = act.action_type if act else "unknown"
            all_actions[f"sync_{aid}"] = Action(
                action_id=f"sync_{aid}", action_type=atype,
            )

        initial = f"{mdp1.initial_state}||{mdp2.initial_state}"
        goal_states = {
            f"{g1}||{g2}"
            for g1 in mdp1.goal_states
            for g2 in mdp2.goal_states
            if f"{g1}||{g2}" in product_states
        }

        composed = MDP(
            states=product_states,
            actions=all_actions,
            transitions=transitions,
            initial_state=initial,
            goal_states=goal_states,
            discount=min(mdp1.discount, mdp2.discount),
        )

        logger.info(
            "Parallel composition: %d × %d → %d states, %d transitions",
            len(states_1), len(states_2),
            composed.n_states, composed.n_transitions,
        )

        return composed


# ---------------------------------------------------------------------------
# Modular state-space reduction
# ---------------------------------------------------------------------------

@dataclass
class ModularReduction:
    """Modular state-space reduction via compositional bisimulation.

    Decomposes the MDP into components, reduces each independently,
    and reconstructs the full quotient.

    Parameters
    ----------
    max_component_size : int
        Maximum number of states per component before splitting.
    overlap : int
        Number of boundary states shared between components.
    """

    max_component_size: int = 100
    overlap: int = 2

    def decompose(self, mdp: MDP) -> list[set[str]]:
        """Decompose the MDP state space into weakly connected components.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        list[set[str]]
            List of state-id sets, one per component.
        """
        state_ids = set(mdp.states.keys())
        visited: set[str] = set()
        components: list[set[str]] = []

        for sid in sorted(state_ids):
            if sid in visited:
                continue
            component: set[str] = set()
            queue = [sid]
            while queue:
                s = queue.pop()
                if s in visited:
                    continue
                visited.add(s)
                component.add(s)
                # Successors and predecessors (undirected connectivity)
                for succ in mdp.get_successors(s):
                    if succ not in visited:
                        queue.append(succ)
                for pred in mdp.get_predecessors(s):
                    if pred not in visited:
                        queue.append(pred)

            if component:
                components.append(component)

        # Split large components
        result: list[set[str]] = []
        for comp in components:
            if len(comp) <= self.max_component_size:
                result.append(comp)
            else:
                result.extend(self._split_component(comp, mdp))

        logger.info(
            "Decomposed %d states into %d components "
            "(sizes: %s)",
            len(state_ids), len(result),
            ", ".join(str(len(c)) for c in result[:5]),
        )
        return result

    def _split_component(
        self, component: set[str], mdp: MDP,
    ) -> list[set[str]]:
        """Split a large component into smaller pieces."""
        states = sorted(component)
        n = len(states)
        chunk_size = self.max_component_size

        chunks: list[set[str]] = []
        for start in range(0, n, chunk_size - self.overlap):
            end = min(start + chunk_size, n)
            chunks.append(set(states[start:end]))

        return chunks

    def reduce_component(
        self,
        component: set[str],
        mdp: MDP,
        partition: Partition,
    ) -> Partition:
        """Reduce a single component using an existing partition.

        Restricts the partition to states in the component and merges
        blocks that are entirely contained within the component.

        Parameters
        ----------
        component : set[str]
        mdp : MDP
        partition : Partition

        Returns
        -------
        Partition
            Restricted partition for this component.
        """
        restricted_blocks: list[frozenset[str]] = []

        for block in partition.blocks:
            intersection = block & component
            if intersection:
                restricted_blocks.append(frozenset(intersection))

        if not restricted_blocks:
            return Partition.trivial(sorted(component))

        return Partition.from_blocks(restricted_blocks)

    def merge_partitions(
        self,
        partitions: list[Partition],
        mdp: MDP,
    ) -> Partition:
        """Merge component partitions into a global partition.

        States appearing in multiple component partitions are resolved
        by taking the finest classification.

        Parameters
        ----------
        partitions : list[Partition]
        mdp : MDP

        Returns
        -------
        Partition
            Global merged partition.
        """
        # Assign each state to its finest block
        state_to_label: dict[str, tuple[int, int]] = {}

        for comp_idx, partition in enumerate(partitions):
            for block_idx, block in enumerate(partition.blocks):
                for state in block:
                    existing = state_to_label.get(state)
                    if existing is None:
                        state_to_label[state] = (comp_idx, block_idx)
                    # If state appears in multiple partitions, create a
                    # compound label for the finest distinction
                    else:
                        state_to_label[state] = (
                            existing[0] * 1000 + comp_idx,
                            existing[1] * 1000 + block_idx,
                        )

        # Group by label
        groups: dict[tuple[int, int], set[str]] = defaultdict(set)
        for state, label in state_to_label.items():
            groups[label].add(state)

        # Add any states from the MDP not covered
        covered = set(state_to_label.keys())
        uncovered = set(mdp.states.keys()) - covered
        if uncovered:
            groups[(-1, -1)] = uncovered

        blocks = [frozenset(g) for g in groups.values() if g]
        return Partition.from_blocks(blocks)
