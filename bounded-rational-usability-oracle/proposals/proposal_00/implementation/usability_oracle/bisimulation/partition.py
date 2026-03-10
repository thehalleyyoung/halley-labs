"""
usability_oracle.bisimulation.partition — Partition refinement algorithm.

Implements the classical partition-refinement loop for bounded-rational
bisimulation:

    1. Initialise with a coarse partition (group by action availability + goal
       status).
    2. For each block, check whether all states have the same π_β distribution
       over abstract actions (actions leading to the same abstract block).
    3. If not, split the block.
    4. Repeat until convergence (no new splits).

The algorithm terminates because each step strictly increases the number of
blocks and the state space is finite.

The key invariant maintained is: within each block, for every action *a* and
every target block *B'*, the bounded-rational transition probabilities are
ε-close.

References
----------
- Givan, Dean & Greig (2003). Equivalence notions and model minimization
  in Markov decision processes. *Artificial Intelligence* 147, 163–223.
- Ferns, Panangaden & Precup (2004). Metrics for finite Markov decision
  processes. *UAI*.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bisimulation.cognitive_distance import (
    CognitiveDistanceComputer,
    _soft_value_iteration,
)
from usability_oracle.bisimulation.models import Partition
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PartitionRefinement
# ---------------------------------------------------------------------------

@dataclass
class PartitionRefinement:
    """Partition refinement for bounded-rational bisimulation.

    Iteratively refines a partition of the MDP state space until states within
    each block are behaviourally equivalent (within tolerance ε) under the
    bounded-rational policy π_β.

    Parameters
    ----------
    max_iterations : int
        Maximum number of refinement rounds (default 500).
    verbose : bool
        If True, log progress at each iteration.
    """

    max_iterations: int = 500
    verbose: bool = False

    _distance_computer: CognitiveDistanceComputer = field(
        default_factory=CognitiveDistanceComputer, repr=False
    )

    # ── Public API --------------------------------------------------------

    def refine(
        self,
        mdp: MDP,
        beta: float,
        epsilon: float = 0.01,
    ) -> Partition:
        """Run partition refinement until convergence.

        Parameters
        ----------
        mdp : MDP
            The MDP to partition.
        beta : float
            Rationality parameter for the bounded-rational policy.
        epsilon : float
            Tolerance for policy distribution differences within a block.

        Returns
        -------
        Partition
            The coarsest partition such that within each block, all states
            have ε-similar bounded-rational policy distributions.
        """
        partition = self._initial_partition(mdp)
        logger.info(
            "Initial partition: %d blocks for %d states",
            partition.n_blocks, len(partition.state_to_block),
        )

        # Pre-compute values and policy distributions
        values = _soft_value_iteration(mdp, beta)

        history: list[int] = [partition.n_blocks]

        for iteration in range(self.max_iterations):
            new_partition = self._refine_step(partition, mdp, beta, epsilon, values)

            history.append(new_partition.n_blocks)

            if self.verbose:
                logger.info(
                    "Iteration %d: %d → %d blocks",
                    iteration, partition.n_blocks, new_partition.n_blocks,
                )

            if self._convergence_check(partition, new_partition):
                logger.info(
                    "Converged after %d iterations with %d blocks",
                    iteration + 1, new_partition.n_blocks,
                )
                return new_partition

            partition = new_partition

        logger.warning(
            "Partition refinement did not converge after %d iterations "
            "(%d blocks)", self.max_iterations, partition.n_blocks,
        )
        return partition

    # ── Initial partition -------------------------------------------------

    def _initial_partition(self, mdp: MDP) -> Partition:
        """Create the initial coarse partition.

        Groups states by:
          1. Goal/terminal status (goal states, terminal non-goal, non-terminal).
          2. Available action set (states with the same set of actions).

        Returns
        -------
        Partition
        """
        groups: dict[tuple, set[str]] = defaultdict(set)

        for sid, state in mdp.states.items():
            # Classify by status
            if state.is_goal:
                status = "goal"
            elif state.is_terminal:
                status = "terminal"
            else:
                status = "active"

            # Classify by available actions
            actions = tuple(sorted(mdp.get_actions(sid)))
            key = (status, actions)
            groups[key].add(sid)

        blocks = [frozenset(g) for g in groups.values() if g]
        partition = Partition.from_blocks(blocks)

        assert partition.is_valid(), "Initial partition is invalid"
        return partition

    # ── Refinement step ---------------------------------------------------

    def _refine_step(
        self,
        partition: Partition,
        mdp: MDP,
        beta: float,
        epsilon: float,
        values: dict[str, float],
    ) -> Partition:
        """Perform one refinement pass over all blocks.

        For each block, compute the abstract policy signature of every state
        (the distribution over target blocks under π_β).  If any state differs
        by more than ε from the block consensus, split the block.

        Parameters
        ----------
        partition : Partition
        mdp : MDP
        beta : float
        epsilon : float
        values : dict[str, float]

        Returns
        -------
        Partition
            The (possibly finer) partition.
        """
        new_blocks: list[frozenset[str]] = []

        for block in partition.blocks:
            if len(block) <= 1:
                new_blocks.append(block)
                continue

            if not self._should_split(block, mdp, beta, epsilon, values, partition):
                new_blocks.append(block)
                continue

            sub_blocks = self._find_split(block, mdp, beta, values, partition)
            new_blocks.extend(sub_blocks)

        return Partition.from_blocks(new_blocks)

    # ── Split detection ---------------------------------------------------

    def _should_split(
        self,
        block: frozenset[str],
        mdp: MDP,
        beta: float,
        epsilon: float,
        values: dict[str, float],
        partition: Partition,
    ) -> bool:
        """Check if a block should be split based on policy divergence.

        Returns True if any pair of states in the block has an abstract policy
        TV-distance exceeding ε.
        """
        if len(block) <= 1:
            return False

        signatures = {}
        for state in block:
            sig = self._abstract_policy_signature(
                state, mdp, beta, values, partition
            )
            signatures[state] = sig

        # Compare all pairs against the first state's signature
        ref_state = next(iter(block))
        ref_sig = signatures[ref_state]
        for state in block:
            if state == ref_state:
                continue
            if self._signature_distance(ref_sig, signatures[state]) > epsilon:
                return True
        return False

    def _find_split(
        self,
        block: frozenset[str],
        mdp: MDP,
        beta: float,
        values: dict[str, float],
        partition: Partition,
    ) -> list[frozenset[str]]:
        """Split a block into sub-blocks of states with similar signatures.

        Uses a greedy clustering approach: pick a seed state, group all states
        whose signature is within ε of the seed, then repeat with remaining
        states.

        Returns
        -------
        list[frozenset[str]]
            At least two non-empty sub-blocks.
        """
        signatures: dict[str, dict[tuple[str, int], float]] = {}
        for state in block:
            signatures[state] = self._abstract_policy_signature(
                state, mdp, beta, values, partition
            )

        # Build distance matrix within block
        states = sorted(block)
        n = len(states)
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._signature_distance(
                    signatures[states[i]], signatures[states[j]]
                )
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Greedy partitioning: find the two states that are most distant
        # and seed two clusters from them
        if n == 2:
            return [frozenset([states[0]]), frozenset([states[1]])]

        # Find maximally distant pair
        flat_idx = int(np.argmax(dist_matrix))
        seed_i, seed_j = divmod(flat_idx, n)

        cluster_a: set[str] = set()
        cluster_b: set[str] = set()

        for k, state in enumerate(states):
            dist_to_a = dist_matrix[k, seed_i]
            dist_to_b = dist_matrix[k, seed_j]
            if dist_to_a <= dist_to_b:
                cluster_a.add(state)
            else:
                cluster_b.add(state)

        result: list[frozenset[str]] = []
        if cluster_a:
            result.append(frozenset(cluster_a))
        if cluster_b:
            result.append(frozenset(cluster_b))

        # Fallback: if one cluster is empty, do a simple first-half / second-half
        if len(result) < 2:
            mid = n // 2
            return [frozenset(states[:mid]), frozenset(states[mid:])]

        return result

    # ── Abstract policy signatures ----------------------------------------

    def _abstract_policy_signature(
        self,
        state: str,
        mdp: MDP,
        beta: float,
        values: dict[str, float],
        partition: Partition,
    ) -> dict[tuple[str, int], float]:
        """Compute the abstract policy signature for a state.

        The signature maps (action, target_block_index) → probability,
        representing how likely each (action, abstract next state) pair is
        under the bounded-rational policy.

        sig[a, B'] = π_β(a|s) · Σ_{s'∈B'} T(s'|s, a)
        """
        actions = mdp.get_actions(state)
        if not actions:
            return {}

        # Compute action probabilities via softmax
        q_values = np.zeros(len(actions), dtype=np.float64)
        gamma = mdp.discount

        for idx, aid in enumerate(actions):
            transitions = mdp.get_transitions(state, aid)
            expected_cost = 0.0
            expected_future = 0.0
            for target, prob, cost in transitions:
                expected_cost += prob * cost
                expected_future += prob * values.get(target, 0.0)
            q_values[idx] = -expected_cost + gamma * expected_future

        # Softmax
        scaled = beta * q_values
        scaled -= np.max(scaled)
        exp_q = np.exp(scaled)
        total_exp = np.sum(exp_q)
        if total_exp <= 0:
            action_probs = np.ones(len(actions)) / len(actions)
        else:
            action_probs = exp_q / total_exp

        # Build signature
        signature: dict[tuple[str, int], float] = {}

        for idx, aid in enumerate(actions):
            transitions = mdp.get_transitions(state, aid)
            for target, trans_prob, _ in transitions:
                block_idx = partition.state_to_block.get(target, -1)
                key = (aid, block_idx)
                prob_contribution = float(action_probs[idx]) * trans_prob
                signature[key] = signature.get(key, 0.0) + prob_contribution

        return signature

    def _signature_distance(
        self,
        sig_a: dict[tuple[str, int], float],
        sig_b: dict[tuple[str, int], float],
    ) -> float:
        """Compute the L1 distance between two abstract policy signatures.

        This is equivalent to total-variation distance over the joint
        (action, block) space.
        """
        all_keys = set(sig_a.keys()) | set(sig_b.keys())
        total = 0.0
        for key in all_keys:
            total += abs(sig_a.get(key, 0.0) - sig_b.get(key, 0.0))
        return 0.5 * total

    # ── Abstract transitions ----------------------------------------------

    def compute_abstract_transitions(
        self,
        partition: Partition,
        mdp: MDP,
    ) -> dict[tuple[int, str, int], float]:
        """Compute abstract transition probabilities.

        T_abs(B_i, a, B_j) = (1/|B_i|) Σ_{s ∈ B_i} Σ_{s' ∈ B_j} T(s'|s, a)

        Parameters
        ----------
        partition : Partition
        mdp : MDP

        Returns
        -------
        dict[tuple[int, str, int], float]
            Maps (source_block, action, target_block) → probability.
        """
        result: dict[tuple[int, str, int], float] = {}

        for bi, block in enumerate(partition.blocks):
            block_size = len(block)
            if block_size == 0:
                continue

            for state in block:
                for aid in mdp.get_actions(state):
                    for target, prob, _ in mdp.get_transitions(state, aid):
                        bj = partition.state_to_block.get(target, -1)
                        key = (bi, aid, bj)
                        result[key] = result.get(key, 0.0) + prob / block_size

        return result

    # ── Convergence -------------------------------------------------------

    def _convergence_check(
        self,
        old_partition: Partition,
        new_partition: Partition,
    ) -> bool:
        """Check if refinement has converged (no new splits).

        Convergence is achieved when the partition is unchanged, i.e. the
        same set of blocks (order-independent).
        """
        if old_partition.n_blocks != new_partition.n_blocks:
            return False
        return set(old_partition.blocks) == set(new_partition.blocks)
