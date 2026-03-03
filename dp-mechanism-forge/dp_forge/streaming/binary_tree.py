"""
Binary tree mechanism for continual observation.

Implements the binary tree aggregation approach of Dwork et al. (2010) and
Chan et al. (2011) for differentially private prefix sums over a data stream.
Each internal node stores a noisy partial sum; a prefix sum is reconstructed
by summing O(log T) nodes, yielding O(log^2 T) total squared error under
pure ε-DP.

References:
    - Dwork, Naor, Pitassi, Rothblum. "Differential Privacy Under Continual
      Observation." STOC 2010.
    - Chan, Shi, Song. "Private and Continual Release of Statistics." ICALP 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.streaming import (
    StreamConfig,
    StreamMechanismType,
    StreamOutput,
    StreamState,
    StreamSummary,
    TreeStructure,
)


# ---------------------------------------------------------------------------
# TreeNode
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in the binary tree aggregation structure.

    Attributes:
        level: Level in the tree (0 = leaf).
        index: Index within its level (0-based).
        true_sum: True partial sum covered by this node.
        noisy_sum: Partial sum after adding Laplace noise.
        noise: Noise that was added.
        left: Left child index or None.
        right: Right child index or None.
        range_start: First leaf index covered (inclusive).
        range_end: Last leaf index covered (exclusive).
    """

    level: int
    index: int
    true_sum: float = 0.0
    noisy_sum: float = 0.0
    noise: float = 0.0
    left: Optional[int] = None
    right: Optional[int] = None
    range_start: int = 0
    range_end: int = 0

    @property
    def range_size(self) -> int:
        return self.range_end - self.range_start

    def __repr__(self) -> str:
        return (
            f"TreeNode(lvl={self.level}, idx={self.index}, "
            f"range=[{self.range_start},{self.range_end}), "
            f"noisy={self.noisy_sum:.4f})"
        )


# ---------------------------------------------------------------------------
# TreeConstruction
# ---------------------------------------------------------------------------


class TreeConstruction:
    """Build and maintain a balanced binary tree over time steps.

    The tree is constructed incrementally: when a new leaf arrives, the
    appropriate internal nodes are updated.
    """

    def __init__(self, max_time: int, branching: int = 2) -> None:
        self.max_time = max_time
        self.branching = branching
        self.height = max(1, math.ceil(math.log(max(max_time, 1)) / math.log(branching)))
        self._nodes: Dict[Tuple[int, int], TreeNode] = {}
        self._num_leaves = 0

    def _get_or_create(self, level: int, index: int) -> TreeNode:
        key = (level, index)
        if key not in self._nodes:
            size = self.branching ** level
            start = index * size
            end = min(start + size, self.max_time)
            self._nodes[key] = TreeNode(
                level=level, index=index,
                range_start=start, range_end=end,
            )
        return self._nodes[key]

    def add_leaf(self, value: float) -> List[TreeNode]:
        """Add a leaf and return the list of nodes updated (leaf → root path)."""
        leaf_index = self._num_leaves
        self._num_leaves += 1
        updated: List[TreeNode] = []
        idx = leaf_index
        for lvl in range(self.height + 1):
            node = self._get_or_create(lvl, idx)
            node.true_sum += value
            updated.append(node)
            idx //= self.branching
        return updated

    def prefix_cover(self, end: int) -> List[TreeNode]:
        """Return the minimal set of nodes covering [0, end).

        This is the canonical decomposition of [0, end) into O(log T) nodes.
        """
        nodes: List[TreeNode] = []
        self._cover_recursive(0, self.branching ** self.height, end, self.height, 0, nodes)
        return nodes

    def _cover_recursive(
        self, start: int, end: int, target: int,
        level: int, index: int, result: List[TreeNode],
    ) -> None:
        if start >= target or start >= self.max_time:
            return
        if end <= target:
            key = (level, index)
            if key in self._nodes:
                result.append(self._nodes[key])
            return
        mid_step = (end - start) // self.branching
        for i in range(self.branching):
            child_start = start + i * mid_step
            child_end = start + (i + 1) * mid_step
            child_index = index * self.branching + i
            self._cover_recursive(child_start, child_end, target, level - 1, child_index, result)

    def get_node(self, level: int, index: int) -> Optional[TreeNode]:
        return self._nodes.get((level, index))

    @property
    def num_leaves(self) -> int:
        return self._num_leaves

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"TreeConstruction(T={self.max_time}, h={self.height}, nodes={self.num_nodes})"


# ---------------------------------------------------------------------------
# NoiseAllocation
# ---------------------------------------------------------------------------


class NoiseAllocation:
    """Optimal noise allocation across tree levels.

    For pure ε-DP, each node receives Lap(Δ·h/ε) noise where h is the
    tree height and Δ is the per-step sensitivity.  For approximate DP,
    Gaussian noise is used with advanced composition.
    """

    def __init__(
        self, height: int, epsilon: float, delta: float = 0.0,
        sensitivity: float = 1.0,
    ) -> None:
        self.height = height
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self._scales = self._compute_scales()

    def _compute_scales(self) -> npt.NDArray[np.float64]:
        """Compute per-level noise scales."""
        num_levels = self.height + 1
        if self.delta == 0.0:
            # Pure DP: each level gets ε/h budget → Lap(sensitivity * h / ε)
            scale = self.sensitivity * num_levels / self.epsilon
            return np.full(num_levels, scale)
        else:
            # Approximate DP: Gaussian with advanced composition
            sigma = (
                self.sensitivity
                * math.sqrt(2.0 * num_levels * math.log(1.0 / self.delta))
                / self.epsilon
            )
            return np.full(num_levels, sigma)

    def scale_for_level(self, level: int) -> float:
        if 0 <= level < len(self._scales):
            return float(self._scales[level])
        return float(self._scales[-1])

    def total_variance_for_prefix(self, num_nodes: int) -> float:
        """Variance of a prefix sum that touches *num_nodes* nodes."""
        if self.delta == 0.0:
            return 2.0 * (self._scales[0] ** 2) * num_nodes
        return (self._scales[0] ** 2) * num_nodes

    def expected_squared_error(self) -> float:
        """Expected squared error of a worst-case prefix sum."""
        num_nodes = self.height + 1
        return self.total_variance_for_prefix(num_nodes)

    def __repr__(self) -> str:
        return (
            f"NoiseAllocation(h={self.height}, ε={self.epsilon}, "
            f"scale_0={self._scales[0]:.4f})"
        )


# ---------------------------------------------------------------------------
# RangeQuery
# ---------------------------------------------------------------------------


class RangeQuery:
    """Answer range queries [l, r) using the binary tree structure.

    A range sum [l, r) = prefix(r) - prefix(l), each computed via O(log T)
    tree nodes, giving O(log^2 T) total squared error.
    """

    def __init__(self, tree: TreeConstruction) -> None:
        self.tree = tree

    def query(self, left: int, right: int) -> float:
        """Return the noisy sum over [left, right).

        Uses prefix(right) - prefix(left) with noisy node sums.
        """
        if left < 0 or right > self.tree.num_leaves or left >= right:
            raise ValueError(f"Invalid range [{left}, {right})")
        sum_right = self._prefix_noisy_sum(right)
        sum_left = self._prefix_noisy_sum(left) if left > 0 else 0.0
        return sum_right - sum_left

    def _prefix_noisy_sum(self, end: int) -> float:
        nodes = self.tree.prefix_cover(end)
        return sum(n.noisy_sum for n in nodes)

    def true_range_sum(self, left: int, right: int) -> float:
        """True (non-private) range sum for error measurement."""
        if left < 0 or right > self.tree.num_leaves or left >= right:
            raise ValueError(f"Invalid range [{left}, {right})")
        sum_right = sum(
            n.true_sum for n in self.tree.prefix_cover(right)
        )
        sum_left = (
            sum(n.true_sum for n in self.tree.prefix_cover(left))
            if left > 0 else 0.0
        )
        return sum_right - sum_left

    def error(self, left: int, right: int) -> float:
        """Absolute error of the noisy range query."""
        return abs(self.query(left, right) - self.true_range_sum(left, right))

    def __repr__(self) -> str:
        return f"RangeQuery(tree={self.tree})"


# ---------------------------------------------------------------------------
# MatrixFactorizationTree
# ---------------------------------------------------------------------------


class MatrixFactorizationTree:
    """Matrix factorization view of the binary tree mechanism.

    The prefix-sum workload W (T×T lower-triangular all-ones) is factored as
    W = B · L where L maps data to noisy tree nodes and B reconstructs
    prefix sums from nodes.  This view connects to the matrix mechanism
    literature (Li et al. 2015).
    """

    def __init__(self, T: int) -> None:
        self.T = T
        self.height = max(1, math.ceil(math.log2(max(T, 1))))
        self._L: Optional[npt.NDArray[np.float64]] = None
        self._B: Optional[npt.NDArray[np.float64]] = None

    def _num_nodes(self) -> int:
        """Total nodes in a complete binary tree with T leaves."""
        cap = 2 ** self.height
        count = 0
        size = 1
        for _ in range(self.height + 1):
            count += (cap + size - 1) // size
            size *= 2
        return min(count, 2 * cap - 1)

    def build_encoder(self) -> npt.NDArray[np.float64]:
        """Build the encoding matrix L (nodes × T).

        L[j, t] = 1 if node j covers time step t.
        """
        if self._L is not None:
            return self._L
        cap = 2 ** self.height
        nodes: List[Tuple[int, int]] = []  # (start, end) ranges
        size = 1
        for lvl in range(self.height + 1):
            idx = 0
            while idx * size < cap:
                start = idx * size
                end = min(start + size, self.T)
                if start < self.T:
                    nodes.append((start, end))
                idx += 1
            size *= 2
        m = len(nodes)
        L = np.zeros((m, self.T), dtype=np.float64)
        for j, (s, e) in enumerate(nodes):
            L[j, s:e] = 1.0
        self._L = L
        return L

    def build_decoder(self) -> npt.NDArray[np.float64]:
        """Build the decoding matrix B (T × nodes).

        B[t, :] sums the nodes in the canonical cover of prefix [0, t+1).
        """
        L = self.build_encoder()
        m = L.shape[0]
        B = np.zeros((self.T, m), dtype=np.float64)
        # For each prefix sum [0, t+1), mark the nodes that cover it
        for t in range(self.T):
            prefix_end = t + 1
            # Greedy covering from largest nodes
            covered = 0
            for j in range(m):
                s = int(np.argmax(L[j] > 0)) if np.any(L[j] > 0) else 0
                e = s + int(np.sum(L[j] > 0))
                if s >= covered and e <= prefix_end and e > s:
                    if s == covered:
                        B[t, j] = 1.0
                        covered = e
            # Fallback: use pseudoinverse-style reconstruction
            if covered < prefix_end:
                target = np.zeros(self.T)
                target[:prefix_end] = 1.0
                B[t, :] = np.linalg.lstsq(L.T, target, rcond=None)[0]
        self._B = B
        return B

    def workload_matrix(self) -> npt.NDArray[np.float64]:
        """The T×T lower-triangular all-ones prefix-sum workload."""
        return np.tril(np.ones((self.T, self.T), dtype=np.float64))

    def reconstruction_error(self, epsilon: float, sensitivity: float = 1.0) -> float:
        """Expected total squared error under this factorization."""
        L = self.build_encoder()
        B = self.build_decoder()
        h = self.height + 1
        # Per-node noise variance (Laplace): 2 * (sensitivity * h / epsilon)^2
        sigma2 = 2.0 * (sensitivity * h / epsilon) ** 2
        # Total error = sum over rows of B of ||b_t||^2 * sigma^2
        row_norms_sq = np.sum(B ** 2, axis=1)
        return float(np.mean(row_norms_sq) * sigma2)

    def __repr__(self) -> str:
        return f"MatrixFactorizationTree(T={self.T}, h={self.height})"


# ---------------------------------------------------------------------------
# HybridTree
# ---------------------------------------------------------------------------


class HybridTree:
    """Hybrid tree for non-power-of-2 stream lengths.

    When T is not a power of 2, a hybrid approach combines multiple
    complete binary trees to cover exactly T leaves with near-optimal error.
    This follows Chan et al. (2011) §4.
    """

    def __init__(self, T: int, epsilon: float, delta: float = 0.0,
                 sensitivity: float = 1.0) -> None:
        self.T = T
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self._subtrees: List[TreeConstruction] = []
        self._subtree_sizes: List[int] = []
        self._noise_alloc: Optional[NoiseAllocation] = None
        self._rng = np.random.default_rng()
        self._build_subtrees()

    def _build_subtrees(self) -> None:
        """Decompose T into powers of 2 and build subtrees."""
        remaining = self.T
        bit = 0
        while remaining > 0:
            if remaining & 1:
                size = 1 << bit
                tree = TreeConstruction(max_time=size)
                self._subtrees.append(tree)
                self._subtree_sizes.append(size)
            remaining >>= 1
            bit += 1
        total_height = max(1, math.ceil(math.log2(max(self.T, 1))))
        self._noise_alloc = NoiseAllocation(
            height=total_height, epsilon=self.epsilon,
            delta=self.delta, sensitivity=self.sensitivity,
        )

    def add_value(self, value: float) -> float:
        """Add a value and return the noisy prefix sum."""
        # Find which subtree this leaf belongs to
        leaf_idx = sum(t.num_leaves for t in self._subtrees)
        if leaf_idx >= self.T:
            raise ValueError("Stream exceeded declared length T")
        cumulative = 0
        target_tree = None
        for tree, size in zip(self._subtrees, self._subtree_sizes):
            if tree.num_leaves < size:
                target_tree = tree
                break
            cumulative += size
        if target_tree is None:
            raise ValueError("All subtrees are full")
        updated = target_tree.add_leaf(value)
        # Add noise to newly created/updated nodes
        for node in updated:
            if node.noise == 0.0:
                scale = self._noise_alloc.scale_for_level(node.level)
                if self.delta == 0.0:
                    node.noise = self._rng.laplace(0, scale)
                else:
                    node.noise = self._rng.normal(0, scale)
                node.noisy_sum = node.true_sum + node.noise
            else:
                # Node already has noise; update the noisy sum
                node.noisy_sum = node.true_sum + node.noise
        return self._compute_prefix_sum()

    def _compute_prefix_sum(self) -> float:
        """Sum all subtree prefix sums to get total prefix sum."""
        total = 0.0
        for tree in self._subtrees:
            if tree.num_leaves > 0:
                nodes = tree.prefix_cover(tree.num_leaves)
                total += sum(n.noisy_sum for n in nodes)
        return total

    def prefix_sum(self, end: int) -> float:
        """Noisy prefix sum [0, end)."""
        if end <= 0:
            return 0.0
        total = 0.0
        remaining = end
        for tree, size in zip(self._subtrees, self._subtree_sizes):
            if remaining <= 0:
                break
            count = min(remaining, tree.num_leaves)
            if count > 0:
                nodes = tree.prefix_cover(count)
                total += sum(n.noisy_sum for n in nodes)
            remaining -= size
        return total

    def expected_error(self) -> float:
        """Expected squared error of a prefix query."""
        if self._noise_alloc is None:
            return 0.0
        return self._noise_alloc.expected_squared_error()

    @property
    def num_subtrees(self) -> int:
        return len(self._subtrees)

    def __repr__(self) -> str:
        return (
            f"HybridTree(T={self.T}, subtrees={self.num_subtrees}, "
            f"sizes={self._subtree_sizes})"
        )


# ---------------------------------------------------------------------------
# BinaryTreeMechanism (full implementation)
# ---------------------------------------------------------------------------


class BinaryTreeMechanism:
    """Binary tree aggregation for continual prefix sums.

    Achieves O(log² T) expected squared error for T time steps while
    maintaining ε-DP.  Each node in the binary tree stores a noisy partial
    sum with independent Laplace (or Gaussian) noise.
    """

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        self.config = config or StreamConfig()
        self._state = StreamState()
        self._rng = np.random.default_rng(self.config.seed)
        self._tree = TreeConstruction(
            max_time=self.config.max_time,
            branching=self.config.branching_factor,
        )
        self._noise_alloc = NoiseAllocation(
            height=self._tree.height,
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            sensitivity=self.config.sensitivity,
        )
        self._outputs: List[StreamOutput] = []
        self._range_query = RangeQuery(self._tree)

    def observe(self, value: float) -> StreamOutput:
        """Process a new value and return the noisy prefix sum."""
        if self._state.current_time >= self.config.max_time:
            raise ValueError("Stream exceeded max_time")
        self._state.running_sum += value
        self._state.num_observations += 1
        updated_nodes = self._tree.add_leaf(value)
        # Add noise to each updated node (only once when created)
        for node in updated_nodes:
            if node.noise == 0.0:
                scale = self._noise_alloc.scale_for_level(node.level)
                if self.config.delta == 0.0:
                    node.noise = self._rng.laplace(0, scale)
                else:
                    node.noise = self._rng.normal(0, scale)
            node.noisy_sum = node.true_sum + node.noise
        # Compute prefix sum from tree
        cover = self._tree.prefix_cover(self._tree.num_leaves)
        noisy_sum = sum(n.noisy_sum for n in cover)
        true_sum = self._state.running_sum
        output = StreamOutput(
            timestamp=self._state.current_time,
            value=noisy_sum,
            true_value=true_sum,
            noise_added=noisy_sum - true_sum,
        )
        self._state.noisy_sum = noisy_sum
        self._state.current_time += 1
        self._outputs.append(output)
        return output

    def query(self) -> StreamOutput:
        """Query the current noisy prefix sum without new observation."""
        if self._tree.num_leaves == 0:
            return StreamOutput(timestamp=0, value=0.0, true_value=0.0, noise_added=0.0)
        cover = self._tree.prefix_cover(self._tree.num_leaves)
        noisy_sum = sum(n.noisy_sum for n in cover)
        return StreamOutput(
            timestamp=self._state.current_time - 1,
            value=noisy_sum,
            true_value=self._state.running_sum,
            noise_added=noisy_sum - self._state.running_sum,
        )

    def range_query(self, left: int, right: int) -> float:
        """Answer a range query [left, right) using the tree."""
        return self._range_query.query(left, right)

    def privacy_spent(self) -> float:
        """Return total privacy budget spent (constant for tree mechanism)."""
        return self.config.epsilon

    def reset(self) -> None:
        """Reset the mechanism."""
        self._state = StreamState()
        self._tree = TreeConstruction(
            max_time=self.config.max_time,
            branching=self.config.branching_factor,
        )
        self._noise_alloc = NoiseAllocation(
            height=self._tree.height,
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            sensitivity=self.config.sensitivity,
        )
        self._outputs = []
        self._range_query = RangeQuery(self._tree)

    @property
    def state(self) -> StreamState:
        return self._state

    def summarize(self) -> StreamSummary:
        """Return summary statistics of the streaming session."""
        if not self._outputs:
            return StreamSummary(
                total_time_steps=0, total_privacy_spent=self.privacy_spent(),
                mean_absolute_error=0.0, max_absolute_error=0.0, rmse=0.0,
                mechanism_type=StreamMechanismType.BINARY_TREE,
            )
        errors = [abs(o.noise_added) for o in self._outputs if o.noise_added is not None]
        if not errors:
            errors = [0.0]
        return StreamSummary(
            total_time_steps=len(self._outputs),
            total_privacy_spent=self.privacy_spent(),
            mean_absolute_error=float(np.mean(errors)),
            max_absolute_error=float(np.max(errors)),
            rmse=float(np.sqrt(np.mean(np.array(errors) ** 2))),
            mechanism_type=StreamMechanismType.BINARY_TREE,
        )

    def __repr__(self) -> str:
        return (
            f"BinaryTreeMechanism(T={self.config.max_time}, "
            f"ε={self.config.epsilon}, h={self._tree.height})"
        )


__all__ = [
    "TreeNode",
    "TreeConstruction",
    "NoiseAllocation",
    "RangeQuery",
    "MatrixFactorizationTree",
    "HybridTree",
    "BinaryTreeMechanism",
]
