"""Property-based tests for bisimulation models (Partition, CognitiveDistanceMatrix).

This module verifies structural invariants of partitions and metric-space
properties of the CognitiveDistanceMatrix using Hypothesis. Properties include
partition validity after construction/merge/split, state coverage, block counts
for trivial and discrete partitions, and symmetry/identity/triangle-inequality
for the cognitive distance metric.
"""

import math

import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats, integers, lists, tuples, sampled_from,
)

from usability_oracle.bisimulation.models import Partition, CognitiveDistanceMatrix


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_state_ids_strategy = lists(
    sampled_from([f"s{i}" for i in range(20)]),
    min_size=2,
    max_size=10,
).map(lambda xs: list(dict.fromkeys(xs)))  # unique, order-preserving


def _make_distance_matrix(n):
    """Build a valid symmetric distance matrix with zero diagonal.

    Uses random non-negative off-diagonal values and enforces symmetry.
    """
    rng = np.random.default_rng(abs(hash(n)) % (2**31))
    d = rng.uniform(0.0, 10.0, (n, n))
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    return d


def _make_metric_matrix(n):
    """Build a distance matrix that satisfies the triangle inequality.

    Constructs the matrix as shortest-path distances in a random graph,
    guaranteeing the triangle inequality holds.
    """
    rng = np.random.default_rng(abs(hash(n)) % (2**31))
    d = rng.uniform(1.0, 10.0, (n, n))
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    # Floyd-Warshall to enforce triangle inequality
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


# ---------------------------------------------------------------------------
# Partition validity after construction
# ---------------------------------------------------------------------------


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_trivial_partition_is_valid(state_ids):
    """A trivial partition (single block) is always valid.

    Partition.trivial(states) must produce a structurally consistent
    partition where is_valid() returns True.
    """
    assume(len(state_ids) >= 2)
    p = Partition.trivial(state_ids)
    assert p.is_valid()


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_discrete_partition_is_valid(state_ids):
    """A discrete partition (one block per state) is always valid.

    Partition.discrete(states) must be structurally consistent.
    """
    assume(len(state_ids) >= 2)
    p = Partition.discrete(state_ids)
    assert p.is_valid()


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_from_blocks_is_valid(state_ids):
    """A partition constructed from explicit blocks is valid.

    Building a partition from a list of frozensets must produce a
    valid structure.
    """
    assume(len(state_ids) >= 2)
    blocks = [frozenset([s]) for s in state_ids]
    p = Partition.from_blocks(blocks)
    assert p.is_valid()


# ---------------------------------------------------------------------------
# Partition covers all states
# ---------------------------------------------------------------------------

@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_trivial_covers_all_states(state_ids):
    """A trivial partition contains every input state.

    partition.states() must equal the set of all input state ids.
    """
    assume(len(state_ids) >= 2)
    p = Partition.trivial(state_ids)
    assert p.states() == frozenset(state_ids)


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_discrete_covers_all_states(state_ids):
    """A discrete partition contains every input state.

    No state may be lost during discrete partitioning.
    """
    assume(len(state_ids) >= 2)
    p = Partition.discrete(state_ids)
    assert p.states() == frozenset(state_ids)


# ---------------------------------------------------------------------------
# Block counts
# ---------------------------------------------------------------------------

@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_trivial_has_one_block(state_ids):
    """A trivial partition has exactly one block.

    All states belong to a single equivalence class.
    """
    assume(len(state_ids) >= 2)
    p = Partition.trivial(state_ids)
    assert p.n_blocks == 1


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_discrete_has_n_blocks(state_ids):
    """A discrete partition has as many blocks as states.

    Each state is in its own equivalence class, so n_blocks == n_states.
    """
    assume(len(state_ids) >= 2)
    p = Partition.discrete(state_ids)
    assert p.n_blocks == len(state_ids)


# ---------------------------------------------------------------------------
# Merge preserves validity
# ---------------------------------------------------------------------------

@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_merge_preserves_validity(state_ids):
    """Merging two blocks in a discrete partition preserves validity.

    After merging block 0 and block 1, the resulting partition must
    still satisfy all structural invariants.
    """
    assume(len(state_ids) >= 3)
    p = Partition.discrete(state_ids)
    merged = p.merge(0, 1)
    assert merged.is_valid()


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_merge_reduces_block_count(state_ids):
    """Merging two distinct blocks reduces the block count by one.

    n_blocks(merged) == n_blocks(original) - 1.
    """
    assume(len(state_ids) >= 3)
    p = Partition.discrete(state_ids)
    merged = p.merge(0, 1)
    assert merged.n_blocks == p.n_blocks - 1


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_merge_preserves_states(state_ids):
    """Merging preserves the set of all states.

    No state is lost or duplicated during a merge operation.
    """
    assume(len(state_ids) >= 3)
    p = Partition.discrete(state_ids)
    merged = p.merge(0, 1)
    assert merged.states() == p.states()


# ---------------------------------------------------------------------------
# Split preserves validity
# ---------------------------------------------------------------------------

@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_split_preserves_validity(state_ids):
    """Splitting a block by a predicate preserves partition validity.

    We split the single block of a trivial partition by a simple criterion.
    """
    assume(len(state_ids) >= 3)
    p = Partition.trivial(state_ids)
    first_state = state_ids[0]
    split_p = p.split(0, lambda s: s == first_state)
    assert split_p.is_valid()


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_split_preserves_states(state_ids):
    """Splitting preserves the full set of states.

    No state is lost or duplicated when a block is split.
    """
    assume(len(state_ids) >= 3)
    p = Partition.trivial(state_ids)
    first_state = state_ids[0]
    split_p = p.split(0, lambda s: s == first_state)
    assert split_p.states() == p.states()


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_split_increases_block_count(state_ids):
    """Splitting a non-singleton block increases the block count.

    A trivial partition of >=2 states split by a singleton criterion
    should yield exactly 2 blocks.
    """
    assume(len(state_ids) >= 3)
    p = Partition.trivial(state_ids)
    first_state = state_ids[0]
    split_p = p.split(0, lambda s: s == first_state)
    assert split_p.n_blocks >= 2


# ---------------------------------------------------------------------------
# get_block consistency
# ---------------------------------------------------------------------------

@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_get_block_contains_state(state_ids):
    """get_block(s) returns a block that actually contains s.

    For every state in the partition, the block returned by get_block
    must contain that state.
    """
    assume(len(state_ids) >= 2)
    p = Partition.discrete(state_ids)
    for s in state_ids:
        block = p.get_block(s)
        assert s in block


@given(_state_ids_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_block_index_consistent(state_ids):
    """block_index is consistent with blocks list.

    p.blocks[p.block_index(s)] should contain s.
    """
    assume(len(state_ids) >= 2)
    p = Partition.discrete(state_ids)
    for s in state_ids:
        idx = p.block_index(s)
        assert s in p.blocks[idx]


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix: symmetry
# ---------------------------------------------------------------------------

@given(integers(min_value=3, max_value=8))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_distance_matrix_symmetry(n):
    """Cognitive distance is symmetric: d(i, j) == d(j, i).

    The distance matrix is constructed to be symmetric, so querying
    d(si, sj) must equal d(sj, si).
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    for i in range(n):
        for j in range(n):
            assert math.isclose(
                cdm.distance(ids[i], ids[j]),
                cdm.distance(ids[j], ids[i]),
                abs_tol=1e-12,
            )


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix: identity of indiscernibles
# ---------------------------------------------------------------------------

@given(integers(min_value=3, max_value=8))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_distance_self_is_zero(n):
    """Cognitive distance from a state to itself is zero: d(i, i) == 0.

    The diagonal of the distance matrix must be identically zero.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    for i in range(n):
        assert cdm.distance(ids[i], ids[i]) == 0.0


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix: triangle inequality
# ---------------------------------------------------------------------------

@given(integers(min_value=3, max_value=8))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_triangle_inequality(n):
    """Cognitive distance satisfies triangle inequality: d(i,k) <= d(i,j) + d(j,k).

    We use a metric-valid matrix (Floyd-Warshall enforced) so the triangle
    inequality holds for all triples.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_metric_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                dij = cdm.distance(ids[i], ids[j])
                djk = cdm.distance(ids[j], ids[k])
                dik = cdm.distance(ids[i], ids[k])
                assert dik <= dij + djk + 1e-9


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix: non-negativity
# ---------------------------------------------------------------------------

@given(integers(min_value=3, max_value=8))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_distances_non_negative(n):
    """All cognitive distances are non-negative.

    A distance metric must satisfy d(i,j) >= 0 for all pairs.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    for i in range(n):
        for j in range(n):
            assert cdm.distance(ids[i], ids[j]) >= -1e-12


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix: diameter
# ---------------------------------------------------------------------------

@given(integers(min_value=3, max_value=8))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_diameter_is_max_distance(n):
    """Diameter equals the maximum pairwise distance.

    diameter() should return the largest element in the distance matrix.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    expected = float(np.max(d))
    assert math.isclose(cdm.diameter(), expected, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix: mean_distance
# ---------------------------------------------------------------------------

@given(integers(min_value=3, max_value=8))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_mean_distance_bounded(n):
    """Mean distance is between 0 and the diameter.

    The average pairwise distance must lie in [0, diameter].
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    md = cdm.mean_distance()
    assert 0.0 <= md + 1e-12
    assert md <= cdm.diameter() + 1e-12


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix: nearest_neighbors
# ---------------------------------------------------------------------------

@given(integers(min_value=4, max_value=8))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_nearest_neighbors_sorted(n):
    """Nearest neighbors are returned in ascending distance order.

    The k nearest neighbors should be sorted by distance.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    nn = cdm.nearest_neighbors(ids[0], k=min(3, n - 1))
    dists = [dist for _, dist in nn]
    assert dists == sorted(dists)


@given(integers(min_value=4, max_value=8))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_nearest_neighbors_exclude_self(n):
    """Nearest neighbors do not include the query state itself.

    The query state should not appear in its own neighbor list.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    nn = cdm.nearest_neighbors(ids[0], k=min(3, n - 1))
    neighbor_ids = [sid for sid, _ in nn]
    assert ids[0] not in neighbor_ids


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix: threshold_partition validity
# ---------------------------------------------------------------------------

@given(integers(min_value=3, max_value=7),
       floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_threshold_partition_valid(n, epsilon):
    """threshold_partition produces a valid Partition.

    Regardless of the epsilon value, the resulting partition must be
    structurally valid and cover all states.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    p = cdm.threshold_partition(epsilon)
    assert p.is_valid()
    assert p.states() == frozenset(ids)


@given(integers(min_value=3, max_value=7))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_threshold_zero_gives_discrete(n):
    """A zero threshold produces a discrete partition.

    When epsilon=0, only identical states (self-distances = 0) should
    be grouped, resulting in the finest partition.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_metric_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    p = cdm.threshold_partition(0.0)
    assert p.n_blocks == n


@given(integers(min_value=3, max_value=7))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_threshold_large_gives_trivial(n):
    """A very large threshold produces a trivial (single-block) partition.

    When epsilon is larger than the diameter, all states are equivalent.
    """
    ids = [f"s{i}" for i in range(n)]
    d = _make_distance_matrix(n)
    cdm = CognitiveDistanceMatrix(distances=d, state_ids=ids)
    huge = cdm.diameter() + 100.0
    p = cdm.threshold_partition(huge)
    assert p.n_blocks == 1
