"""
Tests for DAG path algorithms (causalcert.dag.paths).

Covers directed path enumeration, backdoor paths, path blocking,
mediation paths, instrument paths, and path-based metrics on
canonical DAG examples.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalcert.dag.paths import (
    all_directed_paths,
    all_open_paths,
    all_pairs_shortest_path_lengths,
    backdoor_paths,
    causal_paths,
    count_blocked_paths,
    dag_diameter,
    direct_effect_paths,
    directed_path_lengths,
    has_directed_path,
    indirect_effect_paths,
    instrument_paths,
    is_path_blocked,
    longest_directed_path,
    mediation_paths,
    path_count_matrix,
    reachability_matrix,
    shortest_directed_path,
)
from tests.conftest import _adj, random_dag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain4():
    return _adj(4, [(0, 1), (1, 2), (2, 3)])


@pytest.fixture
def diamond4():
    return _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])


@pytest.fixture
def fork3():
    return _adj(3, [(0, 1), (0, 2)])


@pytest.fixture
def collider3():
    return _adj(3, [(0, 2), (1, 2)])


@pytest.fixture
def confounded3():
    """C→X, C→Y, X→Y.  Nodes: 0=C, 1=X, 2=Y."""
    return _adj(3, [(0, 1), (0, 2), (1, 2)])


@pytest.fixture
def mediator5():
    """X→M1→M2→Y, X→Y.  Direct + indirect paths."""
    return _adj(5, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (4, 3)])


@pytest.fixture
def instrument4():
    """Z→X→Y, C→X, C→Y.  Z is an instrument."""
    return _adj(4, [(0, 1), (1, 2), (3, 1), (3, 2)])


@pytest.fixture
def empty4():
    return _adj(4, [])


@pytest.fixture
def complex7():
    """7-node DAG with multiple paths."""
    return _adj(7, [
        (0, 1), (0, 2), (1, 3), (2, 3), (1, 4),
        (3, 5), (4, 5), (5, 6), (3, 6),
    ])


# ===================================================================
# Directed path enumeration
# ===================================================================

class TestAllDirectedPaths:
    def test_chain_single_path(self, chain4):
        paths = all_directed_paths(chain4, 0, 3)
        assert len(paths) == 1
        assert paths[0] == [0, 1, 2, 3]

    def test_diamond_two_paths(self, diamond4):
        paths = all_directed_paths(diamond4, 0, 3)
        assert len(paths) == 2
        path_sets = {tuple(p) for p in paths}
        assert (0, 1, 3) in path_sets
        assert (0, 2, 3) in path_sets

    def test_no_path(self, chain4):
        paths = all_directed_paths(chain4, 3, 0)
        assert paths == []

    def test_self_path(self, chain4):
        paths = all_directed_paths(chain4, 2, 2)
        assert paths == [[2]]

    def test_adjacent(self, chain4):
        paths = all_directed_paths(chain4, 0, 1)
        assert len(paths) == 1
        assert paths[0] == [0, 1]

    def test_empty_graph(self, empty4):
        paths = all_directed_paths(empty4, 0, 3)
        assert paths == []

    def test_max_length(self, chain4):
        paths = all_directed_paths(chain4, 0, 3, max_length=2)
        assert len(paths) == 0  # shortest path has length 3

    def test_max_length_sufficient(self, chain4):
        paths = all_directed_paths(chain4, 0, 3, max_length=3)
        assert len(paths) == 1

    def test_complex_multiple_paths(self, complex7):
        paths = all_directed_paths(complex7, 0, 6)
        assert len(paths) >= 2
        for p in paths:
            assert p[0] == 0
            assert p[-1] == 6
            assert len(set(p)) == len(p)  # simple paths

    def test_mediator_paths(self, mediator5):
        paths = all_directed_paths(mediator5, 0, 3)
        assert len(paths) == 3  # direct, via M1-M2, via M3
        for p in paths:
            assert p[0] == 0
            assert p[-1] == 3


class TestHasDirectedPath:
    def test_chain_reachable(self, chain4):
        assert has_directed_path(chain4, 0, 3)
        assert has_directed_path(chain4, 0, 1)
        assert has_directed_path(chain4, 1, 3)

    def test_chain_not_reachable(self, chain4):
        assert not has_directed_path(chain4, 3, 0)
        assert not has_directed_path(chain4, 2, 0)

    def test_self(self, chain4):
        assert has_directed_path(chain4, 1, 1)

    def test_fork(self, fork3):
        assert has_directed_path(fork3, 0, 1)
        assert has_directed_path(fork3, 0, 2)
        assert not has_directed_path(fork3, 1, 2)

    def test_empty(self, empty4):
        assert not has_directed_path(empty4, 0, 3)


class TestShortestDirectedPath:
    def test_chain(self, chain4):
        sp = shortest_directed_path(chain4, 0, 3)
        assert sp == [0, 1, 2, 3]

    def test_diamond(self, diamond4):
        sp = shortest_directed_path(diamond4, 0, 3)
        assert sp is not None
        assert len(sp) == 3  # length 2 edges

    def test_no_path(self, chain4):
        sp = shortest_directed_path(chain4, 3, 0)
        assert sp is None

    def test_self(self, chain4):
        sp = shortest_directed_path(chain4, 2, 2)
        assert sp == [2]

    def test_adjacent(self, chain4):
        sp = shortest_directed_path(chain4, 0, 1)
        assert sp == [0, 1]


class TestLongestDirectedPath:
    def test_chain(self, chain4):
        lp = longest_directed_path(chain4, 0, 3)
        assert lp == [0, 1, 2, 3]

    def test_diamond(self, diamond4):
        lp = longest_directed_path(diamond4, 0, 3)
        assert lp is not None
        assert len(lp) == 3  # both paths have same length

    def test_no_path(self, chain4):
        lp = longest_directed_path(chain4, 3, 0)
        assert lp is None


class TestDirectedPathLengths:
    def test_chain(self, chain4):
        lengths = directed_path_lengths(chain4, 0, 3)
        assert lengths == [3]

    def test_diamond(self, diamond4):
        lengths = directed_path_lengths(diamond4, 0, 3)
        assert lengths == [2]

    def test_no_path(self, chain4):
        lengths = directed_path_lengths(chain4, 3, 0)
        assert lengths == []

    def test_mediator_multiple(self, mediator5):
        lengths = directed_path_lengths(mediator5, 0, 3)
        assert len(lengths) >= 2


# ===================================================================
# Backdoor paths
# ===================================================================

class TestBackdoorPaths:
    def test_chain_no_backdoor(self, chain4):
        bd = backdoor_paths(chain4, 0, 3)
        assert len(bd) == 0

    def test_confounded_backdoor(self, confounded3):
        # C→X, C→Y, X→Y.  Treatment=1(X), Outcome=2(Y)
        bd = backdoor_paths(confounded3, 1, 2)
        assert len(bd) == 1
        # Path: X ← C → Y
        assert 0 in bd[0]

    def test_fork_no_backdoor_from_root(self, fork3):
        bd = backdoor_paths(fork3, 0, 1)
        assert len(bd) == 0  # no incoming edges to 0

    def test_diamond_no_backdoor(self, diamond4):
        bd = backdoor_paths(diamond4, 0, 3)
        assert len(bd) == 0  # no incoming edges to 0

    def test_instrument_dag(self, instrument4):
        # Z→X→Y, C→X, C→Y.  Treatment=1(X), Outcome=2(Y)
        bd = backdoor_paths(instrument4, 1, 2)
        # Backdoor via C: X ← C → Y
        assert len(bd) >= 1


class TestCausalPaths:
    def test_chain(self, chain4):
        cp = causal_paths(chain4, 0, 3)
        assert len(cp) == 1

    def test_diamond(self, diamond4):
        cp = causal_paths(diamond4, 0, 3)
        assert len(cp) == 2


# ===================================================================
# Path blocking
# ===================================================================

class TestIsPathBlocked:
    def test_chain_blocked_by_middle(self, chain4):
        path = [0, 1, 2, 3]
        assert is_path_blocked(chain4, path, frozenset({1}))
        assert is_path_blocked(chain4, path, frozenset({2}))

    def test_chain_not_blocked_empty(self, chain4):
        path = [0, 1, 2, 3]
        assert not is_path_blocked(chain4, path, frozenset())

    def test_collider_blocked_unconditional(self, collider3):
        # Path: 0 → 2 ← 1.  Collider at 2.
        path = [0, 2, 1]
        # Without conditioning on collider, path is blocked
        assert is_path_blocked(collider3, path, frozenset())

    def test_collider_unblocked_when_conditioned(self, collider3):
        path = [0, 2, 1]
        # Conditioning on collider 2 opens it
        assert not is_path_blocked(collider3, path, frozenset({2}))

    def test_direct_edge_never_blocked(self, chain4):
        path = [0, 1]
        assert not is_path_blocked(chain4, path, frozenset({0, 1, 2, 3}))

    def test_fork_blocked_by_common_cause(self, fork3):
        # Path: 1 ← 0 → 2.  Non-collider at 0.
        path = [1, 0, 2]
        assert is_path_blocked(fork3, path, frozenset({0}))

    def test_fork_open_without_conditioning(self, fork3):
        path = [1, 0, 2]
        assert not is_path_blocked(fork3, path, frozenset())

    def test_confounded_path_blocked(self, confounded3):
        # Backdoor: X ← C → Y.  Conditioning on C blocks it.
        path = [1, 0, 2]
        assert is_path_blocked(confounded3, path, frozenset({0}))


class TestCountBlockedPaths:
    def test_diamond(self, diamond4):
        paths = [[0, 1, 3], [0, 2, 3]]
        # Conditioning on 1 blocks path through 1 but not 2
        count = count_blocked_paths(diamond4, paths, frozenset({1}))
        assert count == 1

    def test_no_conditioning(self, diamond4):
        paths = [[0, 1, 3], [0, 2, 3]]
        count = count_blocked_paths(diamond4, paths, frozenset())
        assert count == 0


class TestAllOpenPaths:
    def test_chain_all_open(self, chain4):
        open_p = all_open_paths(chain4, 0, 3, frozenset())
        assert len(open_p) >= 1

    def test_chain_blocked_conditioning(self, chain4):
        open_p = all_open_paths(chain4, 0, 3, frozenset({1}))
        # All paths go through 1, so all blocked
        assert len(open_p) == 0


# ===================================================================
# Mediation paths
# ===================================================================

class TestMediationPaths:
    def test_chain_mediation(self, chain4):
        # Mediator=1: path 0→1→2→3
        med = mediation_paths(chain4, 0, 1, 3)
        assert len(med) == 1
        assert med[0] == [0, 1, 2, 3]

    def test_no_mediation(self, chain4):
        # No path 0→3→... so no mediation through 3
        med = mediation_paths(chain4, 0, 3, 2)
        assert len(med) == 0

    def test_diamond_mediation(self, diamond4):
        # Mediator=1: path 0→1→3
        med = mediation_paths(diamond4, 0, 1, 3)
        assert len(med) == 1
        assert med[0] == [0, 1, 3]


class TestDirectEffectPaths:
    def test_diamond_direct(self, diamond4):
        # Mediators: {1}.  Direct paths avoid 1.
        direct = direct_effect_paths(diamond4, 0, 3, frozenset({1}))
        assert len(direct) == 1
        assert 1 not in direct[0][1:-1]

    def test_chain_no_direct(self, chain4):
        # Mediator=1: all paths go through 1
        direct = direct_effect_paths(chain4, 0, 3, frozenset({1}))
        assert len(direct) == 0


class TestIndirectEffectPaths:
    def test_diamond_indirect(self, diamond4):
        indirect = indirect_effect_paths(diamond4, 0, 3, frozenset({1}))
        assert len(indirect) == 1
        assert 1 in indirect[0]

    def test_chain_all_indirect(self, chain4):
        indirect = indirect_effect_paths(chain4, 0, 3, frozenset({1}))
        assert len(indirect) == 1


# ===================================================================
# Instrument paths
# ===================================================================

class TestInstrumentPaths:
    def test_instrument_dag(self, instrument4):
        result = instrument_paths(instrument4, 0, 1, 2)
        assert len(result["instrument_to_treatment"]) >= 1
        assert len(result["treatment_to_outcome"]) >= 1

    def test_valid_instrument(self, instrument4):
        result = instrument_paths(instrument4, 0, 1, 2)
        # Z should not have direct path to Y not through X
        for p in result["instrument_to_outcome_direct"]:
            assert 1 not in p[1:-1]


# ===================================================================
# Path-based metrics
# ===================================================================

class TestAllPairsShortestPaths:
    def test_chain(self, chain4):
        dist = all_pairs_shortest_path_lengths(chain4)
        assert dist[0, 3] == 3
        assert dist[0, 1] == 1
        assert dist[3, 0] == -1  # unreachable

    def test_diagonal(self, chain4):
        dist = all_pairs_shortest_path_lengths(chain4)
        for i in range(4):
            assert dist[i, i] == 0

    def test_empty(self, empty4):
        dist = all_pairs_shortest_path_lengths(empty4)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert dist[i, j] == -1


class TestDagDiameter:
    def test_chain(self, chain4):
        assert dag_diameter(chain4) == 3

    def test_diamond(self, diamond4):
        assert dag_diameter(diamond4) == 2

    def test_empty(self, empty4):
        assert dag_diameter(empty4) == 0


class TestPathCountMatrix:
    def test_chain(self, chain4):
        count = path_count_matrix(chain4)
        assert count[0, 3] == 1
        assert count[0, 1] == 1
        assert count[0, 2] == 1

    def test_diamond(self, diamond4):
        count = path_count_matrix(diamond4)
        assert count[0, 3] == 2  # two paths

    def test_diagonal_is_one(self, chain4):
        count = path_count_matrix(chain4)
        for i in range(4):
            assert count[i, i] == 1


class TestReachabilityMatrix:
    def test_chain(self, chain4):
        reach = reachability_matrix(chain4)
        assert reach[0, 3]
        assert reach[0, 1]
        assert not reach[3, 0]

    def test_self_reachable(self, chain4):
        reach = reachability_matrix(chain4)
        for i in range(4):
            assert reach[i, i]

    def test_empty(self, empty4):
        reach = reachability_matrix(empty4)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert not reach[i, j]


# ===================================================================
# Random DAG tests
# ===================================================================

class TestRandomDAGPaths:
    def test_path_consistency(self):
        """has_directed_path and all_directed_paths should agree."""
        adj = random_dag(8, 0.3, seed=123)
        for s in range(8):
            for t in range(8):
                if s == t:
                    continue
                paths = all_directed_paths(adj, s, t)
                has_path = has_directed_path(adj, s, t)
                assert has_path == (len(paths) > 0)

    def test_shortest_is_shortest(self):
        adj = random_dag(8, 0.4, seed=456)
        for s in range(8):
            for t in range(8):
                if s == t:
                    continue
                sp = shortest_directed_path(adj, s, t)
                all_p = all_directed_paths(adj, s, t)
                if sp is None:
                    assert len(all_p) == 0
                else:
                    min_len = min(len(p) for p in all_p)
                    assert len(sp) == min_len

    def test_reachability_vs_bfs(self):
        adj = random_dag(10, 0.3, seed=789)
        reach = reachability_matrix(adj)
        for s in range(10):
            for t in range(10):
                assert reach[s, t] == has_directed_path(adj, s, t)

    def test_path_count_nonneg(self):
        adj = random_dag(8, 0.3, seed=111)
        count = path_count_matrix(adj)
        assert np.all(count >= 0)
        for i in range(8):
            assert count[i, i] == 1
