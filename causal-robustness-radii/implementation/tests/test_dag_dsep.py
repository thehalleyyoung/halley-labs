"""Tests for causalcert.dag.dsep – d-separation oracle."""

from __future__ import annotations

import numpy as np
import pytest

from causalcert.dag.dsep import DSeparationOracle
from causalcert.dag.graph import CausalDAG
from causalcert.types import AdjacencyMatrix, NodeSet

# ── helper ─────────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


# ═══════════════════════════════════════════════════════════════════════════
# Chain structure: 0 -> 1 -> 2
# ═══════════════════════════════════════════════════════════════════════════


class TestChainDSep:
    """In a chain X -> M -> Y, X _||_ Y | M but X _not_||_ Y | {}."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.adj = _adj(3, [(0, 1), (1, 2)])
        self.oracle = DSeparationOracle(self.adj)

    def test_unconditional_not_dsep(self) -> None:
        assert not self.oracle.is_d_separated(0, 2, frozenset())

    def test_cond_on_mediator_dsep(self) -> None:
        assert self.oracle.is_d_separated(0, 2, frozenset({1}))

    def test_d_connected_unconditional(self) -> None:
        assert self.oracle.is_d_connected(0, 2, frozenset())

    def test_d_connected_cond_mediator(self) -> None:
        assert not self.oracle.is_d_connected(0, 2, frozenset({1}))

    def test_adjacent_nodes_never_dsep(self) -> None:
        assert not self.oracle.is_d_separated(0, 1, frozenset())
        assert not self.oracle.is_d_separated(1, 2, frozenset())


# ═══════════════════════════════════════════════════════════════════════════
# Fork structure: 1 <- 0 -> 2
# ═══════════════════════════════════════════════════════════════════════════


class TestForkDSep:
    """Fork: X <- Z -> Y. X _||_ Y | Z; X _not_||_ Y | {}."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.adj = _adj(3, [(0, 1), (0, 2)])
        self.oracle = DSeparationOracle(self.adj)

    def test_unconditional_not_dsep(self) -> None:
        assert not self.oracle.is_d_separated(1, 2, frozenset())

    def test_cond_on_common_cause_dsep(self) -> None:
        assert self.oracle.is_d_separated(1, 2, frozenset({0}))

    def test_symmetry(self) -> None:
        r1 = self.oracle.is_d_separated(1, 2, frozenset({0}))
        r2 = self.oracle.is_d_separated(2, 1, frozenset({0}))
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════
# Collider structure: 0 -> 2 <- 1
# ═══════════════════════════════════════════════════════════════════════════


class TestColliderDSep:
    """Collider: X -> Z <- Y. X _||_ Y | {} but X _not_||_ Y | Z."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.adj = _adj(3, [(0, 2), (1, 2)])
        self.oracle = DSeparationOracle(self.adj)

    def test_unconditional_dsep(self) -> None:
        assert self.oracle.is_d_separated(0, 1, frozenset())

    def test_cond_on_collider_opens_path(self) -> None:
        assert not self.oracle.is_d_separated(0, 1, frozenset({2}))

    def test_conditioning_on_descendant_of_collider(self) -> None:
        # Add descendant of collider: 2 -> 3
        adj2 = _adj(4, [(0, 2), (1, 2), (2, 3)])
        oracle2 = DSeparationOracle(adj2)
        # Conditioning on descendant of collider also opens path
        assert not oracle2.is_d_separated(0, 1, frozenset({3}))

    def test_conditioning_on_non_collider_stays_dsep(self) -> None:
        # If we condition on nothing, they're d-separated
        assert self.oracle.is_d_separated(0, 1, frozenset())


# ═══════════════════════════════════════════════════════════════════════════
# Diamond / M-structure
# ═══════════════════════════════════════════════════════════════════════════


class TestDiamondDSep:
    """Diamond: 0->1, 0->2, 1->3, 2->3."""

    @pytest.fixture(autouse=True)
    def _setup(self, diamond4_adj: np.ndarray) -> None:
        self.oracle = DSeparationOracle(diamond4_adj)

    def test_0_3_unconditional_not_dsep(self) -> None:
        assert not self.oracle.is_d_separated(0, 3, frozenset())

    def test_0_3_cond_both_mediators_dsep(self) -> None:
        assert self.oracle.is_d_separated(0, 3, frozenset({1, 2}))

    def test_1_2_unconditional(self) -> None:
        # 1 and 2 share common cause 0, so not d-sep
        assert not self.oracle.is_d_separated(1, 2, frozenset())

    def test_1_2_cond_on_0(self) -> None:
        # Conditioning on common cause blocks fork; but 3 is a collider
        # 1 _||_ 2 | 0 (blocked through 0 by conditioning, through 3 by collider)
        assert self.oracle.is_d_separated(1, 2, frozenset({0}))

    def test_1_2_cond_on_0_and_3(self) -> None:
        # Conditioning on collider 3 opens path
        assert not self.oracle.is_d_separated(1, 2, frozenset({0, 3}))


# ═══════════════════════════════════════════════════════════════════════════
# M-bias DAG
# ═══════════════════════════════════════════════════════════════════════════


class TestMBiasDSep:
    """M-bias: 0=U1, 1=X, 2=M, 3=U2, 4=Y; U1->X, U1->M, U2->M, U2->Y, X->Y."""

    @pytest.fixture(autouse=True)
    def _setup(self, mbias5_adj: np.ndarray) -> None:
        self.oracle = DSeparationOracle(mbias5_adj)

    def test_x_y_unconditional_not_dsep(self) -> None:
        # X->Y direct edge
        assert not self.oracle.is_d_separated(1, 4, frozenset())

    def test_x_y_cond_on_m_not_dsep(self) -> None:
        # Conditioning on M (a collider) opens a backdoor path
        assert not self.oracle.is_d_separated(1, 4, frozenset({2}))

    def test_u1_u2_unconditional_dsep(self) -> None:
        # U1 and U2 are d-separated unconditionally (M is a collider)
        assert self.oracle.is_d_separated(0, 3, frozenset())

    def test_u1_u2_cond_on_m_not_dsep(self) -> None:
        # Conditioning on collider M opens path
        assert not self.oracle.is_d_separated(0, 3, frozenset({2}))


# ═══════════════════════════════════════════════════════════════════════════
# CI implications enumeration
# ═══════════════════════════════════════════════════════════════════════════


class TestCIImplications:
    """Enumerate all CI implications."""

    def test_all_ci_implications_chain(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        cis = oracle.all_ci_implications(max_cond_size=2)
        # At least X _||_ Z | Y should be present
        found_xz_y = any(
            (x == 0 and y == 2 and 1 in cs) or (x == 2 and y == 0 and 1 in cs)
            for x, y, cs in cis
        )
        assert found_xz_y

    def test_all_ci_implications_empty_graph(self) -> None:
        adj = _adj(3, [])
        oracle = DSeparationOracle(adj)
        cis = oracle.all_ci_implications(max_cond_size=1)
        # All pairs are d-separated unconditionally
        unconditional = [
            (x, y, cs) for x, y, cs in cis if len(cs) == 0
        ]
        assert len(unconditional) >= 3  # (0,1), (0,2), (1,2)

    def test_all_ci_collider(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        oracle = DSeparationOracle(adj)
        cis = oracle.all_ci_implications(max_cond_size=1)
        # 0 _||_ 1 | {} should be present
        found = any(
            ({x, y} == {0, 1} and len(cs) == 0)
            for x, y, cs in cis
        )
        assert found


# ═══════════════════════════════════════════════════════════════════════════
# Minimal separating sets
# ═══════════════════════════════════════════════════════════════════════════


class TestSeparatingSets:
    def test_find_separating_set_chain(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        sep = oracle.find_separating_set(0, 2)
        assert sep is not None
        assert 1 in sep

    def test_find_separating_set_adjacent(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        # Adjacent nodes cannot be d-separated
        sep = oracle.find_separating_set(0, 1)
        # Should return None (no separating set exists for adjacent nodes)
        assert sep is None

    def test_all_d_separations_chain(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        seps = oracle.all_d_separations(0, 2, max_size=2)
        assert len(seps) >= 1
        assert any(1 in s for s in seps)

    def test_all_d_separations_fork(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2)])
        oracle = DSeparationOracle(adj)
        seps = oracle.all_d_separations(1, 2, max_size=2)
        assert len(seps) >= 1
        assert any(0 in s for s in seps)


# ═══════════════════════════════════════════════════════════════════════════
# d-connected set
# ═══════════════════════════════════════════════════════════════════════════


class TestDConnectedSet:
    def test_d_connected_chain(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        connected = oracle.d_connected_set(0, frozenset())
        # 0 is d-connected to 1 and 2 unconditionally
        assert 1 in connected
        assert 2 in connected

    def test_d_connected_cond_blocks(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        connected = oracle.d_connected_set(0, frozenset({1}))
        # Conditioning on 1 blocks 0 from reaching 2
        assert 2 not in connected


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise d-sep matrix
# ═══════════════════════════════════════════════════════════════════════════


class TestPairwiseMatrix:
    def test_pairwise_collider(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        oracle = DSeparationOracle(adj)
        mat = oracle.pairwise_dsep_matrix(frozenset())
        # 0 and 1 d-sep unconditionally
        assert mat[0, 1] == 1 or mat[0, 1]  # True/1
        # 0 and 2 not d-sep
        assert mat[0, 2] == 0 or not mat[0, 2]

    def test_pairwise_symmetric(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        oracle = DSeparationOracle(adj)
        mat = oracle.pairwise_dsep_matrix(frozenset())
        np.testing.assert_array_equal(mat, mat.T)

    def test_diagonal_is_false(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        mat = oracle.pairwise_dsep_matrix(frozenset())
        for i in range(3):
            assert not mat[i, i]


# ═══════════════════════════════════════════════════════════════════════════
# Markov blanket
# ═══════════════════════════════════════════════════════════════════════════


class TestMarkovBlanket:
    def test_markov_blanket_chain(self) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        oracle = DSeparationOracle(adj)
        mb = oracle.markov_blanket(1)
        assert 0 in mb  # parent
        assert 2 in mb  # child
        # 3 is not in Markov blanket of 1

    def test_markov_blanket_collider(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        oracle = DSeparationOracle(adj)
        mb = oracle.markov_blanket(0)
        assert 2 in mb  # child
        assert 1 in mb  # co-parent through child 2

    def test_markov_blanket_leaf(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2)])
        oracle = DSeparationOracle(adj)
        mb = oracle.markov_blanket(1)
        assert 0 in mb  # parent


# ═══════════════════════════════════════════════════════════════════════════
# Check triples batch
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckTriples:
    def test_batch_check(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        triples = [
            (0, 2, frozenset({1})),  # d-sep
            (0, 2, frozenset()),     # not d-sep
            (0, 1, frozenset()),     # not d-sep (adjacent)
        ]
        results = oracle.check_triples(triples)
        assert results[0] is True
        assert results[1] is False
        assert results[2] is False


# ═══════════════════════════════════════════════════════════════════════════
# Textbook examples
# ═══════════════════════════════════════════════════════════════════════════


class TestTextbookExamples:
    """Verify against known textbook d-separation examples."""

    def test_sprinkler_dag(self) -> None:
        """Classic sprinkler: Season->Rain, Season->Sprinkler,
        Rain->Wet, Sprinkler->Wet.
        0=Season, 1=Rain, 2=Sprinkler, 3=Wet."""
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        oracle = DSeparationOracle(adj)
        # Rain _||_ Sprinkler | Season (fork blocked)
        assert oracle.is_d_separated(1, 2, frozenset({0}))
        # Rain _not_||_ Sprinkler | Season, Wet (collider opened)
        assert not oracle.is_d_separated(1, 2, frozenset({0, 3}))

    def test_backdoor_path_blocked(self) -> None:
        """X<-C->Y, X->Y. Conditioning on C blocks backdoor."""
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])  # C=0, X=1, Y=2
        oracle = DSeparationOracle(adj)
        # 1 not d-sep from 2 unconditionally
        assert not oracle.is_d_separated(1, 2, frozenset())
        # After blocking backdoor through 0, still connected via direct edge
        assert not oracle.is_d_separated(1, 2, frozenset({0}))

    def test_instrumental_variable(self) -> None:
        """Z->X->Y. Z _||_ Y | X."""
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        assert oracle.is_d_separated(0, 2, frozenset({1}))


# ═══════════════════════════════════════════════════════════════════════════
# Property: Pa(X) blocks non-adjacent pairs
# ═══════════════════════════════════════════════════════════════════════════


class TestGlobalMarkovProperty:
    """X _||_ NonDesc(X) \ Pa(X) | Pa(X) in any DAG."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_local_markov_property(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(6, edge_prob=0.3, seed=seed)
        oracle = DSeparationOracle(adj)
        n = adj.shape[0]
        for v in range(n):
            pa_v = frozenset(np.where(adj[:, v] == 1)[0])
            # Non-descendants of v
            desc_v = set()
            stack = [v]
            visited = set()
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                for w in range(n):
                    if adj[u, w] == 1:
                        desc_v.add(w)
                        stack.append(w)
            non_desc = set(range(n)) - desc_v - {v} - set(pa_v)
            for nd in non_desc:
                assert oracle.is_d_separated(v, nd, pa_v), (
                    f"Seed {seed}: node {v} not d-sep from non-desc {nd} given Pa={pa_v}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Active paths
# ═══════════════════════════════════════════════════════════════════════════


class TestActivePaths:
    def test_active_paths_chain(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        paths = oracle.active_paths(0, 2, frozenset())
        assert len(paths) >= 1

    def test_no_active_paths_when_dsep(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        paths = oracle.active_paths(0, 2, frozenset({1}))
        assert len(paths) == 0

    def test_collider_opens_path(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        oracle = DSeparationOracle(adj)
        paths = oracle.active_paths(0, 1, frozenset({2}))
        assert len(paths) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Edge case: empty graph
# ═══════════════════════════════════════════════════════════════════════════


class TestEmptyGraph:
    def test_all_pairs_dsep(self) -> None:
        adj = _adj(4, [])
        oracle = DSeparationOracle(adj)
        for i in range(4):
            for j in range(i + 1, 4):
                assert oracle.is_d_separated(i, j, frozenset())

    def test_no_active_paths(self) -> None:
        adj = _adj(3, [])
        oracle = DSeparationOracle(adj)
        paths = oracle.active_paths(0, 1, frozenset())
        assert len(paths) == 0
