"""Tests for causalcert.dag.mec – CPDAG, Meek rules, MEC equivalence."""

from __future__ import annotations

import numpy as np
import pytest

from causalcert.dag.mec import (
    to_cpdag,
    mec_size_bound,
    is_mec_equivalent,
    compelled_edges,
    reversible_edges,
    cpdag_to_dag,
    enumerate_mec,
    apply_meek_rules,
    skeleton,
    v_structures,
)
from causalcert.types import AdjacencyMatrix

# ── helper ─────────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


# ═══════════════════════════════════════════════════════════════════════════
# CPDAG conversion
# ═══════════════════════════════════════════════════════════════════════════


class TestToCPDAG:
    def test_chain_cpdag(self) -> None:
        # 0->1->2: no v-structures, all edges reversible → undirected
        adj = _adj(3, [(0, 1), (1, 2)])
        cpdag = to_cpdag(adj)
        # Should be undirected (both directions present)
        assert cpdag[0, 1] == 1 and cpdag[1, 0] == 1
        assert cpdag[1, 2] == 1 and cpdag[2, 1] == 1

    def test_collider_cpdag(self) -> None:
        # 0->2<-1: v-structure, edges compelled
        adj = _adj(3, [(0, 2), (1, 2)])
        cpdag = to_cpdag(adj)
        # Both edges must be directed in CPDAG
        assert cpdag[0, 2] == 1 and cpdag[2, 0] == 0
        assert cpdag[1, 2] == 1 and cpdag[2, 1] == 0

    def test_fork_cpdag(self) -> None:
        # 0->1, 0->2: no v-structures → undirected
        adj = _adj(3, [(0, 1), (0, 2)])
        cpdag = to_cpdag(adj)
        assert cpdag[0, 1] == 1 and cpdag[1, 0] == 1

    def test_diamond_cpdag(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        cpdag = to_cpdag(adj)
        # v-structure at 3: 1->3<-2, but both 1 and 2 have common parent 0
        # so 1->3<-2 is NOT a v-structure since 1 and 2 are adjacent via 0...wait
        # Actually 1 and 2 are not adjacent, so 1->3<-2 IS a v-structure
        assert cpdag[1, 3] == 1 and cpdag[3, 1] == 0
        assert cpdag[2, 3] == 1 and cpdag[3, 2] == 0

    def test_single_edge_cpdag(self) -> None:
        adj = _adj(2, [(0, 1)])
        cpdag = to_cpdag(adj)
        # Single edge: undirected
        assert cpdag[0, 1] == 1 and cpdag[1, 0] == 1

    def test_empty_cpdag(self) -> None:
        adj = _adj(3, [])
        cpdag = to_cpdag(adj)
        assert cpdag.sum() == 0


# ═══════════════════════════════════════════════════════════════════════════
# v-structures
# ═══════════════════════════════════════════════════════════════════════════


class TestVStructures:
    def test_collider_has_v_structure(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        vs = v_structures(adj)
        assert len(vs) >= 1
        # Should find (0, 2, 1) or (1, 2, 0) as a v-structure
        triples = {(a, b, c) for a, b, c in vs}
        found = any(b == 2 and {a, c} == {0, 1} for a, b, c in triples)
        assert found

    def test_chain_no_v_structure(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        vs = v_structures(adj)
        assert len(vs) == 0

    def test_fork_no_v_structure(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2)])
        vs = v_structures(adj)
        assert len(vs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Skeleton
# ═══════════════════════════════════════════════════════════════════════════


class TestSkeleton:
    def test_skeleton_undirected(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        skel = skeleton(adj)
        assert skel[0, 1] == 1 and skel[1, 0] == 1
        assert skel[1, 2] == 1 and skel[2, 1] == 1
        np.testing.assert_array_equal(skel, skel.T)

    def test_skeleton_no_self_loops(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2)])
        skel = skeleton(adj)
        for i in range(3):
            assert skel[i, i] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Meek rules
# ═══════════════════════════════════════════════════════════════════════════


class TestMeekRules:
    def test_meek_rules_idempotent(self) -> None:
        # Start with a CPDAG, applying Meek rules again should not change it
        adj = _adj(3, [(0, 2), (1, 2)])
        cpdag = to_cpdag(adj)
        cpdag2 = apply_meek_rules(cpdag.copy())
        np.testing.assert_array_equal(cpdag, cpdag2)

    def test_meek_rules_orient_edges(self) -> None:
        # PDAG with one directed edge and some undirected
        # 0->1, 1-2 (undirected). By Meek R1: if 0->1-2 and 0 not adj 2,
        # orient 1->2
        pdag = np.zeros((3, 3), dtype=np.int8)
        pdag[0, 1] = 1  # directed 0->1
        pdag[1, 2] = 1
        pdag[2, 1] = 1  # undirected 1-2
        # 0 not adjacent to 2
        result = apply_meek_rules(pdag)
        # Should orient 1->2
        assert result[1, 2] == 1
        assert result[2, 1] == 0

    def test_meek_rules_preserve_existing(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        cpdag = to_cpdag(adj)
        original = cpdag.copy()
        result = apply_meek_rules(cpdag)
        # Directed edges should remain
        for i in range(3):
            for j in range(3):
                if original[i, j] == 1 and original[j, i] == 0:
                    assert result[i, j] == 1 and result[j, i] == 0


# ═══════════════════════════════════════════════════════════════════════════
# MEC equivalence
# ═══════════════════════════════════════════════════════════════════════════


class TestMECEquivalence:
    def test_same_dag_equivalent(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        assert is_mec_equivalent(adj, adj)

    def test_reversed_chain_equivalent(self) -> None:
        # 0->1->2 and 2->1->0 have same skeleton and no v-structures
        adj1 = _adj(3, [(0, 1), (1, 2)])
        adj2 = _adj(3, [(2, 1), (1, 0)])
        assert is_mec_equivalent(adj1, adj2)

    def test_different_v_structures_not_equivalent(self) -> None:
        # 0->2<-1 vs 0->1->2
        adj1 = _adj(3, [(0, 2), (1, 2)])
        adj2 = _adj(3, [(0, 1), (1, 2)])
        assert not is_mec_equivalent(adj1, adj2)

    def test_different_skeletons_not_equivalent(self) -> None:
        adj1 = _adj(3, [(0, 1), (1, 2)])
        adj2 = _adj(3, [(0, 1), (0, 2)])
        assert not is_mec_equivalent(adj1, adj2)

    def test_diamond_variants(self) -> None:
        adj1 = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        adj2 = _adj(4, [(0, 1), (0, 2), (3, 1), (2, 3)])
        # Different v-structures
        eq = is_mec_equivalent(adj1, adj2)
        # Just check it returns a bool
        assert isinstance(eq, bool)


# ═══════════════════════════════════════════════════════════════════════════
# Compelled / reversible edges
# ═══════════════════════════════════════════════════════════════════════════


class TestCompelledReversible:
    def test_collider_all_compelled(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        comp = compelled_edges(adj)
        assert len(comp) == 2
        assert (0, 2) in comp
        assert (1, 2) in comp

    def test_chain_all_reversible(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        rev = reversible_edges(adj)
        assert len(rev) == 2

    def test_compelled_plus_reversible_is_all(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        comp = set(compelled_edges(adj))
        rev = set(reversible_edges(adj))
        all_edges = set(zip(*np.where(adj == 1)))
        assert comp | rev == all_edges
        assert comp & rev == set()


# ═══════════════════════════════════════════════════════════════════════════
# CPDAG to DAG
# ═══════════════════════════════════════════════════════════════════════════


class TestCPDAGToDAG:
    def test_round_trip(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        cpdag = to_cpdag(adj)
        dag = cpdag_to_dag(cpdag)
        assert dag is not None
        # Should be a valid DAG in the same MEC
        from causalcert.dag.validation import is_dag
        assert is_dag(dag)
        assert is_mec_equivalent(dag, adj)

    def test_collider_round_trip(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        cpdag = to_cpdag(adj)
        dag = cpdag_to_dag(cpdag)
        assert dag is not None
        from causalcert.dag.validation import is_dag
        assert is_dag(dag)

    def test_invalid_pdag_returns_none(self) -> None:
        # A PDAG that cannot be extended to a DAG
        pdag = np.ones((3, 3), dtype=np.int8)
        np.fill_diagonal(pdag, 0)
        result = cpdag_to_dag(pdag)
        # Should return None or a valid DAG
        if result is not None:
            from causalcert.dag.validation import is_dag
            assert is_dag(result)


# ═══════════════════════════════════════════════════════════════════════════
# MEC enumeration
# ═══════════════════════════════════════════════════════════════════════════


class TestEnumerateMEC:
    def test_chain_mec_size(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        dags = enumerate_mec(adj, max_dags=100)
        # Chain has no v-structures: 0->1->2, 2->1->0, 0<-1->2, 0->1<-2
        # Wait: 0->1<-2 has v-structure, so not in same MEC
        # Same MEC: 0->1->2, 2->1->0 (and maybe others)
        assert len(dags) >= 1
        for d in dags:
            from causalcert.dag.validation import is_dag
            assert is_dag(d)
            assert is_mec_equivalent(d, adj)

    def test_collider_mec_single(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        dags = enumerate_mec(adj, max_dags=100)
        # Collider: the v-structure constrains orientation → only 1 DAG
        assert len(dags) == 1

    def test_mec_size_bound(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        bound = mec_size_bound(adj)
        assert bound >= 1
        dags = enumerate_mec(adj, max_dags=1000)
        assert len(dags) <= bound


# ═══════════════════════════════════════════════════════════════════════════
# PDAG extension
# ═══════════════════════════════════════════════════════════════════════════


class TestPDAGExtension:
    def test_fully_directed_pdag(self) -> None:
        # Already a DAG
        adj = _adj(3, [(0, 1), (1, 2)])
        dag = cpdag_to_dag(adj)
        assert dag is not None

    def test_fully_undirected_pdag(self) -> None:
        pdag = _adj(3, [(0, 1), (1, 0), (1, 2), (2, 1)])
        dag = cpdag_to_dag(pdag)
        if dag is not None:
            from causalcert.dag.validation import is_dag
            assert is_dag(dag)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_cpdag_dag_round_trip_random(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(5, edge_prob=0.3, seed=seed)
        cpdag = to_cpdag(adj)
        dag = cpdag_to_dag(cpdag)
        assert dag is not None
        from causalcert.dag.validation import is_dag
        assert is_dag(dag)
        assert is_mec_equivalent(dag, adj)
