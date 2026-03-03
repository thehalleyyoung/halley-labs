"""Unit tests for cpa.mec.separation – d-separation and Bayes-Ball."""

from __future__ import annotations

import numpy as np
import pytest

from cpa.mec.separation import (
    BayesBall,
    d_separation,
    find_dsep_set,
    find_all_d_separations,
    markov_blanket,
    markov_boundary,
    minimal_d_sep_set,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def chain_adj():
    """Chain: 0 → 1 → 2."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[1, 2] = 1
    return adj


@pytest.fixture
def fork_adj():
    """Fork: 1 ← 0 → 2."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    return adj


@pytest.fixture
def collider_adj():
    """Collider: 0 → 2 ← 1."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 2] = 1
    adj[1, 2] = 1
    return adj


@pytest.fixture
def diamond_adj():
    """Diamond: 0→1, 0→2, 1→3, 2→3."""
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    return adj


@pytest.fixture
def five_node_adj():
    """5-node: 0→1, 0→2, 1→3, 2→3, 3→4."""
    adj = np.zeros((5, 5), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    return adj


# ===================================================================
# Tests – d-separation on known graphs
# ===================================================================


class TestDSeparationChain:
    """Test d-separation on chain structures."""

    def test_chain_unconditioned_dependent(self, chain_adj):
        assert not d_separation(chain_adj, {0}, {2}, set())

    def test_chain_conditioned_on_middle(self, chain_adj):
        assert d_separation(chain_adj, {0}, {2}, {1})

    def test_chain_conditioning_on_leaf(self, chain_adj):
        assert not d_separation(chain_adj, {0}, {1}, {2})

    def test_adjacent_not_separated(self, chain_adj):
        assert not d_separation(chain_adj, {0}, {1}, set())


class TestDSeparationFork:
    """Test d-separation on fork structures."""

    def test_fork_unconditioned_dependent(self, fork_adj):
        assert not d_separation(fork_adj, {1}, {2}, set())

    def test_fork_conditioned_on_root(self, fork_adj):
        assert d_separation(fork_adj, {1}, {2}, {0})


class TestDSeparationCollider:
    """Test d-separation on collider structures."""

    def test_collider_unconditioned_separated(self, collider_adj):
        assert d_separation(collider_adj, {0}, {1}, set())

    def test_collider_conditioned_on_child(self, collider_adj):
        assert not d_separation(collider_adj, {0}, {1}, {2})


class TestDSeparationDiamond:
    """Test d-separation on diamond structures."""

    def test_diamond_unconditioned(self, diamond_adj):
        assert not d_separation(diamond_adj, {1}, {2}, set())

    def test_diamond_conditioned_on_root(self, diamond_adj):
        assert d_separation(diamond_adj, {1}, {2}, {0})

    def test_diamond_conditioned_on_leaf(self, diamond_adj):
        # Conditioning on collider opens path
        assert not d_separation(diamond_adj, {1}, {2}, {3})

    def test_diamond_conditioned_on_root_and_leaf(self, diamond_adj):
        # {0} blocks common cause, {3} opens collider → still dependent
        assert not d_separation(diamond_adj, {1}, {2}, {0, 3})


# ===================================================================
# Tests – BayesBall algorithm
# ===================================================================


class TestBayesBall:
    """Test BayesBall class directly."""

    def test_chain_separated(self, chain_adj):
        bb = BayesBall(chain_adj)
        assert bb.is_d_separated({0}, {2}, {1})

    def test_chain_not_separated(self, chain_adj):
        bb = BayesBall(chain_adj)
        assert not bb.is_d_separated({0}, {2}, set())

    def test_collider_separated(self, collider_adj):
        bb = BayesBall(collider_adj)
        assert bb.is_d_separated({0}, {1}, set())

    def test_collider_opened(self, collider_adj):
        bb = BayesBall(collider_adj)
        assert not bb.is_d_separated({0}, {1}, {2})

    def test_same_node_not_separated(self, chain_adj):
        bb = BayesBall(chain_adj)
        assert not bb.is_d_separated({0}, {0}, set())


# ===================================================================
# Tests – Minimal d-separating set
# ===================================================================


class TestMinimalDSepSet:
    """Test minimal d-separating set computation."""

    def test_chain_dsep_set(self, chain_adj):
        dsep = find_dsep_set(chain_adj, 0, 2)
        assert dsep is not None
        assert 1 in dsep

    def test_fork_dsep_set(self, fork_adj):
        dsep = find_dsep_set(fork_adj, 1, 2)
        assert dsep is not None
        assert 0 in dsep

    def test_adjacent_no_dsep(self, chain_adj):
        dsep = find_dsep_set(chain_adj, 0, 1)
        assert dsep is None

    def test_minimal_dsep_set(self, five_node_adj):
        dsep = minimal_d_sep_set(five_node_adj, 0, 4)
        if dsep is not None:
            # Verify it actually d-separates
            assert d_separation(five_node_adj, {0}, {4}, dsep)

    def test_find_all_d_separations(self, chain_adj):
        all_sets = find_all_d_separations(chain_adj, 0, 2, max_size=3)
        assert len(all_sets) >= 1
        for s in all_sets:
            assert d_separation(chain_adj, {0}, {2}, set(s))


# ===================================================================
# Tests – Markov blanket / boundary
# ===================================================================


class TestMarkovBlanket:
    """Test Markov boundary = parents + children + co-parents."""

    def test_chain_middle_blanket(self, chain_adj):
        mb = markov_blanket(chain_adj, 1)
        assert 0 in mb  # parent
        assert 2 in mb  # child

    def test_chain_root_blanket(self, chain_adj):
        mb = markov_blanket(chain_adj, 0)
        assert 1 in mb  # child

    def test_chain_leaf_blanket(self, chain_adj):
        mb = markov_blanket(chain_adj, 2)
        assert 1 in mb  # parent

    def test_collider_includes_coparents(self, collider_adj):
        # For node 0: child=2, co-parent of 2 is 1
        mb = markov_blanket(collider_adj, 0)
        assert 2 in mb  # child
        assert 1 in mb  # co-parent (spouse)

    def test_diamond_blanket_node3(self, diamond_adj):
        mb = markov_blanket(diamond_adj, 3)
        assert 1 in mb  # parent
        assert 2 in mb  # parent

    def test_markov_boundary_alias(self, chain_adj):
        mb = markov_blanket(chain_adj, 1)
        mbnd = markov_boundary(chain_adj, 1)
        assert mb == mbnd

    def test_node_not_in_own_blanket(self, chain_adj):
        mb = markov_blanket(chain_adj, 1)
        assert 1 not in mb

    def test_five_node_markov_blanket(self, five_node_adj):
        mb = markov_blanket(five_node_adj, 3)
        assert 1 in mb  # parent
        assert 2 in mb  # parent
        assert 4 in mb  # child

    def test_dsep_given_blanket(self, five_node_adj):
        """Node is d-separated from non-blanket given blanket."""
        mb = markov_blanket(five_node_adj, 1)
        non_blanket = set(range(5)) - mb - {1}
        for node in non_blanket:
            assert d_separation(five_node_adj, {1}, {node}, mb)
