"""Shared fixtures for the CausalQD test suite.

Provides DAGs of various sizes, synthetic data generators, pre-populated
archives, and known SCMs that are reused across many test modules.
"""

from __future__ import annotations

import copy
import os
import tempfile
from typing import List, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the package is importable regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causal_qd.core.dag import DAG, DAGError
from causal_qd.types import AdjacencyMatrix, DataMatrix


# ===================================================================
# DAG Fixtures — small (5), medium (10), large (20)
# ===================================================================

@pytest.fixture
def small_dag() -> DAG:
    """5-node chain: 0→1→2→3→4."""
    adj = np.zeros((5, 5), dtype=np.int8)
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    return DAG(adj)


@pytest.fixture
def small_dag_fork() -> DAG:
    """5-node fork: 0→1, 0→2, 0→3, 0→4."""
    adj = np.zeros((5, 5), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[0, 3] = 1
    adj[0, 4] = 1
    return DAG(adj)


@pytest.fixture
def small_dag_collider() -> DAG:
    """5-node collider: 0→2, 1→2, 3→4, 2→4."""
    adj = np.zeros((5, 5), dtype=np.int8)
    adj[0, 2] = 1
    adj[1, 2] = 1
    adj[3, 4] = 1
    adj[2, 4] = 1
    return DAG(adj)


@pytest.fixture
def medium_dag() -> DAG:
    """10-node DAG with a mix of chains, forks, and colliders.

    Structure:
        0→1, 0→2, 1→3, 2→3 (collider at 3)
        3→4, 4→5, 4→6
        5→7, 6→7 (collider at 7)
        7→8, 8→9
    """
    adj = np.zeros((10, 10), dtype=np.int8)
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (3, 4), (4, 5), (4, 6),
        (5, 7), (6, 7),
        (7, 8), (8, 9),
    ]
    for i, j in edges:
        adj[i, j] = 1
    return DAG(adj)


@pytest.fixture
def large_dag() -> DAG:
    """20-node DAG generated with a fixed seed for reproducibility."""
    rng = np.random.default_rng(42)
    return DAG.random_dag(20, edge_prob=0.2, rng=rng)


@pytest.fixture
def empty_dag() -> DAG:
    """5-node DAG with no edges."""
    return DAG.empty(5)


@pytest.fixture
def complete_dag() -> DAG:
    """5-node fully connected DAG (all forward edges in topological order)."""
    n = 5
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = 1
    return DAG(adj)


# ===================================================================
# Adjacency matrix fixtures (raw numpy arrays)
# ===================================================================

@pytest.fixture
def small_adj() -> AdjacencyMatrix:
    """5-node chain as raw adjacency matrix."""
    adj = np.zeros((5, 5), dtype=np.int8)
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    return adj


@pytest.fixture
def medium_adj() -> AdjacencyMatrix:
    """10-node adjacency matrix matching medium_dag."""
    adj = np.zeros((10, 10), dtype=np.int8)
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (3, 4), (4, 5), (4, 6),
        (5, 7), (6, 7),
        (7, 8), (8, 9),
    ]
    for i, j in edges:
        adj[i, j] = 1
    return adj


# ===================================================================
# Data fixtures
# ===================================================================

@pytest.fixture
def random_data() -> DataMatrix:
    """200×5 random Gaussian data matrix."""
    rng = np.random.default_rng(12345)
    return rng.standard_normal((200, 5))


@pytest.fixture
def random_data_10() -> DataMatrix:
    """500×10 random Gaussian data matrix."""
    rng = np.random.default_rng(54321)
    return rng.standard_normal((500, 10))


@pytest.fixture
def gaussian_data() -> Tuple[DataMatrix, AdjacencyMatrix]:
    """Data generated from a known linear Gaussian SCM: 0→1→2→3→4.

    Returns (data, true_adj) where data has 500 samples and 5 variables.
    """
    rng = np.random.default_rng(99)
    n_samples = 500
    n_vars = 5

    # True DAG: chain 0→1→2→3→4
    true_adj = np.zeros((n_vars, n_vars), dtype=np.int8)
    for i in range(n_vars - 1):
        true_adj[i, i + 1] = 1

    # Generate data in topological order
    weights = np.array([0.8, 0.7, 0.9, 0.6])  # edge weights
    data = np.zeros((n_samples, n_vars))
    data[:, 0] = rng.standard_normal(n_samples)
    for i in range(1, n_vars):
        data[:, i] = weights[i - 1] * data[:, i - 1] + rng.standard_normal(n_samples) * 0.5

    return data, true_adj


@pytest.fixture
def gaussian_data_medium() -> Tuple[DataMatrix, AdjacencyMatrix]:
    """Data generated from a 10-node linear Gaussian SCM.

    Returns (data, true_adj) with 1000 samples and 10 variables.
    """
    rng = np.random.default_rng(777)
    n_samples = 1000
    n_vars = 10

    true_adj = np.zeros((n_vars, n_vars), dtype=np.int8)
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (3, 4), (4, 5), (4, 6),
        (5, 7), (6, 7),
        (7, 8), (8, 9),
    ]
    for i, j in edges:
        true_adj[i, j] = 1

    # Generate data
    topo = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data = np.zeros((n_samples, n_vars))
    for node in topo:
        parents = np.where(true_adj[:, node])[0]
        if len(parents) == 0:
            data[:, node] = rng.standard_normal(n_samples)
        else:
            w = rng.uniform(0.5, 1.0, size=len(parents))
            data[:, node] = data[:, parents] @ w + rng.standard_normal(n_samples) * 0.3

    return data, true_adj


@pytest.fixture
def discrete_data() -> DataMatrix:
    """500×4 discrete data (3 categories each)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 3, size=(500, 4)).astype(np.float64)


# ===================================================================
# Archive fixtures
# ===================================================================

@pytest.fixture
def sample_archive():
    """Pre-populated GridArchive with 10 elites."""
    from causal_qd.archive.grid_archive import GridArchive
    from causal_qd.archive.archive_base import ArchiveEntry

    archive = GridArchive(
        dims=(5, 5),
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([1.0, 1.0]),
    )
    rng = np.random.default_rng(42)
    for i in range(10):
        adj = np.zeros((5, 5), dtype=np.int8)
        if i > 0:
            adj[0, i % 5] = 1
        desc = rng.uniform(0, 1, size=2)
        entry = ArchiveEntry(
            solution=adj,
            descriptor=desc,
            quality=float(rng.uniform(-100, -10)),
        )
        archive.add(entry)
    return archive


@pytest.fixture
def empty_archive():
    """Empty GridArchive."""
    from causal_qd.archive.grid_archive import GridArchive

    return GridArchive(
        dims=(5, 5),
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([1.0, 1.0]),
    )


# ===================================================================
# SCM fixtures
# ===================================================================

@pytest.fixture
def known_scm():
    """LinearGaussianSCM with known structure (chain 0→1→2→3→4)."""
    from causal_qd.data.scm import LinearGaussianSCM

    adj = np.zeros((5, 5), dtype=np.int8)
    for i in range(4):
        adj[i, i + 1] = 1
    dag = DAG(adj)

    weights = np.zeros((5, 5), dtype=np.float64)
    weights[0, 1] = 0.8
    weights[1, 2] = 0.7
    weights[2, 3] = 0.9
    weights[3, 4] = 0.6

    noise_std = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    return LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)


@pytest.fixture
def known_scm_fork():
    """LinearGaussianSCM with fork structure: 0→1, 0→2, 0→3."""
    from causal_qd.data.scm import LinearGaussianSCM

    n = 4
    adj = np.zeros((n, n), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[0, 3] = 1
    dag = DAG(adj)

    weights = np.zeros((n, n), dtype=np.float64)
    weights[0, 1] = 0.9
    weights[0, 2] = 0.7
    weights[0, 3] = 0.5

    noise_std = np.ones(n) * 0.5
    return LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)


# ===================================================================
# Scorer fixtures
# ===================================================================

@pytest.fixture
def bic_scorer():
    """BICScore instance with default settings."""
    from causal_qd.scores.bic import BICScore
    return BICScore()


@pytest.fixture
def bdeu_scorer():
    """BDeuScore instance with ESS=1."""
    from causal_qd.scores.bdeu import BDeuScore
    return BDeuScore(equivalent_sample_size=1.0)


@pytest.fixture
def bge_scorer():
    """BGeScore instance with default settings."""
    from causal_qd.scores.bge import BGeScore
    return BGeScore()


# ===================================================================
# RNG fixture
# ===================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


# ===================================================================
# Temporary directory fixture
# ===================================================================

@pytest.fixture
def tmp_dir():
    """Temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d
