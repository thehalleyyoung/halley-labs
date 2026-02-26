"""
Shared pytest fixtures for Causal-Shielded Adaptive Trading test suite.

Provides synthetic data generators, known DAG structures, regime sequences,
transition matrices, configuration objects, and random seed management.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Ensure the implementation package is importable
# ---------------------------------------------------------------------------
_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)


# ===== Random Seed Management =====

@pytest.fixture
def rng():
    """Deterministic NumPy random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def seed():
    """Fixed integer seed for APIs that accept ``seed`` kwargs."""
    return 42


# ===== Small Synthetic Data =====

@pytest.fixture
def small_returns(rng):
    """100-step synthetic return series (univariate)."""
    return rng.normal(0.0, 0.01, size=100)


@pytest.fixture
def small_multivariate_returns(rng):
    """200×5 multivariate returns with mild cross-correlation."""
    cov = np.eye(5) * 0.01
    cov[0, 1] = cov[1, 0] = 0.005
    cov[2, 3] = cov[3, 2] = 0.003
    return rng.multivariate_normal(np.zeros(5), cov, size=200)


@pytest.fixture
def regime_switching_data(rng):
    """300-step data with two clear mean-separated regimes.

    Returns (data, true_labels) where regime 0 has mean 0 and regime 1
    has mean 2.0.
    """
    n = 300
    labels = np.zeros(n, dtype=int)
    labels[100:200] = 1
    data = np.empty(n)
    data[labels == 0] = rng.normal(0.0, 0.5, size=(labels == 0).sum())
    data[labels == 1] = rng.normal(2.0, 0.5, size=(labels == 1).sum())
    return data, labels


@pytest.fixture
def three_regime_data(rng):
    """600-step data with three regimes cycling twice."""
    segment = 100
    labels = np.concatenate([np.full(segment, k) for k in [0, 1, 2, 0, 1, 2]])
    means = {0: 0.0, 1: 3.0, 2: -2.0}
    data = np.array([rng.normal(means[l], 0.4) for l in labels])
    return data, labels


@pytest.fixture
def multivariate_regime_data(rng):
    """400×4 data with 2 regimes differing in covariance structure."""
    n, d = 400, 4
    labels = np.zeros(n, dtype=int)
    labels[200:] = 1
    cov0 = np.eye(d) * 0.5
    cov1 = np.eye(d) * 0.5
    cov1[0, 1] = cov1[1, 0] = 0.4
    data = np.empty((n, d))
    data[:200] = rng.multivariate_normal(np.zeros(d), cov0, size=200)
    data[200:] = rng.multivariate_normal(np.ones(d) * 0.5, cov1, size=200)
    return data, labels


# ===== Known DAG Structures =====

@pytest.fixture
def chain_dag():
    """Simple chain: X0 -> X1 -> X2 -> X3."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return g


@pytest.fixture
def fork_dag():
    """Fork (common cause): X0 -> X1, X0 -> X2."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (0, 2)])
    return g


@pytest.fixture
def collider_dag():
    """Collider: X0 -> X2 <- X1."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 2), (1, 2)])
    return g


@pytest.fixture
def diamond_dag():
    """Diamond: X0 -> X1, X0 -> X2, X1 -> X3, X2 -> X3."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    return g


@pytest.fixture
def named_chain_dag():
    """Chain with named nodes: A -> B -> C -> D."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return g


@pytest.fixture
def five_node_dag():
    """Non-trivial 5-node DAG for structure learning tests."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    return g


# ===== Regime Sequences and Transition Matrices =====

@pytest.fixture
def known_transition_matrix():
    """2-state transition matrix with high self-transition."""
    return np.array([[0.95, 0.05],
                     [0.10, 0.90]])


@pytest.fixture
def three_state_transition_matrix():
    """3-state transition matrix."""
    return np.array([
        [0.90, 0.05, 0.05],
        [0.03, 0.92, 0.05],
        [0.04, 0.06, 0.90],
    ])


@pytest.fixture
def known_regime_sequence(known_transition_matrix, rng):
    """500-step regime sequence from the known 2-state transition matrix."""
    T = known_transition_matrix
    n = 500
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(len(T), p=T[states[t - 1]])
    return states


# ===== Linear SCM Fixtures =====

@pytest.fixture
def linear_scm():
    """A -> B -> C with known linear coefficients."""
    from causal_trading.causal.scm import StructuralCausalModel, LinearEquation

    scm = StructuralCausalModel("test_linear")
    scm.add_variable("A", LinearEquation(weights={}, intercept=0.0, noise_std=1.0))
    scm.add_variable("B", LinearEquation(weights={"A": 0.8}, intercept=0.0, noise_std=0.5))
    scm.add_variable("C", LinearEquation(weights={"B": -0.6}, intercept=1.0, noise_std=0.5))
    scm.add_edge("A", "B")
    scm.add_edge("B", "C")
    return scm


@pytest.fixture
def fork_scm():
    """X -> Y, X -> Z (common cause)."""
    from causal_trading.causal.scm import StructuralCausalModel, LinearEquation

    scm = StructuralCausalModel("fork")
    scm.add_variable("X", LinearEquation(weights={}, intercept=0.0, noise_std=1.0))
    scm.add_variable("Y", LinearEquation(weights={"X": 1.2}, noise_std=0.5))
    scm.add_variable("Z", LinearEquation(weights={"X": -0.9}, noise_std=0.5))
    scm.add_edge("X", "Y")
    scm.add_edge("X", "Z")
    return scm


# ===== Safety Specification Helpers =====

@pytest.fixture
def safe_trajectory():
    """Trajectory that satisfies a 10% drawdown spec."""
    values = [100.0]
    for _ in range(49):
        values.append(values[-1] * 1.001)
    return [{"portfolio_value": v, "position": 10.0, "margin_ratio": 0.5}
            for v in values]


@pytest.fixture
def unsafe_trajectory():
    """Trajectory that violates a 10% drawdown spec."""
    values = [100.0]
    for _ in range(24):
        values.append(values[-1] * 0.99)
    for _ in range(25):
        values.append(values[-1] * 0.97)
    return [{"portfolio_value": v, "position": 10.0, "margin_ratio": 0.5}
            for v in values]


# ===== MDP Fixtures =====

@pytest.fixture
def small_mdp_transition():
    """3-state, 2-action transition tensor for small MDP tests."""
    T = np.zeros((2, 3, 3))
    # Action 0: safe (stays in good states)
    T[0] = np.array([[0.8, 0.15, 0.05],
                      [0.1, 0.7, 0.2],
                      [0.05, 0.15, 0.8]])
    # Action 1: risky (may move to bad state)
    T[1] = np.array([[0.3, 0.3, 0.4],
                      [0.1, 0.3, 0.6],
                      [0.05, 0.05, 0.9]])
    return T


@pytest.fixture
def small_reward_matrix():
    """3-state, 2-action reward matrix."""
    return np.array([[1.0, 2.0],
                     [0.5, 1.5],
                     [-1.0, -0.5]])


# ===== Config Helpers =====

@dataclass
class TestConfig:
    n_regimes: int = 3
    n_features: int = 5
    n_samples: int = 200
    alpha: float = 0.05
    delta: float = 0.05
    horizon: int = 10
    max_drawdown: float = 0.10
    seed: int = 42


@pytest.fixture
def test_config():
    return TestConfig()
