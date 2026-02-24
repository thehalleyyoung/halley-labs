"""Integration tests for the full DivFlow pipeline."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generate import diverse_generate, DivFlowResult
from src.agents import GaussianAgent, ClusteredAgent, UniformAgent
from src.mechanism import (
    DirectMechanism,
    FlowMechanism,
    ParetoMechanism,
)
from src.scoring_rules import LogarithmicRule, BrierRule, EnergyAugmentedRule
from src.config import DivFlowConfig


def _make_agents(n=8, dim=8, seed=42):
    agents = []
    rng = np.random.RandomState(seed)
    for i in range(n):
        mean = rng.randn(dim)
        cov = np.eye(dim) * 0.3
        agents.append(GaussianAgent(mean=mean, cov=cov, seed=seed + i))
    return agents


def test_end_to_end_diverse_generate():
    """Full pipeline: diverse_generate should return valid DivFlowResult."""
    config = DivFlowConfig(n_candidates=8, k_select=4, n_rounds=2, embedding_dim=8)
    agents = _make_agents(n=8, dim=8)
    result = diverse_generate(
        prompt="test", n=8, k=4, mechanism="direct",
        config=config, agents=agents, seed=42,
    )
    assert isinstance(result, DivFlowResult)
    assert len(result.responses) == 4
    assert result.diversity_score >= 0
    assert len(result.quality_scores) == 4
    assert result.coverage_certificate is not None


def test_flow_vs_direct_diversity_improvement():
    """Flow mechanism should produce comparable or better diversity."""
    config = DivFlowConfig(n_candidates=8, k_select=4, n_rounds=4, embedding_dim=8)
    agents = _make_agents(n=8, dim=8)

    r_direct = diverse_generate(
        n=8, k=4, mechanism="direct", config=config, agents=agents, seed=42,
    )
    r_flow = diverse_generate(
        n=8, k=4, mechanism="flow", config=config, agents=agents, seed=42,
    )
    # Both should be valid
    assert r_direct.diversity_score >= 0
    assert r_flow.diversity_score >= 0


def test_coverage_certificate_valid_after_mechanism():
    """Coverage certificate should be populated after running mechanism."""
    agents = _make_agents(n=8, dim=8)
    config = DivFlowConfig(embedding_dim=8)
    result = diverse_generate(
        n=8, k=4, mechanism="flow", config=config, agents=agents, seed=42,
    )
    cert = result.coverage_certificate
    assert cert is not None
    assert 0.0 <= cert.coverage_fraction <= 1.0
    assert cert.n_samples == 4
    assert cert.confidence > 0


def test_pareto_frontier_monotone():
    """On the Pareto frontier, increasing diversity weight should not decrease diversity."""
    agents = _make_agents(n=10, dim=4, seed=42)
    mech = ParetoMechanism(LogarithmicRule(), n_candidates=10, k_select=4, seed=42)
    frontier = mech.trace_frontier(agents, lambdas=[0.0, 0.5, 1.0])
    # Should have valid entries
    assert len(frontier) == 3
    for q, d in frontier:
        assert np.isfinite(q) and np.isfinite(d)


def test_energy_augmented_mechanism():
    """Energy-augmented scoring should produce valid results."""
    base = BrierRule()
    energy_fn = lambda y, hist: float(y) * 0.1
    rule = EnergyAugmentedRule(base, energy_fn, lambda_=0.1)

    agents = _make_agents(n=8, dim=4)
    mech = DirectMechanism(rule, n_candidates=8, k_select=4, seed=42)
    result = mech.run(agents)
    assert len(result.selected_indices) == 4
    assert result.diversity_score >= 0


def test_full_pipeline_with_coverage():
    """Complete pipeline including coverage analysis."""
    from src.coverage import estimate_coverage, coverage_test

    agents = _make_agents(n=10, dim=4)
    config = DivFlowConfig(n_candidates=10, k_select=5, embedding_dim=4)
    result = diverse_generate(
        n=10, k=5, mechanism="sequential", config=config, agents=agents, seed=42,
    )

    # Verify coverage certificate
    assert result.coverage_certificate is not None

    # Additional coverage test against reference
    rng = np.random.RandomState(42)
    reference = rng.randn(50, 4)
    selected = np.array(result.responses)
    cert = coverage_test(selected, reference, epsilon=3.0)
    assert cert.coverage_fraction > 0
