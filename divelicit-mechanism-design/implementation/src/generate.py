"""Main API for diverse generation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .agents import Agent, GaussianAgent, UniformAgent
from .config import DivFlowConfig
from .coverage import CoverageCertificate, estimate_coverage
from .kernels import RBFKernel, AdaptiveRBFKernel
from .mechanism import (
    DirectMechanism,
    FlowMechanism,
    MechanismResult,
    ParetoMechanism,
    SequentialMechanism,
    VCGMechanism,
    BudgetFeasibleMechanism,
)
from .scoring_rules import (
    LogarithmicRule, BrierRule, QualityScore, simulate_quality,
)
from .diversity_metrics import cosine_diversity
from .utils import set_seed


@dataclass
class DivFlowResult:
    """Result of diverse_generate."""
    responses: List[np.ndarray]
    diversity_score: float
    quality_scores: List[float]
    quality_details: Optional[List[Dict[str, float]]] = None
    coverage_certificate: Optional[CoverageCertificate] = None
    ic_verified: bool = False
    ic_violations: int = 0
    payments: Optional[List[float]] = None
    pareto_point: Optional[tuple] = None


def diverse_generate(
    prompt: str = "",
    n: int = 8,
    k: int = 4,
    mechanism: str = "flow",
    config: Optional[DivFlowConfig] = None,
    agents: Optional[List[Agent]] = None,
    seed: int = 42,
) -> DivFlowResult:
    """Generate k diverse responses from n candidates.

    Args:
        prompt: The generation prompt.
        n: Number of candidate responses to generate.
        k: Number of diverse responses to select.
        mechanism: One of "direct", "sequential", "flow", "pareto", "vcg", "budget".
        config: Optional configuration.
        agents: Optional list of agents. If None, uses simulated agents.
        seed: Random seed.

    Returns:
        DivFlowResult with selected responses and metrics.
    """
    set_seed(seed)

    if config is None:
        config = DivFlowConfig(n_candidates=n, k_select=k, seed=seed)

    # Create simulated agents if none provided
    if agents is None:
        dim = config.embedding_dim
        agents = []
        for i in range(n):
            mean = np.random.randn(dim)
            cov = np.eye(dim) * 0.5
            agents.append(GaussianAgent(mean=mean, cov=cov, seed=seed + i))

    scoring_rule = LogarithmicRule()

    if mechanism == "direct":
        mech = DirectMechanism(
            scoring_rule=scoring_rule,
            n_candidates=n, k_select=k, seed=seed,
        )
    elif mechanism == "sequential":
        mech = SequentialMechanism(
            scoring_rule=scoring_rule,
            n_candidates=n, k_select=k,
            n_rounds=config.n_rounds, seed=seed,
        )
    elif mechanism == "flow":
        mech = FlowMechanism(
            scoring_rule=scoring_rule,
            n_candidates=n, k_select=k,
            n_rounds=config.n_rounds, reg=config.sinkhorn_reg, seed=seed,
        )
    elif mechanism == "pareto":
        mech = ParetoMechanism(
            scoring_rule=scoring_rule,
            n_candidates=n, k_select=k, seed=seed,
        )
    elif mechanism == "vcg":
        mech = VCGMechanism(
            scoring_rule=scoring_rule,
            n_candidates=n, k_select=k, seed=seed,
        )
    elif mechanism == "budget":
        mech = BudgetFeasibleMechanism(
            scoring_rule=scoring_rule,
            n_candidates=n, k_select=k, seed=seed,
        )
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    result = mech.run(agents)

    # Compute multi-dimensional quality scores
    rng = np.random.RandomState(seed)
    quality_details = []
    for emb in result.selected_items:
        qs = simulate_quality(emb, rng=rng)
        quality_details.append(qs.to_dict())

    pareto_point = None
    if mechanism == "pareto":
        pareto_point = (float(np.mean(result.quality_scores)), result.diversity_score)

    return DivFlowResult(
        responses=[result.selected_items[i] for i in range(len(result.selected_items))],
        diversity_score=result.diversity_score,
        quality_scores=result.quality_scores,
        quality_details=quality_details,
        coverage_certificate=result.coverage_certificate,
        ic_verified=result.ic_verified,
        ic_violations=result.ic_violations,
        payments=result.payments,
        pareto_point=pareto_point,
    )
