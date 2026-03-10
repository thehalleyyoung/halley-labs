"""
usability_oracle.policy — Bounded-rational policy computation.

Implements the free-energy formulation for bounded-rational decision-making:

    F(π) = E_π[cost] + (1/β) · D_KL(π ‖ p₀)

where β is the *rationality parameter* (inverse temperature), π is the
agent's policy, and p₀ is a prior (default) policy.

Modules
-------
- **models** — Policy, QValues, PolicyResult data structures
- **softmax** — Boltzmann / softmax policy construction
- **free_energy** — Free-energy computation and decomposition
- **value_iteration** — Soft (entropy-regularised) value iteration
- **monte_carlo** — Monte Carlo estimation of value and free energy
- **optimal** — Optimal and bounded-rational policy computation
- **mcts** — Monte Carlo Tree Search with bounded-rational exploration
- **thompson** — Thompson sampling for exploration
- **gradient** — Policy gradient methods (REINFORCE, natural gradient)
- **belief_policy** — Belief-space policies for POMDPs
- **multi_objective** — Multi-objective policy computation and Pareto fronts
- **inverse_rl** — Inverse reinforcement learning for reward recovery

Re-exports
----------
>>> from usability_oracle.policy import Policy, SoftmaxPolicy, FreeEnergyComputer
"""

from __future__ import annotations

from usability_oracle.policy.models import Policy, QValues, PolicyResult
from usability_oracle.policy.softmax import SoftmaxPolicy
from usability_oracle.policy.free_energy import (
    FreeEnergyComputer,
    FreeEnergyDecomposition,
)
from usability_oracle.policy.value_iteration import SoftValueIteration
from usability_oracle.policy.monte_carlo import MonteCarloEstimator
from usability_oracle.policy.optimal import OptimalPolicyComputer
from usability_oracle.policy.mcts import (
    BoundedRationalMCTS,
    MCTSConfig,
    MCTSNode,
)
from usability_oracle.policy.thompson import (
    BetaBernoulliThompson,
    GaussianThompson,
    BoundedRationalThompson,
    InformationDirectedSampler,
    KnowledgeGradient,
    BayesianUIOptimiser,
    UIExplorationTracker,
)
from usability_oracle.policy.gradient import (
    REINFORCE,
    NaturalPolicyGradient,
    SoftmaxPolicyParam,
    CompatibleCritic,
    learn_cognitive_policy,
)
from usability_oracle.policy.belief_policy import (
    BeliefState,
    AlphaVector,
    PWLCValueFunction,
    InformationSeekingPolicy,
    RiskSensitiveBeliefPolicy,
    point_based_value_iteration,
    belief_update,
)
from usability_oracle.policy.multi_objective import (
    MultiObjectiveQValues,
    ParetoPoint,
    ParetoFrontierComputer,
    ConstrainedMDPSolver,
    weighted_sum_scalarisation,
    chebyshev_scalarisation,
    lexicographic_optimise,
    multi_objective_value_iteration,
    pareto_frontier_data,
)
from usability_oracle.policy.inverse_rl import (
    Demonstration,
    IRLResult,
    MaxEntropyIRL,
    FeatureMatchingIRL,
    BayesianIRL,
    BoundedRationalIRL,
    compute_feature_expectations,
)

__all__ = [
    # models
    "Policy",
    "QValues",
    "PolicyResult",
    # softmax
    "SoftmaxPolicy",
    # free_energy
    "FreeEnergyComputer",
    "FreeEnergyDecomposition",
    # value_iteration
    "SoftValueIteration",
    # monte_carlo
    "MonteCarloEstimator",
    # optimal
    "OptimalPolicyComputer",
    # mcts
    "BoundedRationalMCTS",
    "MCTSConfig",
    "MCTSNode",
    # thompson
    "BetaBernoulliThompson",
    "GaussianThompson",
    "BoundedRationalThompson",
    "InformationDirectedSampler",
    "KnowledgeGradient",
    "BayesianUIOptimiser",
    "UIExplorationTracker",
    # gradient
    "REINFORCE",
    "NaturalPolicyGradient",
    "SoftmaxPolicyParam",
    "CompatibleCritic",
    "learn_cognitive_policy",
    # belief_policy
    "BeliefState",
    "AlphaVector",
    "PWLCValueFunction",
    "InformationSeekingPolicy",
    "RiskSensitiveBeliefPolicy",
    "point_based_value_iteration",
    "belief_update",
    # multi_objective
    "MultiObjectiveQValues",
    "ParetoPoint",
    "ParetoFrontierComputer",
    "ConstrainedMDPSolver",
    "weighted_sum_scalarisation",
    "chebyshev_scalarisation",
    "lexicographic_optimise",
    "multi_objective_value_iteration",
    "pareto_frontier_data",
    # inverse_rl
    "Demonstration",
    "IRLResult",
    "MaxEntropyIRL",
    "FeatureMatchingIRL",
    "BayesianIRL",
    "BoundedRationalIRL",
    "compute_feature_expectations",
]
