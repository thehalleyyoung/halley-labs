"""
usability_oracle.mdp — MDP construction and solution for UI accessibility trees.

Provides data models for states, actions, and transitions; builders that
enumerate the MDP state space from an :class:`AccessibilityTree`; classical
solvers (value iteration, policy iteration, LP); trajectory sampling; reward
shaping; feature extraction; and visualisation utilities.

Also provides POMDP support (partial observability), belief state computation,
POMDP solvers (PBVI, Perseus, SARSOP, QMDP, FIB), active information
gathering, factored MDP/POMDP representations, and hierarchical MDP (options
framework).

Re-exports
----------
>>> from usability_oracle.mdp import MDP, MDPBuilder, ValueIterationSolver
>>> from usability_oracle.mdp import POMDP, BeliefState, PBVISolver
"""

from __future__ import annotations

from usability_oracle.mdp.models import (
    State,
    Action,
    Transition,
    MDP,
    MDPStatistics,
)
from usability_oracle.mdp.builder import MDPBuilder
from usability_oracle.mdp.solver import (
    ValueIterationSolver,
    PolicyIterationSolver,
    LinearProgramSolver,
)
from usability_oracle.mdp.trajectory import TrajectorySampler, TrajectoryStats
from usability_oracle.mdp.reward import RewardFunction, TaskRewardShaper
from usability_oracle.mdp.features import StateFeatureExtractor
from usability_oracle.mdp.visualization import MDPVisualizer

# POMDP support
from usability_oracle.mdp.pomdp import (
    Observation,
    ObservationModel,
    BeliefState,
    POMDP,
    POMDPBuilder,
    point_based_beliefs,
)
from usability_oracle.mdp.belief import (
    BeliefUpdater,
    BeliefCompressor,
    ParticleFilter,
    ParticleBeliefState,
    FactoredBeliefState,
    BeliefRewardShaper,
    belief_entropy,
    belief_kl_divergence,
    belief_to_ascii,
)
from usability_oracle.mdp.pomdp_solver import (
    AlphaVector,
    POMDPPolicy,
    QMDPSolver,
    FIBSolver,
    ExactPOMDPSolver,
    PBVISolver,
    PerseusSolver,
    SARSOPSolver,
    BoundedRationalPOMDPPolicy,
    PolicyTreeNode,
    build_policy_tree,
)
from usability_oracle.mdp.information_gathering import (
    ValueOfInformation,
    InformationGain,
    OptimalStopping,
    InformationStrategy,
    EntropyReductionReward,
    ActiveInferenceAgent,
)
from usability_oracle.mdp.factored import (
    Factor,
    ConditionalProbTable,
    ContextSpecificIndependence,
    FactoredMDP,
    FactoredValueFunction,
    ADDNode,
    build_add_from_table,
    factored_belief_update,
    UIStateFactorBuilder,
)
from usability_oracle.mdp.hierarchical import (
    Option,
    OptionExecution,
    OptionExecutor,
    OptionDiscovery,
    HierarchicalValueIteration,
    MAXQNode,
    MAXQDecomposition,
    TaskToOptionMapper,
)

__all__ = [
    # Core MDP
    "State",
    "Action",
    "Transition",
    "MDP",
    "MDPStatistics",
    "MDPBuilder",
    "ValueIterationSolver",
    "PolicyIterationSolver",
    "LinearProgramSolver",
    "TrajectorySampler",
    "TrajectoryStats",
    "RewardFunction",
    "TaskRewardShaper",
    "StateFeatureExtractor",
    "MDPVisualizer",
    # POMDP model
    "Observation",
    "ObservationModel",
    "BeliefState",
    "POMDP",
    "POMDPBuilder",
    "point_based_beliefs",
    # Belief computation
    "BeliefUpdater",
    "BeliefCompressor",
    "ParticleFilter",
    "ParticleBeliefState",
    "FactoredBeliefState",
    "BeliefRewardShaper",
    "belief_entropy",
    "belief_kl_divergence",
    "belief_to_ascii",
    # POMDP solvers
    "AlphaVector",
    "POMDPPolicy",
    "QMDPSolver",
    "FIBSolver",
    "ExactPOMDPSolver",
    "PBVISolver",
    "PerseusSolver",
    "SARSOPSolver",
    "BoundedRationalPOMDPPolicy",
    "PolicyTreeNode",
    "build_policy_tree",
    # Information gathering
    "ValueOfInformation",
    "InformationGain",
    "OptimalStopping",
    "InformationStrategy",
    "EntropyReductionReward",
    "ActiveInferenceAgent",
    # Factored MDP/POMDP
    "Factor",
    "ConditionalProbTable",
    "ContextSpecificIndependence",
    "FactoredMDP",
    "FactoredValueFunction",
    "ADDNode",
    "build_add_from_table",
    "factored_belief_update",
    "UIStateFactorBuilder",
    # Hierarchical MDP
    "Option",
    "OptionExecution",
    "OptionExecutor",
    "OptionDiscovery",
    "HierarchicalValueIteration",
    "MAXQNode",
    "MAXQDecomposition",
    "TaskToOptionMapper",
]
