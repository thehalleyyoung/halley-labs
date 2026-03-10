"""
usability_oracle.mdp — MDP construction and solution for UI accessibility trees.

Provides data models for states, actions, and transitions; builders that
enumerate the MDP state space from an :class:`AccessibilityTree`; classical
solvers (value iteration, policy iteration, LP); trajectory sampling; reward
shaping; feature extraction; and visualisation utilities.

Re-exports
----------
>>> from usability_oracle.mdp import MDP, MDPBuilder, ValueIterationSolver
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

__all__ = [
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
]
