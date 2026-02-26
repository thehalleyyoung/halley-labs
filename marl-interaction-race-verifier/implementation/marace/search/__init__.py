"""
Adversarial schedule search engine for the MARACE system.

Implements Monte Carlo Tree Search (MCTS) over the space of agent schedules
to find adversarial interleavings that trigger interaction races.  Includes
UCB1 variants with abstract safety margins, AI-guided pruning via abstract
interpretation, adversarial replay generation, and schedule optimization.
"""

from marace.search.mcts import (
    MCTS,
    MCTSNode,
    MCTSTree,
    ScheduleAction,
    SearchBudget,
    SearchResult,
)
from marace.search.ucb import (
    AbstractMarginEstimate,
    ExplorationBonus,
    MultiObjectiveUCB,
    UCB1Progressive,
    UCB1Safety,
    UCB1Standard,
)
from marace.search.pruning import (
    AbstractPruner,
    CompositePruner,
    DominancePruner,
    HBConsistencyPruner,
    PruningStatistics,
    SafetyMarginPruner,
    SymmetryPruner,
)
from marace.search.replay import (
    CounterfactualReplay,
    MinimalReplayExtractor,
    ReplayGenerator,
    ReplaySerializer,
    ReplayStep,
    ReplayTrace,
    ReplayVerifier,
    ReplayVisualization,
)
from marace.search.schedule_optimizer import (
    GeneticScheduleSearch,
    GradientScheduleEstimate,
    LocalScheduleSearch,
    ScheduleCandidate,
    ScheduleInterpolation,
    TimingOptimizer,
)

__all__ = [
    # mcts
    "MCTS",
    "MCTSNode",
    "MCTSTree",
    "ScheduleAction",
    "SearchBudget",
    "SearchResult",
    # ucb
    "AbstractMarginEstimate",
    "ExplorationBonus",
    "MultiObjectiveUCB",
    "UCB1Progressive",
    "UCB1Safety",
    "UCB1Standard",
    # pruning
    "AbstractPruner",
    "CompositePruner",
    "DominancePruner",
    "HBConsistencyPruner",
    "PruningStatistics",
    "SafetyMarginPruner",
    "SymmetryPruner",
    # replay
    "CounterfactualReplay",
    "MinimalReplayExtractor",
    "ReplayGenerator",
    "ReplaySerializer",
    "ReplayStep",
    "ReplayTrace",
    "ReplayVerifier",
    "ReplayVisualization",
    # schedule_optimizer
    "GeneticScheduleSearch",
    "GradientScheduleEstimate",
    "LocalScheduleSearch",
    "ScheduleCandidate",
    "ScheduleInterpolation",
    "TimingOptimizer",
]
