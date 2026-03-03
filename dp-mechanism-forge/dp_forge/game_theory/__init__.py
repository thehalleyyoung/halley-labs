"""
Game-theoretic differential privacy mechanism design.

This package formulates DP mechanism synthesis as a two-player zero-sum game
between a mechanism designer (minimising utility loss) and a privacy
adversary (maximising privacy leakage). It supports:

- **Minimax formulation**: The designer chooses a mechanism to minimise
  worst-case privacy loss, while the adversary chooses the worst-case
  adjacent database pair.
- **Stackelberg formulation**: The designer (leader) commits to a mechanism
  first; the adversary (follower) best-responds.
- **Nash equilibrium computation**: Find mixed-strategy equilibria for
  mechanism-adversary games.
- **Correlated equilibrium**: Linear programming relaxation of Nash.

Architecture:
    1. **GameFormulator** — Constructs the payoff matrix from a QuerySpec.
    2. **MinimaxSolver** — Solves the minimax (zero-sum) game via LP duality.
    3. **StackelbergSolver** — Solves bi-level Stackelberg games.
    4. **NashSolver** — Computes Nash equilibria via support enumeration
       or Lemke-Howson.
    5. **GameAnalyzer** — Post-hoc analysis of equilibria and strategies.

Example::

    from dp_forge.game_theory import GameFormulator, MinimaxSolver

    formulator = GameFormulator()
    game = formulator.from_query_spec(query_spec)
    solver = MinimaxSolver()
    result = solver.solve(game)
    print(f"Minimax value: {result.game_value:.6f}")
    print(f"Designer strategy: {result.designer_strategy}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    AdjacencyRelation,
    GameMatrix,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
    SolverBackend,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GameType(Enum):
    """Type of game formulation for DP mechanism design."""

    ZERO_SUM = auto()
    STACKELBERG = auto()
    GENERAL_SUM = auto()
    BAYESIAN = auto()

    def __repr__(self) -> str:
        return f"GameType.{self.name}"


class EquilibriumType(Enum):
    """Type of game-theoretic equilibrium."""

    MINIMAX = auto()
    NASH = auto()
    CORRELATED = auto()
    STACKELBERG = auto()
    DOMINANT_STRATEGY = auto()

    def __repr__(self) -> str:
        return f"EquilibriumType.{self.name}"


class NashAlgorithm(Enum):
    """Algorithm for computing Nash equilibria."""

    SUPPORT_ENUMERATION = auto()
    LEMKE_HOWSON = auto()
    LINEAR_COMPLEMENTARITY = auto()
    VERTEX_ENUMERATION = auto()

    def __repr__(self) -> str:
        return f"NashAlgorithm.{self.name}"


class PlayerRole(Enum):
    """Role of a player in the mechanism design game."""

    DESIGNER = auto()
    ADVERSARY = auto()

    def __repr__(self) -> str:
        return f"PlayerRole.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GameConfig:
    """Configuration for game-theoretic DP mechanism design.

    Attributes:
        game_type: Type of game formulation.
        equilibrium_type: Target equilibrium concept.
        nash_algorithm: Algorithm for Nash equilibrium computation.
        solver: LP solver backend for minimax/correlated equilibrium.
        max_support_size: Maximum support size for Nash equilibrium search.
        convergence_tol: Tolerance for equilibrium approximation.
        verbose: Verbosity level.
    """

    game_type: GameType = GameType.ZERO_SUM
    equilibrium_type: EquilibriumType = EquilibriumType.MINIMAX
    nash_algorithm: NashAlgorithm = NashAlgorithm.SUPPORT_ENUMERATION
    solver: SolverBackend = SolverBackend.AUTO
    max_support_size: int = 50
    convergence_tol: float = 1e-8
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.max_support_size < 1:
            raise ValueError(f"max_support_size must be >= 1, got {self.max_support_size}")
        if self.convergence_tol <= 0:
            raise ValueError(f"convergence_tol must be > 0, got {self.convergence_tol}")

    def __repr__(self) -> str:
        return (
            f"GameConfig(type={self.game_type.name}, "
            f"eq={self.equilibrium_type.name}, "
            f"alg={self.nash_algorithm.name})"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class Strategy:
    """A mixed strategy for a player in the game.

    Attributes:
        player: Which player this strategy belongs to.
        probabilities: Probability distribution over pure strategies.
        support: Indices of pure strategies with nonzero probability.
        labels: Optional labels for the pure strategies.
    """

    player: PlayerRole
    probabilities: npt.NDArray[np.float64]
    support: Optional[List[int]] = None
    labels: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if self.probabilities.ndim != 1:
            raise ValueError(
                f"probabilities must be 1-D, got shape {self.probabilities.shape}"
            )
        total = float(np.sum(self.probabilities))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1, got {total}")
        if self.support is None:
            self.support = list(np.where(self.probabilities > 1e-12)[0])

    @property
    def support_size(self) -> int:
        """Number of pure strategies with positive probability."""
        return len(self.support) if self.support else 0

    @property
    def is_pure(self) -> bool:
        """Whether this is a pure (deterministic) strategy."""
        return self.support_size == 1

    def __repr__(self) -> str:
        kind = "pure" if self.is_pure else f"mixed(|supp|={self.support_size})"
        return f"Strategy(player={self.player.name}, {kind})"


@dataclass
class Equilibrium:
    """A game-theoretic equilibrium.

    Attributes:
        equilibrium_type: Type of equilibrium.
        designer_strategy: The designer's equilibrium strategy.
        adversary_strategy: The adversary's equilibrium strategy.
        game_value: Value of the game at equilibrium.
        is_exact: Whether this is an exact or approximate equilibrium.
        approximation_error: Error bound for approximate equilibria.
    """

    equilibrium_type: EquilibriumType
    designer_strategy: Strategy
    adversary_strategy: Strategy
    game_value: float
    is_exact: bool = True
    approximation_error: float = 0.0

    def __post_init__(self) -> None:
        if self.is_exact and self.approximation_error > 1e-10:
            raise ValueError(
                "approximation_error must be ~0 for exact equilibria, "
                f"got {self.approximation_error}"
            )

    def __repr__(self) -> str:
        exact = "exact" if self.is_exact else f"approx(err={self.approximation_error:.2e})"
        return (
            f"Equilibrium(type={self.equilibrium_type.name}, "
            f"value={self.game_value:.6f}, {exact})"
        )


@dataclass
class GameResult:
    """Result of a game-theoretic mechanism synthesis.

    Attributes:
        mechanism: The n × k probability table.
        equilibrium: The computed equilibrium.
        game_matrix: The payoff matrix used.
        iterations: Number of solver iterations.
        optimality_certificate: Duality-based optimality certificate.
        worst_case_pair: The adversary's worst-case adjacent pair.
    """

    mechanism: npt.NDArray[np.float64]
    equilibrium: Equilibrium
    game_matrix: GameMatrix
    iterations: int = 0
    optimality_certificate: Optional[OptimalityCertificate] = None
    worst_case_pair: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        self.mechanism = np.asarray(self.mechanism, dtype=np.float64)
        if self.mechanism.ndim != 2:
            raise ValueError(f"mechanism must be 2-D, got shape {self.mechanism.shape}")

    @property
    def game_value(self) -> float:
        """Value of the game at equilibrium."""
        return self.equilibrium.game_value

    def __repr__(self) -> str:
        return (
            f"GameResult(n={self.mechanism.shape[0]}, k={self.mechanism.shape[1]}, "
            f"value={self.game_value:.6f}, iter={self.iterations})"
        )


@dataclass
class StackelbergResult:
    """Result of a Stackelberg (leader-follower) game.

    Attributes:
        leader_strategy: The leader's (designer's) committed strategy.
        follower_best_response: The follower's (adversary's) best response.
        leader_utility: Leader's utility at the Stackelberg equilibrium.
        follower_utility: Follower's utility at the Stackelberg equilibrium.
        mechanism: Extracted mechanism from the leader's strategy.
    """

    leader_strategy: Strategy
    follower_best_response: Strategy
    leader_utility: float
    follower_utility: float
    mechanism: Optional[npt.NDArray[np.float64]] = None

    def __repr__(self) -> str:
        return (
            f"StackelbergResult(leader_util={self.leader_utility:.6f}, "
            f"follower_util={self.follower_utility:.6f})"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class GameSolver(Protocol):
    """Protocol for game-theoretic solvers."""

    def solve(self, game: GameMatrix, budget: PrivacyBudget) -> Equilibrium:
        """Solve for an equilibrium of the game."""
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class GameFormulator:
    """Construct game matrices from DP mechanism synthesis problems.

    Translates a QuerySpec into a two-player game where:
    - Row player (designer) chooses mechanism parameters.
    - Column player (adversary) chooses adjacent database pair.
    """

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig()

    def from_query_spec(self, spec: QuerySpec) -> GameMatrix:
        """Construct a game matrix from a query specification.

        Args:
            spec: Query specification.

        Returns:
            GameMatrix with designer strategies as rows, adversary as columns.
        """
        raise NotImplementedError("GameFormulator.from_query_spec")

    def from_mechanism(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
        budget: PrivacyBudget,
    ) -> GameMatrix:
        """Construct a game matrix from an existing mechanism.

        Args:
            mechanism: The n × k probability table.
            adjacency: Adjacency relation.
            budget: Privacy budget.

        Returns:
            GameMatrix analysing the mechanism's game-theoretic properties.
        """
        raise NotImplementedError("GameFormulator.from_mechanism")


class MinimaxSolver:
    """Solve zero-sum games for minimax DP mechanism design.

    Reduces the minimax game to an LP: the designer minimises the maximum
    adversarial payoff, which is dual to the adversary maximising the
    minimum designer payoff (von Neumann's minimax theorem).
    """

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig(equilibrium_type=EquilibriumType.MINIMAX)

    def solve(self, game: GameMatrix) -> GameResult:
        """Compute the minimax equilibrium.

        Args:
            game: The payoff matrix.

        Returns:
            GameResult with minimax strategies and value.
        """
        raise NotImplementedError("MinimaxSolver.solve")

    def solve_from_spec(self, spec: QuerySpec) -> GameResult:
        """End-to-end minimax synthesis from a query specification.

        Args:
            spec: Query specification.

        Returns:
            GameResult with the synthesised mechanism.
        """
        raise NotImplementedError("MinimaxSolver.solve_from_spec")


class StackelbergSolver:
    """Solve Stackelberg (leader-follower) games for DP mechanism design.

    The mechanism designer commits first (leader), then the adversary
    best-responds (follower). This is solved via a bi-level LP.
    """

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig(
            game_type=GameType.STACKELBERG,
            equilibrium_type=EquilibriumType.STACKELBERG,
        )

    def solve(self, game: GameMatrix) -> StackelbergResult:
        """Compute the Stackelberg equilibrium.

        Args:
            game: The payoff matrix.

        Returns:
            StackelbergResult with leader/follower strategies.
        """
        raise NotImplementedError("StackelbergSolver.solve")


class NashSolver:
    """Compute Nash equilibria for general-sum DP games.

    Supports support enumeration, Lemke-Howson, and vertex enumeration
    algorithms.
    """

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig(equilibrium_type=EquilibriumType.NASH)

    def solve(
        self,
        designer_payoff: GameMatrix,
        adversary_payoff: GameMatrix,
    ) -> List[Equilibrium]:
        """Compute all Nash equilibria (up to max_support_size).

        Args:
            designer_payoff: Designer's payoff matrix.
            adversary_payoff: Adversary's payoff matrix.

        Returns:
            List of Nash equilibria found.
        """
        raise NotImplementedError("NashSolver.solve")


class GameAnalyzer:
    """Post-hoc analysis of game-theoretic mechanism design results."""

    def dominated_strategies(self, game: GameMatrix) -> Dict[PlayerRole, List[int]]:
        """Find strictly dominated strategies for each player.

        Args:
            game: The payoff matrix.

        Returns:
            Dict mapping each player to list of dominated strategy indices.
        """
        raise NotImplementedError("GameAnalyzer.dominated_strategies")

    def best_response(
        self,
        game: GameMatrix,
        opponent_strategy: Strategy,
        player: PlayerRole,
    ) -> Strategy:
        """Compute a player's best response to an opponent's strategy.

        Args:
            game: The payoff matrix.
            opponent_strategy: The opponent's mixed strategy.
            player: Which player to compute best response for.

        Returns:
            Best response strategy.
        """
        raise NotImplementedError("GameAnalyzer.best_response")

    def exploitability(
        self,
        game: GameMatrix,
        designer_strategy: Strategy,
        adversary_strategy: Strategy,
    ) -> float:
        """Compute exploitability of a strategy profile.

        Exploitability measures how far a strategy profile is from a
        Nash equilibrium.

        Args:
            game: The payoff matrix.
            designer_strategy: Designer's strategy.
            adversary_strategy: Adversary's strategy.

        Returns:
            Exploitability value (0 means Nash equilibrium).
        """
        raise NotImplementedError("GameAnalyzer.exploitability")


__all__ = [
    # Enums
    "GameType",
    "EquilibriumType",
    "NashAlgorithm",
    "PlayerRole",
    # Config
    "GameConfig",
    # Data types
    "Strategy",
    "Equilibrium",
    "GameResult",
    "StackelbergResult",
    # Protocols
    "GameSolver",
    # Classes
    "GameFormulator",
    "MinimaxSolver",
    "StackelbergSolver",
    "NashSolver",
    "GameAnalyzer",
]

# ---------------------------------------------------------------------------
# Submodule imports (deferred to avoid circular imports at class-definition
# time – the stubs above are used as base types by the submodules).
# ---------------------------------------------------------------------------

from dp_forge.game_theory.minimax import (  # noqa: E402
    MinimaxSolver as _MinimaxSolverImpl,
    SaddlePointComputation,
    MaxMinFair,
    WorstCaseDataset,
    MinimaxLPFormulation,
    RobustOptimization,
)
from dp_forge.game_theory.stackelberg import (  # noqa: E402
    StackelbergSolver as _StackelbergSolverImpl,
    LeaderFollowerGame,
    MultipleFollower,
    BilevelOptimization,
    StrongStackelberg,
    OptimalCommitment,
)
from dp_forge.game_theory.equilibrium import (  # noqa: E402
    NashEquilibrium,
    SupportEnumeration,
    LemkeHowson,
    CorrelatedEquilibrium,
    EvolutionaryDynamics,
    TrembleEquilibrium,
)
from dp_forge.game_theory.mechanism_game import (  # noqa: E402
    PrivacyGame,
    AdversaryModel,
    AdversaryCapabilities,
    UtilityFunction,
    InformationDesign,
    AuctionMechanism,
    BayesianGame,
)

__all__ += [
    # minimax
    "SaddlePointComputation",
    "MaxMinFair",
    "WorstCaseDataset",
    "MinimaxLPFormulation",
    "RobustOptimization",
    # stackelberg
    "LeaderFollowerGame",
    "MultipleFollower",
    "BilevelOptimization",
    "StrongStackelberg",
    "OptimalCommitment",
    # equilibrium
    "NashEquilibrium",
    "SupportEnumeration",
    "LemkeHowson",
    "CorrelatedEquilibrium",
    "EvolutionaryDynamics",
    "TrembleEquilibrium",
    # mechanism_game
    "PrivacyGame",
    "AdversaryModel",
    "AdversaryCapabilities",
    "UtilityFunction",
    "InformationDesign",
    "AuctionMechanism",
    "BayesianGame",
]
