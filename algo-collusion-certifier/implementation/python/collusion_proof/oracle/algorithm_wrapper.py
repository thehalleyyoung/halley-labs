"""Algorithm wrapper for sandboxed execution of pricing algorithms."""

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger("collusion_proof.oracle.wrapper")


# ── Protocol ─────────────────────────────────────────────────────────────────


class PricingAlgorithm(Protocol):
    """Protocol that any pricing algorithm must satisfy."""

    def select_action(self, state: Any) -> float: ...
    def update(
        self, state: Any, action: float, reward: float, next_state: Any
    ) -> None: ...
    def get_state(self) -> Any: ...
    def set_state(self, state: Any) -> None: ...


# ── AlgorithmWrapper ─────────────────────────────────────────────────────────


class AlgorithmWrapper:
    """Wraps a pricing algorithm for sandboxed execution.

    Provides a uniform interface regardless of the underlying
    algorithm's implementation details.
    """

    def __init__(
        self,
        algorithm: Any,
        player_id: int,
        action_space: np.ndarray,
    ) -> None:
        self.algorithm = algorithm
        self.player_id = player_id
        self.action_space = np.asarray(action_space, dtype=float)
        self.num_actions = len(self.action_space)
        self._initial_state: Optional[Any] = None
        # Snapshot the initial state for later resets
        if hasattr(algorithm, "get_state"):
            self._initial_state = copy.deepcopy(algorithm.get_state())

    # ------------------------------------------------------------------
    # Price selection
    # ------------------------------------------------------------------

    def select_price(self, market_state: Dict[str, Any]) -> float:
        """Select a price given the current market state.

        Converts the market state to an internal representation, asks the
        algorithm for a discrete action index, and maps it back to a
        continuous price.
        """
        internal_state = self._encode_state(market_state)
        action = self.algorithm.select_action(internal_state)
        action_idx = int(np.clip(action, 0, self.num_actions - 1))
        return float(self.action_space[action_idx])

    def select_action(self, market_state: Dict[str, Any]) -> int:
        """Select a discrete action index given the market state."""
        internal_state = self._encode_state(market_state)
        action = self.algorithm.select_action(internal_state)
        return int(np.clip(action, 0, self.num_actions - 1))

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update(
        self,
        market_state: Dict[str, Any],
        action: float,
        reward: float,
        next_state: Dict[str, Any],
    ) -> None:
        """Feed an experience tuple to the underlying algorithm."""
        s = self._encode_state(market_state)
        a = self._price_to_action(action)
        ns = self._encode_state(next_state)
        self.algorithm.update(s, a, reward, ns)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def save_state(self) -> Any:
        """Return a deep copy of the algorithm's internal state."""
        return copy.deepcopy(self.algorithm.get_state())

    def restore_state(self, state: Any) -> None:
        """Restore the algorithm to a previously saved state."""
        self.algorithm.set_state(copy.deepcopy(state))

    def reset(self) -> None:
        """Reset the algorithm to its initial state."""
        if self._initial_state is not None:
            self.algorithm.set_state(copy.deepcopy(self._initial_state))

    # ------------------------------------------------------------------
    # Proxied protocol methods
    # ------------------------------------------------------------------

    def get_state(self) -> Any:
        return self.algorithm.get_state()

    def set_state(self, state: Any) -> None:
        self.algorithm.set_state(state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_state(self, market_state: Dict[str, Any]) -> Any:
        """Convert a market-state dict into the representation expected by
        the wrapped algorithm.

        If the market_state is already a scalar (e.g. an int index),
        pass it through unchanged.
        """
        if isinstance(market_state, (int, np.integer)):
            return int(market_state)
        if isinstance(market_state, dict):
            # Use the joint-action index if available
            if "state_index" in market_state:
                return int(market_state["state_index"])
            if "prev_actions" in market_state:
                return self._actions_to_state(market_state["prev_actions"])
            if "prices" in market_state:
                return self._prices_to_state(market_state["prices"])
        return market_state

    def _actions_to_state(self, actions: np.ndarray) -> int:
        """Encode a vector of discrete actions into a single state index
        using a mixed-radix representation."""
        actions = np.asarray(actions, dtype=int)
        state = 0
        for a in actions:
            state = state * self.num_actions + int(np.clip(a, 0, self.num_actions - 1))
        return state

    def _prices_to_state(self, prices: np.ndarray) -> int:
        """Map continuous prices to the nearest action indices and encode."""
        prices = np.asarray(prices, dtype=float)
        actions = np.array(
            [int(np.argmin(np.abs(self.action_space - p))) for p in prices]
        )
        return self._actions_to_state(actions)

    def _price_to_action(self, price: float) -> int:
        """Map a single continuous price to the nearest discrete action."""
        return int(np.argmin(np.abs(self.action_space - price)))

    def __repr__(self) -> str:
        return (
            f"AlgorithmWrapper(player={self.player_id}, "
            f"actions={self.num_actions}, "
            f"algo={type(self.algorithm).__name__})"
        )


# ── MarketSimulator ──────────────────────────────────────────────────────────


class MarketSimulator:
    """Simulate a repeated pricing game.

    Supports linear demand, optional product differentiation, and
    demand noise.
    """

    def __init__(
        self,
        algorithms: List[AlgorithmWrapper],
        game_config: Dict[str, Any],
    ) -> None:
        self.algorithms = algorithms
        self.game_config = game_config
        self.num_players = len(algorithms)

        self.marginal_cost: float = game_config.get("marginal_cost", 1.0)
        self.demand_intercept: float = game_config.get("demand_intercept", 10.0)
        self.demand_slope: float = game_config.get("demand_slope", 1.0)
        self.noise_std: float = game_config.get("noise_std", 0.0)
        self.differentiation: float = game_config.get("differentiation", 0.0)

        a = self.demand_intercept / self.demand_slope
        self.nash_price = (a + self.num_players * self.marginal_cost) / (
            self.num_players + 1
        )
        self.monopoly_price = (a + self.marginal_cost) / 2.0

        self.round_num = 0
        self.price_history: List[np.ndarray] = []
        self.quantity_history: List[np.ndarray] = []
        self.profit_history: List[np.ndarray] = []

        self._prev_market_state: Dict[str, Any] = self._initial_market_state()

    def _initial_market_state(self) -> Dict[str, Any]:
        mid_price = (self.nash_price + self.monopoly_price) / 2.0
        return {
            "round": 0,
            "prices": np.full(self.num_players, mid_price),
            "prev_actions": np.full(self.num_players, 0, dtype=int),
            "state_index": 0,
        }

    # ------------------------------------------------------------------
    # Run full simulation
    # ------------------------------------------------------------------

    def run(
        self,
        num_rounds: int,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """Run simulation for *num_rounds* and return results."""
        for _ in range(num_rounds):
            result = self.step()
            if callbacks:
                for cb in callbacks:
                    cb(result)

        return {
            "prices": np.array(self.price_history),
            "quantities": np.array(self.quantity_history),
            "profits": np.array(self.profit_history),
            "num_rounds": self.round_num,
            "nash_price": self.nash_price,
            "monopoly_price": self.monopoly_price,
        }

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """Execute one round of the pricing game."""
        # 1. Each algorithm selects a price
        prices = np.array(
            [alg.select_price(self._prev_market_state) for alg in self.algorithms]
        )

        # 2. Compute demand & profits
        quantities = self.compute_demand(prices)
        profits = self.compute_profits(prices, quantities)

        # 3. Build next market state
        actions = np.array(
            [alg.select_action(self._prev_market_state) for alg in self.algorithms]
        )
        next_state: Dict[str, Any] = {
            "round": self.round_num + 1,
            "prices": prices.copy(),
            "prev_actions": actions.copy(),
            "state_index": self._compute_joint_state(actions),
        }

        # 4. Update each algorithm
        for i, alg in enumerate(self.algorithms):
            alg.update(
                self._prev_market_state,
                float(prices[i]),
                float(profits[i]),
                next_state,
            )

        # 5. Record
        self.price_history.append(prices)
        self.quantity_history.append(quantities)
        self.profit_history.append(profits)
        self._prev_market_state = next_state
        self.round_num += 1

        return {
            "round": self.round_num - 1,
            "prices": prices,
            "quantities": quantities,
            "profits": profits,
        }

    # ------------------------------------------------------------------
    # Demand model
    # ------------------------------------------------------------------

    def compute_demand(self, prices: np.ndarray) -> np.ndarray:
        """Compute demand for each firm given prices.

        With ``differentiation == 0`` the model is symmetric Bertrand:
        total demand is split equally.  With ``differentiation > 0`` a
        logit-style share model gives higher share to cheaper firms.
        """
        prices = np.asarray(prices, dtype=float)
        avg_price = np.mean(prices)
        total_demand = max(
            self.demand_intercept - self.demand_slope * avg_price, 0.0
        )

        if self.noise_std > 0:
            total_demand = max(
                total_demand + np.random.normal(0, self.noise_std), 0.0
            )

        if self.differentiation > 0 and self.num_players > 1:
            # Logit-style share: share_i ∝ exp(-β * p_i)
            beta = self.differentiation * 10.0
            utilities = -beta * prices
            utilities -= np.max(utilities)  # numerical stability
            exp_u = np.exp(utilities)
            shares = exp_u / np.sum(exp_u)
        else:
            shares = np.full(self.num_players, 1.0 / self.num_players)

        return total_demand * shares

    def compute_profits(
        self, prices: np.ndarray, quantities: np.ndarray
    ) -> np.ndarray:
        """Compute per-firm profits: (p_i - c) * q_i."""
        return (prices - self.marginal_cost) * quantities

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def get_market_state(self) -> Dict[str, Any]:
        """Return the current market state dict."""
        return dict(self._prev_market_state)

    def _compute_joint_state(self, actions: np.ndarray) -> int:
        """Encode joint action vector into a single integer index."""
        num_actions = self.algorithms[0].num_actions if self.algorithms else 1
        state = 0
        for a in actions:
            state = state * num_actions + int(a)
        return state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the simulator (and optionally the algorithms)."""
        self.round_num = 0
        self.price_history.clear()
        self.quantity_history.clear()
        self.profit_history.clear()
        self._prev_market_state = self._initial_market_state()
        for alg in self.algorithms:
            alg.reset()

    def __repr__(self) -> str:
        return (
            f"MarketSimulator(players={self.num_players}, "
            f"rounds={self.round_num}, "
            f"nash={self.nash_price:.3f}, "
            f"monopoly={self.monopoly_price:.3f})"
        )
