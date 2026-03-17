"""Q-Learning implementation for pricing games."""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("collusion_proof.oracle.q_learning")


# ── Tabular Q-Learning ───────────────────────────────────────────────────────


class QLearningAgent:
    """Tabular Q-learning agent for pricing.

    Maintains a Q-table of shape ``(num_states, num_actions)`` and uses
    epsilon-greedy exploration with configurable decay.
    """

    def __init__(
        self,
        num_actions: int,
        num_states: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
    ) -> None:
        self.num_actions = num_actions
        self.num_states = num_states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((num_states, num_actions))
        self.visit_counts = np.zeros((num_states, num_actions), dtype=int)
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        state = int(state) % self.num_states
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        q_values = self.q_table[state]
        # Break ties randomly
        max_q = np.max(q_values)
        candidates = np.where(np.abs(q_values - max_q) < 1e-10)[0]
        return int(np.random.choice(candidates))

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        """Standard Q-learning (off-policy) update."""
        state = int(state) % self.num_states
        action = int(action) % self.num_actions
        next_state = int(next_state) % self.num_states

        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        self.visit_counts[state, action] += 1
        self.total_steps += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "q_table": self.q_table.copy(),
            "visit_counts": self.visit_counts.copy(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.q_table = state["q_table"].copy()
        self.visit_counts = state["visit_counts"].copy()
        self.epsilon = state["epsilon"]
        self.total_steps = state["total_steps"]

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def get_policy(self) -> np.ndarray:
        """Return greedy policy (best action per state)."""
        return np.argmax(self.q_table, axis=1).astype(int)

    def get_value_function(self) -> np.ndarray:
        """Return the value function V(s) = max_a Q(s, a)."""
        return np.max(self.q_table, axis=1)

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(states={self.num_states}, "
            f"actions={self.num_actions}, "
            f"eps={self.epsilon:.4f}, "
            f"steps={self.total_steps})"
        )


# ── Multi-Agent Independent Q-Learning ───────────────────────────────────────


class MultiAgentQLearning:
    """Multi-agent independent Q-learning for pricing games.

    Each player maintains a separate Q-table.  States encode the most
    recent joint action (with configurable memory length).
    """

    def __init__(
        self,
        num_players: int,
        num_actions: int,
        memory_length: int = 1,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
    ) -> None:
        self.num_players = num_players
        self.num_actions = num_actions
        self.memory_length = memory_length

        # State space: (num_actions^num_players)^memory_length
        self.states_per_step = num_actions ** num_players
        self.num_states = self.states_per_step ** memory_length

        self.agents = [
            QLearningAgent(
                num_actions=num_actions,
                num_states=self.num_states,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
            for _ in range(num_players)
        ]

        self._action_history: List[np.ndarray] = []
        self._current_state: List[int] = [0] * num_players

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def state_from_history(self, price_history: np.ndarray) -> List[int]:
        """Convert recent price-action history to state indices for each
        agent.

        All agents share the same state encoding (the joint action
        profile), so this returns a single state index broadcast to a
        list of length ``num_players``.
        """
        if len(price_history) == 0:
            return [0] * self.num_players

        # Take the last ``memory_length`` rows
        tail = price_history[-self.memory_length:]
        state = 0
        for row in tail:
            row = np.asarray(row, dtype=int)
            step_index = 0
            for a in row:
                step_index = step_index * self.num_actions + int(
                    np.clip(a, 0, self.num_actions - 1)
                )
            state = state * self.states_per_step + step_index

        state = state % self.num_states
        return [state] * self.num_players

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, game_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute one round.  Returns (prices, quantities, profits)."""
        marginal_cost = game_config.get("marginal_cost", 1.0)
        demand_intercept = game_config.get("demand_intercept", 10.0)
        demand_slope = game_config.get("demand_slope", 1.0)
        noise_std = game_config.get("noise_std", 0.0)
        a = demand_intercept / demand_slope
        monopoly_price = (a + marginal_cost) / 2.0
        action_space = np.linspace(
            marginal_cost, monopoly_price * 1.2, self.num_actions
        )

        # Select actions
        actions = np.array(
            [
                self.agents[i].select_action(self._current_state[i])
                for i in range(self.num_players)
            ]
        )
        prices = action_space[actions]

        # Demand & profit
        avg_price = np.mean(prices)
        total_demand = max(demand_intercept - demand_slope * avg_price, 0.0)
        if noise_std > 0:
            total_demand = max(
                total_demand + np.random.normal(0, noise_std), 0.0
            )
        quantities = np.full(self.num_players, total_demand / self.num_players)
        profits = (prices - marginal_cost) * quantities

        # Next state
        self._action_history.append(actions)
        next_states = self.state_from_history(
            np.array(self._action_history)
        )

        # Update each agent
        for i in range(self.num_players):
            self.agents[i].update(
                self._current_state[i],
                int(actions[i]),
                float(profits[i]),
                next_states[i],
            )

        self._current_state = next_states
        return prices, quantities, profits

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        game_config: Dict[str, Any],
        num_rounds: int,
        log_interval: int = 10000,
    ) -> Dict[str, Any]:
        """Full training loop.  Returns training results."""
        all_prices = np.zeros((num_rounds, self.num_players))
        all_profits = np.zeros((num_rounds, self.num_players))

        for t in range(num_rounds):
            prices, _quantities, profits = self.step(game_config)
            all_prices[t] = prices
            all_profits[t] = profits

            if log_interval > 0 and (t + 1) % log_interval == 0:
                avg_p = np.mean(all_prices[max(0, t - log_interval) : t + 1])
                eps = self.agents[0].epsilon
                logger.info(
                    "Round %d/%d  avg_price=%.4f  eps=%.4f",
                    t + 1,
                    num_rounds,
                    avg_p,
                    eps,
                )

        convergence_round = self.detect_convergence(all_prices)

        # Compute Nash and monopoly prices for context
        marginal_cost = game_config.get("marginal_cost", 1.0)
        demand_intercept = game_config.get("demand_intercept", 10.0)
        demand_slope = game_config.get("demand_slope", 1.0)
        a = demand_intercept / demand_slope
        nash_price = (a + self.num_players * marginal_cost) / (
            self.num_players + 1
        )
        monopoly_price = (a + marginal_cost) / 2.0

        tail_start = convergence_round if convergence_round is not None else max(0, num_rounds - 1000)
        mean_converged = np.mean(all_prices[tail_start:], axis=0)

        return {
            "prices": all_prices,
            "profits": all_profits,
            "convergence_round": convergence_round,
            "mean_converged_prices": mean_converged.tolist(),
            "nash_price": nash_price,
            "monopoly_price": monopoly_price,
            "final_epsilon": self.agents[0].epsilon,
            "total_steps": self.agents[0].total_steps,
        }

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def detect_convergence(
        self,
        price_history: np.ndarray,
        window: int = 1000,
        threshold: float = 0.001,
    ) -> Optional[int]:
        """Detect the round at which prices converge."""
        if price_history.shape[0] < window:
            return None
        avg = np.mean(price_history, axis=1)
        for start in range(len(avg) - window + 1):
            if np.std(avg[start : start + window]) < threshold:
                return int(start)
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "agent_states": [a.get_state() for a in self.agents],
            "action_history": [h.copy() for h in self._action_history],
            "current_state": list(self._current_state),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        for a, s in zip(self.agents, state["agent_states"]):
            a.set_state(s)
        self._action_history = [h.copy() for h in state["action_history"]]
        self._current_state = list(state["current_state"])

    def __repr__(self) -> str:
        return (
            f"MultiAgentQLearning(players={self.num_players}, "
            f"actions={self.num_actions}, "
            f"memory={self.memory_length})"
        )


# ── Experience Replay Q-Learning ─────────────────────────────────────────────


class ExperienceReplayQLearning(QLearningAgent):
    """Q-learning with experience replay buffer.

    Stores transitions in a circular buffer and periodically samples
    mini-batches for additional Q-value updates.
    """

    def __init__(
        self,
        num_actions: int,
        num_states: int,
        buffer_size: int = 10000,
        batch_size: int = 32,
        replay_frequency: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
    ) -> None:
        super().__init__(
            num_actions=num_actions,
            num_states=num_states,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
        )
        self.buffer: List[Tuple[int, int, float, int]] = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_frequency = replay_frequency
        self._insert_pos = 0

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def store_experience(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        """Store a transition in the replay buffer."""
        transition = (
            int(state) % self.num_states,
            int(action) % self.num_actions,
            float(reward),
            int(next_state) % self.num_states,
        )
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self._insert_pos] = transition
        self._insert_pos = (self._insert_pos + 1) % self.buffer_size

    def replay(self) -> None:
        """Sample and learn from experience buffer."""
        if len(self.buffer) < self.batch_size:
            return
        indices = np.random.randint(0, len(self.buffer), size=self.batch_size)
        for idx in indices:
            s, a, r, ns = self.buffer[idx]
            best_next = np.max(self.q_table[ns])
            td_target = r + self.discount_factor * best_next
            td_error = td_target - self.q_table[s, a]
            self.q_table[s, a] += self.learning_rate * td_error

    # ------------------------------------------------------------------
    # Override update to include buffer + replay
    # ------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        """Q-learning update with experience replay."""
        # Standard online update
        super().update(state, action, reward, next_state)
        # Store and optionally replay
        self.store_experience(state, action, reward, next_state)
        if self.total_steps % self.replay_frequency == 0:
            self.replay()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["buffer"] = list(self.buffer)
        base["insert_pos"] = self._insert_pos
        return base

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state)
        self.buffer = list(state.get("buffer", []))
        self._insert_pos = state.get("insert_pos", 0)

    def __repr__(self) -> str:
        return (
            f"ExperienceReplayQLearning(states={self.num_states}, "
            f"actions={self.num_actions}, "
            f"buffer={len(self.buffer)}/{self.buffer_size}, "
            f"eps={self.epsilon:.4f})"
        )
