"""Multi-armed bandit agents for pricing.

Provides several bandit algorithms suited to stateless or slowly-changing
pricing environments:
- Epsilon-Greedy
- UCB1
- Thompson Sampling (Beta posterior for bounded rewards)
- EXP3 (adversarial bandit)
- Softmax / Boltzmann exploration
- Gradient Bandit with preference-based action selection
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("collusion_proof.oracle.bandits")


# ── Epsilon-Greedy ───────────────────────────────────────────────────────────


class EpsilonGreedyBandit:
    """Epsilon-greedy bandit for pricing.

    Maintains running estimates of each arm's mean reward and
    explores uniformly at random with probability *epsilon*.
    """

    def __init__(
        self,
        num_actions: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        initial_value: float = 0.0,
    ) -> None:
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.counts = np.zeros(num_actions, dtype=int)
        self.values = np.full(num_actions, initial_value, dtype=float)
        self.total_steps = 0

    def select_action(self, state: Any = None) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        max_val = np.max(self.values)
        candidates = np.where(np.abs(self.values - max_val) < 1e-10)[0]
        return int(np.random.choice(candidates))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any = None,
    ) -> None:
        action = int(action)
        self.counts[action] += 1
        # Incremental mean update
        self.values[action] += (reward - self.values[action]) / self.counts[action]
        self.total_steps += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def get_state(self) -> Dict[str, Any]:
        return {
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.counts = state["counts"].copy()
        self.values = state["values"].copy()
        self.epsilon = state["epsilon"]
        self.total_steps = state["total_steps"]

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyBandit(actions={self.num_actions}, "
            f"eps={self.epsilon:.4f})"
        )


# ── UCB1 ─────────────────────────────────────────────────────────────────────


class UCB1Agent:
    """UCB1 bandit algorithm.

    Selects the arm that maximises: mean_reward + c * sqrt(ln(t) / n_i)
    where c is the exploration weight.
    """

    def __init__(
        self,
        num_actions: int,
        exploration_weight: float = 2.0,
    ) -> None:
        self.num_actions = num_actions
        self.exploration_weight = exploration_weight
        self.counts = np.zeros(num_actions, dtype=int)
        self.values = np.zeros(num_actions, dtype=float)
        self.total_steps = 0

    def select_action(self, state: Any = None) -> int:
        # Pull each arm at least once
        for a in range(self.num_actions):
            if self.counts[a] == 0:
                return a

        log_t = np.log(self.total_steps)
        ucb_values = self.values + self.exploration_weight * np.sqrt(
            log_t / self.counts
        )
        return int(np.argmax(ucb_values))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any = None,
    ) -> None:
        action = int(action)
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]
        self.total_steps += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "total_steps": self.total_steps,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.counts = state["counts"].copy()
        self.values = state["values"].copy()
        self.total_steps = state["total_steps"]

    def __repr__(self) -> str:
        return (
            f"UCB1Agent(actions={self.num_actions}, "
            f"c={self.exploration_weight:.2f})"
        )


# ── Thompson Sampling ────────────────────────────────────────────────────────


class ThompsonSamplingAgent:
    """Thompson sampling with Beta priors for bounded rewards.

    Rewards are assumed to lie in ``[reward_min, reward_max]``.  Each
    observation is rescaled to [0, 1] and used to update a
    Beta(alpha, beta) posterior.
    """

    def __init__(
        self,
        num_actions: int,
        reward_min: float = 0.0,
        reward_max: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self.num_actions = num_actions
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.alphas = np.full(num_actions, prior_alpha, dtype=float)
        self.betas = np.full(num_actions, prior_beta, dtype=float)
        self.counts = np.zeros(num_actions, dtype=int)
        self.total_steps = 0

    def _normalise_reward(self, reward: float) -> float:
        span = self.reward_max - self.reward_min
        if span <= 0:
            return 0.5
        return np.clip((reward - self.reward_min) / span, 0.0, 1.0)

    def select_action(self, state: Any = None) -> int:
        samples = np.array(
            [
                np.random.beta(self.alphas[a], self.betas[a])
                for a in range(self.num_actions)
            ]
        )
        return int(np.argmax(samples))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any = None,
    ) -> None:
        action = int(action)
        p = self._normalise_reward(reward)
        self.alphas[action] += p
        self.betas[action] += 1.0 - p
        self.counts[action] += 1
        self.total_steps += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "alphas": self.alphas.copy(),
            "betas": self.betas.copy(),
            "counts": self.counts.copy(),
            "total_steps": self.total_steps,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.alphas = state["alphas"].copy()
        self.betas = state["betas"].copy()
        self.counts = state["counts"].copy()
        self.total_steps = state["total_steps"]

    def __repr__(self) -> str:
        return f"ThompsonSamplingAgent(actions={self.num_actions})"


# ── EXP3 ─────────────────────────────────────────────────────────────────────


class EXP3Agent:
    """EXP3 adversarial bandit algorithm.

    Maintains a probability distribution over actions using
    exponential-weight updates.  The parameter *gamma* controls the
    exploration-exploitation trade-off.
    """

    def __init__(
        self,
        num_actions: int,
        gamma: float = 0.1,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.weights = np.ones(num_actions, dtype=float)
        self.counts = np.zeros(num_actions, dtype=int)
        self.total_steps = 0
        self._last_probs: Optional[np.ndarray] = None

    def _get_probs(self) -> np.ndarray:
        total_w = np.sum(self.weights)
        probs = (1.0 - self.gamma) * (self.weights / total_w) + self.gamma / self.num_actions
        # Ensure valid distribution
        probs = np.maximum(probs, 1e-10)
        probs /= np.sum(probs)
        return probs

    def select_action(self, state: Any = None) -> int:
        self._last_probs = self._get_probs()
        return int(np.random.choice(self.num_actions, p=self._last_probs))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any = None,
    ) -> None:
        action = int(action)
        probs = self._last_probs if self._last_probs is not None else self._get_probs()

        # Importance-weighted reward estimate
        estimated_reward = reward / max(probs[action], 1e-10)
        # Clip to prevent numerical explosion
        estimated_reward = np.clip(estimated_reward, -100.0, 100.0)

        self.weights[action] *= np.exp(self.gamma * estimated_reward / self.num_actions)

        # Normalise weights to prevent overflow
        max_w = np.max(self.weights)
        if max_w > 1e10:
            self.weights /= max_w

        self.counts[action] += 1
        self.total_steps += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.copy(),
            "counts": self.counts.copy(),
            "total_steps": self.total_steps,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.weights = state["weights"].copy()
        self.counts = state["counts"].copy()
        self.total_steps = state["total_steps"]

    def __repr__(self) -> str:
        return (
            f"EXP3Agent(actions={self.num_actions}, gamma={self.gamma:.3f})"
        )


# ── Softmax / Boltzmann ──────────────────────────────────────────────────────


class SoftmaxBandit:
    """Boltzmann / softmax exploration bandit.

    Selects actions according to a Boltzmann distribution over
    estimated Q-values with temperature parameter *tau*.
    """

    def __init__(
        self,
        num_actions: int,
        tau: float = 1.0,
        tau_decay: float = 1.0,
        tau_min: float = 0.01,
    ) -> None:
        self.num_actions = num_actions
        self.tau = tau
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.counts = np.zeros(num_actions, dtype=int)
        self.values = np.zeros(num_actions, dtype=float)
        self.total_steps = 0

    def _softmax_probs(self) -> np.ndarray:
        logits = self.values / max(self.tau, 1e-10)
        logits -= np.max(logits)  # numerical stability
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l)
        return probs

    def select_action(self, state: Any = None) -> int:
        probs = self._softmax_probs()
        return int(np.random.choice(self.num_actions, p=probs))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any = None,
    ) -> None:
        action = int(action)
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]
        self.total_steps += 1
        self.tau = max(self.tau * self.tau_decay, self.tau_min)

    def get_state(self) -> Dict[str, Any]:
        return {
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "tau": self.tau,
            "total_steps": self.total_steps,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.counts = state["counts"].copy()
        self.values = state["values"].copy()
        self.tau = state["tau"]
        self.total_steps = state["total_steps"]

    def __repr__(self) -> str:
        return (
            f"SoftmaxBandit(actions={self.num_actions}, tau={self.tau:.4f})"
        )


# ── Gradient Bandit ──────────────────────────────────────────────────────────


class GradientBandit:
    """Gradient bandit with preference-based action selection.

    Maintains a preference vector *H* and selects actions via
    softmax(H).  Preferences are updated using stochastic gradient
    ascent on expected reward.
    """

    def __init__(
        self,
        num_actions: int,
        step_size: float = 0.1,
        use_baseline: bool = True,
    ) -> None:
        self.num_actions = num_actions
        self.step_size = step_size
        self.use_baseline = use_baseline

        self.preferences = np.zeros(num_actions, dtype=float)
        self.counts = np.zeros(num_actions, dtype=int)
        self.avg_reward = 0.0
        self.total_steps = 0
        self._last_probs: Optional[np.ndarray] = None

    def _probs(self) -> np.ndarray:
        h = self.preferences - np.max(self.preferences)
        exp_h = np.exp(h)
        return exp_h / np.sum(exp_h)

    def select_action(self, state: Any = None) -> int:
        self._last_probs = self._probs()
        return int(np.random.choice(self.num_actions, p=self._last_probs))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any = None,
    ) -> None:
        action = int(action)
        probs = self._last_probs if self._last_probs is not None else self._probs()

        baseline = self.avg_reward if self.use_baseline else 0.0
        advantage = reward - baseline

        # Update preferences
        for a in range(self.num_actions):
            if a == action:
                self.preferences[a] += self.step_size * advantage * (1.0 - probs[a])
            else:
                self.preferences[a] -= self.step_size * advantage * probs[a]

        self.counts[action] += 1
        self.total_steps += 1
        # Incremental average reward for baseline
        self.avg_reward += (reward - self.avg_reward) / self.total_steps

    def get_state(self) -> Dict[str, Any]:
        return {
            "preferences": self.preferences.copy(),
            "counts": self.counts.copy(),
            "avg_reward": self.avg_reward,
            "total_steps": self.total_steps,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.preferences = state["preferences"].copy()
        self.counts = state["counts"].copy()
        self.avg_reward = state["avg_reward"]
        self.total_steps = state["total_steps"]

    def __repr__(self) -> str:
        return (
            f"GradientBandit(actions={self.num_actions}, "
            f"step_size={self.step_size:.3f})"
        )
