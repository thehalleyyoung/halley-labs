"""Deep Q-Network agent implemented in pure NumPy.

Provides a feedforward neural network, experience replay buffer, and a
DQN agent with a target network — all without external deep-learning
frameworks.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("collusion_proof.oracle.dqn")


# ── Neural Network ───────────────────────────────────────────────────────────


class NeuralNetwork:
    """Simple feedforward neural network in NumPy.

    Supports ReLU and Tanh hidden activations with a linear output layer.
    Training is via vanilla backpropagation with optional gradient clipping.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer sizes")

        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Xavier initialisation
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out) * std)
            self.biases.append(np.zeros(fan_out))

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0)
        if self.activation == "tanh":
            return np.tanh(z)
        return z  # linear fallback

    def _activate_deriv(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(float)
        if self.activation == "tanh":
            t = np.tanh(z)
            return 1.0 - t * t
        return np.ones_like(z)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.  *x* has shape ``(batch, input_dim)`` or ``(input_dim,)``."""
        x = np.atleast_2d(x).astype(float)
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = self._activate(x)
        # Output layer — linear
        x = x @ self.weights[-1] + self.biases[-1]
        return x

    # ------------------------------------------------------------------
    # Backward (full backprop)
    # ------------------------------------------------------------------

    def backward(
        self,
        x: np.ndarray,
        target: np.ndarray,
        learning_rate: float = 0.001,
        clip_grad: float = 1.0,
    ) -> float:
        """Backpropagation with MSE loss.  Returns loss."""
        x = np.atleast_2d(x).astype(float)
        target = np.atleast_2d(target).astype(float)
        batch_size = x.shape[0]

        # Forward pass — store pre-activations
        zs: List[np.ndarray] = []  # pre-activation
        activations: List[np.ndarray] = [x]
        a = x
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            zs.append(z)
            a = self._activate(z)
            activations.append(a)
        # Output layer
        z_out = a @ self.weights[-1] + self.biases[-1]
        zs.append(z_out)
        activations.append(z_out)

        # Loss
        error = activations[-1] - target
        loss = float(np.mean(error ** 2))

        # Backward pass
        delta = error / batch_size  # dL/dz for output (linear activation)
        weight_grads: List[np.ndarray] = [None] * len(self.weights)  # type: ignore
        bias_grads: List[np.ndarray] = [None] * len(self.biases)  # type: ignore

        # Output layer gradients
        weight_grads[-1] = activations[-2].T @ delta
        bias_grads[-1] = np.sum(delta, axis=0)

        # Hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * self._activate_deriv(zs[i])
            weight_grads[i] = activations[i].T @ delta
            bias_grads[i] = np.sum(delta, axis=0)

        # Gradient clipping & parameter update
        for i in range(len(self.weights)):
            wg_norm = np.linalg.norm(weight_grads[i])
            if wg_norm > clip_grad:
                weight_grads[i] *= clip_grad / wg_norm
            bg_norm = np.linalg.norm(bias_grads[i])
            if bg_norm > clip_grad:
                bias_grads[i] *= clip_grad / bg_norm

            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]

        return loss

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]

    def copy_from(self, other: "NeuralNetwork") -> None:
        """Copy parameters from *other* into this network."""
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]

    def __repr__(self) -> str:
        return f"NeuralNetwork({self.layer_sizes}, activation={self.activation!r})"


# ── Replay Buffer ────────────────────────────────────────────────────────────


class ReplayBuffer:
    """Circular experience replay buffer."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._pos = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition = (
            np.asarray(state, dtype=float),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=float),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=float),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer({len(self.buffer)}/{self.capacity})"


# ── DQN Agent ────────────────────────────────────────────────────────────────


class DQNAgent:
    """Deep Q-Network agent for pricing games.

    Uses a policy network for action selection and a periodically-synced
    target network for stable TD targets.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_sizes: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 500,
        activation: str = "relu",
    ) -> None:
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        layer_sizes = [state_dim] + hidden_sizes + [num_actions]
        self.policy_net = NeuralNetwork(layer_sizes, activation=activation)
        self.target_net = NeuralNetwork(layer_sizes, activation=activation)
        self.target_net.copy_from(self.policy_net)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: Any) -> int:
        """Epsilon-greedy action selection using the policy network."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))

        state_arr = np.atleast_1d(np.asarray(state, dtype=float))
        q_values = self.policy_net.forward(state_arr)
        return int(np.argmax(q_values.ravel()))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool = False,
    ) -> Optional[float]:
        """Store experience and train.  Returns loss if training occurred."""
        state_arr = np.atleast_1d(np.asarray(state, dtype=float))
        next_arr = np.atleast_1d(np.asarray(next_state, dtype=float))

        self.replay_buffer.push(state_arr, int(action), reward, next_arr, done)
        self.total_steps += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if len(self.replay_buffer) < self.batch_size:
            return None

        loss = self._train_step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.copy_from(self.policy_net)

        return loss

    def _train_step(self) -> float:
        """Single training step from replay buffer."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Compute target Q-values using target network
        next_q = self.target_net.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets_full = self.policy_net.forward(states).copy()

        for i in range(self.batch_size):
            td_target = rewards[i] + (1.0 - dones[i]) * self.discount_factor * max_next_q[i]
            targets_full[i, actions[i]] = td_target

        loss = self.policy_net.backward(
            states, targets_full, learning_rate=self.learning_rate
        )
        return loss

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "policy_params": self.policy_net.get_params(),
            "target_params": self.target_net.get_params(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "buffer": list(self.replay_buffer.buffer),
            "buffer_pos": self.replay_buffer._pos,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.policy_net.set_params(state["policy_params"])
        self.target_net.set_params(state["target_params"])
        self.epsilon = state["epsilon"]
        self.total_steps = state["total_steps"]
        self.replay_buffer.buffer = list(state.get("buffer", []))
        self.replay_buffer._pos = state.get("buffer_pos", 0)

    def __repr__(self) -> str:
        return (
            f"DQNAgent(state_dim={self.state_dim}, "
            f"actions={self.num_actions}, "
            f"eps={self.epsilon:.4f}, "
            f"steps={self.total_steps}, "
            f"buffer={len(self.replay_buffer)})"
        )
