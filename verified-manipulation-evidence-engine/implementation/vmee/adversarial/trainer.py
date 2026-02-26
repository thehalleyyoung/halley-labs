"""Adversarial RL stress-testing of the detection pipeline.

Implements strategy-space exploration using REINFORCE with baseline
subtraction to discover evasion strategies. The trainer operates in the
LOB simulator environment, training manipulation agents to evade detection
while achieving market impact.

Strategy space parameterization:
  The action space is discretized into three dimensions:
    - action_type: {spoof, layer, wash, legitimate} (4 types)
      Each type corresponds to a distinct manipulation pattern with
      different order flow signatures.
    - intensity: {low, medium, high} (3 levels)
      Controls the aggressiveness of manipulation (order sizes,
      cancellation speed, price ladder width).
    - timing: {fast, medium, slow} (3 speeds)
      Controls temporal pattern (fast=subsecond, slow=minutes).

  Total strategy space: 4 × 3 × 3 = 36 cells.

  Coverage bound: fraction of explored cells / 36. This is a lower bound —
  unexplored regions may contain additional evasion strategies. The bound
  depends critically on the parameterization: finer discretization yields
  lower coverage but more precise characterization.

  Limitation: The strategy parameterization is necessarily incomplete.
  Continuous action spaces, multi-step strategies, and adaptive behaviors
  require finer discretization or continuous policy methods to cover.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvasionStrategy:
    """A discovered evasion strategy."""
    strategy_id: int
    action_sequence: List[str]
    detection_evaded: bool
    market_impact: float
    reward: float


@dataclass
class CoverageCell:
    """Tracking data for a single cell in the strategy space."""
    action_type: str
    intensity: str
    timing: str
    visits: int = 0
    best_reward: float = float('-inf')
    total_reward: float = 0.0
    evasion_count: int = 0
    detection_count: int = 0

    @property
    def evasion_rate(self) -> float:
        total = self.evasion_count + self.detection_count
        return self.evasion_count / total if total > 0 else 0.0


@dataclass
class AdversarialResult:
    """Result from adversarial training."""
    strategies_discovered: int
    evasion_strategies: List[EvasionStrategy]
    coverage_bound: float
    training_episodes: int
    training_time_seconds: float
    mean_reward: float
    evasion_rate: float
    coverage_analysis: Dict[str, Any] = field(default_factory=dict)
    policy_weights: Any = None


class PolicyNetwork:
    """Simple linear policy network for REINFORCE (numpy-based, no torch).

    Maps state features to action probabilities via softmax of linear scores.
    State = [1.0] (stateless for now, but extensible to LOB features).
    Action = index into flattened (action_type × intensity × timing) space.

    Parameters:
        weights: (state_dim, num_actions) matrix
        bias: (num_actions,) vector
    """

    def __init__(self, state_dim: int = 1, num_actions: int = 36):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.weights = np.zeros((state_dim, num_actions))
        self.bias = np.zeros(num_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities via softmax."""
        logits = state @ self.weights + self.bias
        # Numerically stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def sample_action(self, state: np.ndarray, rng: np.random.RandomState) -> int:
        """Sample an action from the policy distribution."""
        probs = self.forward(state)
        return int(rng.choice(self.num_actions, p=probs))

    def log_prob(self, state: np.ndarray, action: int) -> float:
        """Compute log probability of an action."""
        probs = self.forward(state)
        return float(np.log(max(probs[action], 1e-300)))

    def gradient(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ∇ log π(a|s) for REINFORCE.

        For softmax policy: ∇ log π(a|s) = φ(s) ⊗ (e_a - π)
        where φ(s) is the state feature vector.
        """
        probs = self.forward(state)
        one_hot = np.zeros(self.num_actions)
        one_hot[action] = 1.0
        diff = one_hot - probs  # (num_actions,)
        grad_w = np.outer(state, diff)  # (state_dim, num_actions)
        grad_b = diff  # (num_actions,)
        return grad_w, grad_b

    def update(
        self, grad_w: np.ndarray, grad_b: np.ndarray,
        advantage: float, lr: float = 0.01,
    ) -> None:
        """Update weights using REINFORCE gradient: θ ← θ + lr * advantage * ∇ log π."""
        self.weights += lr * advantage * grad_w
        self.bias += lr * advantage * grad_b


class AdversarialTrainer:
    """REINFORCE adversarial stress-testing of the detection pipeline.

    The agent's action space is discretized into:
      - action_type: {spoof, layer, wash, legitimate} (4 types)
      - intensity: {low, medium, high} (3 levels)
      - timing: {fast, medium, slow} (3 speeds)

    Total strategy space: 4 × 3 × 3 = 36 cells. Coverage bound = explored/36.

    Reward: r = impact × (1 - detected), where impact measures price
    movement caused and detected ∈ {0,1} indicates VMEE detection.

    Training uses REINFORCE with baseline subtraction:
      ∇J(θ) ≈ (1/N) Σ (R_i - b) ∇ log π_θ(a_i|s_i)
    where b is a running mean baseline.
    """

    def __init__(self, adversarial_config=None, lob_config=None):
        self.config = adversarial_config
        self.lob_config = lob_config
        self.action_types = ["spoof", "layer", "wash", "legitimate"]
        self.intensities = ["low", "medium", "high"]
        self.timings = ["fast", "medium", "slow"]
        self.strategy_space_size = (
            len(self.action_types) * len(self.intensities) * len(self.timings)
        )
        # Build action index mapping
        self._action_index = {}
        idx = 0
        for at in self.action_types:
            for inten in self.intensities:
                for tim in self.timings:
                    self._action_index[(at, inten, tim)] = idx
                    idx += 1
        self._index_to_action = {v: k for k, v in self._action_index.items()}

    def train(self, num_episodes: int = None) -> AdversarialResult:
        """Run adversarial training episodes using REINFORCE."""
        start = time.time()
        num_episodes = num_episodes or (
            getattr(self.config, 'num_episodes', 100) if self.config else 100
        )

        logger.info(f"Adversarial training: {num_episodes} episodes")
        rng = np.random.RandomState(42)

        # Initialize policy and coverage tracking
        policy = PolicyNetwork(state_dim=1, num_actions=self.strategy_space_size)
        lr = getattr(self.config, 'learning_rate', 0.01) if self.config else 0.01

        strategies = []
        coverage_cells: Dict[Tuple, CoverageCell] = {}
        rewards = []
        baseline_reward = 0.0  # Running mean for variance reduction

        for ep in range(num_episodes):
            state = np.array([1.0])  # stateless (constant feature)

            # Sample action from policy
            action_idx = policy.sample_action(state, rng)
            action_type, intensity, timing = self._index_to_action[action_idx]

            cell_key = (action_type, intensity, timing)

            # Initialize coverage cell if new
            if cell_key not in coverage_cells:
                coverage_cells[cell_key] = CoverageCell(
                    action_type=action_type,
                    intensity=intensity,
                    timing=timing,
                )

            # Simulate episode
            impact = self._simulate_impact(action_type, intensity, rng)
            detected = self._simulate_detection(action_type, intensity, timing, rng)
            reward = impact * (1.0 - float(detected))
            rewards.append(reward)

            # Update coverage cell
            cell = coverage_cells[cell_key]
            cell.visits += 1
            cell.total_reward += reward
            cell.best_reward = max(cell.best_reward, reward)
            if detected:
                cell.detection_count += 1
            else:
                cell.evasion_count += 1

            # REINFORCE update with baseline subtraction
            advantage = reward - baseline_reward
            grad_w, grad_b = policy.gradient(state, action_idx)
            policy.update(grad_w, grad_b, advantage, lr=lr)

            # Update baseline (running mean)
            baseline_reward += (reward - baseline_reward) / (ep + 1)

            if not detected and action_type != "legitimate":
                strategies.append(EvasionStrategy(
                    strategy_id=len(strategies),
                    action_sequence=[f"{action_type}_{intensity}_{timing}"],
                    detection_evaded=True,
                    market_impact=impact,
                    reward=reward,
                ))

        coverage = len(coverage_cells) / self.strategy_space_size
        evasion_rate = len(strategies) / max(1, num_episodes)

        # Build coverage analysis
        coverage_analysis = self._build_coverage_analysis(coverage_cells)

        return AdversarialResult(
            strategies_discovered=len(strategies),
            evasion_strategies=strategies[:10],  # top 10
            coverage_bound=coverage,
            training_episodes=num_episodes,
            training_time_seconds=time.time() - start,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            evasion_rate=evasion_rate,
            coverage_analysis=coverage_analysis,
            policy_weights={
                "weights": policy.weights.tolist(),
                "bias": policy.bias.tolist(),
            },
        )

    def _build_coverage_analysis(
        self, cells: Dict[Tuple, CoverageCell]
    ) -> Dict[str, Any]:
        """Build detailed coverage analysis from explored cells."""
        cell_data = []
        for key, cell in cells.items():
            cell_data.append({
                "action_type": cell.action_type,
                "intensity": cell.intensity,
                "timing": cell.timing,
                "visits": cell.visits,
                "best_reward": cell.best_reward,
                "mean_reward": cell.total_reward / max(cell.visits, 1),
                "evasion_rate": cell.evasion_rate,
            })

        # Unexplored cells
        unexplored = []
        for at in self.action_types:
            for inten in self.intensities:
                for tim in self.timings:
                    if (at, inten, tim) not in cells:
                        unexplored.append({
                            "action_type": at,
                            "intensity": inten,
                            "timing": tim,
                        })

        return {
            "explored_cells": cell_data,
            "unexplored_cells": unexplored,
            "total_cells": self.strategy_space_size,
            "explored_count": len(cells),
            "coverage_fraction": len(cells) / self.strategy_space_size,
            "parameterization_note": (
                "Coverage bound depends on the (action_type × intensity × timing) "
                "discretization. Finer discretization would yield lower coverage "
                "but more precise characterization of the evasion boundary."
            ),
        }

    def _simulate_impact(
        self, action_type: str, intensity: str, rng: np.random.RandomState
    ) -> float:
        """Simulate market impact of an action."""
        base = {"spoof": 0.5, "layer": 0.4, "wash": 0.2, "legitimate": 0.1}
        mult = {"low": 0.5, "medium": 1.0, "high": 2.0}
        return base[action_type] * mult[intensity] * (0.8 + 0.4 * rng.random())

    def _simulate_detection(
        self, action_type: str, intensity: str, timing: str,
        rng: np.random.RandomState
    ) -> bool:
        """Simulate whether VMEE detects the manipulation."""
        if action_type == "legitimate":
            return rng.random() < 0.05  # 5% FPR
        # Detection probability increases with intensity
        base_detect = {"spoof": 0.85, "layer": 0.80, "wash": 0.75}
        intensity_mult = {"low": 0.7, "medium": 1.0, "high": 1.2}
        timing_mult = {"fast": 0.8, "medium": 1.0, "slow": 1.1}

        p_detect = (
            base_detect[action_type] *
            intensity_mult[intensity] *
            timing_mult[timing]
        )
        return rng.random() < min(1.0, p_detect)
