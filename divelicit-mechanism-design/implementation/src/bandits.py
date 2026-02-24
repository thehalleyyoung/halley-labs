"""Bandit-based diversity selection algorithms.

Multi-armed, combinatorial, and adversarial bandit algorithms adapted for
iterative diverse subset selection from LLM candidate responses.

Theoretical foundations:
- UCB1 regret bounds: O(sqrt(KT log T))
- Thompson sampling: Bayesian regret O(sqrt(KT))
- CUCB for combinatorial: O(sqrt(mKT log T))
- EXP3 adversarial: O(sqrt(KT log K))
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .kernels import Kernel, RBFKernel
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BanditArm:
    """An arm in the bandit problem."""
    arm_id: int
    embedding: np.ndarray
    true_quality: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class BanditState:
    """Internal state of a bandit algorithm."""
    n_arms: int
    n_rounds: int = 0
    pulls: np.ndarray = field(default_factory=lambda: np.array([]))
    rewards: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_rewards: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_regret: float = 0.0
    regret_history: List[float] = field(default_factory=list)
    selection_history: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        if len(self.pulls) == 0:
            self.pulls = np.zeros(self.n_arms)
            self.rewards = np.zeros(self.n_arms)
            self.cumulative_rewards = np.zeros(self.n_arms)


@dataclass
class BanditResult:
    """Result from running a bandit algorithm."""
    selected_arms: List[int]
    final_selection: List[int]
    total_reward: float
    cumulative_regret: float
    regret_history: List[float]
    diversity_history: List[float]
    reward_history: List[float]
    n_rounds: int
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def diversity_reward(
    selected_indices: List[int],
    arms: List[BanditArm],
    kernel: Optional[Kernel] = None,
    quality_weight: float = 0.5,
    diversity_weight: float = 0.5,
) -> float:
    """Compute reward combining quality and diversity."""
    if len(selected_indices) == 0:
        return 0.0
    selected = [arms[i] for i in selected_indices]
    quality = np.mean([a.true_quality for a in selected])

    if len(selected) < 2:
        diversity = 0.0
    else:
        embeddings = np.array([a.embedding for a in selected])
        if kernel is None:
            kernel = RBFKernel(bandwidth=1.0)
        K = kernel.gram_matrix(embeddings)
        diversity = log_det_safe(K)

    return quality_weight * quality + diversity_weight * max(diversity, 0.0)


def marginal_diversity_reward(
    arm_idx: int,
    current_selection: List[int],
    arms: List[BanditArm],
    kernel: Optional[Kernel] = None,
) -> float:
    """Marginal reward from adding arm to current selection."""
    if kernel is None:
        kernel = RBFKernel(bandwidth=1.0)
    r_without = diversity_reward(current_selection, arms, kernel)
    r_with = diversity_reward(current_selection + [arm_idx], arms, kernel)
    return r_with - r_without


def pairwise_distance_reward(
    selected_indices: List[int],
    arms: List[BanditArm],
) -> float:
    """Sum of pairwise distances reward."""
    if len(selected_indices) < 2:
        return 0.0
    embeddings = np.array([arms[i].embedding for i in selected_indices])
    n = len(embeddings)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(embeddings[i] - embeddings[j])
    return total / (n * (n - 1) / 2)


# ---------------------------------------------------------------------------
# Base bandit class
# ---------------------------------------------------------------------------

class BanditAlgorithm(ABC):
    """Base class for bandit algorithms."""

    def __init__(self, n_arms: int, seed: int = 42):
        self.n_arms = n_arms
        self.state = BanditState(n_arms=n_arms)
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def select_arm(self) -> int:
        """Select an arm to pull."""
        ...

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Update internal state after observing reward."""
        ...

    def reset(self) -> None:
        """Reset the bandit state."""
        self.state = BanditState(n_arms=self.n_arms)


# ---------------------------------------------------------------------------
# UCB1 with Diversity Reward
# ---------------------------------------------------------------------------

class UCB1Diversity(BanditAlgorithm):
    """UCB1 algorithm adapted for diversity-aware arm selection.

    Upper Confidence Bound:
      UCB_i(t) = hat{mu}_i + c * sqrt(2 * ln(t) / N_i(t))

    Diversity bonus is added to the reward signal.
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        exploration_constant: float = 1.0,
        diversity_weight: float = 0.5,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.c = exploration_constant
        self.diversity_weight = diversity_weight
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.current_selection: List[int] = []
        self._empirical_means = np.zeros(n_arms)
        self._ucb_values = np.full(n_arms, float("inf"))

    def select_arm(self) -> int:
        """Select arm with highest UCB value."""
        t = self.state.n_rounds + 1
        for i in range(self.n_arms):
            if self.state.pulls[i] == 0:
                return i
            mean = self.state.cumulative_rewards[i] / self.state.pulls[i]
            bonus = self.c * math.sqrt(2 * math.log(t) / self.state.pulls[i])
            # Add diversity marginal gain
            div_bonus = 0.0
            if len(self.current_selection) > 0 and i not in self.current_selection:
                emb_i = self.arms[i].embedding
                min_dist = min(
                    np.linalg.norm(emb_i - self.arms[j].embedding)
                    for j in self.current_selection
                )
                div_bonus = self.diversity_weight * min_dist
            self._ucb_values[i] = mean + bonus + div_bonus
            self._empirical_means[i] = mean
        return int(np.argmax(self._ucb_values))

    def update(self, arm: int, reward: float) -> None:
        """Update after pulling arm."""
        self.state.pulls[arm] += 1
        self.state.cumulative_rewards[arm] += reward
        self.state.n_rounds += 1
        self._empirical_means[arm] = (
            self.state.cumulative_rewards[arm] / self.state.pulls[arm]
        )

    def run(
        self,
        n_rounds: int,
        k: int = 5,
        reward_fn: Optional[Callable] = None,
    ) -> BanditResult:
        """Run UCB1 for n_rounds, selecting k items."""
        reward_history: List[float] = []
        diversity_history: List[float] = []
        regret_history: List[float] = []
        total_reward = 0.0
        cumulative_regret = 0.0

        # Compute optimal reward for regret
        all_combos_reward = diversity_reward(
            list(range(min(k, self.n_arms))), self.arms, self.kernel,
        )

        for t in range(n_rounds):
            self.current_selection = []
            round_reward = 0.0

            for _ in range(min(k, self.n_arms)):
                arm = self.select_arm()
                if arm in self.current_selection:
                    # Find next best
                    ucb_copy = self._ucb_values.copy()
                    for s in self.current_selection:
                        ucb_copy[s] = -float("inf")
                    arm = int(np.argmax(ucb_copy))

                # Compute reward
                if reward_fn is not None:
                    reward = reward_fn(arm, self.current_selection, self.arms)
                else:
                    reward = marginal_diversity_reward(
                        arm, self.current_selection, self.arms, self.kernel
                    )
                # Add noise
                noisy_reward = reward + self.rng.normal(0, 0.1)
                self.update(arm, noisy_reward)
                self.current_selection.append(arm)
                round_reward += reward

            total_reward += round_reward
            regret = all_combos_reward - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)

            div = diversity_reward(self.current_selection, self.arms, self.kernel)
            diversity_history.append(div)
            reward_history.append(round_reward)

        return BanditResult(
            selected_arms=self.current_selection,
            final_selection=self.current_selection,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "UCB1_diversity"},
        )


# ---------------------------------------------------------------------------
# Thompson Sampling with Diversity Prior
# ---------------------------------------------------------------------------

class ThompsonSamplingDiversity(BanditAlgorithm):
    """Thompson Sampling with Beta posterior and diversity-informed prior.

    Prior: Beta(alpha_0, beta_0) where alpha_0 encodes diversity preference.
    Posterior: Beta(alpha_0 + successes, beta_0 + failures).
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        diversity_prior_strength: float = 0.5,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.diversity_prior_strength = diversity_prior_strength
        # Per-arm Beta parameters
        self.alphas = np.full(n_arms, alpha_prior)
        self.betas = np.full(n_arms, beta_prior)
        self.current_selection: List[int] = []

    def _diversity_prior_bonus(self, arm_idx: int) -> float:
        """Compute diversity-informed prior bonus."""
        if len(self.current_selection) == 0:
            return 0.0
        emb = self.arms[arm_idx].embedding
        min_dist = min(
            np.linalg.norm(emb - self.arms[j].embedding)
            for j in self.current_selection
        )
        return self.diversity_prior_strength * min_dist

    def select_arm(self) -> int:
        """Sample from posterior and add diversity bonus."""
        samples = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if i in self.current_selection:
                samples[i] = -float("inf")
                continue
            # Thompson sample
            ts = self.rng.beta(self.alphas[i], self.betas[i])
            div_bonus = self._diversity_prior_bonus(i)
            samples[i] = ts + div_bonus
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """Update Beta posterior."""
        # Convert reward to Bernoulli outcome
        success = 1 if reward > 0.5 else 0
        self.alphas[arm] += success
        self.betas[arm] += 1 - success
        self.state.pulls[arm] += 1
        self.state.cumulative_rewards[arm] += reward
        self.state.n_rounds += 1

    def run(
        self,
        n_rounds: int,
        k: int = 5,
    ) -> BanditResult:
        """Run Thompson Sampling for n_rounds."""
        reward_history: List[float] = []
        diversity_history: List[float] = []
        regret_history: List[float] = []
        total_reward = 0.0
        cumulative_regret = 0.0

        optimal_reward = diversity_reward(
            list(range(min(k, self.n_arms))), self.arms, self.kernel,
        )

        for t in range(n_rounds):
            self.current_selection = []
            round_reward = 0.0

            for _ in range(min(k, self.n_arms)):
                arm = self.select_arm()
                reward = marginal_diversity_reward(
                    arm, self.current_selection, self.arms, self.kernel
                )
                noisy = reward + self.rng.normal(0, 0.05)
                self.update(arm, max(0, min(1, noisy)))
                self.current_selection.append(arm)
                round_reward += reward

            total_reward += round_reward
            regret = optimal_reward - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)

            div = diversity_reward(self.current_selection, self.arms, self.kernel)
            diversity_history.append(div)
            reward_history.append(round_reward)

        return BanditResult(
            selected_arms=self.current_selection,
            final_selection=self.current_selection,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "thompson_sampling_diversity"},
        )


# ---------------------------------------------------------------------------
# Contextual Bandits
# ---------------------------------------------------------------------------

class ContextualBanditDiversity(BanditAlgorithm):
    """Contextual bandit with linear reward model.

    Reward model: r(a, c) = theta_a^T c + diversity_bonus
    Uses ridge regression to estimate theta_a for each arm.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        arms: List[BanditArm],
        lambda_reg: float = 1.0,
        alpha: float = 1.0,
        diversity_weight: float = 0.3,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.context_dim = context_dim
        self.arms = arms
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.diversity_weight = diversity_weight
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        # Per-arm ridge regression
        self.A = [lambda_reg * np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]
        self.current_selection: List[int] = []
        self._context: Optional[np.ndarray] = None

    def set_context(self, context: np.ndarray) -> None:
        """Set current context vector."""
        self._context = context

    def select_arm(self) -> int:
        """Select arm using LinUCB."""
        if self._context is None:
            return self.rng.randint(self.n_arms)
        c = self._context
        ucb_vals = np.full(self.n_arms, -float("inf"))
        for i in range(self.n_arms):
            if i in self.current_selection:
                continue
            A_inv = np.linalg.inv(self.A[i])
            self.theta[i] = A_inv @ self.b[i]
            pred = self.theta[i] @ c
            conf = self.alpha * math.sqrt(c @ A_inv @ c)
            # Diversity bonus
            div_bonus = 0.0
            if len(self.current_selection) > 0:
                emb = self.arms[i].embedding
                min_dist = min(
                    np.linalg.norm(emb - self.arms[j].embedding)
                    for j in self.current_selection
                )
                div_bonus = self.diversity_weight * min_dist
            ucb_vals[i] = pred + conf + div_bonus
        return int(np.argmax(ucb_vals))

    def update(self, arm: int, reward: float) -> None:
        """Update ridge regression for arm."""
        if self._context is None:
            return
        c = self._context
        self.A[arm] += np.outer(c, c)
        self.b[arm] += reward * c
        self.state.pulls[arm] += 1
        self.state.cumulative_rewards[arm] += reward
        self.state.n_rounds += 1

    def run(
        self,
        n_rounds: int,
        k: int = 5,
        context_generator: Optional[Callable[[], np.ndarray]] = None,
    ) -> BanditResult:
        """Run contextual bandit."""
        if context_generator is None:
            context_generator = lambda: self.rng.randn(self.context_dim)

        reward_history: List[float] = []
        diversity_history: List[float] = []
        regret_history: List[float] = []
        total_reward = 0.0
        cumulative_regret = 0.0
        optimal_reward = diversity_reward(
            list(range(min(k, self.n_arms))), self.arms, self.kernel,
        )

        for t in range(n_rounds):
            context = context_generator()
            self.set_context(context)
            self.current_selection = []
            round_reward = 0.0

            for _ in range(min(k, self.n_arms)):
                arm = self.select_arm()
                reward = marginal_diversity_reward(
                    arm, self.current_selection, self.arms, self.kernel
                )
                noisy = reward + self.rng.normal(0, 0.1)
                self.update(arm, noisy)
                self.current_selection.append(arm)
                round_reward += reward

            total_reward += round_reward
            regret = optimal_reward - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)
            diversity_history.append(
                diversity_reward(self.current_selection, self.arms, self.kernel)
            )
            reward_history.append(round_reward)

        return BanditResult(
            selected_arms=self.current_selection,
            final_selection=self.current_selection,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "contextual_linucb_diversity"},
        )


# ---------------------------------------------------------------------------
# Combinatorial UCB (CUCB)
# ---------------------------------------------------------------------------

class CUCB(BanditAlgorithm):
    """Combinatorial UCB for subset selection.

    At each round, selects a super-arm (subset of size k) that maximizes
    the sum of UCB indices, subject to diversity constraints.

    Regret bound: O(sqrt(m * K * T * log T)) where m = subset size.
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        k: int = 5,
        exploration_constant: float = 1.5,
        diversity_threshold: float = 0.0,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.k = k
        self.c = exploration_constant
        self.diversity_threshold = diversity_threshold
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self._empirical_means = np.zeros(n_arms)

    def _compute_ucb(self) -> np.ndarray:
        """Compute UCB index for each arm."""
        t = max(self.state.n_rounds, 1)
        ucb = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.state.pulls[i] == 0:
                ucb[i] = float("inf")
            else:
                mean = self.state.cumulative_rewards[i] / self.state.pulls[i]
                bonus = self.c * math.sqrt(math.log(t) / self.state.pulls[i])
                ucb[i] = mean + bonus
                self._empirical_means[i] = mean
        return ucb

    def select_arm(self) -> int:
        """Not used directly; use select_subset instead."""
        ucb = self._compute_ucb()
        return int(np.argmax(ucb))

    def select_subset(self) -> List[int]:
        """Select k arms maximizing UCB with diversity constraint."""
        ucb = self._compute_ucb()
        selected: List[int] = []

        for _ in range(min(self.k, self.n_arms)):
            best_arm = -1
            best_score = -float("inf")

            for i in range(self.n_arms):
                if i in selected:
                    continue
                # Check diversity constraint
                if len(selected) > 0 and self.diversity_threshold > 0:
                    emb = self.arms[i].embedding
                    min_dist = min(
                        np.linalg.norm(emb - self.arms[j].embedding)
                        for j in selected
                    )
                    if min_dist < self.diversity_threshold:
                        continue

                score = ucb[i]
                if score > best_score:
                    best_score = score
                    best_arm = i

            if best_arm < 0:
                break
            selected.append(best_arm)

        return selected

    def update(self, arm: int, reward: float) -> None:
        """Update arm statistics."""
        self.state.pulls[arm] += 1
        self.state.cumulative_rewards[arm] += reward
        self.state.n_rounds += 1

    def run(self, n_rounds: int) -> BanditResult:
        """Run CUCB for n_rounds."""
        reward_history: List[float] = []
        diversity_history: List[float] = []
        regret_history: List[float] = []
        total_reward = 0.0
        cumulative_regret = 0.0
        optimal = diversity_reward(
            list(range(min(self.k, self.n_arms))), self.arms, self.kernel,
        )

        for t in range(n_rounds):
            subset = self.select_subset()
            # Get reward for each arm in subset
            round_reward = 0.0
            for arm in subset:
                reward = self.arms[arm].true_quality + self.rng.normal(0, 0.1)
                self.update(arm, max(0, reward))
                round_reward += reward

            # Add diversity bonus
            div = diversity_reward(subset, self.arms, self.kernel)
            round_reward += div
            diversity_history.append(div)

            total_reward += round_reward
            regret = optimal - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)
            reward_history.append(round_reward)

        return BanditResult(
            selected_arms=subset,
            final_selection=subset,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "CUCB"},
        )


# ---------------------------------------------------------------------------
# CombLinUCB
# ---------------------------------------------------------------------------

class CombLinUCB(BanditAlgorithm):
    """Combinatorial Linear UCB for diverse subset selection.

    Models reward as linear: r(S) = theta^T phi(S) where phi(S) is a
    feature representation of subset S.

    Uses ridge regression to estimate theta and selects subsets
    maximizing the UCB of the predicted reward.
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        feature_dim: int,
        k: int = 5,
        lambda_reg: float = 1.0,
        alpha: float = 1.0,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.feature_dim = feature_dim
        self.k = k
        self.alpha = alpha
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.V = lambda_reg * np.eye(feature_dim)
        self.b_vec = np.zeros(feature_dim)
        self.theta_hat = np.zeros(feature_dim)

    def _subset_features(self, subset: List[int]) -> np.ndarray:
        """Compute feature vector for a subset."""
        if len(subset) == 0:
            return np.zeros(self.feature_dim)
        embeddings = np.array([self.arms[i].embedding for i in subset])
        # Features: mean embedding + diversity features
        mean_emb = np.mean(embeddings, axis=0)
        if len(mean_emb) >= self.feature_dim:
            return mean_emb[:self.feature_dim]
        # Pad with diversity statistics
        features = np.zeros(self.feature_dim)
        features[:len(mean_emb)] = mean_emb
        if len(subset) >= 2:
            # Add pairwise distance stats
            dists = []
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    dists.append(np.linalg.norm(embeddings[i] - embeddings[j]))
            if len(mean_emb) < self.feature_dim:
                features[len(mean_emb)] = np.mean(dists)
            if len(mean_emb) + 1 < self.feature_dim:
                features[len(mean_emb) + 1] = np.std(dists)
        return features

    def select_arm(self) -> int:
        """Not used directly."""
        return 0

    def select_subset(self) -> List[int]:
        """Greedy subset selection maximizing linear UCB."""
        V_inv = np.linalg.inv(self.V)
        self.theta_hat = V_inv @ self.b_vec
        selected: List[int] = []

        for _ in range(min(self.k, self.n_arms)):
            best_arm = -1
            best_ucb = -float("inf")

            for i in range(self.n_arms):
                if i in selected:
                    continue
                candidate = selected + [i]
                phi = self._subset_features(candidate)
                pred = self.theta_hat @ phi
                conf = self.alpha * math.sqrt(phi @ V_inv @ phi)
                ucb_val = pred + conf
                if ucb_val > best_ucb:
                    best_ucb = ucb_val
                    best_arm = i

            if best_arm >= 0:
                selected.append(best_arm)

        return selected

    def update_subset(self, subset: List[int], reward: float) -> None:
        """Update after observing reward for selected subset."""
        phi = self._subset_features(subset)
        self.V += np.outer(phi, phi)
        self.b_vec += reward * phi
        self.state.n_rounds += 1

    def update(self, arm: int, reward: float) -> None:
        """Single arm update (wraps update_subset)."""
        self.update_subset([arm], reward)

    def run(self, n_rounds: int) -> BanditResult:
        """Run CombLinUCB."""
        reward_history: List[float] = []
        diversity_history: List[float] = []
        regret_history: List[float] = []
        total_reward = 0.0
        cumulative_regret = 0.0
        optimal = diversity_reward(
            list(range(min(self.k, self.n_arms))), self.arms, self.kernel,
        )

        for t in range(n_rounds):
            subset = self.select_subset()
            reward = diversity_reward(subset, self.arms, self.kernel)
            noisy_reward = reward + self.rng.normal(0, 0.05)
            self.update_subset(subset, noisy_reward)

            total_reward += reward
            regret = optimal - reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)
            diversity_history.append(reward)
            reward_history.append(reward)

        return BanditResult(
            selected_arms=subset,
            final_selection=subset,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "CombLinUCB"},
        )


# ---------------------------------------------------------------------------
# EXP3 (Adversarial Bandits)
# ---------------------------------------------------------------------------

class EXP3Diversity(BanditAlgorithm):
    """EXP3 algorithm for adversarial diversity selection.

    Maintains a probability distribution over arms using exponential weights.
    Robust to adversarial reward sequences.

    Regret bound: O(sqrt(K * T * ln K))
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        gamma: float = 0.1,
        diversity_weight: float = 0.3,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.gamma = gamma
        self.diversity_weight = diversity_weight
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.weights = np.ones(n_arms)
        self.probabilities = np.ones(n_arms) / n_arms
        self.current_selection: List[int] = []

    def _update_probabilities(self) -> None:
        """Update sampling probabilities from weights."""
        w_sum = np.sum(self.weights)
        if w_sum <= 0:
            self.probabilities = np.ones(self.n_arms) / self.n_arms
        else:
            self.probabilities = (
                (1 - self.gamma) * self.weights / w_sum
                + self.gamma / self.n_arms
            )

    def select_arm(self) -> int:
        """Sample arm from EXP3 distribution."""
        self._update_probabilities()
        probs = self.probabilities.copy()
        # Zero out already selected
        for s in self.current_selection:
            probs[s] = 0.0
        p_sum = np.sum(probs)
        if p_sum <= 0:
            remaining = [i for i in range(self.n_arms) if i not in self.current_selection]
            if len(remaining) == 0:
                return 0
            return self.rng.choice(remaining)
        probs /= p_sum
        return int(self.rng.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float) -> None:
        """Update EXP3 weights."""
        self._update_probabilities()
        p = self.probabilities[arm]
        if p > 1e-12:
            # Importance-weighted estimator
            estimated_reward = reward / p
            self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.n_arms)
        self.state.pulls[arm] += 1
        self.state.cumulative_rewards[arm] += reward
        self.state.n_rounds += 1

    def run(
        self,
        n_rounds: int,
        k: int = 5,
    ) -> BanditResult:
        """Run EXP3."""
        reward_history: List[float] = []
        diversity_history: List[float] = []
        regret_history: List[float] = []
        total_reward = 0.0
        cumulative_regret = 0.0
        optimal = diversity_reward(
            list(range(min(k, self.n_arms))), self.arms, self.kernel,
        )

        for t in range(n_rounds):
            self.current_selection = []
            round_reward = 0.0

            for _ in range(min(k, self.n_arms)):
                arm = self.select_arm()
                reward = marginal_diversity_reward(
                    arm, self.current_selection, self.arms, self.kernel
                )
                self.update(arm, max(0, reward))
                self.current_selection.append(arm)
                round_reward += reward

            total_reward += round_reward
            div = diversity_reward(self.current_selection, self.arms, self.kernel)
            diversity_history.append(div)
            reward_history.append(round_reward)
            regret = optimal - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)

        return BanditResult(
            selected_arms=self.current_selection,
            final_selection=self.current_selection,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "EXP3_diversity"},
        )


# ---------------------------------------------------------------------------
# EXP3.P (with high probability guarantees)
# ---------------------------------------------------------------------------

class EXP3P(BanditAlgorithm):
    """EXP3.P: EXP3 with explicit exploration bonus.

    Provides high-probability regret bounds.
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        eta: float = 0.1,
        beta: float = 0.01,
        gamma: float = 0.05,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.eta = eta
        self.beta = beta
        self.gamma = gamma
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.log_weights = np.zeros(n_arms)
        self.current_selection: List[int] = []

    def _get_probabilities(self) -> np.ndarray:
        """Compute arm selection probabilities."""
        max_lw = np.max(self.log_weights)
        shifted = self.log_weights - max_lw
        w = np.exp(shifted)
        w_sum = np.sum(w)
        probs = (1 - self.gamma) * w / w_sum + self.gamma / self.n_arms
        return probs

    def select_arm(self) -> int:
        """Select arm from EXP3.P distribution."""
        probs = self._get_probabilities()
        for s in self.current_selection:
            probs[s] = 0.0
        p_sum = np.sum(probs)
        if p_sum <= 0:
            remaining = [i for i in range(self.n_arms) if i not in self.current_selection]
            return self.rng.choice(remaining) if remaining else 0
        probs /= p_sum
        return int(self.rng.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float) -> None:
        """Update EXP3.P weights."""
        probs = self._get_probabilities()
        p = max(probs[arm], 1e-12)
        estimated = reward / p
        # Exploration bonus
        bonus = self.beta / (p * self.n_arms)
        self.log_weights[arm] += self.eta * (estimated + bonus)
        self.state.pulls[arm] += 1
        self.state.n_rounds += 1

    def run(self, n_rounds: int, k: int = 5) -> BanditResult:
        """Run EXP3.P."""
        reward_history, diversity_history, regret_history = [], [], []
        total_reward = 0.0
        cumulative_regret = 0.0
        optimal = diversity_reward(
            list(range(min(k, self.n_arms))), self.arms, self.kernel,
        )

        for t in range(n_rounds):
            self.current_selection = []
            round_reward = 0.0
            for _ in range(min(k, self.n_arms)):
                arm = self.select_arm()
                reward = marginal_diversity_reward(
                    arm, self.current_selection, self.arms, self.kernel
                )
                self.update(arm, max(0, reward))
                self.current_selection.append(arm)
                round_reward += reward

            total_reward += round_reward
            div = diversity_reward(self.current_selection, self.arms, self.kernel)
            diversity_history.append(div)
            reward_history.append(round_reward)
            regret = optimal - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)

        return BanditResult(
            selected_arms=self.current_selection,
            final_selection=self.current_selection,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "EXP3P"},
        )


# ---------------------------------------------------------------------------
# Online Learning with Expert Advice
# ---------------------------------------------------------------------------

class ExpertAdviceDiversity:
    """Online diversity selection with expert advice (Hedge algorithm).

    Experts are different diversity selection strategies. The algorithm
    maintains weights over experts and follows the best one.
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        experts: List[Callable[[List[BanditArm], int], List[int]]],
        eta: float = 0.1,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        self.n_arms = n_arms
        self.arms = arms
        self.experts = experts
        self.n_experts = len(experts)
        self.eta = eta
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.weights = np.ones(self.n_experts)
        self.rng = np.random.RandomState(seed)

    def _expert_probabilities(self) -> np.ndarray:
        """Compute expert selection probabilities."""
        w_sum = np.sum(self.weights)
        return self.weights / w_sum

    def select_subset(self, k: int) -> Tuple[List[int], int]:
        """Select subset by following an expert.

        Returns (selected_arms, expert_index).
        """
        probs = self._expert_probabilities()
        expert_idx = int(self.rng.choice(self.n_experts, p=probs))
        selection = self.experts[expert_idx](self.arms, k)
        return selection, expert_idx

    def update_experts(self, expert_rewards: List[float]) -> None:
        """Update expert weights based on rewards."""
        for i, reward in enumerate(expert_rewards):
            self.weights[i] *= np.exp(self.eta * reward)
        # Normalize
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
            self.weights *= self.n_experts

    def run(self, n_rounds: int, k: int = 5) -> BanditResult:
        """Run expert advice algorithm."""
        reward_history, diversity_history, regret_history = [], [], []
        total_reward = 0.0
        cumulative_regret = 0.0

        for t in range(n_rounds):
            # Get all expert recommendations
            expert_selections = [
                expert(self.arms, k) for expert in self.experts
            ]
            # Compute rewards for each expert
            expert_rewards = [
                diversity_reward(sel, self.arms, self.kernel)
                for sel in expert_selections
            ]
            # Follow selected expert
            selection, expert_idx = self.select_subset(k)
            round_reward = diversity_reward(selection, self.arms, self.kernel)

            # Update weights
            self.update_experts(expert_rewards)

            total_reward += round_reward
            best_expert_reward = max(expert_rewards)
            regret = best_expert_reward - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)
            diversity_history.append(round_reward)
            reward_history.append(round_reward)

        return BanditResult(
            selected_arms=selection,
            final_selection=selection,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "expert_advice_hedge"},
        )


# ---------------------------------------------------------------------------
# Expert strategies (for ExpertAdviceDiversity)
# ---------------------------------------------------------------------------

def greedy_quality_expert(arms: List[BanditArm], k: int) -> List[int]:
    """Expert that selects highest quality arms."""
    qualities = [(i, arms[i].true_quality) for i in range(len(arms))]
    qualities.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in qualities[:k]]


def greedy_diversity_expert(
    kernel: Optional[Kernel] = None,
) -> Callable[[List[BanditArm], int], List[int]]:
    """Expert that greedily maximizes diversity."""
    if kernel is None:
        kernel = RBFKernel(bandwidth=1.0)

    def _expert(arms: List[BanditArm], k: int) -> List[int]:
        selected: List[int] = []
        for _ in range(min(k, len(arms))):
            best = -1
            best_gain = -float("inf")
            for i in range(len(arms)):
                if i in selected:
                    continue
                gain = marginal_diversity_reward(i, selected, arms, kernel)
                if gain > best_gain:
                    best_gain = gain
                    best = i
            if best >= 0:
                selected.append(best)
        return selected
    return _expert


def random_expert(seed: int = 0) -> Callable[[List[BanditArm], int], List[int]]:
    """Expert that selects randomly."""
    rng = np.random.RandomState(seed)
    def _expert(arms: List[BanditArm], k: int) -> List[int]:
        return rng.choice(len(arms), size=min(k, len(arms)), replace=False).tolist()
    return _expert


def max_dispersion_expert(arms: List[BanditArm], k: int) -> List[int]:
    """Expert that maximizes minimum pairwise distance."""
    if len(arms) <= k:
        return list(range(len(arms)))
    # Start with random arm
    rng = np.random.RandomState(42)
    selected = [rng.randint(len(arms))]
    for _ in range(k - 1):
        best = -1
        best_min_dist = -1.0
        for i in range(len(arms)):
            if i in selected:
                continue
            min_dist = min(
                np.linalg.norm(arms[i].embedding - arms[j].embedding)
                for j in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best = i
        if best >= 0:
            selected.append(best)
    return selected


# ---------------------------------------------------------------------------
# Regret bounds computation
# ---------------------------------------------------------------------------

class RegretAnalyzer:
    """Compute and verify theoretical regret bounds for bandit algorithms."""

    @staticmethod
    def ucb1_regret_bound(n_arms: int, T: int, delta_min: float = 0.01) -> float:
        """Upper bound on UCB1 regret.

        R(T) <= sum_{i: delta_i > 0} (8 ln T / delta_i) + (1 + pi^2/3) * sum delta_i
        """
        if delta_min <= 0:
            return float("inf")
        bound = n_arms * (8 * math.log(T) / delta_min + (1 + math.pi**2 / 3) * delta_min)
        return bound

    @staticmethod
    def thompson_bayesian_regret_bound(n_arms: int, T: int) -> float:
        """Bayesian regret bound for Thompson Sampling.

        BayesRegret(T) <= O(sqrt(K * T * ln T))
        """
        return math.sqrt(n_arms * T * math.log(T + 1))

    @staticmethod
    def exp3_regret_bound(n_arms: int, T: int) -> float:
        """Minimax regret bound for EXP3.

        R(T) <= 2 * sqrt(K * T * ln K)
        """
        return 2 * math.sqrt(n_arms * T * math.log(n_arms))

    @staticmethod
    def cucb_regret_bound(n_arms: int, k: int, T: int, delta_min: float = 0.01) -> float:
        """Regret bound for CUCB.

        R(T) <= O(k * sqrt(K * T * ln T / delta_min))
        """
        return k * math.sqrt(n_arms * T * math.log(T + 1) / delta_min)

    @staticmethod
    def verify_sublinear_regret(regret_history: List[float], T: int) -> Dict[str, float]:
        """Verify that empirical regret is sublinear."""
        if T <= 1:
            return {"is_sublinear": True, "growth_rate": 0.0}
        regret = regret_history[-1] if regret_history else 0.0
        avg_regret = regret / T
        # Fit log-log line to detect growth rate
        n = len(regret_history)
        if n < 10:
            return {"is_sublinear": avg_regret < 1.0, "growth_rate": avg_regret}
        log_t = np.log(np.arange(1, n + 1))
        log_r = np.log(np.maximum(regret_history, 1e-12))
        # Linear regression in log-log space
        A = np.vstack([log_t, np.ones(n)]).T
        slope, intercept = np.linalg.lstsq(A, log_r, rcond=None)[0]
        return {
            "is_sublinear": slope < 1.0 - 1e-6,
            "growth_rate": float(slope),
            "avg_regret": float(avg_regret),
            "total_regret": float(regret),
        }


# ---------------------------------------------------------------------------
# Bandit comparison framework
# ---------------------------------------------------------------------------

class BanditComparison:
    """Compare multiple bandit algorithms on diversity selection tasks."""

    def __init__(
        self,
        arms: List[BanditArm],
        k: int = 5,
        kernel: Optional[Kernel] = None,
    ):
        self.arms = arms
        self.k = k
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n_arms = len(arms)

    def run_comparison(
        self,
        n_rounds: int = 500,
        n_seeds: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Run all algorithms and compare."""
        results: Dict[str, List[BanditResult]] = {}

        for seed in range(n_seeds):
            # UCB1
            ucb = UCB1Diversity(
                self.n_arms, self.arms, kernel=self.kernel, seed=seed
            )
            r = ucb.run(n_rounds, self.k)
            results.setdefault("UCB1", []).append(r)

            # Thompson Sampling
            ts = ThompsonSamplingDiversity(
                self.n_arms, self.arms, kernel=self.kernel, seed=seed
            )
            r = ts.run(n_rounds, self.k)
            results.setdefault("Thompson", []).append(r)

            # CUCB
            cucb = CUCB(
                self.n_arms, self.arms, k=self.k, kernel=self.kernel, seed=seed
            )
            r = cucb.run(n_rounds)
            results.setdefault("CUCB", []).append(r)

            # EXP3
            exp3 = EXP3Diversity(
                self.n_arms, self.arms, kernel=self.kernel, seed=seed
            )
            r = exp3.run(n_rounds, self.k)
            results.setdefault("EXP3", []).append(r)

            # EXP3.P
            exp3p = EXP3P(
                self.n_arms, self.arms, kernel=self.kernel, seed=seed
            )
            r = exp3p.run(n_rounds, self.k)
            results.setdefault("EXP3P", []).append(r)

        # Aggregate
        summary: Dict[str, Dict[str, float]] = {}
        for name, runs in results.items():
            rewards = [r.total_reward for r in runs]
            regrets = [r.cumulative_regret for r in runs]
            final_divs = [r.diversity_history[-1] if r.diversity_history else 0 for r in runs]
            summary[name] = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_regret": float(np.mean(regrets)),
                "std_regret": float(np.std(regrets)),
                "mean_final_diversity": float(np.mean(final_divs)),
            }
        return summary


# ---------------------------------------------------------------------------
# Sliding Window UCB
# ---------------------------------------------------------------------------

class SlidingWindowUCB(BanditAlgorithm):
    """Sliding window UCB for non-stationary diversity rewards.

    Only uses observations from the last W rounds for mean estimation.
    Adapts to changing diversity landscapes.
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        window_size: int = 100,
        exploration_constant: float = 1.0,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.window = window_size
        self.c = exploration_constant
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.history: List[Tuple[int, float]] = []
        self.current_selection: List[int] = []

    def select_arm(self) -> int:
        t = len(self.history) + 1
        # Recent observations within window
        recent = self.history[-self.window:] if len(self.history) > self.window else self.history
        counts = np.zeros(self.n_arms)
        sums = np.zeros(self.n_arms)
        for arm, reward in recent:
            counts[arm] += 1
            sums[arm] += reward

        ucb = np.full(self.n_arms, float("inf"))
        for i in range(self.n_arms):
            if i in self.current_selection:
                ucb[i] = -float("inf")
                continue
            if counts[i] > 0:
                mean = sums[i] / counts[i]
                bonus = self.c * math.sqrt(2 * math.log(t) / counts[i])
                ucb[i] = mean + bonus
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.history.append((arm, reward))
        self.state.n_rounds += 1

    def run(self, n_rounds: int, k: int = 5) -> BanditResult:
        reward_history, diversity_history, regret_history = [], [], []
        total_reward = 0.0
        cumulative_regret = 0.0
        optimal = diversity_reward(
            list(range(min(k, self.n_arms))), self.arms, self.kernel
        )

        for t in range(n_rounds):
            self.current_selection = []
            round_reward = 0.0
            for _ in range(min(k, self.n_arms)):
                arm = self.select_arm()
                reward = marginal_diversity_reward(arm, self.current_selection, self.arms, self.kernel)
                self.update(arm, reward + self.rng.normal(0, 0.1))
                self.current_selection.append(arm)
                round_reward += reward

            total_reward += round_reward
            regret = optimal - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)
            diversity_history.append(diversity_reward(self.current_selection, self.arms, self.kernel))
            reward_history.append(round_reward)

        return BanditResult(
            selected_arms=self.current_selection,
            final_selection=self.current_selection,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "sliding_window_ucb"},
        )


# ---------------------------------------------------------------------------
# Successive Elimination
# ---------------------------------------------------------------------------

class SuccessiveElimination(BanditAlgorithm):
    """Successive elimination algorithm for best-arm identification.

    Progressively eliminates sub-optimal arms and returns the top-k.
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        delta: float = 0.1,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.delta = delta
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.active_arms: List[int] = list(range(n_arms))
        self._arm_rewards: Dict[int, List[float]] = {i: [] for i in range(n_arms)}

    def select_arm(self) -> int:
        if not self.active_arms:
            return 0
        return self.active_arms[self.state.n_rounds % len(self.active_arms)]

    def update(self, arm: int, reward: float) -> None:
        self._arm_rewards[arm].append(reward)
        self.state.n_rounds += 1

    def _eliminate(self) -> None:
        """Eliminate arms with confidence."""
        if len(self.active_arms) <= 1:
            return
        means = {}
        for arm in self.active_arms:
            rewards = self._arm_rewards[arm]
            if len(rewards) > 0:
                means[arm] = np.mean(rewards)
            else:
                means[arm] = 0.0

        best_mean = max(means.values())
        new_active = []
        for arm in self.active_arms:
            n_pulls = len(self._arm_rewards[arm])
            if n_pulls == 0:
                new_active.append(arm)
                continue
            ci_width = math.sqrt(math.log(2 * self.n_arms * n_pulls**2 / self.delta) / (2 * n_pulls))
            if means[arm] + ci_width >= best_mean - ci_width:
                new_active.append(arm)
        self.active_arms = new_active

    def run(self, n_rounds: int, k: int = 5) -> BanditResult:
        reward_history, diversity_history, regret_history = [], [], []
        total_reward = 0.0
        cumulative_regret = 0.0
        optimal = diversity_reward(
            list(range(min(k, self.n_arms))), self.arms, self.kernel
        )

        for t in range(n_rounds):
            arm = self.select_arm()
            reward = self.arms[arm].true_quality + self.rng.normal(0, 0.1)
            self.update(arm, reward)

            if t % 10 == 0:
                self._eliminate()

            # Current best k from active arms
            active_means = {
                a: np.mean(self._arm_rewards[a]) if self._arm_rewards[a] else 0
                for a in self.active_arms
            }
            top_k = sorted(active_means, key=lambda a: active_means[a], reverse=True)[:k]
            div = diversity_reward(top_k, self.arms, self.kernel)
            diversity_history.append(div)
            reward_history.append(reward)
            cumulative_regret += max(optimal - div, 0)
            regret_history.append(cumulative_regret)
            total_reward += reward

        final_means = {
            a: np.mean(self._arm_rewards[a]) if self._arm_rewards[a] else 0
            for a in self.active_arms
        }
        final_k = sorted(final_means, key=lambda a: final_means[a], reverse=True)[:k]

        return BanditResult(
            selected_arms=final_k,
            final_selection=final_k,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "successive_elimination", "active_arms": len(self.active_arms)},
        )


# ---------------------------------------------------------------------------
# Epsilon-Greedy with Diversity
# ---------------------------------------------------------------------------

class EpsilonGreedyDiversity(BanditAlgorithm):
    """Epsilon-greedy with decaying exploration and diversity reward."""

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        epsilon: float = 0.1,
        decay: float = 0.999,
        diversity_weight: float = 0.5,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        super().__init__(n_arms, seed)
        self.arms = arms
        self.epsilon = epsilon
        self.decay = decay
        self.diversity_weight = diversity_weight
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self._means = np.zeros(n_arms)
        self.current_selection: List[int] = []

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            remaining = [i for i in range(self.n_arms) if i not in self.current_selection]
            return self.rng.choice(remaining) if remaining else 0
        scores = self._means.copy()
        for i in range(self.n_arms):
            if i in self.current_selection:
                scores[i] = -float("inf")
                continue
            if self.diversity_weight > 0 and len(self.current_selection) > 0:
                emb = self.arms[i].embedding
                min_d = min(np.linalg.norm(emb - self.arms[j].embedding) for j in self.current_selection)
                scores[i] += self.diversity_weight * min_d
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float) -> None:
        self.state.pulls[arm] += 1
        self.state.cumulative_rewards[arm] += reward
        self._means[arm] = self.state.cumulative_rewards[arm] / self.state.pulls[arm]
        self.state.n_rounds += 1
        self.epsilon *= self.decay

    def run(self, n_rounds: int, k: int = 5) -> BanditResult:
        reward_history, diversity_history, regret_history = [], [], []
        total_reward = 0.0
        cumulative_regret = 0.0
        optimal = diversity_reward(
            list(range(min(k, self.n_arms))), self.arms, self.kernel
        )

        for t in range(n_rounds):
            self.current_selection = []
            round_reward = 0.0
            for _ in range(min(k, self.n_arms)):
                arm = self.select_arm()
                reward = marginal_diversity_reward(arm, self.current_selection, self.arms, self.kernel)
                self.update(arm, reward + self.rng.normal(0, 0.1))
                self.current_selection.append(arm)
                round_reward += reward

            total_reward += round_reward
            div = diversity_reward(self.current_selection, self.arms, self.kernel)
            diversity_history.append(div)
            reward_history.append(round_reward)
            regret = optimal - round_reward
            cumulative_regret += max(regret, 0)
            regret_history.append(cumulative_regret)

        return BanditResult(
            selected_arms=self.current_selection,
            final_selection=self.current_selection,
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            regret_history=regret_history,
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_rounds,
            metadata={"algorithm": "epsilon_greedy_diversity"},
        )


# ---------------------------------------------------------------------------
# Bayesian Optimization for Diversity
# ---------------------------------------------------------------------------

class BayesianOptimizationDiversity:
    """Bayesian optimization approach to diversity selection.

    Uses GP surrogate with expected improvement acquisition function.
    """

    def __init__(
        self,
        n_arms: int,
        arms: List[BanditArm],
        k: int = 5,
        kernel: Optional[Kernel] = None,
        seed: int = 42,
    ):
        self.n_arms = n_arms
        self.arms = arms
        self.k = k
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.rng = np.random.RandomState(seed)
        # GP observations
        self.X_obs: List[np.ndarray] = []
        self.y_obs: List[float] = []

    def _subset_to_features(self, subset: List[int]) -> np.ndarray:
        """Convert subset to feature vector."""
        embs = np.array([self.arms[i].embedding for i in subset])
        mean_emb = np.mean(embs, axis=0)
        # Add diversity features
        if len(subset) >= 2:
            dists = []
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    dists.append(np.linalg.norm(embs[i] - embs[j]))
            features = np.concatenate([mean_emb, [np.mean(dists), np.std(dists)]])
        else:
            features = np.concatenate([mean_emb, [0.0, 0.0]])
        return features

    def _gp_predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Simple GP prediction using kernel regression."""
        if len(self.X_obs) == 0:
            return 0.0, 1.0
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        n = len(y)

        # Kernel matrix with noise
        K_noise = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K_noise[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / 2)
        K_noise += 0.01 * np.eye(n)

        # Cross-kernel
        k_star = np.array([np.exp(-np.linalg.norm(x - X[i])**2 / 2) for i in range(n)])

        try:
            alpha = np.linalg.solve(K_noise, y)
            mu = float(k_star @ alpha)
            v = np.linalg.solve(K_noise, k_star)
            var = max(1.0 - k_star @ v, 1e-6)
            return mu, math.sqrt(var)
        except np.linalg.LinAlgError:
            return float(np.mean(y)), 1.0

    def _expected_improvement(self, x: np.ndarray, best_y: float) -> float:
        """Expected improvement acquisition function."""
        mu, sigma = self._gp_predict(x)
        if sigma < 1e-12:
            return 0.0
        z = (mu - best_y) / sigma
        # Approximate Phi(z) and phi(z)
        phi = math.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        return sigma * (z * Phi + phi)

    def run(self, n_iterations: int = 100) -> BanditResult:
        """Run Bayesian optimization."""
        reward_history, diversity_history = [], []

        # Initial random evaluations
        for _ in range(min(10, n_iterations)):
            subset = sorted(self.rng.choice(self.n_arms, self.k, replace=False).tolist())
            features = self._subset_to_features(subset)
            reward = diversity_reward(subset, self.arms, self.kernel)
            self.X_obs.append(features)
            self.y_obs.append(reward)
            reward_history.append(reward)
            diversity_history.append(reward)

        best_y = max(self.y_obs)
        best_subset = []

        # BO iterations
        for it in range(10, n_iterations):
            best_ei = -float("inf")
            best_candidate = None

            for _ in range(50):
                candidate = sorted(self.rng.choice(self.n_arms, self.k, replace=False).tolist())
                features = self._subset_to_features(candidate)
                ei = self._expected_improvement(features, best_y)
                if ei > best_ei:
                    best_ei = ei
                    best_candidate = candidate

            if best_candidate is None:
                best_candidate = sorted(self.rng.choice(self.n_arms, self.k, replace=False).tolist())

            features = self._subset_to_features(best_candidate)
            reward = diversity_reward(best_candidate, self.arms, self.kernel)
            self.X_obs.append(features)
            self.y_obs.append(reward)
            reward_history.append(reward)
            diversity_history.append(reward)

            if reward > best_y:
                best_y = reward
                best_subset = best_candidate

        return BanditResult(
            selected_arms=best_subset,
            final_selection=best_subset,
            total_reward=sum(reward_history),
            cumulative_regret=0.0,
            regret_history=[],
            diversity_history=diversity_history,
            reward_history=reward_history,
            n_rounds=n_iterations,
            metadata={"algorithm": "bayesian_optimization"},
        )
