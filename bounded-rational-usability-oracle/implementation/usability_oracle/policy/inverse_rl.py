"""
usability_oracle.policy.inverse_rl — Inverse reinforcement learning.

Recovers the implicit reward (or cost) function that rationalises observed
user behaviour, accounting for bounded rationality.  The key insight is that
a bounded-rational user with rationality β acts as if optimising:

    π*(a|s) ∝ p₀(a|s) · exp(β · R(s, a))

so inverse RL reduces to finding R(s, a) such that the induced policy matches
observations.

Methods
-------
- **Maximum-entropy IRL** — finds the reward function that maximises the
  likelihood of observed trajectories under a MaxEnt model (Ziebart et al.).
- **Feature-matching IRL** — matches expected feature counts between the
  learned policy and demonstrations (Abbeel & Ng).
- **Bayesian IRL** — computes a posterior distribution over reward functions
  given observations (Ramachandran & Amir).
- **Bounded-rational IRL** — extends MaxEnt IRL to account for suboptimal
  users with finite β.
- **Confidence bounds** — posterior credible intervals on the recovered
  reward parameters.

References
----------
- Ziebart, B. D. et al. (2008). Maximum entropy inverse reinforcement
  learning. *AAAI*.
- Abbeel, P. & Ng, A. Y. (2004). Apprenticeship learning via inverse
  reinforcement learning. *ICML*.
- Ramachandran, D. & Amir, E. (2007). Bayesian inverse reinforcement
  learning. *IJCAI*.
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import optimize as sp_opt  # type: ignore[import-untyped]

from usability_oracle.policy.models import Policy, QValues

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demonstration data
# ---------------------------------------------------------------------------

@dataclass
class Demonstration:
    """An observed user trajectory for IRL.

    Attributes
    ----------
    states : list[str]
    actions : list[str]
    features : list[np.ndarray]
        Per-step feature vectors φ(s_t, a_t) ∈ ℝ^d.
    """

    states: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.states)


@dataclass
class IRLResult:
    """Result of an inverse reinforcement learning procedure.

    Attributes
    ----------
    reward_weights : np.ndarray
        Learned reward parameter vector θ such that R(s,a) = θ^T φ(s,a).
    reward_map : dict[str, dict[str, float]]
        Recovered reward function R(s, a).
    policy : Policy
        Policy induced by the recovered reward.
    log_likelihood : float
    convergence_info : dict
    confidence_intervals : dict[int, tuple[float, float]]
        Per-feature-dimension 95% credible intervals.
    """

    reward_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    reward_map: dict[str, dict[str, float]] = field(default_factory=dict)
    policy: Policy = field(default_factory=Policy)
    log_likelihood: float = 0.0
    convergence_info: dict[str, Any] = field(default_factory=dict)
    confidence_intervals: dict[int, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def compute_feature_expectations(
    demos: list[Demonstration],
    discount: float = 0.99,
) -> np.ndarray:
    """Compute empirical feature expectations from demonstrations.

    μ̂ = (1/N) Σ_n Σ_t γ^t φ(s_t, a_t)

    Parameters
    ----------
    demos : list[Demonstration]
    discount : float

    Returns
    -------
    np.ndarray
        Feature expectation vector ∈ ℝ^d.
    """
    if not demos or not demos[0].features:
        return np.array([])

    d = len(demos[0].features[0])
    total = np.zeros(d, dtype=np.float64)

    for demo in demos:
        gamma_t = 1.0
        for feat in demo.features:
            total += gamma_t * feat
            gamma_t *= discount

    return total / max(len(demos), 1)


def compute_state_action_features(
    states: list[str],
    actions_per_state: dict[str, list[str]],
    feature_fn: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Organise features into a state-action lookup.

    Parameters
    ----------
    states : list[str]
    actions_per_state : dict[str, list[str]]
    feature_fn : dict[str, dict[str, np.ndarray]]
        ``feature_fn[s][a]`` = φ(s, a).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
    """
    result: dict[str, dict[str, np.ndarray]] = {}
    for s in states:
        result[s] = {}
        for a in actions_per_state.get(s, []):
            result[s][a] = feature_fn.get(s, {}).get(a, np.zeros(1))
    return result


# ---------------------------------------------------------------------------
# Maximum Entropy IRL
# ---------------------------------------------------------------------------

class MaxEntropyIRL:
    """Maximum-entropy inverse reinforcement learning.

    Finds the reward weights θ that maximise the likelihood of
    demonstrations under the soft (Boltzmann) policy:

        max_θ  Σ_n Σ_t [θ^T φ(s_t, a_t) − log Z_θ(s_t)]

    with an optional L2 regulariser on θ.

    Parameters
    ----------
    feature_dim : int
    beta : float
        Rationality parameter (inverse temperature).
    learning_rate : float
    regularisation : float
        L2 penalty weight on θ.
    max_iter : int
    """

    def __init__(
        self,
        feature_dim: int,
        beta: float = 1.0,
        learning_rate: float = 0.01,
        regularisation: float = 0.01,
        max_iter: int = 200,
    ) -> None:
        self.d = feature_dim
        self.beta = beta
        self.lr = learning_rate
        self.reg = regularisation
        self.max_iter = max_iter

    def fit(
        self,
        demos: list[Demonstration],
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
        transition_fn: Optional[dict[str, dict[str, list[tuple[str, float]]]]] = None,
        discount: float = 0.99,
    ) -> IRLResult:
        """Fit reward weights to demonstrations.

        Parameters
        ----------
        demos : list[Demonstration]
        states : list[str]
        actions_per_state : dict[str, list[str]]
        feature_fn : dict[str, dict[str, np.ndarray]]
        transition_fn : dict, optional
            If provided, used for forward RL to compute the induced policy.
        discount : float

        Returns
        -------
        IRLResult
        """
        # Empirical feature expectations
        mu_expert = compute_feature_expectations(demos, discount)

        # Initialise weights
        theta = np.zeros(self.d, dtype=np.float64)

        for iteration in range(self.max_iter):
            # Compute reward function
            rewards = self._reward_from_weights(theta, states, actions_per_state, feature_fn)

            # Compute softmax policy under current reward
            policy = self._softmax_policy(rewards, states, actions_per_state)

            # Compute expected features under current policy
            mu_policy = self._policy_feature_expectations(
                policy, states, actions_per_state, feature_fn
            )

            # Gradient: ∇_θ L = μ_expert − μ_policy − reg·θ
            gradient = mu_expert - mu_policy - self.reg * theta

            # Update
            theta += self.lr * gradient

            grad_norm = float(np.linalg.norm(gradient))
            if grad_norm < 1e-6:
                logger.debug("MaxEnt IRL converged at iteration %d", iteration + 1)
                break

        # Final outputs
        rewards = self._reward_from_weights(theta, states, actions_per_state, feature_fn)
        policy = self._softmax_policy(rewards, states, actions_per_state)
        ll = self._log_likelihood(theta, demos, states, actions_per_state, feature_fn)

        return IRLResult(
            reward_weights=theta,
            reward_map=rewards,
            policy=policy,
            log_likelihood=ll,
            convergence_info={"iterations": iteration + 1, "grad_norm": grad_norm},
        )

    def _reward_from_weights(
        self,
        theta: np.ndarray,
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, float]]:
        """R(s, a) = θ^T φ(s, a)."""
        rewards: dict[str, dict[str, float]] = {}
        for s in states:
            rewards[s] = {}
            for a in actions_per_state.get(s, []):
                phi = feature_fn.get(s, {}).get(a, np.zeros(self.d))
                rewards[s][a] = float(np.dot(theta, phi))
        return rewards

    def _softmax_policy(
        self,
        rewards: dict[str, dict[str, float]],
        states: list[str],
        actions_per_state: dict[str, list[str]],
    ) -> Policy:
        """Compute π(a|s) ∝ exp(β · R(s,a))."""
        state_action_probs: dict[str, dict[str, float]] = {}
        for s in states:
            actions = actions_per_state.get(s, [])
            if not actions:
                continue
            r_vals = np.array(
                [rewards.get(s, {}).get(a, 0.0) for a in actions],
                dtype=np.float64,
            )
            logits = self.beta * r_vals
            logits -= np.max(logits)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum()
            state_action_probs[s] = {
                actions[i]: float(probs[i]) for i in range(len(actions))
            }
        return Policy(state_action_probs=state_action_probs, beta=self.beta)

    def _policy_feature_expectations(
        self,
        policy: Policy,
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Compute E_π[φ(s,a)] averaged over states."""
        total = np.zeros(self.d, dtype=np.float64)
        n = 0
        for s in states:
            dist = policy.state_action_probs.get(s, {})
            for a, pi_a in dist.items():
                phi = feature_fn.get(s, {}).get(a, np.zeros(self.d))
                total += pi_a * phi
            if dist:
                n += 1
        return total / max(n, 1)

    def _log_likelihood(
        self,
        theta: np.ndarray,
        demos: list[Demonstration],
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
    ) -> float:
        """Log-likelihood of demonstrations under the current model."""
        rewards = self._reward_from_weights(theta, states, actions_per_state, feature_fn)
        ll = 0.0
        for demo in demos:
            for t in range(demo.length):
                s, a = demo.states[t], demo.actions[t]
                actions = actions_per_state.get(s, [])
                if not actions:
                    continue
                r_a = rewards.get(s, {}).get(a, 0.0)
                r_all = np.array(
                    [rewards.get(s, {}).get(ac, 0.0) for ac in actions],
                    dtype=np.float64,
                )
                logits = self.beta * r_all
                log_z = float(np.max(logits)) + float(
                    np.log(np.sum(np.exp(logits - np.max(logits))))
                )
                ll += self.beta * r_a - log_z
        return ll


# ---------------------------------------------------------------------------
# Feature-Matching IRL
# ---------------------------------------------------------------------------

class FeatureMatchingIRL:
    """IRL via feature expectation matching (Abbeel & Ng, 2004).

    Iteratively finds a reward weight vector θ such that the induced
    policy's feature expectations match the expert's.

    Parameters
    ----------
    feature_dim : int
    beta : float
    max_iter : int
    tolerance : float
    """

    def __init__(
        self,
        feature_dim: int,
        beta: float = 1.0,
        max_iter: int = 50,
        tolerance: float = 0.01,
    ) -> None:
        self.d = feature_dim
        self.beta = beta
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(
        self,
        demos: list[Demonstration],
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
        discount: float = 0.99,
    ) -> IRLResult:
        """Fit reward via feature matching.

        Returns
        -------
        IRLResult
        """
        mu_expert = compute_feature_expectations(demos, discount)
        theta = np.zeros(self.d, dtype=np.float64)

        # Projection algorithm
        mu_bar: Optional[np.ndarray] = None

        for iteration in range(self.max_iter):
            rewards = {
                s: {
                    a: float(np.dot(theta, feature_fn.get(s, {}).get(a, np.zeros(self.d))))
                    for a in actions_per_state.get(s, [])
                }
                for s in states
            }

            # Softmax policy
            policy = MaxEntropyIRL(self.d, self.beta)._softmax_policy(
                rewards, states, actions_per_state
            )

            mu_pi = MaxEntropyIRL(self.d, self.beta)._policy_feature_expectations(
                policy, states, actions_per_state, feature_fn
            )

            if mu_bar is None:
                mu_bar = mu_pi.copy()
            else:
                # Project mu_bar toward mu_expert
                diff = mu_pi - mu_bar
                dot = np.dot(mu_expert - mu_bar, diff)
                denom = np.dot(diff, diff)
                if denom > 1e-10:
                    t = dot / denom
                    t = np.clip(t, 0, 1)
                    mu_bar = mu_bar + t * diff
                else:
                    mu_bar = mu_pi

            # Update theta toward expert
            theta = mu_expert - mu_bar

            gap = float(np.linalg.norm(mu_expert - mu_bar))
            if gap < self.tolerance:
                logger.debug("Feature matching converged at iteration %d", iteration + 1)
                break

        rewards_final = {
            s: {
                a: float(np.dot(theta, feature_fn.get(s, {}).get(a, np.zeros(self.d))))
                for a in actions_per_state.get(s, [])
            }
            for s in states
        }
        policy = MaxEntropyIRL(self.d, self.beta)._softmax_policy(
            rewards_final, states, actions_per_state
        )

        return IRLResult(
            reward_weights=theta,
            reward_map=rewards_final,
            policy=policy,
            convergence_info={"iterations": iteration + 1, "gap": gap},
        )


# ---------------------------------------------------------------------------
# Bayesian IRL
# ---------------------------------------------------------------------------

class BayesianIRL:
    """Bayesian inverse reinforcement learning.

    Places a Gaussian prior on the reward weights and computes a posterior
    via MCMC (Metropolis-Hastings):

        P(θ | D) ∝ P(D | θ) · P(θ)

    where P(D | θ) = Π_n Π_t π_θ(a_t | s_t) is the likelihood under
    the soft policy.

    Parameters
    ----------
    feature_dim : int
    beta : float
    prior_mean : np.ndarray or None
    prior_precision : float
    n_samples : int
    proposal_std : float
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        feature_dim: int,
        beta: float = 1.0,
        prior_mean: Optional[np.ndarray] = None,
        prior_precision: float = 1.0,
        n_samples: int = 1000,
        proposal_std: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.d = feature_dim
        self.beta = beta
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(feature_dim)
        self.prior_precision = prior_precision
        self.n_samples = n_samples
        self.proposal_std = proposal_std
        self.rng = rng or np.random.default_rng()

    def fit(
        self,
        demos: list[Demonstration],
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
    ) -> IRLResult:
        """Run MCMC to sample from the posterior over reward weights.

        Returns
        -------
        IRLResult
            ``reward_weights`` is the posterior mean; ``confidence_intervals``
            contains per-dimension 95% credible intervals.
        """
        irl_helper = MaxEntropyIRL(self.d, self.beta)

        def log_posterior(theta: np.ndarray) -> float:
            # Log-likelihood
            ll = irl_helper._log_likelihood(
                theta, demos, states, actions_per_state, feature_fn
            )
            # Log-prior: N(prior_mean, (1/prior_precision)·I)
            diff = theta - self.prior_mean
            lp = -0.5 * self.prior_precision * float(np.dot(diff, diff))
            return ll + lp

        # Metropolis-Hastings
        samples: list[np.ndarray] = []
        theta = self.prior_mean.copy()
        log_p = log_posterior(theta)

        n_accept = 0
        for i in range(self.n_samples):
            proposal = theta + self.rng.normal(0, self.proposal_std, size=self.d)
            log_p_prop = log_posterior(proposal)

            log_alpha = log_p_prop - log_p
            if math.log(max(self.rng.random(), 1e-300)) < log_alpha:
                theta = proposal
                log_p = log_p_prop
                n_accept += 1

            samples.append(theta.copy())

        # Burn-in: discard first 20%
        burn = max(self.n_samples // 5, 1)
        posterior_samples = np.array(samples[burn:], dtype=np.float64)

        # Posterior statistics
        posterior_mean = np.mean(posterior_samples, axis=0)
        posterior_std = np.std(posterior_samples, axis=0, ddof=1)

        # 95% credible intervals
        ci: dict[int, tuple[float, float]] = {}
        for dim in range(self.d):
            low = float(np.percentile(posterior_samples[:, dim], 2.5))
            high = float(np.percentile(posterior_samples[:, dim], 97.5))
            ci[dim] = (low, high)

        # Reward map and policy from posterior mean
        rewards = irl_helper._reward_from_weights(
            posterior_mean, states, actions_per_state, feature_fn
        )
        policy = irl_helper._softmax_policy(rewards, states, actions_per_state)

        return IRLResult(
            reward_weights=posterior_mean,
            reward_map=rewards,
            policy=policy,
            log_likelihood=float(log_posterior(posterior_mean)),
            convergence_info={
                "n_samples": self.n_samples,
                "acceptance_rate": n_accept / self.n_samples,
                "posterior_std": posterior_std.tolist(),
            },
            confidence_intervals=ci,
        )


# ---------------------------------------------------------------------------
# Bounded-Rational IRL
# ---------------------------------------------------------------------------

class BoundedRationalIRL:
    """Inverse RL accounting for bounded-rational users.

    Jointly infers the reward function R and the rationality parameter β
    from observed trajectories.  The model is:

        π*(a|s) ∝ p₀(a|s) · exp(β · R(s,a))

    We maximise the joint log-likelihood over (θ, β) using gradient
    ascent, where R(s,a) = θ^T φ(s,a).

    Parameters
    ----------
    feature_dim : int
    initial_beta : float
    learning_rate : float
    max_iter : int
    regularisation : float
    """

    def __init__(
        self,
        feature_dim: int,
        initial_beta: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 300,
        regularisation: float = 0.01,
    ) -> None:
        self.d = feature_dim
        self.initial_beta = initial_beta
        self.lr = learning_rate
        self.max_iter = max_iter
        self.reg = regularisation

    def fit(
        self,
        demos: list[Demonstration],
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
        prior: Optional[Policy] = None,
        discount: float = 0.99,
    ) -> tuple[IRLResult, float]:
        """Jointly learn reward weights and rationality parameter.

        Parameters
        ----------
        demos : list[Demonstration]
        states : list[str]
        actions_per_state : dict[str, list[str]]
        feature_fn : dict[str, dict[str, np.ndarray]]
        prior : Policy, optional
        discount : float

        Returns
        -------
        tuple[IRLResult, float]
            (IRL result, estimated β).
        """
        theta = np.zeros(self.d, dtype=np.float64)
        beta = self.initial_beta
        mu_expert = compute_feature_expectations(demos, discount)

        for iteration in range(self.max_iter):
            # Reward function
            rewards = self._reward_from_weights(theta, states, actions_per_state, feature_fn)

            # Softmax policy with prior
            policy = self._softmax_policy_with_prior(
                rewards, states, actions_per_state, beta, prior
            )

            # Feature expectations under policy
            mu_policy = self._policy_features(
                policy, states, actions_per_state, feature_fn
            )

            # Gradient w.r.t. theta
            grad_theta = beta * (mu_expert - mu_policy) - self.reg * theta

            # Gradient w.r.t. beta (approximate)
            # ∂L/∂β ≈ Σ_n Σ_t [R(s_t,a_t) − E_π[R(s,a)]]
            grad_beta = 0.0
            n_steps = 0
            for demo in demos:
                for t in range(demo.length):
                    s, a = demo.states[t], demo.actions[t]
                    r_a = rewards.get(s, {}).get(a, 0.0)
                    r_expected = sum(
                        pi_a * rewards.get(s, {}).get(ac, 0.0)
                        for ac, pi_a in policy.state_action_probs.get(s, {}).items()
                    )
                    grad_beta += r_a - r_expected
                    n_steps += 1

            if n_steps > 0:
                grad_beta /= n_steps

            theta += self.lr * grad_theta
            beta = max(beta + self.lr * grad_beta, 0.01)

            grad_norm = float(np.linalg.norm(grad_theta))
            if grad_norm < 1e-6 and abs(grad_beta) < 1e-6:
                logger.debug("BR-IRL converged at iteration %d", iteration + 1)
                break

        rewards = self._reward_from_weights(theta, states, actions_per_state, feature_fn)
        policy = self._softmax_policy_with_prior(
            rewards, states, actions_per_state, beta, prior
        )

        result = IRLResult(
            reward_weights=theta,
            reward_map=rewards,
            policy=policy,
            convergence_info={
                "iterations": iteration + 1,
                "estimated_beta": beta,
                "grad_norm": grad_norm,
            },
        )
        return result, beta

    def _reward_from_weights(
        self,
        theta: np.ndarray,
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, float]]:
        rewards: dict[str, dict[str, float]] = {}
        for s in states:
            rewards[s] = {}
            for a in actions_per_state.get(s, []):
                phi = feature_fn.get(s, {}).get(a, np.zeros(self.d))
                rewards[s][a] = float(np.dot(theta, phi))
        return rewards

    def _softmax_policy_with_prior(
        self,
        rewards: dict[str, dict[str, float]],
        states: list[str],
        actions_per_state: dict[str, list[str]],
        beta: float,
        prior: Optional[Policy],
    ) -> Policy:
        """Compute π(a|s) ∝ p₀(a|s) · exp(β · R(s,a))."""
        state_action_probs: dict[str, dict[str, float]] = {}
        for s in states:
            actions = actions_per_state.get(s, [])
            if not actions:
                continue
            n = len(actions)
            r_vals = np.array(
                [rewards.get(s, {}).get(a, 0.0) for a in actions],
                dtype=np.float64,
            )

            if prior is not None:
                p0_dist = prior.state_action_probs.get(s, {})
                p0 = np.array(
                    [p0_dist.get(a, 1.0 / n) for a in actions],
                    dtype=np.float64,
                )
            else:
                p0 = np.ones(n, dtype=np.float64) / n

            p0 = np.maximum(p0, 1e-300)
            p0 /= p0.sum()

            logits = np.log(p0) + beta * r_vals
            logits -= np.max(logits)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum()

            state_action_probs[s] = {
                actions[i]: float(probs[i]) for i in range(n)
            }
        return Policy(state_action_probs=state_action_probs, beta=beta)

    def _policy_features(
        self,
        policy: Policy,
        states: list[str],
        actions_per_state: dict[str, list[str]],
        feature_fn: dict[str, dict[str, np.ndarray]],
    ) -> np.ndarray:
        total = np.zeros(self.d, dtype=np.float64)
        n = 0
        for s in states:
            dist = policy.state_action_probs.get(s, {})
            for a, pi_a in dist.items():
                phi = feature_fn.get(s, {}).get(a, np.zeros(self.d))
                total += pi_a * phi
            if dist:
                n += 1
        return total / max(n, 1)
