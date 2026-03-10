"""
usability_oracle.policy.thompson — Thompson sampling for exploration.

Implements Thompson sampling variants for bounded-rational exploration in UI
evaluation.  Models the user's exploration of an unfamiliar interface as a
multi-armed bandit problem where each action is an "arm" whose reward
distribution is unknown.

Variants
--------
- **Beta-Bernoulli** — for binary success/failure outcomes.
- **Gaussian** — for continuous reward / cost observations.
- **Bounded-rational Thompson sampling** — exploration limited by the
  rationality parameter β (temperature-scaled posterior sampling).
- **Information-directed sampling** — selects actions maximising the ratio
  of expected regret reduction to information gain.
- **Knowledge gradient** — selects the action with the greatest expected
  improvement in the optimal decision after one observation.
- **Bayesian optimisation** — Gaussian-process-based tuning of continuous
  UI parameters (font size, spacing, contrast, etc.).

References
----------
- Thompson, W. R. (1933). On the likelihood that one unknown probability
  exceeds another. *Biometrika*, 25, 285–294.
- Russo, D. & Van Roy, B. (2014). Learning to optimize via
  information-directed sampling. *NeurIPS*.
- Frazier, P. I., Powell, W. B. & Dayanik, S. (2009). The knowledge-gradient
  policy for correlated normal beliefs. *INFORMS J. Comput.*.
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats as sp_stats  # type: ignore[import-untyped]

from usability_oracle.policy.models import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beta-Bernoulli Thompson Sampling
# ---------------------------------------------------------------------------

@dataclass
class BetaBernoulliArm:
    """Posterior state for a single Beta-Bernoulli arm.

    Attributes
    ----------
    alpha : float
        Pseudo-count of successes (prior + observed).
    beta_param : float
        Pseudo-count of failures (prior + observed).
    """

    alpha: float = 1.0
    beta_param: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta_param
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.alpha, self.beta_param))

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta_param += 1.0


class BetaBernoulliThompson:
    """Thompson sampling with Beta-Bernoulli arms.

    Each action is modelled as a Bernoulli arm with unknown success
    probability.  The posterior is Beta(α, β) initialised with a uniform
    prior Beta(1, 1).

    Parameters
    ----------
    actions : list[str]
        Action identifiers.
    prior_alpha : float
    prior_beta : float
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        actions: list[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.arms: dict[str, BetaBernoulliArm] = {
            a: BetaBernoulliArm(alpha=prior_alpha, beta_param=prior_beta)
            for a in actions
        }

    def select_action(self) -> str:
        """Sample from each arm's posterior and select the best."""
        best_action = ""
        best_sample = -float("inf")
        for action, arm in self.arms.items():
            sample = arm.sample(self.rng)
            if sample > best_sample:
                best_sample = sample
                best_action = action
        return best_action

    def update(self, action: str, success: bool) -> None:
        """Update the posterior for *action* after observing *success*."""
        self.arms[action].update(success)

    def action_probabilities(self, n_samples: int = 1000) -> dict[str, float]:
        """Estimate the probability each arm is optimal via MC sampling.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        dict[str, float]
        """
        actions = list(self.arms.keys())
        counts = {a: 0 for a in actions}
        for _ in range(n_samples):
            samples = {a: self.arms[a].sample(self.rng) for a in actions}
            best = max(samples, key=samples.get)  # type: ignore[arg-type]
            counts[best] += 1
        return {a: counts[a] / n_samples for a in actions}


# ---------------------------------------------------------------------------
# Gaussian Thompson Sampling
# ---------------------------------------------------------------------------

@dataclass
class GaussianArm:
    """Normal-Inverse-Gamma posterior for a Gaussian arm.

    Uses the conjugate prior with known precision form:
        θ | data ~ N(mu, 1/(n·tau))

    Attributes
    ----------
    mu : float
        Posterior mean.
    n_obs : int
        Number of observations.
    tau : float
        Precision of the prior.
    sum_x : float
        Sum of observations.
    sum_x2 : float
        Sum of squared observations.
    """

    mu: float = 0.0
    n_obs: int = 0
    tau: float = 1.0
    sum_x: float = 0.0
    sum_x2: float = 0.0

    @property
    def posterior_mean(self) -> float:
        if self.n_obs == 0:
            return self.mu
        return (self.tau * self.mu + self.sum_x) / (self.tau + self.n_obs)

    @property
    def posterior_variance(self) -> float:
        return 1.0 / (self.tau + self.n_obs)

    def sample(self, rng: np.random.Generator) -> float:
        return float(
            rng.normal(self.posterior_mean, math.sqrt(self.posterior_variance))
        )

    def update(self, value: float) -> None:
        self.n_obs += 1
        self.sum_x += value
        self.sum_x2 += value * value


class GaussianThompson:
    """Thompson sampling with Gaussian arms.

    Parameters
    ----------
    actions : list[str]
    prior_mean : float
    prior_precision : float
        Precision (1/variance) of the prior on each arm's mean.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        actions: list[str],
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.arms: dict[str, GaussianArm] = {
            a: GaussianArm(mu=prior_mean, tau=prior_precision)
            for a in actions
        }

    def select_action(self, minimise: bool = True) -> str:
        """Sample from posteriors and select the best action.

        Parameters
        ----------
        minimise : bool
            If True, select the action with the *lowest* sampled cost.
        """
        samples = {a: arm.sample(self.rng) for a, arm in self.arms.items()}
        if minimise:
            return min(samples, key=samples.get)  # type: ignore[arg-type]
        return max(samples, key=samples.get)  # type: ignore[arg-type]

    def update(self, action: str, value: float) -> None:
        """Update posterior after observing *value* for *action*."""
        self.arms[action].update(value)

    def posterior_means(self) -> dict[str, float]:
        """Return the current posterior mean for each arm."""
        return {a: arm.posterior_mean for a, arm in self.arms.items()}


# ---------------------------------------------------------------------------
# Bounded-Rational Thompson Sampling
# ---------------------------------------------------------------------------

class BoundedRationalThompson:
    """Thompson sampling with bounded rationality.

    Instead of selecting the arm with the best posterior sample, the
    selection is softmax-tempered by the rationality parameter β:

        P(select a) ∝ exp(β · sample(a))

    At β → ∞ this recovers standard Thompson sampling (greedy on samples).
    At β → 0 the agent selects uniformly at random regardless of beliefs.

    Parameters
    ----------
    actions : list[str]
    beta : float
        Rationality parameter.
    prior_mean : float
    prior_precision : float
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        actions: list[str],
        beta: float = 1.0,
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.beta = beta
        self.rng = rng or np.random.default_rng()
        self.gaussian = GaussianThompson(
            actions, prior_mean, prior_precision, self.rng
        )

    def select_action(self, minimise: bool = True) -> str:
        """Softmax-tempered Thompson selection.

        Parameters
        ----------
        minimise : bool
            If True, lower cost is better (negate samples for softmax).
        """
        samples = {
            a: arm.sample(self.rng)
            for a, arm in self.gaussian.arms.items()
        }
        actions = list(samples.keys())
        vals = np.array([samples[a] for a in actions], dtype=np.float64)

        if minimise:
            vals = -vals  # negate so higher → better for softmax

        scaled = self.beta * vals
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        total = probs.sum()
        if total <= 0:
            return str(self.rng.choice(actions))
        probs /= total

        idx = self.rng.choice(len(actions), p=probs)
        return actions[idx]

    def update(self, action: str, value: float) -> None:
        self.gaussian.update(action, value)

    def to_policy(self, state: str, n_samples: int = 1000) -> Policy:
        """Estimate the action selection policy at *state* via MC.

        Parameters
        ----------
        state : str
        n_samples : int

        Returns
        -------
        Policy
        """
        actions = list(self.gaussian.arms.keys())
        counts = {a: 0 for a in actions}
        for _ in range(n_samples):
            a = self.select_action()
            counts[a] += 1
        dist = {a: counts[a] / n_samples for a in actions}
        return Policy(state_action_probs={state: dist}, beta=self.beta)


# ---------------------------------------------------------------------------
# Information-Directed Sampling
# ---------------------------------------------------------------------------

class InformationDirectedSampler:
    """Information-directed sampling (IDS).

    Selects actions maximising the ratio of squared expected regret
    reduction to information gain about the optimal action:

        Ψ(a) = Δ(a)² / g(a)

    where Δ(a) is the expected sub-optimality of action a and g(a) is the
    expected mutual information gained about the identity of the optimal
    action by pulling a.

    Parameters
    ----------
    actions : list[str]
    prior_mean : float
    prior_precision : float
    observation_noise : float
        Variance of the observation noise σ².
    rng : np.random.Generator, optional

    References
    ----------
    - Russo, D. & Van Roy, B. (2014). Learning to optimize via
      information-directed sampling. *NeurIPS*.
    """

    def __init__(
        self,
        actions: list[str],
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        observation_noise: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.observation_noise = observation_noise
        self.arms: dict[str, GaussianArm] = {
            a: GaussianArm(mu=prior_mean, tau=prior_precision)
            for a in actions
        }

    def select_action(self) -> str:
        """Select action minimising the information ratio Ψ(a) = Δ²/g."""
        actions = list(self.arms.keys())
        means = np.array(
            [self.arms[a].posterior_mean for a in actions], dtype=np.float64
        )
        variances = np.array(
            [self.arms[a].posterior_variance for a in actions], dtype=np.float64
        )

        # Expected regret: Δ(a) = μ*(best) - μ(a)  (for maximisation)
        best_mean = np.max(means)
        deltas = best_mean - means

        # Information gain: g(a) ≈ 0.5 · log(1 + var(a) / σ²_obs)
        info_gains = 0.5 * np.log1p(variances / self.observation_noise)
        info_gains = np.maximum(info_gains, 1e-10)

        # Information ratio
        ratios = (deltas ** 2) / info_gains

        # Select minimum ratio (most information-efficient)
        # Break ties by preferring higher information gain
        best_idx = int(np.argmin(ratios))
        return actions[best_idx]

    def update(self, action: str, value: float) -> None:
        self.arms[action].update(value)


# ---------------------------------------------------------------------------
# Knowledge Gradient
# ---------------------------------------------------------------------------

class KnowledgeGradient:
    """Knowledge gradient policy for Gaussian beliefs.

    Selects the action with the greatest expected improvement in the
    value of the best decision after observing one additional sample:

        KG(a) = E[max_a' μ'(a') | observe a] − max_a' μ(a')

    Parameters
    ----------
    actions : list[str]
    prior_mean : float
    prior_precision : float
    observation_noise : float
    rng : np.random.Generator, optional

    References
    ----------
    - Frazier, P. I., Powell, W. B. & Dayanik, S. (2009). The knowledge-
      gradient policy for correlated normal beliefs. *INFORMS J. Comput.*.
    """

    def __init__(
        self,
        actions: list[str],
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        observation_noise: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.observation_noise = observation_noise
        self.arms: dict[str, GaussianArm] = {
            a: GaussianArm(mu=prior_mean, tau=prior_precision)
            for a in actions
        }

    def select_action(self) -> str:
        """Select the action with the highest knowledge gradient."""
        actions = list(self.arms.keys())
        kg_values = np.array(
            [self._kg_value(a) for a in actions], dtype=np.float64
        )
        return actions[int(np.argmax(kg_values))]

    def _kg_value(self, action: str) -> float:
        """Compute the knowledge gradient for a single action.

        KG(a) = σ̃(a) · f(|μ(a) − μ*_{-a}| / σ̃(a))

        where f(z) = z·Φ(z) + φ(z) is the standard KG factor and
        σ̃(a) = change in posterior std from one observation.
        """
        arm = self.arms[action]
        current_var = arm.posterior_variance
        # Posterior variance after one more observation
        new_precision = (arm.tau + arm.n_obs) + 1.0 / self.observation_noise
        new_var = 1.0 / new_precision
        sigma_tilde = math.sqrt(max(current_var - new_var, 1e-15))

        if sigma_tilde < 1e-12:
            return 0.0

        # Best mean excluding this arm
        other_means = [
            self.arms[a].posterior_mean
            for a in self.arms
            if a != action
        ]
        if not other_means:
            return sigma_tilde

        best_other = max(other_means)
        z = abs(arm.posterior_mean - best_other) / sigma_tilde

        # f(z) = z·Φ(z) + φ(z)
        kg = sigma_tilde * (z * sp_stats.norm.cdf(z) + sp_stats.norm.pdf(z))
        return float(kg)

    def update(self, action: str, value: float) -> None:
        self.arms[action].update(value)


# ---------------------------------------------------------------------------
# Bayesian Optimisation for UI parameter tuning
# ---------------------------------------------------------------------------

class BayesianUIOptimiser:
    """Gaussian-process-based Bayesian optimisation for UI parameters.

    Tunes continuous UI parameters (font size, spacing, contrast, etc.)
    by modelling the usability score as a Gaussian process and selecting
    the next configuration to evaluate via Expected Improvement.

    Uses a simplified RBF kernel with independent length-scales and
    noise-free posterior updates (sufficient for low-dimensional UI
    parameter spaces).

    Parameters
    ----------
    param_bounds : dict[str, tuple[float, float]]
        Mapping ``parameter_name → (lower, upper)``.
    kernel_lengthscale : float
        RBF length-scale.
    kernel_variance : float
        Signal variance.
    noise_variance : float
        Observation noise variance.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        kernel_lengthscale: float = 1.0,
        kernel_variance: float = 1.0,
        noise_variance: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.param_names = list(param_bounds.keys())
        self.bounds = np.array(
            [param_bounds[p] for p in self.param_names], dtype=np.float64
        )
        self.lengthscale = kernel_lengthscale
        self.signal_var = kernel_variance
        self.noise_var = noise_variance
        self.rng = rng or np.random.default_rng()

        # Observations
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def suggest(self, n_candidates: int = 500) -> dict[str, float]:
        """Suggest the next parameter configuration to evaluate.

        Uses Expected Improvement (EI) over a random candidate set.

        Parameters
        ----------
        n_candidates : int
            Number of random candidates to evaluate EI over.

        Returns
        -------
        dict[str, float]
            Suggested parameter values.
        """
        if len(self.X) < 2:
            # Not enough data — sample randomly
            return self._random_sample()

        candidates = self._generate_candidates(n_candidates)
        ei_values = np.array(
            [self._expected_improvement(c) for c in candidates],
            dtype=np.float64,
        )
        best_idx = int(np.argmax(ei_values))
        x_best = candidates[best_idx]
        return {self.param_names[i]: float(x_best[i]) for i in range(len(self.param_names))}

    def observe(self, params: dict[str, float], score: float) -> None:
        """Record an observation.

        Parameters
        ----------
        params : dict[str, float]
        score : float
            Usability score (higher is better).
        """
        x = np.array([params[p] for p in self.param_names], dtype=np.float64)
        self.X.append(x)
        self.y.append(score)

    def best_params(self) -> dict[str, float]:
        """Return the parameter configuration with the best observed score."""
        if not self.y:
            return self._random_sample()
        idx = int(np.argmax(self.y))
        x = self.X[idx]
        return {self.param_names[i]: float(x[i]) for i in range(len(self.param_names))}

    # ── GP internals ------------------------------------------------------

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBF (squared exponential) kernel."""
        diff = x1 - x2
        return self.signal_var * math.exp(
            -0.5 * float(np.dot(diff, diff)) / (self.lengthscale ** 2)
        )

    def _gp_predict(self, x: np.ndarray) -> tuple[float, float]:
        """GP posterior mean and variance at point x.

        Returns
        -------
        tuple[float, float]
            (posterior_mean, posterior_variance)
        """
        n = len(self.X)
        if n == 0:
            return 0.0, self.signal_var

        # Kernel matrix K + σ²I
        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                K[i, j] = self._rbf_kernel(self.X[i], self.X[j])
        K += self.noise_var * np.eye(n)

        # Cross-covariance k(x, X)
        k_star = np.array(
            [self._rbf_kernel(x, self.X[i]) for i in range(n)],
            dtype=np.float64,
        )

        # Solve K^{-1} y and K^{-1} k*
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.array(self.y)))
            v = np.linalg.solve(L, k_star)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.array(self.y)))
            v = np.linalg.solve(L, k_star)

        mu = float(np.dot(k_star, alpha))
        var = self.signal_var - float(np.dot(v, v))
        var = max(var, 1e-10)

        return mu, var

    def _expected_improvement(self, x: np.ndarray) -> float:
        """Expected Improvement at point x.

        EI(x) = (μ(x) − f*) Φ(z) + σ(x) φ(z)

        where z = (μ(x) − f*) / σ(x) and f* is the best observed value.
        """
        mu, var = self._gp_predict(x)
        sigma = math.sqrt(var)
        if sigma < 1e-12:
            return 0.0

        f_best = max(self.y)
        z = (mu - f_best) / sigma
        ei = (mu - f_best) * sp_stats.norm.cdf(z) + sigma * sp_stats.norm.pdf(z)
        return max(float(ei), 0.0)

    def _random_sample(self) -> dict[str, float]:
        x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
        return {self.param_names[i]: float(x[i]) for i in range(len(self.param_names))}

    def _generate_candidates(self, n: int) -> list[np.ndarray]:
        return [
            self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            for _ in range(n)
        ]


# ---------------------------------------------------------------------------
# Convenience: posterior update tracker for user interactions
# ---------------------------------------------------------------------------

@dataclass
class UIExplorationTracker:
    """Track a user's exploration of an unfamiliar UI via Thompson sampling.

    Models each UI element as a Gaussian arm whose "reward" is the
    probability of successfully completing the task via that element.
    The posterior is updated after each interaction.

    Parameters
    ----------
    actions : list[str]
        UI element / action identifiers.
    beta : float
        User rationality parameter.
    prior_mean : float
    prior_precision : float
    """

    actions: list[str] = field(default_factory=list)
    beta: float = 1.0
    prior_mean: float = 0.0
    prior_precision: float = 1.0
    _sampler: Optional[BoundedRationalThompson] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if self.actions:
            self._sampler = BoundedRationalThompson(
                actions=self.actions,
                beta=self.beta,
                prior_mean=self.prior_mean,
                prior_precision=self.prior_precision,
            )

    def predict_action(self) -> str:
        """Predict the next action the user will take."""
        assert self._sampler is not None, "Tracker not initialised with actions"
        return self._sampler.select_action(minimise=False)

    def observe_interaction(self, action: str, reward: float) -> None:
        """Update beliefs after observing user interaction outcome."""
        assert self._sampler is not None
        self._sampler.update(action, reward)

    def current_beliefs(self) -> dict[str, dict[str, float]]:
        """Return posterior summary for each action."""
        assert self._sampler is not None
        return {
            a: {
                "mean": arm.posterior_mean,
                "variance": arm.posterior_variance,
                "n_obs": arm.n_obs,
            }
            for a, arm in self._sampler.gaussian.arms.items()
        }

    def to_policy(self, state: str, n_samples: int = 1000) -> Policy:
        """Convert current beliefs to a :class:`Policy`."""
        assert self._sampler is not None
        return self._sampler.to_policy(state, n_samples)
