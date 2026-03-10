"""
usability_oracle.policy.belief_policy — Belief-space policies for POMDPs.

Implements policies defined over *belief states* — probability distributions
over the underlying (hidden) state — for settings where the user has only
partial observability of the interface state (e.g., hidden menu items,
ambiguous icons, uncertain system state).

Key components
--------------
- **Alpha-vector representation** — piecewise-linear and convex (PWLC)
  value function over the belief simplex.
- **Belief-dependent action selection** — selects actions based on the
  entire belief distribution, not just the MAP state.
- **Information-seeking policies** — drive the user toward observations
  that maximally reduce state uncertainty.
- **Risk-sensitive belief policies** — CVaR-based pessimistic action
  selection for safety-critical UIs.

References
----------
- Kaelbling, L. P., Littman, M. L. & Cassandra, A. R. (1998). Planning
  and acting in partially observable stochastic domains. *AI*, 101, 99–134.
- Smallwood, R. D. & Sondik, E. J. (1973). The optimal control of partially
  observable Markov processes over a finite horizon. *Operations Research*.
- Araya, M. et al. (2010). A POMDP extension with belief-dependent rewards.
  *NeurIPS*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from usability_oracle.policy.models import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """A probability distribution over hidden states.

    Attributes
    ----------
    state_ids : list[str]
        Ordered state identifiers.
    probabilities : np.ndarray
        Belief vector b ∈ Δ^{|S|-1}, summing to 1.
    """

    state_ids: list[str] = field(default_factory=list)
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        self.probabilities = np.asarray(self.probabilities, dtype=np.float64)
        total = self.probabilities.sum()
        if total > 0 and abs(total - 1.0) > 1e-8:
            self.probabilities = self.probabilities / total

    @property
    def n_states(self) -> int:
        return len(self.state_ids)

    def prob(self, state: str) -> float:
        """Return belief probability for *state*."""
        try:
            idx = self.state_ids.index(state)
            return float(self.probabilities[idx])
        except ValueError:
            return 0.0

    def entropy(self) -> float:
        """Shannon entropy H(b) = −Σ b(s) log b(s)."""
        h = 0.0
        for p in self.probabilities:
            if p > 0:
                h -= p * math.log(p)
        return h

    def max_state(self) -> str:
        """Most probable state (MAP estimate)."""
        idx = int(np.argmax(self.probabilities))
        return self.state_ids[idx]

    def is_certain(self, threshold: float = 0.99) -> bool:
        """True if belief is concentrated on a single state."""
        return bool(np.max(self.probabilities) >= threshold)

    @staticmethod
    def uniform(state_ids: list[str]) -> "BeliefState":
        n = len(state_ids)
        return BeliefState(
            state_ids=state_ids,
            probabilities=np.ones(n, dtype=np.float64) / n,
        )


# ---------------------------------------------------------------------------
# Belief update
# ---------------------------------------------------------------------------

def belief_update(
    belief: BeliefState,
    action: str,
    observation: str,
    transition_fn: dict[str, dict[str, dict[str, float]]],
    observation_fn: dict[str, dict[str, dict[str, float]]],
) -> BeliefState:
    """Bayes-optimal belief update after (action, observation).

    b'(s') = η · O(o|s',a) · Σ_s T(s'|s,a) · b(s)

    Parameters
    ----------
    belief : BeliefState
    action : str
    observation : str
    transition_fn : dict[str, dict[str, dict[str, float]]]
        ``T[s][a][s']`` = P(s'|s,a).
    observation_fn : dict[str, dict[str, dict[str, float]]]
        ``O[s'][a][o]`` = P(o|s',a).

    Returns
    -------
    BeliefState
    """
    states = belief.state_ids
    n = len(states)
    new_probs = np.zeros(n, dtype=np.float64)

    for j, s_prime in enumerate(states):
        # Observation likelihood
        obs_prob = observation_fn.get(s_prime, {}).get(action, {}).get(observation, 0.0)

        # Transition prediction
        pred = 0.0
        for i, s in enumerate(states):
            t_prob = transition_fn.get(s, {}).get(action, {}).get(s_prime, 0.0)
            pred += t_prob * belief.probabilities[i]

        new_probs[j] = obs_prob * pred

    total = new_probs.sum()
    if total > 0:
        new_probs /= total
    else:
        new_probs = belief.probabilities.copy()

    return BeliefState(state_ids=states, probabilities=new_probs)


# ---------------------------------------------------------------------------
# Alpha vectors
# ---------------------------------------------------------------------------

@dataclass
class AlphaVector:
    """An alpha-vector representing a linear value function over the belief simplex.

    V(b) = max_α α^T · b

    Each alpha-vector corresponds to a conditional plan (action + successor
    alpha-vectors for each observation).

    Attributes
    ----------
    vector : np.ndarray
        Value vector α ∈ ℝ^{|S|}.
    action : str
        Action associated with this alpha-vector.
    """

    vector: np.ndarray = field(default_factory=lambda: np.array([]))
    action: str = ""

    def value(self, belief: BeliefState) -> float:
        """Evaluate this alpha-vector at belief b: α^T b."""
        return float(np.dot(self.vector, belief.probabilities))


class PWLCValueFunction:
    """Piecewise-linear and convex value function over the belief simplex.

    V(b) = max_{α ∈ Γ} α^T · b

    where Γ is a set of alpha-vectors.

    Parameters
    ----------
    alpha_vectors : list[AlphaVector]
    """

    def __init__(self, alpha_vectors: Optional[list[AlphaVector]] = None) -> None:
        self.alpha_vectors: list[AlphaVector] = alpha_vectors or []

    def value(self, belief: BeliefState) -> float:
        """Evaluate V(b) = max_α α^T b."""
        if not self.alpha_vectors:
            return 0.0
        return max(av.value(belief) for av in self.alpha_vectors)

    def best_action(self, belief: BeliefState) -> str:
        """Return the action of the maximising alpha-vector at belief b."""
        if not self.alpha_vectors:
            return ""
        best_av = max(self.alpha_vectors, key=lambda av: av.value(belief))
        return best_av.action

    def add(self, alpha: AlphaVector) -> None:
        self.alpha_vectors.append(alpha)

    def prune(self, beliefs: Optional[list[BeliefState]] = None) -> None:
        """Remove dominated alpha-vectors.

        An alpha-vector is dominated if there is no belief point at which
        it is the maximiser.  If a set of witness beliefs is provided,
        pruning is approximate (point-based); otherwise exact.
        """
        if len(self.alpha_vectors) <= 1:
            return

        if beliefs is not None:
            self._point_based_prune(beliefs)
        else:
            self._simple_prune()

    def _point_based_prune(self, beliefs: list[BeliefState]) -> None:
        """Keep only alpha-vectors that are best at some belief point."""
        useful: set[int] = set()
        for b in beliefs:
            values = [av.value(b) for av in self.alpha_vectors]
            useful.add(int(np.argmax(values)))
        self.alpha_vectors = [
            self.alpha_vectors[i] for i in sorted(useful)
        ]

    def _simple_prune(self) -> None:
        """Heuristic pruning: remove vectors dominated at simplex vertices."""
        n = len(self.alpha_vectors)
        if n == 0:
            return
        dim = len(self.alpha_vectors[0].vector)
        vertices = [
            BeliefState(
                state_ids=[f"s{j}" for j in range(dim)],
                probabilities=np.eye(dim)[i],
            )
            for i in range(dim)
        ]
        self._point_based_prune(vertices)


# ---------------------------------------------------------------------------
# Point-based value iteration for belief-space policies
# ---------------------------------------------------------------------------

def point_based_value_iteration(
    state_ids: list[str],
    actions: list[str],
    transition_fn: dict[str, dict[str, dict[str, float]]],
    observation_fn: dict[str, dict[str, dict[str, float]]],
    reward_fn: dict[str, dict[str, float]],
    beliefs: list[BeliefState],
    discount: float = 0.95,
    n_iterations: int = 50,
    epsilon: float = 1e-4,
) -> PWLCValueFunction:
    """Point-based value iteration (PBVI) for a POMDP.

    Computes an approximate PWLC value function by performing Bellman
    backups at a fixed set of belief points.

    Parameters
    ----------
    state_ids : list[str]
    actions : list[str]
    transition_fn : dict
        ``T[s][a][s']`` = P(s'|s,a).
    observation_fn : dict
        ``O[s'][a][o]`` = P(o|s',a).
    reward_fn : dict
        ``R[s][a]`` = reward for taking action a in state s.
    beliefs : list[BeliefState]
        Witness belief points.
    discount : float
    n_iterations : int
    epsilon : float

    Returns
    -------
    PWLCValueFunction
    """
    n_s = len(state_ids)
    s_idx = {s: i for i, s in enumerate(state_ids)}

    # Collect all possible observations
    obs_set: set[str] = set()
    for s in observation_fn:
        for a in observation_fn[s]:
            obs_set.update(observation_fn[s][a].keys())
    observations = sorted(obs_set)

    # Initialise with zero value function
    vf = PWLCValueFunction([
        AlphaVector(vector=np.zeros(n_s, dtype=np.float64), action=actions[0])
    ])

    for iteration in range(n_iterations):
        new_alphas: list[AlphaVector] = []

        for b in beliefs:
            best_alpha: Optional[AlphaVector] = None
            best_val = -float("inf")

            for a in actions:
                # Compute α^a_o for each observation
                alpha_a = np.zeros(n_s, dtype=np.float64)

                # Reward component
                for i, s in enumerate(state_ids):
                    alpha_a[i] += reward_fn.get(s, {}).get(a, 0.0)

                # Future value component
                for o in observations:
                    # Build the belief conditioned on (a, o)
                    b_prime = belief_update(b, a, o, transition_fn, observation_fn)
                    # Find maximising alpha-vector at b'
                    if vf.alpha_vectors:
                        best_av = max(
                            vf.alpha_vectors,
                            key=lambda av: av.value(b_prime),
                        )
                        av_vec = best_av.vector
                    else:
                        av_vec = np.zeros(n_s, dtype=np.float64)

                    # α^a += γ Σ_s T(s'|s,a)·O(o|s',a)·α*(s')
                    for i, s in enumerate(state_ids):
                        for j, s_prime in enumerate(state_ids):
                            t = transition_fn.get(s, {}).get(a, {}).get(s_prime, 0.0)
                            obs_p = observation_fn.get(s_prime, {}).get(a, {}).get(o, 0.0)
                            alpha_a[i] += discount * t * obs_p * av_vec[j]

                val = float(np.dot(alpha_a, b.probabilities))
                if val > best_val:
                    best_val = val
                    best_alpha = AlphaVector(vector=alpha_a, action=a)

            if best_alpha is not None:
                new_alphas.append(best_alpha)

        new_vf = PWLCValueFunction(new_alphas)

        # Check convergence
        max_diff = 0.0
        for b in beliefs:
            diff = abs(new_vf.value(b) - vf.value(b))
            max_diff = max(max_diff, diff)

        vf = new_vf
        if max_diff < epsilon:
            logger.debug("PBVI converged at iteration %d (diff=%.2e)", iteration + 1, max_diff)
            break

    vf.prune(beliefs)
    return vf


# ---------------------------------------------------------------------------
# Information-seeking policy
# ---------------------------------------------------------------------------

class InformationSeekingPolicy:
    """Policy that selects actions maximising expected information gain.

    Chooses actions that maximally reduce belief entropy:

        a* = argmax_a E_o[ H(b) − H(b'|a,o) ]

    Parameters
    ----------
    state_ids : list[str]
    actions : list[str]
    transition_fn : dict
    observation_fn : dict
    reward_weight : float
        Weight on the task reward (vs. information gain).
    """

    def __init__(
        self,
        state_ids: list[str],
        actions: list[str],
        transition_fn: dict[str, dict[str, dict[str, float]]],
        observation_fn: dict[str, dict[str, dict[str, float]]],
        reward_weight: float = 0.0,
        reward_fn: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self.state_ids = state_ids
        self.actions = actions
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.reward_weight = reward_weight
        self.reward_fn = reward_fn or {}

    def select_action(self, belief: BeliefState) -> str:
        """Select the action with highest expected information gain."""
        best_action = self.actions[0]
        best_score = -float("inf")

        current_entropy = belief.entropy()

        for a in self.actions:
            expected_info_gain = self._expected_info_gain(belief, a, current_entropy)
            expected_reward = self._expected_reward(belief, a)
            score = expected_info_gain + self.reward_weight * expected_reward

            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def _expected_info_gain(
        self, belief: BeliefState, action: str, current_entropy: float
    ) -> float:
        """Compute E_o[H(b) − H(b'|a,o)]."""
        obs_set: set[str] = set()
        for s in self.observation_fn:
            obs_set.update(self.observation_fn.get(s, {}).get(action, {}).keys())

        info_gain = 0.0
        for o in obs_set:
            # P(o|a,b) = Σ_{s'} O(o|s',a) Σ_s T(s'|s,a) b(s)
            p_o = 0.0
            for j, s_prime in enumerate(self.state_ids):
                obs_p = self.observation_fn.get(s_prime, {}).get(action, {}).get(o, 0.0)
                pred = 0.0
                for i, s in enumerate(self.state_ids):
                    t = self.transition_fn.get(s, {}).get(action, {}).get(s_prime, 0.0)
                    pred += t * belief.probabilities[i]
                p_o += obs_p * pred

            if p_o < 1e-15:
                continue

            b_prime = belief_update(
                belief, action, o, self.transition_fn, self.observation_fn
            )
            info_gain += p_o * (current_entropy - b_prime.entropy())

        return info_gain

    def _expected_reward(self, belief: BeliefState, action: str) -> float:
        r = 0.0
        for i, s in enumerate(belief.state_ids):
            r += belief.probabilities[i] * self.reward_fn.get(s, {}).get(action, 0.0)
        return r


# ---------------------------------------------------------------------------
# Risk-sensitive belief policy
# ---------------------------------------------------------------------------

class RiskSensitiveBeliefPolicy:
    """CVaR-based risk-sensitive policy over beliefs.

    Instead of maximising expected value, maximises the Conditional
    Value-at-Risk (CVaR) at confidence level α:

        CVaR_α(V) = E[V | V ≤ VaR_α(V)]

    Parameters
    ----------
    value_fn : PWLCValueFunction
    alpha : float
        Risk level in (0, 1].  α=1 is risk-neutral; α→0 is maximally
        pessimistic.
    n_samples : int
        MC samples for CVaR estimation.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        value_fn: PWLCValueFunction,
        alpha: float = 0.1,
        n_samples: int = 500,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.value_fn = value_fn
        self.alpha = alpha
        self.n_samples = n_samples
        self.rng = rng or np.random.default_rng()

    def select_action(
        self,
        belief: BeliefState,
        actions: list[str],
        transition_fn: dict[str, dict[str, dict[str, float]]],
        observation_fn: dict[str, dict[str, dict[str, float]]],
        reward_fn: dict[str, dict[str, float]],
    ) -> str:
        """Select the action maximising CVaR of the successor belief value."""
        best_action = actions[0]
        best_cvar = -float("inf")

        for a in actions:
            cvar = self._estimate_cvar(
                belief, a, transition_fn, observation_fn, reward_fn
            )
            if cvar > best_cvar:
                best_cvar = cvar
                best_action = a

        return best_action

    def _estimate_cvar(
        self,
        belief: BeliefState,
        action: str,
        transition_fn: dict,
        observation_fn: dict,
        reward_fn: dict,
    ) -> float:
        """Estimate CVaR_α for taking *action* from *belief*."""
        values: list[float] = []

        for _ in range(self.n_samples):
            # Sample a true state from belief
            s_idx = self.rng.choice(
                len(belief.state_ids), p=belief.probabilities
            )
            s = belief.state_ids[s_idx]

            # Immediate reward
            r = reward_fn.get(s, {}).get(action, 0.0)

            # Sample next state
            trans = transition_fn.get(s, {}).get(action, {})
            if trans:
                next_states = list(trans.keys())
                probs = np.array([trans[ns] for ns in next_states], dtype=np.float64)
                probs /= probs.sum()
                ns_idx = self.rng.choice(len(next_states), p=probs)
                ns = next_states[ns_idx]
            else:
                ns = s

            # Sample observation
            obs = observation_fn.get(ns, {}).get(action, {})
            if obs:
                obs_keys = list(obs.keys())
                obs_probs = np.array([obs[o] for o in obs_keys], dtype=np.float64)
                obs_probs /= obs_probs.sum()
                o_idx = self.rng.choice(len(obs_keys), p=obs_probs)
                o = obs_keys[o_idx]
            else:
                o = ""

            # Belief update and value
            b_prime = belief_update(belief, action, o, transition_fn, observation_fn)
            v = r + self.value_fn.value(b_prime)
            values.append(v)

        if not values:
            return 0.0

        arr = np.sort(np.array(values))
        n_tail = max(int(self.alpha * len(arr)), 1)
        return float(np.mean(arr[:n_tail]))
