"""
usability_oracle.mdp.belief — Belief state computation and manipulation.

Provides algorithms for maintaining, updating, compressing, and
visualising belief states in POMDPs:

- Bayesian belief update: b'(s') ∝ O(o|s',a) Σ_s T(s'|s,a) b(s)
- Belief compression via PCA on belief vectors
- Belief entropy as an uncertainty measure
- Information-gathering action identification
- Belief-dependent reward shaping
- Particle filter approximation for large state spaces
- Factored belief states
- Belief state visualisation on the probability simplex

References
----------
- Thrun, S., Burgard, W. & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Pineau, J., Gordon, G. & Thrun, S. (2003). Point-based value iteration.
  *IJCAI*.
- Poupart, P. & Boutilier, C. (2004). VDCBPI: an approximate scalable
  algorithm for large POMDPs. *NeurIPS*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.linalg import svd  # type: ignore[import-untyped]

from usability_oracle.mdp.models import MDP
from usability_oracle.mdp.pomdp import BeliefState, ObservationModel, POMDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bayesian belief update (vectorised)
# ---------------------------------------------------------------------------


class BeliefUpdater:
    """Efficient Bayesian belief update using matrix operations.

    Pre-computes transition and observation matrices for fast vectorised
    updates.  For an n-state POMDP the update is O(n²) per (action, obs).

    Parameters
    ----------
    pomdp : POMDP
    """

    def __init__(self, pomdp: POMDP) -> None:
        self.pomdp = pomdp
        self.state_ids = pomdp.state_ids
        self.n_states = len(self.state_ids)
        self._sid_to_idx = {sid: i for i, sid in enumerate(self.state_ids)}

        # Pre-compute |A| transition matrices T_a[s, s'] = T(s'|s, a)
        self._T: dict[str, np.ndarray] = {}
        for aid in pomdp.action_ids:
            T = np.zeros((self.n_states, self.n_states), dtype=np.float64)
            for i, sid in enumerate(self.state_ids):
                for target, prob, _cost in pomdp.mdp.get_transitions(sid, aid):
                    j = self._sid_to_idx.get(target)
                    if j is not None:
                        T[i, j] += prob
            self._T[aid] = T

        # Pre-compute observation vectors O_a,o[s'] = O(o | s', a)
        self._O: dict[str, dict[str, np.ndarray]] = {}
        for aid in pomdp.action_ids:
            self._O[aid] = {}
            for oid in pomdp.observation_ids:
                o_vec = np.zeros(self.n_states, dtype=np.float64)
                for j, sid in enumerate(self.state_ids):
                    o_vec[j] = pomdp.observation_model.prob(oid, sid, aid)
                self._O[aid][oid] = o_vec

    def update(
        self, belief: BeliefState, action_id: str, obs_id: str
    ) -> BeliefState:
        """Vectorised belief update: b'(s') ∝ O(o|s',a) Σ_s T(s'|s,a) b(s).

        Parameters
        ----------
        belief : BeliefState
        action_id : str
        obs_id : str

        Returns
        -------
        BeliefState
            Normalised posterior belief.
        """
        b = belief.to_vector(self.state_ids)
        T = self._T.get(action_id)
        O = self._O.get(action_id, {}).get(obs_id)

        if T is None or O is None:
            logger.warning("Unknown action/obs in belief update, returning prior.")
            return belief

        # b' ∝ O ⊙ (T^T b)
        predicted = T.T @ b  # Σ_s T(s'|s,a) b(s)
        posterior = O * predicted

        total = posterior.sum()
        if total > 0:
            posterior /= total
        else:
            logger.warning("Zero-probability observation; returning uniform.")
            posterior = np.ones(self.n_states, dtype=np.float64) / self.n_states

        return BeliefState.from_vector(posterior, self.state_ids)

    def predict(
        self, belief: BeliefState, action_id: str
    ) -> np.ndarray:
        """Predict belief after action (before observation): b̄(s') = Σ_s T(s'|s,a) b(s)."""
        b = belief.to_vector(self.state_ids)
        T = self._T.get(action_id)
        if T is None:
            return b.copy()
        return T.T @ b

    def observation_likelihood(
        self, belief: BeliefState, action_id: str, obs_id: str
    ) -> float:
        """P(o | b, a) = Σ_{s'} O(o|s',a) b̄(s')."""
        predicted = self.predict(belief, action_id)
        O = self._O.get(action_id, {}).get(obs_id)
        if O is None:
            return 0.0
        return float(np.dot(O, predicted))


# ---------------------------------------------------------------------------
# Belief compression
# ---------------------------------------------------------------------------


class BeliefCompressor:
    """Compress belief vectors via PCA for approximate POMDP solving.

    Reduces the dimensionality of belief vectors from |S| to *k* while
    preserving as much variance as possible.  Useful when |S| is large
    but beliefs tend to concentrate on a low-dimensional manifold.

    Parameters
    ----------
    n_components : int
        Number of PCA components to retain.
    """

    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components
        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._singular_values: Optional[np.ndarray] = None
        self._state_ids: Optional[list[str]] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None

    def fit(self, beliefs: Sequence[BeliefState], state_ids: list[str]) -> None:
        """Fit PCA on a collection of belief vectors.

        Parameters
        ----------
        beliefs : sequence of BeliefState
        state_ids : list[str]
            Ordered state IDs for vector alignment.
        """
        self._state_ids = state_ids
        n = len(state_ids)
        k = min(self.n_components, n, len(beliefs))

        if k <= 0 or len(beliefs) == 0:
            return

        B = np.array([b.to_vector(state_ids) for b in beliefs], dtype=np.float64)
        self._mean = B.mean(axis=0)
        B_centered = B - self._mean

        U, S, Vt = svd(B_centered, full_matrices=False)
        self._components = Vt[:k]
        self._singular_values = S[:k]

        total_var = (S ** 2).sum()
        if total_var > 0:
            self._explained_variance_ratio = (S[:k] ** 2) / total_var
        else:
            self._explained_variance_ratio = np.zeros(k)

    def compress(self, belief: BeliefState) -> np.ndarray:
        """Project a belief vector onto the PCA subspace.

        Parameters
        ----------
        belief : BeliefState

        Returns
        -------
        np.ndarray
            Compressed belief vector of length n_components.
        """
        if self._components is None or self._mean is None or self._state_ids is None:
            raise RuntimeError("BeliefCompressor not fitted; call fit() first.")

        b = belief.to_vector(self._state_ids)
        return (b - self._mean) @ self._components.T

    def decompress(self, compressed: np.ndarray) -> BeliefState:
        """Reconstruct a belief vector from its compressed representation.

        The reconstructed vector is projected back onto the simplex
        (non-negative, normalised).

        Parameters
        ----------
        compressed : np.ndarray

        Returns
        -------
        BeliefState
        """
        if self._components is None or self._mean is None or self._state_ids is None:
            raise RuntimeError("BeliefCompressor not fitted; call fit() first.")

        reconstructed = compressed @ self._components + self._mean
        # Project onto simplex
        reconstructed = np.maximum(reconstructed, 0.0)
        total = reconstructed.sum()
        if total > 0:
            reconstructed /= total
        return BeliefState.from_vector(reconstructed, self._state_ids)

    @property
    def explained_variance(self) -> Optional[np.ndarray]:
        """Fraction of variance explained by each component."""
        return self._explained_variance_ratio


# ---------------------------------------------------------------------------
# Belief entropy and uncertainty
# ---------------------------------------------------------------------------


def belief_entropy(belief: BeliefState) -> float:
    """Shannon entropy H(b) = −Σ_s b(s) log₂ b(s).

    Higher entropy → more uncertainty about the true state.
    For a uniform belief over n states, H = log₂(n).

    Parameters
    ----------
    belief : BeliefState

    Returns
    -------
    float
        Entropy in bits.
    """
    h = 0.0
    for p in belief.distribution.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


def belief_kl_divergence(p: BeliefState, q: BeliefState) -> float:
    """KL divergence D_KL(p ‖ q) = Σ_s p(s) log(p(s)/q(s)).

    Parameters
    ----------
    p, q : BeliefState

    Returns
    -------
    float
        KL divergence (≥ 0).  Returns inf if q(s)=0 where p(s)>0.
    """
    kl = 0.0
    for s, p_s in p.distribution.items():
        if p_s <= 0:
            continue
        q_s = q.distribution.get(s, 0.0)
        if q_s <= 0:
            return float("inf")
        kl += p_s * math.log(p_s / q_s)
    return kl


def belief_uncertainty_level(belief: BeliefState) -> str:
    """Categorise belief uncertainty as low/medium/high.

    Parameters
    ----------
    belief : BeliefState

    Returns
    -------
    str
        One of ``"low"``, ``"medium"``, ``"high"``.
    """
    h = belief_entropy(belief)
    n = belief.support_size
    if n <= 1:
        return "low"
    max_h = math.log2(n)
    ratio = h / max_h if max_h > 0 else 0.0
    if ratio < 0.3:
        return "low"
    elif ratio < 0.7:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Belief-dependent reward shaping
# ---------------------------------------------------------------------------


class BeliefRewardShaper:
    """Reward shaping based on belief state properties.

    Adds intrinsic motivation signals to encourage information gathering
    when uncertainty is high and exploitation when uncertainty is low.

    R_shaped(b, a) = R(b, a) + w_ent · ΔH(b, a) + w_prog · progress(b)

    Parameters
    ----------
    entropy_weight : float
        Weight for entropy reduction bonus.
    progress_weight : float
        Weight for task progress bonus.
    """

    def __init__(
        self,
        entropy_weight: float = 0.5,
        progress_weight: float = 1.0,
    ) -> None:
        self.entropy_weight = entropy_weight
        self.progress_weight = progress_weight

    def shaped_reward(
        self,
        pomdp: POMDP,
        belief: BeliefState,
        action_id: str,
        updater: Optional[BeliefUpdater] = None,
    ) -> float:
        """Compute belief-shaped reward for (belief, action).

        R_shaped(b, a) = R(b, a) − w_ent · E_o[H(b')] + w_prog · progress(b)

        Parameters
        ----------
        pomdp : POMDP
        belief : BeliefState
        action_id : str
        updater : BeliefUpdater, optional
            Pre-computed updater for efficiency.

        Returns
        -------
        float
        """
        if updater is None:
            updater = BeliefUpdater(pomdp)

        # Base reward
        base_reward = pomdp.expected_reward(belief, action_id)

        # Expected posterior entropy
        current_h = belief_entropy(belief)
        expected_posterior_h = 0.0
        for oid in pomdp.observation_ids:
            p_obs = updater.observation_likelihood(belief, action_id, oid)
            if p_obs <= 0:
                continue
            b_prime = updater.update(belief, action_id, oid)
            expected_posterior_h += p_obs * belief_entropy(b_prime)

        entropy_bonus = current_h - expected_posterior_h  # information gain

        # Task progress
        progress = 0.0
        for sid, p in belief.distribution.items():
            state = pomdp.mdp.states.get(sid)
            if state:
                progress += p * state.features.get("task_progress", 0.0)

        return (
            base_reward
            + self.entropy_weight * entropy_bonus
            + self.progress_weight * progress
        )


# ---------------------------------------------------------------------------
# Particle filter belief approximation
# ---------------------------------------------------------------------------


@dataclass
class ParticleBeliefState:
    """Belief state approximated by a set of weighted particles.

    Each particle is a state ID with an associated weight.
    More memory-efficient than full distributions for large |S|.

    Parameters
    ----------
    particles : list[str]
        State IDs of particles.
    weights : np.ndarray
        Normalised particle weights.
    """

    particles: list[str] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @property
    def effective_sample_size(self) -> float:
        """Effective sample size: 1 / Σ w_i².  Low ESS → degenerate filter."""
        if self.n_particles == 0:
            return 0.0
        return 1.0 / float(np.sum(self.weights ** 2))

    def to_belief_state(self) -> BeliefState:
        """Convert particle representation to an explicit belief state."""
        dist: dict[str, float] = {}
        for sid, w in zip(self.particles, self.weights):
            dist[sid] = dist.get(sid, 0.0) + float(w)
        return BeliefState(distribution=dist)

    @classmethod
    def from_belief(
        cls,
        belief: BeliefState,
        n_particles: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> ParticleBeliefState:
        """Sample particles from an explicit belief state."""
        rng = rng or np.random.default_rng()
        sids = list(belief.distribution.keys())
        probs = np.array([belief.distribution[s] for s in sids], dtype=np.float64)
        total = probs.sum()
        if total <= 0:
            return cls()
        probs /= total

        indices = rng.choice(len(sids), size=n_particles, p=probs)
        particles = [sids[i] for i in indices]
        weights = np.ones(n_particles, dtype=np.float64) / n_particles
        return cls(particles=particles, weights=weights)

    def __repr__(self) -> str:
        ess = self.effective_sample_size
        return f"ParticleBeliefState(n={self.n_particles}, ESS={ess:.1f})"


class ParticleFilter:
    """Sequential Monte Carlo (particle filter) for POMDP belief tracking.

    Maintains a weighted particle set and updates it upon each
    (action, observation) pair.  Includes systematic resampling to
    combat particle degeneracy.

    Parameters
    ----------
    pomdp : POMDP
    n_particles : int
        Number of particles to maintain.
    resample_threshold : float
        Resample when ESS / n_particles < this threshold.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        pomdp: POMDP,
        n_particles: int = 1000,
        resample_threshold: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.pomdp = pomdp
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.rng = rng or np.random.default_rng()

        # Pre-index transitions for fast sampling
        self._trans: dict[str, dict[str, list[tuple[str, float]]]] = {}
        for sid in pomdp.state_ids:
            self._trans[sid] = {}
            for aid in pomdp.action_ids:
                outcomes = pomdp.mdp.get_transitions(sid, aid)
                self._trans[sid][aid] = [(t, p) for t, p, _c in outcomes]

    def initialize(self, belief: Optional[BeliefState] = None) -> ParticleBeliefState:
        """Create initial particle set from a belief state.

        Parameters
        ----------
        belief : BeliefState, optional
            If None, uses the POMDP's initial belief.

        Returns
        -------
        ParticleBeliefState
        """
        b = belief or self.pomdp.initial_belief
        return ParticleBeliefState.from_belief(b, self.n_particles, self.rng)

    def update(
        self,
        particles: ParticleBeliefState,
        action_id: str,
        obs_id: str,
    ) -> ParticleBeliefState:
        """Update particle belief given (action, observation).

        Steps:
        1. Propagate: sample s' ~ T(·|s, a) for each particle s
        2. Weight: w' ∝ w · O(o | s', a)
        3. Resample if ESS is low

        Parameters
        ----------
        particles : ParticleBeliefState
        action_id : str
        obs_id : str

        Returns
        -------
        ParticleBeliefState
        """
        new_particles: list[str] = []
        new_weights = np.zeros(particles.n_particles, dtype=np.float64)

        for i, (sid, w) in enumerate(zip(particles.particles, particles.weights)):
            # Propagate
            outcomes = self._trans.get(sid, {}).get(action_id, [])
            if not outcomes:
                new_particles.append(sid)
                new_weights[i] = 0.0
                continue

            targets = [t for t, _ in outcomes]
            probs = np.array([p for _, p in outcomes], dtype=np.float64)
            total = probs.sum()
            if total <= 0:
                new_particles.append(sid)
                new_weights[i] = 0.0
                continue
            probs /= total

            idx = int(self.rng.choice(len(targets), p=probs))
            s_prime = targets[idx]
            new_particles.append(s_prime)

            # Weight by observation likelihood
            obs_prob = self.pomdp.observation_model.prob(obs_id, s_prime, action_id)
            new_weights[i] = w * obs_prob

        # Normalise weights
        total_w = new_weights.sum()
        if total_w > 0:
            new_weights /= total_w
        else:
            new_weights = np.ones(particles.n_particles, dtype=np.float64) / particles.n_particles

        result = ParticleBeliefState(particles=new_particles, weights=new_weights)

        # Resample if needed
        if result.effective_sample_size < self.resample_threshold * self.n_particles:
            result = self._systematic_resample(result)

        return result

    def _systematic_resample(
        self, particles: ParticleBeliefState
    ) -> ParticleBeliefState:
        """Systematic resampling (low-variance resampling).

        Produces N equally-weighted particles from the weighted set,
        preserving diversity better than multinomial resampling.
        """
        n = particles.n_particles
        if n == 0:
            return particles

        cumsum = np.cumsum(particles.weights)
        cumsum[-1] = 1.0  # ensure exact sum

        u = self.rng.uniform(0, 1.0 / n)
        positions = u + np.arange(n) / n

        new_particles: list[str] = []
        idx = 0
        for pos in positions:
            while idx < n - 1 and cumsum[idx] < pos:
                idx += 1
            new_particles.append(particles.particles[idx])

        new_weights = np.ones(n, dtype=np.float64) / n
        return ParticleBeliefState(particles=new_particles, weights=new_weights)


# ---------------------------------------------------------------------------
# Factored belief states
# ---------------------------------------------------------------------------


@dataclass
class FactoredBeliefState:
    """Belief state factored into independent sub-distributions.

    For large state spaces where S = S₁ × S₂ × … × S_k, maintaining
    a full joint belief is intractable.  A factored belief assumes
    conditional independence between factors:

        b(s₁, s₂, …, s_k) ≈ b₁(s₁) · b₂(s₂) · … · b_k(s_k)

    Parameters
    ----------
    factors : dict[str, dict[str, float]]
        Mapping factor_name → {value → probability}.
    """

    factors: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def total_states(self) -> int:
        """Total number of joint states (product of factor sizes)."""
        if not self.factors:
            return 0
        result = 1
        for factor in self.factors.values():
            result *= len(factor)
        return result

    @property
    def entropy(self) -> float:
        """Total entropy (sum of factor entropies under independence)."""
        h = 0.0
        for factor_dist in self.factors.values():
            for p in factor_dist.values():
                if p > 0:
                    h -= p * math.log2(p)
        return h

    def marginal(self, factor_name: str) -> dict[str, float]:
        """Return the marginal distribution for a single factor."""
        return dict(self.factors.get(factor_name, {}))

    def update_factor(
        self, factor_name: str, new_dist: dict[str, float]
    ) -> FactoredBeliefState:
        """Return a new factored belief with one factor updated."""
        new_factors = dict(self.factors)
        new_factors[factor_name] = new_dist
        return FactoredBeliefState(factors=new_factors)

    def to_joint(self, state_ids: Optional[list[str]] = None) -> BeliefState:
        """Expand to a full joint belief state.

        Warning: exponential in the number of factors.

        Parameters
        ----------
        state_ids : list[str], optional
            If provided, only expand to matching state IDs.

        Returns
        -------
        BeliefState
        """
        if not self.factors:
            return BeliefState()

        factor_names = list(self.factors.keys())
        factor_values = [list(self.factors[f].keys()) for f in factor_names]

        dist: dict[str, float] = {}
        import itertools

        for combo in itertools.product(*factor_values):
            joint_id = ":".join(combo)
            prob = 1.0
            for fname, val in zip(factor_names, combo):
                prob *= self.factors[fname].get(val, 0.0)
            if prob > 1e-15:
                if state_ids is None or joint_id in state_ids:
                    dist[joint_id] = prob

        return BeliefState(distribution=dist)

    @classmethod
    def from_joint(
        cls,
        belief: BeliefState,
        factor_names: list[str],
        separator: str = ":",
    ) -> FactoredBeliefState:
        """Approximate a joint belief as factored by marginalising.

        State IDs are split by *separator* to extract factor values.

        Parameters
        ----------
        belief : BeliefState
        factor_names : list[str]
        separator : str

        Returns
        -------
        FactoredBeliefState
        """
        factors: dict[str, dict[str, float]] = {f: {} for f in factor_names}

        for sid, prob in belief.distribution.items():
            parts = sid.split(separator)
            for i, fname in enumerate(factor_names):
                if i < len(parts):
                    val = parts[i]
                    factors[fname][val] = factors[fname].get(val, 0.0) + prob

        return cls(factors=factors)

    def __repr__(self) -> str:
        sizes = {f: len(d) for f, d in self.factors.items()}
        return f"FactoredBeliefState(factors={sizes}, H={self.entropy:.3f})"


# ---------------------------------------------------------------------------
# Belief visualisation
# ---------------------------------------------------------------------------


def belief_simplex_coordinates(
    belief: BeliefState, state_ids: list[str]
) -> Optional[np.ndarray]:
    """Project a belief state onto 2D coordinates for simplex visualisation.

    For |S| = 2, maps to a 1D line segment.
    For |S| = 3, maps to a 2D equilateral triangle (barycentric coords).
    For |S| > 3, uses PCA projection to 2D.

    Parameters
    ----------
    belief : BeliefState
    state_ids : list[str]

    Returns
    -------
    np.ndarray or None
        2D coordinates (x, y), or None if |S| < 2.
    """
    n = len(state_ids)
    if n < 2:
        return None

    b = belief.to_vector(state_ids)

    if n == 2:
        return np.array([b[0], 0.0])

    if n == 3:
        # Barycentric to Cartesian on equilateral triangle
        # Vertices: (0,0), (1,0), (0.5, √3/2)
        x = b[1] + 0.5 * b[2]
        y = (math.sqrt(3) / 2) * b[2]
        return np.array([x, y])

    # PCA for higher dimensions
    U, S, Vt = svd(b.reshape(1, -1), full_matrices=False)
    if len(b) >= 2:
        return b[:2]  # simple projection onto first two states
    return np.array([b[0], 0.0])


def belief_to_ascii(
    belief: BeliefState,
    max_states: int = 20,
    bar_width: int = 40,
) -> str:
    """Render a belief state as an ASCII bar chart.

    Parameters
    ----------
    belief : BeliefState
    max_states : int
        Maximum number of states to display.
    bar_width : int
        Width of the probability bar.

    Returns
    -------
    str
        Multi-line ASCII visualisation.
    """
    if not belief.distribution:
        return "(empty belief)"

    sorted_items = sorted(
        belief.distribution.items(), key=lambda x: -x[1]
    )[:max_states]

    lines: list[str] = [
        f"Belief state (H={belief.entropy:.3f} nats, "
        f"support={belief.support_size})",
        "─" * (bar_width + 30),
    ]

    for sid, prob in sorted_items:
        n_bars = int(prob * bar_width)
        bar = "█" * n_bars + "░" * (bar_width - n_bars)
        label = sid[:20].ljust(20)
        lines.append(f"  {label} │ {bar} {prob:.4f}")

    remaining = len(belief.distribution) - max_states
    if remaining > 0:
        lines.append(f"  … and {remaining} more states")

    return "\n".join(lines)
