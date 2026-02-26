"""
Posterior-predictive shield synthesis for safe trading.

Constructs shields over a posterior distribution of MDPs, filtering actions
that violate safety specifications with high posterior probability.

For each (state, action) pair, computes:
    P(safety | model) weighted by the posterior distribution

An action a is permitted in state s iff:
    P(phi | s, a) >= 1 - delta

where phi is the conjunction of all active safety specifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from scipy.stats import dirichlet

logger = logging.getLogger(__name__)


class ShieldMode(Enum):
    """Shield operation mode."""
    PRE = "pre"       # pre-shield: filter before action selection
    POST = "post"     # post-shield: override after action selection
    ADVISORY = "advisory"  # advisory: warn but don't block


@dataclass
class ShieldQuery:
    """A query to the shield for action permissibility."""
    state: np.ndarray
    state_index: Optional[int] = None
    desired_action: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0


@dataclass
class ShieldResult:
    """Result of a shield query."""
    permitted_actions: np.ndarray  # boolean mask over action space
    safety_probabilities: np.ndarray  # P(safe | s, a) for each action
    most_restrictive_spec: Optional[str] = None
    permissivity_ratio: float = 1.0
    shielded_action: Optional[int] = None
    original_action: Optional[int] = None
    was_overridden: bool = False
    spec_violations: Dict[str, float] = field(default_factory=dict)


@dataclass
class MDPModel:
    """A Markov Decision Process model."""
    n_states: int
    n_actions: int
    transition_matrix: np.ndarray  # (n_states, n_actions, n_states)
    reward_matrix: np.ndarray      # (n_states, n_actions)
    initial_distribution: np.ndarray  # (n_states,)

    def validate(self) -> bool:
        """Check that transition probabilities sum to 1."""
        for s in range(self.n_states):
            for a in range(self.n_actions):
                total = np.sum(self.transition_matrix[s, a, :])
                if not np.isclose(total, 1.0, atol=1e-6):
                    return False
        return True


class PosteriorPredictiveShield:
    """
    Posterior-predictive shield synthesis.

    Constructs a safety shield by computing the posterior probability of
    satisfying safety specifications for each (state, action) pair,
    marginalizing over a posterior distribution of MDP models.

    The shield permits action a in state s iff:
        E_{M ~ posterior}[P_M(phi | s, a)] >= 1 - delta

    where phi is the conjunction of safety specifications and delta
    is the tolerable violation probability.

    Parameters
    ----------
    n_states : int
        Number of states in the MDP.
    n_actions : int
        Number of actions.
    delta : float
        Safety threshold. Actions are permitted iff P(safe) >= 1 - delta.
    horizon : int
        Planning horizon for safety evaluation.
    mode : ShieldMode
        Shield operation mode (pre, post, or advisory).
    n_posterior_samples : int
        Number of posterior samples for Monte Carlo integration.
    cache_size : int
        Maximum number of cached shield queries.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        delta: float = 0.05,
        horizon: int = 10,
        mode: ShieldMode = ShieldMode.PRE,
        n_posterior_samples: int = 200,
        cache_size: int = 10000,
        adaptive_delta: bool = False,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.delta = delta
        self.horizon = horizon
        self.mode = mode
        self.n_posterior_samples = n_posterior_samples
        self.cache_size = cache_size
        self._adaptive_delta = adaptive_delta

        # Safety specifications
        self._specs: List[Any] = []
        self._spec_names: List[str] = []
        self._spec_weights: np.ndarray = np.array([])

        # Shield table: P(safe | s, a) for each (s, a)
        self._safety_table: Optional[np.ndarray] = None  # (n_states, n_actions)
        self._permitted_table: Optional[np.ndarray] = None  # (n_states, n_actions) bool
        self._per_spec_safety: Dict[str, np.ndarray] = {}

        # Query cache using LRU-style dict
        self._cache: Dict[int, ShieldResult] = {}
        self._cache_order: List[int] = []

        # Statistics
        self._n_queries = 0
        self._n_overrides = 0
        self._synthesis_count = 0

        # Posterior parameters (Dirichlet)
        self._prior_counts: Optional[np.ndarray] = None
        self._posterior_counts: Optional[np.ndarray] = None

        logger.info(
            "PosteriorPredictiveShield initialized: %d states, %d actions, "
            "delta=%.4f, horizon=%d",
            n_states, n_actions, delta, horizon,
        )

    def add_spec(self, spec: Any, name: str, weight: float = 1.0) -> None:
        """
        Add a safety specification to the shield.

        Parameters
        ----------
        spec : SafetySpecification
            The safety specification object.
        name : str
            Human-readable name for this specification.
        weight : float
            Relative weight for this specification (used in soft composition).
        """
        self._specs.append(spec)
        self._spec_names.append(name)
        weights_list = list(self._spec_weights) + [weight]
        self._spec_weights = np.array(weights_list)
        # Normalize weights
        self._spec_weights = self._spec_weights / np.sum(self._spec_weights)
        logger.info("Added safety spec '%s' (weight=%.3f)", name, weight)

    def set_prior(self, prior_counts: np.ndarray) -> None:
        """
        Set Dirichlet prior counts for transition probabilities.

        Parameters
        ----------
        prior_counts : np.ndarray
            Shape (n_states, n_actions, n_states). Dirichlet concentration
            parameters for each (state, action) transition distribution.
        """
        assert prior_counts.shape == (self.n_states, self.n_actions, self.n_states)
        self._prior_counts = prior_counts.copy()
        if self._posterior_counts is None:
            self._posterior_counts = prior_counts.copy()

    def update_posterior(self, transitions: np.ndarray) -> None:
        """
        Update posterior with observed transition counts.

        Parameters
        ----------
        transitions : np.ndarray
            Shape (n_states, n_actions, n_states). Observed transition counts.
        """
        assert transitions.shape == (self.n_states, self.n_actions, self.n_states)
        if self._posterior_counts is None:
            if self._prior_counts is not None:
                self._posterior_counts = self._prior_counts.copy()
            else:
                # Uniform Dirichlet prior
                self._posterior_counts = np.ones(
                    (self.n_states, self.n_actions, self.n_states)
                )
        self._posterior_counts += transitions
        # Invalidate cache
        self._cache.clear()
        self._cache_order.clear()

    def _sample_mdp_from_posterior(self, rng: np.random.Generator) -> MDPModel:
        """
        Sample a single MDP from the posterior Dirichlet distribution.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        MDPModel
            A sampled MDP with transitions drawn from the posterior.
        """
        counts = self._posterior_counts
        if counts is None:
            counts = np.ones((self.n_states, self.n_actions, self.n_states))

        T = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                alpha = counts[s, a, :]
                alpha = np.maximum(alpha, 1e-10)
                T[s, a, :] = rng.dirichlet(alpha)

        R = np.zeros((self.n_states, self.n_actions))
        init = np.ones(self.n_states) / self.n_states

        return MDPModel(
            n_states=self.n_states,
            n_actions=self.n_actions,
            transition_matrix=T,
            reward_matrix=R,
            initial_distribution=init,
        )

    def _evaluate_safety_for_model(
        self,
        model: MDPModel,
        spec: Any,
        state: int,
        action: int,
    ) -> float:
        """
        Evaluate safety probability for a single model, spec, state, action.

        Uses forward reachability analysis over the planning horizon to
        compute the probability that the specification is satisfied.

        Parameters
        ----------
        model : MDPModel
            The MDP model to evaluate.
        spec : SafetySpecification
            The safety specification.
        state : int
            Current state index.
        action : int
            Action index.

        Returns
        -------
        float
            Probability of satisfying the specification from (state, action).
        """
        T = model.transition_matrix
        n_s = model.n_states
        horizon = self.horizon

        # Initialize: probability of being in each safe state
        # After taking action in state, distribution over next states:
        next_dist = T[state, action, :].copy()

        # Check which states satisfy the spec constraints
        safe_states = self._get_safe_states(spec)

        # P(safe) via forward propagation
        # At each step, keep only probability mass in safe states
        safe_prob = np.sum(next_dist[safe_states])
        current_dist = next_dist.copy()

        for t in range(1, horizon):
            # Zero out probability in unsafe states
            masked_dist = np.zeros(n_s)
            masked_dist[safe_states] = current_dist[safe_states]

            if np.sum(masked_dist) < 1e-15:
                return 0.0

            # Propagate forward using average policy (uniform)
            next_dist = np.zeros(n_s)
            for s in range(n_s):
                if masked_dist[s] > 1e-15:
                    # Average over all actions (conservative)
                    for a in range(model.n_actions):
                        next_dist += (masked_dist[s] / model.n_actions) * T[s, a, :]

            current_dist = next_dist
            step_safe = np.sum(current_dist[safe_states])
            safe_prob = min(safe_prob, step_safe)

        return float(safe_prob)

    def _get_safe_states(self, spec: Any) -> np.ndarray:
        """
        Get boolean mask of states satisfying a specification's invariant.

        Parameters
        ----------
        spec : SafetySpecification
            The specification to check.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_states,) indicating safe states.
        """
        if hasattr(spec, 'get_safe_state_mask'):
            return spec.get_safe_state_mask(self.n_states)

        # Default: try the constraints method
        if hasattr(spec, 'get_constraints'):
            constraints = spec.get_constraints()
            safe = np.ones(self.n_states, dtype=bool)
            for constraint_fn in constraints:
                for s in range(self.n_states):
                    if not constraint_fn(s):
                        safe[s] = False
            return safe

        # Fallback: all states safe
        return np.ones(self.n_states, dtype=bool)

    def _compute_safety_table_mc(
        self,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute safety table via Monte Carlo over posterior MDP samples.

        Returns
        -------
        safety_table : np.ndarray
            Shape (n_states, n_actions). Posterior mean P(safe | s, a).
        per_spec_safety : dict
            Per-specification safety tables.
        """
        n_s, n_a = self.n_states, self.n_actions
        n_samples = self.n_posterior_samples

        # Accumulate safety probabilities
        safety_accum = np.zeros((n_s, n_a))
        per_spec_accum: Dict[str, np.ndarray] = {
            name: np.zeros((n_s, n_a)) for name in self._spec_names
        }

        for i in range(n_samples):
            model = self._sample_mdp_from_posterior(rng)

            for s in range(n_s):
                for a in range(n_a):
                    # Compute min safety across all specs (conjunction)
                    min_safety = 1.0
                    for spec, name in zip(self._specs, self._spec_names):
                        p_safe = self._evaluate_safety_for_model(
                            model, spec, s, a,
                        )
                        per_spec_accum[name][s, a] += p_safe
                        min_safety = min(min_safety, p_safe)
                    safety_accum[s, a] += min_safety

        # Average over samples
        safety_table = safety_accum / n_samples
        per_spec_safety = {
            name: arr / n_samples for name, arr in per_spec_accum.items()
        }

        return safety_table, per_spec_safety

    def _compute_safety_table_analytic(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute safety table analytically using Dirichlet mean transitions.

        More efficient than MC when the model structure permits closed-form
        computation. Uses the posterior mean transition probabilities.

        Returns
        -------
        safety_table : np.ndarray
            Shape (n_states, n_actions).
        per_spec_safety : dict
            Per-specification safety tables.
        """
        n_s, n_a = self.n_states, self.n_actions
        counts = self._posterior_counts
        if counts is None:
            counts = np.ones((n_s, n_a, n_s))

        # Posterior mean transitions
        T_mean = np.zeros((n_s, n_a, n_s))
        for s in range(n_s):
            for a in range(n_a):
                alpha = counts[s, a, :]
                total = np.sum(alpha)
                T_mean[s, a, :] = alpha / total

        mean_model = MDPModel(
            n_states=n_s,
            n_actions=n_a,
            transition_matrix=T_mean,
            reward_matrix=np.zeros((n_s, n_a)),
            initial_distribution=np.ones(n_s) / n_s,
        )

        safety_table = np.zeros((n_s, n_a))
        per_spec_safety: Dict[str, np.ndarray] = {
            name: np.zeros((n_s, n_a)) for name in self._spec_names
        }

        for s in range(n_s):
            for a in range(n_a):
                min_safety = 1.0
                for spec, name in zip(self._specs, self._spec_names):
                    p_safe = self._evaluate_safety_for_model(
                        mean_model, spec, s, a,
                    )
                    per_spec_safety[name][s, a] = p_safe
                    min_safety = min(min_safety, p_safe)
                safety_table[s, a] = min_safety

        # Apply posterior variance correction (second-order)
        # Var[Dir(alpha)] = alpha_i(alpha_0 - alpha_i) / (alpha_0^2 (alpha_0+1))
        variance_penalty = np.zeros((n_s, n_a))
        for s in range(n_s):
            for a in range(n_a):
                alpha = counts[s, a, :]
                alpha_0 = np.sum(alpha)
                # Average variance of transition probabilities
                var_sum = 0.0
                for j in range(n_s):
                    var_sum += alpha[j] * (alpha_0 - alpha[j]) / (
                        alpha_0 ** 2 * (alpha_0 + 1)
                    )
                variance_penalty[s, a] = var_sum

        # Penalize safety by variance (conservative)
        safety_table = np.maximum(0.0, safety_table - np.sqrt(variance_penalty))

        return safety_table, per_spec_safety

    def synthesize(
        self,
        posterior_counts: Optional[np.ndarray] = None,
        specs: Optional[List[Any]] = None,
        spec_names: Optional[List[str]] = None,
        method: str = "mc",
        seed: int = 42,
    ) -> np.ndarray:
        """
        Synthesize the shield from posterior and safety specifications.

        Parameters
        ----------
        posterior_counts : np.ndarray, optional
            Dirichlet posterior counts. If None, uses stored posterior.
        specs : list, optional
            Safety specifications. If None, uses stored specs.
        spec_names : list, optional
            Names for the specifications.
        method : str
            'mc' for Monte Carlo, 'analytic' for closed-form approximation.
        seed : int
            Random seed for MC sampling.

        Returns
        -------
        np.ndarray
            Boolean permitted action table of shape (n_states, n_actions).
        """
        if posterior_counts is not None:
            self._posterior_counts = posterior_counts.copy()

        if specs is not None:
            self._specs = specs
            if spec_names is not None:
                self._spec_names = spec_names
            else:
                self._spec_names = [f"spec_{i}" for i in range(len(specs))]
            self._spec_weights = np.ones(len(specs)) / len(specs)

        if not self._specs:
            logger.warning("No safety specs provided; shield is trivially permissive.")
            self._safety_table = np.ones((self.n_states, self.n_actions))
            self._permitted_table = np.ones(
                (self.n_states, self.n_actions), dtype=bool
            )
            return self._permitted_table

        rng = np.random.default_rng(seed)

        if method == "mc":
            self._safety_table, self._per_spec_safety = (
                self._compute_safety_table_mc(rng)
            )
        elif method == "analytic":
            self._safety_table, self._per_spec_safety = (
                self._compute_safety_table_analytic()
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply threshold
        self._permitted_table = self._safety_table >= (1.0 - self.delta)

        # Adaptive delta: if posterior is concentrated, relax threshold
        if self._adaptive_delta:
            effective_delta = self._compute_adaptive_delta()
            self._permitted_table = self._safety_table >= (1.0 - effective_delta)
            logger.info(
                "Adaptive delta: base=%.4f, effective=%.4f",
                self.delta, effective_delta,
            )

        # Ensure at least one action permitted per state (liveness fallback)
        for s in range(self.n_states):
            if not np.any(self._permitted_table[s, :]):
                # Permit the safest action
                best_a = np.argmax(self._safety_table[s, :])
                self._permitted_table[s, best_a] = True
                logger.debug(
                    "State %d: no action meets threshold; permitting safest "
                    "action %d (P=%.4f)",
                    s, best_a, self._safety_table[s, best_a],
                )

        self._synthesis_count += 1
        self._cache.clear()
        self._cache_order.clear()

        n_permitted = np.sum(self._permitted_table)
        total = self.n_states * self.n_actions
        logger.info(
            "Shield synthesized (method=%s): %d/%d actions permitted (%.1f%%)",
            method, n_permitted, total, 100 * n_permitted / total,
        )

        return self._permitted_table.copy()

    def query(self, state: np.ndarray, state_index: Optional[int] = None) -> ShieldResult:
        """
        Query the shield for permitted actions in a given state.

        Parameters
        ----------
        state : np.ndarray
            State vector (used for cache key).
        state_index : int, optional
            Discrete state index. If None, uses hash of state vector.

        Returns
        -------
        ShieldResult
            Shield query result with permitted actions and diagnostics.
        """
        self._n_queries += 1

        if state_index is None:
            state_index = self._state_to_index(state)

        # Check cache
        cache_key = state_index
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._safety_table is None or self._permitted_table is None:
            # Shield not synthesized: permit everything
            result = ShieldResult(
                permitted_actions=np.ones(self.n_actions, dtype=bool),
                safety_probabilities=np.ones(self.n_actions),
                permissivity_ratio=1.0,
            )
        else:
            s = min(state_index, self.n_states - 1)
            permitted = self._permitted_table[s, :].copy()
            safety_probs = self._safety_table[s, :].copy()

            # Find most restrictive spec
            most_restrictive = None
            min_safety_across_specs = 1.0
            spec_violations: Dict[str, float] = {}
            for name in self._spec_names:
                if name in self._per_spec_safety:
                    spec_min = np.min(self._per_spec_safety[name][s, :])
                    spec_violations[name] = float(1.0 - spec_min)
                    if spec_min < min_safety_across_specs:
                        min_safety_across_specs = spec_min
                        most_restrictive = name

            n_permitted = int(np.sum(permitted))
            ratio = n_permitted / self.n_actions

            result = ShieldResult(
                permitted_actions=permitted,
                safety_probabilities=safety_probs,
                most_restrictive_spec=most_restrictive,
                permissivity_ratio=ratio,
                spec_violations=spec_violations,
            )

        # Update cache
        self._update_cache(cache_key, result)
        return result

    def get_permitted_actions(self, state: np.ndarray, state_index: Optional[int] = None) -> np.ndarray:
        """
        Get boolean mask of permitted actions for a state.

        Parameters
        ----------
        state : np.ndarray
            State vector.
        state_index : int, optional
            Discrete state index.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_actions,).
        """
        result = self.query(state, state_index)
        return result.permitted_actions

    def shield_action(
        self,
        state: np.ndarray,
        desired_action: int,
        state_index: Optional[int] = None,
    ) -> ShieldResult:
        """
        Apply the shield to a desired action.

        If the desired action is permitted, it passes through. Otherwise,
        the shield selects the safest permitted alternative.

        Parameters
        ----------
        state : np.ndarray
            Current state.
        desired_action : int
            The action the agent wants to take.
        state_index : int, optional
            Discrete state index.

        Returns
        -------
        ShieldResult
            Result including the (possibly overridden) action.
        """
        result = self.query(state, state_index)
        result.original_action = desired_action

        if result.permitted_actions[desired_action]:
            result.shielded_action = desired_action
            result.was_overridden = False
        else:
            # Select safest permitted action
            permitted_indices = np.where(result.permitted_actions)[0]
            if len(permitted_indices) == 0:
                # Fallback: safest overall
                result.shielded_action = int(
                    np.argmax(result.safety_probabilities)
                )
            else:
                # Among permitted, pick one with highest safety probability
                best_idx = permitted_indices[
                    np.argmax(result.safety_probabilities[permitted_indices])
                ]
                result.shielded_action = int(best_idx)

            result.was_overridden = True
            self._n_overrides += 1

            if self.mode == ShieldMode.ADVISORY:
                # In advisory mode, return original but flag it
                result.shielded_action = desired_action
                result.was_overridden = False

            logger.debug(
                "Shield override: state_idx=%s, desired=%d -> shielded=%d",
                state_index, desired_action, result.shielded_action,
            )

        return result

    def get_safety_table(self) -> Optional[np.ndarray]:
        """Return the full safety probability table."""
        if self._safety_table is None:
            return None
        return self._safety_table.copy()

    def get_override_rate(self) -> float:
        """Get the fraction of queries that resulted in overrides."""
        if self._n_queries == 0:
            return 0.0
        return self._n_overrides / self._n_queries

    def get_per_spec_safety(self) -> Dict[str, np.ndarray]:
        """Get per-specification safety tables."""
        return {k: v.copy() for k, v in self._per_spec_safety.items()}

    def _state_to_index(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete index via hashing."""
        # Quantize state to avoid floating-point cache misses
        quantized = np.round(state * 100).astype(np.int64)
        return int(hash(quantized.tobytes()) % self.n_states)

    def _update_cache(self, key: int, result: ShieldResult) -> None:
        """Update the LRU cache."""
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self.cache_size:
            evict_key = self._cache_order.pop(0)
            del self._cache[evict_key]
        self._cache[key] = result
        self._cache_order.append(key)

    def reset_stats(self) -> None:
        """Reset query statistics."""
        self._n_queries = 0
        self._n_overrides = 0

    def _compute_adaptive_delta(self) -> float:
        """Compute adaptive delta based on posterior concentration.

        When the posterior is concentrated (high total counts), the
        uncertainty is low and we can afford a larger delta (more
        permissive shield). When uncertain, we stay conservative.
        """
        if self._posterior_counts is None:
            return self.delta

        # Average concentration across all (state, action) pairs
        total_counts = []
        for s in range(self.n_states):
            for a in range(self.n_actions):
                total_counts.append(np.sum(self._posterior_counts[s, a, :]))
        avg_concentration = np.mean(total_counts)

        # Scale delta based on concentration: more data → relax toward 2*delta
        # concentration_factor in [1.0, 2.0]
        concentration_factor = min(2.0, 1.0 + np.log1p(avg_concentration) / 10.0)
        return min(self.delta * concentration_factor, 0.5)

    def graceful_degradation(
        self,
        state: np.ndarray,
        state_index: Optional[int] = None,
    ) -> ShieldResult:
        """Return the least-restrictive action when no actions pass the threshold.

        Unlike ``query()``, this method always returns at least one permitted
        action (the one with the highest safety probability), but emits a
        warning when the threshold is not met.
        """
        result = self.query(state, state_index)
        if not np.any(result.permitted_actions):
            best_a = int(np.argmax(result.safety_probabilities))
            result.permitted_actions[best_a] = True
            result.permissivity_ratio = 1.0 / self.n_actions
            logger.warning(
                "graceful_degradation: no action meets threshold; "
                "permitting safest action %d (P=%.4f)",
                best_a, result.safety_probabilities[best_a],
            )
        return result

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the shield state."""
        avg_permissivity = 0.0
        if self._permitted_table is not None:
            avg_permissivity = float(np.mean(
                np.sum(self._permitted_table, axis=1) / self.n_actions
            ))

        return {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "delta": self.delta,
            "horizon": self.horizon,
            "mode": self.mode.value,
            "n_specs": len(self._specs),
            "spec_names": self._spec_names,
            "synthesis_count": self._synthesis_count,
            "n_queries": self._n_queries,
            "n_overrides": self._n_overrides,
            "override_rate": self.get_override_rate(),
            "avg_permissivity": avg_permissivity,
            "is_synthesized": self._safety_table is not None,
        }


class ComposedShield:
    """
    Composition of multiple shields via intersection.

    The composed shield permits an action iff ALL component shields permit it.
    This implements the shield intersection operation from the literature on
    compositional shielding.

    Parameters
    ----------
    shields : list of PosteriorPredictiveShield
        Component shields to compose.
    composition_mode : str
        'intersection' (all must permit) or 'union' (any must permit).
    """

    def __init__(
        self,
        shields: List[PosteriorPredictiveShield],
        composition_mode: str = "intersection",
    ) -> None:
        self.shields = shields
        self.composition_mode = composition_mode

        if not shields:
            raise ValueError("Must provide at least one shield.")

        self.n_states = shields[0].n_states
        self.n_actions = shields[0].n_actions
        for sh in shields[1:]:
            if sh.n_states != self.n_states or sh.n_actions != self.n_actions:
                raise ValueError("All shields must have same state/action dimensions.")

        self._n_queries = 0

    def query(self, state: np.ndarray, state_index: Optional[int] = None) -> ShieldResult:
        """
        Query the composed shield.

        Parameters
        ----------
        state : np.ndarray
            Current state.
        state_index : int, optional
            Discrete state index.

        Returns
        -------
        ShieldResult
            Composed shield result.
        """
        self._n_queries += 1
        results = [sh.query(state, state_index) for sh in self.shields]

        if self.composition_mode == "intersection":
            permitted = np.ones(self.n_actions, dtype=bool)
            for r in results:
                permitted &= r.permitted_actions
        else:
            permitted = np.zeros(self.n_actions, dtype=bool)
            for r in results:
                permitted |= r.permitted_actions

        # Combined safety: minimum across shields
        safety_probs = np.ones(self.n_actions)
        for r in results:
            safety_probs = np.minimum(safety_probs, r.safety_probabilities)

        # Ensure at least one action
        if not np.any(permitted):
            best_a = int(np.argmax(safety_probs))
            permitted[best_a] = True

        # Aggregate spec violations
        all_violations: Dict[str, float] = {}
        most_restrictive = None
        min_ratio = 1.0
        for r in results:
            all_violations.update(r.spec_violations)
            if r.permissivity_ratio < min_ratio:
                min_ratio = r.permissivity_ratio
                most_restrictive = r.most_restrictive_spec

        n_permitted = int(np.sum(permitted))

        return ShieldResult(
            permitted_actions=permitted,
            safety_probabilities=safety_probs,
            most_restrictive_spec=most_restrictive,
            permissivity_ratio=n_permitted / self.n_actions,
            spec_violations=all_violations,
        )

    def shield_action(
        self,
        state: np.ndarray,
        desired_action: int,
        state_index: Optional[int] = None,
    ) -> ShieldResult:
        """
        Apply composed shield to a desired action.

        Parameters
        ----------
        state : np.ndarray
            Current state.
        desired_action : int
            Desired action index.
        state_index : int, optional
            Discrete state index.

        Returns
        -------
        ShieldResult
            Shield result with possibly overridden action.
        """
        result = self.query(state, state_index)
        result.original_action = desired_action

        if result.permitted_actions[desired_action]:
            result.shielded_action = desired_action
            result.was_overridden = False
        else:
            permitted_idx = np.where(result.permitted_actions)[0]
            if len(permitted_idx) == 0:
                result.shielded_action = int(
                    np.argmax(result.safety_probabilities)
                )
            else:
                best = permitted_idx[
                    np.argmax(result.safety_probabilities[permitted_idx])
                ]
                result.shielded_action = int(best)
            result.was_overridden = True

        return result

    def get_permissivity_matrix(self) -> np.ndarray:
        """
        Get the permissivity ratio for each state under composition.

        Returns
        -------
        np.ndarray
            Shape (n_states,). Permissivity ratio per state.
        """
        ratios = np.ones(self.n_states)
        for s in range(self.n_states):
            state_vec = np.zeros(1)
            result = self.query(state_vec, state_index=s)
            ratios[s] = result.permissivity_ratio
        return ratios

    def summary(self) -> Dict[str, Any]:
        """Return summary of composed shield."""
        return {
            "n_shields": len(self.shields),
            "composition_mode": self.composition_mode,
            "n_queries": self._n_queries,
            "component_summaries": [sh.summary() for sh in self.shields],
        }
