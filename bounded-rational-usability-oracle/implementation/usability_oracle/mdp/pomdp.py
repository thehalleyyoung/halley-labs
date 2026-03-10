"""
usability_oracle.mdp.pomdp — Partially Observable MDP model.

Extends the standard MDP tuple ⟨S, A, T, R, γ⟩ with observations:

    POMDP = ⟨S, A, T, R, Ω, O, γ⟩

where:
  - Ω is a finite set of observations,
  - O : S' × A → Δ(Ω) is the observation function O(o | s', a).

Provides belief state representation, belief updates, belief-MDP
construction, belief-space discretisation, and POMDP instantiation
from UI accessibility trees with partial observability (scroll-dependent
visibility, loading states, occluded elements).

References
----------
- Kaelbling, L. P., Littman, M. L. & Cassandra, A. R. (1998). Planning
  and acting in partially observable stochastic domains. *AIJ*.
- Smallwood, R. D. & Sondik, E. J. (1973). The optimal control of
  partially observable Markov processes over a finite horizon. *OR*.
- Cassandra, A. R. (1998). Exact and approximate algorithms for
  partially observable Markov decision processes. PhD thesis, Brown.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.spatial import Delaunay  # type: ignore[import-untyped]

from usability_oracle.mdp.models import Action, MDP, State, Transition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """A single observation in the POMDP.

    Observations capture what the user *perceives* about the UI state —
    which elements are visible, their apparent status, and any feedback
    from the last action.

    Parameters
    ----------
    obs_id : str
        Unique identifier.
    features : dict[str, float]
        Numeric observation features (visible element count, scroll pos, …).
    label : str
        Human-readable description.
    metadata : dict
        Arbitrary extra data.
    """

    obs_id: str
    features: dict[str, float] = field(default_factory=dict)
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Observation(id={self.obs_id!r}, label={self.label!r})"


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------


@dataclass
class ObservationModel:
    """Observation function O(o | s', a) for a POMDP.

    Stores the conditional probability of receiving observation *o* when
    the agent transitions to state *s'* via action *a*.

    The model is stored as a nested dict::

        probs[state_id][action_id] -> list[(obs_id, probability)]

    For deterministic observations (e.g. fully visible UI state) each
    (s', a) pair maps to exactly one observation with probability 1.

    Parameters
    ----------
    probs : dict
        Mapping ``state_id -> action_id -> [(obs_id, prob)]``.
    observations : dict[str, Observation]
        All known observations.
    """

    probs: dict[str, dict[str, list[tuple[str, float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    observations: dict[str, Observation] = field(default_factory=dict)

    def prob(self, obs_id: str, state_id: str, action_id: str) -> float:
        """Return O(o | s', a)."""
        for oid, p in self.probs.get(state_id, {}).get(action_id, []):
            if oid == obs_id:
                return p
        return 0.0

    def sample(
        self,
        state_id: str,
        action_id: str,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """Sample an observation from O(· | s', a)."""
        rng = rng or np.random.default_rng()
        entries = self.probs.get(state_id, {}).get(action_id, [])
        if not entries:
            return ""
        obs_ids = [oid for oid, _ in entries]
        probs = np.array([p for _, p in entries], dtype=np.float64)
        total = probs.sum()
        if total <= 0:
            return obs_ids[int(rng.integers(len(obs_ids)))]
        probs /= total
        idx = int(rng.choice(len(obs_ids), p=probs))
        return obs_ids[idx]

    def add(
        self, state_id: str, action_id: str, obs_id: str, probability: float
    ) -> None:
        """Add an observation probability entry."""
        self.probs[state_id][action_id].append((obs_id, probability))
        if obs_id not in self.observations:
            self.observations[obs_id] = Observation(obs_id=obs_id)

    def validate(self) -> list[str]:
        """Validate that observation distributions sum to 1 per (s', a)."""
        errors: list[str] = []
        for sid, act_map in self.probs.items():
            for aid, entries in act_map.items():
                total = sum(p for _, p in entries)
                if abs(total - 1.0) > 1e-6:
                    errors.append(
                        f"O(·|{sid!r},{aid!r}) sums to {total:.6f} ≠ 1"
                    )
        return errors


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """A probability distribution over POMDP states.

    The belief b(s) represents the agent's subjective probability of
    being in each state *s*, given the history of actions and observations.

    Parameters
    ----------
    distribution : dict[str, float]
        Mapping state_id → probability.  Must sum to 1.
    """

    distribution: dict[str, float] = field(default_factory=dict)

    @property
    def entropy(self) -> float:
        """Shannon entropy H(b) = −Σ b(s) log b(s)."""
        h = 0.0
        for p in self.distribution.values():
            if p > 0:
                h -= p * math.log(p)
        return h

    @property
    def max_belief_state(self) -> str:
        """State with highest belief probability."""
        if not self.distribution:
            return ""
        return max(self.distribution, key=self.distribution.get)  # type: ignore[arg-type]

    @property
    def support_size(self) -> int:
        """Number of states with non-zero probability."""
        return sum(1 for p in self.distribution.values() if p > 1e-15)

    def to_vector(self, state_ids: Sequence[str]) -> np.ndarray:
        """Convert to a numpy array aligned with *state_ids*."""
        return np.array(
            [self.distribution.get(sid, 0.0) for sid in state_ids],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray, state_ids: Sequence[str]) -> BeliefState:
        """Construct a BeliefState from a numpy vector and ordered state IDs."""
        dist = {sid: float(vec[i]) for i, sid in enumerate(state_ids) if vec[i] > 1e-15}
        return cls(distribution=dist)

    @classmethod
    def uniform(cls, state_ids: Sequence[str]) -> BeliefState:
        """Create a uniform belief over *state_ids*."""
        n = len(state_ids)
        if n == 0:
            return cls()
        p = 1.0 / n
        return cls(distribution={sid: p for sid in state_ids})

    @classmethod
    def point(cls, state_id: str) -> BeliefState:
        """Create a point-mass belief on a single state."""
        return cls(distribution={state_id: 1.0})

    def validate(self) -> list[str]:
        """Validate belief state constraints."""
        errors: list[str] = []
        total = sum(self.distribution.values())
        if abs(total - 1.0) > 1e-6:
            errors.append(f"Belief probabilities sum to {total:.6f} ≠ 1")
        for sid, p in self.distribution.items():
            if p < -1e-10:
                errors.append(f"Negative belief b({sid!r}) = {p}")
        return errors

    def __repr__(self) -> str:
        n = self.support_size
        h = self.entropy
        top = self.max_belief_state
        return f"BeliefState(support={n}, H={h:.3f}, mode={top!r})"


# ---------------------------------------------------------------------------
# POMDP
# ---------------------------------------------------------------------------


@dataclass
class POMDP:
    """Partially Observable Markov Decision Process.

    Extends :class:`MDP` with an observation set Ω and observation
    model O(o | s', a).  The underlying MDP provides S, A, T, R, γ.

    Parameters
    ----------
    mdp : MDP
        The underlying fully-observable MDP.
    observations : dict[str, Observation]
        Finite observation set Ω.
    observation_model : ObservationModel
        Conditional observation distribution O(o | s', a).
    initial_belief : BeliefState
        Prior belief b₀ over initial states.
    """

    mdp: MDP = field(default_factory=MDP)
    observations: dict[str, Observation] = field(default_factory=dict)
    observation_model: ObservationModel = field(default_factory=ObservationModel)
    initial_belief: BeliefState = field(default_factory=BeliefState)

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def state_ids(self) -> list[str]:
        return list(self.mdp.states.keys())

    @property
    def action_ids(self) -> list[str]:
        return list(self.mdp.actions.keys())

    @property
    def observation_ids(self) -> list[str]:
        return list(self.observations.keys())

    def belief_update(
        self, belief: BeliefState, action_id: str, obs_id: str
    ) -> BeliefState:
        """Bayesian belief update: b'(s') ∝ O(o|s',a) Σ_s T(s'|s,a) b(s).

        Parameters
        ----------
        belief : BeliefState
            Current belief.
        action_id : str
            Action taken.
        obs_id : str
            Observation received.

        Returns
        -------
        BeliefState
            Updated (normalised) belief.
        """
        new_dist: dict[str, float] = {}

        for s_prime in self.mdp.states:
            # O(o | s', a)
            obs_prob = self.observation_model.prob(obs_id, s_prime, action_id)
            if obs_prob <= 0:
                continue

            # Σ_s T(s' | s, a) b(s)
            trans_sum = 0.0
            for s, b_s in belief.distribution.items():
                if b_s <= 0:
                    continue
                transitions = self.mdp.get_transitions(s, action_id)
                for target, prob, _cost in transitions:
                    if target == s_prime:
                        trans_sum += prob * b_s

            if trans_sum > 0:
                new_dist[s_prime] = obs_prob * trans_sum

        # Normalise
        total = sum(new_dist.values())
        if total > 0:
            new_dist = {s: p / total for s, p in new_dist.items()}
        else:
            logger.warning(
                "Belief update produced zero-probability posterior; "
                "returning uniform belief."
            )
            return BeliefState.uniform(self.state_ids)

        return BeliefState(distribution=new_dist)

    def observation_probability(
        self, belief: BeliefState, action_id: str, obs_id: str
    ) -> float:
        """Compute P(o | b, a) = Σ_{s'} O(o|s',a) Σ_s T(s'|s,a) b(s).

        Parameters
        ----------
        belief : BeliefState
        action_id : str
        obs_id : str

        Returns
        -------
        float
        """
        total = 0.0
        for s_prime in self.mdp.states:
            obs_prob = self.observation_model.prob(obs_id, s_prime, action_id)
            if obs_prob <= 0:
                continue
            trans_sum = 0.0
            for s, b_s in belief.distribution.items():
                if b_s <= 0:
                    continue
                for target, prob, _cost in self.mdp.get_transitions(s, action_id):
                    if target == s_prime:
                        trans_sum += prob * b_s
            total += obs_prob * trans_sum
        return total

    def expected_reward(
        self, belief: BeliefState, action_id: str
    ) -> float:
        """Expected immediate reward under belief: R(b, a) = Σ_s b(s) R(s, a).

        Uses transition costs as negative reward: R(s, a) = −Σ_{s'} T(s'|s,a) c(s,a,s').
        """
        total = 0.0
        for s, b_s in belief.distribution.items():
            if b_s <= 0:
                continue
            transitions = self.mdp.get_transitions(s, action_id)
            for _target, prob, cost in transitions:
                total -= b_s * prob * cost
        return total

    def to_belief_mdp(
        self, belief_points: Optional[list[BeliefState]] = None
    ) -> MDP:
        """Construct a belief-MDP approximation from sampled belief points.

        Each belief point becomes a state in the belief MDP. Transitions
        connect belief states via (action, observation) pairs.

        Parameters
        ----------
        belief_points : list[BeliefState], optional
            Sampled belief states.  If None, uses a small grid.

        Returns
        -------
        MDP
            A standard MDP over belief space.
        """
        if belief_points is None:
            belief_points = self._grid_beliefs(resolution=5)

        state_ids_list = self.state_ids
        bp_ids: list[str] = []
        bp_vectors: list[np.ndarray] = []

        # Create belief-MDP states
        b_states: dict[str, State] = {}
        for i, bp in enumerate(belief_points):
            bid = f"b{i}"
            bp_ids.append(bid)
            bp_vectors.append(bp.to_vector(state_ids_list))
            b_states[bid] = State(
                state_id=bid,
                features={"entropy": bp.entropy, "support": float(bp.support_size)},
                label=f"belief_{i}",
                is_terminal=False,
                is_goal=False,
            )

        if not bp_vectors:
            return MDP()

        bp_matrix = np.array(bp_vectors)

        # Create actions (same as underlying MDP)
        b_actions = dict(self.mdp.actions)

        # Create transitions
        b_transitions: list[Transition] = []
        for i, bp in enumerate(belief_points):
            src_id = bp_ids[i]
            for aid in self.mdp.actions:
                for oid in self.observations:
                    p_obs = self.observation_probability(bp, aid, oid)
                    if p_obs < 1e-10:
                        continue

                    new_belief = self.belief_update(bp, aid, oid)
                    new_vec = new_belief.to_vector(state_ids_list)

                    # Find nearest belief point
                    dists = np.linalg.norm(bp_matrix - new_vec, axis=1)
                    nearest_idx = int(np.argmin(dists))
                    target_id = bp_ids[nearest_idx]

                    cost = -self.expected_reward(bp, aid)

                    b_transitions.append(Transition(
                        source=src_id,
                        action=aid,
                        target=target_id,
                        probability=p_obs,
                        cost=max(0.0, cost),
                    ))

        # Find initial belief point
        init_vec = self.initial_belief.to_vector(state_ids_list)
        dists = np.linalg.norm(bp_matrix - init_vec, axis=1)
        initial_idx = int(np.argmin(dists))

        return MDP(
            states=b_states,
            actions=b_actions,
            transitions=b_transitions,
            initial_state=bp_ids[initial_idx],
            goal_states=set(),
            discount=self.mdp.discount,
        )

    def _grid_beliefs(self, resolution: int = 5) -> list[BeliefState]:
        """Generate a uniform grid of belief points on the probability simplex.

        For |S| ≤ 3, produces a regular grid.  For larger state spaces,
        falls back to random Dirichlet samples.

        Parameters
        ----------
        resolution : int
            Number of grid divisions per dimension.

        Returns
        -------
        list[BeliefState]
        """
        sids = self.state_ids
        n = len(sids)

        if n <= 3:
            return self._simplex_grid(sids, resolution)

        # Dirichlet sampling for high dimensions
        rng = np.random.default_rng(42)
        n_samples = min(resolution ** 2, 500)
        points: list[BeliefState] = []
        for _ in range(n_samples):
            vec = rng.dirichlet(np.ones(n))
            dist = {sids[i]: float(vec[i]) for i in range(n) if vec[i] > 1e-15}
            points.append(BeliefState(distribution=dist))

        # Always include uniform and point beliefs
        points.append(BeliefState.uniform(sids))
        for sid in sids:
            points.append(BeliefState.point(sid))

        return points

    @staticmethod
    def _simplex_grid(
        state_ids: list[str], resolution: int
    ) -> list[BeliefState]:
        """Generate a regular grid on the probability simplex for ≤ 3 states."""
        n = len(state_ids)
        if n == 0:
            return []
        if n == 1:
            return [BeliefState.point(state_ids[0])]

        points: list[BeliefState] = []
        # Generate all integer partitions summing to resolution
        for combo in _integer_partitions(resolution, n):
            vec = np.array(combo, dtype=np.float64) / resolution
            dist = {
                state_ids[i]: float(vec[i])
                for i in range(n)
                if vec[i] > 1e-15
            }
            if dist:
                points.append(BeliefState(distribution=dist))
        return points

    def validate(self) -> list[str]:
        """Validate the complete POMDP specification."""
        errors = self.mdp.validate()
        errors.extend(self.observation_model.validate())
        errors.extend(self.initial_belief.validate())
        return errors

    def __repr__(self) -> str:
        return (
            f"POMDP(|S|={self.mdp.n_states}, |A|={self.mdp.n_actions}, "
            f"|Ω|={self.n_observations}, γ={self.mdp.discount})"
        )


# ---------------------------------------------------------------------------
# POMDP builder from accessibility tree
# ---------------------------------------------------------------------------


class POMDPBuilder:
    """Construct a POMDP from an accessibility tree with partial observability.

    Sources of partial observability in UIs:
    1. **Scroll-dependent visibility**: elements outside the viewport
    2. **Uncertain element state**: loading indicators, dynamic content
    3. **Occluded elements**: overlapping elements, modal dialogs

    Parameters
    ----------
    viewport_height : float
        Viewport height in pixels for scroll-visibility computation.
    loading_uncertainty : float
        Probability that a loading element has finished loading.
    occlusion_noise : float
        Probability of observing an occluded element correctly.
    """

    def __init__(
        self,
        viewport_height: float = 1080.0,
        loading_uncertainty: float = 0.3,
        occlusion_noise: float = 0.1,
    ) -> None:
        self.viewport_height = viewport_height
        self.loading_uncertainty = loading_uncertainty
        self.occlusion_noise = occlusion_noise

    def build(self, mdp: MDP, tree: Any) -> POMDP:
        """Build a POMDP from an existing MDP and its accessibility tree.

        Parameters
        ----------
        mdp : MDP
            Fully-observable MDP already constructed from the tree.
        tree : AccessibilityTree
            The accessibility tree (duck-typed).

        Returns
        -------
        POMDP
        """
        observations, obs_model = self._build_observation_model(mdp, tree)
        initial_belief = self._build_initial_belief(mdp, tree)

        return POMDP(
            mdp=mdp,
            observations=observations,
            observation_model=obs_model,
            initial_belief=initial_belief,
        )

    def _build_observation_model(
        self, mdp: MDP, tree: Any
    ) -> tuple[dict[str, Observation], ObservationModel]:
        """Construct observations and O(o | s', a) from the tree.

        Each state produces an observation encoding:
        - visible_elements: set of element IDs visible in viewport
        - element_states: apparent states of visible elements
        - scroll_position: estimated scroll region
        """
        observations: dict[str, Observation] = {}
        model = ObservationModel()

        for sid, state in mdp.states.items():
            node_id = state.metadata.get("node_id", sid.split(":")[0])
            node = tree.get_node(node_id) if tree is not None else None

            # Determine visibility conditions
            is_scrolled = self._is_scroll_dependent(node, tree)
            is_loading = self._is_loading_element(node)
            is_occluded = self._is_occluded(node, tree)

            # Build observation distribution for this state
            for aid in mdp.get_actions(sid):
                obs_entries = self._compute_obs_distribution(
                    sid, aid, node, is_scrolled, is_loading, is_occluded, tree
                )
                for obs_id, prob, obs_features in obs_entries:
                    if obs_id not in observations:
                        observations[obs_id] = Observation(
                            obs_id=obs_id,
                            features=obs_features,
                            label=f"obs_{obs_id}",
                        )
                    model.add(sid, aid, obs_id, prob)

            # Also handle observations for states reached by transitions
            # (O is conditioned on s', so we need entries for target states)
            for aid in mdp.get_actions(sid):
                for target, _prob, _cost in mdp.get_transitions(sid, aid):
                    if target in model.probs and aid in model.probs[target]:
                        continue
                    t_node_id = mdp.states[target].metadata.get(
                        "node_id", target.split(":")[0]
                    )
                    t_node = tree.get_node(t_node_id) if tree is not None else None
                    t_scrolled = self._is_scroll_dependent(t_node, tree)
                    t_loading = self._is_loading_element(t_node)
                    t_occluded = self._is_occluded(t_node, tree)

                    obs_entries = self._compute_obs_distribution(
                        target, aid, t_node,
                        t_scrolled, t_loading, t_occluded, tree,
                    )
                    for obs_id, prob, obs_features in obs_entries:
                        if obs_id not in observations:
                            observations[obs_id] = Observation(
                                obs_id=obs_id,
                                features=obs_features,
                                label=f"obs_{obs_id}",
                            )
                        model.add(target, aid, obs_id, prob)

        model.observations = observations
        return observations, model

    def _compute_obs_distribution(
        self,
        state_id: str,
        action_id: str,
        node: Any,
        is_scrolled: bool,
        is_loading: bool,
        is_occluded: bool,
        tree: Any,
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Compute observation distribution for a (state, action) pair.

        Returns list of (obs_id, probability, features) tuples.
        """
        base_obs_id = f"obs_{state_id}"
        base_features: dict[str, float] = {"visible": 1.0}

        if not is_scrolled and not is_loading and not is_occluded:
            # Fully observable: deterministic observation
            return [(base_obs_id, 1.0, base_features)]

        entries: list[tuple[str, float, dict[str, float]]] = []

        if is_scrolled:
            # Element may or may not be visible depending on scroll
            vis_obs = f"obs_{state_id}_visible"
            hid_obs = f"obs_{state_id}_hidden"
            scroll_offset = getattr(node, "scroll_offset", 0.0) if node else 0.0
            y_pos = getattr(getattr(node, "bounding_box", None), "y", 0.0) if node else 0.0
            vis_prob = self._scroll_visibility_prob(y_pos, scroll_offset)

            entries.append((vis_obs, vis_prob, {"visible": 1.0, "scrolled": 1.0}))
            entries.append((hid_obs, 1.0 - vis_prob, {"visible": 0.0, "scrolled": 1.0}))

        elif is_loading:
            # Element state is uncertain (loaded vs still loading)
            loaded_obs = f"obs_{state_id}_loaded"
            loading_obs = f"obs_{state_id}_loading"
            p_loaded = self.loading_uncertainty
            entries.append((loaded_obs, p_loaded, {"visible": 1.0, "loading": 0.0}))
            entries.append((loading_obs, 1.0 - p_loaded, {"visible": 1.0, "loading": 1.0}))

        elif is_occluded:
            # Element may be visible or occluded by overlapping content
            clear_obs = f"obs_{state_id}_clear"
            occl_obs = f"obs_{state_id}_occluded"
            p_clear = 1.0 - self.occlusion_noise
            entries.append((clear_obs, p_clear, {"visible": 1.0, "occluded": 0.0}))
            entries.append((occl_obs, self.occlusion_noise, {"visible": 0.0, "occluded": 1.0}))

        # Ensure probabilities sum to 1
        total = sum(p for _, p, _ in entries)
        if total > 0 and abs(total - 1.0) > 1e-10:
            entries = [(oid, p / total, f) for oid, p, f in entries]

        return entries

    def _scroll_visibility_prob(
        self, y_position: float, scroll_offset: float
    ) -> float:
        """Probability that an element is visible given scroll state.

        Uses a soft sigmoid based on distance from viewport centre.
        """
        viewport_center = scroll_offset + self.viewport_height / 2
        distance = abs(y_position - viewport_center)
        # Sigmoid: close elements are likely visible
        scale = self.viewport_height / 4
        return 1.0 / (1.0 + math.exp((distance - self.viewport_height / 2) / max(scale, 1.0)))

    def _is_scroll_dependent(self, node: Any, tree: Any) -> bool:
        """Check if a node's visibility depends on scroll position."""
        if node is None:
            return False
        bbox = getattr(node, "bounding_box", None)
        if bbox is None:
            return False
        y = getattr(bbox, "y", 0.0)
        return y > self.viewport_height or y < 0

    def _is_loading_element(self, node: Any) -> bool:
        """Check if a node represents dynamic/loading content."""
        if node is None:
            return False
        role = getattr(node, "role", "")
        state_val = getattr(node, "state", "")
        aria_busy = getattr(node, "aria_busy", False)
        return aria_busy or "loading" in str(state_val).lower() or role == "progressbar"

    def _is_occluded(self, node: Any, tree: Any) -> bool:
        """Check if a node is potentially occluded by overlapping elements."""
        if node is None or tree is None:
            return False
        # Check for modal dialogs or overlapping z-index elements
        metadata = getattr(node, "metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("occluded", False) or metadata.get("behind_modal", False)
        return False

    def _build_initial_belief(self, mdp: MDP, tree: Any) -> BeliefState:
        """Build the initial belief state from the MDP.

        If the initial state is known, use a point belief.  Otherwise
        spread belief over plausible initial states.
        """
        if mdp.initial_state and mdp.initial_state in mdp.states:
            return BeliefState.point(mdp.initial_state)
        return BeliefState.uniform(list(mdp.states.keys()))


# ---------------------------------------------------------------------------
# Belief space discretisation
# ---------------------------------------------------------------------------


def point_based_beliefs(
    pomdp: POMDP,
    n_points: int = 100,
    n_expand_steps: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> list[BeliefState]:
    """Generate belief points via forward reachability from b₀.

    Performs stochastic forward exploration: at each step, pick a random
    action and observation to produce a successor belief.  This is the
    belief-space analogue of random walk exploration.

    Parameters
    ----------
    pomdp : POMDP
    n_points : int
        Target number of belief points.
    n_expand_steps : int
        Number of expansion steps per seed.
    rng : np.random.Generator, optional

    Returns
    -------
    list[BeliefState]
    """
    rng = rng or np.random.default_rng()
    beliefs: list[BeliefState] = [pomdp.initial_belief]
    action_ids = pomdp.action_ids
    obs_ids = pomdp.observation_ids

    if not action_ids or not obs_ids:
        return beliefs

    while len(beliefs) < n_points:
        # Pick a seed belief
        seed = beliefs[int(rng.integers(len(beliefs)))]

        b = seed
        for _ in range(n_expand_steps):
            if len(beliefs) >= n_points:
                break

            aid = action_ids[int(rng.integers(len(action_ids)))]

            # Sample observation weighted by P(o | b, a)
            obs_probs = np.array(
                [pomdp.observation_probability(b, aid, oid) for oid in obs_ids],
                dtype=np.float64,
            )
            total = obs_probs.sum()
            if total <= 0:
                break
            obs_probs /= total
            oid = obs_ids[int(rng.choice(len(obs_ids), p=obs_probs))]

            b = pomdp.belief_update(b, aid, oid)
            beliefs.append(b)

    return beliefs[:n_points]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _integer_partitions(total: int, parts: int) -> list[tuple[int, ...]]:
    """Generate all non-negative integer tuples of length *parts* summing to *total*."""
    if parts == 1:
        return [(total,)]
    result: list[tuple[int, ...]] = []
    for i in range(total + 1):
        for rest in _integer_partitions(total - i, parts - 1):
            result.append((i,) + rest)
    return result
