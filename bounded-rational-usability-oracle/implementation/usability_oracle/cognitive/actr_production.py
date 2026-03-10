"""ACT-R production system for usability modelling.

Implements the procedural module of the ACT-R cognitive architecture
(Anderson, 2007; Anderson & Lebiere, 1998), including production rule
representation, utility-based conflict resolution, utility learning,
expected gain computation, production compilation, and procedural
learning curves.

References
----------
Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical
    Universe?* Oxford University Press.
Anderson, J. R. & Lebiere, C. (1998). *The Atomic Components of Thought*.
    Lawrence Erlbaum Associates.
Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C.,
    & Qin, Y. (2004). An integrated theory of the mind. *Psychological
    Review*, 111(4), 1036-1060.
Taatgen, N. A. & Lee, F. J. (2003). Production compilation: A simple
    mechanism to model complex skill acquisition. *Human Factors*, 45(1),
    61-76.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Buffer state
# ---------------------------------------------------------------------------


@dataclass
class BufferState:
    """State of ACT-R buffers at a point in time.

    Each buffer holds at most one chunk represented as a type plus
    slot–value mapping.

    Attributes
    ----------
    buffers : dict[str, dict[str, Any]]
        Mapping of buffer names to their current contents.  Each value
        is a dict with at least ``"chunk_type"`` and additional slot
        keys.
    """

    buffers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get(self, buffer_name: str) -> Optional[Dict[str, Any]]:
        """Return the content of *buffer_name*, or ``None``."""
        return self.buffers.get(buffer_name)

    def set(self, buffer_name: str, content: Dict[str, Any]) -> None:
        """Set the content of *buffer_name*."""
        self.buffers[buffer_name] = content

    def clear(self, buffer_name: str) -> None:
        """Clear a buffer."""
        self.buffers.pop(buffer_name, None)

    def copy(self) -> "BufferState":
        """Return a deep copy."""
        import copy as _copy
        return BufferState(buffers=_copy.deepcopy(self.buffers))


# ---------------------------------------------------------------------------
# Production rule
# ---------------------------------------------------------------------------


@dataclass
class Production:
    """A single production rule (condition → action).

    Attributes
    ----------
    name : str
        Unique rule name.
    conditions : dict[str, dict[str, Any]]
        Buffer conditions.  Keys are buffer names; values are patterns
        that must match (subset of the buffer content).
    actions : dict[str, dict[str, Any]]
        Buffer modifications.  Keys are buffer names; values are updates
        to apply.
    utility : float
        Current utility estimate for conflict resolution.
    cost : float
        Estimated time to fire this production (seconds).
    reward : float
        Expected reward associated with achieving the goal after this
        production fires.
    creation_time : float
        Simulation time when this production was created.
    fire_count : int
        Number of times this production has fired.
    success_count : int
        Number of successful goal completions after firing.
    failure_count : int
        Number of failures after firing.
    """

    name: str
    conditions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    utility: float = 0.0
    cost: float = 0.050
    reward: float = 0.0
    creation_time: float = 0.0
    fire_count: int = 0
    success_count: int = 0
    failure_count: int = 0


# ---------------------------------------------------------------------------
# Production system
# ---------------------------------------------------------------------------


class ACTRProductionSystem:
    """ACT-R production system with utility learning and compilation.

    Conflict resolution selects the production with the highest expected
    utility *E* (Anderson, 2007, Ch. 4):

    .. math::

        E_i = P_i G_i - C_i + \\varepsilon

    where *P_i* is the estimated probability of success, *G_i* is the
    value of the goal, *C_i* is the estimated cost, and ε is noise.

    Utility learning uses the Rescorla-Wagner-style update:

    .. math::

        U_i(n) = U_i(n-1) + \\alpha (R_i - U_i(n-1))

    Parameters
    ----------
    utility_noise_s : float
        Noise scale for utility selection (default 0.25).
    alpha : float
        Learning rate for utility learning (default 0.2).
    default_utility : float
        Initial utility for new productions (default 0.0).
    egs : float
        Expected Gain S — noise in expected gain (default 0.5).
    production_firing_time : float
        Default production firing time in seconds (default 0.050).
    compilation_threshold : int
        Minimum fire count before a production pair can be compiled
        (default 3).
    """

    def __init__(
        self,
        utility_noise_s: float = 0.25,
        alpha: float = 0.2,
        default_utility: float = 0.0,
        egs: float = 0.5,
        production_firing_time: float = 0.050,
        compilation_threshold: int = 3,
    ) -> None:
        self.utility_noise_s = utility_noise_s
        self.alpha = alpha
        self.default_utility = default_utility
        self.egs = egs
        self.production_firing_time = production_firing_time
        self.compilation_threshold = compilation_threshold

        self._productions: Dict[str, Production] = {}
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Production management
    # ------------------------------------------------------------------ #

    def add_production(self, production: Production) -> None:
        """Add a production rule to the system."""
        self._productions[production.name] = production

    def get_production(self, name: str) -> Optional[Production]:
        """Return a production by name, or ``None``."""
        return self._productions.get(name)

    @property
    def productions(self) -> List[Production]:
        """All productions in the system."""
        return list(self._productions.values())

    @property
    def production_count(self) -> int:
        """Number of productions in the system."""
        return len(self._productions)

    # ------------------------------------------------------------------ #
    # Pattern matching
    # ------------------------------------------------------------------ #

    def matches(
        self,
        production: Production,
        state: BufferState,
    ) -> bool:
        """Check whether *production*'s conditions match *state*.

        Every buffer pattern in the production's conditions must be a
        subset of the corresponding buffer's current content.

        Parameters
        ----------
        production : Production
            The production rule to test.
        state : BufferState
            Current buffer state.

        Returns
        -------
        bool
            True if all conditions are satisfied.
        """
        for buffer_name, pattern in production.conditions.items():
            content = state.get(buffer_name)
            if content is None:
                return False
            for slot, value in pattern.items():
                if slot not in content or content[slot] != value:
                    return False
        return True

    def matching_productions(
        self,
        state: BufferState,
    ) -> List[Production]:
        """Return all productions whose conditions match *state*."""
        return [p for p in self._productions.values() if self.matches(p, state)]

    # ------------------------------------------------------------------ #
    # Conflict resolution
    # ------------------------------------------------------------------ #

    def expected_gain(self, production: Production) -> float:
        """Compute expected gain *E_i* for a production.

        .. math::

            E_i = P_i G_i - C_i

        where *P_i* is the success probability, *G_i* is the goal value
        (= utility + reward), and *C_i* is the cost.

        Parameters
        ----------
        production : Production
            Target production.

        Returns
        -------
        float
            Expected gain (no noise added).
        """
        total = production.fire_count
        if total == 0:
            p_success = 0.5
        else:
            p_success = production.success_count / max(total, 1)

        goal_value = production.utility + production.reward
        return p_success * goal_value - production.cost

    def select_production(
        self,
        state: BufferState,
    ) -> Optional[Production]:
        """Select the best matching production via conflict resolution.

        Adds logistic noise to each production's expected gain and
        returns the production with the highest noisy gain.

        Parameters
        ----------
        state : BufferState
            Current buffer state.

        Returns
        -------
        Production or None
            The winning production, or ``None`` if no productions match.
        """
        candidates = self.matching_productions(state)
        if not candidates:
            return None

        best_prod: Optional[Production] = None
        best_gain = -float("inf")

        for prod in candidates:
            gain = self.expected_gain(prod)
            if self.egs > 0:
                gain += float(self._rng.logistic(0.0, self.egs))
            if gain > best_gain:
                best_gain = gain
                best_prod = prod

        return best_prod

    # ------------------------------------------------------------------ #
    # Fire production
    # ------------------------------------------------------------------ #

    def fire(
        self,
        production: Production,
        state: BufferState,
    ) -> Tuple[BufferState, float]:
        """Fire a production, updating buffer state.

        Applies the production's actions to a copy of *state* and
        returns the new state along with the firing time.

        Parameters
        ----------
        production : Production
            Production to fire.
        state : BufferState
            Current buffer state.

        Returns
        -------
        tuple[BufferState, float]
            (new_state, firing_time_seconds).
        """
        new_state = state.copy()

        for buffer_name, action in production.actions.items():
            current = new_state.get(buffer_name) or {}
            if action.get("_clear"):
                new_state.clear(buffer_name)
            else:
                current.update(action)
                new_state.set(buffer_name, current)

        production.fire_count += 1
        return new_state, self.production_firing_time

    # ------------------------------------------------------------------ #
    # Utility learning
    # ------------------------------------------------------------------ #

    def update_utility(
        self,
        production: Production,
        reward: float,
    ) -> float:
        """Update production utility using Rescorla-Wagner rule.

        .. math::

            U_i(n) = U_i(n-1) + \\alpha (R_i - U_i(n-1))

        Parameters
        ----------
        production : Production
            Production whose utility is updated.
        reward : float
            Observed reward for the outcome.

        Returns
        -------
        float
            Updated utility value.
        """
        production.utility += self.alpha * (reward - production.utility)
        return production.utility

    def update_utilities_batch(
        self,
        fired_sequence: Sequence[Production],
        reward: float,
        discount: float = 1.0,
    ) -> NDArray[np.floating]:
        """Update utilities for a sequence of fired productions.

        Productions fired earlier receive a discounted reward reflecting
        the temporal credit assignment problem.

        Parameters
        ----------
        fired_sequence : sequence of Production
            Productions in the order they fired (earliest first).
        reward : float
            Final reward obtained.
        discount : float
            Per-step discount factor in (0, 1].

        Returns
        -------
        numpy.ndarray
            Updated utility values.
        """
        n = len(fired_sequence)
        utilities = np.empty(n, dtype=np.float64)
        for i, prod in enumerate(reversed(fired_sequence)):
            step_reward = reward * (discount ** i)
            utilities[n - 1 - i] = self.update_utility(prod, step_reward)
        return utilities

    def record_outcome(
        self,
        production: Production,
        success: bool,
        reward: float,
    ) -> None:
        """Record a success/failure outcome and update utility.

        Parameters
        ----------
        production : Production
            The production that was fired.
        success : bool
            Whether the goal was achieved.
        reward : float
            Reward signal.
        """
        if success:
            production.success_count += 1
        else:
            production.failure_count += 1
        self.update_utility(production, reward)

    # ------------------------------------------------------------------ #
    # Production compilation
    # ------------------------------------------------------------------ #

    def compile(
        self,
        prod_a: Production,
        prod_b: Production,
    ) -> Optional[Production]:
        """Compile two sequentially-fired productions into one.

        Production compilation (Taatgen & Lee, 2003) merges two
        productions that fire in sequence, removing the intermediate
        buffer operations and producing a single, faster production.

        Compilation only proceeds if both productions have fired at
        least ``compilation_threshold`` times.

        Parameters
        ----------
        prod_a : Production
            First production in the sequence.
        prod_b : Production
            Second production.

        Returns
        -------
        Production or None
            The compiled production, or ``None`` if compilation
            conditions are not met.
        """
        if (
            prod_a.fire_count < self.compilation_threshold
            or prod_b.fire_count < self.compilation_threshold
        ):
            return None

        # Merge conditions: take prod_a's conditions, add prod_b's that
        # aren't satisfied by prod_a's actions.
        merged_conditions: Dict[str, Dict[str, Any]] = {}
        for buf, pat in prod_a.conditions.items():
            merged_conditions[buf] = dict(pat)
        for buf, pat in prod_b.conditions.items():
            if buf in prod_a.actions:
                # prod_a's action may already satisfy prod_b's condition
                action = prod_a.actions[buf]
                residual = {
                    k: v for k, v in pat.items()
                    if k not in action or action[k] != v
                }
                if residual:
                    existing = merged_conditions.get(buf, {})
                    existing.update(residual)
                    merged_conditions[buf] = existing
            else:
                existing = merged_conditions.get(buf, {})
                existing.update(pat)
                merged_conditions[buf] = existing

        # Merge actions: prod_a's actions overridden by prod_b's
        merged_actions: Dict[str, Dict[str, Any]] = {}
        for buf, act in prod_a.actions.items():
            merged_actions[buf] = dict(act)
        for buf, act in prod_b.actions.items():
            existing = merged_actions.get(buf, {})
            existing.update(act)
            merged_actions[buf] = existing

        compiled = Production(
            name=f"{prod_a.name}+{prod_b.name}",
            conditions=merged_conditions,
            actions=merged_actions,
            utility=(prod_a.utility + prod_b.utility) / 2.0,
            cost=prod_a.cost + prod_b.cost,
        )
        self.add_production(compiled)
        return compiled

    # ------------------------------------------------------------------ #
    # Procedural learning curve
    # ------------------------------------------------------------------ #

    @staticmethod
    def learning_curve(
        n_trials: NDArray[np.integer] | Sequence[int],
        initial_time: float = 2.0,
        learning_rate: float = 0.4,
    ) -> NDArray[np.floating]:
        """Predict production execution time across practice trials.

        Uses the power law of practice (Newell & Rosenbloom, 1981):

        .. math::

            T(n) = T_1 \\cdot n^{-\\alpha}

        Parameters
        ----------
        n_trials : array-like of int
            Trial numbers (>= 1).
        initial_time : float
            Time on the first trial *T₁* (seconds).
        learning_rate : float
            Power law exponent *α* (typically 0.2–0.6).

        Returns
        -------
        numpy.ndarray
            Predicted times for each trial.
        """
        n = np.asarray(n_trials, dtype=np.float64)
        n = np.maximum(n, 1.0)
        return initial_time * np.power(n, -learning_rate)

    # ------------------------------------------------------------------ #
    # Simulation step
    # ------------------------------------------------------------------ #

    def step(
        self,
        state: BufferState,
    ) -> Tuple[BufferState, Optional[Production], float]:
        """Execute one production-system cycle.

        1. Match all productions against *state*.
        2. Select the best via conflict resolution.
        3. Fire the selected production.

        Parameters
        ----------
        state : BufferState
            Current buffer state.

        Returns
        -------
        tuple[BufferState, Production | None, float]
            (new_state, fired_production, cycle_time).  If no production
            matches, the state is unchanged and cycle_time equals the
            default firing time.
        """
        prod = self.select_production(state)
        if prod is None:
            return state, None, self.production_firing_time

        new_state, fire_time = self.fire(prod, state)
        return new_state, prod, fire_time

    def run(
        self,
        state: BufferState,
        max_cycles: int = 100,
    ) -> Tuple[BufferState, List[Production], float]:
        """Run the production system until no productions match.

        Parameters
        ----------
        state : BufferState
            Initial buffer state.
        max_cycles : int
            Safety limit on the number of cycles.

        Returns
        -------
        tuple[BufferState, list[Production], float]
            (final_state, fired_productions, total_time).
        """
        fired: List[Production] = []
        total_time = 0.0

        for _ in range(max_cycles):
            new_state, prod, dt = self.step(state)
            total_time += dt
            if prod is None:
                break
            fired.append(prod)
            state = new_state

        return state, fired, total_time
