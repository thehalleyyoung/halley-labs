"""
Tau-leaping methods for accelerated stochastic simulation.

Provides explicit, implicit, and midpoint tau-leaping with adaptive step
size selection, negative population handling, and automatic SSA switching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

from .ssa import Reaction, StochasticState, TrajectoryRecorder


@dataclass
class LeapCondition:
    """Parameters for the leap condition (Cao et al. 2006).

    The leap condition bounds the relative change in each propensity
    during a leap: |delta a_j| / a_j <= epsilon.
    """

    epsilon: float = 0.03
    # Number of SSA steps before attempting tau-leap again
    n_ssa_steps: int = 100
    # Threshold: if a0*tau < n_critical, use SSA
    n_critical: int = 10


class AdaptiveTauSelector:
    """Selects tau adaptively using Cao et al.'s method.

    Computes the largest tau such that no propensity changes by more than
    epsilon * a0 during the leap, using auxiliary quantities mu_i and
    sigma_i^2.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        epsilon: float = 0.03,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.num_reactions = len(reactions)
        self.epsilon = epsilon
        # Precompute stoichiometry
        self._stoich = np.zeros((self.num_reactions, self.num_species), dtype=int)
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < self.num_species:
                    self._stoich[j, sp] = delta
        # Highest order of each reaction for each species
        self._hor = self._compute_highest_order_reactions()

    def _compute_highest_order_reactions(self) -> Dict[int, Tuple[int, int]]:
        """For each species, find the highest-order reaction it participates in.

        Returns dict mapping species index to (order, reaction_index).
        """
        hor: Dict[int, Tuple[int, int]] = {}
        for sp in range(self.num_species):
            max_order = 0
            max_rxn = -1
            for j, rxn in enumerate(self.reactions):
                if sp in rxn.reactants:
                    order = rxn.reactants[sp]
                    if order > max_order:
                        max_order = order
                        max_rxn = j
            if max_order > 0:
                hor[sp] = (max_order, max_rxn)
        return hor

    def select_tau(
        self, state: np.ndarray, propensities: np.ndarray
    ) -> float:
        """Select tau using Cao et al. adaptive method.

        Computes auxiliary quantities mu_i and sigma_i^2 for each species
        and returns the maximum tau satisfying the leap condition.
        """
        a0 = propensities.sum()
        if a0 <= 0:
            return np.inf

        tau_candidates = []
        for sp in range(self.num_species):
            if sp not in self._hor:
                continue
            hor_order, _ = self._hor[sp]
            x_i = max(float(state[sp]), 1.0)
            # gi factor for highest order reaction
            if hor_order == 1:
                g_i = 1.0
            elif hor_order == 2:
                g_i = 2.0 / (x_i - 1.0) if x_i > 1 else 2.0
            elif hor_order == 3:
                g_i = 3.0 / (x_i - 1.0) if x_i > 1 else 3.0
            else:
                g_i = float(hor_order) / max(x_i - 1.0, 1.0)

            # Compute mu_i = sum_j v_ji * a_j and sigma_i^2 = sum_j v_ji^2 * a_j
            mu_i = 0.0
            sigma_sq_i = 0.0
            for j in range(self.num_reactions):
                v_ji = self._stoich[j, sp]
                if v_ji != 0:
                    mu_i += v_ji * propensities[j]
                    sigma_sq_i += v_ji * v_ji * propensities[j]

            bound = max(self.epsilon * x_i / g_i, 1.0)
            if abs(mu_i) > 0:
                tau_mu = bound / abs(mu_i)
                tau_candidates.append(tau_mu)
            if sigma_sq_i > 0:
                tau_sigma = bound * bound / sigma_sq_i
                tau_candidates.append(tau_sigma)

        if not tau_candidates:
            return np.inf
        return min(tau_candidates)


class StepSizeController:
    """Adaptive step size controller with acceptance/rejection.

    Adjusts tau based on whether steps produce negative populations
    or if propensity changes are too large.
    """

    def __init__(
        self,
        tau_init: float = 0.01,
        safety_factor: float = 0.9,
        min_tau: float = 1e-12,
        max_tau: float = 1e6,
        growth_factor: float = 1.5,
        shrink_factor: float = 0.5,
    ):
        self.tau = tau_init
        self.safety_factor = safety_factor
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self._rejections = 0
        self._accepts = 0

    def accept(self):
        """Record a successful step."""
        self._accepts += 1
        self._rejections = 0
        self.tau = min(self.tau * self.growth_factor, self.max_tau)

    def reject(self):
        """Record a rejected step and shrink tau."""
        self._rejections += 1
        self.tau = max(self.tau * self.shrink_factor, self.min_tau)

    def propose(self, adaptive_tau: float) -> float:
        """Return the step size to use, clamped to [min_tau, max_tau]."""
        proposed = min(self.tau, adaptive_tau) * self.safety_factor
        return max(self.min_tau, min(proposed, self.max_tau))

    @property
    def rejection_rate(self) -> float:
        total = self._accepts + self._rejections
        return self._rejections / total if total > 0 else 0.0


class ExplicitTauLeaping:
    """Explicit tau-leaping with Poisson increments.

    Parameters:
        reactions: List of Reaction objects.
        num_species: Number of species.
        epsilon: Leap condition bound.
        seed: Random seed.
        negative_handling: 'reject' to reject and retry, 'postleap' to
            reduce to valid state.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        epsilon: float = 0.03,
        seed: Optional[int] = None,
        negative_handling: str = "reject",
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.num_reactions = len(reactions)
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.negative_handling = negative_handling
        self.tau_selector = AdaptiveTauSelector(reactions, num_species, epsilon)
        self.step_controller = StepSizeController()
        self._stoich = np.zeros((self.num_reactions, self.num_species), dtype=int)
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < self.num_species:
                    self._stoich[j, sp] = delta

    def _compute_propensities(self, state: np.ndarray) -> np.ndarray:
        props = np.empty(self.num_reactions)
        for j, rxn in enumerate(self.reactions):
            props[j] = rxn.propensity(state)
        return props

    def _postleap_check(
        self,
        state: np.ndarray,
        firings: np.ndarray,
    ) -> np.ndarray:
        """Reduce firings to avoid negative populations.

        Iteratively scales down the largest-firing reactions until
        all populations are non-negative.
        """
        new_state = state.copy()
        adjusted_firings = firings.copy()
        for _ in range(100):
            delta = self._stoich.T @ adjusted_firings
            candidate = new_state + delta
            if np.all(candidate >= 0):
                return adjusted_firings
            # Find species with most negative count
            neg_species = np.where(candidate < 0)[0]
            for sp in neg_species:
                # Find reactions that decrease this species
                for j in range(self.num_reactions):
                    if self._stoich[j, sp] < 0 and adjusted_firings[j] > 0:
                        max_allowed = int(new_state[sp] / abs(self._stoich[j, sp]))
                        for other_j in range(self.num_reactions):
                            if (
                                other_j != j
                                and self._stoich[other_j, sp] < 0
                                and adjusted_firings[other_j] > 0
                            ):
                                max_allowed = max(max_allowed, 0)
                        adjusted_firings[j] = min(adjusted_firings[j], max(max_allowed, 0))
        return adjusted_firings

    def _leap_step(
        self, state: np.ndarray, tau: float, propensities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Execute one tau-leap step.

        Returns (new_state, firings, accepted).
        """
        # Generate Poisson random numbers for each reaction
        lambdas = propensities * tau
        lambdas = np.maximum(lambdas, 0.0)
        firings = np.zeros(self.num_reactions, dtype=np.int64)
        for j in range(self.num_reactions):
            if lambdas[j] > 0:
                firings[j] = self.rng.poisson(lambdas[j])

        if self.negative_handling == "postleap":
            firings = self._postleap_check(state, firings)
            delta = self._stoich.T @ firings
            new_state = state + delta
            return new_state, firings, True

        # Default: rejection
        delta = self._stoich.T @ firings
        new_state = state + delta
        if np.any(new_state < 0):
            return state, firings, False
        return new_state, firings, True

    def simulate(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        max_steps: int = 1_000_000,
        recorder: Optional[TrajectoryRecorder] = None,
    ) -> StochasticState:
        """Run tau-leaping simulation."""
        state = initial_state.astype(np.int64).copy()
        t = t_start

        if recorder is not None:
            recorder.record_initial(t, state)

        for step in range(max_steps):
            if t >= t_end:
                break
            propensities = self._compute_propensities(state)
            a0 = propensities.sum()
            if a0 <= 0:
                break
            adaptive_tau = self.tau_selector.select_tau(state, propensities)
            tau = self.step_controller.propose(adaptive_tau)
            tau = min(tau, t_end - t)

            new_state, firings, accepted = self._leap_step(state, tau, propensities)
            if accepted:
                state = new_state
                t += tau
                self.step_controller.accept()
                if recorder is not None:
                    recorder.record(t, state)
            else:
                self.step_controller.reject()

        return StochasticState(time=t, copy_numbers=state.copy())


class ImplicitTauLeaping:
    """Implicit tau-leaping for stiff stochastic systems.

    Solves the implicit equation:
        X(t+tau) = X(t) + sum_j v_j * Poisson(a_j(X(t)) * tau)
                   + sum_j v_j * [a_j(X(t+tau)) - a_j(X(t))] * tau

    Uses Newton iteration to solve the implicit system.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        epsilon: float = 0.03,
        seed: Optional[int] = None,
        newton_tol: float = 1e-8,
        max_newton_iter: int = 50,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.num_reactions = len(reactions)
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.newton_tol = newton_tol
        self.max_newton_iter = max_newton_iter
        self.tau_selector = AdaptiveTauSelector(reactions, num_species, epsilon)
        self._stoich = np.zeros((self.num_reactions, self.num_species), dtype=int)
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < self.num_species:
                    self._stoich[j, sp] = delta

    def _compute_propensities(self, state: np.ndarray) -> np.ndarray:
        props = np.empty(self.num_reactions)
        for j, rxn in enumerate(self.reactions):
            props[j] = rxn.propensity(state)
        return props

    def _implicit_residual(
        self,
        x_new: np.ndarray,
        x_old: np.ndarray,
        poisson_increments: np.ndarray,
        tau: float,
        propensities_old: np.ndarray,
    ) -> np.ndarray:
        """Residual of the implicit tau-leaping equation."""
        # Explicit Poisson part
        explicit_part = self._stoich.T @ poisson_increments
        # Implicit correction
        props_new = np.empty(self.num_reactions)
        for j, rxn in enumerate(self.reactions):
            props_new[j] = rxn.propensity(np.maximum(x_new, 0))
        implicit_correction = self._stoich.T @ ((props_new - propensities_old) * tau)
        return x_new - x_old - explicit_part - implicit_correction

    def _solve_implicit(
        self,
        x_old: np.ndarray,
        poisson_increments: np.ndarray,
        tau: float,
        propensities_old: np.ndarray,
    ) -> np.ndarray:
        """Solve implicit equation via Newton iteration."""
        # Initial guess: explicit tau-leaping result
        x_guess = x_old + self._stoich.T @ poisson_increments
        x_guess = np.maximum(x_guess.astype(float), 0.0)

        for iteration in range(self.max_newton_iter):
            residual = self._implicit_residual(
                x_guess, x_old, poisson_increments, tau, propensities_old
            )
            if np.linalg.norm(residual) < self.newton_tol:
                break
            # Approximate Jacobian numerically
            n = len(x_guess)
            jac = np.eye(n)
            eps_fd = 1e-6
            for i in range(n):
                x_pert = x_guess.copy()
                x_pert[i] += eps_fd
                res_pert = self._implicit_residual(
                    x_pert, x_old, poisson_increments, tau, propensities_old
                )
                jac[:, i] = (res_pert - residual) / eps_fd
            try:
                delta = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                break
            x_guess += delta
            x_guess = np.maximum(x_guess, 0.0)

        return np.round(x_guess).astype(np.int64)

    def simulate(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        max_steps: int = 1_000_000,
        recorder: Optional[TrajectoryRecorder] = None,
    ) -> StochasticState:
        state = initial_state.astype(np.int64).copy()
        t = t_start

        if recorder is not None:
            recorder.record_initial(t, state)

        for step in range(max_steps):
            if t >= t_end:
                break
            propensities = self._compute_propensities(state)
            a0 = propensities.sum()
            if a0 <= 0:
                break
            tau = self.tau_selector.select_tau(state, propensities)
            tau = min(tau, t_end - t)

            lambdas = np.maximum(propensities * tau, 0.0)
            poisson_increments = np.array(
                [self.rng.poisson(lam) if lam > 0 else 0 for lam in lambdas],
                dtype=np.int64,
            )
            state = self._solve_implicit(
                state, poisson_increments, tau, propensities
            )
            t += tau
            if recorder is not None:
                recorder.record(t, state)

        return StochasticState(time=t, copy_numbers=state.copy())


class MidpointTauLeaping:
    """Midpoint tau-leaping for improved accuracy (second-order).

    Takes a half-step to estimate midpoint propensities, then uses those
    for the full Poisson increments.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        epsilon: float = 0.03,
        seed: Optional[int] = None,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.num_reactions = len(reactions)
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.tau_selector = AdaptiveTauSelector(reactions, num_species, epsilon)
        self._stoich = np.zeros((self.num_reactions, self.num_species), dtype=int)
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < self.num_species:
                    self._stoich[j, sp] = delta

    def _compute_propensities(self, state: np.ndarray) -> np.ndarray:
        props = np.empty(self.num_reactions)
        for j, rxn in enumerate(self.reactions):
            props[j] = rxn.propensity(np.maximum(state, 0))
        return props

    def simulate(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        max_steps: int = 1_000_000,
        recorder: Optional[TrajectoryRecorder] = None,
    ) -> StochasticState:
        state = initial_state.astype(np.int64).copy()
        t = t_start

        if recorder is not None:
            recorder.record_initial(t, state)

        for step in range(max_steps):
            if t >= t_end:
                break
            propensities = self._compute_propensities(state)
            a0 = propensities.sum()
            if a0 <= 0:
                break
            tau = self.tau_selector.select_tau(state, propensities)
            tau = min(tau, t_end - t)

            # Half step: deterministic midpoint estimate
            half_delta = self._stoich.T @ (propensities * tau / 2.0)
            midpoint_state = state + np.round(half_delta).astype(np.int64)
            midpoint_state = np.maximum(midpoint_state, 0)

            # Midpoint propensities
            midpoint_props = self._compute_propensities(midpoint_state)

            # Full step with midpoint propensities
            lambdas = np.maximum(midpoint_props * tau, 0.0)
            firings = np.array(
                [self.rng.poisson(lam) if lam > 0 else 0 for lam in lambdas],
                dtype=np.int64,
            )
            delta = self._stoich.T @ firings
            new_state = state + delta
            if np.any(new_state < 0):
                # Fallback: clamp at zero
                new_state = np.maximum(new_state, 0)
            state = new_state
            t += tau
            if recorder is not None:
                recorder.record(t, state)

        return StochasticState(time=t, copy_numbers=state.copy())


class SSATauLeapingSwitch:
    """Automatic switching between SSA and tau-leaping.

    Uses SSA when propensity magnitudes are small (stiff regime or
    near extinction) and tau-leaping when propensities are large enough.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        epsilon: float = 0.03,
        seed: Optional[int] = None,
        n_critical: int = 10,
        ssa_steps_between_checks: int = 100,
        negative_handling: str = "reject",
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.num_reactions = len(reactions)
        self.n_critical = n_critical
        self.ssa_steps_between_checks = ssa_steps_between_checks
        base_seed = seed if seed is not None else None
        rng = np.random.default_rng(base_seed)
        seed1, seed2 = rng.integers(0, 2**31, size=2)
        self.ssa = _MinimalSSA(reactions, num_species, seed=int(seed1))
        self.tau_leaper = ExplicitTauLeaping(
            reactions, num_species, epsilon, seed=int(seed2),
            negative_handling=negative_handling,
        )
        self.tau_selector = AdaptiveTauSelector(reactions, num_species, epsilon)

    def _should_use_ssa(
        self, state: np.ndarray, propensities: np.ndarray
    ) -> bool:
        a0 = propensities.sum()
        if a0 <= 0:
            return True
        tau = self.tau_selector.select_tau(state, propensities)
        expected_firings = a0 * tau
        return expected_firings < self.n_critical

    def simulate(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        max_steps: int = 10_000_000,
        recorder: Optional[TrajectoryRecorder] = None,
    ) -> StochasticState:
        state = initial_state.astype(np.int64).copy()
        t = t_start
        if recorder is not None:
            recorder.record_initial(t, state)

        step = 0
        while step < max_steps and t < t_end:
            propensities = self.tau_leaper._compute_propensities(state)
            a0 = propensities.sum()
            if a0 <= 0:
                break

            if self._should_use_ssa(state, propensities):
                # Do a burst of SSA steps
                for _ in range(self.ssa_steps_between_checks):
                    if t >= t_end:
                        break
                    dt, j = self.ssa.step(state, propensities)
                    if dt is None:
                        break
                    if t + dt > t_end:
                        break
                    t += dt
                    state += self.ssa._stoich[j]
                    propensities = self.tau_leaper._compute_propensities(state)
                    if recorder is not None:
                        recorder.record(t, state, j)
                    step += 1
            else:
                tau = self.tau_selector.select_tau(state, propensities)
                tau = min(tau, t_end - t)
                new_state, _, accepted = self.tau_leaper._leap_step(
                    state, tau, propensities
                )
                if accepted:
                    state = new_state
                    t += tau
                    if recorder is not None:
                        recorder.record(t, state)
                step += 1

        return StochasticState(time=t, copy_numbers=state.copy())


class _MinimalSSA:
    """Minimal SSA helper for hybrid switching (no trajectory management)."""

    def __init__(
        self, reactions: List[Reaction], num_species: int, seed: int
    ):
        self.reactions = reactions
        self.num_reactions = len(reactions)
        self.rng = np.random.default_rng(seed)
        self._stoich = np.zeros((len(reactions), num_species), dtype=int)
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < num_species:
                    self._stoich[j, sp] = delta

    def step(
        self, state: np.ndarray, propensities: np.ndarray
    ) -> Tuple[Optional[float], int]:
        a0 = propensities.sum()
        if a0 <= 0:
            return None, -1
        u1 = self.rng.random()
        while u1 == 0.0:
            u1 = self.rng.random()
        dt = -math.log(u1) / a0
        u2 = self.rng.random() * a0
        cumsum = 0.0
        j = 0
        for j in range(self.num_reactions):
            cumsum += propensities[j]
            if cumsum >= u2:
                break
        return dt, j
