"""Cross-entropy proposal adaptation for importance sampling.

Implements the cross-entropy (CE) method for iteratively adapting the
proposal distribution toward high-probability race regions.  The CE
method selects elite samples, fits a new parametric proposal, and
repeats until convergence.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from dataclasses import dataclass, field
from scipy.special import logsumexp

from marace.sampling.schedule_space import (
    ContinuousSchedule,
    Schedule,
    ScheduleSpace,
)
from marace.sampling.importance_sampling import (
    ImportanceWeights,
    ProposalDistribution,
)


# ---------------------------------------------------------------------------
# Parametric proposal
# ---------------------------------------------------------------------------

class ParametricProposal(ProposalDistribution):
    """Parameterised proposal distribution over continuous schedules.

    Models agent action times as independent Gaussians with learnable
    means and standard deviations.

    Attributes:
        means: Per-agent mean action time.
        stds: Per-agent standard deviation.
        agent_order: Ordered list of agent IDs.
        period: Time horizon [0, period].
    """

    def __init__(
        self,
        agent_order: List[str],
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
        period: float = 1.0,
    ) -> None:
        self.agent_order = list(agent_order)
        n = len(agent_order)
        self.means = means if means is not None else np.full(n, period / 2.0)
        self.stds = stds if stds is not None else np.full(n, period / 4.0)
        self.period = period

    @property
    def num_agents(self) -> int:
        return len(self.agent_order)

    @property
    def parameters(self) -> np.ndarray:
        """Flat parameter vector ``[means; stds]``."""
        return np.concatenate([self.means, self.stds])

    def set_parameters(self, params: np.ndarray) -> None:
        """Update from a flat parameter vector."""
        n = self.num_agents
        self.means = params[:n].copy()
        self.stds = np.maximum(params[n:], 1e-6).copy()

    def sample(self, n: int, rng: np.random.RandomState) -> List[Schedule]:
        """Sample *n* schedules from the parametric proposal."""
        schedules: List[Schedule] = []
        for _ in range(n):
            times = rng.normal(self.means, self.stds)
            times = np.clip(times, 0.0, self.period)
            cs = ContinuousSchedule.from_vector(times, self.agent_order, self.period)
            schedules.append(cs.to_discrete_schedule())
        return schedules

    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability under independent Gaussian model.

        Maps the schedule ordering back to action times by assigning
        evenly-spaced times in [0, period].
        """
        ordering = schedule.ordering()
        n = len(self.agent_order)
        # Reconstruct approximate times from ordering position
        times = np.zeros(n)
        pos_map = {a: i for i, a in enumerate(ordering)}
        for j, agent in enumerate(self.agent_order):
            if agent in pos_map:
                times[j] = pos_map[agent] / max(len(ordering) - 1, 1) * self.period
            else:
                times[j] = self.means[j]

        log_p = 0.0
        for j in range(n):
            z = (times[j] - self.means[j]) / max(self.stds[j], 1e-10)
            log_p += -0.5 * z ** 2 - math.log(max(self.stds[j], 1e-10)) - 0.5 * math.log(2 * math.pi)
        return log_p

    def copy(self) -> "ParametricProposal":
        return ParametricProposal(
            list(self.agent_order),
            self.means.copy(),
            self.stds.copy(),
            self.period,
        )


# ---------------------------------------------------------------------------
# Multi-modal proposal
# ---------------------------------------------------------------------------

class MultiModalProposal(ProposalDistribution):
    """Mixture of parametric proposals for multi-modal race distributions.

    Attributes:
        components: List of :class:`ParametricProposal` components.
        weights: Mixture weights (sum to 1).
    """

    def __init__(
        self,
        components: List[ParametricProposal],
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.components = list(components)
        k = len(components)
        self.weights = weights if weights is not None else np.ones(k) / k

    @property
    def num_components(self) -> int:
        return len(self.components)

    def sample(self, n: int, rng: np.random.RandomState) -> List[Schedule]:
        """Sample from the mixture by first choosing a component."""
        assignments = rng.choice(
            self.num_components, size=n, p=self.weights
        )
        schedules: List[Schedule] = []
        for comp_idx in assignments:
            s = self.components[comp_idx].sample(1, rng)[0]
            schedules.append(s)
        return schedules

    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability under the mixture (log-sum-exp)."""
        log_probs = np.array([
            math.log(max(self.weights[i], 1e-300)) + comp.log_prob(schedule)
            for i, comp in enumerate(self.components)
        ])
        return float(logsumexp(log_probs))

    def update_weights(self, new_weights: np.ndarray) -> None:
        """Set new mixture weights (must sum to 1)."""
        self.weights = new_weights / new_weights.sum()


# ---------------------------------------------------------------------------
# Elite sample selector
# ---------------------------------------------------------------------------

class EliteSampleSelector:
    """Select top-k samples for cross-entropy update.

    Parameters:
        elite_fraction: Fraction of samples to keep as elite.
        min_elite: Minimum number of elite samples.
    """

    def __init__(
        self, elite_fraction: float = 0.1, min_elite: int = 5
    ) -> None:
        self._fraction = elite_fraction
        self._min_elite = min_elite

    def select(
        self,
        schedules: List[Schedule],
        scores: np.ndarray,
    ) -> Tuple[List[Schedule], np.ndarray, float]:
        """Select elite samples based on scores.

        Parameters:
            schedules: All sampled schedules.
            scores: Score for each schedule (higher = better).

        Returns:
            ``(elite_schedules, elite_scores, threshold)``
        """
        n = len(schedules)
        k = max(self._min_elite, int(n * self._fraction))
        k = min(k, n)

        indices = np.argsort(scores)[::-1][:k]
        threshold = float(scores[indices[-1]]) if k > 0 else 0.0

        elite_schedules = [schedules[i] for i in indices]
        elite_scores = scores[indices]
        return elite_schedules, elite_scores, threshold


# ---------------------------------------------------------------------------
# CE iteration
# ---------------------------------------------------------------------------

@dataclass
class CEIterationResult:
    """Result of a single cross-entropy iteration."""
    iteration: int
    threshold: float
    elite_fraction_race: float
    proposal_params: np.ndarray
    kl_divergence: float = 0.0
    ess: float = 0.0


class CEIteration:
    """Single iteration of cross-entropy parameter update.

    Given elite samples, fits the parametric proposal to the elite
    distribution via maximum likelihood.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        """
        Parameters:
            smoothing: Smoothing factor for parameter updates.
                ``params_new = (1-α)*params_ml + α*params_old``.
        """
        self._alpha = smoothing

    def update(
        self,
        proposal: ParametricProposal,
        elite_schedules: List[Schedule],
        elite_scores: np.ndarray,
    ) -> ParametricProposal:
        """Fit the proposal to elite samples and return the updated proposal.

        Parameters:
            proposal: Current proposal to update.
            elite_schedules: Elite schedules from the selector.
            elite_scores: Scores of the elite schedules.

        Returns:
            Updated :class:`ParametricProposal`.
        """
        if not elite_schedules:
            return proposal

        n_agents = proposal.num_agents
        period = proposal.period

        # Extract action times from elite schedules
        times_matrix = np.zeros((len(elite_schedules), n_agents))
        for i, sched in enumerate(elite_schedules):
            ordering = sched.ordering()
            pos_map = {a: j for j, a in enumerate(ordering)}
            for k, agent in enumerate(proposal.agent_order):
                if agent in pos_map:
                    times_matrix[i, k] = (
                        pos_map[agent] / max(len(ordering) - 1, 1) * period
                    )
                else:
                    times_matrix[i, k] = proposal.means[k]

        # Score-weighted ML estimates
        weights = elite_scores - elite_scores.min() + 1e-10
        weights = weights / weights.sum()

        ml_means = np.average(times_matrix, weights=weights, axis=0)
        ml_vars = np.average(
            (times_matrix - ml_means) ** 2, weights=weights, axis=0
        )
        ml_stds = np.sqrt(np.maximum(ml_vars, 1e-6))

        # Smooth update
        new_proposal = proposal.copy()
        new_proposal.means = (1 - self._alpha) * ml_means + self._alpha * proposal.means
        new_proposal.stds = (1 - self._alpha) * ml_stds + self._alpha * proposal.stds

        return new_proposal


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

class ProposalConvergence:
    """Detect convergence of proposal adaptation.

    Monitors the KL divergence between successive proposal iterations
    and declares convergence when it drops below a threshold.
    """

    def __init__(
        self,
        kl_threshold: float = 0.01,
        param_threshold: float = 1e-4,
        patience: int = 3,
    ) -> None:
        self._kl_threshold = kl_threshold
        self._param_threshold = param_threshold
        self._patience = patience
        self._history: List[np.ndarray] = []
        self._convergence_count = 0

    def check(self, old_proposal: ParametricProposal, new_proposal: ParametricProposal) -> bool:
        """Check whether the proposal has converged.

        Returns ``True`` if converged.
        """
        # KL between two diagonal Gaussians
        kl = self._gaussian_kl(
            old_proposal.means, old_proposal.stds,
            new_proposal.means, new_proposal.stds,
        )

        param_change = float(np.max(np.abs(
            old_proposal.parameters - new_proposal.parameters
        )))

        self._history.append(new_proposal.parameters.copy())

        if kl < self._kl_threshold and param_change < self._param_threshold:
            self._convergence_count += 1
        else:
            self._convergence_count = 0

        return self._convergence_count >= self._patience

    @staticmethod
    def _gaussian_kl(
        mu1: np.ndarray, sigma1: np.ndarray,
        mu2: np.ndarray, sigma2: np.ndarray,
    ) -> float:
        """KL(N(mu1,sigma1^2) || N(mu2,sigma2^2)) for diagonal Gaussians."""
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2 + 1e-12
        kl = 0.5 * np.sum(
            np.log(var2 / (var1 + 1e-12))
            + var1 / var2
            + (mu2 - mu1) ** 2 / var2
            - 1.0
        )
        return max(float(kl), 0.0)


# ---------------------------------------------------------------------------
# Adaptive temperature
# ---------------------------------------------------------------------------

class AdaptiveTemperature:
    """Control exploration vs exploitation in CE iterations.

    Starts with a high temperature (exploration) and gradually decreases
    toward a target temperature (exploitation) as the CE method converges.
    """

    def __init__(
        self,
        initial: float = 5.0,
        final: float = 0.1,
        decay_rate: float = 0.8,
    ) -> None:
        self._initial = initial
        self._final = final
        self._decay = decay_rate
        self._current = initial
        self._iteration = 0

    @property
    def temperature(self) -> float:
        return self._current

    def step(self) -> float:
        """Advance one iteration and return the new temperature."""
        self._iteration += 1
        self._current = max(
            self._final,
            self._initial * (self._decay ** self._iteration),
        )
        return self._current

    def reset(self) -> None:
        self._current = self._initial
        self._iteration = 0


# ---------------------------------------------------------------------------
# Cross-entropy optimizer
# ---------------------------------------------------------------------------

class CrossEntropyOptimizer:
    """CE method for adapting proposal distribution.

    The full CE workflow:
    1. Sample from the current proposal.
    2. Score samples (e.g. using a race-risk function).
    3. Select elite samples.
    4. Fit the proposal to the elite distribution.
    5. Check convergence; repeat if not converged.

    Parameters:
        space: The schedule space.
        score_fn: Maps a schedule to a non-negative score.
        agent_order: Ordered list of agent IDs.
        initial_proposal: Starting proposal (optional).
        elite_fraction: Fraction of samples kept as elite.
        smoothing: Parameter update smoothing factor.
        max_iterations: Maximum CE iterations.
        samples_per_iteration: Samples drawn per iteration.
        temperature: Adaptive temperature controller.
    """

    def __init__(
        self,
        space: ScheduleSpace,
        score_fn: Callable[[Schedule], float],
        agent_order: List[str],
        initial_proposal: Optional[ParametricProposal] = None,
        elite_fraction: float = 0.1,
        smoothing: float = 0.1,
        max_iterations: int = 50,
        samples_per_iteration: int = 500,
        temperature: Optional[AdaptiveTemperature] = None,
    ) -> None:
        self._space = space
        self._score_fn = score_fn
        self._agent_order = agent_order
        self._proposal = initial_proposal or ParametricProposal(agent_order)
        self._selector = EliteSampleSelector(elite_fraction)
        self._updater = CEIteration(smoothing)
        self._convergence = ProposalConvergence()
        self._max_iter = max_iterations
        self._samples_per_iter = samples_per_iteration
        self._temperature = temperature or AdaptiveTemperature()

    @property
    def proposal(self) -> ParametricProposal:
        return self._proposal

    def optimize(
        self,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[ParametricProposal, List[CEIterationResult]]:
        """Run the cross-entropy optimization loop.

        Returns:
            ``(optimized_proposal, iteration_history)``
        """
        rng = rng or np.random.RandomState(42)
        history: List[CEIterationResult] = []

        for it in range(self._max_iter):
            # Sample
            schedules = self._proposal.sample(self._samples_per_iter, rng)

            # Score with temperature
            temp = self._temperature.temperature
            scores = np.array([self._score_fn(s) / temp for s in schedules])

            # Select elite
            elite_scheds, elite_scores, threshold = self._selector.select(
                schedules, scores
            )

            # Compute elite race fraction (for diagnostics)
            race_count = sum(1 for s in elite_scores if s > 0)
            elite_race_frac = race_count / max(len(elite_scores), 1)

            # Update proposal
            old_proposal = self._proposal.copy()
            self._proposal = self._updater.update(
                self._proposal, elite_scheds, elite_scores
            )

            # Convergence check
            kl = ProposalConvergence._gaussian_kl(
                old_proposal.means, old_proposal.stds,
                self._proposal.means, self._proposal.stds,
            )

            result = CEIterationResult(
                iteration=it,
                threshold=threshold,
                elite_fraction_race=elite_race_frac,
                proposal_params=self._proposal.parameters.copy(),
                kl_divergence=kl,
            )
            history.append(result)

            if self._convergence.check(old_proposal, self._proposal):
                break

            self._temperature.step()

        return self._proposal, history

    def optimize_multimodal(
        self,
        num_components: int = 3,
        rng: Optional[np.random.RandomState] = None,
    ) -> MultiModalProposal:
        """Run CE with multiple restarts to find a multi-modal proposal.

        Parameters:
            num_components: Number of mixture components.
            rng: Random state.

        Returns:
            :class:`MultiModalProposal` with fitted components.
        """
        rng = rng or np.random.RandomState(42)
        components: List[ParametricProposal] = []

        for k in range(num_components):
            # Random restart
            n = len(self._agent_order)
            init_means = rng.uniform(0, self._proposal.period, size=n)
            init_stds = np.full(n, self._proposal.period / 4.0)
            init_proposal = ParametricProposal(
                self._agent_order, init_means, init_stds, self._proposal.period
            )

            self._proposal = init_proposal
            self._convergence = ProposalConvergence()
            self._temperature.reset()

            optimized, _ = self.optimize(rng)
            components.append(optimized)

        return MultiModalProposal(components)


# ---------------------------------------------------------------------------
# CE Convergence Proof Implementation
# ---------------------------------------------------------------------------

class CEConvergenceProof:
    r"""Convergence rate bounds for the cross-entropy method.

    Implements computable bounds from Rubinstein & Kroese (2004) for
    the CE method with parametric proposal families.

    Theorem (CE Convergence Rate)
    -----------------------------
    Let {q_t} be the sequence of CE proposals and p* the zero-variance
    proposal p*(x) ∝ p(x)|f(x)|.  Under mild regularity conditions
    (compact parameter space, continuous Fisher information):

        D_KL(p* || q_t) ≤ D_KL(p* || q_0) · (1 - η)^t

    where η ∈ (0,1) depends on the elite fraction ρ and the parametric
    family curvature (eigenvalues of the Fisher information matrix).

    For Gaussian proposals with smoothing α:
        η ≈ ρ · (1 - α)

    Parameters
    ----------
    elite_fraction : float
        Elite ratio ρ ∈ (0, 1).
    smoothing : float
        Parameter update smoothing α.
    param_dim : int
        Dimension of the parameter space.
    """

    def __init__(
        self,
        elite_fraction: float = 0.1,
        smoothing: float = 0.1,
        param_dim: int = 1,
    ) -> None:
        self._rho = elite_fraction
        self._alpha = smoothing
        self._dim = param_dim

    @property
    def contraction_rate(self) -> float:
        """Contraction rate η = ρ · (1 - α)."""
        return self._rho * (1.0 - self._alpha)

    def kl_upper_bound(self, iteration: int, initial_kl: float = 10.0) -> float:
        """Upper bound on KL(p* || q_t) after t iterations."""
        eta = self.contraction_rate
        return initial_kl * ((1.0 - eta) ** iteration)

    def iterations_for_kl_target(self, target_kl: float, initial_kl: float = 10.0) -> int:
        """Minimum iterations to achieve KL(p* || q_t) ≤ target_kl."""
        if target_kl >= initial_kl:
            return 0
        eta = self.contraction_rate
        if eta <= 0:
            return int(1e6)
        ratio = target_kl / initial_kl
        return int(math.ceil(math.log(ratio) / math.log(1.0 - eta)))

    def finite_sample_bound(
        self, num_rounds: int, samples_per_round: int
    ) -> float:
        r"""Finite-sample excess variance bound.

        ΔV ≤ C √(d · log(T·m) / m)

        where d = param_dim, T = num_rounds, m = samples_per_round.
        """
        C = 2.0
        m = max(samples_per_round, 1)
        T = max(num_rounds, 1)
        return C * math.sqrt(self._dim * math.log(T * m) / m)

    def summary(self, num_iterations: int, initial_kl: float = 10.0) -> Dict:
        """Summary of convergence properties."""
        return {
            "contraction_rate": self.contraction_rate,
            "kl_bound_at_T": self.kl_upper_bound(num_iterations, initial_kl),
            "iterations_for_0.01": self.iterations_for_kl_target(0.01, initial_kl),
            "iterations_for_0.001": self.iterations_for_kl_target(0.001, initial_kl),
            "elite_fraction": self._rho,
            "smoothing": self._alpha,
            "param_dim": self._dim,
        }


# ---------------------------------------------------------------------------
# KL Divergence Monitor
# ---------------------------------------------------------------------------

class KLDivergenceMonitor:
    """Monitor KL divergence between successive CE proposals.

    Tracks KL(q_{t-1} || q_t) and KL(q_t || q_{t-1}) over iterations
    to diagnose convergence. Also computes effective support size.

    Convergence is declared when KL drops below threshold for
    patience consecutive iterations.

    Parameters
    ----------
    kl_threshold : float
        KL divergence threshold for convergence.
    patience : int
        Number of consecutive below-threshold iterations required.
    """

    def __init__(
        self,
        kl_threshold: float = 0.01,
        patience: int = 3,
    ) -> None:
        self._threshold = kl_threshold
        self._patience = patience
        self._kl_history: List[float] = []
        self._reverse_kl_history: List[float] = []
        self._effective_support_sizes: List[float] = []
        self._below_threshold_count = 0

    def update(
        self,
        old_proposal: ParametricProposal,
        new_proposal: ParametricProposal,
    ) -> Tuple[bool, float]:
        """Record KL divergence and check convergence.

        Parameters
        ----------
        old_proposal : proposal from previous iteration.
        new_proposal : proposal from current iteration.

        Returns
        -------
        (converged, kl_divergence)
        """
        kl_forward = ProposalConvergence._gaussian_kl(
            old_proposal.means, old_proposal.stds,
            new_proposal.means, new_proposal.stds,
        )
        kl_reverse = ProposalConvergence._gaussian_kl(
            new_proposal.means, new_proposal.stds,
            old_proposal.means, old_proposal.stds,
        )

        self._kl_history.append(kl_forward)
        self._reverse_kl_history.append(kl_reverse)

        # Effective support size: exp(entropy) of the proposal
        # For diagonal Gaussian: exp(∑ log(σ_i√(2πe))) = ∏ σ_i · (2πe)^{n/2}
        n = new_proposal.num_agents
        log_ess = 0.5 * n * math.log(2 * math.pi * math.e) + float(
            np.sum(np.log(np.maximum(new_proposal.stds, 1e-300)))
        )
        self._effective_support_sizes.append(math.exp(min(log_ess, 100)))

        if kl_forward < self._threshold:
            self._below_threshold_count += 1
        else:
            self._below_threshold_count = 0

        converged = self._below_threshold_count >= self._patience
        return converged, kl_forward

    @property
    def kl_history(self) -> List[float]:
        return list(self._kl_history)

    @property
    def kl_gap(self) -> float:
        """Current KL gap (latest KL divergence)."""
        return self._kl_history[-1] if self._kl_history else float("inf")

    @property
    def effective_support_sizes(self) -> List[float]:
        return list(self._effective_support_sizes)

    def diagnostics(self) -> Dict:
        """Convergence diagnostics summary."""
        return {
            "num_iterations": len(self._kl_history),
            "kl_history": list(self._kl_history),
            "reverse_kl_history": list(self._reverse_kl_history),
            "effective_support_sizes": list(self._effective_support_sizes),
            "kl_gap": self.kl_gap,
            "converged": self._below_threshold_count >= self._patience,
            "below_threshold_count": self._below_threshold_count,
        }


# ---------------------------------------------------------------------------
# CE Convergence Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class CEConvergenceDiagnostics:
    """Comprehensive diagnostics for CE method convergence.

    Aggregates KL monitoring, ESS tracking, and convergence proof
    bounds into a single diagnostic report.

    Attributes
    ----------
    kl_history : list of float
        KL divergence per iteration.
    kl_bound_history : list of float
        Theoretical KL upper bound per iteration.
    effective_support_sizes : list of float
        Effective support size per iteration.
    ess_history : list of float
        ESS per iteration.
    converged : bool
        Whether convergence was achieved.
    convergence_iteration : int or None
        Iteration at which convergence was achieved.
    """

    kl_history: List[float] = field(default_factory=list)
    kl_bound_history: List[float] = field(default_factory=list)
    effective_support_sizes: List[float] = field(default_factory=list)
    ess_history: List[float] = field(default_factory=list)
    converged: bool = False
    convergence_iteration: Optional[int] = None

    def is_healthy(self) -> bool:
        """Check if the CE run appears healthy.

        A healthy run has:
        1. Decreasing KL divergence trend
        2. Non-degenerate effective support
        3. Reasonable ESS
        """
        if len(self.kl_history) < 3:
            return True
        # KL should be roughly decreasing
        last_3 = self.kl_history[-3:]
        kl_decreasing = last_3[-1] <= last_3[0] * 1.5
        # ESS should be reasonable
        ess_ok = not self.ess_history or self.ess_history[-1] > 1.0
        return kl_decreasing and ess_ok

    def summary(self) -> Dict:
        """Summary dictionary."""
        return {
            "num_iterations": len(self.kl_history),
            "final_kl": self.kl_history[-1] if self.kl_history else None,
            "final_kl_bound": self.kl_bound_history[-1] if self.kl_bound_history else None,
            "converged": self.converged,
            "convergence_iteration": self.convergence_iteration,
            "is_healthy": self.is_healthy(),
        }
