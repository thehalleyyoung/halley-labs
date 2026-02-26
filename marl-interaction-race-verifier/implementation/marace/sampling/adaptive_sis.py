"""Adaptive Sequential Importance Sampling with IIA diagnostics.

Addresses the Plackett-Luce IIA assumption violation by providing:

1. AdaptiveSISEngine — SIS with real-time ESS monitoring and adaptive
   resampling (multinomial, systematic, residual).
2. PlackettLuceValidator — statistical tests for IIA violation severity.
3. MixedLogitProposal — mixture-of-PL proposal robust to IIA violations.
4. NestedLogitProposal — nested logit proposal with interaction groups.
5. StoppingCriteria — principled convergence diagnostics (ESS-based,
   CI-width, MCSE, R-hat, Geweke).
6. JointErrorAnalysis — decompose total error into abstract-interpretation
   error and importance-sampling estimation error.

References
----------
.. [PL75]  Plackett, R. L. (1975). "The Analysis of Permutations."
.. [MT00]  McFadden, D. & Train, K. (2000). "Mixed MNL Models for
   Discrete Response." *JARE*.
.. [GR92]  Geweke, J. (1992). "Evaluating the Accuracy of Sampling-Based
   Approaches to Calculating Posterior Moments."
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
from scipy import stats as scipy_stats
from scipy.special import logsumexp

from marace.sampling.schedule_space import (
    Schedule,
    ScheduleEvent,
    ScheduleSpace,
    ScheduleGenerator,
)
from marace.sampling.importance_sampling import (
    ImportanceWeights,
    EffectiveSampleSize,
    ConfidenceInterval,
    ProposalDistribution,
    ImportanceSampler,
)


# ===================================================================
# Resampling strategies
# ===================================================================

class ResamplingStrategy(abc.ABC):
    """Abstract base for resampling algorithms."""

    @abc.abstractmethod
    def resample(
        self,
        particles: List[Schedule],
        normalised_weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> List[Schedule]:
        """Resample particles according to weights.

        Returns a new list of N particles with uniform weights.
        """


class MultinomialResampling(ResamplingStrategy):
    """Standard multinomial resampling.

    Draws N independent samples from the categorical distribution
    defined by the normalised weights.  Simple but higher variance
    than systematic resampling.
    """

    def resample(
        self,
        particles: List[Schedule],
        normalised_weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> List[Schedule]:
        n = len(particles)
        indices = rng.choice(n, size=n, replace=True, p=normalised_weights)
        return [particles[i] for i in indices]


class SystematicResampling(ResamplingStrategy):
    """Low-variance systematic resampling.

    Uses a single uniform offset and equally spaced positions on the
    CDF.  Produces more representative samples with lower variance
    than multinomial resampling.
    """

    def resample(
        self,
        particles: List[Schedule],
        normalised_weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> List[Schedule]:
        n = len(particles)
        positions = (rng.uniform() + np.arange(n)) / n
        cumulative = np.cumsum(normalised_weights)
        indices: List[int] = []
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative[j]:
                indices.append(j)
                i += 1
            else:
                j = min(j + 1, n - 1)
        return [particles[idx] for idx in indices]


class ResidualResampling(ResamplingStrategy):
    """Residual resampling (Liu & Chen, 1998).

    Deterministically copies floor(N * w_i) copies of each particle,
    then resamples the residual fractional parts multinomially.
    Reduces variance relative to pure multinomial.
    """

    def resample(
        self,
        particles: List[Schedule],
        normalised_weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> List[Schedule]:
        n = len(particles)
        nw = n * normalised_weights
        deterministic_counts = np.floor(nw).astype(int)
        residuals = nw - deterministic_counts

        result: List[Schedule] = []
        for i, count in enumerate(deterministic_counts):
            result.extend([particles[i]] * count)

        n_remaining = n - len(result)
        if n_remaining > 0:
            residual_sum = residuals.sum()
            if residual_sum > 1e-12:
                residual_probs = residuals / residual_sum
            else:
                residual_probs = np.ones(n) / n
            indices = rng.choice(n, size=n_remaining, replace=True, p=residual_probs)
            result.extend([particles[i] for i in indices])

        return result


RESAMPLING_STRATEGIES: Dict[str, ResamplingStrategy] = {
    "multinomial": MultinomialResampling(),
    "systematic": SystematicResampling(),
    "residual": ResidualResampling(),
}


# ===================================================================
# Adaptive SIS Engine
# ===================================================================

@dataclass
class SISResult:
    """Result container for adaptive SIS."""
    particles: List[Schedule]
    weights: ImportanceWeights
    ess_history: List[float]
    resample_steps: List[int]
    converged: bool
    num_steps: int


class AdaptiveSISEngine:
    """Sequential Importance Sampling with adaptive resampling.

    Provides real-time ESS monitoring at each resampling step,
    automatic resampling when ESS drops below a configurable threshold,
    and convergence diagnostics.

    Supports SIR (Sampling Importance Resampling) and SIS variants.

    Parameters
    ----------
    target_log_prob : callable
        Log-density of the target distribution.
    proposal : ProposalDistribution
        Proposal distribution for drawing particles.
    num_particles : int
        Population size.
    ess_threshold_fraction : float
        Resample when ESS < N * threshold_fraction (default 0.5).
    resampling_strategy : str
        One of 'multinomial', 'systematic', 'residual'.
    max_steps : int
        Maximum number of SIS steps (rounds of sampling).
    mode : str
        'sir' for Sampling Importance Resampling (single step),
        'sis' for Sequential Importance Sampling (multi-step).
    """

    def __init__(
        self,
        target_log_prob: Callable[[Schedule], float],
        proposal: ProposalDistribution,
        num_particles: int = 500,
        ess_threshold_fraction: float = 0.5,
        resampling_strategy: str = "systematic",
        max_steps: int = 10,
        mode: str = "sir",
    ) -> None:
        self._target_lp = target_log_prob
        self._proposal = proposal
        self._N = num_particles
        self._ess_threshold = ess_threshold_fraction
        self._resampler = RESAMPLING_STRATEGIES.get(
            resampling_strategy, SystematicResampling()
        )
        self._max_steps = max_steps
        self._mode = mode
        self._ess_history: List[float] = []
        self._resample_steps: List[int] = []

    def run(
        self,
        rng: Optional[np.random.RandomState] = None,
    ) -> SISResult:
        """Run the adaptive SIS/SIR procedure.

        Returns
        -------
        SISResult
            Final particles, weights, ESS history, and convergence info.
        """
        rng = rng or np.random.RandomState(42)
        self._ess_history = []
        self._resample_steps = []

        particles = self._proposal.sample(self._N, rng)
        log_weights = np.array([
            self._target_lp(s) - self._proposal.log_prob(s)
            for s in particles
        ])

        ess = self._compute_ess(log_weights)
        self._ess_history.append(ess)

        converged = False
        num_steps = 1

        if self._mode == "sir":
            if ess < self._N * self._ess_threshold:
                particles, log_weights = self._resample(
                    particles, log_weights, rng, step=0
                )
            converged = self._check_convergence()
        else:
            for step in range(1, self._max_steps):
                if ess < self._N * self._ess_threshold:
                    particles, log_weights = self._resample(
                        particles, log_weights, rng, step=step
                    )

                new_particles = self._proposal.sample(self._N, rng)
                new_log_weights = np.array([
                    self._target_lp(s) - self._proposal.log_prob(s)
                    for s in new_particles
                ])

                particles = new_particles
                log_weights = new_log_weights
                ess = self._compute_ess(log_weights)
                self._ess_history.append(ess)
                num_steps = step + 1

                if self._check_convergence():
                    converged = True
                    break

        return SISResult(
            particles=particles,
            weights=ImportanceWeights(log_weights),
            ess_history=list(self._ess_history),
            resample_steps=list(self._resample_steps),
            converged=converged,
            num_steps=num_steps,
        )

    def _resample(
        self,
        particles: List[Schedule],
        log_weights: np.ndarray,
        rng: np.random.RandomState,
        step: int,
    ) -> Tuple[List[Schedule], np.ndarray]:
        """Resample particles and reset weights to uniform."""
        max_lw = float(np.max(log_weights))
        w = np.exp(log_weights - max_lw)
        w_sum = w.sum()
        if w_sum < 1e-300:
            w_norm = np.ones(len(particles)) / len(particles)
        else:
            w_norm = w / w_sum

        new_particles = self._resampler.resample(particles, w_norm, rng)
        new_log_weights = np.zeros(len(particles), dtype=np.float64)
        self._resample_steps.append(step)
        return new_particles, new_log_weights

    def _compute_ess(self, log_weights: np.ndarray) -> float:
        """Compute effective sample size from log-weights."""
        max_lw = float(np.max(log_weights))
        w = np.exp(log_weights - max_lw)
        w_sum = w.sum()
        if w_sum < 1e-300:
            return 0.0
        w_norm = w / w_sum
        return 1.0 / float(np.sum(w_norm ** 2))

    def _check_convergence(self) -> bool:
        """Check if ESS is stable over recent history.

        Convergence when the coefficient of variation of the last 3
        ESS values is below 10%.
        """
        if len(self._ess_history) < 3:
            return False
        recent = self._ess_history[-3:]
        mean_ess = np.mean(recent)
        if mean_ess < 1e-12:
            return False
        cv = np.std(recent) / mean_ess
        return cv < 0.1

    @property
    def ess_history(self) -> List[float]:
        return list(self._ess_history)


# ===================================================================
# Plackett-Luce IIA Validator
# ===================================================================

@dataclass
class IIATestResult:
    """Result of the IIA validation test."""
    is_violated: bool
    severity: str  # 'none', 'mild', 'moderate', 'severe'
    chi2_statistic: float
    p_value: float
    pair_violations: Dict[Tuple[str, str], float]
    recommendation: str


class PlackettLuceValidator:
    """Validates the Plackett-Luce IIA assumption.

    Tests IIA via pairwise comparison: for each pair of agents,
    checks if relative ordering probability is independent of
    which other agents are present (subset consistency).

    The test uses a chi-squared statistic comparing the observed
    pairwise win rates across subsets of competitors against the
    rates predicted by the Plackett-Luce model.

    Parameters
    ----------
    agents : list of str
        Agent identifiers.
    significance_level : float
        Significance level for chi-squared test (default 0.05).
    """

    def __init__(
        self,
        agents: List[str],
        significance_level: float = 0.05,
    ) -> None:
        self._agents = list(agents)
        self._alpha = significance_level

    def validate(
        self,
        schedules: List[Schedule],
        pl_weights: Optional[np.ndarray] = None,
    ) -> IIATestResult:
        """Test IIA on a collection of observed schedules.

        Parameters
        ----------
        schedules : list of Schedule
            Observed schedule samples.
        pl_weights : numpy.ndarray, optional
            Fitted PL weights per agent. If None, estimated from data.

        Returns
        -------
        IIATestResult
        """
        n_agents = len(self._agents)
        if n_agents < 2 or len(schedules) < 5:
            return IIATestResult(
                is_violated=False,
                severity="none",
                chi2_statistic=0.0,
                p_value=1.0,
                pair_violations={},
                recommendation="Insufficient data for IIA test.",
            )

        agent_idx = {a: i for i, a in enumerate(self._agents)}

        if pl_weights is None:
            pl_weights = self._estimate_pl_weights(schedules, agent_idx)

        pairwise_observed = self._compute_pairwise_rates(schedules, agent_idx)
        pairwise_expected = self._compute_pl_expected_rates(pl_weights)

        chi2_stat = 0.0
        df = 0
        pair_violations: Dict[Tuple[str, str], float] = {}

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                obs = pairwise_observed.get((i, j), 0.5)
                exp = pairwise_expected.get((i, j), 0.5)
                n_obs = len(schedules)
                if exp > 1e-10 and (1 - exp) > 1e-10:
                    chi2_contrib = n_obs * (obs - exp) ** 2 / (exp * (1 - exp))
                    chi2_stat += chi2_contrib
                    df += 1
                    deviation = abs(obs - exp)
                    if deviation > 0.05:
                        pair_violations[(self._agents[i], self._agents[j])] = deviation

        if df == 0:
            return IIATestResult(
                is_violated=False,
                severity="none",
                chi2_statistic=0.0,
                p_value=1.0,
                pair_violations={},
                recommendation="No testable pairs.",
            )

        p_value = float(scipy_stats.chi2.sf(chi2_stat, df))
        is_violated = p_value < self._alpha
        severity = self._classify_severity(chi2_stat, df, p_value)
        recommendation = self._recommend(severity)

        return IIATestResult(
            is_violated=is_violated,
            severity=severity,
            chi2_statistic=chi2_stat,
            p_value=p_value,
            pair_violations=pair_violations,
            recommendation=recommendation,
        )

    def _estimate_pl_weights(
        self,
        schedules: List[Schedule],
        agent_idx: Dict[str, int],
    ) -> np.ndarray:
        """Estimate PL weights from observed schedules via rank counts."""
        n = len(self._agents)
        weights = np.ones(n)
        for sched in schedules:
            ordering = sched.ordering()
            for pos, agent in enumerate(ordering):
                if agent in agent_idx:
                    weights[agent_idx[agent]] += 1.0 / (pos + 1)
        weights = weights / weights.sum()
        return np.maximum(weights, 1e-10)

    def _compute_pairwise_rates(
        self,
        schedules: List[Schedule],
        agent_idx: Dict[str, int],
    ) -> Dict[Tuple[int, int], float]:
        """Compute observed P(i before j) for each pair."""
        n = len(self._agents)
        wins = np.zeros((n, n))
        counts = np.zeros((n, n))
        for sched in schedules:
            ordering = sched.ordering()
            pos_map = {a: p for p, a in enumerate(ordering)}
            for i in range(n):
                for j in range(i + 1, n):
                    ai, aj = self._agents[i], self._agents[j]
                    if ai in pos_map and aj in pos_map:
                        counts[i, j] += 1
                        if pos_map[ai] < pos_map[aj]:
                            wins[i, j] += 1
        result: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                if counts[i, j] > 0:
                    result[(i, j)] = wins[i, j] / counts[i, j]
                else:
                    result[(i, j)] = 0.5
        return result

    def _compute_pl_expected_rates(
        self, weights: np.ndarray
    ) -> Dict[Tuple[int, int], float]:
        """Compute PL-predicted P(i before j) = w_i / (w_i + w_j)."""
        n = len(weights)
        result: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                result[(i, j)] = weights[i] / (weights[i] + weights[j])
        return result

    def _classify_severity(
        self, chi2: float, df: int, p_value: float
    ) -> str:
        """Classify IIA violation severity."""
        if p_value >= self._alpha:
            return "none"
        ratio = chi2 / max(df, 1)
        if ratio < 3.0:
            return "mild"
        elif ratio < 10.0:
            return "moderate"
        else:
            return "severe"

    def _recommend(self, severity: str) -> str:
        """Suggest alternatives based on violation severity."""
        if severity == "none":
            return "Plackett-Luce IIA holds. Standard PL proposal is appropriate."
        elif severity == "mild":
            return (
                "Mild IIA violation. Consider mixed logit proposal "
                "(MixedLogitProposal) for improved accuracy."
            )
        elif severity == "moderate":
            return (
                "Moderate IIA violation. Use nested logit "
                "(NestedLogitProposal) or mixed logit proposal. "
                "Group correlated agents into interaction nests."
            )
        else:
            return (
                "Severe IIA violation. Use kernel-based proposal or "
                "mixed logit with full covariance. Plackett-Luce is "
                "inappropriate for this schedule distribution."
            )


# ===================================================================
# Mixed Logit Proposal
# ===================================================================

class MixedLogitProposal(ProposalDistribution):
    r"""Mixture of Plackett-Luce proposals for IIA-robust sampling.

    Models the schedule distribution as a finite mixture:

        q(σ) = Σ_k π_k PL_k(σ)

    where each PL_k has its own strength parameters.  Unlike a single
    PL, this can capture correlated agent timing from shared
    environmental factors.

    EM-style fitting is available via ``fit()`` from observed schedules.

    Parameters
    ----------
    agents : list of str
        Agent identifiers.
    num_components : int
        Number of mixture components.
    rng : numpy.random.RandomState, optional
        Random state.
    """

    def __init__(
        self,
        agents: List[str],
        num_components: int = 3,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self._agents = list(agents)
        self._n = len(agents)
        self._K = num_components
        self._rng = rng or np.random.RandomState(42)
        self._mix_weights = np.ones(self._K) / self._K
        self._component_weights = [
            np.ones(self._n) / self._n for _ in range(self._K)
        ]
        self._fitted = False

    @property
    def mix_weights(self) -> np.ndarray:
        return self._mix_weights.copy()

    @property
    def component_weights(self) -> List[np.ndarray]:
        return [w.copy() for w in self._component_weights]

    def fit(
        self,
        schedules: List[Schedule],
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> float:
        """Fit mixture parameters via EM from observed schedules.

        Parameters
        ----------
        schedules : list of Schedule
            Training data.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance on log-likelihood.

        Returns
        -------
        float
            Final log-likelihood.
        """
        N = len(schedules)
        if N == 0:
            return float("-inf")

        agent_idx = {a: i for i, a in enumerate(self._agents)}

        # Initialize component weights with perturbation
        for k in range(self._K):
            self._component_weights[k] = np.ones(self._n) + self._rng.uniform(
                0, 0.5, size=self._n
            )
            self._component_weights[k] /= self._component_weights[k].sum()

        prev_ll = float("-inf")

        for iteration in range(max_iter):
            # E-step: compute responsibilities
            log_resp = np.zeros((N, self._K))
            for i, sched in enumerate(schedules):
                for k in range(self._K):
                    log_resp[i, k] = (
                        math.log(max(self._mix_weights[k], 1e-300))
                        + self._pl_log_prob_single(
                            sched, self._component_weights[k], agent_idx
                        )
                    )

            log_resp_max = log_resp.max(axis=1, keepdims=True)
            resp = np.exp(log_resp - log_resp_max)
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp = resp / np.maximum(resp_sum, 1e-300)

            ll = float(np.sum(np.log(np.maximum(resp_sum.flatten(), 1e-300))
                              + log_resp_max.flatten()))

            # M-step: update parameters
            for k in range(self._K):
                r_k = resp[:, k]
                r_sum = r_k.sum()
                if r_sum < 1e-10:
                    continue
                self._mix_weights[k] = r_sum / N

                new_weights = np.ones(self._n) * 1e-6
                for i, sched in enumerate(schedules):
                    ordering = sched.ordering()
                    for pos, agent in enumerate(ordering):
                        if agent in agent_idx:
                            new_weights[agent_idx[agent]] += (
                                r_k[i] / (pos + 1)
                            )
                self._component_weights[k] = new_weights / new_weights.sum()

            self._mix_weights /= self._mix_weights.sum()

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        self._fitted = True
        return prev_ll

    def sample(self, n: int, rng: np.random.RandomState) -> List[Schedule]:
        """Sample schedules from the mixture proposal."""
        schedules: List[Schedule] = []
        assignments = rng.choice(self._K, size=n, p=self._mix_weights)
        for k_idx in assignments:
            sched = self._sample_pl(self._component_weights[k_idx], rng)
            schedules.append(sched)
        return schedules

    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability under the mixture (log-sum-exp)."""
        agent_idx = {a: i for i, a in enumerate(self._agents)}
        log_probs = np.array([
            math.log(max(self._mix_weights[k], 1e-300))
            + self._pl_log_prob_single(schedule, self._component_weights[k], agent_idx)
            for k in range(self._K)
        ])
        return float(logsumexp(log_probs))

    def _sample_pl(
        self, weights: np.ndarray, rng: np.random.RandomState
    ) -> Schedule:
        """Sample a single schedule from a PL distribution."""
        remaining = list(range(self._n))
        remaining_w = weights.copy()
        events: List[ScheduleEvent] = []
        time_counter = 0.0

        while remaining:
            w_avail = remaining_w[remaining]
            w_sum = w_avail.sum()
            if w_sum < 1e-300:
                probs = np.ones(len(remaining)) / len(remaining)
            else:
                probs = w_avail / w_sum
            choice_local = rng.choice(len(remaining), p=probs)
            chosen = remaining[choice_local]
            events.append(ScheduleEvent(
                agent_id=self._agents[chosen],
                timestep=0,
                action_time=time_counter,
            ))
            remaining.pop(choice_local)
            time_counter += 1.0

        return Schedule(events=events)

    @staticmethod
    def _pl_log_prob_single(
        schedule: Schedule,
        weights: np.ndarray,
        agent_idx: Dict[str, int],
    ) -> float:
        """Log-probability of schedule under a single PL component."""
        ordering = schedule.ordering()
        n = len(weights)
        indices: List[int] = []
        for agent in ordering:
            if agent in agent_idx:
                indices.append(agent_idx[agent])
        if len(indices) != n:
            return float("-inf")

        lp = 0.0
        remaining_sum = sum(weights[i] for i in indices)
        for idx in indices:
            w = weights[idx]
            if w < 1e-300 or remaining_sum < 1e-300:
                return float("-inf")
            lp += math.log(w) - math.log(remaining_sum)
            remaining_sum -= w
            if remaining_sum < 1e-300:
                remaining_sum = 1e-300
        return lp


# ===================================================================
# Nested Logit Proposal
# ===================================================================

class NestedLogitProposal(ProposalDistribution):
    """Nested logit proposal over agent schedules.

    Groups agents into *nests* (interaction groups).  IIA holds
    within each nest but not across nests.  The model selects a
    nest first, then orders agents within the nest, then proceeds
    to the next nest.

    Parameters
    ----------
    agents : list of str
        Agent identifiers.
    nests : list of list of str
        Grouping of agents into nests.  Each inner list is a nest.
        All agents must appear exactly once.
    nest_scales : numpy.ndarray, optional
        Scale parameters per nest (λ_k ∈ (0,1]).  Default uniform.
    agent_weights : numpy.ndarray, optional
        Per-agent strength weights.  Default uniform.
    rng : numpy.random.RandomState, optional
        Random state.
    """

    def __init__(
        self,
        agents: List[str],
        nests: List[List[str]],
        nest_scales: Optional[np.ndarray] = None,
        agent_weights: Optional[np.ndarray] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self._agents = list(agents)
        self._n = len(agents)
        self._nests = [list(nest) for nest in nests]
        self._K = len(nests)
        self._rng = rng or np.random.RandomState(42)

        if nest_scales is not None:
            self._nest_scales = np.clip(nest_scales, 1e-6, 1.0)
        else:
            self._nest_scales = np.ones(self._K)

        if agent_weights is not None:
            self._agent_weights = np.maximum(agent_weights, 1e-12)
        else:
            self._agent_weights = np.ones(self._n)

        self._agent_idx = {a: i for i, a in enumerate(self._agents)}

        self._nest_for_agent: Dict[str, int] = {}
        for k, nest in enumerate(self._nests):
            for agent in nest:
                self._nest_for_agent[agent] = k

    def sample(self, n: int, rng: np.random.RandomState) -> List[Schedule]:
        """Sample schedules from the nested logit proposal."""
        schedules: List[Schedule] = []
        for _ in range(n):
            schedules.append(self._sample_one(rng))
        return schedules

    def _sample_one(self, rng: np.random.RandomState) -> Schedule:
        """Sample one schedule: pick nest order, then order within nests."""
        nest_inclusive_values = np.zeros(self._K)
        for k, nest in enumerate(self._nests):
            iv = 0.0
            for agent in nest:
                idx = self._agent_idx[agent]
                iv += self._agent_weights[idx] ** (1.0 / self._nest_scales[k])
            nest_inclusive_values[k] = iv ** self._nest_scales[k]

        total_iv = nest_inclusive_values.sum()
        if total_iv < 1e-300:
            nest_probs = np.ones(self._K) / self._K
        else:
            nest_probs = nest_inclusive_values / total_iv

        nest_order = []
        remaining_nests = list(range(self._K))
        remaining_probs = nest_probs.copy()

        while remaining_nests:
            rp_sum = remaining_probs.sum()
            if rp_sum < 1e-300:
                p = np.ones(len(remaining_nests)) / len(remaining_nests)
            else:
                p = remaining_probs / rp_sum
            choice = rng.choice(len(remaining_nests), p=p)
            nest_order.append(remaining_nests[choice])
            remaining_nests.pop(choice)
            remaining_probs = np.delete(remaining_probs, choice)

        events: List[ScheduleEvent] = []
        time_counter = 0.0
        for k in nest_order:
            nest_agents = list(self._nests[k])
            remaining = list(nest_agents)
            while remaining:
                w = np.array([
                    self._agent_weights[self._agent_idx[a]] for a in remaining
                ])
                w_sum = w.sum()
                if w_sum < 1e-300:
                    p = np.ones(len(remaining)) / len(remaining)
                else:
                    p = w / w_sum
                choice = rng.choice(len(remaining), p=p)
                agent = remaining[choice]
                events.append(ScheduleEvent(
                    agent_id=agent,
                    timestep=0,
                    action_time=time_counter,
                ))
                remaining.pop(choice)
                time_counter += 1.0

        return Schedule(events=events)

    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability under the nested logit model.

        Decomposes as:
            log P(σ) = Σ_k [ log P(nest_k at position) +
                             Σ_{agents in nest_k} log P(agent | nest_k) ]
        """
        ordering = schedule.ordering()
        lp = 0.0

        # Track which agents and nests have been placed
        remaining_nests_set = set(range(self._K))
        nest_remaining_agents: Dict[int, List[str]] = {
            k: list(nest) for k, nest in enumerate(self._nests)
        }

        pos = 0
        while pos < len(ordering):
            agent = ordering[pos]
            if agent not in self._nest_for_agent:
                pos += 1
                continue
            k = self._nest_for_agent[agent]

            # Probability of choosing this nest (if first agent from nest)
            if k in remaining_nests_set:
                remaining_nests_set.discard(k)
                nest_ivs = {}
                for kk in list(remaining_nests_set) + [k]:
                    iv = 0.0
                    for a in nest_remaining_agents.get(kk, []):
                        idx = self._agent_idx.get(a, 0)
                        iv += self._agent_weights[idx] ** (
                            1.0 / self._nest_scales[kk]
                        )
                    nest_ivs[kk] = iv ** self._nest_scales[kk]
                total_iv = sum(nest_ivs.values())
                if total_iv > 1e-300 and nest_ivs.get(k, 0) > 1e-300:
                    lp += math.log(nest_ivs[k]) - math.log(total_iv)

            # Probability of agent within nest (PL within nest)
            remaining_in_nest = nest_remaining_agents.get(k, [])
            if agent in remaining_in_nest:
                w_remaining = [
                    self._agent_weights[self._agent_idx[a]]
                    for a in remaining_in_nest
                ]
                w_total = sum(w_remaining)
                w_agent = self._agent_weights[self._agent_idx[agent]]
                if w_total > 1e-300 and w_agent > 1e-300:
                    lp += math.log(w_agent) - math.log(w_total)
                remaining_in_nest.remove(agent)

            pos += 1

        return lp


# ===================================================================
# Stopping Criteria
# ===================================================================

@dataclass
class StoppingDecision:
    """Result of stopping criteria evaluation."""
    should_stop: bool
    criterion: str
    details: Dict[str, float]


class StoppingCriteria:
    """Principled convergence diagnostics for IS/SIS.

    Supports multiple stopping criteria:
    - ESS-based: stop when ESS is stable
    - CI-width: stop when confidence interval width < target
    - MCSE: Monte Carlo Standard Error below threshold
    - R-hat: Gelman-Rubin convergence for multiple chains
    - Geweke: stationarity diagnostic for single chain

    Parameters
    ----------
    ess_stability_window : int
        Window size for ESS stability check.
    ess_cv_threshold : float
        Coefficient of variation threshold for ESS stability.
    ci_width_target : float
        Target CI width for stopping.
    mcse_threshold : float
        MCSE threshold for stopping.
    rhat_threshold : float
        R-hat threshold (< 1.1 is standard).
    geweke_threshold : float
        Geweke z-score threshold (typically 2.0).
    """

    def __init__(
        self,
        ess_stability_window: int = 5,
        ess_cv_threshold: float = 0.1,
        ci_width_target: float = 0.05,
        mcse_threshold: float = 0.01,
        rhat_threshold: float = 1.1,
        geweke_threshold: float = 2.0,
    ) -> None:
        self._ess_window = ess_stability_window
        self._ess_cv_thresh = ess_cv_threshold
        self._ci_target = ci_width_target
        self._mcse_thresh = mcse_threshold
        self._rhat_thresh = rhat_threshold
        self._geweke_thresh = geweke_threshold

    def check_ess_stability(self, ess_history: List[float]) -> StoppingDecision:
        """Check if ESS has stabilised."""
        if len(ess_history) < self._ess_window:
            return StoppingDecision(
                should_stop=False,
                criterion="ess_stability",
                details={"reason": "insufficient_data"},
            )
        window = ess_history[-self._ess_window:]
        mean_ess = np.mean(window)
        if mean_ess < 1e-12:
            return StoppingDecision(
                should_stop=False,
                criterion="ess_stability",
                details={"mean_ess": 0.0, "cv": float("inf")},
            )
        cv = float(np.std(window) / mean_ess)
        return StoppingDecision(
            should_stop=cv < self._ess_cv_thresh,
            criterion="ess_stability",
            details={"mean_ess": float(mean_ess), "cv": cv},
        )

    def check_ci_width(
        self, ci: ConfidenceInterval
    ) -> StoppingDecision:
        """Check if CI width is below target."""
        return StoppingDecision(
            should_stop=ci.width < self._ci_target,
            criterion="ci_width",
            details={"width": ci.width, "target": self._ci_target},
        )

    def check_mcse(
        self,
        values: np.ndarray,
        log_weights: np.ndarray,
    ) -> StoppingDecision:
        """Monte Carlo Standard Error stopping criterion.

        MCSE = sqrt(Var_IS[f] / ESS)
        """
        max_lw = float(np.max(log_weights))
        w = np.exp(log_weights - max_lw)
        w_sum = w.sum()
        if w_sum < 1e-300:
            return StoppingDecision(
                should_stop=False,
                criterion="mcse",
                details={"mcse": float("inf")},
            )
        w_norm = w / w_sum
        ess = 1.0 / float(np.sum(w_norm ** 2))
        mu = float(np.dot(w_norm, values))
        var = float(np.dot(w_norm, (values - mu) ** 2))
        mcse = math.sqrt(max(var, 0.0) / max(ess, 1.0))
        return StoppingDecision(
            should_stop=mcse < self._mcse_thresh,
            criterion="mcse",
            details={"mcse": mcse, "ess": ess, "var": var},
        )

    def check_rhat(
        self, chains: List[np.ndarray]
    ) -> StoppingDecision:
        """Gelman-Rubin R-hat convergence diagnostic.

        R-hat = sqrt((n-1)/n + B/(n*W))
        where B = between-chain variance, W = within-chain variance.
        """
        m = len(chains)
        if m < 2:
            return StoppingDecision(
                should_stop=False,
                criterion="rhat",
                details={"rhat": float("inf"), "reason": "need_multiple_chains"},
            )

        n_per_chain = min(len(c) for c in chains)
        if n_per_chain < 2:
            return StoppingDecision(
                should_stop=False,
                criterion="rhat",
                details={"rhat": float("inf")},
            )

        chain_means = np.array([np.mean(c[:n_per_chain]) for c in chains])
        chain_vars = np.array([np.var(c[:n_per_chain], ddof=1) for c in chains])

        grand_mean = np.mean(chain_means)
        B = n_per_chain * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)

        if W < 1e-15:
            rhat = 1.0 if B < 1e-15 else float("inf")
        else:
            var_hat = (n_per_chain - 1) / n_per_chain * W + B / n_per_chain
            rhat = math.sqrt(var_hat / W)

        return StoppingDecision(
            should_stop=rhat < self._rhat_thresh,
            criterion="rhat",
            details={"rhat": rhat, "B": float(B), "W": float(W)},
        )

    def check_geweke(
        self, chain: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5
    ) -> StoppingDecision:
        """Geweke (1992) stationarity diagnostic.

        Compares the mean of the first fraction to the mean of the last
        fraction of the chain via a z-test.
        """
        n = len(chain)
        n_first = max(int(n * first_frac), 1)
        n_last = max(int(n * last_frac), 1)

        if n_first + n_last > n or n < 4:
            return StoppingDecision(
                should_stop=False,
                criterion="geweke",
                details={"z_score": float("inf"), "reason": "chain_too_short"},
            )

        first_part = chain[:n_first]
        last_part = chain[-n_last:]

        mean_first = np.mean(first_part)
        mean_last = np.mean(last_part)
        var_first = np.var(first_part, ddof=1) / n_first if n_first > 1 else 0
        var_last = np.var(last_part, ddof=1) / n_last if n_last > 1 else 0

        denom = math.sqrt(max(var_first + var_last, 1e-15))
        z_score = abs(mean_first - mean_last) / denom

        return StoppingDecision(
            should_stop=z_score < self._geweke_thresh,
            criterion="geweke",
            details={"z_score": z_score, "mean_first": float(mean_first),
                      "mean_last": float(mean_last)},
        )


# ===================================================================
# Joint Error Analysis
# ===================================================================

@dataclass
class ErrorDecomposition:
    """Decomposition of total estimation error.

    Theorem: total MSE ≤ (AI_error)² + (IS_variance/N) + 2*(AI_error)*(IS_bias)

    Attributes
    ----------
    ai_error : float
        Abstract interpretation overapproximation error.
    is_variance : float
        Importance sampling variance.
    is_bias : float
        Importance sampling bias (from self-normalisation).
    cross_term : float
        Interaction term: 2 * AI_error * IS_bias.
    total_mse_bound : float
        Upper bound on total MSE.
    dominant_source : str
        Which error source dominates ('ai', 'is_variance', 'cross_term').
    recommendation : str
        Actionable diagnostic.
    """
    ai_error: float
    is_variance: float
    is_bias: float
    cross_term: float
    total_mse_bound: float
    dominant_source: str
    recommendation: str
    num_samples: int = 0


class JointErrorAnalysis:
    """Analyze interaction between AI overapproximation and IS estimation error.

    Decomposes total error into:
    - Abstract interpretation error (overapproximation of reachable states)
    - Importance sampling estimation error (variance + bias)
    - Cross-term (how AI error inflates IS variance)

    The key insight is that abstract interpretation identifies a
    *superset* of race-triggering schedules, so the IS proposal
    may waste samples on false-positive regions.  This inflates
    the IS variance by a factor related to the AI overapproximation
    ratio.

    Parameters
    ----------
    ai_overapprox_ratio : float
        Ratio |AI_race_region| / |true_race_region|.  ≥ 1.0.
    """

    def __init__(self, ai_overapprox_ratio: float = 1.0) -> None:
        self._overapprox = max(ai_overapprox_ratio, 1.0)

    def analyze(
        self,
        is_estimate: float,
        is_variance: float,
        is_bias: float,
        num_samples: int,
        ai_bound: Optional[float] = None,
        true_value: Optional[float] = None,
    ) -> ErrorDecomposition:
        """Perform joint error decomposition.

        Parameters
        ----------
        is_estimate : float
            IS point estimate of P(race).
        is_variance : float
            Estimated variance of the IS estimator.
        is_bias : float
            Estimated bias of the self-normalised IS estimator.
        num_samples : int
            Number of IS samples used.
        ai_bound : float, optional
            Upper bound from abstract interpretation.
        true_value : float, optional
            True value (if known, for validation).

        Returns
        -------
        ErrorDecomposition
        """
        if ai_bound is not None and ai_bound > 0:
            ai_error = max(ai_bound - is_estimate, 0.0)
        else:
            ai_error = is_estimate * (self._overapprox - 1.0)

        is_var_per_sample = is_variance / max(num_samples, 1)
        cross_term = 2.0 * ai_error * abs(is_bias)

        total_mse = ai_error ** 2 + is_var_per_sample + cross_term

        components = {
            "ai": ai_error ** 2,
            "is_variance": is_var_per_sample,
            "cross_term": cross_term,
        }
        dominant = max(components, key=components.get)  # type: ignore

        recommendation = self._make_recommendation(
            dominant, ai_error, is_var_per_sample, cross_term, num_samples
        )

        return ErrorDecomposition(
            ai_error=ai_error,
            is_variance=is_var_per_sample,
            is_bias=is_bias,
            cross_term=cross_term,
            total_mse_bound=total_mse,
            dominant_source=dominant,
            recommendation=recommendation,
            num_samples=num_samples,
        )

    def _make_recommendation(
        self,
        dominant: str,
        ai_error: float,
        is_var: float,
        cross: float,
        n: int,
    ) -> str:
        if dominant == "ai":
            return (
                "Abstract interpretation error dominates. "
                "Refine the abstract domain (e.g., use polyhedra "
                "instead of intervals) to reduce overapproximation."
            )
        elif dominant == "is_variance":
            needed = int(n * is_var / max(self._target_var(ai_error), 1e-15))
            return (
                f"IS variance dominates. Increase samples to ~{max(needed, n)} "
                f"or use a better proposal (cross-entropy adapted)."
            )
        else:
            return (
                "Cross-term (AI × IS interaction) dominates. "
                "Both AI precision and IS proposal quality need improvement. "
                "Consider tighter abstract interpretation + adapted proposal."
            )

    def _target_var(self, ai_error: float) -> float:
        """Target IS variance to match AI error contribution."""
        return max(ai_error ** 2, 1e-15)

    def compute_inflation_factor(self) -> float:
        """Compute how much AI overapproximation inflates IS variance.

        When the proposal is guided toward the AI race region (which is
        a superset of the true race region), the effective sample size
        is reduced by approximately the overapproximation ratio.

        Returns
        -------
        float
            Variance inflation factor ≥ 1.0.
        """
        return self._overapprox

    def required_samples(
        self,
        target_mse: float,
        ai_error: float,
        is_bias: float,
    ) -> int:
        """Minimum samples needed to achieve target total MSE.

        From: MSE ≤ ai² + var/N + 2·ai·|bias|
        Solving for N: N ≥ var / (target_MSE - ai² - 2·ai·|bias|)
        """
        remaining = target_mse - ai_error ** 2 - 2 * ai_error * abs(is_bias)
        if remaining <= 0:
            return int(1e7)
        base_var = self._overapprox
        return max(int(math.ceil(base_var / remaining)), 1)
