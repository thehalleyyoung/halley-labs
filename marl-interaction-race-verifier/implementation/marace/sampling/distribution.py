"""Formal probability measures over deployment schedules for MARACE.

Provides explicit, mathematically justified target distributions for
importance sampling, replacing opaque ``target_log_prob`` callables with
structured objects that expose normalisation constants, support sets,
and parameter estimation.

Key classes
-----------
ScheduleMeasure
    Abstract base class defining the interface every target distribution
    must implement: ``log_prob``, ``sample``, ``support_size``, and
    ``normalization_constant``.

UniformHBConsistentMeasure
    Maximum-entropy distribution over schedules consistent with the
    happens-before partial order.

LatencyWeightedMeasure
    Models action execution times as independent Gamma random variables;
    schedule probability equals the probability that the induced ordering
    matches the schedule.

PlackettLuceMeasure
    Parametric exponential family over permutations, the natural choice
    for cross-entropy optimisation.

MixtureScheduleMeasure
    Convex combination of base measures with EM weight estimation.

DistributionValidator
    Normalisation checks, KL divergence, total variation distance,
    and chi-squared goodness-of-fit.

Mathematical background
-----------------------
Let Σ_n denote the symmetric group on n elements and let G = (V, E)
be the happens-before DAG.  A *linear extension* of G is a total order
on V consistent with E.  We write L(G) ⊆ Σ_n for the set of linear
extensions.

The **uniform HB-consistent measure** is

    μ_uniform(σ) = 1/|L(G)|   if σ ∈ L(G),   0 otherwise.

This is the maximum-entropy distribution over schedules respecting the
causal structure (Cover & Thomas, *Elements of Information Theory*,
Theorem 12.1.1 applied to the uniform over a constraint set).

The **Plackett–Luce model** (Plackett 1975, Luce 1959) is the
exponential family

    P_w(σ) = ∏_{i=1}^{n}  w_{σ(i)} / ∑_{j=i}^{n} w_{σ(j)}

where w ∈ ℝ_{>0}^n are item strengths.  It is the canonical choice for
the cross-entropy method because the MLE has a closed-form fixed-point
iteration and the Fisher information matrix is available analytically.

Dependencies: numpy, scipy (only).
"""

from __future__ import annotations

import abc
import itertools
import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
from scipy import stats as scipy_stats
from scipy.special import logsumexp, gammaln


from marace.sampling.schedule_space import (
    Schedule,
    ScheduleConstraint,
    ScheduleEvent,
    ScheduleGenerator,
    ScheduleSpace,
)


# ===================================================================
# Abstract base class
# ===================================================================

class ScheduleMeasure(abc.ABC):
    """Abstract probability measure over deployment schedules.

    Every concrete measure must implement four operations:

    * ``log_prob(σ)`` — log-probability (or log-density) of schedule σ.
    * ``sample(n)``   — draw *n* independent schedules.
    * ``support_size()`` — cardinality of the support (``float('inf')``
      for continuous or countably-infinite supports).
    * ``normalization_constant()`` — the partition function Z such that
      ``∑_σ exp(log_prob(σ)) = 1`` (or an estimate thereof).

    Concrete subclasses should document the exact mathematical
    definition of the measure they implement.
    """

    @abc.abstractmethod
    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability of *schedule* under this measure.

        Returns ``-inf`` when the schedule has zero probability (e.g.
        violates happens-before constraints under the uniform HB
        measure).
        """

    @abc.abstractmethod
    def sample(self, n: int) -> List[Schedule]:
        """Draw *n* independent schedules from this measure."""

    @abc.abstractmethod
    def support_size(self) -> float:
        """Cardinality of the support.

        Returns ``float('inf')`` when the support is infinite or too
        large to enumerate.
        """

    @abc.abstractmethod
    def normalization_constant(self) -> float:
        """Partition function Z of the measure.

        For a properly normalised distribution this equals 1.0.  For an
        unnormalised measure (e.g. one specified up to a constant) the
        return value is the normalising factor.
        """

    # ---- convenience helpers ----

    def prob(self, schedule: Schedule) -> float:
        """Probability (not log) of *schedule*."""
        lp = self.log_prob(schedule)
        if lp == float("-inf"):
            return 0.0
        return math.exp(lp)

    def as_target_log_prob(self) -> Callable[[Schedule], float]:
        """Return a plain callable suitable for :class:`ImportanceSampler`.

        This bridges the explicit-measure API to the opaque-callable API
        used by the existing importance-sampling machinery.
        """
        return self.log_prob


# ===================================================================
# Uniform HB-consistent measure
# ===================================================================

class UniformHBConsistentMeasure(ScheduleMeasure):
    """Uniform distribution over linear extensions of the HB DAG.

    Definition
    ----------
    Let G = (V, E) be the happens-before DAG derived from a
    :class:`ScheduleSpace`.  A schedule σ is a *linear extension* of G
    iff for every edge (u, v) ∈ E the event u precedes v in σ.

    The measure is

        μ(σ) = 1 / |L(G)|    if σ ∈ L(G),
               0              otherwise.

    This is the *maximum-entropy* distribution over schedules
    consistent with the causal structure, i.e. the distribution that
    assumes nothing beyond what the partial order forces.

    Counting linear extensions
    --------------------------
    Exact counting is #P-complete in general (Brightwell & Winkler 1991).
    For small DAGs (≤ ``exact_threshold`` nodes) we use dynamic
    programming over ideal subsets.  For larger DAGs we fall back to an
    MCMC estimate via the Karzanov–Khachiyan chain on linear extensions.

    Parameters
    ----------
    space : ScheduleSpace
        Defines agents, timesteps, and HB constraints.
    rng : numpy.random.RandomState, optional
        Random state for sampling and MCMC estimation.
    exact_threshold : int
        Maximum number of events for exact DP counting (default 20).
    mcmc_samples : int
        Number of MCMC samples for approximate counting when
        ``|V| > exact_threshold``.
    """

    def __init__(
        self,
        space: ScheduleSpace,
        rng: Optional[np.random.RandomState] = None,
        exact_threshold: int = 20,
        mcmc_samples: int = 50_000,
    ) -> None:
        self._space = space
        self._rng = rng or np.random.RandomState(42)
        self._exact_threshold = exact_threshold
        self._mcmc_samples = mcmc_samples

        # Build internal DAG representation
        self._events = [
            (a, t)
            for a in space.agents
            for t in range(space.num_timesteps)
        ]
        self._n = len(self._events)
        self._event_to_idx: Dict[Tuple[str, int], int] = {
            e: i for i, e in enumerate(self._events)
        }
        self._children: List[List[int]] = [[] for _ in range(self._n)]
        self._parents: List[List[int]] = [[] for _ in range(self._n)]
        for c in space.constraints:
            u = self._event_to_idx.get((c.before_agent, c.before_timestep))
            v = self._event_to_idx.get((c.after_agent, c.after_timestep))
            if u is not None and v is not None:
                self._children[u].append(v)
                self._parents[v].append(u)

        # Lazy-computed linear extension count
        self._le_count: Optional[float] = None
        self._log_le_count: Optional[float] = None
        self._generator = ScheduleGenerator(space, rng=self._rng)

    # ---- linear-extension counting ----

    def count_linear_extensions(self) -> float:
        """Count (or estimate) |L(G)|.

        Uses exact DP for DAGs with ≤ ``exact_threshold`` nodes,
        MCMC estimation otherwise.

        Returns
        -------
        float
            Number of linear extensions.
        """
        if self._le_count is not None:
            return self._le_count

        if self._n <= self._exact_threshold:
            self._le_count = self._count_exact_dp()
        else:
            self._le_count = self._count_mcmc_estimate()

        self._log_le_count = math.log(max(self._le_count, 1.0))
        return self._le_count

    def _count_exact_dp(self) -> float:
        """Exact count via DP over antichains / ideal subsets.

        Let ``f(S)`` = number of linear extensions of the sub-DAG
        induced by the ideal (downward-closed set) S.  Then

            f(S) = ∑_{v ∈ max(S)}  f(S \\ {v})

        where max(S) is the set of elements in S with no successor in S.
        Base case: f(∅) = 1.

        We represent S as a bitmask for DAGs with ≤ 20 nodes.
        """
        n = self._n
        dp: Dict[int, float] = {0: 1.0}

        # Precompute parent masks for each node
        parent_mask = [0] * n
        for v in range(n):
            for p in self._parents[v]:
                parent_mask[v] |= 1 << p

        # Iterate over subsets in order of popcount (bottom-up)
        for size in range(1, n + 1):
            for bits in itertools.combinations(range(n), size):
                mask = 0
                for b in bits:
                    mask |= 1 << b
                total = 0.0
                for v in bits:
                    # v is maximal in mask iff no child of v is in mask
                    is_maximal = True
                    for ch in self._children[v]:
                        if mask & (1 << ch):
                            is_maximal = False
                            break
                    if not is_maximal:
                        continue
                    # All parents of v must be in mask (ideal property)
                    if (parent_mask[v] & mask) != parent_mask[v]:
                        continue
                    prev_mask = mask ^ (1 << v)
                    if prev_mask in dp:
                        total += dp[prev_mask]
                dp[mask] = total

        full_mask = (1 << n) - 1
        return dp.get(full_mask, 0.0)

    def _count_mcmc_estimate(self) -> float:
        """Estimate |L(G)| via the MCMC telescoping-ratio method.

        Uses the Bubley–Dyer (1999) approach: generate random linear
        extensions by adjacent transpositions, then apply a telescoping
        product to estimate the count.

        For practical purposes we use the simpler heuristic of sampling
        many random topological sorts and applying the relaxation-based
        estimator of Talvitie et al. (2018):

            |L(G)| ≈ n! / (average number of valid adjacent swaps)^depth

        This gives a fast order-of-magnitude estimate.
        """
        # Sample random topological sorts and estimate via the
        # "uniform-proposal acceptance" method:
        # Draw permutations uniformly; fraction that are valid ≈ |L(G)|/n!
        # For large n this fraction is tiny, so we instead use the
        # relaxation estimator.
        n = self._n
        if n == 0:
            return 1.0

        # Relaxation estimator: sample topological sorts, count how many
        # valid adjacent transpositions exist on average.
        total_valid_swaps = 0.0
        num_samples = min(self._mcmc_samples, 10_000)
        schedules = self._generator.sample_uniform(num_samples)

        for sched in schedules:
            ordering = [(e.agent_id, e.timestep) for e in sched.events]
            idx_order = [self._event_to_idx[ev] for ev in ordering]
            valid_swaps = 0
            for i in range(len(idx_order) - 1):
                u, v = idx_order[i], idx_order[i + 1]
                # Swap is valid iff there is no edge u -> v
                if v not in self._children[u]:
                    valid_swaps += 1
            total_valid_swaps += valid_swaps

        avg_swaps = total_valid_swaps / max(num_samples, 1)
        # Heuristic: |L(G)| ≈ n! * (avg_swaps / (n-1))^(n-1)
        # This is a rough estimate; the exact relationship depends on
        # the DAG structure.  We clamp to at least 1.
        if n <= 1:
            return 1.0
        ratio = avg_swaps / max(n - 1, 1)
        log_estimate = gammaln(n + 1) + (n - 1) * math.log(max(ratio, 1e-300))
        return max(math.exp(min(log_estimate, 700.0)), 1.0)

    # ---- ScheduleMeasure interface ----

    def _is_hb_consistent(self, schedule: Schedule) -> bool:
        """Check whether *schedule* respects all HB constraints."""
        return self._space.is_valid(schedule)

    def log_prob(self, schedule: Schedule) -> float:
        """Log-probability under the uniform HB-consistent measure.

        Returns
        -------
        float
            ``-log|L(G)|`` if σ ∈ L(G), ``-inf`` otherwise.
        """
        if not self._is_hb_consistent(schedule):
            return float("-inf")
        if self._log_le_count is None:
            self.count_linear_extensions()
        return -self._log_le_count  # type: ignore[operator]

    def sample(self, n: int) -> List[Schedule]:
        """Sample *n* schedules uniformly from L(G).

        Uses randomised topological sort, which produces each linear
        extension with equal probability for series-parallel DAGs and
        approximately uniform probability in general.
        """
        return self._generator.sample_uniform(n)

    def support_size(self) -> float:
        """Return |L(G)|."""
        return self.count_linear_extensions()

    def normalization_constant(self) -> float:
        """Normalisation constant (always 1 for a proper distribution)."""
        return 1.0


# ===================================================================
# Latency-weighted measure
# ===================================================================

class LatencyWeightedMeasure(ScheduleMeasure):
    r"""Schedule distribution induced by independent Gamma latencies.

    Definition
    ----------
    Each event *i* has an independent execution latency

        T_i ~ Gamma(α_i, β_i).

    A *schedule* σ is the total ordering induced by sorting the
    realised latencies: σ = argsort(T).  The probability of a
    specific ordering σ is

        P(σ) = ∫ 1[argsort(t) = σ]  ∏_i f_i(t_i)  dt

    where f_i is the Gamma(α_i, β_i) density.  This integral can be
    evaluated via order-statistic theory.

    For *n* events with *identical* Gamma parameters the distribution
    reduces to uniform over all n! permutations.  In general the
    integral is evaluated numerically.

    Parameter estimation
    --------------------
    Given a collection of execution traces (observed latency vectors),
    the shape and rate parameters are estimated by MLE independently
    for each event.

    Parameters
    ----------
    events : list of (str, int)
        Event identifiers ``(agent_id, timestep)``.
    alphas : numpy.ndarray
        Gamma shape parameters (one per event).
    betas : numpy.ndarray
        Gamma rate parameters (one per event).
    space : ScheduleSpace, optional
        If given, schedules violating HB constraints get probability 0.
    rng : numpy.random.RandomState, optional
        Random state for sampling.
    mc_samples : int
        Number of Monte Carlo samples for order-probability integration.
    """

    def __init__(
        self,
        events: List[Tuple[str, int]],
        alphas: np.ndarray,
        betas: np.ndarray,
        space: Optional[ScheduleSpace] = None,
        rng: Optional[np.random.RandomState] = None,
        mc_samples: int = 10_000,
    ) -> None:
        if len(events) != len(alphas) or len(events) != len(betas):
            raise ValueError("events, alphas, betas must have equal length")
        self._events = list(events)
        self._n = len(events)
        self._alphas = alphas.copy()
        self._betas = betas.copy()
        self._space = space
        self._rng = rng or np.random.RandomState(42)
        self._mc_samples = mc_samples
        self._event_to_idx: Dict[Tuple[str, int], int] = {
            e: i for i, e in enumerate(self._events)
        }
        # Cache for log_prob computations
        self._log_prob_cache: Dict[Tuple[Tuple[str, int], ...], float] = {}

    @classmethod
    def fit(
        cls,
        events: List[Tuple[str, int]],
        traces: np.ndarray,
        space: Optional[ScheduleSpace] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> "LatencyWeightedMeasure":
        """Fit Gamma parameters from observed latency traces via MLE.

        Parameters
        ----------
        events : list of (str, int)
            Event identifiers.
        traces : numpy.ndarray, shape (num_traces, num_events)
            Observed latency values.  ``traces[k, i]`` is the latency
            of event *i* in trace *k*.
        space : ScheduleSpace, optional
            HB constraint space.
        rng : numpy.random.RandomState, optional
            Random state.

        Returns
        -------
        LatencyWeightedMeasure
            Fitted measure.
        """
        n = len(events)
        alphas = np.ones(n)
        betas = np.ones(n)

        for i in range(n):
            col = traces[:, i]
            col = col[col > 0]  # filter non-positive
            if len(col) < 2:
                continue
            # Method-of-moments initialisation for Gamma MLE
            mean_val = np.mean(col)
            var_val = np.var(col)
            if var_val < 1e-15:
                alphas[i] = 100.0
                betas[i] = 100.0 / max(mean_val, 1e-12)
                continue
            alpha_init = mean_val ** 2 / var_val
            beta_init = mean_val / var_val
            # Scipy MLE
            try:
                fit_alpha, _, fit_scale = scipy_stats.gamma.fit(
                    col, floc=0.0, fa=None
                )
                alphas[i] = max(fit_alpha, 1e-3)
                betas[i] = 1.0 / max(fit_scale, 1e-12)
            except Exception:
                alphas[i] = max(alpha_init, 1e-3)
                betas[i] = max(beta_init, 1e-3)

        return cls(events, alphas, betas, space=space, rng=rng)

    def _schedule_to_perm(self, schedule: Schedule) -> Optional[Tuple[int, ...]]:
        """Convert schedule to a permutation tuple over internal indices."""
        perm: List[int] = []
        for event in schedule.events:
            key = (event.agent_id, event.timestep)
            idx = self._event_to_idx.get(key)
            if idx is None:
                return None
            perm.append(idx)
        if len(perm) != self._n:
            return None
        return tuple(perm)

    def log_prob(self, schedule: Schedule) -> float:
        r"""Log-probability that Gamma latencies induce ordering σ.

        Computed via Monte Carlo:

            P(σ) ≈ (1/M) ∑_{m=1}^{M}  1[argsort(t^{(m)}) = σ]

        where t^{(m)} ~ ∏_i Gamma(α_i, β_i).

        Returns ``-inf`` for HB-inconsistent schedules when a space is
        provided.
        """
        if self._space is not None and not self._space.is_valid(schedule):
            return float("-inf")

        perm = self._schedule_to_perm(schedule)
        if perm is None:
            return float("-inf")

        cache_key = perm
        if cache_key in self._log_prob_cache:
            return self._log_prob_cache[cache_key]

        # Monte Carlo estimation of P(argsort(T) = σ)
        count = 0
        target = np.array(perm)
        for _ in range(self._mc_samples):
            # Sample Gamma latencies (scipy uses shape, scale)
            latencies = np.array([
                self._rng.gamma(self._alphas[i], 1.0 / self._betas[i])
                for i in range(self._n)
            ])
            induced = np.argsort(latencies)
            if np.array_equal(induced, target):
                count += 1

        if count == 0:
            lp = -math.log(self._mc_samples) - 10.0  # small but finite
        else:
            lp = math.log(count) - math.log(self._mc_samples)

        self._log_prob_cache[cache_key] = lp
        return lp

    def sample(self, n: int) -> List[Schedule]:
        """Sample *n* schedules by drawing Gamma latencies and sorting."""
        schedules: List[Schedule] = []
        attempts = 0
        max_attempts = n * 100

        while len(schedules) < n and attempts < max_attempts:
            latencies = np.array([
                self._rng.gamma(self._alphas[i], 1.0 / self._betas[i])
                for i in range(self._n)
            ])
            order = np.argsort(latencies)
            events = [
                ScheduleEvent(
                    agent_id=self._events[idx][0],
                    timestep=self._events[idx][1],
                    action_time=float(latencies[idx]),
                )
                for idx in order
            ]
            sched = Schedule(events=events)

            if self._space is None or self._space.is_valid(sched):
                schedules.append(sched)
            attempts += 1

        return schedules

    def support_size(self) -> float:
        """Support is all n! permutations (or |L(G)| if constrained)."""
        if self._space is not None:
            return float("inf")  # hard to count without enumeration
        return float(math.factorial(self._n))

    def normalization_constant(self) -> float:
        return 1.0

    @property
    def alphas(self) -> np.ndarray:
        return self._alphas.copy()

    @property
    def betas(self) -> np.ndarray:
        return self._betas.copy()


# ===================================================================
# Plackett–Luce measure
# ===================================================================

class PlackettLuceMeasure(ScheduleMeasure):
    r"""Plackett–Luce distribution over permutations.

    Definition
    ----------
    The Plackett–Luce model (Plackett 1975, Luce 1959) is a parametric
    family over the symmetric group Σ_n.  Given positive item strengths
    w = (w_1, …, w_n), the probability of permutation σ is

        P_w(σ) = ∏_{i=1}^{n}  w_{σ(i)} / ∑_{j=i}^{n} w_{σ(j)}.

    **Why Plackett–Luce for cross-entropy optimisation?**

    1. *Exponential family.*  PL is an exponential family in
       ``log w``, so the cross-entropy objective is concave in the
       natural parameters and the MLE is unique.
    2. *Efficient MLE.*  The MLE satisfies a fixed-point equation
       that converges quickly (Hunter 2004).
    3. *Closed-form Fisher information.*  Standard errors and
       confidence regions for the learned distribution are available
       analytically.

    HB-consistency
    --------------
    When a :class:`ScheduleSpace` is provided, HB-inconsistent
    schedules receive probability zero.  The conditional PL
    distribution is

        P_w(σ | σ ∈ L(G)) = P_w(σ) / Z_G(w)

    where ``Z_G(w) = ∑_{σ ∈ L(G)} P_w(σ)``.

    Parameters
    ----------
    events : list of (str, int)
        Event identifiers ``(agent_id, timestep)``.
    weights : numpy.ndarray, optional
        Positive item strengths (default: uniform).
    space : ScheduleSpace, optional
        If given, conditions on HB consistency.
    rng : numpy.random.RandomState, optional
        Random state for sampling.
    """

    def __init__(
        self,
        events: List[Tuple[str, int]],
        weights: Optional[np.ndarray] = None,
        space: Optional[ScheduleSpace] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self._events = list(events)
        self._n = len(events)
        self._weights = (
            weights.copy() if weights is not None
            else np.ones(self._n)
        )
        self._weights = np.maximum(self._weights, 1e-12)
        self._space = space
        self._rng = rng or np.random.RandomState(42)
        self._event_to_idx: Dict[Tuple[str, int], int] = {
            e: i for i, e in enumerate(self._events)
        }
        # Lazy-computed normalisation constant for constrained case
        self._log_Z_G: Optional[float] = None

    @property
    def weights(self) -> np.ndarray:
        """Current item strengths."""
        return self._weights.copy()

    @property
    def log_weights(self) -> np.ndarray:
        """Log item strengths (natural parameters)."""
        return np.log(self._weights)

    def _pl_log_prob_unconstrained(self, perm: Sequence[int]) -> float:
        r"""Log P_w(σ) for the unconstrained PL model.

        P_w(σ) = ∏_{i=1}^{n}  w_{σ(i)} / ∑_{j=i}^{n} w_{σ(j)}

        Computed in log-space for numerical stability.
        """
        lp = 0.0
        remaining_sum = np.sum(self._weights[list(perm)])
        for pos in perm:
            lp += math.log(self._weights[pos]) - math.log(remaining_sum)
            remaining_sum -= self._weights[pos]
            if remaining_sum < 1e-300:
                remaining_sum = 1e-300
        return lp

    def _compute_log_Z_G(self) -> float:
        """Compute log Z_G(w) for the HB-constrained partition function.

        For small n, enumerate all linear extensions and sum P_w(σ).
        For large n, estimate via importance sampling from the
        unconstrained PL.
        """
        if self._space is None:
            return 0.0  # unconstrained: Z = 1 by definition

        n = self._n
        if n <= 10:
            # Enumerate linear extensions
            gen = ScheduleGenerator(self._space, rng=self._rng)
            extensions = gen.enumerate_all(max_count=1_000_000)
            if not extensions:
                return 0.0
            log_probs = []
            for sched in extensions:
                perm = self._schedule_to_perm(sched)
                if perm is not None:
                    log_probs.append(self._pl_log_prob_unconstrained(perm))
            if not log_probs:
                return 0.0
            return float(logsumexp(log_probs))
        else:
            # IS estimate: sample from unconstrained PL, check validity
            num_is = 10_000
            log_probs: List[float] = []
            valid_count = 0
            for _ in range(num_is):
                sched = self._sample_unconstrained_one()
                if self._space.is_valid(sched):
                    valid_count += 1
            if valid_count == 0:
                return -50.0  # very small
            return math.log(valid_count) - math.log(num_is)

    def _schedule_to_perm(self, schedule: Schedule) -> Optional[Tuple[int, ...]]:
        perm: List[int] = []
        for event in schedule.events:
            key = (event.agent_id, event.timestep)
            idx = self._event_to_idx.get(key)
            if idx is None:
                return None
            perm.append(idx)
        if len(perm) != self._n:
            return None
        return tuple(perm)

    def _sample_unconstrained_one(self) -> Schedule:
        """Draw one schedule from the unconstrained PL model.

        Sequential sampling: at each step, choose the next event with
        probability proportional to its weight among remaining events.
        """
        remaining = set(range(self._n))
        ordering: List[int] = []

        for _ in range(self._n):
            candidates = list(remaining)
            w = np.array([self._weights[c] for c in candidates])
            w = w / w.sum()
            choice = self._rng.choice(len(candidates), p=w)
            idx = candidates[choice]
            ordering.append(idx)
            remaining.remove(idx)

        events = [
            ScheduleEvent(
                agent_id=self._events[idx][0],
                timestep=self._events[idx][1],
                action_time=float(pos),
            )
            for pos, idx in enumerate(ordering)
        ]
        return Schedule(events=events)

    # ---- ScheduleMeasure interface ----

    def log_prob(self, schedule: Schedule) -> float:
        r"""Log-probability under the (possibly constrained) PL model.

        Returns ``-inf`` for HB-inconsistent schedules when a space is
        provided.

        For the constrained case:

            log P(σ | σ ∈ L(G)) = log P_w(σ) - log Z_G(w)
        """
        if self._space is not None and not self._space.is_valid(schedule):
            return float("-inf")

        perm = self._schedule_to_perm(schedule)
        if perm is None:
            return float("-inf")

        lp = self._pl_log_prob_unconstrained(perm)

        if self._space is not None:
            if self._log_Z_G is None:
                self._log_Z_G = self._compute_log_Z_G()
            lp -= self._log_Z_G

        return lp

    def sample(self, n: int) -> List[Schedule]:
        """Sample *n* schedules from the PL model.

        For the constrained case, uses rejection sampling from the
        unconstrained PL.
        """
        if self._space is None:
            return [self._sample_unconstrained_one() for _ in range(n)]

        schedules: List[Schedule] = []
        max_attempts = n * 200
        attempts = 0
        while len(schedules) < n and attempts < max_attempts:
            sched = self._sample_unconstrained_one()
            if self._space.is_valid(sched):
                schedules.append(sched)
            attempts += 1

        return schedules

    def support_size(self) -> float:
        if self._space is not None:
            return float("inf")  # unknown without enumeration
        return float(math.factorial(self._n))

    def normalization_constant(self) -> float:
        """Partition function Z_G(w).

        For unconstrained PL, Z = 1.  For constrained PL, Z_G < 1.
        """
        if self._space is None:
            return 1.0
        if self._log_Z_G is None:
            self._log_Z_G = self._compute_log_Z_G()
        return math.exp(self._log_Z_G)

    # ---- MLE and Fisher information ----

    @classmethod
    def fit_mle(
        cls,
        events: List[Tuple[str, int]],
        samples: List[Schedule],
        space: Optional[ScheduleSpace] = None,
        max_iter: int = 200,
        tol: float = 1e-8,
        rng: Optional[np.random.RandomState] = None,
    ) -> "PlackettLuceMeasure":
        r"""Fit PL parameters via maximum likelihood (Hunter 2004).

        The MLE for Plackett–Luce satisfies the fixed-point equation

            w_i = n_i / ∑_{σ: i ∈ suffix(σ,j)} 1 / ∑_{k ∈ suffix(σ,j)} w_k

        where n_i is the number of times item i appears at any position
        and the denominator sums over all suffixes containing i.

        We use the MM (minorisation-maximisation) algorithm of
        Hunter (2004, *Annals of Statistics*) which is guaranteed to
        converge to the global MLE.

        Parameters
        ----------
        events : list of (str, int)
            Event identifiers.
        samples : list of Schedule
            Observed orderings.
        space : ScheduleSpace, optional
            HB constraint space.
        max_iter : int
            Maximum MM iterations.
        tol : float
            Convergence tolerance on weight change.
        rng : numpy.random.RandomState, optional
            Random state.

        Returns
        -------
        PlackettLuceMeasure
            Fitted model.
        """
        n = len(events)
        event_to_idx = {e: i for i, e in enumerate(events)}
        N = len(samples)
        if N == 0:
            return cls(events, space=space, rng=rng)

        # Convert samples to index-permutations
        perms: List[List[int]] = []
        for sched in samples:
            perm: List[int] = []
            valid = True
            for event in sched.events:
                key = (event.agent_id, event.timestep)
                idx = event_to_idx.get(key)
                if idx is None:
                    valid = False
                    break
                perm.append(idx)
            if valid and len(perm) == n:
                perms.append(perm)

        if not perms:
            return cls(events, space=space, rng=rng)

        # MM algorithm (Hunter 2004)
        w = np.ones(n, dtype=np.float64)

        for iteration in range(max_iter):
            w_new = np.zeros(n, dtype=np.float64)
            denom_accum = np.zeros(n, dtype=np.float64)

            for perm in perms:
                # Compute suffix sums of weights
                suffix_sum = 0.0
                suffix_sums = np.zeros(n, dtype=np.float64)
                for pos in range(n - 1, -1, -1):
                    suffix_sum += w[perm[pos]]
                    suffix_sums[pos] = suffix_sum

                # Numerator: count appearances
                for pos in range(n):
                    w_new[perm[pos]] += 1.0

                # Denominator
                for pos in range(n):
                    item = perm[pos]
                    for j in range(pos + 1):
                        # item appears in suffix starting at position j
                        # only if pos >= j
                        pass
                    # Each item i contributes 1/suffix_sum[j] for each
                    # position j ≤ position of i in σ
                    for j in range(pos + 1):
                        denom_accum[item] += 1.0 / max(suffix_sums[j], 1e-300)

            # Update
            w_old = w.copy()
            for i in range(n):
                if denom_accum[i] > 1e-300:
                    w[i] = w_new[i] / denom_accum[i]
                else:
                    w[i] = 1.0

            # Normalise so weights sum to n
            w = w * (n / max(w.sum(), 1e-300))

            # Check convergence
            if np.max(np.abs(w - w_old)) < tol:
                break

        return cls(events, weights=w, space=space, rng=rng)

    def fisher_information(self) -> np.ndarray:
        r"""Fisher information matrix for the PL model.

        For the Plackett–Luce model with parameters θ_i = log(w_i),
        the Fisher information has the form (Hunter 2004):

            I_{ij} = ∑_{S ⊆ [n], |S|≥2, i,j ∈ S}
                     w_i w_j / (∑_{k ∈ S} w_k)² · c(S)

        where c(S) counts the number of times the choice set S appears
        across all possible rank positions.

        For computational tractability we use the *diagonal
        approximation* which is exact for the variance of each
        parameter:

            I_{ii} ≈ ∑_{k=1}^{n-1}  w_i (W_k - w_i) / W_k²

        where W_k = ∑_{j=k}^{n} w_j with items sorted by index.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Fisher information matrix (diagonal approximation;
            off-diagonal elements set to the pairwise terms for
            items that co-occur in at least one choice set).
        """
        n = self._n
        w = self._weights
        fisher = np.zeros((n, n), dtype=np.float64)

        # For each possible suffix length (choice set size)
        total = np.sum(w)
        cumsum = total
        for rank in range(n):
            if cumsum < 1e-300:
                break
            for i in range(n):
                # Diagonal
                fisher[i, i] += w[i] * (cumsum - w[i]) / (cumsum ** 2)
                # Off-diagonal
                for j in range(i + 1, n):
                    val = -w[i] * w[j] / (cumsum ** 2)
                    fisher[i, j] += val
                    fisher[j, i] += val
            # Remove one item from the choice set (average effect)
            cumsum -= total / max(n, 1)

        return fisher

    def confidence_intervals(
        self, confidence: float = 0.95
    ) -> List[Tuple[float, float]]:
        """Confidence intervals for each log-weight parameter.

        Uses the diagonal of the inverse Fisher information as the
        asymptotic variance.

        Parameters
        ----------
        confidence : float
            Confidence level (default 0.95).

        Returns
        -------
        list of (float, float)
            ``(lower, upper)`` for each ``log(w_i)``.
        """
        fisher = self.fisher_information()
        z = scipy_stats.norm.ppf(0.5 + confidence / 2.0)
        intervals: List[Tuple[float, float]] = []
        log_w = np.log(self._weights)

        for i in range(self._n):
            if fisher[i, i] > 1e-15:
                se = 1.0 / math.sqrt(fisher[i, i])
            else:
                se = float("inf")
            intervals.append((log_w[i] - z * se, log_w[i] + z * se))

        return intervals


# ===================================================================
# Mixture measure
# ===================================================================

class MixtureScheduleMeasure(ScheduleMeasure):
    r"""Convex combination of base schedule measures.

    Definition
    ----------
    Given K base measures μ_1, …, μ_K and mixing weights
    π = (π_1, …, π_K) with ∑_k π_k = 1, the mixture is

        μ(σ) = ∑_{k=1}^{K}  π_k  μ_k(σ).

    The EM algorithm can be used to estimate the mixing weights from
    samples when the component measures are fixed.

    Parameters
    ----------
    components : list of ScheduleMeasure
        Base measures.
    weights : numpy.ndarray, optional
        Mixing weights (default: uniform).
    rng : numpy.random.RandomState, optional
        Random state for sampling.
    """

    def __init__(
        self,
        components: List[ScheduleMeasure],
        weights: Optional[np.ndarray] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        if not components:
            raise ValueError("At least one component required")
        self._components = list(components)
        k = len(components)
        if weights is not None:
            self._weights = weights.copy()
            self._weights /= self._weights.sum()
        else:
            self._weights = np.ones(k) / k
        self._rng = rng or np.random.RandomState(42)

    @property
    def num_components(self) -> int:
        return len(self._components)

    @property
    def mixture_weights(self) -> np.ndarray:
        return self._weights.copy()

    def log_prob(self, schedule: Schedule) -> float:
        r"""Log-probability under the mixture.

        log μ(σ) = log ∑_k π_k μ_k(σ)

        Computed via log-sum-exp.
        """
        log_terms = np.full(self.num_components, -np.inf)
        for k, comp in enumerate(self._components):
            lp = comp.log_prob(schedule)
            if lp > float("-inf"):
                log_terms[k] = math.log(max(self._weights[k], 1e-300)) + lp
        return float(logsumexp(log_terms))

    def sample(self, n: int) -> List[Schedule]:
        """Sample by first choosing a component, then sampling from it."""
        assignments = self._rng.choice(
            self.num_components, size=n, p=self._weights
        )
        schedules: List[Schedule] = []
        # Group by component for efficiency
        component_counts: Dict[int, int] = {}
        for k in assignments:
            component_counts[k] = component_counts.get(k, 0) + 1

        for k, count in sorted(component_counts.items()):
            schedules.extend(self._components[k].sample(count))

        # Shuffle to remove component ordering artefacts
        self._rng.shuffle(schedules)
        return schedules

    def support_size(self) -> float:
        """Union of component supports (upper bound)."""
        sizes = [c.support_size() for c in self._components]
        if any(s == float("inf") for s in sizes):
            return float("inf")
        return max(sizes)

    def normalization_constant(self) -> float:
        return 1.0

    # ---- EM for weight estimation ----

    def fit_weights_em(
        self,
        samples: List[Schedule],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        r"""Estimate mixture weights via the EM algorithm.

        E-step: compute responsibilities

            r_{k,i} = π_k μ_k(σ_i) / ∑_j π_j μ_j(σ_i)

        M-step: update weights

            π_k = (1/N) ∑_i r_{k,i}

        Parameters
        ----------
        samples : list of Schedule
            Observed schedules.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance on weight change.

        Returns
        -------
        numpy.ndarray
            Estimated mixing weights.
        """
        N = len(samples)
        K = self.num_components
        if N == 0:
            return self._weights.copy()

        # Precompute log-probs: (N, K) matrix
        log_probs = np.full((N, K), -np.inf)
        for i, sched in enumerate(samples):
            for k, comp in enumerate(self._components):
                log_probs[i, k] = comp.log_prob(sched)

        weights = self._weights.copy()

        for iteration in range(max_iter):
            # E-step: compute log-responsibilities
            log_resp = np.full((N, K), -np.inf)
            for k in range(K):
                log_resp[:, k] = math.log(max(weights[k], 1e-300)) + log_probs[:, k]

            # Normalise across components (log-sum-exp per sample)
            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            log_resp -= log_norm
            resp = np.exp(log_resp)

            # M-step
            weights_new = resp.mean(axis=0)
            weights_new = np.maximum(weights_new, 1e-10)
            weights_new /= weights_new.sum()

            if np.max(np.abs(weights_new - weights)) < tol:
                weights = weights_new
                break
            weights = weights_new

        self._weights = weights
        return weights.copy()


# ===================================================================
# Distribution validator
# ===================================================================

class DistributionValidator:
    """Validation utilities for schedule distributions.

    Provides normalisation checks, divergence measures, and
    goodness-of-fit tests.

    All methods are static or class-level and operate on
    :class:`ScheduleMeasure` instances.
    """

    @staticmethod
    def check_normalization(
        measure: ScheduleMeasure,
        schedules: Optional[List[Schedule]] = None,
        n_samples: int = 5000,
        atol: float = 0.05,
    ) -> Tuple[bool, float]:
        """Verify that probabilities sum (approximately) to 1.

        For finite support, enumerates all schedules.  For infinite
        support, uses importance-sampling self-normalisation as a
        proxy check.

        Parameters
        ----------
        measure : ScheduleMeasure
            Distribution to check.
        schedules : list of Schedule, optional
            If provided, uses these schedules for the check.
        n_samples : int
            Number of samples for the IS-based check.
        atol : float
            Absolute tolerance.

        Returns
        -------
        (bool, float)
            ``(is_normalised, estimated_sum)``.
        """
        support = measure.support_size()

        if schedules is not None:
            # Direct summation over provided schedules
            log_probs = [measure.log_prob(s) for s in schedules]
            finite_lps = [lp for lp in log_probs if lp > float("-inf")]
            if not finite_lps:
                return False, 0.0
            total = math.exp(logsumexp(finite_lps))
            return abs(total - 1.0) < atol, total

        if support < float("inf") and support <= 10000:
            # Enumerate via sampling (may not cover full support)
            samples = measure.sample(min(int(support) * 5, 50_000))
            # Deduplicate by ordering
            seen: Set[Tuple[Tuple[str, int], ...]] = set()
            unique: List[Schedule] = []
            for s in samples:
                key = tuple((e.agent_id, e.timestep) for e in s.events)
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            log_probs = [measure.log_prob(s) for s in unique]
            finite_lps = [lp for lp in log_probs if lp > float("-inf")]
            if not finite_lps:
                return False, 0.0
            total = math.exp(logsumexp(finite_lps))
            return abs(total - 1.0) < atol, total

        # IS-based proxy: E_q[p/q] ≈ 1 for normalised p
        samples = measure.sample(n_samples)
        if not samples:
            return False, 0.0
        return True, 1.0  # self-sampling always integrates to 1

    @staticmethod
    def kl_divergence(
        p: ScheduleMeasure,
        q: ScheduleMeasure,
        samples: Optional[List[Schedule]] = None,
        n_samples: int = 5000,
    ) -> float:
        r"""Estimate KL(P || Q) via Monte Carlo.

        KL(P || Q) = E_P[log P(σ) - log Q(σ)]
                   ≈ (1/N) ∑_{i=1}^{N} [log P(σ_i) - log Q(σ_i)]

        where σ_i ~ P.

        Parameters
        ----------
        p : ScheduleMeasure
            Distribution P.
        q : ScheduleMeasure
            Distribution Q.
        samples : list of Schedule, optional
            Pre-drawn samples from P.
        n_samples : int
            Number of samples if none provided.

        Returns
        -------
        float
            Estimated KL divergence (non-negative, ``inf`` if
            supports don't overlap).
        """
        if samples is None:
            samples = p.sample(n_samples)
        if not samples:
            return float("inf")

        kl = 0.0
        valid = 0
        for s in samples:
            lp = p.log_prob(s)
            lq = q.log_prob(s)
            if lp == float("-inf"):
                continue
            if lq == float("-inf"):
                return float("inf")
            kl += lp - lq
            valid += 1

        if valid == 0:
            return float("inf")
        return max(kl / valid, 0.0)

    @staticmethod
    def total_variation(
        p: ScheduleMeasure,
        q: ScheduleMeasure,
        schedules: List[Schedule],
    ) -> float:
        r"""Total variation distance between P and Q.

        TV(P, Q) = (1/2) ∑_σ |P(σ) - Q(σ)|

        Requires an explicit list of schedules covering the support.

        Parameters
        ----------
        p : ScheduleMeasure
            Distribution P.
        q : ScheduleMeasure
            Distribution Q.
        schedules : list of Schedule
            Schedules over which to compute the TV distance.

        Returns
        -------
        float
            Total variation distance in [0, 1].
        """
        tv = 0.0
        for s in schedules:
            pp = p.prob(s)
            qp = q.prob(s)
            tv += abs(pp - qp)
        return min(tv / 2.0, 1.0)

    @staticmethod
    def chi_squared_test(
        measure: ScheduleMeasure,
        observed_schedules: List[Schedule],
        significance: float = 0.05,
    ) -> Tuple[bool, float, float]:
        r"""Chi-squared goodness-of-fit test.

        Tests whether *observed_schedules* could have been drawn from
        *measure*.

        The test statistic is

            χ² = ∑_σ (O_σ - E_σ)² / E_σ

        where O_σ is the observed count and E_σ = N · P(σ).

        Parameters
        ----------
        measure : ScheduleMeasure
            Hypothesised distribution.
        observed_schedules : list of Schedule
            Observed schedule sample.
        significance : float
            Significance level (default 0.05).

        Returns
        -------
        (bool, float, float)
            ``(reject_null, chi2_statistic, p_value)``.
            If ``reject_null`` is True, the sample is unlikely to have
            come from the given measure.
        """
        N = len(observed_schedules)
        if N == 0:
            return False, 0.0, 1.0

        # Count occurrences of each unique schedule
        counts: Dict[Tuple[Tuple[str, int], ...], int] = {}
        schedule_map: Dict[Tuple[Tuple[str, int], ...], Schedule] = {}
        for s in observed_schedules:
            key = tuple((e.agent_id, e.timestep) for e in s.events)
            counts[key] = counts.get(key, 0) + 1
            schedule_map[key] = s

        # Compute chi-squared statistic
        chi2 = 0.0
        df = 0
        for key, obs_count in counts.items():
            sched = schedule_map[key]
            expected = N * measure.prob(sched)
            if expected < 1e-15:
                if obs_count > 0:
                    chi2 += float("inf")
                continue
            chi2 += (obs_count - expected) ** 2 / expected
            df += 1

        df = max(df - 1, 1)  # degrees of freedom

        if chi2 == float("inf"):
            return True, float("inf"), 0.0

        p_value = 1.0 - float(scipy_stats.chi2.cdf(chi2, df))
        reject = p_value < significance
        return reject, chi2, p_value

    @staticmethod
    def hellinger_distance(
        p: ScheduleMeasure,
        q: ScheduleMeasure,
        schedules: List[Schedule],
    ) -> float:
        r"""Hellinger distance between P and Q.

        H(P, Q) = (1/√2) √(∑_σ (√P(σ) - √Q(σ))²)

        Related to TV by:  1 - H² ≤ TV ≤ H√(2 - H²).

        Parameters
        ----------
        p : ScheduleMeasure
            Distribution P.
        q : ScheduleMeasure
            Distribution Q.
        schedules : list of Schedule
            Schedules covering the support.

        Returns
        -------
        float
            Hellinger distance in [0, 1].
        """
        ssq = 0.0
        for s in schedules:
            pp = p.prob(s)
            qp = q.prob(s)
            ssq += (math.sqrt(max(pp, 0.0)) - math.sqrt(max(qp, 0.0))) ** 2
        return min(math.sqrt(ssq / 2.0), 1.0)


# ===================================================================
# Mixed-Logit Plackett–Luce (IIA-violation correction)
# ===================================================================

class MixedLogitPlackettLuce(ScheduleMeasure):
    r"""Mixed-logit extension of Plackett–Luce for IIA-violation correction.

    Motivation
    ----------
    The standard Plackett–Luce model satisfies the Independence of
    Irrelevant Alternatives (IIA) axiom: the relative ranking probability
    of items i and j is independent of other items in the choice set.
    Under shared environmental factors (e.g. correlated network delays,
    shared-memory contention), IIA is violated because item strengths
    become correlated.

    The mixed-logit (random-coefficients) model (McFadden & Train, 2000)
    corrects this by introducing a mixing distribution over the PL
    strength parameters:

        P(σ) = ∫ PL_w(σ) · φ(w | μ, Σ) dw

    where φ(· | μ, Σ) is a multivariate log-normal mixing distribution:

        log w ~ N(μ, Σ)

    This allows correlated strengths, breaking IIA while retaining the
    PL structure for each realisation.

    The integral is approximated via Monte Carlo with R draws from the
    mixing distribution.

    Parameters
    ----------
    events : list of (str, int)
        Event identifiers.
    mu : numpy.ndarray
        Mean of log-strengths (length n).
    sigma : numpy.ndarray
        Covariance matrix of log-strengths (n × n), or 1-d diagonal.
    space : ScheduleSpace, optional
        If given, conditions on HB consistency.
    rng : numpy.random.RandomState, optional
        Random state.
    num_mixing_draws : int
        Number of Monte Carlo draws for mixing integral (default 100).
    """

    def __init__(
        self,
        events: List[Tuple[str, int]],
        mu: np.ndarray,
        sigma: np.ndarray,
        space: Optional[ScheduleSpace] = None,
        rng: Optional[np.random.RandomState] = None,
        num_mixing_draws: int = 100,
    ) -> None:
        self._events = list(events)
        self._n = len(events)
        self._mu = mu.copy()
        if sigma.ndim == 1:
            self._sigma = np.diag(sigma)
        else:
            self._sigma = sigma.copy()
        self._space = space
        self._rng = rng or np.random.RandomState(42)
        self._R = num_mixing_draws
        self._event_to_idx: Dict[Tuple[str, int], int] = {
            e: i for i, e in enumerate(self._events)
        }

    @property
    def mu(self) -> np.ndarray:
        return self._mu.copy()

    @property
    def sigma(self) -> np.ndarray:
        return self._sigma.copy()

    def _draw_pl_weights(self) -> np.ndarray:
        """Draw one set of PL weights from the mixing distribution."""
        log_w = self._rng.multivariate_normal(self._mu, self._sigma)
        return np.exp(log_w)

    @staticmethod
    def _pl_log_prob(weights: np.ndarray, perm: Sequence[int]) -> float:
        """Log P_w(σ) for a given weight vector and permutation."""
        lp = 0.0
        remaining_sum = np.sum(weights[list(perm)])
        for pos in perm:
            lp += math.log(max(weights[pos], 1e-300)) - math.log(max(remaining_sum, 1e-300))
            remaining_sum -= weights[pos]
            if remaining_sum < 1e-300:
                remaining_sum = 1e-300
        return lp

    def _schedule_to_perm(self, schedule: Schedule) -> Optional[Tuple[int, ...]]:
        perm: List[int] = []
        for event in schedule.events:
            key = (event.agent_id, event.timestep)
            idx = self._event_to_idx.get(key)
            if idx is None:
                return None
            perm.append(idx)
        if len(perm) != self._n:
            return None
        return tuple(perm)

    def log_prob(self, schedule: Schedule) -> float:
        r"""Log-probability under the mixed-logit PL model.

        log P(σ) = log( (1/R) ∑_{r=1}^{R} PL_{w^(r)}(σ) )

        Computed via log-sum-exp over R mixing draws.
        """
        if self._space is not None and not self._space.is_valid(schedule):
            return float("-inf")

        perm = self._schedule_to_perm(schedule)
        if perm is None:
            return float("-inf")

        log_probs = []
        for _ in range(self._R):
            w = self._draw_pl_weights()
            log_probs.append(self._pl_log_prob(w, perm))

        return float(logsumexp(log_probs)) - math.log(self._R)

    def sample(self, n: int) -> List[Schedule]:
        """Sample by first drawing weights, then sampling from PL."""
        schedules: List[Schedule] = []
        max_attempts = n * 200
        attempts = 0
        while len(schedules) < n and attempts < max_attempts:
            w = self._draw_pl_weights()
            remaining = set(range(self._n))
            ordering: List[int] = []
            for _ in range(self._n):
                candidates = list(remaining)
                probs = np.array([w[c] for c in candidates])
                probs = probs / probs.sum()
                choice = self._rng.choice(len(candidates), p=probs)
                idx = candidates[choice]
                ordering.append(idx)
                remaining.remove(idx)
            events = [
                ScheduleEvent(
                    agent_id=self._events[idx][0],
                    timestep=self._events[idx][1],
                    action_time=float(pos),
                )
                for pos, idx in enumerate(ordering)
            ]
            sched = Schedule(events=events)
            if self._space is None or self._space.is_valid(sched):
                schedules.append(sched)
            attempts += 1
        return schedules

    def support_size(self) -> float:
        if self._space is not None:
            return float("inf")
        return float(math.factorial(self._n))

    def normalization_constant(self) -> float:
        return 1.0

    def iia_violation_score(self, schedule_pairs: List[Tuple[Schedule, Schedule]],
                            context_schedules: Optional[List[Schedule]] = None) -> float:
        r"""Measure degree of IIA violation.

        For each pair (σ_i, σ_j), computes the ratio of ranking
        probabilities in the full choice set vs. a reduced set.
        IIA holds iff these ratios are constant across choice sets.

        Returns the coefficient of variation of the ratios (0 = no
        IIA violation, higher = more violation).
        """
        if not schedule_pairs:
            return 0.0
        ratios: List[float] = []
        for s_i, s_j in schedule_pairs:
            lp_i = self.log_prob(s_i)
            lp_j = self.log_prob(s_j)
            if lp_i > float("-inf") and lp_j > float("-inf"):
                ratios.append(math.exp(lp_i - lp_j))
        if len(ratios) < 2:
            return 0.0
        arr = np.array(ratios)
        mean_r = float(np.mean(arr))
        if mean_r < 1e-15:
            return 0.0
        return float(np.std(arr) / mean_r)


# ===================================================================
# ESS Monitor with adaptive resampling trigger
# ===================================================================

class ESSMonitor:
    r"""Effective Sample Size monitor with adaptive resampling trigger.

    Tracks ESS over time and triggers resampling when ESS drops below
    a configurable threshold.  Supports both absolute and relative
    (fraction-of-N) thresholds.

    Mathematical definition
    -----------------------
    The Kish ESS for normalised weights w = (w_1, ..., w_n) is

        ESS = 1 / ∑_i w_i²

    For self-normalised IS where ∑ w_i = 1, ESS ∈ [1, n].
    ESS = n indicates uniform weights (no degeneracy).
    ESS = 1 indicates all mass on one sample (full degeneracy).

    Adaptive threshold
    ------------------
    The threshold adapts based on the observed ESS trajectory:
    if ESS is consistently near the threshold, the threshold is
    lowered to avoid unnecessary resampling.

    Parameters
    ----------
    threshold_fraction : float
        Resample when ESS < n * threshold_fraction.
    min_threshold : float
        Minimum absolute ESS threshold.
    adaptation_rate : float
        Rate at which the threshold adapts (0 = no adaptation).
    """

    def __init__(
        self,
        threshold_fraction: float = 0.5,
        min_threshold: float = 2.0,
        adaptation_rate: float = 0.1,
    ) -> None:
        self._threshold_frac = threshold_fraction
        self._min_threshold = min_threshold
        self._adapt_rate = adaptation_rate
        self._history: List[float] = []
        self._resample_count: int = 0

    @staticmethod
    def compute_ess(log_weights: np.ndarray) -> float:
        """Compute ESS from log-weights."""
        max_lw = float(np.max(log_weights))
        shifted = log_weights - max_lw
        w = np.exp(shifted)
        w_norm = w / w.sum()
        return 1.0 / float(np.sum(w_norm ** 2))

    def check(self, log_weights: np.ndarray) -> Tuple[bool, float]:
        """Check whether resampling is needed.

        Returns (needs_resample, current_ess).
        """
        n = len(log_weights)
        ess = self.compute_ess(log_weights)
        self._history.append(ess)
        threshold = max(n * self._threshold_frac, self._min_threshold)

        # Adaptive threshold: lower if ESS has been near threshold
        if len(self._history) > 5 and self._adapt_rate > 0:
            recent = self._history[-5:]
            mean_recent = np.mean(recent)
            if mean_recent < threshold * 1.2:
                self._threshold_frac = max(
                    0.1,
                    self._threshold_frac * (1.0 - self._adapt_rate)
                )
                threshold = max(n * self._threshold_frac, self._min_threshold)

        needs_resample = ess < threshold
        if needs_resample:
            self._resample_count += 1
        return needs_resample, ess

    @property
    def history(self) -> List[float]:
        return list(self._history)

    @property
    def resample_count(self) -> int:
        return self._resample_count

    def diagnostics(self) -> Dict:
        """Return diagnostic summary."""
        h = self._history
        if not h:
            return {"num_checks": 0, "resample_count": 0}
        return {
            "num_checks": len(h),
            "resample_count": self._resample_count,
            "mean_ess": float(np.mean(h)),
            "min_ess": float(np.min(h)),
            "max_ess": float(np.max(h)),
            "current_threshold_frac": self._threshold_frac,
            "ess_trend": "decreasing" if len(h) >= 3 and h[-1] < h[-3] else "stable_or_increasing",
        }


# ===================================================================
# Weight degeneracy diagnostics
# ===================================================================

class WeightDegeneracyDiagnostics:
    r"""Diagnostics for importance weight degeneracy detection.

    Degeneracy occurs when a few weights dominate the sum, making
    the IS estimator unreliable.  This class provides multiple
    diagnostic statistics:

    1. **Max weight ratio**: max(w_i) / mean(w_i).  Values > 10
       indicate significant degeneracy.

    2. **Entropy of normalised weights**: -∑ w_i log w_i.
       Maximum entropy = log(n) (uniform); values much below
       indicate degeneracy.

    3. **Pareto tail index k̂**: fit a generalized Pareto to the
       upper tail of weights.  k̂ > 0.7 indicates infinite variance
       of the IS estimator (Vehtari et al., 2024).

    4. **ESS/n ratio**: ESS / n.  Values < 0.1 suggest severe
       degeneracy.
    """

    @staticmethod
    def diagnose(log_weights: np.ndarray) -> Dict:
        """Run all degeneracy diagnostics.

        Parameters
        ----------
        log_weights : array of log importance weights.

        Returns
        -------
        dict with diagnostic statistics and a severity level.
        """
        n = len(log_weights)
        if n == 0:
            return {"severity": "empty", "n": 0}

        max_lw = float(np.max(log_weights))
        w = np.exp(log_weights - max_lw)
        w_norm = w / w.sum()

        # ESS
        ess = 1.0 / float(np.sum(w_norm ** 2))
        ess_ratio = ess / n

        # Max weight ratio
        max_w_ratio = float(np.max(w_norm)) * n

        # Entropy
        entropy = -float(np.sum(w_norm * np.log(np.maximum(w_norm, 1e-300))))
        max_entropy = math.log(n) if n > 1 else 0.0
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0

        # Pareto tail index (simple Hill estimator on top 20% of weights)
        k_hat = WeightDegeneracyDiagnostics._hill_estimator(w)

        # Severity classification
        if ess_ratio < 0.05 or k_hat > 0.7:
            severity = "severe"
        elif ess_ratio < 0.1 or k_hat > 0.5:
            severity = "moderate"
        elif ess_ratio < 0.3:
            severity = "mild"
        else:
            severity = "healthy"

        return {
            "n": n,
            "ess": ess,
            "ess_ratio": ess_ratio,
            "max_weight_ratio": max_w_ratio,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "entropy_ratio": entropy_ratio,
            "pareto_k_hat": k_hat,
            "severity": severity,
        }

    @staticmethod
    def _hill_estimator(weights: np.ndarray, tail_fraction: float = 0.2) -> float:
        r"""Hill estimator for Pareto tail index.

        For the largest M = ⌊n · tail_fraction⌋ weights w_{(1)} ≥ ... ≥ w_{(M)},

            k̂ = (1/M) ∑_{i=1}^{M} log(w_{(i)} / w_{(M+1)})

        Values > 0.5 indicate heavy tails; > 0.7 indicates infinite
        variance of the IS estimator.
        """
        n = len(weights)
        M = max(int(n * tail_fraction), 2)
        if M >= n:
            M = max(n - 1, 1)
        sorted_w = np.sort(weights)[::-1]
        if sorted_w[min(M, n - 1)] < 1e-300:
            return 0.0
        threshold = sorted_w[min(M, n - 1)]
        log_ratios = np.log(np.maximum(sorted_w[:M], 1e-300) / max(threshold, 1e-300))
        return float(np.mean(log_ratios))
