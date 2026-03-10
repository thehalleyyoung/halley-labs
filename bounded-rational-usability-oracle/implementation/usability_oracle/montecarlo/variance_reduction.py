"""
usability_oracle.montecarlo.variance_reduction — Variance reduction techniques.

Provides control variates, antithetic variates, stratified sampling,
importance sampling, and Rao–Blackwellisation for Monte Carlo trajectory
estimation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.montecarlo.types import (
    ImportanceWeight,
    TrajectoryBundle,
    VarianceEstimate,
)


# ═══════════════════════════════════════════════════════════════════════════
# ControlVariates
# ═══════════════════════════════════════════════════════════════════════════

class ControlVariates:
    """Variance reduction via control variates.

    Given an estimator X and a control variate Z with known expectation E[Z],
    the adjusted estimator is:

        X̂_cv = X - c*(Z - E[Z])

    where the optimal coefficient c* = Cov(X,Z) / Var(Z) minimises variance.

    The variance reduction factor is  1 - ρ²(X,Z)  where ρ is the correlation.
    """

    def apply(
        self,
        raw_estimates: Sequence[float],
        control_values: Sequence[float],
        control_expectation: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Apply control-variate adjustment.

        Parameters:
            raw_estimates: Original per-sample cost estimates X.
            control_values: Per-sample values of control variate Z.
            control_expectation: Known E[Z].

        Returns:
            ``(adjusted_estimates, optimal_c, variance_reduction_ratio)``
        """
        x = np.asarray(raw_estimates, dtype=np.float64)
        z = np.asarray(control_values, dtype=np.float64)
        n = len(x)

        if n < 2:
            return x.copy(), 0.0, 1.0

        cov_xz = np.cov(x, z, ddof=1)[0, 1]
        var_z = np.var(z, ddof=1)

        if var_z < 1e-15:
            return x.copy(), 0.0, 1.0

        c_star = cov_xz / var_z
        adjusted = x - c_star * (z - control_expectation)

        # Variance reduction ratio: Var(X_cv) / Var(X)
        var_x = np.var(x, ddof=1)
        if var_x < 1e-15:
            ratio = 1.0
        else:
            rho_sq = cov_xz ** 2 / (var_x * var_z)
            ratio = max(1.0 - rho_sq, 0.0)

        return adjusted, float(c_star), float(ratio)

    def estimate_variance(
        self,
        adjusted: np.ndarray,
        confidence_level: float = 0.95,
    ) -> VarianceEstimate:
        """Compute variance estimate for adjusted samples."""
        return _compute_variance_estimate(adjusted, confidence_level)


# ═══════════════════════════════════════════════════════════════════════════
# AntitheticVariates
# ═══════════════════════════════════════════════════════════════════════════

class AntitheticVariates:
    """Variance reduction via antithetic variates.

    For each pair (X, X'), the antithetic estimator is:
        Ŷ = (X + X') / 2

    Variance is reduced when Cov(X, X') < 0.  The reduction factor is:
        Var(Ŷ) = (1/4)[Var(X) + Var(X') + 2·Cov(X,X')]
    """

    def pair_estimates(
        self,
        originals: Sequence[float],
        antithetics: Sequence[float],
    ) -> Tuple[np.ndarray, float]:
        """Combine original and antithetic samples.

        Parameters:
            originals: Estimates from original samples.
            antithetics: Estimates from antithetic (complementary) samples.

        Returns:
            ``(paired_means, variance_reduction_ratio)``
        """
        x = np.asarray(originals, dtype=np.float64)
        x_prime = np.asarray(antithetics, dtype=np.float64)
        n = min(len(x), len(x_prime))

        if n == 0:
            return np.array([], dtype=np.float64), 1.0

        x, x_prime = x[:n], x_prime[:n]
        paired = (x + x_prime) / 2.0

        if n < 2:
            return paired, 1.0

        var_x = np.var(x, ddof=1)
        var_paired = np.var(paired, ddof=1)

        ratio = var_paired / var_x if var_x > 1e-15 else 1.0
        return paired, float(ratio)

    def estimate_variance(
        self,
        paired: np.ndarray,
        confidence_level: float = 0.95,
    ) -> VarianceEstimate:
        """Compute variance estimate for paired antithetic samples."""
        return _compute_variance_estimate(paired, confidence_level)


# ═══════════════════════════════════════════════════════════════════════════
# StratifiedSampling
# ═══════════════════════════════════════════════════════════════════════════

class StratifiedSampling:
    """Stratified sampling for variance reduction.

    Partitions the initial-state space into strata and allocates samples
    proportional to each stratum's weight.  The stratified estimator is:

        μ̂_st = Σₕ wₕ · X̄ₕ

    Variance is reduced because within-stratum variance is typically
    smaller than total variance:
        Var(μ̂_st) = Σₕ wₕ² · σₕ² / nₕ  ≤  Var(μ̂_mc) = σ² / n
    """

    def allocate_samples(
        self,
        strata_weights: Dict[str, float],
        total_samples: int,
        min_per_stratum: int = 1,
    ) -> Dict[str, int]:
        """Proportional allocation of samples to strata.

        Parameters:
            strata_weights: Weight (probability) of each stratum.
            total_samples: Total sample budget.
            min_per_stratum: Minimum samples allocated to each stratum.

        Returns:
            Mapping stratum → number of samples.
        """
        if not strata_weights:
            return {}

        keys = list(strata_weights.keys())
        weights = np.array([strata_weights[k] for k in keys], dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        total_w = weights.sum()
        if total_w <= 0:
            # Uniform allocation
            per = max(total_samples // len(keys), min_per_stratum)
            return {k: per for k in keys}

        weights /= total_w

        # Proportional allocation with minimum guarantee
        alloc = np.maximum(np.round(weights * total_samples).astype(int), min_per_stratum)
        diff = total_samples - alloc.sum()

        # Adjust to match total
        if diff > 0:
            indices = np.argsort(-weights)
            for i in range(diff):
                alloc[indices[i % len(indices)]] += 1
        elif diff < 0:
            indices = np.argsort(weights)
            for i in range(-diff):
                idx = indices[i % len(indices)]
                if alloc[idx] > min_per_stratum:
                    alloc[idx] -= 1

        return {k: int(alloc[i]) for i, k in enumerate(keys)}

    def combine_strata(
        self,
        strata_means: Dict[str, float],
        strata_variances: Dict[str, float],
        strata_weights: Dict[str, float],
        strata_sizes: Dict[str, int],
    ) -> Tuple[float, float]:
        """Combine stratum-level estimates into overall estimate.

        Parameters:
            strata_means: X̄ₕ per stratum.
            strata_variances: σ̂ₕ² per stratum.
            strata_weights: wₕ per stratum.
            strata_sizes: nₕ per stratum.

        Returns:
            ``(overall_mean, overall_variance)``
        """
        keys = list(strata_weights.keys())
        if not keys:
            return 0.0, 0.0

        w = np.array([strata_weights.get(k, 0.0) for k in keys], dtype=np.float64)
        w_total = w.sum()
        if w_total <= 0:
            return 0.0, 0.0
        w /= w_total

        means = np.array([strata_means.get(k, 0.0) for k in keys])
        variances = np.array([strata_variances.get(k, 0.0) for k in keys])
        sizes = np.array([strata_sizes.get(k, 1) for k in keys], dtype=np.float64)

        overall_mean = float(np.sum(w * means))
        overall_var = float(np.sum(w ** 2 * variances / np.maximum(sizes, 1.0)))

        return overall_mean, overall_var

    def optimal_allocation(
        self,
        strata_weights: Dict[str, float],
        strata_std: Dict[str, float],
        total_samples: int,
        min_per_stratum: int = 1,
    ) -> Dict[str, int]:
        """Neyman optimal allocation — allocate proportional to wₕ·σₕ.

        Minimises Var(μ̂_st) given a fixed total sample budget.

        Parameters:
            strata_weights: wₕ per stratum.
            strata_std: σₕ per stratum.
            total_samples: Total budget.
            min_per_stratum: Minimum per stratum.

        Returns:
            Optimal allocation mapping stratum → nₕ.
        """
        keys = list(strata_weights.keys())
        if not keys:
            return {}

        w = np.array([strata_weights.get(k, 0.0) for k in keys])
        s = np.array([strata_std.get(k, 0.0) for k in keys])
        ws = w * s
        ws_total = ws.sum()

        if ws_total <= 0:
            return self.allocate_samples(strata_weights, total_samples, min_per_stratum)

        alloc = np.maximum(np.round(ws / ws_total * total_samples).astype(int), min_per_stratum)
        diff = total_samples - alloc.sum()
        if diff > 0:
            indices = np.argsort(-ws)
            for i in range(diff):
                alloc[indices[i % len(indices)]] += 1
        elif diff < 0:
            indices = np.argsort(ws)
            for i in range(-diff):
                idx = indices[i % len(indices)]
                if alloc[idx] > min_per_stratum:
                    alloc[idx] -= 1

        return {k: int(alloc[i]) for i, k in enumerate(keys)}


# ═══════════════════════════════════════════════════════════════════════════
# ImportanceSampling
# ═══════════════════════════════════════════════════════════════════════════

class ImportanceSampling:
    """Importance sampling utilities.

    Provides computation of optimal proposal distributions, effective
    sample size, and self-normalised importance weights.
    """

    def compute_proposal(
        self,
        target_policy: Dict[str, Dict[str, float]],
        cost_model: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute an approximately optimal proposal distribution.

        The zero-variance proposal in state s is proportional to:
            q*(a|s) ∝ π(a|s) · |C(s,a)|

        This shifts sampling towards high-cost actions, reducing variance
        of the cost estimator.

        Parameters:
            target_policy: π(a|s).
            cost_model: C(s,a).

        Returns:
            Proposal q(a|s) in the same format.
        """
        proposal: Dict[str, Dict[str, float]] = {}

        for state, action_probs in target_policy.items():
            state_costs = cost_model.get(state, {})
            weighted: Dict[str, float] = {}
            for action, prob in action_probs.items():
                cost = abs(state_costs.get(action, 0.0)) + 1e-10
                weighted[action] = prob * cost

            # Normalise
            total = sum(weighted.values())
            if total > 0:
                proposal[state] = {a: w / total for a, w in weighted.items()}
            else:
                proposal[state] = dict(action_probs)

        return proposal

    def compute_weights(
        self,
        target_policy: Dict[str, Dict[str, float]],
        proposal_policy: Dict[str, Dict[str, float]],
        trajectories_states: Sequence[Sequence[str]],
        trajectories_actions: Sequence[Sequence[str]],
    ) -> List[ImportanceWeight]:
        """Compute importance weights for sampled trajectories.

        w(τ) = ∏_t π(aₜ|sₜ) / q(aₜ|sₜ)

        Parameters:
            target_policy: π(a|s).
            proposal_policy: q(a|s).
            trajectories_states: State sequences for each trajectory.
            trajectories_actions: Action sequences for each trajectory.

        Returns:
            List of :class:`ImportanceWeight` objects.
        """
        log_weights: List[float] = []

        for states, actions in zip(trajectories_states, trajectories_actions):
            log_w = 0.0
            for s, a in zip(states, actions):
                pi_probs = target_policy.get(s, {})
                q_probs = proposal_policy.get(s, {})
                pi_a = pi_probs.get(a, 0.0)
                q_a = q_probs.get(a, 1e-15)
                if pi_a > 0 and q_a > 0:
                    log_w += math.log(pi_a) - math.log(q_a)
                elif pi_a > 0:
                    log_w = float('inf')
                # If pi_a == 0, contribution is -inf but weight is 0
            log_weights.append(log_w)

        return _normalise_log_weights_list(log_weights)

    def estimate_variance(
        self,
        bundle: TrajectoryBundle,
    ) -> VarianceEstimate:
        """Compute variance diagnostics for a trajectory bundle.

        Parameters:
            bundle: A sampled trajectory bundle (may have importance weights).

        Returns:
            Detailed :class:`VarianceEstimate`.
        """
        costs = np.array(bundle.costs, dtype=np.float64)
        n = len(costs)
        if n == 0:
            return VarianceEstimate(
                sample_variance=0.0, standard_error=0.0,
                coefficient_of_variation=0.0,
                effective_sample_size=0.0, ess_ratio=0.0,
            )

        if bundle.importance_weights is not None:
            w = np.array([iw.normalised_weight for iw in bundle.importance_weights])
            mean = float(np.sum(w * costs))
            var = float(np.sum(w * (costs - mean) ** 2))
            ess = compute_effective_sample_size(w)
        else:
            mean = float(np.mean(costs))
            var = float(np.var(costs, ddof=1)) if n > 1 else 0.0
            ess = float(n)

        se = math.sqrt(max(var, 0.0) / n) if n > 0 else 0.0
        cv = math.sqrt(max(var, 0.0)) / abs(mean) if abs(mean) > 1e-15 else 0.0

        return VarianceEstimate(
            sample_variance=var,
            standard_error=se,
            coefficient_of_variation=cv,
            effective_sample_size=ess,
            ess_ratio=ess / n if n > 0 else 0.0,
        )

    def apply_control_variate(
        self,
        raw_estimates: Sequence[float],
        control_values: Sequence[float],
        control_expectation: float,
    ) -> Sequence[float]:
        """Apply a control-variate adjustment (delegates to ControlVariates)."""
        cv = ControlVariates()
        adjusted, _, _ = cv.apply(raw_estimates, control_values, control_expectation)
        return adjusted.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Module-level functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size from normalised importance weights.

    ESS = (Σ wᵢ)² / Σ wᵢ²

    For self-normalised weights (Σ wᵢ = 1):  ESS = 1 / Σ wᵢ²

    Parameters:
        weights: Array of (normalised) importance weights.

    Returns:
        Effective sample size.  For uniform weights ESS ≈ n.
    """
    w = np.asarray(weights, dtype=np.float64)
    if len(w) == 0:
        return 0.0
    w_sum = w.sum()
    w_sq_sum = np.sum(w ** 2)
    if w_sq_sum < 1e-30:
        return 0.0
    return float(w_sum ** 2 / w_sq_sum)


def compute_optimal_proposal(
    target_policy: Dict[str, Dict[str, float]],
    cost_model: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Compute proposal distribution that minimises importance-sampling variance.

    Convenience wrapper around :meth:`ImportanceSampling.compute_proposal`.
    """
    return ImportanceSampling().compute_proposal(target_policy, cost_model)


def rao_blackwellize(
    samples: Sequence[float],
    sufficient_statistics: Dict[str, Sequence[float]],
) -> np.ndarray:
    """Rao–Blackwell improvement using sufficient statistics.

    For each sample Xᵢ, we compute E[Xᵢ | T(X)] where T is a sufficient
    statistic.  This reduces variance because Var(E[X|T]) ≤ Var(X).

    In practice, we group samples by binned sufficient statistic and
    replace each sample with its conditional mean within the bin.

    Parameters:
        samples: Raw Monte Carlo samples.
        sufficient_statistics: Named statistics, each a sequence of
            per-sample values.  Used to define conditioning bins.

    Returns:
        Improved estimates with Var ≤ Var(samples).
    """
    x = np.asarray(samples, dtype=np.float64)
    n = len(x)
    if n < 2 or not sufficient_statistics:
        return x.copy()

    # Use the first sufficient statistic for binning
    stat_name = next(iter(sufficient_statistics))
    stat_vals = np.asarray(sufficient_statistics[stat_name], dtype=np.float64)

    if len(stat_vals) != n:
        return x.copy()

    # Bin by quantiles of the statistic
    n_bins = max(min(int(np.sqrt(n)), 50), 2)
    try:
        bin_edges = np.percentile(stat_vals, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(stat_vals, bin_edges[1:-1])
    except (ValueError, IndexError):
        return x.copy()

    improved = x.copy()
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            improved[mask] = np.mean(x[mask])

    return improved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_log_weights_list(log_weights: List[float]) -> List[ImportanceWeight]:
    """Convert log weights to normalised ImportanceWeight objects."""
    if not log_weights:
        return []

    log_w = np.array(log_weights, dtype=np.float64)
    # Handle inf/-inf
    finite_mask = np.isfinite(log_w)
    if not np.any(finite_mask):
        n = len(log_weights)
        uniform = 1.0 / n if n > 0 else 0.0
        return [
            ImportanceWeight(sample_id=i, raw_weight=uniform,
                             log_weight=0.0, normalised_weight=uniform)
            for i in range(n)
        ]

    max_log = np.max(log_w[finite_mask])
    # Clamp infinities
    log_w = np.clip(log_w, max_log - 700, max_log + 700)
    log_sum = max_log + np.log(np.sum(np.exp(log_w - max_log)))
    log_norm = log_w - log_sum

    raw = np.exp(log_w)
    normalised = np.exp(log_norm)

    return [
        ImportanceWeight(
            sample_id=i,
            raw_weight=float(raw[i]),
            log_weight=float(log_w[i]),
            normalised_weight=float(normalised[i]),
        )
        for i in range(len(log_weights))
    ]


def _compute_variance_estimate(
    samples: np.ndarray,
    confidence_level: float = 0.95,
) -> VarianceEstimate:
    """Compute variance estimate for an array of samples."""
    n = len(samples)
    if n == 0:
        return VarianceEstimate(
            sample_variance=0.0, standard_error=0.0,
            coefficient_of_variation=0.0,
            effective_sample_size=0.0, ess_ratio=0.0,
        )

    mean = float(np.mean(samples))
    var = float(np.var(samples, ddof=1)) if n > 1 else 0.0
    se = math.sqrt(max(var, 0.0) / n) if n > 0 else 0.0
    cv = math.sqrt(max(var, 0.0)) / abs(mean) if abs(mean) > 1e-15 else 0.0

    return VarianceEstimate(
        sample_variance=var,
        standard_error=se,
        coefficient_of_variation=cv,
        effective_sample_size=float(n),
        ess_ratio=1.0,
    )
