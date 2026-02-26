"""
Shield safety and permissivity metrics.

Evaluates the trade-off between safety (violation avoidance) and
permissivity (return sacrifice) of the runtime shield, on a per-regime
and per-state basis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ShieldSafetyMetrics:
    """Safety-related metrics for the shield."""
    # Rate at which unshielded actions would violate safety spec
    unshielded_violation_rate: float
    # Rate at which the shield intervenes (modifies the proposed action)
    shield_intervention_rate: float
    # Rate at which even the shielded action still violates
    shielded_violation_rate: float
    # Certified upper bound on violation rate (from verification)
    certified_violation_bound: float
    # Whether empirical rate respects certified bound
    bound_respected: bool
    # Per-regime breakdown
    per_regime_violation_rate: Dict[int, float] = field(default_factory=dict)
    per_regime_intervention_rate: Dict[int, float] = field(default_factory=dict)
    # Severity-weighted violation rate
    severity_weighted_violation_rate: float = 0.0
    # Number of timesteps analysed
    n_timesteps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Shield Safety ===",
            f"Unshielded violation rate:  {self.unshielded_violation_rate:.6f}",
            f"Shield intervention rate:   {self.shield_intervention_rate:.6f}",
            f"Shielded violation rate:    {self.shielded_violation_rate:.6f}",
            f"Certified bound:            {self.certified_violation_bound:.6f}",
            f"Bound respected:            {self.bound_respected}",
            f"Severity-weighted viol.:    {self.severity_weighted_violation_rate:.6f}",
            f"Timesteps analysed:         {self.n_timesteps}",
        ]
        for r in sorted(self.per_regime_violation_rate):
            lines.append(
                f"  Regime {r}: viol={self.per_regime_violation_rate[r]:.6f}  "
                f"interv={self.per_regime_intervention_rate.get(r, 0.0):.6f}"
            )
        return "\n".join(lines)


@dataclass
class ShieldPermissivityMetrics:
    """Permissivity and cost-of-safety metrics."""
    # Overall permissivity ratio (fraction of actions left unchanged)
    overall_permissivity: float
    # Per-regime permissivity
    per_regime_permissivity: Dict[int, float] = field(default_factory=dict)
    # Per-state permissivity (discretised states)
    per_state_permissivity: Dict[str, float] = field(default_factory=dict)
    # Return difference: shielded return - unshielded return
    return_difference: float = 0.0
    # Cost of safety: fraction of return sacrificed
    cost_of_safety: float = 0.0
    # Risk-adjusted cost: Sharpe difference
    sharpe_difference: float = 0.0
    # Average action distance (L2 norm between proposed and shielded action)
    mean_action_distance: float = 0.0
    max_action_distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Shield Permissivity ===",
            f"Overall permissivity:  {self.overall_permissivity:.4f}",
            f"Return difference:     {self.return_difference:+.6f}",
            f"Cost of safety:        {self.cost_of_safety:.4%}",
            f"Sharpe difference:     {self.sharpe_difference:+.4f}",
            f"Mean action distance:  {self.mean_action_distance:.6f}",
            f"Max action distance:   {self.max_action_distance:.6f}",
        ]
        for r in sorted(self.per_regime_permissivity):
            lines.append(f"  Regime {r}: permissivity={self.per_regime_permissivity[r]:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ShieldMetricsEvaluator:
    """Evaluates shield safety, permissivity, and cost of safety.

    Parameters
    ----------
    certified_bound : verified upper bound on violation probability
    severity_weights : optional per-violation-type severity weight
    state_discretiser : callable that maps a continuous state vector to a
        string key for per-state permissivity aggregation
    """

    def __init__(
        self,
        certified_bound: float = 0.0,
        severity_weights: Optional[NDArray[np.float64]] = None,
        state_discretiser: Optional[Any] = None,
    ) -> None:
        self._certified_bound = certified_bound
        self._severity_weights = severity_weights
        self._state_disc = state_discretiser
        self._safety: Optional[ShieldSafetyMetrics] = None
        self._permissivity: Optional[ShieldPermissivityMetrics] = None

    # ---- public API --------------------------------------------------------

    def evaluate(
        self,
        proposed_actions: NDArray[np.float64],
        shielded_actions: NDArray[np.float64],
        violations_unshielded: NDArray[np.bool_],
        violations_shielded: NDArray[np.bool_],
        regimes: Optional[NDArray[np.int64]] = None,
        states: Optional[NDArray[np.float64]] = None,
        returns_shielded: Optional[NDArray[np.float64]] = None,
        returns_unshielded: Optional[NDArray[np.float64]] = None,
        violation_severities: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[ShieldSafetyMetrics, ShieldPermissivityMetrics]:
        """Full evaluation of shield behaviour.

        Parameters
        ----------
        proposed_actions : (T, A) array of actions proposed by the strategy
        shielded_actions : (T, A) array of actions after shield correction
        violations_unshielded : (T,) bool – would proposed action violate spec?
        violations_shielded : (T,) bool – does shielded action still violate?
        regimes : (T,) int array of regime labels (optional)
        states : (T, S) state vectors (optional, for per-state analysis)
        returns_shielded : (T,) per-step returns with shield
        returns_unshielded : (T,) per-step returns without shield
        violation_severities : (T,) severity of each violation (optional)

        Returns
        -------
        (ShieldSafetyMetrics, ShieldPermissivityMetrics)
        """
        proposed_actions = np.atleast_2d(proposed_actions)
        shielded_actions = np.atleast_2d(shielded_actions)
        T = proposed_actions.shape[0]

        violations_u = np.asarray(violations_unshielded, dtype=bool).ravel()
        violations_s = np.asarray(violations_shielded, dtype=bool).ravel()

        # Intervention = shield changed the action
        diffs = np.linalg.norm(proposed_actions - shielded_actions, axis=1)
        interventions = diffs > 1e-10

        self._safety = self._compute_safety(
            violations_u, violations_s, interventions,
            regimes, T, violation_severities,
        )
        self._permissivity = self._compute_permissivity(
            proposed_actions, shielded_actions, interventions,
            regimes, states,
            returns_shielded, returns_unshielded, diffs, T,
        )
        return self._safety, self._permissivity

    def get_safety_metrics(self) -> ShieldSafetyMetrics:
        if self._safety is None:
            raise RuntimeError("Call evaluate() first.")
        return self._safety

    def get_permissivity_metrics(self) -> ShieldPermissivityMetrics:
        if self._permissivity is None:
            raise RuntimeError("Call evaluate() first.")
        return self._permissivity

    # ---- safety computation ------------------------------------------------

    def _compute_safety(
        self,
        violations_u: NDArray[np.bool_],
        violations_s: NDArray[np.bool_],
        interventions: NDArray[np.bool_],
        regimes: Optional[NDArray[np.int64]],
        T: int,
        severities: Optional[NDArray[np.float64]],
    ) -> ShieldSafetyMetrics:
        unsh_rate = float(np.mean(violations_u))
        sh_rate = float(np.mean(violations_s))
        interv_rate = float(np.mean(interventions))

        # Severity-weighted
        sev_rate = 0.0
        if severities is not None:
            sev = np.asarray(severities, dtype=np.float64).ravel()
            sev_rate = float(np.mean(violations_s.astype(np.float64) * sev))

        bound_ok = sh_rate <= self._certified_bound + 1e-12

        # Per-regime
        per_r_viol: Dict[int, float] = {}
        per_r_interv: Dict[int, float] = {}
        if regimes is not None:
            regimes = np.asarray(regimes, dtype=np.int64).ravel()
            for r in np.unique(regimes):
                mask = regimes == r
                per_r_viol[int(r)] = float(np.mean(violations_s[mask])) if np.any(mask) else 0.0
                per_r_interv[int(r)] = float(np.mean(interventions[mask])) if np.any(mask) else 0.0

        return ShieldSafetyMetrics(
            unshielded_violation_rate=unsh_rate,
            shield_intervention_rate=interv_rate,
            shielded_violation_rate=sh_rate,
            certified_violation_bound=self._certified_bound,
            bound_respected=bound_ok,
            per_regime_violation_rate=per_r_viol,
            per_regime_intervention_rate=per_r_interv,
            severity_weighted_violation_rate=sev_rate,
            n_timesteps=T,
        )

    # ---- permissivity computation ------------------------------------------

    def _compute_permissivity(
        self,
        proposed: NDArray[np.float64],
        shielded: NDArray[np.float64],
        interventions: NDArray[np.bool_],
        regimes: Optional[NDArray[np.int64]],
        states: Optional[NDArray[np.float64]],
        returns_s: Optional[NDArray[np.float64]],
        returns_u: Optional[NDArray[np.float64]],
        diffs: NDArray[np.float64],
        T: int,
    ) -> ShieldPermissivityMetrics:
        overall_perm = 1.0 - float(np.mean(interventions))

        # Per-regime permissivity
        per_r_perm: Dict[int, float] = {}
        if regimes is not None:
            regimes = np.asarray(regimes, dtype=np.int64).ravel()
            for r in np.unique(regimes):
                mask = regimes == r
                per_r_perm[int(r)] = 1.0 - float(np.mean(interventions[mask]))

        # Per-state permissivity
        per_s_perm: Dict[str, float] = {}
        if states is not None and self._state_disc is not None:
            state_keys = [self._state_disc(states[t]) for t in range(T)]
            from collections import defaultdict
            buckets: Dict[str, List[bool]] = defaultdict(list)
            for t, key in enumerate(state_keys):
                buckets[key].append(bool(not interventions[t]))
            per_s_perm = {k: float(np.mean(v)) for k, v in buckets.items()}

        # Return difference and cost of safety
        ret_diff = 0.0
        cost = 0.0
        sharpe_diff = 0.0
        if returns_s is not None and returns_u is not None:
            rs = np.asarray(returns_s, dtype=np.float64).ravel()
            ru = np.asarray(returns_u, dtype=np.float64).ravel()
            cum_s = float(np.sum(rs))
            cum_u = float(np.sum(ru))
            ret_diff = cum_s - cum_u
            if abs(cum_u) > 1e-12:
                cost = -ret_diff / abs(cum_u)  # positive means safety costs

            # Sharpe difference
            sharpe_s = self._annualised_sharpe(rs)
            sharpe_u = self._annualised_sharpe(ru)
            sharpe_diff = sharpe_s - sharpe_u

        mean_dist = float(np.mean(diffs))
        max_dist = float(np.max(diffs)) if len(diffs) > 0 else 0.0

        return ShieldPermissivityMetrics(
            overall_permissivity=overall_perm,
            per_regime_permissivity=per_r_perm,
            per_state_permissivity=per_s_perm,
            return_difference=ret_diff,
            cost_of_safety=cost,
            sharpe_difference=sharpe_diff,
            mean_action_distance=mean_dist,
            max_action_distance=max_dist,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _annualised_sharpe(
        returns: NDArray[np.float64],
        ann: int = 252,
    ) -> float:
        if len(returns) < 2:
            return 0.0
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma < 1e-12:
            return 0.0
        return mu / sigma * np.sqrt(ann)

    # ---- cost of safety analysis -------------------------------------------

    @staticmethod
    def cost_of_safety_analysis(
        returns_shielded: NDArray[np.float64],
        returns_unshielded: NDArray[np.float64],
        violations_shielded: NDArray[np.bool_],
        violations_unshielded: NDArray[np.bool_],
        n_bootstrap: int = 5000,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Detailed cost-of-safety analysis with bootstrap CIs.

        Returns
        -------
        dict with return_diff, sharpe_diff, violation_diff,
        and bootstrap confidence intervals for each.
        """
        rs = np.asarray(returns_shielded, dtype=np.float64).ravel()
        ru = np.asarray(returns_unshielded, dtype=np.float64).ravel()
        vs = np.asarray(violations_shielded, dtype=bool).ravel()
        vu = np.asarray(violations_unshielded, dtype=bool).ravel()
        n = len(rs)

        rng = np.random.default_rng(42)

        boot_ret_diff = np.empty(n_bootstrap, dtype=np.float64)
        boot_viol_diff = np.empty(n_bootstrap, dtype=np.float64)
        boot_sharpe_diff = np.empty(n_bootstrap, dtype=np.float64)

        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            rs_b, ru_b = rs[idx], ru[idx]
            vs_b, vu_b = vs[idx], vu[idx]

            boot_ret_diff[b] = float(np.mean(rs_b) - np.mean(ru_b))
            boot_viol_diff[b] = float(np.mean(vu_b.astype(float)) - np.mean(vs_b.astype(float)))

            sig_s = np.std(rs_b, ddof=1)
            sig_u = np.std(ru_b, ddof=1)
            sh_s = np.mean(rs_b) / max(sig_s, 1e-12)
            sh_u = np.mean(ru_b) / max(sig_u, 1e-12)
            boot_sharpe_diff[b] = sh_s - sh_u

        alpha = (1 - confidence) / 2.0

        return {
            "return_diff": float(np.mean(rs) - np.mean(ru)),
            "return_diff_ci": (
                float(np.percentile(boot_ret_diff, 100 * alpha)),
                float(np.percentile(boot_ret_diff, 100 * (1 - alpha))),
            ),
            "violation_reduction": float(np.mean(vu.astype(float)) - np.mean(vs.astype(float))),
            "violation_reduction_ci": (
                float(np.percentile(boot_viol_diff, 100 * alpha)),
                float(np.percentile(boot_viol_diff, 100 * (1 - alpha))),
            ),
            "sharpe_diff": float(boot_sharpe_diff.mean()),
            "sharpe_diff_ci": (
                float(np.percentile(boot_sharpe_diff, 100 * alpha)),
                float(np.percentile(boot_sharpe_diff, 100 * (1 - alpha))),
            ),
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
        }

    @staticmethod
    def shield_efficiency_frontier(
        intervention_rates: NDArray[np.float64],
        violation_rates: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> Dict[str, NDArray[np.float64]]:
        """Compute the shield efficiency frontier.

        For varying shield strictness levels (parameterised by
        intervention rate), compute the resulting violation rate and
        return.  Returns the Pareto-optimal frontier.

        Parameters
        ----------
        intervention_rates : (K,) intervention rate at each strictness level
        violation_rates : (K,) violation rate at each level
        returns : (K,) total return at each level

        Returns
        -------
        dict with pareto_intervention, pareto_violation, pareto_return
        """
        K = len(intervention_rates)
        # Pareto front: minimise violation rate, maximise return
        dominated = np.zeros(K, dtype=bool)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                if violation_rates[j] <= violation_rates[i] and returns[j] >= returns[i]:
                    if violation_rates[j] < violation_rates[i] or returns[j] > returns[i]:
                        dominated[i] = True
                        break

        mask = ~dominated
        order = np.argsort(intervention_rates[mask])
        return {
            "pareto_intervention": intervention_rates[mask][order],
            "pareto_violation": violation_rates[mask][order],
            "pareto_return": returns[mask][order],
        }
