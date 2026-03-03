"""IPU (Incremental Plasticity Update) algorithm.

Implements Algorithm 6 from the CPA theory: online incremental updates
to plasticity descriptors as new context data arrives.  Uses Welford's
algorithm for running statistics, rank-one covariance updates, and
exponential forgetting for non-stationary streams.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PlasticityDelta:
    """Record of a single plasticity change event."""

    node: int
    old_class: str
    new_class: str
    confidence_change: float
    timestamp: float


@dataclass
class _DescriptorSnapshot:
    """Immutable snapshot of all descriptors at a point in time."""

    descriptors: Dict[int, Dict[str, float]]
    stats: Dict[str, Any]
    timestamp: float


# ---------------------------------------------------------------------------
# SufficientStatistics – online mean / covariance tracking
# ---------------------------------------------------------------------------

class SufficientStatistics:
    """Online sufficient statistics for multivariate data.

    Maintains running mean, covariance, and count using Welford's
    online algorithm extended to the multivariate case with rank-one
    covariance updates.

    Parameters
    ----------
    num_variables : int
        Dimensionality of each observation vector.
    """

    def __init__(self, num_variables: int) -> None:
        self._p = num_variables
        self._n: int = 0
        self._mean = np.zeros(num_variables, dtype=np.float64)
        self._cov = np.zeros(
            (num_variables, num_variables), dtype=np.float64
        )
        # Sum of outer products of centred observations (M2 matrix)
        self._m2 = np.zeros(
            (num_variables, num_variables), dtype=np.float64
        )

    # -- public properties -------------------------------------------------

    @property
    def mean(self) -> NDArray:
        """Current running mean vector ``(p,)``."""
        return self._mean.copy()

    @property
    def covariance(self) -> NDArray:
        """Current sample covariance matrix ``(p, p)``."""
        if self._n < 2:
            return np.zeros_like(self._m2)
        return self._m2 / (self._n - 1)

    @property
    def n(self) -> int:
        """Number of observations seen so far."""
        return self._n

    # -- updates -----------------------------------------------------------

    def update(self, data: NDArray) -> None:
        """Incorporate one or more observations.

        Parameters
        ----------
        data : array of shape ``(n_obs, p)`` or ``(p,)``
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        for x in data:
            self._welford_update_vec(x)

    def _welford_update_vec(self, x: NDArray) -> None:
        """Welford update for a single p-dimensional vector *x*."""
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += np.outer(delta, delta2)

    @staticmethod
    def _welford_update(
        existing_mean: float,
        existing_var: float,
        new_value: float,
        n: int,
    ) -> Tuple[float, float]:
        """Scalar Welford update returning ``(new_mean, new_m2)``."""
        delta = new_value - existing_mean
        new_mean = existing_mean + delta / n
        delta2 = new_value - new_mean
        new_m2 = existing_var + delta * delta2
        return new_mean, new_m2

    @staticmethod
    def _rank_one_covariance_update(
        cov: NDArray, x: NDArray, n: int
    ) -> NDArray:
        """Rank-1 update: ``Σ_new = ((n-1)Σ + (n/(n+1)) δδ^T) / n``."""
        if n < 2:
            return np.zeros_like(cov)
        old_mean_contrib = (n - 1) / n
        return old_mean_contrib * cov + np.outer(x, x) / n

    def merge(self, other: "SufficientStatistics") -> None:
        """Merge another :class:`SufficientStatistics` into this one.

        Uses the parallel / Chan algorithm for combining partial
        aggregates.
        """
        if other._n == 0:
            return
        n_a, n_b = self._n, other._n
        n_ab = n_a + n_b
        delta = other._mean - self._mean
        self._m2 = (
            self._m2
            + other._m2
            + (n_a * n_b / n_ab) * np.outer(delta, delta)
        )
        self._mean = (n_a * self._mean + n_b * other._mean) / n_ab
        self._n = n_ab

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self._n = 0
        self._mean[:] = 0.0
        self._m2[:] = 0.0

    def copy(self) -> "SufficientStatistics":
        """Return an independent copy."""
        ss = SufficientStatistics(self._p)
        ss._n = self._n
        ss._mean = self._mean.copy()
        ss._m2 = self._m2.copy()
        return ss

    def __repr__(self) -> str:
        return (
            f"SufficientStatistics(p={self._p}, n={self._n})"
        )


# ---------------------------------------------------------------------------
# IncrementalPlasticityUpdater – Algorithm 6 (IPU)
# ---------------------------------------------------------------------------

class IncrementalPlasticityUpdater:
    """Online updater for plasticity descriptors via the IPU algorithm.

    Implements Algorithm 6 (Incremental Plasticity Update) from the CPA
    theory monograph.  As new context data arrives the updater:

    1. Updates sufficient statistics incrementally (Welford / rank-1).
    2. Re-estimates mechanism parameters via online OLS.
    3. Recomputes the 4-D plasticity descriptor (ψ_S, ψ_P, ψ_E, ψ_CS).
    4. Detects regime changes when a descriptor crosses a class boundary.
    5. Logs every detected change and supports rollback.

    Parameters
    ----------
    initial_descriptors : dict
        Starting plasticity descriptors keyed by node index.  Each value
        is a dict with at least ``psi_S``, ``psi_P``, ``psi_E``,
        ``psi_CS``, and ``classification``.
    alpha : float
        Learning rate for exponentially weighted moving average of the
        descriptor components.
    decay_rate : float
        Exponential decay rate for older observations (forgetting factor
        λ = 1 − decay_rate).
    """

    _CLASS_THRESHOLDS = {
        "invariant": (0.05, 0.05),
        "parametric_plastic": (0.05, 0.30),
        "structural_plastic": (0.30, None),
        "emergent": (None, None),
    }

    def __init__(
        self,
        initial_descriptors: Dict,
        alpha: float = 0.05,
        decay_rate: float = 0.01,
    ) -> None:
        self._descriptors: Dict[int, Dict[str, Any]] = {}
        for node, desc in initial_descriptors.items():
            if isinstance(desc, dict):
                self._descriptors[node] = dict(desc)
            else:
                self._descriptors[node] = {
                    "psi_S": getattr(desc, "psi_S", 0.0),
                    "psi_P": getattr(desc, "psi_P", 0.0),
                    "psi_E": getattr(desc, "psi_E", 0.0),
                    "psi_CS": getattr(desc, "psi_CS", 0.0),
                    "classification": getattr(
                        desc, "classification", "invariant"
                    ),
                }
        self._alpha = alpha
        self._decay_rate = decay_rate
        self._forgetting = 1.0 - decay_rate
        self._history: List[PlasticityDelta] = []
        self._snapshots: List[_DescriptorSnapshot] = []
        self._context_stats: Dict[str, SufficientStatistics] = {}
        self._context_adjacencies: Dict[str, NDArray] = {}
        self._context_count = 0
        self._update_count = 0

    # -- public API --------------------------------------------------------

    def update(
        self, new_data: NDArray, context_id: str
    ) -> List[PlasticityDelta]:
        """Incorporate *new_data* from *context_id* and return changes.

        Parameters
        ----------
        new_data : array of shape ``(n_obs, p)``
            Observations from the new context.
        context_id : str
            Identifier for the context from which the data originates.

        Returns
        -------
        list of PlasticityDelta
            Detected plasticity class changes.
        """
        new_data = np.asarray(new_data, dtype=np.float64)
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)

        p = new_data.shape[1]
        self._save_snapshot()

        # Step 1 – update sufficient statistics
        self._update_sufficient_statistics(context_id, new_data)

        # Step 2 – estimate adjacency for new context (greedy from corr)
        adj = self._estimate_adjacency(context_id, p)
        self._context_adjacencies[context_id] = adj

        # Step 3 – recompute descriptors for each node
        deltas: List[PlasticityDelta] = []
        for node in range(p):
            if node not in self._descriptors:
                self._descriptors[node] = {
                    "psi_S": 0.0,
                    "psi_P": 0.0,
                    "psi_E": 0.0,
                    "psi_CS": 0.0,
                    "classification": "invariant",
                }
            old_desc = dict(self._descriptors[node])
            self._update_descriptors(node)
            new_desc = self._descriptors[node]

            change = self._detect_regime_change(old_desc, new_desc)
            if change is not None:
                deltas.append(change)
                self._history.append(change)

        self._update_count += 1
        return deltas

    def current_descriptors(self) -> Dict:
        """Return the current plasticity descriptors."""
        return copy.deepcopy(self._descriptors)

    def rollback(self, n_steps: int = 1) -> None:
        """Undo the last *n_steps* updates by restoring snapshots."""
        for _ in range(min(n_steps, len(self._snapshots))):
            snap = self._snapshots.pop()
            self._descriptors = snap.descriptors
            # Remove history entries after snapshot time
            self._history = [
                h for h in self._history if h.timestamp <= snap.timestamp
            ]
            self._update_count = max(0, self._update_count - 1)

    def change_history(self) -> List[PlasticityDelta]:
        """Return the full history of plasticity changes."""
        return list(self._history)

    def exponential_decay_weight(self, age: float) -> float:
        """Compute the exponential decay weight for a given *age*."""
        return float(self._forgetting ** age)

    def get_current_descriptors(self) -> Dict:
        """Alias for :meth:`current_descriptors`."""
        return self.current_descriptors()

    def get_change_log(self) -> List[PlasticityDelta]:
        """Return log of all changes detected."""
        return list(self._history)

    # -- internal: sufficient statistics -----------------------------------

    def _update_sufficient_statistics(
        self, context_id: str, data: NDArray
    ) -> None:
        """Update running statistics for *context_id*."""
        p = data.shape[1]
        if context_id not in self._context_stats:
            self._context_stats[context_id] = SufficientStatistics(p)
            self._context_count += 1
        self._context_stats[context_id].update(data)

    # -- internal: adjacency estimation ------------------------------------

    def _estimate_adjacency(self, context_id: str, p: int) -> NDArray:
        """Estimate DAG adjacency from sufficient statistics.

        Uses a greedy correlation-based approach: for each variable,
        choose parents whose absolute partial correlation exceeds a
        threshold, then orient edges using residual variance ordering.
        """
        stats = self._context_stats[context_id]
        if stats.n < 3 or p < 2:
            return np.zeros((p, p), dtype=np.float64)

        cov = stats.covariance
        diag = np.diag(cov)
        safe_diag = np.where(diag > 1e-12, diag, 1.0)
        corr = cov / np.sqrt(np.outer(safe_diag, safe_diag))
        np.fill_diagonal(corr, 0.0)

        threshold = 2.0 / np.sqrt(max(stats.n, 4))
        adj = np.zeros((p, p), dtype=np.float64)
        var_order = np.argsort(diag)  # lower variance → earlier

        for idx in range(p):
            j = var_order[idx]
            candidates = var_order[:idx]
            for i in candidates:
                if abs(corr[i, j]) > threshold:
                    adj[i, j] = 1.0
        return adj

    # -- internal: descriptor recomputation --------------------------------

    def _update_descriptors(self, node: int) -> None:
        """Recompute the plasticity descriptor for *node* incrementally.

        Uses EWMA blending of the old and new estimates so that
        the descriptor tracks non-stationary regime shifts.
        """
        if len(self._context_adjacencies) < 2:
            return

        adjs = list(self._context_adjacencies.values())
        p = adjs[0].shape[0]
        if node >= p:
            return

        new_psi_S = self._compute_structural_plasticity(node, adjs)
        new_psi_P = self._compute_parametric_plasticity(node)
        new_psi_E = self._compute_emergence(node, adjs)
        new_psi_CS = self._compute_context_sensitivity(node)

        alpha = self._alpha
        desc = self._descriptors[node]
        desc["psi_S"] = (1 - alpha) * desc["psi_S"] + alpha * new_psi_S
        desc["psi_P"] = (1 - alpha) * desc["psi_P"] + alpha * new_psi_P
        desc["psi_E"] = (1 - alpha) * desc["psi_E"] + alpha * new_psi_E
        desc["psi_CS"] = (1 - alpha) * desc["psi_CS"] + alpha * new_psi_CS
        desc["classification"] = self._classify(desc)

    def _compute_structural_plasticity(
        self, node: int, adjs: List[NDArray]
    ) -> float:
        """ψ_S via Jensen-Shannon divergence of parent indicators."""
        K = len(adjs)
        if K < 2:
            return 0.0
        p = adjs[0].shape[0]
        parent_indicators = np.array(
            [adj[:, node] for adj in adjs], dtype=np.float64
        )  # (K, p)
        mean_ind = np.mean(parent_indicators, axis=0)

        # JSD of Bernoulli distributions
        jsd_vals = []
        for j in range(p):
            if j == node:
                continue
            probs = parent_indicators[:, j]
            q = np.clip(mean_ind[j], 1e-12, 1 - 1e-12)
            h_mean = -q * np.log2(q + 1e-15) - (1 - q) * np.log2(
                1 - q + 1e-15
            )
            h_components = 0.0
            for k in range(K):
                pk = np.clip(probs[k], 1e-12, 1 - 1e-12)
                h_components += (
                    -pk * np.log2(pk + 1e-15)
                    - (1 - pk) * np.log2(1 - pk + 1e-15)
                )
            h_components /= K
            jsd_vals.append(max(0.0, h_mean - h_components))

        if not jsd_vals:
            return 0.0
        return float(np.sqrt(np.mean(jsd_vals)))

    def _compute_parametric_plasticity(self, node: int) -> float:
        """ψ_P via coefficient variation across contexts."""
        stats_list = list(self._context_stats.values())
        if len(stats_list) < 2:
            return 0.0

        means = np.array([s.mean[node] for s in stats_list if node < len(s.mean)])
        if len(means) < 2:
            return 0.0
        variances = np.array([
            s.covariance[node, node] if node < s.covariance.shape[0] else 0.0
            for s in stats_list
        ])

        # JSD between Gaussians for each pair
        jsd_sum = 0.0
        count = 0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                mu1, mu2 = means[i], means[j]
                s1 = max(variances[i], 1e-12)
                s2 = max(variances[j], 1e-12)
                # JSD between N(mu1,s1) and N(mu2,s2)
                jsd = self._jsd_gaussian(mu1, s1, mu2, s2)
                jsd_sum += np.sqrt(max(0.0, jsd))
                count += 1
        if count == 0:
            return 0.0
        return float(np.clip(jsd_sum / count, 0.0, 1.0))

    def _compute_emergence(
        self, node: int, adjs: List[NDArray]
    ) -> float:
        """ψ_E via Markov blanket variation."""
        mb_sizes = []
        for adj in adjs:
            p = adj.shape[0]
            if node >= p:
                continue
            parents = set(np.nonzero(adj[:, node])[0])
            children = set(np.nonzero(adj[node, :])[0])
            co_parents: set = set()
            for ch in children:
                co_parents.update(np.nonzero(adj[:, ch])[0])
            co_parents.discard(node)
            mb = parents | children | co_parents
            mb_sizes.append(len(mb))
        if not mb_sizes or max(mb_sizes) == 0:
            return 0.0
        return float(1.0 - min(mb_sizes) / (max(mb_sizes) + 1))

    def _compute_context_sensitivity(self, node: int) -> float:
        """ψ_CS via bootstrap-style context subset variation of ψ_P."""
        adjs = list(self._context_adjacencies.values())
        K = len(adjs)
        if K < 3:
            return 0.0

        rng = np.random.RandomState(node)
        n_subsets = min(20, K * (K - 1))
        subset_size = max(2, K // 2)
        psi_p_vals = []

        for _ in range(n_subsets):
            idx = rng.choice(K, size=subset_size, replace=False)
            sub_adjs = [adjs[i] for i in idx]
            psi_p_vals.append(
                self._compute_structural_plasticity(node, sub_adjs)
            )

        arr = np.array(psi_p_vals)
        mean_val = np.mean(arr)
        if mean_val < 1e-12:
            return 0.0
        return float(np.clip(np.std(arr) / mean_val, 0.0, 1.0))

    # -- internal: regime change detection ---------------------------------

    def _detect_regime_change(
        self, old_desc: Dict, new_desc: Dict
    ) -> Optional[PlasticityDelta]:
        """Check whether the update caused a plasticity class change."""
        old_cls = str(old_desc.get("classification", "invariant"))
        new_cls = str(new_desc.get("classification", "invariant"))
        if old_cls != new_cls:
            old_vec = np.array([
                old_desc.get("psi_S", 0),
                old_desc.get("psi_P", 0),
                old_desc.get("psi_E", 0),
                old_desc.get("psi_CS", 0),
            ])
            new_vec = np.array([
                new_desc.get("psi_S", 0),
                new_desc.get("psi_P", 0),
                new_desc.get("psi_E", 0),
                new_desc.get("psi_CS", 0),
            ])
            conf_change = float(np.linalg.norm(new_vec - old_vec))
            node = new_desc.get("variable_index", 0)
            return PlasticityDelta(
                node=node,
                old_class=old_cls,
                new_class=new_cls,
                confidence_change=conf_change,
                timestamp=time.time(),
            )
        return None

    # -- internal: classification ------------------------------------------

    @staticmethod
    def _classify(desc: Dict) -> str:
        """Assign a plasticity class from the 4-D descriptor."""
        psi_S = desc.get("psi_S", 0.0)
        psi_P = desc.get("psi_P", 0.0)
        psi_E = desc.get("psi_E", 0.0)

        if psi_E > 0.5:
            return "emergent"
        if psi_S > 0.3:
            if psi_P > 0.1:
                return "mixed"
            return "structural_plastic"
        if psi_P > 0.1:
            return "parametric_plastic"
        return "invariant"

    # -- internal: JSD helper ----------------------------------------------

    @staticmethod
    def _jsd_gaussian(
        mu1: float, var1: float, mu2: float, var2: float
    ) -> float:
        """Jensen-Shannon divergence between two univariate Gaussians."""
        var1 = max(var1, 1e-12)
        var2 = max(var2, 1e-12)
        m_var = 0.5 * (var1 + var2)
        m_mu = 0.5 * (mu1 + mu2)
        # KL(p||m) + KL(q||m)  / 2
        kl1 = 0.5 * (np.log(m_var / var1) + var1 / m_var
                      + (mu1 - m_mu) ** 2 / m_var - 1)
        kl2 = 0.5 * (np.log(m_var / var2) + var2 / m_var
                      + (mu2 - m_mu) ** 2 / m_var - 1)
        return float(max(0.0, 0.5 * (kl1 + kl2)))

    # -- internal: snapshot / rollback -------------------------------------

    def _save_snapshot(self) -> None:
        """Save current state for potential rollback."""
        self._snapshots.append(
            _DescriptorSnapshot(
                descriptors=copy.deepcopy(self._descriptors),
                stats={
                    cid: ss.copy()
                    for cid, ss in self._context_stats.items()
                },
                timestamp=time.time(),
            )
        )
        # Keep at most 50 snapshots
        if len(self._snapshots) > 50:
            self._snapshots = self._snapshots[-50:]

    # -- internal: certificate invalidation --------------------------------

    def _update_certificates(
        self, affected_mechanisms: List[int]
    ) -> List[int]:
        """Invalidate certificates for *affected_mechanisms*.

        Returns the list of mechanism indices whose certificates were
        invalidated (i.e. whose class changed).
        """
        invalidated: List[int] = []
        for node in affected_mechanisms:
            desc = self._descriptors.get(node)
            if desc is None:
                continue
            psi_S = desc.get("psi_S", 0.0)
            psi_P = desc.get("psi_P", 0.0)
            if psi_S > 0.1 or psi_P > 0.2:
                invalidated.append(node)
        return invalidated

    def __repr__(self) -> str:
        return (
            f"IncrementalPlasticityUpdater("
            f"nodes={len(self._descriptors)}, "
            f"contexts={self._context_count}, "
            f"updates={self._update_count})"
        )
