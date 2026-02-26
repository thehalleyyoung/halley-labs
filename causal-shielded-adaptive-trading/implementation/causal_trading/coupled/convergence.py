"""
Fixed-point convergence analysis for coupled regime-causal inference.

Monitors Hamming distance between successive regime assignments,
DAG edit distance between successive causal graphs, and provides
Lyapunov-function verification and convergence-rate estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import optimize, stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceDiagnostics:
    """Diagnostics summary for the convergence analyser."""
    converged: bool
    n_iterations: int
    final_ll_change: float
    final_hamming: float
    final_dag_distance: float
    estimated_rate: Optional[float]
    lyapunov_decreasing: bool
    lyapunov_values: List[float]
    ll_trace: List[float]
    hamming_trace: List[float]
    dag_distance_trace: List[float]


@dataclass
class _SnapshotEntry:
    """Single iteration snapshot used for convergence tracking."""
    iteration: int
    regime_assignments: np.ndarray
    causal_graphs: Dict[int, nx.DiGraph]
    log_likelihood: float


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Hamming distance between two integer label arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape")
    if a.size == 0:
        return 0.0
    return float(np.mean(a != b))


def dag_edit_distance(g1: nx.DiGraph, g2: nx.DiGraph) -> int:
    """Structural Hamming Distance between two DAGs.

    Counts the number of edge additions, deletions, and reversals
    needed to turn g1 into g2.
    """
    edges1 = set(g1.edges())
    edges2 = set(g2.edges())

    # Extra / missing edges
    added = edges2 - edges1
    removed = edges1 - edges2

    # Count reversals (edges in removed whose reverse is in added)
    reversals = set()
    for u, v in removed:
        if (v, u) in added:
            reversals.add((min(u, v), max(u, v)))

    n_reversals = len(reversals)
    n_pure_add = len(added) - n_reversals
    n_pure_del = len(removed) - n_reversals

    return n_pure_add + n_pure_del + n_reversals


def total_dag_edit_distance(
    graphs1: Dict[int, nx.DiGraph],
    graphs2: Dict[int, nx.DiGraph],
) -> float:
    """Sum of DAG edit distances across all regimes."""
    all_keys = set(graphs1) | set(graphs2)
    total = 0
    for k in all_keys:
        g1 = graphs1.get(k, nx.DiGraph())
        g2 = graphs2.get(k, nx.DiGraph())
        total += dag_edit_distance(g1, g2)
    return float(total)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ConvergenceAnalyzer:
    """Monitor and analyse convergence of the coupled EM loop.

    Maintains a history of regime assignments and causal graphs to compute
    distances, estimate convergence rates, and verify Lyapunov conditions.

    Parameters
    ----------
    hamming_tol : float
        Threshold on normalised Hamming distance for regime convergence.
    dag_tol : float
        Threshold on aggregate DAG edit distance for graph convergence.
    ll_rel_tol : float
        Threshold on relative log-likelihood change.
    window : int
        Number of most recent iterations to consider for rate estimation.
    """

    def __init__(
        self,
        hamming_tol: float = 0.01,
        dag_tol: float = 1.0,
        ll_rel_tol: float = 1e-5,
        window: int = 10,
    ) -> None:
        self.hamming_tol = hamming_tol
        self.dag_tol = dag_tol
        self.ll_rel_tol = ll_rel_tol
        self.window = window

        self._snapshots: List[_SnapshotEntry] = []
        self._hamming_trace: List[float] = []
        self._dag_dist_trace: List[float] = []
        self._ll_trace: List[float] = []
        self._lyapunov_values: List[float] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        iteration: int,
        regime_assignments: np.ndarray,
        causal_graphs: Dict[int, nx.DiGraph],
        log_likelihood: float,
    ) -> None:
        """Record a snapshot from one EM iteration."""
        entry = _SnapshotEntry(
            iteration=iteration,
            regime_assignments=regime_assignments.copy(),
            causal_graphs={k: g.copy() for k, g in causal_graphs.items()},
            log_likelihood=log_likelihood,
        )
        self._snapshots.append(entry)
        self._ll_trace.append(log_likelihood)

        if len(self._snapshots) >= 2:
            prev = self._snapshots[-2]
            h = hamming_distance(regime_assignments, prev.regime_assignments)
            d = total_dag_edit_distance(causal_graphs, prev.causal_graphs)
            self._hamming_trace.append(h)
            self._dag_dist_trace.append(d)
        else:
            self._hamming_trace.append(1.0)
            self._dag_dist_trace.append(float("inf"))

        # Lyapunov function: negative log-likelihood (should decrease)
        lyap = -log_likelihood
        self._lyapunov_values.append(lyap)

    def record_from_coupled(self, model: Any) -> None:
        """Convenience: record from a CoupledInference model after an iteration."""
        history = model.get_history()
        if not history:
            return
        last = history[-1]
        self.record(
            iteration=last.iteration,
            regime_assignments=model.get_regimes(),
            causal_graphs=model.get_causal_graphs(),
            log_likelihood=last.log_likelihood,
        )

    # ------------------------------------------------------------------
    # Convergence checking
    # ------------------------------------------------------------------

    def check_convergence(
        self,
        history: Optional[List[Any]] = None,
    ) -> bool:
        """Check whether the EM loop has converged.

        Uses the internal trace (from ``record()`` calls) unless `history`
        is provided explicitly (list of IterationRecord from CoupledInference).

        Returns True if ALL of:
          - normalised Hamming distance < hamming_tol
          - total DAG edit distance < dag_tol
          - relative LL change < ll_rel_tol
        """
        if history is not None:
            self._ingest_history(history)

        if len(self._ll_trace) < 3:
            return False

        h = self._hamming_trace[-1] if self._hamming_trace else 1.0
        d = self._dag_dist_trace[-1] if self._dag_dist_trace else float("inf")
        ll_prev = self._ll_trace[-2]
        ll_curr = self._ll_trace[-1]
        ll_rel = abs((ll_curr - ll_prev) / (abs(ll_prev) + 1e-10))

        converged = h < self.hamming_tol and d < self.dag_tol and ll_rel < self.ll_rel_tol
        return converged

    def _ingest_history(self, history: List[Any]) -> None:
        """Populate internal traces from IterationRecord list."""
        self._ll_trace = [r.log_likelihood for r in history]
        self._hamming_trace = [r.regime_hamming for r in history]
        self._dag_dist_trace = [r.dag_edit_distance for r in history]
        self._lyapunov_values = [-r.log_likelihood for r in history]

    # ------------------------------------------------------------------
    # Rate estimation
    # ------------------------------------------------------------------

    def estimate_rate(self) -> Optional[float]:
        """Estimate the linear convergence rate from the Hamming distance trace.

        Fits an exponential decay model  h(t) = a * rho^t  and returns rho.
        rho < 1 indicates linear convergence; values closer to 0 are faster.

        Returns None if insufficient data or the fit fails.
        """
        trace = np.array(self._hamming_trace)
        if len(trace) < 4:
            return None

        # Use the last `window` entries
        trace = trace[-self.window:]
        trace = np.maximum(trace, 1e-15)
        log_trace = np.log(trace)
        t = np.arange(len(log_trace))

        # Fit log(h) = log(a) + t * log(rho)
        try:
            slope, intercept, _, _, _ = stats.linregress(t, log_trace)
            rho = float(np.exp(slope))
        except Exception:
            return None

        if not np.isfinite(rho) or rho <= 0:
            return None
        return min(rho, 2.0)  # cap at 2 for sanity

    def estimate_rate_dag(self) -> Optional[float]:
        """Estimate convergence rate from DAG edit-distance trace."""
        trace = np.array(self._dag_dist_trace)
        if len(trace) < 4:
            return None
        trace = trace[-self.window:]
        trace = np.maximum(trace, 0.1)
        log_trace = np.log(trace)
        t = np.arange(len(log_trace))
        try:
            slope, _, _, _, _ = stats.linregress(t, log_trace)
            rho = float(np.exp(slope))
        except Exception:
            return None
        if not np.isfinite(rho) or rho <= 0:
            return None
        return min(rho, 2.0)

    # ------------------------------------------------------------------
    # Lyapunov analysis
    # ------------------------------------------------------------------

    def lyapunov_decreasing(self, strict: bool = False) -> bool:
        """Check if the Lyapunov function (negative LL) is monotonically decreasing.

        Parameters
        ----------
        strict : bool
            If True, require strict decrease at every step.
        """
        vals = np.array(self._lyapunov_values)
        if len(vals) < 2:
            return True

        diffs = np.diff(vals)
        if strict:
            return bool(np.all(diffs < 0))
        else:
            # Allow small increases (numerical noise)
            return bool(np.all(diffs < 1e-3 * np.abs(vals[:-1]).mean()))

    def lyapunov_function_values(self) -> np.ndarray:
        """Return the Lyapunov function trace."""
        return np.array(self._lyapunov_values)

    # ------------------------------------------------------------------
    # Full diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> ConvergenceDiagnostics:
        """Return a comprehensive diagnostics summary."""
        converged = self.check_convergence()
        rate = self.estimate_rate()
        lyap_dec = self.lyapunov_decreasing()

        n_iter = len(self._ll_trace)
        final_ll_change = 0.0
        if n_iter >= 2:
            final_ll_change = abs(
                (self._ll_trace[-1] - self._ll_trace[-2])
                / (abs(self._ll_trace[-2]) + 1e-10)
            )

        return ConvergenceDiagnostics(
            converged=converged,
            n_iterations=n_iter,
            final_ll_change=final_ll_change,
            final_hamming=self._hamming_trace[-1] if self._hamming_trace else 1.0,
            final_dag_distance=self._dag_dist_trace[-1] if self._dag_dist_trace else float("inf"),
            estimated_rate=rate,
            lyapunov_decreasing=lyap_dec,
            lyapunov_values=list(self._lyapunov_values),
            ll_trace=list(self._ll_trace),
            hamming_trace=list(self._hamming_trace),
            dag_distance_trace=list(self._dag_dist_trace),
        )

    # ------------------------------------------------------------------
    # Multi-restart selector
    # ------------------------------------------------------------------

    @staticmethod
    def select_best_restart(
        models: Sequence[Any],
    ) -> Tuple[Any, int]:
        """Select the best model among multiple restarts by final LL.

        Parameters
        ----------
        models : sequence of fitted CoupledInference models

        Returns
        -------
        best_model : CoupledInference
        best_index : int
        """
        best_idx = -1
        best_ll = -np.inf
        for i, m in enumerate(models):
            lls = m.log_likelihoods()
            if len(lls) == 0:
                continue
            if lls[-1] > best_ll:
                best_ll = lls[-1]
                best_idx = i
        if best_idx < 0:
            raise ValueError("No valid model found among restarts")
        return models[best_idx], best_idx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded snapshots and traces."""
        self._snapshots.clear()
        self._hamming_trace.clear()
        self._dag_dist_trace.clear()
        self._ll_trace.clear()
        self._lyapunov_values.clear()

    def n_recorded(self) -> int:
        """Number of recorded iterations."""
        return len(self._snapshots)

    def plot_traces(self) -> Dict[str, np.ndarray]:
        """Return trace arrays suitable for plotting.

        Returns a dict with keys 'log_likelihood', 'hamming', 'dag_distance',
        'lyapunov', each mapping to a 1-D numpy array.
        """
        return {
            "log_likelihood": np.array(self._ll_trace),
            "hamming": np.array(self._hamming_trace),
            "dag_distance": np.array(self._dag_dist_trace),
            "lyapunov": np.array(self._lyapunov_values),
        }

    def iterations_to_convergence(self, tol_hamming: Optional[float] = None) -> Optional[int]:
        """Return the first iteration at which Hamming distance dropped below threshold.

        Returns None if the threshold was never reached.
        """
        tol = tol_hamming or self.hamming_tol
        for i, h in enumerate(self._hamming_trace):
            if h < tol:
                return i
        return None

    def stationarity_test(self, last_n: int = 10) -> Tuple[float, bool]:
        """Run a simple stationarity test on the last `last_n` LL values.

        Uses the Augmented Dickey-Fuller-like approach: regress ΔLL on LL_{t-1}
        and test whether the coefficient is significantly negative (stationary).

        Returns
        -------
        t_stat : float
        is_stationary : bool  (True if t_stat < -2.86, the 5% critical value for ADF)
        """
        lls = np.array(self._ll_trace[-last_n:])
        if len(lls) < 5:
            return 0.0, False

        delta = np.diff(lls)
        lagged = lls[:-1]
        # Regress delta on lagged
        X = np.column_stack([np.ones(len(lagged)), lagged])
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, delta, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, False

        if len(residuals) == 0:
            sigma = np.std(delta - X @ beta)
        else:
            sigma = np.sqrt(residuals[0] / (len(delta) - 2)) if residuals[0] > 0 else 1e-10

        se_beta1 = sigma / (np.sqrt(np.sum((lagged - lagged.mean()) ** 2)) + 1e-10)
        t_stat = beta[1] / (se_beta1 + 1e-10)

        return float(t_stat), t_stat < -2.86


# ---------------------------------------------------------------------------
# Formal EM convergence analysis
# ---------------------------------------------------------------------------

class EMConvergenceTheorem:
    """Formal analysis of EM alternation convergence guarantees.

    The coupled regime-causal inference alternates:
      E-step: Fix causal structure G, update regime posteriors Z
      M-step: Fix regime assignments Z, run causal discovery -> G

    **Theorem (Monotone Convergence with Caveats):**
    Let L(Z, G) = log p(X | Z, G) + log p(Z | G) + log p(G) be the
    penalized complete-data log-likelihood. Under the following conditions:

    1. The HDP-HMM E-step computes the exact conditional posterior
       p(Z | X, G) (or a valid lower bound via variational inference)
    2. The PC algorithm M-step is order-independent (PC-stable variant)
    3. The DAG space is restricted to a finite set of candidate graphs

    Then:
    - L(Z^(t), G^(t)) is non-decreasing in t
    - The sequence converges to a fixed point (Z*, G*)
    - The fixed point satisfies local optimality: no single-edge
      change in G and no single-assignment change in Z improves L

    **Caveat:** This does NOT guarantee convergence to the global optimum.
    Multiple restarts with different initializations are required.
    The EM_alternation implementation uses n_restarts > 1 to mitigate
    local optima.

    This class provides empirical verification of these theoretical
    properties on actual runs.
    """

    @staticmethod
    def verify_monotonicity(ll_trace: list) -> dict:
        """Verify that the log-likelihood sequence is non-decreasing.

        Parameters
        ----------
        ll_trace : list of float
            Log-likelihood values from successive EM iterations.

        Returns
        -------
        dict with:
            monotone : bool
            n_violations : int
            max_violation : float (magnitude of worst decrease)
            violation_indices : list of int
        """
        ll = np.array(ll_trace)
        if len(ll) < 2:
            return {"monotone": True, "n_violations": 0,
                    "max_violation": 0.0, "violation_indices": []}

        diffs = np.diff(ll)
        violations = np.where(diffs < -1e-10)[0]
        max_viol = float(np.min(diffs)) if len(diffs) > 0 else 0.0

        return {
            "monotone": len(violations) == 0,
            "n_violations": len(violations),
            "max_violation": abs(min(max_viol, 0.0)),
            "violation_indices": violations.tolist(),
        }

    @staticmethod
    def verify_fixed_point(
        hamming_trace: list,
        dag_distance_trace: list,
        hamming_tol: float = 0.01,
        dag_tol: float = 1.0,
    ) -> dict:
        """Verify convergence to a fixed point.

        Parameters
        ----------
        hamming_trace : list of float
            Normalized Hamming distances between successive regime assignments.
        dag_distance_trace : list of float
            DAG edit distances between successive causal graphs.

        Returns
        -------
        dict with:
            reached_fixed_point : bool
            regime_converged_at : int or None
            dag_converged_at : int or None
            final_hamming : float
            final_dag_distance : float
        """
        h = np.array(hamming_trace)
        d = np.array(dag_distance_trace)

        regime_conv = np.where(h < hamming_tol)[0]
        dag_conv = np.where(d <= dag_tol)[0]

        regime_at = int(regime_conv[0]) if len(regime_conv) > 0 else None
        dag_at = int(dag_conv[0]) if len(dag_conv) > 0 else None

        reached = (regime_at is not None and dag_at is not None
                   and len(h) > 0 and h[-1] < hamming_tol
                   and len(d) > 0 and d[-1] <= dag_tol)

        return {
            "reached_fixed_point": reached,
            "regime_converged_at": regime_at,
            "dag_converged_at": dag_at,
            "final_hamming": float(h[-1]) if len(h) > 0 else float('inf'),
            "final_dag_distance": float(d[-1]) if len(d) > 0 else float('inf'),
        }

    @staticmethod
    def estimate_contraction_rate(ll_trace: list, window: int = 10) -> float:
        """Estimate the contraction rate of the EM iteration.

        Fits L^(t+1) - L* ≈ r * (L^(t) - L*) where L* is estimated
        as the final log-likelihood value.

        Parameters
        ----------
        ll_trace : list of float
        window : int
            Number of final iterations to use for rate estimation.

        Returns
        -------
        float
            Estimated contraction rate r in [0, 1].
            r < 1 indicates geometric convergence.
        """
        ll = np.array(ll_trace)
        if len(ll) < window + 2:
            return 1.0

        L_star = ll[-1]
        gaps = L_star - ll[-(window+1):-1]
        gaps_next = L_star - ll[-window:]

        # Avoid division by zero
        valid = np.abs(gaps) > 1e-15
        if not np.any(valid):
            return 0.0

        ratios = gaps_next[valid] / gaps[valid]
        rate = float(np.median(np.clip(ratios, 0, 2)))
        return min(max(rate, 0.0), 2.0)
