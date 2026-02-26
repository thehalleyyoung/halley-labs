"""
Causal graph stability monitoring for Causal-Shielded Adaptive Trading.

Tracks the stability of identified causal edges over time using rolling
HSIC statistics, detects edge breakage (previously invariant edges that
become variant), computes DAG edit distances, and generates structural
change alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class StructuralAlertSeverity(Enum):
    """Severity for causal structural alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class StructuralAlert:
    """Alert generated when causal graph structure changes."""
    timestamp: int
    severity: StructuralAlertSeverity
    message: str
    affected_edges: List[Tuple[str, str]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeStatistics:
    """Rolling statistics for a single causal edge."""
    source: str
    target: str
    hsic_values: List[float] = field(default_factory=list)
    p_values: List[float] = field(default_factory=list)
    is_invariant: bool = True
    last_updated: int = 0
    first_breakage_time: Optional[int] = None

    @property
    def edge_key(self) -> Tuple[str, str]:
        return (self.source, self.target)

    def mean_hsic(self, window: int = 50) -> float:
        if not self.hsic_values:
            return 0.0
        vals = self.hsic_values[-window:]
        return float(np.mean(vals))

    def hsic_trend(self, window: int = 50) -> float:
        """Compute linear trend of HSIC over the window (positive = strengthening)."""
        vals = self.hsic_values[-window:]
        if len(vals) < 5:
            return 0.0
        x = np.arange(len(vals), dtype=np.float64)
        slope, _, _, _, _ = sp_stats.linregress(x, vals)
        return float(slope)


def _rbf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix between x and y."""
    if sigma is None:
        dists = np.sqrt(
            np.sum(x ** 2, axis=1, keepdims=True)
            - 2.0 * x @ y.T
            + np.sum(y ** 2, axis=1, keepdims=True).T
        )
        sigma = float(np.median(dists[dists > 0])) + 1e-8
    sq_dist = (
        np.sum(x ** 2, axis=1, keepdims=True)
        - 2.0 * x @ y.T
        + np.sum(y ** 2, axis=1, keepdims=True).T
    )
    return np.exp(-sq_dist / (2.0 * sigma ** 2))


def compute_hsic(
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) between
    two sets of observations and an approximate p-value via permutation.

    Parameters
    ----------
    x : np.ndarray, shape (n, dx)
    y : np.ndarray, shape (n, dy)

    Returns
    -------
    hsic_value : float
    p_value : float
        Permutation-based p-value (100 permutations).
    """
    n = x.shape[0]
    if n < 5:
        return 0.0, 1.0

    x = x.reshape(n, -1)
    y = y.reshape(n, -1)

    K = _rbf_kernel(x, x, sigma_x)
    L = _rbf_kernel(y, y, sigma_y)

    H = np.eye(n) - np.ones((n, n)) / n
    HKH = H @ K @ H
    HLH = H @ L @ H

    hsic_val = float(np.trace(HKH @ HLH)) / (n - 1) ** 2

    # Permutation test
    n_perm = 100
    count_ge = 0
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        L_perm = L[np.ix_(perm, perm)]
        HLH_perm = H @ L_perm @ H
        hsic_perm = float(np.trace(HKH @ HLH_perm)) / (n - 1) ** 2
        if hsic_perm >= hsic_val:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perm + 1)
    return hsic_val, p_value


def dag_edit_distance(
    edges_a: Set[Tuple[str, str]],
    edges_b: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Compute edit distance between two DAGs represented as edge sets.

    Returns the number of edge additions and deletions, the symmetric
    difference, and the Jaccard distance.
    """
    added = edges_b - edges_a
    removed = edges_a - edges_b
    symmetric_diff = added | removed
    union = edges_a | edges_b
    jaccard = len(symmetric_diff) / max(len(union), 1)

    return {
        "added": list(added),
        "removed": list(removed),
        "n_added": len(added),
        "n_removed": len(removed),
        "edit_distance": len(symmetric_diff),
        "jaccard_distance": jaccard,
    }


class CausalGraphMonitor:
    """
    Monitor stability of a causal graph over streaming data.

    Maintains rolling HSIC statistics for each edge, detects edge
    breakage (previously invariant edge becomes variant), and
    generates structural change alerts.

    Parameters
    ----------
    edges : list of (str, str)
        Initial list of causal edges (source, target).
    window_size : int
        Rolling window size for HSIC computation.
    invariance_threshold : float
        p-value threshold below which an edge is considered significant
        (i.e., dependence exists).
    breakage_threshold : float
        p-value above which a previously invariant edge is declared broken.
    min_samples : int
        Minimum samples before running stability checks.
    """

    def __init__(
        self,
        edges: List[Tuple[str, str]],
        window_size: int = 100,
        invariance_threshold: float = 0.05,
        breakage_threshold: float = 0.3,
        min_samples: int = 30,
    ) -> None:
        self.window_size = window_size
        self.invariance_threshold = invariance_threshold
        self.breakage_threshold = breakage_threshold
        self.min_samples = min_samples

        self._edge_stats: Dict[Tuple[str, str], EdgeStatistics] = {}
        for src, tgt in edges:
            self._edge_stats[(src, tgt)] = EdgeStatistics(source=src, target=tgt)

        self._all_nodes: Set[str] = set()
        for src, tgt in edges:
            self._all_nodes.add(src)
            self._all_nodes.add(tgt)

        self._data_buffer: List[Dict[str, np.ndarray]] = []
        self._t: int = 0
        self._alerts: List[StructuralAlert] = []
        self._graph_snapshots: List[Tuple[int, Set[Tuple[str, str]]]] = []
        self._edit_distances: List[Tuple[int, float]] = []

    def update(self, data_batch: Dict[str, np.ndarray]) -> List[StructuralAlert]:
        """
        Update with a new batch of data and check all edges.

        Parameters
        ----------
        data_batch : dict
            Mapping from variable name to observation array. Each array
            should have shape (batch_size,) or (batch_size, d).

        Returns
        -------
        alerts : list of StructuralAlert
            Any new alerts generated.
        """
        self._data_buffer.append(data_batch)
        if len(self._data_buffer) > self.window_size:
            self._data_buffer = self._data_buffer[-self.window_size:]

        self._t += 1
        new_alerts: List[StructuralAlert] = []

        if len(self._data_buffer) < self.min_samples:
            return new_alerts

        # Concatenate windowed data
        windowed = self._concatenate_window()

        for edge_key, edge_stat in self._edge_stats.items():
            src, tgt = edge_key
            if src not in windowed or tgt not in windowed:
                continue

            x_data = windowed[src]
            y_data = windowed[tgt]

            n = min(len(x_data), len(y_data))
            if n < 10:
                continue

            hsic_val, p_val = compute_hsic(x_data[:n], y_data[:n])
            edge_stat.hsic_values.append(hsic_val)
            edge_stat.p_values.append(p_val)
            edge_stat.last_updated = self._t

            # Check for edge breakage
            if edge_stat.is_invariant and p_val > self.breakage_threshold:
                if len(edge_stat.p_values) >= 3:
                    recent_p = edge_stat.p_values[-3:]
                    if all(p > self.breakage_threshold for p in recent_p):
                        edge_stat.is_invariant = False
                        edge_stat.first_breakage_time = self._t
                        alert = StructuralAlert(
                            timestamp=self._t,
                            severity=StructuralAlertSeverity.CRITICAL,
                            message=f"Edge breakage detected: {src} -> {tgt}",
                            affected_edges=[edge_key],
                            details={
                                "hsic": hsic_val,
                                "p_value": p_val,
                                "recent_p_values": recent_p,
                            },
                        )
                        new_alerts.append(alert)
                        self._alerts.append(alert)

            # Check for edge strengthening (previously broken edge recovers)
            elif not edge_stat.is_invariant and p_val < self.invariance_threshold:
                if len(edge_stat.p_values) >= 3:
                    recent_p = edge_stat.p_values[-3:]
                    if all(p < self.invariance_threshold for p in recent_p):
                        edge_stat.is_invariant = True
                        alert = StructuralAlert(
                            timestamp=self._t,
                            severity=StructuralAlertSeverity.INFO,
                            message=f"Edge recovered: {src} -> {tgt}",
                            affected_edges=[edge_key],
                            details={"hsic": hsic_val, "p_value": p_val},
                        )
                        new_alerts.append(alert)
                        self._alerts.append(alert)

        # Snapshot current graph and compute edit distance
        current_edges = self._get_active_edges()
        self._graph_snapshots.append((self._t, current_edges))
        if len(self._graph_snapshots) >= 2:
            prev_edges = self._graph_snapshots[-2][1]
            dist_info = dag_edit_distance(prev_edges, current_edges)
            self._edit_distances.append((self._t, dist_info["jaccard_distance"]))

            if dist_info["edit_distance"] > 0:
                severity = (
                    StructuralAlertSeverity.WARNING
                    if dist_info["edit_distance"] <= 2
                    else StructuralAlertSeverity.CRITICAL
                )
                alert = StructuralAlert(
                    timestamp=self._t,
                    severity=severity,
                    message=f"Graph structure changed: {dist_info['edit_distance']} edge(s) differ",
                    affected_edges=list(dist_info["added"]) + list(dist_info["removed"]),
                    details=dist_info,
                )
                new_alerts.append(alert)
                self._alerts.append(alert)

        return new_alerts

    def check_stability(self) -> Dict[str, Any]:
        """
        Return overall stability assessment of the causal graph.

        Returns
        -------
        report : dict
            Includes fraction of invariant edges, mean HSIC, trend info,
            and unstable edge list.
        """
        total = len(self._edge_stats)
        if total == 0:
            return {"stable": True, "invariant_fraction": 1.0}

        invariant_count = sum(
            1 for e in self._edge_stats.values() if e.is_invariant
        )
        mean_hsic_vals = [
            e.mean_hsic() for e in self._edge_stats.values()
            if e.hsic_values
        ]
        trends = [
            e.hsic_trend() for e in self._edge_stats.values()
            if len(e.hsic_values) >= 5
        ]

        inv_frac = invariant_count / total
        return {
            "stable": inv_frac >= 0.8,
            "invariant_fraction": inv_frac,
            "invariant_count": invariant_count,
            "total_edges": total,
            "mean_hsic": float(np.mean(mean_hsic_vals)) if mean_hsic_vals else 0.0,
            "mean_hsic_trend": float(np.mean(trends)) if trends else 0.0,
            "unstable_edges": [
                e.edge_key for e in self._edge_stats.values() if not e.is_invariant
            ],
        }

    def get_unstable_edges(self) -> List[Tuple[str, str]]:
        """Return list of edges that are currently not invariant."""
        return [
            e.edge_key for e in self._edge_stats.values() if not e.is_invariant
        ]

    def get_edge_report(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """Return detailed statistics for a specific edge."""
        key = (source, target)
        edge = self._edge_stats.get(key)
        if edge is None:
            return None
        return {
            "source": edge.source,
            "target": edge.target,
            "is_invariant": edge.is_invariant,
            "mean_hsic": edge.mean_hsic(),
            "hsic_trend": edge.hsic_trend(),
            "n_observations": len(edge.hsic_values),
            "last_updated": edge.last_updated,
            "first_breakage_time": edge.first_breakage_time,
            "recent_p_values": edge.p_values[-5:] if edge.p_values else [],
        }

    def get_alerts(self, last_n: Optional[int] = None) -> List[StructuralAlert]:
        """Return structural alerts, optionally limited to the last n."""
        if last_n is not None:
            return self._alerts[-last_n:]
        return list(self._alerts)

    def get_edit_distance_series(self) -> List[Tuple[int, float]]:
        """Return the time series of DAG edit distances (Jaccard)."""
        return list(self._edit_distances)

    def add_edge(self, source: str, target: str) -> None:
        """Add a new edge to monitor."""
        key = (source, target)
        if key not in self._edge_stats:
            self._edge_stats[key] = EdgeStatistics(source=source, target=target)
            self._all_nodes.add(source)
            self._all_nodes.add(target)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from monitoring."""
        key = (source, target)
        self._edge_stats.pop(key, None)

    def windowed_graph_comparison(
        self,
        window1: int = 50,
        window2: int = 50,
    ) -> Dict[str, Any]:
        """
        Compare causal graph structure between two consecutive windows.

        Uses the graph snapshots to compare the edge sets in the first
        window (older) vs. the second window (newer).
        """
        if len(self._graph_snapshots) < 2:
            return {"comparable": False, "reason": "Insufficient snapshots"}

        n_snaps = len(self._graph_snapshots)
        mid = max(1, n_snaps - window2)

        # Aggregate edges in each window via majority vote
        recent_edges = self._aggregate_edges(
            self._graph_snapshots[mid:]
        )
        older_start = max(0, mid - window1)
        older_edges = self._aggregate_edges(
            self._graph_snapshots[older_start:mid]
        )

        dist = dag_edit_distance(older_edges, recent_edges)
        dist["comparable"] = True
        dist["older_window_size"] = mid - older_start
        dist["recent_window_size"] = n_snaps - mid
        return dist

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _concatenate_window(self) -> Dict[str, np.ndarray]:
        """Concatenate data from the buffer window into arrays per variable."""
        result: Dict[str, List[np.ndarray]] = {}
        for batch in self._data_buffer:
            for var_name, arr in batch.items():
                if var_name not in result:
                    result[var_name] = []
                result[var_name].append(np.atleast_2d(arr))

        return {
            k: np.vstack(v) for k, v in result.items()
        }

    def _get_active_edges(self) -> Set[Tuple[str, str]]:
        """Return set of currently invariant edges."""
        return {
            key for key, stat in self._edge_stats.items() if stat.is_invariant
        }

    def _aggregate_edges(
        self,
        snapshots: List[Tuple[int, Set[Tuple[str, str]]]],
    ) -> Set[Tuple[str, str]]:
        """Return edges present in more than half of the snapshots."""
        if not snapshots:
            return set()
        edge_counts: Dict[Tuple[str, str], int] = {}
        for _, edges in snapshots:
            for e in edges:
                edge_counts[e] = edge_counts.get(e, 0) + 1
        threshold = len(snapshots) / 2.0
        return {e for e, c in edge_counts.items() if c > threshold}
