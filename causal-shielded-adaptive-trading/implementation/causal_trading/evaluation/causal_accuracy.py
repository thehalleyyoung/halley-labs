"""
Causal discovery accuracy evaluation.

Computes Structural Hamming Distance, edge-level precision/recall/F1,
adjacency and arrowhead accuracy, and per-edge confidence analysis
for comparing estimated causal DAGs against ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Edge = Tuple[int, int]  # (source, target)


@dataclass
class CausalAccuracyMetrics:
    """Container for causal discovery accuracy metrics."""
    # Structural Hamming Distance
    shd: int
    shd_missing: int
    shd_extra: int
    shd_reversed: int

    # Edge-level P/R/F1
    edge_precision: float
    edge_recall: float
    edge_f1: float

    # Adjacency (ignoring orientation)
    adjacency_precision: float
    adjacency_recall: float
    adjacency_f1: float

    # Arrowhead (orientation correctness for correct adjacencies)
    arrowhead_precision: float
    arrowhead_recall: float
    arrowhead_f1: float

    # Invariant vs regime-specific edge metrics
    invariant_edge_precision: float
    invariant_edge_recall: float
    invariant_edge_f1: float
    regime_specific_edge_precision: float
    regime_specific_edge_recall: float
    regime_specific_edge_f1: float

    # Counts
    n_true_edges: int
    n_estimated_edges: int
    n_correct_edges: int
    n_true_nodes: int

    # Per-edge data
    per_edge_confidence_accuracy: Optional[NDArray[np.float64]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Causal Accuracy ===",
            f"SHD:                  {self.shd:>6d}  (miss={self.shd_missing}  extra={self.shd_extra}  rev={self.shd_reversed})",
            f"Edge P/R/F1:          {self.edge_precision:.4f} / {self.edge_recall:.4f} / {self.edge_f1:.4f}",
            f"Adjacency P/R/F1:     {self.adjacency_precision:.4f} / {self.adjacency_recall:.4f} / {self.adjacency_f1:.4f}",
            f"Arrowhead P/R/F1:     {self.arrowhead_precision:.4f} / {self.arrowhead_recall:.4f} / {self.arrowhead_f1:.4f}",
            f"Invariant P/R/F1:     {self.invariant_edge_precision:.4f} / {self.invariant_edge_recall:.4f} / {self.invariant_edge_f1:.4f}",
            f"Regime-spec P/R/F1:   {self.regime_specific_edge_precision:.4f} / {self.regime_specific_edge_recall:.4f} / {self.regime_specific_edge_f1:.4f}",
            f"True edges:           {self.n_true_edges:>6d}",
            f"Estimated edges:      {self.n_estimated_edges:>6d}",
            f"Correct edges:        {self.n_correct_edges:>6d}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DAG helpers
# ---------------------------------------------------------------------------

def _adj_matrix_to_edge_set(adj: NDArray) -> Set[Edge]:
    """Convert adjacency matrix to set of directed edges."""
    edges: Set[Edge] = set()
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] != 0:
                edges.add((i, j))
    return edges


def _edge_set_to_skeleton(edges: Set[Edge]) -> Set[frozenset]:
    """Convert directed edge set to undirected skeleton."""
    return {frozenset((u, v)) for u, v in edges}


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall, F1 from counts."""
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return float(prec), float(rec), float(f1)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CausalAccuracyEvaluator:
    """Evaluates causal DAG recovery accuracy.

    Compares an estimated adjacency matrix (or edge set) against the
    true DAG using SHD, precision/recall/F1 at edge and adjacency
    level, arrowhead accuracy, and separate metrics for invariant
    versus regime-specific edges.

    Parameters
    ----------
    n_nodes : number of variables (inferred from adjacency if not given)
    """

    def __init__(self, n_nodes: Optional[int] = None) -> None:
        self._n_nodes = n_nodes
        self._metrics: Optional[CausalAccuracyMetrics] = None

    # ---- public API --------------------------------------------------------

    def evaluate(
        self,
        true_dag: NDArray,
        estimated_dag: NDArray,
        invariant_edges: Optional[Set[Edge]] = None,
        regime_specific_edges: Optional[Set[Edge]] = None,
        edge_confidences: Optional[Dict[Edge, float]] = None,
    ) -> CausalAccuracyMetrics:
        """Compare estimated DAG to ground truth.

        Parameters
        ----------
        true_dag : (n, n) binary adjacency matrix of the true DAG
        estimated_dag : (n, n) adjacency matrix (binary or weighted)
        invariant_edges : set of edges that should be present in all regimes
        regime_specific_edges : set of edges specific to current regime
        edge_confidences : mapping from estimated edge to confidence score

        Returns
        -------
        CausalAccuracyMetrics
        """
        true_dag = np.asarray(true_dag)
        estimated_dag = np.asarray(estimated_dag)
        n = true_dag.shape[0]
        if estimated_dag.shape[0] != n:
            raise ValueError("DAG dimension mismatch.")

        # Binarise estimated DAG
        est_bin = (np.abs(estimated_dag) > 1e-10).astype(np.int64)
        true_bin = (np.abs(true_dag) > 1e-10).astype(np.int64)

        true_edges = _adj_matrix_to_edge_set(true_bin)
        est_edges = _adj_matrix_to_edge_set(est_bin)

        # SHD
        shd_miss, shd_extra, shd_rev = self._compute_shd_components(
            true_edges, est_edges,
        )
        shd = shd_miss + shd_extra + shd_rev

        # Edge-level P/R/F1 (directed)
        tp_edge = len(true_edges & est_edges)
        fp_edge = len(est_edges - true_edges)
        fn_edge = len(true_edges - est_edges)
        ep, er, ef = _prf(tp_edge, fp_edge, fn_edge)

        # Adjacency P/R/F1 (skeleton)
        true_skel = _edge_set_to_skeleton(true_edges)
        est_skel = _edge_set_to_skeleton(est_edges)
        tp_adj = len(true_skel & est_skel)
        fp_adj = len(est_skel - true_skel)
        fn_adj = len(true_skel - est_skel)
        ap, ar, af = _prf(tp_adj, fp_adj, fn_adj)

        # Arrowhead P/R/F1 (orientation among correct adjacencies)
        ah_tp, ah_fp, ah_fn = self._arrowhead_counts(true_edges, est_edges)
        ahp, ahr, ahf = _prf(ah_tp, ah_fp, ah_fn)

        # Invariant vs regime-specific
        inv_p, inv_r, inv_f = 0.0, 0.0, 0.0
        rs_p, rs_r, rs_f = 0.0, 0.0, 0.0
        if invariant_edges is not None:
            inv_p, inv_r, inv_f = self._subset_prf(invariant_edges, est_edges)
        if regime_specific_edges is not None:
            rs_p, rs_r, rs_f = self._subset_prf(regime_specific_edges, est_edges)

        # Per-edge confidence vs accuracy
        conf_acc = None
        if edge_confidences:
            conf_acc = self._confidence_accuracy_curve(
                edge_confidences, true_edges,
            )

        self._metrics = CausalAccuracyMetrics(
            shd=shd,
            shd_missing=shd_miss,
            shd_extra=shd_extra,
            shd_reversed=shd_rev,
            edge_precision=ep,
            edge_recall=er,
            edge_f1=ef,
            adjacency_precision=ap,
            adjacency_recall=ar,
            adjacency_f1=af,
            arrowhead_precision=ahp,
            arrowhead_recall=ahr,
            arrowhead_f1=ahf,
            invariant_edge_precision=inv_p,
            invariant_edge_recall=inv_r,
            invariant_edge_f1=inv_f,
            regime_specific_edge_precision=rs_p,
            regime_specific_edge_recall=rs_r,
            regime_specific_edge_f1=rs_f,
            n_true_edges=len(true_edges),
            n_estimated_edges=len(est_edges),
            n_correct_edges=tp_edge,
            n_true_nodes=n,
            per_edge_confidence_accuracy=conf_acc,
        )
        return self._metrics

    def get_metrics(self) -> CausalAccuracyMetrics:
        if self._metrics is None:
            raise RuntimeError("Call evaluate() first.")
        return self._metrics

    # ---- SHD ---------------------------------------------------------------

    @staticmethod
    def _compute_shd_components(
        true_edges: Set[Edge],
        est_edges: Set[Edge],
    ) -> Tuple[int, int, int]:
        """Decompose SHD into missing, extra, and reversed edges.

        A reversed edge (u→v estimated as v→u) counts as 1 SHD error
        rather than 2 (one missing + one extra).

        Returns
        -------
        (n_missing, n_extra, n_reversed)
        """
        true_skel = _edge_set_to_skeleton(true_edges)
        est_skel = _edge_set_to_skeleton(est_edges)

        missing_skel = true_skel - est_skel
        extra_skel = est_skel - true_skel
        common_skel = true_skel & est_skel

        n_reversed = 0
        for pair in common_skel:
            u, v = tuple(pair)
            # Check orientation mismatch
            true_dir = (u, v) in true_edges or (v, u) in true_edges
            if true_dir:
                if (u, v) in true_edges and (u, v) not in est_edges:
                    n_reversed += 1
                elif (v, u) in true_edges and (v, u) not in est_edges:
                    n_reversed += 1

        n_missing = len(missing_skel)
        n_extra = len(extra_skel)
        return n_missing, n_extra, n_reversed

    # ---- Arrowhead ---------------------------------------------------------

    @staticmethod
    def _arrowhead_counts(
        true_edges: Set[Edge],
        est_edges: Set[Edge],
    ) -> Tuple[int, int, int]:
        """Compute arrowhead TP, FP, FN.

        For each pair (i,j) that appears in both skeletons, check if the
        orientation (direction) matches.

        Returns (tp, fp, fn)
        """
        true_skel = _edge_set_to_skeleton(true_edges)
        est_skel = _edge_set_to_skeleton(est_edges)
        common = true_skel & est_skel

        tp = 0
        fp = 0
        fn = 0
        for pair in common:
            nodes = tuple(pair)
            u, v = nodes[0], nodes[1] if len(nodes) == 2 else nodes[0]
            if len(nodes) < 2:
                continue
            u, v = nodes

            true_uv = (u, v) in true_edges
            true_vu = (v, u) in true_edges
            est_uv = (u, v) in est_edges
            est_vu = (v, u) in est_edges

            # Arrowhead at v for edge u→v
            if true_uv and est_uv:
                tp += 1
            elif true_uv and not est_uv:
                fn += 1

            if true_vu and est_vu:
                tp += 1
            elif true_vu and not est_vu:
                fn += 1

            if est_uv and not true_uv:
                fp += 1
            if est_vu and not true_vu:
                fp += 1

        return tp, fp, fn

    # ---- Subset P/R/F1 (invariant / regime-specific) -----------------------

    @staticmethod
    def _subset_prf(
        target_edges: Set[Edge],
        est_edges: Set[Edge],
    ) -> Tuple[float, float, float]:
        """P/R/F1 restricted to a subset of target edges."""
        if not target_edges:
            return 0.0, 0.0, 0.0
        tp = len(target_edges & est_edges)
        fn = len(target_edges - est_edges)
        # FP: estimated edges that are in the target set's complement
        fp = 0  # we cannot know FP without the complement context
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return float(prec), float(rec), float(f1)

    # ---- Confidence vs accuracy curve --------------------------------------

    @staticmethod
    def _confidence_accuracy_curve(
        edge_confidences: Dict[Edge, float],
        true_edges: Set[Edge],
        n_bins: int = 20,
    ) -> NDArray[np.float64]:
        """Compute calibration curve: for each confidence bin, what
        fraction of edges are actually correct?

        Returns
        -------
        (n_bins, 3) array with columns [bin_center, accuracy, count].
        """
        if not edge_confidences:
            return np.zeros((0, 3), dtype=np.float64)

        confs = np.array(list(edge_confidences.values()), dtype=np.float64)
        correct = np.array(
            [1.0 if e in true_edges else 0.0 for e in edge_confidences],
            dtype=np.float64,
        )

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        result = np.zeros((n_bins, 3), dtype=np.float64)
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (confs >= lo) & (confs < hi) if b < n_bins - 1 else (confs >= lo) & (confs <= hi)
            result[b, 0] = (lo + hi) / 2.0
            if np.any(mask):
                result[b, 1] = float(np.mean(correct[mask]))
                result[b, 2] = float(np.sum(mask))
        return result

    # ---- DAG comparison utilities ------------------------------------------

    @staticmethod
    def dag_to_cpdag(adj: NDArray) -> NDArray:
        """Convert a DAG adjacency matrix to its CPDAG (completed PDAG).

        Edges that are part of a v-structure or can be oriented by
        Meek's rules are directed; all others become undirected (both
        directions present).

        Parameters
        ----------
        adj : (n, n) binary adjacency matrix

        Returns
        -------
        (n, n) adjacency matrix of the CPDAG
        """
        n = adj.shape[0]
        cpdag = np.zeros_like(adj)
        adj_bin = (np.abs(adj) > 1e-10).astype(int)

        # Identify v-structures: i → j ← k where i and k are not adjacent
        compelled: Set[Edge] = set()
        for j in range(n):
            parents = [i for i in range(n) if adj_bin[i, j] and not adj_bin[j, i]]
            for pi in range(len(parents)):
                for pk in range(pi + 1, len(parents)):
                    i, k = parents[pi], parents[pk]
                    if not adj_bin[i, k] and not adj_bin[k, i]:
                        compelled.add((i, j))
                        compelled.add((k, j))

        # Propagate via Meek's rules (simplified)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(n):
                    if adj_bin[i, j] and (i, j) not in compelled and (j, i) not in compelled:
                        # Rule 1: i → j – k (chain) and i ≠ k not adjacent
                        for k in range(n):
                            if k != i and (j, k) in compelled and not adj_bin[i, k] and not adj_bin[k, i]:
                                compelled.add((i, j))
                                changed = True
                                break

        for i in range(n):
            for j in range(n):
                if adj_bin[i, j]:
                    if (i, j) in compelled:
                        cpdag[i, j] = 1
                    else:
                        cpdag[i, j] = 1
                        cpdag[j, i] = 1
        return cpdag

    @staticmethod
    def random_dag(n_nodes: int, edge_prob: float = 0.3, seed: int = 42) -> NDArray:
        """Generate a random DAG via topological ordering.

        Parameters
        ----------
        n_nodes : number of nodes
        edge_prob : probability of edge between any ordered pair
        seed : random seed

        Returns
        -------
        (n, n) binary adjacency matrix
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_nodes)
        adj = np.zeros((n_nodes, n_nodes), dtype=np.int64)
        for idx_i in range(n_nodes):
            for idx_j in range(idx_i + 1, n_nodes):
                if rng.random() < edge_prob:
                    adj[order[idx_i], order[idx_j]] = 1
        return adj
