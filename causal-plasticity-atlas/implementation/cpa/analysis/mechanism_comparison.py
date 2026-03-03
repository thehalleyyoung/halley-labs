"""Cross-context mechanism comparison.

Compares causal mechanisms across contexts by quantifying structural
and parametric distances at each node, producing plasticity scores
and statistical significance tests.

Provides:

* :class:`MechanismComparator` – compare mechanisms node-by-node
  across SCMs using Frobenius/KL/MMD distance, the Chow test for
  structural breaks, and likelihood-ratio tests.
* :class:`MechanismComparisonResult` – container for results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from cpa.core.scm import StructuralCausalModel


# ===================================================================
# Dataclass
# ===================================================================


@dataclass
class MechanismComparisonResult:
    """Result of a cross-context mechanism comparison.

    Attributes
    ----------
    nodes : list of int
        Node indices that were compared.
    structural_distances : NDArray
        Pairwise structural distance matrix, shape (n_contexts, n_contexts)
        averaged over nodes, or (n_nodes,) per node for single-pair.
    parametric_distances : NDArray
        Pairwise parametric distance matrix.
    plasticity_scores : NDArray
        Per-node plasticity scores (higher = more plastic).
    p_values : NDArray
        Per-node p-values from statistical tests.
    node_details : dict
        Per-node detailed comparison results.
    """

    nodes: List[int]
    structural_distances: NDArray
    parametric_distances: NDArray
    plasticity_scores: NDArray
    p_values: NDArray
    node_details: Dict[int, Dict[str, Any]] = field(default_factory=dict)


# ===================================================================
# MechanismComparator
# ===================================================================


class MechanismComparator:
    """Compare causal mechanisms across contexts.

    Parameters
    ----------
    distance_metric : str
        Distance metric for parametric comparison.
        One of ``"frobenius"``, ``"kl"``, ``"mmd"``, ``"jsd"``.
    significance_level : float
        Significance level for hypothesis tests.
    n_samples_dist : int
        Number of samples for distributional distance estimation.
    """

    def __init__(
        self,
        distance_metric: str = "frobenius",
        significance_level: float = 0.05,
        n_samples_dist: int = 2000,
    ) -> None:
        valid = ("frobenius", "kl", "mmd", "jsd")
        if distance_metric not in valid:
            raise ValueError(
                f"distance_metric must be one of {valid}, got {distance_metric!r}"
            )
        self._metric = distance_metric
        self._alpha = significance_level
        self._n_samples = n_samples_dist

    # -----------------------------------------------------------------
    # Pairwise comparison (single node, two SCMs)
    # -----------------------------------------------------------------

    def compare_pair(
        self,
        scm1: StructuralCausalModel,
        scm2: StructuralCausalModel,
        node: int,
    ) -> Dict[str, float]:
        """Compare a single mechanism between two SCMs.

        Parameters
        ----------
        scm1, scm2 : StructuralCausalModel
        node : int
            Variable index whose mechanism is compared.

        Returns
        -------
        dict
            Keys: ``"structural_distance"``, ``"parametric_distance"``,
            ``"distributional_distance"``, ``"chow_pvalue"``,
            ``"lr_pvalue"``, ``"plasticity_score"``.
        """
        sd = self._structural_distance(scm1, scm2, node)
        pd = self._parameter_distance(scm1, scm2, node)
        dd = self._distributional_distance(scm1, scm2, node, self._n_samples)

        parents1 = scm1.parents(node)
        parents2 = scm2.parents(node)
        all_parents = sorted(set(parents1) | set(parents2))

        data1 = scm1.sample(self._n_samples)
        data2 = scm2.sample(self._n_samples)

        chow_p = self._chow_test(data1, data2, node, all_parents)
        lr_p = self._likelihood_ratio_test(data1, data2, node, all_parents)

        plasticity = sd + pd + dd

        return {
            "structural_distance": sd,
            "parametric_distance": pd,
            "distributional_distance": dd,
            "chow_pvalue": chow_p,
            "lr_pvalue": lr_p,
            "plasticity_score": plasticity,
        }

    # -----------------------------------------------------------------
    # Full comparison across contexts
    # -----------------------------------------------------------------

    def compare_all(
        self,
        scms: List[StructuralCausalModel],
        nodes: Optional[List[int]] = None,
    ) -> MechanismComparisonResult:
        """Compare mechanisms across all context pairs and nodes.

        Parameters
        ----------
        scms : list of StructuralCausalModel
        nodes : list of int or None
            Node indices to compare.  ``None`` compares all nodes.

        Returns
        -------
        MechanismComparisonResult
        """
        K = len(scms)
        if K < 2:
            raise ValueError("Need at least 2 SCMs to compare")

        if nodes is None:
            nodes = list(range(scms[0].num_variables))

        n_nodes = len(nodes)
        struct_dists = np.zeros((K, K), dtype=np.float64)
        param_dists = np.zeros((K, K), dtype=np.float64)
        plasticity_scores = np.zeros(n_nodes, dtype=np.float64)
        p_values = np.ones(n_nodes, dtype=np.float64)
        node_details: Dict[int, Dict[str, Any]] = {}

        for ni, node in enumerate(nodes):
            node_struct = np.zeros((K, K), dtype=np.float64)
            node_param = np.zeros((K, K), dtype=np.float64)
            pvals_node: List[float] = []

            for i in range(K):
                for j in range(i + 1, K):
                    result = self.compare_pair(scms[i], scms[j], node)
                    sd = result["structural_distance"]
                    pd_val = result["parametric_distance"]
                    node_struct[i, j] = node_struct[j, i] = sd
                    node_param[i, j] = node_param[j, i] = pd_val
                    struct_dists[i, j] += sd / n_nodes
                    struct_dists[j, i] += sd / n_nodes
                    param_dists[i, j] += pd_val / n_nodes
                    param_dists[j, i] += pd_val / n_nodes
                    pvals_node.append(min(result["chow_pvalue"], result["lr_pvalue"]))

            plasticity_scores[ni] = np.mean(node_struct) + np.mean(node_param)
            if pvals_node:
                combined = _combine_pvalues(pvals_node)
                p_values[ni] = combined

            node_details[node] = {
                "structural_distance_matrix": node_struct,
                "parametric_distance_matrix": node_param,
                "mean_structural": float(np.mean(node_struct[np.triu_indices(K, k=1)])),
                "mean_parametric": float(np.mean(node_param[np.triu_indices(K, k=1)])),
            }

        return MechanismComparisonResult(
            nodes=nodes,
            structural_distances=struct_dists,
            parametric_distances=param_dists,
            plasticity_scores=plasticity_scores,
            p_values=p_values,
            node_details=node_details,
        )

    def significance_matrix(
        self,
        scms: List[StructuralCausalModel],
        nodes: Optional[List[int]] = None,
    ) -> NDArray:
        """Matrix of pairwise p-values averaged over nodes.

        Parameters
        ----------
        scms : list of StructuralCausalModel
        nodes : list of int or None

        Returns
        -------
        NDArray, shape (K, K)
        """
        K = len(scms)
        if nodes is None:
            nodes = list(range(scms[0].num_variables))

        pmat = np.ones((K, K), dtype=np.float64)
        for i in range(K):
            for j in range(i + 1, K):
                pvals: List[float] = []
                for node in nodes:
                    res = self.compare_pair(scms[i], scms[j], node)
                    pvals.append(min(res["chow_pvalue"], res["lr_pvalue"]))
                combined = _combine_pvalues(pvals)
                pmat[i, j] = pmat[j, i] = combined
        return pmat

    # -----------------------------------------------------------------
    # Distance metrics
    # -----------------------------------------------------------------

    def structural_distance(
        self,
        parents1: NDArray,
        parents2: NDArray,
    ) -> float:
        """Compute structural distance between two parent indicator vectors."""
        p1 = np.asarray(parents1, dtype=np.float64).ravel()
        p2 = np.asarray(parents2, dtype=np.float64).ravel()
        max_len = max(len(p1), len(p2))
        v1 = np.zeros(max_len)
        v2 = np.zeros(max_len)
        v1[: len(p1)] = p1
        v2[: len(p2)] = p2
        return float(np.sum(np.abs(v1 - v2)))

    def parametric_distance(
        self,
        params1: NDArray,
        params2: NDArray,
    ) -> float:
        """Compute parametric distance between two parameter vectors."""
        p1 = np.asarray(params1, dtype=np.float64).ravel()
        p2 = np.asarray(params2, dtype=np.float64).ravel()
        max_len = max(len(p1), len(p2))
        v1 = np.zeros(max_len)
        v2 = np.zeros(max_len)
        v1[: len(p1)] = p1
        v2[: len(p2)] = p2

        if self._metric == "frobenius":
            return float(np.sqrt(np.sum((v1 - v2) ** 2)))
        elif self._metric in ("kl", "jsd"):
            return _jsd_1d(v1, v2)
        elif self._metric == "mmd":
            return _mmd_linear(v1.reshape(-1, 1), v2.reshape(-1, 1))
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))

    # -----------------------------------------------------------------
    # Internal comparison helpers
    # -----------------------------------------------------------------

    def _structural_distance(
        self,
        scm1: StructuralCausalModel,
        scm2: StructuralCausalModel,
        node: int,
    ) -> float:
        """Distance between parent sets of *node* in two SCMs."""
        p = max(scm1.num_variables, scm2.num_variables)
        ind1 = np.zeros(p, dtype=np.float64)
        ind2 = np.zeros(p, dtype=np.float64)
        for pa in scm1.parents(node):
            ind1[pa] = 1.0
        for pa in scm2.parents(node):
            ind2[pa] = 1.0
        return float(np.sum(np.abs(ind1 - ind2)))

    def _parameter_distance(
        self,
        scm1: StructuralCausalModel,
        scm2: StructuralCausalModel,
        node: int,
    ) -> float:
        """Distance between regression coefficients for *node*."""
        coefs1 = scm1.regression_coefficients[:, node]
        coefs2 = scm2.regression_coefficients[:, node]
        return self.parametric_distance(coefs1, coefs2)

    def _distributional_distance(
        self,
        scm1: StructuralCausalModel,
        scm2: StructuralCausalModel,
        node: int,
        n_samples: int,
    ) -> float:
        """Distributional distance for *node* via sampling."""
        rng = np.random.default_rng(0)
        data1 = scm1.sample(n_samples, rng=rng)[:, node]
        data2 = scm2.sample(n_samples, rng=rng)[:, node]

        if self._metric in ("jsd", "kl"):
            return _jsd_samples(data1, data2)
        elif self._metric == "mmd":
            return _mmd_linear(data1.reshape(-1, 1), data2.reshape(-1, 1))
        return _jsd_samples(data1, data2)

    @staticmethod
    def _chow_test(
        data1: NDArray,
        data2: NDArray,
        node: int,
        parents: List[int],
    ) -> float:
        """Chow test for structural break between two datasets.

        Tests whether the linear regression of *node* on *parents*
        has the same coefficients in both datasets.

        Returns
        -------
        float
            p-value (small → different mechanisms).
        """
        if len(parents) == 0:
            stat, pval = sp_stats.ttest_ind(data1[:, node], data2[:, node])
            return float(pval)

        n1, n2 = len(data1), len(data2)
        X1 = np.column_stack([np.ones(n1), data1[:, parents]])
        X2 = np.column_stack([np.ones(n2), data2[:, parents]])
        y1 = data1[:, node]
        y2 = data2[:, node]

        Xp = np.vstack([X1, X2])
        yp = np.concatenate([y1, y2])

        k = X1.shape[1]

        def _rss(X: NDArray, y: NDArray) -> float:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            return float(np.dot(resid, resid))

        rss_p = _rss(Xp, yp)
        rss_1 = _rss(X1, y1)
        rss_2 = _rss(X2, y2)
        rss_u = rss_1 + rss_2

        df_num = k
        df_den = n1 + n2 - 2 * k

        if df_den <= 0 or rss_u < 1e-15:
            return 1.0

        F_stat = ((rss_p - rss_u) / df_num) / (rss_u / df_den)
        F_stat = max(F_stat, 0.0)
        pval = float(1.0 - sp_stats.f.cdf(F_stat, df_num, df_den))
        return pval

    @staticmethod
    def _likelihood_ratio_test(
        data1: NDArray,
        data2: NDArray,
        node: int,
        parents: List[int],
    ) -> float:
        """Likelihood-ratio test for parameter equality.

        Compares the log-likelihood of a pooled model vs separate
        models for each dataset.

        Returns
        -------
        float
            p-value.
        """
        if len(parents) == 0:
            stat, pval = sp_stats.ttest_ind(data1[:, node], data2[:, node])
            return float(pval)

        n1, n2 = len(data1), len(data2)
        n = n1 + n2

        X1 = np.column_stack([np.ones(n1), data1[:, parents]])
        X2 = np.column_stack([np.ones(n2), data2[:, parents]])
        y1 = data1[:, node]
        y2 = data2[:, node]
        k = X1.shape[1]

        def _log_lik(X: NDArray, y: NDArray) -> float:
            nn = len(y)
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            sigma2 = np.dot(resid, resid) / max(nn, 1)
            if sigma2 < 1e-15:
                sigma2 = 1e-15
            return -0.5 * nn * (np.log(2 * np.pi * sigma2) + 1.0)

        Xp = np.vstack([X1, X2])
        yp = np.concatenate([y1, y2])
        ll_pooled = _log_lik(Xp, yp)
        ll_sep = _log_lik(X1, y1) + _log_lik(X2, y2)

        lr_stat = 2.0 * (ll_sep - ll_pooled)
        lr_stat = max(lr_stat, 0.0)
        df = k
        pval = float(1.0 - sp_stats.chi2.cdf(lr_stat, df))
        return pval


# ===================================================================
# Utility functions
# ===================================================================


def _combine_pvalues(pvalues: List[float]) -> float:
    """Combine p-values using Fisher's method."""
    pvals = np.asarray(pvalues, dtype=np.float64)
    pvals = np.clip(pvals, 1e-300, 1.0)
    stat = -2.0 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    return float(1.0 - sp_stats.chi2.cdf(stat, df))


def _jsd_samples(x: NDArray, y: NDArray, n_bins: int = 50) -> float:
    """Jensen-Shannon divergence between two 1-D sample arrays."""
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if hi - lo < 1e-12:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(x, bins=bins, density=True)
    q, _ = np.histogram(y, bins=bins, density=True)

    p = p.astype(np.float64) + 1e-12
    q = q.astype(np.float64) + 1e-12
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _jsd_1d(p: NDArray, q: NDArray) -> float:
    """JSD between two non-negative vectors (treated as unnormalised)."""
    p = np.abs(p) + 1e-12
    q = np.abs(q) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * kl_pm + 0.5 * kl_qm


def _mmd_linear(X: NDArray, Y: NDArray) -> float:
    """Linear-time MMD estimate with RBF kernel."""
    n = min(len(X), len(Y))
    if n < 2:
        return 0.0
    X = X[:n]
    Y = Y[:n]

    sigma2 = float(np.median(np.sum((X - Y) ** 2, axis=1)))
    if sigma2 < 1e-12:
        sigma2 = 1.0

    def k(a: NDArray, b: NDArray) -> float:
        return float(np.exp(-np.sum((a - b) ** 2) / (2.0 * sigma2)))

    stat = 0.0
    pairs = n // 2
    for i in range(pairs):
        i1, i2 = 2 * i, 2 * i + 1
        stat += k(X[i1], X[i2]) + k(Y[i1], Y[i2]) - k(X[i1], Y[i2]) - k(X[i2], Y[i1])
    return float(max(stat / max(pairs, 1), 0.0))
