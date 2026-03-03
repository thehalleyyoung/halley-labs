"""
Alignment scoring utilities for the Causal-Plasticity Atlas.

Provides the statistical scoring components used by the CADA alignment
algorithm (ALG1, Phases 2-3):

    - CIFingerprintScorer: compute and compare CI fingerprints (Phase 3, weight 0.6)
    - MarkovBlanketOverlap: Jaccard-based MB overlap scoring (Phase 2)
    - DistributionShapeSimilarity: KL-based shape comparison (Phase 3, weight 0.4)
    - ScoreMatrix: construct and normalize alignment score matrices
    - AnchorValidator: validate anchor consistency and conflicts

References:
    ALG1 Phase 2: Markov blanket candidate generation (Jaccard >= 0.3)
    ALG1 Phase 3: Statistical compatibility scoring
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_EPSILON = 1e-12
_MIN_VARIANCE = 1e-14
_REGULARIZATION = 1e-10


# ---------------------------------------------------------------------------
# Graceful imports from sibling modules
# ---------------------------------------------------------------------------
try:
    from cpa.core.types import MechanismParams, VariableID
except ImportError:

    @dataclass
    class MechanismParams:
        """Parameters of a linear-Gaussian conditional."""
        coeffs: NDArray[np.floating]
        intercept: float = 0.0
        noise_var: float = 1.0

    VariableID = Union[int, str]

from cpa.core.mechanism_distance import (
    _scm_adjacency,
    _scm_implied_covariance,
    _scm_markov_blanket,
    _scm_mechanism_params,
    _scm_n_vars,
    _scm_parents,
)


# ===================================================================
#  CIFingerprintScorer
# ===================================================================
class CIFingerprintScorer:
    """Score alignment candidates using CI-fingerprint similarity (Phase 3).

    A CI-fingerprint for variable X_i in context c is the vector of partial
    correlations rho(X_i, X_j | X_{-i,-j}) for all j. Two variables from
    different contexts with similar fingerprints likely represent the same
    causal mechanism.

    This scorer supports three comparison modes:
        1. Cosine similarity on aligned partial correlation vectors
        2. Rank correlation (Spearman) on partial correlation magnitudes
        3. Overlap of significant partial correlations (binary)

    Parameters
    ----------
    method : str
        Comparison method: 'cosine', 'spearman', or 'overlap'. Default 'cosine'.
    significance_threshold : float
        Threshold for considering a partial correlation significant. Default 0.05.
    regularization : float
        Ridge regularization for covariance inversion.
    """

    def __init__(
        self,
        method: str = "cosine",
        significance_threshold: float = 0.05,
        regularization: float = _REGULARIZATION,
    ) -> None:
        if method not in ("cosine", "spearman", "overlap"):
            raise ValueError(f"Unknown method {method!r}; use 'cosine', 'spearman', or 'overlap'")
        self.method = method
        self.significance_threshold = significance_threshold
        self.regularization = regularization
        self._precision_cache: Dict[str, NDArray] = {}

    def _compute_precision(
        self,
        cov: NDArray[np.floating],
        cache_key: Optional[str] = None,
    ) -> NDArray[np.floating]:
        """Compute precision matrix from covariance with regularization.

        Parameters
        ----------
        cov : NDArray, shape (n, n)
            Covariance matrix.
        cache_key : str or None
            Optional cache key.

        Returns
        -------
        NDArray, shape (n, n)
            Precision matrix.
        """
        if cache_key and cache_key in self._precision_cache:
            return self._precision_cache[cache_key]

        cov = np.asarray(cov, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)

        eigvals = la.eigvalsh(cov)
        if eigvals[0] < self.regularization:
            cov = cov + (self.regularization - min(eigvals[0], 0.0) + self.regularization) * np.eye(
                cov.shape[0]
            )

        try:
            precision = la.inv(cov)
        except la.LinAlgError:
            cov += self.regularization * 100 * np.eye(cov.shape[0])
            precision = la.inv(cov)

        precision = 0.5 * (precision + precision.T)

        if cache_key:
            self._precision_cache[cache_key] = precision

        return precision

    def partial_correlations(
        self,
        var_idx: int,
        cov: NDArray[np.floating],
        cache_key: Optional[str] = None,
    ) -> NDArray[np.floating]:
        """Compute partial correlations for variable *var_idx*.

        Uses the precision matrix:
            rho(X_i, X_j | X_{-i,-j}) = -Theta[i,j] / sqrt(Theta[i,i]*Theta[j,j])

        Parameters
        ----------
        var_idx : int
            Variable index.
        cov : NDArray, shape (n, n)
            Covariance matrix.
        cache_key : str or None
            Cache key for precision matrix.

        Returns
        -------
        NDArray, shape (n,)
            Partial correlations (0 at var_idx).
        """
        n = cov.shape[0]
        precision = self._compute_precision(cov, cache_key)

        pcorrs = np.zeros(n, dtype=np.float64)
        diag_i = abs(precision[var_idx, var_idx])

        for j in range(n):
            if j == var_idx:
                continue
            diag_j = abs(precision[j, j])
            denom = np.sqrt(diag_i * diag_j)
            if denom < _EPSILON:
                pcorrs[j] = 0.0
            else:
                pcorrs[j] = -precision[var_idx, j] / denom

        return np.clip(pcorrs, -1.0, 1.0)

    def fingerprint_similarity(
        self,
        pcorrs_a: NDArray[np.floating],
        pcorrs_b: NDArray[np.floating],
        alignment: Optional[Dict[int, Optional[int]]] = None,
    ) -> float:
        """Compute similarity between two partial-correlation vectors.

        Parameters
        ----------
        pcorrs_a, pcorrs_b : NDArray
            Partial correlation vectors.
        alignment : dict or None
            Maps indices in a to indices in b.

        Returns
        -------
        float
            Similarity in [0, 1].
        """
        if alignment is not None:
            pairs_a = []
            pairs_b = []
            for i, j in alignment.items():
                if j is not None and i < len(pcorrs_a) and j < len(pcorrs_b):
                    pairs_a.append(pcorrs_a[i])
                    pairs_b.append(pcorrs_b[j])
            if not pairs_a:
                return 0.0
            va = np.array(pairs_a)
            vb = np.array(pairs_b)
        else:
            n = min(len(pcorrs_a), len(pcorrs_b))
            if n == 0:
                return 0.0
            va = pcorrs_a[:n]
            vb = pcorrs_b[:n]

        if self.method == "cosine":
            return self._cosine_similarity(va, vb)
        elif self.method == "spearman":
            return self._spearman_similarity(va, vb)
        else:  # overlap
            return self._overlap_similarity(va, vb)

    def _cosine_similarity(self, a: NDArray, b: NDArray) -> float:
        """Cosine similarity mapped to [0, 1]."""
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < _EPSILON and norm_b < _EPSILON:
            return 1.0
        if norm_a < _EPSILON or norm_b < _EPSILON:
            return 0.0
        cosine = float(a @ b) / (norm_a * norm_b)
        return float(np.clip(0.5 * (1.0 + cosine), 0.0, 1.0))

    def _spearman_similarity(self, a: NDArray, b: NDArray) -> float:
        """Spearman rank correlation mapped to [0, 1]."""
        if len(a) < 3:
            return self._cosine_similarity(a, b)
        rho, _ = sp_stats.spearmanr(a, b)
        if np.isnan(rho):
            return 0.5
        return float(np.clip(0.5 * (1.0 + rho), 0.0, 1.0))

    def _overlap_similarity(self, a: NDArray, b: NDArray) -> float:
        """Fraction of shared significant partial correlations."""
        sig_a = set(i for i, v in enumerate(a) if abs(v) > self.significance_threshold)
        sig_b = set(i for i, v in enumerate(b) if abs(v) > self.significance_threshold)
        if not sig_a and not sig_b:
            return 1.0
        if not sig_a or not sig_b:
            return 0.0
        intersection = len(sig_a & sig_b)
        union = len(sig_a | sig_b)
        return intersection / union if union > 0 else 0.0

    def score_pair(
        self,
        var_a: int,
        var_b: int,
        cov_a: NDArray[np.floating],
        cov_b: NDArray[np.floating],
        alignment: Optional[Dict[int, Optional[int]]] = None,
        ctx_a: str = "",
        ctx_b: str = "",
    ) -> float:
        """Score a candidate pair (var_a in ctx_a, var_b in ctx_b).

        Parameters
        ----------
        var_a, var_b : int
            Variable indices.
        cov_a, cov_b : NDArray
            Covariance matrices for each context.
        alignment : dict or None
            Variable alignment.
        ctx_a, ctx_b : str
            Context identifiers (for caching).

        Returns
        -------
        float
            Similarity score in [0, 1].
        """
        pcorrs_a = self.partial_correlations(var_a, cov_a, cache_key=f"{ctx_a}")
        pcorrs_b = self.partial_correlations(var_b, cov_b, cache_key=f"{ctx_b}")
        return self.fingerprint_similarity(pcorrs_a, pcorrs_b, alignment)

    def clear_cache(self) -> None:
        """Clear precision matrix cache."""
        self._precision_cache.clear()


# ===================================================================
#  MarkovBlanketOverlap
# ===================================================================
class MarkovBlanketOverlap:
    """Jaccard-based Markov blanket overlap scoring (Phase 2).

    For each unanchored variable, computes the overlap between its Markov
    blanket and the Markov blankets of candidate variables in the other
    context, using already-anchored variables as the reference frame.

    Parameters
    ----------
    overlap_threshold : float
        Minimum Jaccard overlap to consider a candidate. Default 0.3.
    """

    def __init__(self, overlap_threshold: float = 0.3) -> None:
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError(f"overlap_threshold must be in [0,1], got {overlap_threshold}")
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def markov_blanket_from_adjacency(
        adj: NDArray[np.floating],
        var_idx: int,
    ) -> Set[int]:
        """Compute the Markov blanket from an adjacency matrix.

        MB(X_i) = parents(X_i) ∪ children(X_i) ∪ co-parents of children(X_i)

        Parameters
        ----------
        adj : NDArray, shape (n, n)
            Adjacency matrix where adj[i,j] != 0 means i -> j.
        var_idx : int
            Variable index.

        Returns
        -------
        set of int
            Markov blanket variable indices (excluding var_idx).
        """
        n = adj.shape[0]
        parents = set(int(j) for j in range(n) if adj[j, var_idx] != 0)
        children = set(int(j) for j in range(n) if adj[var_idx, j] != 0)

        co_parents: Set[int] = set()
        for child in children:
            for j in range(n):
                if j != var_idx and adj[j, child] != 0:
                    co_parents.add(j)

        mb = (parents | children | co_parents) - {var_idx}
        return mb

    def jaccard_index(self, set_a: Set[int], set_b: Set[int]) -> float:
        """Compute the Jaccard index between two sets.

        J(A, B) = |A ∩ B| / |A ∪ B|

        Parameters
        ----------
        set_a, set_b : set of int
            Two sets.

        Returns
        -------
        float
            Jaccard index in [0, 1].
        """
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def anchored_jaccard(
        self,
        mb_a: Set[int],
        mb_b: Set[int],
        anchor_map_a_to_b: Dict[int, int],
    ) -> float:
        """Compute Jaccard overlap using only anchored variables.

        Translates mb_a indices through anchor_map before comparison.

        Parameters
        ----------
        mb_a : set of int
            Markov blanket in context A.
        mb_b : set of int
            Markov blanket in context B.
        anchor_map_a_to_b : dict
            Maps anchored variable indices from A to B.

        Returns
        -------
        float
            Jaccard index computed over anchored variables only.
        """
        # Translate mb_a through anchor map
        translated_a = set()
        for v in mb_a:
            if v in anchor_map_a_to_b:
                translated_a.add(anchor_map_a_to_b[v])

        # Restrict mb_b to anchored variables too
        anchored_b_vars = set(anchor_map_a_to_b.values())
        restricted_b = mb_b & anchored_b_vars

        return self.jaccard_index(translated_a, restricted_b)

    def generate_candidates(
        self,
        adj_a: NDArray[np.floating],
        adj_b: NDArray[np.floating],
        unanchored_a: List[int],
        unanchored_b: List[int],
        anchor_map: Dict[int, int],
    ) -> List[Tuple[int, int, float]]:
        """Generate candidate pairs based on Markov blanket overlap.

        For each unanchored variable in context A, computes MB overlap
        with each unanchored variable in context B, using anchored variables
        as the reference frame.

        Parameters
        ----------
        adj_a, adj_b : NDArray
            Adjacency matrices for contexts A and B.
        unanchored_a, unanchored_b : list of int
            Unanchored variable indices.
        anchor_map : dict
            Maps variable indices in A to B.

        Returns
        -------
        list of (var_a, var_b, overlap_score) tuples
            Sorted by decreasing overlap. Only includes pairs with
            overlap >= self.overlap_threshold.
        """
        candidates = []

        for va in unanchored_a:
            mb_a = self.markov_blanket_from_adjacency(adj_a, va)

            for vb in unanchored_b:
                mb_b = self.markov_blanket_from_adjacency(adj_b, vb)

                overlap = self.anchored_jaccard(mb_a, mb_b, anchor_map)

                if overlap >= self.overlap_threshold:
                    candidates.append((va, vb, overlap))

        candidates.sort(key=lambda x: -x[2])
        return candidates

    def mb_overlap_matrix(
        self,
        adj_a: NDArray[np.floating],
        adj_b: NDArray[np.floating],
        vars_a: List[int],
        vars_b: List[int],
        anchor_map: Dict[int, int],
    ) -> NDArray[np.floating]:
        """Compute full MB overlap matrix.

        Parameters
        ----------
        adj_a, adj_b : NDArray
            Adjacency matrices.
        vars_a, vars_b : list of int
            Variable index lists.
        anchor_map : dict
            Anchor mapping from A to B.

        Returns
        -------
        NDArray, shape (len(vars_a), len(vars_b))
            Overlap scores.
        """
        n_a = len(vars_a)
        n_b = len(vars_b)
        mat = np.zeros((n_a, n_b), dtype=np.float64)

        for i, va in enumerate(vars_a):
            mb_a = self.markov_blanket_from_adjacency(adj_a, va)
            for j, vb in enumerate(vars_b):
                mb_b = self.markov_blanket_from_adjacency(adj_b, vb)
                mat[i, j] = self.anchored_jaccard(mb_a, mb_b, anchor_map)

        return mat


# ===================================================================
#  DistributionShapeSimilarity
# ===================================================================
class DistributionShapeSimilarity:
    """KL-based shape comparison for alignment scoring (Phase 3, weight 0.4).

    Compares the *shape* of local conditional distributions, focusing on
    variance ratios and coefficient patterns rather than exact parameter values.

    Parameters
    ----------
    method : str
        'kl_symmetric' (default): symmetric KL divergence.
        'jsd': Jensen-Shannon divergence.
        'variance_ratio': simple variance ratio comparison.
    regularization : float
        Regularization for numerical stability.
    """

    def __init__(
        self,
        method: str = "kl_symmetric",
        regularization: float = _REGULARIZATION,
    ) -> None:
        if method not in ("kl_symmetric", "jsd", "variance_ratio"):
            raise ValueError(f"Unknown method {method!r}")
        self.method = method
        self.regularization = regularization

    def _kl_gaussian_1d(
        self,
        mu1: float,
        var1: float,
        mu2: float,
        var2: float,
    ) -> float:
        """KL(N(mu1,var1) || N(mu2,var2))."""
        var1 = max(var1, _MIN_VARIANCE)
        var2 = max(var2, _MIN_VARIANCE)
        return 0.5 * (np.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / var2 - 1.0)

    def symmetric_kl(
        self,
        mu1: float,
        var1: float,
        mu2: float,
        var2: float,
    ) -> float:
        """Symmetric KL divergence: (KL(P||Q) + KL(Q||P)) / 2.

        Parameters
        ----------
        mu1, mu2 : float
            Means.
        var1, var2 : float
            Variances.

        Returns
        -------
        float
            Symmetric KL (non-negative).
        """
        return 0.5 * (self._kl_gaussian_1d(mu1, var1, mu2, var2) +
                       self._kl_gaussian_1d(mu2, var2, mu1, var1))

    def jsd_1d(
        self,
        mu1: float,
        var1: float,
        mu2: float,
        var2: float,
    ) -> float:
        """Jensen-Shannon divergence for 1D Gaussians (moment-matched)."""
        mu_m = 0.5 * (mu1 + mu2)
        var_m = 0.5 * (var1 + var2) + 0.25 * (mu1 - mu2) ** 2
        kl_p_m = self._kl_gaussian_1d(mu1, var1, mu_m, var_m)
        kl_q_m = self._kl_gaussian_1d(mu2, var2, mu_m, var_m)
        return max(0.5 * kl_p_m + 0.5 * kl_q_m, 0.0)

    def variance_ratio_similarity(self, var1: float, var2: float) -> float:
        """Similarity based on variance ratio.

        sim = 1 - |log(var1/var2)| / (|log(var1/var2)| + 1)

        Maps to [0, 1] with 1 meaning identical variances.

        Parameters
        ----------
        var1, var2 : float
            Variances.

        Returns
        -------
        float
            Similarity in [0, 1].
        """
        var1 = max(var1, _MIN_VARIANCE)
        var2 = max(var2, _MIN_VARIANCE)
        log_ratio = abs(np.log(var1 / var2))
        return 1.0 / (1.0 + log_ratio)

    def score(
        self,
        params_a: MechanismParams,
        params_b: MechanismParams,
        parent_cov: Optional[NDArray[np.floating]] = None,
    ) -> float:
        """Score the shape similarity between two mechanisms.

        Parameters
        ----------
        params_a, params_b : MechanismParams
            Mechanism parameters.
        parent_cov : NDArray or None
            Parent covariance for computing marginal distributions.

        Returns
        -------
        float
            Similarity in [0, 1].
        """
        c_a = np.asarray(params_a.coeffs, dtype=np.float64).ravel()
        c_b = np.asarray(params_b.coeffs, dtype=np.float64).ravel()

        # Compute marginal means and variances
        d = max(len(c_a), len(c_b))
        ca_padded = np.zeros(d)
        cb_padded = np.zeros(d)
        ca_padded[: len(c_a)] = c_a
        cb_padded[: len(c_b)] = c_b

        if parent_cov is not None and d > 0:
            Sigma_pa = np.asarray(parent_cov, dtype=np.float64)
            if Sigma_pa.shape[0] < d:
                new_cov = np.eye(d)
                old_d = Sigma_pa.shape[0]
                new_cov[:old_d, :old_d] = Sigma_pa
                Sigma_pa = new_cov
            mu_pa = np.zeros(d)
        elif d > 0:
            Sigma_pa = np.eye(d)
            mu_pa = np.zeros(d)
        else:
            Sigma_pa = np.array([[1.0]])
            mu_pa = np.zeros(0)

        if d > 0:
            mu_a = params_a.intercept + float(ca_padded @ mu_pa)
            mu_b = params_b.intercept + float(cb_padded @ mu_pa)
            var_a = float(ca_padded @ Sigma_pa[:d, :d] @ ca_padded) + params_a.noise_var
            var_b = float(cb_padded @ Sigma_pa[:d, :d] @ cb_padded) + params_b.noise_var
        else:
            mu_a = params_a.intercept
            mu_b = params_b.intercept
            var_a = params_a.noise_var
            var_b = params_b.noise_var

        var_a = max(var_a, _MIN_VARIANCE)
        var_b = max(var_b, _MIN_VARIANCE)

        if self.method == "kl_symmetric":
            divergence = self.symmetric_kl(mu_a, var_a, mu_b, var_b)
            return 1.0 / (1.0 + divergence)
        elif self.method == "jsd":
            divergence = self.jsd_1d(mu_a, var_a, mu_b, var_b)
            return 1.0 / (1.0 + divergence)
        else:
            # Variance ratio + coefficient similarity
            var_sim = self.variance_ratio_similarity(var_a, var_b)
            if d > 0:
                norm_a = float(np.linalg.norm(ca_padded))
                norm_b = float(np.linalg.norm(cb_padded))
                if norm_a > _EPSILON and norm_b > _EPSILON:
                    coeff_cos = float(ca_padded @ cb_padded) / (norm_a * norm_b)
                    coeff_sim = 0.5 * (1.0 + coeff_cos)
                elif norm_a < _EPSILON and norm_b < _EPSILON:
                    coeff_sim = 1.0
                else:
                    coeff_sim = 0.0
                return 0.5 * var_sim + 0.5 * coeff_sim
            return var_sim


# ===================================================================
#  ScoreMatrix
# ===================================================================
class ScoreMatrix:
    """Construct and normalize alignment score matrices.

    Combines CI-fingerprint similarity and distribution shape similarity
    into a single score matrix for the Hungarian algorithm.

    Parameters
    ----------
    ci_weight : float
        Weight for CI-fingerprint similarity. Default 0.6.
    shape_weight : float
        Weight for distribution shape similarity. Default 0.4.
    mb_overlap_weight : float
        Weight for Markov blanket overlap bonus. Default 0.0 (used as filter only).
    """

    def __init__(
        self,
        ci_weight: float = 0.6,
        shape_weight: float = 0.4,
        mb_overlap_weight: float = 0.0,
    ) -> None:
        total = ci_weight + shape_weight + mb_overlap_weight
        if abs(total) < _EPSILON:
            raise ValueError("At least one weight must be non-zero")
        self.ci_weight = ci_weight / total
        self.shape_weight = shape_weight / total
        self.mb_overlap_weight = mb_overlap_weight / total

    def build(
        self,
        ci_scores: NDArray[np.floating],
        shape_scores: NDArray[np.floating],
        mb_overlap_scores: Optional[NDArray[np.floating]] = None,
        candidate_mask: Optional[NDArray[np.bool_]] = None,
    ) -> NDArray[np.floating]:
        """Build combined score matrix.

        Parameters
        ----------
        ci_scores : NDArray, shape (m, n)
            CI-fingerprint similarity scores.
        shape_scores : NDArray, shape (m, n)
            Distribution shape similarity scores.
        mb_overlap_scores : NDArray or None, shape (m, n)
            MB overlap scores (used as weight if mb_overlap_weight > 0).
        candidate_mask : NDArray of bool or None, shape (m, n)
            If provided, zero out non-candidate pairs.

        Returns
        -------
        NDArray, shape (m, n)
            Combined score matrix.
        """
        m, n = ci_scores.shape
        if shape_scores.shape != (m, n):
            raise ValueError(
                f"Shape mismatch: ci_scores={ci_scores.shape}, shape_scores={shape_scores.shape}"
            )

        combined = self.ci_weight * ci_scores + self.shape_weight * shape_scores

        if mb_overlap_scores is not None and self.mb_overlap_weight > 0:
            if mb_overlap_scores.shape != (m, n):
                raise ValueError("mb_overlap_scores shape mismatch")
            combined += self.mb_overlap_weight * mb_overlap_scores

        if candidate_mask is not None:
            combined = combined * candidate_mask.astype(float)

        return combined

    def normalize(
        self,
        scores: NDArray[np.floating],
        method: str = "min_max",
    ) -> NDArray[np.floating]:
        """Normalize score matrix.

        Parameters
        ----------
        scores : NDArray, shape (m, n)
            Raw score matrix.
        method : str
            'min_max': scale to [0, 1] based on min/max.
            'rank': replace values with rank percentiles.
            'softmax': apply softmax per row.

        Returns
        -------
        NDArray, shape (m, n)
            Normalized score matrix.
        """
        scores = np.asarray(scores, dtype=np.float64)

        if method == "min_max":
            smin = scores.min()
            smax = scores.max()
            if smax - smin < _EPSILON:
                return np.ones_like(scores) * 0.5
            return (scores - smin) / (smax - smin)

        elif method == "rank":
            flat = scores.ravel()
            ranks = sp_stats.rankdata(flat) / len(flat)
            return ranks.reshape(scores.shape)

        elif method == "softmax":
            # Row-wise softmax
            result = np.zeros_like(scores)
            for i in range(scores.shape[0]):
                row = scores[i]
                shifted = row - np.max(row)
                exp_row = np.exp(shifted)
                s = exp_row.sum()
                if s > _EPSILON:
                    result[i] = exp_row / s
                else:
                    result[i] = np.ones_like(row) / len(row)
            return result

        else:
            raise ValueError(f"Unknown normalization method {method!r}")

    def apply_candidate_filter(
        self,
        scores: NDArray[np.floating],
        mb_overlaps: NDArray[np.floating],
        overlap_threshold: float = 0.3,
    ) -> NDArray[np.floating]:
        """Zero out entries below MB overlap threshold.

        Parameters
        ----------
        scores : NDArray, shape (m, n)
            Score matrix.
        mb_overlaps : NDArray, shape (m, n)
            MB overlap scores.
        overlap_threshold : float
            Minimum overlap to keep.

        Returns
        -------
        NDArray, shape (m, n)
            Filtered score matrix.
        """
        mask = mb_overlaps >= overlap_threshold
        return scores * mask.astype(float)


# ===================================================================
#  AnchorValidator
# ===================================================================
class AnchorValidator:
    """Validate anchor consistency and detect conflicts.

    Anchors are known variable correspondences between contexts.
    This validator checks:
        1. No variable is anchored to multiple variables (bijectivity)
        2. Anchored variables have compatible Markov blankets
        3. Anchored variables have compatible distributions (within tolerance)

    Parameters
    ----------
    mb_consistency_threshold : float
        Minimum MB overlap for anchor consistency. Default 0.2.
    distribution_tolerance : float
        Maximum distance for distribution compatibility. Default 0.5.
    """

    def __init__(
        self,
        mb_consistency_threshold: float = 0.2,
        distribution_tolerance: float = 0.5,
    ) -> None:
        self.mb_consistency_threshold = mb_consistency_threshold
        self.distribution_tolerance = distribution_tolerance

    def check_bijectivity(
        self,
        anchors: Dict[int, int],
    ) -> Tuple[bool, List[str]]:
        """Check that anchor map is a partial bijection.

        Parameters
        ----------
        anchors : dict
            Maps variable indices in A to variable indices in B.

        Returns
        -------
        (is_valid, errors) tuple.
        """
        errors = []

        # Check injectivity: no two A-vars map to same B-var
        targets = list(anchors.values())
        target_counts: Dict[int, int] = {}
        for t in targets:
            target_counts[t] = target_counts.get(t, 0) + 1

        for t, count in target_counts.items():
            if count > 1:
                sources = [s for s, tgt in anchors.items() if tgt == t]
                errors.append(
                    f"Multiple sources {sources} map to target {t}"
                )

        return len(errors) == 0, errors

    def check_mb_consistency(
        self,
        anchors: Dict[int, int],
        adj_a: NDArray[np.floating],
        adj_b: NDArray[np.floating],
    ) -> Tuple[bool, List[str], Dict[int, float]]:
        """Check Markov blanket consistency of anchors.

        For each anchor pair (i, j), checks that the translated Markov
        blankets have sufficient overlap.

        Parameters
        ----------
        anchors : dict
            Anchor mapping A -> B.
        adj_a, adj_b : NDArray
            Adjacency matrices.

        Returns
        -------
        (is_consistent, warnings, overlap_scores) tuple.
        """
        warns = []
        overlaps: Dict[int, float] = {}
        mb_helper = MarkovBlanketOverlap()

        for va, vb in anchors.items():
            mb_a = mb_helper.markov_blanket_from_adjacency(adj_a, va)
            mb_b = mb_helper.markov_blanket_from_adjacency(adj_b, vb)

            overlap = mb_helper.anchored_jaccard(mb_a, mb_b, anchors)
            overlaps[va] = overlap

            if overlap < self.mb_consistency_threshold:
                warns.append(
                    f"Anchor ({va}->{vb}) has low MB overlap {overlap:.3f} "
                    f"< {self.mb_consistency_threshold}"
                )

        is_consistent = len(warns) == 0
        return is_consistent, warns, overlaps

    def check_distribution_consistency(
        self,
        anchors: Dict[int, int],
        scm_a: Any,
        scm_b: Any,
    ) -> Tuple[bool, List[str], Dict[int, float]]:
        """Check distribution consistency of anchors.

        For each anchor pair, checks that mechanism distances are within tolerance.

        Parameters
        ----------
        anchors : dict
            Anchor mapping A -> B.
        scm_a, scm_b : StructuralCausalModel (or compatible)
            SCMs for the two contexts.

        Returns
        -------
        (is_consistent, warnings, distances) tuple.
        """
        from cpa.core.mechanism_distance import MechanismDistanceComputer, _scm_mechanism_params, _scm_n_vars

        computer = MechanismDistanceComputer(cache_enabled=False)
        warns = []
        distances: Dict[int, float] = {}

        n_a = _scm_n_vars(scm_a)
        n_b = _scm_n_vars(scm_b)

        for va, vb in anchors.items():
            if va >= n_a or vb >= n_b:
                warns.append(f"Anchor ({va}->{vb}) out of range")
                distances[va] = 1.0
                continue

            params_a = _scm_mechanism_params(scm_a, va)
            params_b = _scm_mechanism_params(scm_b, vb)

            dist = computer.sqrt_jsd_conditional(
                params_a.coeffs, params_a.noise_var,
                params_b.coeffs, params_b.noise_var,
                intercept1=params_a.intercept,
                intercept2=params_b.intercept,
            )
            distances[va] = dist

            if dist > self.distribution_tolerance:
                warns.append(
                    f"Anchor ({va}->{vb}) has high distance {dist:.3f} "
                    f"> {self.distribution_tolerance}"
                )

        is_consistent = len(warns) == 0
        return is_consistent, warns, distances

    def validate(
        self,
        anchors: Dict[int, int],
        adj_a: Optional[NDArray[np.floating]] = None,
        adj_b: Optional[NDArray[np.floating]] = None,
        scm_a: Optional[LinearGaussianSCM] = None,
        scm_b: Optional[LinearGaussianSCM] = None,
    ) -> Dict[str, Any]:
        """Run all validation checks on anchors.

        Parameters
        ----------
        anchors : dict
            Anchor mapping.
        adj_a, adj_b : NDArray or None
            Adjacency matrices (for MB check).
        scm_a, scm_b : LinearGaussianSCM or None
            SCMs (for distribution check).

        Returns
        -------
        dict with keys:
            - 'valid': bool
            - 'bijectivity': (bool, list of str)
            - 'mb_consistency': (bool, list of str, dict) or None
            - 'distribution_consistency': (bool, list of str, dict) or None
        """
        bij_ok, bij_errors = self.check_bijectivity(anchors)
        result: Dict[str, Any] = {
            "valid": bij_ok,
            "bijectivity": (bij_ok, bij_errors),
            "mb_consistency": None,
            "distribution_consistency": None,
        }

        if adj_a is not None and adj_b is not None:
            mb_ok, mb_warns, mb_overlaps = self.check_mb_consistency(anchors, adj_a, adj_b)
            result["mb_consistency"] = (mb_ok, mb_warns, mb_overlaps)
            if not mb_ok:
                result["valid"] = False

        if scm_a is not None and scm_b is not None:
            dist_ok, dist_warns, dist_scores = self.check_distribution_consistency(
                anchors, scm_a, scm_b
            )
            result["distribution_consistency"] = (dist_ok, dist_warns, dist_scores)
            if not dist_ok:
                result["valid"] = False

        return result


# ===================================================================
#  AlignmentScorer — unified scoring for candidate pairs
# ===================================================================
class AlignmentScorer:
    """Unified alignment scorer combining all scoring components.

    Orchestrates CI-fingerprint, distribution shape, and MB overlap
    scoring into a single interface.

    Parameters
    ----------
    ci_weight : float
        Weight for CI-fingerprint similarity. Default 0.6.
    shape_weight : float
        Weight for distribution shape similarity. Default 0.4.
    ci_method : str
        CI-fingerprint comparison method. Default 'cosine'.
    shape_method : str
        Distribution shape comparison method. Default 'kl_symmetric'.
    mb_overlap_threshold : float
        Minimum MB overlap for candidate acceptance. Default 0.3.
    significance_threshold : float
        Threshold for significant partial correlations. Default 0.05.
    """

    def __init__(
        self,
        ci_weight: float = 0.6,
        shape_weight: float = 0.4,
        ci_method: str = "cosine",
        shape_method: str = "kl_symmetric",
        mb_overlap_threshold: float = 0.3,
        significance_threshold: float = 0.05,
    ) -> None:
        self.ci_weight = ci_weight
        self.shape_weight = shape_weight
        self.ci_scorer = CIFingerprintScorer(
            method=ci_method,
            significance_threshold=significance_threshold,
        )
        self.shape_scorer = DistributionShapeSimilarity(method=shape_method)
        self.mb_overlap = MarkovBlanketOverlap(overlap_threshold=mb_overlap_threshold)
        self.score_builder = ScoreMatrix(ci_weight=ci_weight, shape_weight=shape_weight)

    def score_candidate_pair(
        self,
        var_a: int,
        var_b: int,
        scm_a: Any,
        scm_b: Any,
        anchors: Optional[Dict[int, int]] = None,
        context_a: str = "",
        context_b: str = "",
    ) -> Dict[str, float]:
        """Score a single candidate pair across all dimensions.

        Parameters
        ----------
        var_a, var_b : int
            Variable indices.
        scm_a, scm_b : StructuralCausalModel (or compatible)
            SCMs for each context.
        anchors : dict or None
            Anchor mapping.
        context_a, context_b : str
            Context identifiers.

        Returns
        -------
        dict with keys:
            - 'combined': float -- weighted combined score
            - 'ci_fingerprint': float -- CI-fingerprint similarity
            - 'distribution_shape': float -- distribution shape similarity
            - 'mb_overlap': float -- Markov blanket overlap
        """
        cov_a = _scm_implied_covariance(scm_a)
        cov_b = _scm_implied_covariance(scm_b)

        # CI-fingerprint similarity
        ci_score = self.ci_scorer.score_pair(
            var_a, var_b, cov_a, cov_b,
            alignment=anchors,
            ctx_a=context_a,
            ctx_b=context_b,
        )

        # Distribution shape similarity
        params_a = _scm_mechanism_params(scm_a, var_a)
        params_b = _scm_mechanism_params(scm_b, var_b)
        shape_score = self.shape_scorer.score(params_a, params_b)

        # MB overlap
        adj_a = _scm_adjacency(scm_a)
        adj_b = _scm_adjacency(scm_b)
        mb_a = self.mb_overlap.markov_blanket_from_adjacency(adj_a, var_a)
        mb_b = self.mb_overlap.markov_blanket_from_adjacency(adj_b, var_b)
        overlap = self.mb_overlap.anchored_jaccard(
            mb_a, mb_b, anchors if anchors else {}
        )

        combined = self.ci_weight * ci_score + self.shape_weight * shape_score

        return {
            "combined": combined,
            "ci_fingerprint": ci_score,
            "distribution_shape": shape_score,
            "mb_overlap": overlap,
        }

    def score_all_candidates(
        self,
        unanchored_a: List[int],
        unanchored_b: List[int],
        scm_a: Any,
        scm_b: Any,
        anchors: Optional[Dict[int, int]] = None,
        context_a: str = "",
        context_b: str = "",
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Score all candidate pairs.

        Parameters
        ----------
        unanchored_a, unanchored_b : list of int
            Unanchored variable indices.
        scm_a, scm_b : StructuralCausalModel (or compatible)
            SCMs.
        anchors : dict or None
            Anchor mapping.
        context_a, context_b : str
            Context identifiers.

        Returns
        -------
        (ci_scores, shape_scores, mb_overlaps) tuple of NDArrays,
            each with shape (len(unanchored_a), len(unanchored_b)).
        """
        n_a = len(unanchored_a)
        n_b = len(unanchored_b)

        ci_scores = np.zeros((n_a, n_b), dtype=np.float64)
        shape_scores = np.zeros((n_a, n_b), dtype=np.float64)
        mb_overlaps = np.zeros((n_a, n_b), dtype=np.float64)

        cov_a = _scm_implied_covariance(scm_a)
        cov_b = _scm_implied_covariance(scm_b)
        adj_a = _scm_adjacency(scm_a)
        adj_b = _scm_adjacency(scm_b)

        for i, va in enumerate(unanchored_a):
            mb_a = self.mb_overlap.markov_blanket_from_adjacency(adj_a, va)
            params_a = _scm_mechanism_params(scm_a, va)

            for j, vb in enumerate(unanchored_b):
                # CI fingerprint
                ci_scores[i, j] = self.ci_scorer.score_pair(
                    va, vb, cov_a, cov_b,
                    alignment=anchors,
                    ctx_a=context_a,
                    ctx_b=context_b,
                )

                # Shape
                params_b = _scm_mechanism_params(scm_b, vb)
                shape_scores[i, j] = self.shape_scorer.score(params_a, params_b)

                # MB overlap
                mb_b = self.mb_overlap.markov_blanket_from_adjacency(adj_b, vb)
                mb_overlaps[i, j] = self.mb_overlap.anchored_jaccard(
                    mb_a, mb_b, anchors if anchors else {}
                )

        return ci_scores, shape_scores, mb_overlaps

    def build_score_matrix(
        self,
        ci_scores: NDArray[np.floating],
        shape_scores: NDArray[np.floating],
        mb_overlaps: Optional[NDArray[np.floating]] = None,
        mb_threshold: Optional[float] = None,
    ) -> NDArray[np.floating]:
        """Build combined score matrix with optional MB filtering.

        Parameters
        ----------
        ci_scores, shape_scores : NDArray
            Individual score matrices.
        mb_overlaps : NDArray or None
            MB overlap scores.
        mb_threshold : float or None
            If set, zero out pairs below this MB overlap.

        Returns
        -------
        NDArray
            Combined score matrix.
        """
        combined = self.score_builder.build(ci_scores, shape_scores)

        if mb_overlaps is not None and mb_threshold is not None:
            mask = mb_overlaps >= mb_threshold
            combined = combined * mask.astype(float)

        return combined
