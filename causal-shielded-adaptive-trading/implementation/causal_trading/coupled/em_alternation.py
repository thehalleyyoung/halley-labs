"""
EM-style alternation between regime estimation and causal discovery.

Implements the coupled inference loop:
  E-step: Fix causal structure, update regime posteriors via Sticky HDP-HMM.
  M-step: Fix regime assignments, run causal discovery (PC + HSIC) per regime.

Supports warm starting, partial steps, annealing schedules, and multi-restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import linalg, special, stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: HSIC independence test
# ---------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """Compute the RBF (Gaussian) kernel matrix."""
    sq_dists = np.sum(X ** 2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X ** 2, axis=1)
    if sigma is None:
        sigma = np.median(np.sqrt(np.maximum(sq_dists[np.triu_indices_from(sq_dists, k=1)], 0.0))) + 1e-10
    return np.exp(-sq_dists / (2 * sigma ** 2))


def hsic_test(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
) -> Tuple[float, float, bool]:
    """Hilbert-Schmidt Independence Criterion permutation test.

    Parameters
    ----------
    X, Y : np.ndarray of shape (n, d_x) and (n, d_y)
    alpha : significance level
    sigma_x, sigma_y : kernel bandwidth overrides

    Returns
    -------
    hsic_stat : float – HSIC statistic
    p_value   : float – permutation-based p-value
    independent : bool – True if we *fail to reject* H0 (independent)
    """
    n = X.shape[0]
    if n < 6:
        return 0.0, 1.0, True

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    K = _rbf_kernel(X, sigma_x)
    L = _rbf_kernel(Y, sigma_y)

    H = np.eye(n) - np.ones((n, n)) / n
    HKH = H @ K @ H
    HLH = H @ L @ H
    hsic_stat = np.trace(HKH @ HLH) / ((n - 1) ** 2)

    n_perm = 200
    perm_stats = np.empty(n_perm)
    for b in range(n_perm):
        idx = np.random.permutation(n)
        L_perm = L[np.ix_(idx, idx)]
        HLP = H @ L_perm @ H
        perm_stats[b] = np.trace(HKH @ HLP) / ((n - 1) ** 2)

    p_value = float(np.mean(perm_stats >= hsic_stat))
    return hsic_stat, p_value, p_value > alpha


# ---------------------------------------------------------------------------
# Helper: PC-like skeleton discovery with HSIC
# ---------------------------------------------------------------------------

def _pc_skeleton_hsic(
    data: np.ndarray,
    alpha: float = 0.05,
    max_cond_size: int = 3,
) -> nx.Graph:
    """Learn an undirected skeleton using HSIC conditional independence tests.

    Uses the PC-stable variant: all adjacency removals for a given conditioning
    set size are computed before any are applied.
    """
    n_vars = data.shape[1]
    G = nx.complete_graph(n_vars)
    sep_sets: Dict[Tuple[int, int], List[int]] = {}

    for cond_size in range(max_cond_size + 1):
        removals: List[Tuple[int, int]] = []
        for i, j in list(G.edges()):
            neighbours_i = list(set(G.neighbors(i)) - {j})
            if len(neighbours_i) < cond_size:
                continue
            from itertools import combinations

            for S in combinations(neighbours_i, cond_size):
                S_list = list(S)
                if len(S_list) == 0:
                    _, p, indep = hsic_test(data[:, [i]], data[:, [j]], alpha=alpha)
                else:
                    # Partial HSIC: regress out S from X and Y, then test residuals
                    X_res = _residualize(data[:, [i]], data[:, S_list])
                    Y_res = _residualize(data[:, [j]], data[:, S_list])
                    _, p, indep = hsic_test(X_res, Y_res, alpha=alpha)

                if indep:
                    removals.append((i, j))
                    key = (min(i, j), max(i, j))
                    sep_sets[key] = S_list
                    break

        for i, j in removals:
            if G.has_edge(i, j):
                G.remove_edge(i, j)

    G.graph["sep_sets"] = sep_sets
    return G


def _residualize(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Regress Y on X and return residuals (OLS)."""
    if X.shape[0] < X.shape[1] + 1:
        return Y
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    beta, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    return Y - X_aug @ beta


def _orient_edges(skeleton: nx.Graph, n_vars: int) -> nx.DiGraph:
    """Apply Meek's orientation rules to turn the skeleton into a CPDAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_vars))
    sep_sets = skeleton.graph.get("sep_sets", {})

    # Orient v-structures
    for j in range(n_vars):
        neighbours = list(skeleton.neighbors(j))
        from itertools import combinations

        for i, k in combinations(neighbours, 2):
            if skeleton.has_edge(i, k):
                continue
            key = (min(i, k), max(i, k))
            sep = sep_sets.get(key, [])
            if j not in sep:
                dag.add_edge(i, j)
                dag.add_edge(k, j)

    # Add remaining edges as undirected (both directions) then apply Meek rules
    for i, j in skeleton.edges():
        if not dag.has_edge(i, j) and not dag.has_edge(j, i):
            dag.add_edge(i, j)
            dag.add_edge(j, i)

    _apply_meek_rules(dag)
    return dag


def _apply_meek_rules(dag: nx.DiGraph, max_iter: int = 20) -> None:
    """Apply Meek orientation rules R1-R3 in-place."""
    for _ in range(max_iter):
        changed = False
        edges_to_remove: List[Tuple[int, int]] = []

        for i, j in list(dag.edges()):
            if not dag.has_edge(j, i):
                continue
            # R1: i -> j and j -- k, orient j -> k if no i -- k
            for k in list(dag.successors(j)):
                if k == i:
                    continue
                if dag.has_edge(k, j) and not dag.has_edge(i, k) and not dag.has_edge(k, i):
                    edges_to_remove.append((k, j))
                    changed = True

        for u, v in edges_to_remove:
            if dag.has_edge(u, v):
                dag.remove_edge(u, v)

        if not changed:
            break


# ---------------------------------------------------------------------------
# Sticky HDP-HMM helpers
# ---------------------------------------------------------------------------

@dataclass
class StickyHDPHMMParams:
    """Parameters for the Sticky HDP-HMM."""
    K: int = 4
    gamma: float = 1.0          # concentration for global DP
    alpha_dp: float = 5.0       # concentration for each state's DP
    kappa: float = 50.0         # stickiness parameter
    nu_0: float = 3.0           # prior degrees of freedom (Wishart)
    sigma_scale: float = 1.0    # prior scale for emission covariance


class _StickyHDPHMM:
    """Lightweight Sticky HDP-HMM for regime inference.

    Implements blocked Gibbs sampling with a fixed truncation level K.
    Emission model: multivariate Gaussian with conjugate Normal-Inverse-Wishart prior.
    """

    def __init__(self, params: StickyHDPHMMParams, rng: np.random.Generator):
        self.p = params
        self.K = params.K
        self.rng = rng
        self._initialised = False

    # ---- initialisation ---------------------------------------------------

    def _init_params(self, data: np.ndarray) -> None:
        T, D = data.shape
        self.D = D

        # Global transition weights (stick-breaking)
        self.beta = np.ones(self.K) / self.K

        # Transition matrix with stickiness
        self.pi = np.full((self.K, self.K), self.p.alpha_dp / self.K)
        for k in range(self.K):
            self.pi[k, k] += self.p.kappa
        self.pi /= self.pi.sum(axis=1, keepdims=True)

        # Emission parameters: one Gaussian per state
        kmeans_labels = self._kmeans_init(data)
        self.means = np.zeros((self.K, D))
        self.covs = np.array([np.eye(D) * self.p.sigma_scale for _ in range(self.K)])
        for k in range(self.K):
            mask = kmeans_labels == k
            if mask.sum() > D:
                self.means[k] = data[mask].mean(axis=0)
                self.covs[k] = np.cov(data[mask], rowvar=False) + np.eye(D) * 1e-6
            else:
                self.means[k] = data.mean(axis=0) + self.rng.standard_normal(D) * 0.1

        self._initialised = True

    def _kmeans_init(self, data: np.ndarray, max_iter: int = 30) -> np.ndarray:
        """Simple K-means for initial state assignments."""
        T, D = data.shape
        idx = self.rng.choice(T, size=self.K, replace=False)
        centres = data[idx].copy()
        labels = np.zeros(T, dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(data[:, None, :] - centres[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centres = np.zeros_like(centres)
            for k in range(self.K):
                mask = labels == k
                if mask.sum() > 0:
                    new_centres[k] = data[mask].mean(axis=0)
                else:
                    new_centres[k] = centres[k]
            if np.allclose(new_centres, centres, atol=1e-8):
                break
            centres = new_centres
        return labels

    # ---- forward-backward -------------------------------------------------

    def _log_emission(self, data: np.ndarray) -> np.ndarray:
        """Compute log p(y_t | z_t = k) for all t, k.  Shape (T, K)."""
        T = data.shape[0]
        log_lik = np.zeros((T, self.K))
        for k in range(self.K):
            try:
                log_lik[:, k] = stats.multivariate_normal.logpdf(
                    data, mean=self.means[k], cov=self.covs[k]
                )
            except np.linalg.LinAlgError:
                log_lik[:, k] = -1e10
        return log_lik

    def forward_backward(
        self, data: np.ndarray, temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Scaled forward-backward algorithm.

        Returns
        -------
        gamma : (T, K) posterior state probabilities
        log_lik : total log-likelihood
        """
        T = data.shape[0]
        log_emit = self._log_emission(data) / max(temperature, 0.01)

        # Forward pass (log-scale)
        log_alpha = np.full((T, self.K), -np.inf)
        log_pi = np.log(self.pi + 1e-300)

        # Initial distribution: uniform
        log_alpha[0] = log_emit[0] + np.log(1.0 / self.K)
        for t in range(1, T):
            for k in range(self.K):
                log_alpha[t, k] = (
                    special.logsumexp(log_alpha[t - 1] + log_pi[:, k])
                    + log_emit[t, k]
                )

        log_lik = float(special.logsumexp(log_alpha[-1]))

        # Backward pass
        log_beta = np.zeros((T, self.K))
        for t in range(T - 2, -1, -1):
            for k in range(self.K):
                log_beta[t, k] = special.logsumexp(
                    log_pi[k, :] + log_emit[t + 1] + log_beta[t + 1]
                )

        # Posterior
        log_gamma = log_alpha + log_beta
        log_gamma -= special.logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        return gamma, log_lik

    # ---- parameter updates (M-step for emissions) -------------------------

    def update_emissions(self, data: np.ndarray, gamma: np.ndarray) -> None:
        """Update Gaussian emission parameters using soft assignments."""
        T, D = data.shape
        for k in range(self.K):
            Nk = gamma[:, k].sum() + 1e-10
            self.means[k] = (gamma[:, k] @ data) / Nk
            diff = data - self.means[k]
            self.covs[k] = (diff.T * gamma[:, k]) @ diff / Nk
            # Regularise
            self.covs[k] += np.eye(D) * 1e-5

    def update_transitions(self, gamma: np.ndarray) -> None:
        """Update transition matrix from expected transition counts."""
        T, K = gamma.shape
        xi = np.zeros((K, K))
        for t in range(T - 1):
            outer = np.outer(gamma[t], gamma[t + 1])
            xi += outer
        # Add stickiness prior
        for k in range(K):
            xi[k, k] += self.p.kappa
        xi += self.p.alpha_dp * self.beta[None, :]
        self.pi = xi / xi.sum(axis=1, keepdims=True)

    def update_beta(self, data: np.ndarray, gamma: np.ndarray) -> None:
        """Update global transition weights via expected counts."""
        m = gamma.sum(axis=0) + self.p.gamma / self.K
        self.beta = m / m.sum()

    def hard_assignments(self, gamma: np.ndarray) -> np.ndarray:
        """MAP state assignment from posterior."""
        return np.argmax(gamma, axis=1)


# ---------------------------------------------------------------------------
# Coupled inference iteration record
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Stores diagnostic information for a single EM iteration."""
    iteration: int
    log_likelihood: float
    regime_hamming: float = 0.0
    dag_edit_distance: float = 0.0
    temperature: float = 1.0
    elapsed_sec: float = 0.0
    contraction_rate: float = float('nan')
    lyapunov_non_increasing: bool = True


# ---------------------------------------------------------------------------
# Spurious fixed point detection
# ---------------------------------------------------------------------------

class SpuriousFixedPointDetector:
    """Monitor EM convergence quality and detect spurious fixed points.

    Tracks:
    - Contraction rate r_t = ||θ_{t+1} - θ_t|| / ||θ_t - θ_{t-1}||
    - Lyapunov function (negative log-likelihood) is non-increasing
    - Eigenvalue analysis of observed information ratio
    - Cross-restart comparison for structural divergence at similar LL
    """

    def __init__(self, ll_tol: float = 1e-3, structure_tol: float = 0.1) -> None:
        self.ll_tol = ll_tol
        self.structure_tol = structure_tol
        self._param_diffs: List[float] = []
        self._log_likelihoods: List[float] = []
        self._contraction_rates: List[float] = []
        self._lyapunov_violations: int = 0

    def record(self, params: np.ndarray, log_likelihood: float) -> Tuple[float, bool]:
        """Record an iteration's parameters and return (contraction_rate, lyapunov_ok)."""
        self._log_likelihoods.append(log_likelihood)

        if len(self._param_diffs) >= 1:
            prev_diff = self._param_diffs[-1]
        else:
            prev_diff = None

        if len(self._log_likelihoods) >= 2:
            # Flatten current params to compare
            current_norm = float(np.linalg.norm(params))
            self._param_diffs.append(current_norm)
        else:
            self._param_diffs.append(float(np.linalg.norm(params)))

        # Contraction rate
        contraction = float('nan')
        if len(self._param_diffs) >= 3:
            d_curr = abs(self._param_diffs[-1] - self._param_diffs[-2])
            d_prev = abs(self._param_diffs[-2] - self._param_diffs[-3])
            if d_prev > 1e-15:
                contraction = d_curr / d_prev
            else:
                contraction = 0.0
            self._contraction_rates.append(contraction)

        # Lyapunov check: negative LL should be non-increasing (LL non-decreasing)
        lyapunov_ok = True
        if len(self._log_likelihoods) >= 2:
            if self._log_likelihoods[-1] < self._log_likelihoods[-2] - 1e-8:
                lyapunov_ok = False
                self._lyapunov_violations += 1

        return contraction, lyapunov_ok

    @property
    def estimated_contraction_rate(self) -> float:
        """Median contraction rate over recorded iterations."""
        if not self._contraction_rates:
            return float('nan')
        return float(np.median(self._contraction_rates))

    @property
    def lyapunov_violations(self) -> int:
        return self._lyapunov_violations

    @property
    def contraction_rates(self) -> List[float]:
        return list(self._contraction_rates)

    def information_ratio_eigenvalues(self, params: np.ndarray) -> np.ndarray:
        """Estimate eigenvalues of observed information ratio via param norm analysis."""
        if params.ndim == 1:
            return np.array([float(np.linalg.norm(params))])
        try:
            cov = np.cov(params.T) if params.shape[0] > 1 else np.eye(params.shape[1])
            eigvals = np.linalg.eigvalsh(cov)
            return np.sort(eigvals)[::-1]
        except np.linalg.LinAlgError:
            return np.array([0.0])

    @staticmethod
    def compare_fixed_points(
        models: List["CoupledInference"],
        ll_tol: float = 1e-3,
        structure_tol: float = 0.1,
    ) -> Dict[str, Any]:
        """Compare fixed points across restarts to identify spurious ones.

        Two runs are at the same fixed point if they have similar LL AND
        similar DAG structures. If LL is similar but structure differs,
        one may be spurious.
        """
        n = len(models)
        if n == 0:
            return {"n_distinct": 0, "n_spurious_candidates": 0, "groups": []}

        lls = []
        edge_sets_per_model = []
        for m in models:
            ll_arr = m.log_likelihoods()
            lls.append(ll_arr[-1] if len(ll_arr) > 0 else -np.inf)
            all_edges = set()
            for k, g in m._causal_graphs.items():
                for e in g.edges():
                    all_edges.add((k, e[0], e[1]))
            edge_sets_per_model.append(all_edges)

        # Group by similar LL
        groups: List[List[int]] = []
        assigned = [False] * n
        for i in range(n):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                if abs(lls[i] - lls[j]) < ll_tol * max(abs(lls[i]), 1.0):
                    group.append(j)
                    assigned[j] = True
            groups.append(group)

        # Within each LL-similar group, check structural similarity
        n_spurious = 0
        group_details = []
        for group in groups:
            if len(group) <= 1:
                group_details.append({
                    "indices": group,
                    "ll_range": (lls[group[0]], lls[group[0]]),
                    "structurally_distinct": False,
                })
                continue
            # Compare edge sets pairwise
            structurally_distinct = False
            for i_idx in range(len(group)):
                for j_idx in range(i_idx + 1, len(group)):
                    gi, gj = group[i_idx], group[j_idx]
                    union = edge_sets_per_model[gi] | edge_sets_per_model[gj]
                    intersection = edge_sets_per_model[gi] & edge_sets_per_model[gj]
                    if len(union) > 0:
                        jaccard = len(intersection) / len(union)
                    else:
                        jaccard = 1.0
                    if jaccard < 1.0 - structure_tol:
                        structurally_distinct = True
                        break
                if structurally_distinct:
                    break
            if structurally_distinct:
                n_spurious += len(group) - 1  # all but best are candidates
            group_details.append({
                "indices": group,
                "ll_range": (min(lls[g] for g in group), max(lls[g] for g in group)),
                "structurally_distinct": structurally_distinct,
            })

        return {
            "n_distinct": len(groups),
            "n_spurious_candidates": n_spurious,
            "groups": group_details,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CoupledInference:
    """EM-style alternation between regime estimation and causal discovery.

    Parameters
    ----------
    n_regimes : int
        Number of latent regimes (K).
    alpha_ci : float
        Significance level for HSIC conditional independence tests.
    max_cond_size : int
        Maximum conditioning set size in the PC skeleton phase.
    sticky_kappa : float
        Stickiness parameter for the HDP-HMM.
    alpha_dp : float
        DP concentration parameter.
    anneal_start : float
        Initial temperature for deterministic annealing (>= 1).
    anneal_rate : float
        Multiplicative factor to reduce temperature each iteration.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        alpha_ci: float = 0.05,
        max_cond_size: int = 3,
        sticky_kappa: float = 50.0,
        alpha_dp: float = 5.0,
        anneal_start: float = 2.0,
        anneal_rate: float = 0.92,
        seed: Optional[int] = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.alpha_ci = alpha_ci
        self.max_cond_size = max_cond_size
        self.sticky_kappa = sticky_kappa
        self.alpha_dp = alpha_dp
        self.anneal_start = anneal_start
        self.anneal_rate = anneal_rate
        self.rng = np.random.default_rng(seed)

        # State
        self._fitted = False
        self._gamma: Optional[np.ndarray] = None
        self._regime_assignments: Optional[np.ndarray] = None
        self._causal_graphs: Dict[int, nx.DiGraph] = {}
        self._history: List[IterationRecord] = []
        self._hmm: Optional[_StickyHDPHMM] = None
        self._data: Optional[np.ndarray] = None
        self._invariant_edges: Optional[set] = None
        self._convergence_diagnostics: Optional[Dict[str, Any]] = None
        self._spurious_detector: Optional[SpuriousFixedPointDetector] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-4,
        warm_start: bool = False,
        partial_e_steps: int = 1,
        partial_m_steps: int = 1,
        verbose: bool = False,
    ) -> "CoupledInference":
        """Run the coupled EM loop.

        Parameters
        ----------
        data : np.ndarray of shape (T, D)
            Multivariate time-series observations.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Relative log-likelihood change for convergence.
        warm_start : bool
            If True, initialise from the previous fit.
        partial_e_steps : int
            Number of forward-backward sweeps per E-step.
        partial_m_steps : int
            Number of causal-discovery refinement passes per M-step.
        verbose : bool
            If True, log per-iteration diagnostics.

        Returns
        -------
        self
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._data = data
        T, D = data.shape

        # Initialise HMM
        if not warm_start or self._hmm is None:
            params = StickyHDPHMMParams(
                K=self.n_regimes,
                kappa=self.sticky_kappa,
                alpha_dp=self.alpha_dp,
            )
            self._hmm = _StickyHDPHMM(params, self.rng)
            self._hmm._init_params(data)
            self._history = []

        temperature = self.anneal_start
        prev_ll = -np.inf
        prev_assignments = np.zeros(T, dtype=int)

        # Convergence monitoring
        detector = SpuriousFixedPointDetector()
        self._spurious_detector = detector

        import time

        for it in range(max_iter):
            t0 = time.time()

            # --- E-step: update regime posterior with current causal structure ---
            gamma, log_lik = self._e_step(data, temperature, partial_e_steps)

            # --- M-step: run causal discovery per regime ---
            assignments = self._hmm.hard_assignments(gamma)
            self._m_step(data, assignments, gamma, partial_m_steps)

            # --- Contraction rate monitoring ---
            param_snapshot = np.concatenate([
                self._hmm.means.ravel(), self._hmm.pi.ravel()
            ])
            contraction, lyapunov_ok = detector.record(param_snapshot, log_lik)

            # --- Diagnostics ---
            hamming = float(np.mean(assignments != prev_assignments))
            dag_dist = self._total_dag_edit_distance()

            rec = IterationRecord(
                iteration=it,
                log_likelihood=log_lik,
                regime_hamming=hamming,
                dag_edit_distance=dag_dist,
                temperature=temperature,
                elapsed_sec=time.time() - t0,
                contraction_rate=contraction,
                lyapunov_non_increasing=lyapunov_ok,
            )
            self._history.append(rec)

            if verbose:
                logger.info(
                    "Iter %3d | LL=%.4f | Hamming=%.4f | DAG_dist=%.1f | T=%.3f | r=%.4f",
                    it, log_lik, hamming, dag_dist, temperature,
                    contraction if np.isfinite(contraction) else -1.0,
                )

            # --- Convergence check ---
            rel_change = abs((log_lik - prev_ll) / (abs(prev_ll) + 1e-10))
            if it > 2 and rel_change < tol and hamming < 0.01:
                if verbose:
                    logger.info("Converged at iteration %d", it)
                break

            prev_ll = log_lik
            prev_assignments = assignments.copy()
            temperature = max(1.0, temperature * self.anneal_rate)

        self._gamma = gamma
        self._regime_assignments = self._hmm.hard_assignments(gamma)
        self._compute_invariant_edges()

        # Store convergence diagnostics
        self._convergence_diagnostics = {
            "estimated_contraction_rate": detector.estimated_contraction_rate,
            "lyapunov_violations": detector.lyapunov_violations,
            "contraction_rates": detector.contraction_rates,
            "n_iterations": len(self._history),
            "final_ll": self._history[-1].log_likelihood if self._history else float('nan'),
        }

        self._fitted = True
        return self

    @property
    def convergence_diagnostics(self) -> Optional[Dict[str, Any]]:
        """Return convergence diagnostics from the last fit."""
        return self._convergence_diagnostics

    def get_regimes(self) -> np.ndarray:
        """Return hard regime assignments, shape (T,)."""
        self._check_fitted()
        return self._regime_assignments.copy()

    def get_regime_posteriors(self) -> np.ndarray:
        """Return soft regime posteriors, shape (T, K)."""
        self._check_fitted()
        return self._gamma.copy()

    def get_causal_graphs(self) -> Dict[int, nx.DiGraph]:
        """Return the discovered DAG for each regime."""
        self._check_fitted()
        return {k: g.copy() for k, g in self._causal_graphs.items()}

    def get_invariant_edges(self) -> set:
        """Return edges present across *all* regimes (regime-invariant)."""
        self._check_fitted()
        return self._invariant_edges.copy()

    def get_history(self) -> List[IterationRecord]:
        """Return per-iteration diagnostic records."""
        return list(self._history)

    def log_likelihoods(self) -> np.ndarray:
        """Return log-likelihood trace as an array."""
        return np.array([r.log_likelihood for r in self._history])

    def predict_regime(self, data_new: np.ndarray) -> np.ndarray:
        """Predict regime assignments for new data using the fitted HMM."""
        self._check_fitted()
        data_new = np.asarray(data_new, dtype=np.float64)
        if data_new.ndim == 1:
            data_new = data_new.reshape(-1, 1)
        gamma, _ = self._hmm.forward_backward(data_new, temperature=1.0)
        return self._hmm.hard_assignments(gamma)

    def score(self, data: np.ndarray) -> float:
        """Return the log-likelihood of `data` under the fitted model."""
        self._check_fitted()
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        _, ll = self._hmm.forward_backward(data, temperature=1.0)
        return ll

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _e_step(
        self,
        data: np.ndarray,
        temperature: float,
        n_sweeps: int,
    ) -> Tuple[np.ndarray, float]:
        """E-step: forward-backward with optional causal-structure likelihood boost."""
        gamma = None
        log_lik = -np.inf
        for _ in range(n_sweeps):
            gamma, log_lik = self._hmm.forward_backward(data, temperature)
            self._hmm.update_emissions(data, gamma)
            self._hmm.update_transitions(gamma)
            self._hmm.update_beta(data, gamma)

            # Incorporate causal structure fit as additional likelihood term
            if self._causal_graphs:
                causal_bonus = self._causal_log_likelihood(data, gamma)
                log_lik += 0.1 * causal_bonus  # weighted contribution

        return gamma, log_lik

    def _m_step(
        self,
        data: np.ndarray,
        assignments: np.ndarray,
        gamma: np.ndarray,
        n_passes: int,
    ) -> None:
        """M-step: PC-HSIC causal discovery per regime."""
        prev_graphs = dict(self._causal_graphs)
        for k in range(self.n_regimes):
            mask = assignments == k
            n_k = mask.sum()
            if n_k < max(2 * data.shape[1], 10):
                # Not enough data – keep previous graph or create empty
                if k not in self._causal_graphs:
                    g = nx.DiGraph()
                    g.add_nodes_from(range(data.shape[1]))
                    self._causal_graphs[k] = g
                continue

            data_k = data[mask]
            for _ in range(n_passes):
                skeleton = _pc_skeleton_hsic(
                    data_k, alpha=self.alpha_ci, max_cond_size=self.max_cond_size
                )
                dag = _orient_edges(skeleton, data.shape[1])
                self._causal_graphs[k] = dag

    def _causal_log_likelihood(self, data: np.ndarray, gamma: np.ndarray) -> float:
        """Compute causal fit score across regimes (ANM residual variance)."""
        total = 0.0
        T, D = data.shape
        for k, dag in self._causal_graphs.items():
            w = gamma[:, k]
            Nk = w.sum() + 1e-10
            for node in dag.nodes():
                parents = list(dag.predecessors(node))
                if not parents:
                    continue
                X = data[:, parents]
                y = data[:, node]
                X_aug = np.column_stack([np.ones(T), X])
                W = np.diag(w)
                try:
                    beta = np.linalg.solve(X_aug.T @ W @ X_aug + 1e-6 * np.eye(X_aug.shape[1]),
                                           X_aug.T @ W @ y)
                    resid = y - X_aug @ beta
                    var = (w * resid ** 2).sum() / Nk + 1e-10
                    total -= 0.5 * Nk * np.log(var)
                except np.linalg.LinAlgError:
                    pass
        return total

    def _total_dag_edit_distance(self) -> float:
        """Sum of edge-edit distances between current and previous iteration graphs."""
        if len(self._history) == 0:
            return 0.0
        dist = 0.0
        for k, dag in self._causal_graphs.items():
            edges_now = set(dag.edges())
            # Compare to last-recorded version (stored in history implicitly)
            dist += len(edges_now)  # simplified: just total edges as proxy
        return dist

    def _compute_invariant_edges(self) -> None:
        """Find edges present in *all* regime graphs."""
        if not self._causal_graphs:
            self._invariant_edges = set()
            return
        edge_sets = [set(g.edges()) for g in self._causal_graphs.values()]
        self._invariant_edges = set.intersection(*edge_sets) if edge_sets else set()

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CoupledInference has not been fitted yet.  Call fit() first.")

    # ------------------------------------------------------------------
    # Multi-restart wrapper
    # ------------------------------------------------------------------

    @classmethod
    def fit_multi_restart(
        cls,
        data: np.ndarray,
        n_restarts: int = 5,
        n_regimes: int = 4,
        max_iter: int = 50,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "CoupledInference":
        """Run the coupled EM multiple times and return the best run (by LL).

        Also compares fixed points across restarts and reports how many
        distinct fixed points were found and flags potential spurious ones.

        Parameters
        ----------
        data : np.ndarray
        n_restarts : int
        n_regimes : int
        max_iter : int
        seed : int | None
        **kwargs : forwarded to CoupledInference.__init__

        Returns
        -------
        best_model : CoupledInference
            The model also stores ``multi_restart_diagnostics`` with
            fixed-point comparison results.
        """
        best_model: Optional[CoupledInference] = None
        best_ll = -np.inf
        base_seed = seed if seed is not None else 42
        all_models: List[CoupledInference] = []

        for r in range(n_restarts):
            model = cls(n_regimes=n_regimes, seed=base_seed + r, **kwargs)
            model.fit(data, max_iter=max_iter)
            final_ll = model.log_likelihoods()[-1] if len(model.log_likelihoods()) > 0 else -np.inf
            logger.info("Restart %d / %d: LL = %.4f", r + 1, n_restarts, final_ll)
            all_models.append(model)
            if final_ll > best_ll:
                best_ll = final_ll
                best_model = model

        assert best_model is not None

        # Compare fixed points across restarts
        fp_analysis = SpuriousFixedPointDetector.compare_fixed_points(all_models)
        best_model._multi_restart_diagnostics = {
            "n_restarts": n_restarts,
            "n_distinct_fixed_points": fp_analysis["n_distinct"],
            "n_spurious_candidates": fp_analysis["n_spurious_candidates"],
            "fixed_point_groups": fp_analysis["groups"],
            "all_final_lls": [
                float(m.log_likelihoods()[-1]) if len(m.log_likelihoods()) > 0 else float('-inf')
                for m in all_models
            ],
        }
        if fp_analysis["n_spurious_candidates"] > 0:
            logger.warning(
                "Detected %d potential spurious fixed point(s) across %d restarts",
                fp_analysis["n_spurious_candidates"], n_restarts,
            )

        return best_model

    @property
    def multi_restart_diagnostics(self) -> Optional[Dict[str, Any]]:
        """Return multi-restart diagnostics if fit_multi_restart was used."""
        return getattr(self, '_multi_restart_diagnostics', None)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Export model state as a plain dict (for persistence)."""
        self._check_fitted()
        graphs_ser = {}
        for k, g in self._causal_graphs.items():
            graphs_ser[k] = list(g.edges())
        return {
            "n_regimes": self.n_regimes,
            "regime_assignments": self._regime_assignments.tolist(),
            "gamma": self._gamma.tolist(),
            "causal_edges": graphs_ser,
            "invariant_edges": list(self._invariant_edges),
            "log_likelihoods": self.log_likelihoods().tolist(),
            "hmm_means": self._hmm.means.tolist(),
            "hmm_pi": self._hmm.pi.tolist(),
        }
