"""
Joint posterior over regimes and causal structures.

Computes P(regimes, DAG | data), marginal posteriors, MAP estimates, and
supports MCMC sampling over the joint space for Bayesian model comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import special, stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PosteriorSample:
    """A single sample from the joint posterior."""
    regime_assignments: np.ndarray
    causal_graph: Dict[int, nx.DiGraph]
    log_joint: float
    log_likelihood: float
    log_prior: float


@dataclass
class PosteriorSummary:
    """Summary statistics of the posterior distribution."""
    map_regimes: np.ndarray
    map_graphs: Dict[int, nx.DiGraph]
    map_log_joint: float
    marginal_regime_probs: np.ndarray  # (T, K) marginal P(z_t = k | data)
    edge_inclusion_probs: Dict[int, np.ndarray]  # per-regime (D, D) matrix
    n_samples: int
    effective_sample_size: float
    acceptance_rate: float


@dataclass
class ModelComparisonResult:
    """Result of comparing different (K, DAG) configurations."""
    model_index: int
    n_regimes: int
    log_marginal_likelihood: float
    bic: float
    aic: float
    n_params: int
    description: str = ""


# ---------------------------------------------------------------------------
# DAG prior helpers
# ---------------------------------------------------------------------------

def _log_dag_prior(dag: nx.DiGraph, n_vars: int, edge_penalty: float = 1.0) -> float:
    """Log prior over DAGs: penalise number of edges.

    Uses a sparsity-inducing prior: P(G) ∝ exp(-edge_penalty * |E|).
    Also returns -inf for cyclic graphs.
    """
    if not nx.is_directed_acyclic_graph(dag):
        return -np.inf
    n_edges = dag.number_of_edges()
    return -edge_penalty * n_edges


def _log_regime_prior(
    assignments: np.ndarray,
    K: int,
    alpha: float = 1.0,
    kappa: float = 10.0,
) -> float:
    """Log prior over regime assignments using a Sticky-HMM-like prior.

    Favours temporal smoothness: sequential same-regime transitions are
    rewarded by kappa.
    """
    T = len(assignments)
    if T == 0:
        return 0.0

    # Dirichlet-Multinomial for state proportions
    counts = np.bincount(assignments, minlength=K).astype(float)
    log_p = float(special.gammaln(alpha * K) - special.gammaln(T + alpha * K))
    for k in range(K):
        log_p += float(special.gammaln(counts[k] + alpha) - special.gammaln(alpha))

    # Stickiness bonus
    n_same = int(np.sum(assignments[1:] == assignments[:-1]))
    log_p += kappa * n_same / T

    return log_p


# ---------------------------------------------------------------------------
# Likelihood computation
# ---------------------------------------------------------------------------

def _log_likelihood_gaussian(
    data: np.ndarray,
    assignments: np.ndarray,
    K: int,
) -> float:
    """Compute log-likelihood under regime-specific Gaussians (MLE params)."""
    T, D = data.shape
    ll = 0.0
    for k in range(K):
        mask = assignments == k
        n_k = mask.sum()
        if n_k < D + 1:
            ll += -1e6  # heavy penalty for too-few-sample regimes
            continue
        data_k = data[mask]
        mu = data_k.mean(axis=0)
        cov = np.cov(data_k, rowvar=False) + np.eye(D) * 1e-6
        try:
            ll += float(np.sum(stats.multivariate_normal.logpdf(data_k, mean=mu, cov=cov)))
        except np.linalg.LinAlgError:
            ll += -1e6
    return ll


def _log_likelihood_dag(
    data: np.ndarray,
    assignments: np.ndarray,
    graphs: Dict[int, nx.DiGraph],
    K: int,
) -> float:
    """Compute log-likelihood incorporating DAG structure (ANM residual model)."""
    T, D = data.shape
    ll = 0.0
    for k in range(K):
        mask = assignments == k
        n_k = mask.sum()
        if n_k < D + 1:
            ll += -1e6
            continue
        data_k = data[mask]
        dag = graphs.get(k)
        if dag is None:
            # Fall back to Gaussian
            mu = data_k.mean(axis=0)
            cov = np.cov(data_k, rowvar=False) + np.eye(D) * 1e-6
            try:
                ll += float(np.sum(stats.multivariate_normal.logpdf(data_k, mean=mu, cov=cov)))
            except np.linalg.LinAlgError:
                ll += -1e6
            continue

        for j in range(D):
            parents = list(dag.predecessors(j))
            y = data_k[:, j]
            if parents:
                X = np.column_stack([np.ones(n_k), data_k[:, parents]])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X @ beta
                except np.linalg.LinAlgError:
                    resid = y - y.mean()
            else:
                resid = y - y.mean()

            var = np.var(resid) + 1e-10
            ll += float(-0.5 * n_k * np.log(2 * np.pi * var) - 0.5 * np.sum(resid ** 2) / var)

    return ll


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class JointPosterior:
    """Joint posterior P(regimes, DAG | data) with MCMC sampling.

    Parameters
    ----------
    n_regimes : int
        Number of regimes (K).
    edge_penalty : float
        Prior penalty per DAG edge (sparsity).
    alpha : float
        Dirichlet concentration for regime proportions.
    kappa : float
        Stickiness bonus for regime smoothness.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        edge_penalty: float = 1.0,
        alpha: float = 1.0,
        kappa: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.edge_penalty = edge_penalty
        self.alpha = alpha
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

        self._samples: List[PosteriorSample] = []
        self._summary: Optional[PosteriorSummary] = None
        self._data: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Joint probability
    # ------------------------------------------------------------------

    def log_joint(
        self,
        data: np.ndarray,
        assignments: np.ndarray,
        graphs: Dict[int, nx.DiGraph],
    ) -> Tuple[float, float, float]:
        """Compute log P(regimes, DAG, data) = log P(data | regimes, DAG) + log P(regimes) + log P(DAG).

        Returns (log_joint, log_likelihood, log_prior).
        """
        D = data.shape[1]
        K = self.n_regimes

        log_lik = _log_likelihood_dag(data, assignments, graphs, K)
        log_pr_regime = _log_regime_prior(assignments, K, self.alpha, self.kappa)
        log_pr_dag = sum(
            _log_dag_prior(g, D, self.edge_penalty) for g in graphs.values()
        )

        log_prior = log_pr_regime + log_pr_dag
        return log_lik + log_prior, log_lik, log_prior

    # ------------------------------------------------------------------
    # MAP estimation
    # ------------------------------------------------------------------

    def map_estimate(
        self,
        data: np.ndarray,
        init_assignments: np.ndarray,
        init_graphs: Dict[int, nx.DiGraph],
        n_iter: int = 100,
    ) -> Tuple[np.ndarray, Dict[int, nx.DiGraph], float]:
        """Find MAP estimate via greedy local search.

        Alternates between:
          1. For each time step, reassign to the regime maximising log-joint.
          2. For each regime graph, greedily add/remove edges.

        Returns (assignments, graphs, log_joint).
        """
        data = np.asarray(data, dtype=np.float64)
        T, D = data.shape
        K = self.n_regimes

        assignments = init_assignments.copy()
        graphs = {k: g.copy() for k, g in init_graphs.items()}
        best_lj, _, _ = self.log_joint(data, assignments, graphs)

        for it in range(n_iter):
            improved = False

            # --- Regime reassignment sweeps ---
            order = self.rng.permutation(T)
            for t in order:
                current_k = assignments[t]
                best_k = current_k
                best_local = best_lj

                for k in range(K):
                    if k == current_k:
                        continue
                    assignments[t] = k
                    lj, _, _ = self.log_joint(data, assignments, graphs)
                    if lj > best_local:
                        best_local = lj
                        best_k = k

                assignments[t] = best_k
                if best_k != current_k:
                    best_lj = best_local
                    improved = True

            # --- Graph edge edits ---
            for k in range(K):
                dag = graphs[k]
                nodes = list(dag.nodes())
                for u in nodes:
                    for v in nodes:
                        if u == v:
                            continue
                        has_edge = dag.has_edge(u, v)
                        if has_edge:
                            dag.remove_edge(u, v)
                        else:
                            dag.add_edge(u, v)

                        if nx.is_directed_acyclic_graph(dag):
                            lj, _, _ = self.log_joint(data, assignments, graphs)
                            if lj > best_lj:
                                best_lj = lj
                                improved = True
                            else:
                                # Revert
                                if has_edge:
                                    dag.add_edge(u, v)
                                else:
                                    dag.remove_edge(u, v)
                        else:
                            # Revert (cycle introduced)
                            if has_edge:
                                dag.add_edge(u, v)
                            else:
                                dag.remove_edge(u, v)

            if not improved:
                break

        return assignments, graphs, best_lj

    # ------------------------------------------------------------------
    # MCMC sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        data: np.ndarray,
        init_assignments: np.ndarray,
        init_graphs: Dict[int, nx.DiGraph],
        n_samples: int = 500,
        burn_in: int = 100,
        thin: int = 5,
    ) -> List[PosteriorSample]:
        """Metropolis-Hastings MCMC over the joint (regimes, DAG) space.

        Proposal moves:
          - Single time-step regime reassignment
          - Single edge addition/removal in a regime graph

        Parameters
        ----------
        data : np.ndarray, shape (T, D)
        init_assignments : starting regime labels
        init_graphs : starting DAGs per regime
        n_samples : number of post-burn-in samples to collect
        burn_in : number of initial samples to discard
        thin : keep every `thin`-th sample

        Returns
        -------
        List[PosteriorSample]
        """
        data = np.asarray(data, dtype=np.float64)
        self._data = data
        T, D = data.shape
        K = self.n_regimes

        assignments = init_assignments.copy()
        graphs = {k: g.copy() for k, g in init_graphs.items()}
        curr_lj, curr_ll, curr_lp = self.log_joint(data, assignments, graphs)

        samples: List[PosteriorSample] = []
        total_steps = burn_in + n_samples * thin
        n_accepted = 0

        for step in range(total_steps):
            # Choose move type
            if self.rng.random() < 0.6:
                # Regime reassignment proposal
                t = self.rng.integers(0, T)
                old_k = assignments[t]
                new_k = self.rng.integers(0, K)
                if new_k == old_k:
                    continue
                assignments[t] = new_k
                prop_lj, prop_ll, prop_lp = self.log_joint(data, assignments, graphs)
                log_alpha = prop_lj - curr_lj

                if np.log(self.rng.random() + 1e-300) < log_alpha:
                    curr_lj, curr_ll, curr_lp = prop_lj, prop_ll, prop_lp
                    n_accepted += 1
                else:
                    assignments[t] = old_k
            else:
                # Edge edit proposal
                k = self.rng.integers(0, K)
                dag = graphs[k]
                u, v = int(self.rng.integers(0, D)), int(self.rng.integers(0, D))
                if u == v:
                    continue
                had_edge = dag.has_edge(u, v)
                if had_edge:
                    dag.remove_edge(u, v)
                else:
                    dag.add_edge(u, v)

                if not nx.is_directed_acyclic_graph(dag):
                    if had_edge:
                        dag.add_edge(u, v)
                    else:
                        dag.remove_edge(u, v)
                    continue

                prop_lj, prop_ll, prop_lp = self.log_joint(data, assignments, graphs)
                log_alpha = prop_lj - curr_lj

                if np.log(self.rng.random() + 1e-300) < log_alpha:
                    curr_lj, curr_ll, curr_lp = prop_lj, prop_ll, prop_lp
                    n_accepted += 1
                else:
                    if had_edge:
                        dag.add_edge(u, v)
                    else:
                        dag.remove_edge(u, v)

            # Collect sample
            if step >= burn_in and (step - burn_in) % thin == 0:
                s = PosteriorSample(
                    regime_assignments=assignments.copy(),
                    causal_graph={kk: gg.copy() for kk, gg in graphs.items()},
                    log_joint=curr_lj,
                    log_likelihood=curr_ll,
                    log_prior=curr_lp,
                )
                samples.append(s)

        self._samples = samples
        acceptance_rate = n_accepted / max(total_steps, 1)
        self._build_summary(data, samples, acceptance_rate)
        return samples

    # ------------------------------------------------------------------
    # Marginals
    # ------------------------------------------------------------------

    def marginal_regime_posterior(self) -> np.ndarray:
        """Compute marginal P(z_t = k | data) from MCMC samples.

        Returns array of shape (T, K).
        """
        if not self._samples:
            raise RuntimeError("No samples available. Call sample() first.")

        T = self._samples[0].regime_assignments.shape[0]
        K = self.n_regimes
        counts = np.zeros((T, K))
        for s in self._samples:
            for t in range(T):
                counts[t, s.regime_assignments[t]] += 1
        return counts / counts.sum(axis=1, keepdims=True)

    def marginal_edge_posterior(self) -> Dict[int, np.ndarray]:
        """Compute marginal P(edge (i,j) ∈ G_k | data) for each regime.

        Returns dict mapping regime index to (D, D) probability matrix.
        """
        if not self._samples:
            raise RuntimeError("No samples available. Call sample() first.")

        D = self._data.shape[1] if self._data is not None else 0
        if D == 0:
            raise RuntimeError("No data stored – call sample() with data first.")

        edge_probs: Dict[int, np.ndarray] = {}
        for k in range(self.n_regimes):
            mat = np.zeros((D, D))
            for s in self._samples:
                g = s.causal_graph.get(k)
                if g is None:
                    continue
                for u, v in g.edges():
                    mat[u, v] += 1
            mat /= len(self._samples)
            edge_probs[k] = mat
        return edge_probs

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        data: np.ndarray,
        model_specs: List[Tuple[int, Dict[int, nx.DiGraph], np.ndarray]],
    ) -> List[ModelComparisonResult]:
        """Compare candidate (K, DAG, assignments) triples.

        Each entry in model_specs is (n_regimes, graphs, assignments).

        Returns a list of ModelComparisonResult sorted by BIC (lower is better).
        """
        data = np.asarray(data, dtype=np.float64)
        T, D = data.shape
        results: List[ModelComparisonResult] = []

        for idx, (K, graphs, assignments) in enumerate(model_specs):
            self_tmp = JointPosterior(n_regimes=K, edge_penalty=self.edge_penalty,
                                      alpha=self.alpha, kappa=self.kappa)
            lj, ll, lp = self_tmp.log_joint(data, assignments, graphs)

            # Count parameters
            n_params = self._count_params(K, D, graphs)
            bic = -2 * ll + n_params * np.log(T)
            aic = -2 * ll + 2 * n_params

            # Laplace approximation for marginal likelihood
            log_ml = ll + lp - 0.5 * n_params * np.log(T / (2 * np.pi))

            results.append(ModelComparisonResult(
                model_index=idx,
                n_regimes=K,
                log_marginal_likelihood=log_ml,
                bic=bic,
                aic=aic,
                n_params=n_params,
                description=f"K={K}, edges={sum(g.number_of_edges() for g in graphs.values())}",
            ))

        results.sort(key=lambda r: r.bic)
        return results

    @staticmethod
    def _count_params(K: int, D: int, graphs: Dict[int, nx.DiGraph]) -> int:
        """Count free parameters in the model."""
        # Transition matrix: K*(K-1)
        n_trans = K * (K - 1)
        # Emission params per regime: mean (D) + covariance (D*(D+1)/2)
        n_emit = K * (D + D * (D + 1) // 2)
        # DAG: each edge adds 1 regression coefficient
        n_dag = sum(g.number_of_edges() for g in graphs.values())
        return n_trans + n_emit + n_dag

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        data: np.ndarray,
        samples: List[PosteriorSample],
        acceptance_rate: float,
    ) -> None:
        """Build posterior summary from MCMC samples."""
        if not samples:
            return

        T = samples[0].regime_assignments.shape[0]
        K = self.n_regimes
        D = data.shape[1]

        # MAP
        map_idx = int(np.argmax([s.log_joint for s in samples]))
        map_sample = samples[map_idx]

        # Marginal regime
        regime_counts = np.zeros((T, K))
        for s in samples:
            for t in range(T):
                regime_counts[t, s.regime_assignments[t]] += 1
        marginal_regimes = regime_counts / regime_counts.sum(axis=1, keepdims=True)

        # Edge inclusion
        edge_probs: Dict[int, np.ndarray] = {}
        for k in range(K):
            mat = np.zeros((D, D))
            for s in samples:
                g = s.causal_graph.get(k)
                if g is None:
                    continue
                for u, v in g.edges():
                    mat[u, v] += 1
            mat /= len(samples)
            edge_probs[k] = mat

        # ESS via autocorrelation of log-joint
        lj_trace = np.array([s.log_joint for s in samples])
        ess = self._effective_sample_size(lj_trace)

        self._summary = PosteriorSummary(
            map_regimes=map_sample.regime_assignments.copy(),
            map_graphs={k: g.copy() for k, g in map_sample.causal_graph.items()},
            map_log_joint=map_sample.log_joint,
            marginal_regime_probs=marginal_regimes,
            edge_inclusion_probs=edge_probs,
            n_samples=len(samples),
            effective_sample_size=ess,
            acceptance_rate=acceptance_rate,
        )

    def get_summary(self) -> PosteriorSummary:
        """Return the most recently computed posterior summary."""
        if self._summary is None:
            raise RuntimeError("No summary available. Call sample() first.")
        return self._summary

    @staticmethod
    def _effective_sample_size(trace: np.ndarray) -> float:
        """Estimate ESS from a 1-D trace using autocorrelation."""
        n = len(trace)
        if n < 4:
            return float(n)

        centered = trace - trace.mean()
        var = np.var(centered)
        if var < 1e-15:
            return float(n)

        acf = np.correlate(centered, centered, mode="full")[n - 1:]
        acf = acf / (var * n)

        # Sum autocorrelations until first negative pair
        tau = 1.0
        for lag in range(1, n // 2):
            if acf[lag] < 0:
                break
            tau += 2 * acf[lag]

        return n / tau

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_samples(self) -> List[PosteriorSample]:
        """Return stored MCMC samples."""
        return list(self._samples)

    def log_joint_trace(self) -> np.ndarray:
        """Return the log-joint values across stored samples."""
        return np.array([s.log_joint for s in self._samples])

    def acceptance_rate(self) -> float:
        """Return MCMC acceptance rate."""
        if self._summary is None:
            return 0.0
        return self._summary.acceptance_rate
