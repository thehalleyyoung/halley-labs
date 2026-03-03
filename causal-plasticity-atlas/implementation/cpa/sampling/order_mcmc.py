"""Order MCMC sampler (Friedman & Koller, 2003).

Samples over linear orderings of nodes and, for each ordering,
finds the optimal DAG consistent with that order.  Proposals swap
adjacent elements in the current permutation.

The key idea is that the DAG space is searched indirectly by sampling
linear orderings (permutations) of nodes.  For each ordering, the
optimal parent set for every node is chosen among its predecessors in
the ordering, yielding the highest-scoring DAG consistent with that
ordering.  Proposals swap two adjacent elements, yielding a symmetric
random walk on the space of permutations.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass
class OrderSample:
    """A single sample from the order MCMC chain.

    Attributes
    ----------
    order : List[int]
        Node permutation defining a topological ordering.
    score : float
        Log-score of the optimal DAG consistent with this order.
    dag : Optional[NDArray]
        Adjacency matrix of the optimal DAG (if retained).
    """

    order: List[int]
    score: float
    dag: Optional[NDArray] = None


@dataclass
class DAGPosteriorSamples:
    """Aggregated results from an Order MCMC run.

    Attributes
    ----------
    dags : List[NDArray]
        Sampled adjacency matrices.
    scores : List[float]
        Log-scores of each sampled DAG.
    orders : List[List[int]]
        Sampled orderings.
    acceptance_rate : float
        Fraction of proposals accepted.
    burn_in : int
        Number of burn-in iterations discarded.
    """

    dags: List[NDArray] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    orders: List[List[int]] = field(default_factory=list)
    acceptance_rate: float = 0.0
    burn_in: int = 0


# -------------------------------------------------------------------
# Score cache
# -------------------------------------------------------------------

class _ScoreCache:
    """Cache for local score evaluations to avoid recomputation."""

    def __init__(
        self,
        score_fn: Callable[[int, Sequence[int]], float],
    ) -> None:
        self._score_fn = score_fn
        self._cache: Dict[Tuple[int, FrozenSet[int]], float] = {}

    def local_score(self, node: int, parents: Sequence[int]) -> float:
        key = (node, frozenset(parents))
        if key not in self._cache:
            self._cache[key] = self._score_fn(node, list(parents))
        return self._cache[key]


# -------------------------------------------------------------------
# Helper: enumerate parent subsets
# -------------------------------------------------------------------

def _subsets(items: List[int], max_size: int) -> List[Tuple[int, ...]]:
    """Return all subsets of *items* with size <= *max_size*."""
    result: List[Tuple[int, ...]] = [()]
    for k in range(1, min(max_size, len(items)) + 1):
        result.extend(itertools.combinations(items, k))
    return result


# -------------------------------------------------------------------
# OrderMCMC
# -------------------------------------------------------------------

class OrderMCMC:
    """Order MCMC sampler over DAG space.

    Explores the posterior over DAGs by sampling linear orderings
    of nodes using the Metropolis-Hastings algorithm.  For each
    sampled ordering, the optimal DAG consistent with that ordering
    is computed by choosing the best parent set for each node from
    its predecessors in the ordering.

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Local score function with signature ``(node, parents) -> float``.
        Must be decomposable (total DAG score = sum of local scores).
    n_nodes : int
        Number of variables in the model.
    max_parents : Optional[int]
        Upper bound on the parent-set size.  Defaults to
        ``min(n_nodes - 1, 5)``.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        score_fn: Callable[[int, Sequence[int]], float],
        n_nodes: int,
        max_parents: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.score_fn = score_fn
        self.n_nodes = n_nodes
        self.max_parents = max_parents if max_parents is not None else min(n_nodes - 1, 5)
        self._rng = np.random.default_rng(seed)
        self._cache = _ScoreCache(score_fn)
        # Per-node best-parent cache keyed on (node, frozenset(candidates))
        self._best_parent_cache: Dict[Tuple[int, FrozenSet[int]], Tuple[float, List[int]]] = {}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(
        self,
        n_iterations: int,
        burnin: int = 0,
        thin: int = 1,
    ) -> List[OrderSample]:
        """Run the MCMC chain and return collected samples.

        Parameters
        ----------
        n_iterations : int
            Total number of MCMC iterations (including burn-in).
        burnin : int
            Number of initial iterations to discard.
        thin : int
            Keep every *thin*-th sample after burn-in.

        Returns
        -------
        List[OrderSample]
            Collected order samples after burn-in and thinning.
        """
        current_order = list(self._rng.permutation(self.n_nodes))
        current_score = self._compute_order_score(current_order)

        samples: List[OrderSample] = []
        accepted = 0
        total_proposals = 0

        for it in range(n_iterations):
            proposed_order = self.propose_order(current_order)
            proposed_score = self._compute_order_score(proposed_order)
            total_proposals += 1

            alpha = self.acceptance_ratio(current_score, proposed_score)
            if self._rng.random() < alpha:
                current_order = proposed_order
                current_score = proposed_score
                accepted += 1

            if it >= burnin and (it - burnin) % thin == 0:
                dag = self.sample_dag_from_order(current_order)
                samples.append(
                    OrderSample(
                        order=list(current_order),
                        score=current_score,
                        dag=dag,
                    )
                )

        # Store diagnostics on last sample (or return separately)
        self._last_acceptance_rate = accepted / max(total_proposals, 1)
        return samples

    def run_full(
        self,
        n_iterations: int = 10000,
        burn_in: int = 1000,
        thin: int = 1,
        seed: Optional[int] = None,
    ) -> DAGPosteriorSamples:
        """Run Order MCMC and return aggregated posterior samples.

        Parameters
        ----------
        n_iterations : int
            Total MCMC iterations.
        burn_in : int
            Burn-in iterations to discard.
        thin : int
            Thinning interval.
        seed : int, optional
            If provided, reseeds the RNG.

        Returns
        -------
        DAGPosteriorSamples
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        samples = self.run(n_iterations, burnin=burn_in, thin=thin)

        result = DAGPosteriorSamples(
            dags=[s.dag for s in samples if s.dag is not None],
            scores=[s.score for s in samples],
            orders=[s.order for s in samples],
            acceptance_rate=getattr(self, "_last_acceptance_rate", 0.0),
            burn_in=burn_in,
        )
        return result

    def propose_order(self, current_order: List[int]) -> List[int]:
        """Propose a new ordering by swapping two adjacent elements.

        Parameters
        ----------
        current_order : List[int]
            Current node permutation.

        Returns
        -------
        List[int]
            New permutation with two adjacent elements swapped.
        """
        new_order = list(current_order)
        n = len(new_order)
        if n < 2:
            return new_order
        idx = int(self._rng.integers(0, n - 1))
        new_order[idx], new_order[idx + 1] = new_order[idx + 1], new_order[idx]
        return new_order

    def sample_dag_from_order(self, order: List[int]) -> NDArray:
        """Return the optimal DAG consistent with *order*.

        For each node, selects the parent set among its predecessors
        in the ordering that maximises the local score.

        Parameters
        ----------
        order : List[int]
            A permutation of node indices.

        Returns
        -------
        NDArray
            Binary adjacency matrix of shape ``(n_nodes, n_nodes)``.
        """
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for pos, node in enumerate(order):
            candidates = list(order[:pos])
            _, best_parents = self._best_parents(node, candidates)
            for p in best_parents:
                adj[p, node] = 1.0
        return adj

    def acceptance_ratio(
        self, current_score: float, proposed_score: float
    ) -> float:
        """Return the Metropolis-Hastings acceptance probability.

        For symmetric proposals on orderings this is simply
        ``min(1, exp(proposed_score - current_score))``.

        Parameters
        ----------
        current_score : float
            Log-score of the current state.
        proposed_score : float
            Log-score of the proposed state.

        Returns
        -------
        float
            Acceptance probability in [0, 1].
        """
        log_ratio = proposed_score - current_score
        if log_ratio >= 0:
            return 1.0
        return math.exp(log_ratio)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _compute_order_score(self, order: List[int]) -> float:
        """Compute score of the best DAG consistent with *order*.

        This is the sum of the best local scores for each node given
        its predecessors in the ordering.

        Parameters
        ----------
        order : List[int]
            A permutation of node indices.

        Returns
        -------
        float
            Total log-score.
        """
        total = 0.0
        for pos, node in enumerate(order):
            candidates = list(order[:pos])
            best_score, _ = self._best_parents(node, candidates)
            total += best_score
        return total

    def _best_parents(
        self, node: int, candidates: List[int]
    ) -> Tuple[float, List[int]]:
        """Find the best parent set for *node* among *candidates*.

        Enumerates all subsets of *candidates* up to size
        ``max_parents`` and returns the one with the highest local
        score.

        Parameters
        ----------
        node : int
            Target node.
        candidates : List[int]
            Potential parent nodes (predecessors in ordering).

        Returns
        -------
        Tuple[float, List[int]]
            ``(best_score, best_parent_set)``.
        """
        cache_key = (node, frozenset(candidates))
        if cache_key in self._best_parent_cache:
            return self._best_parent_cache[cache_key]

        best_score = -math.inf
        best_parents: List[int] = []

        # If candidates are too many for exhaustive search, limit to
        # a heuristic forward selection approach
        if len(candidates) > 15:
            best_score, best_parents = self._greedy_parents(node, candidates)
        else:
            for subset in _subsets(candidates, self.max_parents):
                s = self._cache.local_score(node, list(subset))
                if s > best_score:
                    best_score = s
                    best_parents = list(subset)

        self._best_parent_cache[cache_key] = (best_score, best_parents)
        return best_score, best_parents

    def _greedy_parents(
        self, node: int, candidates: List[int]
    ) -> Tuple[float, List[int]]:
        """Greedy forward selection of parents when exhaustive search is too expensive.

        Iteratively adds the candidate that most improves the local
        score, stopping when no improvement is possible or the maximum
        parent set size is reached.

        Parameters
        ----------
        node : int
            Target node.
        candidates : List[int]
            Potential parent nodes.

        Returns
        -------
        Tuple[float, List[int]]
            ``(best_score, best_parent_set)``.
        """
        current_parents: List[int] = []
        current_score = self._cache.local_score(node, [])
        remaining = set(candidates)

        for _ in range(self.max_parents):
            best_add_score = current_score
            best_add_node: Optional[int] = None

            for c in remaining:
                trial = current_parents + [c]
                s = self._cache.local_score(node, trial)
                if s > best_add_score:
                    best_add_score = s
                    best_add_node = c

            if best_add_node is None:
                break
            current_parents.append(best_add_node)
            remaining.discard(best_add_node)
            current_score = best_add_score

        return current_score, current_parents

    # -----------------------------------------------------------------
    # Convergence diagnostics
    # -----------------------------------------------------------------

    @staticmethod
    def effective_sample_size(scores: List[float]) -> float:
        """Estimate effective sample size from autocorrelation of scores.

        Uses the initial positive sequence estimator.

        Parameters
        ----------
        scores : List[float]
            Sequence of log-scores from the chain.

        Returns
        -------
        float
            Estimated effective sample size.
        """
        n = len(scores)
        if n < 4:
            return float(n)
        x = np.asarray(scores, dtype=np.float64)
        x = x - x.mean()
        var = np.var(x, ddof=0)
        if var < 1e-300:
            return float(n)

        # Compute autocorrelation using FFT
        fft_x = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)

        # Initial positive sequence estimator (Geyer 1992)
        running_sum = 0.0
        for lag in range(1, n, 2):
            pair = acf[lag] + (acf[lag + 1] if lag + 1 < n else 0.0)
            if pair < 0:
                break
            running_sum += pair

        tau = 1.0 + 2.0 * running_sum
        return n / max(tau, 1.0)

    @staticmethod
    def gelman_rubin(chains: List[List[float]]) -> float:
        """Compute the Gelman-Rubin R-hat diagnostic for multiple chains.

        Parameters
        ----------
        chains : List[List[float]]
            List of score sequences, one per chain.

        Returns
        -------
        float
            R-hat statistic.  Values close to 1.0 indicate convergence.
        """
        m = len(chains)
        if m < 2:
            return float("nan")
        ns = [len(c) for c in chains]
        n = min(ns)
        if n < 2:
            return float("nan")

        chain_means = np.array([np.mean(c[:n]) for c in chains])
        chain_vars = np.array([np.var(c[:n], ddof=1) for c in chains])

        grand_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)

        if W < 1e-300:
            return 1.0

        var_est = (1 - 1 / n) * W + (1 / n) * B
        r_hat = math.sqrt(var_est / W)
        return r_hat

    @staticmethod
    def trace_plot_data(samples: List[OrderSample]) -> Tuple[NDArray, NDArray]:
        """Extract iteration indices and scores for a trace plot.

        Parameters
        ----------
        samples : List[OrderSample]
            Collected MCMC samples.

        Returns
        -------
        Tuple[NDArray, NDArray]
            ``(iterations, scores)`` arrays.
        """
        iterations = np.arange(len(samples), dtype=np.float64)
        scores = np.array([s.score for s in samples], dtype=np.float64)
        return iterations, scores

    @staticmethod
    def edge_posterior_probabilities(
        samples: List[OrderSample],
    ) -> NDArray:
        """Compute posterior edge inclusion probabilities.

        Parameters
        ----------
        samples : List[OrderSample]
            Collected MCMC samples with DAGs.

        Returns
        -------
        NDArray
            Matrix of shape ``(n_nodes, n_nodes)`` with empirical
            edge inclusion frequencies.
        """
        dags = [s.dag for s in samples if s.dag is not None]
        if not dags:
            return np.empty((0, 0))
        n = dags[0].shape[0]
        prob = np.zeros((n, n), dtype=np.float64)
        for dag in dags:
            prob += (dag != 0).astype(np.float64)
        prob /= len(dags)
        return prob

    @staticmethod
    def map_dag(samples: List[OrderSample]) -> Tuple[NDArray, float]:
        """Return the maximum a posteriori DAG from samples.

        Parameters
        ----------
        samples : List[OrderSample]
            Collected MCMC samples with DAGs.

        Returns
        -------
        Tuple[NDArray, float]
            ``(dag, score)`` of the highest-scoring sampled DAG.
        """
        best_idx = max(range(len(samples)), key=lambda i: samples[i].score)
        best = samples[best_idx]
        return best.dag.copy() if best.dag is not None else np.empty((0, 0)), best.score
