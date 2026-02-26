"""
Score-based causal structure learning.

Provides BIC, BDeu, and BGe scoring functions for DAGs, greedy hill-climbing
search over the DAG space (with edge additions, deletions, and reversals),
score equivalence utilities, penalised likelihood scoring, and a simple
Bayesian model averaging scheme over DAGs.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Sequence, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import gammaln


# ====================================================================
# Abstract score
# ====================================================================

class DAGScore(ABC):
    """Abstract interface for a decomposable DAG score."""

    @abstractmethod
    def local_score(
        self,
        node: int,
        parents: Sequence[int],
        data: NDArray,
    ) -> float:
        """Score a single node given its parent set."""

    def score(self, dag: nx.DiGraph, data: NDArray) -> float:
        """Total decomposable score: sum of local scores."""
        total = 0.0
        for node in dag.nodes:
            parents = sorted(dag.predecessors(node))
            total += self.local_score(node, parents, data)
        return total


# ====================================================================
# BIC score
# ====================================================================

class BICScore(DAGScore):
    """Bayesian Information Criterion for linear-Gaussian models.

    local_score(j, Pa_j) = -n/2 * log(σ²_j|Pa_j) - |Pa_j|/2 * log(n)

    Parameters
    ----------
    penalty_discount : float
        Multiplier on the penalty term (default 1.0 = standard BIC).
    """

    def __init__(self, penalty_discount: float = 1.0) -> None:
        self.penalty_discount = penalty_discount

    def local_score(
        self,
        node: int,
        parents: Sequence[int],
        data: NDArray,
    ) -> float:
        n = data.shape[0]
        y = data[:, node]

        if len(parents) == 0:
            residual_var = np.var(y, ddof=1)
        else:
            X = data[:, list(parents)]
            X = np.column_stack([X, np.ones(n)])
            beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
            residual_var = np.var(residuals, ddof=1)

        residual_var = max(residual_var, 1e-12)
        k = len(parents) + 1  # number of parameters (coefficients + variance)
        ll = -0.5 * n * np.log(2 * np.pi * residual_var) - 0.5 * n
        return float(ll - 0.5 * self.penalty_discount * k * np.log(n))


# ====================================================================
# BDeu score
# ====================================================================

class BDeuScore(DAGScore):
    """Bayesian Dirichlet equivalent uniform (BDeu) score for discrete data.

    Parameters
    ----------
    equivalent_sample_size : float
        The equivalent sample size (α) for the Dirichlet prior.
    n_categories : int or dict
        Number of categories per variable (uniform or per-variable).
    """

    def __init__(
        self,
        equivalent_sample_size: float = 10.0,
        n_categories: int | Dict[int, int] = 2,
    ) -> None:
        self.ess = equivalent_sample_size
        self._n_categories = n_categories

    def _get_n_categories(self, node: int) -> int:
        if isinstance(self._n_categories, dict):
            return self._n_categories.get(node, 2)
        return self._n_categories

    def local_score(
        self,
        node: int,
        parents: Sequence[int],
        data: NDArray,
    ) -> float:
        n = data.shape[0]
        r_j = self._get_n_categories(node)

        if len(parents) == 0:
            # No parents: single multinomial
            counts = np.bincount(data[:, node].astype(int), minlength=r_j)
            alpha_ij = self.ess / r_j
            score = gammaln(self.ess) - gammaln(self.ess + n)
            for k in range(r_j):
                score += gammaln(alpha_ij + counts[k]) - gammaln(alpha_ij)
            return float(score)

        # Count joint configurations of parents
        parent_list = list(parents)
        parent_cards = [self._get_n_categories(p) for p in parent_list]
        q_j = int(np.prod(parent_cards))

        # Encode parent configurations as integers
        parent_data = data[:, parent_list].astype(int)
        multipliers = np.ones(len(parent_list), dtype=int)
        for idx in range(len(parent_list) - 2, -1, -1):
            multipliers[idx] = multipliers[idx + 1] * parent_cards[idx + 1]
        config = parent_data @ multipliers

        alpha_ijk = self.ess / (r_j * q_j)
        alpha_ij = self.ess / q_j

        score = 0.0
        for qi in range(q_j):
            mask = config == qi
            n_ij = np.sum(mask)
            score += gammaln(alpha_ij) - gammaln(alpha_ij + n_ij)
            if n_ij == 0:
                continue
            child_vals = data[mask, node].astype(int)
            counts = np.bincount(child_vals, minlength=r_j)
            for k in range(r_j):
                score += gammaln(alpha_ijk + counts[k]) - gammaln(alpha_ijk)

        return float(score)


# ====================================================================
# BGe score
# ====================================================================

class BGeScore(DAGScore):
    """Bayesian Gaussian equivalent (BGe) score (Geiger & Heckerman 2002).

    Parameters
    ----------
    alpha_mu : float
        Prior precision for the mean (ν in some notations).
    alpha_w : int
        Degrees of freedom of the Wishart prior (must be ≥ p).
    prior_scale : float
        Scale multiplier for the prior scatter matrix T = prior_scale * I.
    """

    def __init__(
        self,
        alpha_mu: float = 1.0,
        alpha_w: Optional[int] = None,
        prior_scale: float = 1.0,
    ) -> None:
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w  # defaults to p + 2 if None
        self.prior_scale = prior_scale

    def local_score(
        self,
        node: int,
        parents: Sequence[int],
        data: NDArray,
    ) -> float:
        n, p = data.shape
        alpha_w = self.alpha_w if self.alpha_w is not None else p + 2
        variables = list(parents) + [node]
        d = len(variables)

        X = data[:, variables]
        mu_0 = np.zeros(d)
        T_0 = self.prior_scale * np.eye(d)

        S = X - X.mean(axis=0)
        S_n = S.T @ S

        alpha_n = self.alpha_mu + n
        T_n = T_0 + S_n + (self.alpha_mu * n / alpha_n) * np.outer(
            X.mean(axis=0) - mu_0, X.mean(axis=0) - mu_0
        )

        # Log marginal likelihood (up to constants per variable set)
        score = 0.0
        score += -0.5 * n * d * np.log(np.pi)
        score += 0.5 * d * np.log(self.alpha_mu / alpha_n)
        for j in range(d):
            score += gammaln(0.5 * (alpha_w + n - j)) - gammaln(
                0.5 * (alpha_w - j)
            )
        score += 0.5 * alpha_w * np.linalg.slogdet(T_0)[1]
        score -= 0.5 * (alpha_w + n) * np.linalg.slogdet(T_n)[1]

        # For decomposability, subtract the parents-only score
        if len(parents) > 0:
            d_p = len(parents)
            X_p = data[:, list(parents)]
            mu_0p = np.zeros(d_p)
            T_0p = self.prior_scale * np.eye(d_p)
            S_p = X_p - X_p.mean(axis=0)
            S_np = S_p.T @ S_p
            alpha_np = self.alpha_mu + n
            T_np = T_0p + S_np + (self.alpha_mu * n / alpha_np) * np.outer(
                X_p.mean(axis=0) - mu_0p, X_p.mean(axis=0) - mu_0p
            )
            score_p = 0.0
            score_p += -0.5 * n * d_p * np.log(np.pi)
            score_p += 0.5 * d_p * np.log(self.alpha_mu / alpha_np)
            for j in range(d_p):
                score_p += gammaln(0.5 * (alpha_w + n - j)) - gammaln(
                    0.5 * (alpha_w - j)
                )
            score_p += 0.5 * alpha_w * np.linalg.slogdet(T_0p)[1]
            score_p -= 0.5 * (alpha_w + n) * np.linalg.slogdet(T_np)[1]
            score -= score_p

        return float(score)


# ====================================================================
# DAG operations (for structure search)
# ====================================================================

@dataclass
class DAGOperation:
    """A single modification to a DAG."""
    op_type: str  # "add", "delete", "reverse"
    edge: Tuple[int, int]
    score_delta: float = 0.0


def _enumerate_operations(
    dag: nx.DiGraph,
    p: int,
    scorer: DAGScore,
    data: NDArray,
    tabu: Optional[Set[Tuple[str, Tuple[int, int]]]] = None,
) -> List[DAGOperation]:
    """Enumerate all valid single-edge operations and compute score deltas."""
    if tabu is None:
        tabu = set()
    ops: List[DAGOperation] = []
    nodes = list(range(p))

    for i in nodes:
        for j in nodes:
            if i == j:
                continue

            if dag.has_edge(i, j):
                # --- DELETE ---
                if ("delete", (i, j)) in tabu:
                    continue
                old_parents_j = sorted(dag.predecessors(j))
                new_parents_j = [x for x in old_parents_j if x != i]
                delta = (
                    scorer.local_score(j, new_parents_j, data)
                    - scorer.local_score(j, old_parents_j, data)
                )
                ops.append(DAGOperation("delete", (i, j), delta))

                # --- REVERSE ---
                if ("reverse", (i, j)) in tabu:
                    continue
                test_dag = dag.copy()
                test_dag.remove_edge(i, j)
                test_dag.add_edge(j, i)
                if nx.is_directed_acyclic_graph(test_dag):
                    old_parents_i = sorted(dag.predecessors(i))
                    new_parents_i = sorted(test_dag.predecessors(i))
                    new_parents_j2 = sorted(test_dag.predecessors(j))
                    delta = (
                        scorer.local_score(j, new_parents_j2, data)
                        + scorer.local_score(i, new_parents_i, data)
                        - scorer.local_score(j, old_parents_j, data)
                        - scorer.local_score(i, old_parents_i, data)
                    )
                    ops.append(DAGOperation("reverse", (i, j), delta))
            else:
                # --- ADD ---
                if ("add", (i, j)) in tabu:
                    continue
                test_dag = dag.copy()
                test_dag.add_edge(i, j)
                if not nx.is_directed_acyclic_graph(test_dag):
                    continue
                old_parents_j = sorted(dag.predecessors(j))
                new_parents_j = sorted(old_parents_j) + [i]
                new_parents_j.sort()
                delta = (
                    scorer.local_score(j, new_parents_j, data)
                    - scorer.local_score(j, old_parents_j, data)
                )
                ops.append(DAGOperation("add", (i, j), delta))

    return ops


# ====================================================================
# Greedy Hill Climbing
# ====================================================================

class GreedyHillClimbing:
    """Greedy hill-climbing search over the DAG space.

    Parameters
    ----------
    scorer : DAGScore
        Scoring function (BIC, BDeu, BGe).
    max_iter : int
        Maximum number of hill-climbing steps.
    max_parents : int
        Maximum in-degree for any node.
    tabu_length : int
        Length of tabu list (0 = no tabu search).
    restarts : int
        Number of random restarts.
    """

    def __init__(
        self,
        scorer: Optional[DAGScore] = None,
        max_iter: int = 1000,
        max_parents: int = 5,
        tabu_length: int = 10,
        restarts: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self.scorer = scorer or BICScore()
        self.max_iter = max_iter
        self.max_parents = max_parents
        self.tabu_length = tabu_length
        self.restarts = restarts
        self.seed = seed

        self.best_dag_: Optional[nx.DiGraph] = None
        self.best_score_: float = -np.inf
        self.score_history_: List[float] = []

    def fit(
        self,
        data: NDArray,
        variable_names: Optional[List[str]] = None,
        initial_dag: Optional[nx.DiGraph] = None,
    ) -> "GreedyHillClimbing":
        """Run greedy hill-climbing.

        Parameters
        ----------
        data : ndarray, shape (n, p)
        variable_names : list of str, optional
        initial_dag : nx.DiGraph, optional
            Starting graph; defaults to empty.
        """
        data = np.asarray(data, dtype=np.float64)
        n, p = data.shape
        rng = np.random.default_rng(self.seed)

        best_dag = None
        best_score = -np.inf

        for restart in range(self.restarts + 1):
            if initial_dag is not None and restart == 0:
                dag = initial_dag.copy()
            elif restart > 0:
                dag = self._random_dag(p, rng)
            else:
                dag = nx.DiGraph()
                dag.add_nodes_from(range(p))

            current_score = self.scorer.score(dag, data)
            tabu_list: List[Tuple[str, Tuple[int, int]]] = []
            history = [current_score]

            for step in range(self.max_iter):
                tabu_set = set(tabu_list[-self.tabu_length:]) if self.tabu_length > 0 else set()
                ops = _enumerate_operations(dag, p, self.scorer, data, tabu_set)

                # Filter by max_parents constraint
                valid_ops = []
                for op in ops:
                    if op.op_type == "add":
                        _, j = op.edge
                        if dag.in_degree(j) >= self.max_parents:
                            continue
                    valid_ops.append(op)

                if not valid_ops:
                    break

                best_op = max(valid_ops, key=lambda o: o.score_delta)
                if best_op.score_delta <= 0:
                    break

                self._apply_operation(dag, best_op)
                tabu_list.append((best_op.op_type, best_op.edge))
                current_score += best_op.score_delta
                history.append(current_score)

            if current_score > best_score:
                best_score = current_score
                best_dag = dag.copy()
                self.score_history_ = history

        # Relabel to variable names
        if variable_names is not None and best_dag is not None:
            mapping = {i: variable_names[i] for i in range(p)}
            best_dag = nx.relabel_nodes(best_dag, mapping)

        self.best_dag_ = best_dag
        self.best_score_ = best_score
        return self

    @staticmethod
    def _apply_operation(dag: nx.DiGraph, op: DAGOperation) -> None:
        i, j = op.edge
        if op.op_type == "add":
            dag.add_edge(i, j)
        elif op.op_type == "delete":
            dag.remove_edge(i, j)
        elif op.op_type == "reverse":
            dag.remove_edge(i, j)
            dag.add_edge(j, i)

    @staticmethod
    def _random_dag(p: int, rng: np.random.Generator) -> nx.DiGraph:
        """Generate a random sparse DAG."""
        dag = nx.DiGraph()
        dag.add_nodes_from(range(p))
        perm = rng.permutation(p)
        for i in range(p):
            for j in range(i + 1, p):
                if rng.random() < 2.0 / p:
                    dag.add_edge(int(perm[i]), int(perm[j]))
        return dag

    def get_dag(self) -> nx.DiGraph:
        if self.best_dag_ is None:
            raise RuntimeError("Call fit() first.")
        return self.best_dag_


# ====================================================================
# Bayesian Model Averaging
# ====================================================================

class BayesianModelAveraging:
    """Approximate Bayesian model averaging over DAGs using an MCMC
    (Metropolis-Hastings) sampler.

    Parameters
    ----------
    scorer : DAGScore
        Log-score function.
    n_samples : int
        Number of MCMC samples.
    burn_in : int
        Burn-in period.
    seed : int
    """

    def __init__(
        self,
        scorer: Optional[DAGScore] = None,
        n_samples: int = 1000,
        burn_in: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        self.scorer = scorer or BICScore()
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.seed = seed
        self.edge_probs_: Optional[NDArray] = None
        self.dag_scores_: List[float] = []

    def fit(self, data: NDArray) -> "BayesianModelAveraging":
        """Run MCMC structure sampling."""
        data = np.asarray(data, dtype=np.float64)
        n, p = data.shape
        rng = np.random.default_rng(self.seed)

        # Start from empty DAG
        dag = nx.DiGraph()
        dag.add_nodes_from(range(p))
        current_score = self.scorer.score(dag, data)

        edge_counts = np.zeros((p, p))
        total_samples = 0

        for step in range(self.n_samples + self.burn_in):
            # Propose a random operation
            op_type = rng.choice(["add", "delete", "reverse"])
            i, j = int(rng.integers(p)), int(rng.integers(p))
            if i == j:
                continue

            proposal = dag.copy()
            if op_type == "add" and not proposal.has_edge(i, j):
                proposal.add_edge(i, j)
            elif op_type == "delete" and proposal.has_edge(i, j):
                proposal.remove_edge(i, j)
            elif op_type == "reverse" and proposal.has_edge(i, j):
                proposal.remove_edge(i, j)
                proposal.add_edge(j, i)
            else:
                continue

            if not nx.is_directed_acyclic_graph(proposal):
                continue

            proposal_score = self.scorer.score(proposal, data)
            log_accept = proposal_score - current_score
            if log_accept > 0 or np.log(rng.random()) < log_accept:
                dag = proposal
                current_score = proposal_score

            if step >= self.burn_in:
                for u, v in dag.edges():
                    edge_counts[u, v] += 1
                total_samples += 1
                self.dag_scores_.append(current_score)

        if total_samples > 0:
            self.edge_probs_ = edge_counts / total_samples
        else:
            self.edge_probs_ = np.zeros((p, p))
        return self

    def get_edge_probabilities(
        self, variable_names: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], float]:
        """Return posterior edge inclusion probabilities."""
        if self.edge_probs_ is None:
            raise RuntimeError("Call fit() first.")
        p = self.edge_probs_.shape[0]
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(p)]
        result: Dict[Tuple[str, str], float] = {}
        for i in range(p):
            for j in range(p):
                if self.edge_probs_[i, j] > 0:
                    result[(variable_names[i], variable_names[j])] = float(
                        self.edge_probs_[i, j]
                    )
        return result

    def get_median_dag(
        self,
        threshold: float = 0.5,
        variable_names: Optional[List[str]] = None,
    ) -> nx.DiGraph:
        """Return the median probability DAG (edges with P > threshold)."""
        if self.edge_probs_ is None:
            raise RuntimeError("Call fit() first.")
        p = self.edge_probs_.shape[0]
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(p)]
        G = nx.DiGraph()
        G.add_nodes_from(variable_names)

        # Add edges in decreasing probability, skip if cycle
        edges = []
        for i in range(p):
            for j in range(p):
                if self.edge_probs_[i, j] > threshold:
                    edges.append((i, j, self.edge_probs_[i, j]))
        edges.sort(key=lambda x: x[2], reverse=True)
        for i, j, prob in edges:
            G.add_edge(variable_names[i], variable_names[j], probability=prob)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(variable_names[i], variable_names[j])
        return G
