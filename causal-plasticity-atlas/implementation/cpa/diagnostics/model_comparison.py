"""Model comparison and equivalence class analysis.

Provides tools for comparing causal models using information criteria,
cross-validation, and Bayes factors, as well as analysing Markov
equivalence classes.

Classes
-------
ModelSelector
    BIC/AIC-based model comparison and cross-validation.
EquivalenceClassAnalyzer
    CPDAG analysis and equivalence class estimation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import stats as sp_stats

from cpa.utils.logging import get_logger

logger = get_logger("diagnostics.model_comparison")


# ---------------------------------------------------------------------------
# Model comparison result
# ---------------------------------------------------------------------------


@dataclass
class ModelScore:
    """Score for a single causal model.

    Attributes
    ----------
    model_id : str
    adj_matrix : np.ndarray
    bic : float
    aic : float
    log_likelihood : float
    n_params : int
    n_samples : int
    """

    model_id: str
    adj_matrix: np.ndarray
    bic: float = np.inf
    aic: float = np.inf
    log_likelihood: float = -np.inf
    n_params: int = 0
    n_samples: int = 0


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------


class ModelSelector:
    """BIC/AIC-based model comparison for causal models.

    Compares candidate causal structures by fitting linear Gaussian
    models and computing information criteria.

    Parameters
    ----------
    criterion : str
        Scoring criterion: 'bic', 'aic', or 'both'.
    n_folds : int
        Number of cross-validation folds (for CV-based comparison).

    Examples
    --------
    >>> selector = ModelSelector(criterion='bic')
    >>> scores = selector.score_models([adj1, adj2], data)
    >>> best = selector.select_best(scores)
    """

    def __init__(
        self,
        criterion: str = "bic",
        n_folds: int = 5,
    ) -> None:
        self.criterion = criterion
        self.n_folds = n_folds

    def score_model(
        self,
        adj_matrix: np.ndarray,
        data: np.ndarray,
        model_id: str = "model",
    ) -> ModelScore:
        """Score a single causal model.

        Fits a linear Gaussian model consistent with the DAG structure
        and computes BIC and AIC scores.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Binary adjacency matrix (n_vars x n_vars).
        data : np.ndarray
            Observational data (n_samples x n_vars).
        model_id : str
            Identifier for this model.

        Returns
        -------
        ModelScore
        """
        n_samples, n_vars = data.shape
        total_ll = 0.0
        total_params = 0

        for j in range(n_vars):
            parents = np.where(adj_matrix[:, j] != 0)[0]
            n_pa = len(parents)

            if n_pa == 0:
                # No parents: use marginal distribution
                y = data[:, j]
                mu = np.mean(y)
                var = np.var(y, ddof=1) if n_samples > 1 else np.var(y) + 1e-10
                if var < 1e-15:
                    var = 1e-15
                ll = -0.5 * n_samples * (np.log(2 * np.pi * var) + 1.0)
                total_ll += ll
                total_params += 2  # mean + variance
            else:
                # Regression on parents
                X = data[:, parents]
                y = data[:, j]

                # OLS fit
                X_aug = np.column_stack([np.ones(n_samples), X])
                try:
                    beta, residuals, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                    y_pred = X_aug @ beta
                    sse = np.sum((y - y_pred) ** 2)
                except np.linalg.LinAlgError:
                    sse = np.sum((y - np.mean(y)) ** 2)

                var = sse / n_samples
                if var < 1e-15:
                    var = 1e-15
                ll = -0.5 * n_samples * (np.log(2 * np.pi * var) + 1.0)
                total_ll += ll
                total_params += n_pa + 2  # coefficients + intercept + variance

        # BIC and AIC
        bic = -2 * total_ll + total_params * np.log(n_samples)
        aic = -2 * total_ll + 2 * total_params

        return ModelScore(
            model_id=model_id,
            adj_matrix=adj_matrix,
            bic=float(bic),
            aic=float(aic),
            log_likelihood=float(total_ll),
            n_params=total_params,
            n_samples=n_samples,
        )

    def score_models(
        self,
        adj_matrices: Sequence[np.ndarray],
        data: np.ndarray,
        model_ids: Optional[Sequence[str]] = None,
    ) -> List[ModelScore]:
        """Score multiple candidate models.

        Parameters
        ----------
        adj_matrices : sequence of np.ndarray
        data : np.ndarray
        model_ids : sequence of str, optional

        Returns
        -------
        list of ModelScore
        """
        if model_ids is None:
            model_ids = [f"model_{i}" for i in range(len(adj_matrices))]

        scores = []
        for adj, mid in zip(adj_matrices, model_ids):
            score = self.score_model(adj, data, model_id=mid)
            scores.append(score)

        return scores

    def select_best(
        self,
        scores: Sequence[ModelScore],
        criterion: Optional[str] = None,
    ) -> ModelScore:
        """Select the best model by criterion.

        Parameters
        ----------
        scores : sequence of ModelScore
        criterion : str, optional
            Override the default criterion.

        Returns
        -------
        ModelScore
        """
        crit = criterion or self.criterion
        if crit == "bic":
            return min(scores, key=lambda s: s.bic)
        elif crit == "aic":
            return min(scores, key=lambda s: s.aic)
        else:
            return min(scores, key=lambda s: s.bic)

    def rank_models(
        self,
        scores: Sequence[ModelScore],
        criterion: Optional[str] = None,
    ) -> List[Tuple[ModelScore, float]]:
        """Rank models by criterion and compute relative weights.

        Computes Akaike/BIC weights for model averaging.

        Parameters
        ----------
        scores : sequence of ModelScore
        criterion : str, optional

        Returns
        -------
        list of (ModelScore, weight)
            Sorted by criterion (best first) with model weights.
        """
        crit = criterion or self.criterion

        if crit == "bic":
            values = np.array([s.bic for s in scores])
        else:
            values = np.array([s.aic for s in scores])

        # Compute delta-IC and weights
        min_val = np.min(values)
        deltas = values - min_val
        exp_deltas = np.exp(-0.5 * deltas)
        weights = exp_deltas / (np.sum(exp_deltas) + 1e-15)

        # Sort by weight (descending)
        order = np.argsort(weights)[::-1]
        return [(scores[i], float(weights[i])) for i in order]

    def cross_validate(
        self,
        adj_matrix: np.ndarray,
        data: np.ndarray,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Cross-validated model assessment.

        Parameters
        ----------
        adj_matrix : np.ndarray
        data : np.ndarray
        seed : int, optional

        Returns
        -------
        dict
            cv_score (mean), cv_std, fold_scores.
        """
        rng = np.random.default_rng(seed)
        n_samples = data.shape[0]
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        fold_size = n_samples // self.n_folds
        fold_scores = []

        for fold in range(self.n_folds):
            start = fold * fold_size
            if fold == self.n_folds - 1:
                end = n_samples
            else:
                end = start + fold_size

            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            train_data = data[train_idx]
            test_data = data[test_idx]

            # Fit on train, evaluate on test
            ll = self._compute_test_log_likelihood(adj_matrix, train_data, test_data)
            fold_scores.append(ll)

        scores = np.array(fold_scores)
        return {
            "cv_score": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
            "fold_scores": scores.tolist(),
        }

    def _compute_test_log_likelihood(
        self,
        adj: np.ndarray,
        train: np.ndarray,
        test: np.ndarray,
    ) -> float:
        """Compute log-likelihood of test data given model fit on training data.

        Parameters
        ----------
        adj : np.ndarray
        train : np.ndarray
        test : np.ndarray

        Returns
        -------
        float
        """
        n_test = test.shape[0]
        n_vars = adj.shape[0]
        total_ll = 0.0

        for j in range(n_vars):
            parents = np.where(adj[:, j] != 0)[0]

            if len(parents) == 0:
                mu = np.mean(train[:, j])
                var = np.var(train[:, j], ddof=1)
                if var < 1e-15:
                    var = 1e-15
                ll = -0.5 * np.sum(
                    np.log(2 * np.pi * var) + (test[:, j] - mu) ** 2 / var
                )
            else:
                X_train = np.column_stack([np.ones(train.shape[0]), train[:, parents]])
                y_train = train[:, j]

                try:
                    beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
                except np.linalg.LinAlgError:
                    beta = np.zeros(X_train.shape[1])

                train_pred = X_train @ beta
                var = np.mean((y_train - train_pred) ** 2)
                if var < 1e-15:
                    var = 1e-15

                X_test = np.column_stack([np.ones(n_test), test[:, parents]])
                test_pred = X_test @ beta
                ll = -0.5 * np.sum(
                    np.log(2 * np.pi * var) + (test[:, j] - test_pred) ** 2 / var
                )

            total_ll += ll

        return float(total_ll)

    def bayes_factor_approximation(
        self,
        score1: ModelScore,
        score2: ModelScore,
    ) -> Dict[str, Any]:
        """Approximate the Bayes factor between two models using BIC.

        BF ≈ exp(-0.5 * (BIC_1 - BIC_2))

        Parameters
        ----------
        score1, score2 : ModelScore

        Returns
        -------
        dict
            log_bf, bf, interpretation.
        """
        delta_bic = score1.bic - score2.bic
        log_bf = -0.5 * delta_bic

        # Clamp for numerical stability
        log_bf_clamped = np.clip(log_bf, -500, 500)
        bf = float(np.exp(log_bf_clamped))

        # Kass-Raftery interpretation
        if abs(log_bf) < 0.5:
            interpretation = "Not worth more than a bare mention"
        elif abs(log_bf) < 1.0:
            interpretation = "Substantial evidence"
        elif abs(log_bf) < 2.0:
            interpretation = "Strong evidence"
        else:
            interpretation = "Decisive evidence"

        if log_bf > 0:
            favored = score1.model_id
        else:
            favored = score2.model_id

        return {
            "log_bayes_factor": float(log_bf),
            "bayes_factor": bf,
            "favored_model": favored,
            "interpretation": interpretation,
        }


# ---------------------------------------------------------------------------
# Equivalence class analyzer
# ---------------------------------------------------------------------------


class EquivalenceClassAnalyzer:
    """Analyze Markov equivalence classes of causal DAGs.

    Provides tools for working with CPDAGs (Completed Partially
    Directed Acyclic Graphs) and estimating equivalence class sizes.

    Examples
    --------
    >>> analyzer = EquivalenceClassAnalyzer()
    >>> cpdag = analyzer.dag_to_cpdag(adj_matrix)
    >>> size = analyzer.estimate_class_size(cpdag)
    """

    def dag_to_cpdag(self, adj: np.ndarray) -> np.ndarray:
        """Convert a DAG to its CPDAG (equivalence class representative).

        Applies Meek's orientation rules to identify compelled and
        reversible edges.

        Parameters
        ----------
        adj : np.ndarray
            Binary DAG adjacency matrix.

        Returns
        -------
        np.ndarray
            CPDAG adjacency matrix where:
            - adj[i,j]=1 and adj[j,i]=0 → compelled directed edge i→j
            - adj[i,j]=1 and adj[j,i]=1 → reversible (undirected) edge
        """
        n = adj.shape[0]
        cpdag = np.zeros((n, n), dtype=int)

        # Start with skeleton
        for i in range(n):
            for j in range(n):
                if adj[i, j] != 0:
                    cpdag[i, j] = 1

        # Identify v-structures (compelled orientations)
        compelled = np.zeros((n, n), dtype=bool)
        for j in range(n):
            parents = np.where(adj[:, j] != 0)[0]
            for i_idx in range(len(parents)):
                for k_idx in range(i_idx + 1, len(parents)):
                    i = parents[i_idx]
                    k = parents[k_idx]
                    # i → j ← k is a v-structure if i and k are not adjacent
                    if adj[i, k] == 0 and adj[k, i] == 0:
                        compelled[i, j] = True
                        compelled[k, j] = True

        # Apply Meek rules for additional compelled orientations
        changed = True
        max_iters = n * n
        iteration = 0
        while changed and iteration < max_iters:
            changed = False
            iteration += 1

            for i in range(n):
                for j in range(n):
                    if adj[i, j] == 0 or compelled[i, j]:
                        continue

                    # Rule 1: i→j is compelled if ∃k: k→i compelled, k not adj j
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if compelled[k, i] and adj[k, j] == 0 and adj[j, k] == 0:
                            compelled[i, j] = True
                            changed = True
                            break

                    if compelled[i, j]:
                        continue

                    # Rule 2: i→j is compelled if ∃k: i→k→j, both compelled
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if compelled[i, k] and compelled[k, j]:
                            compelled[i, j] = True
                            changed = True
                            break

                    if compelled[i, j]:
                        continue

                    # Rule 3: i→j is compelled if ∃k,l: k→j, l→j both compelled,
                    # k-i and l-i both undirected, k and l not adjacent
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if not compelled[k, j]:
                            continue
                        # k-i undirected?
                        if not (adj[k, i] != 0 and adj[i, k] != 0
                                and not compelled[k, i] and not compelled[i, k]):
                            continue
                        for l in range(k + 1, n):
                            if l == i or l == j:
                                continue
                            if not compelled[l, j]:
                                continue
                            if not (adj[l, i] != 0 and adj[i, l] != 0
                                    and not compelled[l, i] and not compelled[i, l]):
                                continue
                            if adj[k, l] == 0 and adj[l, k] == 0:
                                compelled[i, j] = True
                                changed = True
                                break
                        if compelled[i, j]:
                            break

        # Build CPDAG: compelled edges stay directed, others become undirected
        result = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if adj[i, j] != 0:
                    if compelled[i, j]:
                        result[i, j] = 1
                    else:
                        result[i, j] = 1
                        result[j, i] = 1

        return result

    def cpdag_to_skeleton(self, cpdag: np.ndarray) -> np.ndarray:
        """Extract the skeleton (undirected graph) from a CPDAG.

        Parameters
        ----------
        cpdag : np.ndarray

        Returns
        -------
        np.ndarray
            Symmetric binary adjacency matrix.
        """
        return ((cpdag != 0) | (cpdag.T != 0)).astype(int)

    def identify_compelled_edges(self, cpdag: np.ndarray) -> List[Tuple[int, int]]:
        """Identify compelled (directed) edges in the CPDAG.

        Parameters
        ----------
        cpdag : np.ndarray

        Returns
        -------
        list of (int, int)
            Compelled directed edges.
        """
        n = cpdag.shape[0]
        compelled = []
        for i in range(n):
            for j in range(n):
                if cpdag[i, j] != 0 and cpdag[j, i] == 0:
                    compelled.append((i, j))
        return compelled

    def identify_reversible_edges(
        self, cpdag: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Identify reversible (undirected) edges in the CPDAG.

        Parameters
        ----------
        cpdag : np.ndarray

        Returns
        -------
        list of (int, int)
            Reversible edges (each pair listed once, i < j).
        """
        n = cpdag.shape[0]
        reversible = []
        for i in range(n):
            for j in range(i + 1, n):
                if cpdag[i, j] != 0 and cpdag[j, i] != 0:
                    reversible.append((i, j))
        return reversible

    def estimate_class_size(self, cpdag: np.ndarray) -> int:
        """Estimate the size of the Markov equivalence class.

        For small graphs, this enumerates all valid DAG orientations.
        For larger graphs, uses a heuristic estimate.

        Parameters
        ----------
        cpdag : np.ndarray

        Returns
        -------
        int
            Estimated number of DAGs in the equivalence class.
        """
        reversible = self.identify_reversible_edges(cpdag)
        n_rev = len(reversible)

        if n_rev == 0:
            return 1

        # For small graphs, enumerate
        if n_rev <= 15:
            return self._enumerate_class_size(cpdag, reversible)

        # Heuristic: upper bound is 2^n_rev, but acyclicity constrains it
        # Rough estimate: ~2^(n_rev * 0.7) due to cycle constraints
        return max(1, int(2 ** (n_rev * 0.7)))

    def _enumerate_class_size(
        self,
        cpdag: np.ndarray,
        reversible: List[Tuple[int, int]],
    ) -> int:
        """Enumerate valid DAGs in the equivalence class.

        Parameters
        ----------
        cpdag : np.ndarray
        reversible : list of (int, int)

        Returns
        -------
        int
        """
        n = cpdag.shape[0]
        compelled = self.identify_compelled_edges(cpdag)
        count = 0

        for orientation in itertools.product([0, 1], repeat=len(reversible)):
            # Build candidate DAG
            dag = np.zeros((n, n), dtype=int)

            # Add compelled edges
            for i, j in compelled:
                dag[i, j] = 1

            # Orient reversible edges
            for idx, (i, j) in enumerate(reversible):
                if orientation[idx] == 0:
                    dag[i, j] = 1
                else:
                    dag[j, i] = 1

            # Check acyclicity
            if not self._is_dag(dag):
                continue

            count += 1

        return max(1, count)

    @staticmethod
    def _is_dag(adj: np.ndarray) -> bool:
        """Check if adjacency matrix is a DAG using topological sort.

        Parameters
        ----------
        adj : np.ndarray

        Returns
        -------
        bool
        """
        n = adj.shape[0]
        in_degree = (adj != 0).astype(int).sum(axis=0)
        queue = list(np.where(in_degree == 0)[0])
        visited = 0

        while queue:
            node = queue.pop(0)
            visited += 1
            for child in range(n):
                if adj[node, child] != 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        return visited == n

    def identifiability_assessment(
        self,
        adj: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Assess the identifiability of each edge in the DAG.

        Parameters
        ----------
        adj : np.ndarray
        variable_names : list of str, optional

        Returns
        -------
        dict
            Per-edge identifiability status and overall metrics.
        """
        n = adj.shape[0]
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(n)]

        cpdag = self.dag_to_cpdag(adj)
        compelled = self.identify_compelled_edges(cpdag)
        reversible = self.identify_reversible_edges(cpdag)
        class_size = self.estimate_class_size(cpdag)

        compelled_set = set(compelled)
        reversible_set = set()
        for i, j in reversible:
            reversible_set.add((i, j))
            reversible_set.add((j, i))

        edges = list(zip(*np.where(adj != 0)))
        edge_info = []
        for i, j in edges:
            if (i, j) in compelled_set:
                status = "identifiable"
            elif (min(i, j), max(i, j)) in {(min(a, b), max(a, b)) for a, b in reversible}:
                status = "non-identifiable"
            else:
                status = "identifiable"

            edge_info.append({
                "from": variable_names[i],
                "to": variable_names[j],
                "status": status,
            })

        n_identifiable = sum(1 for e in edge_info if e["status"] == "identifiable")
        n_total = len(edge_info)

        return {
            "equivalence_class_size": class_size,
            "n_compelled_edges": len(compelled),
            "n_reversible_edges": len(reversible),
            "n_total_edges": n_total,
            "identifiability_ratio": n_identifiable / max(n_total, 1),
            "edge_identifiability": edge_info,
            "fully_identifiable": class_size == 1,
        }

    def compare_equivalence_classes(
        self,
        adj1: np.ndarray,
        adj2: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare the equivalence classes of two DAGs.

        Parameters
        ----------
        adj1, adj2 : np.ndarray

        Returns
        -------
        dict
        """
        cpdag1 = self.dag_to_cpdag(adj1)
        cpdag2 = self.dag_to_cpdag(adj2)

        # Check if they're in the same equivalence class
        same_class = np.array_equal(cpdag1, cpdag2)

        # Skeleton comparison
        skel1 = self.cpdag_to_skeleton(cpdag1)
        skel2 = self.cpdag_to_skeleton(cpdag2)
        same_skeleton = np.array_equal(skel1, skel2)

        # Structural differences
        n = adj1.shape[0]
        shared_compelled = 0
        different_orientation = 0

        for i in range(n):
            for j in range(n):
                if cpdag1[i, j] != 0 and cpdag1[j, i] == 0:
                    if cpdag2[i, j] != 0 and cpdag2[j, i] == 0:
                        shared_compelled += 1
                    elif cpdag2[j, i] != 0 and cpdag2[i, j] == 0:
                        different_orientation += 1

        return {
            "same_equivalence_class": same_class,
            "same_skeleton": same_skeleton,
            "shared_compelled_edges": shared_compelled,
            "different_orientations": different_orientation,
            "class_size_1": self.estimate_class_size(cpdag1),
            "class_size_2": self.estimate_class_size(cpdag2),
        }
