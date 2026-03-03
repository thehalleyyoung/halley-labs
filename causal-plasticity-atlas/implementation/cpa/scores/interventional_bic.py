"""Interventional BIC score for multi-context causal discovery.

Extends the BIC score to handle data from multiple experimental
contexts (observational + interventional).  For each node, the score
compares a *pooled* model (same linear-Gaussian mechanism across
observational contexts) with *context-specific* models (separate
parameters per context).  Model selection between pooled and
heterogeneous models is done via BIC.

This directly supports the Causal Plasticity Atlas (CPA) goal:
detecting which mechanisms change across experimental conditions.

Key idea
--------
For a node X_i with parents Pa_i:
- Observational contexts: those where X_i was NOT intervened upon.
  These share the same causal mechanism and can be pooled.
- Interventional contexts: those where X_i WAS intervened upon.
  The mechanism may be different, so each gets its own parameters.

The total score for a family (i, Pa_i) is:

    Score = sum_{c in obs_contexts} LL_c(i, Pa_i | theta_pooled)
          + sum_{c in int_contexts} LL_c(i, Pa_i | theta_c)
          - penalty

Optionally, we also compare pooled vs heterogeneous on the
observational contexts to detect *soft* mechanism changes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helpers (self-contained — no dependency on bic.py)
# ---------------------------------------------------------------------------

def _ols_fit(
    y: NDArray, X: NDArray, rcond: float = 1e-12
) -> Tuple[NDArray, NDArray, float]:
    """OLS regression.  Returns (coefficients, residuals, mle_variance)."""
    n = y.shape[0]
    if X.shape[1] == 0:
        residuals = y - np.mean(y)
        var = float(np.sum(residuals ** 2) / max(n, 1))
        return np.array([]), residuals, max(var, 1e-300)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=rcond)
    residuals = y - X @ coeffs
    var = float(np.sum(residuals ** 2) / max(n, 1))
    return coeffs, residuals, max(var, 1e-300)


def _design_matrix(
    data: NDArray, parents: Sequence[int]
) -> NDArray:
    """Build design matrix with intercept column."""
    n = data.shape[0]
    if len(parents) == 0:
        return np.ones((n, 1), dtype=data.dtype)
    X = data[:, list(parents)]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(n, dtype=data.dtype), X])


def _gaussian_ll(residuals: NDArray, var: float, n: int) -> float:
    """Gaussian log-likelihood at MLE."""
    if var <= 0:
        var = 1e-300
    return -0.5 * n * math.log(2.0 * math.pi) \
           - 0.5 * n * math.log(var) \
           - 0.5 * np.sum(residuals ** 2) / var


def _bic_penalty(num_params: int, n: int, weight: float = 1.0) -> float:
    if n <= 1:
        return 0.0
    return 0.5 * num_params * math.log(n) * weight


# ---------------------------------------------------------------------------
# InterventionalBICScore
# ---------------------------------------------------------------------------

class InterventionalBICScore:
    """BIC score for multi-context interventional data.

    Parameters
    ----------
    datasets : List[NDArray]
        One observation matrix ``(n_samples_c, n_variables)`` per context.
    intervention_targets : List[Set[int]]
        Per-context sets of intervened-upon variable indices.
        For observational data use an empty set ``set()``.
    context_labels : Optional[List[str]]
        Human-readable labels for each context.
    penalty_weight : float
        Multiplier for the BIC penalty.
    """

    def __init__(
        self,
        datasets: List[NDArray],
        intervention_targets: List[Set[int]],
        context_labels: Optional[List[str]] = None,
        penalty_weight: float = 1.0,
    ) -> None:
        self.datasets = [np.asarray(d, dtype=np.float64) for d in datasets]
        self.intervention_targets = [set(t) for t in intervention_targets]
        self.penalty_weight = penalty_weight
        self.n_contexts = len(datasets)
        self.n_variables = self.datasets[0].shape[1]
        self.context_labels = context_labels or [
            f"ctx_{c}" for c in range(self.n_contexts)
        ]

        if len(self.intervention_targets) != self.n_contexts:
            raise ValueError(
                "Number of intervention target sets must match number of datasets"
            )
        for d in self.datasets:
            if d.shape[1] != self.n_variables:
                raise ValueError("All datasets must have the same number of variables")

        # Pre-compute per-context sample sizes
        self._context_sizes = [d.shape[0] for d in self.datasets]
        self._total_n = sum(self._context_sizes)

    # ---- Observational / interventional context helpers ------------------

    def _obs_contexts(self, node: int) -> List[int]:
        """Return indices of contexts where *node* is NOT intervened upon."""
        return [
            c for c in range(self.n_contexts)
            if node not in self.intervention_targets[c]
        ]

    def _int_contexts(self, node: int) -> List[int]:
        """Return indices of contexts where *node* IS intervened upon."""
        return [
            c for c in range(self.n_contexts)
            if node in self.intervention_targets[c]
        ]

    # ---- Context-specific log-likelihood --------------------------------

    def _context_specific_ll(
        self, node: int, parents: Sequence[int], context: int
    ) -> Tuple[float, int]:
        """Log-likelihood for *node* | *parents* in a single context.

        Returns (log_likelihood, n_samples).
        """
        data_c = self.datasets[context]
        n_c = data_c.shape[0]
        if n_c == 0:
            return 0.0, 0
        y = data_c[:, node]
        X = _design_matrix(data_c, parents)
        _, residuals, var = _ols_fit(y, X)
        ll = _gaussian_ll(residuals, var, n_c)
        return float(ll), n_c

    # ---- Pooled score ---------------------------------------------------

    def pooled_score(self, node: int, parents: Sequence[int]) -> float:
        """BIC score pooling all observational contexts for *node*.

        Uses a single set of regression parameters across all
        non-intervened contexts.
        """
        parents = list(parents)
        obs = self._obs_contexts(node)
        if not obs:
            return 0.0

        # Stack data from observational contexts
        data_obs = np.vstack([self.datasets[c] for c in obs])
        n_obs = data_obs.shape[0]
        if n_obs == 0:
            return 0.0

        y = data_obs[:, node]
        X = _design_matrix(data_obs, parents)
        _, residuals, var = _ols_fit(y, X)
        ll = _gaussian_ll(residuals, var, n_obs)
        k = len(parents) + 2  # intercept + parents + variance
        pen = _bic_penalty(k, n_obs, self.penalty_weight)
        return float(ll - pen)

    def _pooled_score(self, node: int, parents: Sequence[int]) -> float:
        """Alias for pooled_score."""
        return self.pooled_score(node, parents)

    # ---- Heterogeneous score --------------------------------------------

    def _heterogeneous_score(
        self, node: int, parents: Sequence[int]
    ) -> float:
        """Score allowing different mechanisms per observational context.

        Each observational context gets its own regression parameters.
        """
        parents = list(parents)
        obs = self._obs_contexts(node)
        if not obs:
            return 0.0

        total = 0.0
        for c in obs:
            ll_c, n_c = self._context_specific_ll(node, parents, c)
            if n_c > 0:
                k = len(parents) + 2
                pen = _bic_penalty(k, n_c, self.penalty_weight)
                total += ll_c - pen
        return total

    # ---- Context-specific score (public) --------------------------------

    def context_specific_score(
        self, node: int, parents: Sequence[int], context: int
    ) -> float:
        """Return the BIC score for *node* | *parents* in a single context."""
        parents = list(parents)
        ll_c, n_c = self._context_specific_ll(node, parents, context)
        if n_c == 0:
            return 0.0
        k = len(parents) + 2
        pen = _bic_penalty(k, n_c, self.penalty_weight)
        return float(ll_c - pen)

    # ---- Model selection (pooled vs heterogeneous) ----------------------

    def _model_selection(
        self, node: int, parents: Sequence[int]
    ) -> Tuple[str, float]:
        """Select between pooled and heterogeneous models via BIC.

        Returns
        -------
        model : str  ('pooled' or 'heterogeneous')
        score : float  (the winning score)
        """
        parents = list(parents)
        pooled = self.pooled_score(node, parents)
        hetero = self._heterogeneous_score(node, parents)

        if pooled >= hetero:
            return "pooled", pooled
        else:
            return "heterogeneous", hetero

    # ---- Main local_score ----------------------------------------------

    def local_score(
        self,
        node: int,
        parents: Sequence[int],
        context: Optional[int] = None,
    ) -> float:
        """Return the local interventional BIC score.

        If *context* is given, returns the score for that single context.
        Otherwise, returns the combined score:
          - For observational contexts: best of pooled vs heterogeneous.
          - For interventional contexts: context-specific scores.
        """
        parents = list(parents)
        self._validate(node, parents)

        if context is not None:
            return self.context_specific_score(node, parents, context)

        # Observational part
        _, obs_score = self._model_selection(node, parents)

        # Interventional part
        int_score = 0.0
        for c in self._int_contexts(node):
            int_score += self.context_specific_score(node, parents, c)

        return obs_score + int_score

    # ---- Full DAG score ------------------------------------------------

    def score_dag(self, adj_matrix: NDArray) -> float:
        """Return the total interventional BIC score of a DAG."""
        adj = np.asarray(adj_matrix)
        total = 0.0
        for j in range(self.n_variables):
            parents = list(np.nonzero(adj[:, j])[0])
            total += self.local_score(j, parents)
        return total

    # ---- Mechanism change detection ------------------------------------

    def detect_intervention_targets(
        self, node: int, parents: Sequence[int]
    ) -> Dict[str, object]:
        """Detect which contexts have different mechanisms for *node*.

        Compares per-context BIC to the pooled BIC to identify contexts
        where the mechanism has changed (even among non-intervened contexts).

        Returns
        -------
        dict with keys:
            'model': 'pooled' or 'heterogeneous'
            'pooled_score': float
            'heterogeneous_score': float
            'context_scores': list of float  (per-context scores)
            'changed_contexts': list of int  (contexts with different mechanisms)
            'mechanism_params': list of dict  (per-context regression params)
        """
        parents = list(parents)
        obs = self._obs_contexts(node)

        pooled = self.pooled_score(node, parents)
        hetero = self._heterogeneous_score(node, parents)

        context_scores = []
        mechanism_params = []
        for c in range(self.n_contexts):
            cs = self.context_specific_score(node, parents, c)
            context_scores.append(cs)

            # Extract regression parameters for this context
            data_c = self.datasets[c]
            n_c = data_c.shape[0]
            if n_c > 0:
                y = data_c[:, node]
                X = _design_matrix(data_c, parents)
                coeffs, residuals, var = _ols_fit(y, X)
                mechanism_params.append({
                    "coefficients": coeffs.tolist(),
                    "variance": var,
                    "n_samples": n_c,
                })
            else:
                mechanism_params.append({"coefficients": [], "variance": 0.0, "n_samples": 0})

        # Detect changed contexts: if heterogeneous wins, check which
        # observational contexts differ from the pooled parameter estimates
        changed = list(self._int_contexts(node))
        if len(obs) > 1:
            data_pooled = np.vstack([self.datasets[c] for c in obs])
            y_pooled = data_pooled[:, node]
            X_pooled = _design_matrix(data_pooled, parents)
            coeffs_pooled, _, var_pooled = _ols_fit(y_pooled, X_pooled)

            for c in obs:
                if len(mechanism_params[c]["coefficients"]) == 0:
                    continue
                coeffs_c = np.array(mechanism_params[c]["coefficients"])
                var_c = mechanism_params[c]["variance"]
                n_c = mechanism_params[c]["n_samples"]
                if n_c < 3:
                    continue
                # Simple test: compare parameter vectors using a
                # likelihood ratio between pooled and context-specific
                ll_pooled_c = self._eval_model_on_context(
                    node, parents, coeffs_pooled, var_pooled, c
                )
                ll_specific_c, _ = self._context_specific_ll(node, parents, c)
                lr = 2.0 * (ll_specific_c - ll_pooled_c)
                k = len(parents) + 2
                if lr > math.log(n_c) * k:
                    if c not in changed:
                        changed.append(c)

        model = "pooled" if pooled >= hetero else "heterogeneous"
        return {
            "model": model,
            "pooled_score": pooled,
            "heterogeneous_score": hetero,
            "context_scores": context_scores,
            "changed_contexts": sorted(changed),
            "mechanism_params": mechanism_params,
        }

    def _eval_model_on_context(
        self,
        node: int,
        parents: Sequence[int],
        coefficients: NDArray,
        variance: float,
        context: int,
    ) -> float:
        """Evaluate a fixed model (coefficients, variance) on context data."""
        data_c = self.datasets[context]
        n_c = data_c.shape[0]
        if n_c == 0:
            return 0.0
        y = data_c[:, node]
        X = _design_matrix(data_c, parents)
        if len(coefficients) == 0 or X.shape[1] != len(coefficients):
            return -math.inf
        residuals = y - X @ coefficients
        return float(_gaussian_ll(residuals, variance, n_c))

    # ---- Score comparison across penalty weights -----------------------

    def score_dag_range(
        self, adj_matrix: NDArray, penalty_weights: Sequence[float]
    ) -> Dict[float, float]:
        """Score DAG under multiple penalty weights."""
        results = {}
        saved = self.penalty_weight
        for pw in penalty_weights:
            self.penalty_weight = pw
            results[pw] = self.score_dag(adj_matrix)
        self.penalty_weight = saved
        return results

    # ---- Validation ----------------------------------------------------

    def _validate(self, node: int, parents: Sequence[int]) -> None:
        if not 0 <= node < self.n_variables:
            raise ValueError(f"node {node} out of range")
        for p in parents:
            if not 0 <= p < self.n_variables:
                raise ValueError(f"parent {p} out of range")
            if p == node:
                raise ValueError("node cannot be its own parent")

    def __repr__(self) -> str:
        ctx_sizes = [d.shape[0] for d in self.datasets]
        return (
            f"InterventionalBICScore(n_contexts={self.n_contexts}, "
            f"context_sizes={ctx_sizes}, n_variables={self.n_variables})"
        )
