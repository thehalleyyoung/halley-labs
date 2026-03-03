"""Invariant Causal Prediction (ICP) baseline (BL3).

Implements the ICP procedure of Peters, Bühlmann & Meinshausen (2016).
ICP identifies the set of direct causes (invariant parents) of a target
variable by testing invariance of the conditional distribution across
contexts/environments.

The core idea:  S* = ∩ { S : residuals of Y ~ X_S are invariant across
environments }.  Invariance is tested via the Levene test (equal variances)
and the KS test (equal distributions) on the residual distributions.

References
----------
Peters, Bühlmann & Meinshausen (2016).  Causal inference by using
invariant prediction: identification and confidence intervals.
*Journal of the Royal Statistical Society B*, 78(5), 947-1012.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from cpa.core.types import PlasticityClass


# -------------------------------------------------------------------
# Invariance tests
# -------------------------------------------------------------------


def _ols_residuals(
    X: NDArray, y: NDArray,
) -> Tuple[NDArray, NDArray]:
    """Compute OLS residuals and coefficients.

    Parameters
    ----------
    X : (n, k) predictor matrix (may be empty → k=0)
    y : (n,) response

    Returns
    -------
    residuals : (n,)
    coefficients : (k,)
    """
    n = y.shape[0]
    if X.shape[1] == 0:
        return y - np.mean(y), np.array([], dtype=np.float64)
    X_aug = np.column_stack([X, np.ones(n)])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    residuals = y - X_aug @ beta
    return residuals, beta[:-1]


def _levene_test_across_contexts(
    residual_groups: List[NDArray],
) -> float:
    """Levene's test for equality of variances across groups.

    Returns the p-value.
    """
    if len(residual_groups) < 2:
        return 1.0
    groups = [g for g in residual_groups if len(g) > 1]
    if len(groups) < 2:
        return 1.0
    stat, p_value = sp_stats.levene(*groups)
    return float(p_value)


def _ks_test_pairwise(
    residual_groups: List[NDArray],
) -> float:
    """Two-sample KS test between all pairs; return minimum p-value.

    The minimum p-value across all pairs is a conservative test for
    distributional invariance.
    """
    if len(residual_groups) < 2:
        return 1.0
    min_p = 1.0
    for a, b in itertools.combinations(range(len(residual_groups)), 2):
        if len(residual_groups[a]) < 2 or len(residual_groups[b]) < 2:
            continue
        _, p_val = sp_stats.ks_2samp(residual_groups[a], residual_groups[b])
        min_p = min(min_p, float(p_val))
    return min_p


def _f_test_coefficients(
    datasets_list: List[NDArray],
    parent_indices: List[int],
    target: int,
) -> float:
    """Chow-type F-test for equality of regression coefficients.

    Tests whether the linear coefficients X_S → Y are stable across
    all environments.  Returns the p-value.
    """
    if len(datasets_list) < 2 or len(parent_indices) == 0:
        return 1.0

    k = len(parent_indices)
    n_envs = len(datasets_list)

    # Pooled regression
    X_all = np.vstack([d[:, parent_indices] for d in datasets_list])
    y_all = np.concatenate([d[:, target] for d in datasets_list])
    n_total = y_all.shape[0]
    X_aug = np.column_stack([X_all, np.ones(n_total)])
    beta_all, _, _, _ = np.linalg.lstsq(X_aug, y_all, rcond=None)
    rss_pooled = float(np.sum((y_all - X_aug @ beta_all) ** 2))

    # Per-environment regressions
    rss_separate = 0.0
    n_params_separate = 0
    for d in datasets_list:
        X_e = d[:, parent_indices]
        y_e = d[:, target]
        n_e = y_e.shape[0]
        if n_e <= k + 1:
            continue
        X_e_aug = np.column_stack([X_e, np.ones(n_e)])
        beta_e, _, _, _ = np.linalg.lstsq(X_e_aug, y_e, rcond=None)
        rss_separate += float(np.sum((y_e - X_e_aug @ beta_e) ** 2))
        n_params_separate += k + 1

    df_num = n_params_separate - (k + 1)
    df_den = n_total - n_params_separate
    if df_num <= 0 or df_den <= 0 or rss_separate < 1e-15:
        return 1.0

    f_stat = ((rss_pooled - rss_separate) / df_num) / (rss_separate / df_den)
    if f_stat < 0:
        return 1.0
    p_value = float(1.0 - sp_stats.f.cdf(f_stat, df_num, df_den))
    return p_value


# -------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------


class ICPBaseline:
    """Invariant Causal Prediction baseline (BL3).

    Identifies the invariant parent set S* for each target variable by
    testing subsets of predictors for invariance across environments.

    Parameters
    ----------
    significance_level : float
        Alpha for invariance tests.
    max_set_size : int or None
        Maximum size of candidate parent sets to enumerate.
        ``None`` defaults to ``min(p-1, 4)``.
    target : int or None
        Default target variable index.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_set_size: Optional[int] = None,
        target: Optional[int] = None,
    ) -> None:
        self._alpha = significance_level
        self._max_set_size = max_set_size
        self._target = target
        self._datasets: Dict[str, NDArray] = {}
        self._datasets_list: List[NDArray] = []
        self._context_keys: List[str] = []
        self._n_vars: int = 0
        self._invariant_parents: Dict[int, Set[int]] = {}
        self._p_values: Dict[int, Dict[str, float]] = {}
        self._accepted_sets: Dict[int, List[Tuple[List[int], float]]] = {}
        self._fitted: bool = False

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(
        self,
        datasets: Dict[str, NDArray],
        context_labels: Optional[List[str]] = None,
        target: Optional[int] = None,
        intervention_targets: Optional[List[int]] = None,
    ) -> "ICPBaseline":
        """Run ICP across multiple environments.

        Parameters
        ----------
        datasets : Dict[str, NDArray]
            ``{context_label: (n_samples, n_vars)}`` arrays.
        context_labels : list of str, optional
            Explicit context ordering.
        target : int, optional
            If given, only run ICP for this target.
        intervention_targets : list of int, optional
            Known intervention targets (unused, for API compat).

        Returns
        -------
        self
        """
        if not datasets:
            raise ValueError("datasets must be non-empty")
        if isinstance(datasets, list):
            datasets = {f"ctx_{i}": d for i, d in enumerate(datasets)}
        if len(datasets) < 2:
            raise ValueError("ICP requires at least 2 environments")

        self._datasets = dict(datasets)
        self._context_keys = sorted(datasets.keys())
        self._datasets_list = [datasets[k] for k in self._context_keys]
        self._n_vars = self._datasets_list[0].shape[1]

        for k, d in datasets.items():
            if d.shape[1] != self._n_vars:
                raise ValueError(
                    f"Context {k!r}: {d.shape[1]} vars, expected {self._n_vars}"
                )

        effective_target = target if target is not None else self._target

        if effective_target is not None:
            inv_parents, pvals, accepted = self._run_icp_single(
                effective_target,
            )
            self._invariant_parents[effective_target] = inv_parents
            self._p_values[effective_target] = pvals
            self._accepted_sets[effective_target] = accepted
        else:
            for t in range(self._n_vars):
                inv_parents, pvals, accepted = self._run_icp_single(t)
                self._invariant_parents[t] = inv_parents
                self._p_values[t] = pvals
                self._accepted_sets[t] = accepted

        self._fitted = True
        return self

    def invariant_parents(self, target: int) -> Set[int]:
        """Return the invariant parent set for *target*."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return set(self._invariant_parents.get(target, set()))

    def invariant_set(self) -> Dict[int, Set[int]]:
        """Return invariant parent sets for all variables."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return {k: set(v) for k, v in self._invariant_parents.items()}

    def predict_invariant_parents(self, target: int) -> Set[int]:
        """Alias for invariant_parents."""
        return self.invariant_parents(target)

    def predict_plasticity_all_targets(
        self,
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Run ICP for all variables and classify edges.

        Returns
        -------
        Dict[Tuple[int, int], PlasticityClass]
            Edge-level plasticity classifications.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self._classify_from_invariant_parents()

    def predict_plasticity(
        self,
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Return plasticity classifications derived from ICP."""
        return self.predict_plasticity_all_targets()

    def p_values(self) -> Dict[int, Dict[str, float]]:
        """Return per-variable p-values from invariance tests."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._p_values)

    def accepted_sets(self) -> Dict[int, List[Tuple[List[int], float]]]:
        """Return accepted invariant sets with their p-values."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._accepted_sets)

    # ---------------------------------------------------------------
    # Core ICP algorithm
    # ---------------------------------------------------------------

    def _run_icp_single(
        self, target: int,
    ) -> Tuple[Set[int], Dict[str, float], List[Tuple[List[int], float]]]:
        """Run ICP for a single target variable.

        Returns (invariant_parents, p_values_dict, accepted_sets_list).
        """
        predictors = [j for j in range(self._n_vars) if j != target]
        max_size = self._max_set_size
        if max_size is None:
            max_size = min(len(predictors), 4)

        accepted_sets: List[Tuple[List[int], float]] = []
        all_p_values: Dict[str, float] = {}

        # Test each subset S of predictors
        for size in range(max_size + 1):
            for subset in itertools.combinations(predictors, size):
                subset_list = list(subset)
                p_val = self._test_invariance(subset_list, target)
                subset_key = str(sorted(subset_list))
                all_p_values[subset_key] = p_val

                if p_val > self._alpha:
                    accepted_sets.append((subset_list, p_val))

        # S* = intersection of all accepted sets
        invariant_parents = self._intersection_of_accepted(accepted_sets)

        return invariant_parents, all_p_values, accepted_sets

    def _test_invariance(
        self,
        parent_set: List[int],
        target: int,
    ) -> float:
        """Test whether parent_set gives invariant predictions for target.

        Combines the residual invariance test (Levene + KS) with a
        coefficient equality test (Chow-type F-test).  Returns the
        minimum p-value across all tests.
        """
        p_residual = self._residual_invariance_test(parent_set, target)
        p_coeff = _f_test_coefficients(
            self._datasets_list, parent_set, target,
        )
        return min(p_residual, p_coeff)

    def _residual_invariance_test(
        self,
        parent_set: List[int],
        target: int,
    ) -> float:
        """Compare residual distributions across environments.

        Computes OLS residuals of Y ~ X_S in each environment, then
        tests equality via Levene's test and pairwise KS tests.
        """
        residual_groups: List[NDArray] = []

        for data in self._datasets_list:
            y = data[:, target]
            if len(parent_set) == 0:
                X = np.empty((data.shape[0], 0))
            else:
                X = data[:, parent_set]
            residuals, _ = _ols_residuals(X, y)
            residual_groups.append(residuals)

        p_levene = _levene_test_across_contexts(residual_groups)
        p_ks = _ks_test_pairwise(residual_groups)

        # Combine: reject invariance if either test rejects
        return min(p_levene, p_ks)

    def _enumerate_subsets(
        self, variables: List[int], max_size: int,
    ) -> List[List[int]]:
        """Enumerate all subsets of *variables* up to *max_size*."""
        subsets: List[List[int]] = []
        for size in range(max_size + 1):
            for combo in itertools.combinations(variables, size):
                subsets.append(list(combo))
        return subsets

    def _intersection_of_accepted(
        self,
        accepted_sets: List[Tuple[List[int], float]],
    ) -> Set[int]:
        """Compute the intersection of all accepted parent sets.

        S* = ∩ { S : S is accepted by the invariance test }.
        If no set is accepted, return the empty set.
        """
        if len(accepted_sets) == 0:
            return set()

        # Start with the first accepted set, intersect with all others
        result: Optional[Set[int]] = None
        for subset, _ in accepted_sets:
            s = set(subset)
            if result is None:
                result = s
            else:
                result = result & s
        return result if result is not None else set()

    # ---------------------------------------------------------------
    # Plasticity classification from ICP results
    # ---------------------------------------------------------------

    def _classify_from_invariant_parents(
        self,
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Derive edge plasticity from invariant parent sets.

        Rules:
        - If i is an invariant parent of j → (i, j) is INVARIANT
        - If i is a parent in some regression but not invariant →
          STRUCTURAL_PLASTIC or PARAMETRIC_PLASTIC
        - If i appears as parent only in one context → EMERGENT
        """
        classifications: Dict[Tuple[int, int], PlasticityClass] = {}

        # Collect per-context regression significance to determine
        # which edges exist per context
        per_context_edges: Dict[str, Set[Tuple[int, int]]] = {}
        for ctx_key, data in zip(self._context_keys, self._datasets_list):
            edges: Set[Tuple[int, int]] = set()
            n = data.shape[0]
            for target in range(self._n_vars):
                for j in range(self._n_vars):
                    if j == target:
                        continue
                    # Simple marginal regression test
                    X = data[:, [j]]
                    X_aug = np.column_stack([X, np.ones(n)])
                    beta, _, _, _ = np.linalg.lstsq(
                        X_aug, data[:, target], rcond=None,
                    )
                    residuals = data[:, target] - X_aug @ beta
                    rss = float(np.sum(residuals ** 2))
                    tss = float(np.sum(
                        (data[:, target] - np.mean(data[:, target])) ** 2
                    ))
                    if tss < 1e-15:
                        continue
                    r2 = 1.0 - rss / tss
                    # F-test for significance
                    if n > 2 and r2 > 0:
                        f_stat = (r2 / 1.0) / ((1 - r2) / max(n - 2, 1))
                        p_val = 1.0 - sp_stats.f.cdf(f_stat, 1, n - 2)
                        if p_val < self._alpha:
                            edges.add((j, target))
            per_context_edges[ctx_key] = edges

        # Collect all edges across contexts
        all_edges: Set[Tuple[int, int]] = set()
        for edges in per_context_edges.values():
            all_edges |= edges

        n_ctx = len(self._context_keys)

        for i, j in all_edges:
            if (i, j) in classifications or (j, i) in classifications:
                continue

            is_invariant_parent = (
                j in self._invariant_parents
                and i in self._invariant_parents[j]
            )
            # Count contexts where edge appears
            n_present = sum(
                1 for ctx_key in self._context_keys
                if (i, j) in per_context_edges[ctx_key]
            )

            if is_invariant_parent:
                classifications[(i, j)] = PlasticityClass.INVARIANT
            elif n_present == 1:
                classifications[(i, j)] = PlasticityClass.EMERGENT
            elif n_present == n_ctx:
                classifications[(i, j)] = PlasticityClass.PARAMETRIC_PLASTIC
            else:
                classifications[(i, j)] = PlasticityClass.STRUCTURAL_PLASTIC

        return classifications
