"""Interventional query processing.

Implements truncated factorization, back-door and front-door adjustment
formulae, inverse-probability weighting, doubly-robust estimation,
instrumental-variable estimation, and average/conditional treatment
effect computation for estimating causal effects from observational data.

All estimators work with numpy arrays and use scipy for statistical
computations.  No external causal inference libraries are required.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


# ---------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------

@dataclass
class InterventionalQuery:
    """Specification of an interventional query do(X=x)."""

    target_vars: list[int] = field(default_factory=list)
    intervention_vars: list[int] = field(default_factory=list)
    intervention_values: list[float] = field(default_factory=list)
    conditioning_vars: list[int] = field(default_factory=list)


# ---------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------

def _parents_of(adj: NDArray, j: int) -> List[int]:
    return list(np.nonzero(adj[:, j])[0])


def _children_of(adj: NDArray, i: int) -> List[int]:
    return list(np.nonzero(adj[i, :])[0])


def _topological_sort(adj: NDArray) -> List[int]:
    p = adj.shape[0]
    binary = (adj != 0).astype(int)
    in_deg = binary.sum(axis=0).tolist()
    queue: deque[int] = deque(i for i in range(p) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for ch in range(p):
            if binary[node, ch]:
                in_deg[ch] -= 1
                if in_deg[ch] == 0:
                    queue.append(ch)
    if len(order) != p:
        raise ValueError("Graph contains a cycle")
    return order


def _ancestors_of(adj: NDArray, nodes: Set[int]) -> Set[int]:
    p = adj.shape[0]
    result: set[int] = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for par in range(p):
            if adj[par, n] != 0 and par not in result:
                result.add(par)
                stack.append(par)
    return result


def _descendants_of(adj: NDArray, nodes: Set[int]) -> Set[int]:
    p = adj.shape[0]
    result: set[int] = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for ch in range(p):
            if adj[n, ch] != 0 and ch not in result:
                result.add(ch)
                stack.append(ch)
    return result


def _d_separated(
    adj: NDArray, x: Set[int], y: Set[int], z: Set[int]
) -> bool:
    """Test d-separation X ⊥ Y | Z using Bayes-Ball."""
    if x & y:
        return False
    p = adj.shape[0]
    visited: set[tuple[int, str]] = set()
    queue: deque[tuple[int, str]] = deque()
    reachable: set[int] = set()
    for s in x:
        queue.append((s, "up"))
    while queue:
        node, direction = queue.popleft()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))
        if node not in x:
            reachable.add(node)
        if direction == "up" and node not in z:
            for par in _parents_of(adj, node):
                if (par, "up") not in visited:
                    queue.append((par, "up"))
            for ch in _children_of(adj, node):
                if (ch, "down") not in visited:
                    queue.append((ch, "down"))
        elif direction == "down":
            if node not in z:
                for ch in _children_of(adj, node):
                    if (ch, "down") not in visited:
                        queue.append((ch, "down"))
            if node in z:
                for par in _parents_of(adj, node):
                    if (par, "up") not in visited:
                        queue.append((par, "up"))
    return len(reachable & y) == 0


def _build_mutilated_graph(
    adj: NDArray, intervention_nodes: Set[int]
) -> NDArray[np.float64]:
    g = np.array(adj, dtype=np.float64).copy()
    for node in intervention_nodes:
        g[:, node] = 0
    return g


# ---------------------------------------------------------------
# Interventional Estimator
# ---------------------------------------------------------------

class InterventionalEstimator:
    """Estimator for interventional distributions.

    Provides back-door adjustment, front-door adjustment,
    instrumental-variable estimation, inverse-probability weighting,
    doubly-robust estimation, and average / conditional treatment
    effects.

    Parameters
    ----------
    alpha : float
        Significance level for confidence intervals.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    # -----------------------------------------------------------------
    # Truncated factorization
    # -----------------------------------------------------------------

    def truncated_factorization(
        self,
        scm: Any,
        intervention: InterventionalQuery,
    ) -> NDArray[np.float64]:
        """Compute the post-intervention covariance via truncated factorization.

        P(v \\\\ x | do(x)) = ∏_{i ∉ X} P(v_i | pa_i)

        For linear-Gaussian SCMs this amounts to computing the implied
        covariance of the mutilated model.

        Parameters
        ----------
        scm : StructuralCausalModel
        intervention : InterventionalQuery

        Returns
        -------
        ndarray
            Implied covariance matrix of the mutilated model.
        """
        if not intervention.intervention_vars:
            return scm.implied_covariance()

        interv_dict: dict[int, float] = {}
        for idx, val in zip(
            intervention.intervention_vars, intervention.intervention_values
        ):
            interv_dict[idx] = val

        mutilated = scm.do_intervention(interv_dict)
        return mutilated.implied_covariance()

    # -----------------------------------------------------------------
    # Find valid adjustment set (back-door criterion)
    # -----------------------------------------------------------------

    def _find_valid_adjustment_set(
        self,
        adj: NDArray[np.int_],
        treatment: int,
        outcome: int,
    ) -> Optional[Set[int]]:
        """Find a minimal valid adjustment set using the back-door criterion.

        A set Z satisfies the back-door criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X.
        2. Z blocks every path between X and Y that contains an arrow
           into X (i.e. back-door paths).

        We search among non-descendants of X, starting from small sets.

        Parameters
        ----------
        adj : ndarray
            Adjacency matrix.
        treatment : int
        outcome : int

        Returns
        -------
        set of int or None
        """
        adj = np.asarray(adj, dtype=np.float64)
        p = adj.shape[0]
        desc_x = _descendants_of(adj, {treatment})
        forbidden = desc_x | {treatment, outcome}
        candidates = sorted(set(range(p)) - forbidden)

        g_mut = _build_mutilated_graph(adj, {treatment})

        # Try empty set
        if _d_separated(g_mut, {treatment}, {outcome}, set()):
            return set()

        # Try single variables
        for c in candidates:
            z = {c}
            if _d_separated(g_mut, {treatment}, {outcome}, z):
                return z

        # Try pairs
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                z = {candidates[i], candidates[j]}
                if _d_separated(g_mut, {treatment}, {outcome}, z):
                    return z

        # Try parents of treatment (often the minimal set)
        parents_x = set(_parents_of(adj, treatment))
        valid_parents = parents_x - forbidden
        if valid_parents and _d_separated(
            g_mut, {treatment}, {outcome}, valid_parents
        ):
            return valid_parents

        # Try full candidate set
        full = set(candidates)
        if full and _d_separated(g_mut, {treatment}, {outcome}, full):
            return full

        return None

    # -----------------------------------------------------------------
    # Back-door adjustment
    # -----------------------------------------------------------------

    def backdoor_adjustment(
        self,
        graph: NDArray[np.int_],
        treatment: int,
        outcome: int,
        adjustment_set: set[int],
        data: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Estimate causal effect using the back-door adjustment formula.

        E[Y | do(X=x)] = Σ_z E[Y | X=x, Z=z] P(Z=z)

        For continuous data, implements this via OLS regression of Y on
        X and Z, returning the coefficient of X as the causal effect
        estimate.

        Parameters
        ----------
        graph : ndarray
            Adjacency matrix.
        treatment : int
            Treatment variable index.
        outcome : int
            Outcome variable index.
        adjustment_set : set of int
            Back-door adjustment variables.
        data : ndarray
            Observational data of shape ``(n, p)``.

        Returns
        -------
        (effect, se) : tuple of float
            Estimated causal effect and standard error.
        """
        data = np.asarray(data, dtype=np.float64)
        n, p = data.shape

        if treatment < 0 or treatment >= p:
            raise ValueError(f"treatment {treatment} out of range [0, {p})")
        if outcome < 0 or outcome >= p:
            raise ValueError(f"outcome {outcome} out of range [0, {p})")

        # Build design matrix: [X, Z_1, ..., Z_k, 1]
        z_list = sorted(adjustment_set - {treatment, outcome})
        col_indices = [treatment] + z_list
        X_design = data[:, col_indices]
        X_design = np.column_stack([X_design, np.ones(n)])

        y = data[:, outcome]

        # OLS
        beta, residuals, rank, sv = np.linalg.lstsq(X_design, y, rcond=None)

        effect = float(beta[0])  # coefficient of treatment

        # Standard error
        y_hat = X_design @ beta
        resid = y - y_hat
        dof = max(1, n - X_design.shape[1])
        sigma2 = float(np.sum(resid ** 2) / dof)

        try:
            cov_beta = sigma2 * np.linalg.inv(X_design.T @ X_design)
            se = float(np.sqrt(max(cov_beta[0, 0], 0)))
        except np.linalg.LinAlgError:
            se = float("nan")

        return (effect, se)

    # -----------------------------------------------------------------
    # Front-door adjustment
    # -----------------------------------------------------------------

    def frontdoor_adjustment(
        self,
        graph: NDArray[np.int_],
        treatment: int,
        outcome: int,
        mediator_set: set[int],
        data: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Estimate causal effect using the front-door adjustment formula.

        E[Y | do(X=x)] = Σ_m P(M=m | X=x) Σ_{x'} P(Y | M=m, X=x') P(X=x')

        For continuous data, implements via two-stage regression:
        Stage 1: regress M on X → β_XM
        Stage 2: regress Y on M controlling for X → β_MY
        Causal effect = β_XM × β_MY

        Parameters
        ----------
        graph : ndarray
            Adjacency matrix.
        treatment : int
        outcome : int
        mediator_set : set of int
            Mediator variables.
        data : ndarray
            Observational data.

        Returns
        -------
        (effect, se) : tuple of float
        """
        data = np.asarray(data, dtype=np.float64)
        n, p = data.shape
        mediators = sorted(mediator_set)

        if not mediators:
            raise ValueError("mediator_set cannot be empty")

        total_effect = 0.0
        total_se_sq = 0.0

        for med in mediators:
            # Stage 1: M = α_0 + α_1 X + ε
            X1 = np.column_stack([data[:, treatment], np.ones(n)])
            y1 = data[:, med]
            beta1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
            alpha_xm = float(beta1[0])

            resid1 = y1 - X1 @ beta1
            dof1 = max(1, n - 2)
            se1_sq = float(np.sum(resid1 ** 2) / dof1)
            try:
                var_alpha = se1_sq * np.linalg.inv(X1.T @ X1)[0, 0]
            except np.linalg.LinAlgError:
                var_alpha = float("nan")

            # Stage 2: Y = β_0 + β_1 M + β_2 X + ε
            X2 = np.column_stack([data[:, med], data[:, treatment], np.ones(n)])
            y2 = data[:, outcome]
            beta2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
            beta_my = float(beta2[0])

            resid2 = y2 - X2 @ beta2
            dof2 = max(1, n - 3)
            se2_sq = float(np.sum(resid2 ** 2) / dof2)
            try:
                var_beta = se2_sq * np.linalg.inv(X2.T @ X2)[0, 0]
            except np.linalg.LinAlgError:
                var_beta = float("nan")

            # Product of coefficients
            total_effect += alpha_xm * beta_my

            # Delta method SE: Var(αβ) ≈ α²Var(β) + β²Var(α)
            if not (math.isnan(var_alpha) or math.isnan(var_beta)):
                total_se_sq += (
                    alpha_xm ** 2 * var_beta + beta_my ** 2 * var_alpha
                )
            else:
                total_se_sq = float("nan")

        se = (
            float(np.sqrt(total_se_sq))
            if not math.isnan(total_se_sq)
            else float("nan")
        )
        return (total_effect, se)

    # -----------------------------------------------------------------
    # Instrumental variable estimation
    # -----------------------------------------------------------------

    def instrumental_variable(
        self,
        graph: NDArray[np.int_],
        instrument: int,
        treatment: int,
        outcome: int,
        data: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Estimate causal effect using instrumental variable (IV / 2SLS).

        Two-stage least squares:
        Stage 1: X̂ = π₀ + π₁ Z
        Stage 2: Y = β₀ + β₁ X̂

        The IV estimate is β₁ = Cov(Y, Z) / Cov(X, Z).

        Parameters
        ----------
        graph : ndarray
            Adjacency matrix.
        instrument : int
            Instrument variable index.
        treatment : int
            Treatment variable index.
        outcome : int
            Outcome variable index.
        data : ndarray
            Observational data.

        Returns
        -------
        (effect, se) : tuple of float
        """
        data = np.asarray(data, dtype=np.float64)
        n = data.shape[0]

        z = data[:, instrument]
        x = data[:, treatment]
        y = data[:, outcome]

        # Wald / IV estimate
        cov_yz = float(np.cov(y, z, ddof=1)[0, 1])
        cov_xz = float(np.cov(x, z, ddof=1)[0, 1])

        if abs(cov_xz) < 1e-12:
            raise ValueError(
                "Weak instrument: Cov(X, Z) ≈ 0; IV estimate undefined"
            )

        iv_estimate = cov_yz / cov_xz

        # 2SLS for proper SE
        # Stage 1: regress X on Z
        Z1 = np.column_stack([z, np.ones(n)])
        beta1, _, _, _ = np.linalg.lstsq(Z1, x, rcond=None)
        x_hat = Z1 @ beta1

        # Stage 2: regress Y on X̂
        X2 = np.column_stack([x_hat, np.ones(n)])
        beta2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)

        # SE using original X residuals
        resid = y - np.column_stack([x, np.ones(n)]) @ np.array(
            [beta2[0], beta2[1]]
        )
        dof = max(1, n - 2)
        sigma2 = float(np.sum(resid ** 2) / dof)
        try:
            var_beta = sigma2 * np.linalg.inv(X2.T @ X2)[0, 0]
            se = float(np.sqrt(max(var_beta, 0)))
        except np.linalg.LinAlgError:
            se = float("nan")

        return (iv_estimate, se)

    # -----------------------------------------------------------------
    # Inverse-probability weighting
    # -----------------------------------------------------------------

    def ip_weighting(
        self,
        treatment: NDArray[np.float64],
        outcome: NDArray[np.float64],
        propensity_scores: NDArray[np.float64],
        data: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Inverse-probability weighting (IPW) estimator.

        ATE_IPW = (1/n) Σ [ T_i Y_i / e(X_i) - (1-T_i) Y_i / (1-e(X_i)) ]

        where e(X_i) is the propensity score P(T=1 | X_i).

        Parameters
        ----------
        treatment : ndarray
            Binary treatment indicator (0/1), shape ``(n,)``.
        outcome : ndarray
            Outcome variable, shape ``(n,)``.
        propensity_scores : ndarray
            Propensity scores e(X), shape ``(n,)``.
        data : ndarray
            Full data matrix (unused but kept for API consistency).

        Returns
        -------
        (ate, se) : tuple of float
            IPW estimate of ATE and standard error.
        """
        treatment = np.asarray(treatment, dtype=np.float64).ravel()
        outcome = np.asarray(outcome, dtype=np.float64).ravel()
        ps = np.asarray(propensity_scores, dtype=np.float64).ravel()
        n = len(treatment)

        if n == 0:
            raise ValueError("Empty data")
        if len(outcome) != n or len(ps) != n:
            raise ValueError("Input arrays must have same length")

        # Clip propensity scores to avoid extreme weights
        eps = 1e-6
        ps_clipped = np.clip(ps, eps, 1.0 - eps)

        # IPW estimator
        w1 = treatment / ps_clipped
        w0 = (1.0 - treatment) / (1.0 - ps_clipped)

        mu1 = np.sum(w1 * outcome) / np.sum(w1) if np.sum(w1) > 0 else 0.0
        mu0 = np.sum(w0 * outcome) / np.sum(w0) if np.sum(w0) > 0 else 0.0

        ate = float(mu1 - mu0)

        # SE via influence function
        phi = w1 * (outcome - mu1) - w0 * (outcome - mu0)
        se = float(np.std(phi, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

        return (ate, se)

    # -----------------------------------------------------------------
    # Doubly-robust estimator
    # -----------------------------------------------------------------

    def doubly_robust(
        self,
        treatment_col: NDArray[np.float64],
        outcome_col: NDArray[np.float64],
        propensity_scores: NDArray[np.float64],
        outcome_model_pred: NDArray[np.float64],
        data: NDArray[np.float64],
        *,
        outcome_model_pred_0: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[float, float]:
        """Doubly-robust (augmented IPW) estimator.

        ATE_DR = (1/n) Σ [ μ̂₁(X_i) - μ̂₀(X_i)
                 + T_i(Y_i - μ̂₁(X_i))/e(X_i)
                 - (1-T_i)(Y_i - μ̂₀(X_i))/(1-e(X_i)) ]

        Consistent if either the propensity model *or* the outcome model
        is correctly specified.

        Parameters
        ----------
        treatment_col : ndarray
            Binary treatment (0/1).
        outcome_col : ndarray
            Observed outcome.
        propensity_scores : ndarray
            P(T=1 | X).
        outcome_model_pred : ndarray
            μ̂₁(X): predicted outcome under treatment.
        data : ndarray
            Full data matrix.
        outcome_model_pred_0 : ndarray, optional
            μ̂₀(X): predicted outcome under control.  If None, set to 0.

        Returns
        -------
        (ate, se) : tuple of float
        """
        t = np.asarray(treatment_col, dtype=np.float64).ravel()
        y = np.asarray(outcome_col, dtype=np.float64).ravel()
        ps = np.asarray(propensity_scores, dtype=np.float64).ravel()
        mu1 = np.asarray(outcome_model_pred, dtype=np.float64).ravel()
        n = len(t)

        if outcome_model_pred_0 is not None:
            mu0 = np.asarray(outcome_model_pred_0, dtype=np.float64).ravel()
        else:
            mu0 = np.zeros(n, dtype=np.float64)

        eps = 1e-6
        ps_c = np.clip(ps, eps, 1.0 - eps)

        # DR estimator
        dr_scores = (
            mu1
            - mu0
            + t * (y - mu1) / ps_c
            - (1.0 - t) * (y - mu0) / (1.0 - ps_c)
        )

        ate = float(np.mean(dr_scores))
        se = float(np.std(dr_scores, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

        return (ate, se)

    # -----------------------------------------------------------------
    # Average Treatment Effect
    # -----------------------------------------------------------------

    def compute_ate(
        self,
        scm: Any,
        treatment: int,
        outcome: int,
        data: Optional[NDArray[np.float64]] = None,
        *,
        n_samples: int = 10_000,
    ) -> Tuple[float, float]:
        """Compute the Average Treatment Effect (ATE).

        ATE = E[Y | do(X=1)] - E[Y | do(X=0)]

        Uses the back-door adjustment formula if a valid adjustment set
        exists; otherwise falls back to simulation from the SCM.

        Parameters
        ----------
        scm : StructuralCausalModel
        treatment, outcome : int
        data : ndarray, optional
            Observational data.  Generated from SCM if None.
        n_samples : int

        Returns
        -------
        (ate, se) : tuple of float
        """
        adj = scm.adjacency_matrix
        p = scm.num_variables

        if treatment < 0 or treatment >= p:
            raise ValueError(f"treatment {treatment} out of range [0, {p})")
        if outcome < 0 or outcome >= p:
            raise ValueError(f"outcome {outcome} out of range [0, {p})")

        # Try analytic computation for linear-Gaussian SCMs
        ate_analytic = self._ate_analytic(scm, treatment, outcome)
        if ate_analytic is not None:
            return ate_analytic

        # Try back-door adjustment with data
        if data is None:
            data = scm.sample(n_samples)

        adj_set = self._find_valid_adjustment_set(adj, treatment, outcome)
        if adj_set is not None:
            return self.backdoor_adjustment(
                adj, treatment, outcome, adj_set, data
            )

        # Fallback: simulation
        return self._ate_simulation(scm, treatment, outcome, n_samples)

    def _ate_analytic(
        self, scm: Any, treatment: int, outcome: int
    ) -> Optional[Tuple[float, float]]:
        """Analytic ATE for linear-Gaussian SCMs.

        In a linear SCM, ATE = total causal effect = sum of products of
        coefficients along all directed paths from treatment to outcome.
        """
        try:
            adj = scm.adjacency_matrix
            coefs = scm.regression_coefficients
            p = scm.num_variables

            # Compute total effect via (I - B)^{-1}
            I = np.eye(p, dtype=np.float64)
            B = coefs.T  # B[j,i] = coef of i in eq for j
            IminusB = I - B
            total = np.linalg.inv(IminusB)
            # total[outcome, treatment] gives the total causal effect
            ate = float(total[outcome, treatment])

            # SE from residual variances (approximation)
            resid = scm.residual_variances
            n = max(scm.sample_size, 1)
            se = float(np.sqrt(resid[outcome] / n))

            return (ate, se)
        except Exception:
            return None

    def _ate_simulation(
        self,
        scm: Any,
        treatment: int,
        outcome: int,
        n_samples: int,
    ) -> Tuple[float, float]:
        """ATE by simulating from do(X=1) and do(X=0)."""
        data_do1 = scm.sample(n_samples, interventions={treatment: 1.0})
        data_do0 = scm.sample(n_samples, interventions={treatment: 0.0})

        y1 = data_do1[:, outcome]
        y0 = data_do0[:, outcome]
        ate = float(np.mean(y1) - np.mean(y0))
        se = float(np.sqrt(np.var(y1, ddof=1) / n_samples + np.var(y0, ddof=1) / n_samples))
        return (ate, se)

    # -----------------------------------------------------------------
    # Conditional Average Treatment Effect
    # -----------------------------------------------------------------

    def compute_cate(
        self,
        scm: Any,
        treatment: int,
        outcome: int,
        covariates: Dict[int, float],
        data: Optional[NDArray[np.float64]] = None,
        *,
        n_samples: int = 10_000,
        bandwidth: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute the Conditional Average Treatment Effect (CATE).

        CATE(x_cov) = E[Y | do(X=1), X_cov=x_cov] - E[Y | do(X=0), X_cov=x_cov]

        Uses kernel weighting around the covariate values.

        Parameters
        ----------
        scm : StructuralCausalModel
        treatment, outcome : int
        covariates : dict
            ``{variable_index: value}`` conditioning values.
        data : ndarray, optional
        n_samples : int
        bandwidth : float, optional
            Kernel bandwidth.  If None, uses Silverman's rule.

        Returns
        -------
        (cate, se) : tuple of float
        """
        p = scm.num_variables
        if data is None:
            data = scm.sample(n_samples)

        n = data.shape[0]
        cov_indices = sorted(covariates.keys())

        if not cov_indices:
            return self.compute_ate(scm, treatment, outcome, data)

        # Compute kernel weights
        if bandwidth is None:
            # Silverman's rule
            std_vals = [
                max(float(np.std(data[:, c])), 1e-6) for c in cov_indices
            ]
            bw = np.mean(std_vals) * (n ** (-1 / (len(cov_indices) + 4)))
            bandwidth = max(bw, 1e-6)

        dists = np.zeros(n, dtype=np.float64)
        for c in cov_indices:
            dists += ((data[:, c] - covariates[c]) / bandwidth) ** 2
        weights = np.exp(-0.5 * dists)
        weights /= max(np.sum(weights), 1e-12)

        # Weighted regression: Y = β₀ + β₁ X + Σ β_c X_c + ε
        adj = scm.adjacency_matrix
        adj_set = self._find_valid_adjustment_set(adj, treatment, outcome)
        z_list = sorted((adj_set or set()) - {treatment, outcome})

        col_indices = [treatment] + z_list
        X_design = np.column_stack([data[:, col_indices], np.ones(n)])
        y = data[:, outcome]

        # Weighted least squares
        W = np.diag(np.sqrt(weights))
        Xw = W @ X_design
        yw = W @ y

        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        cate = float(beta[0])

        # SE
        y_hat = X_design @ beta
        resid = y - y_hat
        dof = max(1, n - X_design.shape[1])
        sigma2 = float(np.sum(weights * resid ** 2) / max(np.sum(weights), 1e-12))
        try:
            XwX = X_design.T @ np.diag(weights) @ X_design
            cov_beta = sigma2 * np.linalg.inv(XwX)
            se = float(np.sqrt(max(cov_beta[0, 0], 0)))
        except np.linalg.LinAlgError:
            se = float("nan")

        return (cate, se)

    # -----------------------------------------------------------------
    # Adjustment formula (generic)
    # -----------------------------------------------------------------

    def adjustment_formula(
        self,
        scm: Any,
        treatment: int,
        outcome: int,
        adjustment_set: Optional[Set[int]] = None,
        data: Optional[NDArray[np.float64]] = None,
        *,
        n_samples: int = 10_000,
    ) -> Tuple[float, float]:
        """Estimate causal effect via the adjustment formula.

        If *adjustment_set* is None, automatically finds a valid one.

        Parameters
        ----------
        scm : StructuralCausalModel
        treatment, outcome : int
        adjustment_set : set of int, optional
        data : ndarray, optional
        n_samples : int

        Returns
        -------
        (effect, se) : tuple of float
        """
        adj = scm.adjacency_matrix
        if adjustment_set is None:
            adjustment_set = self._find_valid_adjustment_set(
                adj, treatment, outcome
            )
            if adjustment_set is None:
                raise ValueError(
                    "No valid adjustment set exists for the back-door criterion"
                )

        if data is None:
            data = scm.sample(n_samples)

        return self.backdoor_adjustment(
            adj, treatment, outcome, adjustment_set, data
        )
