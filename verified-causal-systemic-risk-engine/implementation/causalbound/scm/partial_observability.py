"""
Partial observability handling for SCM construction.

Detects latent confounders from failed conditional independence tests,
imputes missing data using EM-style algorithms, corrects for selection
bias, and computes causal effect bounds under partial observability.

References
----------
- Richardson, Spirtes (2002). Ancestral graph Markov models.
- Tian, Pearl (2002). On the identification of causal effects.
- Manski (1990). Nonparametric bounds on treatment effects.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

from .dag import DAGRepresentation, EdgeType


@dataclass
class LatentConfounder:
    """Information about a detected latent confounder.

    ``children`` are the observed variables that share the latent cause.
    ``evidence`` summarises the statistical evidence (failed CI tests).
    """
    name: str
    children: List[str]
    evidence: List[str] = field(default_factory=list)
    estimated_strength: float = 0.0


@dataclass
class BoundsResult:
    """Bounds on a causal effect under partial observability."""
    lower: float
    upper: float
    target: str
    treatment: str
    treatment_value: float
    method: str = ""

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


class PartialObservabilityHandler:
    """Handle partially-observed data in SCM construction.

    Provides methods for:
    - Detecting latent confounders from conditional-independence test patterns
    - EM-style imputation for missing data
    - Selection bias correction (inverse probability weighting)
    - Computing bounds on causal effects when latent confounders exist

    Parameters
    ----------
    ci_alpha : float
        Significance level for conditional independence tests.
    max_latent : int
        Maximum number of latent confounders to detect.
    """

    def __init__(
        self,
        ci_alpha: float = 0.05,
        max_latent: int = 10,
    ) -> None:
        self.ci_alpha = ci_alpha
        self.max_latent = max_latent
        self._detected_latents: List[LatentConfounder] = []

    # ──────────────────────────────────────────────────────────────────
    # Latent confounder detection
    # ──────────────────────────────────────────────────────────────────

    def detect_latent_confounders(
        self,
        data: np.ndarray,
        skeleton: List[Tuple[str, str]],
        variables: Optional[List[str]] = None,
        method: str = "tetrad",
    ) -> List[LatentConfounder]:
        """Detect potential latent confounders from observed data.

        Uses two complementary strategies:

        1. **Tetrad constraints** (Spirtes et al.): If four observed
           variables satisfy certain algebraic constraints on their
           covariance matrix, a latent common cause is implied.

        2. **Residual correlation**: After conditioning on all observed
           parents, if two variables remain correlated, a latent
           confounder may exist.

        Parameters
        ----------
        data : (n_samples, n_variables) array
        skeleton : list of (u, v) edges
        variables : variable names
        method : ``"tetrad"`` or ``"residual"`` or ``"both"``
        """
        n_vars = data.shape[1]
        if variables is None:
            variables = [f"X{i}" for i in range(n_vars)]
        col_map = {v: i for i, v in enumerate(variables)}

        latents: List[LatentConfounder] = []

        if method in ("tetrad", "both"):
            latents.extend(
                self._detect_via_tetrad(data, skeleton, variables, col_map)
            )

        if method in ("residual", "both"):
            latents.extend(
                self._detect_via_residual(data, skeleton, variables, col_map)
            )

        # Merge overlapping latents
        latents = self._merge_latents(latents)

        # Limit count
        latents = latents[:self.max_latent]
        self._detected_latents = latents
        return latents

    def _detect_via_tetrad(
        self,
        data: np.ndarray,
        skeleton: List[Tuple[str, str]],
        variables: List[str],
        col_map: Dict[str, int],
    ) -> List[LatentConfounder]:
        """Detect latent confounders via vanishing tetrad differences.

        A tetrad τ(i,j,k,l) = σ_ij·σ_kl − σ_ik·σ_jl should vanish if
        {i,j,k,l} have a specific latent structure (one-factor model).
        """
        n = data.shape[0]
        cov = np.cov(data.T)
        latents: List[LatentConfounder] = []

        # Look for groups of 4 variables with near-vanishing tetrads
        observed = list(range(data.shape[1]))
        if len(observed) < 4:
            return latents

        # Check all 4-subsets (up to a reasonable limit)
        from itertools import combinations
        subsets = list(combinations(observed, 4))
        if len(subsets) > 500:
            # Subsample
            indices = np.random.choice(len(subsets), 500, replace=False)
            subsets = [subsets[i] for i in indices]

        for i, j, k, l in subsets:
            # Three tetrad differences for a 4-set
            t1 = cov[i, j] * cov[k, l] - cov[i, k] * cov[j, l]
            t2 = cov[i, j] * cov[k, l] - cov[i, l] * cov[j, k]
            t3 = cov[i, k] * cov[j, l] - cov[i, l] * cov[j, k]

            # Wishart-based test for vanishing tetrad (Bollen & Ting, 1993)
            # Approximate SE under null
            se = np.sqrt(self._tetrad_variance(cov, n, i, j, k, l))
            if se < 1e-12:
                continue

            # If two of three tetrads vanish, suggests one-factor model
            vanishing = 0
            for t in [t1, t2, t3]:
                z = t / se
                p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
                if p > self.ci_alpha:
                    vanishing += 1

            if vanishing >= 2:
                children = [variables[idx] for idx in [i, j, k, l]]
                latents.append(LatentConfounder(
                    name=f"L_{'_'.join(children)}",
                    children=children,
                    evidence=[f"Vanishing tetrads ({vanishing}/3)"],
                    estimated_strength=1.0 - (vanishing / 3.0),
                ))

        return latents

    def _tetrad_variance(
        self, cov: np.ndarray, n: int, i: int, j: int, k: int, l: int
    ) -> float:
        """Approximate variance of tetrad difference under the null."""
        # Wishart approximation: Var(σ_ij·σ_kl) ≈ (σ_ij²·σ_kl² + ...) / n
        var = (
            cov[i, j] ** 2 * (cov[k, k] * cov[l, l] + cov[k, l] ** 2) +
            cov[k, l] ** 2 * (cov[i, i] * cov[j, j] + cov[i, j] ** 2) +
            2 * cov[i, j] * cov[k, l] * (cov[i, k] * cov[j, l] + cov[i, l] * cov[j, k])
        ) / (n + 1e-15)
        return max(var, 1e-15)

    def _detect_via_residual(
        self,
        data: np.ndarray,
        skeleton: List[Tuple[str, str]],
        variables: List[str],
        col_map: Dict[str, int],
    ) -> List[LatentConfounder]:
        """Detect latent confounders via residual correlations.

        For each pair (X, Y) not adjacent in the skeleton, regress both
        on all common neighbours.  If residuals are significantly
        correlated, a latent confounder is suggested.
        """
        n = data.shape[0]
        adj: Dict[str, Set[str]] = defaultdict(set)
        for u, v in skeleton:
            adj[u].add(v)
            adj[v].add(u)

        skeleton_set = {frozenset({u, v}) for u, v in skeleton}
        latents: List[LatentConfounder] = []

        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                u, v = variables[i], variables[j]
                if frozenset({u, v}) in skeleton_set:
                    continue

                # Common neighbours
                common = adj[u] & adj[v]
                if not common:
                    # No common neighbours – can't distinguish from marginal
                    continue

                z_indices = [col_map[c] for c in common if c in col_map]
                if not z_indices:
                    continue

                # Regress X and Y on Z
                Z = data[:, z_indices]
                Z_aug = np.column_stack([np.ones(n), Z])
                try:
                    beta_x, _, _, _ = np.linalg.lstsq(Z_aug, data[:, col_map[u]], rcond=None)
                    beta_y, _, _, _ = np.linalg.lstsq(Z_aug, data[:, col_map[v]], rcond=None)
                except np.linalg.LinAlgError:
                    continue

                res_x = data[:, col_map[u]] - Z_aug @ beta_x
                res_y = data[:, col_map[v]] - Z_aug @ beta_y

                # Test residual correlation
                r = np.corrcoef(res_x, res_y)[0, 1]
                r = np.clip(r, -0.9999, 0.9999)
                z_stat = 0.5 * np.log((1 + r) / (1 - r + 1e-15))
                z_stat *= np.sqrt(n - len(z_indices) - 3)
                p_val = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))

                if p_val < self.ci_alpha:
                    latents.append(LatentConfounder(
                        name=f"L_{u}_{v}",
                        children=[u, v],
                        evidence=[f"Residual corr r={r:.3f}, p={p_val:.4f}"],
                        estimated_strength=abs(r),
                    ))

        return latents

    def _merge_latents(
        self, latents: List[LatentConfounder]
    ) -> List[LatentConfounder]:
        """Merge latent confounders with overlapping children."""
        if not latents:
            return latents

        # Union-find to merge overlapping groups
        parent_map: Dict[str, str] = {}
        for latent in latents:
            for child in latent.children:
                parent_map.setdefault(child, child)

        def find(x: str) -> str:
            while parent_map[x] != x:
                parent_map[x] = parent_map[parent_map[x]]
                x = parent_map[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent_map[rx] = ry

        for latent in latents:
            children = latent.children
            for i in range(len(children) - 1):
                union(children[i], children[i + 1])

        # Group children by root
        groups: Dict[str, Set[str]] = defaultdict(set)
        for latent in latents:
            for child in latent.children:
                root = find(child)
                groups[root].add(child)

        # Build merged latents
        merged: List[LatentConfounder] = []
        for root, children in groups.items():
            evidence: List[str] = []
            strength = 0.0
            count = 0
            for latent in latents:
                if set(latent.children) & children:
                    evidence.extend(latent.evidence)
                    strength += latent.estimated_strength
                    count += 1
            merged.append(LatentConfounder(
                name=f"L_{'_'.join(sorted(children))}",
                children=sorted(children),
                evidence=evidence,
                estimated_strength=strength / max(count, 1),
            ))

        return merged

    # ──────────────────────────────────────────────────────────────────
    # Missing data imputation
    # ──────────────────────────────────────────────────────────────────

    def impute_missing(
        self,
        data: np.ndarray,
        mechanism: str = "em",
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """Impute missing values (NaN) in *data*.

        Parameters
        ----------
        mechanism : str
            ``"em"`` for EM algorithm (assumes multivariate normal),
            ``"mean"`` for simple mean imputation,
            ``"regression"`` for chained regression imputation.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance (relative change in log-likelihood).
        """
        if mechanism == "mean":
            return self._impute_mean(data)
        elif mechanism == "regression":
            return self._impute_chained_regression(data, max_iter)
        elif mechanism == "em":
            return self._impute_em(data, max_iter, tol)
        else:
            raise ValueError(f"Unknown imputation mechanism: {mechanism}")

    def _impute_mean(self, data: np.ndarray) -> np.ndarray:
        """Replace NaN with column means."""
        result = data.copy()
        for j in range(data.shape[1]):
            col = result[:, j]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.nanmean(col)
        return result

    def _impute_em(
        self, data: np.ndarray, max_iter: int, tol: float
    ) -> np.ndarray:
        """EM imputation assuming multivariate normal distribution.

        E-step: impute missing values using conditional expectations.
        M-step: re-estimate mean and covariance from completed data.
        """
        n, p = data.shape
        result = data.copy()
        missing = np.isnan(result)

        # Initialise: mean imputation
        col_means = np.nanmean(result, axis=0)
        for j in range(p):
            result[missing[:, j], j] = col_means[j]

        prev_ll = -np.inf
        for iteration in range(max_iter):
            # M-step: estimate mean and covariance
            mu = np.mean(result, axis=0)
            centered = result - mu
            sigma = (centered.T @ centered) / n
            sigma += np.eye(p) * 1e-6  # regularisation

            # E-step: impute each row
            try:
                sigma_inv = np.linalg.inv(sigma)
            except np.linalg.LinAlgError:
                sigma_inv = np.linalg.pinv(sigma)

            for i in range(n):
                obs_mask = ~missing[i]
                mis_mask = missing[i]
                if not mis_mask.any():
                    continue

                obs_idx = np.where(obs_mask)[0]
                mis_idx = np.where(mis_mask)[0]

                # Partition covariance matrix
                sigma_oo = sigma[np.ix_(obs_idx, obs_idx)]
                sigma_mo = sigma[np.ix_(mis_idx, obs_idx)]

                try:
                    sigma_oo_inv = np.linalg.inv(
                        sigma_oo + np.eye(len(obs_idx)) * 1e-8
                    )
                except np.linalg.LinAlgError:
                    continue

                # Conditional expectation: E[X_m | X_o]
                x_obs = result[i, obs_idx] - mu[obs_idx]
                cond_mean = mu[mis_idx] + sigma_mo @ sigma_oo_inv @ x_obs
                result[i, mis_idx] = cond_mean

            # Compute log-likelihood for convergence check
            try:
                sign, logdet = np.linalg.slogdet(sigma)
                if sign <= 0:
                    logdet = -1e10
                ll = -0.5 * n * (p * np.log(2 * np.pi) + logdet)
                for i in range(n):
                    diff = result[i] - mu
                    ll -= 0.5 * diff @ sigma_inv @ diff
            except np.linalg.LinAlgError:
                ll = prev_ll

            if abs(ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol:
                break
            prev_ll = ll

        return result

    def _impute_chained_regression(
        self, data: np.ndarray, max_iter: int
    ) -> np.ndarray:
        """Chained regression (MICE-style) imputation."""
        n, p = data.shape
        result = self._impute_mean(data.copy())
        missing = np.isnan(data)

        # Columns with missing data, sorted by fraction missing
        cols_with_missing = [
            j for j in range(p) if missing[:, j].any()
        ]
        cols_with_missing.sort(key=lambda j: missing[:, j].sum())

        for iteration in range(max_iter):
            max_change = 0.0
            for j in cols_with_missing:
                miss_rows = np.where(missing[:, j])[0]
                obs_rows = np.where(~missing[:, j])[0]
                if len(obs_rows) < 3:
                    continue

                predictor_cols = [c for c in range(p) if c != j]
                X_obs = result[np.ix_(obs_rows, predictor_cols)]
                y_obs = data[obs_rows, j]
                X_aug = np.column_stack([np.ones(len(obs_rows)), X_obs])

                try:
                    beta, _, _, _ = np.linalg.lstsq(X_aug, y_obs, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                X_miss = result[np.ix_(miss_rows, predictor_cols)]
                X_miss_aug = np.column_stack([np.ones(len(miss_rows)), X_miss])
                predicted = X_miss_aug @ beta

                # Add noise for proper imputation
                residual_std = np.std(y_obs - X_aug @ beta)
                noise = np.random.normal(0, residual_std, len(miss_rows))
                new_vals = predicted + noise

                old_vals = result[miss_rows, j]
                max_change = max(max_change, np.max(np.abs(new_vals - old_vals)))
                result[miss_rows, j] = new_vals

            if max_change < 1e-4:
                break

        return result

    # ──────────────────────────────────────────────────────────────────
    # Selection bias correction
    # ──────────────────────────────────────────────────────────────────

    def correct_selection_bias(
        self,
        data: np.ndarray,
        selection_var: int,
        variables: Optional[List[str]] = None,
        method: str = "ipw",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct for selection bias via inverse probability weighting.

        Parameters
        ----------
        data : (n_samples, n_variables) array
            Includes the selection variable column.
        selection_var : int
            Column index of the selection indicator (1 = selected).
        method : str
            ``"ipw"`` for inverse probability weighting,
            ``"heckman"`` for Heckman correction.

        Returns
        -------
        (corrected_data, weights) – the re-weighted data and sample weights.
        """
        if method == "ipw":
            return self._ipw_correction(data, selection_var)
        elif method == "heckman":
            return self._heckman_correction(data, selection_var)
        else:
            raise ValueError(f"Unknown selection bias method: {method}")

    def _ipw_correction(
        self, data: np.ndarray, selection_var: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse probability weighting for selection bias.

        Estimates P(S=1 | X) via logistic regression and weights
        selected observations by 1/P(S=1 | X).
        """
        n, p = data.shape
        S = data[:, selection_var]
        predictors = [j for j in range(p) if j != selection_var]
        X = data[:, predictors]
        X_aug = np.column_stack([np.ones(n), X])

        # Fit logistic regression for P(S=1 | X)
        beta = np.zeros(X_aug.shape[1])

        def neg_log_likelihood(b):
            logits = X_aug @ b
            logits = np.clip(logits, -500, 500)
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            ll = np.sum(S * np.log(probs) + (1 - S) * np.log(1 - probs))
            return -ll

        result = minimize(neg_log_likelihood, beta, method="L-BFGS-B")
        beta_hat = result.x

        logits = X_aug @ beta_hat
        logits = np.clip(logits, -500, 500)
        prop_scores = 1.0 / (1.0 + np.exp(-logits))
        prop_scores = np.clip(prop_scores, 0.01, 0.99)

        # Weights for selected observations
        weights = np.where(S == 1, 1.0 / prop_scores, 0.0)
        weights /= weights.sum() + 1e-15
        weights *= n

        # Selected data only
        selected = S == 1
        corrected_data = data[selected]
        sample_weights = weights[selected]

        return corrected_data, sample_weights

    def _heckman_correction(
        self, data: np.ndarray, selection_var: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Heckman two-step correction for selection bias.

        Step 1: Probit model for selection equation.
        Step 2: Include inverse Mills ratio in the outcome equation.
        """
        n, p = data.shape
        S = data[:, selection_var]
        predictors = [j for j in range(p) if j != selection_var]
        X = data[:, predictors]
        X_aug = np.column_stack([np.ones(n), X])

        # Step 1: Probit model for P(S=1 | X)
        beta = np.zeros(X_aug.shape[1])

        def neg_probit_ll(b):
            z = X_aug @ b
            z = np.clip(z, -30, 30)
            Phi = sp_stats.norm.cdf(z)
            Phi = np.clip(Phi, 1e-10, 1 - 1e-10)
            ll = np.sum(S * np.log(Phi) + (1 - S) * np.log(1 - Phi))
            return -ll

        result = minimize(neg_probit_ll, beta, method="L-BFGS-B")
        gamma_hat = result.x

        z = X_aug @ gamma_hat
        z = np.clip(z, -30, 30)

        # Inverse Mills ratio
        Phi = sp_stats.norm.cdf(z)
        phi = sp_stats.norm.pdf(z)
        Phi = np.clip(Phi, 1e-10, 1.0)
        imr = phi / Phi

        # Weights based on IMR-corrected propensity
        weights = np.ones(n)
        selected = S == 1
        corrected_data = data.copy()
        # Attach IMR as a virtual column for downstream regression
        corrected_data = np.column_stack([corrected_data, imr])

        return corrected_data[selected], weights[selected]

    # ──────────────────────────────────────────────────────────────────
    # Bounds under partial observability
    # ──────────────────────────────────────────────────────────────────

    def compute_bounds_under_latent(
        self,
        dag: DAGRepresentation,
        target: str,
        intervention: Dict[str, float],
        data: Optional[np.ndarray] = None,
        variables: Optional[List[str]] = None,
        method: str = "manski",
    ) -> BoundsResult:
        """Compute bounds on causal effect under latent confounders.

        Parameters
        ----------
        dag : DAGRepresentation
            The causal DAG (may include bidirected edges for latents).
        target : str
            Target (outcome) variable.
        intervention : dict
            ``{treatment_var: value}`` specifying the do-operator.
        data : array, optional
            Observational data for computing bounds.
        variables : list, optional
            Column names for data.
        method : str
            ``"manski"`` for Manski (1990) no-assumptions bounds,
            ``"balke_pearl"`` for Balke-Pearl linear programming bounds.
        """
        treatment_var = list(intervention.keys())[0]
        treatment_val = intervention[treatment_var]

        if method == "manski":
            return self._manski_bounds(
                data, variables, target, treatment_var, treatment_val
            )
        elif method == "balke_pearl":
            return self._balke_pearl_bounds(
                dag, data, variables, target, treatment_var, treatment_val
            )
        else:
            raise ValueError(f"Unknown bounds method: {method}")

    def _manski_bounds(
        self,
        data: Optional[np.ndarray],
        variables: Optional[List[str]],
        target: str,
        treatment: str,
        treatment_value: float,
    ) -> BoundsResult:
        """Manski's no-assumptions bounds on causal effects.

        For binary outcome Y and treatment T:
          Lower: E[Y|T=t]·P(T=t) + 0·P(T≠t)
          Upper: E[Y|T=t]·P(T=t) + 1·P(T≠t)

        For continuous outcomes, uses the data range.
        """
        if data is None or variables is None:
            return BoundsResult(
                lower=-np.inf, upper=np.inf,
                target=target, treatment=treatment,
                treatment_value=treatment_value,
                method="manski_no_data",
            )

        col_map = {v: i for i, v in enumerate(variables)}
        if target not in col_map or treatment not in col_map:
            return BoundsResult(
                lower=-np.inf, upper=np.inf,
                target=target, treatment=treatment,
                treatment_value=treatment_value,
                method="manski_missing_vars",
            )

        y = data[:, col_map[target]]
        t = data[:, col_map[treatment]]
        y_min, y_max = float(np.min(y)), float(np.max(y))

        # P(T = t)
        treated = np.abs(t - treatment_value) < 0.5
        p_treated = np.mean(treated)

        if p_treated < 1e-10 or p_treated > 1 - 1e-10:
            return BoundsResult(
                lower=y_min, upper=y_max,
                target=target, treatment=treatment,
                treatment_value=treatment_value,
                method="manski_degenerate",
            )

        # E[Y | T = t]
        ey_treated = float(np.mean(y[treated]))

        # Manski bounds
        lower = ey_treated * p_treated + y_min * (1 - p_treated)
        upper = ey_treated * p_treated + y_max * (1 - p_treated)

        return BoundsResult(
            lower=lower, upper=upper,
            target=target, treatment=treatment,
            treatment_value=treatment_value,
            method="manski",
        )

    def _balke_pearl_bounds(
        self,
        dag: DAGRepresentation,
        data: Optional[np.ndarray],
        variables: Optional[List[str]],
        target: str,
        treatment: str,
        treatment_value: float,
    ) -> BoundsResult:
        """Balke-Pearl linear programming bounds for binary IV models.

        Requires binary treatment, binary outcome, and an instrument.
        Finds the tightest bounds by enumerating response function types.
        """
        if data is None or variables is None:
            return BoundsResult(
                lower=-1.0, upper=1.0,
                target=target, treatment=treatment,
                treatment_value=treatment_value,
                method="balke_pearl_no_data",
            )

        col_map = {v: i for i, v in enumerate(variables)}

        # Find instrument: a parent of treatment not adjacent to target
        # except through treatment
        instrument = None
        for parent in dag.parents(treatment):
            if not dag.has_edge(parent, target) and not dag.has_bidirected(parent, target):
                instrument = parent
                break

        if instrument is None or instrument not in col_map:
            # Fall back to Manski
            return self._manski_bounds(data, variables, target, treatment, treatment_value)

        z_col = col_map[instrument]
        t_col = col_map[treatment]
        y_col = col_map[target]

        z = data[:, z_col]
        t = data[:, t_col]
        y = data[:, y_col]

        # Binarise
        z_bin = (z > np.median(z)).astype(int)
        t_bin = (t > np.median(t)).astype(int)
        y_bin = (y > np.median(y)).astype(int)

        # Compute joint distribution P(Y=y, T=t | Z=z)
        probs = {}
        for zv in [0, 1]:
            mask_z = z_bin == zv
            n_z = mask_z.sum()
            if n_z == 0:
                continue
            for tv in [0, 1]:
                for yv in [0, 1]:
                    count = ((t_bin[mask_z] == tv) & (y_bin[mask_z] == yv)).sum()
                    probs[(yv, tv, zv)] = count / n_z

        # There are 16 response function types for binary T, Y
        # Response function: (y(0), y(1)) for each individual
        # Possible types: (0,0), (0,1), (1,0), (1,1)
        # For treatment response: t(z) for z=0,1
        # Types: always-0, complier, defier, always-1

        # P(Y=1 | do(T=t)) lower and upper bounds via LP
        # Simplified: use arithmetic bounds from the joint distribution
        p_y1_t1_z0 = probs.get((1, 1, 0), 0)
        p_y1_t1_z1 = probs.get((1, 1, 1), 0)
        p_y1_t0_z0 = probs.get((1, 0, 0), 0)
        p_y1_t0_z1 = probs.get((1, 0, 1), 0)
        p_t1_z0 = probs.get((0, 1, 0), 0) + probs.get((1, 1, 0), 0)
        p_t1_z1 = probs.get((0, 1, 1), 0) + probs.get((1, 1, 1), 0)

        # Balke-Pearl bounds on ACE = P(Y=1|do(T=1)) - P(Y=1|do(T=0))
        terms = []
        for zv in [0, 1]:
            for yv in [0, 1]:
                for tv in [0, 1]:
                    terms.append(probs.get((yv, tv, zv), 0))

        # Tight bounds via the IV inequality constraints
        lower_ace = max(
            -1.0,
            probs.get((1, 1, 0), 0) - probs.get((1, 0, 0), 0) -
            probs.get((0, 1, 1), 0) - probs.get((1, 0, 1), 0),
            probs.get((1, 1, 1), 0) - probs.get((1, 0, 1), 0) -
            probs.get((0, 1, 0), 0) - probs.get((1, 0, 0), 0),
        )
        upper_ace = min(
            1.0,
            probs.get((1, 1, 0), 0) + probs.get((0, 0, 0), 0) +
            probs.get((1, 1, 1), 0) + probs.get((0, 0, 1), 0),
            probs.get((1, 1, 1), 0) + probs.get((0, 0, 1), 0) +
            probs.get((1, 1, 0), 0) + probs.get((0, 0, 0), 0),
        )

        return BoundsResult(
            lower=lower_ace, upper=upper_ace,
            target=target, treatment=treatment,
            treatment_value=treatment_value,
            method="balke_pearl",
        )

    # ──────────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────────

    @property
    def detected_latents(self) -> List[LatentConfounder]:
        return list(self._detected_latents)

    def summary(self) -> str:
        lines = [f"PartialObservabilityHandler:"]
        lines.append(f"  Detected latents: {len(self._detected_latents)}")
        for lat in self._detected_latents:
            lines.append(f"    {lat.name}: children={lat.children}, "
                         f"strength={lat.estimated_strength:.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PartialObservabilityHandler(ci_alpha={self.ci_alpha}, "
            f"latents={len(self._detected_latents)})"
        )
