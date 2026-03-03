"""
Confidence interval computation for plasticity descriptors.

Provides three complementary strategies for assessing descriptor
reliability:

1. StabilitySelector   — Subsample-based structural stability.
2. ParametricBootstrap — Bootstrap CIs for parametric components.
3. PermutationCalibrator — Null-distribution threshold calibration.

These are consumed by PlasticityComputer (ALG2, Step 5) to produce
confidence intervals on the four descriptor components.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StabilitySelectionResult:
    """Result from stability-based structural CI computation."""

    selection_probabilities: NDArray
    structural_plasticity_samples: NDArray
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_rounds: int
    subsample_fraction: float
    mean_estimate: float
    median_estimate: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BootstrapCIResult:
    """Result from parametric bootstrap CI computation."""

    point_estimate: float
    bootstrap_distribution: NDArray
    ci_lower: float
    ci_upper: float
    ci_level: float
    method: str
    n_bootstrap: int
    se: float
    bias: float
    metadata: dict = field(default_factory=dict)


@dataclass
class PermutationResult:
    """Result from permutation-based calibration."""

    null_distribution: NDArray
    observed_statistic: float
    p_value: float
    threshold: float
    significance_level: float
    n_permutations: int
    fdr_adjusted_p: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DescriptorCI:
    """Confidence intervals for all four plasticity descriptor components."""

    psi_S_ci: tuple[float, float]
    psi_P_ci: tuple[float, float]
    psi_E_ci: tuple[float, float]
    psi_CS_ci: tuple[float, float]
    ci_level: float
    structural_stability: Optional[StabilitySelectionResult] = None
    parametric_bootstrap: Optional[BootstrapCIResult] = None

    def all_cis(self) -> dict[str, tuple[float, float]]:
        """Return dict of all CIs."""
        return {
            "psi_S": self.psi_S_ci,
            "psi_P": self.psi_P_ci,
            "psi_E": self.psi_E_ci,
            "psi_CS": self.psi_CS_ci,
        }

    def ci_widths(self) -> dict[str, float]:
        """Return widths of all CIs."""
        return {k: v[1] - v[0] for k, v in self.all_cis().items()}


# ---------------------------------------------------------------------------
# StabilitySelector
# ---------------------------------------------------------------------------

class StabilitySelector:
    """Subsample-based structural stability assessment for plasticity.

    Repeatedly subsamples 50% of data per context, re-estimates the
    DAG structure, and recomputes structural plasticity (psi_S) to
    assess variability due to finite-sample DAG estimation.

    Parameters
    ----------
    n_rounds : int
        Number of subsampling rounds (default 100).
    subsample_fraction : float
        Fraction of data to subsample per round (default 0.5).
    ci_level : float
        Confidence level for intervals (default 0.95).
    random_state : int or None
        Random seed for reproducibility.
    correction : str
        Multiple testing correction: "bonferroni", "holm", or "none".
    """

    def __init__(
        self,
        n_rounds: int = 100,
        subsample_fraction: float = 0.5,
        ci_level: float = 0.95,
        random_state: Optional[int] = None,
        correction: str = "none",
    ):
        if n_rounds < 10:
            raise ValueError("n_rounds should be >= 10 for reliable CIs.")
        if not 0 < subsample_fraction < 1:
            raise ValueError("subsample_fraction must be in (0, 1).")
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be in (0, 1).")
        if correction not in ("bonferroni", "holm", "none"):
            raise ValueError(f"Unknown correction: {correction}")

        self.n_rounds = n_rounds
        self.subsample_fraction = subsample_fraction
        self.ci_level = ci_level
        self.random_state = random_state
        self.correction = correction

    def compute_structural_ci(
        self,
        datasets: list[NDArray],
        target_idx: int,
        dag_learner: Callable[[NDArray], NDArray],
        n_variables: int,
    ) -> StabilitySelectionResult:
        """Compute CI for structural plasticity (psi_S) via stability selection.

        Parameters
        ----------
        datasets : list of (n_k, n_vars) data arrays, one per context
        target_idx : index of the target variable
        dag_learner : function(data) -> adjacency matrix
        n_variables : number of variables in the system

        Returns
        -------
        StabilitySelectionResult
        """
        K = len(datasets)
        if K < 2:
            return StabilitySelectionResult(
                selection_probabilities=np.ones((1, n_variables)),
                structural_plasticity_samples=np.zeros(1),
                ci_lower=0.0,
                ci_upper=0.0,
                ci_level=self.ci_level,
                n_rounds=0,
                subsample_fraction=self.subsample_fraction,
                mean_estimate=0.0,
                median_estimate=0.0,
                metadata={"warning": "Fewer than 2 contexts"},
            )

        rng = np.random.default_rng(self.random_state)
        psi_S_samples = np.zeros(self.n_rounds, dtype=np.float64)
        all_sel_probs = np.zeros((self.n_rounds, K, n_variables), dtype=np.float64)

        for r in range(self.n_rounds):
            parent_indicators = np.zeros((K, n_variables), dtype=np.float64)

            for k, data in enumerate(datasets):
                n_k = data.shape[0]
                sub_size = max(2, int(n_k * self.subsample_fraction))
                if sub_size >= n_k:
                    sub_size = max(2, n_k - 1)

                indices = rng.choice(n_k, size=sub_size, replace=False)
                sub_data = data[indices]

                try:
                    adj = np.asarray(dag_learner(sub_data), dtype=np.float64)
                    # Parent indicators: which variables are parents of target?
                    parent_indicators[k] = adj[:, target_idx]
                except Exception:
                    parent_indicators[k] = 0.0

            all_sel_probs[r] = parent_indicators

            # Compute structural plasticity from these subsampled parent sets
            psi_S = self._compute_structural_plasticity(parent_indicators)
            psi_S_samples[r] = psi_S

        # Compute CI
        alpha = 1 - self.ci_level
        ci_lower = float(np.percentile(psi_S_samples, 100 * alpha / 2))
        ci_upper = float(np.percentile(psi_S_samples, 100 * (1 - alpha / 2)))

        # Mean selection probabilities
        mean_sel_probs = np.mean(all_sel_probs, axis=0)

        return StabilitySelectionResult(
            selection_probabilities=mean_sel_probs,
            structural_plasticity_samples=psi_S_samples,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            n_rounds=self.n_rounds,
            subsample_fraction=self.subsample_fraction,
            mean_estimate=float(np.mean(psi_S_samples)),
            median_estimate=float(np.median(psi_S_samples)),
            metadata={
                "target_idx": target_idx,
                "n_contexts": K,
                "n_variables": n_variables,
            },
        )

    def compute_emergence_ci(
        self,
        datasets: list[NDArray],
        target_idx: int,
        dag_learner: Callable[[NDArray], NDArray],
        n_variables: int,
    ) -> tuple[float, float]:
        """Compute CI for emergence (psi_E) via stability selection.

        Returns (ci_lower, ci_upper) for psi_E.
        """
        K = len(datasets)
        rng = np.random.default_rng(
            self.random_state + 1000 if self.random_state else None
        )
        psi_E_samples = np.zeros(self.n_rounds, dtype=np.float64)

        for r in range(self.n_rounds):
            mb_sizes = np.zeros(K, dtype=np.float64)

            for k, data in enumerate(datasets):
                n_k = data.shape[0]
                sub_size = max(2, int(n_k * self.subsample_fraction))
                if sub_size >= n_k:
                    sub_size = max(2, n_k - 1)

                indices = rng.choice(n_k, size=sub_size, replace=False)
                sub_data = data[indices]

                try:
                    adj = np.asarray(dag_learner(sub_data), dtype=np.float64)
                    mb = self._markov_blanket_size(adj, target_idx)
                    mb_sizes[k] = mb
                except Exception:
                    mb_sizes[k] = 0

            psi_E_samples[r] = self._compute_emergence(mb_sizes)

        alpha = 1 - self.ci_level
        ci_lower = float(np.percentile(psi_E_samples, 100 * alpha / 2))
        ci_upper = float(np.percentile(psi_E_samples, 100 * (1 - alpha / 2)))
        return ci_lower, ci_upper

    def threshold_calibration(
        self,
        datasets: list[NDArray],
        target_idx: int,
        dag_learner: Callable[[NDArray], NDArray],
        n_variables: int,
        n_calibration: int = 50,
    ) -> dict[str, float]:
        """Calibrate structural plasticity threshold via subsampling.

        Estimates the expected psi_S under the null (identical structures)
        by permuting context labels. Returns calibrated thresholds.

        Parameters
        ----------
        datasets, target_idx, dag_learner, n_variables : as above
        n_calibration : number of calibration rounds

        Returns
        -------
        dict with "threshold_95", "threshold_99", "null_mean", "null_std"
        """
        K = len(datasets)
        all_data = np.vstack(datasets)
        n_total = all_data.shape[0]
        context_sizes = [d.shape[0] for d in datasets]

        rng = np.random.default_rng(
            self.random_state + 2000 if self.random_state else None
        )
        null_psi_S = np.zeros(n_calibration, dtype=np.float64)

        for c in range(n_calibration):
            # Permute context labels
            perm = rng.permutation(n_total)
            perm_data = all_data[perm]

            # Split back into contexts
            perm_datasets = []
            idx = 0
            for size in context_sizes:
                perm_datasets.append(perm_data[idx:idx + size])
                idx += size

            parent_indicators = np.zeros((K, n_variables), dtype=np.float64)
            for k, data in enumerate(perm_datasets):
                try:
                    adj = np.asarray(dag_learner(data), dtype=np.float64)
                    parent_indicators[k] = adj[:, target_idx]
                except Exception:
                    parent_indicators[k] = 0.0

            null_psi_S[c] = self._compute_structural_plasticity(parent_indicators)

        return {
            "threshold_95": float(np.percentile(null_psi_S, 95)),
            "threshold_99": float(np.percentile(null_psi_S, 99)),
            "null_mean": float(np.mean(null_psi_S)),
            "null_std": float(np.std(null_psi_S)),
        }

    def apply_correction(
        self,
        p_values: NDArray,
    ) -> NDArray:
        """Apply multiple testing correction to p-values.

        Parameters
        ----------
        p_values : array of p-values

        Returns
        -------
        Corrected p-values
        """
        p = np.asarray(p_values, dtype=np.float64).ravel()
        m = len(p)

        if self.correction == "bonferroni":
            return np.minimum(p * m, 1.0)
        elif self.correction == "holm":
            return self._holm_correction(p)
        else:
            return p

    @staticmethod
    def _holm_correction(p_values: NDArray) -> NDArray:
        """Holm-Bonferroni step-down correction."""
        m = len(p_values)
        order = np.argsort(p_values)
        corrected = np.zeros(m)
        cummax = 0.0
        for rank, idx in enumerate(order):
            adjusted = p_values[idx] * (m - rank)
            cummax = max(cummax, adjusted)
            corrected[idx] = min(cummax, 1.0)
        return corrected

    @staticmethod
    def _compute_structural_plasticity(parent_indicators: NDArray) -> float:
        """Compute psi_S = sqrt(JSD) over parent indicator distributions.

        parent_indicators: (K, n_variables) binary matrix
        """
        K, d = parent_indicators.shape
        if K < 2:
            return 0.0

        # Treat each row as a discrete distribution over 2^d configurations
        # For computational tractability, compute variable-wise JSD
        jsd_sum = 0.0
        count = 0
        for j in range(d):
            col = parent_indicators[:, j]
            # Each context gives a Bernoulli parameter
            probs = col  # probability of parent presence
            # JSD of K Bernoulli distributions
            mean_p = np.mean(probs)
            if mean_p <= 0 or mean_p >= 1:
                # All same => JSD = 0
                continue

            # H(mean) - mean(H(p_k))
            h_mean = -mean_p * np.log2(mean_p + 1e-15) - (1 - mean_p) * np.log2(1 - mean_p + 1e-15)
            h_individual = np.zeros(K)
            for k in range(K):
                pk = probs[k]
                if 0 < pk < 1:
                    h_individual[k] = -pk * np.log2(pk) - (1 - pk) * np.log2(1 - pk)
                elif pk <= 0 or pk >= 1:
                    h_individual[k] = 0.0
            mean_h = np.mean(h_individual)
            jsd_j = max(h_mean - mean_h, 0.0)
            jsd_sum += jsd_j
            count += 1

        if count == 0:
            return 0.0

        jsd = jsd_sum / count
        return math.sqrt(max(jsd, 0.0))

    @staticmethod
    def _markov_blanket_size(adj: NDArray, target: int) -> int:
        """Compute Markov blanket size from adjacency matrix."""
        n = adj.shape[0]
        mb = set()
        # Parents
        for i in range(n):
            if adj[i, target] != 0:
                mb.add(i)
        # Children
        for j in range(n):
            if adj[target, j] != 0:
                mb.add(j)
        # Parents of children (co-parents)
        children = [j for j in range(n) if adj[target, j] != 0]
        for child in children:
            for i in range(n):
                if adj[i, child] != 0 and i != target:
                    mb.add(i)
        mb.discard(target)
        return len(mb)

    @staticmethod
    def _compute_emergence(mb_sizes: NDArray) -> float:
        """Compute psi_E from Markov blanket sizes."""
        min_mb = np.min(mb_sizes)
        max_mb = np.max(mb_sizes)
        if max_mb + 1 == 0:
            return 0.0
        return 1.0 - min_mb / (max_mb + 1)


# ---------------------------------------------------------------------------
# ParametricBootstrap
# ---------------------------------------------------------------------------

class ParametricBootstrap:
    """Parametric bootstrap for regression parameters and JSD.

    Generates bootstrap replicates by simulating from the fitted
    linear Gaussian model, re-estimating parameters, and computing
    the distribution of parametric plasticity (psi_P).

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap replicates (default 200).
    ci_level : float
        Confidence level (default 0.95).
    ci_method : str
        CI method: "percentile" or "bca" (default "percentile").
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        ci_method: str = "percentile",
        random_state: Optional[int] = None,
    ):
        if n_bootstrap < 10:
            raise ValueError("n_bootstrap must be >= 10.")
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be in (0, 1).")
        if ci_method not in ("percentile", "bca"):
            raise ValueError("ci_method must be 'percentile' or 'bca'.")

        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.ci_method = ci_method
        self.random_state = random_state

    def compute_parametric_ci(
        self,
        datasets: list[NDArray],
        target_idx: int,
        parent_sets: list[list[int]],
    ) -> BootstrapCIResult:
        """Compute CI for parametric plasticity (psi_P) via bootstrap.

        For each context with the same parent set, perturbs regression
        coefficients and residual variance, then recomputes psi_P.

        Parameters
        ----------
        datasets : list of (n_k, n_vars) data arrays per context
        target_idx : target variable index
        parent_sets : list of parent index lists per context

        Returns
        -------
        BootstrapCIResult
        """
        K = len(datasets)
        if K < 2:
            return BootstrapCIResult(
                point_estimate=0.0,
                bootstrap_distribution=np.zeros(1),
                ci_lower=0.0,
                ci_upper=0.0,
                ci_level=self.ci_level,
                method=self.ci_method,
                n_bootstrap=0,
                se=0.0,
                bias=0.0,
                metadata={"warning": "Fewer than 2 contexts"},
            )

        # Fit regression models per context
        models = []
        for k in range(K):
            data = datasets[k]
            parents = parent_sets[k]
            model = self._fit_regression(data, target_idx, parents)
            models.append(model)

        # Compute point estimate of psi_P
        psi_P_point = self._compute_psi_P(models, parent_sets)

        # Bootstrap
        rng = np.random.default_rng(self.random_state)
        psi_P_boot = np.zeros(self.n_bootstrap, dtype=np.float64)

        for b in range(self.n_bootstrap):
            boot_models = []
            for k in range(K):
                bm = self._perturb_model(models[k], rng, datasets[k].shape[0])
                boot_models.append(bm)
            psi_P_boot[b] = self._compute_psi_P(boot_models, parent_sets)

        # CI computation
        alpha = 1 - self.ci_level
        if self.ci_method == "percentile":
            ci_lower = float(np.percentile(psi_P_boot, 100 * alpha / 2))
            ci_upper = float(np.percentile(psi_P_boot, 100 * (1 - alpha / 2)))
        else:
            ci_lower, ci_upper = self._bca_ci_scalar(psi_P_point, psi_P_boot)

        se = float(np.std(psi_P_boot, ddof=1))
        bias = float(np.mean(psi_P_boot) - psi_P_point)

        return BootstrapCIResult(
            point_estimate=psi_P_point,
            bootstrap_distribution=psi_P_boot,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method=self.ci_method,
            n_bootstrap=self.n_bootstrap,
            se=se,
            bias=bias,
            metadata={
                "target_idx": target_idx,
                "n_contexts": K,
                "n_groups": len(set(tuple(sorted(ps)) for ps in parent_sets)),
            },
        )

    def compute_context_sensitivity_ci(
        self,
        datasets: list[NDArray],
        target_idx: int,
        parent_sets: list[list[int]],
        n_subsets: int = 100,
    ) -> BootstrapCIResult:
        """Compute CI for context sensitivity (psi_CS) via bootstrap.

        Parameters
        ----------
        datasets : per-context data
        target_idx : target variable
        parent_sets : per-context parent sets
        n_subsets : number of random subsets for psi_CS

        Returns
        -------
        BootstrapCIResult
        """
        K = len(datasets)
        if K < 3:
            return BootstrapCIResult(
                point_estimate=0.0,
                bootstrap_distribution=np.zeros(1),
                ci_lower=0.0,
                ci_upper=0.0,
                ci_level=self.ci_level,
                method=self.ci_method,
                n_bootstrap=0,
                se=0.0,
                bias=0.0,
                metadata={"warning": "Fewer than 3 contexts for psi_CS"},
            )

        # Point estimate
        psi_CS_point = self._compute_psi_CS(datasets, target_idx, parent_sets, n_subsets)

        # Bootstrap
        rng = np.random.default_rng(self.random_state)
        psi_CS_boot = np.zeros(self.n_bootstrap, dtype=np.float64)

        for b in range(self.n_bootstrap):
            # Resample within each context
            boot_datasets = []
            for k in range(K):
                n_k = datasets[k].shape[0]
                indices = rng.choice(n_k, size=n_k, replace=True)
                boot_datasets.append(datasets[k][indices])

            psi_CS_boot[b] = self._compute_psi_CS(
                boot_datasets, target_idx, parent_sets, n_subsets
            )

        alpha = 1 - self.ci_level
        ci_lower = float(np.percentile(psi_CS_boot, 100 * alpha / 2))
        ci_upper = float(np.percentile(psi_CS_boot, 100 * (1 - alpha / 2)))
        se = float(np.std(psi_CS_boot, ddof=1))
        bias = float(np.mean(psi_CS_boot) - psi_CS_point)

        return BootstrapCIResult(
            point_estimate=psi_CS_point,
            bootstrap_distribution=psi_CS_boot,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method=self.ci_method,
            n_bootstrap=self.n_bootstrap,
            se=se,
            bias=bias,
        )

    def bootstrap_distribution_summary(
        self,
        result: BootstrapCIResult,
    ) -> dict:
        """Generate summary statistics for bootstrap distribution.

        Returns dict with mean, median, std, skewness, kurtosis, quantiles.
        """
        d = result.bootstrap_distribution
        n = len(d)
        mean = float(np.mean(d))
        median = float(np.median(d))
        std = float(np.std(d, ddof=1)) if n > 1 else 0.0
        q05 = float(np.percentile(d, 5))
        q25 = float(np.percentile(d, 25))
        q75 = float(np.percentile(d, 75))
        q95 = float(np.percentile(d, 95))

        # Skewness and kurtosis
        if std > 1e-10:
            skew = float(np.mean(((d - mean) / std) ** 3))
            kurt = float(np.mean(((d - mean) / std) ** 4) - 3.0)
        else:
            skew = 0.0
            kurt = 0.0

        return {
            "mean": mean,
            "median": median,
            "std": std,
            "skewness": skew,
            "kurtosis": kurt,
            "q05": q05,
            "q25": q25,
            "q75": q75,
            "q95": q95,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "ci_width": result.ci_upper - result.ci_lower,
            "n_bootstrap": result.n_bootstrap,
        }

    # ---- Internal helpers ----

    @staticmethod
    def _fit_regression(
        data: NDArray,
        target_idx: int,
        parents: list[int],
    ) -> dict:
        """Fit OLS regression: target ~ parents.

        Returns dict with coefficients, residual_var, intercept, se.
        """
        y = data[:, target_idx]
        n = len(y)

        if len(parents) == 0:
            return {
                "coefficients": np.array([]),
                "intercept": float(np.mean(y)),
                "residual_var": float(np.var(y, ddof=1)) if n > 1 else float(np.var(y)),
                "se_coefficients": np.array([]),
                "se_intercept": float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "n_samples": n,
                "n_parents": 0,
                "parents": parents,
            }

        X = data[:, parents]
        if X.ndim == 1:
            X = X[:, np.newaxis]

        # Add intercept
        X_aug = np.column_stack([np.ones(n), X])
        p = X_aug.shape[1]

        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X_aug) @ y
            residuals = None

        y_hat = X_aug @ beta
        resid = y - y_hat

        if n > p:
            res_var = float(np.sum(resid ** 2) / (n - p))
        else:
            res_var = float(np.var(resid))

        # Standard errors
        try:
            cov_matrix = res_var * np.linalg.inv(X_aug.T @ X_aug)
            se = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(p, np.sqrt(res_var / max(n, 1)))

        return {
            "coefficients": beta[1:],
            "intercept": float(beta[0]),
            "residual_var": max(res_var, 1e-15),
            "se_coefficients": se[1:],
            "se_intercept": float(se[0]),
            "n_samples": n,
            "n_parents": len(parents),
            "parents": parents,
        }

    @staticmethod
    def _perturb_model(model: dict, rng: np.random.Generator, n: int) -> dict:
        """Generate a parametric bootstrap replicate of a regression model.

        Perturbs coefficients by their SEs and residual variance by chi-squared.
        """
        perturbed = dict(model)
        p = model["n_parents"]

        if p > 0:
            coef_noise = rng.normal(0, model["se_coefficients"])
            perturbed["coefficients"] = model["coefficients"] + coef_noise

        intercept_noise = rng.normal(0, model["se_intercept"])
        perturbed["intercept"] = model["intercept"] + intercept_noise

        # Chi-squared perturbation for variance
        df = max(n - p - 1, 1)
        chi2 = rng.chisquare(df)
        perturbed["residual_var"] = max(model["residual_var"] * chi2 / df, 1e-15)

        return perturbed

    @staticmethod
    @staticmethod
    def _gaussian_jsd(model_a: dict, model_b: dict) -> float:
        """Compute JSD between two Gaussian regression models (same parents).

        For Gaussian with same parent set:
            JSD ≈ closed-form from means (intercepts + coefficients) and variances.
        """
        mu_a = model_a["intercept"]
        mu_b = model_b["intercept"]
        var_a = model_a["residual_var"]
        var_b = model_b["residual_var"]

        # Also account for coefficient differences in the mean
        coef_a = model_a["coefficients"]
        coef_b = model_b["coefficients"]

        if len(coef_a) > 0 and len(coef_b) > 0 and len(coef_a) == len(coef_b):
            # Mean difference including coefficient effect (unit X)
            coef_diff = np.sum((coef_a - coef_b) ** 2)
        else:
            coef_diff = 0.0

        mean_diff_sq = (mu_a - mu_b) ** 2 + coef_diff

        # KL(a||b) for univariate Gaussians
        if var_b < 1e-15 or var_a < 1e-15:
            return 0.0

        kl_ab = 0.5 * (np.log(var_b / var_a) + var_a / var_b + mean_diff_sq / var_b - 1)
        kl_ba = 0.5 * (np.log(var_a / var_b) + var_b / var_a + mean_diff_sq / var_a - 1)

        jsd = 0.5 * (kl_ab + kl_ba)
        return max(jsd / np.log(2), 0.0)  # Convert to bits

    def _compute_psi_P(
        self,
        models: list[dict],
        parent_sets: list[list[int]],
    ) -> float:
        """Compute parametric plasticity from fitted models.

        Groups contexts by parent set, computes pairwise sqrt(JSD)
        within each group, averages over all within-group pairs.
        """
        K = len(models)
        if K < 2:
            return 0.0

        # Group by parent set
        groups: dict[tuple, list[int]] = {}
        for k in range(K):
            key = tuple(sorted(parent_sets[k]))
            if key not in groups:
                groups[key] = []
            groups[key].append(k)

        total_jsd = 0.0
        n_pairs = 0

        for key, members in groups.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    jsd = self._gaussian_jsd(models[members[i]], models[members[j]])
                    total_jsd += math.sqrt(max(jsd, 0.0))
                    n_pairs += 1

        if n_pairs == 0:
            return 0.0
        return total_jsd / n_pairs

    def _compute_psi_CS(
        self,
        datasets: list[NDArray],
        target_idx: int,
        parent_sets: list[list[int]],
        n_subsets: int,
    ) -> float:
        """Compute context sensitivity: CV of psi_P across context subsets."""
        K = len(datasets)
        if K < 3:
            return 0.0

        subset_size = max(2, math.ceil(K / 2))
        rng = np.random.default_rng(
            self.random_state + 5000 if self.random_state is not None else None
        )

        psi_P_values = np.zeros(n_subsets, dtype=np.float64)
        for s in range(n_subsets):
            indices = rng.choice(K, size=subset_size, replace=False)
            sub_datasets = [datasets[k] for k in indices]
            sub_parents = [parent_sets[k] for k in indices]

            sub_models = []
            for k_idx in range(len(sub_datasets)):
                model = self._fit_regression(
                    sub_datasets[k_idx], target_idx, sub_parents[k_idx]
                )
                sub_models.append(model)

            psi_P_values[s] = self._compute_psi_P(sub_models, sub_parents)

        mean_psi = np.mean(psi_P_values)
        std_psi = np.std(psi_P_values, ddof=1) if n_subsets > 1 else 0.0

        if abs(mean_psi) < 1e-10:
            return 0.0
        return std_psi / abs(mean_psi)

    def _bca_ci_scalar(
        self,
        point: float,
        boot_samples: NDArray,
    ) -> tuple[float, float]:
        """BCa CI for a scalar statistic."""
        from scipy.stats import norm

        B = len(boot_samples)
        alpha = 1 - self.ci_level

        # Bias correction z0
        prop_less = np.mean(boot_samples < point)
        prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
        z0 = norm.ppf(prop_less)

        # Acceleration (simplified: 0)
        a = 0.0

        z_lo = norm.ppf(alpha / 2)
        z_hi = norm.ppf(1 - alpha / 2)

        denom_lo = 1 - a * (z0 + z_lo)
        denom_hi = 1 - a * (z0 + z_hi)
        if abs(denom_lo) < 1e-10:
            denom_lo = 1e-10
        if abs(denom_hi) < 1e-10:
            denom_hi = 1e-10

        alpha_lo = norm.cdf(z0 + (z0 + z_lo) / denom_lo)
        alpha_hi = norm.cdf(z0 + (z0 + z_hi) / denom_hi)
        alpha_lo = np.clip(alpha_lo, 0.001, 0.999)
        alpha_hi = np.clip(alpha_hi, 0.001, 0.999)

        return (
            float(np.percentile(boot_samples, 100 * alpha_lo)),
            float(np.percentile(boot_samples, 100 * alpha_hi)),
        )


# ---------------------------------------------------------------------------
# PermutationCalibrator
# ---------------------------------------------------------------------------

class PermutationCalibrator:
    """Null-distribution generation via permutation for threshold calibration.

    Generates a null distribution for each plasticity descriptor by
    permuting context labels and recomputing the descriptor. Provides
    calibrated thresholds and p-values with FDR control.

    Parameters
    ----------
    n_permutations : int
        Number of permutations (default 999).
    significance_level : float
        Significance level for threshold (default 0.05).
    random_state : int or None
        Random seed.
    fdr_method : str
        FDR control method: "bh" (Benjamini-Hochberg) or "by"
        (Benjamini-Yekutieli) or "none".
    """

    def __init__(
        self,
        n_permutations: int = 999,
        significance_level: float = 0.05,
        random_state: Optional[int] = None,
        fdr_method: str = "bh",
    ):
        if n_permutations < 10:
            raise ValueError("n_permutations must be >= 10.")
        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be in (0, 1).")
        if fdr_method not in ("bh", "by", "none"):
            raise ValueError("fdr_method must be 'bh', 'by', or 'none'.")

        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.random_state = random_state
        self.fdr_method = fdr_method

    def calibrate_structural(
        self,
        datasets: list[NDArray],
        target_idx: int,
        dag_learner: Callable[[NDArray], NDArray],
        observed_psi_S: float,
    ) -> PermutationResult:
        """Calibrate structural plasticity threshold via permutation.

        Permutes context labels (shuffles data across contexts) and
        recomputes psi_S under the null of no structural differences.

        Parameters
        ----------
        datasets : per-context data arrays
        target_idx : target variable
        dag_learner : DAG learning function
        observed_psi_S : observed structural plasticity value

        Returns
        -------
        PermutationResult
        """
        K = len(datasets)
        n_vars = datasets[0].shape[1]
        all_data = np.vstack(datasets)
        n_total = all_data.shape[0]
        context_sizes = [d.shape[0] for d in datasets]

        rng = np.random.default_rng(self.random_state)
        null_values = np.zeros(self.n_permutations, dtype=np.float64)

        for b in range(self.n_permutations):
            perm = rng.permutation(n_total)
            perm_data = all_data[perm]

            perm_datasets = []
            idx = 0
            for size in context_sizes:
                perm_datasets.append(perm_data[idx:idx + size])
                idx += size

            parent_indicators = np.zeros((K, n_vars), dtype=np.float64)
            for k, data in enumerate(perm_datasets):
                try:
                    adj = np.asarray(dag_learner(data), dtype=np.float64)
                    parent_indicators[k] = adj[:, target_idx]
                except Exception:
                    parent_indicators[k] = 0.0

            null_values[b] = StabilitySelector._compute_structural_plasticity(
                parent_indicators
            )

        p_value = self._compute_p_value(observed_psi_S, null_values)
        threshold = float(
            np.percentile(null_values, 100 * (1 - self.significance_level))
        )

        return PermutationResult(
            null_distribution=null_values,
            observed_statistic=observed_psi_S,
            p_value=p_value,
            threshold=threshold,
            significance_level=self.significance_level,
            n_permutations=self.n_permutations,
        )

    def calibrate_parametric(
        self,
        datasets: list[NDArray],
        target_idx: int,
        parent_sets: list[list[int]],
        observed_psi_P: float,
    ) -> PermutationResult:
        """Calibrate parametric plasticity threshold via permutation.

        Permutes samples within same-structure groups and recomputes psi_P.

        Parameters
        ----------
        datasets : per-context data arrays
        target_idx : target variable
        parent_sets : per-context parent sets
        observed_psi_P : observed parametric plasticity

        Returns
        -------
        PermutationResult
        """
        K = len(datasets)
        rng = np.random.default_rng(self.random_state)
        null_values = np.zeros(self.n_permutations, dtype=np.float64)

        # Group contexts by parent set
        groups: dict[tuple, list[int]] = {}
        for k in range(K):
            key = tuple(sorted(parent_sets[k]))
            if key not in groups:
                groups[key] = []
            groups[key].append(k)

        for b in range(self.n_permutations):
            # For each group, pool and re-split data
            perm_datasets = list(datasets)  # shallow copy
            for key, members in groups.items():
                if len(members) < 2:
                    continue
                pooled = np.vstack([datasets[m] for m in members])
                perm = rng.permutation(pooled.shape[0])
                pooled = pooled[perm]
                idx = 0
                for m in members:
                    n_m = datasets[m].shape[0]
                    perm_datasets[m] = pooled[idx:idx + n_m]
                    idx += n_m

            # Recompute psi_P
            models = []
            for k in range(K):
                model = ParametricBootstrap._fit_regression(
                    perm_datasets[k], target_idx, parent_sets[k]
                )
                models.append(model)

            # Use a temporary ParametricBootstrap to call _compute_psi_P
            _temp_pb = ParametricBootstrap.__new__(ParametricBootstrap)
            psi_P = _temp_pb._compute_psi_P(models, parent_sets)
            null_values[b] = psi_P

        p_value = self._compute_p_value(observed_psi_P, null_values)
        threshold = float(
            np.percentile(null_values, 100 * (1 - self.significance_level))
        )

        return PermutationResult(
            null_distribution=null_values,
            observed_statistic=observed_psi_P,
            p_value=p_value,
            threshold=threshold,
            significance_level=self.significance_level,
            n_permutations=self.n_permutations,
        )

    def batch_calibrate(
        self,
        p_values: list[float],
        statistics: list[float],
        null_distributions: list[NDArray],
    ) -> list[PermutationResult]:
        """Apply FDR control to multiple tests.

        Parameters
        ----------
        p_values : list of raw p-values
        statistics : list of observed statistics
        null_distributions : list of null distributions

        Returns
        -------
        list of PermutationResult with FDR-adjusted p-values
        """
        raw_p = np.array(p_values)
        adjusted_p = self.fdr_correction(raw_p)

        results = []
        for i in range(len(p_values)):
            threshold = float(
                np.percentile(
                    null_distributions[i],
                    100 * (1 - self.significance_level),
                )
            )
            results.append(PermutationResult(
                null_distribution=null_distributions[i],
                observed_statistic=statistics[i],
                p_value=p_values[i],
                threshold=threshold,
                significance_level=self.significance_level,
                n_permutations=len(null_distributions[i]),
                fdr_adjusted_p=float(adjusted_p[i]),
            ))
        return results

    def fdr_correction(self, p_values: NDArray) -> NDArray:
        """Apply FDR correction to p-values.

        Parameters
        ----------
        p_values : array of raw p-values

        Returns
        -------
        array of adjusted p-values
        """
        p = np.asarray(p_values, dtype=np.float64).ravel()
        m = len(p)

        if self.fdr_method == "none" or m == 0:
            return p

        if self.fdr_method == "bh":
            return self._benjamini_hochberg(p)
        elif self.fdr_method == "by":
            return self._benjamini_yekutieli(p)
        else:
            return p

    @staticmethod
    def _compute_p_value(observed: float, null: NDArray) -> float:
        """Compute permutation p-value.

        p = (#{null >= observed} + 1) / (n_perm + 1)
        """
        count = np.sum(null >= observed)
        return float((count + 1) / (len(null) + 1))

    @staticmethod
    def _benjamini_hochberg(p_values: NDArray) -> NDArray:
        """Benjamini-Hochberg FDR correction."""
        m = len(p_values)
        if m == 0:
            return p_values

        order = np.argsort(p_values)
        adjusted = np.zeros(m)

        # Process from largest to smallest
        cummin = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            adjusted_val = p_values[idx] * m / (rank + 1)
            cummin = min(cummin, adjusted_val)
            adjusted[idx] = min(cummin, 1.0)

        return adjusted

    @staticmethod
    def _benjamini_yekutieli(p_values: NDArray) -> NDArray:
        """Benjamini-Yekutieli FDR correction (handles dependence)."""
        m = len(p_values)
        if m == 0:
            return p_values

        # c(m) = sum(1/k for k=1..m)
        c_m = np.sum(1.0 / np.arange(1, m + 1))

        order = np.argsort(p_values)
        adjusted = np.zeros(m)

        cummin = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            adjusted_val = p_values[idx] * m * c_m / (rank + 1)
            cummin = min(cummin, adjusted_val)
            adjusted[idx] = min(cummin, 1.0)

        return adjusted
