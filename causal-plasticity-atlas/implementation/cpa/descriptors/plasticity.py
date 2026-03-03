"""
Unified 4D Plasticity Descriptor Computation (ALG2).

Computes the four-component plasticity descriptor vector for each
variable in a multi-context structural causal model:

    Ψ(i) = (ψ_S, ψ_P, ψ_E, ψ_CS)

Components:
    ψ_S  — Structural Plasticity:  sqrt(JSD) over parent-set indicators.
    ψ_P  — Parametric Plasticity:  within-group sqrt(JSD) of conditionals.
    ψ_E  — Emergence:              Markov-blanket size variation.
    ψ_CS — Context Sensitivity:    CV of ψ_P across context subsets.

Provides:
    PlasticityComputer      — Single-variable descriptor computation.
    BatchPlasticityComputer  — Batch computation across all variables.

Theory reference: ALG2 in the CPA specification.
"""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from cpa.descriptors.classification import (
    BatchClassificationResult,
    ClassificationResult,
    ClassificationThresholds,
    PlasticityCategory,
    PlasticityClassifier,
)
from cpa.descriptors.confidence import (
    BootstrapCIResult,
    DescriptorCI,
    ParametricBootstrap,
    PermutationCalibrator,
    StabilitySelector,
    StabilitySelectionResult,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlasticityConfig:
    """Configuration for plasticity descriptor computation.

    Parameters
    ----------
    n_stability_rounds : int
        Stability selection rounds for structural CI (default 100).
    n_bootstrap : int
        Parametric bootstrap replicates (default 200).
    n_context_subsets : int
        Monte Carlo subsets for context sensitivity (default 100).
    subsample_fraction : float
        Subsample fraction for stability selection (default 0.5).
    ci_level : float
        Confidence level for intervals (default 0.95).
    ci_method : str
        CI method for bootstrap: "percentile" or "bca" (default "percentile").
    tau_S : float
        Structural plasticity threshold (default 0.1).
    tau_P : float
        Parametric plasticity threshold (default 0.5).
    tau_E : float
        Emergence threshold (default 0.5).
    use_ci_lower : bool
        Use CI lower bounds for classification (default True).
    compute_cis : bool
        Whether to compute confidence intervals (default True).
    compute_classification : bool
        Whether to compute classification (default True).
    random_state : int or None
        Random seed (default None).
    min_samples_warning : int
        Warn if any context has fewer samples (default 30).
    n_permutations : int
        Permutations for calibration (default 999).
    """

    n_stability_rounds: int = 100
    n_bootstrap: int = 200
    n_context_subsets: int = 100
    subsample_fraction: float = 0.5
    ci_level: float = 0.95
    ci_method: str = "percentile"
    tau_S: float = 0.1
    tau_P: float = 0.5
    tau_E: float = 0.5
    use_ci_lower: bool = True
    compute_cis: bool = True
    compute_classification: bool = True
    random_state: Optional[int] = None
    min_samples_warning: int = 30
    n_permutations: int = 999

    def thresholds(self) -> ClassificationThresholds:
        """Return ClassificationThresholds from config."""
        return ClassificationThresholds(
            tau_S=self.tau_S,
            tau_P=self.tau_P,
            tau_E=self.tau_E,
            use_ci_lower=self.use_ci_lower,
            ci_level=self.ci_level,
        )


# ---------------------------------------------------------------------------
# Plasticity Descriptor dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlasticityDescriptor:
    """Complete plasticity descriptor for a single variable.

    Contains the 4D descriptor vector, confidence intervals,
    classification result, and computation metadata.
    """

    variable_idx: int
    variable_name: Optional[str]

    # Core 4D descriptor
    psi_S: float    # Structural plasticity
    psi_P: float    # Parametric plasticity
    psi_E: float    # Emergence
    psi_CS: float   # Context sensitivity

    # Confidence intervals (optional)
    psi_S_ci: Optional[tuple[float, float]] = None
    psi_P_ci: Optional[tuple[float, float]] = None
    psi_E_ci: Optional[tuple[float, float]] = None
    psi_CS_ci: Optional[tuple[float, float]] = None

    # Classification (optional)
    classification: Optional[ClassificationResult] = None

    # Intermediate data
    parent_sets: Optional[list[list[int]]] = None
    markov_blanket_sizes: Optional[list[int]] = None
    psi_P_per_group: Optional[dict[str, float]] = None
    psi_P_subset_values: Optional[NDArray] = None

    # Metadata
    n_contexts: int = 0
    computation_time: float = 0.0
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def descriptor_vector(self) -> NDArray:
        """Return 4D descriptor as numpy array."""
        return np.array([self.psi_S, self.psi_P, self.psi_E, self.psi_CS])

    @property
    def has_cis(self) -> bool:
        """True if confidence intervals are available."""
        return self.psi_S_ci is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            "variable_idx": self.variable_idx,
            "variable_name": self.variable_name,
            "psi_S": self.psi_S,
            "psi_P": self.psi_P,
            "psi_E": self.psi_E,
            "psi_CS": self.psi_CS,
            "n_contexts": self.n_contexts,
        }
        if self.psi_S_ci is not None:
            d["psi_S_ci"] = list(self.psi_S_ci)
            d["psi_P_ci"] = list(self.psi_P_ci)
            d["psi_E_ci"] = list(self.psi_E_ci)
            d["psi_CS_ci"] = list(self.psi_CS_ci)
        if self.classification is not None:
            d["classification"] = self.classification.primary_category.value
            d["confidence"] = self.classification.confidence
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PlasticityDescriptor:
        """Create from dictionary."""
        desc = cls(
            variable_idx=d["variable_idx"],
            variable_name=d.get("variable_name"),
            psi_S=d["psi_S"],
            psi_P=d["psi_P"],
            psi_E=d["psi_E"],
            psi_CS=d["psi_CS"],
            n_contexts=d.get("n_contexts", 0),
        )
        if "psi_S_ci" in d:
            desc.psi_S_ci = tuple(d["psi_S_ci"])
            desc.psi_P_ci = tuple(d["psi_P_ci"])
            desc.psi_E_ci = tuple(d["psi_E_ci"])
            desc.psi_CS_ci = tuple(d["psi_CS_ci"])
        return desc

    def summary(self) -> str:
        """One-line summary."""
        name = self.variable_name or f"X_{self.variable_idx}"
        cat = self.classification.primary_category.value if self.classification else "unclassified"
        return (
            f"{name}: [{cat}] ψ_S={self.psi_S:.3f} ψ_P={self.psi_P:.3f} "
            f"ψ_E={self.psi_E:.3f} ψ_CS={self.psi_CS:.3f}"
        )


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence utilities
# ---------------------------------------------------------------------------

def _kl_divergence_bernoulli(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q))."""
    if p <= 0:
        return (1 - p) * math.log((1 - p) / max(1 - q, 1e-15)) if (1 - p) > 0 else 0.0
    if p >= 1:
        return p * math.log(p / max(q, 1e-15)) if p > 0 else 0.0
    if q <= 0 or q >= 1:
        return float("inf")
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def _binary_entropy(p: float) -> float:
    """H(Bernoulli(p)) in nats."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def _jsd_bernoulli_multi(probs: NDArray) -> float:
    """Multi-distribution JSD of K Bernoulli distributions.

    JSD(p_1, ..., p_K) = H(mean(p)) - mean(H(p_k))

    Parameters
    ----------
    probs : (K,) array of Bernoulli probabilities

    Returns
    -------
    JSD in nats
    """
    K = len(probs)
    if K < 2:
        return 0.0

    mean_p = np.mean(probs)
    h_mean = _binary_entropy(mean_p)
    h_individuals = np.array([_binary_entropy(p) for p in probs])
    jsd = h_mean - np.mean(h_individuals)
    return max(jsd, 0.0)


def _jsd_gaussian(
    mu1: float,
    var1: float,
    mu2: float,
    var2: float,
) -> float:
    """Jensen-Shannon divergence between two univariate Gaussians.

    Uses JSD = 0.5 * (KL(p||m) + KL(q||m)) where m is the mixture.
    For Gaussians this doesn't have a closed form, so we use the
    symmetric KL approximation: JSD ≈ 0.5 * KL_sym.
    """
    if var1 < 1e-15 or var2 < 1e-15:
        return 0.0

    # Symmetric KL
    kl_12 = 0.5 * (math.log(var2 / var1) + var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1)
    kl_21 = 0.5 * (math.log(var1 / var2) + var2 / var1 + (mu2 - mu1) ** 2 / var1 - 1)
    jsd = 0.5 * (kl_12 + kl_21)
    return max(jsd, 0.0)


def _jsd_gaussian_regression(
    coefs_a: NDArray,
    intercept_a: float,
    var_a: float,
    coefs_b: NDArray,
    intercept_b: float,
    var_b: float,
) -> float:
    """JSD between two conditional Gaussian distributions.

    P(Y | X, theta_a) ~ N(intercept_a + X @ coefs_a, var_a)
    P(Y | X, theta_b) ~ N(intercept_b + X @ coefs_b, var_b)

    We integrate out X assuming X is standardized (zero mean, unit variance).
    The effective mean difference is just the intercept difference + coefficient
    differences (since E[X] = 0).
    """
    if var_a < 1e-15 or var_b < 1e-15:
        return 0.0

    # Mean difference at E[X] = 0
    mean_diff = intercept_a - intercept_b

    # Additional variance from coefficient differences
    coef_diff = np.asarray(coefs_a) - np.asarray(coefs_b)
    # Under unit-variance X: E[(X @ delta)^2] = ||delta||^2
    mean_diff_var = float(np.sum(coef_diff ** 2))

    # Effective parameters for JSD
    effective_mean_diff_sq = mean_diff ** 2 + mean_diff_var

    kl_ab = 0.5 * (math.log(var_b / var_a) + var_a / var_b + effective_mean_diff_sq / var_b - 1)
    kl_ba = 0.5 * (math.log(var_a / var_b) + var_b / var_a + effective_mean_diff_sq / var_a - 1)
    jsd = 0.5 * (kl_ab + kl_ba)
    return max(jsd, 0.0)


def _jsd_discrete_multi(distributions: list[NDArray]) -> float:
    """Multi-distribution JSD for discrete distributions.

    Parameters
    ----------
    distributions : list of K probability vectors (each sums to 1)

    Returns
    -------
    JSD in nats
    """
    K = len(distributions)
    if K < 2:
        return 0.0

    # Pad to same length
    max_len = max(len(d) for d in distributions)
    padded = np.zeros((K, max_len), dtype=np.float64)
    for k, d in enumerate(distributions):
        padded[k, :len(d)] = d

    # Normalize
    row_sums = padded.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    padded = padded / row_sums

    # Mixture distribution
    mixture = np.mean(padded, axis=0)

    # H(mixture) - mean(H(p_k))
    h_mix = -np.sum(mixture * np.log(mixture + 1e-15))
    h_individual = -np.sum(padded * np.log(padded + 1e-15), axis=1)
    mean_h = np.mean(h_individual)

    return max(h_mix - mean_h, 0.0)


# ---------------------------------------------------------------------------
# Regression utilities
# ---------------------------------------------------------------------------

def _fit_ols(
    X: NDArray,
    y: NDArray,
) -> tuple[NDArray, float, float]:
    """Fit OLS regression y ~ X (with intercept).

    Returns (coefficients, intercept, residual_variance).
    coefficients does NOT include intercept.
    """
    n = len(y)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    p = X.shape[1]

    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_aug) @ y

    resid = y - X_aug @ beta
    df = max(n - p - 1, 1)
    res_var = float(np.sum(resid ** 2) / df)
    return beta[1:], float(beta[0]), max(res_var, 1e-15)


def _fit_ols_no_parents(y: NDArray) -> tuple[NDArray, float, float]:
    """OLS for a variable with no parents."""
    n = len(y)
    intercept = float(np.mean(y))
    if n > 1:
        res_var = float(np.var(y, ddof=1))
    else:
        res_var = float(np.var(y))
    return np.array([]), intercept, max(res_var, 1e-15)


def _coefficient_se(X: NDArray, y: NDArray, res_var: float) -> NDArray:
    """Standard errors for OLS coefficients."""
    n = len(y)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    X_aug = np.column_stack([np.ones(n), X])
    try:
        cov = res_var * np.linalg.inv(X_aug.T @ X_aug)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return se[1:]
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], math.sqrt(res_var / max(n, 1)))


# ---------------------------------------------------------------------------
# Markov blanket utilities
# ---------------------------------------------------------------------------

def _markov_blanket(adj: NDArray, target: int) -> set[int]:
    """Compute Markov blanket from adjacency matrix.

    MB(target) = parents(target) ∪ children(target) ∪ co-parents(children).
    """
    n = adj.shape[0]
    mb = set()

    # Parents: adj[i, target] != 0
    for i in range(n):
        if adj[i, target] != 0:
            mb.add(i)

    # Children: adj[target, j] != 0
    children = []
    for j in range(n):
        if adj[target, j] != 0:
            mb.add(j)
            children.append(j)

    # Co-parents of children
    for child in children:
        for i in range(n):
            if adj[i, child] != 0 and i != target:
                mb.add(i)

    mb.discard(target)
    return mb


def _parent_set(adj: NDArray, target: int) -> list[int]:
    """Extract parent set of target from adjacency matrix."""
    n = adj.shape[0]
    return sorted([i for i in range(n) if adj[i, target] != 0])


# ---------------------------------------------------------------------------
# PlasticityComputer
# ---------------------------------------------------------------------------

class PlasticityComputer:
    """Compute the 4D plasticity descriptor for a single variable.

    Implements Algorithm 2 (ALG2) steps 1–6:
        1. Structural Plasticity (ψ_S)
        2. Parametric Plasticity (ψ_P)
        3. Emergence (ψ_E)
        4. Context Sensitivity (ψ_CS)
        5. Confidence Intervals
        6. Classification

    Parameters
    ----------
    config : PlasticityConfig
        Configuration parameters.

    Examples
    --------
    >>> computer = PlasticityComputer()
    >>> desc = computer.compute(
    ...     adjacencies=[adj1, adj2, adj3],
    ...     datasets=[data1, data2, data3],
    ...     target_idx=0,
    ... )
    >>> print(desc.summary())
    """

    def __init__(self, config: Optional[PlasticityConfig] = None):
        self.config = config or PlasticityConfig()
        self._classifier = PlasticityClassifier(thresholds=self.config.thresholds())

    def compute(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        target_idx: int,
        variable_name: Optional[str] = None,
        dag_learner: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> PlasticityDescriptor:
        """Compute full 4D plasticity descriptor for a variable.

        Parameters
        ----------
        adjacencies : list of K adjacency matrices, shape (n_vars, n_vars)
            adjacencies[k][i, j] != 0 means i -> j in context k.
        datasets : list of K data arrays, shape (n_k, n_vars)
        target_idx : index of the variable to describe
        variable_name : optional name for the variable
        dag_learner : optional DAG learning function for stability selection

        Returns
        -------
        PlasticityDescriptor
        """
        import time
        t0 = time.perf_counter()

        # Validate inputs
        K = len(adjacencies)
        warn_msgs = self._validate_inputs(adjacencies, datasets, target_idx, K)
        n_vars = adjacencies[0].shape[0]

        # Ensure proper numpy arrays
        adjacencies = [np.asarray(a, dtype=np.float64) for a in adjacencies]
        datasets = [np.asarray(d, dtype=np.float64) for d in datasets]

        # Step 1: Structural Plasticity
        psi_S, parent_sets = self._compute_structural_plasticity(
            adjacencies, target_idx, n_vars, K
        )

        # Step 2: Parametric Plasticity
        psi_P, psi_P_per_group = self._compute_parametric_plasticity(
            datasets, target_idx, parent_sets, K
        )

        # Step 3: Emergence
        psi_E, mb_sizes = self._compute_emergence(adjacencies, target_idx, K)

        # Step 4: Context Sensitivity
        psi_CS, psi_P_subset_values = self._compute_context_sensitivity(
            datasets, target_idx, parent_sets, K
        )

        # Step 5: Confidence Intervals (optional)
        ci = None
        psi_S_ci = None
        psi_P_ci = None
        psi_E_ci = None
        psi_CS_ci = None

        if self.config.compute_cis:
            ci = self._compute_confidence_intervals(
                adjacencies, datasets, target_idx, parent_sets,
                psi_S, psi_P, psi_E, psi_CS,
                n_vars, K, dag_learner,
            )
            psi_S_ci = ci.psi_S_ci
            psi_P_ci = ci.psi_P_ci
            psi_E_ci = ci.psi_E_ci
            psi_CS_ci = ci.psi_CS_ci

        # Step 6: Classification (optional)
        classification = None
        if self.config.compute_classification:
            classification = self._classifier.classify(
                psi_S=psi_S,
                psi_P=psi_P,
                psi_E=psi_E,
                psi_CS=psi_CS,
                variable_idx=target_idx,
                psi_S_ci=psi_S_ci,
                psi_P_ci=psi_P_ci,
                psi_E_ci=psi_E_ci,
                psi_CS_ci=psi_CS_ci,
            )

        elapsed = time.perf_counter() - t0

        return PlasticityDescriptor(
            variable_idx=target_idx,
            variable_name=variable_name,
            psi_S=psi_S,
            psi_P=psi_P,
            psi_E=psi_E,
            psi_CS=psi_CS,
            psi_S_ci=psi_S_ci,
            psi_P_ci=psi_P_ci,
            psi_E_ci=psi_E_ci,
            psi_CS_ci=psi_CS_ci,
            classification=classification,
            parent_sets=parent_sets,
            markov_blanket_sizes=mb_sizes,
            psi_P_per_group=psi_P_per_group,
            psi_P_subset_values=psi_P_subset_values,
            n_contexts=K,
            computation_time=elapsed,
            warnings=warn_msgs,
        )

    # ------------------------------------------------------------------
    # Step 1: Structural Plasticity (ψ_S)
    # ------------------------------------------------------------------

    def _compute_structural_plasticity(
        self,
        adjacencies: list[NDArray],
        target_idx: int,
        n_vars: int,
        K: int,
    ) -> tuple[float, list[list[int]]]:
        """Step 1: Compute structural plasticity ψ_S.

        For each variable i, compute K binary parent indicator vectors.
        Multi-distribution JSD over parent indicators (discrete distributions).
        Apply sqrt for proper metric: ψ_S = sqrt(JSD).

        Parameters
        ----------
        adjacencies : K adjacency matrices
        target_idx : target variable
        n_vars : number of variables
        K : number of contexts

        Returns
        -------
        (psi_S, parent_sets) where parent_sets[k] = list of parent indices
        """
        if K < 2:
            parent_sets = [_parent_set(adjacencies[0], target_idx)] if K == 1 else [[]]
            return 0.0, parent_sets

        # Extract parent indicator vectors: indicator[k][j] = 1 if j is parent of target in context k
        parent_indicators = np.zeros((K, n_vars), dtype=np.float64)
        parent_sets = []
        for k in range(K):
            ps = _parent_set(adjacencies[k], target_idx)
            parent_sets.append(ps)
            for j in ps:
                parent_indicators[k, j] = 1.0

        # Check for identical parent sets (ψ_S = 0)
        if self._all_parent_sets_identical(parent_sets):
            return 0.0, parent_sets

        # Check for completely different parent sets
        # (each context has a unique set)

        # Compute variable-wise JSD and average
        jsd_values = []
        for j in range(n_vars):
            if j == target_idx:
                continue
            col = parent_indicators[:, j]
            # Skip if all same
            if np.all(col == col[0]):
                jsd_values.append(0.0)
                continue
            jsd_j = _jsd_bernoulli_multi(col)
            jsd_values.append(jsd_j)

        if len(jsd_values) == 0:
            return 0.0, parent_sets

        # Average JSD across variables, then sqrt for metric
        mean_jsd = float(np.mean(jsd_values))
        psi_S = math.sqrt(max(mean_jsd, 0.0))

        return psi_S, parent_sets

    @staticmethod
    def _all_parent_sets_identical(parent_sets: list[list[int]]) -> bool:
        """Check if all parent sets are identical."""
        if len(parent_sets) <= 1:
            return True
        first = set(parent_sets[0])
        return all(set(ps) == first for ps in parent_sets[1:])

    # ------------------------------------------------------------------
    # Step 2: Parametric Plasticity (ψ_P)
    # ------------------------------------------------------------------

    def _compute_parametric_plasticity(
        self,
        datasets: list[NDArray],
        target_idx: int,
        parent_sets: list[list[int]],
        K: int,
    ) -> tuple[float, dict[str, float]]:
        """Step 2: Compute parametric plasticity ψ_P.

        Groups contexts by parent set (same structure).
        Within each group, computes pairwise sqrt(JSD) between
        conditional distributions (Gaussian closed-form).
        Averages over all within-group pairs.

        Parameters
        ----------
        datasets : K data arrays
        target_idx : target variable
        parent_sets : parent set per context
        K : number of contexts

        Returns
        -------
        (psi_P, psi_P_per_group) where psi_P_per_group maps
        group_key -> within-group mean sqrt(JSD)
        """
        if K < 2:
            return 0.0, {}

        # Group contexts by parent set
        groups: dict[tuple[int, ...], list[int]] = {}
        for k in range(K):
            key = tuple(sorted(parent_sets[k]))
            if key not in groups:
                groups[key] = []
            groups[key].append(k)

        # Fit regression models per context
        models = {}
        for k in range(K):
            parents = parent_sets[k]
            data = datasets[k]
            y = data[:, target_idx]

            if len(parents) == 0:
                coefs, intercept, res_var = _fit_ols_no_parents(y)
            else:
                X = data[:, parents]
                coefs, intercept, res_var = _fit_ols(X, y)

            models[k] = {
                "coefficients": coefs,
                "intercept": intercept,
                "residual_var": res_var,
                "parents": parents,
                "n_samples": data.shape[0],
            }

        # Within-group pairwise sqrt(JSD)
        total_sqrt_jsd = 0.0
        n_pairs = 0
        psi_P_per_group = {}

        for key, members in groups.items():
            if len(members) < 2:
                psi_P_per_group[str(key)] = 0.0
                continue

            group_sqrt_jsd = 0.0
            group_pairs = 0

            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    ki, kj = members[i], members[j]
                    mi, mj = models[ki], models[kj]

                    # Handle groups with different numbers of parents
                    # (shouldn't happen within a group, but be safe)
                    if len(mi["coefficients"]) != len(mj["coefficients"]):
                        warnings.warn(
                            f"Coefficient length mismatch in group {key}: "
                            f"{len(mi['coefficients'])} vs {len(mj['coefficients'])}",
                            stacklevel=2,
                        )
                        continue

                    if len(mi["coefficients"]) > 0:
                        jsd = _jsd_gaussian_regression(
                            mi["coefficients"], mi["intercept"], mi["residual_var"],
                            mj["coefficients"], mj["intercept"], mj["residual_var"],
                        )
                    else:
                        jsd = _jsd_gaussian(
                            mi["intercept"], mi["residual_var"],
                            mj["intercept"], mj["residual_var"],
                        )

                    group_sqrt_jsd += math.sqrt(max(jsd, 0.0))
                    group_pairs += 1

            if group_pairs > 0:
                psi_P_per_group[str(key)] = group_sqrt_jsd / group_pairs
                total_sqrt_jsd += group_sqrt_jsd
                n_pairs += group_pairs
            else:
                psi_P_per_group[str(key)] = 0.0

        if n_pairs == 0:
            return 0.0, psi_P_per_group

        psi_P = total_sqrt_jsd / n_pairs
        return psi_P, psi_P_per_group

    # ------------------------------------------------------------------
    # Step 3: Emergence (ψ_E)
    # ------------------------------------------------------------------

    def _compute_emergence(
        self,
        adjacencies: list[NDArray],
        target_idx: int,
        K: int,
    ) -> tuple[float, list[int]]:
        """Step 3: Compute emergence ψ_E.

        ψ_E = 1 - min(|MB|) / (max(|MB|) + 1)

        High ψ_E means the variable is causally isolated in some contexts
        but highly connected in others.

        Handles empty Markov blankets gracefully.

        Parameters
        ----------
        adjacencies : K adjacency matrices
        target_idx : target variable
        K : number of contexts

        Returns
        -------
        (psi_E, markov_blanket_sizes)
        """
        if K == 0:
            return 0.0, []

        mb_sizes = []
        for k in range(K):
            mb = _markov_blanket(adjacencies[k], target_idx)
            mb_sizes.append(len(mb))

        if K == 1:
            return 0.0, mb_sizes

        min_mb = min(mb_sizes)
        max_mb = max(mb_sizes)

        if max_mb == 0:
            # Empty Markov blanket in all contexts
            return 0.0, mb_sizes

        psi_E = 1.0 - min_mb / (max_mb + 1)
        return psi_E, mb_sizes

    # ------------------------------------------------------------------
    # Step 4: Context Sensitivity (ψ_CS)
    # ------------------------------------------------------------------

    def _compute_context_sensitivity(
        self,
        datasets: list[NDArray],
        target_idx: int,
        parent_sets: list[list[int]],
        K: int,
    ) -> tuple[float, Optional[NDArray]]:
        """Step 4: Compute context sensitivity ψ_CS.

        Coefficient of variation of ψ_P across context subsets.
        Uses subsets of size ceil(K/2), Monte Carlo sampling of
        n_context_subsets random subsets.

        High ψ_CS = plasticity concentrated in specific contexts.

        Parameters
        ----------
        datasets : K data arrays
        target_idx : target variable
        parent_sets : per-context parent sets
        K : number of contexts

        Returns
        -------
        (psi_CS, psi_P_subset_values)
        """
        if K < 3:
            return 0.0, None

        n_subsets = self.config.n_context_subsets
        subset_size = max(2, math.ceil(K / 2))

        rng = np.random.default_rng(self.config.random_state)
        psi_P_values = np.zeros(n_subsets, dtype=np.float64)

        for s in range(n_subsets):
            indices = rng.choice(K, size=subset_size, replace=False)
            sub_datasets = [datasets[k] for k in indices]
            sub_parents = [parent_sets[k] for k in indices]

            psi_P_sub, _ = self._compute_parametric_plasticity(
                sub_datasets, target_idx, sub_parents, len(indices)
            )
            psi_P_values[s] = psi_P_sub

        mean_psi = float(np.mean(psi_P_values))
        if n_subsets > 1:
            std_psi = float(np.std(psi_P_values, ddof=1))
        else:
            std_psi = 0.0

        if abs(mean_psi) < 1e-10:
            return 0.0, psi_P_values

        psi_CS = std_psi / abs(mean_psi)
        return psi_CS, psi_P_values

    # ------------------------------------------------------------------
    # Step 5: Confidence Intervals
    # ------------------------------------------------------------------

    def _compute_confidence_intervals(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        target_idx: int,
        parent_sets: list[list[int]],
        psi_S: float,
        psi_P: float,
        psi_E: float,
        psi_CS: float,
        n_vars: int,
        K: int,
        dag_learner: Optional[Callable] = None,
    ) -> DescriptorCI:
        """Step 5: Compute confidence intervals via stability selection
        (structural) and parametric bootstrap (parametric).

        Parameters
        ----------
        adjacencies, datasets, target_idx, parent_sets : from previous steps
        psi_S, psi_P, psi_E, psi_CS : point estimates
        n_vars, K : dimensions
        dag_learner : optional DAG learner for structural CI

        Returns
        -------
        DescriptorCI
        """
        # Structural CI via stability selection (if DAG learner available)
        stab_result = None
        psi_S_ci = (psi_S, psi_S)
        psi_E_ci = (psi_E, psi_E)

        if dag_learner is not None:
            try:
                stab_selector = StabilitySelector(
                    n_rounds=self.config.n_stability_rounds,
                    subsample_fraction=self.config.subsample_fraction,
                    ci_level=self.config.ci_level,
                    random_state=self.config.random_state,
                )
                stab_result = stab_selector.compute_structural_ci(
                    datasets=datasets,
                    target_idx=target_idx,
                    dag_learner=dag_learner,
                    n_variables=n_vars,
                )
                psi_S_ci = (stab_result.ci_lower, stab_result.ci_upper)

                # Emergence CI
                psi_E_ci = stab_selector.compute_emergence_ci(
                    datasets=datasets,
                    target_idx=target_idx,
                    dag_learner=dag_learner,
                    n_variables=n_vars,
                )
            except Exception as e:
                warnings.warn(f"Structural CI computation failed: {e}", stacklevel=2)
        else:
            # Without DAG learner, use bootstrap perturbation of adjacencies
            psi_S_ci = self._bootstrap_structural_ci(
                adjacencies, target_idx, n_vars, K
            )
            psi_E_ci = self._bootstrap_emergence_ci(
                adjacencies, target_idx, K
            )

        # Parametric CI via bootstrap
        boot_result = None
        psi_P_ci = (psi_P, psi_P)
        psi_CS_ci = (psi_CS, psi_CS)

        try:
            bootstrap = ParametricBootstrap(
                n_bootstrap=self.config.n_bootstrap,
                ci_level=self.config.ci_level,
                ci_method=self.config.ci_method,
                random_state=self.config.random_state,
            )
            boot_result = bootstrap.compute_parametric_ci(
                datasets=datasets,
                target_idx=target_idx,
                parent_sets=parent_sets,
            )
            psi_P_ci = (boot_result.ci_lower, boot_result.ci_upper)

            # Context sensitivity CI
            cs_result = bootstrap.compute_context_sensitivity_ci(
                datasets=datasets,
                target_idx=target_idx,
                parent_sets=parent_sets,
                n_subsets=self.config.n_context_subsets,
            )
            psi_CS_ci = (cs_result.ci_lower, cs_result.ci_upper)
        except Exception as e:
            warnings.warn(f"Parametric CI computation failed: {e}", stacklevel=2)

        return DescriptorCI(
            psi_S_ci=psi_S_ci,
            psi_P_ci=psi_P_ci,
            psi_E_ci=psi_E_ci,
            psi_CS_ci=psi_CS_ci,
            ci_level=self.config.ci_level,
            structural_stability=stab_result,
            parametric_bootstrap=boot_result,
        )

    def _bootstrap_structural_ci(
        self,
        adjacencies: list[NDArray],
        target_idx: int,
        n_vars: int,
        K: int,
    ) -> tuple[float, float]:
        """Bootstrap CI for structural plasticity when no DAG learner is available.

        Uses perturbation of adjacency matrices by flipping edges with
        small probability.
        """
        rng = np.random.default_rng(self.config.random_state)
        n_boot = min(self.config.n_bootstrap, 100)
        psi_S_samples = np.zeros(n_boot, dtype=np.float64)

        flip_prob = 0.05  # 5% edge flip probability

        for b in range(n_boot):
            perturbed_adjs = []
            for k in range(K):
                adj = adjacencies[k].copy()
                # Random edge flips
                flips = rng.random(adj.shape) < flip_prob
                np.fill_diagonal(flips, False)
                adj = np.where(flips, 1.0 - adj, adj)
                perturbed_adjs.append(adj)

            psi_S_b, _ = self._compute_structural_plasticity(
                perturbed_adjs, target_idx, n_vars, K
            )
            psi_S_samples[b] = psi_S_b

        alpha = 1 - self.config.ci_level
        return (
            float(np.percentile(psi_S_samples, 100 * alpha / 2)),
            float(np.percentile(psi_S_samples, 100 * (1 - alpha / 2))),
        )

    def _bootstrap_emergence_ci(
        self,
        adjacencies: list[NDArray],
        target_idx: int,
        K: int,
    ) -> tuple[float, float]:
        """Bootstrap CI for emergence."""
        rng = np.random.default_rng(
            self.config.random_state + 3000 if self.config.random_state else None
        )
        n_boot = min(self.config.n_bootstrap, 100)
        psi_E_samples = np.zeros(n_boot, dtype=np.float64)

        flip_prob = 0.05

        for b in range(n_boot):
            perturbed_adjs = []
            for k in range(K):
                adj = adjacencies[k].copy()
                flips = rng.random(adj.shape) < flip_prob
                np.fill_diagonal(flips, False)
                adj = np.where(flips, 1.0 - adj, adj)
                perturbed_adjs.append(adj)

            psi_E_b, _ = self._compute_emergence(perturbed_adjs, target_idx, K)
            psi_E_samples[b] = psi_E_b

        alpha = 1 - self.config.ci_level
        return (
            float(np.percentile(psi_E_samples, 100 * alpha / 2)),
            float(np.percentile(psi_E_samples, 100 * (1 - alpha / 2))),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        target_idx: int,
        K: int,
    ) -> list[str]:
        """Validate inputs and return list of warning messages."""
        warn_msgs = []

        if K == 0:
            raise ValueError("At least one context is required.")

        if len(adjacencies) != len(datasets):
            raise ValueError(
                f"Number of adjacencies ({len(adjacencies)}) != "
                f"number of datasets ({len(datasets)})."
            )

        n_vars = adjacencies[0].shape[0]
        for k, adj in enumerate(adjacencies):
            if adj.shape != (n_vars, n_vars):
                raise ValueError(
                    f"Adjacency {k} has shape {adj.shape}, expected ({n_vars}, {n_vars})."
                )

        for k, data in enumerate(datasets):
            if data.shape[1] != n_vars:
                raise ValueError(
                    f"Dataset {k} has {data.shape[1]} columns, expected {n_vars}."
                )
            if data.shape[0] < self.config.min_samples_warning:
                warn_msgs.append(
                    f"Context {k}: n={data.shape[0]} < {self.config.min_samples_warning}"
                )

        if not 0 <= target_idx < n_vars:
            raise ValueError(
                f"target_idx={target_idx} out of range [0, {n_vars})."
            )

        if K < 2:
            warn_msgs.append("Single context: all plasticity measures will be 0.")

        # Check for zero-variance variables
        for k, data in enumerate(datasets):
            vars_k = np.var(data, axis=0)
            zero_var = np.where(vars_k < 1e-15)[0]
            if len(zero_var) > 0:
                warn_msgs.append(
                    f"Context {k}: zero-variance variables at indices {zero_var.tolist()}"
                )

        return warn_msgs


# ---------------------------------------------------------------------------
# BatchPlasticityComputer
# ---------------------------------------------------------------------------

@dataclass
class BatchPlasticityResult:
    """Result from batch plasticity computation."""

    descriptors: list[PlasticityDescriptor]
    batch_classification: Optional[BatchClassificationResult]
    n_variables: int
    n_contexts: int
    summary_stats: dict
    computation_time: float
    metadata: dict = field(default_factory=dict)

    def descriptor_matrix(self) -> NDArray:
        """Return (n_vars, 4) matrix of descriptor vectors."""
        return np.array([d.descriptor_vector for d in self.descriptors])

    def by_category(
        self, category: PlasticityCategory | str
    ) -> list[PlasticityDescriptor]:
        """Return descriptors for a specific category."""
        cat = PlasticityCategory(category) if isinstance(category, str) else category
        return [
            d for d in self.descriptors
            if d.classification is not None
            and d.classification.primary_category == cat
        ]

    def most_plastic(self, n: int = 5) -> list[PlasticityDescriptor]:
        """Return the n most plastic variables."""
        return sorted(
            self.descriptors,
            key=lambda d: max(d.psi_S, d.psi_P, d.psi_E),
            reverse=True,
        )[:n]

    def invariant_variables(self) -> list[PlasticityDescriptor]:
        """Return invariant descriptors."""
        return [
            d for d in self.descriptors
            if d.classification is not None
            and d.classification.primary_category == PlasticityCategory.INVARIANT
        ]


class BatchPlasticityComputer:
    """Compute plasticity descriptors for all variables in an MCCM.

    Orchestrates PlasticityComputer across all variables, with optional
    parallelism, progress reporting, and summary statistics.

    Parameters
    ----------
    config : PlasticityConfig
    n_jobs : int
        Number of parallel workers (default 1, -1 for all CPUs).
    progress : bool
        Whether to show progress bar (default True).
    variable_names : list of variable name strings (optional).

    Examples
    --------
    >>> batch = BatchPlasticityComputer(n_jobs=4)
    >>> result = batch.compute(
    ...     adjacencies=[adj1, adj2, adj3],
    ...     datasets=[data1, data2, data3],
    ... )
    >>> for desc in result.most_plastic(3):
    ...     print(desc.summary())
    """

    def __init__(
        self,
        config: Optional[PlasticityConfig] = None,
        n_jobs: int = 1,
        progress: bool = True,
        variable_names: Optional[list[str]] = None,
    ):
        self.config = config or PlasticityConfig()
        self.n_jobs = n_jobs
        self.progress = progress
        self.variable_names = variable_names

    def compute(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        variable_indices: Optional[list[int]] = None,
        dag_learner: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> BatchPlasticityResult:
        """Compute plasticity descriptors for all (or selected) variables.

        Parameters
        ----------
        adjacencies : list of K adjacency matrices
        datasets : list of K data arrays
        variable_indices : optional subset of variables to compute
        dag_learner : optional DAG learner for stability selection

        Returns
        -------
        BatchPlasticityResult
        """
        import time
        t0 = time.perf_counter()

        K = len(adjacencies)
        n_vars = adjacencies[0].shape[0]

        if variable_indices is None:
            variable_indices = list(range(n_vars))

        # Generate variable names
        names = self.variable_names
        if names is None:
            names = [f"X_{i}" for i in range(n_vars)]

        computer = PlasticityComputer(config=self.config)

        # Compute descriptors
        if self.n_jobs == 1:
            descriptors = self._compute_sequential(
                computer, adjacencies, datasets, variable_indices, names, dag_learner
            )
        else:
            descriptors = self._compute_parallel(
                computer, adjacencies, datasets, variable_indices, names,
                dag_learner, self.n_jobs
            )

        # Batch classification
        batch_class = None
        if self.config.compute_classification:
            classifier = PlasticityClassifier(
                thresholds=self.config.thresholds(),
                variable_names=names,
            )
            desc_dicts = [
                {
                    "psi_S": d.psi_S,
                    "psi_P": d.psi_P,
                    "psi_E": d.psi_E,
                    "psi_CS": d.psi_CS,
                    "variable_idx": d.variable_idx,
                    "psi_S_ci": d.psi_S_ci,
                    "psi_P_ci": d.psi_P_ci,
                    "psi_E_ci": d.psi_E_ci,
                    "psi_CS_ci": d.psi_CS_ci,
                }
                for d in descriptors
            ]
            batch_class = classifier.classify_batch(desc_dicts)

        # Summary statistics
        summary = self._compute_summary(descriptors, batch_class)

        elapsed = time.perf_counter() - t0

        return BatchPlasticityResult(
            descriptors=descriptors,
            batch_classification=batch_class,
            n_variables=len(variable_indices),
            n_contexts=K,
            summary_stats=summary,
            computation_time=elapsed,
        )

    def _compute_sequential(
        self,
        computer: PlasticityComputer,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        variable_indices: list[int],
        names: list[str],
        dag_learner: Optional[Callable],
    ) -> list[PlasticityDescriptor]:
        """Compute descriptors sequentially."""
        descriptors = []
        iterator = variable_indices

        try:
            from tqdm import tqdm
            if self.progress:
                iterator = tqdm(variable_indices, desc="Computing plasticity descriptors")
        except ImportError:
            pass

        for idx in iterator:
            name = names[idx] if idx < len(names) else f"X_{idx}"
            desc = computer.compute(
                adjacencies=adjacencies,
                datasets=datasets,
                target_idx=idx,
                variable_name=name,
                dag_learner=dag_learner,
            )
            descriptors.append(desc)

        return descriptors

    def _compute_parallel(
        self,
        computer: PlasticityComputer,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        variable_indices: list[int],
        names: list[str],
        dag_learner: Optional[Callable],
        n_jobs: int,
    ) -> list[PlasticityDescriptor]:
        """Compute descriptors in parallel using thread pool.

        Note: We use ThreadPoolExecutor because numpy releases the GIL.
        """
        import os

        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1

        descriptors = [None] * len(variable_indices)

        def _compute_one(i: int, idx: int) -> tuple[int, PlasticityDescriptor]:
            name = names[idx] if idx < len(names) else f"X_{idx}"
            desc = computer.compute(
                adjacencies=adjacencies,
                datasets=datasets,
                target_idx=idx,
                variable_name=name,
                dag_learner=dag_learner,
            )
            return i, desc

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_compute_one, i, idx): i
                for i, idx in enumerate(variable_indices)
            }

            try:
                from tqdm import tqdm
                if self.progress:
                    pbar = tqdm(total=len(variable_indices), desc="Computing descriptors")
                else:
                    pbar = None
            except ImportError:
                pbar = None

            for future in as_completed(futures):
                i, desc = future.result()
                descriptors[i] = desc
                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

        return descriptors

    def _compute_summary(
        self,
        descriptors: list[PlasticityDescriptor],
        batch_class: Optional[BatchClassificationResult],
    ) -> dict:
        """Compute summary statistics from batch results."""
        if not descriptors:
            return {"n_variables": 0}

        psi_S = np.array([d.psi_S for d in descriptors])
        psi_P = np.array([d.psi_P for d in descriptors])
        psi_E = np.array([d.psi_E for d in descriptors])
        psi_CS = np.array([d.psi_CS for d in descriptors])

        summary = {
            "n_variables": len(descriptors),
            "psi_S": {
                "mean": float(np.mean(psi_S)),
                "std": float(np.std(psi_S, ddof=1)) if len(psi_S) > 1 else 0.0,
                "min": float(np.min(psi_S)),
                "max": float(np.max(psi_S)),
                "median": float(np.median(psi_S)),
            },
            "psi_P": {
                "mean": float(np.mean(psi_P)),
                "std": float(np.std(psi_P, ddof=1)) if len(psi_P) > 1 else 0.0,
                "min": float(np.min(psi_P)),
                "max": float(np.max(psi_P)),
                "median": float(np.median(psi_P)),
            },
            "psi_E": {
                "mean": float(np.mean(psi_E)),
                "std": float(np.std(psi_E, ddof=1)) if len(psi_E) > 1 else 0.0,
                "min": float(np.min(psi_E)),
                "max": float(np.max(psi_E)),
                "median": float(np.median(psi_E)),
            },
            "psi_CS": {
                "mean": float(np.mean(psi_CS)),
                "std": float(np.std(psi_CS, ddof=1)) if len(psi_CS) > 1 else 0.0,
                "min": float(np.min(psi_CS)),
                "max": float(np.max(psi_CS)),
                "median": float(np.median(psi_CS)),
            },
        }

        # Correlations between descriptor components
        if len(descriptors) > 2:
            desc_matrix = np.column_stack([psi_S, psi_P, psi_E, psi_CS])
            corr = np.corrcoef(desc_matrix.T)
            summary["correlations"] = {
                "S_P": float(corr[0, 1]),
                "S_E": float(corr[0, 2]),
                "S_CS": float(corr[0, 3]),
                "P_E": float(corr[1, 2]),
                "P_CS": float(corr[1, 3]),
                "E_CS": float(corr[2, 3]),
            }

        if batch_class is not None:
            summary["category_distribution"] = batch_class.category_distribution

        return summary

    def visualization_data(
        self,
        result: BatchPlasticityResult,
    ) -> dict:
        """Prepare data for visualization.

        Returns dict with arrays suitable for scatter plots, heatmaps, etc.
        """
        descriptors = result.descriptors

        data = {
            "variable_names": [d.variable_name or f"X_{d.variable_idx}" for d in descriptors],
            "variable_indices": [d.variable_idx for d in descriptors],
            "psi_S": [d.psi_S for d in descriptors],
            "psi_P": [d.psi_P for d in descriptors],
            "psi_E": [d.psi_E for d in descriptors],
            "psi_CS": [d.psi_CS for d in descriptors],
            "descriptor_matrix": result.descriptor_matrix().tolist(),
        }

        # Categories
        if result.batch_classification:
            data["categories"] = [
                r.primary_category.value
                for r in result.batch_classification.results
            ]
            data["confidences"] = [
                r.confidence for r in result.batch_classification.results
            ]

        # CIs
        if any(d.has_cis for d in descriptors):
            data["psi_S_ci_lower"] = [d.psi_S_ci[0] if d.psi_S_ci else d.psi_S for d in descriptors]
            data["psi_S_ci_upper"] = [d.psi_S_ci[1] if d.psi_S_ci else d.psi_S for d in descriptors]
            data["psi_P_ci_lower"] = [d.psi_P_ci[0] if d.psi_P_ci else d.psi_P for d in descriptors]
            data["psi_P_ci_upper"] = [d.psi_P_ci[1] if d.psi_P_ci else d.psi_P for d in descriptors]

        return data
