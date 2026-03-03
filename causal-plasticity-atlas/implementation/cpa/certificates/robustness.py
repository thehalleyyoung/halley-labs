"""
Robustness Certificate Generation (ALG5).

Generates certificates of mechanism stability by combining stability
selection for structural invariance with parametric bootstrap for
upper confidence bounds on mechanism divergence.

Certificate types (ordered by strength):
    STRONG_INVARIANCE   — UCB < 0.01: mechanism is structurally and
                          parametrically invariant with high confidence.
    PARAMETRIC_STABILITY — UCB ≤ τ: mechanism parameters are stable
                          within tolerance τ.
    CANNOT_ISSUE         — UCB > τ: insufficient evidence of stability.

Classes
-------
CertificateGenerator   — Full ALG5 pipeline.
CertificateValidator   — Certificate assumption validation.
CertificateReport      — Human-readable certificate reports.

Theory reference: ALG5 in the CPA specification.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from cpa.certificates.stability import (
    BootstrapEngine,
    BootstrapResult,
    StabilitySelectionEngine,
    StabilityResult,
)


# ---------------------------------------------------------------------------
# Configuration & enumerations
# ---------------------------------------------------------------------------

class CertificateType(str, Enum):
    """Certificate strength levels."""

    STRONG_INVARIANCE = "strong_invariance"
    PARAMETRIC_STABILITY = "parametric_stability"
    CANNOT_ISSUE = "cannot_issue"

    @property
    def strength_level(self) -> int:
        return {
            "strong_invariance": 2,
            "parametric_stability": 1,
            "cannot_issue": 0,
        }[self.value]

    def __lt__(self, other: CertificateType) -> bool:
        return self.strength_level < other.strength_level

    def __le__(self, other: CertificateType) -> bool:
        return self.strength_level <= other.strength_level


@dataclass
class CertificateConfig:
    """Configuration for certificate generation.

    Parameters
    ----------
    n_stability_rounds : int
        Stability selection rounds (default 100).
    subsample_fraction : float
        Subsample fraction (default 0.5).
    stability_upper_threshold : float
        Selection probability above which an edge is declared stable (default 0.6).
    stability_lower_threshold : float
        Selection probability below which an edge is absent (default 0.4).
    n_bootstrap : int
        Bootstrap replicates for UCB (default 1000).
    beta : float
        Confidence level for UCB: (1 - beta)-quantile (default 0.05).
    tau : float
        Tolerance threshold for stability certificate (default 0.5).
    strong_invariance_threshold : float
        UCB threshold for strong invariance (default 0.01).
    min_samples_warning : int
        Sample size warning threshold (default 200).
    random_state : int or None
        Random seed.
    """

    n_stability_rounds: int = 100
    subsample_fraction: float = 0.5
    stability_upper_threshold: float = 0.6
    stability_lower_threshold: float = 0.4
    n_bootstrap: int = 1000
    beta: float = 0.05
    tau: float = 0.5
    strong_invariance_threshold: float = 0.01
    min_samples_warning: int = 200
    random_state: Optional[int] = None


# ---------------------------------------------------------------------------
# Certificate dataclass
# ---------------------------------------------------------------------------

@dataclass
class RobustnessCertificate:
    """A robustness certificate for a mechanism.

    Contains the certificate type, upper confidence bound, margin,
    and all evidence used to generate the certificate.
    """

    variable_idx: int
    variable_name: Optional[str]
    certificate_type: CertificateType
    ucb: float                        # Upper confidence bound on max sqrt(JSD)
    tau: float                        # Tolerance threshold
    robustness_margin: float          # tau - ucb (positive = certified)
    beta: float                       # Confidence level

    # Structural invariance evidence
    structural_invariance: bool
    stability_probabilities: Optional[NDArray] = None
    stable_edges: Optional[list[tuple[int, int]]] = None
    unstable_edges: Optional[list[tuple[int, int]]] = None

    # Parametric stability evidence
    pairwise_sqrt_jsd: Optional[NDArray] = None
    max_sqrt_jsd: float = 0.0
    bootstrap_ucb_distribution: Optional[NDArray] = None

    # Metadata
    n_contexts: int = 0
    n_samples_per_context: Optional[list[int]] = None
    assumptions: list[str] = field(default_factory=list)
    warnings_list: list[str] = field(default_factory=list)
    computation_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def is_certified(self) -> bool:
        """True if a certificate was issued."""
        return self.certificate_type != CertificateType.CANNOT_ISSUE

    @property
    def is_strongly_invariant(self) -> bool:
        """True if mechanism has strong invariance certificate."""
        return self.certificate_type == CertificateType.STRONG_INVARIANCE

    def to_dict(self) -> dict:
        """Serialize certificate to dictionary."""
        d = {
            "variable_idx": self.variable_idx,
            "variable_name": self.variable_name,
            "certificate_type": self.certificate_type.value,
            "ucb": self.ucb,
            "tau": self.tau,
            "robustness_margin": self.robustness_margin,
            "beta": self.beta,
            "structural_invariance": self.structural_invariance,
            "max_sqrt_jsd": self.max_sqrt_jsd,
            "n_contexts": self.n_contexts,
            "assumptions": self.assumptions,
            "warnings": self.warnings_list,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RobustnessCertificate:
        """Deserialize from dictionary."""
        return cls(
            variable_idx=d["variable_idx"],
            variable_name=d.get("variable_name"),
            certificate_type=CertificateType(d["certificate_type"]),
            ucb=d["ucb"],
            tau=d["tau"],
            robustness_margin=d["robustness_margin"],
            beta=d["beta"],
            structural_invariance=d["structural_invariance"],
            max_sqrt_jsd=d.get("max_sqrt_jsd", 0.0),
            n_contexts=d.get("n_contexts", 0),
            assumptions=d.get("assumptions", []),
            warnings_list=d.get("warnings", []),
        )

    def summary(self) -> str:
        """One-line summary."""
        name = self.variable_name or f"X_{self.variable_idx}"
        return (
            f"{name}: {self.certificate_type.value} "
            f"(UCB={self.ucb:.4f}, τ={self.tau:.3f}, "
            f"margin={self.robustness_margin:+.4f})"
        )


@dataclass
class BatchCertificateResult:
    """Result from batch certificate generation."""

    certificates: list[RobustnessCertificate]
    n_variables: int
    n_certified: int
    n_strong: int
    n_parametric: int
    n_failed: int
    summary_stats: dict
    computation_time: float
    metadata: dict = field(default_factory=dict)

    def by_type(
        self, cert_type: CertificateType | str
    ) -> list[RobustnessCertificate]:
        """Return certificates of a specific type."""
        ct = CertificateType(cert_type) if isinstance(cert_type, str) else cert_type
        return [c for c in self.certificates if c.certificate_type == ct]

    def certified_variables(self) -> list[RobustnessCertificate]:
        """Return all certified variables."""
        return [c for c in self.certificates if c.is_certified]


# ---------------------------------------------------------------------------
# CertificateGenerator — ALG5
# ---------------------------------------------------------------------------

class CertificateGenerator:
    """Robustness Certificate Generation (ALG5).

    Generates certificates of mechanism stability via:
        Step 1: Structural invariance via stability selection
        Step 2: Parametric stability check (pairwise sqrt(JSD))
        Step 3: Bootstrap upper confidence bound (UCB)
        Step 4: Certificate decision
        Step 5: Metadata and warnings

    Parameters
    ----------
    config : CertificateConfig
        Configuration parameters.

    Examples
    --------
    >>> generator = CertificateGenerator()
    >>> cert = generator.generate(
    ...     adjacencies=[adj1, adj2, adj3],
    ...     datasets=[data1, data2, data3],
    ...     target_idx=0,
    ...     dag_learner=my_learner,
    ... )
    >>> print(cert.summary())
    """

    def __init__(self, config: Optional[CertificateConfig] = None):
        self.config = config or CertificateConfig()

    def generate(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        target_idx: int,
        dag_learner: Callable[[NDArray], NDArray],
        variable_name: Optional[str] = None,
    ) -> RobustnessCertificate:
        """Generate a robustness certificate for a single mechanism.

        Parameters
        ----------
        adjacencies : list of K adjacency matrices
        datasets : list of K data arrays
        target_idx : index of the variable
        dag_learner : function(data) -> adjacency matrix
        variable_name : optional variable name

        Returns
        -------
        RobustnessCertificate
        """
        import time
        t0 = time.perf_counter()

        K = len(adjacencies)
        n_vars = adjacencies[0].shape[0]
        warn_msgs = self._validate_inputs(adjacencies, datasets, target_idx, K)

        # Ensure numpy arrays
        adjacencies = [np.asarray(a, dtype=np.float64) for a in adjacencies]
        datasets = [np.asarray(d, dtype=np.float64) for d in datasets]

        # Track assumptions
        assumptions = [
            "Linear Gaussian SCM assumed for each context",
            f"Stability selection: {self.config.n_stability_rounds} rounds, "
            f"{self.config.subsample_fraction:.0%} subsample",
            f"Bootstrap UCB: {self.config.n_bootstrap} replicates, "
            f"β={self.config.beta:.2f}",
            f"Tolerance threshold τ = {self.config.tau:.3f}",
        ]

        # Step 1: Structural Invariance via Stability Selection
        (
            is_structurally_invariant,
            stability_probs,
            stable_edges,
            unstable_edges,
        ) = self._step1_structural_invariance(
            datasets, target_idx, dag_learner, n_vars, K
        )

        if not is_structurally_invariant:
            # Cannot certify — structure changes across contexts
            elapsed = time.perf_counter() - t0
            return RobustnessCertificate(
                variable_idx=target_idx,
                variable_name=variable_name,
                certificate_type=CertificateType.CANNOT_ISSUE,
                ucb=float("inf"),
                tau=self.config.tau,
                robustness_margin=-float("inf"),
                beta=self.config.beta,
                structural_invariance=False,
                stability_probabilities=stability_probs,
                stable_edges=stable_edges,
                unstable_edges=unstable_edges,
                n_contexts=K,
                n_samples_per_context=[d.shape[0] for d in datasets],
                assumptions=assumptions,
                warnings_list=warn_msgs + ["Structural invariance not established"],
                computation_time=elapsed,
            )

        # Step 2: Parametric Stability Check
        parent_set = self._consensus_parent_set(
            adjacencies, target_idx, n_vars
        )
        pairwise_jsd, max_jsd = self._step2_parametric_stability(
            datasets, target_idx, parent_set, K
        )

        # Step 3: Bootstrap UCB
        ucb, boot_distribution = self._step3_bootstrap_ucb(
            datasets, target_idx, parent_set, K
        )

        # Step 4: Certificate Decision
        cert_type, margin = self._step4_decision(ucb)

        # Step 5: Metadata
        elapsed = time.perf_counter() - t0

        return RobustnessCertificate(
            variable_idx=target_idx,
            variable_name=variable_name,
            certificate_type=cert_type,
            ucb=ucb,
            tau=self.config.tau,
            robustness_margin=margin,
            beta=self.config.beta,
            structural_invariance=True,
            stability_probabilities=stability_probs,
            stable_edges=stable_edges,
            unstable_edges=unstable_edges,
            pairwise_sqrt_jsd=pairwise_jsd,
            max_sqrt_jsd=max_jsd,
            bootstrap_ucb_distribution=boot_distribution,
            n_contexts=K,
            n_samples_per_context=[d.shape[0] for d in datasets],
            assumptions=assumptions,
            warnings_list=warn_msgs,
            computation_time=elapsed,
        )

    def generate_batch(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        dag_learner: Callable[[NDArray], NDArray],
        variable_indices: Optional[list[int]] = None,
        variable_names: Optional[list[str]] = None,
    ) -> BatchCertificateResult:
        """Generate certificates for multiple variables.

        Parameters
        ----------
        adjacencies, datasets : as above
        dag_learner : DAG learning function
        variable_indices : optional subset of variables
        variable_names : optional variable names

        Returns
        -------
        BatchCertificateResult
        """
        import time
        t0 = time.perf_counter()

        n_vars = adjacencies[0].shape[0]
        if variable_indices is None:
            variable_indices = list(range(n_vars))

        names = variable_names or [f"X_{i}" for i in range(n_vars)]

        certificates = []
        for idx in variable_indices:
            name = names[idx] if idx < len(names) else f"X_{idx}"
            cert = self.generate(
                adjacencies, datasets, idx, dag_learner, name
            )
            certificates.append(cert)

        n_strong = sum(1 for c in certificates if c.is_strongly_invariant)
        n_param = sum(
            1 for c in certificates
            if c.certificate_type == CertificateType.PARAMETRIC_STABILITY
        )
        n_failed = sum(
            1 for c in certificates
            if c.certificate_type == CertificateType.CANNOT_ISSUE
        )
        n_certified = n_strong + n_param

        # Summary statistics
        ucbs = [c.ucb for c in certificates if c.ucb < float("inf")]
        summary = {
            "n_variables": len(variable_indices),
            "n_certified": n_certified,
            "n_strong_invariance": n_strong,
            "n_parametric_stability": n_param,
            "n_cannot_issue": n_failed,
            "certification_rate": n_certified / max(len(variable_indices), 1),
        }
        if ucbs:
            summary["ucb_mean"] = float(np.mean(ucbs))
            summary["ucb_median"] = float(np.median(ucbs))
            summary["ucb_max"] = float(np.max(ucbs))

        elapsed = time.perf_counter() - t0

        return BatchCertificateResult(
            certificates=certificates,
            n_variables=len(variable_indices),
            n_certified=n_certified,
            n_strong=n_strong,
            n_parametric=n_param,
            n_failed=n_failed,
            summary_stats=summary,
            computation_time=elapsed,
        )

    # ---- Step 1: Structural Invariance ----

    def _step1_structural_invariance(
        self,
        datasets: list[NDArray],
        target_idx: int,
        dag_learner: Callable[[NDArray], NDArray],
        n_vars: int,
        K: int,
    ) -> tuple[bool, NDArray, list[tuple[int, int]], list[tuple[int, int]]]:
        """Step 1: Assess structural invariance via stability selection.

        Subsample 50% of data, re-estimate DAG, compute selection
        probabilities for each potential parent. Declare structural
        invariance if probabilities consistently > 0.6 or < 0.4 across
        all contexts.

        Returns
        -------
        (is_invariant, stability_probs, stable_edges, unstable_edges)
        """
        engine = StabilitySelectionEngine(
            n_rounds=self.config.n_stability_rounds,
            subsample_fraction=self.config.subsample_fraction,
            upper_threshold=self.config.stability_upper_threshold,
            lower_threshold=self.config.stability_lower_threshold,
            random_state=self.config.random_state,
        )

        # Run stability selection for each context
        per_context_probs = np.zeros((K, n_vars), dtype=np.float64)
        per_context_stable = [set() for _ in range(K)]
        per_context_unstable = [set() for _ in range(K)]

        # Map predictor indices back to variable indices
        predictor_to_var = [v for v in range(n_vars) if v != target_idx]

        for k in range(K):
            probs, stable, unstable = engine.run_variable_selection(
                data=datasets[k],
                target_idx=target_idx,
                selector_fn=self._make_parent_selector(dag_learner, target_idx),
            )
            # Map predictor-space probs to variable-space
            for pi, prob_val in enumerate(probs):
                if pi < len(predictor_to_var):
                    per_context_probs[k, predictor_to_var[pi]] = prob_val
            # Map predictor-space indices to variable indices
            per_context_stable[k] = {predictor_to_var[s] for s in stable
                                     if s < len(predictor_to_var)}
            per_context_unstable[k] = {predictor_to_var[u] for u in unstable
                                       if u < len(predictor_to_var)}

        # Check consistency across contexts
        # A parent is declared invariantly present if stable in ALL contexts
        # A parent is declared invariantly absent if unstable in ALL contexts
        all_vars = set(range(n_vars)) - {target_idx}
        consistent_present = set()
        consistent_absent = set()
        inconsistent = set()

        for j in all_vars:
            all_stable = all(j in per_context_stable[k] for k in range(K))
            all_unstable = all(j in per_context_unstable[k] for k in range(K))
            if all_stable:
                consistent_present.add(j)
            elif all_unstable:
                consistent_absent.add(j)
            else:
                inconsistent.add(j)

        is_invariant = len(inconsistent) == 0
        mean_probs = np.mean(per_context_probs, axis=0)

        stable_edges = [(j, target_idx) for j in sorted(consistent_present)]
        unstable_edges = [(j, target_idx) for j in sorted(consistent_absent)]

        return is_invariant, mean_probs, stable_edges, unstable_edges

    # ---- Step 2: Parametric Stability ----

    def _step2_parametric_stability(
        self,
        datasets: list[NDArray],
        target_idx: int,
        parent_set: list[int],
        K: int,
    ) -> tuple[NDArray, float]:
        """Step 2: Compute pairwise sqrt(JSD) for structurally invariant mechanism.

        Uses Gaussian closed-form for regression models.

        Returns
        -------
        (pairwise_jsd_matrix, max_sqrt_jsd)
        """
        # Fit regression per context
        models = []
        for k in range(K):
            data = datasets[k]
            y = data[:, target_idx]

            if len(parent_set) > 0:
                X = data[:, parent_set]
                coefs, intercept, res_var = self._fit_ols(X, y)
            else:
                coefs = np.array([])
                intercept = float(np.mean(y))
                res_var = float(np.var(y, ddof=1)) if len(y) > 1 else 1e-10

            models.append({
                "coefficients": coefs,
                "intercept": intercept,
                "residual_var": max(res_var, 1e-15),
            })

        # Pairwise sqrt(JSD)
        pairwise = np.zeros((K, K), dtype=np.float64)
        for a in range(K):
            for b in range(a + 1, K):
                ma, mb = models[a], models[b]
                if len(ma["coefficients"]) > 0 and len(mb["coefficients"]) > 0:
                    jsd = self._gaussian_regression_jsd(
                        ma["coefficients"], ma["intercept"], ma["residual_var"],
                        mb["coefficients"], mb["intercept"], mb["residual_var"],
                    )
                else:
                    jsd = self._gaussian_jsd(
                        ma["intercept"], ma["residual_var"],
                        mb["intercept"], mb["residual_var"],
                    )
                sqrt_jsd = math.sqrt(max(jsd, 0.0))
                pairwise[a, b] = sqrt_jsd
                pairwise[b, a] = sqrt_jsd

        max_sqrt_jsd = float(np.max(pairwise)) if K > 1 else 0.0
        return pairwise, max_sqrt_jsd

    # ---- Step 3: Bootstrap UCB ----

    def _step3_bootstrap_ucb(
        self,
        datasets: list[NDArray],
        target_idx: int,
        parent_set: list[int],
        K: int,
    ) -> tuple[float, NDArray]:
        """Step 3: Compute bootstrap upper confidence bound.

        Parametric bootstrap (B=1000):
        - Perturb regression coefficients by standard errors
        - Chi-squared resampling for residual variance
        - UCB = (1-β)-quantile of max sqrt(JSD)

        Returns
        -------
        (ucb, bootstrap_distribution)
        """
        rng = np.random.default_rng(self.config.random_state)
        B = self.config.n_bootstrap

        # Fit original models with standard errors
        models = []
        for k in range(K):
            data = datasets[k]
            y = data[:, target_idx]
            n_k = data.shape[0]

            if len(parent_set) > 0:
                X = data[:, parent_set]
                coefs, intercept, res_var = self._fit_ols(X, y)
                se_coefs = self._coefficient_se(X, y, res_var)
                p = len(parent_set)
            else:
                coefs = np.array([])
                intercept = float(np.mean(y))
                res_var = float(np.var(y, ddof=1)) if n_k > 1 else 1e-10
                se_coefs = np.array([])
                p = 0

            se_intercept = math.sqrt(max(res_var / max(n_k, 1), 1e-15))

            models.append({
                "coefficients": coefs,
                "intercept": intercept,
                "residual_var": max(res_var, 1e-15),
                "se_coefficients": se_coefs,
                "se_intercept": se_intercept,
                "n_samples": n_k,
                "n_params": p + 1,
            })

        # Bootstrap
        max_jsd_dist = np.zeros(B, dtype=np.float64)

        for b in range(B):
            # Perturb each model
            boot_models = []
            for k in range(K):
                m = models[k]
                boot_m = {}

                # Perturb coefficients
                if len(m["coefficients"]) > 0:
                    noise = rng.normal(0, m["se_coefficients"])
                    boot_m["coefficients"] = m["coefficients"] + noise
                else:
                    boot_m["coefficients"] = np.array([])

                # Perturb intercept
                boot_m["intercept"] = m["intercept"] + rng.normal(0, m["se_intercept"])

                # Chi-squared resampling for variance
                df = max(m["n_samples"] - m["n_params"], 1)
                chi2 = rng.chisquare(df)
                boot_m["residual_var"] = max(m["residual_var"] * chi2 / df, 1e-15)

                boot_models.append(boot_m)

            # Compute max pairwise sqrt(JSD)
            max_jsd = 0.0
            for a in range(K):
                for bb in range(a + 1, K):
                    ma, mb = boot_models[a], boot_models[bb]
                    if len(ma["coefficients"]) > 0 and len(mb["coefficients"]) > 0:
                        jsd = self._gaussian_regression_jsd(
                            ma["coefficients"], ma["intercept"], ma["residual_var"],
                            mb["coefficients"], mb["intercept"], mb["residual_var"],
                        )
                    else:
                        jsd = self._gaussian_jsd(
                            ma["intercept"], ma["residual_var"],
                            mb["intercept"], mb["residual_var"],
                        )
                    sqrt_jsd = math.sqrt(max(jsd, 0.0))
                    max_jsd = max(max_jsd, sqrt_jsd)

            max_jsd_dist[b] = max_jsd

        # UCB = (1-β)-quantile
        ucb = float(np.percentile(max_jsd_dist, 100 * (1 - self.config.beta)))
        return ucb, max_jsd_dist

    # ---- Step 4: Certificate Decision ----

    def _step4_decision(
        self,
        ucb: float,
    ) -> tuple[CertificateType, float]:
        """Step 4: Issue certificate based on UCB.

        STRONG_INVARIANCE if UCB < 0.01
        PARAMETRIC_STABILITY if UCB <= τ
        CANNOT_ISSUE if UCB > τ

        Returns (certificate_type, robustness_margin)
        """
        tau = self.config.tau
        strong_thresh = self.config.strong_invariance_threshold

        if ucb < strong_thresh:
            return CertificateType.STRONG_INVARIANCE, tau - ucb
        elif ucb <= tau:
            return CertificateType.PARAMETRIC_STABILITY, tau - ucb
        else:
            return CertificateType.CANNOT_ISSUE, tau - ucb

    # ---- Validation ----

    def _validate_inputs(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        target_idx: int,
        K: int,
    ) -> list[str]:
        """Validate inputs and return warning messages."""
        warn_msgs = []

        if K < 2:
            raise ValueError("At least 2 contexts required for certificate generation.")

        if len(adjacencies) != len(datasets):
            raise ValueError("Number of adjacencies must match number of datasets.")

        n_vars = adjacencies[0].shape[0]
        if not 0 <= target_idx < n_vars:
            raise ValueError(f"target_idx={target_idx} out of range [0, {n_vars}).")

        for k in range(K):
            n_k = datasets[k].shape[0]
            if n_k < self.config.min_samples_warning:
                warn_msgs.append(
                    f"Context {k}: n={n_k} < {self.config.min_samples_warning} "
                    f"(small sample warning)"
                )

        return warn_msgs

    # ---- Helper methods ----

    @staticmethod
    def _make_parent_selector(
        dag_learner: Callable[[NDArray], NDArray],
        target_idx: int,
    ) -> Callable[[NDArray, NDArray], NDArray]:
        """Create a parent selector function from a DAG learner."""
        def selector(X: NDArray, y: NDArray) -> NDArray:
            full_data = np.column_stack([X, y])
            # Reorder so target_idx is correct
            n_predictors = X.shape[1]
            # Build full data with target at correct position
            cols = list(range(n_predictors + 1))
            # Insert y at target_idx position
            reordered = np.zeros((full_data.shape[0], n_predictors + 1))
            pred_idx = 0
            for i in range(n_predictors + 1):
                if i == target_idx:
                    reordered[:, i] = y
                else:
                    if pred_idx < n_predictors:
                        reordered[:, i] = X[:, pred_idx]
                        pred_idx += 1
            try:
                adj = dag_learner(reordered)
                adj = np.asarray(adj, dtype=np.float64)
                # Extract parent indicators (column target_idx)
                parents = adj[:, target_idx]
                # Remove target's own entry
                result = np.delete(parents, target_idx)
                return result
            except Exception:
                return np.zeros(n_predictors)
        return selector

    @staticmethod
    def _consensus_parent_set(
        adjacencies: list[NDArray],
        target_idx: int,
        n_vars: int,
    ) -> list[int]:
        """Get consensus parent set from adjacencies.

        A variable is a consensus parent if it's a parent in the majority
        of contexts.
        """
        K = len(adjacencies)
        counts = np.zeros(n_vars, dtype=np.int64)
        for k in range(K):
            for j in range(n_vars):
                if adjacencies[k][j, target_idx] != 0:
                    counts[j] += 1
        return sorted([j for j in range(n_vars) if counts[j] > K / 2 and j != target_idx])

    @staticmethod
    def _fit_ols(X: NDArray, y: NDArray) -> tuple[NDArray, float, float]:
        """OLS regression with intercept."""
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

    @staticmethod
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

    @staticmethod
    def _gaussian_jsd(mu1: float, var1: float, mu2: float, var2: float) -> float:
        """JSD between two univariate Gaussians."""
        if var1 < 1e-15 or var2 < 1e-15:
            return 0.0
        kl_12 = 0.5 * (math.log(var2 / var1) + var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1)
        kl_21 = 0.5 * (math.log(var1 / var2) + var2 / var1 + (mu2 - mu1) ** 2 / var1 - 1)
        return max(0.5 * (kl_12 + kl_21), 0.0)

    @staticmethod
    def _gaussian_regression_jsd(
        coefs_a: NDArray, int_a: float, var_a: float,
        coefs_b: NDArray, int_b: float, var_b: float,
    ) -> float:
        """JSD between two conditional Gaussians."""
        if var_a < 1e-15 or var_b < 1e-15:
            return 0.0
        mean_diff = int_a - int_b
        coef_diff = np.asarray(coefs_a) - np.asarray(coefs_b)
        effective_sq = mean_diff ** 2 + float(np.sum(coef_diff ** 2))
        kl_ab = 0.5 * (math.log(var_b / var_a) + var_a / var_b + effective_sq / var_b - 1)
        kl_ba = 0.5 * (math.log(var_a / var_b) + var_b / var_a + effective_sq / var_a - 1)
        return max(0.5 * (kl_ab + kl_ba), 0.0)


# ---------------------------------------------------------------------------
# CertificateValidator
# ---------------------------------------------------------------------------

class CertificateValidator:
    """Validate certificate assumptions and cross-check with data.

    Checks that the assumptions underlying a certificate still hold,
    provides invalidation logic, and computes risk assessments.

    Parameters
    ----------
    normality_test_alpha : float
        Significance level for Gaussian assumption test (default 0.05).
    """

    def __init__(self, normality_test_alpha: float = 0.05):
        self.normality_test_alpha = normality_test_alpha

    def validate(
        self,
        certificate: RobustnessCertificate,
        datasets: list[NDArray],
        adjacencies: Optional[list[NDArray]] = None,
    ) -> dict:
        """Validate a certificate against empirical data.

        Parameters
        ----------
        certificate : the certificate to validate
        datasets : the data used to generate it (or new data)
        adjacencies : optional adjacencies for structural checks

        Returns
        -------
        dict with validation results
        """
        target_idx = certificate.variable_idx
        K = len(datasets)
        n_vars = datasets[0].shape[1]

        validation = {
            "certificate_type": certificate.certificate_type.value,
            "valid": True,
            "checks": {},
            "warnings": [],
        }

        # Check 1: Sample size adequacy
        sample_sizes = [d.shape[0] for d in datasets]
        min_n = min(sample_sizes)
        validation["checks"]["sample_size"] = {
            "min_n": min_n,
            "adequate": min_n >= 200,
        }
        if min_n < 200:
            validation["warnings"].append(
                f"Minimum sample size {min_n} < 200: certificate may be unreliable."
            )

        # Check 2: Gaussianity assumption
        gaussianity_ok = True
        for k in range(K):
            y = datasets[k][:, target_idx]
            if len(y) >= 20:
                # Shapiro-Wilk test (only for moderate n)
                try:
                    from scipy.stats import shapiro
                    stat, p_val = shapiro(y[:min(len(y), 5000)])
                    if p_val < self.normality_test_alpha:
                        gaussianity_ok = False
                        validation["warnings"].append(
                            f"Context {k}: Gaussianity rejected (p={p_val:.4f})"
                        )
                except ImportError:
                    pass

        validation["checks"]["gaussianity"] = {"ok": gaussianity_ok}

        # Check 3: Structural consistency
        if adjacencies is not None and certificate.structural_invariance:
            parent_sets = []
            for k in range(K):
                ps = sorted(
                    j for j in range(n_vars) if adjacencies[k][j, target_idx] != 0
                )
                parent_sets.append(ps)

            all_same = all(ps == parent_sets[0] for ps in parent_sets[1:])
            validation["checks"]["structural_consistency"] = {
                "all_same": all_same,
                "parent_sets": parent_sets,
            }
            if not all_same and certificate.structural_invariance:
                validation["valid"] = False
                validation["warnings"].append(
                    "Structural invariance claimed but parent sets differ "
                    "in the provided adjacencies."
                )

        # Check 4: UCB reproducibility
        if certificate.ucb < float("inf"):
            validation["checks"]["ucb"] = {
                "ucb": certificate.ucb,
                "tau": certificate.tau,
                "margin": certificate.robustness_margin,
                "margin_positive": certificate.robustness_margin > 0,
            }

        # Check 5: Multicollinearity
        if adjacencies is not None:
            parent_set = sorted(
                j for j in range(n_vars)
                if any(adjacencies[k][j, target_idx] != 0 for k in range(K))
                and j != target_idx
            )
            if len(parent_set) > 1:
                for k in range(K):
                    X = datasets[k][:, parent_set]
                    if X.shape[0] > X.shape[1]:
                        try:
                            corr = np.corrcoef(X.T)
                            max_corr = np.max(np.abs(corr - np.eye(len(parent_set))))
                            if max_corr > 0.9:
                                validation["warnings"].append(
                                    f"Context {k}: High multicollinearity "
                                    f"(max |r|={max_corr:.3f})"
                                )
                        except Exception:
                            pass

        # Overall validity
        if validation["warnings"]:
            # Warnings don't necessarily invalidate but flag concerns
            if not validation["checks"].get("structural_consistency", {}).get("all_same", True):
                validation["valid"] = False

        return validation

    def check_expiration(
        self,
        certificate: RobustnessCertificate,
        new_datasets: list[NDArray],
        new_adjacencies: list[NDArray],
    ) -> dict:
        """Check if a certificate should be invalidated given new data.

        Parameters
        ----------
        certificate : existing certificate
        new_datasets : new context data
        new_adjacencies : new context adjacencies

        Returns
        -------
        dict with expiration assessment
        """
        target_idx = certificate.variable_idx
        n_vars = new_adjacencies[0].shape[0]
        K_new = len(new_datasets)

        result = {
            "expired": False,
            "reasons": [],
        }

        if not certificate.is_certified:
            result["expired"] = True
            result["reasons"].append("Certificate was never issued.")
            return result

        # Check structural invariance in new data
        if certificate.structural_invariance:
            parent_sets = []
            for k in range(K_new):
                ps = sorted(
                    j for j in range(n_vars)
                    if new_adjacencies[k][j, target_idx] != 0
                )
                parent_sets.append(ps)

            if not all(ps == parent_sets[0] for ps in parent_sets[1:]):
                result["expired"] = True
                result["reasons"].append(
                    "Structural invariance no longer holds in new data."
                )

        # Check parametric stability in new data
        consensus_parents = certificate.stable_edges or []
        parent_set = sorted(set(e[0] for e in consensus_parents))

        if len(parent_set) > 0 and K_new >= 2:
            # Quick parametric check
            max_jsd = 0.0
            for a in range(K_new):
                for b in range(a + 1, K_new):
                    y_a = new_datasets[a][:, target_idx]
                    y_b = new_datasets[b][:, target_idx]
                    mu_a = float(np.mean(y_a))
                    mu_b = float(np.mean(y_b))
                    var_a = max(float(np.var(y_a, ddof=1)), 1e-15)
                    var_b = max(float(np.var(y_b, ddof=1)), 1e-15)
                    jsd = CertificateGenerator._gaussian_jsd(mu_a, var_a, mu_b, var_b)
                    max_jsd = max(max_jsd, math.sqrt(max(jsd, 0.0)))

            if max_jsd > certificate.tau:
                result["expired"] = True
                result["reasons"].append(
                    f"Parametric divergence ({max_jsd:.4f}) exceeds τ ({certificate.tau:.3f})."
                )

        return result

    def risk_assessment(
        self,
        certificate: RobustnessCertificate,
    ) -> dict:
        """Compute risk assessment for a certificate.

        Evaluates the risk that the certificate is incorrect based on
        available evidence.

        Returns dict with risk level ("low", "medium", "high") and factors.
        """
        risk_factors = []
        risk_score = 0.0

        # Factor 1: Robustness margin
        if certificate.robustness_margin < 0:
            risk_factors.append("Negative robustness margin")
            risk_score += 3.0
        elif certificate.robustness_margin < 0.1:
            risk_factors.append("Small robustness margin")
            risk_score += 1.5
        elif certificate.robustness_margin < 0.2:
            risk_factors.append("Moderate robustness margin")
            risk_score += 0.5

        # Factor 2: Sample size
        if certificate.n_samples_per_context:
            min_n = min(certificate.n_samples_per_context)
            if min_n < 50:
                risk_factors.append(f"Very small samples (min n={min_n})")
                risk_score += 2.0
            elif min_n < 200:
                risk_factors.append(f"Small samples (min n={min_n})")
                risk_score += 1.0

        # Factor 3: Number of contexts
        if certificate.n_contexts < 3:
            risk_factors.append("Few contexts")
            risk_score += 1.0

        # Factor 4: Warnings
        n_warnings = len(certificate.warnings_list)
        if n_warnings > 2:
            risk_factors.append(f"{n_warnings} warnings")
            risk_score += 0.5 * n_warnings

        # Determine risk level
        if risk_score >= 3.0:
            level = "high"
        elif risk_score >= 1.5:
            level = "medium"
        else:
            level = "low"

        return {
            "risk_level": level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
        }


# ---------------------------------------------------------------------------
# CertificateReport
# ---------------------------------------------------------------------------

class CertificateReport:
    """Generate human-readable certificate reports.

    Parameters
    ----------
    max_detail_variables : int
        Max variables to show in detail (default 20).
    """

    def __init__(self, max_detail_variables: int = 20):
        self.max_detail_variables = max_detail_variables

    def generate(
        self,
        batch_result: BatchCertificateResult,
        title: str = "Robustness Certificate Report",
    ) -> str:
        """Generate a full certificate report.

        Parameters
        ----------
        batch_result : BatchCertificateResult
        title : report title

        Returns
        -------
        str : formatted report text
        """
        lines = []
        lines.append("=" * 72)
        lines.append(title.center(72))
        lines.append("=" * 72)
        lines.append("")

        # Overview
        lines.extend(self._overview_section(batch_result))
        lines.append("")

        # Certificate breakdown
        lines.extend(self._breakdown_section(batch_result))
        lines.append("")

        # Variable details
        lines.extend(self._variable_details(batch_result))
        lines.append("")

        # Risk assessment
        lines.extend(self._risk_section(batch_result))
        lines.append("")

        lines.append("=" * 72)
        lines.append("End of Report".center(72))
        lines.append("=" * 72)

        return "\n".join(lines)

    def single_certificate_report(
        self,
        certificate: RobustnessCertificate,
    ) -> str:
        """Generate report for a single certificate."""
        lines = []
        name = certificate.variable_name or f"Variable {certificate.variable_idx}"
        lines.append(f"--- Certificate: {name} ---")
        lines.append(f"  Type:         {certificate.certificate_type.value}")
        lines.append(f"  UCB:          {certificate.ucb:.6f}")
        lines.append(f"  Threshold τ:  {certificate.tau:.4f}")
        lines.append(f"  Margin:       {certificate.robustness_margin:+.6f}")
        lines.append(f"  β (conf):     {certificate.beta:.3f}")
        lines.append(f"  Structural:   {'invariant' if certificate.structural_invariance else 'variable'}")
        lines.append(f"  Max √JSD:     {certificate.max_sqrt_jsd:.6f}")
        lines.append(f"  Contexts:     {certificate.n_contexts}")

        if certificate.n_samples_per_context:
            ns = certificate.n_samples_per_context
            lines.append(f"  Sample sizes: {ns} (min={min(ns)}, max={max(ns)})")

        if certificate.assumptions:
            lines.append(f"  Assumptions:")
            for a in certificate.assumptions:
                lines.append(f"    - {a}")

        if certificate.warnings_list:
            lines.append(f"  Warnings:")
            for w in certificate.warnings_list:
                lines.append(f"    ⚠ {w}")

        return "\n".join(lines)

    def risk_summary(
        self,
        batch_result: BatchCertificateResult,
    ) -> str:
        """Generate risk assessment summary."""
        validator = CertificateValidator()
        lines = ["RISK ASSESSMENT SUMMARY", "-" * 40]

        risk_levels = {"low": 0, "medium": 0, "high": 0}
        for cert in batch_result.certificates:
            ra = validator.risk_assessment(cert)
            risk_levels[ra["risk_level"]] += 1

        for level, count in risk_levels.items():
            pct = 100 * count / max(len(batch_result.certificates), 1)
            lines.append(f"  {level:8s}: {count:3d} ({pct:5.1f}%)")

        return "\n".join(lines)

    def _overview_section(self, result: BatchCertificateResult) -> list[str]:
        lines = ["OVERVIEW", "-" * 40]
        lines.append(f"Total variables:         {result.n_variables}")
        lines.append(f"Certified:               {result.n_certified} "
                     f"({100*result.n_certified/max(result.n_variables,1):.1f}%)")
        lines.append(f"  Strong invariance:     {result.n_strong}")
        lines.append(f"  Parametric stability:  {result.n_parametric}")
        lines.append(f"Cannot issue:            {result.n_failed}")
        lines.append(f"Computation time:        {result.computation_time:.2f}s")
        return lines

    def _breakdown_section(self, result: BatchCertificateResult) -> list[str]:
        lines = ["CERTIFICATE BREAKDOWN", "-" * 40]
        n = max(result.n_variables, 1)

        for ct in CertificateType:
            count = sum(1 for c in result.certificates if c.certificate_type == ct)
            pct = 100 * count / n
            bar = "█" * int(pct / 2)
            lines.append(f"  {ct.value:24s}: {count:3d} ({pct:5.1f}%) {bar}")

        return lines

    def _variable_details(self, result: BatchCertificateResult) -> list[str]:
        lines = ["VARIABLE DETAILS", "-" * 40]

        sorted_certs = sorted(
            result.certificates,
            key=lambda c: (-c.certificate_type.strength_level, -c.robustness_margin),
        )

        shown = min(len(sorted_certs), self.max_detail_variables)
        for cert in sorted_certs[:shown]:
            lines.append(self.single_certificate_report(cert))
            lines.append("")

        if len(sorted_certs) > shown:
            lines.append(f"  ... and {len(sorted_certs) - shown} more variables")

        return lines

    def _risk_section(self, result: BatchCertificateResult) -> list[str]:
        lines = [self.risk_summary(result)]
        return lines
