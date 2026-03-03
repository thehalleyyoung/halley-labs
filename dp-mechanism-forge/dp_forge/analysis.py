"""
Deep analysis module for synthesised DP mechanisms.

Provides comprehensive tools for evaluating the quality, optimality,
robustness, and statistical properties of mechanisms produced by the
DP-Forge CEGIS pipeline.

Analysis Capabilities:
    - **Utility analysis**: Monte Carlo MSE, MAE, variance, bias, tail
      probabilities, and error quantiles.
    - **Privacy curve analysis**: Full (ε, δ) privacy curves, Rényi DP
      curves, f-DP trade-off functions, and multi-mechanism comparison.
    - **Optimality analysis**: Gap to known information-theoretic lower
      bounds, comparison with Laplace/Staircase baselines, improvement
      ratios with confidence intervals.
    - **Robustness analysis**: Sensitivity of utility to privacy parameters,
      discretization resolution, and domain size.  Numerical stability
      diagnostics.
    - **Statistical analysis**: Bootstrap confidence intervals, permutation
      tests for mechanism comparison, power analysis, and sample size
      recommendations.
    - **Report generation**: LaTeX, Markdown, and JSON reports with tables,
      figures, and statistical summaries.

Classes:
    MechanismAnalyzer      — utility metrics (MSE, MAE, quantiles, etc.)
    PrivacyCurveAnalyzer   — (ε,δ), RDP, and f-DP curves
    OptimalityAnalyzer     — gap to lower bounds and baselines
    RobustnessAnalyzer     — sensitivity and stability checks
    StatisticalAnalyzer    — bootstrap CI, permutation tests, power
    ReportGenerator        — LaTeX / Markdown / JSON reports
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)
from dp_forge.types import (
    CompositionType,
    ExtractedMechanism,
    LossFunction,
    PrivacyBudget,
    QuerySpec,
    WorkloadSpec,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Analysis result containers
# ---------------------------------------------------------------------------


@dataclass
class UtilityMetrics:
    """Collection of utility metrics for a mechanism.

    Attributes:
        mse: Mean squared error (Monte Carlo estimate).
        mae: Mean absolute error (Monte Carlo estimate).
        variance: Variance of the noise distribution.
        bias: Bias of the mechanism (expected output − true value).
        max_error: Maximum observed absolute error.
        n_samples: Number of Monte Carlo samples used.
        quantiles: Dict mapping quantile levels to error values.
        tail_probabilities: Dict mapping thresholds to P(|error| > threshold).
    """

    mse: float
    mae: float
    variance: float
    bias: float
    max_error: float
    n_samples: int
    quantiles: Dict[float, float] = field(default_factory=dict)
    tail_probabilities: Dict[float, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"UtilityMetrics(mse={self.mse:.6f}, mae={self.mae:.6f}, "
            f"var={self.variance:.6f}, bias={self.bias:.6f}, n={self.n_samples})"
        )


@dataclass
class PrivacyCurve:
    """Privacy curve data.

    Attributes:
        epsilons: Array of epsilon values.
        deltas: Array of delta values.
        curve_type: Type of curve ('eps_delta', 'rdp', 'fdp').
        alphas: RDP orders (for RDP curves only).
        metadata: Additional curve metadata.
    """

    epsilons: FloatArray
    deltas: FloatArray
    curve_type: str = "eps_delta"
    alphas: Optional[FloatArray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PrivacyCurve(type={self.curve_type!r}, "
            f"n_points={len(self.epsilons)})"
        )


@dataclass
class OptimalityReport:
    """Report on mechanism optimality.

    Attributes:
        mechanism_mse: MSE of the analysed mechanism.
        lower_bound_mse: Known lower bound on MSE.
        baseline_mse: MSE of the baseline (Laplace/Gaussian).
        optimality_gap: Ratio mechanism_mse / lower_bound_mse.
        improvement_factor: Ratio baseline_mse / mechanism_mse.
        improvement_ci: 95% confidence interval for improvement_factor.
        metadata: Additional information.
    """

    mechanism_mse: float
    lower_bound_mse: float
    baseline_mse: float
    optimality_gap: float
    improvement_factor: float
    improvement_ci: Tuple[float, float] = (0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"OptimalityReport(gap={self.optimality_gap:.4f}, "
            f"improvement={self.improvement_factor:.4f}, "
            f"ci=({self.improvement_ci[0]:.4f}, {self.improvement_ci[1]:.4f}))"
        )


@dataclass
class RobustnessReport:
    """Report on mechanism robustness.

    Attributes:
        eps_sensitivity: How MSE changes per unit change in epsilon.
        k_sensitivity: How MSE changes per unit change in k.
        n_sensitivity: How MSE changes per unit change in n.
        condition_number: Condition number of the probability table.
        numerical_issues: List of detected numerical issues.
        metadata: Additional robustness data.
    """

    eps_sensitivity: float
    k_sensitivity: float
    n_sensitivity: float
    condition_number: float
    numerical_issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_stable(self) -> bool:
        """Whether no numerical issues were detected."""
        return len(self.numerical_issues) == 0

    def __repr__(self) -> str:
        status = "stable" if self.is_stable else f"{len(self.numerical_issues)} issues"
        return (
            f"RobustnessReport(eps_sens={self.eps_sensitivity:.4f}, "
            f"cond={self.condition_number:.2e}, {status})"
        )


@dataclass
class StatisticalTestResult:
    """Result of a statistical test.

    Attributes:
        test_name: Name of the test (e.g., 'permutation', 'bootstrap').
        statistic: Test statistic value.
        p_value: p-value of the test.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        significant: Whether the result is significant at the chosen alpha.
        n_samples: Number of samples or permutations used.
        metadata: Additional test metadata.
    """

    test_name: str
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"StatisticalTestResult({self.test_name!r}, stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, {sig})"
        )


@dataclass
class AnalysisReport:
    """Comprehensive analysis report combining all analyses.

    Attributes:
        utility: Utility metrics.
        privacy_curve: Privacy curve data.
        optimality: Optimality report.
        robustness: Robustness report.
        statistical_tests: List of statistical test results.
        generation_time: Time taken to generate the report, in seconds.
        metadata: Report-level metadata.
    """

    utility: Optional[UtilityMetrics] = None
    privacy_curve: Optional[PrivacyCurve] = None
    optimality: Optional[OptimalityReport] = None
    robustness: Optional[RobustnessReport] = None
    statistical_tests: List[StatisticalTestResult] = field(default_factory=list)
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = []
        if self.utility:
            parts.append(f"mse={self.utility.mse:.6f}")
        if self.optimality:
            parts.append(f"gap={self.optimality.optimality_gap:.4f}")
        if self.robustness:
            parts.append("stable" if self.robustness.is_stable else "unstable")
        return f"AnalysisReport({', '.join(parts)})"


# =========================================================================
# Internal sampling helpers
# =========================================================================


def _sample_mechanism(
    p_table: FloatArray,
    y_grid: FloatArray,
    input_idx: int,
    n_samples: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample n_samples outputs from mechanism row p_table[input_idx].

    Args:
        p_table: Probability table of shape (n, k).
        y_grid: Output grid of shape (k,).
        input_idx: Row index to sample from.
        n_samples: Number of samples.
        rng: Random number generator.

    Returns:
        Array of sampled output values.
    """
    probs = p_table[input_idx]
    probs = np.maximum(probs, 0.0)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        raise InvalidMechanismError(
            f"Row {input_idx} of probability table sums to {prob_sum}",
            reason="zero probability row",
        )
    probs = probs / prob_sum
    indices = rng.choice(len(y_grid), size=n_samples, p=probs)
    return y_grid[indices]


def _build_output_grid(spec: QuerySpec) -> FloatArray:
    """Build an output grid from a QuerySpec.

    Creates a uniform grid spanning the range of query values with
    some padding, using k bins.

    Args:
        spec: Query specification.

    Returns:
        Output grid array of shape (k,).
    """
    q_min = float(np.min(spec.query_values))
    q_max = float(np.max(spec.query_values))
    q_range = q_max - q_min
    if q_range == 0:
        q_range = 1.0
    padding = q_range * 0.5
    return np.linspace(q_min - padding, q_max + padding, spec.k)


# =========================================================================
# 1. MechanismAnalyzer
# =========================================================================


class MechanismAnalyzer:
    """Utility analysis for synthesised DP mechanisms.

    Computes MSE, MAE, variance, bias, tail probabilities, and error
    quantiles via Monte Carlo sampling.  All metrics are averaged over
    the input distribution (uniform over databases by default).

    Usage::

        analyzer = MechanismAnalyzer(seed=42)
        metrics = analyzer.compute_mse(mechanism, spec, n_samples=100_000)
        full = analyzer.analyze_comprehensive(mechanism, spec)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize analyser with a random seed.

        Args:
            seed: Random seed for reproducibility.
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def analyze_comprehensive(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        n_samples: int = 100_000,
    ) -> UtilityMetrics:
        """Run a comprehensive utility analysis.

        Computes all utility metrics: MSE, MAE, variance, bias, max error,
        error quantiles at [0.5, 0.9, 0.95, 0.99], and tail probabilities.

        Args:
            mechanism: The mechanism to analyse.
            spec: Query specification used to synthesise the mechanism.
            n_samples: Number of Monte Carlo samples per database.

        Returns:
            UtilityMetrics with all computed values.
        """
        y_grid = _build_output_grid(spec)
        p_table = mechanism.p_final
        n = min(p_table.shape[0], spec.n)

        all_errors: list[float] = []
        all_squared_errors: list[float] = []
        all_abs_errors: list[float] = []
        all_outputs: list[float] = []
        true_values_list: list[float] = []

        for i in range(n):
            true_val = spec.query_values[i]
            samples = _sample_mechanism(p_table, y_grid, i, n_samples, self._rng)
            errors = samples - true_val
            all_errors.extend(errors.tolist())
            all_squared_errors.extend((errors ** 2).tolist())
            all_abs_errors.extend(np.abs(errors).tolist())
            all_outputs.extend(samples.tolist())
            true_values_list.extend([true_val] * n_samples)

        errors_arr = np.array(all_errors, dtype=np.float64)
        sq_errors_arr = np.array(all_squared_errors, dtype=np.float64)
        abs_errors_arr = np.array(all_abs_errors, dtype=np.float64)

        mse = float(np.mean(sq_errors_arr))
        mae = float(np.mean(abs_errors_arr))
        variance = float(np.var(errors_arr))
        bias = float(np.mean(errors_arr))
        max_error = float(np.max(abs_errors_arr))

        # Quantiles
        quantile_levels = [0.5, 0.9, 0.95, 0.99]
        quantiles = {
            q: float(np.quantile(abs_errors_arr, q))
            for q in quantile_levels
        }

        # Tail probabilities
        thresholds = [1.0, 2.0, 5.0, 10.0]
        tail_probs = {
            t: float(np.mean(abs_errors_arr > t))
            for t in thresholds
        }

        return UtilityMetrics(
            mse=mse,
            mae=mae,
            variance=variance,
            bias=bias,
            max_error=max_error,
            n_samples=n_samples * n,
            quantiles=quantiles,
            tail_probabilities=tail_probs,
        )

    def compute_mse(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        n_samples: int = 100_000,
    ) -> float:
        """Compute Monte Carlo estimate of mean squared error.

        MSE = (1/n) Σ_i E[(M(x_i) - f(x_i))²]

        where the expectation is over the mechanism's randomness.

        Args:
            mechanism: The mechanism to evaluate.
            spec: Query specification.
            n_samples: Number of samples per database.

        Returns:
            Estimated MSE.
        """
        y_grid = _build_output_grid(spec)
        p_table = mechanism.p_final
        n = min(p_table.shape[0], spec.n)

        total_mse = 0.0
        for i in range(n):
            true_val = spec.query_values[i]
            samples = _sample_mechanism(p_table, y_grid, i, n_samples, self._rng)
            total_mse += float(np.mean((samples - true_val) ** 2))

        return total_mse / n

    def compute_mae(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        n_samples: int = 100_000,
    ) -> float:
        """Compute Monte Carlo estimate of mean absolute error.

        MAE = (1/n) Σ_i E[|M(x_i) - f(x_i)|]

        Args:
            mechanism: The mechanism to evaluate.
            spec: Query specification.
            n_samples: Number of samples per database.

        Returns:
            Estimated MAE.
        """
        y_grid = _build_output_grid(spec)
        p_table = mechanism.p_final
        n = min(p_table.shape[0], spec.n)

        total_mae = 0.0
        for i in range(n):
            true_val = spec.query_values[i]
            samples = _sample_mechanism(p_table, y_grid, i, n_samples, self._rng)
            total_mae += float(np.mean(np.abs(samples - true_val)))

        return total_mae / n

    def compute_variance(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        n_samples: int = 100_000,
    ) -> FloatArray:
        """Compute per-input variance of the mechanism.

        For each database x_i, computes Var[M(x_i)] = E[M(x_i)²] − E[M(x_i)]².

        Args:
            mechanism: The mechanism to evaluate.
            spec: Query specification.
            n_samples: Number of samples per database.

        Returns:
            Array of per-input variances, shape (n,).
        """
        y_grid = _build_output_grid(spec)
        p_table = mechanism.p_final
        n = min(p_table.shape[0], spec.n)

        variances = np.empty(n, dtype=np.float64)
        for i in range(n):
            samples = _sample_mechanism(p_table, y_grid, i, n_samples, self._rng)
            variances[i] = float(np.var(samples))

        return variances

    def compute_bias(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        n_samples: int = 100_000,
    ) -> FloatArray:
        """Compute per-input bias of the mechanism.

        Bias_i = E[M(x_i)] − f(x_i).

        An unbiased mechanism has Bias_i = 0 for all i.

        Args:
            mechanism: The mechanism to evaluate.
            spec: Query specification.
            n_samples: Number of samples per database.

        Returns:
            Array of per-input biases, shape (n,).
        """
        y_grid = _build_output_grid(spec)
        p_table = mechanism.p_final
        n = min(p_table.shape[0], spec.n)

        biases = np.empty(n, dtype=np.float64)
        for i in range(n):
            true_val = spec.query_values[i]
            samples = _sample_mechanism(p_table, y_grid, i, n_samples, self._rng)
            biases[i] = float(np.mean(samples) - true_val)

        return biases

    def tail_probability(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        threshold: float,
        n_samples: int = 100_000,
    ) -> float:
        """Compute P(|error| > threshold) averaged over inputs.

        Tail probabilities measure how likely the mechanism is to produce
        a large error.  Lower values indicate a mechanism whose noise is
        more concentrated.

        Args:
            mechanism: The mechanism to evaluate.
            spec: Query specification.
            threshold: Error threshold.
            n_samples: Number of samples per database.

        Returns:
            Average tail probability.
        """
        y_grid = _build_output_grid(spec)
        p_table = mechanism.p_final
        n = min(p_table.shape[0], spec.n)

        total_tail = 0.0
        for i in range(n):
            true_val = spec.query_values[i]
            samples = _sample_mechanism(p_table, y_grid, i, n_samples, self._rng)
            total_tail += float(np.mean(np.abs(samples - true_val) > threshold))

        return total_tail / n

    def quantile_analysis(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        quantiles: Optional[List[float]] = None,
        n_samples: int = 100_000,
    ) -> Dict[float, float]:
        """Compute error quantiles averaged over inputs.

        For each quantile level q, computes the q-th quantile of |M(x) − f(x)|
        pooled across all inputs.

        Args:
            mechanism: The mechanism to evaluate.
            spec: Query specification.
            quantiles: List of quantile levels in [0, 1].
                Defaults to [0.5, 0.75, 0.9, 0.95, 0.99].
            n_samples: Number of samples per database.

        Returns:
            Dict mapping quantile levels to error values.
        """
        if quantiles is None:
            quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]

        y_grid = _build_output_grid(spec)
        p_table = mechanism.p_final
        n = min(p_table.shape[0], spec.n)

        all_abs_errors = []
        for i in range(n):
            true_val = spec.query_values[i]
            samples = _sample_mechanism(p_table, y_grid, i, n_samples, self._rng)
            all_abs_errors.extend(np.abs(samples - true_val).tolist())

        abs_errors = np.array(all_abs_errors, dtype=np.float64)
        return {q: float(np.quantile(abs_errors, q)) for q in quantiles}


# =========================================================================
# 2. PrivacyCurveAnalyzer
# =========================================================================


class PrivacyCurveAnalyzer:
    """Analyse and compare privacy curves of mechanisms.

    Computes the full (ε, δ) privacy curve, Rényi DP curve, and
    f-DP trade-off function for mechanisms.  Supports multi-mechanism
    comparison for evaluating synthesised vs. baseline mechanisms.

    Usage::

        analyzer = PrivacyCurveAnalyzer()
        curve = analyzer.compute_privacy_curve(mechanism, edges)
        rdp = analyzer.compute_rdp_curve(mechanism, edges, alphas)
    """

    def compute_privacy_curve(
        self,
        mechanism: ExtractedMechanism,
        edges: Optional[List[Tuple[int, int]]] = None,
        n_points: int = 100,
    ) -> PrivacyCurve:
        """Compute the full (ε, δ) privacy curve.

        For each candidate ε, computes the minimum δ such that the
        mechanism satisfies (ε, δ)-DP.  This is determined by the
        privacy loss distribution over all adjacent pairs.

        The privacy curve δ(ε) is defined as:
            δ(ε) = max_{(i,i')} max_j [p[i,j] − e^ε · p[i',j]]

        summed over output bins where the quantity is positive.

        Args:
            mechanism: The mechanism to analyse.
            edges: List of adjacent database pairs. If None, uses
                consecutive pairs (0,1), (1,2), ...
            n_points: Number of epsilon values to evaluate.

        Returns:
            PrivacyCurve with epsilon and delta arrays.
        """
        p_table = mechanism.p_final
        n, k = p_table.shape

        if edges is None:
            edges = [(i, i + 1) for i in range(n - 1)]

        # Sweep epsilon from 0 to a reasonable maximum
        max_eps = float(np.max(np.log(
            np.maximum(p_table, 1e-300) / np.maximum(np.min(p_table[p_table > 0]), 1e-300)
        )))
        max_eps = max(max_eps, 5.0)

        eps_values = np.linspace(0.0, max_eps, n_points)
        delta_values = np.empty(n_points, dtype=np.float64)

        for idx, eps in enumerate(eps_values):
            max_delta = 0.0
            for i, ip in edges:
                if i >= n or ip >= n:
                    continue
                # δ(ε) for this pair: Σ_j max(0, p[i,j] - e^ε · p[i',j])
                diff = p_table[i] - math.exp(eps) * p_table[ip]
                delta_pair = float(np.sum(np.maximum(diff, 0.0)))
                # Also check the reverse direction
                diff_rev = p_table[ip] - math.exp(eps) * p_table[i]
                delta_pair_rev = float(np.sum(np.maximum(diff_rev, 0.0)))
                max_delta = max(max_delta, delta_pair, delta_pair_rev)

            delta_values[idx] = max_delta

        return PrivacyCurve(
            epsilons=eps_values,
            deltas=delta_values,
            curve_type="eps_delta",
        )

    def compute_rdp_curve(
        self,
        mechanism: ExtractedMechanism,
        edges: Optional[List[Tuple[int, int]]] = None,
        alphas: Optional[FloatArray] = None,
    ) -> PrivacyCurve:
        """Compute the Rényi DP curve.

        For each order α > 1, computes the RDP guarantee ε(α) defined as:
            ε(α) = max_{(i,i')} (1/(α−1)) · log E_{j~p[i']} [(p[i,j]/p[i',j])^α]

        This is the Rényi divergence D_α(p[i] || p[i']) maximised over
        adjacent pairs.

        Args:
            mechanism: The mechanism to analyse.
            edges: Adjacent pairs. Defaults to consecutive.
            alphas: RDP orders to evaluate. Defaults to standard grid.

        Returns:
            PrivacyCurve with RDP values (epsilons are RDP epsilons, deltas
            unused, alphas stored in the curve).
        """
        p_table = mechanism.p_final
        n, k = p_table.shape

        if edges is None:
            edges = [(i, i + 1) for i in range(n - 1)]

        if alphas is None:
            alphas = np.concatenate([
                np.arange(1.5, 10.0, 0.5),
                np.arange(10.0, 50.0, 5.0),
                np.array([64.0, 128.0, 256.0]),
            ])

        rdp_values = np.empty(len(alphas), dtype=np.float64)

        for a_idx, alpha in enumerate(alphas):
            max_rdp = 0.0
            for i, ip in edges:
                if i >= n or ip >= n:
                    continue
                # D_alpha(p[i] || p[i'])
                pi = np.maximum(p_table[i], 1e-300)
                pip = np.maximum(p_table[ip], 1e-300)

                # (1/(α-1)) · log(Σ_j p[i',j] · (p[i,j]/p[i',j])^α)
                # = (1/(α-1)) · log(Σ_j p[i,j]^α · p[i',j]^{1-α})
                log_terms = alpha * np.log(pi) + (1.0 - alpha) * np.log(pip)
                log_rdp = float(np.max(log_terms)) + np.log(
                    np.sum(np.exp(log_terms - np.max(log_terms)))
                )
                rdp = float(log_rdp) / (alpha - 1.0)

                # Also check reverse direction
                log_terms_rev = alpha * np.log(pip) + (1.0 - alpha) * np.log(pi)
                log_rdp_rev = float(np.max(log_terms_rev)) + np.log(
                    np.sum(np.exp(log_terms_rev - np.max(log_terms_rev)))
                )
                rdp_rev = float(log_rdp_rev) / (alpha - 1.0)

                max_rdp = max(max_rdp, rdp, rdp_rev)

            rdp_values[a_idx] = max_rdp

        return PrivacyCurve(
            epsilons=rdp_values,
            deltas=np.zeros_like(rdp_values),
            curve_type="rdp",
            alphas=alphas,
        )

    def compute_fdp_curve(
        self,
        mechanism: ExtractedMechanism,
        edges: Optional[List[Tuple[int, int]]] = None,
        n_points: int = 100,
    ) -> PrivacyCurve:
        """Compute the f-DP trade-off function.

        f-DP (Dong, Roth, Su 2019) characterises a mechanism by its
        trade-off function f(α) = inf{β : ∃ test with type-I ≤ α, type-II ≤ β}
        where α and β are type-I and type-II error probabilities for
        distinguishing adjacent databases.

        The trade-off function is computed from the privacy loss distribution:
            f(α) = max_{(i,i')} min_S {Pr_{p[i']}[S] : Pr_{p[i]}[S] ≥ 1 − α}

        which is equivalently the Neyman-Pearson curve.

        Args:
            mechanism: The mechanism to analyse.
            edges: Adjacent pairs.
            n_points: Number of points on the f-DP curve.

        Returns:
            PrivacyCurve with alpha values in epsilons and f(alpha) in deltas.
        """
        p_table = mechanism.p_final
        n, k = p_table.shape

        if edges is None:
            edges = [(i, i + 1) for i in range(n - 1)]

        alpha_values = np.linspace(0.0, 1.0, n_points)
        f_values = np.ones(n_points, dtype=np.float64)

        for i, ip in edges:
            if i >= n or ip >= n:
                continue

            pi = p_table[i]
            pip = p_table[ip]

            # Likelihood ratio
            safe_pip = np.maximum(pip, 1e-300)
            lr = pi / safe_pip

            # Sort by likelihood ratio (descending) for Neyman-Pearson
            sorted_idx = np.argsort(-lr)
            pi_sorted = pi[sorted_idx]
            pip_sorted = pip[sorted_idx]

            # Build the ROC curve: as we include more bins (high LR first),
            # type-I error decreases and type-II error increases.
            cum_pi = np.cumsum(pi_sorted)
            cum_pip = np.cumsum(pip_sorted)

            for a_idx, alpha in enumerate(alpha_values):
                # Find threshold such that Pr_{p[i]}[reject] >= 1 - alpha
                # i.e., include enough bins that their total probability under
                # p[i] is >= 1 - alpha
                target_power = 1.0 - alpha
                if target_power <= 0:
                    f_val = 1.0
                elif target_power >= 1.0:
                    f_val = 0.0
                else:
                    # Find the cutoff
                    idx_cutoff = np.searchsorted(cum_pi, target_power, side="left")
                    idx_cutoff = min(idx_cutoff, k - 1)
                    # Type-II error = Pr_{p[i']}[accept] = 1 - Pr_{p[i']}[reject]
                    f_val = 1.0 - cum_pip[idx_cutoff]
                    f_val = max(f_val, 0.0)

                f_values[a_idx] = min(f_values[a_idx], f_val)

        return PrivacyCurve(
            epsilons=alpha_values,
            deltas=f_values,
            curve_type="fdp",
            metadata={"x_label": "type_I_error", "y_label": "type_II_error"},
        )

    def compare_curves(
        self,
        mechanisms: List[ExtractedMechanism],
        labels: Optional[List[str]] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
        n_points: int = 100,
    ) -> List[PrivacyCurve]:
        """Compare privacy curves of multiple mechanisms.

        Computes the (ε, δ) privacy curve for each mechanism and returns
        them in a list for side-by-side comparison.

        Args:
            mechanisms: List of mechanisms to compare.
            labels: Optional labels for each mechanism.
            edges: Adjacent pairs (shared across all mechanisms).
            n_points: Number of points per curve.

        Returns:
            List of PrivacyCurves, one per mechanism.
        """
        if labels is None:
            labels = [f"mechanism_{i}" for i in range(len(mechanisms))]

        curves = []
        for mech, label in zip(mechanisms, labels):
            curve = self.compute_privacy_curve(mech, edges, n_points)
            curve.metadata["label"] = label
            curves.append(curve)

        return curves


# =========================================================================
# 3. OptimalityAnalyzer
# =========================================================================


class OptimalityAnalyzer:
    """Analyse how close a mechanism is to information-theoretic optimality.

    Compares synthesised mechanisms against known lower bounds and
    baseline mechanisms (Laplace, Staircase, Gaussian) to quantify
    the improvement achieved by the CEGIS synthesis.

    Usage::

        analyzer = OptimalityAnalyzer(seed=42)
        report = analyzer.optimality_gap(mechanism, spec, lower_bound=0.5)
        factor = analyzer.improvement_factor(mechanism, baseline, spec)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize with a random seed for Monte Carlo estimates.

        Args:
            seed: Random seed.
        """
        self._analyzer = MechanismAnalyzer(seed=seed)
        self._seed = seed

    def optimality_gap(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        lower_bound: Optional[float] = None,
        n_samples: int = 100_000,
    ) -> OptimalityReport:
        """Compute the optimality gap relative to a known lower bound.

        The gap is defined as mechanism_mse / lower_bound_mse.  A value of
        1.0 means the mechanism achieves the lower bound (is optimal).

        Args:
            mechanism: The mechanism to evaluate.
            spec: Query specification.
            lower_bound: Known MSE lower bound. If None, uses the
                theoretical_lower_bound method.
            n_samples: Monte Carlo samples for MSE estimation.

        Returns:
            OptimalityReport with gap and related metrics.
        """
        mse = self._analyzer.compute_mse(mechanism, spec, n_samples)

        if lower_bound is None:
            lower_bound = self.theoretical_lower_bound(spec)

        baseline_mse = self.compare_to_optimal_laplace(mechanism, spec)

        gap = mse / max(lower_bound, 1e-15)
        improvement = baseline_mse / max(mse, 1e-15)

        return OptimalityReport(
            mechanism_mse=mse,
            lower_bound_mse=lower_bound,
            baseline_mse=baseline_mse,
            optimality_gap=gap,
            improvement_factor=improvement,
            metadata={"n_samples": n_samples},
        )

    def compare_to_optimal_laplace(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
    ) -> float:
        """Compute the MSE of the optimal Laplace mechanism for comparison.

        The Laplace mechanism with scale b = Δf/ε has MSE = 2b² = 2(Δf/ε)².

        Args:
            mechanism: The synthesised mechanism (used for epsilon extraction).
            spec: Query specification.

        Returns:
            Laplace mechanism MSE.
        """
        b = spec.sensitivity / spec.epsilon
        return 2.0 * b ** 2

    def compare_to_optimal_staircase(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
    ) -> float:
        """Compute the MSE of the optimal Staircase mechanism for comparison.

        The Staircase mechanism (Geng & Viswanath 2014) is optimal for
        counting queries under L1 loss.  For L2 loss, its MSE can be
        computed numerically.

        For a counting query with sensitivity Δ and privacy ε, the Staircase
        mechanism's MSE is approximately:

            MSE ≈ 2Δ²/ε² × (1 − (1−e^{−ε})/(2ε))

        For small ε, this is close to the Laplace MSE.

        Args:
            mechanism: The synthesised mechanism (for epsilon extraction).
            spec: Query specification.

        Returns:
            Approximate Staircase mechanism MSE.
        """
        eps = spec.epsilon
        delta_f = spec.sensitivity

        # Staircase optimal parameter
        if eps >= 30:
            # For large ε, Staircase ≈ Laplace
            return 2.0 * (delta_f / eps) ** 2

        # Geng & Viswanath (2014) formula for optimal γ
        gamma = 1.0 / (1.0 + math.exp(eps / 2.0))

        # MSE of the Staircase mechanism with optimal γ
        b = delta_f / eps
        # The Staircase MSE is always ≤ Laplace MSE
        exp_neg_eps = math.exp(-eps)
        staircase_mse = 2.0 * b ** 2 * (
            1.0 - (1.0 - exp_neg_eps) / (2.0 * eps)
        )

        return max(staircase_mse, 0.0)

    def improvement_factor(
        self,
        mechanism: ExtractedMechanism,
        baseline: ExtractedMechanism,
        spec: QuerySpec,
        n_samples: int = 100_000,
        n_bootstrap: int = 1000,
    ) -> StatisticalTestResult:
        """Compute the improvement factor with bootstrap confidence interval.

        Improvement = MSE(baseline) / MSE(mechanism).
        Values > 1 indicate the synthesised mechanism is better.

        Args:
            mechanism: The synthesised mechanism.
            baseline: The baseline mechanism.
            spec: Query specification.
            n_samples: Monte Carlo samples for MSE.
            n_bootstrap: Bootstrap resamples for CI.

        Returns:
            StatisticalTestResult with improvement factor and CI.
        """
        rng = np.random.default_rng(self._seed)
        y_grid = _build_output_grid(spec)
        n = min(mechanism.p_final.shape[0], baseline.p_final.shape[0], spec.n)

        # Collect paired MSE samples for bootstrap
        mech_errors = []
        base_errors = []

        samples_per = max(n_samples // n, 100)

        for i in range(n):
            true_val = spec.query_values[i]
            m_samp = _sample_mechanism(mechanism.p_final, y_grid, i, samples_per, rng)
            b_samp = _sample_mechanism(baseline.p_final, y_grid, i, samples_per, rng)
            mech_errors.extend(((m_samp - true_val) ** 2).tolist())
            base_errors.extend(((b_samp - true_val) ** 2).tolist())

        mech_errors_arr = np.array(mech_errors, dtype=np.float64)
        base_errors_arr = np.array(base_errors, dtype=np.float64)

        mech_mse = float(np.mean(mech_errors_arr))
        base_mse = float(np.mean(base_errors_arr))
        improvement = base_mse / max(mech_mse, 1e-15)

        # Bootstrap CI
        boot_improvements = []
        total = len(mech_errors_arr)
        for _ in range(n_bootstrap):
            idx = rng.choice(total, size=total, replace=True)
            m_boot = float(np.mean(mech_errors_arr[idx]))
            b_boot = float(np.mean(base_errors_arr[idx]))
            boot_improvements.append(b_boot / max(m_boot, 1e-15))

        boot_arr = np.array(boot_improvements, dtype=np.float64)
        ci_lower = float(np.percentile(boot_arr, 2.5))
        ci_upper = float(np.percentile(boot_arr, 97.5))

        return StatisticalTestResult(
            test_name="improvement_factor",
            statistic=improvement,
            p_value=0.0,  # Not a hypothesis test
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=ci_lower > 1.0,  # Significant if CI excludes 1.0
            n_samples=total,
            metadata={
                "mechanism_mse": mech_mse,
                "baseline_mse": base_mse,
                "n_bootstrap": n_bootstrap,
            },
        )

    def theoretical_lower_bound(self, spec: QuerySpec) -> float:
        """Compute known information-theoretic MSE lower bounds.

        For counting queries with sensitivity Δ under ε-DP:
            MSE ≥ Δ² / (2ε²)    (for small ε)

        This is the Hardt-Talwar (2010) lower bound for linear queries.

        For (ε, δ)-DP with δ > 0:
            MSE ≥ Δ² · min(1/(2ε²), σ²_gaussian)
            where σ_gaussian = Δ · √(2 ln(1.25/δ)) / ε

        Args:
            spec: Query specification.

        Returns:
            Lower bound on MSE.
        """
        delta_f = spec.sensitivity
        eps = spec.epsilon

        if spec.delta == 0:
            # Pure DP lower bound (based on Staircase optimality for counting)
            return delta_f ** 2 / (2.0 * eps ** 2)
        else:
            # Approximate DP: the lower bound is determined by the Gaussian
            # mechanism's optimal MSE (which is near-optimal for Gaussian)
            sigma = delta_f * math.sqrt(2.0 * math.log(1.25 / spec.delta)) / eps
            gaussian_mse = sigma ** 2

            pure_lb = delta_f ** 2 / (2.0 * eps ** 2)
            return min(pure_lb, gaussian_mse)


# =========================================================================
# 4. RobustnessAnalyzer
# =========================================================================


class RobustnessAnalyzer:
    """Analyse robustness and numerical stability of mechanisms.

    Measures how sensitive the mechanism's utility is to changes in
    privacy parameters, discretization resolution, and domain size.
    Also performs numerical stability checks on the probability table.

    Usage::

        analyzer = RobustnessAnalyzer(seed=42)
        report = analyzer.sensitivity_to_epsilon(mechanism, spec)
        stability = analyzer.numerical_stability_check(mechanism)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize with a random seed.

        Args:
            seed: Random seed for Monte Carlo estimates.
        """
        self._analyzer = MechanismAnalyzer(seed=seed)
        self._seed = seed

    def sensitivity_to_epsilon(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        eps_range: Optional[FloatArray] = None,
        n_samples: int = 50_000,
    ) -> Dict[str, Any]:
        """Analyse how MSE changes with epsilon.

        Computes the Laplace MSE = 2(Δf/ε)² at several epsilon values
        for comparison, and estimates the derivative dMSE/dε at the
        mechanism's operating point.

        Args:
            mechanism: The mechanism to analyse.
            spec: Query specification.
            eps_range: Epsilon values to evaluate. Defaults to a range
                around spec.epsilon.
            n_samples: Monte Carlo samples.

        Returns:
            Dict with 'eps_values', 'laplace_mse', 'mech_mse', 'sensitivity'.
        """
        if eps_range is None:
            center = spec.epsilon
            eps_range = np.linspace(
                max(center * 0.5, 0.01), center * 2.0, 20
            )

        laplace_mse = 2.0 * (spec.sensitivity / eps_range) ** 2

        # Mechanism MSE at the synthesis epsilon
        mech_mse = self._analyzer.compute_mse(mechanism, spec, n_samples)

        # Numerical sensitivity: dMSE/dε ≈ -4Δ²/ε³ for Laplace
        sensitivity = -4.0 * spec.sensitivity ** 2 / spec.epsilon ** 3

        return {
            "eps_values": eps_range,
            "laplace_mse": laplace_mse,
            "mechanism_mse": mech_mse,
            "sensitivity_dMSE_deps": sensitivity,
        }

    def sensitivity_to_k(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        k_range: Optional[List[int]] = None,
        n_samples: int = 50_000,
    ) -> Dict[str, Any]:
        """Analyse how MSE depends on discretization parameter k.

        Since the mechanism was synthesised with a specific k, this method
        evaluates the Laplace baseline at different k values to show how
        discretization affects utility.  The mechanism's own MSE is fixed.

        Args:
            mechanism: The mechanism to analyse.
            spec: Query specification.
            k_range: Values of k to evaluate.
            n_samples: Monte Carlo samples.

        Returns:
            Dict with 'k_values', 'discretization_error', 'mechanism_mse'.
        """
        if k_range is None:
            k_range = [10, 20, 50, 100, 200, 500]

        mech_mse = self._analyzer.compute_mse(mechanism, spec, n_samples)

        # Discretization error ~ (range / k)² for uniform grids
        q_range = float(np.max(spec.query_values) - np.min(spec.query_values))
        if q_range == 0:
            q_range = 1.0

        disc_errors = [(q_range / k) ** 2 for k in k_range]

        return {
            "k_values": k_range,
            "discretization_error": disc_errors,
            "mechanism_mse": mech_mse,
            "mechanism_k": spec.k,
        }

    def sensitivity_to_n(
        self,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        n_range: Optional[List[int]] = None,
        n_samples: int = 50_000,
    ) -> Dict[str, Any]:
        """Analyse how MSE depends on domain size n.

        Evaluates how the mechanism's MSE changes with the number of
        possible database inputs.  The mechanism's own MSE is fixed,
        but the Laplace baseline may vary with n.

        Args:
            mechanism: The mechanism to analyse.
            spec: Query specification.
            n_range: Domain sizes to evaluate.
            n_samples: Monte Carlo samples.

        Returns:
            Dict with 'n_values', 'laplace_mse', 'mechanism_mse'.
        """
        if n_range is None:
            n_range = [2, 5, 10, 20, 50, 100]

        mech_mse = self._analyzer.compute_mse(mechanism, spec, n_samples)

        # For Laplace, MSE doesn't depend on n (only on sensitivity and ε)
        laplace_mse = 2.0 * (spec.sensitivity / spec.epsilon) ** 2

        return {
            "n_values": n_range,
            "laplace_mse": [laplace_mse] * len(n_range),
            "mechanism_mse": mech_mse,
            "mechanism_n": spec.n,
        }

    def numerical_stability_check(
        self,
        mechanism: ExtractedMechanism,
        tol: float = 1e-10,
    ) -> RobustnessReport:
        """Check the mechanism's probability table for numerical issues.

        Checks for:
        - Negative probabilities.
        - Rows not summing to 1.
        - Very small but non-zero probabilities (underflow risk).
        - Very large probability ratios (overflow risk in log-space).
        - Poor conditioning of the probability table.

        Args:
            mechanism: The mechanism to check.
            tol: Tolerance for numerical comparisons.

        Returns:
            RobustnessReport with detected issues.
        """
        p_table = mechanism.p_final
        issues: List[str] = []

        # Check negative probabilities
        min_val = float(np.min(p_table))
        if min_val < -tol:
            issues.append(f"Negative probabilities detected (min={min_val:.2e})")

        # Check row sums
        row_sums = p_table.sum(axis=1)
        max_deviation = float(np.max(np.abs(row_sums - 1.0)))
        if max_deviation > tol:
            issues.append(
                f"Row sums deviate from 1 (max deviation={max_deviation:.2e})"
            )

        # Check for underflow-prone values
        positive_vals = p_table[p_table > 0]
        if len(positive_vals) > 0:
            min_positive = float(np.min(positive_vals))
            if min_positive < 1e-300:
                issues.append(
                    f"Extremely small probabilities (min={min_positive:.2e})"
                )

        # Check probability ratios (privacy-critical)
        max_ratio = 0.0
        n, k = p_table.shape
        for i in range(n - 1):
            for j in range(k):
                if p_table[i, j] > tol and p_table[i + 1, j] > tol:
                    ratio = p_table[i, j] / p_table[i + 1, j]
                    max_ratio = max(max_ratio, ratio, 1.0 / ratio)

        if max_ratio > 1e15:
            issues.append(
                f"Very large probability ratio (max={max_ratio:.2e})"
            )

        # Condition number of the probability table
        try:
            cond = float(np.linalg.cond(p_table))
        except np.linalg.LinAlgError:
            cond = float("inf")
            issues.append("Could not compute condition number")

        if cond > 1e12:
            issues.append(f"High condition number ({cond:.2e})")

        return RobustnessReport(
            eps_sensitivity=0.0,
            k_sensitivity=0.0,
            n_sensitivity=0.0,
            condition_number=cond,
            numerical_issues=issues,
        )


# =========================================================================
# 5. StatisticalAnalyzer
# =========================================================================


class StatisticalAnalyzer:
    """Statistical tests and tools for mechanism analysis.

    Provides bootstrap confidence intervals, permutation tests for
    comparing mechanisms, power analysis for experiment design, and
    sample size recommendations.

    Usage::

        analyzer = StatisticalAnalyzer(seed=42)
        ci = analyzer.bootstrap_ci(estimator_fn, data)
        test = analyzer.permutation_test(mech1, mech2, spec)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize with a random seed.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)

    def bootstrap_ci(
        self,
        estimator: Callable[[FloatArray], float],
        data: FloatArray,
        B: int = 1000,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Compute a bootstrap confidence interval for an estimator.

        Uses the percentile bootstrap: resample data B times with
        replacement, compute the estimator on each resample, and take
        the alpha/2 and 1-alpha/2 percentiles.

        Args:
            estimator: Function that takes a data array and returns a scalar.
            data: Data array to bootstrap.
            B: Number of bootstrap resamples.
            alpha: Significance level (default 0.05 for 95% CI).

        Returns:
            StatisticalTestResult with CI bounds.
        """
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        point_estimate = estimator(data)

        boot_estimates = np.empty(B, dtype=np.float64)
        for b in range(B):
            resample = data[self._rng.choice(n, size=n, replace=True)]
            boot_estimates[b] = estimator(resample)

        ci_lower = float(np.percentile(boot_estimates, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_estimates, 100 * (1 - alpha / 2)))
        se = float(np.std(boot_estimates))

        return StatisticalTestResult(
            test_name="bootstrap_ci",
            statistic=point_estimate,
            p_value=0.0,  # CI, not a hypothesis test
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=True,  # CI is always valid
            n_samples=B,
            metadata={"alpha": alpha, "standard_error": se},
        )

    def permutation_test(
        self,
        mechanism1: ExtractedMechanism,
        mechanism2: ExtractedMechanism,
        spec: QuerySpec,
        n_samples: int = 10_000,
        n_permutations: int = 1000,
    ) -> StatisticalTestResult:
        """Two-sample permutation test comparing two mechanisms' MSE.

        Tests H0: MSE(mechanism1) = MSE(mechanism2) against H1: they differ.
        The test statistic is the difference in MSE estimates.

        Args:
            mechanism1: First mechanism.
            mechanism2: Second mechanism.
            spec: Query specification.
            n_samples: Monte Carlo samples for MSE estimation.
            n_permutations: Number of permutations.

        Returns:
            StatisticalTestResult with test statistic and p-value.
        """
        y_grid = _build_output_grid(spec)
        n = min(mechanism1.p_final.shape[0], mechanism2.p_final.shape[0], spec.n)

        errors1 = []
        errors2 = []
        samples_per = max(n_samples // n, 100)

        for i in range(n):
            true_val = spec.query_values[i]
            s1 = _sample_mechanism(mechanism1.p_final, y_grid, i, samples_per, self._rng)
            s2 = _sample_mechanism(mechanism2.p_final, y_grid, i, samples_per, self._rng)
            errors1.extend(((s1 - true_val) ** 2).tolist())
            errors2.extend(((s2 - true_val) ** 2).tolist())

        errors1_arr = np.array(errors1, dtype=np.float64)
        errors2_arr = np.array(errors2, dtype=np.float64)

        observed_diff = float(np.mean(errors1_arr) - np.mean(errors2_arr))

        # Pool and permute
        pooled = np.concatenate([errors1_arr, errors2_arr])
        n1 = len(errors1_arr)
        n_total = len(pooled)

        count_extreme = 0
        for _ in range(n_permutations):
            perm = self._rng.permutation(n_total)
            perm_diff = float(np.mean(pooled[perm[:n1]]) - np.mean(pooled[perm[n1:]]))
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1

        p_value = (count_extreme + 1) / (n_permutations + 1)

        return StatisticalTestResult(
            test_name="permutation_test",
            statistic=observed_diff,
            p_value=p_value,
            ci_lower=0.0,
            ci_upper=0.0,
            significant=p_value < 0.05,
            n_samples=n_permutations,
            metadata={
                "mse1": float(np.mean(errors1_arr)),
                "mse2": float(np.mean(errors2_arr)),
            },
        )

    def power_analysis(
        self,
        effect_size: float,
        n_samples: int,
        alpha: float = 0.05,
    ) -> float:
        """Compute statistical power for a two-sample z-test.

        Given an effect size (Cohen's d), sample size, and significance
        level, compute the probability of detecting a true difference.

        Power = Φ(z_{1-α/2} - d · √n) + Φ(-z_{1-α/2} - d · √n)

        where Φ is the standard normal CDF.

        Args:
            effect_size: Expected effect size (Cohen's d).
            n_samples: Sample size per group.
            alpha: Significance level.

        Returns:
            Statistical power in [0, 1].
        """
        from scipy import stats as sp_stats

        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        ncp = effect_size * math.sqrt(n_samples)  # non-centrality parameter

        power = (
            sp_stats.norm.cdf(-z_alpha + ncp)
            + sp_stats.norm.cdf(-z_alpha - ncp)
        )
        return float(power)

    def sample_size_recommendation(
        self,
        desired_width: float,
        mechanism: ExtractedMechanism,
        spec: QuerySpec,
        confidence: float = 0.95,
        pilot_n: int = 1000,
    ) -> int:
        """Recommend the number of Monte Carlo samples needed.

        Uses a pilot study to estimate the variance of the estimator, then
        computes the sample size needed to achieve a confidence interval
        of the desired width.

        n ≈ (z_{α/2} · σ / (width/2))²

        Args:
            desired_width: Desired CI width (e.g., 0.01 for ±0.005).
            mechanism: Mechanism to estimate variance from.
            spec: Query specification.
            confidence: Confidence level (default 0.95).
            pilot_n: Pilot sample size for variance estimation.

        Returns:
            Recommended total sample size.
        """
        from scipy import stats as sp_stats

        alpha = 1 - confidence
        z = sp_stats.norm.ppf(1 - alpha / 2)

        # Pilot study: estimate variance of squared error
        y_grid = _build_output_grid(spec)
        p_table = mechanism.p_final
        n_inputs = min(p_table.shape[0], spec.n)

        all_sq_errors = []
        pilot_per = max(pilot_n // n_inputs, 10)
        for i in range(n_inputs):
            true_val = spec.query_values[i]
            samples = _sample_mechanism(p_table, y_grid, i, pilot_per, self._rng)
            sq_errors = (samples - true_val) ** 2
            all_sq_errors.extend(sq_errors.tolist())

        sigma = float(np.std(all_sq_errors))
        half_width = desired_width / 2.0

        if half_width <= 0:
            return pilot_n

        n_needed = math.ceil((z * sigma / half_width) ** 2)
        return max(n_needed, pilot_n)


# =========================================================================
# 6. ReportGenerator
# =========================================================================


class ReportGenerator:
    """Generate analysis reports in multiple formats.

    Produces LaTeX, Markdown, and JSON reports from analysis results.
    Reports include tables, figures (as descriptions), and statistical
    summaries.

    Usage::

        gen = ReportGenerator()
        latex = gen.generate_latex_report(analysis)
        md = gen.generate_markdown_report(analysis)
        json_str = gen.generate_json_report(analysis)
    """

    def generate_latex_report(
        self,
        analysis: AnalysisReport,
        title: str = "DP-Forge Mechanism Analysis",
    ) -> str:
        """Generate a full LaTeX report from analysis results.

        Produces a standalone LaTeX document with sections for utility
        metrics, privacy analysis, optimality, and robustness.

        Args:
            analysis: Analysis report to format.
            title: Report title.

        Returns:
            LaTeX document as a string.
        """
        lines = [
            r"\documentclass{article}",
            r"\usepackage{booktabs, amsmath, graphicx}",
            r"\title{" + title + "}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            "",
        ]

        # Utility section
        if analysis.utility is not None:
            u = analysis.utility
            lines.extend([
                r"\section{Utility Metrics}",
                r"\begin{table}[h]",
                r"\centering",
                r"\begin{tabular}{lr}",
                r"\toprule",
                r"Metric & Value \\",
                r"\midrule",
                f"MSE & {u.mse:.6f} \\\\",
                f"MAE & {u.mae:.6f} \\\\",
                f"Variance & {u.variance:.6f} \\\\",
                f"Bias & {u.bias:.6f} \\\\",
                f"Max Error & {u.max_error:.6f} \\\\",
                f"Samples & {u.n_samples:,} \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Utility metrics (Monte Carlo estimates).}",
                r"\end{table}",
                "",
            ])

            if u.quantiles:
                lines.extend([
                    r"\subsection{Error Quantiles}",
                    r"\begin{table}[h]",
                    r"\centering",
                    r"\begin{tabular}{rr}",
                    r"\toprule",
                    r"Quantile & $|$Error$|$ \\",
                    r"\midrule",
                ])
                for q, v in sorted(u.quantiles.items()):
                    lines.append(f"{q:.0%} & {v:.6f} \\\\")
                lines.extend([
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\caption{Error quantiles.}",
                    r"\end{table}",
                    "",
                ])

        # Optimality section
        if analysis.optimality is not None:
            o = analysis.optimality
            lines.extend([
                r"\section{Optimality Analysis}",
                r"\begin{itemize}",
                f"  \\item Mechanism MSE: {o.mechanism_mse:.6f}",
                f"  \\item Lower Bound MSE: {o.lower_bound_mse:.6f}",
                f"  \\item Baseline (Laplace) MSE: {o.baseline_mse:.6f}",
                f"  \\item Optimality Gap: {o.optimality_gap:.4f}",
                f"  \\item Improvement Factor: {o.improvement_factor:.4f}",
                f"  \\item 95\\% CI: [{o.improvement_ci[0]:.4f}, {o.improvement_ci[1]:.4f}]",
                r"\end{itemize}",
                "",
            ])

        # Robustness section
        if analysis.robustness is not None:
            r = analysis.robustness
            lines.extend([
                r"\section{Robustness Analysis}",
                f"Condition number: {r.condition_number:.2e}",
                "",
            ])
            if r.numerical_issues:
                lines.append(r"\subsection{Numerical Issues}")
                lines.append(r"\begin{itemize}")
                for issue in r.numerical_issues:
                    lines.append(f"  \\item {issue}")
                lines.append(r"\end{itemize}")
            else:
                lines.append("No numerical issues detected.")
            lines.append("")

        # Statistical tests
        if analysis.statistical_tests:
            lines.extend([
                r"\section{Statistical Tests}",
                r"\begin{table}[h]",
                r"\centering",
                r"\begin{tabular}{lrrrl}",
                r"\toprule",
                r"Test & Statistic & p-value & CI & Significant \\",
                r"\midrule",
            ])
            for test in analysis.statistical_tests:
                sig = "Yes" if test.significant else "No"
                ci_str = f"[{test.ci_lower:.4f}, {test.ci_upper:.4f}]"
                lines.append(
                    f"{test.test_name} & {test.statistic:.4f} & "
                    f"{test.p_value:.4f} & {ci_str} & {sig} \\\\"
                )
            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Statistical test results.}",
                r"\end{table}",
                "",
            ])

        lines.extend([
            r"\end{document}",
        ])

        return "\n".join(lines)

    def generate_markdown_report(
        self,
        analysis: AnalysisReport,
        title: str = "DP-Forge Mechanism Analysis",
    ) -> str:
        """Generate a Markdown report from analysis results.

        Args:
            analysis: Analysis report to format.
            title: Report title.

        Returns:
            Markdown document as a string.
        """
        lines = [f"# {title}", ""]

        # Utility section
        if analysis.utility is not None:
            u = analysis.utility
            lines.extend([
                "## Utility Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| MSE | {u.mse:.6f} |",
                f"| MAE | {u.mae:.6f} |",
                f"| Variance | {u.variance:.6f} |",
                f"| Bias | {u.bias:.6f} |",
                f"| Max Error | {u.max_error:.6f} |",
                f"| Samples | {u.n_samples:,} |",
                "",
            ])

            if u.quantiles:
                lines.extend([
                    "### Error Quantiles",
                    "",
                    "| Quantile | |Error| |",
                    "|----------|---------|",
                ])
                for q, v in sorted(u.quantiles.items()):
                    lines.append(f"| {q:.0%} | {v:.6f} |")
                lines.append("")

            if u.tail_probabilities:
                lines.extend([
                    "### Tail Probabilities",
                    "",
                    "| Threshold | P(|error| > t) |",
                    "|-----------|----------------|",
                ])
                for t, p in sorted(u.tail_probabilities.items()):
                    lines.append(f"| {t:.1f} | {p:.6f} |")
                lines.append("")

        # Optimality section
        if analysis.optimality is not None:
            o = analysis.optimality
            lines.extend([
                "## Optimality Analysis",
                "",
                f"- **Mechanism MSE**: {o.mechanism_mse:.6f}",
                f"- **Lower Bound MSE**: {o.lower_bound_mse:.6f}",
                f"- **Baseline (Laplace) MSE**: {o.baseline_mse:.6f}",
                f"- **Optimality Gap**: {o.optimality_gap:.4f}",
                f"- **Improvement Factor**: {o.improvement_factor:.4f}",
                f"- **95% CI**: [{o.improvement_ci[0]:.4f}, {o.improvement_ci[1]:.4f}]",
                "",
            ])

        # Robustness section
        if analysis.robustness is not None:
            r = analysis.robustness
            lines.extend([
                "## Robustness Analysis",
                "",
                f"- **Condition Number**: {r.condition_number:.2e}",
                f"- **Status**: {'Stable ✓' if r.is_stable else 'Issues Detected ✗'}",
                "",
            ])
            if r.numerical_issues:
                lines.append("### Numerical Issues")
                for issue in r.numerical_issues:
                    lines.append(f"- {issue}")
                lines.append("")

        # Statistical tests
        if analysis.statistical_tests:
            lines.extend([
                "## Statistical Tests",
                "",
                "| Test | Statistic | p-value | CI | Significant |",
                "|------|-----------|---------|-----|-------------|",
            ])
            for test in analysis.statistical_tests:
                sig = "✓" if test.significant else "✗"
                ci_str = f"[{test.ci_lower:.4f}, {test.ci_upper:.4f}]"
                lines.append(
                    f"| {test.test_name} | {test.statistic:.4f} | "
                    f"{test.p_value:.4f} | {ci_str} | {sig} |"
                )
            lines.append("")

        if analysis.generation_time > 0:
            lines.append(f"*Report generated in {analysis.generation_time:.2f}s*")

        return "\n".join(lines)

    def generate_json_report(
        self,
        analysis: AnalysisReport,
    ) -> Dict[str, Any]:
        """Generate a machine-readable JSON report.

        Args:
            analysis: Analysis report.

        Returns:
            Dict suitable for JSON serialisation.
        """
        report: Dict[str, Any] = {}

        if analysis.utility is not None:
            u = analysis.utility
            report["utility"] = {
                "mse": u.mse,
                "mae": u.mae,
                "variance": u.variance,
                "bias": u.bias,
                "max_error": u.max_error,
                "n_samples": u.n_samples,
                "quantiles": u.quantiles,
                "tail_probabilities": u.tail_probabilities,
            }

        if analysis.privacy_curve is not None:
            c = analysis.privacy_curve
            report["privacy_curve"] = {
                "type": c.curve_type,
                "epsilons": c.epsilons.tolist(),
                "deltas": c.deltas.tolist(),
                "n_points": len(c.epsilons),
            }
            if c.alphas is not None:
                report["privacy_curve"]["alphas"] = c.alphas.tolist()

        if analysis.optimality is not None:
            o = analysis.optimality
            report["optimality"] = {
                "mechanism_mse": o.mechanism_mse,
                "lower_bound_mse": o.lower_bound_mse,
                "baseline_mse": o.baseline_mse,
                "optimality_gap": o.optimality_gap,
                "improvement_factor": o.improvement_factor,
                "improvement_ci": list(o.improvement_ci),
            }

        if analysis.robustness is not None:
            r = analysis.robustness
            report["robustness"] = {
                "condition_number": r.condition_number,
                "is_stable": r.is_stable,
                "numerical_issues": r.numerical_issues,
                "eps_sensitivity": r.eps_sensitivity,
                "k_sensitivity": r.k_sensitivity,
                "n_sensitivity": r.n_sensitivity,
            }

        if analysis.statistical_tests:
            report["statistical_tests"] = [
                {
                    "test_name": t.test_name,
                    "statistic": t.statistic,
                    "p_value": t.p_value,
                    "ci_lower": t.ci_lower,
                    "ci_upper": t.ci_upper,
                    "significant": t.significant,
                    "n_samples": t.n_samples,
                }
                for t in analysis.statistical_tests
            ]

        report["generation_time"] = analysis.generation_time
        report["metadata"] = analysis.metadata

        return report

    def generate_comparison_table(
        self,
        results: Dict[str, AnalysisReport],
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Generate a side-by-side comparison table in Markdown.

        Args:
            results: Dict mapping mechanism names to their analysis reports.
            metrics: Metrics to include. Defaults to ['mse', 'mae', 'variance'].

        Returns:
            Markdown table as a string.
        """
        if metrics is None:
            metrics = ["mse", "mae", "variance", "bias"]

        names = list(results.keys())
        header = "| Metric | " + " | ".join(names) + " |"
        separator = "|--------" + "|-------" * len(names) + "|"

        lines = [header, separator]

        for metric in metrics:
            row = f"| {metric} |"
            for name in names:
                report = results[name]
                if report.utility is not None and hasattr(report.utility, metric):
                    val = getattr(report.utility, metric)
                    row += f" {val:.6f} |"
                else:
                    row += " N/A |"
            lines.append(row)

        return "\n".join(lines)
