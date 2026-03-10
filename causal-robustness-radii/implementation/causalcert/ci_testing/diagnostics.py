"""
CI test diagnostics and calibration assessment.

Provides tools for assessing CI test quality:
- Power curve estimation via bootstrap
- Effect size quantification
- Calibration assessment (uniform p-values under the null)
- QQ plot data generation
- Test agreement matrix across ensemble members
- Sensitivity to sample size

These diagnostics help users understand test reliability and choose
appropriate significance levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from causalcert.ci_testing.base import BaseCITest
from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Diagnostic result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PowerCurvePoint:
    """A single point on a power curve.

    Attributes
    ----------
    effect_size : float
        Alternative-hypothesis effect size.
    power : float
        Estimated power at this effect size.
    se : float
        Standard error of the power estimate.
    """

    effect_size: float
    power: float
    se: float


@dataclass(slots=True)
class CalibrationResult:
    """Result of a calibration assessment.

    Attributes
    ----------
    ks_statistic : float
        Kolmogorov–Smirnov statistic against Uniform(0, 1).
    ks_pvalue : float
        KS test p-value (low ⇒ miscalibrated).
    mean_pvalue : float
        Mean of observed p-values (should be ≈ 0.5 under null).
    n_tests : int
        Number of p-values used.
    p_values : np.ndarray
        Collected p-values.
    """

    ks_statistic: float
    ks_pvalue: float
    mean_pvalue: float
    n_tests: int
    p_values: np.ndarray


@dataclass(slots=True)
class QQPlotData:
    """Data for a QQ plot of p-values against Uniform(0, 1).

    Attributes
    ----------
    theoretical : np.ndarray
        Expected quantiles.
    observed : np.ndarray
        Observed p-value quantiles.
    ci_lower : np.ndarray
        Lower 95 % confidence band.
    ci_upper : np.ndarray
        Upper 95 % confidence band.
    """

    theoretical: np.ndarray
    observed: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray


@dataclass(slots=True)
class AgreementMatrix:
    """Pairwise agreement rates among ensemble CI tests.

    Attributes
    ----------
    methods : list[CITestMethod]
        Test methods (row/column labels).
    agreement : np.ndarray
        ``(K, K)`` matrix where entry ``(i, j)`` is the fraction of
        tests on which methods *i* and *j* agree.
    n_triples : int
        Number of CI triples tested.
    """

    methods: list[CITestMethod]
    agreement: np.ndarray
    n_triples: int


@dataclass(slots=True)
class SensitivityPoint:
    """Sensitivity of a test to sample size at a single n.

    Attributes
    ----------
    n : int
        Sample size.
    rejection_rate : float
        Fraction of bootstrap replicates that reject.
    mean_pvalue : float
        Mean p-value across replicates.
    se_pvalue : float
        Standard error of the mean p-value.
    """

    n: int
    rejection_rate: float
    mean_pvalue: float
    se_pvalue: float


# ---------------------------------------------------------------------------
# CIDiagnostics class
# ---------------------------------------------------------------------------


class CIDiagnostics:
    """Diagnostic suite for conditional-independence tests.

    All methods are static or take explicit arguments so that the class
    can be used without instantiation for convenience.

    Parameters
    ----------
    seed : int
        Random seed for all stochastic diagnostics.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    # ------------------------------------------------------------------
    # Power curve estimation
    # ------------------------------------------------------------------

    def power_curve(
        self,
        test: BaseCITest,
        n: int,
        conditioning_size: int,
        effect_sizes: Sequence[float] | None = None,
        *,
        n_simulations: int = 200,
    ) -> list[PowerCurvePoint]:
        """Estimate the power curve via Monte-Carlo simulation.

        For each effect size, generates data under the alternative
        (partial correlation = effect_size) and estimates the rejection
        rate.

        Parameters
        ----------
        test : BaseCITest
            CI test to evaluate.
        n : int
            Sample size per simulation.
        conditioning_size : int
            Number of conditioning variables.
        effect_sizes : Sequence[float] | None
            Grid of effect sizes.  ``None`` ⇒ ``[0, 0.05, ..., 0.5]``.
        n_simulations : int
            Simulations per effect size.

        Returns
        -------
        list[PowerCurvePoint]
        """
        if effect_sizes is None:
            effect_sizes = np.linspace(0.0, 0.5, 11).tolist()

        rng = np.random.default_rng(self.seed)
        k = conditioning_size
        points: list[PowerCurvePoint] = []

        for es in effect_sizes:
            rejections = 0
            for _ in range(n_simulations):
                data = self._generate_data(n, k, es, rng)
                x, y = 0, 1
                cond = frozenset(range(2, 2 + k))
                try:
                    result = test.test(x, y, cond, data)
                    if result.reject:
                        rejections += 1
                except Exception:
                    pass

            power = rejections / n_simulations
            se = math.sqrt(power * (1 - power) / max(n_simulations, 1))
            points.append(PowerCurvePoint(
                effect_size=es, power=power, se=se,
            ))

        return points

    # ------------------------------------------------------------------
    # Effect size quantification
    # ------------------------------------------------------------------

    @staticmethod
    def effect_size_from_result(result: CITestResult) -> float:
        """Estimate effect size from a CI test result.

        For test statistics that can be interpreted as correlations,
        converts the statistic to Cohen's f².

        Parameters
        ----------
        result : CITestResult
            CI test result.

        Returns
        -------
        float
            Estimated Cohen's f².
        """
        s = abs(result.statistic)
        if result.method == CITestMethod.PARTIAL_CORRELATION:
            r2 = min(s ** 2, 1.0 - _EPS) if s < 1 else 1.0 - _EPS
            return r2 / (1.0 - r2)
        elif result.method == CITestMethod.RANK:
            r2 = min(s ** 2, 1.0 - _EPS) if s < 1 else 1.0 - _EPS
            return r2 / (1.0 - r2)
        # For non-correlation-based tests, return the statistic itself
        return s

    @staticmethod
    def cohens_d_from_f2(f2: float) -> float:
        """Convert Cohen's f² to Cohen's d.

        Parameters
        ----------
        f2 : float
            Cohen's f².

        Returns
        -------
        float
            Cohen's d.
        """
        return math.sqrt(f2) * 2.0

    # ------------------------------------------------------------------
    # Calibration assessment
    # ------------------------------------------------------------------

    def calibration_assessment(
        self,
        test: BaseCITest,
        n: int,
        conditioning_size: int,
        *,
        n_simulations: int = 200,
    ) -> CalibrationResult:
        """Assess calibration by testing under the null.

        Generates data where X ⊥ Y | Z (effect_size = 0), runs the
        test repeatedly, and checks whether p-values are uniform.

        Parameters
        ----------
        test : BaseCITest
            CI test to assess.
        n : int
            Sample size.
        conditioning_size : int
            Number of conditioning variables.
        n_simulations : int
            Number of replications.

        Returns
        -------
        CalibrationResult
        """
        rng = np.random.default_rng(self.seed)
        k = conditioning_size
        p_values: list[float] = []

        for _ in range(n_simulations):
            data = self._generate_data(n, k, 0.0, rng)
            x, y = 0, 1
            cond = frozenset(range(2, 2 + k))
            try:
                result = test.test(x, y, cond, data)
                p_values.append(result.p_value)
            except Exception:
                continue

        p_arr = np.array(p_values) if p_values else np.array([0.5])
        ks_stat, ks_p = stats.kstest(p_arr, "uniform")

        return CalibrationResult(
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_p),
            mean_pvalue=float(np.mean(p_arr)),
            n_tests=len(p_arr),
            p_values=p_arr,
        )

    # ------------------------------------------------------------------
    # QQ plot data
    # ------------------------------------------------------------------

    @staticmethod
    def qq_plot_data(p_values: np.ndarray) -> QQPlotData:
        """Generate QQ plot data for p-values vs Uniform(0, 1).

        Parameters
        ----------
        p_values : np.ndarray
            Observed p-values.

        Returns
        -------
        QQPlotData
        """
        n = len(p_values)
        if n == 0:
            return QQPlotData(
                theoretical=np.array([]),
                observed=np.array([]),
                ci_lower=np.array([]),
                ci_upper=np.array([]),
            )

        sorted_p = np.sort(p_values)
        theoretical = (np.arange(1, n + 1) - 0.5) / n

        # 95% pointwise confidence band using the order-statistic
        # distribution: U_(i) ~ Beta(i, n-i+1)
        ci_lower = np.array([
            stats.beta.ppf(0.025, i, n - i + 1)
            for i in range(1, n + 1)
        ])
        ci_upper = np.array([
            stats.beta.ppf(0.975, i, n - i + 1)
            for i in range(1, n + 1)
        ])

        return QQPlotData(
            theoretical=theoretical,
            observed=sorted_p,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    # ------------------------------------------------------------------
    # Agreement matrix
    # ------------------------------------------------------------------

    def agreement_matrix(
        self,
        tests: Sequence[BaseCITest],
        triples: list[tuple[NodeId, NodeId, NodeSet]],
        data: pd.DataFrame,
    ) -> AgreementMatrix:
        """Compute pairwise agreement rates among CI tests.

        For each pair of tests, computes the fraction of triples on
        which both agree (both reject or both fail to reject).

        Parameters
        ----------
        tests : Sequence[BaseCITest]
            CI tests to compare.
        triples : list[tuple[NodeId, NodeId, NodeSet]]
            CI test triples.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        AgreementMatrix
        """
        K = len(tests)
        n_triples = len(triples)
        methods = [t.method for t in tests]

        # Collect decisions
        decisions = np.zeros((K, n_triples), dtype=bool)
        for i, test in enumerate(tests):
            for j, (x, y, s) in enumerate(triples):
                try:
                    result = test.test(x, y, s, data)
                    decisions[i, j] = result.reject
                except Exception:
                    decisions[i, j] = False

        # Pairwise agreement
        agreement = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            for j in range(K):
                if n_triples > 0:
                    agreement[i, j] = float(
                        np.mean(decisions[i] == decisions[j])
                    )
                else:
                    agreement[i, j] = 1.0 if i == j else 0.0

        return AgreementMatrix(
            methods=methods,
            agreement=agreement,
            n_triples=n_triples,
        )

    # ------------------------------------------------------------------
    # Sensitivity to sample size
    # ------------------------------------------------------------------

    def sensitivity_to_sample_size(
        self,
        test: BaseCITest,
        data: pd.DataFrame,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        sample_sizes: Sequence[int] | None = None,
        *,
        n_bootstrap: int = 100,
    ) -> list[SensitivityPoint]:
        """Assess how test results change with sample size.

        For each target sample size, draws bootstrap sub-samples and
        reports the rejection rate and mean p-value.

        Parameters
        ----------
        test : BaseCITest
            CI test.
        data : pd.DataFrame
            Full dataset.
        x, y : NodeId
            Variables.
        conditioning_set : NodeSet
            Conditioning set.
        sample_sizes : Sequence[int] | None
            Sample sizes to evaluate.  ``None`` ⇒ auto-generated.
        n_bootstrap : int
            Bootstrap replicates per sample size.

        Returns
        -------
        list[SensitivityPoint]
        """
        n_full = len(data)
        if sample_sizes is None:
            sample_sizes = [
                s for s in [
                    20, 50, 100, 200, 500, 1000, 2000,
                ]
                if s <= n_full
            ]
            if not sample_sizes:
                sample_sizes = [n_full]

        rng = np.random.default_rng(self.seed)
        points: list[SensitivityPoint] = []

        for ns in sample_sizes:
            p_values: list[float] = []
            rejections = 0
            for _ in range(n_bootstrap):
                idx = rng.choice(n_full, size=min(ns, n_full), replace=True)
                sub = data.iloc[idx].reset_index(drop=True)
                try:
                    result = test.test(x, y, conditioning_set, sub)
                    p_values.append(result.p_value)
                    if result.reject:
                        rejections += 1
                except Exception:
                    p_values.append(1.0)

            p_arr = np.array(p_values)
            points.append(SensitivityPoint(
                n=ns,
                rejection_rate=rejections / max(n_bootstrap, 1),
                mean_pvalue=float(np.mean(p_arr)),
                se_pvalue=float(np.std(p_arr) / math.sqrt(len(p_arr))),
            ))

        return points

    # ------------------------------------------------------------------
    # Internal data generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_data(
        n: int,
        k: int,
        effect_size: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate synthetic data for power / calibration analysis.

        Creates a dataset where the partial correlation between columns
        0 and 1 given columns 2..k+1 equals ``effect_size``.

        Parameters
        ----------
        n : int
            Sample size.
        k : int
            Number of conditioning variables.
        effect_size : float
            Target partial correlation.
        rng : np.random.Generator
            RNG.

        Returns
        -------
        pd.DataFrame
            Data with columns ``0, 1, ..., k+1``.
        """
        Z = rng.standard_normal((n, k)) if k > 0 else np.empty((n, 0))

        beta_x = rng.standard_normal(k) * 0.3 if k > 0 else np.array([])
        eps_x = rng.standard_normal(n)
        X = (Z @ beta_x if k > 0 else 0.0) + eps_x

        beta_y = rng.standard_normal(k) * 0.3 if k > 0 else np.array([])
        eps_y = rng.standard_normal(n)
        Y = effect_size * X + (Z @ beta_y if k > 0 else 0.0) + eps_y

        cols = {0: X, 1: Y}
        for j in range(k):
            cols[2 + j] = Z[:, j]

        return pd.DataFrame(cols)
