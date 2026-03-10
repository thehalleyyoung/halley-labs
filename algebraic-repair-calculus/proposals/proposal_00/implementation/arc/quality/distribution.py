"""
Distribution shift detection using statistical tests.

:class:`DistributionAnalyzer` provides:

* **Kolmogorov-Smirnov test** – non-parametric two-sample test for
  continuous distributions.
* **Population Stability Index (PSI)** – binned divergence metric
  commonly used in credit-risk modelling.
* **Chi-squared test** – goodness-of-fit test for categorical data.
* **Jensen-Shannon divergence** – symmetric KL-based divergence.
* **Column profiling** and **profile comparison**.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from arc.types.base import (
    Anomaly,
    ChiSquaredResult,
    ColumnProfile,
    KSResult,
    ProfileDiff,
    ShiftResult,
)

logger = logging.getLogger(__name__)

_EPS = 1e-12  # prevent log(0)


class DistributionAnalyzer:
    """Statistical tests for distribution shift detection.

    All methods accept numpy arrays (or array-like objects).
    NaN values are dropped before analysis.
    """

    # ── Kolmogorov-Smirnov test ────────────────────────────────────────

    @staticmethod
    def ks_test(
        sample1: Any,
        sample2: Any,
    ) -> KSResult:
        """Two-sample Kolmogorov-Smirnov test.

        Parameters
        ----------
        sample1, sample2:
            Array-like samples to compare.

        Returns
        -------
        KSResult
            Test statistic and p-value.
        """
        a = np.asarray(sample1, dtype=float)
        b = np.asarray(sample2, dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

        n1, n2 = len(a), len(b)
        if n1 == 0 or n2 == 0:
            return KSResult(statistic=0.0, p_value=1.0, sample_size_1=n1, sample_size_2=n2)

        try:
            from scipy.stats import ks_2samp
            stat, pval = ks_2samp(a, b)
            return KSResult(
                statistic=float(stat),
                p_value=float(pval),
                sample_size_1=n1,
                sample_size_2=n2,
            )
        except ImportError:
            # Pure-numpy fallback
            stat = _numpy_ks_statistic(a, b)
            # Approximate p-value using the asymptotic distribution
            en = math.sqrt(n1 * n2 / (n1 + n2))
            pval = _ks_pvalue(stat * en)
            return KSResult(
                statistic=float(stat),
                p_value=float(pval),
                sample_size_1=n1,
                sample_size_2=n2,
            )

    # ── Population Stability Index ─────────────────────────────────────

    @staticmethod
    def psi_score(
        expected: Any,
        actual: Any,
        bins: int = 10,
    ) -> float:
        """Population Stability Index between two distributions.

        PSI = Σ (actual_i - expected_i) × ln(actual_i / expected_i)

        A PSI < 0.1 indicates no significant shift.
        PSI ∈ [0.1, 0.25) indicates moderate shift.
        PSI ≥ 0.25 indicates significant shift.

        Parameters
        ----------
        expected:
            The reference (baseline) distribution.
        actual:
            The new distribution to compare.
        bins:
            Number of equal-frequency bins.

        Returns
        -------
        float
            The PSI score.
        """
        exp = np.asarray(expected, dtype=float)
        act = np.asarray(actual, dtype=float)
        exp = exp[~np.isnan(exp)]
        act = act[~np.isnan(act)]

        if len(exp) == 0 or len(act) == 0:
            return 0.0

        # Create bins from the expected distribution
        breakpoints = np.percentile(exp, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        exp_counts = np.histogram(exp, bins=breakpoints)[0].astype(float)
        act_counts = np.histogram(act, bins=breakpoints)[0].astype(float)

        # Normalise to proportions
        exp_props = exp_counts / max(exp_counts.sum(), 1)
        act_props = act_counts / max(act_counts.sum(), 1)

        # Clip to avoid log(0)
        exp_props = np.clip(exp_props, _EPS, 1.0)
        act_props = np.clip(act_props, _EPS, 1.0)

        psi = float(np.sum((act_props - exp_props) * np.log(act_props / exp_props)))
        return max(psi, 0.0)

    # ── Chi-squared test ───────────────────────────────────────────────

    @staticmethod
    def chi_squared_test(
        observed: Any,
        expected: Any,
    ) -> ChiSquaredResult:
        """Chi-squared goodness-of-fit test.

        Parameters
        ----------
        observed:
            Observed frequency counts.
        expected:
            Expected frequency counts.

        Returns
        -------
        ChiSquaredResult
        """
        obs = np.asarray(observed, dtype=float)
        exp = np.asarray(expected, dtype=float)

        if len(obs) != len(exp) or len(obs) == 0:
            return ChiSquaredResult(statistic=0.0, p_value=1.0, degrees_of_freedom=0)

        # Scale expected to match observed total
        obs_total = obs.sum()
        exp_total = exp.sum()
        if exp_total > 0:
            exp = exp * (obs_total / exp_total)

        exp = np.clip(exp, _EPS, None)
        dof = max(len(obs) - 1, 1)

        try:
            from scipy.stats import chisquare
            stat, pval = chisquare(obs, f_exp=exp)
            return ChiSquaredResult(
                statistic=float(stat),
                p_value=float(pval),
                degrees_of_freedom=dof,
            )
        except ImportError:
            stat = float(np.sum((obs - exp) ** 2 / exp))
            # Approximate p-value using chi-squared CDF
            pval = _chi2_pvalue(stat, dof)
            return ChiSquaredResult(
                statistic=stat,
                p_value=pval,
                degrees_of_freedom=dof,
            )

    # ── Jensen-Shannon divergence ──────────────────────────────────────

    @staticmethod
    def jensen_shannon_divergence(
        p: Any,
        q: Any,
    ) -> float:
        """Jensen-Shannon divergence between two probability distributions.

        JSD(P || Q) = 0.5 × KL(P || M) + 0.5 × KL(Q || M)
        where M = 0.5 × (P + Q).

        Parameters
        ----------
        p, q:
            Probability distributions (must sum to 1).

        Returns
        -------
        float
            JSD in nats ∈ [0, ln(2)].
        """
        p_arr = np.asarray(p, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        if len(p_arr) != len(q_arr) or len(p_arr) == 0:
            return 0.0

        # Normalise
        p_arr = np.clip(p_arr, _EPS, None)
        q_arr = np.clip(q_arr, _EPS, None)
        p_arr = p_arr / p_arr.sum()
        q_arr = q_arr / q_arr.sum()

        m = 0.5 * (p_arr + q_arr)
        kl_pm = float(np.sum(p_arr * np.log(p_arr / m)))
        kl_qm = float(np.sum(q_arr * np.log(q_arr / m)))
        return 0.5 * kl_pm + 0.5 * kl_qm

    # ── Multi-column shift detection ───────────────────────────────────

    def detect_shift(
        self,
        old_data: Any,
        new_data: Any,
        columns: list[str],
        threshold: float = 0.1,
    ) -> list[ShiftResult]:
        """Detect distribution shifts across multiple columns.

        For each column, runs a KS test and computes PSI.  Reports a
        shift if either metric exceeds *threshold*.

        Parameters
        ----------
        old_data:
            The baseline dataset.
        new_data:
            The new dataset.
        columns:
            Columns to test.
        threshold:
            PSI threshold for reporting a shift.

        Returns
        -------
        list[ShiftResult]
        """
        results: list[ShiftResult] = []

        for col in columns:
            try:
                old_arr = np.asarray(old_data[col], dtype=float)
                new_arr = np.asarray(new_data[col], dtype=float)
            except (ValueError, TypeError, KeyError):
                results.append(ShiftResult(
                    column_name=col,
                    test_name="skip",
                    statistic=0.0,
                    p_value=1.0,
                    shifted=False,
                    threshold=threshold,
                    message=f"column {col} is not numeric or missing",
                ))
                continue

            old_clean = old_arr[~np.isnan(old_arr)]
            new_clean = new_arr[~np.isnan(new_arr)]

            if len(old_clean) == 0 or len(new_clean) == 0:
                results.append(ShiftResult(
                    column_name=col,
                    test_name="skip",
                    statistic=0.0,
                    p_value=1.0,
                    shifted=False,
                    threshold=threshold,
                    message=f"column {col} has no valid values",
                ))
                continue

            # KS test
            ks = self.ks_test(old_clean, new_clean)
            psi = self.psi_score(old_clean, new_clean)

            shifted = psi >= threshold or ks.p_value < 0.05
            results.append(ShiftResult(
                column_name=col,
                test_name="ks+psi",
                statistic=ks.statistic,
                p_value=ks.p_value,
                shifted=shifted,
                threshold=threshold,
                message=(
                    f"KS={ks.statistic:.4f} (p={ks.p_value:.4f}), PSI={psi:.4f}"
                ),
            ))

        return results

    # ── Column profiling ───────────────────────────────────────────────

    @staticmethod
    def compute_column_profile(
        data: Any,
        column: str,
    ) -> ColumnProfile:
        """Compute a statistical profile for a single column.

        Parameters
        ----------
        data:
            The dataset.
        column:
            Column name.

        Returns
        -------
        ColumnProfile
        """
        try:
            arr = np.asarray(data[column] if not isinstance(data, np.ndarray) else data)
        except Exception:
            return ColumnProfile(column_name=column)

        count = len(arr)
        null_mask = _null_mask(arr)
        null_count = int(np.sum(null_mask))
        non_null = arr[~null_mask]

        unique_count = len(set(non_null.tolist())) if len(non_null) > 0 else 0

        # Type detection
        dtype_str = str(arr.dtype)

        mean_val: float | None = None
        std_val: float | None = None
        min_val: Any = None
        max_val: Any = None
        percentiles: dict[int, float] = {}
        hist_counts: tuple[float, ...] = ()
        hist_edges: tuple[float, ...] = ()

        try:
            numeric = non_null.astype(float)
            numeric = numeric[~np.isnan(numeric)]
            if len(numeric) > 0:
                mean_val = float(np.mean(numeric))
                std_val = float(np.std(numeric))
                min_val = float(np.min(numeric))
                max_val = float(np.max(numeric))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                    percentiles[p] = float(np.percentile(numeric, p))
                if len(numeric) >= 2:
                    counts, edges = np.histogram(numeric, bins=min(20, len(set(numeric.tolist()))))
                    hist_counts = tuple(float(c) for c in counts)
                    hist_edges = tuple(float(e) for e in edges)
        except (ValueError, TypeError):
            # Non-numeric column
            if len(non_null) > 0:
                min_val = non_null.min()
                max_val = non_null.max()

        # Most common values
        most_common: tuple[tuple[Any, int], ...] = ()
        if len(non_null) > 0:
            from collections import Counter
            counter = Counter(non_null.tolist())
            most_common = tuple(counter.most_common(10))

        return ColumnProfile(
            column_name=column,
            dtype=dtype_str,
            count=count,
            null_count=null_count,
            unique_count=unique_count,
            mean=mean_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            percentiles=percentiles,
            histogram_counts=hist_counts,
            histogram_edges=hist_edges,
            most_common=most_common,
        )

    # ── Profile comparison ─────────────────────────────────────────────

    def compare_profiles(
        self,
        old_profile: ColumnProfile,
        new_profile: ColumnProfile,
    ) -> ProfileDiff:
        """Compare two column profiles and detect anomalies.

        Returns
        -------
        ProfileDiff
        """
        anomalies: list[Anomaly] = []

        count_diff = new_profile.count - old_profile.count
        null_diff = new_profile.null_count - old_profile.null_count
        unique_diff = new_profile.unique_count - old_profile.unique_count

        mean_diff: float | None = None
        std_diff: float | None = None
        min_diff: float | None = None
        max_diff: float | None = None

        if old_profile.mean is not None and new_profile.mean is not None:
            mean_diff = new_profile.mean - old_profile.mean
            if old_profile.std is not None and old_profile.std > 0:
                z_score = abs(mean_diff) / old_profile.std
                if z_score > 3.0:
                    anomalies.append(Anomaly(
                        column_name=old_profile.column_name,
                        anomaly_type="mean_shift",
                        message=f"Mean shifted by {z_score:.1f} std devs",
                        severity="warning" if z_score < 5.0 else "error",
                        old_value=old_profile.mean,
                        new_value=new_profile.mean,
                        score=z_score,
                    ))

        if old_profile.std is not None and new_profile.std is not None:
            std_diff = new_profile.std - old_profile.std
            if old_profile.std > 0:
                ratio = new_profile.std / old_profile.std
                if ratio > 2.0 or ratio < 0.5:
                    anomalies.append(Anomaly(
                        column_name=old_profile.column_name,
                        anomaly_type="variance_change",
                        message=f"Std dev ratio {ratio:.2f}",
                        severity="warning",
                        old_value=old_profile.std,
                        new_value=new_profile.std,
                        score=abs(math.log(ratio)),
                    ))

        if old_profile.min_val is not None and new_profile.min_val is not None:
            try:
                min_diff = float(new_profile.min_val) - float(old_profile.min_val)
            except (ValueError, TypeError):
                pass

        if old_profile.max_val is not None and new_profile.max_val is not None:
            try:
                max_diff = float(new_profile.max_val) - float(old_profile.max_val)
            except (ValueError, TypeError):
                pass

        # Check null rate change
        old_null_rate = old_profile.null_rate
        new_null_rate = new_profile.null_rate
        null_rate_change = new_null_rate - old_null_rate
        if abs(null_rate_change) > 0.1:
            anomalies.append(Anomaly(
                column_name=old_profile.column_name,
                anomaly_type="null_rate_change",
                message=f"Null rate changed by {null_rate_change:+.2%}",
                severity="warning",
                old_value=old_null_rate,
                new_value=new_null_rate,
                score=abs(null_rate_change),
            ))

        # Check uniqueness change
        old_uniq = old_profile.uniqueness
        new_uniq = new_profile.uniqueness
        if old_uniq > 0 and abs(new_uniq - old_uniq) > 0.2:
            anomalies.append(Anomaly(
                column_name=old_profile.column_name,
                anomaly_type="uniqueness_change",
                message=f"Uniqueness changed from {old_uniq:.2f} to {new_uniq:.2f}",
                severity="warning",
                old_value=old_uniq,
                new_value=new_uniq,
                score=abs(new_uniq - old_uniq),
            ))

        # Distribution shift via histogram comparison
        shift: ShiftResult | None = None
        if old_profile.histogram_counts and new_profile.histogram_counts:
            old_h = np.array(old_profile.histogram_counts)
            new_h = np.array(new_profile.histogram_counts)
            min_len = min(len(old_h), len(new_h))
            if min_len > 0:
                old_h = old_h[:min_len]
                new_h = new_h[:min_len]
                old_p = old_h / max(old_h.sum(), 1)
                new_p = new_h / max(new_h.sum(), 1)
                jsd = self.jensen_shannon_divergence(old_p, new_p)
                shifted = jsd > 0.1
                shift = ShiftResult(
                    column_name=old_profile.column_name,
                    test_name="jsd",
                    statistic=jsd,
                    p_value=1.0 - jsd,
                    shifted=shifted,
                    threshold=0.1,
                    message=f"JSD={jsd:.4f}",
                )
                if shifted:
                    anomalies.append(Anomaly(
                        column_name=old_profile.column_name,
                        anomaly_type="distribution_shift",
                        message=f"JSD={jsd:.4f} exceeds threshold",
                        severity="warning",
                        score=jsd,
                    ))

        return ProfileDiff(
            column_name=old_profile.column_name,
            count_diff=count_diff,
            null_count_diff=null_diff,
            unique_count_diff=unique_diff,
            mean_diff=mean_diff,
            std_diff=std_diff,
            min_diff=min_diff,
            max_diff=max_diff,
            distribution_shift=shift,
            anomalies=tuple(anomalies),
        )

    def __repr__(self) -> str:
        return "DistributionAnalyzer()"


# ── Private helper functions ──────────────────────────────────────────

def _null_mask(arr: np.ndarray) -> np.ndarray:
    """Create a boolean mask for null/NaN values."""
    mask = np.zeros(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        if v is None:
            mask[i] = True
        else:
            try:
                if np.isnan(float(v)):
                    mask[i] = True
            except (ValueError, TypeError):
                pass
    return mask


def _numpy_ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Pure-numpy two-sample KS statistic."""
    combined = np.concatenate([a, b])
    combined.sort()
    n1, n2 = len(a), len(b)
    cdf1 = np.searchsorted(np.sort(a), combined, side="right") / n1
    cdf2 = np.searchsorted(np.sort(b), combined, side="right") / n2
    return float(np.max(np.abs(cdf1 - cdf2)))


def _ks_pvalue(z: float) -> float:
    """Approximate KS p-value from the scaled statistic z."""
    if z <= 0:
        return 1.0
    # Kolmogorov asymptotic formula
    total = 0.0
    for k in range(1, 20):
        total += (-1) ** (k - 1) * math.exp(-2.0 * k * k * z * z)
    return max(min(2.0 * total, 1.0), 0.0)


def _chi2_pvalue(stat: float, dof: int) -> float:
    """Rough chi-squared p-value approximation."""
    if dof <= 0:
        return 1.0
    # Wilson-Hilferty approximation
    z = ((stat / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(2.0 / (9.0 * dof))
    # Standard normal CDF approximation
    return max(1.0 - _normal_cdf(z), 0.0)


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
