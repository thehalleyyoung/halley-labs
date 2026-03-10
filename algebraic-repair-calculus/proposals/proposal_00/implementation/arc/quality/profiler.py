"""
Comprehensive data profiling for quality baselines.

:class:`DataProfiler` computes per-column statistical profiles,
detects anomalies against a baseline, computes a composite data-quality
score, and generates human-readable reports.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from arc.quality.distribution import DistributionAnalyzer
from arc.types.base import (
    Anomaly,
    ColumnProfile,
    TableProfile,
)

logger = logging.getLogger(__name__)


class DataProfiler:
    """Comprehensive data profiling for quality baselines.

    Parameters
    ----------
    analyzer:
        Distribution analyzer for statistical tests.
    """

    def __init__(self, analyzer: DistributionAnalyzer | None = None) -> None:
        self._analyzer = analyzer or DistributionAnalyzer()

    # ── Table profiling ────────────────────────────────────────────────

    def profile_table(
        self,
        data: Any,
        table_name: str = "",
        sample_size: int | None = None,
    ) -> TableProfile:
        """Compute a comprehensive profile of the dataset.

        Parameters
        ----------
        data:
            The dataset (dict-of-arrays, DataFrame, or similar).
        table_name:
            Name of the table (for metadata).
        sample_size:
            If set, profile a random sample instead of the full dataset.

        Returns
        -------
        TableProfile
        """
        start = time.time()
        columns = _get_columns(data)
        total_rows = _data_len(data)

        # Sampling
        sampled = data
        actual_sample = None
        if sample_size is not None and sample_size < total_rows:
            indices = np.random.choice(total_rows, sample_size, replace=False)
            sampled = _sample_data(data, indices)
            actual_sample = sample_size

        # Profile each column
        col_profiles: dict[str, ColumnProfile] = {}
        for col in columns:
            try:
                profile = self.profile_column(sampled, col)
                col_profiles[col] = profile
            except Exception as exc:
                logger.warning("Failed to profile column %s: %s", col, exc)
                col_profiles[col] = ColumnProfile(column_name=col)

        # Estimate size
        size_bytes = 0
        for cp in col_profiles.values():
            if cp.mean is not None:
                size_bytes += total_rows * 8
            elif cp.dtype and "object" in cp.dtype:
                size_bytes += total_rows * 64
            else:
                size_bytes += total_rows * 8

        return TableProfile(
            table_name=table_name,
            row_count=total_rows,
            column_count=len(columns),
            column_profiles=col_profiles,
            size_bytes=size_bytes,
            profiled_at=start,
            sample_size=actual_sample,
        )

    # ── Column profiling ───────────────────────────────────────────────

    def profile_column(
        self,
        data: Any,
        column: str,
    ) -> ColumnProfile:
        """Compute a statistical profile for a single column.

        Delegates to :meth:`DistributionAnalyzer.compute_column_profile`.
        """
        return DistributionAnalyzer.compute_column_profile(data, column)

    # ── Anomaly detection ──────────────────────────────────────────────

    def detect_anomalies(
        self,
        profile: TableProfile,
        baseline_profile: TableProfile,
    ) -> list[Anomaly]:
        """Detect anomalies by comparing a profile against a baseline.

        Parameters
        ----------
        profile:
            The new profile.
        baseline_profile:
            The baseline (reference) profile.

        Returns
        -------
        list[Anomaly]
        """
        anomalies: list[Anomaly] = []

        # Row count anomaly
        if baseline_profile.row_count > 0:
            ratio = profile.row_count / baseline_profile.row_count
            if ratio < 0.5 or ratio > 2.0:
                anomalies.append(Anomaly(
                    column_name="__table__",
                    anomaly_type="row_count_change",
                    message=(
                        f"Row count changed from {baseline_profile.row_count} "
                        f"to {profile.row_count} ({ratio:.1f}x)"
                    ),
                    severity="warning" if 0.3 < ratio < 3.0 else "error",
                    old_value=baseline_profile.row_count,
                    new_value=profile.row_count,
                    score=abs(ratio - 1.0),
                ))

        # Column count anomaly
        if profile.column_count != baseline_profile.column_count:
            anomalies.append(Anomaly(
                column_name="__table__",
                anomaly_type="column_count_change",
                message=(
                    f"Column count changed from {baseline_profile.column_count} "
                    f"to {profile.column_count}"
                ),
                severity="warning",
                old_value=baseline_profile.column_count,
                new_value=profile.column_count,
                score=abs(profile.column_count - baseline_profile.column_count),
            ))

        # Per-column anomalies
        for col_name, new_cp in profile.column_profiles.items():
            old_cp = baseline_profile.column_profiles.get(col_name)
            if old_cp is None:
                anomalies.append(Anomaly(
                    column_name=col_name,
                    anomaly_type="new_column",
                    message=f"Column {col_name} is new (not in baseline)",
                    severity="info",
                ))
                continue

            diff = self._analyzer.compare_profiles(old_cp, new_cp)
            anomalies.extend(diff.anomalies)

        # Detect removed columns
        for col_name in baseline_profile.column_profiles:
            if col_name not in profile.column_profiles:
                anomalies.append(Anomaly(
                    column_name=col_name,
                    anomaly_type="removed_column",
                    message=f"Column {col_name} was removed",
                    severity="warning",
                ))

        return anomalies

    # ── Data quality score ─────────────────────────────────────────────

    def compute_data_quality_score(
        self,
        profile: TableProfile,
        constraints: list[dict[str, Any]] | None = None,
    ) -> float:
        """Compute a composite data quality score ∈ [0, 1].

        The score is based on:
        * Completeness (null rate across columns).
        * Uniqueness (where applicable).
        * Consistency (value distribution normality).

        Parameters
        ----------
        profile:
            The table profile to score.
        constraints:
            Optional list of constraint specs to evaluate.

        Returns
        -------
        float
            Quality score in [0, 1].
        """
        if not profile.column_profiles:
            return 1.0

        scores: list[float] = []

        # Completeness score
        completeness = profile.completeness_score
        scores.append(completeness)

        # Uniqueness score (for columns with high uniqueness)
        uniqueness_scores: list[float] = []
        for cp in profile.column_profiles.values():
            if cp.uniqueness > 0:
                uniqueness_scores.append(min(cp.uniqueness, 1.0))
        if uniqueness_scores:
            scores.append(sum(uniqueness_scores) / len(uniqueness_scores))

        # Value distribution score (penalise extreme skew)
        distribution_scores: list[float] = []
        for cp in profile.column_profiles.values():
            if cp.std is not None and cp.mean is not None:
                if cp.mean != 0:
                    cv = abs(cp.std / cp.mean)
                    # Lower CV is better; penalise CV > 3
                    dist_score = max(0.0, 1.0 - cv / 10.0)
                    distribution_scores.append(dist_score)
        if distribution_scores:
            scores.append(sum(distribution_scores) / len(distribution_scores))

        # Constraint satisfaction score
        if constraints:
            from arc.quality.constraints import ConstraintEngine
            engine = ConstraintEngine()
            # We can't evaluate here without data, so skip
            pass

        return sum(scores) / len(scores) if scores else 1.0

    # ── Reporting ──────────────────────────────────────────────────────

    def generate_profile_report(self, profile: TableProfile) -> str:
        """Generate a human-readable profile report.

        Parameters
        ----------
        profile:
            The table profile.

        Returns
        -------
        str
            Formatted report string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"Data Profile: {profile.table_name or 'unnamed'}")
        lines.append("=" * 60)
        lines.append(f"Rows:    {profile.row_count:,}")
        lines.append(f"Columns: {profile.column_count}")
        if profile.sample_size:
            lines.append(f"Sample:  {profile.sample_size:,}")
        lines.append(f"Quality: {self.compute_data_quality_score(profile):.2%}")
        lines.append("")

        for col_name in sorted(profile.column_profiles.keys()):
            cp = profile.column_profiles[col_name]
            lines.append(f"  Column: {col_name}")
            lines.append(f"    Type:       {cp.dtype}")
            lines.append(f"    Count:      {cp.count:,}")
            lines.append(f"    Nulls:      {cp.null_count:,} ({cp.null_rate:.1%})")
            lines.append(f"    Unique:     {cp.unique_count:,} ({cp.uniqueness:.1%})")

            if cp.mean is not None:
                lines.append(f"    Mean:       {cp.mean:.4f}")
                lines.append(f"    Std:        {cp.std:.4f}" if cp.std is not None else "")
                lines.append(f"    Min:        {cp.min_val}")
                lines.append(f"    Max:        {cp.max_val}")

                if cp.percentiles:
                    pcts = ", ".join(f"p{k}={v:.2f}" for k, v in sorted(cp.percentiles.items()))
                    lines.append(f"    Pctiles:    {pcts}")

            if cp.most_common:
                top3 = cp.most_common[:3]
                mc_str = ", ".join(f"{v!r}({c})" for v, c in top3)
                lines.append(f"    Top values: {mc_str}")

            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "DataProfiler()"


# ── Utilities ──────────────────────────────────────────────────────────

def _get_columns(data: Any) -> list[str]:
    if isinstance(data, dict):
        return list(data.keys())
    if hasattr(data, "columns"):
        return list(data.columns)
    return []


def _data_len(data: Any) -> int:
    if isinstance(data, dict):
        for v in data.values():
            return len(v)
        return 0
    if hasattr(data, "__len__"):
        return len(data)
    if hasattr(data, "shape"):
        return data.shape[0]
    return 0


def _sample_data(data: Any, indices: np.ndarray) -> Any:
    """Sample rows from a dataset by index."""
    if isinstance(data, dict):
        return {k: np.asarray(v)[indices] for k, v in data.items()}
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data
