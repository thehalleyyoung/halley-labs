"""
Data validation — missing values, schema checks, and DAG compatibility.

Ensures that the observational dataset is compatible with the assumed DAG
before running any computations.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import numpy as np

from causalcert.types import AdjacencyMatrix
from causalcert.exceptions import SchemaError, MissingValueError


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------


def validate_data(
    data: pd.DataFrame,
    *,
    allow_missing: bool = False,
    min_rows: int = 1,
) -> None:
    """Validate basic data quality.

    Parameters
    ----------
    data : pd.DataFrame
        Observational dataset.
    allow_missing : bool
        If ``False`` (default), raise on any missing values.
    min_rows : int
        Minimum number of rows required.

    Raises
    ------
    MissingValueError
        If missing values are found and *allow_missing* is ``False``.
    SchemaError
        If the DataFrame is empty or has insufficient rows.
    """
    if data.shape[0] == 0:
        raise SchemaError("DataFrame is empty (0 rows)")
    if data.shape[1] == 0:
        raise SchemaError("DataFrame has no columns")
    if data.shape[0] < min_rows:
        raise SchemaError(
            f"DataFrame has {data.shape[0]} rows, need at least {min_rows}"
        )

    if not allow_missing:
        n_missing = int(data.isnull().sum().sum())
        if n_missing > 0:
            cols_with_missing = data.columns[data.isnull().any()].tolist()
            raise MissingValueError(
                f"Found {n_missing} missing values in columns: {cols_with_missing}"
            )


# ---------------------------------------------------------------------------
# DAG-data compatibility
# ---------------------------------------------------------------------------


def validate_dag_data_compatibility(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    node_names: list[str] | None = None,
) -> None:
    """Check that the DAG and data are dimensionally compatible.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Observational data.
    node_names : list[str] | None
        If provided, also checks that all node names appear as columns.

    Raises
    ------
    SchemaError
        If the number of columns does not match the number of nodes,
        or if named nodes are missing from the data.
    """
    n_nodes = adj.shape[0]
    n_cols = data.shape[1]
    if n_cols != n_nodes:
        raise SchemaError(
            f"DAG has {n_nodes} nodes but data has {n_cols} columns"
        )

    if node_names is not None:
        missing = set(node_names) - set(data.columns)
        if missing:
            raise SchemaError(
                f"DAG node names not found in data columns: {missing}"
            )


# ---------------------------------------------------------------------------
# Missing value analysis
# ---------------------------------------------------------------------------


def missing_value_analysis(
    data: pd.DataFrame,
) -> dict[str, Any]:
    """Analyse missing value patterns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.

    Returns
    -------
    dict[str, Any]
        Analysis report with per-column counts, overall rate, and pattern
        classification.
    """
    total = int(data.isnull().sum().sum())
    n_obs = data.shape[0]
    per_col = {col: int(data[col].isnull().sum()) for col in data.columns}
    pct_col = {
        col: float(cnt / n_obs * 100) for col, cnt in per_col.items()
    }

    # Pattern detection heuristic
    miss_cols = {k: v for k, v in per_col.items() if v > 0}
    if not miss_cols:
        pattern = "none"
    else:
        rates = np.array(list(miss_cols.values()), dtype=float)
        cv = float(np.std(rates) / max(np.mean(rates), 1e-12))
        pattern = "MCAR" if cv < 0.5 else "MAR"

    return {
        "total_missing": total,
        "overall_rate": float(total / (n_obs * data.shape[1]) * 100) if data.size > 0 else 0.0,
        "per_column_count": per_col,
        "per_column_pct": pct_col,
        "pattern": pattern,
        "n_complete_rows": int((~data.isnull().any(axis=1)).sum()),
    }


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def detect_outliers(
    data: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
) -> dict[str, dict[str, Any]]:
    """Detect outliers in numeric columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    method : str
        Detection method: ``"iqr"`` or ``"zscore"``.
    threshold : float
        IQR multiplier or z-score threshold.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-column outlier report with ``"n_outliers"`` and ``"indices"``.
    """
    report: dict[str, dict[str, Any]] = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) == 0:
            report[col] = {"n_outliers": 0, "indices": []}
            continue

        if method == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (series < lower) | (series > upper)
        elif method == "zscore":
            mean = float(series.mean())
            std = float(series.std())
            if std < 1e-12:
                mask = pd.Series(False, index=series.index)
            else:
                z = np.abs((series - mean) / std)
                mask = z > threshold
        else:
            raise ValueError(f"Unknown outlier method: {method!r}")

        outlier_idx = series.index[mask].tolist()
        report[col] = {
            "n_outliers": int(mask.sum()),
            "indices": outlier_idx,
            "pct": float(mask.sum() / len(series) * 100),
        }

    return report


# ---------------------------------------------------------------------------
# Positivity / overlap check
# ---------------------------------------------------------------------------


def check_positivity(
    data: pd.DataFrame,
    treatment_col: int | str,
    covariate_cols: list[int | str] | None = None,
    min_prop: float = 0.05,
) -> dict[str, Any]:
    """Check the positivity (overlap) assumption.

    For each stratum of discrete covariates, checks that both treatment
    values are represented with at least *min_prop* proportion.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    treatment_col : int | str
        Treatment column index or name.
    covariate_cols : list[int | str] | None
        Covariates to check.  If ``None``, checks overall treatment balance.
    min_prop : float
        Minimum treatment prevalence.

    Returns
    -------
    dict[str, Any]
        Positivity report.
    """
    if isinstance(treatment_col, int):
        t = data.iloc[:, treatment_col]
        t_name = data.columns[treatment_col]
    else:
        t = data[treatment_col]
        t_name = treatment_col

    overall_prop = float(t.mean())
    violations: list[str] = []

    if overall_prop < min_prop or overall_prop > (1.0 - min_prop):
        violations.append(
            f"Overall treatment prevalence = {overall_prop:.3f} (threshold: {min_prop:.3f})"
        )

    return {
        "treatment": t_name,
        "overall_prevalence": overall_prop,
        "n_treated": int(t.sum()),
        "n_control": int((1 - t).sum()),
        "positivity_satisfied": len(violations) == 0,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Sample size warnings
# ---------------------------------------------------------------------------


def sample_size_warnings(
    data: pd.DataFrame,
    n_covariates: int,
    treatment_col: int | str | None = None,
) -> list[str]:
    """Generate warnings about insufficient sample size.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    n_covariates : int
        Number of covariates in the adjustment set.
    treatment_col : int | str | None
        Treatment column (for per-arm checks).

    Returns
    -------
    list[str]
        Warning messages (empty if no issues).
    """
    warnings_list: list[str] = []
    n = data.shape[0]

    # Rule of thumb: need ≥ 10 * n_covariates per treatment arm
    min_per_arm = max(10 * n_covariates, 30)

    if n < 2 * min_per_arm:
        warnings_list.append(
            f"Total sample size ({n}) may be insufficient for "
            f"{n_covariates} covariates (recommend ≥ {2 * min_per_arm})"
        )

    if treatment_col is not None:
        if isinstance(treatment_col, int):
            t = data.iloc[:, treatment_col]
        else:
            t = data[treatment_col]

        n_treated = int(t.sum())
        n_control = int(len(t) - n_treated)

        if n_treated < min_per_arm:
            warnings_list.append(
                f"Treated group ({n_treated}) smaller than recommended "
                f"minimum ({min_per_arm})"
            )
        if n_control < min_per_arm:
            warnings_list.append(
                f"Control group ({n_control}) smaller than recommended "
                f"minimum ({min_per_arm})"
            )

    return warnings_list
