"""
Variable type inference from data.

Heuristically classifies DataFrame columns as continuous, ordinal, nominal,
or binary based on value counts, dtype, and cardinality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causalcert.types import VariableType


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def infer_variable_types(
    data: pd.DataFrame,
    *,
    max_nominal_cardinality: int = 10,
    overrides: dict[str, VariableType] | None = None,
) -> dict[str, VariableType]:
    """Infer the statistical type of each column.

    Heuristic rules:
    1. If the column has exactly 2 unique non-null values → ``BINARY``.
    2. If the column is categorical, object, or string dtype → ``NOMINAL``.
    3. If the column is integer and unique values ≤ max_nominal_cardinality → ``ORDINAL``.
    4. If the column is integer with > max_nominal_cardinality unique values → ``CONTINUOUS``.
    5. If the column is float → ``CONTINUOUS``.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    max_nominal_cardinality : int
        Maximum number of unique values for a column to be classified
        as nominal (otherwise treated as continuous).
    overrides : dict[str, VariableType] | None
        Manual type overrides.  Keys are column names.

    Returns
    -------
    dict[str, VariableType]
        Mapping from column name to inferred type.
    """
    result: dict[str, VariableType] = {}
    overrides = overrides or {}

    for col in data.columns:
        if col in overrides:
            result[col] = overrides[col]
            continue

        series = data[col]
        result[col] = _infer_single(series, max_nominal_cardinality)

    return result


def _infer_single(
    series: pd.Series,
    max_nominal_cardinality: int,
) -> VariableType:
    """Infer the type of a single Series."""
    non_null = series.dropna()
    n_unique = non_null.nunique()

    # Binary check
    if n_unique == 2:
        return VariableType.BINARY

    # Boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return VariableType.BINARY

    # Categorical or string types
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        if n_unique <= max_nominal_cardinality:
            return VariableType.NOMINAL
        return VariableType.NOMINAL

    # Numeric types
    if pd.api.types.is_float_dtype(series):
        # Could be ordinal if few unique values
        if n_unique <= max_nominal_cardinality:
            # Check if values look like integers
            if non_null.apply(lambda x: float(x).is_integer()).all():
                return VariableType.ORDINAL
        return VariableType.CONTINUOUS

    if pd.api.types.is_integer_dtype(series):
        if n_unique <= max_nominal_cardinality:
            return VariableType.ORDINAL
        return VariableType.CONTINUOUS

    return VariableType.NOMINAL


def is_binary(series: pd.Series) -> bool:
    """Check whether a Series contains exactly two unique non-null values.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    bool
    """
    return series.dropna().nunique() == 2


# ---------------------------------------------------------------------------
# Type constraints from DAG
# ---------------------------------------------------------------------------


def refine_types_from_dag(
    types: dict[str, VariableType],
    adj: np.ndarray,
    node_names: list[str],
) -> dict[str, VariableType]:
    """Refine inferred types using DAG structure hints.

    If a variable has binary children (according to the DAG), it is more
    likely to be binary or categorical itself if it has few unique values.

    Parameters
    ----------
    types : dict[str, VariableType]
        Current type assignments.
    adj : np.ndarray
        DAG adjacency matrix.
    node_names : list[str]
        Node names corresponding to matrix indices.

    Returns
    -------
    dict[str, VariableType]
        Refined type assignments.
    """
    refined = dict(types)
    n = adj.shape[0]

    for i in range(n):
        name = node_names[i]
        if refined.get(name) in (VariableType.BINARY, VariableType.NOMINAL):
            continue

        # Check if all children are binary
        children = np.nonzero(adj[i])[0]
        if len(children) == 0:
            continue

        all_binary = all(
            refined.get(node_names[int(c)]) == VariableType.BINARY
            for c in children
        )
        # If current variable is ordinal with few values and all children
        # are binary, it might be nominal
        if all_binary and refined.get(name) == VariableType.ORDINAL:
            refined[name] = VariableType.NOMINAL

    return refined


# ---------------------------------------------------------------------------
# Summary statistics per type
# ---------------------------------------------------------------------------


def type_summary(
    data: pd.DataFrame,
    types: dict[str, VariableType],
) -> dict[str, dict[str, Any]]:
    """Compute summary statistics appropriate for each variable type.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    types : dict[str, VariableType]
        Variable type assignments.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-column summaries.
    """
    summaries: dict[str, dict[str, Any]] = {}

    for col, vtype in types.items():
        if col not in data.columns:
            continue
        series = data[col]

        if vtype == VariableType.CONTINUOUS:
            summaries[col] = {
                "type": "continuous",
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "n_missing": int(series.isnull().sum()),
                "skewness": float(series.skew()),
            }
        elif vtype == VariableType.BINARY:
            vals = series.dropna()
            summaries[col] = {
                "type": "binary",
                "values": sorted(vals.unique().tolist()),
                "prevalence": float(vals.value_counts(normalize=True).iloc[0]),
                "n_missing": int(series.isnull().sum()),
            }
        elif vtype in (VariableType.ORDINAL, VariableType.NOMINAL):
            summaries[col] = {
                "type": vtype.value,
                "n_unique": int(series.nunique()),
                "mode": series.mode().iloc[0] if not series.mode().empty else None,
                "value_counts": series.value_counts().to_dict(),
                "n_missing": int(series.isnull().sum()),
            }

    return summaries
