"""
Data preprocessing — standardisation and categorical encoding.

Transforms raw data into a form suitable for CI testing and causal
estimation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causalcert.types import VariableType


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------


def standardize(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "zscore",
) -> pd.DataFrame:
    """Standardise continuous columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to standardise.  ``None`` for all numeric columns.
    method : str
        ``"zscore"`` (default) or ``"minmax"``.

    Returns
    -------
    pd.DataFrame
        Copy with standardised columns.
    """
    df = data.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype(np.float64)

        if method == "zscore":
            mean = series.mean()
            std = series.std()
            if std > 1e-12:
                df[col] = (series - mean) / std
            else:
                df[col] = 0.0
        elif method == "minmax":
            mn = series.min()
            mx = series.max()
            rng = mx - mn
            if rng > 1e-12:
                df[col] = (series - mn) / rng
            else:
                df[col] = 0.0
        else:
            raise ValueError(f"Unknown standardisation method: {method!r}")

    return df


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def encode_categorical(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "onehot",
    drop_first: bool = True,
) -> pd.DataFrame:
    """Encode categorical columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to encode.  ``None`` infers from dtype.
    method : str
        Encoding method: ``"onehot"`` or ``"ordinal"``.
    drop_first : bool
        If ``True`` and method is ``"onehot"``, drop the first level to
        avoid multicollinearity.

    Returns
    -------
    pd.DataFrame
        Encoded data (may have more columns if one-hot).
    """
    df = data.copy()
    if columns is None:
        columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    if not columns:
        return df

    if method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=float)
    elif method == "ordinal":
        for col in columns:
            if col not in df.columns:
                continue
            cats = sorted(df[col].dropna().unique())
            mapping = {v: i for i, v in enumerate(cats)}
            df[col] = df[col].map(mapping)
    else:
        raise ValueError(f"Unknown encoding method: {method!r}")

    return df


# ---------------------------------------------------------------------------
# Missing value imputation
# ---------------------------------------------------------------------------


def impute_missing(
    data: pd.DataFrame,
    method: str = "mean",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Impute missing values.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    method : str
        ``"mean"``, ``"median"``, or ``"mice"`` (iterative imputation).
    columns : list[str] | None
        Columns to impute.  ``None`` for all columns with missing values.

    Returns
    -------
    pd.DataFrame
        Data with missing values imputed.
    """
    df = data.copy()
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()

    if not columns:
        return df

    if method == "mean":
        for col in columns:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
    elif method == "median":
        for col in columns:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
    elif method == "mice":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            imp = IterativeImputer(max_iter=10, random_state=42)
            df[numeric_cols] = imp.fit_transform(df[numeric_cols])
        # Non-numeric: fill with mode
        non_numeric = [c for c in columns if c not in numeric_cols]
        for col in non_numeric:
            if col in df.columns:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
                df[col] = df[col].fillna(mode_val)
    else:
        raise ValueError(f"Unknown imputation method: {method!r}")

    return df


# ---------------------------------------------------------------------------
# Winsorisation
# ---------------------------------------------------------------------------


def winsorize(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorise outliers by clipping to quantile boundaries.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to winsorise.  ``None`` for all numeric columns.
    lower : float
        Lower quantile (values below are clipped).
    upper : float
        Upper quantile (values above are clipped).

    Returns
    -------
    pd.DataFrame
    """
    df = data.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        q_low = float(df[col].quantile(lower))
        q_high = float(df[col].quantile(upper))
        df[col] = df[col].clip(q_low, q_high)

    return df


# ---------------------------------------------------------------------------
# Variable selection helpers
# ---------------------------------------------------------------------------


def select_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return names of numeric columns."""
    return data.select_dtypes(include=[np.number]).columns.tolist()


def select_categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return names of categorical/object columns."""
    return data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def drop_constant_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with zero variance (constant values)."""
    nunique = data.nunique()
    to_drop = nunique[nunique <= 1].index.tolist()
    return data.drop(columns=to_drop)


def drop_high_cardinality(
    data: pd.DataFrame,
    threshold: int = 50,
) -> pd.DataFrame:
    """Drop categorical columns with cardinality above *threshold*."""
    cat_cols = select_categorical_columns(data)
    to_drop = [c for c in cat_cols if data[c].nunique() > threshold]
    return data.drop(columns=to_drop)


# ---------------------------------------------------------------------------
# Combined preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess(
    data: pd.DataFrame,
    *,
    impute: str | None = "mean",
    standardize_numeric: bool = True,
    encode_categoricals: bool = True,
    winsorize_numeric: bool = False,
    drop_constants: bool = True,
    standardize_method: str = "zscore",
) -> pd.DataFrame:
    """Full preprocessing pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data.
    impute : str | None
        Imputation method.  ``None`` to skip.
    standardize_numeric : bool
        Whether to standardise numeric columns.
    encode_categoricals : bool
        Whether to encode categorical columns.
    winsorize_numeric : bool
        Whether to winsorise numeric outliers.
    drop_constants : bool
        Whether to drop constant columns.
    standardize_method : str
        Standardisation method.

    Returns
    -------
    pd.DataFrame
        Preprocessed data.
    """
    df = data.copy()

    if drop_constants:
        df = drop_constant_columns(df)

    if impute is not None:
        df = impute_missing(df, method=impute)

    if winsorize_numeric:
        df = winsorize(df)

    if encode_categoricals:
        df = encode_categorical(df)

    if standardize_numeric:
        df = standardize(df, method=standardize_method)

    return df
