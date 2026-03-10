"""
CSV and Parquet data loading with schema validation.

Provides a unified interface for loading observational datasets from
CSV and Parquet files with optional column selection and type coercion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from causalcert.exceptions import DataError, SchemaError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_csv(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    dtype: dict[str, str] | None = None,
    na_values: list[str] | None = None,
    nrows: int | None = None,
    delimiter: str = ",",
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    columns : list[str] | None
        Subset of columns to load.
    dtype : dict[str, str] | None
        Column type overrides.
    na_values : list[str] | None
        Additional strings to recognise as NA.
    nrows : int | None
        Maximum number of rows to read.
    delimiter : str
        Column delimiter.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    DataError
        If the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    if not path.suffix.lower() in (".csv", ".tsv", ".txt"):
        logger.warning("File %s does not have a CSV extension", path)

    try:
        df = pd.read_csv(
            path,
            usecols=columns,
            dtype=dtype,
            na_values=na_values,
            nrows=nrows,
            delimiter=delimiter,
        )
    except Exception as exc:
        raise DataError(f"Failed to read CSV {path}: {exc}") from exc

    logger.info("Loaded CSV %s: %d rows × %d columns", path, *df.shape)
    return df


def load_parquet(
    path: str | Path,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the Parquet file.
    columns : list[str] | None
        Subset of columns to load.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    DataError
        If the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")

    try:
        df = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise DataError(f"Failed to read Parquet {path}: {exc}") from exc

    logger.info("Loaded Parquet %s: %d rows × %d columns", path, *df.shape)
    return df


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def load_auto(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Auto-detect file format and load.

    Dispatches to :func:`load_csv` or :func:`load_parquet` based on the
    file extension.

    Parameters
    ----------
    path : str | Path
        File path.
    columns : list[str] | None
        Columns to load.
    dtype : dict[str, str] | None
        Column type overrides (CSV only).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"):
        return load_parquet(path, columns=columns)
    elif ext in (".csv", ".tsv", ".txt"):
        return load_csv(path, columns=columns, dtype=dtype)
    else:
        # Try CSV first, then Parquet
        try:
            return load_csv(path, columns=columns, dtype=dtype)
        except DataError:
            return load_parquet(path, columns=columns)


# ---------------------------------------------------------------------------
# Streaming for large files
# ---------------------------------------------------------------------------


def stream_csv(
    path: str | Path,
    *,
    chunksize: int = 10000,
    columns: list[str] | None = None,
    dtype: dict[str, str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream a CSV file in chunks for memory-efficient processing.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    chunksize : int
        Number of rows per chunk.
    columns : list[str] | None
        Columns to load.
    dtype : dict[str, str] | None
        Type overrides.

    Yields
    ------
    pd.DataFrame
        One chunk at a time.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")

    reader = pd.read_csv(path, chunksize=chunksize, usecols=columns, dtype=dtype)
    for chunk in reader:
        yield chunk


# ---------------------------------------------------------------------------
# Missing value analysis
# ---------------------------------------------------------------------------


def missing_value_report(data: pd.DataFrame) -> dict[str, Any]:
    """Analyse missing values in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.

    Returns
    -------
    dict[str, Any]
        Contains ``"total_missing"``, ``"per_column"`` (dict of counts),
        ``"pct_missing"`` (per column), and ``"pattern"``
        (``"MCAR"``, ``"MAR"``, or ``"unknown"``).
    """
    total = int(data.isnull().sum().sum())
    per_col = data.isnull().sum().to_dict()
    pct = (data.isnull().sum() / len(data) * 100).to_dict()

    # Simple MCAR heuristic: if missing values are roughly equally distributed
    # across all columns with missing data, guess MCAR
    miss_cols = {k: v for k, v in per_col.items() if v > 0}
    if not miss_cols:
        pattern = "none"
    elif len(miss_cols) == 1:
        pattern = "MAR"
    else:
        rates = list(miss_cols.values())
        cv = np.std(rates) / max(np.mean(rates), 1e-12)
        pattern = "MCAR" if cv < 0.5 else "MAR"

    return {
        "total_missing": total,
        "per_column": per_col,
        "pct_missing": pct,
        "pattern": pattern,
    }


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------


def select_columns(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
    numeric_only: bool = False,
) -> pd.DataFrame:
    """Select a subset of columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to keep.
    exclude : list[str] | None
        Columns to exclude (applied after *columns*).
    numeric_only : bool
        If ``True``, keep only numeric columns.

    Returns
    -------
    pd.DataFrame
    """
    df = data
    if columns is not None:
        missing = set(columns) - set(df.columns)
        if missing:
            raise SchemaError(f"Columns not found in data: {missing}")
        df = df[columns]
    if exclude is not None:
        df = df.drop(columns=[c for c in exclude if c in df.columns])
    if numeric_only:
        df = df.select_dtypes(include=[np.number])
    return df
