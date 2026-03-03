"""Data readers for the CPA engine.

Provides readers for loading multi-context observational data from
CSV files, numpy arrays, pandas DataFrames, and synthetic benchmark
generators.  All readers produce a :class:`MultiContextDataset`.
"""

from __future__ import annotations

import csv
import os
import re
import warnings
from pathlib import Path
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

from cpa.utils.logging import get_logger

logger = get_logger("io.readers")


# =====================================================================
# MultiContextDataset (canonical container)
# =====================================================================


class MultiContextDataset:
    """Unified container for multi-context observational data.

    Holds data matrices for K contexts with p shared variables.
    Each context may have a different number of samples.

    Parameters
    ----------
    context_data : dict of str → np.ndarray
        Mapping from context identifier to (n_i, p) data matrix.
    variable_names : list of str, optional
        Ordered variable names. Auto-generated if None.
    context_ids : list of str, optional
        Ordered context identifiers. Sorted keys if None.
    context_metadata : dict, optional
        Optional per-context metadata.

    Raises
    ------
    ValueError
        If context_data is empty.
    """

    def __init__(
        self,
        context_data: Dict[str, np.ndarray],
        variable_names: Optional[List[str]] = None,
        context_ids: Optional[List[str]] = None,
        context_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not context_data:
            raise ValueError("context_data must be non-empty")

        self.context_data = {k: np.asarray(v) for k, v in context_data.items()}
        self.context_ids = list(context_ids) if context_ids else sorted(
            self.context_data.keys()
        )

        first = self.context_data[self.context_ids[0]]
        p = first.shape[1] if first.ndim == 2 else 1

        if variable_names is not None:
            if len(variable_names) != p:
                raise ValueError(
                    f"variable_names has {len(variable_names)} entries "
                    f"but data has {p} columns"
                )
            self.variable_names = list(variable_names)
        else:
            self.variable_names = [f"X{i}" for i in range(p)]

        self.context_metadata = context_metadata or {}

    @property
    def n_contexts(self) -> int:
        """Number of contexts."""
        return len(self.context_ids)

    @property
    def n_variables(self) -> int:
        """Number of variables (columns)."""
        return len(self.variable_names)

    def get_data(self, context_id: str) -> np.ndarray:
        """Return (n, p) data matrix for a context."""
        return self.context_data[context_id]

    def sample_sizes(self) -> Dict[str, int]:
        """Sample sizes per context."""
        return {
            cid: d.shape[0] for cid, d in self.context_data.items()
        }

    def total_samples(self) -> int:
        """Total samples across all contexts."""
        return sum(d.shape[0] for d in self.context_data.values())

    def validate(self) -> List[str]:
        """Return list of validation error messages (empty if valid)."""
        return DataValidator.validate_dataset(self)

    def pooled_data(self) -> np.ndarray:
        """Concatenate all contexts into a single (N, p) matrix."""
        return np.vstack([
            self.context_data[cid] for cid in self.context_ids
        ])

    def context_labels(self) -> np.ndarray:
        """Return context label array matching pooled_data rows."""
        labels: List[str] = []
        for cid in self.context_ids:
            n = self.context_data[cid].shape[0]
            labels.extend([cid] * n)
        return np.array(labels)

    def subset_contexts(self, ids: Sequence[str]) -> "MultiContextDataset":
        """Return dataset with a subset of contexts."""
        return MultiContextDataset(
            context_data={c: self.context_data[c] for c in ids},
            variable_names=self.variable_names,
            context_ids=list(ids),
            context_metadata={
                c: self.context_metadata.get(c, {}) for c in ids
            },
        )

    def subset_variables(
        self, indices: Sequence[int]
    ) -> "MultiContextDataset":
        """Return dataset with a subset of variables."""
        idx = list(indices)
        return MultiContextDataset(
            context_data={
                c: d[:, idx] for c, d in self.context_data.items()
            },
            variable_names=[self.variable_names[i] for i in idx],
            context_ids=self.context_ids,
            context_metadata=self.context_metadata,
        )

    def standardize(self, per_context: bool = True) -> "MultiContextDataset":
        """Return z-scored dataset.

        Parameters
        ----------
        per_context : bool
            If True, standardize each context independently.
            If False, compute global mean/std from pooled data.

        Returns
        -------
        MultiContextDataset
        """
        if per_context:
            new_data = {}
            for cid, data in self.context_data.items():
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                std[std == 0] = 1.0
                new_data[cid] = (data - mean) / std
        else:
            pooled = self.pooled_data()
            mean = np.mean(pooled, axis=0)
            std = np.std(pooled, axis=0)
            std[std == 0] = 1.0
            new_data = {
                cid: (data - mean) / std
                for cid, data in self.context_data.items()
            }

        return MultiContextDataset(
            context_data=new_data,
            variable_names=self.variable_names,
            context_ids=self.context_ids,
            context_metadata=self.context_metadata,
        )

    def __repr__(self) -> str:
        sizes = self.sample_sizes()
        n_range = f"{min(sizes.values())}–{max(sizes.values())}" if sizes else "0"
        return (
            f"MultiContextDataset(K={self.n_contexts}, "
            f"p={self.n_variables}, n={n_range})"
        )


# =====================================================================
# Data validation
# =====================================================================


class DataValidator:
    """Validate multi-context datasets for CPA pipeline compatibility."""

    @staticmethod
    def validate_dataset(dataset: MultiContextDataset) -> List[str]:
        """Validate a MultiContextDataset.

        Checks for:
        - Non-empty data
        - Consistent number of variables
        - Minimum sample sizes
        - NaN and Inf values
        - Constant columns

        Parameters
        ----------
        dataset : MultiContextDataset
            Dataset to validate.

        Returns
        -------
        list of str
            Validation error messages (empty if valid).
        """
        errors: List[str] = []
        p = dataset.n_variables

        if dataset.n_contexts == 0:
            errors.append("No contexts in dataset")
            return errors

        for cid in dataset.context_ids:
            if cid not in dataset.context_data:
                errors.append(
                    f"Context {cid!r} listed but not found in context_data"
                )
                continue

            data = dataset.context_data[cid]

            if data.ndim != 2:
                errors.append(
                    f"Context {cid!r}: expected 2D array, got shape {data.shape}"
                )
                continue

            if data.shape[1] != p:
                errors.append(
                    f"Context {cid!r}: expected {p} columns, got {data.shape[1]}"
                )

            if data.shape[0] < 2:
                errors.append(
                    f"Context {cid!r}: need >= 2 samples, got {data.shape[0]}"
                )

            n_nan = int(np.sum(np.isnan(data)))
            if n_nan > 0:
                errors.append(f"Context {cid!r}: {n_nan} NaN values")

            n_inf = int(np.sum(np.isinf(data)))
            if n_inf > 0:
                errors.append(f"Context {cid!r}: {n_inf} Inf values")

        return errors

    @staticmethod
    def validate_and_clean(
        dataset: MultiContextDataset,
        drop_nan: bool = True,
        drop_constant: bool = False,
        min_samples: int = 2,
    ) -> Tuple[MultiContextDataset, List[str]]:
        """Validate and optionally clean a dataset.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input dataset.
        drop_nan : bool
            Drop rows with NaN values.
        drop_constant : bool
            Drop constant (zero-variance) columns.
        min_samples : int
            Minimum samples per context.

        Returns
        -------
        dataset : MultiContextDataset
            Cleaned dataset.
        warnings : list of str
            Warning messages about changes made.
        """
        warns: List[str] = []
        new_data: Dict[str, np.ndarray] = {}
        new_ids: List[str] = []

        for cid in dataset.context_ids:
            data = dataset.context_data[cid].copy()

            if drop_nan and np.any(np.isnan(data)):
                mask = ~np.any(np.isnan(data), axis=1)
                n_before = data.shape[0]
                data = data[mask]
                n_dropped = n_before - data.shape[0]
                if n_dropped > 0:
                    warns.append(
                        f"Context {cid!r}: dropped {n_dropped} rows with NaN"
                    )

            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            if data.shape[0] >= min_samples:
                new_data[cid] = data
                new_ids.append(cid)
            else:
                warns.append(
                    f"Context {cid!r}: dropped (only {data.shape[0]} samples)"
                )

        if not new_data:
            raise ValueError("No contexts remain after cleaning")

        variable_names = list(dataset.variable_names)
        keep_cols: Optional[List[int]] = None

        if drop_constant:
            pooled = np.vstack(list(new_data.values()))
            col_std = np.std(pooled, axis=0)
            keep_cols = [i for i in range(len(variable_names)) if col_std[i] > 1e-10]

            n_dropped = len(variable_names) - len(keep_cols)
            if n_dropped > 0:
                warns.append(f"Dropped {n_dropped} constant columns")
                variable_names = [variable_names[i] for i in keep_cols]
                new_data = {
                    cid: data[:, keep_cols] for cid, data in new_data.items()
                }

        return (
            MultiContextDataset(
                context_data=new_data,
                variable_names=variable_names,
                context_ids=new_ids,
                context_metadata=dataset.context_metadata,
            ),
            warns,
        )

    @staticmethod
    def check_sample_sizes(
        dataset: MultiContextDataset,
        min_n: int = 30,
        warn_ratio: float = 5.0,
    ) -> List[str]:
        """Check sample size adequacy.

        Parameters
        ----------
        dataset : MultiContextDataset
            Dataset to check.
        min_n : int
            Minimum recommended sample size.
        warn_ratio : float
            Warn if max/min sample ratio exceeds this.

        Returns
        -------
        list of str
            Warning messages.
        """
        warns: List[str] = []
        sizes = dataset.sample_sizes()
        p = dataset.n_variables

        for cid, n in sizes.items():
            if n < min_n:
                warns.append(
                    f"Context {cid!r}: n={n} < {min_n} (recommended minimum)"
                )
            if n < p + 2:
                warns.append(
                    f"Context {cid!r}: n={n} < p+2={p + 2} "
                    "(underdetermined for correlation-based methods)"
                )

        if len(sizes) >= 2:
            ratio = max(sizes.values()) / max(min(sizes.values()), 1)
            if ratio > warn_ratio:
                warns.append(
                    f"Sample size ratio {ratio:.1f} exceeds {warn_ratio:.1f}; "
                    "consider balancing or adjusting significance levels"
                )

        return warns


# =====================================================================
# CSVReader
# =====================================================================


class CSVReader:
    """Read multi-context data from CSV files.

    Supports two layouts:
    1. **Directory of CSVs**: one file per context in a directory.
    2. **Single CSV with context column**: one file with a column
       identifying the context for each row.

    Parameters
    ----------
    path : str or Path
        Path to CSV file or directory of CSV files.
    context_column : str, optional
        Column name that identifies the context (single-file mode).
    delimiter : str
        CSV delimiter character.
    has_header : bool
        Whether CSV files have a header row.
    variable_columns : list of str, optional
        Specific columns to load (None = all non-context columns).
    dtype : np.dtype, optional
        Data type for loaded arrays.

    Examples
    --------
    >>> reader = CSVReader("data/contexts/")
    >>> dataset = reader.read()

    >>> reader = CSVReader("data/all.csv", context_column="environment")
    >>> dataset = reader.read()
    """

    def __init__(
        self,
        path: Union[str, Path],
        context_column: Optional[str] = None,
        delimiter: str = ",",
        has_header: bool = True,
        variable_columns: Optional[List[str]] = None,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        self._path = Path(path)
        self._context_column = context_column
        self._delimiter = delimiter
        self._has_header = has_header
        self._variable_columns = variable_columns
        self._dtype = dtype or np.float64

    def read(self) -> MultiContextDataset:
        """Read and return a MultiContextDataset.

        Returns
        -------
        MultiContextDataset

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        ValueError
            If data format is invalid.
        """
        if self._path.is_dir():
            return self._read_directory()
        elif self._path.is_file():
            return self._read_single_file()
        else:
            raise FileNotFoundError(f"Path not found: {self._path}")

    def _read_directory(self) -> MultiContextDataset:
        """Read one CSV file per context from a directory."""
        csv_files = sorted(self._path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self._path}")

        context_data: Dict[str, np.ndarray] = {}
        variable_names: Optional[List[str]] = None

        for csv_file in csv_files:
            context_id = csv_file.stem

            data, header = self._load_csv_file(csv_file)

            if variable_names is None:
                if header is not None:
                    cols = header
                    if self._variable_columns:
                        cols = self._variable_columns
                    variable_names = cols
                else:
                    variable_names = [f"X{i}" for i in range(data.shape[1])]

            if self._variable_columns and header:
                col_indices = [
                    header.index(c) for c in self._variable_columns
                    if c in header
                ]
                data = data[:, col_indices]

            context_data[context_id] = data

        return MultiContextDataset(
            context_data=context_data,
            variable_names=variable_names,
        )

    def _read_single_file(self) -> MultiContextDataset:
        """Read a single CSV with a context column."""
        data, header = self._load_csv_file(self._path)

        if self._context_column is None:
            raise ValueError(
                "context_column must be specified for single-file mode"
            )

        if header is None:
            raise ValueError(
                "Header row required when using context_column"
            )

        if self._context_column not in header:
            raise ValueError(
                f"Context column {self._context_column!r} not found. "
                f"Available: {header}"
            )

        ctx_col_idx = header.index(self._context_column)

        raw_data, raw_header = self._load_csv_file(
            self._path, return_strings=True
        )

        context_labels = raw_data[:, ctx_col_idx]
        value_cols = [
            i for i in range(len(header)) if i != ctx_col_idx
        ]

        if self._variable_columns:
            value_cols = [
                header.index(c) for c in self._variable_columns
                if c in header and header.index(c) != ctx_col_idx
            ]

        var_names = [header[i] for i in value_cols]

        numeric_data = np.zeros(
            (raw_data.shape[0], len(value_cols)), dtype=self._dtype
        )
        for out_i, col_i in enumerate(value_cols):
            for row in range(raw_data.shape[0]):
                try:
                    numeric_data[row, out_i] = float(raw_data[row, col_i])
                except (ValueError, TypeError):
                    numeric_data[row, out_i] = np.nan

        unique_contexts = sorted(set(context_labels))
        context_data: Dict[str, np.ndarray] = {}
        for ctx in unique_contexts:
            mask = context_labels == ctx
            context_data[str(ctx)] = numeric_data[mask]

        return MultiContextDataset(
            context_data=context_data,
            variable_names=var_names,
        )

    def _load_csv_file(
        self, path: Path, return_strings: bool = False
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Load a CSV file into an array.

        Parameters
        ----------
        path : Path
            File path.
        return_strings : bool
            If True, return string array instead of float.

        Returns
        -------
        data : np.ndarray
            Data array.
        header : list of str or None
            Column names if header present.
        """
        header: Optional[List[str]] = None
        rows: List[List[str]] = []

        with open(path, "r", newline="") as f:
            reader = csv.reader(f, delimiter=self._delimiter)

            if self._has_header:
                header_row = next(reader, None)
                if header_row:
                    header = [h.strip() for h in header_row]

            for row in reader:
                if row:
                    rows.append([v.strip() for v in row])

        if not rows:
            raise ValueError(f"No data rows in {path}")

        if return_strings:
            return np.array(rows, dtype=str), header

        n_cols = len(rows[0])
        data = np.zeros((len(rows), n_cols), dtype=self._dtype)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                if j < n_cols:
                    try:
                        data[i, j] = float(val)
                    except (ValueError, TypeError):
                        data[i, j] = np.nan

        return data, header


# =====================================================================
# NumpyReader
# =====================================================================


class NumpyReader:
    """Read multi-context data from numpy arrays or .npz files.

    Parameters
    ----------
    source : dict, str, or Path
        Either a dict of context_id → np.ndarray, or path to .npz file.
    variable_names : list of str, optional
        Variable names.
    context_ids : list of str, optional
        Context identifiers.

    Examples
    --------
    >>> reader = NumpyReader({"ctx0": X0, "ctx1": X1})
    >>> dataset = reader.read()

    >>> reader = NumpyReader("data.npz")
    >>> dataset = reader.read()
    """

    def __init__(
        self,
        source: Union[Dict[str, np.ndarray], str, Path],
        variable_names: Optional[List[str]] = None,
        context_ids: Optional[List[str]] = None,
    ) -> None:
        self._source = source
        self._variable_names = variable_names
        self._context_ids = context_ids

    def read(self) -> MultiContextDataset:
        """Read and return a MultiContextDataset."""
        if isinstance(self._source, dict):
            context_data = {
                k: np.asarray(v, dtype=np.float64)
                for k, v in self._source.items()
            }
        elif isinstance(self._source, (str, Path)):
            path = Path(self._source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if path.suffix == ".npz":
                npz = np.load(path, allow_pickle=False)
                context_data = {key: npz[key] for key in npz.files}
            elif path.suffix == ".npy":
                arr = np.load(path, allow_pickle=False)
                context_data = {"context_0": arr}
            else:
                raise ValueError(
                    f"Unsupported numpy file format: {path.suffix}"
                )
        else:
            raise TypeError(
                f"source must be dict, str, or Path, got {type(self._source)}"
            )

        return MultiContextDataset(
            context_data=context_data,
            variable_names=self._variable_names,
            context_ids=self._context_ids,
        )


# =====================================================================
# PandasReader
# =====================================================================


class PandasReader:
    """Read multi-context data from pandas DataFrames.

    Parameters
    ----------
    source : DataFrame or dict of str → DataFrame
        Single DataFrame with context column, or dict of DataFrames.
    context_column : str, optional
        Column identifying contexts (single DataFrame mode).
    variable_columns : list of str, optional
        Columns to use as variables (None = all numeric columns).

    Examples
    --------
    >>> reader = PandasReader(df, context_column="group")
    >>> dataset = reader.read()

    >>> reader = PandasReader({"g1": df1, "g2": df2})
    >>> dataset = reader.read()
    """

    def __init__(
        self,
        source: Any,
        context_column: Optional[str] = None,
        variable_columns: Optional[List[str]] = None,
    ) -> None:
        self._source = source
        self._context_column = context_column
        self._variable_columns = variable_columns

    def read(self) -> MultiContextDataset:
        """Read and return a MultiContextDataset."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for PandasReader: pip install pandas"
            ) from exc

        if isinstance(self._source, dict):
            return self._read_dict_of_frames()
        elif isinstance(self._source, pd.DataFrame):
            return self._read_single_frame()
        else:
            raise TypeError(
                f"source must be DataFrame or dict, got {type(self._source)}"
            )

    def _read_dict_of_frames(self) -> MultiContextDataset:
        """Read from dict of DataFrames."""
        import pandas as pd

        context_data: Dict[str, np.ndarray] = {}
        variable_names: Optional[List[str]] = None

        for ctx_id, df in self._source.items():
            if self._variable_columns:
                cols = self._variable_columns
            else:
                cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if variable_names is None:
                variable_names = list(cols)

            context_data[str(ctx_id)] = df[cols].values.astype(np.float64)

        return MultiContextDataset(
            context_data=context_data,
            variable_names=variable_names,
        )

    def _read_single_frame(self) -> MultiContextDataset:
        """Read from a single DataFrame with context column."""
        import pandas as pd

        df = self._source

        if self._context_column is None:
            raise ValueError("context_column required for single DataFrame")
        if self._context_column not in df.columns:
            raise ValueError(
                f"Column {self._context_column!r} not in DataFrame"
            )

        if self._variable_columns:
            var_cols = self._variable_columns
        else:
            var_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != self._context_column
            ]

        variable_names = list(var_cols)
        context_data: Dict[str, np.ndarray] = {}

        for ctx_val, group in df.groupby(self._context_column):
            context_data[str(ctx_val)] = group[var_cols].values.astype(
                np.float64
            )

        return MultiContextDataset(
            context_data=context_data,
            variable_names=variable_names,
        )


# =====================================================================
# ParquetReader
# =====================================================================


class ParquetReader:
    """Read multi-context data from Parquet files.

    Supports two layouts:
    1. **Directory of Parquet files**: one file per context.
    2. **Single Parquet file with context column**: one file with a
       column identifying the context for each row.

    Parameters
    ----------
    path : str or Path
        Path to a ``.parquet`` / ``.pq`` file or a directory of them.
    context_column : str, optional
        Column name that identifies the context (single-file mode).
    variable_columns : list of str, optional
        Specific columns to load (None = all numeric columns).

    Notes
    -----
    Requires ``pyarrow`` **or** ``pandas`` with a Parquet engine.
    Users can create compatible files with ``df.to_parquet('data.parquet')``.

    Examples
    --------
    >>> reader = ParquetReader("data.parquet", context_column="env")
    >>> dataset = reader.read()

    >>> reader = ParquetReader("contexts/")
    >>> dataset = reader.read()
    """

    _EXTENSIONS = {".parquet", ".pq"}

    def __init__(
        self,
        path: Union[str, Path],
        context_column: Optional[str] = None,
        variable_columns: Optional[List[str]] = None,
    ) -> None:
        self._path = Path(path)
        self._context_column = context_column
        self._variable_columns = variable_columns

    def read(self) -> MultiContextDataset:
        """Read and return a MultiContextDataset.

        Returns
        -------
        MultiContextDataset

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        ImportError
            If neither pyarrow nor pandas is available.
        ValueError
            If data format is invalid.
        """
        self._ensure_backend()

        if self._path.is_dir():
            return self._read_directory()
        elif self._path.is_file():
            return self._read_single_file()
        else:
            raise FileNotFoundError(f"Path not found: {self._path}")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_backend() -> None:
        """Raise a helpful error if no Parquet backend is installed."""
        try:
            import pyarrow  # noqa: F401
            return
        except ImportError:
            pass
        try:
            import pandas  # noqa: F401
            return
        except ImportError:
            pass
        raise ImportError(
            "Parquet support requires pyarrow or pandas. "
            "Install one with:  pip install pyarrow"
        )

    @staticmethod
    def _read_parquet_file(path: Path) -> "pd.DataFrame":
        """Read a single Parquet file into a pandas DataFrame."""
        import pandas as pd
        return pd.read_parquet(path)

    def _read_directory(self) -> MultiContextDataset:
        """Read one Parquet file per context from a directory."""
        pq_files = sorted(
            f for f in self._path.iterdir()
            if f.suffix in self._EXTENSIONS
        )
        if not pq_files:
            raise ValueError(
                f"No Parquet files (.parquet/.pq) found in {self._path}"
            )

        context_data: Dict[str, np.ndarray] = {}
        variable_names: Optional[List[str]] = None

        for pq_file in pq_files:
            df = self._read_parquet_file(pq_file)
            context_id = pq_file.stem

            if self._variable_columns:
                cols = self._variable_columns
            else:
                cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if variable_names is None:
                variable_names = list(cols)

            context_data[context_id] = df[cols].values.astype(np.float64)

        return MultiContextDataset(
            context_data=context_data,
            variable_names=variable_names,
        )

    def _read_single_file(self) -> MultiContextDataset:
        """Read a single Parquet file, optionally split by context column."""
        df = self._read_parquet_file(self._path)

        if self._context_column is None:
            # Treat entire file as one context
            if self._variable_columns:
                cols = self._variable_columns
            else:
                cols = df.select_dtypes(include=[np.number]).columns.tolist()

            return MultiContextDataset(
                context_data={"context_0": df[cols].values.astype(np.float64)},
                variable_names=list(cols),
            )

        if self._context_column not in df.columns:
            raise ValueError(
                f"Context column {self._context_column!r} not found. "
                f"Available: {list(df.columns)}"
            )

        if self._variable_columns:
            var_cols = self._variable_columns
        else:
            var_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != self._context_column
            ]

        context_data: Dict[str, np.ndarray] = {}
        for ctx_val, group in df.groupby(self._context_column):
            context_data[str(ctx_val)] = group[var_cols].values.astype(
                np.float64
            )

        return MultiContextDataset(
            context_data=context_data,
            variable_names=list(var_cols),
        )


# =====================================================================
# SyntheticReader
# =====================================================================


class SyntheticReader:
    """Read from synthetic benchmark generators.

    Wraps benchmark generators to produce MultiContextDataset instances
    for testing and evaluation.

    Parameters
    ----------
    generator : str or callable
        Generator name ('fsvp', 'csvm', 'tps') or callable.
    **kwargs
        Keyword arguments passed to the generator.

    Examples
    --------
    >>> reader = SyntheticReader("fsvp", p=5, K=3, n=200, seed=42)
    >>> dataset = reader.read()
    """

    _BUILTIN_GENERATORS = {"fsvp", "csvm", "tps"}

    def __init__(
        self,
        generator: Union[str, Callable[..., MultiContextDataset]],
        **kwargs: Any,
    ) -> None:
        self._generator = generator
        self._kwargs = kwargs

    def read(self) -> MultiContextDataset:
        """Generate and return a MultiContextDataset."""
        if callable(self._generator):
            result = self._generator(**self._kwargs)
            if isinstance(result, MultiContextDataset):
                return result
            if isinstance(result, dict):
                return MultiContextDataset(context_data=result)
            raise TypeError(
                f"Generator returned {type(result)}, expected "
                "MultiContextDataset or dict"
            )

        if isinstance(self._generator, str):
            name = self._generator.lower()
            if name in self._BUILTIN_GENERATORS:
                return self._run_builtin(name)
            raise ValueError(
                f"Unknown generator: {name!r}. "
                f"Available: {self._BUILTIN_GENERATORS}"
            )

        raise TypeError(
            f"generator must be str or callable, got {type(self._generator)}"
        )

    def _run_builtin(self, name: str) -> MultiContextDataset:
        """Run a built-in generator."""
        try:
            from benchmarks.generators import (
                FSVPGenerator,
                CSVMGenerator,
                TPSGenerator,
            )

            gen_map = {
                "fsvp": FSVPGenerator,
                "csvm": CSVMGenerator,
                "tps": TPSGenerator,
            }
            gen_cls = gen_map[name]
            gen = gen_cls(**self._kwargs)
            result = gen.generate()

            if isinstance(result, MultiContextDataset):
                return result
            if isinstance(result, tuple) and len(result) >= 1:
                data = result[0]
                if isinstance(data, dict):
                    return MultiContextDataset(context_data=data)

            raise TypeError(f"Generator returned unexpected type: {type(result)}")

        except ImportError:
            return self._simple_synthetic(**self._kwargs)

    def _simple_synthetic(self, **kwargs: Any) -> MultiContextDataset:
        """Simple built-in synthetic data generation."""
        p = kwargs.get("p", 5)
        K = kwargs.get("K", 3)
        n = kwargs.get("n", 200)
        seed = kwargs.get("seed", 42)

        rng = np.random.RandomState(seed)

        context_data: Dict[str, np.ndarray] = {}
        for k in range(K):
            data = rng.randn(n, p)
            shift = rng.uniform(-1, 1, p) * (k / max(K - 1, 1))
            data += shift
            context_data[f"context_{k}"] = data

        variable_names = [f"X{i}" for i in range(p)]

        return MultiContextDataset(
            context_data=context_data,
            variable_names=variable_names,
        )
