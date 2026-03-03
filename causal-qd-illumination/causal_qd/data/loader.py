"""Data loading and validation utilities for causal discovery.

Supports CSV, NumPy arrays, pandas DataFrames, and built-in benchmarks.
Provides data validation, type checking, missing data detection, and
variable type inference (continuous, discrete, mixed).
"""
from __future__ import annotations

import enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, DataMatrix


class VariableType(enum.Enum):
    """Inferred type of a variable (column)."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"
    CONSTANT = "constant"


class DataValidationResult:
    """Result of validating a data matrix.

    Attributes
    ----------
    is_valid : bool
    n_samples : int
    n_variables : int
    has_missing : bool
    missing_count : int
    missing_fraction : float
    has_constant_columns : bool
    constant_columns : List[int]
    variable_types : Dict[int, VariableType]
    warnings : List[str]
    """

    def __init__(
        self,
        is_valid: bool,
        n_samples: int,
        n_variables: int,
        has_missing: bool,
        missing_count: int,
        has_constant_columns: bool,
        constant_columns: List[int],
        variable_types: Dict[int, VariableType],
        warnings: List[str],
    ) -> None:
        self.is_valid = is_valid
        self.n_samples = n_samples
        self.n_variables = n_variables
        self.has_missing = has_missing
        self.missing_count = missing_count
        self.missing_fraction = missing_count / max(n_samples * n_variables, 1)
        self.has_constant_columns = has_constant_columns
        self.constant_columns = constant_columns
        self.variable_types = variable_types
        self.warnings = warnings

    def __repr__(self) -> str:
        return (
            f"DataValidationResult(valid={self.is_valid}, "
            f"shape=({self.n_samples}, {self.n_variables}), "
            f"missing={self.missing_count}, "
            f"warnings={len(self.warnings)})"
        )


class DataLoader:
    """Load observational data (and optionally ground-truth graphs) from disk.

    Supports CSV, NumPy ``.npy`` / ``.npz`` files, and pandas DataFrames.
    """

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_csv(
        path: str,
        delimiter: str = ",",
        has_header: Optional[bool] = None,
    ) -> DataMatrix:
        """Load a data matrix from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        delimiter : str
            Column delimiter.
        has_header : bool or None
            If ``None`` (default), auto-detect the header.

        Returns
        -------
        DataMatrix
        """
        filepath = Path(path)
        if has_header is True:
            data = np.loadtxt(
                filepath, delimiter=delimiter, dtype=np.float64, skiprows=1
            )
        elif has_header is False:
            data = np.loadtxt(filepath, delimiter=delimiter, dtype=np.float64)
        else:
            # Auto-detect: try without skipping, then with
            try:
                data = np.loadtxt(filepath, delimiter=delimiter, dtype=np.float64)
            except ValueError:
                data = np.loadtxt(
                    filepath, delimiter=delimiter, dtype=np.float64, skiprows=1
                )
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    # ------------------------------------------------------------------
    # NumPy loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_numpy(path: str) -> DataMatrix:
        """Load a data matrix from a ``.npy`` file.

        Parameters
        ----------
        path : str

        Returns
        -------
        DataMatrix
        """
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = list(data.keys())
            if len(keys) == 1:
                return data[keys[0]].astype(np.float64)
            if "data" in keys:
                return data["data"].astype(np.float64)
            return data[keys[0]].astype(np.float64)
        return data.astype(np.float64)

    # ------------------------------------------------------------------
    # Parquet loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_parquet(path: str) -> DataMatrix:
        """Load a data matrix from a Parquet file.

        Parameters
        ----------
        path : str
            Path to the Parquet file.

        Returns
        -------
        DataMatrix

        Raises
        ------
        ImportError
            If ``pyarrow`` or ``pandas`` is not installed.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pandas is required for Parquet support. "
                "Install with: pip install pandas pyarrow"
            )
        df = pd.read_parquet(path)
        numeric = df.select_dtypes(include=[np.number])
        data = numeric.to_numpy(dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    # ------------------------------------------------------------------
    # Pandas loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_dataframe(df: "object") -> DataMatrix:
        """Convert a pandas DataFrame to a DataMatrix.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        DataMatrix

        Raises
        ------
        TypeError
            If *df* is not a DataFrame.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("pandas is required for load_dataframe")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

        # Select only numeric columns
        numeric = df.select_dtypes(include=[np.number])
        return numeric.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(data: DataMatrix) -> DataValidationResult:
        """Validate a data matrix and infer variable types.

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        DataValidationResult
        """
        warnings: List[str] = []
        is_valid = True

        if data.ndim != 2:
            is_valid = False
            warnings.append(f"Expected 2D array, got {data.ndim}D")
            return DataValidationResult(
                is_valid=False,
                n_samples=0,
                n_variables=0,
                has_missing=False,
                missing_count=0,
                has_constant_columns=False,
                constant_columns=[],
                variable_types={},
                warnings=warnings,
            )

        n_samples, n_vars = data.shape

        if n_samples < 2:
            warnings.append(f"Very few samples: {n_samples}")
        if n_vars < 2:
            warnings.append(f"Very few variables: {n_vars}")

        # Missing data
        missing_mask = np.isnan(data)
        missing_count = int(missing_mask.sum())
        has_missing = missing_count > 0
        if has_missing:
            warnings.append(
                f"Missing values: {missing_count} "
                f"({100 * missing_count / (n_samples * n_vars):.1f}%)"
            )

        # Constant columns
        constant_cols: List[int] = []
        for j in range(n_vars):
            col = data[:, j]
            col_clean = col[~np.isnan(col)]
            if len(col_clean) > 0 and np.all(col_clean == col_clean[0]):
                constant_cols.append(j)
        if constant_cols:
            warnings.append(f"Constant columns: {constant_cols}")

        # Variable type inference
        var_types: Dict[int, VariableType] = {}
        for j in range(n_vars):
            col = data[:, j]
            col_clean = col[~np.isnan(col)]
            if len(col_clean) == 0:
                var_types[j] = VariableType.CONSTANT
            elif np.all(col_clean == col_clean[0]):
                var_types[j] = VariableType.CONSTANT
            else:
                unique = np.unique(col_clean)
                if len(unique) == 2:
                    var_types[j] = VariableType.BINARY
                elif len(unique) <= min(20, n_samples * 0.05):
                    # Heuristic: ≤20 unique values or ≤5% of samples
                    var_types[j] = VariableType.DISCRETE
                else:
                    var_types[j] = VariableType.CONTINUOUS

        # Inf values
        if np.any(np.isinf(data)):
            warnings.append("Data contains infinite values")

        return DataValidationResult(
            is_valid=is_valid and not has_missing,
            n_samples=n_samples,
            n_variables=n_vars,
            has_missing=has_missing,
            missing_count=missing_count,
            has_constant_columns=bool(constant_cols),
            constant_columns=constant_cols,
            variable_types=var_types,
            warnings=warnings,
        )

    @staticmethod
    def infer_variable_types(
        data: DataMatrix,
    ) -> Dict[int, VariableType]:
        """Infer the type of each variable.

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        Dict[int, VariableType]
        """
        result = DataLoader.validate(data)
        return result.variable_types

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    @staticmethod
    def load_benchmark(
        name: str,
    ) -> Tuple[DataMatrix, Optional[AdjacencyMatrix]]:
        """Load a built-in benchmark dataset by name.

        Currently supported: ``"sachs"``.

        Parameters
        ----------
        name : str

        Returns
        -------
        Tuple[DataMatrix, Optional[AdjacencyMatrix]]
        """
        name_lower = name.lower()
        if name_lower == "sachs":
            return _load_sachs()
        raise ValueError(
            f"Unknown benchmark dataset: {name!r}. Available: 'sachs'."
        )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _load_sachs() -> Tuple[DataMatrix, Optional[AdjacencyMatrix]]:
    """Load the Sachs benchmark from common locations."""
    candidate_dirs = [
        Path(__file__).resolve().parent / "benchmarks",
        Path.cwd() / "data" / "sachs",
    ]
    for d in candidate_dirs:
        data_path = d / "sachs_data.csv"
        graph_path = d / "sachs_graph.csv"
        if data_path.exists():
            data = np.loadtxt(
                data_path, delimiter=",", dtype=np.float64, skiprows=1
            )
            adj: Optional[AdjacencyMatrix] = None
            if graph_path.exists():
                adj = np.loadtxt(
                    graph_path, delimiter=",", dtype=np.int8, skiprows=1
                )
            return data, adj

    raise FileNotFoundError(
        "Sachs benchmark files not found. Place 'sachs_data.csv' in "
        "one of: " + ", ".join(str(d) for d in candidate_dirs)
    )
