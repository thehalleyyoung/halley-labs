"""
taintflow.integrations.formats – Parquet and CSV file format support.

Provides provenance-aware loading of ML datasets from Parquet and CSV
files.  Each loader wraps the underlying pandas/pyarrow read and
annotates the resulting DataFrame with row-level partition provenance,
enabling TaintFlow to track data lineage from the point of ingestion.

Usage::

    from taintflow.integrations.formats import (
        load_csv,
        load_parquet,
        FormatDetector,
    )

    # Load with automatic provenance annotation
    df = load_csv("train.csv", partition="train")
    df_test = load_parquet("test.parquet", partition="test")

    # Or auto-detect format
    df = FormatDetector.load("data.parquet", partition="train")
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from taintflow.core.types import Origin, ProvenanceInfo

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False


# ===================================================================
#  Data ingestion record
# ===================================================================


@dataclass
class IngestionRecord:
    """Metadata recorded when a dataset file is loaded."""

    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    file_path: str = ""
    file_format: str = ""
    partition: str = "unknown"
    n_rows: int = 0
    n_columns: int = 0
    columns: List[str] = field(default_factory=list)
    file_size_bytes: int = 0
    load_time_seconds: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "file_path": self.file_path,
            "file_format": self.file_format,
            "partition": self.partition,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "file_size_bytes": self.file_size_bytes,
            "load_time_seconds": self.load_time_seconds,
        }


# ===================================================================
#  CSV loading
# ===================================================================


def load_csv(
    path: Union[str, Path],
    partition: str = "unknown",
    **kwargs: Any,
) -> Any:
    """Load a CSV file with provenance annotation.

    Parameters
    ----------
    path:
        Path to the CSV file.
    partition:
        Partition label: ``"train"``, ``"test"``, or ``"unknown"``.
    **kwargs:
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame.  A ``_taintflow_provenance`` attribute
        is attached with an :class:`IngestionRecord`.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for CSV loading. "
            "Install with: pip install taintflow[pandas]"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    start = time.monotonic()
    df = pd.read_csv(path, **kwargs)
    elapsed = time.monotonic() - start

    record = IngestionRecord(
        file_path=str(path),
        file_format="csv",
        partition=partition,
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=list(df.columns),
        file_size_bytes=path.stat().st_size,
        load_time_seconds=elapsed,
    )

    df.attrs["_taintflow_provenance"] = record
    df.attrs["_taintflow_partition"] = partition

    logger.info(
        "Loaded CSV %s: %d rows × %d cols (partition=%s, %.3fs)",
        path.name, len(df), len(df.columns), partition, elapsed,
    )

    return df


# ===================================================================
#  Parquet loading
# ===================================================================


def load_parquet(
    path: Union[str, Path],
    partition: str = "unknown",
    columns: Optional[List[str]] = None,
    use_pyarrow: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a Parquet file with provenance annotation.

    Parameters
    ----------
    path:
        Path to the Parquet file.
    partition:
        Partition label: ``"train"``, ``"test"``, or ``"unknown"``.
    columns:
        Optional list of column names to read (column pruning).
    use_pyarrow:
        If True and pyarrow is available, read directly with pyarrow
        for better performance.  Otherwise falls back to pandas.
    **kwargs:
        Additional keyword arguments forwarded to the reader.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame with provenance annotations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    start = time.monotonic()

    if use_pyarrow and _PYARROW_AVAILABLE:
        table = pq.read_table(path, columns=columns, **kwargs)
        df = table.to_pandas()
    elif _PANDAS_AVAILABLE:
        df = pd.read_parquet(path, columns=columns, **kwargs)
    else:
        raise ImportError(
            "Either pyarrow or pandas is required for Parquet loading. "
            "Install with: pip install taintflow[formats]"
        )

    elapsed = time.monotonic() - start

    record = IngestionRecord(
        file_path=str(path),
        file_format="parquet",
        partition=partition,
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=list(df.columns),
        file_size_bytes=path.stat().st_size,
        load_time_seconds=elapsed,
    )

    df.attrs["_taintflow_provenance"] = record
    df.attrs["_taintflow_partition"] = partition

    logger.info(
        "Loaded Parquet %s: %d rows × %d cols (partition=%s, %.3fs)",
        path.name, len(df), len(df.columns), partition, elapsed,
    )

    return df


# ===================================================================
#  Parquet writing (with provenance metadata)
# ===================================================================


def save_parquet(
    df: Any,
    path: Union[str, Path],
    partition: Optional[str] = None,
    **kwargs: Any,
) -> IngestionRecord:
    """Save a DataFrame to Parquet with TaintFlow metadata.

    Parameters
    ----------
    df:
        The pandas DataFrame to save.
    path:
        Output file path.
    partition:
        Partition label to embed in Parquet metadata.
    **kwargs:
        Additional keyword arguments forwarded to the writer.

    Returns
    -------
    IngestionRecord
        Record of the write operation.
    """
    path = Path(path)

    if _PYARROW_AVAILABLE:
        table = pa.Table.from_pandas(df)
        metadata = table.schema.metadata or {}
        if partition:
            metadata[b"taintflow_partition"] = partition.encode()
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, path, **kwargs)
    elif _PANDAS_AVAILABLE:
        df.to_parquet(path, **kwargs)
    else:
        raise ImportError(
            "Either pyarrow or pandas is required for Parquet writing."
        )

    record = IngestionRecord(
        file_path=str(path),
        file_format="parquet",
        partition=partition or "unknown",
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=list(df.columns),
        file_size_bytes=path.stat().st_size,
    )

    logger.info(
        "Saved Parquet %s: %d rows × %d cols",
        path.name, len(df), len(df.columns),
    )

    return record


# ===================================================================
#  Format detection
# ===================================================================


class FormatDetector:
    """Auto-detect file format and load with provenance tracking."""

    _EXTENSIONS: Dict[str, str] = {
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".parq": "parquet",
    }

    @classmethod
    def detect(cls, path: Union[str, Path]) -> str:
        """Detect the file format from the extension.

        Returns
        -------
        str
            One of ``"csv"``, ``"parquet"``, or ``"unknown"``.
        """
        ext = Path(path).suffix.lower()
        return cls._EXTENSIONS.get(ext, "unknown")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        partition: str = "unknown",
        **kwargs: Any,
    ) -> Any:
        """Load a file with automatic format detection.

        Parameters
        ----------
        path:
            Path to the data file.
        partition:
            Partition label for provenance tracking.
        **kwargs:
            Additional keyword arguments forwarded to the loader.

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame with provenance annotations.

        Raises
        ------
        ValueError
            If the file format cannot be detected.
        """
        fmt = cls.detect(path)
        if fmt == "csv":
            return load_csv(path, partition=partition, **kwargs)
        elif fmt == "parquet":
            return load_parquet(path, partition=partition, **kwargs)
        else:
            raise ValueError(
                f"Cannot detect format for {path}. "
                "Supported formats: .csv, .tsv, .parquet, .pq"
            )


# ===================================================================
#  Provenance extraction utilities
# ===================================================================


def get_provenance(df: Any) -> Optional[IngestionRecord]:
    """Extract the TaintFlow provenance record from a DataFrame.

    Returns None if the DataFrame was not loaded via TaintFlow.
    """
    return getattr(df, "attrs", {}).get("_taintflow_provenance")


def get_partition(df: Any) -> str:
    """Extract the partition label from a DataFrame.

    Returns ``"unknown"`` if the DataFrame was not loaded via TaintFlow.
    """
    return getattr(df, "attrs", {}).get("_taintflow_partition", "unknown")


# ===================================================================
#  Public exports
# ===================================================================

__all__: list[str] = [
    "IngestionRecord",
    "FormatDetector",
    "load_csv",
    "load_parquet",
    "save_parquet",
    "get_provenance",
    "get_partition",
]
