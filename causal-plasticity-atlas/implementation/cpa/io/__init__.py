"""CPA I/O — Data readers, writers, and serialization utilities.

Provides unified interfaces for loading multi-context datasets from
various formats (CSV, numpy, pandas, synthetic), writing results to
disk, and serializing/deserializing CPA data structures.

Modules
-------
readers
    Data readers for CSV, numpy, pandas, and synthetic data.
writers
    Output writers for JSON, CSV, numpy, and full atlas reports.
serialization
    Serialization utilities for numpy arrays, SCMs, and atlases.
"""

from cpa.io.readers import (
    CSVReader,
    NumpyReader,
    PandasReader,
    SyntheticReader,
    MultiContextDataset,
    DataValidator,
)
from cpa.io.writers import (
    JSONWriter,
    CSVWriter,
    NumpyWriter,
    AtlasWriter,
    ReportWriter,
)
from cpa.io.serialization import (
    serialize_numpy,
    deserialize_numpy,
    serialize_atlas,
    deserialize_atlas,
)

__all__ = [
    "CSVReader",
    "NumpyReader",
    "PandasReader",
    "SyntheticReader",
    "MultiContextDataset",
    "DataValidator",
    "JSONWriter",
    "CSVWriter",
    "NumpyWriter",
    "AtlasWriter",
    "ReportWriter",
    "serialize_numpy",
    "deserialize_numpy",
    "serialize_atlas",
    "deserialize_atlas",
]
