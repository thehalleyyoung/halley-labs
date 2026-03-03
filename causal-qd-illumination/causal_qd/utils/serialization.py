"""Efficient serialization for DAGs, archives, and experiment results.

Provides compact serialization of graph structures using sparse formats,
archive compression, and export to multiple formats (JSON, CSV, HDF5).

Key classes
-----------
* :class:`DAGSerializer` – efficient DAG serialization (sparse format)
* :class:`ArchiveSerializer` – serialize/deserialize full archives
* :class:`CompressedArchive` – compress archive for storage
* :class:`ResultExporter` – export results to JSON, CSV, HDF5
* :class:`ExperimentLogger` – log experiment parameters and results
"""

from __future__ import annotations

import gzip
import io
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, QualityScore

__all__ = [
    "DAGSerializer",
    "ArchiveSerializer",
    "CompressedArchive",
    "ResultExporter",
    "ExperimentLogger",
]


# ---------------------------------------------------------------------------
# DAGSerializer
# ---------------------------------------------------------------------------


class DAGSerializer:
    """Efficient DAG serialization using sparse edge-list format.

    For typical sparse DAGs (density < 0.3), storing as an edge list is
    much more compact than the full adjacency matrix.

    Serialization format
    --------------------
    A dict with keys:

    * ``n``: number of nodes
    * ``edges``: list of ``[source, target]`` pairs
    * ``format``: ``"sparse"`` or ``"dense"``

    Examples
    --------
    >>> s = DAGSerializer()
    >>> data = s.serialize(adj)
    >>> restored = s.deserialize(data)
    >>> np.array_equal(adj, restored)
    True
    """

    DENSITY_THRESHOLD: float = 0.3

    def serialize(
        self,
        adj: AdjacencyMatrix,
        force_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize an adjacency matrix.

        Parameters
        ----------
        adj : np.ndarray
            ``(n, n)`` adjacency matrix.
        force_format : str, optional
            Force ``"sparse"`` or ``"dense"`` format.

        Returns
        -------
        dict
            Serialized representation.
        """
        n = adj.shape[0]
        n_edges = int(np.sum(adj))
        max_edges = n * (n - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0

        use_sparse = force_format == "sparse" or (
            force_format is None and density < self.DENSITY_THRESHOLD
        )

        if use_sparse:
            rows, cols = np.nonzero(adj)
            edges = [[int(r), int(c)] for r, c in zip(rows, cols)]
            return {"n": n, "edges": edges, "format": "sparse"}
        else:
            return {
                "n": n,
                "matrix": adj.astype(np.int8).tolist(),
                "format": "dense",
            }

    def deserialize(self, data: Dict[str, Any]) -> AdjacencyMatrix:
        """Deserialize an adjacency matrix.

        Parameters
        ----------
        data : dict
            Serialized representation.

        Returns
        -------
        np.ndarray
            ``(n, n)`` adjacency matrix.
        """
        n = data["n"]
        fmt = data.get("format", "sparse")

        if fmt == "sparse":
            adj = np.zeros((n, n), dtype=np.int8)
            for src, tgt in data["edges"]:
                adj[src, tgt] = 1
            return adj
        else:
            return np.array(data["matrix"], dtype=np.int8)

    def to_bytes(self, adj: AdjacencyMatrix) -> bytes:
        """Serialize to compact binary format.

        Format: [n (4 bytes)] [n_edges (4 bytes)] [edge pairs (4 bytes each)]
        """
        n = adj.shape[0]
        rows, cols = np.nonzero(adj)
        n_edges = len(rows)

        buf = np.array([n, n_edges], dtype=np.int32).tobytes()
        if n_edges > 0:
            edges = np.column_stack([rows, cols]).astype(np.int16)
            buf += edges.tobytes()
        return buf

    def from_bytes(self, data: bytes) -> AdjacencyMatrix:
        """Deserialize from compact binary format."""
        header = np.frombuffer(data[:8], dtype=np.int32)
        n, n_edges = int(header[0]), int(header[1])

        adj = np.zeros((n, n), dtype=np.int8)
        if n_edges > 0:
            edges = np.frombuffer(data[8:], dtype=np.int16).reshape(-1, 2)
            for src, tgt in edges:
                adj[src, tgt] = 1
        return adj

    def batch_serialize(
        self, adjs: Sequence[AdjacencyMatrix]
    ) -> List[Dict[str, Any]]:
        """Serialize multiple DAGs."""
        return [self.serialize(adj) for adj in adjs]

    def batch_deserialize(
        self, data_list: List[Dict[str, Any]]
    ) -> List[AdjacencyMatrix]:
        """Deserialize multiple DAGs."""
        return [self.deserialize(d) for d in data_list]


# ---------------------------------------------------------------------------
# ArchiveSerializer
# ---------------------------------------------------------------------------


class ArchiveSerializer:
    """Serialize and deserialize full MAP-Elites archives.

    Stores archive entries (solution, quality, descriptor) along with
    archive configuration (dimensions, bounds).

    Examples
    --------
    >>> serializer = ArchiveSerializer()
    >>> serializer.save(archive, "archive.npz")
    >>> loaded = serializer.load("archive.npz")
    """

    def __init__(self) -> None:
        self._dag_serializer = DAGSerializer()

    def save(self, archive: Any, path: str) -> None:
        """Save archive to disk.

        Parameters
        ----------
        archive : Archive-like
            Must have ``elites()``, ``descriptor_bounds``,
            and optional ``dims`` attribute.
        path : str
            Output file path. Uses ``.npz`` format.
        """
        save_dict: Dict[str, Any] = {}

        # Bounds
        if hasattr(archive, "descriptor_bounds"):
            lower, upper = archive.descriptor_bounds
            save_dict["lower_bounds"] = np.asarray(lower)
            save_dict["upper_bounds"] = np.asarray(upper)

        # Dims
        if hasattr(archive, "dims"):
            save_dict["dims"] = np.array(archive.dims)

        # Entries
        elites = archive.elites() if callable(archive.elites) else archive.elites
        n_elites = len(elites)

        if n_elites > 0:
            # Assume all solutions have the same shape
            first = elites[0]
            if hasattr(first, "solution"):
                solutions = np.array([e.solution for e in elites], dtype=np.int8)
                qualities = np.array([e.quality for e in elites], dtype=np.float64)
                descriptors = np.array(
                    [e.descriptor for e in elites], dtype=np.float64
                )
            else:
                # Tuple format: (solution, quality, descriptor)
                solutions = np.array([e[0] for e in elites], dtype=np.int8)
                qualities = np.array([e[1] for e in elites], dtype=np.float64)
                descriptors = np.array([e[2] for e in elites], dtype=np.float64)

            save_dict["solutions"] = solutions
            save_dict["qualities"] = qualities
            save_dict["descriptors"] = descriptors

        np.savez_compressed(path, **save_dict)

    def load(self, path: str) -> Dict[str, Any]:
        """Load archive data from disk.

        Parameters
        ----------
        path : str
            Input file path (``.npz`` format).

        Returns
        -------
        dict
            Keys: ``solutions``, ``qualities``, ``descriptors``,
            ``lower_bounds``, ``upper_bounds``, ``dims`` (if present).
        """
        data = np.load(path, allow_pickle=True)
        result: Dict[str, Any] = {}

        for key in data.files:
            result[key] = data[key]

        return result

    def to_json(self, archive: Any) -> str:
        """Serialize archive to JSON string.

        Only stores qualities and descriptors (not full solutions) for
        lightweight transfer.
        """
        elites = archive.elites() if callable(archive.elites) else archive.elites
        entries = []
        for e in elites:
            if hasattr(e, "quality"):
                entries.append({
                    "quality": float(e.quality),
                    "descriptor": e.descriptor.tolist(),
                })
            else:
                entries.append({
                    "quality": float(e[1]),
                    "descriptor": e[2].tolist(),
                })

        obj = {
            "n_elites": len(entries),
            "entries": entries,
        }

        if hasattr(archive, "descriptor_bounds"):
            lower, upper = archive.descriptor_bounds
            obj["lower_bounds"] = np.asarray(lower).tolist()
            obj["upper_bounds"] = np.asarray(upper).tolist()

        return json.dumps(obj, indent=2)


# ---------------------------------------------------------------------------
# CompressedArchive
# ---------------------------------------------------------------------------


class CompressedArchive:
    """Compress and decompress archive data for efficient storage.

    Uses gzip compression on the binary representation.

    Examples
    --------
    >>> ca = CompressedArchive()
    >>> compressed = ca.compress(archive_data)
    >>> original = ca.decompress(compressed)
    """

    def __init__(self, compression_level: int = 6) -> None:
        self._level = compression_level

    def compress(self, archive_data: Dict[str, Any]) -> bytes:
        """Compress archive data.

        Parameters
        ----------
        archive_data : dict
            Archive data with numpy arrays.

        Returns
        -------
        bytes
            Gzip-compressed bytes.
        """
        buf = io.BytesIO()
        np.savez(buf, **archive_data)
        raw = buf.getvalue()
        return gzip.compress(raw, compresslevel=self._level)

    def decompress(self, data: bytes) -> Dict[str, Any]:
        """Decompress archive data.

        Parameters
        ----------
        data : bytes
            Gzip-compressed bytes.

        Returns
        -------
        dict
            Archive data with numpy arrays.
        """
        raw = gzip.decompress(data)
        buf = io.BytesIO(raw)
        npz = np.load(buf, allow_pickle=True)
        return {key: npz[key] for key in npz.files}

    def save_compressed(
        self, archive_data: Dict[str, Any], path: str
    ) -> int:
        """Save compressed archive to file.

        Returns the file size in bytes.
        """
        compressed = self.compress(archive_data)
        with open(path, "wb") as f:
            f.write(compressed)
        return len(compressed)

    def load_compressed(self, path: str) -> Dict[str, Any]:
        """Load compressed archive from file."""
        with open(path, "rb") as f:
            data = f.read()
        return self.decompress(data)

    def compression_ratio(
        self, archive_data: Dict[str, Any]
    ) -> float:
        """Compute compression ratio.

        Returns ratio of compressed size to uncompressed size.
        """
        buf = io.BytesIO()
        np.savez(buf, **archive_data)
        raw_size = len(buf.getvalue())
        compressed = self.compress(archive_data)
        return len(compressed) / raw_size if raw_size > 0 else 1.0


# ---------------------------------------------------------------------------
# ResultExporter
# ---------------------------------------------------------------------------


class ResultExporter:
    """Export experiment results to various formats.

    Supports JSON, CSV, and HDF5 (if h5py is installed).

    Examples
    --------
    >>> exporter = ResultExporter()
    >>> exporter.to_json(results, "results.json")
    >>> exporter.to_csv(results, "results.csv")
    """

    def to_json(
        self,
        data: Dict[str, Any],
        path: str,
        indent: int = 2,
    ) -> None:
        """Export to JSON.

        Parameters
        ----------
        data : dict
            Data to export. Numpy arrays are converted to lists.
        path : str
            Output file path.
        indent : int
            JSON indentation.
        """

        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if is_dataclass(obj) and not isinstance(obj, type):
                return asdict(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        with open(path, "w") as f:
            json.dump(_convert(data), f, indent=indent)

    def to_csv(
        self,
        records: List[Dict[str, Any]],
        path: str,
    ) -> None:
        """Export list of records to CSV.

        Parameters
        ----------
        records : list of dict
            Each dict is a row.
        path : str
            Output file path.
        """
        if not records:
            return

        import csv

        keys = list(records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for record in records:
                row = {}
                for k, v in record.items():
                    if isinstance(v, np.ndarray):
                        row[k] = v.tolist()
                    elif isinstance(v, (np.integer,)):
                        row[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        row[k] = float(v)
                    else:
                        row[k] = v
                writer.writerow(row)

    def to_hdf5(
        self,
        data: Dict[str, Any],
        path: str,
    ) -> None:
        """Export to HDF5 format.

        Parameters
        ----------
        data : dict
            Data to export. Values should be numpy arrays or scalars.
        path : str
            Output file path.

        Raises
        ------
        ImportError
            If h5py is not installed.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 export.")

        with h5py.File(path, "w") as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression="gzip")
                elif isinstance(value, (int, float, str, bool)):
                    f.attrs[key] = value
                elif isinstance(value, dict):
                    group = f.create_group(key)
                    for k2, v2 in value.items():
                        if isinstance(v2, np.ndarray):
                            group.create_dataset(
                                k2, data=v2, compression="gzip"
                            )
                        else:
                            group.attrs[k2] = v2

    def from_hdf5(self, path: str) -> Dict[str, Any]:
        """Load data from HDF5 format.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        dict
            Loaded data.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 import.")

        result: Dict[str, Any] = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    result[key] = f[key][()]
                elif isinstance(f[key], h5py.Group):
                    group_data: Dict[str, Any] = {}
                    for k2 in f[key].keys():
                        group_data[k2] = f[key][k2][()]
                    for k2, v2 in f[key].attrs.items():
                        group_data[k2] = v2
                    result[key] = group_data
            for key, val in f.attrs.items():
                result[key] = val

        return result


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------


class ExperimentLogger:
    """Log experiment parameters, metrics, and artifacts.

    Provides structured logging for reproducible experiments.

    Parameters
    ----------
    log_dir : str
        Directory for log files and artifacts.
    experiment_name : str
        Name of the experiment.

    Examples
    --------
    >>> logger = ExperimentLogger("logs", "experiment_01")
    >>> logger.log_params({"n_nodes": 10, "n_iter": 500})
    >>> logger.log_metric("best_quality", -123.4, step=100)
    >>> logger.save()
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "experiment",
    ) -> None:
        self._log_dir = Path(log_dir)
        self._experiment_name = experiment_name
        self._params: Dict[str, Any] = {}
        self._metrics: Dict[str, List[Tuple[int, float]]] = {}
        self._artifacts: List[str] = []
        self._start_time: float = time.time()
        self._notes: List[str] = []

        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters.

        Parameters
        ----------
        params : dict
            Parameter name-value pairs.
        """
        self._params.update(params)

    def log_metric(
        self, name: str, value: float, step: int = 0
    ) -> None:
        """Log a metric value.

        Parameters
        ----------
        name : str
            Metric name.
        value : float
            Metric value.
        step : int
            Step/iteration number.
        """
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append((step, float(value)))

    def log_metrics(
        self, metrics: Dict[str, float], step: int = 0
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_artifact(self, path: str) -> None:
        """Register an artifact file path."""
        self._artifacts.append(path)

    def add_note(self, note: str) -> None:
        """Add a text note to the experiment log."""
        self._notes.append(note)

    def save(self) -> str:
        """Save experiment log to disk.

        Returns
        -------
        str
            Path to the saved log file.
        """
        log_data = {
            "experiment_name": self._experiment_name,
            "start_time": self._start_time,
            "duration_seconds": time.time() - self._start_time,
            "params": self._params,
            "metrics": {
                name: [{"step": s, "value": v} for s, v in values]
                for name, values in self._metrics.items()
            },
            "artifacts": self._artifacts,
            "notes": self._notes,
        }

        log_path = str(
            self._log_dir / f"{self._experiment_name}.json"
        )

        exporter = ResultExporter()
        exporter.to_json(log_data, log_path)
        return log_path

    def get_metric_history(
        self, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get metric history as numpy arrays.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            ``(steps, values)`` arrays.
        """
        if name not in self._metrics:
            return np.array([]), np.array([])
        entries = self._metrics[name]
        steps = np.array([s for s, _ in entries])
        values = np.array([v for _, v in entries])
        return steps, values

    def summary(self) -> Dict[str, Any]:
        """Experiment summary."""
        summary: Dict[str, Any] = {
            "name": self._experiment_name,
            "params": self._params,
            "duration": time.time() - self._start_time,
        }
        for name, entries in self._metrics.items():
            if entries:
                values = [v for _, v in entries]
                summary[f"{name}_final"] = values[-1]
                summary[f"{name}_best"] = max(values)
        return summary
