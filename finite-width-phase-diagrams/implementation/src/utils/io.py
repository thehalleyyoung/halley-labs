"""I/O utilities for the finite-width phase diagram system.

Save/load phase diagrams, kernel matrices, calibration results.
HDF5 support for large arrays, JSON for metadata, numpy serialisation.
Checkpoint saving/loading for long computations.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .logging import get_logger

_log = get_logger("fwpd.io")


# ---------------------------------------------------------------------------
# Numpy array serialisation helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_array(arr: np.ndarray, path: Union[str, Path]) -> None:
    """Save a numpy array to ``.npy`` format."""
    p = _ensure_dir(path)
    np.save(str(p), arr)


def load_array(path: Union[str, Path]) -> np.ndarray:
    """Load a numpy array from ``.npy`` format."""
    return np.load(str(path), allow_pickle=False)


def save_arrays(arrays: Dict[str, np.ndarray], path: Union[str, Path]) -> None:
    """Save multiple arrays to a ``.npz`` archive."""
    p = _ensure_dir(path)
    np.savez_compressed(str(p), **arrays)


def load_arrays(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load arrays from a ``.npz`` archive."""
    with np.load(str(path), allow_pickle=False) as data:
        return dict(data)


# ---------------------------------------------------------------------------
# JSON metadata helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save data to a JSON file with numpy support."""
    p = _ensure_dir(path)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# HDF5 helpers (optional dependency)
# ---------------------------------------------------------------------------

def _require_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 support: pip install h5py")


def save_hdf5(
    arrays: Dict[str, np.ndarray],
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = "gzip",
) -> None:
    """Save arrays and metadata to an HDF5 file."""
    h5py = _require_h5py()
    p = _ensure_dir(path)
    with h5py.File(str(p), "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=arr, compression=compression)
        if metadata:
            meta_grp = f.create_group("metadata")
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta_grp.attrs[k] = v
                else:
                    meta_grp.attrs[k] = json.dumps(v, cls=_NumpyEncoder)


def load_hdf5(
    path: Union[str, Path],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load arrays and metadata from an HDF5 file."""
    h5py = _require_h5py()
    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        for name in f:
            if name == "metadata":
                continue
            arrays[name] = f[name][:]
        if "metadata" in f:
            for k, v in f["metadata"].attrs.items():
                if isinstance(v, str):
                    try:
                        metadata[k] = json.loads(v)
                    except json.JSONDecodeError:
                        metadata[k] = v
                else:
                    metadata[k] = v
    return arrays, metadata


# ---------------------------------------------------------------------------
# Phase diagram I/O
# ---------------------------------------------------------------------------

def save_phase_diagram(
    diagram: Any,
    path: Union[str, Path],
    format: str = "npz",
) -> None:
    """Save a PhaseDiagram object.

    Parameters
    ----------
    diagram : PhaseDiagram
        Phase diagram to save.
    path : str or Path
        Output path (extension auto-added if missing).
    format : str
        ``"npz"``, ``"hdf5"``, or ``"json"``.
    """
    p = Path(path)
    d = diagram.to_dict() if hasattr(diagram, "to_dict") else diagram

    arrays: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            arrays[k] = np.array(v)
        else:
            meta[k] = v

    if format == "npz":
        save_arrays(arrays, p.with_suffix(".npz"))
        if meta:
            save_json(meta, p.with_suffix(".json"))
    elif format == "hdf5":
        save_hdf5(arrays, p.with_suffix(".h5"), metadata=meta)
    elif format == "json":
        save_json(d, p.with_suffix(".json"))
    else:
        raise ValueError(f"Unknown format: {format}")

    _log.info("Saved phase diagram to %s (format=%s)", p, format)


def load_phase_diagram(
    path: Union[str, Path],
    format: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a phase diagram from file.

    Returns a dict; caller reconstructs PhaseDiagram via ``from_dict``.
    """
    p = Path(path)
    if format is None:
        if p.suffix in (".h5", ".hdf5"):
            format = "hdf5"
        elif p.suffix == ".json":
            format = "json"
        else:
            format = "npz"

    if format == "npz":
        result: Dict[str, Any] = {}
        result.update(load_arrays(p.with_suffix(".npz")))
        json_path = p.with_suffix(".json")
        if json_path.exists():
            result.update(load_json(json_path))
        return result
    elif format == "hdf5":
        arrays, meta = load_hdf5(p.with_suffix(".h5"))
        arrays.update(meta)
        return arrays
    elif format == "json":
        return load_json(p.with_suffix(".json"))
    else:
        raise ValueError(f"Unknown format: {format}")


# ---------------------------------------------------------------------------
# Kernel matrix I/O
# ---------------------------------------------------------------------------

def save_kernel_matrix(
    K: np.ndarray,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "npz",
) -> None:
    """Save a kernel matrix with optional metadata."""
    p = Path(path)
    if format == "hdf5":
        arrays = {"kernel": K}
        save_hdf5(arrays, p.with_suffix(".h5"), metadata=metadata)
    else:
        save_arrays({"kernel": K}, p.with_suffix(".npz"))
        if metadata:
            save_json(metadata, p.with_suffix(".meta.json"))
    _log.info("Saved kernel matrix (%s) to %s", K.shape, p)


def load_kernel_matrix(
    path: Union[str, Path],
    format: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a kernel matrix and its metadata."""
    p = Path(path)
    if format is None:
        format = "hdf5" if p.suffix in (".h5", ".hdf5") else "npz"

    if format == "hdf5":
        arrays, meta = load_hdf5(p.with_suffix(".h5"))
        return arrays["kernel"], meta
    else:
        arrays = load_arrays(p.with_suffix(".npz"))
        meta: Dict[str, Any] = {}
        meta_path = p.with_suffix(".meta.json")
        if meta_path.exists():
            meta = load_json(meta_path)
        return arrays["kernel"], meta


# ---------------------------------------------------------------------------
# Calibration result I/O
# ---------------------------------------------------------------------------

def save_calibration(
    result: Any,
    path: Union[str, Path],
) -> None:
    """Save calibration results (regression + bootstrap)."""
    p = _ensure_dir(Path(path))
    d = result.to_dict() if hasattr(result, "to_dict") else result
    if not isinstance(d, dict):
        d = asdict(d) if hasattr(d, "__dataclass_fields__") else {"data": d}

    arrays: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            meta[k] = v

    if arrays:
        save_arrays(arrays, p.with_suffix(".npz"))
    save_json(meta, p.with_suffix(".json"))
    _log.info("Saved calibration result to %s", p)


def load_calibration(
    path: Union[str, Path],
) -> Dict[str, Any]:
    """Load calibration results."""
    p = Path(path)
    result: Dict[str, Any] = {}
    npz_path = p.with_suffix(".npz")
    if npz_path.exists():
        result.update(load_arrays(npz_path))
    json_path = p.with_suffix(".json")
    if json_path.exists():
        result.update(load_json(json_path))
    return result


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save and resume long computations from checkpoints.

    Each checkpoint stores step index, data arrays, and metadata.
    Supports automatic cleanup of old checkpoints.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory for checkpoint files.
    max_checkpoints : int
        Maximum number of checkpoints to retain.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        max_checkpoints: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._log = get_logger("fwpd.checkpoint")

    def save(
        self,
        step: int,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint for the given step.

        Parameters
        ----------
        step : int
            Pipeline step index.
        data : dict
            Data to checkpoint (may contain numpy arrays).
        metadata : dict, optional
            Extra metadata (JSON-serialisable).

        Returns
        -------
        Path to the checkpoint directory.
        """
        ts = int(time.time())
        ckpt_name = f"step_{step:04d}_{ts}"
        ckpt_dir = self.checkpoint_dir / ckpt_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {}
        meta: Dict[str, Any] = {"step": step, "timestamp": ts}
        if metadata:
            meta.update(metadata)

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                arrays[k] = v
            else:
                meta[k] = v

        if arrays:
            save_arrays(arrays, ckpt_dir / "data.npz")
        save_json(meta, ckpt_dir / "meta.json")

        self._log.info("Checkpoint saved: step=%d -> %s", step, ckpt_dir)
        self._cleanup()
        return ckpt_dir

    def load_latest(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Load the most recent checkpoint.

        Returns
        -------
        (step, data_dict) or None if no checkpoints exist.
        """
        ckpt_dirs = sorted(self.checkpoint_dir.iterdir())
        ckpt_dirs = [d for d in ckpt_dirs if d.is_dir() and d.name.startswith("step_")]
        if not ckpt_dirs:
            return None
        latest = ckpt_dirs[-1]
        return self._load_checkpoint(latest)

    def load_step(self, step: int) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Load a specific step checkpoint."""
        prefix = f"step_{step:04d}_"
        for d in sorted(self.checkpoint_dir.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith(prefix):
                return self._load_checkpoint(d)
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        result = []
        for d in sorted(self.checkpoint_dir.iterdir()):
            if d.is_dir() and d.name.startswith("step_"):
                meta_path = d / "meta.json"
                if meta_path.exists():
                    meta = load_json(meta_path)
                    result.append({"dir": str(d), **meta})
        return result

    def _load_checkpoint(self, ckpt_dir: Path) -> Tuple[int, Dict[str, Any]]:
        data: Dict[str, Any] = {}
        npz_path = ckpt_dir / "data.npz"
        if npz_path.exists():
            data.update(load_arrays(npz_path))
        meta_path = ckpt_dir / "meta.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            step = meta.pop("step", 0)
            data.update(meta)
        else:
            step = 0
        self._log.info("Checkpoint loaded: step=%d from %s", step, ckpt_dir)
        return step, data

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        ckpt_dirs = sorted(
            [d for d in self.checkpoint_dir.iterdir()
             if d.is_dir() and d.name.startswith("step_")]
        )
        while len(ckpt_dirs) > self.max_checkpoints:
            old = ckpt_dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            self._log.debug("Removed old checkpoint: %s", old)

    def clear(self) -> None:
        """Remove all checkpoints."""
        for d in self.checkpoint_dir.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                shutil.rmtree(d, ignore_errors=True)
        self._log.info("All checkpoints cleared")
