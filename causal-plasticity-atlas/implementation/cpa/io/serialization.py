"""Serialization utilities for the CPA engine.

Provides functions for converting CPA data structures (numpy arrays,
SCMs, MCCMs, atlas results) to and from JSON-safe formats with
version tagging for forward compatibility.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("io.serialization")


# Version tag for serialization format
_SERIALIZATION_VERSION = "1.0.0"


# =====================================================================
# Numpy serialization
# =====================================================================


def serialize_numpy(
    arr: np.ndarray,
    encoding: str = "list",
) -> Dict[str, Any]:
    """Serialize a numpy array to a JSON-safe dictionary.

    Parameters
    ----------
    arr : np.ndarray
        Array to serialize.
    encoding : str
        Encoding method: 'list' (nested lists), 'base64' (compact binary),
        or 'flat' (flat list with shape).

    Returns
    -------
    dict
        JSON-safe representation with type and shape metadata.

    Examples
    --------
    >>> d = serialize_numpy(np.eye(3))
    >>> arr = deserialize_numpy(d)
    """
    result: Dict[str, Any] = {
        "__type__": "ndarray",
        "__version__": _SERIALIZATION_VERSION,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }

    if encoding == "list":
        result["encoding"] = "list"
        result["data"] = arr.tolist()
    elif encoding == "base64":
        result["encoding"] = "base64"
        raw = np.ascontiguousarray(arr).tobytes()
        result["data"] = base64.b64encode(raw).decode("ascii")
        result["byte_order"] = "C"
    elif encoding == "flat":
        result["encoding"] = "flat"
        result["data"] = arr.ravel().tolist()
    else:
        raise ValueError(
            f"encoding must be 'list', 'base64', or 'flat', got {encoding!r}"
        )

    return result


def deserialize_numpy(d: Dict[str, Any]) -> np.ndarray:
    """Deserialize a numpy array from a JSON-safe dictionary.

    Parameters
    ----------
    d : dict
        Dictionary produced by :func:`serialize_numpy`.

    Returns
    -------
    np.ndarray
    """
    if "__type__" not in d or d["__type__"] != "ndarray":
        raise ValueError("Not a serialized ndarray")

    dtype = np.dtype(d["dtype"])
    shape = tuple(d["shape"])
    encoding = d.get("encoding", "list")

    if encoding == "list":
        return np.array(d["data"], dtype=dtype).reshape(shape)
    elif encoding == "base64":
        raw = base64.b64decode(d["data"])
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    elif encoding == "flat":
        return np.array(d["data"], dtype=dtype).reshape(shape)
    else:
        raise ValueError(f"Unknown encoding: {encoding!r}")


def serialize_sparse_matrix(
    arr: np.ndarray, threshold: float = 0.0
) -> Dict[str, Any]:
    """Serialize a sparse-ish matrix using COO format.

    Parameters
    ----------
    arr : np.ndarray
        2D array (typically sparse adjacency or parameter matrix).
    threshold : float
        Values with abs <= threshold are treated as zero.

    Returns
    -------
    dict
        Compact JSON-safe representation.
    """
    rows, cols = np.where(np.abs(arr) > threshold)
    values = arr[rows, cols].tolist()

    return {
        "__type__": "sparse_matrix",
        "__version__": _SERIALIZATION_VERSION,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "rows": rows.tolist(),
        "cols": cols.tolist(),
        "values": values,
        "nnz": len(values),
    }


def deserialize_sparse_matrix(d: Dict[str, Any]) -> np.ndarray:
    """Deserialize a sparse matrix from COO format.

    Parameters
    ----------
    d : dict
        Dictionary produced by :func:`serialize_sparse_matrix`.

    Returns
    -------
    np.ndarray
    """
    shape = tuple(d["shape"])
    dtype = np.dtype(d["dtype"])
    arr = np.zeros(shape, dtype=dtype)

    rows = d["rows"]
    cols = d["cols"]
    values = d["values"]

    for r, c, v in zip(rows, cols, values):
        arr[r, c] = v

    return arr


# =====================================================================
# SCM serialization
# =====================================================================


def serialize_scm(scm: Any) -> Dict[str, Any]:
    """Serialize an SCM result to a JSON-safe dictionary.

    Parameters
    ----------
    scm : SCMResult or dict
        SCM to serialize.

    Returns
    -------
    dict
        JSON-safe representation.
    """
    if hasattr(scm, "to_dict"):
        d = scm.to_dict()
    elif isinstance(scm, dict):
        d = dict(scm)
    else:
        raise TypeError(f"Cannot serialize SCM of type {type(scm)}")

    result: Dict[str, Any] = {
        "__type__": "scm",
        "__version__": _SERIALIZATION_VERSION,
    }

    for key, value in d.items():
        if isinstance(value, np.ndarray):
            result[key] = serialize_numpy(value)
        else:
            result[key] = value

    return result


def deserialize_scm(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize an SCM from a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary produced by :func:`serialize_scm`.

    Returns
    -------
    dict
        Deserialized SCM data with numpy arrays restored.
    """
    result: Dict[str, Any] = {}

    for key, value in d.items():
        if key.startswith("__"):
            continue
        if isinstance(value, dict) and value.get("__type__") == "ndarray":
            result[key] = deserialize_numpy(value)
        elif isinstance(value, list) and key in ("adjacency", "parameters"):
            result[key] = np.array(value)
        else:
            result[key] = value

    return result


# =====================================================================
# MCCM serialization
# =====================================================================


def serialize_mccm(mccm: Any) -> Dict[str, Any]:
    """Serialize a Multi-Context Causal Model (MCCM).

    An MCCM consists of multiple SCMs (one per context) plus
    alignment information.

    Parameters
    ----------
    mccm : dict or object
        MCCM containing context_scms and alignments.

    Returns
    -------
    dict
        JSON-safe representation.
    """
    if hasattr(mccm, "to_dict"):
        data = mccm.to_dict()
    elif isinstance(mccm, dict):
        data = dict(mccm)
    else:
        raise TypeError(f"Cannot serialize MCCM of type {type(mccm)}")

    result: Dict[str, Any] = {
        "__type__": "mccm",
        "__version__": _SERIALIZATION_VERSION,
        "context_ids": data.get("context_ids", []),
        "variable_names": data.get("variable_names", []),
    }

    scms = data.get("scm_results", data.get("context_scms", {}))
    result["scm_results"] = {
        cid: serialize_scm(scm) for cid, scm in scms.items()
    }

    alignments = data.get("alignment_results", data.get("alignments", {}))
    serialized_alignments: Dict[str, Any] = {}
    for key, alignment in alignments.items():
        str_key = (
            f"{key[0]}__{key[1]}" if isinstance(key, tuple) else str(key)
        )
        if hasattr(alignment, "to_dict"):
            al_dict = alignment.to_dict()
        elif isinstance(alignment, dict):
            al_dict = dict(alignment)
        else:
            al_dict = {}

        for ak, av in al_dict.items():
            if isinstance(av, np.ndarray):
                al_dict[ak] = serialize_numpy(av)
        serialized_alignments[str_key] = al_dict

    result["alignment_results"] = serialized_alignments

    descriptors = data.get("descriptors", {})
    if descriptors:
        result["descriptors"] = {}
        for var, desc in descriptors.items():
            if hasattr(desc, "to_dict"):
                result["descriptors"][var] = desc.to_dict()
            elif isinstance(desc, dict):
                result["descriptors"][var] = dict(desc)

    return result


def deserialize_mccm(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize an MCCM from a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary produced by :func:`serialize_mccm`.

    Returns
    -------
    dict
        Deserialized MCCM with numpy arrays restored.
    """
    result: Dict[str, Any] = {
        "context_ids": d.get("context_ids", []),
        "variable_names": d.get("variable_names", []),
    }

    result["scm_results"] = {
        cid: deserialize_scm(scm_d)
        for cid, scm_d in d.get("scm_results", {}).items()
    }

    alignments: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, al_d in d.get("alignment_results", {}).items():
        if "__" in key:
            parts = key.split("__", 1)
            pair = (parts[0], parts[1])
        else:
            pair = (key, "")

        restored: Dict[str, Any] = {}
        for ak, av in al_d.items():
            if isinstance(av, dict) and av.get("__type__") == "ndarray":
                restored[ak] = deserialize_numpy(av)
            elif isinstance(av, list) and ak == "permutation":
                restored[ak] = np.array(av)
            else:
                restored[ak] = av
        alignments[pair] = restored

    result["alignment_results"] = alignments
    result["descriptors"] = d.get("descriptors", {})

    return result


# =====================================================================
# Full atlas serialization
# =====================================================================


def serialize_atlas(atlas: Any, encoding: str = "list") -> Dict[str, Any]:
    """Serialize a complete AtlasResult to a JSON-safe dictionary.

    Parameters
    ----------
    atlas : AtlasResult or dict
        Atlas to serialize.
    encoding : str
        Numpy encoding method ('list', 'base64', 'flat').

    Returns
    -------
    dict
        Fully JSON-safe atlas representation.
    """
    if hasattr(atlas, "to_dict"):
        data = atlas.to_dict()
    elif isinstance(atlas, dict):
        data = dict(atlas)
    else:
        raise TypeError(f"Cannot serialize atlas of type {type(atlas)}")

    result: Dict[str, Any] = {
        "__type__": "atlas",
        "__version__": _SERIALIZATION_VERSION,
        "metadata": data.get("metadata", {}),
        "config": data.get("config", {}),
        "summary": data.get("summary", {}),
    }

    def _convert_arrays(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return serialize_numpy(obj, encoding=encoding)
        if isinstance(obj, dict):
            return {k: _convert_arrays(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert_arrays(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, "value"):
            return obj.value
        return obj

    if "foundation" in data:
        result["foundation"] = _convert_arrays(data["foundation"])
    if "exploration" in data:
        result["exploration"] = _convert_arrays(data["exploration"])
    if "validation" in data:
        result["validation"] = _convert_arrays(data["validation"])

    return result


def deserialize_atlas(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize an atlas from a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary produced by :func:`serialize_atlas`.

    Returns
    -------
    dict
        Deserialized atlas with numpy arrays restored.
    """
    def _restore_arrays(obj: Any) -> Any:
        if isinstance(obj, dict):
            if obj.get("__type__") == "ndarray":
                return deserialize_numpy(obj)
            if obj.get("__type__") == "sparse_matrix":
                return deserialize_sparse_matrix(obj)
            return {k: _restore_arrays(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_restore_arrays(v) for v in obj]
        return obj

    result: Dict[str, Any] = {
        "metadata": d.get("metadata", {}),
        "config": d.get("config", {}),
        "summary": d.get("summary", {}),
    }

    if "foundation" in d:
        result["foundation"] = _restore_arrays(d["foundation"])
    if "exploration" in d:
        result["exploration"] = _restore_arrays(d["exploration"])
    if "validation" in d:
        result["validation"] = _restore_arrays(d["validation"])

    return result


def save_atlas(
    atlas: Any,
    path: Union[str, Path],
    encoding: str = "list",
) -> Path:
    """Serialize and save an atlas to a JSON file.

    Parameters
    ----------
    atlas : AtlasResult or dict
        Atlas to save.
    path : str or Path
        Output file path.
    encoding : str
        Numpy encoding method.

    Returns
    -------
    Path
    """
    serialized = serialize_atlas(atlas, encoding=encoding)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(serialized, indent=2, default=str)
    out.write_text(text)
    return out.resolve()


def load_atlas(path: Union[str, Path]) -> Dict[str, Any]:
    """Load an atlas from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file.

    Returns
    -------
    dict
        Deserialized atlas.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Atlas file not found: {p}")
    data = json.loads(p.read_text())
    return deserialize_atlas(data)


# =====================================================================
# Version checking
# =====================================================================


def check_version(d: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if a serialized object's version is compatible.

    Parameters
    ----------
    d : dict
        Serialized object with __version__ key.

    Returns
    -------
    compatible : bool
        Whether the version is compatible.
    message : str
        Version information message.
    """
    stored = d.get("__version__", "0.0.0")
    current = _SERIALIZATION_VERSION

    stored_parts = [int(x) for x in stored.split(".")]
    current_parts = [int(x) for x in current.split(".")]

    if stored_parts[0] != current_parts[0]:
        return False, (
            f"Major version mismatch: file={stored}, current={current}"
        )

    if stored_parts[1] > current_parts[1]:
        return True, (
            f"File uses newer minor version: file={stored}, current={current}. "
            "Some features may not be available."
        )

    return True, f"Version compatible: file={stored}, current={current}"
