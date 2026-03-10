"""
usability_oracle.utils.serialization — JSON serialisation utilities.

Custom encoders for numpy arrays, dataclasses, and enums, plus
convenience functions for saving/loading objects to/from files.
"""

from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np


# ---------------------------------------------------------------------------
# NumpyEncoder
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types.

    Converts:
      - ``np.ndarray`` → list (nested)
      - ``np.integer`` → int
      - ``np.floating`` → float
      - ``np.bool_`` → bool
      - ``np.complexfloating`` → dict with "real" and "imag"
      - ``np.void`` → None
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            val = float(o)
            if np.isnan(val):
                return None
            if np.isinf(val):
                return 1e308 if val > 0 else -1e308
            return val
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.complexfloating):
            return {"real": float(o.real), "imag": float(o.imag)}
        if isinstance(o, np.void):
            return None
        return super().default(o)


# ---------------------------------------------------------------------------
# DataclassEncoder
# ---------------------------------------------------------------------------

class DataclassEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses and enums.

    Falls back to :class:`NumpyEncoder` for numpy types.
    """

    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, enum.Enum):
            return o.value
        if isinstance(o, set):
            return sorted(o, key=str)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return super().default(o)


# ---------------------------------------------------------------------------
# Combined encoder
# ---------------------------------------------------------------------------

class _CombinedEncoder(json.JSONEncoder):
    """Combines Numpy + Dataclass encoding."""

    def default(self, o: Any) -> Any:
        # Dataclass
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        # Enum
        if isinstance(o, enum.Enum):
            return o.value
        # Set
        if isinstance(o, set):
            return sorted(o, key=str)
        # Path
        if isinstance(o, Path):
            return str(o)
        # Numpy
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            val = float(o)
            return None if np.isnan(val) else val
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.complexfloating):
            return {"real": float(o.real), "imag": float(o.imag)}
        # Custom to_dict
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return super().default(o)


# ---------------------------------------------------------------------------
# Serialisation functions
# ---------------------------------------------------------------------------

def serialize_to_json(obj: Any, indent: int = 2) -> str:
    """Serialise *obj* to a JSON string, handling numpy/dataclass/enum types."""
    return json.dumps(obj, cls=_CombinedEncoder, indent=indent, ensure_ascii=False)


def deserialize_from_json(json_str: str, cls: Optional[Type] = None) -> Any:
    """Deserialise a JSON string, optionally constructing a *cls* instance.

    If *cls* is a dataclass, it will be instantiated from the parsed dict.
    Otherwise the raw parsed value is returned.
    """
    data = json.loads(json_str)
    if cls is None:
        return data
    if dataclasses.is_dataclass(cls):
        return _dict_to_dataclass(data, cls)
    return cls(data) if callable(cls) else data


def _dict_to_dataclass(data: dict[str, Any], cls: Type) -> Any:
    """Recursively convert a dict to a dataclass instance."""
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in field_types:
            kwargs[key] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_to_file(obj: Any, path: Path, indent: int = 2) -> None:
    """Serialise *obj* and write to *path* as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = serialize_to_json(obj, indent=indent)
    path.write_text(text, encoding="utf-8")


def load_from_file(path: Path, cls: Optional[Type] = None) -> Any:
    """Read JSON from *path* and deserialise.

    Parameters:
        path: File path.
        cls: Optional target type for deserialisation.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return deserialize_from_json(text, cls=cls)
