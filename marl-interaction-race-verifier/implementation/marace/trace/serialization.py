"""Trace serialization: JSON, binary, streaming, compression, fingerprinting.

Supports two formats:
  * **JSON** — human-readable, used for debugging and interchange.
  * **Binary** — compact msgpack-style encoding (uses ``struct`` + ``zlib``
    so no external dependency is required).

Both formats embed enough metadata to reconstruct the full
``ExecutionTrace`` including numpy arrays.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import struct
import time
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, TextIO, Tuple

import numpy as np

from .events import Event, event_from_dict
from .trace import ExecutionTrace


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, frozenset):
            return list(obj)
        return super().default(obj)


def _numpy_hook(d: Dict[str, Any]) -> Any:
    if "__ndarray__" in d:
        return np.array(d["data"], dtype=d["dtype"])
    return d


class TraceSerialization:
    """Serialize / deserialize ``ExecutionTrace`` objects."""

    # ---- JSON -------------------------------------------------------------
    @staticmethod
    def to_json(trace: ExecutionTrace, *, indent: int = 2) -> str:
        """Serialize *trace* to a JSON string."""
        payload = {
            "trace_id": trace.trace_id,
            "agents": sorted(trace.agents),
            "event_count": len(trace),
            "events": [e.to_dict() for e in trace],
        }
        return json.dumps(payload, cls=_NumpyEncoder, indent=indent)

    @staticmethod
    def from_json(data: str) -> ExecutionTrace:
        """Deserialize an ``ExecutionTrace`` from a JSON string."""
        payload = json.loads(data, object_hook=_numpy_hook)
        trace = ExecutionTrace(
            trace_id=payload.get("trace_id", ""),
            agents=payload.get("agents"),
        )
        for ed in payload["events"]:
            trace.append_event(event_from_dict(ed))
        return trace

    # ---- JSON file ---------------------------------------------------------
    @staticmethod
    def save_json(trace: ExecutionTrace, path: str, *, compress: bool = False) -> None:
        """Write *trace* to a JSON file, optionally gzip-compressed."""
        raw = TraceSerialization.to_json(trace).encode("utf-8")
        if compress:
            with gzip.open(path, "wb") as f:
                f.write(raw)
        else:
            with open(path, "w") as f:
                f.write(raw.decode("utf-8"))

    @staticmethod
    def load_json(path: str, *, compressed: bool = False) -> ExecutionTrace:
        if compressed:
            with gzip.open(path, "rb") as f:
                raw = f.read().decode("utf-8")
        else:
            with open(path, "r") as f:
                raw = f.read()
        return TraceSerialization.from_json(raw)

    # ---- Binary format -----------------------------------------------------
    @staticmethod
    def to_binary(trace: ExecutionTrace) -> bytes:
        """Compact binary encoding using JSON + zlib (no msgpack dependency).

        Layout::
            [4 bytes: magic 'MRCB']
            [4 bytes: version (uint32 big-endian)]
            [4 bytes: uncompressed size (uint32 big-endian)]
            [rest   : zlib-compressed JSON payload]
        """
        import zlib

        json_bytes = TraceSerialization.to_json(trace, indent=None).encode("utf-8")
        compressed = zlib.compress(json_bytes, level=6)
        header = b"MRCB" + struct.pack(">II", 1, len(json_bytes))
        return header + compressed

    @staticmethod
    def from_binary(data: bytes) -> ExecutionTrace:
        import zlib

        if data[:4] != b"MRCB":
            raise ValueError("Invalid binary trace: bad magic bytes")
        version, raw_len = struct.unpack(">II", data[4:12])
        if version != 1:
            raise ValueError(f"Unsupported binary trace version {version}")
        decompressed = zlib.decompress(data[12:])
        if len(decompressed) != raw_len:
            raise ValueError(
                f"Size mismatch: expected {raw_len}, got {len(decompressed)}"
            )
        return TraceSerialization.from_json(decompressed.decode("utf-8"))

    @staticmethod
    def save_binary(trace: ExecutionTrace, path: str) -> None:
        with open(path, "wb") as f:
            f.write(TraceSerialization.to_binary(trace))

    @staticmethod
    def load_binary(path: str) -> ExecutionTrace:
        with open(path, "rb") as f:
            return TraceSerialization.from_binary(f.read())

    # ---- Fingerprint -------------------------------------------------------
    @staticmethod
    def fingerprint(trace: ExecutionTrace) -> str:
        """SHA-256 fingerprint of the trace content.

        The fingerprint is computed over a deterministic canonical JSON
        representation so that two traces with identical events (same ids,
        same order) always produce the same hash.
        """
        canonical = TraceSerialization.to_json(trace, indent=None)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Streaming serializer / deserializer
# ---------------------------------------------------------------------------
class StreamingTraceWriter:
    """Write events one-at-a-time to a file (JSON Lines format).

    Useful for very large traces that should not be held in memory.
    Each line is a self-contained JSON object.
    """

    def __init__(self, path: str, *, compress: bool = False):
        self._compress = compress
        if compress:
            self._fh: Any = gzip.open(path, "wt", encoding="utf-8")
        else:
            self._fh = open(path, "w")
        # Write header line
        self._fh.write(json.dumps({"__header__": True, "version": 1}) + "\n")
        self._count = 0

    def write_event(self, event: Event) -> None:
        line = json.dumps(event.to_dict(), cls=_NumpyEncoder)
        self._fh.write(line + "\n")
        self._count += 1

    def close(self) -> None:
        self._fh.write(
            json.dumps({"__footer__": True, "event_count": self._count}) + "\n"
        )
        self._fh.close()

    def __enter__(self) -> "StreamingTraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class StreamingTraceReader:
    """Read events one-at-a-time from a JSON Lines file.

    Yields ``Event`` objects via ``__iter__`` without loading the
    entire file into memory.
    """

    def __init__(self, path: str, *, compressed: bool = False):
        self._path = path
        self._compressed = compressed

    def __iter__(self) -> Iterator[Event]:
        if self._compressed:
            fh: Any = gzip.open(self._path, "rt", encoding="utf-8")
        else:
            fh = open(self._path, "r")

        try:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line, object_hook=_numpy_hook)
                if "__header__" in obj or "__footer__" in obj:
                    continue
                yield event_from_dict(obj)
        finally:
            fh.close()

    def to_trace(self, trace_id: str = "") -> ExecutionTrace:
        """Materialise the full trace into memory."""
        trace = ExecutionTrace(trace_id=trace_id)
        for event in self:
            trace.append_event(event)
        return trace


# ---------------------------------------------------------------------------
# Trace comparison utilities
# ---------------------------------------------------------------------------
def traces_equivalent(a: ExecutionTrace, b: ExecutionTrace) -> bool:
    """Quick check: same fingerprint ⟹ same content."""
    return TraceSerialization.fingerprint(a) == TraceSerialization.fingerprint(b)


def trace_diff_summary(
    a: ExecutionTrace, b: ExecutionTrace
) -> Dict[str, Any]:
    """Return a summary of differences between two traces."""
    ids_a = {e.event_id for e in a}
    ids_b = {e.event_id for e in b}
    return {
        "only_in_a": len(ids_a - ids_b),
        "only_in_b": len(ids_b - ids_a),
        "common": len(ids_a & ids_b),
        "same_fingerprint": traces_equivalent(a, b),
    }
