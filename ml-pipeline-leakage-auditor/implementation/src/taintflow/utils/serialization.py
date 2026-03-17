"""
taintflow.utils.serialization – Serialization utilities for PI-DAG and reports.

Supports MessagePack (via ``msgpack``), JSON, and pickle formats with
schema versioning, optional compression (gzip, lz4), and streaming
serialization for large DAGs.
"""

from __future__ import annotations

import enum
import gzip
import io
import json
import math
import pickle
import struct
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

__all__ = [
    "Serializer",
    "MsgPackSerializer",
    "JSONSerializer",
    "PickleSerializer",
    "serialize_dag",
    "deserialize_dag",
    "serialize_report",
    "deserialize_report",
    "validate_schema",
    "CompressionKind",
]

# ---------------------------------------------------------------------------
# Constants / version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
MAGIC_BYTES = b"TFLO"  # TaintFlow magic header

# ---------------------------------------------------------------------------
# Compression support
# ---------------------------------------------------------------------------


class CompressionKind(enum.Enum):
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


def _compress(data: bytes, kind: CompressionKind, level: int = 6) -> bytes:
    """Compress *data* with the given algorithm."""
    if kind == CompressionKind.NONE:
        return data
    if kind == CompressionKind.GZIP:
        return gzip.compress(data, compresslevel=level)
    if kind == CompressionKind.LZ4:
        try:
            import lz4.frame  # type: ignore[import-untyped]

            return lz4.frame.compress(data, compression_level=level)
        except ImportError:
            # fallback to gzip
            return gzip.compress(data, compresslevel=level)
    raise ValueError(f"unknown compression: {kind}")


def _decompress(data: bytes, kind: CompressionKind) -> bytes:
    """Decompress *data*."""
    if kind == CompressionKind.NONE:
        return data
    if kind == CompressionKind.GZIP:
        return gzip.decompress(data)
    if kind == CompressionKind.LZ4:
        try:
            import lz4.frame  # type: ignore[import-untyped]

            return lz4.frame.decompress(data)
        except ImportError:
            # Might have been compressed with gzip fallback
            return gzip.decompress(data)
    raise ValueError(f"unknown compression: {kind}")


# ---------------------------------------------------------------------------
# Custom JSON encoder
# ---------------------------------------------------------------------------


class TaintFlowJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types, enums, dataclasses, frozensets."""

    def default(self, obj: Any) -> Any:
        # Enums
        if isinstance(obj, enum.Enum):
            return {"__enum__": f"{type(obj).__qualname__}.{obj.name}"}

        # Dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return {
                "__dataclass__": type(obj).__qualname__,
                **{f.name: getattr(obj, f.name) for f in fields(obj)},
            }

        # frozenset
        if isinstance(obj, frozenset):
            return {"__frozenset__": sorted(self.default(x) if not isinstance(x, (str, int, float, bool)) else x for x in obj)}

        # set
        if isinstance(obj, set):
            return {"__set__": sorted(self.default(x) if not isinstance(x, (str, int, float, bool)) else x for x in obj)}

        # bytes
        if isinstance(obj, bytes):
            import base64
            return {"__bytes__": base64.b64encode(obj).decode("ascii")}

        # numpy scalar types
        try:
            import numpy as np  # type: ignore[import-untyped]

            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return {"__ndarray__": obj.tolist(), "__dtype__": str(obj.dtype)}
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass

        # inf / nan
        if isinstance(obj, float):
            if math.isinf(obj):
                return {"__float__": "inf" if obj > 0 else "-inf"}
            if math.isnan(obj):
                return {"__float__": "nan"}

        return super().default(obj)


class TaintFlowJSONDecoder(json.JSONDecoder):
    """JSON decoder that reconstructs custom types."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["object_hook"] = self._object_hook
        super().__init__(**kwargs)

    @staticmethod
    def _object_hook(obj: Dict[str, Any]) -> Any:
        if "__frozenset__" in obj:
            return frozenset(obj["__frozenset__"])
        if "__set__" in obj:
            return set(obj["__set__"])
        if "__bytes__" in obj:
            import base64
            return base64.b64decode(obj["__bytes__"])
        if "__float__" in obj:
            val = obj["__float__"]
            if val == "inf":
                return float("inf")
            if val == "-inf":
                return float("-inf")
            if val == "nan":
                return float("nan")
        if "__ndarray__" in obj:
            try:
                import numpy as np  # type: ignore[import-untyped]
                return np.array(obj["__ndarray__"], dtype=obj.get("__dtype__"))
            except ImportError:
                return obj["__ndarray__"]
        if "__enum__" in obj:
            return obj  # Leave as tagged dict for consumer to resolve
        if "__dataclass__" in obj:
            return obj  # Leave as tagged dict for consumer to resolve
        return obj


# ---------------------------------------------------------------------------
# Serializer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Serializer(Protocol):
    """Protocol for serialization backends."""

    def dumps(self, obj: Any) -> bytes: ...
    def loads(self, data: bytes) -> Any: ...
    @property
    def format_tag(self) -> bytes: ...


# ---------------------------------------------------------------------------
# MsgPackSerializer
# ---------------------------------------------------------------------------


class MsgPackSerializer:
    """Serializer using MessagePack format.

    Requires the ``msgpack`` library (falls back to JSON if unavailable).
    """

    def __init__(self) -> None:
        try:
            import msgpack  # type: ignore[import-untyped]

            self._msgpack = msgpack
            self._available = True
        except ImportError:
            self._available = False

    @property
    def format_tag(self) -> bytes:
        return b"MP"

    def _encode_hook(self, obj: Any) -> Any:
        """Custom encoder for msgpack."""
        if isinstance(obj, enum.Enum):
            return {"__enum__": f"{type(obj).__qualname__}.{obj.name}"}
        if is_dataclass(obj) and not isinstance(obj, type):
            d = {"__dataclass__": type(obj).__qualname__}
            for f in fields(obj):
                d[f.name] = getattr(obj, f.name)
            return d
        if isinstance(obj, frozenset):
            return {"__frozenset__": sorted(str(x) for x in obj)}
        if isinstance(obj, set):
            return {"__set__": sorted(str(x) for x in obj)}
        try:
            import numpy as np  # type: ignore[import-untyped]

            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return {"__ndarray__": obj.tolist(), "__dtype__": str(obj.dtype)}
        except ImportError:
            pass
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            if math.isinf(obj):
                return {"__float__": "inf" if obj > 0 else "-inf"}
            return {"__float__": "nan"}
        return obj

    def _decode_hook(self, obj: Any) -> Any:
        """Custom decoder for msgpack."""
        if isinstance(obj, dict):
            if "__frozenset__" in obj:
                return frozenset(obj["__frozenset__"])
            if "__set__" in obj:
                return set(obj["__set__"])
            if "__float__" in obj:
                v = obj["__float__"]
                if v == "inf":
                    return float("inf")
                if v == "-inf":
                    return float("-inf")
                return float("nan")
            if "__ndarray__" in obj:
                try:
                    import numpy as np  # type: ignore[import-untyped]
                    return np.array(obj["__ndarray__"], dtype=obj.get("__dtype__"))
                except ImportError:
                    return obj["__ndarray__"]
        return obj

    def _recursive_encode(self, obj: Any) -> Any:
        """Recursively encode an object tree."""
        if isinstance(obj, dict):
            return {k: self._recursive_encode(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._recursive_encode(v) for v in obj]
        return self._encode_hook(obj)

    def _recursive_decode(self, obj: Any) -> Any:
        """Recursively decode an object tree."""
        if isinstance(obj, dict):
            decoded = {k: self._recursive_decode(v) for k, v in obj.items()}
            return self._decode_hook(decoded)
        if isinstance(obj, (list, tuple)):
            return [self._recursive_decode(v) for v in obj]
        return obj

    def dumps(self, obj: Any) -> bytes:
        encoded = self._recursive_encode(obj)
        if self._available:
            return self._msgpack.packb(encoded, use_bin_type=True)
        # Fallback to JSON
        return json.dumps(encoded, cls=TaintFlowJSONEncoder).encode("utf-8")

    def loads(self, data: bytes) -> Any:
        if self._available:
            raw = self._msgpack.unpackb(data, raw=False)
            return self._recursive_decode(raw)
        # Fallback
        raw = json.loads(data.decode("utf-8"), cls=TaintFlowJSONDecoder)
        return self._recursive_decode(raw)


# ---------------------------------------------------------------------------
# JSONSerializer
# ---------------------------------------------------------------------------


class JSONSerializer:
    """Serializer using JSON with custom type encoding."""

    @property
    def format_tag(self) -> bytes:
        return b"JS"

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(
            obj,
            cls=TaintFlowJSONEncoder,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def loads(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"), cls=TaintFlowJSONDecoder)


# ---------------------------------------------------------------------------
# PickleSerializer
# ---------------------------------------------------------------------------


class PickleSerializer:
    """Serializer using Python pickle (protocol 5)."""

    def __init__(self, protocol: int = 5) -> None:
        self._protocol = min(protocol, pickle.HIGHEST_PROTOCOL)

    @property
    def format_tag(self) -> bytes:
        return b"PK"

    def dumps(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=self._protocol)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)  # noqa: S301


# ---------------------------------------------------------------------------
# Wire format: MAGIC(4) | VERSION(2) | FORMAT(2) | COMPRESSION(1) | LEN(4) | DATA
# ---------------------------------------------------------------------------

_HEADER_FMT = "<4sHH1sI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _pack_envelope(
    payload: bytes,
    serializer: Serializer,
    compression: CompressionKind = CompressionKind.NONE,
) -> bytes:
    """Wrap *payload* in the TaintFlow envelope."""
    compressed = _compress(payload, compression)
    comp_byte = {
        CompressionKind.NONE: b"\x00",
        CompressionKind.GZIP: b"\x01",
        CompressionKind.LZ4: b"\x02",
    }[compression]
    header = struct.pack(
        _HEADER_FMT,
        MAGIC_BYTES,
        SCHEMA_VERSION,
        int.from_bytes(serializer.format_tag, "little"),
        comp_byte,
        len(compressed),
    )
    return header + compressed


def _unpack_envelope(data: bytes) -> Tuple[int, int, CompressionKind, bytes]:
    """Unpack envelope; return (version, format_code, compression, payload)."""
    if len(data) < _HEADER_SIZE:
        raise ValueError("data too short for TaintFlow envelope")
    magic, version, fmt_code, comp_byte, length = struct.unpack_from(
        _HEADER_FMT, data, 0
    )
    if magic != MAGIC_BYTES:
        raise ValueError(f"bad magic bytes: {magic!r}")
    comp_map = {b"\x00": CompressionKind.NONE, b"\x01": CompressionKind.GZIP, b"\x02": CompressionKind.LZ4}
    compression = comp_map.get(comp_byte, CompressionKind.NONE)
    payload = data[_HEADER_SIZE : _HEADER_SIZE + length]
    decompressed = _decompress(payload, compression)
    return version, fmt_code, compression, decompressed


def _get_serializer_for_fmt(fmt_code: int) -> Serializer:
    """Return a serializer matching *fmt_code*."""
    tag = fmt_code.to_bytes(2, "little")
    if tag == b"MP":
        return MsgPackSerializer()
    if tag == b"JS":
        return JSONSerializer()
    if tag == b"PK":
        return PickleSerializer()
    raise ValueError(f"unknown format tag: {tag!r}")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_DAG_REQUIRED_KEYS = {"nodes", "edges", "metadata"}
_NODE_REQUIRED_KEYS = {"id", "op_type"}
_EDGE_REQUIRED_KEYS = {"src", "dst"}


def validate_schema(obj: Any, kind: str = "dag") -> List[str]:
    """Validate a deserialized object against the TaintFlow schema.

    Parameters
    ----------
    obj : dict
        Deserialized dictionary.
    kind : str
        ``"dag"`` or ``"report"``.

    Returns
    -------
    list of str
        Validation error messages (empty if valid).
    """
    errors: List[str] = []
    if not isinstance(obj, dict):
        errors.append("root must be a dict")
        return errors

    if kind == "dag":
        missing = _DAG_REQUIRED_KEYS - set(obj)
        if missing:
            errors.append(f"missing top-level keys: {missing}")
        nodes = obj.get("nodes", [])
        if not isinstance(nodes, list):
            errors.append("'nodes' must be a list")
        else:
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    errors.append(f"node[{i}] must be a dict")
                    continue
                nmiss = _NODE_REQUIRED_KEYS - set(node)
                if nmiss:
                    errors.append(f"node[{i}] missing keys: {nmiss}")
        edges_list = obj.get("edges", [])
        if not isinstance(edges_list, list):
            errors.append("'edges' must be a list")
        else:
            for i, edge in enumerate(edges_list):
                if not isinstance(edge, dict):
                    errors.append(f"edge[{i}] must be a dict")
                    continue
                emiss = _EDGE_REQUIRED_KEYS - set(edge)
                if emiss:
                    errors.append(f"edge[{i}] missing keys: {emiss}")
        meta = obj.get("metadata")
        if meta is not None and not isinstance(meta, dict):
            errors.append("'metadata' must be a dict or null")

    elif kind == "report":
        for key in ("summary", "stages"):
            if key not in obj:
                errors.append(f"missing top-level key: {key!r}")
        stages = obj.get("stages", [])
        if not isinstance(stages, list):
            errors.append("'stages' must be a list")
        else:
            for i, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    errors.append(f"stage[{i}] must be a dict")
    else:
        errors.append(f"unknown schema kind: {kind!r}")

    return errors


# ---------------------------------------------------------------------------
# High-level API: serialize / deserialize DAG
# ---------------------------------------------------------------------------


def serialize_dag(
    dag: Dict[str, Any],
    serializer: Optional[Serializer] = None,
    compression: CompressionKind = CompressionKind.GZIP,
) -> bytes:
    """Serialize a PI-DAG dictionary to bytes with envelope.

    Parameters
    ----------
    dag : dict
        Must contain ``nodes``, ``edges``, ``metadata``.
    serializer : Serializer or None
        Defaults to :class:`MsgPackSerializer`.
    compression : CompressionKind
        Compression algorithm.

    Returns
    -------
    bytes
        Envelope + compressed payload.
    """
    if serializer is None:
        serializer = MsgPackSerializer()
    payload = serializer.dumps(dag)
    return _pack_envelope(payload, serializer, compression)


def deserialize_dag(
    data: bytes,
    validate: bool = True,
) -> Dict[str, Any]:
    """Deserialize a PI-DAG from bytes.

    Parameters
    ----------
    data : bytes
        Must begin with the TaintFlow magic header.
    validate : bool
        If True, validate against the DAG schema.

    Returns
    -------
    dict
        The deserialized DAG dictionary.

    Raises
    ------
    ValueError
        If validation fails.
    """
    version, fmt_code, compression, payload = _unpack_envelope(data)
    ser = _get_serializer_for_fmt(fmt_code)
    obj = ser.loads(payload)
    if validate:
        errors = validate_schema(obj, "dag")
        if errors:
            raise ValueError(f"DAG schema validation failed: {errors}")
    return obj


# ---------------------------------------------------------------------------
# High-level API: serialize / deserialize report
# ---------------------------------------------------------------------------


def serialize_report(
    report: Dict[str, Any],
    serializer: Optional[Serializer] = None,
    compression: CompressionKind = CompressionKind.GZIP,
) -> bytes:
    """Serialize a leakage report to bytes."""
    if serializer is None:
        serializer = JSONSerializer()
    payload = serializer.dumps(report)
    return _pack_envelope(payload, serializer, compression)


def deserialize_report(
    data: bytes,
    validate: bool = True,
) -> Dict[str, Any]:
    """Deserialize a leakage report from bytes."""
    version, fmt_code, compression, payload = _unpack_envelope(data)
    ser = _get_serializer_for_fmt(fmt_code)
    obj = ser.loads(payload)
    if validate:
        errors = validate_schema(obj, "report")
        if errors:
            raise ValueError(f"Report schema validation failed: {errors}")
    return obj


# ---------------------------------------------------------------------------
# Streaming serialization for large DAGs
# ---------------------------------------------------------------------------


class StreamingSerializer:
    """Write DAG nodes/edges incrementally to a binary stream.

    Useful for very large graphs that should not be fully materialised
    in memory.

    Wire format per record::

        TYPE(1) | LENGTH(4) | JSON_PAYLOAD(LENGTH)

    Where TYPE is ``0x01`` for a node, ``0x02`` for an edge, ``0x03`` for
    metadata, ``0xFF`` for end-of-stream.
    """

    NODE_TYPE = 0x01
    EDGE_TYPE = 0x02
    META_TYPE = 0x03
    EOS_TYPE = 0xFF

    def __init__(
        self,
        stream: BinaryIO,
        compression: CompressionKind = CompressionKind.NONE,
    ) -> None:
        self._stream = stream
        self._compression = compression
        self._encoder = TaintFlowJSONEncoder(separators=(",", ":"))
        # Write header
        self._stream.write(MAGIC_BYTES)
        self._stream.write(struct.pack("<H", SCHEMA_VERSION))
        comp_byte = {
            CompressionKind.NONE: 0,
            CompressionKind.GZIP: 1,
            CompressionKind.LZ4: 2,
        }[compression]
        self._stream.write(struct.pack("<B", comp_byte))
        self._count = 0

    def _write_record(self, rtype: int, obj: Any) -> None:
        payload = self._encoder.encode(obj).encode("utf-8")
        if self._compression != CompressionKind.NONE:
            payload = _compress(payload, self._compression, level=1)
        self._stream.write(struct.pack("<BI", rtype, len(payload)))
        self._stream.write(payload)
        self._count += 1

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        self._write_record(self.META_TYPE, metadata)

    def write_node(self, node: Dict[str, Any]) -> None:
        self._write_record(self.NODE_TYPE, node)

    def write_edge(self, edge: Dict[str, Any]) -> None:
        self._write_record(self.EDGE_TYPE, edge)

    def finish(self) -> None:
        """Write end-of-stream marker and flush."""
        self._stream.write(struct.pack("<BI", self.EOS_TYPE, 0))
        self._stream.flush()

    @property
    def records_written(self) -> int:
        return self._count


class StreamingDeserializer:
    """Read DAG nodes/edges incrementally from a binary stream."""

    NODE_TYPE = StreamingSerializer.NODE_TYPE
    EDGE_TYPE = StreamingSerializer.EDGE_TYPE
    META_TYPE = StreamingSerializer.META_TYPE
    EOS_TYPE = StreamingSerializer.EOS_TYPE

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        # Read header
        magic = stream.read(4)
        if magic != MAGIC_BYTES:
            raise ValueError(f"bad magic: {magic!r}")
        (self._version,) = struct.unpack("<H", stream.read(2))
        (comp_byte,) = struct.unpack("<B", stream.read(1))
        comp_map = {0: CompressionKind.NONE, 1: CompressionKind.GZIP, 2: CompressionKind.LZ4}
        self._compression = comp_map.get(comp_byte, CompressionKind.NONE)

    def __iter__(self) -> Iterator[Tuple[int, Any]]:
        return self

    def __next__(self) -> Tuple[int, Any]:
        header = self._stream.read(5)
        if len(header) < 5:
            raise StopIteration
        rtype, length = struct.unpack("<BI", header)
        if rtype == self.EOS_TYPE:
            raise StopIteration
        payload = self._stream.read(length)
        if self._compression != CompressionKind.NONE:
            payload = _decompress(payload, self._compression)
        obj = json.loads(payload.decode("utf-8"), cls=TaintFlowJSONDecoder)
        return rtype, obj

    def read_all(self) -> Dict[str, Any]:
        """Read entire stream into a DAG dictionary."""
        nodes: List[Any] = []
        edges_list: List[Any] = []
        metadata: Dict[str, Any] = {}
        for rtype, obj in self:
            if rtype == self.NODE_TYPE:
                nodes.append(obj)
            elif rtype == self.EDGE_TYPE:
                edges_list.append(obj)
            elif rtype == self.META_TYPE:
                metadata.update(obj)
        return {"nodes": nodes, "edges": edges_list, "metadata": metadata}
