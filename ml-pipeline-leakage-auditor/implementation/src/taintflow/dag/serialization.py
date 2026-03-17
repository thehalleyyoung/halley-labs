"""
taintflow.dag.serialization -- DAG serialization and deserialization.

Supports JSON and MessagePack formats with schema versioning, backward
compatibility, compression, and streaming serialization for large DAGs.
"""

from __future__ import annotations

import gzip
import io
import json
import struct
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

from taintflow.core.types import (
    ColumnSchema,
    EdgeKind,
    OpType,
    Origin,
    ProvenanceInfo,
    ShapeMetadata,
)
from taintflow.core.errors import DAGConstructionError
from taintflow.dag.node import PipelineNode, SourceLocation
from taintflow.dag.edge import PipelineEdge, EdgeSet
from taintflow.dag.pidag import PIDAG

# Schema version for serialized DAGs
SCHEMA_VERSION = "1.0.0"
_SUPPORTED_VERSIONS = {"1.0.0", "0.9.0", "0.8.0"}
_MAGIC_BYTES = b"TFDAG"
_MSGPACK_HEADER_SIZE = 16


# ===================================================================
#  JSON serialization
# ===================================================================


def serialize_pidag_json(
    dag: PIDAG,
    *,
    indent: int | None = 2,
    include_metadata: bool = True,
    compact: bool = False,
) -> str:
    """Serialize a PIDAG to a JSON string.

    Parameters
    ----------
    dag : PIDAG
        The DAG to serialize.
    indent : int | None
        JSON indentation level. None for compact.
    include_metadata : bool
        Whether to include node/edge metadata.
    compact : bool
        If True, use minimal keys and no whitespace.
    """
    data = _dag_to_serializable(dag, include_metadata=include_metadata)
    if compact:
        data = _compact_dict(data)
        return json.dumps(data, separators=(",", ":"), default=_json_default)
    return json.dumps(data, indent=indent, default=_json_default, sort_keys=False)


def deserialize_pidag_json(data: str) -> PIDAG:
    """Deserialize a PIDAG from a JSON string."""
    raw = json.loads(data)
    return _serializable_to_dag(raw)


def serialize_pidag_json_stream(dag: PIDAG, stream: TextIO) -> None:
    """Serialize a PIDAG to a text stream in JSON format.

    For large DAGs this avoids building the full string in memory
    by writing nodes and edges incrementally.
    """
    stream.write('{"schema_version":"')
    stream.write(SCHEMA_VERSION)
    stream.write('","n_nodes":')
    stream.write(str(dag.n_nodes))
    stream.write(',"n_edges":')
    stream.write(str(dag.n_edges))
    stream.write(',"nodes":{')

    first = True
    for nid, node in dag.nodes.items():
        if not first:
            stream.write(",")
        first = False
        stream.write(json.dumps(nid))
        stream.write(":")
        stream.write(json.dumps(node.to_dict(), default=_json_default))

    stream.write('},"edges":[')

    first = True
    for edge in dag.edges:
        if not first:
            stream.write(",")
        first = False
        stream.write(json.dumps(edge.to_dict(), default=_json_default))

    stream.write('],"metadata":')
    stream.write(json.dumps(dag.metadata, default=_json_default))
    stream.write("}")


def deserialize_pidag_json_stream(stream: TextIO) -> PIDAG:
    """Deserialize a PIDAG from a JSON text stream."""
    raw = json.load(stream)
    return _serializable_to_dag(raw)


# ===================================================================
#  MessagePack serialization
# ===================================================================


def serialize_pidag_msgpack(
    dag: PIDAG,
    *,
    compress: bool = True,
    include_metadata: bool = True,
) -> bytes:
    """Serialize a PIDAG to MessagePack bytes.

    Parameters
    ----------
    dag : PIDAG
        The DAG to serialize.
    compress : bool
        Whether to gzip-compress the output.
    include_metadata : bool
        Whether to include node/edge metadata.
    """
    data = _dag_to_serializable(dag, include_metadata=include_metadata)
    payload = _encode_msgpack(data)

    header = bytearray()
    header.extend(_MAGIC_BYTES)
    version_bytes = SCHEMA_VERSION.encode("utf-8")[:8]
    header.extend(version_bytes)
    header.extend(b"\x00" * (8 - len(version_bytes)))
    flags = 0
    if compress:
        flags |= 0x01
    header.append(flags)
    header.extend(b"\x00" * (16 - len(header)))

    if compress:
        payload = gzip.compress(payload, compresslevel=6)

    size_bytes = struct.pack("<I", len(payload))
    return bytes(header) + size_bytes + payload


def deserialize_pidag_msgpack(data: bytes) -> PIDAG:
    """Deserialize a PIDAG from MessagePack bytes."""
    if len(data) < _MSGPACK_HEADER_SIZE + 4:
        raise DAGConstructionError("Data too short for PIDAG msgpack format")

    header = data[:_MSGPACK_HEADER_SIZE]
    if header[:5] != _MAGIC_BYTES:
        raise DAGConstructionError("Invalid magic bytes in PIDAG msgpack data")

    version_raw = header[5:13].rstrip(b"\x00").decode("utf-8")
    if version_raw not in _SUPPORTED_VERSIONS:
        raise DAGConstructionError(f"Unsupported schema version: {version_raw!r}")

    flags = header[13]
    is_compressed = bool(flags & 0x01)

    size = struct.unpack("<I", data[_MSGPACK_HEADER_SIZE:_MSGPACK_HEADER_SIZE + 4])[0]
    payload = data[_MSGPACK_HEADER_SIZE + 4:_MSGPACK_HEADER_SIZE + 4 + size]

    if is_compressed:
        payload = gzip.decompress(payload)

    raw = _decode_msgpack(payload)

    if version_raw != SCHEMA_VERSION:
        raw = _migrate_schema(raw, version_raw, SCHEMA_VERSION)

    return _serializable_to_dag(raw)


def serialize_pidag_msgpack_stream(dag: PIDAG, stream: BinaryIO, *, compress: bool = True) -> None:
    """Serialize a PIDAG to a binary stream in msgpack format."""
    encoded = serialize_pidag_msgpack(dag, compress=compress)
    stream.write(encoded)


def deserialize_pidag_msgpack_stream(stream: BinaryIO) -> PIDAG:
    """Deserialize a PIDAG from a binary stream."""
    data = stream.read()
    return deserialize_pidag_msgpack(data)


# ===================================================================
#  Compressed JSON
# ===================================================================


def serialize_pidag_json_compressed(dag: PIDAG) -> bytes:
    """Serialize to gzip-compressed JSON bytes."""
    json_str = serialize_pidag_json(dag, indent=None, compact=True)
    return gzip.compress(json_str.encode("utf-8"), compresslevel=6)


def deserialize_pidag_json_compressed(data: bytes) -> PIDAG:
    """Deserialize from gzip-compressed JSON bytes."""
    json_str = gzip.decompress(data).decode("utf-8")
    return deserialize_pidag_json(json_str)


# ===================================================================
#  Internal helpers
# ===================================================================


def _dag_to_serializable(
    dag: PIDAG,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Convert a PIDAG to a JSON-serializable dictionary."""
    nodes_dict: dict[str, Any] = {}
    for nid, node in dag.nodes.items():
        nd = node.to_dict()
        if not include_metadata:
            nd.pop("metadata", None)
        nodes_dict[nid] = nd

    edges_list: list[dict[str, Any]] = []
    for edge in dag.edges:
        ed = edge.to_dict()
        if not include_metadata:
            ed.pop("metadata", None)
        edges_list.append(ed)

    return {
        "schema_version": SCHEMA_VERSION,
        "n_nodes": dag.n_nodes,
        "n_edges": dag.n_edges,
        "nodes": nodes_dict,
        "edges": edges_list,
        "metadata": dag.metadata if include_metadata else {},
    }


def _serializable_to_dag(raw: dict[str, Any]) -> PIDAG:
    """Convert a deserialized dictionary back to a PIDAG."""
    version = raw.get("schema_version", SCHEMA_VERSION)
    if version not in _SUPPORTED_VERSIONS:
        raise DAGConstructionError(f"Unsupported schema version: {version!r}")

    if version != SCHEMA_VERSION:
        raw = _migrate_schema(raw, version, SCHEMA_VERSION)

    return PIDAG.from_dict(raw)


def _migrate_schema(
    data: dict[str, Any],
    from_version: str,
    to_version: str,
) -> dict[str, Any]:
    """Migrate serialized data between schema versions."""
    result = dict(data)

    if from_version == "0.8.0":
        result = _migrate_0_8_to_0_9(result)
        from_version = "0.9.0"

    if from_version == "0.9.0":
        result = _migrate_0_9_to_1_0(result)
        from_version = "1.0.0"

    result["schema_version"] = to_version
    return result


def _migrate_0_8_to_0_9(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from 0.8.0 to 0.9.0 schema.

    0.9.0 added:
    - provenance_fraction to edges (default 0.0)
    - metadata field on nodes (default {})
    """
    result = dict(data)
    edges = result.get("edges", [])
    for e in edges:
        if "provenance_fraction" not in e:
            e["provenance_fraction"] = 0.0
    nodes = result.get("nodes", {})
    for nid, nd in nodes.items():
        if "metadata" not in nd:
            nd["metadata"] = {}
    result["schema_version"] = "0.9.0"
    return result


def _migrate_0_9_to_1_0(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from 0.9.0 to 1.0.0 schema.

    1.0.0 added:
    - node_type field (default PipelineNode)
    - edge capacity field (default 0.0)
    """
    result = dict(data)
    nodes = result.get("nodes", {})
    for nid, nd in nodes.items():
        if "node_type" not in nd:
            nd["node_type"] = "PipelineNode"
    edges = result.get("edges", [])
    for e in edges:
        if "capacity" not in e:
            e["capacity"] = 0.0
    result["schema_version"] = "1.0.0"
    return result


def _compact_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Create a compact version of a serialized dict with short keys."""
    key_map = {
        "schema_version": "sv",
        "n_nodes": "nn",
        "n_edges": "ne",
        "nodes": "ns",
        "edges": "es",
        "metadata": "m",
        "node_type": "nt",
        "node_id": "id",
        "op_type": "op",
        "source_location": "sl",
        "input_schema": "is",
        "output_schema": "os",
        "shape": "sh",
        "provenance": "pv",
        "timestamp": "ts",
        "source_id": "si",
        "target_id": "ti",
        "columns": "c",
        "edge_kind": "ek",
        "capacity": "ca",
        "provenance_fraction": "pf",
    }
    return _remap_keys(d, key_map)


def _remap_keys(obj: Any, key_map: dict[str, str]) -> Any:
    """Recursively remap dictionary keys."""
    if isinstance(obj, dict):
        return {key_map.get(k, k): _remap_keys(v, key_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_remap_keys(item, key_map) for item in obj]
    return obj


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for non-standard types."""
    if isinstance(obj, frozenset):
        return sorted(str(x) for x in obj)
    if isinstance(obj, set):
        return sorted(str(x) for x in obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


# ===================================================================
#  Pure-Python msgpack encoder/decoder
# ===================================================================


def _encode_msgpack(obj: Any) -> bytes:
    """Encode a Python object to msgpack-compatible bytes.

    This is a minimal pure-Python implementation covering the types
    used by PI-DAG serialization (dict, list, str, int, float, bool, None, bytes).
    """
    buf = io.BytesIO()
    _pack(buf, obj)
    return buf.getvalue()


def _decode_msgpack(data: bytes) -> Any:
    """Decode msgpack bytes to a Python object."""
    buf = io.BytesIO(data)
    return _unpack(buf)


def _pack(buf: BinaryIO, obj: Any) -> None:
    """Pack a single object into the buffer."""
    if obj is None:
        buf.write(b"\xc0")
    elif isinstance(obj, bool):
        buf.write(b"\xc3" if obj else b"\xc2")
    elif isinstance(obj, int):
        _pack_int(buf, obj)
    elif isinstance(obj, float):
        buf.write(b"\xcb")
        buf.write(struct.pack(">d", obj))
    elif isinstance(obj, str):
        encoded = obj.encode("utf-8")
        _pack_str(buf, encoded)
    elif isinstance(obj, bytes):
        _pack_bin(buf, obj)
    elif isinstance(obj, (list, tuple)):
        _pack_array(buf, obj)
    elif isinstance(obj, dict):
        _pack_map(buf, obj)
    else:
        _pack(buf, str(obj))


def _pack_int(buf: BinaryIO, n: int) -> None:
    if 0 <= n <= 0x7F:
        buf.write(struct.pack("B", n))
    elif -32 <= n < 0:
        buf.write(struct.pack("b", n))
    elif 0 <= n <= 0xFF:
        buf.write(b"\xcc")
        buf.write(struct.pack("B", n))
    elif 0 <= n <= 0xFFFF:
        buf.write(b"\xcd")
        buf.write(struct.pack(">H", n))
    elif 0 <= n <= 0xFFFFFFFF:
        buf.write(b"\xce")
        buf.write(struct.pack(">I", n))
    elif 0 <= n <= 0xFFFFFFFFFFFFFFFF:
        buf.write(b"\xcf")
        buf.write(struct.pack(">Q", n))
    elif -0x80 <= n < 0:
        buf.write(b"\xd0")
        buf.write(struct.pack(">b", n))
    elif -0x8000 <= n < 0:
        buf.write(b"\xd1")
        buf.write(struct.pack(">h", n))
    elif -0x80000000 <= n < 0:
        buf.write(b"\xd2")
        buf.write(struct.pack(">i", n))
    else:
        buf.write(b"\xd3")
        buf.write(struct.pack(">q", n))


def _pack_str(buf: BinaryIO, encoded: bytes) -> None:
    n = len(encoded)
    if n <= 31:
        buf.write(struct.pack("B", 0xA0 | n))
    elif n <= 0xFF:
        buf.write(b"\xd9")
        buf.write(struct.pack("B", n))
    elif n <= 0xFFFF:
        buf.write(b"\xda")
        buf.write(struct.pack(">H", n))
    else:
        buf.write(b"\xdb")
        buf.write(struct.pack(">I", n))
    buf.write(encoded)


def _pack_bin(buf: BinaryIO, data: bytes) -> None:
    n = len(data)
    if n <= 0xFF:
        buf.write(b"\xc4")
        buf.write(struct.pack("B", n))
    elif n <= 0xFFFF:
        buf.write(b"\xc5")
        buf.write(struct.pack(">H", n))
    else:
        buf.write(b"\xc6")
        buf.write(struct.pack(">I", n))
    buf.write(data)


def _pack_array(buf: BinaryIO, arr: Sequence[Any]) -> None:
    n = len(arr)
    if n <= 15:
        buf.write(struct.pack("B", 0x90 | n))
    elif n <= 0xFFFF:
        buf.write(b"\xdc")
        buf.write(struct.pack(">H", n))
    else:
        buf.write(b"\xdd")
        buf.write(struct.pack(">I", n))
    for item in arr:
        _pack(buf, item)


def _pack_map(buf: BinaryIO, d: Mapping[str, Any]) -> None:
    n = len(d)
    if n <= 15:
        buf.write(struct.pack("B", 0x80 | n))
    elif n <= 0xFFFF:
        buf.write(b"\xde")
        buf.write(struct.pack(">H", n))
    else:
        buf.write(b"\xdf")
        buf.write(struct.pack(">I", n))
    for k, v in d.items():
        _pack(buf, k)
        _pack(buf, v)


def _unpack(buf: BinaryIO) -> Any:
    """Unpack a single object from the buffer."""
    b = buf.read(1)
    if not b:
        raise DAGConstructionError("Unexpected end of msgpack data")
    tag = b[0]

    # Positive fixint
    if tag <= 0x7F:
        return tag
    # Fixmap
    if 0x80 <= tag <= 0x8F:
        return _unpack_map(buf, tag & 0x0F)
    # Fixarray
    if 0x90 <= tag <= 0x9F:
        return _unpack_array(buf, tag & 0x0F)
    # Fixstr
    if 0xA0 <= tag <= 0xBF:
        length = tag & 0x1F
        return buf.read(length).decode("utf-8")
    # Nil
    if tag == 0xC0:
        return None
    # Bool
    if tag == 0xC2:
        return False
    if tag == 0xC3:
        return True
    # Bin8
    if tag == 0xC4:
        length = struct.unpack("B", buf.read(1))[0]
        return buf.read(length)
    # Bin16
    if tag == 0xC5:
        length = struct.unpack(">H", buf.read(2))[0]
        return buf.read(length)
    # Bin32
    if tag == 0xC6:
        length = struct.unpack(">I", buf.read(4))[0]
        return buf.read(length)
    # Float32
    if tag == 0xCA:
        return struct.unpack(">f", buf.read(4))[0]
    # Float64
    if tag == 0xCB:
        return struct.unpack(">d", buf.read(8))[0]
    # Uint8
    if tag == 0xCC:
        return struct.unpack("B", buf.read(1))[0]
    # Uint16
    if tag == 0xCD:
        return struct.unpack(">H", buf.read(2))[0]
    # Uint32
    if tag == 0xCE:
        return struct.unpack(">I", buf.read(4))[0]
    # Uint64
    if tag == 0xCF:
        return struct.unpack(">Q", buf.read(8))[0]
    # Int8
    if tag == 0xD0:
        return struct.unpack(">b", buf.read(1))[0]
    # Int16
    if tag == 0xD1:
        return struct.unpack(">h", buf.read(2))[0]
    # Int32
    if tag == 0xD2:
        return struct.unpack(">i", buf.read(4))[0]
    # Int64
    if tag == 0xD3:
        return struct.unpack(">q", buf.read(8))[0]
    # Str8
    if tag == 0xD9:
        length = struct.unpack("B", buf.read(1))[0]
        return buf.read(length).decode("utf-8")
    # Str16
    if tag == 0xDA:
        length = struct.unpack(">H", buf.read(2))[0]
        return buf.read(length).decode("utf-8")
    # Str32
    if tag == 0xDB:
        length = struct.unpack(">I", buf.read(4))[0]
        return buf.read(length).decode("utf-8")
    # Array16
    if tag == 0xDC:
        length = struct.unpack(">H", buf.read(2))[0]
        return _unpack_array(buf, length)
    # Array32
    if tag == 0xDD:
        length = struct.unpack(">I", buf.read(4))[0]
        return _unpack_array(buf, length)
    # Map16
    if tag == 0xDE:
        length = struct.unpack(">H", buf.read(2))[0]
        return _unpack_map(buf, length)
    # Map32
    if tag == 0xDF:
        length = struct.unpack(">I", buf.read(4))[0]
        return _unpack_map(buf, length)
    # Negative fixint
    if tag >= 0xE0:
        return tag - 256

    raise DAGConstructionError(f"Unknown msgpack tag: 0x{tag:02X}")


def _unpack_array(buf: BinaryIO, length: int) -> list[Any]:
    return [_unpack(buf) for _ in range(length)]


def _unpack_map(buf: BinaryIO, length: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for _ in range(length):
        key = _unpack(buf)
        value = _unpack(buf)
        result[str(key)] = value
    return result
