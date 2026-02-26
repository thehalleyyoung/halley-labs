"""
Witness deserialization for CoaCert-TLA bisimulation witnesses.

Supports both binary witness format (.cwit) and JSON witness format (.json).
Binary format layout:
  [Header: 16 bytes] [Section Directory] [Sections...]

Header:
  magic:   4 bytes  "CWIT"
  version: 2 bytes  (major.minor)
  flags:   2 bytes
  num_sections: 4 bytes
  total_size:   4 bytes

Section kinds:
  0x01 - Equivalence bindings
  0x02 - Transition witnesses
  0x03 - Fairness data
  0x04 - Hash chain
  0x05 - Metadata
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC = b"CWIT"
SUPPORTED_VERSIONS = {(1, 0), (1, 1), (2, 0)}
HEADER_SIZE = 16
SECTION_DIR_ENTRY_SIZE = 12  # kind(2) + offset(4) + length(4) + checksum(2)


class SectionKind(IntEnum):
    EQUIVALENCE = 0x01
    TRANSITION = 0x02
    FAIRNESS = 0x03
    HASH_CHAIN = 0x04
    METADATA = 0x05


class WitnessFlag(IntEnum):
    STUTTERING = 0x01
    COMPRESSED = 0x02
    FAIRNESS_PRESENT = 0x04
    MERKLE_HASHED = 0x08


# ---------------------------------------------------------------------------
# Data containers (frozen dataclasses, following project convention)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WitnessHeader:
    """Parsed witness file header."""
    version_major: int
    version_minor: int
    flags: int
    num_sections: int
    total_size: int

    @property
    def has_stuttering(self) -> bool:
        return bool(self.flags & WitnessFlag.STUTTERING)

    @property
    def has_fairness(self) -> bool:
        return bool(self.flags & WitnessFlag.FAIRNESS_PRESENT)

    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & WitnessFlag.COMPRESSED)

    @property
    def is_merkle_hashed(self) -> bool:
        return bool(self.flags & WitnessFlag.MERKLE_HASHED)


@dataclass(frozen=True)
class SectionEntry:
    """Directory entry for one section."""
    kind: SectionKind
    offset: int
    length: int
    checksum: int


@dataclass(frozen=True)
class EquivalenceBinding:
    """One equivalence class: representative + members."""
    class_id: int
    representative: str
    members: Tuple[str, ...]
    ap_labels: Tuple[str, ...]


@dataclass(frozen=True)
class TransitionWitness:
    """Witness for one transition-matching obligation."""
    source_class: int
    target_class: int
    original_source: str
    original_target: str
    matching_path: Tuple[str, ...]
    is_stutter: bool = False
    stutter_depth: int = 0


@dataclass(frozen=True)
class FairnessBinding:
    """Fairness acceptance pair binding."""
    pair_id: int
    b_set_classes: Tuple[int, ...]
    g_set_classes: Tuple[int, ...]


@dataclass(frozen=True)
class HashBlock:
    """One block in the hash chain."""
    index: int
    prev_hash: bytes
    payload_hash: bytes
    block_hash: bytes
    merkle_root: Optional[bytes] = None
    merkle_proof: Optional[Tuple[bytes, ...]] = None


@dataclass
class WitnessData:
    """Complete deserialized witness."""
    header: WitnessHeader
    equivalences: List[EquivalenceBinding] = field(default_factory=list)
    transitions: List[TransitionWitness] = field(default_factory=list)
    fairness: List[FairnessBinding] = field(default_factory=list)
    hash_chain: List[HashBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeserializationError(Exception):
    """Raised when witness data cannot be parsed."""

    def __init__(self, message: str, offset: int = -1):
        self.offset = offset
        prefix = f"[offset 0x{offset:08X}] " if offset >= 0 else ""
        super().__init__(f"{prefix}{message}")


# ---------------------------------------------------------------------------
# Binary deserializer
# ---------------------------------------------------------------------------


class WitnessDeserializer:
    """Deserializes CoaCert-TLA bisimulation witnesses from binary or JSON."""

    def __init__(self, *, strict: bool = True, chunk_size: int = 65536):
        self._strict = strict
        self._chunk_size = chunk_size
        self._errors: List[str] = []

    @property
    def errors(self) -> Sequence[str]:
        return list(self._errors)

    # -- public API ---------------------------------------------------------

    def deserialize_file(self, path: str) -> WitnessData:
        """Deserialize a witness file (auto-detects binary vs JSON)."""
        with open(path, "rb") as fh:
            peek = fh.read(4)
            fh.seek(0)
            if peek == MAGIC:
                data = self._parse_binary(fh)
            else:
                fh.close()
                data = self._parse_json_file(path)
        data.source_path = path
        return data

    def deserialize_bytes(self, raw: bytes) -> WitnessData:
        """Deserialize from an in-memory byte buffer."""
        return self._parse_binary(io.BytesIO(raw))

    def deserialize_json(self, text: str) -> WitnessData:
        """Deserialize from a JSON string."""
        return self._parse_json_obj(json.loads(text))

    def stream_deserialize(self, stream: BinaryIO) -> Iterator[
        Union[WitnessHeader, EquivalenceBinding, TransitionWitness,
              FairnessBinding, HashBlock]
    ]:
        """Streaming deserialization — yields objects as they are parsed."""
        header = self._read_header(stream)
        yield header
        directory = self._read_directory(stream, header.num_sections)
        for entry in directory:
            stream.seek(entry.offset)
            section_bytes = stream.read(entry.length)
            if len(section_bytes) < entry.length:
                raise DeserializationError(
                    f"Truncated section {entry.kind.name}: expected {entry.length} "
                    f"bytes, got {len(section_bytes)}",
                    offset=entry.offset,
                )
            self._verify_section_checksum(entry, section_bytes)
            buf = io.BytesIO(section_bytes)
            if entry.kind == SectionKind.EQUIVALENCE:
                yield from self._iter_equivalences(buf, entry.offset)
            elif entry.kind == SectionKind.TRANSITION:
                yield from self._iter_transitions(buf, entry.offset)
            elif entry.kind == SectionKind.FAIRNESS:
                yield from self._iter_fairness(buf, entry.offset)
            elif entry.kind == SectionKind.HASH_CHAIN:
                yield from self._iter_hash_chain(buf, entry.offset)

    # -- binary parsing internals -------------------------------------------

    def _parse_binary(self, stream: BinaryIO) -> WitnessData:
        header = self._read_header(stream)
        directory = self._read_directory(stream, header.num_sections)
        witness = WitnessData(header=header)
        for entry in directory:
            self._parse_section(stream, entry, witness)
        return witness

    def _read_header(self, stream: BinaryIO) -> WitnessHeader:
        offset = stream.tell()
        raw = stream.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise DeserializationError(
                f"File too short for header: {len(raw)} bytes", offset=offset
            )
        magic = raw[0:4]
        if magic != MAGIC:
            raise DeserializationError(
                f"Bad magic: expected {MAGIC!r}, got {magic!r}", offset=offset
            )
        ver_major, ver_minor = struct.unpack_from(">BB", raw, 4)
        if (ver_major, ver_minor) not in SUPPORTED_VERSIONS:
            if self._strict:
                raise DeserializationError(
                    f"Unsupported version {ver_major}.{ver_minor}", offset=offset + 4
                )
            self._errors.append(f"Unknown version {ver_major}.{ver_minor}")
        flags, = struct.unpack_from(">H", raw, 6)
        num_sections, = struct.unpack_from(">I", raw, 8)
        total_size, = struct.unpack_from(">I", raw, 12)
        return WitnessHeader(
            version_major=ver_major,
            version_minor=ver_minor,
            flags=flags,
            num_sections=num_sections,
            total_size=total_size,
        )

    def _read_directory(self, stream: BinaryIO,
                        num_sections: int) -> List[SectionEntry]:
        entries: List[SectionEntry] = []
        base = stream.tell()
        for i in range(num_sections):
            offset = base + i * SECTION_DIR_ENTRY_SIZE
            raw = stream.read(SECTION_DIR_ENTRY_SIZE)
            if len(raw) < SECTION_DIR_ENTRY_SIZE:
                raise DeserializationError(
                    f"Truncated section directory at entry {i}", offset=offset
                )
            kind_raw, sec_offset, sec_length, checksum = struct.unpack(
                ">HIIH", raw
            )
            try:
                kind = SectionKind(kind_raw)
            except ValueError:
                if self._strict:
                    raise DeserializationError(
                        f"Unknown section kind 0x{kind_raw:04X}", offset=offset
                    )
                self._errors.append(f"Unknown section kind 0x{kind_raw:04X}")
                continue
            entries.append(SectionEntry(kind, sec_offset, sec_length, checksum))
        return entries

    def _verify_section_checksum(self, entry: SectionEntry,
                                 data: bytes) -> None:
        computed = self._fletcher16(data)
        if computed != entry.checksum:
            msg = (
                f"Section {entry.kind.name} checksum mismatch: "
                f"expected 0x{entry.checksum:04X}, got 0x{computed:04X}"
            )
            if self._strict:
                raise DeserializationError(msg, offset=entry.offset)
            self._errors.append(msg)

    @staticmethod
    def _fletcher16(data: bytes) -> int:
        """Fletcher-16 checksum for section integrity."""
        s1 = 0
        s2 = 0
        for byte in data:
            s1 = (s1 + byte) % 255
            s2 = (s2 + s1) % 255
        return (s2 << 8) | s1

    def _parse_section(self, stream: BinaryIO, entry: SectionEntry,
                       witness: WitnessData) -> None:
        stream.seek(entry.offset)
        data = stream.read(entry.length)
        if len(data) < entry.length:
            raise DeserializationError(
                f"Truncated section {entry.kind.name}", offset=entry.offset
            )
        self._verify_section_checksum(entry, data)
        buf = io.BytesIO(data)
        if entry.kind == SectionKind.EQUIVALENCE:
            witness.equivalences = list(
                self._iter_equivalences(buf, entry.offset)
            )
        elif entry.kind == SectionKind.TRANSITION:
            witness.transitions = list(
                self._iter_transitions(buf, entry.offset)
            )
        elif entry.kind == SectionKind.FAIRNESS:
            witness.fairness = list(self._iter_fairness(buf, entry.offset))
        elif entry.kind == SectionKind.HASH_CHAIN:
            witness.hash_chain = list(
                self._iter_hash_chain(buf, entry.offset)
            )
        elif entry.kind == SectionKind.METADATA:
            witness.metadata = self._parse_metadata(buf, entry.offset)

    # -- equivalence section ------------------------------------------------

    def _iter_equivalences(self, buf: BinaryIO,
                           base_offset: int) -> Iterator[EquivalenceBinding]:
        num_classes = self._read_u32(buf, base_offset)
        for _ in range(num_classes):
            off = base_offset + buf.tell()
            class_id = self._read_u32(buf, off)
            representative = self._read_string(buf, off)
            num_members = self._read_u32(buf, off)
            members: List[str] = []
            for _ in range(num_members):
                members.append(self._read_string(buf, off))
            num_aps = self._read_u32(buf, off)
            aps: List[str] = []
            for _ in range(num_aps):
                aps.append(self._read_string(buf, off))
            yield EquivalenceBinding(
                class_id=class_id,
                representative=representative,
                members=tuple(members),
                ap_labels=tuple(aps),
            )

    # -- transition section -------------------------------------------------

    def _iter_transitions(self, buf: BinaryIO,
                          base_offset: int) -> Iterator[TransitionWitness]:
        num_transitions = self._read_u32(buf, base_offset)
        for _ in range(num_transitions):
            off = base_offset + buf.tell()
            src_class = self._read_u32(buf, off)
            tgt_class = self._read_u32(buf, off)
            orig_src = self._read_string(buf, off)
            orig_tgt = self._read_string(buf, off)
            path_len = self._read_u32(buf, off)
            path: List[str] = []
            for _ in range(path_len):
                path.append(self._read_string(buf, off))
            flags = self._read_u8(buf, off)
            is_stutter = bool(flags & 0x01)
            stutter_depth = self._read_u16(buf, off) if is_stutter else 0
            yield TransitionWitness(
                source_class=src_class,
                target_class=tgt_class,
                original_source=orig_src,
                original_target=orig_tgt,
                matching_path=tuple(path),
                is_stutter=is_stutter,
                stutter_depth=stutter_depth,
            )

    # -- fairness section ---------------------------------------------------

    def _iter_fairness(self, buf: BinaryIO,
                       base_offset: int) -> Iterator[FairnessBinding]:
        num_pairs = self._read_u32(buf, base_offset)
        for _ in range(num_pairs):
            off = base_offset + buf.tell()
            pair_id = self._read_u32(buf, off)
            b_count = self._read_u32(buf, off)
            b_classes = [self._read_u32(buf, off) for _ in range(b_count)]
            g_count = self._read_u32(buf, off)
            g_classes = [self._read_u32(buf, off) for _ in range(g_count)]
            yield FairnessBinding(
                pair_id=pair_id,
                b_set_classes=tuple(b_classes),
                g_set_classes=tuple(g_classes),
            )

    # -- hash chain section -------------------------------------------------

    def _iter_hash_chain(self, buf: BinaryIO,
                         base_offset: int) -> Iterator[HashBlock]:
        num_blocks = self._read_u32(buf, base_offset)
        for _ in range(num_blocks):
            off = base_offset + buf.tell()
            index = self._read_u32(buf, off)
            prev_hash = self._read_hash(buf, off)
            payload_hash = self._read_hash(buf, off)
            block_hash = self._read_hash(buf, off)
            flags = self._read_u8(buf, off)
            merkle_root: Optional[bytes] = None
            merkle_proof: Optional[Tuple[bytes, ...]] = None
            if flags & 0x01:
                merkle_root = self._read_hash(buf, off)
                proof_len = self._read_u16(buf, off)
                proof_nodes: List[bytes] = []
                for _ in range(proof_len):
                    proof_nodes.append(self._read_hash(buf, off))
                merkle_proof = tuple(proof_nodes)
            yield HashBlock(
                index=index,
                prev_hash=prev_hash,
                payload_hash=payload_hash,
                block_hash=block_hash,
                merkle_root=merkle_root,
                merkle_proof=merkle_proof,
            )

    # -- metadata section ---------------------------------------------------

    def _parse_metadata(self, buf: BinaryIO, base_offset: int) -> Dict[str, Any]:
        raw = buf.read()
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DeserializationError(
                f"Invalid metadata JSON: {exc}", offset=base_offset
            )

    # -- primitive readers --------------------------------------------------

    def _read_u8(self, buf: BinaryIO, ctx_offset: int) -> int:
        raw = buf.read(1)
        if len(raw) < 1:
            raise DeserializationError("Unexpected end of section", offset=ctx_offset)
        return raw[0]

    def _read_u16(self, buf: BinaryIO, ctx_offset: int) -> int:
        raw = buf.read(2)
        if len(raw) < 2:
            raise DeserializationError("Unexpected end of section", offset=ctx_offset)
        return struct.unpack(">H", raw)[0]

    def _read_u32(self, buf: BinaryIO, ctx_offset: int) -> int:
        raw = buf.read(4)
        if len(raw) < 4:
            raise DeserializationError("Unexpected end of section", offset=ctx_offset)
        return struct.unpack(">I", raw)[0]

    def _read_string(self, buf: BinaryIO, ctx_offset: int) -> str:
        length = self._read_u16(buf, ctx_offset)
        raw = buf.read(length)
        if len(raw) < length:
            raise DeserializationError(
                f"Truncated string (expected {length}, got {len(raw)})",
                offset=ctx_offset,
            )
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DeserializationError(
                f"Invalid UTF-8 in string: {exc}", offset=ctx_offset
            )

    def _read_hash(self, buf: BinaryIO, ctx_offset: int) -> bytes:
        raw = buf.read(32)
        if len(raw) < 32:
            raise DeserializationError(
                "Truncated hash (expected 32 bytes)", offset=ctx_offset
            )
        return raw

    # -- JSON parsing -------------------------------------------------------

    def _parse_json_file(self, path: str) -> WitnessData:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        return self._parse_json_obj(obj)

    def _parse_json_obj(self, obj: Dict[str, Any]) -> WitnessData:
        if not isinstance(obj, dict):
            raise DeserializationError("Top-level JSON must be an object")
        header = self._json_parse_header(obj.get("header", {}))
        witness = WitnessData(header=header)
        if "equivalences" in obj:
            witness.equivalences = [
                self._json_parse_equivalence(e) for e in obj["equivalences"]
            ]
        if "transitions" in obj:
            witness.transitions = [
                self._json_parse_transition(t) for t in obj["transitions"]
            ]
        if "fairness" in obj:
            witness.fairness = [
                self._json_parse_fairness(f) for f in obj["fairness"]
            ]
        if "hash_chain" in obj:
            witness.hash_chain = [
                self._json_parse_hash_block(b) for b in obj["hash_chain"]
            ]
        if "metadata" in obj:
            witness.metadata = obj["metadata"]
        return witness

    def _json_parse_header(self, obj: Dict[str, Any]) -> WitnessHeader:
        return WitnessHeader(
            version_major=obj.get("version_major", 1),
            version_minor=obj.get("version_minor", 0),
            flags=obj.get("flags", 0),
            num_sections=obj.get("num_sections", 0),
            total_size=obj.get("total_size", 0),
        )

    def _json_parse_equivalence(self, obj: Dict[str, Any]) -> EquivalenceBinding:
        return EquivalenceBinding(
            class_id=obj["class_id"],
            representative=obj["representative"],
            members=tuple(obj.get("members", [])),
            ap_labels=tuple(obj.get("ap_labels", [])),
        )

    def _json_parse_transition(self, obj: Dict[str, Any]) -> TransitionWitness:
        return TransitionWitness(
            source_class=obj["source_class"],
            target_class=obj["target_class"],
            original_source=obj["original_source"],
            original_target=obj["original_target"],
            matching_path=tuple(obj.get("matching_path", [])),
            is_stutter=obj.get("is_stutter", False),
            stutter_depth=obj.get("stutter_depth", 0),
        )

    def _json_parse_fairness(self, obj: Dict[str, Any]) -> FairnessBinding:
        return FairnessBinding(
            pair_id=obj["pair_id"],
            b_set_classes=tuple(obj.get("b_set_classes", [])),
            g_set_classes=tuple(obj.get("g_set_classes", [])),
        )

    def _json_parse_hash_block(self, obj: Dict[str, Any]) -> HashBlock:
        def _hex(s: Optional[str]) -> Optional[bytes]:
            return bytes.fromhex(s) if s else None

        merkle_proof = None
        if "merkle_proof" in obj and obj["merkle_proof"] is not None:
            merkle_proof = tuple(bytes.fromhex(p) for p in obj["merkle_proof"])
        return HashBlock(
            index=obj["index"],
            prev_hash=bytes.fromhex(obj["prev_hash"]),
            payload_hash=bytes.fromhex(obj["payload_hash"]),
            block_hash=bytes.fromhex(obj["block_hash"]),
            merkle_root=_hex(obj.get("merkle_root")),
            merkle_proof=merkle_proof,
        )
