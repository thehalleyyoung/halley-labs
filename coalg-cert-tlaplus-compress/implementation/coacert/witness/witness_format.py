"""
Witness serialization format for bisimulation certificates.

Defines a binary format with header, four payload sections (equivalence
binding, transition witnesses, fairness witnesses, hash chain), and a
footer containing the overall Merkle root and integrity checksum.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from .equivalence_binding import EquivalenceBinding
from .hash_chain import HashChain
from .merkle_tree import EMPTY_HASH, HASH_LEN, MerkleTree, sha256
from .transition_witness import WitnessSet

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

FORMAT_MAGIC = b"CWIT"  # CoaCert WITness
FORMAT_VERSION = 1
SECTION_EQUIVALENCE = 0x01
SECTION_TRANSITIONS = 0x02
SECTION_FAIRNESS = 0x03
SECTION_HASHCHAIN = 0x04
FOOTER_MAGIC = b"CEND"

# Header layout (fixed 64 bytes):
#   magic          4B
#   version        2B
#   flags          2B
#   timestamp      8B  (float64, seconds since epoch)
#   spec_hash     32B  (SHA-256 of the original specification)
#   orig_states    4B  (uint32, state count of original system)
#   quot_states    4B  (uint32, state count of quotient)
#   section_count  2B
#   reserved       6B
HEADER_SIZE = 64

# Footer layout (fixed 40 bytes):
#   footer_magic   4B
#   merkle_root   32B
#   checksum       4B  (CRC-32 of everything before the footer)
FOOTER_SIZE = 40

FLAG_COMPRESSED = 0x0001


# ---------------------------------------------------------------------------
# Section descriptor
# ---------------------------------------------------------------------------


@dataclass
class _SectionDescriptor:
    section_type: int
    offset: int
    length: int

    def to_bytes(self) -> bytes:
        return struct.pack(">BII", self.section_type, self.offset, self.length)

    @classmethod
    def from_bytes(cls, buf: bytes, offset: int = 0) -> Tuple["_SectionDescriptor", int]:
        stype, soff, slen = struct.unpack_from(">BII", buf, offset)
        return cls(section_type=stype, offset=soff, length=slen), offset + 9


# ---------------------------------------------------------------------------
# WitnessFormat
# ---------------------------------------------------------------------------


class WitnessFormat:
    """
    Serialization and deserialization of bisimulation witness certificates.
    """

    def __init__(
        self,
        equivalence: EquivalenceBinding,
        witnesses: WitnessSet,
        chain: HashChain,
        spec_hash: bytes = EMPTY_HASH,
        original_state_count: int = 0,
        quotient_state_count: int = 0,
        version: int = FORMAT_VERSION,
    ) -> None:
        self.equivalence = equivalence
        self.witnesses = witnesses
        self.chain = chain
        self.spec_hash = spec_hash
        self.original_state_count = original_state_count
        self.quotient_state_count = quotient_state_count
        self.version = version
        self.timestamp = time.time()
        self._flags = 0

    # -- header packing -----------------------------------------------------

    def _pack_header(self, section_count: int) -> bytes:
        header = bytearray(HEADER_SIZE)
        struct.pack_into(">4s", header, 0, FORMAT_MAGIC)
        struct.pack_into(">H", header, 4, self.version)
        struct.pack_into(">H", header, 6, self._flags)
        struct.pack_into(">d", header, 8, self.timestamp)
        header[16:48] = self.spec_hash[:HASH_LEN].ljust(HASH_LEN, b"\x00")
        struct.pack_into(">I", header, 48, self.original_state_count)
        struct.pack_into(">I", header, 52, self.quotient_state_count)
        struct.pack_into(">H", header, 56, section_count)
        return bytes(header)

    @staticmethod
    def _unpack_header(buf: bytes) -> Dict[str, Any]:
        magic = buf[0:4]
        if magic != FORMAT_MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        version = struct.unpack_from(">H", buf, 4)[0]
        flags = struct.unpack_from(">H", buf, 6)[0]
        timestamp = struct.unpack_from(">d", buf, 8)[0]
        spec_hash = buf[16:48]
        orig = struct.unpack_from(">I", buf, 48)[0]
        quot = struct.unpack_from(">I", buf, 52)[0]
        section_count = struct.unpack_from(">H", buf, 56)[0]
        return {
            "version": version,
            "flags": flags,
            "timestamp": timestamp,
            "spec_hash": spec_hash,
            "original_state_count": orig,
            "quotient_state_count": quot,
            "section_count": section_count,
        }

    # -- footer packing -----------------------------------------------------

    @staticmethod
    def _pack_footer(merkle_root: bytes, body: bytes) -> bytes:
        import binascii
        crc = binascii.crc32(body) & 0xFFFFFFFF
        footer = bytearray(FOOTER_SIZE)
        struct.pack_into(">4s", footer, 0, FOOTER_MAGIC)
        footer[4:36] = merkle_root
        struct.pack_into(">I", footer, 36, crc)
        return bytes(footer)

    @staticmethod
    def _unpack_footer(buf: bytes, offset: int) -> Dict[str, Any]:
        magic = buf[offset : offset + 4]
        if magic != FOOTER_MAGIC:
            raise ValueError(f"Bad footer magic: {magic!r}")
        merkle_root = buf[offset + 4 : offset + 36]
        (crc,) = struct.unpack_from(">I", buf, offset + 36)
        return {"merkle_root": merkle_root, "checksum": crc}

    # -- overall Merkle root ------------------------------------------------

    def _compute_overall_root(self) -> bytes:
        """Merkle root over equivalence root, witness root, chain root."""
        roots = [
            self.equivalence.root,
            self.witnesses.root,
            self.chain.merkle_root,
        ]
        tree = MerkleTree(roots)
        return tree.root

    # -- serialize to file --------------------------------------------------

    def serialize(self, path: Union[str, Path], compress: bool = False) -> int:
        """
        Write the witness to a binary file.

        Returns the number of bytes written.
        """
        path = Path(path)

        # Build section payloads
        eq_data = self.equivalence.to_bytes()
        tw_data = self.witnesses.to_bytes()
        # fairness is embedded in the witness set; write empty section
        fair_data = b""
        chain_data = self.chain.to_bytes()

        sections = [
            (SECTION_EQUIVALENCE, eq_data),
            (SECTION_TRANSITIONS, tw_data),
            (SECTION_FAIRNESS, fair_data),
            (SECTION_HASHCHAIN, chain_data),
        ]

        # Descriptor table sits right after the header
        desc_table_size = len(sections) * 9
        body_offset = HEADER_SIZE + desc_table_size

        # Compute section offsets
        descriptors: list[_SectionDescriptor] = []
        cursor = body_offset
        for stype, sdata in sections:
            descriptors.append(_SectionDescriptor(stype, cursor, len(sdata)))
            cursor += len(sdata)

        # Assemble body (header + descriptors + sections)
        header = self._pack_header(section_count=len(sections))
        desc_bytes = b"".join(d.to_bytes() for d in descriptors)
        payload = b"".join(sdata for _, sdata in sections)
        body = header + desc_bytes + payload

        # Footer
        overall_root = self._compute_overall_root()
        footer = self._pack_footer(overall_root, body)
        full = body + footer

        if compress:
            self._flags |= FLAG_COMPRESSED
            # re-pack header with updated flags
            header = self._pack_header(section_count=len(sections))
            body = header + desc_bytes + payload
            footer = self._pack_footer(overall_root, body)
            full = body + footer
            full = gzip.compress(full, compresslevel=6)

        path.write_bytes(full)
        return len(full)

    # -- deserialize from file ----------------------------------------------

    @classmethod
    def deserialize(cls, path: Union[str, Path]) -> "WitnessFormat":
        """Read a witness from a binary file."""
        path = Path(path)
        raw = path.read_bytes()

        # Try gzip decompression
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)

        hdr = cls._unpack_header(raw)
        section_count = hdr["section_count"]

        # Parse descriptor table
        pos = HEADER_SIZE
        descriptors: list[_SectionDescriptor] = []
        for _ in range(section_count):
            desc, pos = _SectionDescriptor.from_bytes(raw, pos)
            descriptors.append(desc)

        # Extract section payloads
        section_data: Dict[int, bytes] = {}
        for desc in descriptors:
            section_data[desc.section_type] = raw[desc.offset : desc.offset + desc.length]

        # Footer verification
        footer_offset = len(raw) - FOOTER_SIZE
        ftr = cls._unpack_footer(raw, footer_offset)
        body = raw[:footer_offset]
        import binascii
        expected_crc = binascii.crc32(body) & 0xFFFFFFFF
        if expected_crc != ftr["checksum"]:
            raise ValueError(
                f"Checksum mismatch: expected {expected_crc:#x}, "
                f"got {ftr['checksum']:#x}"
            )

        # Reconstruct components
        eq = EquivalenceBinding.from_bytes(section_data.get(SECTION_EQUIVALENCE, b"\x00\x00\x00\x00" + EMPTY_HASH))
        ws = WitnessSet.from_bytes(section_data.get(SECTION_TRANSITIONS, b"\x00" * 12 + EMPTY_HASH))
        ch = HashChain.from_bytes(section_data.get(SECTION_HASHCHAIN, b"\x00\x00\x00\x00"))

        wf = cls(
            equivalence=eq,
            witnesses=ws,
            chain=ch,
            spec_hash=hdr["spec_hash"],
            original_state_count=hdr["original_state_count"],
            quotient_state_count=hdr["quotient_state_count"],
            version=hdr["version"],
        )
        wf.timestamp = hdr["timestamp"]
        wf._flags = hdr["flags"]
        return wf

    # -- JSON export --------------------------------------------------------

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """
        Export a human-readable JSON representation.

        If *path* is given, write to that file and return the JSON string.
        """
        data = {
            "format_version": self.version,
            "timestamp": self.timestamp,
            "spec_hash": self.spec_hash.hex(),
            "original_state_count": self.original_state_count,
            "quotient_state_count": self.quotient_state_count,
            "equivalence": self.equivalence.to_dict(),
            "witnesses": self.witnesses.to_dict(),
            "hash_chain": self.chain.to_dict(),
            "overall_merkle_root": self._compute_overall_root().hex(),
        }
        text = json.dumps(data, indent=indent, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    # -- size computation ---------------------------------------------------

    @property
    def witness_size(self) -> Dict[str, int]:
        """Breakdown of serialized sizes by section."""
        eq_sz = len(self.equivalence.to_bytes())
        tw_sz = len(self.witnesses.to_bytes())
        ch_sz = len(self.chain.to_bytes())
        total = HEADER_SIZE + 4 * 9 + eq_sz + tw_sz + ch_sz + FOOTER_SIZE
        return {
            "header": HEADER_SIZE,
            "descriptors": 4 * 9,
            "equivalence": eq_sz,
            "transitions": tw_sz,
            "hash_chain": ch_sz,
            "footer": FOOTER_SIZE,
            "total": total,
        }

    @property
    def compressed_size(self) -> int:
        """Estimated compressed size."""
        eq_data = self.equivalence.to_bytes()
        tw_data = self.witnesses.to_bytes()
        ch_data = self.chain.to_bytes()
        raw = eq_data + tw_data + ch_data
        return len(gzip.compress(raw, compresslevel=6))

    # -- format version handling --------------------------------------------

    @staticmethod
    def supported_versions() -> List[int]:
        return [1]

    def check_version(self) -> bool:
        return self.version in self.supported_versions()

    # -- verification -------------------------------------------------------

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Run all integrity checks on the witness."""
        errors: List[str] = []

        # Equivalence partition
        ok, errs = self.equivalence.verify_partition()
        errors.extend(errs)

        # Hash chain
        ok, errs = self.chain.verify()
        errors.extend(errs)

        # Binding hashes
        if not self.equivalence.verify_binding_hashes():
            errors.append("Equivalence binding hash verification failed")

        return (len(errors) == 0, errors)

    def __repr__(self) -> str:
        sizes = self.witness_size
        return (
            f"WitnessFormat(v{self.version}, "
            f"orig={self.original_state_count}, "
            f"quot={self.quotient_state_count}, "
            f"size={sizes['total']}B)"
        )
