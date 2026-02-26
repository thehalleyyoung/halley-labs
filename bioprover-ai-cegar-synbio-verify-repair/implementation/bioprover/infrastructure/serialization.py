"""
Serialization and checkpoint/restart for BioProver.

Provides a generic serializer (pickle + JSON fallback), a checkpoint manager
for long-running CEGAR / CEGIS loops, versioned serialisation format,
compressed checkpoints, and incremental checkpointing that only persists
changed state.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Current wire-format version – bump when the checkpoint layout changes.
FORMAT_VERSION = 2
FORMAT_MAGIC = b"BPCK"  # BioProver ChecKpoint

# ---------------------------------------------------------------------------
# Versioned on-disk format
# ---------------------------------------------------------------------------

@dataclass
class VersionedFormat:
    """Envelope that wraps serialised payloads with metadata.

    On-disk layout (binary):
        4 bytes   magic  (``BPCK``)
        4 bytes   format version (big-endian uint32)
        32 bytes  SHA-256 of the payload
        N bytes   gzip-compressed payload (pickle or JSON)
    """

    version: int = FORMAT_VERSION
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    HEADER_SIZE: int = 4 + 4 + 32  # magic + version + hash

    @classmethod
    def write(
        cls,
        path: Path,
        payload: bytes,
        *,
        compress: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write *payload* to *path* with version header and integrity hash."""
        digest = hashlib.sha256(payload).digest()
        body = gzip.compress(payload, compresslevel=6) if compress else payload

        with open(path, "wb") as fh:
            fh.write(FORMAT_MAGIC)
            fh.write(FORMAT_VERSION.to_bytes(4, "big"))
            fh.write(digest)
            fh.write(body)

        if metadata is not None:
            meta_path = path.with_suffix(path.suffix + ".meta")
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path, *, compressed: bool = True) -> bytes:
        """Read and verify a checkpoint file, returning the raw payload bytes."""
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != FORMAT_MAGIC:
                raise ValueError(f"Invalid checkpoint magic in {path}")
            version_bytes = fh.read(4)
            version = int.from_bytes(version_bytes, "big")
            if version > FORMAT_VERSION:
                raise ValueError(
                    f"Checkpoint version {version} is newer than supported "
                    f"version {FORMAT_VERSION}; upgrade BioProver."
                )
            expected_hash = fh.read(32)
            body = fh.read()

        payload = gzip.decompress(body) if compressed else body
        actual_hash = hashlib.sha256(payload).digest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checkpoint integrity check failed for {path}. "
                "The file may be corrupted."
            )
        return payload


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

class Serializer:
    """Generic serializer with pickle as primary and JSON as fallback.

    The JSON fallback is used when objects are not picklable (e.g. for
    export / interop).  Custom encoders/decoders can be registered.
    """

    def __init__(self) -> None:
        self._json_encoders: Dict[type, Callable[[Any], Any]] = {}
        self._json_decoders: Dict[str, Callable[[Any], Any]] = {}

    # -- registration -------------------------------------------------------

    def register_json_encoder(
        self, cls: type, encoder: Callable[[Any], Any]
    ) -> None:
        """Register a JSON encoder for *cls*."""
        self._json_encoders[cls] = encoder

    def register_json_decoder(
        self, type_tag: str, decoder: Callable[[Any], Any]
    ) -> None:
        """Register a JSON decoder keyed by a *type_tag* string."""
        self._json_decoders[type_tag] = decoder

    # -- pickle -------------------------------------------------------------

    def to_bytes(self, obj: Any) -> bytes:
        """Serialize *obj* to bytes using pickle (protocol 5)."""
        return pickle.dumps(obj, protocol=5)

    def from_bytes(self, data: bytes) -> Any:
        """Deserialize bytes produced by :meth:`to_bytes`."""
        return pickle.loads(data)  # noqa: S301

    # -- JSON ---------------------------------------------------------------

    def to_json(self, obj: Any) -> str:
        """Serialize *obj* to a JSON string (uses registered encoders)."""
        return json.dumps(obj, default=self._default_encode, indent=2)

    def from_json(self, text: str) -> Any:
        """Deserialize a JSON string (uses registered decoders)."""
        return json.loads(text, object_hook=self._object_hook)

    # -- file I/O -----------------------------------------------------------

    def save(self, obj: Any, path: Path, *, compress: bool = True) -> None:
        """Persist *obj* as a versioned checkpoint file."""
        payload = self.to_bytes(obj)
        VersionedFormat.write(path, payload, compress=compress)
        logger.debug("Saved checkpoint to %s (%d bytes)", path, len(payload))

    def load(self, path: Path, *, compressed: bool = True) -> Any:
        """Load an object from a versioned checkpoint file."""
        payload = VersionedFormat.read(path, compressed=compressed)
        obj = self.from_bytes(payload)
        logger.debug("Loaded checkpoint from %s", path)
        return obj

    # -- internal -----------------------------------------------------------

    def _default_encode(self, obj: Any) -> Any:
        for cls, enc in self._json_encoders.items():
            if isinstance(obj, cls):
                return enc(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return {"__type__": type(obj).__name__, **asdict(obj)}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

    def _object_hook(self, d: Dict[str, Any]) -> Any:
        tag = d.pop("__type__", None)
        if tag is not None and tag in self._json_decoders:
            return self._json_decoders[tag](d)
        return d


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

@dataclass
class StateSnapshot:
    """Immutable snapshot of a piece of verification state.

    Used by the checkpoint manager to track what has changed between saves.
    """

    label: str
    data: Any
    fingerprint: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        raw = pickle.dumps(self.data, protocol=5)
        return hashlib.sha256(raw).hexdigest()[:16]

    @property
    def size_bytes(self) -> int:
        return len(pickle.dumps(self.data, protocol=5))


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages checkpoint files for long-running CEGAR / CEGIS loops.

    Checkpoints are stored in a directory with numbered files.  Old checkpoints
    beyond *keep_last* are automatically pruned.
    """

    def __init__(
        self,
        directory: Path,
        keep_last: int = 5,
        serializer: Optional[Serializer] = None,
    ) -> None:
        self.directory = Path(directory)
        self.keep_last = keep_last
        self._serializer = serializer or Serializer()
        self._counter: int = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        self._counter = self._discover_latest()

    # -- public API ---------------------------------------------------------

    def save(self, state: Any, *, label: str = "") -> Path:
        """Save a checkpoint and return its path."""
        self._counter += 1
        name = f"ckpt_{self._counter:06d}.bp"
        path = self.directory / name
        meta = {"counter": self._counter, "label": label, "time": time.time()}
        payload = self._serializer.to_bytes(state)
        VersionedFormat.write(path, payload, metadata=meta)
        logger.info("Checkpoint %d saved to %s", self._counter, path)
        self._prune()
        return path

    def load_latest(self) -> Any:
        """Load the most recent checkpoint."""
        path = self._latest_path()
        if path is None:
            raise FileNotFoundError("No checkpoints found.")
        return self._serializer.load(path)

    def load(self, index: int) -> Any:
        """Load a checkpoint by its sequence number."""
        name = f"ckpt_{index:06d}.bp"
        path = self.directory / name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {index} not found at {path}")
        return self._serializer.load(path)

    def list_checkpoints(self) -> List[Path]:
        """Return sorted list of checkpoint paths (oldest first)."""
        return sorted(self.directory.glob("ckpt_*.bp"))

    @property
    def latest_index(self) -> int:
        return self._counter

    # -- internal -----------------------------------------------------------

    def _discover_latest(self) -> int:
        existing = self.list_checkpoints()
        if not existing:
            return 0
        last = existing[-1].stem  # e.g. "ckpt_000003"
        return int(last.split("_")[1])

    def _latest_path(self) -> Optional[Path]:
        existing = self.list_checkpoints()
        return existing[-1] if existing else None

    def _prune(self) -> None:
        """Delete old checkpoints beyond *keep_last*."""
        existing = self.list_checkpoints()
        while len(existing) > self.keep_last:
            victim = existing.pop(0)
            victim.unlink(missing_ok=True)
            meta = victim.with_suffix(victim.suffix + ".meta")
            meta.unlink(missing_ok=True)
            logger.debug("Pruned old checkpoint %s", victim)


# ---------------------------------------------------------------------------
# Incremental checkpointer
# ---------------------------------------------------------------------------

class IncrementalCheckpointer:
    """Only checkpoints state components whose fingerprints have changed.

    Useful when the overall state is large but only a small part changes
    each CEGAR iteration (e.g. the predicate set, but not the model).
    """

    def __init__(
        self,
        directory: Path,
        serializer: Optional[Serializer] = None,
        keep_last: int = 3,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._serializer = serializer or Serializer()
        self._keep_last = keep_last
        self._last_fingerprints: Dict[str, str] = {}
        self._generation: int = 0

    def checkpoint(self, snapshots: Dict[str, StateSnapshot]) -> Path:
        """Save only the components that changed since the last call.

        Returns the directory for this generation.
        """
        self._generation += 1
        gen_dir = self.directory / f"gen_{self._generation:06d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        changed: List[str] = []
        unchanged: List[str] = []

        for label, snap in snapshots.items():
            prev_fp = self._last_fingerprints.get(label)
            if prev_fp == snap.fingerprint:
                unchanged.append(label)
                # Symlink to previous generation
                prev_gen = self._generation - 1
                prev_path = (
                    self.directory / f"gen_{prev_gen:06d}" / f"{label}.bp"
                )
                link_path = gen_dir / f"{label}.bp"
                if prev_path.exists() and not link_path.exists():
                    try:
                        link_path.symlink_to(prev_path.resolve())
                    except OSError:
                        # Fallback: copy if symlinks unsupported
                        shutil.copy2(prev_path, link_path)
            else:
                changed.append(label)
                payload = self._serializer.to_bytes(snap.data)
                VersionedFormat.write(gen_dir / f"{label}.bp", payload)
                self._last_fingerprints[label] = snap.fingerprint

        # Write manifest
        manifest = {
            "generation": self._generation,
            "time": time.time(),
            "changed": changed,
            "unchanged": unchanged,
        }
        (gen_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info(
            "Incremental checkpoint gen %d: %d changed, %d reused",
            self._generation,
            len(changed),
            len(unchanged),
        )
        self._prune_generations()
        return gen_dir

    def restore(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Restore all state components from a generation."""
        if generation is None:
            generation = self._generation
        gen_dir = self.directory / f"gen_{generation:06d}"
        if not gen_dir.exists():
            raise FileNotFoundError(f"Generation {generation} not found")

        result: Dict[str, Any] = {}
        for bp_file in gen_dir.glob("*.bp"):
            label = bp_file.stem
            payload = VersionedFormat.read(bp_file)
            result[label] = self._serializer.from_bytes(payload)
        return result

    @property
    def current_generation(self) -> int:
        return self._generation

    def _prune_generations(self) -> None:
        gens = sorted(self.directory.glob("gen_*"))
        while len(gens) > self._keep_last:
            victim = gens.pop(0)
            shutil.rmtree(victim, ignore_errors=True)
            logger.debug("Pruned generation %s", victim.name)
