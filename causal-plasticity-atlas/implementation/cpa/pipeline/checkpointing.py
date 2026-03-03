"""Checkpoint management for the CPA pipeline.

Provides :class:`CheckpointManager` for saving and resuming pipeline
state between and within phases.  Uses atomic writes and integrity
verification to ensure checkpoint reliability.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("pipeline.checkpointing")


# =====================================================================
# Checkpoint metadata
# =====================================================================

_FORMAT_VERSION = "1.0.0"
_CHECKPOINT_PREFIX = "cpa_checkpoint"
_MANIFEST_NAME = "manifest.json"
_ARRAYS_DIR = "arrays"


@dataclass
class CheckpointManifest:
    """Metadata for a checkpoint directory.

    Attributes
    ----------
    format_version : str
        Checkpoint format version for compatibility checks.
    phase : int
        Pipeline phase (1, 2, or 3) that produced this checkpoint.
    step : int
        Sub-step within the phase.
    timestamp : float
        Unix timestamp when checkpoint was created.
    config_hash : str
        Hash of the pipeline configuration for consistency checks.
    array_files : List[str]
        List of numpy array files stored in the checkpoint.
    json_keys : List[str]
        List of JSON-serialized state keys.
    checksum : str
        SHA-256 checksum of the manifest content for integrity.
    description : str
        Human-readable description.
    """

    format_version: str = _FORMAT_VERSION
    phase: int = 0
    step: int = 0
    timestamp: float = 0.0
    config_hash: str = ""
    array_files: List[str] = field(default_factory=list)
    json_keys: List[str] = field(default_factory=list)
    checksum: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "phase": self.phase,
            "step": self.step,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "array_files": self.array_files,
            "json_keys": self.json_keys,
            "checksum": self.checksum,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointManifest":
        return cls(
            format_version=d.get("format_version", _FORMAT_VERSION),
            phase=d.get("phase", 0),
            step=d.get("step", 0),
            timestamp=d.get("timestamp", 0.0),
            config_hash=d.get("config_hash", ""),
            array_files=d.get("array_files", []),
            json_keys=d.get("json_keys", []),
            checksum=d.get("checksum", ""),
            description=d.get("description", ""),
        )


# =====================================================================
# CheckpointManager
# =====================================================================


class CheckpointManager:
    """Save and restore pipeline state with atomic writes and integrity checks.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Root directory for storing checkpoints.
    max_checkpoints : int
        Maximum number of checkpoints to keep (oldest are removed).
    config_hash : str
        Hash of the pipeline configuration, used to verify that a
        checkpoint is compatible with the current run.

    Examples
    --------
    >>> cm = CheckpointManager("/tmp/cpa_checkpoints", config_hash="abc123")
    >>> cm.save(phase=1, step=0, state={"scm_results": {...}},
    ...         arrays={"adj_0": adj_matrix})
    >>> state, arrays = cm.load_latest()
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        config_hash: str = "",
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_checkpoints = max(1, max_checkpoints)
        self._config_hash = config_hash

    @property
    def checkpoint_dir(self) -> Path:
        """Root checkpoint directory."""
        return self._dir

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    def save(
        self,
        phase: int,
        step: int,
        state: Dict[str, Any],
        arrays: Optional[Dict[str, np.ndarray]] = None,
        description: str = "",
    ) -> Path:
        """Save a checkpoint atomically.

        Creates a temporary directory, writes all data, then renames
        to the final location to ensure atomicity.

        Parameters
        ----------
        phase : int
            Pipeline phase number (1, 2, or 3).
        step : int
            Sub-step within the phase.
        state : dict
            JSON-serializable pipeline state.
        arrays : dict of str → np.ndarray, optional
            Named numpy arrays to store.
        description : str
            Human-readable description.

        Returns
        -------
        Path
            Path to the created checkpoint directory.
        """
        arrays = arrays or {}
        ts = time.time()
        ckpt_name = f"{_CHECKPOINT_PREFIX}_p{phase}_s{step}_{int(ts)}"
        final_path = self._dir / ckpt_name

        tmp_dir = Path(tempfile.mkdtemp(
            dir=self._dir, prefix=f".tmp_{ckpt_name}_"
        ))

        try:
            arr_dir = tmp_dir / _ARRAYS_DIR
            arr_dir.mkdir()

            array_files: List[str] = []
            for name, arr in arrays.items():
                safe_name = _sanitize_name(name) + ".npy"
                np.save(arr_dir / safe_name, arr)
                array_files.append(safe_name)

            state_path = tmp_dir / "state.json"
            state_json = json.dumps(
                state, indent=2, default=_checkpoint_json_default
            )
            state_path.write_text(state_json)

            manifest = CheckpointManifest(
                phase=phase,
                step=step,
                timestamp=ts,
                config_hash=self._config_hash,
                array_files=array_files,
                json_keys=list(state.keys()),
                description=description,
            )

            manifest_content = json.dumps(manifest.to_dict(), sort_keys=True)
            manifest.checksum = hashlib.sha256(
                manifest_content.encode()
            ).hexdigest()

            manifest_path = tmp_dir / _MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(manifest.to_dict(), indent=2)
            )

            if final_path.exists():
                shutil.rmtree(final_path)
            tmp_dir.rename(final_path)

            logger.info(
                "Checkpoint saved: phase=%d step=%d path=%s",
                phase, step, final_path,
            )

            self._cleanup_old_checkpoints()

            return final_path

        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    def load(
        self, path: Union[str, Path]
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], CheckpointManifest]:
        """Load a checkpoint from a specific directory.

        Parameters
        ----------
        path : str or Path
            Path to checkpoint directory.

        Returns
        -------
        state : dict
            JSON-deserialized pipeline state.
        arrays : dict of str → np.ndarray
            Named numpy arrays.
        manifest : CheckpointManifest
            Checkpoint metadata.

        Raises
        ------
        FileNotFoundError
            If checkpoint directory or manifest not found.
        ValueError
            If integrity check fails.
        """
        ckpt_dir = Path(path)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        manifest = self._load_manifest(ckpt_dir)
        self._verify_integrity(ckpt_dir, manifest)

        state_path = ckpt_dir / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"state.json not found in {ckpt_dir}")
        state = json.loads(state_path.read_text())

        arrays: Dict[str, np.ndarray] = {}
        arr_dir = ckpt_dir / _ARRAYS_DIR
        if arr_dir.exists():
            for arr_file in manifest.array_files:
                arr_path = arr_dir / arr_file
                if arr_path.exists():
                    name = arr_file.rsplit(".npy", 1)[0]
                    arrays[name] = np.load(arr_path, allow_pickle=False)

        logger.info(
            "Checkpoint loaded: phase=%d step=%d path=%s",
            manifest.phase, manifest.step, ckpt_dir,
        )
        return state, arrays, manifest

    def load_latest(
        self,
        phase: Optional[int] = None,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray], CheckpointManifest]]:
        """Load the most recent checkpoint.

        Parameters
        ----------
        phase : int, optional
            If given, only consider checkpoints from this phase.

        Returns
        -------
        tuple or None
            (state, arrays, manifest) or None if no checkpoint found.
        """
        ckpts = self.list_checkpoints(phase=phase)
        if not ckpts:
            return None
        latest = ckpts[-1]
        return self.load(latest[1])

    def load_phase(
        self, phase: int
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray], CheckpointManifest]]:
        """Load the latest checkpoint for a specific phase.

        Parameters
        ----------
        phase : int
            Phase number (1, 2, or 3).

        Returns
        -------
        tuple or None
        """
        return self.load_latest(phase=phase)

    # -----------------------------------------------------------------
    # List and inspect
    # -----------------------------------------------------------------

    def list_checkpoints(
        self, phase: Optional[int] = None
    ) -> List[Tuple[CheckpointManifest, Path]]:
        """List available checkpoints sorted by timestamp.

        Parameters
        ----------
        phase : int, optional
            Filter to checkpoints from this phase only.

        Returns
        -------
        list of (CheckpointManifest, Path)
            Sorted by timestamp (oldest first).
        """
        results: List[Tuple[CheckpointManifest, Path]] = []

        if not self._dir.exists():
            return results

        for child in self._dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                continue
            if not child.name.startswith(_CHECKPOINT_PREFIX):
                continue
            manifest_path = child / _MANIFEST_NAME
            if not manifest_path.exists():
                continue
            try:
                manifest = self._load_manifest(child)
                if phase is not None and manifest.phase != phase:
                    continue
                results.append((manifest, child))
            except Exception:
                logger.warning("Skipping corrupted checkpoint: %s", child)
                continue

        results.sort(key=lambda x: x[0].timestamp)
        return results

    def has_checkpoint(self, phase: Optional[int] = None) -> bool:
        """Check whether any checkpoint exists.

        Parameters
        ----------
        phase : int, optional
            If given, check only for this phase.

        Returns
        -------
        bool
        """
        return len(self.list_checkpoints(phase=phase)) > 0

    def latest_phase(self) -> int:
        """Return the latest phase number with a checkpoint.

        Returns
        -------
        int
            Phase number (0 if no checkpoints exist).
        """
        ckpts = self.list_checkpoints()
        if not ckpts:
            return 0
        return ckpts[-1][0].phase

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max_checkpoints."""
        ckpts = self.list_checkpoints()
        while len(ckpts) > self._max_checkpoints:
            _, oldest_path = ckpts.pop(0)
            try:
                shutil.rmtree(oldest_path)
                logger.debug("Removed old checkpoint: %s", oldest_path)
            except OSError as e:
                logger.warning(
                    "Failed to remove checkpoint %s: %s", oldest_path, e
                )

    def clear(self, phase: Optional[int] = None) -> int:
        """Remove checkpoints.

        Parameters
        ----------
        phase : int, optional
            If given, only remove checkpoints for this phase.

        Returns
        -------
        int
            Number of checkpoints removed.
        """
        ckpts = self.list_checkpoints(phase=phase)
        removed = 0
        for _, path in ckpts:
            try:
                shutil.rmtree(path)
                removed += 1
            except OSError:
                pass
        return removed

    # -----------------------------------------------------------------
    # Integrity
    # -----------------------------------------------------------------

    def _load_manifest(self, ckpt_dir: Path) -> CheckpointManifest:
        """Load and parse the manifest file."""
        manifest_path = ckpt_dir / _MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )
        data = json.loads(manifest_path.read_text())
        return CheckpointManifest.from_dict(data)

    def _verify_integrity(
        self, ckpt_dir: Path, manifest: CheckpointManifest
    ) -> None:
        """Verify checkpoint integrity via manifest checksum.

        Parameters
        ----------
        ckpt_dir : Path
            Checkpoint directory.
        manifest : CheckpointManifest
            Loaded manifest.

        Raises
        ------
        ValueError
            If integrity check fails.
        """
        stored_checksum = manifest.checksum
        manifest_copy = CheckpointManifest.from_dict(manifest.to_dict())
        manifest_copy.checksum = ""
        content = json.dumps(manifest_copy.to_dict(), sort_keys=True)
        computed = hashlib.sha256(content.encode()).hexdigest()

        if stored_checksum and computed != stored_checksum:
            raise ValueError(
                f"Checkpoint integrity check failed for {ckpt_dir}. "
                f"Expected checksum {stored_checksum}, got {computed}"
            )

        if self._config_hash and manifest.config_hash:
            if manifest.config_hash != self._config_hash:
                logger.warning(
                    "Checkpoint config hash mismatch: checkpoint=%s current=%s. "
                    "The checkpoint may be from a different configuration.",
                    manifest.config_hash,
                    self._config_hash,
                )

        arr_dir = ckpt_dir / _ARRAYS_DIR
        for arr_file in manifest.array_files:
            arr_path = arr_dir / arr_file
            if not arr_path.exists():
                raise ValueError(
                    f"Missing array file: {arr_path}"
                )

    def verify_checkpoint(self, path: Union[str, Path]) -> bool:
        """Verify a checkpoint's integrity without loading all data.

        Parameters
        ----------
        path : str or Path
            Checkpoint directory.

        Returns
        -------
        bool
            True if checkpoint passes integrity checks.
        """
        try:
            ckpt_dir = Path(path)
            manifest = self._load_manifest(ckpt_dir)
            self._verify_integrity(ckpt_dir, manifest)
            return True
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            return False

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    def disk_usage(self) -> int:
        """Total bytes used by all checkpoints.

        Returns
        -------
        int
            Total size in bytes.
        """
        total = 0
        for _, path in self.list_checkpoints():
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    def __repr__(self) -> str:
        n = len(self.list_checkpoints())
        return (
            f"CheckpointManager(dir={str(self._dir)!r}, "
            f"checkpoints={n}, max={self._max_checkpoints})"
        )


# =====================================================================
# Helpers
# =====================================================================


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a file path component."""
    safe = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in "_-.")
    return safe or "unnamed"


def _checkpoint_json_default(obj: Any) -> Any:
    """JSON default handler for checkpoint serialization."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute a stable hash for a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    content = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
