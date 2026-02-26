"""
Checkpoint/restart module for the CausalBound pipeline.

Provides atomic, crash-safe checkpoint persistence with integrity verification,
version management, and pipeline resume capabilities.
"""

import enum
import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = "1.3.0"
CHECKPOINT_EXTENSION = ".ckpt.gz"
CHECKPOINT_MAGIC = b"CBOUND_CKPT"
_VERSION_COMPAT_MAJOR = 1


class PipelineStage(enum.Enum):
    """Ordered stages of the CausalBound pipeline."""
    DECOMPOSITION = "decomposition"
    LP_SOLVING = "lp_solving"
    INFERENCE = "inference"
    VERIFICATION = "verification"
    MCTS_SEARCH = "mcts_search"
    COMPOSITION = "composition"

    @classmethod
    def ordered(cls) -> List["PipelineStage"]:
        return [
            cls.DECOMPOSITION,
            cls.LP_SOLVING,
            cls.INFERENCE,
            cls.VERIFICATION,
            cls.MCTS_SEARCH,
            cls.COMPOSITION,
        ]

    @classmethod
    def from_string(cls, name: str) -> "PipelineStage":
        name_lower = name.lower().strip()
        for stage in cls:
            if stage.value == name_lower:
                return stage
        raise ValueError(f"Unknown pipeline stage: {name!r}")

    def next_stage(self) -> Optional["PipelineStage"]:
        ordered = self.ordered()
        idx = ordered.index(self)
        if idx + 1 < len(ordered):
            return ordered[idx + 1]
        return None

    def progress_fraction(self) -> float:
        ordered = self.ordered()
        idx = ordered.index(self)
        return (idx + 1) / len(ordered)


class CheckpointError(Exception):
    """Raised when checkpoint operations fail."""

    def __init__(self, message: str, path: Optional[str] = None, cause: Optional[Exception] = None):
        self.path = path
        self.cause = cause
        full_msg = message
        if path:
            full_msg = f"{message} [path={path}]"
        if cause:
            full_msg = f"{full_msg} (caused by {type(cause).__name__}: {cause})"
        super().__init__(full_msg)


@dataclass
class CheckpointInfo:
    """Summary information about a checkpoint file."""
    path: str
    size_bytes: int
    timestamp: float
    version: str
    stage: Optional[str] = None
    is_valid: bool = True
    checksum: Optional[str] = None

    @property
    def size_human(self) -> str:
        size = self.size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "timestamp": self.timestamp,
            "version": self.version,
            "stage": self.stage,
            "is_valid": self.is_valid,
            "checksum": self.checksum,
        }


@dataclass
class CheckpointData:
    """Loaded checkpoint with its full state and metadata."""
    state: Dict[str, Any]
    metadata: Dict[str, Any]
    is_valid: bool
    info: CheckpointInfo

    def get_stage(self) -> Optional[PipelineStage]:
        stage_str = self.metadata.get("pipeline_stage")
        if stage_str is None:
            return None
        try:
            return PipelineStage.from_string(stage_str)
        except ValueError:
            return None

    @property
    def version(self) -> str:
        return self.metadata.get("version", "0.0.0")

    @property
    def timestamp(self) -> float:
        return self.metadata.get("timestamp", 0.0)


@dataclass
class PipelineResumeInfo:
    """Information needed to resume a pipeline from a checkpoint."""
    stage: PipelineStage
    state: Dict[str, Any]
    remaining_stages: List[PipelineStage]
    estimated_progress: float
    checkpoint_path: str
    is_partial: bool = False
    partial_keys: List[str] = field(default_factory=list)

    @property
    def num_remaining(self) -> int:
        return len(self.remaining_stages)

    def summary(self) -> str:
        status = "partial" if self.is_partial else "complete"
        return (
            f"Resume from {self.stage.value} ({status}), "
            f"{self.num_remaining} stages remaining, "
            f"{self.estimated_progress:.0%} overall progress"
        )


@dataclass
class CheckpointDiff:
    """Result of comparing two checkpoints."""
    keys_added: List[str]
    keys_removed: List[str]
    keys_changed: List[str]
    keys_unchanged: List[str]
    numeric_deltas: Dict[str, Tuple[float, float, float]]
    stage_transition: Optional[Tuple[str, str]] = None

    @property
    def has_changes(self) -> bool:
        return bool(self.keys_added or self.keys_removed or self.keys_changed)

    def summary(self) -> str:
        parts = []
        if self.keys_added:
            parts.append(f"{len(self.keys_added)} added")
        if self.keys_removed:
            parts.append(f"{len(self.keys_removed)} removed")
        if self.keys_changed:
            parts.append(f"{len(self.keys_changed)} changed")
        if self.keys_unchanged:
            parts.append(f"{len(self.keys_unchanged)} unchanged")
        if self.stage_transition:
            parts.append(f"stage: {self.stage_transition[0]} -> {self.stage_transition[1]}")
        return ", ".join(parts) if parts else "no differences"


def _compute_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_version(version_str: str) -> Tuple[int, int, int]:
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version_str!r}, expected X.Y.Z")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError(f"Non-integer version component in {version_str!r}") from exc


def _check_version_compatibility(version: str) -> bool:
    try:
        major, minor, _patch = _parse_version(version)
    except ValueError:
        return False
    current_major, current_minor, _ = _parse_version(CHECKPOINT_VERSION)
    if major != current_major:
        return False
    if major == 0:
        return minor == current_minor
    return minor <= current_minor


def _migrate_v1_0_to_v1_1(metadata: Dict[str, Any], state: Dict[str, Any]) -> Tuple[Dict, Dict]:
    if "pipeline_stage" not in metadata and "stage" in metadata:
        metadata["pipeline_stage"] = metadata.pop("stage")
    if "checksum" not in metadata:
        metadata["checksum"] = ""
    return metadata, state


def _migrate_v1_1_to_v1_2(metadata: Dict[str, Any], state: Dict[str, Any]) -> Tuple[Dict, Dict]:
    if "config" in state and isinstance(state["config"], list):
        state["config"] = {f"param_{i}": v for i, v in enumerate(state["config"])}
    if "format_version" not in metadata:
        metadata["format_version"] = 2
    return metadata, state


def _migrate_v1_2_to_v1_3(metadata: Dict[str, Any], state: Dict[str, Any]) -> Tuple[Dict, Dict]:
    if "bounds" in state and isinstance(state["bounds"], dict):
        if "history" not in state["bounds"]:
            state["bounds"]["history"] = []
    metadata["format_version"] = 3
    return metadata, state


_MIGRATIONS = [
    ("1.0", "1.1", _migrate_v1_0_to_v1_1),
    ("1.1", "1.2", _migrate_v1_1_to_v1_2),
    ("1.2", "1.3", _migrate_v1_2_to_v1_3),
]


def _apply_migrations(
    version: str, metadata: Dict[str, Any], state: Dict[str, Any]
) -> Tuple[Dict, Dict, str]:
    major, minor, _patch = _parse_version(version)
    current_major, current_minor, _ = _parse_version(CHECKPOINT_VERSION)
    if major != current_major:
        return metadata, state, version
    for from_ver, to_ver, migrate_fn in _MIGRATIONS:
        from_maj, from_min, _ = _parse_version(from_ver + ".0")
        if major == from_maj and minor <= from_min:
            logger.info("Migrating checkpoint from v%s to v%s", from_ver, to_ver)
            metadata, state = migrate_fn(metadata, state)
            to_maj, to_min, _ = _parse_version(to_ver + ".0")
            major, minor = to_maj, to_min
    final_version = f"{major}.{minor}.0"
    metadata["version"] = final_version
    return metadata, state, final_version


def _detect_stage_keys(state: Dict[str, Any]) -> Optional[PipelineStage]:
    stage_key_map = {
        PipelineStage.COMPOSITION: ["composed_result", "final_bound"],
        PipelineStage.MCTS_SEARCH: ["mcts_progress", "mcts_tree", "search_result"],
        PipelineStage.VERIFICATION: ["verification_result", "proof_status"],
        PipelineStage.INFERENCE: ["inference_result", "posterior"],
        PipelineStage.LP_SOLVING: ["lp_solutions", "lp_result", "optimal_value"],
        PipelineStage.DECOMPOSITION: ["decomposition", "subproblems", "graph"],
    }
    for stage, keys in stage_key_map.items():
        if any(k in state for k in keys):
            return stage
    return None


def _expected_keys_for_stage(stage: PipelineStage) -> List[str]:
    key_map = {
        PipelineStage.DECOMPOSITION: ["decomposition"],
        PipelineStage.LP_SOLVING: ["decomposition", "lp_solutions"],
        PipelineStage.INFERENCE: ["decomposition", "lp_solutions", "inference_result"],
        PipelineStage.VERIFICATION: ["decomposition", "lp_solutions", "inference_result", "verification_result"],
        PipelineStage.MCTS_SEARCH: ["decomposition", "lp_solutions", "mcts_progress"],
        PipelineStage.COMPOSITION: ["decomposition", "lp_solutions", "composed_result"],
    }
    return key_map.get(stage, [])


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class CheckpointManager:
    """Manages checkpoint save/load/resume for the CausalBound pipeline."""

    def __init__(self, base_dir: Optional[str] = None, compress: bool = True):
        self.base_dir = Path(base_dir) if base_dir else None
        self.compress = compress
        self._write_count = 0
        self._read_count = 0
        logger.debug(
            "CheckpointManager initialized (base_dir=%s, compress=%s)",
            self.base_dir,
            self.compress,
        )

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointInfo:
        """Save pipeline state atomically to a compressed checkpoint file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        meta = dict(metadata) if metadata else {}
        meta["timestamp"] = time.time()
        meta["version"] = CHECKPOINT_VERSION
        meta["format_version"] = 3

        if "pipeline_stage" not in meta:
            detected = _detect_stage_keys(state)
            if detected is not None:
                meta["pipeline_stage"] = detected.value

        state_keys = sorted(state.keys())
        meta["state_keys"] = state_keys
        meta["num_state_keys"] = len(state_keys)

        payload = {
            "magic": CHECKPOINT_MAGIC,
            "state": state,
            "metadata": meta,
        }

        raw_data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        checksum = _compute_checksum(raw_data)
        meta["checksum"] = checksum

        payload["metadata"] = meta
        raw_data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

        if self.compress:
            write_data = gzip.compress(raw_data, compresslevel=6)
        else:
            write_data = raw_data

        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=".ckpt_tmp_",
            suffix=".partial",
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(write_data)
                f.flush()
                os.fsync(f.fileno())
            shutil.move(tmp_path, str(target))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        file_size = target.stat().st_size
        self._write_count += 1

        info = CheckpointInfo(
            path=str(target),
            size_bytes=file_size,
            timestamp=meta["timestamp"],
            version=CHECKPOINT_VERSION,
            stage=meta.get("pipeline_stage"),
            is_valid=True,
            checksum=checksum,
        )
        logger.info(
            "Checkpoint saved: %s (%s, stage=%s)",
            target.name,
            info.size_human,
            info.stage,
        )
        return info

    def load_checkpoint(self, path: str) -> CheckpointData:
        """Load a checkpoint file with integrity and version checks."""
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise CheckpointError(f"Checkpoint file not found", path=path)

        file_size = ckpt_path.stat().st_size
        if file_size == 0:
            raise CheckpointError("Checkpoint file is empty", path=path)

        try:
            raw_bytes = ckpt_path.read_bytes()
        except OSError as exc:
            raise CheckpointError("Failed to read checkpoint file", path=path, cause=exc)

        try:
            decompressed = gzip.decompress(raw_bytes)
        except gzip.BadGzipFile:
            decompressed = raw_bytes
        except OSError as exc:
            raise CheckpointError("Decompression failed", path=path, cause=exc)

        try:
            payload = pickle.loads(decompressed)
        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            raise CheckpointError("Failed to unpickle checkpoint", path=path, cause=exc)

        if not isinstance(payload, dict):
            raise CheckpointError("Invalid checkpoint structure: expected dict", path=path)

        if payload.get("magic") != CHECKPOINT_MAGIC:
            raise CheckpointError("Invalid checkpoint magic bytes", path=path)

        state = payload.get("state", {})
        metadata = payload.get("metadata", {})

        stored_checksum = metadata.get("checksum", "")
        if stored_checksum:
            verify_meta = dict(metadata)
            verify_meta.pop("checksum", None)
            verify_payload = {
                "magic": CHECKPOINT_MAGIC,
                "state": state,
                "metadata": verify_meta,
            }
            verify_data = pickle.dumps(verify_payload, protocol=pickle.HIGHEST_PROTOCOL)
            computed = _compute_checksum(verify_data)
            if computed != stored_checksum:
                raise CheckpointError(
                    f"Checksum mismatch: expected {stored_checksum[:16]}..., "
                    f"got {computed[:16]}...",
                    path=path,
                )

        version = metadata.get("version", "0.0.0")
        if not _check_version_compatibility(version):
            raise CheckpointError(
                f"Incompatible checkpoint version {version} "
                f"(current: {CHECKPOINT_VERSION})",
                path=path,
            )

        if version != CHECKPOINT_VERSION:
            metadata, state, version = _apply_migrations(version, metadata, state)
            logger.info("Migrated checkpoint to version %s", version)

        self._read_count += 1

        info = CheckpointInfo(
            path=str(ckpt_path),
            size_bytes=file_size,
            timestamp=metadata.get("timestamp", 0.0),
            version=version,
            stage=metadata.get("pipeline_stage"),
            is_valid=True,
            checksum=stored_checksum,
        )

        return CheckpointData(
            state=state,
            metadata=metadata,
            is_valid=True,
            info=info,
        )

    def get_latest_checkpoint(self, directory: str) -> Optional[CheckpointInfo]:
        """Find the most recent valid checkpoint in a directory."""
        ckpts = self.list_checkpoints(directory)
        valid = [c for c in ckpts if c.is_valid]
        if not valid:
            logger.debug("No valid checkpoints found in %s", directory)
            return None

        valid.sort(key=lambda c: c.timestamp, reverse=True)

        for candidate in valid:
            try:
                self.load_checkpoint(candidate.path)
                logger.info("Latest valid checkpoint: %s", candidate.path)
                return candidate
            except CheckpointError as exc:
                logger.warning(
                    "Checkpoint %s appeared valid but failed load: %s",
                    candidate.path,
                    exc,
                )
                continue

        logger.warning("No loadable checkpoints found in %s", directory)
        return None

    def resume_pipeline(self, checkpoint: CheckpointData) -> PipelineResumeInfo:
        """Determine how to resume the pipeline from a loaded checkpoint."""
        stage = checkpoint.get_stage()
        if stage is None:
            stage = _detect_stage_keys(checkpoint.state)
        if stage is None:
            stage = PipelineStage.DECOMPOSITION
            logger.warning("Could not determine stage; defaulting to DECOMPOSITION")

        ordered = PipelineStage.ordered()
        stage_idx = ordered.index(stage)

        is_complete = self._is_stage_complete(stage, checkpoint.state)
        if is_complete and stage.next_stage() is not None:
            resume_stage = stage.next_stage()
            remaining = ordered[ordered.index(resume_stage):]
        else:
            resume_stage = stage
            remaining = ordered[stage_idx:]

        expected = _expected_keys_for_stage(stage)
        present = [k for k in expected if k in checkpoint.state]
        is_partial = len(present) < len(expected) if expected else False
        partial_keys = [k for k in expected if k not in checkpoint.state]

        progress = stage.progress_fraction()
        if is_complete and resume_stage != stage:
            progress = resume_stage.progress_fraction() - (1.0 / len(ordered))

        resume_info = PipelineResumeInfo(
            stage=resume_stage,
            state=checkpoint.state,
            remaining_stages=remaining,
            estimated_progress=progress,
            checkpoint_path=checkpoint.info.path,
            is_partial=is_partial,
            partial_keys=partial_keys,
        )
        logger.info("Pipeline resume: %s", resume_info.summary())
        return resume_info

    def _is_stage_complete(self, stage: PipelineStage, state: Dict[str, Any]) -> bool:
        completeness_indicators = {
            PipelineStage.DECOMPOSITION: lambda s: (
                "decomposition" in s and isinstance(s["decomposition"], dict)
                and s["decomposition"].get("complete", False)
            ),
            PipelineStage.LP_SOLVING: lambda s: (
                "lp_solutions" in s and isinstance(s["lp_solutions"], (dict, list))
                and bool(s["lp_solutions"])
            ),
            PipelineStage.INFERENCE: lambda s: "inference_result" in s,
            PipelineStage.VERIFICATION: lambda s: (
                "verification_result" in s
                and s.get("verification_result", {}).get("verified", False)
            ),
            PipelineStage.MCTS_SEARCH: lambda s: (
                "mcts_progress" in s
                and isinstance(s["mcts_progress"], dict)
                and s["mcts_progress"].get("finished", False)
            ),
            PipelineStage.COMPOSITION: lambda s: "composed_result" in s,
        }
        checker = completeness_indicators.get(stage)
        if checker is None:
            return False
        try:
            return checker(state)
        except (TypeError, KeyError, AttributeError):
            return False

    def list_checkpoints(
        self, directory: str, stage_filter: Optional[str] = None
    ) -> List[CheckpointInfo]:
        """List all checkpoint files in a directory, sorted by timestamp."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.debug("Directory does not exist: %s", directory)
            return []

        patterns = ["*.ckpt", "*.ckpt.gz"]
        found_files = []
        for pattern in patterns:
            found_files.extend(dir_path.glob(pattern))

        seen_paths = set()
        unique_files = []
        for f in found_files:
            resolved = str(f.resolve())
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                unique_files.append(f)

        results = []
        for ckpt_file in unique_files:
            info = self._probe_checkpoint(ckpt_file)
            if info is not None:
                if stage_filter is not None:
                    if info.stage != stage_filter:
                        continue
                results.append(info)

        results.sort(key=lambda c: c.timestamp)
        logger.debug("Found %d checkpoints in %s", len(results), directory)
        return results

    def _probe_checkpoint(self, path: Path) -> Optional[CheckpointInfo]:
        """Extract info from a checkpoint without fully loading state."""
        try:
            file_size = path.stat().st_size
            if file_size == 0:
                return CheckpointInfo(
                    path=str(path), size_bytes=0, timestamp=0.0,
                    version="unknown", is_valid=False,
                )

            raw = path.read_bytes()

            try:
                decompressed = gzip.decompress(raw)
            except (gzip.BadGzipFile, OSError):
                decompressed = raw

            payload = pickle.loads(decompressed)

            if not isinstance(payload, dict) or payload.get("magic") != CHECKPOINT_MAGIC:
                return CheckpointInfo(
                    path=str(path), size_bytes=file_size, timestamp=0.0,
                    version="unknown", is_valid=False,
                )

            meta = payload.get("metadata", {})
            return CheckpointInfo(
                path=str(path),
                size_bytes=file_size,
                timestamp=meta.get("timestamp", os.path.getmtime(str(path))),
                version=meta.get("version", "unknown"),
                stage=meta.get("pipeline_stage"),
                is_valid=True,
                checksum=meta.get("checksum"),
            )
        except Exception as exc:
            logger.debug("Failed to probe checkpoint %s: %s", path, exc)
            return CheckpointInfo(
                path=str(path),
                size_bytes=path.stat().st_size if path.exists() else 0,
                timestamp=0.0,
                version="unknown",
                is_valid=False,
            )

    def cleanup_old_checkpoints(self, directory: str, keep_n: int = 5) -> List[str]:
        """Remove old checkpoints, keeping the N most recent valid ones."""
        if keep_n < 0:
            raise ValueError("keep_n must be non-negative")

        all_ckpts = self.list_checkpoints(directory)
        if len(all_ckpts) <= keep_n:
            logger.debug(
                "Only %d checkpoints found, nothing to clean (keep_n=%d)",
                len(all_ckpts),
                keep_n,
            )
            return []

        by_time = sorted(all_ckpts, key=lambda c: c.timestamp, reverse=True)
        to_keep = by_time[:keep_n]
        to_remove = by_time[keep_n:]
        keep_paths = {c.path for c in to_keep}

        removed = []
        for ckpt in to_remove:
            if ckpt.path in keep_paths:
                continue
            try:
                os.unlink(ckpt.path)
                removed.append(ckpt.path)
                logger.info("Removed old checkpoint: %s", ckpt.path)
            except OSError as exc:
                logger.warning("Failed to remove checkpoint %s: %s", ckpt.path, exc)

        logger.info(
            "Cleanup complete: removed %d, kept %d checkpoints",
            len(removed),
            len(to_keep),
        )
        return removed

    def diff_checkpoints(self, ckpt1_path: str, ckpt2_path: str) -> CheckpointDiff:
        """Compare two checkpoints and report differences."""
        data1 = self.load_checkpoint(ckpt1_path)
        data2 = self.load_checkpoint(ckpt2_path)

        keys1 = set(data1.state.keys())
        keys2 = set(data2.state.keys())

        added = sorted(keys2 - keys1)
        removed = sorted(keys1 - keys2)
        common = keys1 & keys2

        changed = []
        unchanged = []
        numeric_deltas = {}

        for key in sorted(common):
            val1 = data1.state[key]
            val2 = data2.state[key]
            if self._values_equal(val1, val2):
                unchanged.append(key)
            else:
                changed.append(key)
                if _is_numeric(val1) and _is_numeric(val2):
                    delta = float(val2) - float(val1)
                    numeric_deltas[key] = (float(val1), float(val2), delta)
                elif isinstance(val1, dict) and isinstance(val2, dict):
                    self._collect_nested_numeric_deltas(
                        val1, val2, key, numeric_deltas
                    )

        stage1 = data1.metadata.get("pipeline_stage")
        stage2 = data2.metadata.get("pipeline_stage")
        stage_transition = None
        if stage1 != stage2 and stage1 is not None and stage2 is not None:
            stage_transition = (stage1, stage2)

        diff = CheckpointDiff(
            keys_added=added,
            keys_removed=removed,
            keys_changed=changed,
            keys_unchanged=unchanged,
            numeric_deltas=numeric_deltas,
            stage_transition=stage_transition,
        )
        logger.info("Checkpoint diff: %s", diff.summary())
        return diff

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        if type(val1) != type(val2):
            return False
        if _is_numeric(val1) and _is_numeric(val2):
            return abs(float(val1) - float(val2)) < 1e-12
        try:
            return val1 == val2
        except Exception:
            return False

    def _collect_nested_numeric_deltas(
        self,
        dict1: Dict[str, Any],
        dict2: Dict[str, Any],
        prefix: str,
        deltas: Dict[str, Tuple[float, float, float]],
    ) -> None:
        common_keys = set(dict1.keys()) & set(dict2.keys())
        for key in common_keys:
            v1 = dict1[key]
            v2 = dict2[key]
            full_key = f"{prefix}.{key}"
            if _is_numeric(v1) and _is_numeric(v2):
                if abs(float(v1) - float(v2)) > 1e-12:
                    deltas[full_key] = (float(v1), float(v2), float(v2) - float(v1))
            elif isinstance(v1, dict) and isinstance(v2, dict):
                self._collect_nested_numeric_deltas(v1, v2, full_key, deltas)

    def generate_checkpoint_path(
        self, directory: str, stage: Optional[str] = None, suffix: str = ""
    ) -> str:
        """Generate a unique checkpoint filename with timestamp."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        parts = ["ckpt", str(ts)]
        if stage:
            parts.insert(1, stage)
        if suffix:
            parts.append(suffix)
        filename = "_".join(parts) + CHECKPOINT_EXTENSION
        return str(dir_path / filename)

    def export_metadata_json(self, path: str, output_path: str) -> str:
        """Export checkpoint metadata as a human-readable JSON file."""
        data = self.load_checkpoint(path)
        meta_export = {
            "checkpoint_file": path,
            "version": data.version,
            "timestamp": data.timestamp,
            "timestamp_human": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(data.timestamp)
            ),
            "pipeline_stage": data.metadata.get("pipeline_stage"),
            "state_keys": sorted(data.state.keys()),
            "num_state_keys": len(data.state),
            "checksum": data.metadata.get("checksum", ""),
            "is_valid": data.is_valid,
            "file_size": data.info.size_bytes,
            "file_size_human": data.info.size_human,
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(meta_export, f, indent=2, default=str)
        logger.info("Exported metadata to %s", output_path)
        return str(out)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "writes": self._write_count,
            "reads": self._read_count,
        }

    def verify_checkpoint(self, path: str) -> Tuple[bool, str]:
        """Verify a checkpoint's integrity without fully loading state into memory."""
        try:
            data = self.load_checkpoint(path)
            if not data.is_valid:
                return False, "Checkpoint data marked as invalid"
            if not data.state:
                return False, "Checkpoint state is empty"
            version = data.metadata.get("version", "0.0.0")
            if not _check_version_compatibility(version):
                return False, f"Incompatible version: {version}"
            return True, "Checkpoint is valid"
        except CheckpointError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Unexpected error: {exc}"

    def create_snapshot(
        self,
        state: Dict[str, Any],
        directory: str,
        stage: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointInfo:
        """Convenience: generate a path and save a checkpoint in one call."""
        path = self.generate_checkpoint_path(directory, stage=stage)
        metadata = {"pipeline_stage": stage}
        if extra_metadata:
            metadata.update(extra_metadata)
        return self.save_checkpoint(state, path, metadata=metadata)
