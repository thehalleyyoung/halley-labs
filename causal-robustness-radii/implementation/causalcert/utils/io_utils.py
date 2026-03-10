"""I/O helpers: file-format detection, serialisation, checksums, progress."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
from numpy.typing import NDArray


# ====================================================================
# 1. File-Format Detection
# ====================================================================

_FORMAT_SIGNATURES: dict[str, list[bytes]] = {
    "parquet": [b"PAR1"],
    "gzip": [b"\x1f\x8b"],
    "zip": [b"PK\x03\x04"],
    "csv": [],  # fallback
}


def detect_format(path: str | Path) -> str:
    """Detect file format from magic bytes or extension.

    Returns one of: ``"parquet"``, ``"csv"``, ``"json"``, ``"dot"``,
    ``"gml"``, ``"gzip"``, ``"zip"``, ``"unknown"``.
    """
    p = Path(path)

    # Extension-based shortcuts
    ext_map = {
        ".parquet": "parquet",
        ".csv": "csv",
        ".tsv": "csv",
        ".json": "json",
        ".dot": "dot",
        ".gv": "dot",
        ".gml": "gml",
        ".graphml": "gml",
        ".bif": "bif",
    }
    ext = p.suffix.lower()
    if ext in ext_map:
        return ext_map[ext]

    # Magic-byte detection
    try:
        with open(p, "rb") as f:
            header = f.read(8)
    except OSError:
        return "unknown"

    for fmt, sigs in _FORMAT_SIGNATURES.items():
        for sig in sigs:
            if header.startswith(sig):
                return fmt

    # Heuristic: try to read as JSON
    try:
        with open(p) as f:
            json.load(f)
        return "json"
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        pass

    # Default
    return "csv" if ext in {".txt", ""} else "unknown"


def file_size_human(path: str | Path) -> str:
    """Return human-readable file size (e.g. '1.2 MB')."""
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


# ====================================================================
# 2. Temporary File Management
# ====================================================================

class TempFileManager:
    """Context manager that creates a temporary directory and cleans up."""

    def __init__(self, prefix: str = "causalcert_") -> None:
        self._prefix = prefix
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._tmpdir = tempfile.TemporaryDirectory(prefix=self._prefix)
        return Path(self._tmpdir.name)

    def __exit__(self, *exc: Any) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def temp_path(self, name: str) -> Path:
        """Return a path inside the managed directory."""
        if self._tmpdir is None:
            raise RuntimeError("TempFileManager not entered")
        return Path(self._tmpdir.name) / name


def atomic_write(path: str | Path, content: str | bytes) -> None:
    """Write *content* to *path* atomically (write-then-rename)."""
    p = Path(path)
    mode = "wb" if isinstance(content, bytes) else "w"
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, mode) as f:
            f.write(content)
        os.replace(tmp, p)
    except BaseException:
        os.unlink(tmp)
        raise


# ====================================================================
# 3. Serialisation Helpers
# ====================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def to_json(obj: Any, indent: int = 2) -> str:
    """Serialise *obj* to JSON, handling NumPy types."""
    return json.dumps(obj, cls=NumpyEncoder, indent=indent)


def from_json_file(path: str | Path) -> Any:
    """Load JSON from a file."""
    with open(path) as f:
        return json.load(f)


def save_adjacency_npz(adj: NDArray, path: str | Path) -> None:
    """Save an adjacency matrix as a compressed .npz file."""
    np.savez_compressed(path, adj=adj)


def load_adjacency_npz(path: str | Path) -> NDArray:
    """Load an adjacency matrix from a .npz file."""
    data = np.load(path)
    return data["adj"]


# ====================================================================
# 4. Checksum Computation
# ====================================================================

def sha256_file(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """Deterministic hash of a JSON-serialisable object."""
    s = json.dumps(obj, cls=NumpyEncoder, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ====================================================================
# 5. Progress Reporting
# ====================================================================

class ProgressReporter:
    """Simple text-based progress bar for long-running operations."""

    def __init__(
        self,
        total: int,
        description: str = "",
        bar_width: int = 40,
        file: Any = None,
    ) -> None:
        self.total = total
        self.description = description
        self.bar_width = bar_width
        self.file = file or sys.stderr
        self._current = 0
        self._start_time = time.monotonic()

    def update(self, n: int = 1) -> None:
        """Advance the progress bar by *n* steps."""
        self._current = min(self._current + n, self.total)
        self._render()

    def _render(self) -> None:
        frac = self._current / self.total if self.total > 0 else 1.0
        filled = int(self.bar_width * frac)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        elapsed = time.monotonic() - self._start_time
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0

        line = (
            f"\r{self.description} |{bar}| "
            f"{self._current}/{self.total} "
            f"[{elapsed:.0f}s<{eta:.0f}s]"
        )
        self.file.write(line)
        self.file.flush()
        if self._current >= self.total:
            self.file.write("\n")

    def close(self) -> None:
        """Ensure the bar completes."""
        if self._current < self.total:
            self._current = self.total
            self._render()


def iterate_with_progress(
    iterable: Iterator,
    total: int,
    description: str = "",
) -> Iterator:
    """Wrap an iterator with a progress bar."""
    reporter = ProgressReporter(total, description)
    for item in iterable:
        yield item
        reporter.update()
    reporter.close()
