"""
Atomic checkpointing and rollback for repair execution.

:class:`CheckpointManager` provides transaction-like semantics on top
of DuckDB, using temporary table snapshots for safe rollback.

Workflow::

    mgr = CheckpointManager(engine)
    cp = mgr.create_checkpoint()
    mgr.save_table_state(cp, "my_table")
    try:
        ... execute actions ...
        mgr.commit_checkpoint(cp)
    except Exception:
        mgr.restore_checkpoint(cp)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

from arc.types.base import CheckpointInfo

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Atomic checkpointing and rollback for repair execution.

    Uses DuckDB temporary tables to snapshot table state before
    mutations.  On rollback, the snapshot is copied back.

    Parameters
    ----------
    engine:
        The execution engine whose database to checkpoint.
    prefix:
        Table-name prefix for checkpoint snapshots.
    """

    def __init__(
        self,
        engine: Any,
        prefix: str = "_arc_ckpt_",
    ) -> None:
        self._engine = engine
        self._conn: duckdb.DuckDBPyConnection = engine.connection
        self._prefix = prefix
        self._checkpoints: dict[str, CheckpointInfo] = {}
        self._snapshot_tables: dict[str, list[str]] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def create_checkpoint(self, metadata: dict[str, Any] | None = None) -> str:
        """Create a new checkpoint and return its ID.

        The checkpoint is initially empty; call :meth:`save_table_state`
        to snapshot specific tables.
        """
        cp_id = uuid.uuid4().hex[:16]
        info = CheckpointInfo(
            checkpoint_id=cp_id,
            created_at=time.time(),
            metadata=metadata or {},
        )
        self._checkpoints[cp_id] = info
        self._snapshot_tables[cp_id] = []
        logger.info("Created checkpoint %s", cp_id)
        return cp_id

    def save_table_state(self, checkpoint_id: str, table_name: str) -> None:
        """Snapshot *table_name* under the given checkpoint.

        The snapshot is a CREATE TABLE AS SELECT into a temporary table.
        If the source table does not exist, a marker is stored so that
        rollback will drop any table created after the checkpoint.

        Parameters
        ----------
        checkpoint_id:
            ID of an active checkpoint.
        table_name:
            Name of the table to snapshot.

        Raises
        ------
        KeyError
            If the checkpoint ID is unknown.
        """
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")

        snap_name = f"{self._prefix}{checkpoint_id}_{table_name}"

        try:
            self._conn.execute(
                f'CREATE TABLE "{snap_name}" AS SELECT * FROM "{table_name}"'
            )
            self._snapshot_tables[checkpoint_id].append(table_name)
            logger.debug("Saved snapshot of %s as %s", table_name, snap_name)
        except duckdb.CatalogException:
            # Table doesn't exist yet — record a "does not exist" marker
            self._conn.execute(
                f'CREATE TABLE "{snap_name}" (___arc_marker___ BOOLEAN)'
            )
            self._snapshot_tables[checkpoint_id].append(table_name)
            logger.debug(
                "Table %s does not exist; marker created at %s",
                table_name, snap_name,
            )

        # Update checkpoint info
        info = self._checkpoints[checkpoint_id]
        saved = set(info.tables_saved)
        saved.add(table_name)
        self._checkpoints[checkpoint_id] = CheckpointInfo(
            checkpoint_id=info.checkpoint_id,
            created_at=info.created_at,
            tables_saved=tuple(sorted(saved)),
            size_bytes=info.size_bytes,
            metadata=info.metadata,
        )

    def restore_checkpoint(self, checkpoint_id: str) -> None:
        """Rollback to the given checkpoint.

        For each snapshotted table:
        * If the snapshot is a marker (table didn't exist), drop the
          current table.
        * Otherwise, replace the current table with the snapshot.

        Parameters
        ----------
        checkpoint_id:
            ID of the checkpoint to restore.

        Raises
        ------
        KeyError
            If the checkpoint ID is unknown.
        """
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")

        tables = self._snapshot_tables.get(checkpoint_id, [])

        for table_name in tables:
            snap_name = f"{self._prefix}{checkpoint_id}_{table_name}"

            try:
                # Check if it's a marker
                result = self._conn.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{snap_name}' AND column_name = '___arc_marker___'"
                )
                is_marker = len(result.fetchall()) > 0

                if is_marker:
                    # Table didn't exist at checkpoint time — drop it
                    self._conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                    logger.debug("Rollback: dropped %s (didn't exist at checkpoint)", table_name)
                else:
                    # Restore from snapshot
                    self._conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                    self._conn.execute(
                        f'CREATE TABLE "{table_name}" AS SELECT * FROM "{snap_name}"'
                    )
                    logger.debug("Rollback: restored %s from %s", table_name, snap_name)

            except Exception as exc:
                logger.error("Rollback failed for %s: %s", table_name, exc)

        # Clean up snapshot tables
        self._cleanup_snapshot_tables(checkpoint_id)
        logger.info("Restored checkpoint %s (%d tables)", checkpoint_id, len(tables))

    def commit_checkpoint(self, checkpoint_id: str) -> None:
        """Commit the checkpoint, making all changes permanent.

        This simply cleans up the snapshot tables.

        Parameters
        ----------
        checkpoint_id:
            ID of the checkpoint to commit.
        """
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")

        self._cleanup_snapshot_tables(checkpoint_id)
        del self._checkpoints[checkpoint_id]
        del self._snapshot_tables[checkpoint_id]
        logger.info("Committed checkpoint %s", checkpoint_id)

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """Return all active checkpoints."""
        return list(self._checkpoints.values())

    def cleanup_old_checkpoints(self, max_age_seconds: float) -> int:
        """Remove checkpoints older than *max_age_seconds*.

        Returns the number of checkpoints removed.
        """
        now = time.time()
        to_remove: list[str] = []
        for cp_id, info in self._checkpoints.items():
            if now - info.created_at > max_age_seconds:
                to_remove.append(cp_id)

        for cp_id in to_remove:
            try:
                self._cleanup_snapshot_tables(cp_id)
            except Exception:
                pass
            self._checkpoints.pop(cp_id, None)
            self._snapshot_tables.pop(cp_id, None)

        if to_remove:
            logger.info("Cleaned up %d old checkpoints", len(to_remove))
        return len(to_remove)

    def get_checkpoint_size(self, checkpoint_id: str) -> int:
        """Estimate the total size of checkpoint snapshots in bytes.

        This is a rough estimate based on row counts and assumed
        row widths.
        """
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")

        total_bytes = 0
        for table_name in self._snapshot_tables.get(checkpoint_id, []):
            snap_name = f"{self._prefix}{checkpoint_id}_{table_name}"
            try:
                result = self._conn.execute(f'SELECT COUNT(*) FROM "{snap_name}"')
                row = result.fetchone()
                if row is not None:
                    total_bytes += row[0] * 64  # rough estimate: 64 bytes/row
            except Exception:
                pass

        return total_bytes

    # ── Private helpers ────────────────────────────────────────────────

    def _cleanup_snapshot_tables(self, checkpoint_id: str) -> None:
        """Drop all snapshot tables for a checkpoint."""
        for table_name in self._snapshot_tables.get(checkpoint_id, []):
            snap_name = f"{self._prefix}{checkpoint_id}_{table_name}"
            try:
                self._conn.execute(f'DROP TABLE IF EXISTS "{snap_name}"')
            except Exception:
                pass

    def __repr__(self) -> str:
        return f"CheckpointManager(active={len(self._checkpoints)})"
