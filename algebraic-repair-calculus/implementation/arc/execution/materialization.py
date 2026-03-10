"""
Materialization Manager
========================

Manages materialized views for efficient incremental repair. Tracks
which pipeline nodes have been materialized, supports incremental and
full refreshes, estimates storage costs, and provides cleanup of stale
materialized views.

Materialized views are DuckDB tables that cache the output of pipeline
nodes, enabling fast incremental updates instead of full recomputation.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    HAS_DUCKDB = False

logger = logging.getLogger(__name__)


# =====================================================================
# Materialized View State
# =====================================================================


class ViewState(Enum):
    """State of a materialized view."""
    VALID = "valid"
    STALE = "stale"
    REFRESHING = "refreshing"
    INVALIDATED = "invalidated"
    ERROR = "error"


class RefreshMode(Enum):
    """How a materialized view should be refreshed."""
    FULL = "full"
    INCREMENTAL = "incremental"
    MERGE = "merge"


@dataclass
class MaterializedView:
    """A materialized intermediate result in the pipeline.

    Attributes
    ----------
    view_id : str
        Unique identifier for this view.
    node_id : str
        Pipeline node whose output this materializes.
    table_name : str
        DuckDB table name storing the data.
    query : str
        The SQL query that defines this view.
    state : ViewState
        Current state of the materialization.
    refresh_mode : RefreshMode
        How this view should be refreshed.
    created_at : datetime
        When the view was first created.
    last_refreshed : datetime | None
        When the view was last refreshed.
    refresh_count : int
        Number of times the view has been refreshed.
    row_count : int
        Approximate number of rows.
    storage_bytes : int
        Estimated storage size in bytes.
    schema_hash : str
        Hash of the schema for change detection.
    metadata : dict
        Additional metadata.
    """
    view_id: str = ""
    node_id: str = ""
    table_name: str = ""
    query: str = ""
    state: ViewState = ViewState.VALID
    refresh_mode: RefreshMode = RefreshMode.FULL
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_refreshed: Optional[datetime] = None
    refresh_count: int = 0
    row_count: int = 0
    storage_bytes: int = 0
    schema_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.view_id:
            self.view_id = str(uuid.uuid4())[:8]
        if not self.table_name:
            self.table_name = f"_arc_mat_{self.node_id}_{self.view_id}"

    @property
    def age_seconds(self) -> float:
        """Seconds since last refresh (or creation if never refreshed)."""
        ref_time = self.last_refreshed or self.created_at
        return (datetime.utcnow() - ref_time).total_seconds()

    @property
    def is_valid(self) -> bool:
        return self.state == ViewState.VALID

    @property
    def is_stale(self) -> bool:
        return self.state == ViewState.STALE

    def __repr__(self) -> str:
        return (
            f"MaterializedView({self.node_id}, state={self.state.value}, "
            f"rows={self.row_count}, age={self.age_seconds:.0f}s)"
        )


@dataclass
class RefreshResult:
    """Result of a view refresh operation.

    Attributes
    ----------
    view_id : str
        The view that was refreshed.
    success : bool
        Whether the refresh succeeded.
    mode : RefreshMode
        The refresh mode used.
    rows_affected : int
        Number of rows inserted/updated/deleted.
    execution_time_ms : float
        Time for the refresh in milliseconds.
    error_message : str
        Error message if the refresh failed.
    """
    view_id: str = ""
    success: bool = True
    mode: RefreshMode = RefreshMode.FULL
    rows_affected: int = 0
    execution_time_ms: float = 0.0
    error_message: str = ""


@dataclass
class StorageEstimate:
    """Estimated storage cost for a materialized view.

    Attributes
    ----------
    row_count : int
        Number of rows.
    avg_row_bytes : int
        Average bytes per row.
    total_bytes : int
        Total estimated storage.
    index_bytes : int
        Estimated index storage.
    compression_ratio : float
        Expected compression ratio.
    """
    row_count: int = 0
    avg_row_bytes: int = 100
    total_bytes: int = 0
    index_bytes: int = 0
    compression_ratio: float = 0.5

    @property
    def compressed_bytes(self) -> int:
        return int(self.total_bytes * self.compression_ratio)


@dataclass
class AccessPattern:
    """Access pattern statistics for materialization decisions.

    Attributes
    ----------
    read_count : int
        Number of times the node's output was read.
    write_count : int
        Number of times the node was recomputed.
    avg_compute_time_ms : float
        Average compute time for the node.
    last_accessed : datetime | None
        When the node was last accessed.
    """
    read_count: int = 0
    write_count: int = 0
    avg_compute_time_ms: float = 0.0
    last_accessed: Optional[datetime] = None

    @property
    def read_write_ratio(self) -> float:
        if self.write_count == 0:
            return float("inf") if self.read_count > 0 else 0.0
        return self.read_count / self.write_count


# =====================================================================
# Materialization Manager
# =====================================================================


class MaterializationManager:
    """Manage materialized views for efficient incremental repair.

    Provides methods to materialize pipeline node outputs, refresh them
    incrementally or fully, estimate storage costs, and make materialization
    decisions based on access patterns.

    Parameters
    ----------
    engine : Any
        The execution engine (DuckDB-based) for creating/managing tables.
    prefix : str
        Table name prefix for materialized views.
    max_views : int
        Maximum number of materialized views to maintain.
    auto_cleanup_age : int
        Auto-cleanup views older than this many seconds (0 = disabled).
    """

    def __init__(
        self,
        engine: Any = None,
        prefix: str = "_arc_mat_",
        max_views: int = 100,
        auto_cleanup_age: int = 0,
    ) -> None:
        self._engine = engine
        self._conn: Optional[Any] = None
        self._prefix = prefix
        self._max_views = max_views
        self._auto_cleanup_age = auto_cleanup_age
        self._views: Dict[str, MaterializedView] = OrderedDict()
        self._access_patterns: Dict[str, AccessPattern] = defaultdict(AccessPattern)

        if engine is not None and hasattr(engine, "connection"):
            self._conn = engine.connection
        elif HAS_DUCKDB and engine is None:
            self._conn = duckdb.connect(":memory:")

    # ── Core Operations ───────────────────────────────────────────

    def materialize(
        self,
        engine: Any,
        node_id: str,
        query: str,
        refresh_mode: RefreshMode = RefreshMode.FULL,
    ) -> MaterializedView:
        """Materialize a pipeline node's output.

        Creates a new table containing the query results and registers
        it as a materialized view.

        Parameters
        ----------
        engine : Any
            Execution engine.
        node_id : str
            Pipeline node id.
        query : str
            SQL query defining the view.
        refresh_mode : RefreshMode
            How the view should be refreshed.

        Returns
        -------
        MaterializedView
        """
        conn = self._get_connection(engine)

        view = MaterializedView(
            node_id=node_id,
            query=query,
            refresh_mode=refresh_mode,
        )

        try:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {view.table_name} AS {query}")

            row_count = conn.execute(
                f"SELECT COUNT(*) FROM {view.table_name}"
            ).fetchone()[0]
            view.row_count = row_count

            view.storage_bytes = self._estimate_table_size(conn, view.table_name)
            view.schema_hash = self._compute_schema_hash(conn, view.table_name)
            view.state = ViewState.VALID
            view.last_refreshed = datetime.utcnow()

        except Exception as exc:
            logger.error("Failed to materialize %s: %s", node_id, exc)
            view.state = ViewState.ERROR
            view.metadata["error"] = str(exc)

        self._views[view.view_id] = view

        if len(self._views) > self._max_views:
            self._evict_oldest()

        logger.info(
            "Materialized %s -> %s (%d rows)",
            node_id, view.table_name, view.row_count,
        )
        return view

    def refresh_incremental(
        self,
        view: MaterializedView,
        delta_query: str,
        key_columns: Optional[List[str]] = None,
    ) -> RefreshResult:
        """Incrementally refresh a materialized view.

        Applies only the changed rows instead of full recomputation.

        Parameters
        ----------
        view : MaterializedView
            The view to refresh.
        delta_query : str
            Query that selects changed rows.
        key_columns : list[str], optional
            Key columns for identifying rows to update.

        Returns
        -------
        RefreshResult
        """
        start = time.time()
        conn = self._get_connection()

        if conn is None:
            return RefreshResult(
                view_id=view.view_id,
                success=False,
                error_message="No database connection",
            )

        try:
            view.state = ViewState.REFRESHING

            if key_columns:
                delete_cond = " AND ".join(
                    f"t.{k} = d.{k}" for k in key_columns
                )
                conn.execute(
                    f"DELETE FROM {view.table_name} t "
                    f"USING ({delta_query}) d "
                    f"WHERE {delete_cond}"
                )

            conn.execute(
                f"INSERT INTO {view.table_name} {delta_query}"
            )

            row_count = conn.execute(
                f"SELECT COUNT(*) FROM {view.table_name}"
            ).fetchone()[0]

            rows_affected = abs(row_count - view.row_count)
            view.row_count = row_count
            view.state = ViewState.VALID
            view.last_refreshed = datetime.utcnow()
            view.refresh_count += 1

            elapsed = (time.time() - start) * 1000

            return RefreshResult(
                view_id=view.view_id,
                success=True,
                mode=RefreshMode.INCREMENTAL,
                rows_affected=rows_affected,
                execution_time_ms=elapsed,
            )

        except Exception as exc:
            view.state = ViewState.ERROR
            elapsed = (time.time() - start) * 1000
            logger.error("Incremental refresh failed for %s: %s", view.view_id, exc)
            return RefreshResult(
                view_id=view.view_id,
                success=False,
                mode=RefreshMode.INCREMENTAL,
                execution_time_ms=elapsed,
                error_message=str(exc),
            )

    def refresh_full(self, view: MaterializedView) -> RefreshResult:
        """Fully refresh a materialized view by re-executing the query.

        Parameters
        ----------
        view : MaterializedView
            The view to refresh.

        Returns
        -------
        RefreshResult
        """
        start = time.time()
        conn = self._get_connection()

        if conn is None:
            return RefreshResult(
                view_id=view.view_id,
                success=False,
                error_message="No database connection",
            )

        try:
            view.state = ViewState.REFRESHING

            conn.execute(f"DROP TABLE IF EXISTS {view.table_name}")
            conn.execute(
                f"CREATE TABLE {view.table_name} AS {view.query}"
            )

            row_count = conn.execute(
                f"SELECT COUNT(*) FROM {view.table_name}"
            ).fetchone()[0]

            view.row_count = row_count
            view.storage_bytes = self._estimate_table_size(conn, view.table_name)
            view.state = ViewState.VALID
            view.last_refreshed = datetime.utcnow()
            view.refresh_count += 1

            elapsed = (time.time() - start) * 1000

            return RefreshResult(
                view_id=view.view_id,
                success=True,
                mode=RefreshMode.FULL,
                rows_affected=row_count,
                execution_time_ms=elapsed,
            )

        except Exception as exc:
            view.state = ViewState.ERROR
            elapsed = (time.time() - start) * 1000
            logger.error("Full refresh failed for %s: %s", view.view_id, exc)
            return RefreshResult(
                view_id=view.view_id,
                success=False,
                mode=RefreshMode.FULL,
                execution_time_ms=elapsed,
                error_message=str(exc),
            )

    def invalidate(self, view: MaterializedView) -> None:
        """Mark a materialized view as invalid (needs refresh).

        Parameters
        ----------
        view : MaterializedView
            The view to invalidate.
        """
        view.state = ViewState.STALE
        logger.debug("Invalidated view %s for node %s", view.view_id, view.node_id)

    def invalidate_by_node(self, node_id: str) -> int:
        """Invalidate all views for a specific node.

        Parameters
        ----------
        node_id : str
            The pipeline node.

        Returns
        -------
        int
            Number of views invalidated.
        """
        count = 0
        for view in self._views.values():
            if view.node_id == node_id and view.state == ViewState.VALID:
                view.state = ViewState.STALE
                count += 1
        return count

    def get_materialized_data(
        self,
        view: MaterializedView,
    ) -> Any:
        """Retrieve the data stored in a materialized view.

        Parameters
        ----------
        view : MaterializedView
            The view to read.

        Returns
        -------
        Any
            The data (DuckDB relation or fetchall result).
        """
        conn = self._get_connection()
        if conn is None:
            raise RuntimeError("No database connection")

        if view.state == ViewState.INVALIDATED:
            raise ValueError(f"View {view.view_id} has been invalidated")

        self._record_access(view.node_id)

        try:
            result = conn.execute(f"SELECT * FROM {view.table_name}")
            return result.fetchall()
        except Exception as exc:
            logger.error("Failed to read view %s: %s", view.view_id, exc)
            raise

    def drop_view(self, view: MaterializedView) -> None:
        """Drop a materialized view and its data.

        Parameters
        ----------
        view : MaterializedView
            The view to drop.
        """
        conn = self._get_connection()
        if conn is not None:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {view.table_name}")
            except Exception as exc:
                logger.warning("Failed to drop table %s: %s", view.table_name, exc)

        self._views.pop(view.view_id, None)
        view.state = ViewState.INVALIDATED
        logger.debug("Dropped view %s for node %s", view.view_id, view.node_id)

    # ── Storage Cost Estimation ───────────────────────────────────

    def estimate_storage_cost(
        self,
        view: MaterializedView,
    ) -> StorageEstimate:
        """Estimate the storage cost of a materialized view.

        Parameters
        ----------
        view : MaterializedView
            The view to estimate.

        Returns
        -------
        StorageEstimate
        """
        avg_row_bytes = 100
        if view.row_count > 0 and view.storage_bytes > 0:
            avg_row_bytes = view.storage_bytes // view.row_count

        total = view.row_count * avg_row_bytes
        index_bytes = int(total * 0.1)

        return StorageEstimate(
            row_count=view.row_count,
            avg_row_bytes=avg_row_bytes,
            total_bytes=total,
            index_bytes=index_bytes,
            compression_ratio=0.5,
        )

    def total_storage_bytes(self) -> int:
        """Get total storage used by all materialized views."""
        return sum(v.storage_bytes for v in self._views.values())

    # ── Materialization Decision ──────────────────────────────────

    def should_materialize(
        self,
        node: Any,
        access_pattern: Optional[AccessPattern] = None,
    ) -> bool:
        """Decide whether a node should be materialized.

        A node should be materialized if:
        - It has high fan-out (output consumed by many downstream nodes)
        - It is expensive to recompute
        - It is frequently read relative to writes
        - The data size is manageable

        Parameters
        ----------
        node : Any
            The pipeline node.
        access_pattern : AccessPattern, optional
            Access statistics for this node.

        Returns
        -------
        bool
        """
        node_id = getattr(node, "node_id", "")
        pattern = access_pattern or self._access_patterns.get(node_id, AccessPattern())

        if pattern.read_write_ratio > 2.0:
            return True

        if hasattr(node, "cost_estimate"):
            cost = node.cost_estimate.total_weighted_cost
            if cost > 10.0:
                return True

        row_est = 0
        if hasattr(node, "cost_estimate"):
            row_est = node.cost_estimate.row_estimate

        if row_est > 10_000_000:
            return False

        if pattern.avg_compute_time_ms > 1000:
            return True

        return False

    # ── View Lookup ───────────────────────────────────────────────

    def get_view_for_node(self, node_id: str) -> Optional[MaterializedView]:
        """Get the current materialized view for a node, if any."""
        for view in reversed(list(self._views.values())):
            if view.node_id == node_id and view.state in (ViewState.VALID, ViewState.STALE):
                return view
        return None

    def get_all_views(self) -> List[MaterializedView]:
        """Get all registered materialized views."""
        return list(self._views.values())

    def get_valid_views(self) -> List[MaterializedView]:
        """Get all views in VALID state."""
        return [v for v in self._views.values() if v.state == ViewState.VALID]

    def get_stale_views(self) -> List[MaterializedView]:
        """Get all views in STALE state."""
        return [v for v in self._views.values() if v.state == ViewState.STALE]

    # ── Cleanup ───────────────────────────────────────────────────

    def cleanup_stale_views(self, max_age_seconds: int = 3600) -> int:
        """Clean up stale views older than the specified age.

        Parameters
        ----------
        max_age_seconds : int
            Maximum age in seconds before a stale view is dropped.

        Returns
        -------
        int
            Number of views cleaned up.
        """
        to_remove: List[MaterializedView] = []

        for view in self._views.values():
            if view.state in (ViewState.STALE, ViewState.ERROR):
                if view.age_seconds > max_age_seconds:
                    to_remove.append(view)

        for view in to_remove:
            self.drop_view(view)

        logger.info("Cleaned up %d stale views", len(to_remove))
        return len(to_remove)

    def cleanup_all(self) -> int:
        """Drop all materialized views."""
        count = len(self._views)
        for view in list(self._views.values()):
            self.drop_view(view)
        return count

    # ── Internal Helpers ──────────────────────────────────────────

    def _get_connection(self, engine: Any = None) -> Optional[Any]:
        """Get a database connection."""
        if engine is not None and hasattr(engine, "connection"):
            return engine.connection
        return self._conn

    def _estimate_table_size(self, conn: Any, table_name: str) -> int:
        """Estimate the size of a table in bytes."""
        try:
            result = conn.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()
            row_count = result[0] if result else 0

            col_result = conn.execute(
                f"PRAGMA table_info('{table_name}')"
            ).fetchall()
            col_count = len(col_result) if col_result else 5

            return row_count * col_count * 20

        except Exception:
            return 0

    def _compute_schema_hash(self, conn: Any, table_name: str) -> str:
        """Compute a hash of the table schema."""
        try:
            cols = conn.execute(
                f"PRAGMA table_info('{table_name}')"
            ).fetchall()
            schema_str = "|".join(
                f"{c[1]}:{c[2]}" for c in cols
            )
            return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
        except Exception:
            return ""

    def _record_access(self, node_id: str) -> None:
        """Record an access to a node's materialized data."""
        pattern = self._access_patterns[node_id]
        pattern.read_count += 1
        pattern.last_accessed = datetime.utcnow()

    def _evict_oldest(self) -> None:
        """Evict the oldest materialized view to stay under the limit."""
        if not self._views:
            return

        oldest_id = next(iter(self._views))
        oldest = self._views[oldest_id]
        self.drop_view(oldest)
        logger.debug("Evicted oldest view %s", oldest_id)


# =====================================================================
# Convenience Functions
# =====================================================================


def materialize_node(
    engine: Any,
    node_id: str,
    query: str,
) -> MaterializedView:
    """Convenience: materialize a node's output."""
    manager = MaterializationManager(engine=engine)
    return manager.materialize(engine, node_id, query)


def refresh_view(
    engine: Any,
    view: MaterializedView,
    mode: RefreshMode = RefreshMode.FULL,
    delta_query: Optional[str] = None,
    key_columns: Optional[List[str]] = None,
) -> RefreshResult:
    """Convenience: refresh a materialized view."""
    manager = MaterializationManager(engine=engine)
    if mode == RefreshMode.INCREMENTAL and delta_query:
        return manager.refresh_incremental(view, delta_query, key_columns)
    return manager.refresh_full(view)
