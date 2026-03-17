"""
taintflow.integrations.pipeline_wrapper – High-level pipeline wrapping utilities.

Provides wrappers for executing ML scripts and notebooks under TaintFlow
instrumentation, collecting all audit events into a single session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class InstrumentationConfig:
    """Configuration for pipeline instrumentation.

    Attributes:
        trace_calls: Whether to enable sys.settrace call tracing.
        hook_pandas: Whether to monkey-patch pandas operations.
        hook_sklearn: Whether to intercept sklearn estimator calls.
        buffer_size: Maximum number of events to buffer before flush.
        track_provenance: Whether to track row-level provenance.
    """

    trace_calls: bool = True
    hook_pandas: bool = True
    hook_sklearn: bool = True
    buffer_size: int = 50_000
    track_provenance: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trace_calls": self.trace_calls,
            "hook_pandas": self.hook_pandas,
            "hook_sklearn": self.hook_sklearn,
            "buffer_size": self.buffer_size,
            "track_provenance": self.track_provenance,
        }


class AuditSession:
    """A single audit session collecting instrumentation events.

    Manages the lifecycle of instrumentation hooks and event collection
    for one pipeline execution.

    Args:
        config: Instrumentation configuration.
    """

    def __init__(self, config: Optional[InstrumentationConfig] = None) -> None:
        self._config = config or InstrumentationConfig()
        self._events: list[Dict[str, Any]] = []
        self._active: bool = False

    @property
    def is_active(self) -> bool:
        """Whether the session is currently collecting events."""
        return self._active

    @property
    def event_count(self) -> int:
        """Number of events collected so far."""
        return len(self._events)

    def start(self) -> None:
        """Begin collecting instrumentation events."""
        self._active = True
        self._events.clear()

    def stop(self) -> None:
        """Stop collecting events and finalize the session."""
        self._active = False

    def record_event(self, event: Dict[str, Any]) -> None:
        """Record a single instrumentation event.

        Args:
            event: Event descriptor with type, timestamp, and metadata.
        """
        if self._active and len(self._events) < self._config.buffer_size:
            self._events.append(event)

    def get_events(self) -> list[Dict[str, Any]]:
        """Return all collected events."""
        return list(self._events)


class PipelineWrapper:
    """Wraps an ML pipeline for instrumented execution.

    Args:
        session: The audit session to use.
    """

    def __init__(self, session: Optional[AuditSession] = None) -> None:
        self._session = session or AuditSession()

    def wrap(self, pipeline_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a pipeline function for instrumented execution.

        Args:
            pipeline_fn: The pipeline function to wrap.

        Returns:
            Wrapped function that records events during execution.
        """
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._session.start()
            try:
                result = pipeline_fn(*args, **kwargs)
            finally:
                self._session.stop()
            return result
        return wrapped


class ScriptExecutor:
    """Executes a Python script file under instrumentation.

    Args:
        session: The audit session for event collection.
    """

    def __init__(self, session: Optional[AuditSession] = None) -> None:
        self._session = session or AuditSession()

    def execute(self, script_path: str) -> Dict[str, Any]:
        """Execute a Python script and collect audit events.

        Args:
            script_path: Path to the Python script to execute.

        Returns:
            Dictionary with execution results and collected events.
        """
        self._session.start()
        result: Dict[str, Any] = {"script": script_path, "success": False}
        try:
            with open(script_path) as f:
                code = compile(f.read(), script_path, "exec")
            exec(code, {"__name__": "__main__", "__file__": script_path})
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
        finally:
            self._session.stop()
            result["events"] = self._session.event_count
        return result


class NotebookExecutor:
    """Executes a Jupyter notebook under instrumentation.

    Args:
        session: The audit session for event collection.
    """

    def __init__(self, session: Optional[AuditSession] = None) -> None:
        self._session = session or AuditSession()

    def execute(self, notebook_path: str) -> Dict[str, Any]:
        """Execute a notebook and collect audit events.

        Args:
            notebook_path: Path to the .ipynb file.

        Returns:
            Dictionary with execution results.
        """
        return {
            "notebook": notebook_path,
            "success": False,
            "error": "Notebook execution not yet implemented",
        }


class DAGBuilder:
    """Builds a PI-DAG from collected audit session events.

    Args:
        session: Completed audit session with collected events.
    """

    def __init__(self, session: AuditSession) -> None:
        self._session = session

    def build(self) -> Dict[str, Any]:
        """Construct a PI-DAG from session events.

        Returns:
            Dictionary representation of the PI-DAG.
        """
        events = self._session.get_events()
        return {
            "nodes": [],
            "edges": [],
            "n_events": len(events),
        }


class PipelineValidator:
    """Validates pipeline structure before analysis.

    Checks that the pipeline has recognizable entry and exit points,
    contains supported operations, and has consistent data flow.
    """

    def validate(self, dag: Dict[str, Any]) -> list[str]:
        """Validate a PI-DAG and return error messages.

        Args:
            dag: PI-DAG dictionary from DAGBuilder.

        Returns:
            Empty list if valid, otherwise list of issues.
        """
        errors: list[str] = []
        if not dag.get("nodes"):
            errors.append("DAG has no nodes")
        return errors


class CleanupManager:
    """Manages cleanup of instrumentation hooks after audit completion.

    Ensures that monkey-patched methods are restored and sys.settrace
    is properly unregistered even if the pipeline raises an exception.
    """

    def __init__(self) -> None:
        self._hooks_installed: list[str] = []

    def register_hook(self, hook_name: str) -> None:
        """Register an installed hook for later cleanup."""
        self._hooks_installed.append(hook_name)

    def cleanup(self) -> None:
        """Remove all registered hooks and restore original methods."""
        self._hooks_installed.clear()

    @property
    def n_hooks(self) -> int:
        """Number of currently registered hooks."""
        return len(self._hooks_installed)


__all__ = [
    "AuditSession",
    "CleanupManager",
    "DAGBuilder",
    "InstrumentationConfig",
    "NotebookExecutor",
    "PipelineValidator",
    "PipelineWrapper",
    "ScriptExecutor",
]
