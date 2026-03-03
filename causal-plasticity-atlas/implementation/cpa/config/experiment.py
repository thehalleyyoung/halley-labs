"""Experiment configuration and management.

Provides data-class based experiment configuration and an experiment
manager that handles setup, execution, and artifact persistence.

Classes
-------
ExperimentConfig
    Immutable experiment configuration with validation and serialization.
ExperimentRunner
    Sets up seeds, output directories, runs pipelines with timing.
ResultsTracker
    Lightweight metric and artifact logger.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Immutable configuration for a single experiment run.

    Parameters
    ----------
    name : str
        Unique experiment name.
    description : str
        Human-readable description.
    seed : int
        Master random seed.
    n_contexts : int
        Number of contexts.
    n_nodes : int
        Number of variables.
    n_samples : int
        Samples per context.
    methods : List[str]
        Methods to evaluate.
    metrics : List[str]
        Metrics to compute.
    output_dir : str
        Root output directory.
    """

    name: str = "experiment"
    description: str = ""
    seed: int = 42
    n_contexts: int = 2
    n_nodes: int = 5
    n_samples: int = 500
    methods: List[str] = field(default_factory=lambda: ["pc"])
    metrics: List[str] = field(default_factory=lambda: ["shd"])
    output_dir: str = "./results"
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- Factory methods --

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Create an ExperimentConfig from a dictionary.

        Unknown keys are stored in the ``extra`` field.
        """
        known = {
            "name", "description", "seed", "n_contexts", "n_nodes",
            "n_samples", "methods", "metrics", "output_dir", "extra",
        }
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in config_dict.items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        if extra:
            kwargs.setdefault("extra", {})
            kwargs["extra"].update(extra)
        return cls(**kwargs)

    @classmethod
    def from_yaml_string(cls, yaml_str: str) -> ExperimentConfig:
        """Parse a simple YAML-like config string.

        Supports ``key: value`` lines (one level) where values are
        auto-cast to int / float / list / str.
        """
        config: Dict[str, Any] = {}
        for line in yaml_str.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            config[key] = _auto_cast(val)
        return cls.from_dict(config)

    # -- Serialization --

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        d = asdict(self)
        # numpy types → python builtins
        return _convert_numpy(d)

    # -- Validation --

    def validate(self) -> List[str]:
        """Validate configuration; return list of error messages (empty if OK)."""
        errors: List[str] = []
        if not self.name or not self.name.strip():
            errors.append("name must be non-empty")
        if self.seed < 0:
            errors.append("seed must be non-negative")
        if self.n_contexts < 1:
            errors.append("n_contexts must be >= 1")
        if self.n_nodes < 1:
            errors.append("n_nodes must be >= 1")
        if self.n_samples < 1:
            errors.append("n_samples must be >= 1")
        if not self.methods:
            errors.append("methods must be non-empty")
        if not self.metrics:
            errors.append("metrics must be non-empty")
        return errors

    # -- Derived configs --

    def with_overrides(self, **kwargs: Any) -> ExperimentConfig:
        """Return a copy with selected fields overridden."""
        d = self.to_dict()
        d.update(kwargs)
        return ExperimentConfig.from_dict(d)

    def hash(self) -> str:
        """Deterministic SHA-256 hash of the configuration."""
        d = self.to_dict()
        raw = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Run an experiment pipeline with seed management, timing, and I/O.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    output_dir : str or None
        Override output directory (default: config.output_dir).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir or config.output_dir
        self.tracker = ResultsTracker()
        self._run_dir: Optional[str] = None
        self._timings: Dict[str, float] = {}

    def setup(self) -> str:
        """Initialize experiment: set seeds, create output directory.

        Returns the path to the run-specific output directory.
        """
        self._set_seeds(self.config.seed)
        self._run_dir = self._create_output_dir(self.config.name)
        self.save_config("config.json")
        return self._run_dir

    def run(self, pipeline_fn: Callable[[ExperimentConfig, ResultsTracker], Any]) -> Any:
        """Run *pipeline_fn* with tracking and timing.

        Parameters
        ----------
        pipeline_fn : callable(config, tracker) -> results

        Returns
        -------
        Whatever *pipeline_fn* returns.
        """
        if self._run_dir is None:
            self.setup()

        self.tracker.log_metric("start_time", time.time())
        t0 = time.perf_counter()
        try:
            result = pipeline_fn(self.config, self.tracker)
            self.tracker.log_metric("status", 1.0)
        except Exception as exc:
            self.tracker.log_metric("status", 0.0)
            self.tracker.log_artifact("error", str(exc))
            raise
        finally:
            elapsed = time.perf_counter() - t0
            self.tracker.log_metric("elapsed_seconds", elapsed)
            self.tracker.log_metric("end_time", time.time())
        return result

    # -- Seed management --

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Set numpy and stdlib random seeds for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)

    # -- Output directory --

    def _create_output_dir(self, name: str) -> str:
        """Create a timestamped output directory and return its path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"{name}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    # -- Persistence --

    def save_results(self, results: Any, filename: str = "results.json") -> str:
        """Serialize *results* to JSON in the run directory.

        Returns the path to the saved file.
        """
        if self._run_dir is None:
            self.setup()
        path = os.path.join(self._run_dir, filename)  # type: ignore[arg-type]
        data = _convert_numpy(results) if isinstance(results, dict) else results
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        return path

    def save_config(self, filename: str = "config.json") -> str:
        """Save the experiment config for reproducibility."""
        if self._run_dir is None:
            self.setup()
        path = os.path.join(self._run_dir, filename)  # type: ignore[arg-type]
        with open(path, "w") as fh:
            json.dump(self.config.to_dict(), fh, indent=2, default=str)
        return path

    # -- Timing context manager --

    @contextmanager
    def _timing_context(self, name: str) -> Generator[None, None, None]:
        """Context manager that records wall-clock time under *name*."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._timings[name] = elapsed
            self.tracker.log_metric(f"time_{name}", elapsed)


# ---------------------------------------------------------------------------
# Results Tracker
# ---------------------------------------------------------------------------

class ResultsTracker:
    """Lightweight metric and artifact logger.

    Collects metrics (numeric values, optionally with step) and
    artifacts (arbitrary objects keyed by name).
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._artifacts: Dict[str, Any] = {}

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a numeric metric, optionally with a step index."""
        entry: Dict[str, Any] = {"value": float(value), "timestamp": time.time()}
        if step is not None:
            entry["step"] = step
        self._metrics.setdefault(name, []).append(entry)

    def log_artifact(self, name: str, data: Any) -> None:
        """Log an artifact (dict, array, string, etc.)."""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        self._artifacts[name] = data

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics for all logged metrics."""
        out: Dict[str, Any] = {}
        for name, entries in self._metrics.items():
            vals = [e["value"] for e in entries]
            out[name] = {
                "last": vals[-1],
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
            }
        return out

    def get_metric(self, name: str) -> List[Dict[str, Any]]:
        """Return all entries for a metric."""
        return list(self._metrics.get(name, []))

    def get_artifact(self, name: str) -> Any:
        """Return an artifact by name."""
        return self._artifacts.get(name)

    def to_dataframe(self) -> Any:
        """Convert metrics to a pandas DataFrame (name, value, step, timestamp).

        Returns a list-of-dicts fallback if pandas is unavailable.
        """
        rows: List[Dict[str, Any]] = []
        for name, entries in self._metrics.items():
            for entry in entries:
                rows.append({"name": name, **entry})
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows


# Kept for backward compatibility with __init__.py imports.
ExperimentManager = ExperimentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_cast(val: str) -> Any:
    """Cast a YAML-like value string to int, float, bool, list, or str."""
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    if val.startswith("[") and val.endswith("]"):
        items = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        return [_auto_cast(i) for i in items]
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val.strip("'\"")


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to Python builtins."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
