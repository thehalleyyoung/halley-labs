"""Experiment configuration, execution, and analysis for the MARACE system.

Provides infrastructure for running reproducible multi-agent race condition
verification experiments with result persistence, statistical comparison,
parameter sweeps, and progress tracking.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import resource as _resource

    def _get_peak_memory_mb() -> float:
        """Return peak RSS in MB via the resource module."""
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes; on Linux it is in KB.
        import sys

        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
except ImportError:
    def _get_peak_memory_mb() -> float:  # type: ignore[misc]
        """Fallback when resource module is unavailable."""
        return 0.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All parameters that fully specify a single experiment."""

    experiment_id: str
    name: str
    description: str = ""
    env_config: dict = field(default_factory=dict)
    policy_configs: list[dict] = field(default_factory=list)
    specification: str = ""
    num_runs: int = 1
    seed: int = 42
    timeout_s: float = 3600.0
    pipeline_config: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    output_dir: str = "results"

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        """Construct an ``ExperimentConfig`` from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentResult:
    """Outcome of a single experiment run."""

    experiment_id: str
    run_id: int
    config: ExperimentConfig
    races_found: list[dict]
    time_s: float
    memory_peak_mb: float
    states_explored: int
    coverage: float
    converged: bool
    timestamp: datetime
    error: str | None = None

    def to_dict(self) -> dict:
        """Serialize the result to a JSON-safe dictionary."""
        d: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "races_found": self.races_found,
            "time_s": self.time_s,
            "memory_peak_mb": self.memory_peak_mb,
            "states_explored": self.states_explored,
            "coverage": self.coverage,
            "converged": self.converged,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentResult:
        """Reconstruct an ``ExperimentResult`` from a dictionary."""
        config = ExperimentConfig.from_dict(data["config"])
        return cls(
            experiment_id=data["experiment_id"],
            run_id=data["run_id"],
            config=config,
            races_found=data["races_found"],
            time_s=data["time_s"],
            memory_peak_mb=data["memory_peak_mb"],
            states_explored=data["states_explored"],
            coverage=data["coverage"],
            converged=data["converged"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """Wraps an ``ExperimentConfig`` and drives repeated execution."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._results: list[ExperimentResult] = []

    def run(self, run_fn: Callable) -> list[ExperimentResult]:
        """Execute *run_fn* ``num_runs`` times with per-run seeds.

        Args:
            run_fn: A callable ``(config: ExperimentConfig, run_id: int) ->
                ExperimentResult``.  The experiment config is augmented with a
                per-run seed before each invocation.

        Returns:
            A list of ``ExperimentResult`` instances (one per run).
        """
        self._results = []
        for run_id in range(self.config.num_runs):
            result = self._single_run(run_id, run_fn)
            self._results.append(result)
        return self._results

    def _single_run(self, run_id: int, run_fn: Callable) -> ExperimentResult:
        """Execute a single run, injecting the per-run seed.

        The seed for each run is ``base_seed + run_id`` so that repeated
        executions of the same experiment remain deterministic while still
        varying across runs.
        """
        cfg = copy.deepcopy(self.config)
        cfg.seed = self.config.seed + run_id
        return run_fn(cfg, run_id)

    def summary(self) -> str:
        """Return a human-readable summary of accumulated results."""
        if not self._results:
            return f"Experiment '{self.config.name}': no results yet."

        times = [r.time_s for r in self._results]
        coverages = [r.coverage for r in self._results]
        races = [len(r.races_found) for r in self._results]
        errors = sum(1 for r in self._results if r.error is not None)

        lines = [
            f"Experiment: {self.config.name} ({self.config.experiment_id})",
            f"  Runs       : {len(self._results)}",
            f"  Errors     : {errors}",
            f"  Time (avg) : {sum(times) / len(times):.3f}s",
            f"  Coverage   : {sum(coverages) / len(coverages):.4f}",
            f"  Races (avg): {sum(races) / len(races):.1f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Convenience runner that handles timing, memory tracking, and persistence."""

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self, experiment: Experiment, run_fn: Callable
    ) -> list[ExperimentResult]:
        """Run *experiment* using *run_fn*, recording timing and memory.

        *run_fn* has the signature ``(config, run_id) -> ExperimentResult``.
        This wrapper adds wall-clock timing and peak-memory measurements, then
        persists results to disk.
        """
        results: list[ExperimentResult] = []
        for run_id in range(experiment.config.num_runs):
            mem_before = self._track_memory()
            t0 = time.monotonic()

            try:
                result = run_fn(experiment.config, run_id)
            except Exception as exc:  # noqa: BLE001
                result = ExperimentResult(
                    experiment_id=experiment.config.experiment_id,
                    run_id=run_id,
                    config=experiment.config,
                    races_found=[],
                    time_s=time.monotonic() - t0,
                    memory_peak_mb=self._track_memory(),
                    states_explored=0,
                    coverage=0.0,
                    converged=False,
                    timestamp=datetime.now(timezone.utc),
                    error=str(exc),
                )

            elapsed = time.monotonic() - t0
            mem_after = self._track_memory()

            result.time_s = elapsed
            result.memory_peak_mb = max(mem_after, mem_before)
            results.append(result)

        self._save_results(results, experiment.config)
        return results

    @staticmethod
    def _track_memory() -> float:
        """Return current peak RSS in MB."""
        return _get_peak_memory_mb()

    def _save_results(
        self, results: list[ExperimentResult], config: ExperimentConfig
    ) -> None:
        """Persist *results* to a JSON file under ``output_dir``."""
        out_path = self.output_dir / f"{config.experiment_id}.json"
        payload = {
            "config": config.to_dict(),
            "results": [r.to_dict() for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# ResultDatabase
# ---------------------------------------------------------------------------

class ResultDatabase:
    """SQLite-backed store for ``ExperimentResult`` records."""

    def __init__(self, db_path: str = "results.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create the schema if it does not already exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                run_id      INTEGER NOT NULL,
                config_json TEXT    NOT NULL,
                races_json  TEXT    NOT NULL,
                time_s      REAL    NOT NULL,
                memory_peak_mb REAL NOT NULL,
                states_explored INTEGER NOT NULL,
                coverage    REAL    NOT NULL,
                converged   INTEGER NOT NULL,
                timestamp   TEXT    NOT NULL,
                error       TEXT,
                tags_json   TEXT    NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_experiment_id
                ON results (experiment_id);
            """
        )
        self._conn.commit()

    def store(self, result: ExperimentResult) -> None:
        """Insert a single ``ExperimentResult`` into the database."""
        self._conn.execute(
            """
            INSERT INTO results
                (experiment_id, run_id, config_json, races_json,
                 time_s, memory_peak_mb, states_explored, coverage,
                 converged, timestamp, error, tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.experiment_id,
                result.run_id,
                json.dumps(result.config.to_dict()),
                json.dumps(result.races_found),
                result.time_s,
                result.memory_peak_mb,
                result.states_explored,
                result.coverage,
                int(result.converged),
                result.timestamp.isoformat(),
                result.error,
                json.dumps(result.config.tags),
            ),
        )
        self._conn.commit()

    def _row_to_result(self, row: sqlite3.Row) -> ExperimentResult:
        """Convert a database row to an ``ExperimentResult``."""
        config = ExperimentConfig.from_dict(json.loads(row["config_json"]))
        return ExperimentResult(
            experiment_id=row["experiment_id"],
            run_id=row["run_id"],
            config=config,
            races_found=json.loads(row["races_json"]),
            time_s=row["time_s"],
            memory_peak_mb=row["memory_peak_mb"],
            states_explored=row["states_explored"],
            coverage=row["coverage"],
            converged=bool(row["converged"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            error=row["error"],
        )

    def load(self, experiment_id: str) -> list[ExperimentResult]:
        """Load all results for *experiment_id*."""
        cur = self._conn.execute(
            "SELECT * FROM results WHERE experiment_id = ? ORDER BY run_id",
            (experiment_id,),
        )
        return [self._row_to_result(r) for r in cur.fetchall()]

    def load_all(self) -> list[ExperimentResult]:
        """Load every result in the database."""
        cur = self._conn.execute("SELECT * FROM results ORDER BY timestamp")
        return [self._row_to_result(r) for r in cur.fetchall()]

    def query(
        self,
        tags: list[str] | None = None,
        min_coverage: float | None = None,
    ) -> list[ExperimentResult]:
        """Flexible query by tags and/or minimum coverage.

        Tags are matched with an AND semantic: all specified tags must be
        present in the result's tag list.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if min_coverage is not None:
            clauses.append("coverage >= ?")
            params.append(min_coverage)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self._conn.execute(
            f"SELECT * FROM results {where} ORDER BY timestamp", params  # noqa: S608
        )
        rows = cur.fetchall()
        results = [self._row_to_result(r) for r in rows]

        if tags:
            tag_set = set(tags)
            results = [
                r for r in results if tag_set.issubset(set(r.config.tags))
            ]

        return results

    def delete(self, experiment_id: str) -> None:
        """Remove all results for *experiment_id*."""
        self._conn.execute(
            "DELETE FROM results WHERE experiment_id = ?", (experiment_id,)
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# ExperimentComparator
# ---------------------------------------------------------------------------

class ExperimentComparator:
    """Statistical comparison across multiple experiments."""

    def __init__(self) -> None:
        self._data: dict[str, list[ExperimentResult]] = {}

    def add(self, experiment_id: str, results: list[ExperimentResult]) -> None:
        """Register *results* under *experiment_id* for later comparison."""
        self._data[experiment_id] = list(results)

    def compare(self) -> dict:
        """Compute per-experiment aggregate statistics.

        Returns a dictionary keyed by experiment id, each containing mean
        time, mean coverage, and mean number of races found.
        """
        summary: dict[str, dict[str, float]] = {}
        for eid, results in self._data.items():
            if not results:
                continue
            n = len(results)
            summary[eid] = {
                "mean_time_s": sum(r.time_s for r in results) / n,
                "mean_coverage": sum(r.coverage for r in results) / n,
                "mean_races_found": sum(len(r.races_found) for r in results) / n,
                "num_runs": n,
            }
        return summary

    def significance_test(
        self, exp_a: str, exp_b: str, metric: str
    ) -> dict:
        """Welch's t-test between two experiments on a given *metric*.

        *metric* must be one of ``"time_s"``, ``"coverage"``, or
        ``"races_found"``.  Returns a dictionary with the t-statistic and
        approximate two-sided p-value.
        """
        def _extract(results: list[ExperimentResult]) -> list[float]:
            if metric == "time_s":
                return [r.time_s for r in results]
            if metric == "coverage":
                return [r.coverage for r in results]
            if metric == "races_found":
                return [float(len(r.races_found)) for r in results]
            raise ValueError(f"Unknown metric: {metric}")

        a_vals = _extract(self._data[exp_a])
        b_vals = _extract(self._data[exp_b])

        t_stat, p_value = _welch_t_test(a_vals, b_vals)
        return {
            "metric": metric,
            "exp_a": exp_a,
            "exp_b": exp_b,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_005": p_value < 0.05 if p_value is not None else None,
        }

    def summary_table(self) -> str:
        """Return a formatted text table comparing all experiments."""
        comp = self.compare()
        if not comp:
            return "No experiments to compare."

        header = f"{'Experiment':<30} {'Runs':>5} {'Time(s)':>10} {'Coverage':>10} {'Races':>8}"
        sep = "-" * len(header)
        lines = [header, sep]
        for eid, stats in comp.items():
            lines.append(
                f"{eid:<30} {int(stats['num_runs']):>5} "
                f"{stats['mean_time_s']:>10.3f} "
                f"{stats['mean_coverage']:>10.4f} "
                f"{stats['mean_races_found']:>8.1f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Welch's t-test (no scipy dependency)
# ---------------------------------------------------------------------------

def _welch_t_test(
    a: list[float], b: list[float]
) -> tuple[float | None, float | None]:
    """Compute Welch's t-test statistic and approximate p-value.

    Returns ``(None, None)`` when either sample has fewer than two elements or
    has zero variance.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None

    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)

    se = var_a / na + var_b / nb
    if se <= 0:
        return None, None

    t_stat = (mean_a - mean_b) / math.sqrt(se)

    # Welch–Satterthwaite degrees of freedom
    df_num = se ** 2
    df_den = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    if df_den <= 0:
        return t_stat, None
    df = df_num / df_den

    # Approximate two-tailed p-value via the regularised incomplete beta
    # function (good enough for large df; falls back gracefully).
    p_value = _approx_t_pvalue(abs(t_stat), df)
    return t_stat, p_value


def _approx_t_pvalue(t: float, df: float) -> float:
    """Rough two-tailed p-value for the Student-t distribution.

    Uses the relationship ``p = I_{df/(df+t^2)}(df/2, 1/2)`` where *I* is the
    regularised incomplete beta function, approximated here with a continued-
    fraction expansion.
    """
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    # Regularised incomplete beta via a simple series expansion
    p = _regularized_incomplete_beta(x, a, b)
    return max(0.0, min(1.0, p))


def _regularized_incomplete_beta(
    x: float, a: float, b: float, max_iter: int = 200, tol: float = 1e-12
) -> float:
    """Evaluate the regularised incomplete beta function I_x(a, b).

    Uses Lentz's continued-fraction algorithm.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Front factor: x^a (1-x)^b / (a * B(a,b))
    log_front = (
        a * math.log(x)
        + b * math.log(1 - x)
        - math.log(a)
        - _log_beta(a, b)
    )
    front = math.exp(log_front)

    # Continued fraction (modified Lentz)
    tiny = 1e-30
    f = tiny
    c = tiny
    d = 0.0

    for m in range(max_iter):
        if m == 0:
            alpha_m = 1.0
        else:
            k = m
            if k % 2 == 0:
                j = k // 2
                alpha_m = (j * (b - j) * x) / ((a + 2 * j - 1) * (a + 2 * j))
            else:
                j = (k - 1) // 2 + 1
                alpha_m = -((a + j - 1 + j) * (a + j) * x) / (  # noqa: E501
                    (a + 2 * j - 1) * (a + 2 * j)
                )
                # Correction for the odd terms
                alpha_m = -(
                    (a + j - 1) * (a + b + j - 1) * x
                ) / ((a + 2 * j - 2) * (a + 2 * j - 1))

        d = 1.0 + alpha_m * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = 1.0 + alpha_m / c
        if abs(c) < tiny:
            c = tiny

        delta = c * d
        f *= delta
        if abs(delta - 1.0) < tol:
            break

    return front * f


def _log_beta(a: float, b: float) -> float:
    """Logarithm of the Beta function via log-gamma."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ---------------------------------------------------------------------------
# ReproducibilityChecker
# ---------------------------------------------------------------------------

class ReproducibilityChecker:
    """Utilities for checking whether experiments are reproducible."""

    def check(
        self,
        results_a: list[ExperimentResult],
        results_b: list[ExperimentResult],
    ) -> dict:
        """Compare two sets of results for reproducibility.

        Returns a dictionary describing:
        * ``race_sets_match`` – whether the same race conditions were found,
        * ``timing_within_tolerance`` – whether wall-clock times are within 20%,
        * ``coverage_match`` – whether coverage values are identical.
        """
        races_a = [
            sorted(json.dumps(r, sort_keys=True) for r in res.races_found)
            for res in results_a
        ]
        races_b = [
            sorted(json.dumps(r, sort_keys=True) for r in res.races_found)
            for res in results_b
        ]
        race_sets_match = races_a == races_b

        timing_ok = True
        for ra, rb in zip(results_a, results_b):
            mean_t = (ra.time_s + rb.time_s) / 2.0
            if mean_t > 0 and abs(ra.time_s - rb.time_s) / mean_t > 0.20:
                timing_ok = False
                break

        coverage_match = all(
            ra.coverage == rb.coverage
            for ra, rb in zip(results_a, results_b)
        )

        return {
            "race_sets_match": race_sets_match,
            "timing_within_tolerance": timing_ok,
            "coverage_match": coverage_match,
            "reproducible": race_sets_match and coverage_match,
        }

    def is_deterministic(self, results: list[ExperimentResult]) -> bool:
        """Return ``True`` if all runs discovered the same set of races.

        Determinism is judged solely on the *content* of the race conditions
        found (ignoring wall-clock timing).
        """
        if len(results) <= 1:
            return True

        canonical = sorted(
            json.dumps(r, sort_keys=True) for r in results[0].races_found
        )
        return all(
            sorted(json.dumps(r, sort_keys=True) for r in res.races_found)
            == canonical
            for res in results[1:]
        )


# ---------------------------------------------------------------------------
# ExperimentSweep
# ---------------------------------------------------------------------------

class ExperimentSweep:
    """Generate experiment configs via parameter sweeps."""

    def __init__(self, base_config: ExperimentConfig) -> None:
        self.base_config = base_config

    def sweep(self, param_name: str, values: list) -> list[ExperimentConfig]:
        """Create one config per value, varying *param_name*."""
        configs: list[ExperimentConfig] = []
        for i, val in enumerate(values):
            cfg = copy.deepcopy(self.base_config)
            if hasattr(cfg, param_name):
                setattr(cfg, param_name, val)
            else:
                cfg.pipeline_config[param_name] = val
            cfg.experiment_id = f"{self.base_config.experiment_id}_{param_name}_{i}"
            configs.append(cfg)
        return configs

    def grid_sweep(self, params: dict[str, list]) -> list[ExperimentConfig]:
        """Cartesian-product sweep over multiple parameters."""
        keys = list(params.keys())
        configs: list[ExperimentConfig] = []
        for i, combo in enumerate(itertools.product(*params.values())):
            cfg = copy.deepcopy(self.base_config)
            for key, val in zip(keys, combo):
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
                else:
                    cfg.pipeline_config[key] = val
            tag = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
            cfg.experiment_id = f"{self.base_config.experiment_id}_grid_{i}_{tag}"
            configs.append(cfg)
        return configs

    def run_sweep(
        self,
        configs: list[ExperimentConfig],
        run_fn: Callable,
    ) -> list[list[ExperimentResult]]:
        """Execute experiments for every config in *configs*.

        Returns a list of result lists, one per config.
        """
        all_results: list[list[ExperimentResult]] = []
        for cfg in configs:
            exp = Experiment(cfg)
            results = exp.run(run_fn)
            all_results.append(results)
        return all_results


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Simple progress tracker with ETA estimation."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self._start = time.monotonic()

    def update(self, completed: int = 1) -> None:
        """Mark *completed* additional items as done."""
        self.completed += completed

    def eta(self) -> float:
        """Estimate seconds remaining based on average throughput."""
        if self.completed <= 0:
            return float("inf")
        elapsed = self.elapsed()
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.completed
        return remaining / rate if rate > 0 else float("inf")

    def elapsed(self) -> float:
        """Seconds since the tracker was created."""
        return time.monotonic() - self._start

    def progress_bar(self, width: int = 40) -> str:
        """Render a text-based progress bar."""
        frac = self.completed / self.total if self.total > 0 else 0.0
        filled = int(width * frac)
        bar = "█" * filled + "░" * (width - filled)
        pct = frac * 100
        return f"|{bar}| {pct:5.1f}% ({self.completed}/{self.total})"

    def __str__(self) -> str:
        eta_s = self.eta()
        eta_str = f"{eta_s:.1f}s" if math.isfinite(eta_s) else "∞"
        return (
            f"{self.progress_bar()}  "
            f"elapsed={self.elapsed():.1f}s  ETA={eta_str}"
        )
