"""Benchmarks for evaluating MARACE race-condition detection.

Each benchmark plants known race conditions into a multi-agent environment
and measures how well a detector recovers them (recall, precision, F1).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration & planted-race descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration for a single benchmark run."""

    num_agents: int
    state_dim: int
    action_dim: int
    horizon: int = 50
    race_probability: float = 0.01
    seed: int = 42
    timeout_s: float = 300.0


@dataclass
class PlantedRace:
    """Description of a race condition intentionally embedded in a benchmark.

    Attributes:
        race_type: One of ``"collision"``, ``"deadlock"``,
            ``"merge_conflict"``, or ``"correlated_liquidation"``.
        agents_involved: Indices of the agents that participate in this race.
        trigger_condition: Human-readable description of when the race fires.
        probability: Probability that the race manifests per time-step when
            agents are in the trigger region.
        state_region: Axis-aligned bounding box that activates the race.
            Maps dimension index (as string key) to ``{"lo": float, "hi":
            float}`` bounds.
    """

    VALID_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"collision", "deadlock", "merge_conflict", "correlated_liquidation"}
    )

    race_type: str
    agents_involved: List[int]
    trigger_condition: str
    probability: float
    state_region: Dict[str, Dict[str, float]]

    def __post_init__(self) -> None:
        if self.race_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown race_type {self.race_type!r}; "
                f"expected one of {sorted(self.VALID_TYPES)}"
            )
        if not self.agents_involved:
            raise ValueError("agents_involved must be non-empty")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")


# ---------------------------------------------------------------------------
# Abstract benchmark
# ---------------------------------------------------------------------------


class Benchmark(ABC):
    """Abstract base for all MARACE benchmarks.

    Sub-classes plant one or more :class:`PlantedRace` instances inside
    :meth:`setup` and implement domain-specific evaluation logic in
    :meth:`evaluate_results`.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._planted_races: List[PlantedRace] = []
        self._environment_config: Dict[str, Any] = {}
        self._policy_configs: List[Dict[str, Any]] = []
        self._is_setup = False

    # -- properties ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the benchmark."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description."""

    # -- lifecycle -----------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """Create environment, policies, and plant race conditions."""

    def get_planted_races(self) -> List[PlantedRace]:
        """Return the list of planted races (available after :meth:`setup`)."""
        return list(self._planted_races)

    def get_environment_config(self) -> Dict[str, Any]:
        """Return the environment configuration dictionary."""
        return dict(self._environment_config)

    def get_policy_configs(self) -> List[Dict[str, Any]]:
        """Return per-agent policy configurations."""
        return [dict(pc) for pc in self._policy_configs]

    def evaluate_results(self, detected_races: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score *detected_races* against the planted ground truth.

        Returns a dictionary with at least ``"recall"``, ``"precision"``,
        ``"f1"``, ``"true_positives"``, ``"false_positives"``, and
        ``"false_negatives"`` keys.
        """
        planted = self._planted_races
        matched_planted: set[int] = set()
        matched_detected: set[int] = set()

        for d_idx, det in enumerate(detected_races):
            for p_idx, pla in enumerate(planted):
                if p_idx in matched_planted:
                    continue
                if self._race_matches(det, pla):
                    matched_planted.add(p_idx)
                    matched_detected.add(d_idx)
                    break

        tp = len(matched_planted)
        fp = len(detected_races) - len(matched_detected)
        fn = len(planted) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "benchmark": self.name,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_planted": len(planted),
            "num_detected": len(detected_races),
        }

    def teardown(self) -> None:
        """Release resources acquired during :meth:`setup`."""
        self._planted_races.clear()
        self._environment_config.clear()
        self._policy_configs.clear()
        self._is_setup = False

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _race_matches(detected: Dict[str, Any], planted: PlantedRace) -> bool:
        """Return ``True`` when *detected* plausibly corresponds to *planted*.

        Matching criteria (all must hold):
        1. ``race_type`` values agree (if provided in *detected*).
        2. The set of involved agents overlaps by at least 50 %.
        3. If *detected* carries a ``state_region``, the bounding boxes
           overlap in every shared dimension.
        """
        det_type = detected.get("race_type")
        if det_type is not None and det_type != planted.race_type:
            return False

        det_agents = set(detected.get("agents_involved", []))
        planted_agents = set(planted.agents_involved)
        if not det_agents:
            return False
        overlap = len(det_agents & planted_agents)
        if overlap < 0.5 * len(planted_agents):
            return False

        det_region = detected.get("state_region")
        if det_region is not None:
            for dim, bounds in planted.state_region.items():
                if dim not in det_region:
                    continue
                d_lo = det_region[dim].get("lo", float("-inf"))
                d_hi = det_region[dim].get("hi", float("inf"))
                p_lo = bounds["lo"]
                p_hi = bounds["hi"]
                if d_hi < p_lo or d_lo > p_hi:
                    return False

        return True

    def _make_grid_state_space(
        self,
        low: float = 0.0,
        high: float = 1.0,
    ) -> np.ndarray:
        """Return bounds array of shape ``(state_dim, 2)``."""
        bounds = np.empty((self._config.state_dim, 2), dtype=np.float64)
        bounds[:, 0] = low
        bounds[:, 1] = high
        return bounds


# ---------------------------------------------------------------------------
# Concrete benchmarks
# ---------------------------------------------------------------------------


class HighwayIntersectionBenchmark(Benchmark):
    """Four agents driving toward a shared intersection on a 2-D grid.

    A collision race is planted between two agents approaching from
    perpendicular directions.  Detection should flag the intersection
    region as a collision hazard.
    """

    @property
    def name(self) -> str:
        return "highway_intersection"

    @property
    def description(self) -> str:
        return (
            "4-agent intersection with planted collision between "
            "perpendicular approaches"
        )

    def setup(self) -> None:
        cfg = self._config
        state_bounds = self._make_grid_state_space(0.0, 10.0)

        intersection_lo = 4.5
        intersection_hi = 5.5
        intersection_region: Dict[str, Dict[str, float]] = {
            "0": {"lo": intersection_lo, "hi": intersection_hi},
            "1": {"lo": intersection_lo, "hi": intersection_hi},
        }

        self._planted_races = [
            PlantedRace(
                race_type="collision",
                agents_involved=[0, 2],
                trigger_condition=(
                    "Agents 0 (eastbound) and 2 (northbound) simultaneously "
                    "enter the intersection cell [4.5, 5.5]^2"
                ),
                probability=cfg.race_probability,
                state_region=intersection_region,
            )
        ]

        self._environment_config = {
            "type": "grid_2d",
            "state_bounds": state_bounds.tolist(),
            "num_agents": cfg.num_agents,
            "horizon": cfg.horizon,
            "intersection_region": intersection_region,
        }

        directions = ["east", "west", "north", "south"]
        start_positions = [
            [0.0, 5.0],
            [10.0, 5.0],
            [5.0, 0.0],
            [5.0, 10.0],
        ]
        self._policy_configs = [
            {
                "agent_id": i,
                "type": "linear_drive",
                "direction": directions[i],
                "start_position": start_positions[i],
                "speed": 1.0,
                "action_dim": cfg.action_dim,
            }
            for i in range(cfg.num_agents)
        ]

        self._is_setup = True


class HighwayMergingBenchmark(Benchmark):
    """Six agents merging into a shared lane on a 1-D highway segment.

    A merge-conflict race is planted when two agents attempt to occupy the
    same lane position at the merge point.
    """

    @property
    def name(self) -> str:
        return "highway_merging"

    @property
    def description(self) -> str:
        return (
            "6-agent highway merging with planted merge-conflict at the "
            "lane-join point"
        )

    def setup(self) -> None:
        cfg = self._config
        state_bounds = self._make_grid_state_space(0.0, 20.0)

        merge_region: Dict[str, Dict[str, float]] = {
            "0": {"lo": 9.0, "hi": 11.0},
        }

        self._planted_races = [
            PlantedRace(
                race_type="merge_conflict",
                agents_involved=[1, 4],
                trigger_condition=(
                    "Agents 1 (left lane) and 4 (right lane) arrive at the "
                    "merge zone x ∈ [9, 11] within the same time-step"
                ),
                probability=cfg.race_probability,
                state_region=merge_region,
            )
        ]

        self._environment_config = {
            "type": "highway_1d",
            "state_bounds": state_bounds.tolist(),
            "num_agents": cfg.num_agents,
            "horizon": cfg.horizon,
            "merge_region": merge_region,
            "num_lanes": 2,
        }

        self._policy_configs = []
        for i in range(cfg.num_agents):
            lane = i % 2
            start_x = self._rng.uniform(0.0, 5.0)
            self._policy_configs.append(
                {
                    "agent_id": i,
                    "type": "constant_velocity",
                    "lane": lane,
                    "start_position": [start_x, float(lane)],
                    "speed": self._rng.uniform(0.8, 1.2),
                    "action_dim": cfg.action_dim,
                }
            )

        self._is_setup = True


class WarehouseCorridorBenchmark(Benchmark):
    """Eight agents navigating a warehouse with narrow corridors.

    A deadlock race is planted when two agents approach the same corridor
    from opposite ends and neither can pass.
    """

    @property
    def name(self) -> str:
        return "warehouse_corridor"

    @property
    def description(self) -> str:
        return (
            "8-agent warehouse with planted corridor deadlock from "
            "opposing approaches"
        )

    def setup(self) -> None:
        cfg = self._config
        state_bounds = self._make_grid_state_space(0.0, 15.0)

        corridor_region: Dict[str, Dict[str, float]] = {
            "0": {"lo": 6.0, "hi": 9.0},
            "1": {"lo": 4.5, "hi": 5.5},
        }

        self._planted_races = [
            PlantedRace(
                race_type="deadlock",
                agents_involved=[2, 5],
                trigger_condition=(
                    "Agents 2 and 5 enter the narrow corridor "
                    "(x ∈ [6, 9], y ∈ [4.5, 5.5]) from opposite ends and "
                    "block each other"
                ),
                probability=cfg.race_probability,
                state_region=corridor_region,
            )
        ]

        self._environment_config = {
            "type": "warehouse_grid",
            "state_bounds": state_bounds.tolist(),
            "num_agents": cfg.num_agents,
            "horizon": cfg.horizon,
            "corridor_region": corridor_region,
            "corridor_width": 1.0,
        }

        waypoints_pool = [
            [1.0, 1.0],
            [14.0, 1.0],
            [1.0, 14.0],
            [14.0, 14.0],
            [7.5, 1.0],
            [7.5, 14.0],
            [1.0, 7.5],
            [14.0, 7.5],
        ]
        self._policy_configs = []
        for i in range(cfg.num_agents):
            wp = waypoints_pool[i % len(waypoints_pool)]
            self._policy_configs.append(
                {
                    "agent_id": i,
                    "type": "waypoint_follower",
                    "start_position": wp,
                    "goal_position": [15.0 - wp[0], 15.0 - wp[1]],
                    "speed": 0.8,
                    "action_dim": cfg.action_dim,
                }
            )

        self._is_setup = True


class TradingBenchmark(Benchmark):
    """Four trading agents that may trigger a correlated liquidation.

    The planted race fires when all four agents issue *sell* orders in
    the same time-step, causing a cascading price drop.
    """

    @property
    def name(self) -> str:
        return "trading"

    @property
    def description(self) -> str:
        return (
            "4-agent trading with planted correlated-liquidation when all "
            "agents sell simultaneously"
        )

    def setup(self) -> None:
        cfg = self._config
        state_bounds = self._make_grid_state_space(0.0, 100.0)

        panic_region: Dict[str, Dict[str, float]] = {
            "0": {"lo": 0.0, "hi": 20.0},
        }

        self._planted_races = [
            PlantedRace(
                race_type="correlated_liquidation",
                agents_involved=list(range(cfg.num_agents)),
                trigger_condition=(
                    "All agents observe the price state in [0, 20] and "
                    "simultaneously issue sell orders, causing a cascade"
                ),
                probability=cfg.race_probability,
                state_region=panic_region,
            )
        ]

        self._environment_config = {
            "type": "order_book",
            "state_bounds": state_bounds.tolist(),
            "num_agents": cfg.num_agents,
            "horizon": cfg.horizon,
            "panic_region": panic_region,
            "initial_price": 50.0,
            "tick_size": 0.01,
        }

        strategies = ["momentum", "mean_reversion", "market_making", "trend"]
        self._policy_configs = [
            {
                "agent_id": i,
                "type": "trading_strategy",
                "strategy": strategies[i % len(strategies)],
                "risk_limit": self._rng.uniform(0.05, 0.20),
                "action_dim": cfg.action_dim,
            }
            for i in range(cfg.num_agents)
        ]

        self._is_setup = True


class ScalabilityBenchmark(Benchmark):
    """Parameterised benchmark that sweeps agent count from 2 to 12.

    At each scale a simple collision race is planted between agents 0
    and 1.  The primary output is wall-clock timing data for each *N*.
    """

    MIN_AGENTS: int = 2
    MAX_AGENTS: int = 12

    @property
    def name(self) -> str:
        return "scalability"

    @property
    def description(self) -> str:
        return (
            f"Scalability sweep from {self.MIN_AGENTS} to {self.MAX_AGENTS} "
            "agents with planted collision"
        )

    def setup(self) -> None:
        cfg = self._config
        state_bounds = self._make_grid_state_space(0.0, 10.0)

        collision_region: Dict[str, Dict[str, float]] = {
            "0": {"lo": 4.0, "hi": 6.0},
            "1": {"lo": 4.0, "hi": 6.0},
        }

        self._planted_races = [
            PlantedRace(
                race_type="collision",
                agents_involved=[0, 1],
                trigger_condition=(
                    "Agents 0 and 1 enter the collision region "
                    "[4, 6]^2 simultaneously"
                ),
                probability=cfg.race_probability,
                state_region=collision_region,
            )
        ]

        self._environment_config = {
            "type": "grid_2d",
            "state_bounds": state_bounds.tolist(),
            "num_agents": cfg.num_agents,
            "horizon": cfg.horizon,
            "collision_region": collision_region,
        }

        self._policy_configs = [
            {
                "agent_id": i,
                "type": "random_walk",
                "start_position": self._rng.uniform(0.0, 10.0, size=2).tolist(),
                "speed": 1.0,
                "action_dim": cfg.action_dim,
            }
            for i in range(cfg.num_agents)
        ]

        self._is_setup = True

    def run_scaling_sweep(
        self,
        runner_fn: Callable[[Benchmark], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Run the detector at each agent count and collect timing data.

        Parameters
        ----------
        runner_fn:
            A callable that accepts a :class:`Benchmark` (already set up)
            and returns a list of detected-race dictionaries.

        Returns
        -------
        list[dict]
            One entry per agent-count with keys ``"num_agents"``,
            ``"elapsed_s"``, ``"num_detected"``, ``"recall"``,
            ``"precision"``, and ``"f1"``.
        """
        results: List[Dict[str, Any]] = []

        for n in range(self.MIN_AGENTS, self.MAX_AGENTS + 1):
            scaled_cfg = BenchmarkConfig(
                num_agents=n,
                state_dim=self._config.state_dim,
                action_dim=self._config.action_dim,
                horizon=self._config.horizon,
                race_probability=self._config.race_probability,
                seed=self._config.seed + n,
                timeout_s=self._config.timeout_s,
            )
            bench = ScalabilityBenchmark(scaled_cfg)
            bench.setup()

            t0 = time.monotonic()
            detected = runner_fn(bench)
            elapsed = time.monotonic() - t0

            scores = bench.evaluate_results(detected)
            results.append(
                {
                    "num_agents": n,
                    "elapsed_s": elapsed,
                    "num_detected": len(detected),
                    "recall": scores["recall"],
                    "precision": scores["precision"],
                    "f1": scores["f1"],
                }
            )

            bench.teardown()

        return results


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Ordered collection of :class:`Benchmark` instances.

    Provides helpers for batch execution, default construction, and
    human-readable summaries.
    """

    def __init__(self, configs: Optional[List[BenchmarkConfig]] = None) -> None:
        self._benchmarks: List[Benchmark] = []
        if configs is not None:
            suite = self.__class__.get_default_suite(configs)
            self._benchmarks = suite._benchmarks

    @property
    def benchmarks(self) -> List[Benchmark]:
        """The benchmarks registered in this suite (insertion order)."""
        return list(self._benchmarks)

    def add(self, benchmark: Benchmark) -> None:
        """Append *benchmark* to the suite."""
        self._benchmarks.append(benchmark)

    def run_all(
        self,
        runner_fn: Callable[[Benchmark], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Set up, run, evaluate, and tear down every benchmark.

        Parameters
        ----------
        runner_fn:
            Callable accepting a :class:`Benchmark` (already set up) and
            returning a list of detected-race dictionaries.

        Returns
        -------
        list[dict]
            One evaluation-result dictionary per benchmark.
        """
        all_results: List[Dict[str, Any]] = []

        for bench in self._benchmarks:
            bench.setup()
            try:
                detected = runner_fn(bench)
                result = bench.evaluate_results(detected)
                result["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                result = {
                    "benchmark": bench.name,
                    "status": "error",
                    "error": str(exc),
                }
            finally:
                bench.teardown()

            all_results.append(result)

        return all_results

    @classmethod
    def get_default_suite(
        cls,
        configs: Optional[List[BenchmarkConfig]] = None,
    ) -> "BenchmarkSuite":
        """Create a suite with the four domain benchmarks plus scalability.

        If *configs* is ``None`` sensible defaults are used.  Otherwise the
        list must contain exactly five configs corresponding to:

        1. :class:`HighwayIntersectionBenchmark`
        2. :class:`HighwayMergingBenchmark`
        3. :class:`WarehouseCorridorBenchmark`
        4. :class:`TradingBenchmark`
        5. :class:`ScalabilityBenchmark`
        """
        if configs is None:
            configs = [
                BenchmarkConfig(num_agents=4, state_dim=2, action_dim=2),
                BenchmarkConfig(num_agents=6, state_dim=2, action_dim=2),
                BenchmarkConfig(num_agents=8, state_dim=2, action_dim=2),
                BenchmarkConfig(num_agents=4, state_dim=1, action_dim=3),
                BenchmarkConfig(num_agents=2, state_dim=2, action_dim=2),
            ]

        benchmark_classes: List[type[Benchmark]] = [
            HighwayIntersectionBenchmark,
            HighwayMergingBenchmark,
            WarehouseCorridorBenchmark,
            TradingBenchmark,
            ScalabilityBenchmark,
        ]

        if len(configs) != len(benchmark_classes):
            raise ValueError(
                f"Expected {len(benchmark_classes)} configs, got {len(configs)}"
            )

        suite = cls.__new__(cls)
        suite._benchmarks = [
            klass(cfg) for klass, cfg in zip(benchmark_classes, configs)
        ]
        return suite

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the suite."""
        lines = [f"BenchmarkSuite ({len(self._benchmarks)} benchmarks)"]
        lines.append("=" * len(lines[0]))
        for idx, bench in enumerate(self._benchmarks, 1):
            lines.append(f"  {idx}. {bench.name}: {bench.description}")
        return "\n".join(lines)
