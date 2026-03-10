"""
usability_oracle.simulation.engine — Discrete-event cognitive simulation engine.

Orchestrates cognitive processors (perceptual, motor, cognitive control,
working memory, visual attention) via an event-driven loop.  The engine
supports deterministic replay, checkpointing, resource contention, and
multi-trial experiment execution.

Architecture follows the MHP/EPIC/ACT-R tradition of parallel asynchronous
processors communicating through event messages.

References:
    Card, S. K., Moran, T. P., & Newell, A. (1983).
        *The Psychology of Human-Computer Interaction*. Erlbaum.
    Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical
        Universe?* Oxford University Press.
    Kieras, D. E., & Meyer, D. E. (1997). An overview of the EPIC
        architecture. *Cognitive Science*, 21(2), 135-183.
    Banks, J. et al. (2010). *Discrete-Event System Simulation* (5th ed.).
        Prentice Hall.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time as wall_time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from usability_oracle.simulation.event_queue import (
    ConditionalEvent,
    EventFilter,
    EventPriority,
    EventQueue,
    EventType,
    RecurringEvent,
    SimulationEvent,
)
from usability_oracle.simulation.processors import (
    CognitiveControlProcessor,
    CognitiveProcessor,
    MotorProcessor,
    PerceptualProcessor,
    ProcessorState,
    VisualAttentionProcessor,
    WorkingMemoryProcessor,
)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """Configuration for the discrete-event simulation engine.

    Attributes:
        max_time: Maximum simulated time (seconds) before forced stop.
        max_events: Maximum events to process before forced stop.
        seed: Master RNG seed for reproducibility.
        enable_tracing: Whether to record full event trace.
        enable_checkpoints: Whether to take periodic snapshots.
        checkpoint_interval: Simulated time between checkpoints (seconds).
        time_resolution: Minimum time step granularity (seconds).
        resource_contention: Enable bus contention between processors.
        noise_enabled: Enable stochastic noise in processor timing.
        wall_time_limit: Real-time limit (seconds, 0 = unlimited).
    """
    max_time: float = 300.0
    max_events: int = 100_000
    seed: int = 42
    enable_tracing: bool = True
    enable_checkpoints: bool = False
    checkpoint_interval: float = 1.0
    time_resolution: float = 0.0001
    resource_contention: bool = True
    noise_enabled: bool = True
    wall_time_limit: float = 0.0


@dataclass
class CognitiveProcessorConfig:
    """Configuration for individual cognitive processors.

    Uses published parameter defaults from the HCI literature.
    """
    # Perceptual (Card et al. 1983; Kieras & Meyer 1997)
    perceptual_encoding_time: float = 0.100
    eccentricity_slope: float = 0.010
    complexity_factor: float = 1.0

    # Motor (Fitts 1954; MacKenzie 1992; Card et al. 1983)
    fitts_a: float = 0.050
    fitts_b: float = 0.150
    motor_preparation_time: float = 0.150
    keystroke_time: float = 0.280

    # Cognitive control (Anderson 2007)
    production_cycle_time: float = 0.050
    utility_noise: float = 0.0

    # Working memory (Anderson 2007; Cowan 2001)
    wm_capacity: int = 4
    wm_decay_rate: float = 0.5
    retrieval_threshold: float = -0.5
    latency_factor: float = 0.630
    latency_exponent: float = 1.0

    # Visual attention (Posner 1980; Wolfe 1994)
    covert_shift_time: float = 0.050
    saccade_time: float = 0.200
    efficient_search_slope: float = 0.010
    inefficient_search_slope: float = 0.035


# ═══════════════════════════════════════════════════════════════════════════
# Simulation state snapshot
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationState:
    """Complete snapshot of the simulation for checkpointing.

    Captures enough state to resume simulation from this point.
    """
    clock: float = 0.0
    events_processed: int = 0
    pending_events: List[Dict[str, Any]] = field(default_factory=list)
    processor_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    task_state: Dict[str, Any] = field(default_factory=dict)
    rng_state: Optional[Any] = None
    wall_time_elapsed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clock": self.clock,
            "events_processed": self.events_processed,
            "pending_events": self.pending_events,
            "processor_states": self.processor_states,
            "task_state": self.task_state,
            "wall_time_elapsed": self.wall_time_elapsed,
            "metadata": self.metadata,
        }

    def checksum(self) -> str:
        """SHA-256 hash of the state for integrity verification."""
        data = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Simulation trace
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationTrace:
    """Full event log for post-hoc analysis and deterministic replay.

    Stores every event that was processed, along with timing metadata.
    """
    events: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[SimulationState] = field(default_factory=list)
    start_wall_time: float = 0.0
    end_wall_time: float = 0.0
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_event(self, event: SimulationEvent, wall_time_offset: float = 0.0) -> None:
        """Record a processed event."""
        entry = event.to_dict()
        entry["_wall_time_offset"] = wall_time_offset
        self.events.append(entry)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def duration(self) -> float:
        """Simulated time span of the trace."""
        if not self.events:
            return 0.0
        return self.events[-1].get("timestamp", 0.0) - self.events[0].get("timestamp", 0.0)

    @property
    def wall_duration(self) -> float:
        return self.end_wall_time - self.start_wall_time

    def filter_events(self, filt: EventFilter) -> List[Dict[str, Any]]:
        """Filter trace events using an EventFilter (operates on dicts)."""
        filtered = []
        for entry in self.events:
            try:
                evt = SimulationEvent(
                    event_id=entry.get("event_id", 0),
                    timestamp=entry.get("timestamp", 0.0),
                    event_type=EventType[entry.get("event_type", "SIMULATION_START")],
                    source_processor=entry.get("source_processor", ""),
                    target_processor=entry.get("target_processor", ""),
                    payload=entry.get("payload", {}),
                    cancelled=entry.get("cancelled", False),
                )
                if filt.matches(evt):
                    filtered.append(entry)
            except (KeyError, ValueError):
                continue
        return filtered

    def event_counts_by_type(self) -> Dict[str, int]:
        """Count events by type."""
        counts: Dict[str, int] = {}
        for entry in self.events:
            etype = entry.get("event_type", "UNKNOWN")
            counts[etype] = counts.get(etype, 0) + 1
        return counts

    def processor_timeline(self, processor_name: str) -> List[Dict[str, Any]]:
        """Extract events for a specific processor (source or target)."""
        return [
            e for e in self.events
            if e.get("source_processor") == processor_name
            or e.get("target_processor") == processor_name
        ]

    def inter_event_times(self) -> List[float]:
        """Compute inter-event time deltas."""
        if len(self.events) < 2:
            return []
        times = [e.get("timestamp", 0.0) for e in self.events]
        return [times[i + 1] - times[i] for i in range(len(times) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_events": self.n_events,
            "duration": self.duration,
            "wall_duration": self.wall_duration,
            "events": self.events,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "config": self.config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationTrace:
        """Reconstruct a trace from a dict (for replay)."""
        trace = cls()
        trace.events = data.get("events", [])
        trace.config = data.get("config")
        trace.metadata = data.get("metadata", {})
        return trace


# ═══════════════════════════════════════════════════════════════════════════
# Resource bus (contention model)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceBus:
    """Shared communication bus between processors with contention.

    When two processors need the same bus simultaneously, the lower-priority
    request is queued.

    Reference: Kieras & Meyer (1997), EPIC resource contention.
    """
    name: str = ""
    busy_until: float = 0.0
    transfer_time: float = 0.010  # 10 ms bus transfer
    queue: List[Tuple[float, SimulationEvent]] = field(default_factory=list)

    def request(self, event: SimulationEvent, current_time: float) -> float:
        """Request bus access. Returns the time the transfer will complete."""
        available_at = max(current_time, self.busy_until)
        completion = available_at + self.transfer_time
        self.busy_until = completion
        return completion

    def is_available(self, current_time: float) -> bool:
        return current_time >= self.busy_until

    def reset(self) -> None:
        self.busy_until = 0.0
        self.queue.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Trial result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """Result of a single simulation trial."""
    trial_id: int = 0
    task_completed: bool = False
    completion_time: float = 0.0
    n_events: int = 0
    n_errors: int = 0
    processor_utilizations: Dict[str, float] = field(default_factory=dict)
    trace: Optional[SimulationTrace] = None
    final_state: Optional[SimulationState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Aggregated results from multiple trials."""
    trials: List[TrialResult] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def completion_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.task_completed) / len(self.trials)

    @property
    def mean_completion_time(self) -> float:
        completed = [t.completion_time for t in self.trials if t.task_completed]
        return float(np.mean(completed)) if completed else 0.0

    @property
    def std_completion_time(self) -> float:
        completed = [t.completion_time for t in self.trials if t.task_completed]
        return float(np.std(completed)) if len(completed) > 1 else 0.0

    @property
    def median_completion_time(self) -> float:
        completed = [t.completion_time for t in self.trials if t.task_completed]
        return float(np.median(completed)) if completed else 0.0

    @property
    def p95_completion_time(self) -> float:
        completed = [t.completion_time for t in self.trials if t.task_completed]
        return float(np.percentile(completed, 95)) if completed else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "completion_rate": self.completion_rate,
            "mean_time": self.mean_completion_time,
            "std_time": self.std_completion_time,
            "median_time": self.median_completion_time,
            "p95_time": self.p95_completion_time,
        }


# ═══════════════════════════════════════════════════════════════════════════
# DiscreteEventSimulator
# ═══════════════════════════════════════════════════════════════════════════

class DiscreteEventSimulator:
    """Main discrete-event cognitive simulation engine.

    Manages a set of cognitive processors that communicate via events
    routed through a priority queue.  The simulation loop:

    1. Pop the next event from the queue.
    2. Advance the simulation clock to the event's timestamp.
    3. Route the event to its target processor.
    4. Collect output events from the processor and insert them.
    5. Evaluate conditional events.
    6. Generate next instances of recurring events.
    7. Check termination conditions.

    Usage::

        sim = DiscreteEventSimulator()
        sim.initialize(task_spec, cognitive_config=CognitiveProcessorConfig())
        result = sim.run(max_time=30.0)
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self._config = config or SimulationConfig()
        self._queue = EventQueue()
        self._clock: float = 0.0
        self._events_processed: int = 0
        self._processors: Dict[str, CognitiveProcessor] = {}
        self._buses: Dict[str, ResourceBus] = {}
        self._conditional_events: List[ConditionalEvent] = []
        self._recurring_events: List[RecurringEvent] = []
        self._trace = SimulationTrace()
        self._checkpoints: List[SimulationState] = []
        self._last_checkpoint_time: float = 0.0
        self._task_state: Dict[str, Any] = {}
        self._completion_check: Optional[Callable[[Dict[str, Any]], bool]] = None
        self._event_hooks: Dict[EventType, List[Callable]] = {}
        self._rng = random.Random(self._config.seed)
        self._running: bool = False
        self._wall_start: float = 0.0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(
        self,
        task_spec: Optional[Dict[str, Any]] = None,
        ui_mdp: Optional[Any] = None,
        cognitive_config: Optional[CognitiveProcessorConfig] = None,
    ) -> None:
        """Set up processors, buses, and initial events.

        Parameters:
            task_spec: Description of the task (goals, steps, elements).
            ui_mdp: Optional MDP representation of the UI.
            cognitive_config: Processor parameter configuration.
        """
        cfg = cognitive_config or CognitiveProcessorConfig()
        task = task_spec or {}
        seed = self._config.seed

        # Create processors
        self._processors = {
            "perceptual": PerceptualProcessor(
                base_encoding_time=cfg.perceptual_encoding_time,
                eccentricity_slope=cfg.eccentricity_slope,
                complexity_factor=cfg.complexity_factor,
                noise_sigma=0.015 if self._config.noise_enabled else 0.0,
                seed=seed,
            ),
            "motor": MotorProcessor(
                fitts_a=cfg.fitts_a,
                fitts_b=cfg.fitts_b,
                preparation_time=cfg.motor_preparation_time,
                keystroke_time=cfg.keystroke_time,
                noise_sigma=0.020 if self._config.noise_enabled else 0.0,
                seed=seed + 1 if seed else None,
            ),
            "cognitive_control": CognitiveControlProcessor(
                cycle_time=cfg.production_cycle_time,
                utility_noise=cfg.utility_noise,
                noise_sigma=0.010 if self._config.noise_enabled else 0.0,
                seed=seed + 2 if seed else None,
            ),
            "working_memory": WorkingMemoryProcessor(
                capacity=cfg.wm_capacity,
                decay_rate=cfg.wm_decay_rate,
                retrieval_threshold=cfg.retrieval_threshold,
                latency_factor=cfg.latency_factor,
                latency_exponent=cfg.latency_exponent,
                noise_sigma=0.25 if self._config.noise_enabled else 0.0,
                seed=seed + 3 if seed else None,
            ),
            "visual_attention": VisualAttentionProcessor(
                covert_shift_time=cfg.covert_shift_time,
                saccade_time=cfg.saccade_time,
                efficient_slope=cfg.efficient_search_slope,
                inefficient_slope=cfg.inefficient_search_slope,
                noise_sigma=0.012 if self._config.noise_enabled else 0.0,
                seed=seed + 4 if seed else None,
            ),
        }

        # Create resource buses
        self._buses = {
            "visual_bus": ResourceBus(name="visual_bus", transfer_time=0.010),
            "motor_bus": ResourceBus(name="motor_bus", transfer_time=0.010),
            "cognitive_bus": ResourceBus(name="cognitive_bus", transfer_time=0.005),
        }

        # Store task state
        self._task_state = {
            "task_name": task.get("name", "unnamed"),
            "goal_elements": set(task.get("goal_elements", [])),
            "completed_elements": set(),
            "current_step": 0,
            "total_steps": len(task.get("steps", [])),
            "steps": task.get("steps", []),
            "errors": 0,
            "task_completed": False,
        }

        # Set up completion check
        if "completion_check" in task:
            self._completion_check = task["completion_check"]
        else:
            self._completion_check = self._default_completion_check

        # Reset state
        self._clock = 0.0
        self._events_processed = 0
        self._queue.clear()
        self._trace = SimulationTrace(config=self._config.__dict__)
        self._checkpoints.clear()
        self._last_checkpoint_time = 0.0
        self._conditional_events.clear()
        self._recurring_events.clear()

        # Insert initial events
        self._queue.insert(SimulationEvent(
            timestamp=0.0,
            event_type=EventType.SIMULATION_START,
            source_processor="engine",
            target_processor="engine",
            payload={"task": task.get("name", "")},
            priority=EventPriority.CRITICAL,
        ))

        # Schedule initial task events
        if task.get("initial_elements"):
            for i, elem in enumerate(task["initial_elements"]):
                self._queue.insert(SimulationEvent(
                    timestamp=0.001 + i * 0.001,
                    event_type=EventType.VISUAL_ONSET,
                    source_processor="environment",
                    target_processor="visual_attention",
                    payload=elem,
                    priority=EventPriority.PERCEPTUAL,
                ))

        # Set up recurring WM decay
        self._recurring_events.append(RecurringEvent(
            event_template=SimulationEvent(
                event_type=EventType.WM_DECAY_TICK,
                source_processor="engine",
                target_processor="working_memory",
                payload={"operation": "decay_tick"},
            ),
            interval=1.0,
            start_time=1.0,
            stop_time=self._config.max_time,
        ))

    def _default_completion_check(self, state: Dict[str, Any]) -> bool:
        """Default task completion: all goal elements visited."""
        goals = state.get("goal_elements", set())
        if not goals:
            return False
        completed = state.get("completed_elements", set())
        return goals.issubset(completed)

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    def _route_event(self, event: SimulationEvent) -> List[SimulationEvent]:
        """Route an event to its target processor and collect outputs."""
        target = event.target_processor

        # Engine-level events
        if target == "engine":
            return self._handle_engine_event(event)

        # Environment events
        if target == "environment":
            return self._handle_environment_event(event)

        # Trace-only events
        if target == "trace":
            return []

        # Processor events
        processor = self._processors.get(target)
        if processor is None:
            return []

        # Resource contention check
        if self._config.resource_contention:
            bus = self._get_bus_for_processor(target)
            if bus and not bus.is_available(event.timestamp):
                completion = bus.request(event, event.timestamp)
                delayed = SimulationEvent(
                    timestamp=completion,
                    event_type=event.event_type,
                    source_processor=event.source_processor,
                    target_processor=event.target_processor,
                    payload=event.payload,
                    priority=event.priority,
                    metadata={**event.metadata, "_delayed_by_contention": True},
                )
                return [delayed]
            if bus:
                bus.request(event, event.timestamp)

        return processor.receive_event(event)

    def _get_bus_for_processor(self, processor_name: str) -> Optional[ResourceBus]:
        """Map processors to their resource bus."""
        mapping = {
            "perceptual": "visual_bus",
            "visual_attention": "visual_bus",
            "motor": "motor_bus",
            "cognitive_control": "cognitive_bus",
            "working_memory": "cognitive_bus",
        }
        bus_name = mapping.get(processor_name)
        return self._buses.get(bus_name, None) if bus_name else None

    def _handle_engine_event(self, event: SimulationEvent) -> List[SimulationEvent]:
        """Handle simulation-level events."""
        if event.event_type == EventType.SIMULATION_START:
            return []
        if event.event_type == EventType.TIMEOUT:
            self._running = False
            return []
        return []

    def _handle_environment_event(self, event: SimulationEvent) -> List[SimulationEvent]:
        """Handle events that modify the environment state."""
        element_id = event.payload.get("target_element", "")
        if element_id:
            self._task_state["completed_elements"].add(element_id)
            self._task_state["current_step"] += 1

        # Generate system response event
        response_delay = event.payload.get("system_response_time", 0.100)
        return [SimulationEvent(
            timestamp=event.timestamp + response_delay,
            event_type=EventType.SYSTEM_RESPONSE,
            source_processor="environment",
            target_processor="perceptual",
            payload={
                "response_type": "feedback",
                "element_id": element_id,
                "success": True,
            },
            priority=EventPriority.SYSTEM,
        )]

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def register_hook(self, event_type: EventType, callback: Callable[[SimulationEvent], None]) -> None:
        """Register a callback to be invoked when events of a type are processed."""
        if event_type not in self._event_hooks:
            self._event_hooks[event_type] = []
        self._event_hooks[event_type].append(callback)

    def _fire_hooks(self, event: SimulationEvent) -> None:
        """Invoke registered hooks for an event."""
        callbacks = self._event_hooks.get(event.event_type, [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Step and run
    # ------------------------------------------------------------------

    def step(self) -> Optional[SimulationEvent]:
        """Advance simulation by one event.

        Returns the processed event, or None if simulation is complete.
        """
        event = self._queue.pop()
        if event is None:
            self._running = False
            return None

        # Advance clock
        self._clock = event.timestamp
        self._events_processed += 1

        # Record trace
        if self._config.enable_tracing:
            offset = wall_time.time() - self._wall_start if self._wall_start > 0 else 0.0
            self._trace.record_event(event, offset)

        # Fire hooks
        self._fire_hooks(event)

        # Route event and collect outputs
        output_events = self._route_event(event)

        # Insert output events
        for out_event in output_events:
            if out_event.timestamp < self._clock:
                out_event.timestamp = self._clock + self._config.time_resolution
            self._queue.insert(out_event)

        # Evaluate conditional events
        self._evaluate_conditionals()

        # Generate recurring events
        self._generate_recurring()

        # Checkpoint if needed
        if self._config.enable_checkpoints:
            if self._clock - self._last_checkpoint_time >= self._config.checkpoint_interval:
                self._take_checkpoint()

        # Check task completion
        if self._completion_check and self._completion_check(self._task_state):
            self._task_state["task_completed"] = True
            self._queue.insert(SimulationEvent(
                timestamp=self._clock + self._config.time_resolution,
                event_type=EventType.TASK_COMPLETE,
                source_processor="engine",
                target_processor="engine",
                payload={"completion_time": self._clock},
                priority=EventPriority.CRITICAL,
            ))
            self._running = False

        return event

    def run(
        self,
        max_time: Optional[float] = None,
        max_events: Optional[int] = None,
    ) -> SimulationTrace:
        """Run simulation to completion or limit.

        Parameters:
            max_time: Override config max_time.
            max_events: Override config max_events.

        Returns:
            The complete simulation trace.
        """
        effective_max_time = max_time if max_time is not None else self._config.max_time
        effective_max_events = max_events if max_events is not None else self._config.max_events

        self._running = True
        self._wall_start = wall_time.time()
        self._trace.start_wall_time = self._wall_start

        while self._running:
            # Time limit
            if self._clock >= effective_max_time:
                break
            # Event limit
            if self._events_processed >= effective_max_events:
                break
            # Wall time limit
            if self._config.wall_time_limit > 0:
                if (wall_time.time() - self._wall_start) >= self._config.wall_time_limit:
                    break
            # Queue empty
            if self._queue.is_empty:
                break

            event = self.step()
            if event is None:
                break

        self._trace.end_wall_time = wall_time.time()
        self._running = False
        return self._trace

    def run_trial(
        self,
        task_spec: Dict[str, Any],
        n_repetitions: int = 1,
        cognitive_config: Optional[CognitiveProcessorConfig] = None,
    ) -> ExperimentResult:
        """Run multiple trials and collect statistics.

        Parameters:
            task_spec: Task specification.
            n_repetitions: Number of independent trials.
            cognitive_config: Processor parameters.

        Returns:
            Aggregated experiment results.
        """
        result = ExperimentResult(
            config={"task": task_spec, "n_repetitions": n_repetitions},
        )

        for trial_idx in range(n_repetitions):
            # Vary seed per trial for stochastic variation
            trial_config = SimulationConfig(
                max_time=self._config.max_time,
                max_events=self._config.max_events,
                seed=self._config.seed + trial_idx,
                enable_tracing=self._config.enable_tracing,
                noise_enabled=self._config.noise_enabled,
                resource_contention=self._config.resource_contention,
                wall_time_limit=self._config.wall_time_limit,
            )

            sim = DiscreteEventSimulator(config=trial_config)
            sim.initialize(task_spec=task_spec, cognitive_config=cognitive_config)
            trace = sim.run()

            trial_result = TrialResult(
                trial_id=trial_idx,
                task_completed=sim._task_state.get("task_completed", False),
                completion_time=sim._clock,
                n_events=sim._events_processed,
                n_errors=sim._task_state.get("errors", 0),
                processor_utilizations={
                    name: proc.utilization
                    for name, proc in sim._processors.items()
                },
                trace=trace if self._config.enable_tracing else None,
                final_state=sim.snapshot(),
            )
            result.trials.append(trial_result)

        return result

    # ------------------------------------------------------------------
    # Conditional and recurring event management
    # ------------------------------------------------------------------

    def add_conditional_event(self, cond_event: ConditionalEvent) -> None:
        """Register a conditional event."""
        self._conditional_events.append(cond_event)

    def add_recurring_event(self, rec_event: RecurringEvent) -> None:
        """Register a recurring event."""
        self._recurring_events.append(rec_event)

    def _evaluate_conditionals(self) -> None:
        """Evaluate all pending conditional events."""
        for ce in self._conditional_events:
            if ce.is_expired:
                continue
            event = ce.evaluate(self._task_state, self._clock)
            if event is not None:
                self._queue.insert(event)
        # Clean up expired
        self._conditional_events = [ce for ce in self._conditional_events if not ce.is_expired]

    def _generate_recurring(self) -> None:
        """Generate next instances of recurring events."""
        for re in self._recurring_events:
            if not re.active:
                continue
            if re.next_firing_time <= self._clock + self._config.time_resolution:
                jitter = self._rng.uniform(-re.jitter, re.jitter) if re.jitter > 0 else 0.0
                event = re.generate_next(jitter)
                if event is not None:
                    self._queue.insert(event)
        # Clean up inactive
        self._recurring_events = [re for re in self._recurring_events if re.active]

    # ------------------------------------------------------------------
    # Checkpointing and replay
    # ------------------------------------------------------------------

    def snapshot(self) -> SimulationState:
        """Take a complete simulation snapshot."""
        state = SimulationState(
            clock=self._clock,
            events_processed=self._events_processed,
            pending_events=self._queue.snapshot(),
            processor_states={
                name: proc.to_dict()
                for name, proc in self._processors.items()
            },
            task_state={
                k: (list(v) if isinstance(v, set) else v)
                for k, v in self._task_state.items()
            },
            wall_time_elapsed=wall_time.time() - self._wall_start if self._wall_start > 0 else 0.0,
        )
        return state

    def _take_checkpoint(self) -> None:
        """Record a checkpoint at the current time."""
        state = self.snapshot()
        self._checkpoints.append(state)
        self._trace.checkpoints.append(state)
        self._last_checkpoint_time = self._clock

    @staticmethod
    def replay(trace: SimulationTrace) -> List[SimulationEvent]:
        """Deterministic replay from a saved trace.

        Reconstructs the event sequence without re-running processors.

        Returns:
            List of reconstructed SimulationEvent objects.
        """
        events = []
        for entry in trace.events:
            try:
                event = SimulationEvent(
                    event_id=entry.get("event_id", 0),
                    timestamp=entry.get("timestamp", 0.0),
                    event_type=EventType[entry.get("event_type", "SIMULATION_START")],
                    source_processor=entry.get("source_processor", ""),
                    target_processor=entry.get("target_processor", ""),
                    payload=entry.get("payload", {}),
                    cancelled=entry.get("cancelled", False),
                )
                events.append(event)
            except (KeyError, ValueError):
                continue
        return events

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def clock(self) -> float:
        return self._clock

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def task_state(self) -> Dict[str, Any]:
        return dict(self._task_state)

    @property
    def processors(self) -> Dict[str, CognitiveProcessor]:
        return dict(self._processors)

    @property
    def trace(self) -> SimulationTrace:
        return self._trace

    def get_processor(self, name: str) -> Optional[CognitiveProcessor]:
        return self._processors.get(name)

    def reset(self) -> None:
        """Reset the entire simulation."""
        self._clock = 0.0
        self._events_processed = 0
        self._queue.clear()
        self._running = False
        self._trace = SimulationTrace()
        self._checkpoints.clear()
        for proc in self._processors.values():
            proc.reset()
        for bus in self._buses.values():
            bus.reset()
        self._task_state.clear()
        self._conditional_events.clear()
        self._recurring_events.clear()
