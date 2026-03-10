"""
usability_oracle.simulation.processors — Cognitive processor models.

Implements a multi-processor cognitive architecture inspired by ACT-R
(Anderson, 2007) and EPIC (Kieras & Meyer, 1997).  Each processor has
input/output buffers, a state machine, and timing models derived from
the experimental literature.

Processors communicate exclusively via events routed through the
discrete-event simulation engine's EventQueue.

References:
    Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical
        Universe?* Oxford University Press.
    Anderson, J. R. et al. (2004). An integrated theory of the mind.
        *Psychological Review*, 111(4), 1036-1060.
    Kieras, D. E., & Meyer, D. E. (1997). An overview of the EPIC
        architecture. *Cognitive Science*, 21(2), 135-183.
    Card, S. K., Moran, T. P., & Newell, A. (1983).
        *The Psychology of Human-Computer Interaction*. Erlbaum.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from usability_oracle.simulation.event_queue import (
    EventPriority,
    EventType,
    SimulationEvent,
)


# ═══════════════════════════════════════════════════════════════════════════
# Processor state machine
# ═══════════════════════════════════════════════════════════════════════════

@unique
class ProcessorState(Enum):
    """State machine for cognitive processors.

    Each processor cycles: IDLE → PREPARING → PROCESSING → RESPONDING → IDLE.
    The BUSY state is used when a processor is occupied and cannot accept
    new inputs (resource contention).

    Reference: Kieras & Meyer (1997), EPIC processor lifecycle.
    """
    IDLE = auto()
    PREPARING = auto()
    PROCESSING = auto()
    RESPONDING = auto()
    BUSY = auto()
    ERROR = auto()


# ═══════════════════════════════════════════════════════════════════════════
# Buffer model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessorBuffer:
    """Bounded FIFO buffer for inter-processor communication.

    Models the input and output buffers of cognitive processors.
    In EPIC, buffers hold at most one item; we generalize to a configurable
    capacity.

    Attributes:
        name: Buffer identifier (e.g., 'visual_input', 'motor_output').
        capacity: Maximum number of items.
        items: Current buffer contents.
    """
    name: str = ""
    capacity: int = 1
    items: List[Dict[str, Any]] = field(default_factory=list)

    def put(self, item: Dict[str, Any]) -> bool:
        """Place an item in the buffer. Returns False if full."""
        if len(self.items) >= self.capacity:
            return False
        self.items.append(item)
        return True

    def get(self) -> Optional[Dict[str, Any]]:
        """Remove and return the oldest item, or None if empty."""
        if not self.items:
            return None
        return self.items.pop(0)

    def peek(self) -> Optional[Dict[str, Any]]:
        """Return the oldest item without removing, or None if empty."""
        return self.items[0] if self.items else None

    @property
    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    @property
    def size(self) -> int:
        return len(self.items)

    def clear(self) -> None:
        self.items.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "capacity": self.capacity, "size": self.size}


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveProcessor — abstract base class
# ═══════════════════════════════════════════════════════════════════════════

class CognitiveProcessor(ABC):
    """Abstract base class for all cognitive processors.

    Each processor has:
    - An input buffer that receives events from other processors
    - An output buffer that emits events to downstream processors
    - A state machine controlling its lifecycle
    - A timing model that determines processing duration

    Subclasses must implement ``_compute_processing_time`` and
    ``_process_item`` to define processor-specific behavior.

    Reference: Anderson et al. (2004), ACT-R module architecture.
    """

    def __init__(
        self,
        name: str,
        input_capacity: int = 1,
        output_capacity: int = 4,
        noise_sigma: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self._name = name
        self._state = ProcessorState.IDLE
        self._input_buffer = ProcessorBuffer(name=f"{name}_input", capacity=input_capacity)
        self._output_buffer = ProcessorBuffer(name=f"{name}_output", capacity=output_capacity)
        self._noise_sigma = noise_sigma
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        self._current_item: Optional[Dict[str, Any]] = None
        self._processing_start_time: float = 0.0
        self._processing_end_time: float = 0.0
        self._total_busy_time: float = 0.0
        self._items_processed: int = 0
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> ProcessorState:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == ProcessorState.IDLE

    @property
    def is_busy(self) -> bool:
        return self._state in (ProcessorState.PREPARING, ProcessorState.PROCESSING,
                                ProcessorState.RESPONDING, ProcessorState.BUSY)

    @property
    def input_buffer(self) -> ProcessorBuffer:
        return self._input_buffer

    @property
    def output_buffer(self) -> ProcessorBuffer:
        return self._output_buffer

    @property
    def utilization(self) -> float:
        """Fraction of time spent processing (0-1)."""
        total = self._processing_end_time  # last known time
        if total <= 0:
            return 0.0
        return min(self._total_busy_time / total, 1.0)

    @property
    def items_processed(self) -> int:
        return self._items_processed

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def receive_event(self, event: SimulationEvent) -> List[SimulationEvent]:
        """Accept an incoming event and return any output events generated.

        This is the main entry point called by the simulation engine.
        """
        # Place event payload in input buffer
        item = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            **event.payload,
        }

        if self._input_buffer.is_full:
            # Buffer overflow — processor is busy, re-queue after delay
            return [SimulationEvent(
                timestamp=event.timestamp + 0.050,
                event_type=event.event_type,
                source_processor=event.source_processor,
                target_processor=self._name,
                payload=event.payload,
                priority=event.priority,
                metadata={"_retransmit": True},
            )]

        self._input_buffer.put(item)

        # If idle, start processing immediately
        if self._state == ProcessorState.IDLE:
            return self._begin_processing(event.timestamp)
        return []

    def _begin_processing(self, current_time: float) -> List[SimulationEvent]:
        """Start processing the next item in the input buffer."""
        item = self._input_buffer.get()
        if item is None:
            self._state = ProcessorState.IDLE
            return []

        self._state = ProcessorState.PREPARING
        self._current_item = item
        self._processing_start_time = current_time

        # Compute processing time
        base_time = self._compute_processing_time(item)
        noise = self._noise_sigma * self._np_rng.logistic(0, 1) if self._noise_sigma > 0 else 0.0
        processing_time = max(base_time + noise, 0.001)
        self._processing_end_time = current_time + processing_time

        # Transition to processing state
        self._state = ProcessorState.PROCESSING

        # Process and generate output events
        output_events = self._process_item(item, current_time, processing_time)

        # Record history
        self._history.append({
            "item": item,
            "start_time": current_time,
            "processing_time": processing_time,
            "n_outputs": len(output_events),
        })

        self._total_busy_time += processing_time
        self._items_processed += 1
        self._state = ProcessorState.RESPONDING

        # Schedule transition back to idle
        completion_events = output_events + [SimulationEvent(
            timestamp=current_time + processing_time,
            event_type=EventType.CHECKPOINT,
            source_processor=self._name,
            target_processor=self._name,
            payload={"_processor_idle": True},
            priority=EventPriority.LOW,
        )]

        return completion_events

    def complete_cycle(self, current_time: float) -> List[SimulationEvent]:
        """Complete the current processing cycle and start next if available."""
        self._state = ProcessorState.IDLE
        self._current_item = None
        if not self._input_buffer.is_empty:
            return self._begin_processing(current_time)
        return []

    @abstractmethod
    def _compute_processing_time(self, item: Dict[str, Any]) -> float:
        """Compute the processing time for a given input item (seconds)."""
        ...

    @abstractmethod
    def _process_item(
        self,
        item: Dict[str, Any],
        current_time: float,
        processing_time: float,
    ) -> List[SimulationEvent]:
        """Process an item and return output events."""
        ...

    def reset(self) -> None:
        """Reset processor to initial state."""
        self._state = ProcessorState.IDLE
        self._input_buffer.clear()
        self._output_buffer.clear()
        self._current_item = None
        self._total_busy_time = 0.0
        self._items_processed = 0
        self._processing_start_time = 0.0
        self._processing_end_time = 0.0
        self._history.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "state": self._state.name,
            "items_processed": self._items_processed,
            "total_busy_time": self._total_busy_time,
            "input_buffer": self._input_buffer.to_dict(),
            "output_buffer": self._output_buffer.to_dict(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# PerceptualProcessor
# ═══════════════════════════════════════════════════════════════════════════

class PerceptualProcessor(CognitiveProcessor):
    """Visual encoding processor with eccentricity-dependent latency.

    Models the time to encode a visual stimulus into a cognitive
    representation.  Encoding time increases with retinal eccentricity
    following the relationship from Anstis (1974) and the EPIC
    architecture (Kieras & Meyer, 1997):

        T_encode = base_time + eccentricity_slope * eccentricity

    where eccentricity is in degrees of visual angle and base_time is
    the foveal encoding time (~100 ms for simple stimuli, ~340 ms for
    complex objects; Card et al. 1983, p. 63).

    Parameters:
        base_encoding_time: Foveal encoding latency (seconds).
            Default: 0.100 s (Card et al. 1983, Processor τ_p).
        eccentricity_slope: Additional time per degree of eccentricity.
            Default: 0.010 s/degree (Kieras & Meyer 1997).
        complexity_factor: Multiplier for complex stimuli (icons vs text).
            Default: 1.0 (simple); use ~2.0 for complex objects.
        fixation_position: Current gaze position (x, y) in pixels.
        pixels_per_degree: Conversion factor (default: 38 px/deg at 60 cm
            viewing distance on a 96 PPI display).
    """

    def __init__(
        self,
        name: str = "perceptual",
        base_encoding_time: float = 0.100,
        eccentricity_slope: float = 0.010,
        complexity_factor: float = 1.0,
        fixation_position: Tuple[float, float] = (512.0, 384.0),
        pixels_per_degree: float = 38.0,
        noise_sigma: float = 0.015,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, input_capacity=2, output_capacity=4,
                         noise_sigma=noise_sigma, seed=seed)
        self._base_encoding_time = base_encoding_time
        self._eccentricity_slope = eccentricity_slope
        self._complexity_factor = complexity_factor
        self._fixation = fixation_position
        self._ppd = pixels_per_degree

    def set_fixation(self, x: float, y: float) -> None:
        """Update current gaze fixation position."""
        self._fixation = (x, y)

    def _eccentricity_deg(self, target_x: float, target_y: float) -> float:
        """Compute retinal eccentricity in degrees of visual angle."""
        dx = target_x - self._fixation[0]
        dy = target_y - self._fixation[1]
        distance_px = math.sqrt(dx * dx + dy * dy)
        return distance_px / self._ppd

    def _compute_processing_time(self, item: Dict[str, Any]) -> float:
        """Encoding time = base + eccentricity_slope * eccentricity * complexity."""
        target_x = item.get("target_x", self._fixation[0])
        target_y = item.get("target_y", self._fixation[1])
        ecc = self._eccentricity_deg(target_x, target_y)
        return (self._base_encoding_time + self._eccentricity_slope * ecc) * self._complexity_factor

    def _process_item(
        self,
        item: Dict[str, Any],
        current_time: float,
        processing_time: float,
    ) -> List[SimulationEvent]:
        """Produce a VISUAL_ENCODING_COMPLETE event after encoding."""
        return [SimulationEvent(
            timestamp=current_time + processing_time,
            event_type=EventType.VISUAL_ENCODING_COMPLETE,
            source_processor=self._name,
            target_processor="cognitive_control",
            payload={
                "encoded_element": item.get("element_id", ""),
                "element_name": item.get("element_name", ""),
                "element_role": item.get("element_role", ""),
                "encoding_time": processing_time,
                "eccentricity_deg": self._eccentricity_deg(
                    item.get("target_x", 0), item.get("target_y", 0)
                ),
            },
            priority=EventPriority.PERCEPTUAL,
        )]


# ═══════════════════════════════════════════════════════════════════════════
# MotorProcessor
# ═══════════════════════════════════════════════════════════════════════════

class MotorProcessor(CognitiveProcessor):
    """Motor execution processor implementing Fitts' law timing.

    Models a two-phase motor action: preparation + execution.

    Preparation time:
        T_prep = 0.150 s (Card et al. 1983, Processor τ_m preparation).

    Execution time (pointing):
        T_exec = a + b * log2(D / W + 1)   (Fitts, 1954; Shannon form)

    where D is movement distance, W is target width, a and b are
    empirically derived coefficients.

    Keystroke time:
        T_key = 0.280 s (average typist; Card et al. 1983, p. 264).
        Expert typist: 0.120 s; worst-case: 0.500 s.

    References:
        Fitts, P. M. (1954). The information capacity of the human motor
            system. *Journal of Experimental Psychology*, 47(6), 381-391.
        Card, S. K., Moran, T. P., & Newell, A. (1983). Ch. 2.
    """

    # Published parameter defaults
    DEFAULT_FITTS_A: float = 0.050   # seconds (MacKenzie 1992 median)
    DEFAULT_FITTS_B: float = 0.150   # seconds/bit (MacKenzie 1992 median)
    DEFAULT_PREP_TIME: float = 0.150  # seconds (Card et al. 1983)
    DEFAULT_KEYSTROKE_TIME: float = 0.280  # seconds (average typist)

    def __init__(
        self,
        name: str = "motor",
        fitts_a: float = DEFAULT_FITTS_A,
        fitts_b: float = DEFAULT_FITTS_B,
        preparation_time: float = DEFAULT_PREP_TIME,
        keystroke_time: float = DEFAULT_KEYSTROKE_TIME,
        current_position: Tuple[float, float] = (0.0, 0.0),
        noise_sigma: float = 0.020,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, input_capacity=2, output_capacity=4,
                         noise_sigma=noise_sigma, seed=seed)
        self._fitts_a = fitts_a
        self._fitts_b = fitts_b
        self._prep_time = preparation_time
        self._keystroke_time = keystroke_time
        self._position = current_position

    def _fitts_movement_time(self, distance: float, target_width: float) -> float:
        """Compute Fitts' law movement time (Shannon formulation).

        MT = a + b * log2(D/W + 1)

        Reference: MacKenzie, I. S. (1992).
        """
        if target_width < 1.0:
            target_width = 1.0
        id_bits = math.log2(distance / target_width + 1.0)
        return self._fitts_a + self._fitts_b * id_bits

    def _compute_processing_time(self, item: Dict[str, Any]) -> float:
        """Total motor time = preparation + execution."""
        action_type = item.get("action_type", "click")

        if action_type == "keystroke":
            return self._prep_time + self._keystroke_time

        # Pointing movement
        target_x = item.get("target_x", 0.0)
        target_y = item.get("target_y", 0.0)
        target_w = max(item.get("target_width", 20.0), 1.0)

        dx = target_x - self._position[0]
        dy = target_y - self._position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1.0:
            return self._prep_time + self._fitts_a

        return self._prep_time + self._fitts_movement_time(distance, target_w)

    def _process_item(
        self,
        item: Dict[str, Any],
        current_time: float,
        processing_time: float,
    ) -> List[SimulationEvent]:
        """Generate preparation-complete and execution-complete events."""
        action_type = item.get("action_type", "click")
        prep_end = current_time + self._prep_time
        exec_end = current_time + processing_time

        # Update hand position
        if action_type != "keystroke":
            self._position = (item.get("target_x", 0.0), item.get("target_y", 0.0))

        events: List[SimulationEvent] = []

        # Preparation complete event
        events.append(SimulationEvent(
            timestamp=prep_end,
            event_type=EventType.MOTOR_PREPARATION_COMPLETE,
            source_processor=self._name,
            target_processor=self._name,
            payload={"action_type": action_type, "phase": "preparation"},
            priority=EventPriority.MOTOR,
        ))

        # Execution complete event
        motor_event_type = {
            "click": EventType.MOUSE_CLICK,
            "keystroke": EventType.KEYSTROKE,
            "move": EventType.MOUSE_MOVE,
        }.get(action_type, EventType.MOTOR_EXECUTION_COMPLETE)

        events.append(SimulationEvent(
            timestamp=exec_end,
            event_type=motor_event_type,
            source_processor=self._name,
            target_processor="environment",
            payload={
                "action_type": action_type,
                "position": self._position,
                "target_element": item.get("element_id", ""),
                "motor_time": processing_time,
                "phase": "execution",
            },
            priority=EventPriority.MOTOR,
        ))

        return events


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveControlProcessor
# ═══════════════════════════════════════════════════════════════════════════

class CognitiveControlProcessor(CognitiveProcessor):
    """Simplified production-rule system inspired by ACT-R's procedural module.

    Implements a conflict-resolution / fire cycle:
    1. Match: find all productions whose conditions match the current state.
    2. Conflict resolution: select the production with highest utility.
    3. Fire: execute the selected production's action.

    Production cycle time: 50 ms (Anderson 2007, p. 40).

    Each production is a (condition, action, utility) triple:
    - condition: callable(state) -> bool
    - action: callable(state) -> list of output events
    - utility: float (learned or fixed)

    Reference:
        Anderson, J. R. (2007). Ch. 2, Production system.
    """

    # ACT-R default production cycle time
    PRODUCTION_CYCLE_TIME: float = 0.050  # 50 ms (Anderson 2007)

    @dataclass
    class Production:
        """A single production rule."""
        name: str
        condition: Callable[[Dict[str, Any]], bool]
        action: Callable[[Dict[str, Any], float], List[SimulationEvent]]
        utility: float = 1.0
        firing_count: int = 0
        last_fired: float = 0.0

    def __init__(
        self,
        name: str = "cognitive_control",
        cycle_time: float = PRODUCTION_CYCLE_TIME,
        utility_noise: float = 0.0,
        noise_sigma: float = 0.010,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, input_capacity=4, output_capacity=8,
                         noise_sigma=noise_sigma, seed=seed)
        self._cycle_time = cycle_time
        self._utility_noise = utility_noise
        self._productions: List[CognitiveControlProcessor.Production] = []
        self._goal_stack: List[Dict[str, Any]] = []
        self._current_state: Dict[str, Any] = {}

    def add_production(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        action: Callable[[Dict[str, Any], float], List[SimulationEvent]],
        utility: float = 1.0,
    ) -> None:
        """Register a production rule."""
        self._productions.append(self.Production(
            name=name, condition=condition, action=action, utility=utility,
        ))

    def push_goal(self, goal: Dict[str, Any]) -> None:
        """Push a goal onto the goal stack."""
        self._goal_stack.append(goal)

    def pop_goal(self) -> Optional[Dict[str, Any]]:
        """Pop the top goal from the stack."""
        return self._goal_stack.pop() if self._goal_stack else None

    @property
    def current_goal(self) -> Optional[Dict[str, Any]]:
        return self._goal_stack[-1] if self._goal_stack else None

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Merge updates into the current cognitive state."""
        self._current_state.update(updates)

    def _match_productions(self) -> List[Production]:
        """Find all productions whose conditions match current state."""
        state = {**self._current_state}
        if self._goal_stack:
            state["goal"] = self._goal_stack[-1]
        matched = []
        for prod in self._productions:
            try:
                if prod.condition(state):
                    matched.append(prod)
            except Exception:
                continue
        return matched

    def _resolve_conflict(self, matched: List[Production]) -> Optional[Production]:
        """Select the highest-utility production (with optional noise)."""
        if not matched:
            return None
        if self._utility_noise > 0:
            noisy_utilities = [
                (p, p.utility + self._np_rng.logistic(0, self._utility_noise))
                for p in matched
            ]
            return max(noisy_utilities, key=lambda x: x[1])[0]
        return max(matched, key=lambda p: p.utility)

    def _compute_processing_time(self, item: Dict[str, Any]) -> float:
        """Production cycle time (fixed 50 ms per ACT-R)."""
        return self._cycle_time

    def _process_item(
        self,
        item: Dict[str, Any],
        current_time: float,
        processing_time: float,
    ) -> List[SimulationEvent]:
        """Run one production cycle: match → resolve → fire."""
        # Update state from incoming event
        self._current_state.update({
            k: v for k, v in item.items()
            if not k.startswith("_") and k not in ("event_id", "timestamp")
        })

        # Match
        matched = self._match_productions()
        if not matched:
            return []

        # Resolve conflict
        selected = self._resolve_conflict(matched)
        if selected is None:
            return []

        # Fire
        selected.firing_count += 1
        selected.last_fired = current_time

        fire_time = current_time + processing_time
        output_events = selected.action(self._current_state, fire_time)

        # Emit production-fire event for tracing
        output_events.append(SimulationEvent(
            timestamp=fire_time,
            event_type=EventType.PRODUCTION_FIRE,
            source_processor=self._name,
            target_processor="trace",
            payload={
                "production_name": selected.name,
                "utility": selected.utility,
                "firing_count": selected.firing_count,
            },
            priority=EventPriority.COGNITIVE,
        ))

        return output_events


# ═══════════════════════════════════════════════════════════════════════════
# WorkingMemoryProcessor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryChunk:
    """A chunk in working memory with activation-based dynamics.

    Reference: Anderson (2007), Ch. 3 — Declarative memory.
    """
    chunk_id: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    base_level: float = 0.0
    creation_time: float = 0.0
    access_times: List[float] = field(default_factory=list)
    activation: float = 0.0
    in_focus: bool = False


class WorkingMemoryProcessor(CognitiveProcessor):
    """Working memory processor with activation-based storage and decay.

    Implements ACT-R declarative memory dynamics:
    - Base-level activation with power-law decay:  B_i = ln(Σ t_j^{-d})
    - Retrieval latency:  T = F * exp(-f * A_i)
    - Retrieval probability from activation vs threshold

    Parameters:
        capacity: Maximum chunks (Cowan 2001: ~4).
        decay_rate: Power-law decay exponent d (default 0.5; Anderson 2007).
        retrieval_threshold: Minimum activation for successful retrieval (τ).
        latency_factor: F in retrieval time equation (default 0.63 s).
        latency_exponent: f in retrieval time equation (default 1.0).
        mismatch_penalty: P in partial matching (default 1.5).

    References:
        Anderson, J. R. (2007). Ch. 3.
        Cowan, N. (2001). *Behavioral and Brain Sciences*, 24(1), 87-114.
    """

    def __init__(
        self,
        name: str = "working_memory",
        capacity: int = 4,
        decay_rate: float = 0.5,
        retrieval_threshold: float = -0.5,
        latency_factor: float = 0.630,
        latency_exponent: float = 1.0,
        mismatch_penalty: float = 1.5,
        noise_sigma: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, input_capacity=4, output_capacity=4,
                         noise_sigma=noise_sigma, seed=seed)
        self._capacity = capacity
        self._decay = decay_rate
        self._threshold = retrieval_threshold
        self._latency_f = latency_factor
        self._latency_exp = latency_exponent
        self._mismatch_penalty = mismatch_penalty
        self._chunks: Dict[str, MemoryChunk] = {}
        self._focus_set: List[str] = []

    @property
    def chunks(self) -> Dict[str, MemoryChunk]:
        return dict(self._chunks)

    @property
    def load(self) -> int:
        return len(self._chunks)

    def store_chunk(self, chunk_id: str, content: Dict[str, Any], current_time: float) -> bool:
        """Store or refresh a chunk. Returns False if at capacity and eviction fails."""
        if chunk_id in self._chunks:
            self._chunks[chunk_id].access_times.append(current_time)
            self._chunks[chunk_id].content.update(content)
            return True
        if len(self._chunks) >= self._capacity:
            # Evict lowest-activation chunk
            if not self._evict_lowest(current_time):
                return False
        chunk = MemoryChunk(
            chunk_id=chunk_id,
            content=content,
            creation_time=current_time,
            access_times=[current_time],
            base_level=0.0,
        )
        self._chunks[chunk_id] = chunk
        return True

    def _evict_lowest(self, current_time: float) -> bool:
        """Evict the chunk with lowest activation."""
        if not self._chunks:
            return False
        self._update_activations(current_time)
        lowest_id = min(self._chunks, key=lambda cid: self._chunks[cid].activation)
        del self._chunks[lowest_id]
        if lowest_id in self._focus_set:
            self._focus_set.remove(lowest_id)
        return True

    def _base_level_activation(self, chunk: MemoryChunk, current_time: float) -> float:
        """Compute base-level activation: B_i = ln(Σ t_j^{-d}).

        Reference: Anderson (2007), Eq. 3.2.
        """
        if not chunk.access_times:
            return -float("inf")
        total = 0.0
        for t_j in chunk.access_times:
            age = current_time - t_j
            if age > 0:
                total += age ** (-self._decay)
        if total <= 0:
            return -10.0
        return math.log(total)

    def _update_activations(self, current_time: float) -> None:
        """Recompute activations for all chunks."""
        for chunk in self._chunks.values():
            chunk.base_level = self._base_level_activation(chunk, current_time)
            noise = self._np_rng.logistic(0, self._noise_sigma) if self._noise_sigma > 0 else 0.0
            chunk.activation = chunk.base_level + noise

    def retrieve(self, probe: Dict[str, Any], current_time: float) -> Tuple[Optional[MemoryChunk], float]:
        """Attempt retrieval; return (chunk, latency) or (None, failure_latency).

        Retrieval time: T = F * exp(-f * A_i)  (Anderson 2007, Eq. 3.4).
        """
        self._update_activations(current_time)

        best_chunk: Optional[MemoryChunk] = None
        best_activation = -float("inf")

        for chunk in self._chunks.values():
            match_score = self._partial_match_score(chunk, probe)
            effective_activation = chunk.activation + match_score
            if effective_activation > best_activation:
                best_activation = effective_activation
                best_chunk = chunk

        if best_chunk is None or best_activation < self._threshold:
            failure_latency = self._latency_f * math.exp(-self._latency_exp * self._threshold)
            return None, failure_latency

        latency = self._latency_f * math.exp(-self._latency_exp * best_activation)
        best_chunk.access_times.append(current_time)
        return best_chunk, latency

    def _partial_match_score(self, chunk: MemoryChunk, probe: Dict[str, Any]) -> float:
        """Partial matching penalty: Σ P * Sim(probe_j, chunk_j).

        Reference: Anderson (2007), Eq. 3.5.
        """
        if not probe:
            return 0.0
        penalty = 0.0
        for key, probe_val in probe.items():
            chunk_val = chunk.content.get(key)
            if chunk_val is None:
                penalty -= self._mismatch_penalty
            elif chunk_val != probe_val:
                penalty -= self._mismatch_penalty * 0.5
        return penalty

    def _compute_processing_time(self, item: Dict[str, Any]) -> float:
        """Memory operations: store ~ 50ms, retrieve ~ variable."""
        op = item.get("operation", "store")
        if op == "retrieve":
            return 0.100  # initial estimate; actual via retrieve()
        return 0.050  # store operation

    def _process_item(
        self,
        item: Dict[str, Any],
        current_time: float,
        processing_time: float,
    ) -> List[SimulationEvent]:
        """Handle store or retrieve operations."""
        op = item.get("operation", "store")

        if op == "store":
            chunk_id = item.get("chunk_id", f"chunk_{self._items_processed}")
            content = item.get("content", {})
            self.store_chunk(chunk_id, content, current_time)
            return [SimulationEvent(
                timestamp=current_time + processing_time,
                event_type=EventType.WM_STORE,
                source_processor=self._name,
                target_processor="cognitive_control",
                payload={"chunk_id": chunk_id, "load": self.load},
                priority=EventPriority.MEMORY,
            )]

        if op == "retrieve":
            probe = item.get("probe", {})
            chunk, latency = self.retrieve(probe, current_time)
            if chunk is not None:
                return [SimulationEvent(
                    timestamp=current_time + latency,
                    event_type=EventType.RETRIEVAL_COMPLETE,
                    source_processor=self._name,
                    target_processor="cognitive_control",
                    payload={"chunk_id": chunk.chunk_id, "content": chunk.content,
                             "activation": chunk.activation, "latency": latency},
                    priority=EventPriority.MEMORY,
                )]
            else:
                return [SimulationEvent(
                    timestamp=current_time + latency,
                    event_type=EventType.RETRIEVAL_FAILURE,
                    source_processor=self._name,
                    target_processor="cognitive_control",
                    payload={"probe": probe, "latency": latency},
                    priority=EventPriority.MEMORY,
                )]

        return []


# ═══════════════════════════════════════════════════════════════════════════
# VisualAttentionProcessor
# ═══════════════════════════════════════════════════════════════════════════

class VisualAttentionProcessor(CognitiveProcessor):
    """Visual attention with spotlight model and shift cost.

    Models covert attention shifts and overt eye movements.

    Attention shift time:
        T_shift = base_shift_time + distance_factor * eccentricity

    where base_shift_time is ~50 ms for covert and ~200 ms for overt
    (saccade) shifts.

    Visual search:
        Efficient (pop-out): 5-15 ms/item (Treisman & Gelade 1980)
        Inefficient (conjunction): 20-50 ms/item

    References:
        Posner, M. I. (1980). Orienting of attention. *Quarterly Journal
            of Experimental Psychology*, 32(1), 3-25.
        Treisman, A., & Gelade, G. (1980). A feature integration theory
            of attention. *Cognitive Psychology*, 12(1), 97-136.
        Wolfe, J. M. (1994). Guided Search 2.0. *Psychonomic Bulletin
            & Review*, 1(2), 202-238.
    """

    # Published parameter defaults
    COVERT_SHIFT_TIME: float = 0.050   # 50 ms (Posner 1980)
    SACCADE_TIME: float = 0.200        # 200 ms typical saccade (Rayner 1998)
    EFFICIENT_SEARCH_SLOPE: float = 0.010   # 10 ms/item (Treisman & Gelade 1980)
    INEFFICIENT_SEARCH_SLOPE: float = 0.035  # 35 ms/item (Wolfe 1994)

    def __init__(
        self,
        name: str = "visual_attention",
        covert_shift_time: float = COVERT_SHIFT_TIME,
        saccade_time: float = SACCADE_TIME,
        efficient_slope: float = EFFICIENT_SEARCH_SLOPE,
        inefficient_slope: float = INEFFICIENT_SEARCH_SLOPE,
        spotlight_radius: float = 50.0,
        pixels_per_degree: float = 38.0,
        noise_sigma: float = 0.012,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, input_capacity=2, output_capacity=4,
                         noise_sigma=noise_sigma, seed=seed)
        self._covert_shift = covert_shift_time
        self._saccade_time = saccade_time
        self._efficient_slope = efficient_slope
        self._inefficient_slope = inefficient_slope
        self._spotlight_radius = spotlight_radius
        self._ppd = pixels_per_degree
        self._focus_position: Tuple[float, float] = (512.0, 384.0)
        self._attended_elements: List[str] = []

    @property
    def focus_position(self) -> Tuple[float, float]:
        return self._focus_position

    def visual_search_time(self, n_items: int, efficient: bool = False) -> float:
        """Predict visual search time for n items.

        Efficient search: RT = intercept + efficient_slope * n
        Inefficient search: RT = intercept + inefficient_slope * n

        Reference: Wolfe (1994).
        """
        slope = self._efficient_slope if efficient else self._inefficient_slope
        intercept = 0.200  # ~200 ms base RT
        return intercept + slope * n_items

    def _shift_time(self, target_x: float, target_y: float) -> float:
        """Time to shift attention to target location."""
        dx = target_x - self._focus_position[0]
        dy = target_y - self._focus_position[1]
        distance_px = math.sqrt(dx * dx + dy * dy)

        if distance_px < self._spotlight_radius:
            return self._covert_shift
        distance_deg = distance_px / self._ppd
        return self._saccade_time + 0.002 * distance_deg

    def _compute_processing_time(self, item: Dict[str, Any]) -> float:
        """Attention shift time + any visual search."""
        target_x = item.get("target_x", self._focus_position[0])
        target_y = item.get("target_y", self._focus_position[1])
        shift = self._shift_time(target_x, target_y)

        n_distractors = item.get("n_distractors", 0)
        if n_distractors > 0:
            efficient = item.get("efficient_search", False)
            search = self.visual_search_time(n_distractors, efficient)
            return shift + search

        return shift

    def _process_item(
        self,
        item: Dict[str, Any],
        current_time: float,
        processing_time: float,
    ) -> List[SimulationEvent]:
        """Shift attention and emit encoding request."""
        target_x = item.get("target_x", self._focus_position[0])
        target_y = item.get("target_y", self._focus_position[1])
        self._focus_position = (target_x, target_y)

        element_id = item.get("element_id", "")
        if element_id:
            self._attended_elements.append(element_id)

        events: List[SimulationEvent] = []

        # Attention shift complete
        events.append(SimulationEvent(
            timestamp=current_time + processing_time,
            event_type=EventType.VISUAL_ATTENTION_SHIFT,
            source_processor=self._name,
            target_processor="perceptual",
            payload={
                "element_id": element_id,
                "target_x": target_x,
                "target_y": target_y,
                "shift_time": processing_time,
            },
            priority=EventPriority.PERCEPTUAL,
        ))

        # Request perceptual encoding of attended element
        events.append(SimulationEvent(
            timestamp=current_time + processing_time,
            event_type=EventType.VISUAL_ONSET,
            source_processor=self._name,
            target_processor="perceptual",
            payload={
                "element_id": element_id,
                "element_name": item.get("element_name", ""),
                "element_role": item.get("element_role", ""),
                "target_x": target_x,
                "target_y": target_y,
            },
            priority=EventPriority.PERCEPTUAL,
        ))

        return events
