"""ACT-R integration with the bounded-rational usability oracle.

Bridges the ACT-R cognitive architecture modules (declarative memory,
production system, visual module, motor module) with the oracle's
accessibility-tree analysis, MDP formulation, and regression detection
pipeline.

References
----------
Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical
    Universe?* Oxford University Press.
Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C.,
    & Qin, Y. (2004). An integrated theory of the mind. *Psychological
    Review*, 111(4), 1036-1060.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.cognitive.actr_memory import (
    ACTRDeclarativeMemory,
    Chunk,
)
from usability_oracle.cognitive.actr_production import (
    ACTRProductionSystem,
    BufferState,
    Production,
)
from usability_oracle.cognitive.actr_visual import (
    ACTRVisualModule,
    EMMAParams,
    VisualObject,
)
from usability_oracle.cognitive.actr_motor import (
    ACTRMotorModule,
    Hand,
)
from usability_oracle.cognitive.models import BoundingBox, Point2D
from usability_oracle.cognitive.parameters import CognitiveParameters


# ---------------------------------------------------------------------------
# Simulation trace
# ---------------------------------------------------------------------------


@dataclass
class ACTRTraceEntry:
    """A single entry in an ACT-R simulation trace.

    Attributes
    ----------
    time : float
        Simulation time (seconds).
    module : str
        Module that generated this event.
    action : str
        Description of the action.
    duration : float
        Duration of this step (seconds).
    details : dict
        Additional details.
    """

    time: float
    module: str
    action: str
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveCostMetrics:
    """Cognitive cost metrics extracted from an ACT-R trace.

    Attributes
    ----------
    total_time : float
        Total task completion time (seconds).
    cognitive_time : float
        Time spent on cognitive operations (seconds).
    perceptual_time : float
        Time in visual encoding and attention (seconds).
    motor_time : float
        Time in motor actions (seconds).
    memory_retrievals : int
        Number of memory retrieval attempts.
    retrieval_failures : int
        Number of failed retrievals.
    productions_fired : int
        Number of productions fired.
    fixation_count : int
        Number of visual fixations.
    mean_retrieval_time : float
        Average retrieval latency (seconds).
    mean_motor_time : float
        Average motor action time (seconds).
    """

    total_time: float = 0.0
    cognitive_time: float = 0.0
    perceptual_time: float = 0.0
    motor_time: float = 0.0
    memory_retrievals: int = 0
    retrieval_failures: int = 0
    productions_fired: int = 0
    fixation_count: int = 0
    mean_retrieval_time: float = 0.0
    mean_motor_time: float = 0.0


# ---------------------------------------------------------------------------
# ACT-R Model (integrated)
# ---------------------------------------------------------------------------


class ACTRModel:
    """Integrated ACT-R model coupling all modules.

    This is the main interface for building an ACT-R model from an
    accessibility tree and task specification, running simulations, and
    extracting cognitive cost metrics.

    Parameters
    ----------
    dm_params : dict, optional
        Keyword arguments for :class:`ACTRDeclarativeMemory`.
    ps_params : dict, optional
        Keyword arguments for :class:`ACTRProductionSystem`.
    visual_params : EMMAParams, optional
        Visual module parameters.
    motor_params : dict, optional
        Keyword arguments for :class:`ACTRMotorModule`.
    """

    def __init__(
        self,
        dm_params: Optional[Dict[str, Any]] = None,
        ps_params: Optional[Dict[str, Any]] = None,
        visual_params: Optional[EMMAParams] = None,
        motor_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dm = ACTRDeclarativeMemory(**(dm_params or {}))
        self.ps = ACTRProductionSystem(**(ps_params or {}))
        self.visual = ACTRVisualModule(params=visual_params)
        self.motor = ACTRMotorModule(**(motor_params or {}))

        self._trace: List[ACTRTraceEntry] = []
        self._clock: float = 0.0
        self._state = BufferState()

    @property
    def clock(self) -> float:
        """Current simulation time."""
        return self._clock

    @property
    def trace(self) -> List[ACTRTraceEntry]:
        """Full simulation trace."""
        return list(self._trace)

    @property
    def state(self) -> BufferState:
        """Current buffer state."""
        return self._state

    # ------------------------------------------------------------------ #
    # Build model from accessibility tree
    # ------------------------------------------------------------------ #

    def build_from_accessibility_tree(
        self,
        nodes: Sequence[Dict[str, Any]],
        task_steps: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the model from accessibility tree nodes.

        Creates visual objects in the visual module and declarative
        memory chunks for each UI element.  Optionally loads task steps
        as goal-related productions.

        Parameters
        ----------
        nodes : sequence of dict
            Accessibility tree nodes with ``name``, ``role``,
            ``bounds``, etc.
        task_steps : sequence of dict, optional
            Task steps, each with ``action``, ``target``, and optional
            ``description``.
        """
        # Populate visicon
        self.visual.from_accessibility_tree(nodes, self._clock)

        # Create declarative memory chunks
        for node in nodes:
            chunk = Chunk(
                name=str(node.get("name", "")),
                chunk_type=str(node.get("role", "element")),
                slots={
                    "role": node.get("role", ""),
                    "label": node.get("label", ""),
                    "x": node.get("bounds", {}).get("x", 0),
                    "y": node.get("bounds", {}).get("y", 0),
                },
                creation_time=self._clock,
            )
            self.dm.merge_chunk(chunk, self._clock)

        # Create task productions
        if task_steps:
            self._create_task_productions(task_steps)

    def _create_task_productions(
        self,
        task_steps: Sequence[Dict[str, Any]],
    ) -> None:
        """Generate productions for a sequence of task steps."""
        for i, step in enumerate(task_steps):
            conditions: Dict[str, Dict[str, Any]] = {
                "goal": {"step": i},
            }
            if step.get("target"):
                conditions["visual"] = {"label": step["target"]}

            actions: Dict[str, Dict[str, Any]] = {
                "goal": {"step": i + 1},
            }
            action_type = step.get("action", "click")
            if action_type in ("click", "tap"):
                actions["motor"] = {"command": action_type}

            prod = Production(
                name=f"step-{i}-{action_type}",
                conditions=conditions,
                actions=actions,
                utility=0.0,
                cost=self.ps.production_firing_time,
                creation_time=self._clock,
            )
            self.ps.add_production(prod)

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        task_steps: Sequence[Dict[str, Any]],
        max_time: float = 60.0,
    ) -> CognitiveCostMetrics:
        """Run an ACT-R simulation for a task and return cost metrics.

        Processes each task step through the visual → cognitive → motor
        pipeline, accumulating time and trace entries.

        Parameters
        ----------
        task_steps : sequence of dict
            Task steps with ``action``, ``target``, and ``bounds``.
        max_time : float
            Maximum simulation time (seconds).

        Returns
        -------
        CognitiveCostMetrics
            Extracted cost metrics.
        """
        metrics = CognitiveCostMetrics()
        self._state.set("goal", {"step": 0})

        for i, step in enumerate(task_steps):
            if self._clock >= max_time:
                break

            # --- Visual: find and attend target ---
            target_label = step.get("target", "")
            visual_time = 0.0

            if target_label:
                result = self.visual.guided_search(
                    {"label": target_label}
                )
                found, search_time, n_fix = result
                visual_time += search_time
                metrics.fixation_count += n_fix
            else:
                visual_time += 0.250  # default fixation

            self._log("visual", f"attend-{target_label}", visual_time)
            metrics.perceptual_time += visual_time
            self._clock += visual_time

            # --- Cognitive: retrieve relevant chunk ---
            request = {"label": target_label, "role": step.get("role", "")}
            chunk, rt = self.dm.retrieve(request, self._clock, partial=True)
            metrics.memory_retrievals += 1
            if chunk is None:
                metrics.retrieval_failures += 1
            metrics.mean_retrieval_time = (
                (metrics.mean_retrieval_time * (metrics.memory_retrievals - 1)
                 + rt) / metrics.memory_retrievals
            )

            self._log("declarative", f"retrieve-{target_label}", rt)
            metrics.cognitive_time += rt
            self._clock += rt

            # --- Production fire ---
            new_state, prod, prod_time = self.ps.step(self._state)
            if prod is not None:
                self._state = new_state
                metrics.productions_fired += 1
                self._log("procedural", f"fire-{prod.name}", prod_time)
            else:
                prod_time = self.ps.production_firing_time
                self._log("procedural", "no-match", prod_time)
            metrics.cognitive_time += prod_time
            self._clock += prod_time

            # --- Motor action ---
            action = step.get("action", "click")
            bounds = step.get("bounds", {})
            target_pt = Point2D(
                float(bounds.get("x", 0)) + float(bounds.get("width", 10)) / 2,
                float(bounds.get("y", 0)) + float(bounds.get("height", 10)) / 2,
            )
            width = float(bounds.get("width", 10))

            if action in ("click", "tap"):
                motor_time = self.motor.click(target_pt, width)
            elif action == "type":
                motor_time = self.motor.typing_time(
                    step.get("text", ""), skill_level=1.0
                )
            else:
                motor_time = self.motor.click(target_pt, width)

            self._log("motor", f"{action}-{target_label}", motor_time)
            metrics.motor_time += motor_time
            if metrics.productions_fired > 0:
                metrics.mean_motor_time = (
                    metrics.motor_time / metrics.productions_fired
                )
            self._clock += motor_time

            self._state.set("goal", {"step": i + 1})

        metrics.total_time = self._clock
        return metrics

    def _log(self, module: str, action: str, duration: float) -> None:
        """Append a trace entry."""
        self._trace.append(
            ACTRTraceEntry(
                time=self._clock,
                module=module,
                action=action,
                duration=duration,
            )
        )

    # ------------------------------------------------------------------ #
    # Cost metric extraction
    # ------------------------------------------------------------------ #

    def extract_metrics(self) -> CognitiveCostMetrics:
        """Extract cost metrics from the current trace.

        Returns
        -------
        CognitiveCostMetrics
            Aggregated metrics.
        """
        metrics = CognitiveCostMetrics()
        retrieval_times: List[float] = []
        motor_times: List[float] = []

        for entry in self._trace:
            if entry.module == "visual":
                metrics.perceptual_time += entry.duration
                metrics.fixation_count += 1
            elif entry.module == "declarative":
                metrics.cognitive_time += entry.duration
                metrics.memory_retrievals += 1
                retrieval_times.append(entry.duration)
            elif entry.module == "procedural":
                metrics.cognitive_time += entry.duration
                metrics.productions_fired += 1
            elif entry.module == "motor":
                metrics.motor_time += entry.duration
                motor_times.append(entry.duration)

        metrics.total_time = (
            metrics.cognitive_time
            + metrics.perceptual_time
            + metrics.motor_time
        )
        if retrieval_times:
            metrics.mean_retrieval_time = float(np.mean(retrieval_times))
        if motor_times:
            metrics.mean_motor_time = float(np.mean(motor_times))

        return metrics

    # ------------------------------------------------------------------ #
    # Comparison with simplified cognitive models
    # ------------------------------------------------------------------ #

    @staticmethod
    def compare_predictions(
        actr_time: float,
        simplified_time: float,
    ) -> Dict[str, float]:
        """Compare ACT-R prediction with a simplified model prediction.

        Parameters
        ----------
        actr_time : float
            Task time predicted by ACT-R simulation.
        simplified_time : float
            Task time from a simplified model (e.g. KLM).

        Returns
        -------
        dict[str, float]
            ``"absolute_error"``, ``"relative_error"``, ``"ratio"``.
        """
        abs_err = abs(actr_time - simplified_time)
        denom = max(simplified_time, 0.001)
        return {
            "absolute_error": abs_err,
            "relative_error": abs_err / denom,
            "ratio": actr_time / denom,
        }

    # ------------------------------------------------------------------ #
    # Parameter calibration from CognitiveParameters
    # ------------------------------------------------------------------ #

    @staticmethod
    def calibrate_from_parameters(
        params: CognitiveParameters,
        percentile: float = 50.0,
    ) -> Dict[str, Any]:
        """Map CognitiveParameters to ACT-R module parameters.

        Parameters
        ----------
        params : CognitiveParameters
            Oracle cognitive parameters.
        percentile : float
            Population percentile.

        Returns
        -------
        dict[str, Any]
            ACT-R parameter dict with keys ``"dm"``, ``"ps"``,
            ``"visual"``, ``"motor"``.
        """
        pset = params.get_parameter_set(percentile)

        # Map keystroke time to typing rate
        typing_rate = pset.get("keystroke", 0.200)
        fitts_b = pset.get("fitts_b", 0.150)
        mental_prep = pset.get("mental_prep", 1.200)

        # Scale ACT-R latency factor by mental preparation time
        latency_factor = mental_prep / 1.2  # normalised

        return {
            "dm": {
                "decay": 0.5,
                "latency_factor": latency_factor,
                "noise_s": 0.25,
            },
            "ps": {
                "utility_noise_s": 0.25,
                "production_firing_time": 0.050,
            },
            "visual": {
                "encoding_factor": 0.006 * latency_factor,
            },
            "motor": {
                "fitts_b": fitts_b,
                "typing_rate": typing_rate,
            },
        }

    # ------------------------------------------------------------------ #
    # Regression detection via ACT-R comparison
    # ------------------------------------------------------------------ #

    def detect_regression(
        self,
        baseline_metrics: CognitiveCostMetrics,
        current_metrics: CognitiveCostMetrics,
        threshold: float = 0.10,
    ) -> Dict[str, Any]:
        """Detect usability regressions by comparing ACT-R metrics.

        A regression is flagged when any cost metric increases by more
        than *threshold* (proportion).

        Parameters
        ----------
        baseline_metrics : CognitiveCostMetrics
            Metrics from the baseline UI.
        current_metrics : CognitiveCostMetrics
            Metrics from the current UI.
        threshold : float
            Proportional increase threshold (default 10 %).

        Returns
        -------
        dict[str, Any]
            ``"is_regression"`` (bool), ``"regressions"`` (list of
            field names), ``"details"`` (per-field comparison).
        """
        fields = [
            "total_time", "cognitive_time", "perceptual_time",
            "motor_time", "retrieval_failures",
        ]
        regressions: List[str] = []
        details: Dict[str, Dict[str, float]] = {}

        for f in fields:
            baseline_val = float(getattr(baseline_metrics, f))
            current_val = float(getattr(current_metrics, f))
            denom = max(baseline_val, 0.001)
            change = (current_val - baseline_val) / denom

            details[f] = {
                "baseline": baseline_val,
                "current": current_val,
                "change": change,
            }
            if change > threshold:
                regressions.append(f)

        return {
            "is_regression": len(regressions) > 0,
            "regressions": regressions,
            "details": details,
        }

    # ------------------------------------------------------------------ #
    # Bridge ACT-R chunks to MDP states
    # ------------------------------------------------------------------ #

    def chunks_to_mdp_states(self) -> List[Dict[str, Any]]:
        """Convert declarative memory chunks to MDP state descriptions.

        Each chunk becomes an MDP state with features derived from its
        slots.  This bridges the ACT-R representation with the oracle's
        MDP-based analysis.

        Returns
        -------
        list[dict[str, Any]]
            MDP state descriptions.
        """
        states: List[Dict[str, Any]] = []
        for chunk in self.dm.chunks:
            state: Dict[str, Any] = {
                "name": chunk.name,
                "type": chunk.chunk_type,
                "features": dict(chunk.slots),
                "activation": self.dm.base_level_activation(
                    chunk, self._clock
                ),
                "retrieval_prob": self.dm.retrieval_probability(
                    self.dm.base_level_activation(chunk, self._clock)
                ),
            }
            states.append(state)
        return states

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the model to its initial state."""
        self._trace.clear()
        self._clock = 0.0
        self._state = BufferState()
        self.dm = ACTRDeclarativeMemory(
            decay=self.dm.decay,
            latency_factor=self.dm.latency_factor,
            noise_s=self.dm.noise_s,
        )
        self.ps = ACTRProductionSystem(
            utility_noise_s=self.ps.utility_noise_s,
            alpha=self.ps.alpha,
            production_firing_time=self.ps.production_firing_time,
        )
        self.visual = ACTRVisualModule(params=self.visual.params)
        self.motor = ACTRMotorModule(
            fitts_a=self.motor.fitts_a,
            fitts_b=self.motor.fitts_b,
        )
