"""
usability_oracle.pipeline.stages — Stage executor protocol and registry.

Each pipeline stage is represented by a :class:`StageExecutor` that wraps
the corresponding module's logic with standardised error handling, logging,
and timing instrumentation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from usability_oracle.core.enums import PipelineStage
from usability_oracle.core.errors import StageError
from usability_oracle.pipeline.config import StageConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StageExecutor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StageExecutor(Protocol):
    """Protocol for pipeline stage executors."""

    @property
    def stage(self) -> PipelineStage: ...

    def execute(self, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Base implementation
# ---------------------------------------------------------------------------

class BaseStageExecutor:
    """Base class for stage executors with timing and error handling."""

    _stage: PipelineStage

    def __init__(self, config: StageConfig | None = None) -> None:
        self.config = config or StageConfig()
        self._last_timing: float = 0.0

    @property
    def stage(self) -> PipelineStage:
        return self._stage

    def execute(self, **kwargs: Any) -> Any:
        """Execute the stage with timing and error handling."""
        if not self.config.enabled:
            logger.info("Stage %s is disabled, skipping", self._stage.value)
            return None

        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.config.retry:
            try:
                t0 = time.monotonic()
                result = self._run(**kwargs)
                self._last_timing = time.monotonic() - t0
                logger.info(
                    "Stage %s completed in %.3fs",
                    self._stage.value, self._last_timing,
                )
                return result
            except Exception as exc:
                last_error = exc
                attempt += 1
                if attempt <= self.config.retry:
                    logger.warning(
                        "Stage %s attempt %d failed: %s; retrying",
                        self._stage.value, attempt, exc,
                    )
                else:
                    logger.error(
                        "Stage %s failed after %d attempts: %s",
                        self._stage.value, attempt, exc,
                    )

        raise StageError(
            f"Stage {self._stage.value} failed: {last_error}"
        ) from last_error

    def _run(self, **kwargs: Any) -> Any:
        """Override in subclasses to implement stage logic."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete stage executors
# ---------------------------------------------------------------------------

class ParseStageExecutor(BaseStageExecutor):
    """Parse HTML/JSON into an AccessibilityTree."""

    _stage = PipelineStage.PARSE

    def _run(self, **kwargs: Any) -> Any:
        source = kwargs.get("source")
        config = kwargs.get("parser_config")

        if source is None:
            raise StageError("Parse stage requires 'source' input")

        from usability_oracle.accessibility import (
            HTMLAccessibilityParser,
            JSONAccessibilityParser,
        )

        source_str = str(source)
        if source_str.strip().startswith("{") or source_str.strip().startswith("["):
            parser = JSONAccessibilityParser()
        else:
            parser = HTMLAccessibilityParser()

        return parser.parse(source_str)


class AlignStageExecutor(BaseStageExecutor):
    """Align two accessibility trees."""

    _stage = PipelineStage.ALIGN

    def _run(self, **kwargs: Any) -> Any:
        tree_a = kwargs.get("tree_a")
        tree_b = kwargs.get("tree_b")

        if tree_a is None or tree_b is None:
            raise StageError("Align stage requires 'tree_a' and 'tree_b'")

        # Alignment module integration point
        # Returns a simple dict-based alignment result
        from usability_oracle.accessibility.models import AccessibilityTree

        matched_nodes: list[tuple[str, str]] = []
        for nid_a in tree_a.node_index:
            node_a = tree_a.get_node(nid_a)
            # Match by role + name
            for nid_b in tree_b.node_index:
                node_b = tree_b.get_node(nid_b)
                if (node_a and node_b
                        and node_a.role == node_b.role
                        and node_a.name == node_b.name):
                    matched_nodes.append((nid_a, nid_b))
                    break

        return {
            "matched": matched_nodes,
            "unmatched_a": [
                nid for nid in tree_a.node_index
                if nid not in {m[0] for m in matched_nodes}
            ],
            "unmatched_b": [
                nid for nid in tree_b.node_index
                if nid not in {m[1] for m in matched_nodes}
            ],
            "similarity": len(matched_nodes) / max(
                1, len(tree_a.node_index), len(tree_b.node_index)
            ),
        }


class CostStageExecutor(BaseStageExecutor):
    """Compute cognitive costs for tree nodes."""

    _stage = PipelineStage.COST

    def _run(self, **kwargs: Any) -> Any:
        tree = kwargs.get("tree")
        task_spec = kwargs.get("task_spec")

        if tree is None:
            raise StageError("Cost stage requires 'tree'")

        import math
        from usability_oracle.core.config import CognitiveConfig

        config = kwargs.get("cognitive_config", CognitiveConfig())
        costs: dict[str, dict[str, float]] = {}

        interactive = tree.get_interactive_nodes()
        n = max(1, len(interactive))
        hick_cost = config.hick_a + config.hick_b * math.log2(n + 1)

        for node in interactive:
            node_cost: dict[str, float] = {}
            node_cost["hick"] = hick_cost

            if node.bounding_box and node.bounding_box.width > 0:
                d = 200.0  # assumed average distance
                w = node.bounding_box.width
                fitts_id = math.log2(1 + d / w)
                node_cost["fitts"] = config.fitts_a + config.fitts_b * fitts_id
            else:
                node_cost["fitts"] = 0.5

            node_cost["total"] = node_cost["hick"] + node_cost["fitts"]
            costs[node.id] = node_cost

        return {"node_costs": costs, "n_interactive": n}


class MDPStageExecutor(BaseStageExecutor):
    """Build MDP from accessibility tree and task spec."""

    _stage = PipelineStage.MDP_BUILD

    def _run(self, **kwargs: Any) -> Any:
        tree = kwargs.get("tree")
        task_spec = kwargs.get("task_spec")

        if tree is None:
            raise StageError("MDP stage requires 'tree'")

        from usability_oracle.mdp.models import MDP, State, Action, Transition

        states: dict[str, State] = {}
        actions: dict[str, Action] = {}
        transitions: list[Transition] = []

        interactive = tree.get_interactive_nodes()
        for node in interactive:
            sid = f"s_{node.id}"
            states[sid] = State(
                state_id=sid,
                label=f"{node.role}:{node.name}",
                metadata={"node_id": node.id},
            )

        # Add initial and goal states
        if interactive:
            initial_id = f"s_{interactive[0].id}"
            goal_ids = set()
            if len(interactive) > 1:
                goal_ids.add(f"s_{interactive[-1].id}")

            # Create actions and transitions between consecutive nodes
            for i in range(len(interactive) - 1):
                src = f"s_{interactive[i].id}"
                dst = f"s_{interactive[i + 1].id}"
                aid = f"a_{interactive[i].id}_to_{interactive[i + 1].id}"

                actions[aid] = Action(
                    action_id=aid,
                    action_type="click",
                    target_node_id=interactive[i + 1].id,
                )
                transitions.append(Transition(
                    source=src, action=aid, target=dst,
                    probability=1.0, cost=0.3,
                ))

            return MDP(
                states=states,
                actions=actions,
                transitions=transitions,
                initial_state=initial_id,
                goal_states=goal_ids,
            )

        return MDP()


class BisimulationStageExecutor(BaseStageExecutor):
    """Compute bisimulation quotient of MDP."""

    _stage = PipelineStage.BISIMULATE

    def _run(self, **kwargs: Any) -> Any:
        mdp = kwargs.get("mdp")
        if mdp is None:
            raise StageError("Bisimulation stage requires 'mdp'")

        # Simplified partition: group states by features
        partitions: dict[str, list[str]] = {}
        for sid, state in mdp.states.items():
            key = state.label.split(":")[0] if state.label else "unknown"
            if key not in partitions:
                partitions[key] = []
            partitions[key].append(sid)

        return {
            "partitions": partitions,
            "n_partitions": len(partitions),
            "quotient_mdp": mdp,  # pass through for now
        }


class PolicyStageExecutor(BaseStageExecutor):
    """Compute optimal / bounded-rational policy."""

    _stage = PipelineStage.POLICY

    def _run(self, **kwargs: Any) -> Any:
        mdp = kwargs.get("mdp")
        beta = kwargs.get("beta", 1.0)

        if mdp is None:
            raise StageError("Policy stage requires 'mdp'")

        import math

        policy: dict[str, dict[str, float]] = {}
        for sid in mdp.states:
            available = mdp.get_actions(sid)
            if not available:
                continue

            # Softmax (bounded-rational) policy
            costs: dict[str, float] = {}
            for aid in available:
                transitions = mdp.get_transitions(sid, aid)
                avg_cost = sum(c for _, _, c in transitions) / max(1, len(transitions))
                costs[aid] = avg_cost

            if costs:
                min_cost = min(costs.values())
                exp_vals = {
                    aid: math.exp(-beta * (c - min_cost))
                    for aid, c in costs.items()
                }
                total = sum(exp_vals.values())
                policy[sid] = {
                    aid: ev / total for aid, ev in exp_vals.items()
                }

        return {"policy": policy, "beta": beta, "n_states": len(policy)}


class ComparisonStageExecutor(BaseStageExecutor):
    """Compare two policies / cost annotations."""

    _stage = PipelineStage.COMPARE

    def _run(self, **kwargs: Any) -> Any:
        policy_a = kwargs.get("policy_a")
        policy_b = kwargs.get("policy_b")

        result = {
            "verdict": "no_change",
            "details": {},
        }

        if policy_a is None or policy_b is None:
            return result

        pa = policy_a.get("policy", {}) if isinstance(policy_a, dict) else {}
        pb = policy_b.get("policy", {}) if isinstance(policy_b, dict) else {}

        common_states = set(pa.keys()) & set(pb.keys())
        if not common_states:
            result["verdict"] = "inconclusive"
            return result

        total_diff = 0.0
        for sid in common_states:
            for aid in set(pa[sid].keys()) | set(pb[sid].keys()):
                diff = abs(pa[sid].get(aid, 0) - pb[sid].get(aid, 0))
                total_diff += diff

        avg_diff = total_diff / max(1, len(common_states))

        if avg_diff > 0.1:
            result["verdict"] = "regression"
        elif avg_diff < -0.05:
            result["verdict"] = "improvement"

        result["details"] = {
            "common_states": len(common_states),
            "avg_policy_diff": avg_diff,
        }
        return result


class BottleneckStageExecutor(BaseStageExecutor):
    """Detect cognitive bottlenecks."""

    _stage = PipelineStage.BOTTLENECK

    def _run(self, **kwargs: Any) -> Any:
        mdp = kwargs.get("mdp")
        policy = kwargs.get("policy")

        if mdp is None:
            raise StageError("Bottleneck stage requires 'mdp'")

        bottlenecks: list[dict[str, Any]] = []

        for sid, state in mdp.states.items():
            actions = mdp.get_actions(sid)
            if len(actions) > 7:
                bottlenecks.append({
                    "bottleneck_type": "choice_paralysis",
                    "state_id": sid,
                    "severity": "high",
                    "cost_contribution": 0.5,
                    "node_ids": [state.node_id],
                    "description": f"State {sid} has {len(actions)} choices",
                })

            if state.working_memory_load > 4:
                bottlenecks.append({
                    "bottleneck_type": "memory_decay",
                    "state_id": sid,
                    "severity": "medium",
                    "cost_contribution": 0.3,
                    "node_ids": [state.node_id],
                    "description": f"State {sid} requires {state.working_memory_load} memory chunks",
                })

        return bottlenecks


class RepairStageExecutor(BaseStageExecutor):
    """Synthesise repairs for detected bottlenecks."""

    _stage = PipelineStage.REPAIR

    def _run(self, **kwargs: Any) -> Any:
        mdp = kwargs.get("mdp")
        bottlenecks = kwargs.get("bottlenecks", [])

        if mdp is None:
            raise StageError("Repair stage requires 'mdp'")

        from usability_oracle.repair.synthesizer import RepairSynthesizer

        synthesizer = RepairSynthesizer()
        return synthesizer.synthesize(mdp, bottlenecks)


class OutputStageExecutor(BaseStageExecutor):
    """Format and output results."""

    _stage = PipelineStage.OUTPUT

    def _run(self, **kwargs: Any) -> Any:
        result = kwargs.get("result")
        output_format = kwargs.get("output_format", "json")

        if result is None:
            return {"formatted": "No results"}

        import json
        if output_format == "json":
            if hasattr(result, "to_dict"):
                return {"formatted": json.dumps(result.to_dict(), indent=2)}
            return {"formatted": json.dumps(result, indent=2, default=str)}

        return {"formatted": str(result)}


# ---------------------------------------------------------------------------
# Stage Registry
# ---------------------------------------------------------------------------

class StageRegistry:
    """Registry mapping PipelineStage → StageExecutor instances.

    Usage::

        registry = StageRegistry.default()
        executor = registry.get(PipelineStage.PARSE)
        result = executor.execute(source=html_content)
    """

    def __init__(self) -> None:
        self._executors: dict[PipelineStage, BaseStageExecutor] = {}

    def register(
        self, stage: PipelineStage, executor: BaseStageExecutor
    ) -> None:
        self._executors[stage] = executor

    def get(self, stage: PipelineStage) -> BaseStageExecutor:
        if stage not in self._executors:
            raise KeyError(f"No executor registered for stage {stage.value}")
        return self._executors[stage]

    def has(self, stage: PipelineStage) -> bool:
        return stage in self._executors

    @property
    def stages(self) -> list[PipelineStage]:
        return list(self._executors.keys())

    @classmethod
    def default(cls, stage_configs: dict[str, StageConfig] | None = None) -> StageRegistry:
        """Create a registry with all default stage executors."""
        configs = stage_configs or {}
        registry = cls()

        stage_executor_map: list[tuple[PipelineStage, type[BaseStageExecutor]]] = [
            (PipelineStage.PARSE, ParseStageExecutor),
            (PipelineStage.ALIGN, AlignStageExecutor),
            (PipelineStage.COST, CostStageExecutor),
            (PipelineStage.MDP_BUILD, MDPStageExecutor),
            (PipelineStage.BISIMULATE, BisimulationStageExecutor),
            (PipelineStage.POLICY, PolicyStageExecutor),
            (PipelineStage.COMPARE, ComparisonStageExecutor),
            (PipelineStage.BOTTLENECK, BottleneckStageExecutor),
            (PipelineStage.REPAIR, RepairStageExecutor),
            (PipelineStage.OUTPUT, OutputStageExecutor),
        ]

        for stage, executor_cls in stage_executor_map:
            sc = configs.get(stage.value, StageConfig())
            registry.register(stage, executor_cls(config=sc))

        return registry
