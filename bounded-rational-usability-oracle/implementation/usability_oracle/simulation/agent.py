"""
usability_oracle.simulation.agent — Simulated bounded-rational user agent.

Implements a softmax-rational agent that interacts with UI elements
using Fitts' law pointing, Hick-Hyman decision-making, and working
memory constraints.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from usability_oracle.simulation.interaction import InteractionEvent


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a simulated user agent.

    Attributes:
        beta: Softmax rationality parameter (higher = more rational).
        fitts_a: Fitts' law intercept (seconds).
        fitts_b: Fitts' law slope (seconds/bit).
        hick_a: Hick-Hyman intercept (seconds).
        hick_b: Hick-Hyman slope (seconds/bit).
        wm_capacity: Working memory capacity (chunks).
        wm_decay_rate: Decay rate for working memory items.
        error_rate_base: Base probability of motor error.
        visual_search_rate: Milliseconds per item for visual search.
        max_steps: Maximum interaction steps before timeout.
        seed: RNG seed for reproducibility.
    """
    beta: float = 2.0
    fitts_a: float = 0.05
    fitts_b: float = 0.15
    hick_a: float = 0.2
    hick_b: float = 0.15
    wm_capacity: int = 4
    wm_decay_rate: float = 0.1
    error_rate_base: float = 0.02
    visual_search_rate: float = 50.0
    max_steps: int = 100
    seed: int | None = None


# ---------------------------------------------------------------------------
# Working memory model
# ---------------------------------------------------------------------------

@dataclass
class MemoryItem:
    """An item in working memory."""
    content: str
    activation: float = 1.0
    age: int = 0


class WorkingMemory:
    """Simple working memory model with capacity and decay."""

    def __init__(self, capacity: int = 4, decay_rate: float = 0.1) -> None:
        self._capacity = capacity
        self._decay = decay_rate
        self._items: list[MemoryItem] = []

    def store(self, content: str) -> bool:
        """Store an item in working memory. Returns False if at capacity."""
        # Check if already stored
        for item in self._items:
            if item.content == content:
                item.activation = 1.0
                item.age = 0
                return True

        if len(self._items) >= self._capacity:
            # Evict least active item
            self._items.sort(key=lambda x: x.activation)
            self._items.pop(0)

        self._items.append(MemoryItem(content=content))
        return True

    def recall(self, content: str) -> float:
        """Try to recall an item. Returns activation level (0 if forgotten)."""
        for item in self._items:
            if item.content == content:
                return item.activation
        return 0.0

    def tick(self) -> None:
        """Advance time by one step, decaying all items."""
        for item in self._items:
            item.activation *= (1.0 - self._decay)
            item.age += 1
        # Remove items below threshold
        self._items = [i for i in self._items if i.activation > 0.01]

    @property
    def load(self) -> int:
        return len(self._items)

    @property
    def contents(self) -> list[str]:
        return [i.content for i in self._items]


# ---------------------------------------------------------------------------
# SimulatedAgent
# ---------------------------------------------------------------------------

class SimulatedAgent:
    """A bounded-rational simulated user agent.

    The agent uses softmax action selection with cognitive costs
    derived from Fitts' law, Hick-Hyman law, and working memory
    limitations.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()
        self._rng = random.Random(self._config.seed)
        self._np_rng = np.random.RandomState(self._config.seed)
        self._memory = WorkingMemory(self._config.wm_capacity, self._config.wm_decay_rate)
        self._position = (0.0, 0.0)
        self._step_count = 0
        self._total_time = 0.0
        self._errors = 0
        self._history: list[InteractionEvent] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        available_actions: list[dict[str, Any]],
        goal: str = "",
    ) -> dict[str, Any]:
        """Select an action using softmax rationality.

        Each action dict should have: id, name, target_x, target_y,
        target_width, target_height.
        """
        if not available_actions:
            return {}

        # Compute Q-values for each action
        q_values = np.zeros(len(available_actions))
        for i, action in enumerate(available_actions):
            q_values[i] = self._compute_q_value(action, goal)

        # Softmax selection
        beta = self._config.beta
        logits = beta * q_values
        logits -= np.max(logits)  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()

        idx = self._np_rng.choice(len(available_actions), p=probs)
        return available_actions[idx]

    def _compute_q_value(self, action: dict[str, Any], goal: str) -> float:
        """Compute the expected value of taking an action.

        Higher Q-value = better action (less cost, more goal relevance).
        """
        # Motor cost (Fitts' law)
        motor_cost = self._fitts_time(action)

        # Decision cost (Hick-Hyman)
        n_options = action.get("n_alternatives", 1)
        decision_cost = self._hick_time(n_options)

        # Memory cost
        memory_cost = 0.0
        action_name = action.get("name", "")
        recall = self._memory.recall(action_name)
        if recall < 0.5:
            memory_cost = 0.5 * (1.0 - recall)

        # Goal relevance (simple string matching)
        relevance = 0.0
        if goal:
            name = action.get("name", "").lower()
            if goal.lower() in name or name in goal.lower():
                relevance = 1.0
            elif any(w in name for w in goal.lower().split()):
                relevance = 0.5

        # Q = relevance - total_cost
        total_cost = motor_cost + decision_cost + memory_cost
        return relevance - total_cost

    # ------------------------------------------------------------------
    # Cost models
    # ------------------------------------------------------------------

    def _fitts_time(self, action: dict[str, Any]) -> float:
        """Fitts' law pointing time."""
        tx = action.get("target_x", 0) + action.get("target_width", 20) / 2
        ty = action.get("target_y", 0) + action.get("target_height", 20) / 2
        tw = max(action.get("target_width", 20), 1)
        th = max(action.get("target_height", 20), 1)

        distance = math.sqrt((tx - self._position[0]) ** 2 + (ty - self._position[1]) ** 2)
        target_size = min(tw, th)

        if target_size < 1:
            return self._config.fitts_a + self._config.fitts_b * 10
        id_val = math.log2(2.0 * max(distance, 1.0) / target_size)
        return self._config.fitts_a + self._config.fitts_b * max(id_val, 0)

    def _hick_time(self, n_options: int) -> float:
        """Hick-Hyman decision time."""
        return self._config.hick_a + self._config.hick_b * math.log2(max(n_options, 1) + 1)

    def _visual_search_time(self, n_elements: int) -> float:
        """Linear visual search time."""
        return (self._config.visual_search_rate * n_elements) / 1000.0

    # ------------------------------------------------------------------
    # Execute action
    # ------------------------------------------------------------------

    def execute_action(self, action: dict[str, Any]) -> InteractionEvent:
        """Execute an action and record the interaction event."""
        self._step_count += 1
        self._memory.tick()

        # Calculate timing
        motor_time = self._fitts_time(action)
        decision_time = self._hick_time(action.get("n_alternatives", 1))

        # Motor error
        error_occurred = self._rng.random() < self._config.error_rate_base
        if error_occurred:
            self._errors += 1
            motor_time *= 1.5  # Error correction penalty

        total_time = motor_time + decision_time
        self._total_time += total_time

        # Update position
        tx = action.get("target_x", 0) + action.get("target_width", 20) / 2
        ty = action.get("target_y", 0) + action.get("target_height", 20) / 2
        self._position = (tx, ty)

        # Store in working memory
        self._memory.store(action.get("name", ""))

        event = InteractionEvent(
            step=self._step_count,
            action_id=action.get("id", ""),
            action_name=action.get("name", ""),
            timestamp=self._total_time,
            motor_time=motor_time,
            decision_time=decision_time,
            error=error_occurred,
            position=self._position,
            wm_load=self._memory.load,
        )

        self._history.append(event)
        return event

    # ------------------------------------------------------------------
    # Run full task
    # ------------------------------------------------------------------

    def run_task(
        self,
        action_provider: Callable[[int], list[dict[str, Any]]],
        goal: str = "",
        done_check: Callable[[InteractionEvent], bool] | None = None,
    ) -> list[InteractionEvent]:
        """Run a complete task simulation.

        Parameters:
            action_provider: Function(step) -> list of available actions.
            goal: Goal description for action selection.
            done_check: Function(event) -> bool to check task completion.
        """
        events: list[InteractionEvent] = []

        for step in range(self._config.max_steps):
            actions = action_provider(step)
            if not actions:
                break

            selected = self.select_action(actions, goal)
            if not selected:
                break

            event = self.execute_action(selected)
            events.append(event)

            if done_check and done_check(event):
                break

        return events

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def error_count(self) -> int:
        return self._errors

    @property
    def history(self) -> list[InteractionEvent]:
        return list(self._history)

    def reset(self) -> None:
        """Reset agent state."""
        self._position = (0.0, 0.0)
        self._step_count = 0
        self._total_time = 0.0
        self._errors = 0
        self._history.clear()
        self._memory = WorkingMemory(self._config.wm_capacity, self._config.wm_decay_rate)
