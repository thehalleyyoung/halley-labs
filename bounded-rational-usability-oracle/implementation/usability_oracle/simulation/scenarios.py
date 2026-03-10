"""
usability_oracle.simulation.scenarios — Pre-built simulation scenarios.

Provides ready-made scenarios for common UI tasks: form submission,
navigation, search, settings changes, and multi-step workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.simulation.agent import AgentConfig, SimulatedAgent
from usability_oracle.simulation.environment import UIEnvironment
from usability_oracle.simulation.interaction import InteractionSequence
from usability_oracle.simulation.metrics import SimulationMetrics


@dataclass
class Scenario:
    """A pre-built simulation scenario."""
    name: str
    description: str
    tree_generator: str
    generator_kwargs: dict[str, Any] = field(default_factory=dict)
    goal_elements: list[str] = field(default_factory=list)
    task_steps: list[str] = field(default_factory=list)
    agent_config: AgentConfig = field(default_factory=AgentConfig)


class ScenarioLibrary:
    """Library of pre-built simulation scenarios."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._generator = SyntheticUIGenerator(seed=seed)

    # ------------------------------------------------------------------
    # Pre-built scenarios
    # ------------------------------------------------------------------

    def form_submission(
        self,
        n_fields: int = 6,
        complexity: str = "medium",
    ) -> Scenario:
        """Scenario: Fill and submit a form."""
        return Scenario(
            name="Form Submission",
            description=f"Fill {n_fields} fields and submit a {complexity} form",
            tree_generator="form",
            generator_kwargs={"n_fields": n_fields, "complexity": complexity},
            goal_elements=["submit-btn"],
            task_steps=[f"input-{i}" for i in range(n_fields)] + ["submit-btn"],
        )

    def navigation_task(
        self,
        n_items: int = 8,
        target_depth: int = 2,
    ) -> Scenario:
        """Scenario: Navigate to a specific page."""
        return Scenario(
            name="Navigation Task",
            description=f"Navigate through a menu with {n_items} items to depth {target_depth}",
            tree_generator="navigation",
            generator_kwargs={"n_items": n_items, "depth": target_depth},
            goal_elements=["nav-sub-0-0"],
        )

    def dashboard_review(self, n_widgets: int = 6) -> Scenario:
        """Scenario: Review dashboard widgets."""
        return Scenario(
            name="Dashboard Review",
            description=f"Review {n_widgets} dashboard widgets",
            tree_generator="dashboard",
            generator_kwargs={"n_widgets": n_widgets},
            goal_elements=[f"widget-{i}" for i in range(n_widgets)],
        )

    def search_task(self, n_results: int = 10) -> Scenario:
        """Scenario: Search and select a result."""
        return Scenario(
            name="Search Task",
            description=f"Enter search query and select from {n_results} results",
            tree_generator="search_results",
            generator_kwargs={"n_results": n_results},
            goal_elements=["search-btn", "result-title-0"],
            task_steps=["search-bar", "search-btn", "result-title-0"],
        )

    def settings_change(self, n_settings: int = 10) -> Scenario:
        """Scenario: Find and change a specific setting."""
        return Scenario(
            name="Settings Change",
            description=f"Find and toggle a setting among {n_settings} options",
            tree_generator="settings_page",
            generator_kwargs={"n_settings": n_settings},
            goal_elements=["setting-toggle-0", "save-btn"],
            task_steps=["setting-toggle-0", "save-btn"],
        )

    # ------------------------------------------------------------------
    # Run scenario
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario: Scenario,
        n_runs: int = 10,
        agent_config: AgentConfig | None = None,
    ) -> list[InteractionSequence]:
        """Execute a scenario multiple times and collect results."""
        config = agent_config or scenario.agent_config

        # Generate the UI tree
        gen_method = getattr(self._generator, f"generate_{scenario.tree_generator}", None)
        if gen_method is None:
            raise ValueError(f"Unknown generator: {scenario.tree_generator}")
        tree = gen_method(**scenario.generator_kwargs)

        sequences: list[InteractionSequence] = []

        for run_idx in range(n_runs):
            run_config = AgentConfig(
                beta=config.beta,
                fitts_a=config.fitts_a,
                fitts_b=config.fitts_b,
                hick_a=config.hick_a,
                hick_b=config.hick_b,
                wm_capacity=config.wm_capacity,
                wm_decay_rate=config.wm_decay_rate,
                error_rate_base=config.error_rate_base,
                visual_search_rate=config.visual_search_rate,
                max_steps=config.max_steps,
                seed=(self._seed + run_idx) if self._seed is not None else None,
            )
            agent = SimulatedAgent(run_config)
            env = UIEnvironment(tree, goal_elements=scenario.goal_elements, task_steps=scenario.task_steps)

            def action_provider(step: int) -> list[dict[str, Any]]:
                actions = env.get_available_actions()
                return [a.to_dict() for a in actions]

            def done_check(event: Any) -> bool:
                return env.is_goal_reached()

            events = agent.run_task(action_provider, goal=scenario.name, done_check=done_check)

            seq = InteractionSequence(
                events=events,
                task_name=scenario.name,
                goal_reached=env.is_goal_reached(),
                agent_config={"beta": config.beta, "wm_capacity": config.wm_capacity},
            )
            sequences.append(seq)

        return sequences

    # ------------------------------------------------------------------
    # Compare scenarios
    # ------------------------------------------------------------------

    def compare_scenarios(
        self,
        scenario_a: Scenario,
        scenario_b: Scenario,
        n_runs: int = 20,
    ) -> dict[str, Any]:
        """Run two scenarios and compare their metrics."""
        seqs_a = self.run_scenario(scenario_a, n_runs)
        seqs_b = self.run_scenario(scenario_b, n_runs)
        return SimulationMetrics.compare(seqs_a, seqs_b)

    # ------------------------------------------------------------------
    # All scenarios
    # ------------------------------------------------------------------

    def all_scenarios(self) -> list[Scenario]:
        """Return all pre-built scenarios."""
        return [
            self.form_submission(),
            self.navigation_task(),
            self.dashboard_review(),
            self.search_task(),
            self.settings_change(),
        ]
