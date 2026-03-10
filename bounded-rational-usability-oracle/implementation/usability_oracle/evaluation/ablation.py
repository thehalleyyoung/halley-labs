"""
usability_oracle.evaluation.ablation — Ablation study framework.

Systematically disables individual pipeline components and measures the
impact on overall performance, enabling contribution analysis.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from usability_oracle.core.enums import RegressionVerdict


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of an ablation study."""

    full_score: float = 0.0
    ablated_scores: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Ablation study (full score = {self.full_score:.4f}):"]
        for comp in sorted(self.contributions, key=lambda k: self.contributions[k], reverse=True):
            lines.append(
                f"  {comp:30s}  ablated={self.ablated_scores[comp]:.4f}  "
                f"contribution={self.contributions[comp]:+.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------

class AblationStudy:
    """Run an ablation study by disabling pipeline components one at a time.

    Parameters:
        pipeline_fn: Callable that accepts ``(config_dict, dataset)`` and
            returns a float score (e.g. accuracy).
        score_fn: Optional function that computes a scalar score from pipeline
            outputs; if provided, ``pipeline_fn`` should return raw outputs.
    """

    def __init__(
        self,
        pipeline_fn: Optional[Callable[..., Any]] = None,
        score_fn: Optional[Callable[[Any], float]] = None,
    ) -> None:
        self._pipeline_fn = pipeline_fn
        self._score_fn = score_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        pipeline_config: dict[str, Any],
        dataset: list[Any],
        components: list[str],
    ) -> AblationResult:
        """Disable each component in *components* one at a time and measure impact.

        Parameters:
            pipeline_config: Base configuration dict.
            dataset: Evaluation dataset.
            components: List of component names to ablate.
        """
        full_result = self._measure_impact(pipeline_config, dataset)
        full_score = full_result["score"]

        ablated_scores: dict[str, float] = {}
        contributions: dict[str, float] = {}
        timing: dict[str, float] = {}

        for component in components:
            ablated_config = self._disable_component(pipeline_config, component)
            t0 = time.perf_counter()
            ablated_result = self._measure_impact(ablated_config, dataset)
            elapsed = time.perf_counter() - t0
            ablated_scores[component] = ablated_result["score"]
            timing[component] = elapsed

        contributions = self._contribution_analysis(full_score, ablated_scores)

        return AblationResult(
            full_score=full_score,
            ablated_scores=ablated_scores,
            contributions=contributions,
            timing=timing,
            metadata={
                "n_components": len(components),
                "n_dataset": len(dataset),
                "components": components,
            },
        )

    # ------------------------------------------------------------------
    # Disable component
    # ------------------------------------------------------------------

    @staticmethod
    def _disable_component(config: dict[str, Any], component: str) -> dict[str, Any]:
        """Create a copy of *config* with *component* disabled.

        Looks for nested keys like ``"component_name.enabled"`` or top-level
        boolean flags, and sets them to ``False`` / removes them.
        """
        new_config = copy.deepcopy(config)

        # Try nested dict
        if component in new_config and isinstance(new_config[component], dict):
            new_config[component]["enabled"] = False
            return new_config

        # Try flat key
        key_enabled = f"{component}_enabled"
        if key_enabled in new_config:
            new_config[key_enabled] = False
            return new_config

        # Try "disabled_components" list
        disabled = new_config.setdefault("disabled_components", [])
        if component not in disabled:
            disabled.append(component)

        return new_config

    # ------------------------------------------------------------------
    # Measure impact
    # ------------------------------------------------------------------

    def _measure_impact(self, config: dict[str, Any], dataset: list[Any]) -> dict[str, Any]:
        """Run the pipeline with *config* on *dataset* and compute score."""
        if self._pipeline_fn is None:
            return self._default_measure(config, dataset)

        raw = self._pipeline_fn(config, dataset)
        if self._score_fn is not None:
            score = self._score_fn(raw)
        elif isinstance(raw, (int, float)):
            score = float(raw)
        elif isinstance(raw, dict) and "score" in raw:
            score = float(raw["score"])
        else:
            score = 0.0
        return {"score": score, "raw": raw}

    @staticmethod
    def _default_measure(config: dict[str, Any], dataset: list[Any]) -> dict[str, Any]:
        """Fallback measure when no pipeline function is provided.

        Computes a dummy score based on the number of enabled components.
        """
        disabled = set(config.get("disabled_components", []))
        all_components = {
            "parse", "align", "cost", "mdp", "policy",
            "bisimulation", "comparison", "bottleneck",
        }
        enabled = all_components - disabled
        # Also check individual enabled flags
        for key, val in config.items():
            if key.endswith("_enabled") and val is False:
                comp = key.replace("_enabled", "")
                enabled.discard(comp)
            elif isinstance(val, dict) and val.get("enabled") is False:
                enabled.discard(key)
        base_score = len(enabled) / len(all_components) if all_components else 0.0
        return {"score": base_score}

    # ------------------------------------------------------------------
    # Contribution analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _contribution_analysis(
        full_score: float,
        ablated_scores: dict[str, float],
    ) -> dict[str, float]:
        """Compute each component's contribution as score drop when removed.

        ``contribution[c] = full_score − ablated_score[c]``

        A positive contribution means removing the component *hurts*
        performance.
        """
        contributions: dict[str, float] = {}
        for component, ablated in ablated_scores.items():
            contributions[component] = full_score - ablated
        return contributions
