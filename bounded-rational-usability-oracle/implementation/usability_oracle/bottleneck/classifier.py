"""
usability_oracle.bottleneck.classifier — Main 5-type bottleneck classifier.

Orchestrates the five specialised detectors to classify usability bottlenecks:

  1. PerceptualOverloadDetector
  2. ChoiceParalysisDetector
  3. MotorDifficultyDetector
  4. MemoryDecayDetector
  5. CrossChannelInterferenceDetector

For each MDP state, the classifier computes an information-theoretic signature,
runs all applicable detectors, aggregates results, and ranks by severity.

The cost impact of each bottleneck is estimated as the expected additional
cognitive cost compared to a "clean" baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from usability_oracle.bottleneck.choice import ChoiceParalysisDetector
from usability_oracle.bottleneck.interference import CrossChannelInterferenceDetector
from usability_oracle.bottleneck.memory import MemoryDecayDetector
from usability_oracle.bottleneck.models import (
    BottleneckReport,
    BottleneckResult,
    BottleneckSignature,
)
from usability_oracle.bottleneck.motor import MotorDifficultyDetector
from usability_oracle.bottleneck.perceptual import PerceptualOverloadDetector
from usability_oracle.bottleneck.signatures import SignatureComputer
from usability_oracle.core.enums import BottleneckType, Severity
from usability_oracle.mdp.models import MDP
from usability_oracle.policy.models import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BottleneckClassifier
# ---------------------------------------------------------------------------

@dataclass
class BottleneckClassifier:
    """Classify cognitive bottlenecks in a UI-MDP using information-theoretic
    signatures and specialised detectors.

    Parameters
    ----------
    beta : float
        Rationality parameter for bounded-rational analysis.
    min_confidence : float
        Minimum confidence threshold for reporting a bottleneck.
    max_bottlenecks : int
        Maximum number of bottlenecks to report per analysis.
    """

    beta: float = 5.0
    min_confidence: float = 0.3
    max_bottlenecks: int = 50

    _signature_computer: SignatureComputer = field(
        default_factory=SignatureComputer, repr=False
    )
    _perceptual_detector: PerceptualOverloadDetector = field(
        default_factory=PerceptualOverloadDetector, repr=False
    )
    _choice_detector: ChoiceParalysisDetector = field(
        default_factory=ChoiceParalysisDetector, repr=False
    )
    _motor_detector: MotorDifficultyDetector = field(
        default_factory=MotorDifficultyDetector, repr=False
    )
    _memory_detector: MemoryDecayDetector = field(
        default_factory=MemoryDecayDetector, repr=False
    )
    _interference_detector: CrossChannelInterferenceDetector = field(
        default_factory=CrossChannelInterferenceDetector, repr=False
    )

    # ── Public API --------------------------------------------------------

    def classify(
        self,
        mdp: MDP,
        policy: Policy,
        trajectory_stats: Optional[dict[str, Any]] = None,
        cost_breakdown: Optional[dict[str, Any]] = None,
    ) -> list[BottleneckResult]:
        """Run all detectors and return classified bottlenecks.

        Parameters
        ----------
        mdp : MDP
            The UI-MDP to analyse.
        policy : Policy
            The bounded-rational policy π_β.
        trajectory_stats : dict, optional
            Statistics from trajectory sampling (visit counts, etc.).
        cost_breakdown : dict, optional
            Per-state / per-action cost decomposition.

        Returns
        -------
        list[BottleneckResult]
            Detected bottlenecks sorted by severity (highest first).
        """
        trajectory_stats = trajectory_stats or {}
        cost_breakdown = cost_breakdown or {}

        # Phase 1: Compute signatures for all states
        signatures = self._compute_signatures(mdp, policy)

        # Phase 2: Run per-state classification
        per_state_results: list[BottleneckResult] = []
        for state_id in mdp.states:
            state = mdp.states[state_id]
            if state.is_terminal or state.is_goal:
                continue

            sig = signatures.get(state_id)
            context = {
                "trajectory_stats": trajectory_stats,
                "cost_breakdown": cost_breakdown,
                "signature": sig,
                "mdp": mdp,
                "policy": policy,
            }

            results = self._classify_state(state_id, sig, context)
            per_state_results.extend(results)

        # Phase 3: Aggregate and de-duplicate
        aggregated = self._aggregate_bottlenecks(per_state_results)

        # Phase 4: Compute cost impact
        for bottleneck in aggregated:
            bottleneck.evidence["cost_impact"] = self._compute_cost_impact(
                bottleneck, mdp, policy
            )

        # Phase 5: Rank and filter
        ranked = self._rank_by_severity(aggregated)
        filtered = [b for b in ranked if b.confidence >= self.min_confidence]

        return filtered[: self.max_bottlenecks]

    def classify_to_report(
        self,
        mdp: MDP,
        policy: Policy,
        trajectory_stats: Optional[dict[str, Any]] = None,
        cost_breakdown: Optional[dict[str, Any]] = None,
    ) -> BottleneckReport:
        """Classify and return a full report.

        Same as :meth:`classify` but wraps results in a
        :class:`BottleneckReport` with summary statistics.
        """
        bottlenecks = self.classify(mdp, policy, trajectory_stats, cost_breakdown)
        total_impact = sum(
            b.evidence.get("cost_impact", 0.0) for b in bottlenecks
        )
        report = BottleneckReport(
            bottlenecks=bottlenecks,
            total_cost_impact=total_impact,
        )
        report.generate_summary()
        return report

    # ── Signature computation ---------------------------------------------

    def _compute_signatures(
        self,
        mdp: MDP,
        policy: Policy,
    ) -> dict[str, BottleneckSignature]:
        """Compute information-theoretic signatures for all states.

        Parameters
        ----------
        mdp : MDP
        policy : Policy

        Returns
        -------
        dict[str, BottleneckSignature]
        """
        signatures: dict[str, BottleneckSignature] = {}
        for state_id in mdp.states:
            state = mdp.states[state_id]
            if state.is_terminal or state.is_goal:
                continue
            signatures[state_id] = self._signature_computer.compute(
                mdp, policy, state_id
            )
        return signatures

    # ── Per-state classification -------------------------------------------

    def _classify_state(
        self,
        state: str,
        signature: Optional[BottleneckSignature],
        context: dict[str, Any],
    ) -> list[BottleneckResult]:
        """Run all detectors on a single state and collect results.

        Parameters
        ----------
        state : str
        signature : BottleneckSignature or None
        context : dict

        Returns
        -------
        list[BottleneckResult]
        """
        results: list[BottleneckResult] = []
        mdp: MDP = context["mdp"]
        policy: Policy = context["policy"]

        # 1. Perceptual overload
        state_obj = mdp.states.get(state)
        features = state_obj.features if state_obj else {}
        result = self._perceptual_detector.detect(state, mdp, features)
        if result is not None:
            results.append(result)

        # 2. Choice paralysis
        result = self._choice_detector.detect(state, mdp, policy, self.beta)
        if result is not None:
            results.append(result)

        # 3. Motor difficulty — for each action
        for aid in mdp.get_actions(state):
            action_obj = mdp.actions.get(aid)
            action_features = {}
            if action_obj is not None:
                action_features = dict(getattr(action_obj, "preconditions", []))
                # Use action features from the Action object
                # Features might be on the transitions or the action itself
            result = self._motor_detector.detect(state, aid, mdp, features)
            if result is not None:
                results.append(result)

        # 4. Memory decay
        trajectory = context.get("trajectory_stats", {}).get("trajectory", [])
        result = self._memory_detector.detect(trajectory, state, mdp)
        if result is not None:
            results.append(result)

        # 5. Cross-channel interference
        actions = mdp.get_actions(state)
        result = self._interference_detector.detect(state, actions, mdp)
        if result is not None:
            results.append(result)

        return results

    # ── Aggregation -------------------------------------------------------

    def _aggregate_bottlenecks(
        self,
        per_state: list[BottleneckResult],
    ) -> list[BottleneckResult]:
        """Aggregate per-state results, merging similar bottlenecks.

        Bottlenecks of the same type affecting adjacent states are merged
        into a single result with combined affected_states.

        Parameters
        ----------
        per_state : list[BottleneckResult]

        Returns
        -------
        list[BottleneckResult]
        """
        groups: dict[tuple[str, str], list[BottleneckResult]] = {}
        for result in per_state:
            key = (result.bottleneck_type.value, result.channel)
            groups.setdefault(key, []).append(result)

        aggregated: list[BottleneckResult] = []
        for (btype_val, channel), group in groups.items():
            if len(group) == 1:
                aggregated.append(group[0])
                continue

            # Merge: take max severity & confidence, union of states/actions
            all_states: set[str] = set()
            all_actions: set[str] = set()
            max_severity = Severity.INFO
            max_confidence = 0.0
            merged_evidence: dict[str, float] = {}

            severity_order = [
                Severity.INFO, Severity.LOW, Severity.MEDIUM,
                Severity.HIGH, Severity.CRITICAL,
            ]

            for r in group:
                all_states.update(r.affected_states)
                all_actions.update(r.affected_actions)
                if severity_order.index(r.severity) > severity_order.index(max_severity):
                    max_severity = r.severity
                max_confidence = max(max_confidence, r.confidence)
                for k, v in r.evidence.items():
                    merged_evidence[k] = max(merged_evidence.get(k, 0.0), v)

            # Use the first result as template
            template = group[0]
            aggregated.append(
                BottleneckResult(
                    bottleneck_type=template.bottleneck_type,
                    severity=max_severity,
                    confidence=max_confidence,
                    affected_states=sorted(all_states),
                    affected_actions=sorted(all_actions),
                    cognitive_law=template.cognitive_law,
                    channel=channel,
                    evidence=merged_evidence,
                    description=template.description,
                    recommendation=template.recommendation,
                    repair_hints=template.repair_hints,
                )
            )

        return aggregated

    # ── Ranking -----------------------------------------------------------

    def _rank_by_severity(
        self,
        bottlenecks: list[BottleneckResult],
    ) -> list[BottleneckResult]:
        """Sort bottlenecks by impact score (severity × confidence), descending.

        Parameters
        ----------
        bottlenecks : list[BottleneckResult]

        Returns
        -------
        list[BottleneckResult]
        """
        return sorted(bottlenecks, key=lambda b: b.impact_score, reverse=True)

    # ── Cost impact estimation --------------------------------------------

    def _compute_cost_impact(
        self,
        bottleneck: BottleneckResult,
        mdp: MDP,
        policy: Policy,
    ) -> float:
        """Estimate the cognitive cost impact of a bottleneck.

        The impact is approximated as the expected additional cost incurred
        at the affected states, weighted by the policy's visit probability.

        Parameters
        ----------
        bottleneck : BottleneckResult
        mdp : MDP
        policy : Policy

        Returns
        -------
        float
            Estimated additional cost in seconds.
        """
        total_impact = 0.0

        for state_id in bottleneck.affected_states:
            state = mdp.states.get(state_id)
            if state is None:
                continue

            # Compute expected cost at this state under the policy
            action_probs = policy.action_probs(state_id)
            expected_cost = 0.0
            for aid, prob in action_probs.items():
                for target, trans_prob, cost in mdp.get_transitions(state_id, aid):
                    expected_cost += prob * trans_prob * cost

            # Impact proportional to severity
            severity_multiplier = bottleneck.severity_score / 4.0
            total_impact += expected_cost * severity_multiplier

        return total_impact
