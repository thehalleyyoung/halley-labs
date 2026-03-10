"""
usability_oracle.bottleneck.choice — Choice paralysis detector.

Detects states where the user faces too many equally-attractive options,
leading to high decision cost via the Hick-Hyman law:

    T_decision = a + b · H(choices)

where H(choices) = log₂(n + 1) for n equally-probable choices, or more
generally:

    H(π(·|s)) = -Σ_a π(a|s) log₂ π(a|s)

The "effective number of choices" is:

    n_eff = 2^{H(π(·|s))}

Choice paralysis occurs when n_eff exceeds a threshold and the entropy
ratio H(π)/H_max approaches 1 (near-uniform policy).

References
----------
- Hick, W. E. (1952). On the rate of gain of information. *QJEP*.
- Schwartz, B. (2004). *The Paradox of Choice*.
- Iyengar, S. & Lepper, M. (2000). When choice is demotivating. *JPSP*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import MDP
from usability_oracle.policy.models import Policy


# ---------------------------------------------------------------------------
# ChoiceParalysisDetector
# ---------------------------------------------------------------------------

@dataclass
class ChoiceParalysisDetector:
    """Detect choice paralysis at UI states.

    Choice paralysis occurs when the bounded-rational policy has high
    entropy (many near-equal options) and the decision cost exceeds a
    threshold.

    Parameters
    ----------
    PARALYSIS_THRESHOLD : float
        Entropy ratio H(π)/H_max above which paralysis is flagged.
        Default 0.8 (policy is at least 80% as uncertain as uniform).
    MIN_ACTIONS : int
        Minimum number of actions before choice paralysis is possible.
    HICK_A : float
        Hick-Hyman intercept (seconds).
    HICK_B : float
        Hick-Hyman slope (seconds per bit).
    """

    PARALYSIS_THRESHOLD: float = 0.8
    MIN_ACTIONS: int = 4
    HICK_A: float = 0.2
    HICK_B: float = 0.15

    # ── Public API --------------------------------------------------------

    def detect(
        self,
        state: str,
        mdp: MDP,
        policy: Policy,
        beta: float,
    ) -> Optional[BottleneckResult]:
        """Detect choice paralysis at *state*.

        Parameters
        ----------
        state : str
            State identifier.
        mdp : MDP
        policy : Policy
        beta : float
            Rationality parameter.

        Returns
        -------
        BottleneckResult or None
        """
        action_probs = policy.action_probs(state)
        n_actions = len(action_probs)

        if n_actions < self.MIN_ACTIONS:
            return None

        # Compute decision entropy
        entropy = self._decision_entropy(action_probs)

        # Max possible entropy (uniform distribution)
        max_entropy = math.log(n_actions) if n_actions > 0 else 0.0

        if max_entropy <= 0:
            return None

        # Entropy ratio: how close to uniform?
        entropy_ratio = self._policy_entropy_ratio(entropy, max_entropy)

        # Effective number of choices
        n_eff = self._effective_choices(action_probs)

        # Hick-Hyman decision time
        decision_time = self._choice_cost(n_eff)

        # Build evidence
        evidence: dict[str, float] = {
            "decision_entropy": entropy,
            "max_entropy": max_entropy,
            "entropy_ratio": entropy_ratio,
            "effective_choices": n_eff,
            "decision_time_seconds": decision_time,
            "n_actions": float(n_actions),
            "hick_bits": entropy / math.log(2) if entropy > 0 else 0.0,
        }

        # Detection logic
        is_paralysed = entropy_ratio > self.PARALYSIS_THRESHOLD

        if not is_paralysed:
            return None

        # Confidence scales with how far above the threshold
        confidence = min(1.0, (entropy_ratio - self.PARALYSIS_THRESHOLD) / (
            1.0 - self.PARALYSIS_THRESHOLD
        ) * 0.7 + 0.3)

        # Also consider absolute number of effective choices
        if n_eff > 10:
            confidence = min(1.0, confidence + 0.1)

        severity = self._severity_from_choices(n_eff, entropy_ratio)

        return BottleneckResult(
            bottleneck_type=BottleneckType.CHOICE_PARALYSIS,
            severity=severity,
            confidence=confidence,
            affected_states=[state],
            affected_actions=list(action_probs.keys()),
            cognitive_law=CognitiveLaw.HICK_HYMAN,
            channel="cognitive",
            evidence=evidence,
            description=(
                f"Choice paralysis at state {state!r}: "
                f"{n_eff:.1f} effective choices "
                f"(entropy ratio={entropy_ratio:.2f}, "
                f"decision time={decision_time:.2f}s)"
            ),
            recommendation=(
                "Reduce number of options through progressive disclosure, "
                "default recommendations, or categorical grouping."
            ),
            repair_hints=[
                "Reduce visible options to 3-5 primary choices",
                "Add progressive disclosure for secondary actions",
                "Provide a default or recommended option",
                "Group related actions into categories/menus",
                "Use visual hierarchy to differentiate primary from secondary",
            ],
        )

    # ── Information-theoretic measures ------------------------------------

    def _decision_entropy(self, action_probs: dict[str, float]) -> float:
        """Compute the decision entropy H(π(·|s)).

        H(π) = -Σ_a π(a|s) ln π(a|s)

        Parameters
        ----------
        action_probs : dict[str, float]
            Action probability distribution.

        Returns
        -------
        float
            Entropy in nats.
        """
        if not action_probs:
            return 0.0
        h = 0.0
        for p in action_probs.values():
            if p > 0:
                h -= p * math.log(p)
        return h

    def _effective_choices(self, action_probs: dict[str, float]) -> float:
        """Compute the effective number of choices: n_eff = exp(H).

        This is the perplexity of the action distribution — the number of
        equally-probable choices that would yield the same entropy.

        Parameters
        ----------
        action_probs : dict[str, float]

        Returns
        -------
        float
        """
        h = self._decision_entropy(action_probs)
        return math.exp(h)

    def _choice_cost(self, n_choices: float) -> float:
        """Compute Hick-Hyman decision time.

        T = a + b · log₂(n)

        Consistent with :class:`~usability_oracle.cognitive.hick.HickHymanLaw`
        which uses the Hyman (1953) formulation.

        Parameters
        ----------
        n_choices : float
            Number of (effective) choices.

        Returns
        -------
        float
            Decision time in seconds.
        """
        if n_choices <= 1:
            return self.HICK_A
        return self.HICK_A + self.HICK_B * math.log2(n_choices)

    def _policy_entropy_ratio(
        self,
        policy_entropy: float,
        max_entropy: float,
    ) -> float:
        """Compute the entropy ratio H(π) / H_max.

        A ratio close to 1.0 means the policy is nearly uniform — the agent
        has no strong preference among actions.

        Parameters
        ----------
        policy_entropy : float
        max_entropy : float

        Returns
        -------
        float
            Ratio ∈ [0, 1].
        """
        if max_entropy <= 0:
            return 0.0
        return min(1.0, policy_entropy / max_entropy)

    # ── Helpers -----------------------------------------------------------

    def _severity_from_choices(
        self,
        n_eff: float,
        entropy_ratio: float,
    ) -> Severity:
        """Map effective choices and entropy ratio to severity."""
        # Combined score
        score = (entropy_ratio * 2.0) + (n_eff / 10.0)
        if score > 4.0:
            return Severity.CRITICAL
        elif score > 3.0:
            return Severity.HIGH
        elif score > 2.0:
            return Severity.MEDIUM
        elif score > 1.0:
            return Severity.LOW
        return Severity.INFO
