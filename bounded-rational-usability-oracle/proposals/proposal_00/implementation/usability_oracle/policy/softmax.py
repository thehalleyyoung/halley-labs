"""
usability_oracle.policy.softmax — Softmax / Boltzmann policy construction.

Constructs stochastic policies from Q-values using the Boltzmann
(softmax) distribution parameterised by the rationality parameter β:

    π_β(a|s) = p₀(a|s) · exp(−β · Q(s,a)) / Z(s)

where Z(s) = Σ_a p₀(a|s) · exp(−β · Q(s,a)) is the partition function
and p₀ is a prior policy (uniform by default).

The parameter β ∈ [0, ∞) interpolates between:
  - β = 0 : prior policy (no optimisation, purely habitual)
  - β → ∞ : deterministic optimal policy (perfect rationality)

This is the core of the bounded-rational decision theory (Ortega & Braun, 2013).

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
- Todorov, E. (2007). Linearly-solvable Markov decision problems. *NIPS*.
- Levine, S. (2018). Reinforcement learning and control as probabilistic
  inference. *arXiv:1805.00909*.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from usability_oracle.policy.models import Policy, QValues


class SoftmaxPolicy:
    """Construct and analyse softmax (Boltzmann) policies.

    All methods are stateless class/static methods operating on
    :class:`QValues` and :class:`Policy` objects.
    """

    # ── Construction ------------------------------------------------------

    @staticmethod
    def from_q_values(
        q_values: QValues,
        beta: float,
        prior: Optional[Policy] = None,
    ) -> Policy:
        """Create a softmax policy from Q-values.

        π_β(a|s) ∝ p₀(a|s) · exp(−β · Q(s,a))

        Parameters
        ----------
        q_values : QValues
        beta : float
            Rationality parameter (inverse temperature).
        prior : Policy, optional
            Prior policy p₀.  Defaults to uniform over available actions.

        Returns
        -------
        Policy
        """
        state_action_probs: dict[str, dict[str, float]] = {}

        for state, q_s in q_values.values.items():
            if not q_s:
                continue

            actions = list(q_s.keys())
            n = len(actions)

            # Q-value array (costs — lower is better, so we negate for softmax)
            q_arr = np.array([-q_s[a] for a in actions], dtype=np.float64)

            # Prior probabilities
            if prior is not None:
                prior_dist = prior.state_action_probs.get(state, {})
                p0 = np.array(
                    [prior_dist.get(a, 1.0 / n) for a in actions],
                    dtype=np.float64,
                )
            else:
                p0 = np.ones(n, dtype=np.float64) / n

            # Ensure prior sums to 1
            p0_sum = p0.sum()
            if p0_sum > 0:
                p0 /= p0_sum
            else:
                p0 = np.ones(n, dtype=np.float64) / n

            # Compute log-probabilities: log p₀(a|s) + β·(−Q(s,a))
            log_probs = np.log(np.maximum(p0, 1e-300)) + beta * q_arr

            # Numerically stable softmax
            probs = SoftmaxPolicy._softmax_from_logits(log_probs)

            state_action_probs[state] = {
                actions[i]: float(probs[i]) for i in range(n)
            }

        return Policy(state_action_probs=state_action_probs)

    # ── Numerics ----------------------------------------------------------

    @staticmethod
    def _softmax(values: np.ndarray, beta: float) -> np.ndarray:
        """Numerically stable softmax: softmax(β · v).

        Uses the log-sum-exp trick to prevent overflow:
            softmax(x)_i = exp(x_i − max(x)) / Σ_j exp(x_j − max(x))

        Parameters
        ----------
        values : np.ndarray
            Raw values (e.g., −Q(s,a) for each action).
        beta : float
            Inverse temperature.

        Returns
        -------
        np.ndarray
            Probability vector summing to 1.
        """
        x = beta * values
        return SoftmaxPolicy._softmax_from_logits(x)

    @staticmethod
    def _softmax_from_logits(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax from log-space inputs."""
        x_max = np.max(logits)
        exp_x = np.exp(logits - x_max)
        total = exp_x.sum()
        if total <= 0:
            return np.ones_like(logits) / len(logits)
        return exp_x / total

    @staticmethod
    def _log_partition(values: np.ndarray, beta: float) -> float:
        """Log partition function: log Z = log Σ_a exp(β · v_a).

        Uses log-sum-exp for numerical stability.

        Parameters
        ----------
        values : np.ndarray
            Value vector.
        beta : float

        Returns
        -------
        float
            log Z(s).
        """
        x = beta * values
        x_max = float(np.max(x))
        return x_max + float(np.log(np.sum(np.exp(x - x_max))))

    # ── Information-theoretic measures ------------------------------------

    @staticmethod
    def kl_divergence(
        policy: Policy, prior: Policy, state: str
    ) -> float:
        """KL divergence D_KL(π(·|s) ‖ p₀(·|s)).

        D_KL(π ‖ p₀) = Σ_a π(a|s) · log(π(a|s) / p₀(a|s))

        Parameters
        ----------
        policy : Policy
        prior : Policy
        state : str

        Returns
        -------
        float
            KL divergence in nats (≥ 0).
        """
        pi_dist = policy.state_action_probs.get(state, {})
        p0_dist = prior.state_action_probs.get(state, {})

        if not pi_dist:
            return 0.0

        kl = 0.0
        for a, pi_a in pi_dist.items():
            if pi_a <= 0:
                continue
            p0_a = p0_dist.get(a, 1e-10)
            p0_a = max(p0_a, 1e-10)  # avoid log(0)
            kl += pi_a * math.log(pi_a / p0_a)

        return max(kl, 0.0)

    @staticmethod
    def mutual_information(policy: Policy, prior: Policy) -> float:
        """Mutual information I(S; A) under the policy.

        Averaged KL divergence across all states:
        I = (1/|S|) Σ_s D_KL(π(·|s) ‖ p₀(·|s))

        This quantifies the total information the policy uses beyond
        the prior — the *decision complexity*.

        Parameters
        ----------
        policy : Policy
        prior : Policy

        Returns
        -------
        float
            Mutual information in nats.
        """
        states = list(policy.state_action_probs.keys())
        if not states:
            return 0.0

        total_kl = 0.0
        for s in states:
            total_kl += SoftmaxPolicy.kl_divergence(policy, prior, s)

        return total_kl / len(states)

    # ── Beta sweep --------------------------------------------------------

    @staticmethod
    def beta_sweep(
        q_values: QValues,
        betas: list[float],
        prior: Optional[Policy] = None,
    ) -> list[Policy]:
        """Compute policies for a range of β values.

        Useful for tracing the rate-distortion curve from random (β=0)
        to optimal (β→∞).

        Parameters
        ----------
        q_values : QValues
        betas : list[float]
        prior : Policy, optional

        Returns
        -------
        list[Policy]
            One policy per β value.
        """
        return [
            SoftmaxPolicy.from_q_values(q_values, beta, prior)
            for beta in betas
        ]

    # ── Utility -----------------------------------------------------------

    @staticmethod
    def effective_rationality(policy: Policy, prior: Policy) -> float:
        """Estimate the effective β from an observed policy.

        Uses the relationship β ≈ 1 / H_excess where H_excess is the
        difference between prior entropy and policy entropy, averaged
        across states.

        Parameters
        ----------
        policy : Policy
        prior : Policy

        Returns
        -------
        float
            Estimated β.
        """
        states = list(policy.state_action_probs.keys())
        if not states:
            return 0.0

        total_mi = SoftmaxPolicy.mutual_information(policy, prior)
        if total_mi <= 1e-10:
            return 0.0

        # Rough estimate: β ≈ MI / mean_advantage
        # Simplified: just return 1/MI as a relative measure
        return 1.0 / total_mi
