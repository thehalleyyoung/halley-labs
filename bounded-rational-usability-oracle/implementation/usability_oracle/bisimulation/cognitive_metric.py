"""
usability_oracle.bisimulation.cognitive_metric — Cognitive bisimulation metric.

Implements the bounded-rational cognitive distance:

    d_cog(s₁, s₂) = sup_{β' ≤ β}  d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))

and related tools for cognitive-distance-based state aggregation, kernel
construction, and policy sensitivity analysis.

Key capabilities:
  - Efficient d_cog computation via policy sensitivity
  - Connection to free-energy difference
  - Cognitive metric as kernel for GP regression
  - Metric-based state aggregation
  - Application: merge states that are cognitively indistinguishable

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
- Ferns, N., Panangaden, P. & Precup, D. (2004). Metrics for finite
  Markov decision processes. *UAI*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usability_oracle.bisimulation.cognitive_distance import (
    CognitiveDistanceComputer,
    _soft_value_iteration,
)
from usability_oracle.bisimulation.models import (
    BisimulationResult,
    CognitiveDistanceMatrix,
    Partition,
)
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy sensitivity analysis
# ---------------------------------------------------------------------------

@dataclass
class PolicySensitivity:
    """Analyse how sensitive the bounded-rational policy is to state changes.

    The sensitivity at (s, β) measures the gradient of the policy distribution
    with respect to state perturbations, indicating which states are near
    decision boundaries.

    Parameters
    ----------
    n_beta_samples : int
        Number of β values to sample in [0, β_max].
    """

    n_beta_samples: int = 20

    def compute_sensitivity(
        self,
        mdp: MDP,
        beta_max: float,
    ) -> dict[str, float]:
        """Compute per-state policy sensitivity.

        Sensitivity is measured as the maximum total-variation distance
        between the policy at the state and its neighbours' policies,
        across all β' ≤ β_max.

        Parameters
        ----------
        mdp : MDP
        beta_max : float

        Returns
        -------
        dict[str, float]
            Maps state_id → sensitivity score ∈ [0, 1].
        """
        state_ids = sorted(mdp.states.keys())
        cdc = CognitiveDistanceComputer(
            n_grid=self.n_beta_samples, refine=False,
        )

        betas = np.linspace(0.01, beta_max, self.n_beta_samples)
        sensitivity: dict[str, float] = {s: 0.0 for s in state_ids}

        for beta_val in betas:
            values = _soft_value_iteration(mdp, float(beta_val))

            for sid in state_ids:
                actions = mdp.get_actions(sid)
                if not actions:
                    continue
                action_order = sorted(actions)

                pi_s = cdc._policy_at_state_ordered(
                    mdp, sid, float(beta_val), values, action_order,
                )

                # Compare with neighbours
                max_tv = 0.0
                for succ in mdp.get_successors(sid):
                    pi_n = cdc._policy_at_state_ordered(
                        mdp, succ, float(beta_val), values, action_order,
                    )
                    tv = 0.5 * float(np.sum(np.abs(pi_s - pi_n)))
                    max_tv = max(max_tv, tv)

                sensitivity[sid] = max(sensitivity[sid], max_tv)

        return sensitivity

    def high_sensitivity_states(
        self,
        mdp: MDP,
        beta_max: float,
        threshold: float = 0.3,
    ) -> list[str]:
        """Return states with sensitivity above the threshold.

        These are states near decision boundaries where small changes
        in the state can lead to large policy changes.

        Parameters
        ----------
        mdp : MDP
        beta_max : float
        threshold : float

        Returns
        -------
        list[str]
            State IDs sorted by decreasing sensitivity.
        """
        sens = self.compute_sensitivity(mdp, beta_max)
        high = [(s, v) for s, v in sens.items() if v >= threshold]
        high.sort(key=lambda x: -x[1])
        return [s for s, _ in high]


# ---------------------------------------------------------------------------
# Free-energy distance
# ---------------------------------------------------------------------------

@dataclass
class FreeEnergyDistance:
    """Distance based on free-energy differences.

    The free energy at state s under rationality β is:

        F(s, β) = −(1/β) log Σ_a exp(β · Q(s, a))

    The free-energy distance is |F(s₁, β) − F(s₂, β)|.

    Parameters
    ----------
    n_beta_samples : int
        Number of β values for supremum computation.
    """

    n_beta_samples: int = 30

    def compute_matrix(
        self,
        mdp: MDP,
        beta_max: float,
    ) -> CognitiveDistanceMatrix:
        """Compute pairwise free-energy distance matrix.

        Takes the supremum over β' ≤ β_max for robustness.

        Parameters
        ----------
        mdp : MDP
        beta_max : float

        Returns
        -------
        CognitiveDistanceMatrix
        """
        state_ids = sorted(mdp.states.keys())
        n = len(state_ids)
        distances = np.zeros((n, n), dtype=np.float64)

        betas = np.linspace(0.01, beta_max, self.n_beta_samples)

        for beta_val in betas:
            fe = self._free_energies(mdp, float(beta_val), state_ids)

            for i in range(n):
                for j in range(i + 1, n):
                    d = abs(fe[i] - fe[j])
                    distances[i, j] = max(distances[i, j], d)
                    distances[j, i] = distances[i, j]

        # Normalise to [0, 1]
        diam = distances.max()
        if diam > 0:
            distances /= diam

        return CognitiveDistanceMatrix(distances=distances, state_ids=state_ids)

    def _free_energies(
        self,
        mdp: MDP,
        beta: float,
        state_ids: list[str],
    ) -> np.ndarray:
        """Compute free energy F(s, β) for all states."""
        values = _soft_value_iteration(mdp, beta)
        fe = np.zeros(len(state_ids), dtype=np.float64)

        for i, sid in enumerate(state_ids):
            # F(s) = -V(s) in the soft-value formulation
            fe[i] = -values.get(sid, 0.0)

        return fe


# ---------------------------------------------------------------------------
# Cognitive kernel for GP regression
# ---------------------------------------------------------------------------

@dataclass
class CognitiveKernel:
    """Cognitive metric as a kernel for Gaussian Process regression.

    Converts the cognitive distance into a positive semi-definite kernel:

        k(s₁, s₂) = σ² · exp(−d_cog(s₁, s₂)² / (2ℓ²))

    where ℓ is the length scale and σ² is the signal variance.

    Parameters
    ----------
    length_scale : float
        Kernel length scale ℓ.
    signal_variance : float
        Signal variance σ².
    """

    length_scale: float = 0.5
    signal_variance: float = 1.0

    def compute_kernel_matrix(
        self,
        distance_matrix: CognitiveDistanceMatrix,
    ) -> np.ndarray:
        """Compute the kernel (Gram) matrix from a cognitive distance matrix.

        Parameters
        ----------
        distance_matrix : CognitiveDistanceMatrix

        Returns
        -------
        np.ndarray
            Positive semi-definite kernel matrix (n, n).
        """
        d = distance_matrix.distances
        K = self.signal_variance * np.exp(
            -d ** 2 / (2.0 * self.length_scale ** 2)
        )
        return K

    def predict(
        self,
        distance_matrix: CognitiveDistanceMatrix,
        observed_indices: list[int],
        observed_values: np.ndarray,
        noise_variance: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GP regression: predict values at all states from observations.

        Parameters
        ----------
        distance_matrix : CognitiveDistanceMatrix
        observed_indices : list[int]
            Indices of observed states.
        observed_values : np.ndarray
            Observed values at those states.
        noise_variance : float
            Observation noise σ²_n.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean, variance) predictions at all states.
        """
        K = self.compute_kernel_matrix(distance_matrix)
        n = K.shape[0]
        obs = np.array(observed_indices)

        if len(obs) == 0:
            return np.zeros(n), self.signal_variance * np.ones(n)

        K_obs = K[np.ix_(obs, obs)] + noise_variance * np.eye(len(obs))
        K_star = K[:, obs]

        try:
            L = np.linalg.cholesky(K_obs)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, observed_values))
            mean = K_star @ alpha

            V = np.linalg.solve(L, K_star.T)
            variance = np.diag(K) - np.sum(V ** 2, axis=0)
            variance = np.maximum(variance, 0.0)
        except np.linalg.LinAlgError:
            # Fallback: pseudoinverse
            K_inv = np.linalg.pinv(K_obs)
            mean = K_star @ K_inv @ observed_values
            variance = np.diag(K) - np.diag(K_star @ K_inv @ K_star.T)
            variance = np.maximum(variance, 0.0)

        return mean, variance


# ---------------------------------------------------------------------------
# Metric-based state aggregation
# ---------------------------------------------------------------------------

@dataclass
class CognitiveAggregation:
    """Merge cognitively indistinguishable states.

    States with d_cog(s₁, s₂) ≤ ε are merged, producing a quotient
    MDP where the bounded-rational agent cannot tell merged states apart.

    Parameters
    ----------
    epsilon : float
        Cognitive indistinguishability threshold.
    beta : float
        Rationality parameter.
    n_grid : int
        Grid points for cognitive distance computation.
    """

    epsilon: float = 0.05
    beta: float = 1.0
    n_grid: int = 30

    def aggregate(self, mdp: MDP) -> BisimulationResult:
        """Perform cognitive aggregation.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        BisimulationResult
        """
        cdc = CognitiveDistanceComputer(n_grid=self.n_grid, refine=True)
        dm = cdc.compute_distance_matrix(mdp, self.beta)
        partition = dm.threshold_partition(self.epsilon)

        # Build quotient
        from usability_oracle.bisimulation.quotient import QuotientMDPBuilder
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)

        # Compute abstraction error
        values = _soft_value_iteration(mdp, self.beta)
        max_error = 0.0
        for block in partition.blocks:
            if len(block) <= 1:
                continue
            block_vals = [values.get(s, 0.0) for s in block]
            max_error = max(max_error, max(block_vals) - min(block_vals))

        logger.info(
            "Cognitive aggregation: ε=%.4f, β=%.2f, "
            "%d→%d states, error=%.4f",
            self.epsilon, self.beta,
            len(mdp.states), partition.n_blocks, max_error,
        )

        return BisimulationResult(
            partition=partition,
            quotient_mdp=quotient,
            abstraction_error=max_error,
            beta_used=self.beta,
            iterations=1,
            refinement_history=[len(mdp.states), partition.n_blocks],
            metadata={
                "method": "cognitive_aggregation",
                "epsilon": self.epsilon,
                "distance_diameter": dm.diameter(),
                "mean_distance": dm.mean_distance(),
            },
        )

    def sensitivity_guided_aggregation(
        self,
        mdp: MDP,
    ) -> BisimulationResult:
        """Aggregation with per-state adaptive thresholds.

        Uses policy sensitivity to set tighter thresholds near decision
        boundaries and looser thresholds in policy-stable regions.

        Parameters
        ----------
        mdp : MDP

        Returns
        -------
        BisimulationResult
        """
        ps = PolicySensitivity(n_beta_samples=self.n_grid)
        sensitivity = ps.compute_sensitivity(mdp, self.beta)

        cdc = CognitiveDistanceComputer(n_grid=self.n_grid, refine=True)
        dm = cdc.compute_distance_matrix(mdp, self.beta)

        state_ids = dm.state_ids
        n = len(state_ids)

        # Adaptive thresholds: tighter for high-sensitivity states
        max_sens = max(sensitivity.values()) if sensitivity else 1.0
        if max_sens < 1e-10:
            max_sens = 1.0

        # Build adaptive distance matrix
        adaptive_d = dm.distances.copy()
        for i in range(n):
            for j in range(i + 1, n):
                s_i = sensitivity.get(state_ids[i], 0.0)
                s_j = sensitivity.get(state_ids[j], 0.0)
                # Scale distance by sensitivity
                scale = 1.0 + (s_i + s_j) / (2.0 * max_sens)
                adaptive_d[i, j] *= scale
                adaptive_d[j, i] = adaptive_d[i, j]

        adaptive_dm = CognitiveDistanceMatrix(
            distances=adaptive_d, state_ids=state_ids,
        )
        partition = adaptive_dm.threshold_partition(self.epsilon)

        from usability_oracle.bisimulation.quotient import QuotientMDPBuilder
        builder = QuotientMDPBuilder(verify=False)
        quotient = builder.build(mdp, partition)

        values = _soft_value_iteration(mdp, self.beta)
        max_error = 0.0
        for block in partition.blocks:
            if len(block) <= 1:
                continue
            block_vals = [values.get(s, 0.0) for s in block]
            max_error = max(max_error, max(block_vals) - min(block_vals))

        return BisimulationResult(
            partition=partition,
            quotient_mdp=quotient,
            abstraction_error=max_error,
            beta_used=self.beta,
            iterations=1,
            refinement_history=[len(mdp.states), partition.n_blocks],
            metadata={
                "method": "sensitivity_guided_aggregation",
                "epsilon": self.epsilon,
                "mean_sensitivity": float(np.mean(list(sensitivity.values()))),
            },
        )
