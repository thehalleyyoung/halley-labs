"""
SeparatorConsistencyChecker: verify and repair consistency of marginal
distributions on separator variables shared between subgraphs.

When subgraphs are solved independently, their solutions may assign
different marginal distributions to shared separator variables. This
module detects such inconsistencies, quantifies them (KL divergence,
total variation distance), and projects solutions to the nearest
consistent point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr

logger = logging.getLogger(__name__)

EPSILON = 1e-15


@dataclass
class ConsistencyReport:
    """Report on the consistency of separator marginals."""
    is_consistent: bool
    max_inconsistency: float
    mean_inconsistency: float
    per_separator: Dict[int, float]
    per_pair: Dict[Tuple[int, int], float]
    metric_used: str
    n_violations: int


@dataclass
class RepairResult:
    """Result of marginal repair."""
    repaired_marginals: Dict[int, Dict[int, np.ndarray]]
    total_adjustment: float
    per_separator_adjustment: Dict[int, float]
    n_iterations: int
    converged: bool


@dataclass
class SeparatorSpec:
    """Specification of a separator."""
    separator_id: int
    variable_indices: List[int]
    adjacent_subgraphs: List[int]
    cardinality: int = 2


class SeparatorConsistencyChecker:
    """
    Checks and repairs consistency of marginal distributions on separator
    variables between adjacent subgraphs.

    Two subgraphs sharing a separator must agree on the marginal distribution
    of the separator variables. This checker:
    1. Detects violations (marginals disagree beyond tolerance)
    2. Quantifies inconsistency (KL, TV, Hellinger distances)
    3. Repairs by projecting to the nearest consistent point
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        metric: str = "tv",
        max_repair_iterations: int = 100,
        repair_step_size: float = 0.5,
    ):
        """
        Args:
            tolerance: Maximum allowable inconsistency.
            metric: Distance metric ('tv', 'kl', 'hellinger', 'js').
            max_repair_iterations: Maximum repair iterations.
            repair_step_size: Step size for iterative repair.
        """
        if metric not in ("tv", "kl", "hellinger", "js"):
            raise ValueError(f"Unknown metric: {metric}")
        self.tolerance = tolerance
        self.metric = metric
        self.max_repair_iterations = max_repair_iterations
        self.repair_step_size = repair_step_size

    def check_consistency(
        self,
        subgraph_marginals: Dict[int, Dict[int, np.ndarray]],
        separators: List[SeparatorSpec],
    ) -> ConsistencyReport:
        """
        Check consistency of marginal distributions across separators.

        Args:
            subgraph_marginals: {subgraph_id: {separator_id: marginal_array}}
            separators: List of separator specifications.

        Returns:
            ConsistencyReport with per-separator and per-pair inconsistencies.
        """
        per_separator: Dict[int, float] = {}
        per_pair: Dict[Tuple[int, int], float] = {}
        n_violations = 0

        for sep in separators:
            adj = sep.adjacent_subgraphs
            if len(adj) < 2:
                per_separator[sep.separator_id] = 0.0
                continue

            max_incon = 0.0
            for i_idx in range(len(adj)):
                for j_idx in range(i_idx + 1, len(adj)):
                    sg_i, sg_j = adj[i_idx], adj[j_idx]

                    m_i = self._get_marginal(subgraph_marginals, sg_i, sep.separator_id)
                    m_j = self._get_marginal(subgraph_marginals, sg_j, sep.separator_id)

                    if m_i is None or m_j is None:
                        continue

                    m_i, m_j = self._align_marginals(m_i, m_j)
                    d = self.compute_inconsistency(m_i, m_j)
                    per_pair[(sg_i, sg_j)] = d
                    max_incon = max(max_incon, d)

                    if d > self.tolerance:
                        n_violations += 1

            per_separator[sep.separator_id] = max_incon

        all_vals = list(per_pair.values())
        max_inc = max(all_vals) if all_vals else 0.0
        mean_inc = float(np.mean(all_vals)) if all_vals else 0.0

        return ConsistencyReport(
            is_consistent=(n_violations == 0),
            max_inconsistency=max_inc,
            mean_inconsistency=mean_inc,
            per_separator=per_separator,
            per_pair=per_pair,
            metric_used=self.metric,
            n_violations=n_violations,
        )

    def compute_inconsistency(
        self, marginal1: np.ndarray, marginal2: np.ndarray
    ) -> float:
        """
        Compute inconsistency between two marginal distributions.

        Args:
            marginal1: First marginal distribution (probability vector).
            marginal2: Second marginal distribution (probability vector).

        Returns:
            Distance between the marginals.
        """
        p = self._normalize(marginal1)
        q = self._normalize(marginal2)

        if self.metric == "tv":
            return self._total_variation(p, q)
        elif self.metric == "kl":
            return self._kl_divergence(p, q)
        elif self.metric == "hellinger":
            return self._hellinger_distance(p, q)
        elif self.metric == "js":
            return self._jensen_shannon(p, q)
        else:
            return self._total_variation(p, q)

    def repair_marginals(
        self,
        subgraph_marginals: Dict[int, Dict[int, np.ndarray]],
        separators: List[SeparatorSpec],
    ) -> RepairResult:
        """
        Repair inconsistent marginals by projecting to consistent space.

        Uses iterative averaging with step size control to find the nearest
        consistent set of marginals. For each separator, compute the average
        marginal and move each subgraph's marginal toward it.

        Args:
            subgraph_marginals: {subgraph_id: {separator_id: marginal_array}}
            separators: List of separator specifications.

        Returns:
            RepairResult with repaired marginals.
        """
        repaired = self._deep_copy_marginals(subgraph_marginals)
        per_sep_adjustment: Dict[int, float] = {}
        converged = False
        total_adj = 0.0

        for it in range(self.max_repair_iterations):
            max_change = 0.0

            for sep in separators:
                adj = sep.adjacent_subgraphs
                if len(adj) < 2:
                    continue

                # Collect current marginals for this separator
                current = []
                valid_sgs = []
                for sg in adj:
                    m = self._get_marginal(repaired, sg, sep.separator_id)
                    if m is not None:
                        current.append(self._normalize(m))
                        valid_sgs.append(sg)

                if len(current) < 2:
                    continue

                # Compute target: the average marginal
                max_len = max(len(c) for c in current)
                aligned = [self._pad_or_trim(c, max_len) for c in current]
                target = np.mean(aligned, axis=0)
                target = self._normalize(target)

                # Move each marginal toward the target
                sep_adj = 0.0
                for idx, sg in enumerate(valid_sgs):
                    old = aligned[idx]
                    new = (1 - self.repair_step_size) * old + self.repair_step_size * target
                    new = self._normalize(new)

                    change = float(np.sum(np.abs(new - old)))
                    max_change = max(max_change, change)
                    sep_adj += change

                    if sg not in repaired:
                        repaired[sg] = {}
                    repaired[sg][sep.separator_id] = new

                per_sep_adjustment[sep.separator_id] = sep_adj
                total_adj += sep_adj

            if max_change < self.tolerance:
                converged = True
                break

        return RepairResult(
            repaired_marginals=repaired,
            total_adjustment=total_adj,
            per_separator_adjustment=per_sep_adjustment,
            n_iterations=it + 1,
            converged=converged,
        )

    def project_to_consistent(
        self,
        marginals: Dict[int, Dict[int, np.ndarray]],
        separators: List[SeparatorSpec],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Project marginals to the nearest consistent point via optimization.

        Solves: min sum_{(i,j) in boundaries} ||p_i - q||^2 + ||p_j - q||^2
        s.t.   q is a valid probability distribution
               q = average(p_i, p_j) at each separator

        This is the I-projection (information projection) onto the consistent
        polytope.

        Args:
            marginals: {subgraph_id: {separator_id: marginal_array}}
            separators: Separator specifications.

        Returns:
            Projected marginals that are globally consistent.
        """
        projected = self._deep_copy_marginals(marginals)

        for sep in separators:
            adj = sep.adjacent_subgraphs
            if len(adj) < 2:
                continue

            current = []
            valid_sgs = []
            for sg in adj:
                m = self._get_marginal(marginals, sg, sep.separator_id)
                if m is not None:
                    current.append(self._normalize(m))
                    valid_sgs.append(sg)

            if len(current) < 2:
                continue

            max_len = max(len(c) for c in current)
            aligned = [self._pad_or_trim(c, max_len) for c in current]
            n_distributions = len(aligned)
            n_bins = max_len

            # Solve the projection via QP
            # Variables: q (consensus distribution, n_bins)
            # Minimize: sum_i ||aligned_i - q||^2
            # s.t.: sum(q) = 1, q >= 0

            stacked = np.array(aligned)
            mean_dist = np.mean(stacked, axis=0)

            # The optimal q minimizing sum of squared distances is the mean
            # projected onto the simplex
            q_star = self._project_to_simplex(mean_dist)

            for idx, sg in enumerate(valid_sgs):
                if sg not in projected:
                    projected[sg] = {}
                projected[sg][sep.separator_id] = q_star.copy()

        return projected

    def compute_joint_inconsistency(
        self,
        subgraph_marginals: Dict[int, Dict[int, np.ndarray]],
        separators: List[SeparatorSpec],
    ) -> float:
        """
        Compute the total joint inconsistency across all separators.

        This is the sum of pairwise inconsistencies, normalized by the
        number of separator boundaries.
        """
        total = 0.0
        count = 0

        for sep in separators:
            adj = sep.adjacent_subgraphs
            for i_idx in range(len(adj)):
                for j_idx in range(i_idx + 1, len(adj)):
                    sg_i, sg_j = adj[i_idx], adj[j_idx]
                    m_i = self._get_marginal(subgraph_marginals, sg_i, sep.separator_id)
                    m_j = self._get_marginal(subgraph_marginals, sg_j, sep.separator_id)

                    if m_i is not None and m_j is not None:
                        m_i, m_j = self._align_marginals(m_i, m_j)
                        total += self.compute_inconsistency(m_i, m_j)
                        count += 1

        return total / max(count, 1)

    def gradient_of_inconsistency(
        self,
        subgraph_marginals: Dict[int, Dict[int, np.ndarray]],
        separators: List[SeparatorSpec],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Compute the gradient of the total inconsistency w.r.t. each marginal.

        For TV distance, the gradient is the sign of (p - q) at each separator.
        For KL divergence, the gradient is log(p/q) + 1.
        """
        gradients: Dict[int, Dict[int, np.ndarray]] = {}

        for sep in separators:
            adj = sep.adjacent_subgraphs
            if len(adj) < 2:
                continue

            current = []
            valid_sgs = []
            for sg in adj:
                m = self._get_marginal(subgraph_marginals, sg, sep.separator_id)
                if m is not None:
                    current.append(self._normalize(m))
                    valid_sgs.append(sg)

            if len(current) < 2:
                continue

            max_len = max(len(c) for c in current)
            aligned = [self._pad_or_trim(c, max_len) for c in current]
            mean_m = np.mean(aligned, axis=0)

            for idx, sg in enumerate(valid_sgs):
                p = aligned[idx]
                if self.metric == "tv":
                    grad = np.sign(p - mean_m) / len(valid_sgs)
                elif self.metric == "kl":
                    safe_mean = np.maximum(mean_m, EPSILON)
                    safe_p = np.maximum(p, EPSILON)
                    grad = (np.log(safe_p / safe_mean) + 1) / len(valid_sgs)
                elif self.metric == "hellinger":
                    safe_p = np.maximum(p, EPSILON)
                    safe_mean = np.maximum(mean_m, EPSILON)
                    grad = (1.0 - np.sqrt(safe_mean / safe_p)) / (
                        2.0 * len(valid_sgs)
                    )
                else:
                    grad = (p - mean_m) / len(valid_sgs)

                if sg not in gradients:
                    gradients[sg] = {}
                gradients[sg][sep.separator_id] = grad

        return gradients

    # ----------------------------------------------------------------
    # Distance metrics
    # ----------------------------------------------------------------

    @staticmethod
    def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
        """Total variation distance: TV(p,q) = 0.5 * sum |p_i - q_i|."""
        return 0.5 * float(np.sum(np.abs(p - q)))

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence: KL(p||q) = sum p_i * log(p_i / q_i)."""
        p_safe = np.maximum(p, EPSILON)
        q_safe = np.maximum(q, EPSILON)
        return float(np.sum(rel_entr(p_safe, q_safe)))

    @staticmethod
    def _hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Hellinger distance: H(p,q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2."""
        return float(
            np.sqrt(0.5 * np.sum((np.sqrt(np.maximum(p, 0)) - np.sqrt(np.maximum(q, 0))) ** 2))
        )

    @staticmethod
    def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence (symmetric KL)."""
        p_safe = np.maximum(p, EPSILON)
        q_safe = np.maximum(q, EPSILON)
        return float(jensenshannon(p_safe, q_safe) ** 2)

    # ----------------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------------

    @staticmethod
    def _normalize(p: np.ndarray) -> np.ndarray:
        """Normalize to a valid probability distribution."""
        p = np.maximum(p, 0.0)
        s = np.sum(p)
        if s < EPSILON:
            return np.ones_like(p) / len(p)
        return p / s

    @staticmethod
    def _align_marginals(
        m1: np.ndarray, m2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure two marginals have the same length by padding shorter one."""
        if len(m1) == len(m2):
            return m1, m2
        max_len = max(len(m1), len(m2))
        p1 = np.zeros(max_len)
        p2 = np.zeros(max_len)
        p1[: len(m1)] = m1
        p2[: len(m2)] = m2
        return p1, p2

    @staticmethod
    def _pad_or_trim(p: np.ndarray, length: int) -> np.ndarray:
        """Pad or trim a distribution to the given length."""
        if len(p) >= length:
            return p[:length]
        result = np.zeros(length)
        result[: len(p)] = p
        return result

    @staticmethod
    def _get_marginal(
        marginals: Dict[int, Dict[int, np.ndarray]],
        subgraph_id: int,
        separator_id: int,
    ) -> Optional[np.ndarray]:
        """Safely get a marginal distribution."""
        sg_data = marginals.get(subgraph_id)
        if sg_data is None:
            return None
        return sg_data.get(separator_id)

    @staticmethod
    def _deep_copy_marginals(
        marginals: Dict[int, Dict[int, np.ndarray]]
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Deep copy the marginals dictionary."""
        return {
            sg: {sep: m.copy() for sep, m in seps.items()}
            for sg, seps in marginals.items()
        }

    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        """
        Project vector v onto the probability simplex.

        Uses the algorithm from "Efficient Projections onto the l1-Ball
        for Learning in High Dimensions" (Duchi et al., 2008).
        """
        n = len(v)
        if n == 0:
            return v

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho_candidates = u - cssv / (np.arange(1, n + 1))
        rho = int(np.max(np.where(rho_candidates > 0)[0])) if np.any(rho_candidates > 0) else 0
        theta = cssv[rho] / (rho + 1.0)
        return np.maximum(v - theta, 0.0)
