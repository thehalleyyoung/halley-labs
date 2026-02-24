"""Fairness-aware diversity selection.

Implements group fairness, individual fairness, calibrated fairness,
and fairness-aware mechanism design for diverse LLM response selection.

Mathematical foundations:
- Group fairness: |P(selected|g) - P(selected)| <= epsilon
- Individual fairness: d_selection(x,y) <= L * d_input(x,y)
- Envy-freeness: u_i(A_i) >= u_i(A_j) for all i, j
- Proportionality: u_i(A_i) >= v(N) / n for all i
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .kernels import Kernel, RBFKernel
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FairItem:
    """An item with group membership information."""
    index: int
    embedding: np.ndarray
    quality: float
    group: int  # group membership
    sensitive_attributes: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class FairnessReport:
    """Report on fairness properties of a selection."""
    group_fairness: Dict[str, float]
    individual_fairness: Dict[str, float]
    selection: List[int]
    metadata: Dict = field(default_factory=dict)


@dataclass
class FairSelectionResult:
    """Result from a fairness-aware selection mechanism."""
    selected: List[int]
    quality: float
    diversity: float
    fairness_metrics: Dict[str, float]
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Group Fairness
# ---------------------------------------------------------------------------

class GroupFairness:
    """Group fairness constraints for diverse selection.

    Ensures proportional representation of different groups
    (topics, perspectives, demographic groups of content).
    """

    def __init__(
        self,
        items: List[FairItem],
        groups: Optional[Dict[int, List[int]]] = None,
    ):
        self.items = items
        self.n = len(items)
        if groups is not None:
            self.groups = groups
        else:
            self.groups = self._infer_groups()

    def _infer_groups(self) -> Dict[int, List[int]]:
        """Infer groups from item group labels."""
        groups: Dict[int, List[int]] = {}
        for item in self.items:
            g = item.group
            if g not in groups:
                groups[g] = []
            groups[g].append(item.index)
        return groups

    def proportional_selection(
        self,
        k: int,
        quality_weight: float = 0.5,
        kernel: Optional[Kernel] = None,
    ) -> FairSelectionResult:
        """Select k items with proportional group representation.

        Each group gets k * |group| / n items (rounded to nearest integer).
        Within each group, select by quality-diversity tradeoff.
        """
        kernel = kernel or RBFKernel(bandwidth=1.0)
        total = sum(len(g) for g in self.groups.values())
        quotas: Dict[int, int] = {}
        assigned = 0
        for gid, members in sorted(self.groups.items(), key=lambda x: len(x[1])):
            quota = max(1, round(k * len(members) / total))
            quotas[gid] = min(quota, len(members))
            assigned += quotas[gid]

        # Adjust for rounding
        while assigned > k:
            for gid in sorted(quotas, key=lambda g: quotas[g], reverse=True):
                if quotas[gid] > 1:
                    quotas[gid] -= 1
                    assigned -= 1
                if assigned <= k:
                    break
        while assigned < k:
            for gid in sorted(quotas, key=lambda g: len(self.groups[g]) - quotas[g], reverse=True):
                if quotas[gid] < len(self.groups[gid]):
                    quotas[gid] += 1
                    assigned += 1
                if assigned >= k:
                    break

        # Select within each group
        selected: List[int] = []
        for gid, quota in quotas.items():
            members = self.groups[gid]
            if quota >= len(members):
                selected.extend(members)
                continue
            # Greedy quality-diversity within group
            group_selected = self._greedy_within_group(
                members, quota, quality_weight, kernel
            )
            selected.extend(group_selected)

        return self._build_result(selected, kernel, quotas)

    def _greedy_within_group(
        self,
        members: List[int],
        quota: int,
        quality_weight: float,
        kernel: Kernel,
    ) -> List[int]:
        """Greedy selection within a group."""
        selected: List[int] = []
        for _ in range(min(quota, len(members))):
            best_idx = -1
            best_score = -float("inf")
            for idx in members:
                if idx in selected:
                    continue
                quality = self.items[idx].quality
                div_gain = 0.0
                if len(selected) > 0:
                    emb = self.items[idx].embedding
                    min_dist = min(
                        np.linalg.norm(emb - self.items[s].embedding)
                        for s in selected
                    )
                    div_gain = min_dist
                score = quality_weight * quality + (1 - quality_weight) * div_gain
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
        return selected

    def _build_result(
        self,
        selected: List[int],
        kernel: Kernel,
        quotas: Optional[Dict[int, int]] = None,
    ) -> FairSelectionResult:
        """Build result with fairness metrics."""
        quality = float(np.mean([self.items[i].quality for i in selected])) if selected else 0.0
        diversity = 0.0
        if len(selected) >= 2:
            embs = np.array([self.items[i].embedding for i in selected])
            K = kernel.gram_matrix(embs)
            diversity = log_det_safe(K)

        fairness_metrics = self.evaluate_fairness(selected)
        return FairSelectionResult(
            selected=selected,
            quality=quality,
            diversity=diversity,
            fairness_metrics=fairness_metrics,
            metadata={"quotas": quotas},
        )

    def evaluate_fairness(self, selected: List[int]) -> Dict[str, float]:
        """Evaluate group fairness metrics."""
        if len(selected) == 0:
            return {"demographic_parity": 0.0, "max_group_deviation": 0.0}
        k = len(selected)
        group_counts: Dict[int, int] = {}
        for idx in selected:
            g = self.items[idx].group
            group_counts[g] = group_counts.get(g, 0) + 1

        # Demographic parity: selection rate per group
        deviations = []
        expected_rate = k / self.n
        for gid, members in self.groups.items():
            actual_rate = group_counts.get(gid, 0) / max(len(members), 1)
            deviations.append(abs(actual_rate - expected_rate))

        max_dev = max(deviations) if deviations else 0.0
        mean_dev = float(np.mean(deviations)) if deviations else 0.0

        # Representation ratio
        rep_ratios = []
        for gid, members in self.groups.items():
            expected = k * len(members) / self.n
            actual = group_counts.get(gid, 0)
            if expected > 0:
                rep_ratios.append(actual / expected)
            else:
                rep_ratios.append(1.0)

        return {
            "demographic_parity_gap": mean_dev,
            "max_group_deviation": max_dev,
            "min_representation_ratio": float(min(rep_ratios)) if rep_ratios else 0.0,
            "max_representation_ratio": float(max(rep_ratios)) if rep_ratios else 0.0,
            "group_counts": {str(k): v for k, v in group_counts.items()},
        }

    def statistical_parity(
        self,
        selected: List[int],
        threshold: float = 0.1,
    ) -> bool:
        """Check if selection satisfies statistical parity.

        |P(selected|g) - P(selected)| <= threshold for all groups.
        """
        metrics = self.evaluate_fairness(selected)
        return metrics["max_group_deviation"] <= threshold


# ---------------------------------------------------------------------------
# Individual Fairness
# ---------------------------------------------------------------------------

class IndividualFairness:
    """Individual fairness: similar inputs get similar selection probabilities.

    Lipschitz constraint: |P(select x) - P(select y)| <= L * d(x, y)
    """

    def __init__(
        self,
        items: List[FairItem],
        lipschitz_constant: float = 1.0,
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        self.items = items
        self.n = len(items)
        self.L = lipschitz_constant
        self.distance_fn = distance_fn or self._euclidean_distance

    @staticmethod
    def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.linalg.norm(x - y))

    def compute_selection_probabilities(
        self,
        n_trials: int = 100,
        k: int = 5,
        selection_fn: Optional[Callable[[], List[int]]] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Estimate selection probabilities via Monte Carlo."""
        rng = np.random.RandomState(seed)
        counts = np.zeros(self.n)

        for _ in range(n_trials):
            if selection_fn is not None:
                selected = selection_fn()
            else:
                selected = rng.choice(self.n, size=min(k, self.n), replace=False).tolist()
            for idx in selected:
                counts[idx] += 1

        return counts / n_trials

    def check_individual_fairness(
        self,
        selection_probs: np.ndarray,
    ) -> Tuple[bool, List[Tuple[int, int, float]]]:
        """Check Lipschitz fairness constraint.

        Returns (is_fair, list of violations as (i, j, violation_amount)).
        """
        violations: List[Tuple[int, int, float]] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d_input = self.distance_fn(
                    self.items[i].embedding, self.items[j].embedding
                )
                d_selection = abs(selection_probs[i] - selection_probs[j])
                max_allowed = self.L * d_input
                if d_selection > max_allowed + 1e-8:
                    violations.append((i, j, d_selection - max_allowed))

        return len(violations) == 0, violations

    def fair_selection(
        self,
        k: int,
        quality_weight: float = 0.5,
        kernel: Optional[Kernel] = None,
        n_restarts: int = 10,
        seed: int = 42,
    ) -> FairSelectionResult:
        """Select k items satisfying individual fairness via soft constraint."""
        kernel = kernel or RBFKernel(bandwidth=1.0)
        rng = np.random.RandomState(seed)

        # Compute pairwise distances
        dists = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = self.distance_fn(self.items[i].embedding, self.items[j].embedding)
                dists[i, j] = d
                dists[j, i] = d

        best_selected: List[int] = []
        best_score = -float("inf")

        for restart in range(n_restarts):
            # Random initial selection
            selected = sorted(rng.choice(self.n, size=k, replace=False).tolist())

            # Local search with fairness penalty
            for _ in range(100):
                improved = False
                for pos in range(k):
                    best_swap = -1
                    best_swap_score = -float("inf")

                    for candidate in range(self.n):
                        if candidate in selected:
                            continue
                        trial = selected.copy()
                        trial[pos] = candidate
                        trial.sort()

                        # Quality
                        q = np.mean([self.items[i].quality for i in trial])
                        # Diversity
                        embs = np.array([self.items[i].embedding for i in trial])
                        K = kernel.gram_matrix(embs)
                        div = log_det_safe(K)

                        # Fairness penalty: similar items should have similar
                        # selection status. Penalize selecting item far from
                        # unselected similar items.
                        fairness_penalty = 0.0
                        for i in trial:
                            for j in range(self.n):
                                if j in trial:
                                    continue
                                d = dists[i, j]
                                if d < 0.5:  # Similar items
                                    fairness_penalty += (1 - d) * 0.1

                        score = (quality_weight * q
                                 + (1 - quality_weight) * div
                                 - fairness_penalty)

                        if score > best_swap_score:
                            best_swap_score = score
                            best_swap = candidate

                    if best_swap >= 0 and best_swap_score > self._score_selection(
                        selected, quality_weight, kernel, dists
                    ):
                        selected[pos] = best_swap
                        selected.sort()
                        improved = True

                if not improved:
                    break

            score = self._score_selection(selected, quality_weight, kernel, dists)
            if score > best_score:
                best_score = score
                best_selected = selected

        quality = float(np.mean([self.items[i].quality for i in best_selected]))
        embs = np.array([self.items[i].embedding for i in best_selected])
        K = kernel.gram_matrix(embs)
        diversity = log_det_safe(K)

        return FairSelectionResult(
            selected=best_selected,
            quality=quality,
            diversity=diversity,
            fairness_metrics={"fairness_penalty": -best_score + quality + diversity},
        )

    def _score_selection(
        self,
        selected: List[int],
        quality_weight: float,
        kernel: Kernel,
        dists: np.ndarray,
    ) -> float:
        q = np.mean([self.items[i].quality for i in selected])
        embs = np.array([self.items[i].embedding for i in selected])
        K = kernel.gram_matrix(embs)
        div = log_det_safe(K)
        penalty = 0.0
        for i in selected:
            for j in range(self.n):
                if j in selected:
                    continue
                d = dists[i, j]
                if d < 0.5:
                    penalty += (1 - d) * 0.1
        return quality_weight * q + (1 - quality_weight) * div - penalty


# ---------------------------------------------------------------------------
# Calibrated Fairness
# ---------------------------------------------------------------------------

class CalibratedFairness:
    """Calibrated fairness: quality-adjusted representation.

    Higher quality groups should have proportionally higher representation,
    but no group should be completely excluded.
    """

    def __init__(
        self,
        items: List[FairItem],
        groups: Optional[Dict[int, List[int]]] = None,
        min_representation: float = 0.05,
    ):
        self.items = items
        self.n = len(items)
        self.min_rep = min_representation
        if groups is not None:
            self.groups = groups
        else:
            self.groups = {}
            for item in items:
                g = item.group
                if g not in self.groups:
                    self.groups[g] = []
                self.groups[g].append(item.index)

    def _compute_quality_adjusted_quotas(
        self, k: int, alpha: float = 0.5,
    ) -> Dict[int, int]:
        """Compute quotas balancing group size and quality.

        quota_g = k * (alpha * |g|/n + (1-alpha) * mean_quality_g / sum_quality)
        """
        n_groups = len(self.groups)
        group_qualities: Dict[int, float] = {}
        for gid, members in self.groups.items():
            group_qualities[gid] = float(np.mean([
                self.items[i].quality for i in members
            ]))

        total_quality = sum(group_qualities.values())
        quotas: Dict[int, int] = {}
        assigned = 0

        for gid, members in self.groups.items():
            size_fraction = len(members) / self.n
            quality_fraction = (
                group_qualities[gid] / total_quality if total_quality > 0 else 1 / n_groups
            )
            target = k * (alpha * size_fraction + (1 - alpha) * quality_fraction)
            quota = max(
                max(1, int(round(k * self.min_rep))),
                min(int(round(target)), len(members)),
            )
            quotas[gid] = quota
            assigned += quota

        # Normalize
        while assigned > k:
            for gid in sorted(quotas, key=lambda g: quotas[g], reverse=True):
                if quotas[gid] > max(1, int(k * self.min_rep)):
                    quotas[gid] -= 1
                    assigned -= 1
                if assigned <= k:
                    break
        while assigned < k:
            for gid in sorted(
                quotas,
                key=lambda g: len(self.groups[g]) - quotas[g],
                reverse=True,
            ):
                if quotas[gid] < len(self.groups[gid]):
                    quotas[gid] += 1
                    assigned += 1
                if assigned >= k:
                    break

        return quotas

    def select(
        self,
        k: int,
        alpha: float = 0.5,
        kernel: Optional[Kernel] = None,
    ) -> FairSelectionResult:
        """Select with calibrated fairness."""
        kernel = kernel or RBFKernel(bandwidth=1.0)
        quotas = self._compute_quality_adjusted_quotas(k, alpha)

        selected: List[int] = []
        for gid, quota in quotas.items():
            members = self.groups[gid]
            # Select top quality within group
            member_qualities = [(idx, self.items[idx].quality) for idx in members]
            member_qualities.sort(key=lambda x: x[1], reverse=True)
            group_selected = [idx for idx, _ in member_qualities[:quota]]
            selected.extend(group_selected)

        quality = float(np.mean([self.items[i].quality for i in selected])) if selected else 0.0
        diversity = 0.0
        if len(selected) >= 2:
            embs = np.array([self.items[i].embedding for i in selected])
            K = kernel.gram_matrix(embs)
            diversity = log_det_safe(K)

        group_counts: Dict[int, int] = {}
        for idx in selected:
            g = self.items[idx].group
            group_counts[g] = group_counts.get(g, 0) + 1

        return FairSelectionResult(
            selected=selected,
            quality=quality,
            diversity=diversity,
            fairness_metrics={
                "quotas": {str(k): v for k, v in quotas.items()},
                "group_counts": {str(k): v for k, v in group_counts.items()},
                "calibration_alpha": alpha,
            },
        )


# ---------------------------------------------------------------------------
# Fair Division of Diversity Budget
# ---------------------------------------------------------------------------

class FairDivision:
    """Fair division of diversity budget across groups.

    Implements:
    - Proportional allocation
    - Maximin share
    - Competitive equilibrium from equal incomes (CEEI)
    """

    def __init__(
        self,
        items: List[FairItem],
        groups: Dict[int, List[int]],
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.n = len(items)
        self.groups = groups
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def proportional_division(
        self,
        k: int,
    ) -> Dict[int, List[int]]:
        """Proportional fair division: each group gets k/n_groups items."""
        n_groups = len(self.groups)
        per_group = max(1, k // n_groups)

        allocation: Dict[int, List[int]] = {}
        used: Set[int] = set()

        for gid, members in self.groups.items():
            group_items = [i for i in members if i not in used]
            group_items.sort(key=lambda i: self.items[i].quality, reverse=True)
            selected = group_items[:per_group]
            allocation[gid] = selected
            used.update(selected)

        return allocation

    def maximin_share(
        self,
        k: int,
    ) -> Dict[int, float]:
        """Compute maximin share for each group.

        MMS_g = max over all partitions of items into |G| bundles, min value of a bundle.
        Approximated via greedy round-robin.
        """
        n_groups = len(self.groups)
        items_by_quality = sorted(range(self.n), key=lambda i: self.items[i].quality, reverse=True)

        # Round-robin allocation
        bundles: Dict[int, List[int]] = {gid: [] for gid in self.groups}
        group_ids = list(self.groups.keys())
        idx = 0
        for i, item_idx in enumerate(items_by_quality[:k]):
            gid = group_ids[idx % n_groups]
            bundles[gid].append(item_idx)
            idx += 1

        # Compute values
        mms: Dict[int, float] = {}
        for gid, bundle in bundles.items():
            if len(bundle) == 0:
                mms[gid] = 0.0
                continue
            quality = float(np.mean([self.items[i].quality for i in bundle]))
            div = 0.0
            if len(bundle) >= 2:
                embs = np.array([self.items[i].embedding for i in bundle])
                K = self.kernel.gram_matrix(embs)
                div = log_det_safe(K)
            mms[gid] = 0.5 * quality + 0.5 * max(div, 0)

        return mms

    def ceei_approximation(
        self,
        k: int,
        n_iterations: int = 100,
        seed: int = 42,
    ) -> Dict[int, List[int]]:
        """Approximate CEEI (Competitive Equilibrium from Equal Incomes).

        Each group has budget 1. Items have prices. Groups buy items.
        """
        rng = np.random.RandomState(seed)
        n_groups = len(self.groups)

        # Initialize prices
        prices = np.ones(self.n) * 0.1
        budgets = {gid: 1.0 for gid in self.groups}

        for iteration in range(n_iterations):
            # Each group demands items within budget
            demands: Dict[int, List[int]] = {}
            for gid in self.groups:
                budget = budgets[gid]
                # Value / price ratio
                ratios = []
                for i in range(self.n):
                    val = self.items[i].quality
                    ratios.append((i, val / max(prices[i], 1e-12)))
                ratios.sort(key=lambda x: x[1], reverse=True)

                selected = []
                spent = 0.0
                for item_idx, ratio in ratios:
                    if spent + prices[item_idx] <= budget and len(selected) < k // n_groups + 1:
                        selected.append(item_idx)
                        spent += prices[item_idx]
                demands[gid] = selected

            # Update prices: excess demand increases price
            demand_count = np.zeros(self.n)
            for selected in demands.values():
                for idx in selected:
                    demand_count[idx] += 1

            for i in range(self.n):
                if demand_count[i] > 1:
                    prices[i] *= 1.1
                elif demand_count[i] == 0:
                    prices[i] *= 0.9

        # Final allocation: resolve conflicts
        allocation: Dict[int, List[int]] = {}
        used: Set[int] = set()
        for gid in sorted(self.groups.keys()):
            selected = [i for i in demands.get(gid, []) if i not in used]
            per_group = max(1, k // n_groups)
            selected = selected[:per_group]
            allocation[gid] = selected
            used.update(selected)

        return allocation


# ---------------------------------------------------------------------------
# Envy-Free Selection
# ---------------------------------------------------------------------------

class EnvyFreeSelection:
    """Envy-free diverse selection mechanism.

    No group should prefer another group's allocation.
    """

    def __init__(
        self,
        items: List[FairItem],
        groups: Dict[int, List[int]],
        value_fn: Optional[Callable[[List[int]], float]] = None,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.n = len(items)
        self.groups = groups
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        if value_fn is not None:
            self.value_fn = value_fn
        else:
            self.value_fn = self._default_value

    def _default_value(self, indices: List[int]) -> float:
        if len(indices) == 0:
            return 0.0
        quality = float(np.mean([self.items[i].quality for i in indices]))
        div = 0.0
        if len(indices) >= 2:
            embs = np.array([self.items[i].embedding for i in indices])
            K = self.kernel.gram_matrix(embs)
            div = log_det_safe(K)
        return 0.5 * quality + 0.5 * max(div, 0)

    def check_envy_free(
        self,
        allocation: Dict[int, List[int]],
    ) -> Tuple[bool, List[Tuple[int, int, float]]]:
        """Check if allocation is envy-free.

        Returns (is_ef, list of (envying_group, envied_group, envy_amount)).
        """
        envies: List[Tuple[int, int, float]] = []
        for g1 in self.groups:
            v_own = self.value_fn(allocation.get(g1, []))
            for g2 in self.groups:
                if g1 == g2:
                    continue
                v_other = self.value_fn(allocation.get(g2, []))
                if v_other > v_own + 1e-8:
                    envies.append((g1, g2, v_other - v_own))
        return len(envies) == 0, envies

    def find_ef_allocation(
        self,
        k: int,
        n_restarts: int = 20,
        seed: int = 42,
    ) -> Tuple[Dict[int, List[int]], bool]:
        """Find an envy-free allocation via local search."""
        rng = np.random.RandomState(seed)
        n_groups = len(self.groups)
        per_group = max(1, k // n_groups)

        best_allocation: Dict[int, List[int]] = {}
        best_envy = float("inf")

        for restart in range(n_restarts):
            # Random partition
            items = rng.permutation(self.n).tolist()
            allocation: Dict[int, List[int]] = {}
            idx = 0
            for gid in self.groups:
                allocation[gid] = items[idx:idx + per_group]
                idx += per_group

            # Local search to reduce envy
            for _ in range(200):
                is_ef, envies = self.check_envy_free(allocation)
                if is_ef:
                    return allocation, True

                if len(envies) == 0:
                    break
                # Try to reduce worst envy by swapping
                g_envy, g_envied, amount = max(envies, key=lambda x: x[2])
                if allocation.get(g_envy) and allocation.get(g_envied):
                    # Swap an item
                    i1 = rng.choice(allocation[g_envy])
                    i2 = rng.choice(allocation[g_envied])
                    new_alloc = {g: list(items) for g, items in allocation.items()}
                    new_alloc[g_envy].remove(i1)
                    new_alloc[g_envy].append(i2)
                    new_alloc[g_envied].remove(i2)
                    new_alloc[g_envied].append(i1)

                    _, new_envies = self.check_envy_free(new_alloc)
                    new_envy_total = sum(e for _, _, e in new_envies)
                    old_envy_total = sum(e for _, _, e in envies)
                    if new_envy_total < old_envy_total:
                        allocation = new_alloc

                total_envy = sum(e for _, _, e in envies)
                if total_envy < best_envy:
                    best_envy = total_envy
                    best_allocation = {g: list(items) for g, items in allocation.items()}

        is_ef, _ = self.check_envy_free(best_allocation)
        return best_allocation, is_ef

    def ef1_allocation(
        self,
        k: int,
        seed: int = 42,
    ) -> Dict[int, List[int]]:
        """EF1 (envy-free up to one item) allocation via round-robin.

        Guarantees that removing at most one item from the envied bundle
        eliminates envy.
        """
        rng = np.random.RandomState(seed)
        group_ids = list(self.groups.keys())
        n_groups = len(group_ids)

        # Sort items by quality
        sorted_items = sorted(range(self.n), key=lambda i: self.items[i].quality, reverse=True)

        allocation: Dict[int, List[int]] = {gid: [] for gid in group_ids}
        per_group = max(1, k // n_groups)

        # Round-robin with quality ordering
        idx = 0
        total_assigned = 0
        while total_assigned < k and idx < len(sorted_items):
            for gid in group_ids:
                if idx >= len(sorted_items) or total_assigned >= k:
                    break
                if len(allocation[gid]) >= per_group:
                    continue
                allocation[gid].append(sorted_items[idx])
                idx += 1
                total_assigned += 1

        return allocation


# ---------------------------------------------------------------------------
# Proportionality Guarantees
# ---------------------------------------------------------------------------

class ProportionalityGuarantees:
    """Check and enforce proportionality in diversity selection."""

    def __init__(
        self,
        items: List[FairItem],
        groups: Dict[int, List[int]],
        value_fn: Optional[Callable[[List[int]], float]] = None,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.n = len(items)
        self.groups = groups
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        if value_fn is not None:
            self.value_fn = value_fn
        else:
            self.value_fn = self._default_value

    def _default_value(self, indices: List[int]) -> float:
        if len(indices) == 0:
            return 0.0
        quality = float(np.mean([self.items[i].quality for i in indices]))
        return quality

    def check_proportionality(
        self,
        allocation: Dict[int, List[int]],
    ) -> Dict[str, float]:
        """Check proportionality: each group gets >= v(N)/n_groups.

        Returns per-group proportionality ratios.
        """
        all_items = list(range(self.n))
        total_value = self.value_fn(all_items)
        n_groups = len(self.groups)
        fair_share = total_value / n_groups

        ratios: Dict[str, float] = {}
        for gid, bundle in allocation.items():
            bundle_value = self.value_fn(bundle)
            ratio = bundle_value / max(fair_share, 1e-12)
            ratios[f"group_{gid}_ratio"] = ratio

        ratios["min_ratio"] = min(ratios.values()) if ratios else 0.0
        ratios["fair_share"] = fair_share
        return ratios

    def proportional_up_to_one(
        self,
        allocation: Dict[int, List[int]],
    ) -> bool:
        """Check PROP1: each group's value >= fair share - max single item value."""
        all_items = list(range(self.n))
        total_value = self.value_fn(all_items)
        n_groups = len(self.groups)
        fair_share = total_value / n_groups

        max_single = max(self.value_fn([i]) for i in range(self.n))

        for gid, bundle in allocation.items():
            bundle_value = self.value_fn(bundle)
            if bundle_value < fair_share - max_single - 1e-8:
                return False
        return True


# ---------------------------------------------------------------------------
# Comprehensive fairness evaluation
# ---------------------------------------------------------------------------

class FairnessEvaluator:
    """Evaluate all fairness criteria for a diversity selection."""

    def __init__(
        self,
        items: List[FairItem],
        groups: Dict[int, List[int]],
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.groups = groups
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def full_evaluation(
        self,
        selected: List[int],
        allocation: Optional[Dict[int, List[int]]] = None,
    ) -> FairnessReport:
        """Run all fairness checks."""
        # Group fairness
        gf = GroupFairness(self.items, self.groups)
        group_metrics = gf.evaluate_fairness(selected)

        # Individual fairness
        ifair = IndividualFairness(self.items)
        probs = np.zeros(len(self.items))
        for idx in selected:
            probs[idx] = 1.0
        is_if, violations = ifair.check_individual_fairness(probs)
        individual_metrics = {
            "is_individually_fair": float(is_if),
            "n_violations": len(violations),
            "max_violation": float(max(v for _, _, v in violations)) if violations else 0.0,
        }

        # Envy-freeness (if allocation provided)
        if allocation is not None:
            ef = EnvyFreeSelection(self.items, self.groups, kernel=self.kernel)
            is_ef, envies = ef.check_envy_free(allocation)
            group_metrics["envy_free"] = float(is_ef)
            group_metrics["total_envy"] = sum(e for _, _, e in envies)

            # Proportionality
            pg = ProportionalityGuarantees(self.items, self.groups, kernel=self.kernel)
            prop_metrics = pg.check_proportionality(allocation)
            group_metrics.update(prop_metrics)
            group_metrics["prop1"] = float(pg.proportional_up_to_one(allocation))

        return FairnessReport(
            group_fairness=group_metrics,
            individual_fairness=individual_metrics,
            selection=selected,
        )


# ---------------------------------------------------------------------------
# Fair diversity selection pipeline
# ---------------------------------------------------------------------------

def fair_diverse_selection(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    group_labels: np.ndarray,
    k: int = 5,
    method: str = "proportional",
    quality_weight: float = 0.5,
    kernel: Optional[Kernel] = None,
    seed: int = 42,
    **kwargs,
) -> FairSelectionResult:
    """End-to-end fair diverse selection.

    Args:
        embeddings: (n, d) embeddings
        quality_scores: (n,) quality scores
        group_labels: (n,) group membership
        k: number to select
        method: "proportional", "calibrated", "ef1", "individual"
        quality_weight: weight for quality vs diversity
        kernel: diversity kernel
    """
    n = len(embeddings)
    items = [
        FairItem(
            index=i,
            embedding=embeddings[i],
            quality=float(quality_scores[i]),
            group=int(group_labels[i]),
        )
        for i in range(n)
    ]
    groups: Dict[int, List[int]] = {}
    for item in items:
        if item.group not in groups:
            groups[item.group] = []
        groups[item.group].append(item.index)

    kernel = kernel or RBFKernel(bandwidth=1.0)

    if method == "proportional":
        gf = GroupFairness(items, groups)
        return gf.proportional_selection(k, quality_weight, kernel)
    elif method == "calibrated":
        cf = CalibratedFairness(items, groups)
        return cf.select(k, alpha=kwargs.get("alpha", 0.5), kernel=kernel)
    elif method == "ef1":
        ef = EnvyFreeSelection(items, groups, kernel=kernel)
        allocation = ef.ef1_allocation(k, seed=seed)
        selected = []
        for bundle in allocation.values():
            selected.extend(bundle)
        quality = float(np.mean([items[i].quality for i in selected])) if selected else 0.0
        div = 0.0
        if len(selected) >= 2:
            embs = np.array([items[i].embedding for i in selected])
            K = kernel.gram_matrix(embs)
            div = log_det_safe(K)
        evaluator = FairnessEvaluator(items, groups, kernel)
        report = evaluator.full_evaluation(selected, allocation)
        return FairSelectionResult(
            selected=selected,
            quality=quality,
            diversity=div,
            fairness_metrics={**report.group_fairness, **report.individual_fairness},
            metadata={"allocation": allocation},
        )
    elif method == "individual":
        ifair = IndividualFairness(items)
        return ifair.fair_selection(k, quality_weight, kernel, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Demographic Parity Optimizer
# ---------------------------------------------------------------------------

class DemographicParityOptimizer:
    """Optimize selection subject to demographic parity constraints.

    Ensures selection rates are equal across groups while maximizing
    quality-diversity objective.
    """

    def __init__(
        self,
        items: List[FairItem],
        groups: Dict[int, List[int]],
        parity_threshold: float = 0.1,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.n = len(items)
        self.groups = groups
        self.parity_threshold = parity_threshold
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _compute_parity_violation(
        self,
        selected: List[int],
    ) -> float:
        """Compute demographic parity violation."""
        k = len(selected)
        if k == 0:
            return 0.0
        rates = []
        for gid, members in self.groups.items():
            n_selected = sum(1 for i in selected if i in members)
            rate = n_selected / max(len(members), 1)
            rates.append(rate)
        return max(rates) - min(rates) if rates else 0.0

    def optimize(
        self,
        k: int,
        quality_weight: float = 0.5,
        n_restarts: int = 20,
        seed: int = 42,
    ) -> FairSelectionResult:
        """Lagrangian optimization for DP-constrained selection."""
        rng = np.random.RandomState(seed)
        best_selected: List[int] = []
        best_objective = -float("inf")

        for restart in range(n_restarts):
            # Initialize
            selected = sorted(rng.choice(self.n, size=k, replace=False).tolist())
            lagrange_multiplier = 1.0

            for iteration in range(100):
                # Compute objective: quality + diversity - lambda * parity_violation
                quality = np.mean([self.items[i].quality for i in selected])
                if len(selected) >= 2:
                    embs = np.array([self.items[i].embedding for i in selected])
                    K = self.kernel.gram_matrix(embs)
                    diversity = log_det_safe(K)
                else:
                    diversity = 0.0
                violation = self._compute_parity_violation(selected)
                objective = (quality_weight * quality + (1 - quality_weight) * diversity
                             - lagrange_multiplier * violation)

                # Try swaps
                improved = False
                for pos in range(k):
                    for candidate in range(self.n):
                        if candidate in selected:
                            continue
                        trial = selected.copy()
                        trial[pos] = candidate
                        trial.sort()

                        q = np.mean([self.items[i].quality for i in trial])
                        if len(trial) >= 2:
                            embs = np.array([self.items[i].embedding for i in trial])
                            K_t = self.kernel.gram_matrix(embs)
                            d = log_det_safe(K_t)
                        else:
                            d = 0.0
                        v = self._compute_parity_violation(trial)
                        obj = quality_weight * q + (1 - quality_weight) * d - lagrange_multiplier * v

                        if obj > objective:
                            objective = obj
                            selected = trial
                            improved = True
                            break
                    if improved:
                        break

                # Update Lagrange multiplier
                if violation > self.parity_threshold:
                    lagrange_multiplier *= 1.1
                else:
                    lagrange_multiplier *= 0.95

                if not improved:
                    break

            if objective > best_objective:
                best_objective = objective
                best_selected = selected

        quality = float(np.mean([self.items[i].quality for i in best_selected]))
        diversity = 0.0
        if len(best_selected) >= 2:
            embs = np.array([self.items[i].embedding for i in best_selected])
            K = self.kernel.gram_matrix(embs)
            diversity = log_det_safe(K)

        gf = GroupFairness(self.items, self.groups)
        metrics = gf.evaluate_fairness(best_selected)

        return FairSelectionResult(
            selected=best_selected,
            quality=quality,
            diversity=diversity,
            fairness_metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Equalized Odds Selection
# ---------------------------------------------------------------------------

class EqualizedOddsSelection:
    """Selection satisfying equalized odds: selection rate conditioned on
    quality should be equal across groups.

    P(selected | quality=q, group=g) = P(selected | quality=q) for all g, q
    """

    def __init__(
        self,
        items: List[FairItem],
        groups: Dict[int, List[int]],
        n_quality_bins: int = 5,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.n = len(items)
        self.groups = groups
        self.n_bins = n_quality_bins
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _bin_items(self) -> Dict[int, List[int]]:
        """Bin items by quality."""
        qualities = np.array([item.quality for item in self.items])
        bin_edges = np.linspace(np.min(qualities), np.max(qualities) + 1e-8, self.n_bins + 1)
        bins: Dict[int, List[int]] = {b: [] for b in range(self.n_bins)}
        for i, q in enumerate(qualities):
            b = np.searchsorted(bin_edges[1:], q)
            b = min(b, self.n_bins - 1)
            bins[b].append(i)
        return bins

    def select(
        self,
        k: int,
        quality_weight: float = 0.5,
    ) -> FairSelectionResult:
        """Select k items with equalized odds constraint."""
        bins = self._bin_items()
        n_groups = len(self.groups)

        # For each quality bin, select proportionally from each group
        selected: List[int] = []
        per_bin = max(1, k // self.n_bins)

        for bin_id, bin_items in bins.items():
            if not bin_items:
                continue
            bin_selected: List[int] = []
            per_group = max(1, per_bin // n_groups)

            for gid, members in self.groups.items():
                group_bin_items = [i for i in bin_items if i in members]
                if not group_bin_items:
                    continue
                # Select top quality within group-bin
                group_bin_items.sort(key=lambda i: self.items[i].quality, reverse=True)
                bin_selected.extend(group_bin_items[:per_group])

            selected.extend(bin_selected)

        selected = selected[:k]
        quality = float(np.mean([self.items[i].quality for i in selected])) if selected else 0.0
        diversity = 0.0
        if len(selected) >= 2:
            embs = np.array([self.items[i].embedding for i in selected])
            K = self.kernel.gram_matrix(embs)
            diversity = log_det_safe(K)

        gf = GroupFairness(self.items, self.groups)
        metrics = gf.evaluate_fairness(selected)

        return FairSelectionResult(
            selected=selected,
            quality=quality,
            diversity=diversity,
            fairness_metrics=metrics,
            metadata={"method": "equalized_odds"},
        )


# ---------------------------------------------------------------------------
# Counterfactual Fairness
# ---------------------------------------------------------------------------

class CounterfactualFairness:
    """Counterfactual fairness: selection should not change if we counterfactually
    change the group membership of an item.

    P(selected | do(group=g)) = P(selected | do(group=g')) for all g, g'
    """

    def __init__(
        self,
        items: List[FairItem],
        groups: Dict[int, List[int]],
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.n = len(items)
        self.groups = groups
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _group_blind_score(self, item_idx: int) -> float:
        """Compute group-blind quality score."""
        return self.items[item_idx].quality

    def _diversity_contribution(
        self,
        item_idx: int,
        selected: List[int],
    ) -> float:
        """Group-blind diversity contribution."""
        if len(selected) == 0:
            return 0.0
        emb = self.items[item_idx].embedding
        return min(
            np.linalg.norm(emb - self.items[j].embedding)
            for j in selected
        )

    def select(
        self,
        k: int,
        quality_weight: float = 0.5,
    ) -> FairSelectionResult:
        """Counterfactually fair selection (group-blind greedy)."""
        selected: List[int] = []
        for _ in range(k):
            best_idx = -1
            best_score = -float("inf")
            for i in range(self.n):
                if i in selected:
                    continue
                q = self._group_blind_score(i)
                d = self._diversity_contribution(i, selected)
                score = quality_weight * q + (1 - quality_weight) * d
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx >= 0:
                selected.append(best_idx)

        quality = float(np.mean([self.items[i].quality for i in selected])) if selected else 0.0
        diversity = 0.0
        if len(selected) >= 2:
            embs = np.array([self.items[i].embedding for i in selected])
            K = self.kernel.gram_matrix(embs)
            diversity = log_det_safe(K)

        gf = GroupFairness(self.items, self.groups)
        metrics = gf.evaluate_fairness(selected)
        metrics["counterfactually_fair"] = True

        return FairSelectionResult(
            selected=selected,
            quality=quality,
            diversity=diversity,
            fairness_metrics=metrics,
            metadata={"method": "counterfactual_fairness"},
        )

    def evaluate_counterfactual(
        self,
        selection_fn: Callable[[List[FairItem]], List[int]],
    ) -> Dict[str, float]:
        """Evaluate counterfactual fairness of a selection function."""
        # Original selection
        original_selected = selection_fn(self.items)

        # For each item, counterfactually change its group
        changes = 0
        total_tests = 0
        for i in range(self.n):
            if i not in original_selected:
                continue
            original_group = self.items[i].group
            for target_group in self.groups:
                if target_group == original_group:
                    continue
                # Create counterfactual items
                cf_items = []
                for j in range(self.n):
                    if j == i:
                        cf_item = FairItem(
                            index=j,
                            embedding=self.items[j].embedding.copy(),
                            quality=self.items[j].quality,
                            group=target_group,
                        )
                    else:
                        cf_item = self.items[j]
                    cf_items.append(cf_item)

                cf_selected = selection_fn(cf_items)
                if (i in original_selected) != (i in cf_selected):
                    changes += 1
                total_tests += 1

        cf_violation_rate = changes / max(total_tests, 1)
        return {
            "counterfactual_violation_rate": cf_violation_rate,
            "n_tests": total_tests,
            "n_changes": changes,
        }


# ---------------------------------------------------------------------------
# Fairness comparison framework
# ---------------------------------------------------------------------------

class FairnessComparison:
    """Compare multiple fairness methods on the same data."""

    def __init__(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray,
        group_labels: np.ndarray,
        kernel: Optional[Kernel] = None,
    ):
        self.embeddings = embeddings
        self.quality = quality_scores
        self.groups_arr = group_labels
        self.n = len(embeddings)
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.items = make_fair_items_internal(embeddings, quality_scores, group_labels)
        self.groups = _build_groups(self.items)

    def compare(
        self,
        k: int = 10,
        seed: int = 42,
    ) -> Dict[str, FairSelectionResult]:
        """Compare all fairness methods."""
        results: Dict[str, FairSelectionResult] = {}

        for method in ["proportional", "calibrated", "ef1"]:
            try:
                results[method] = fair_diverse_selection(
                    self.embeddings, self.quality, self.groups_arr,
                    k=k, method=method, kernel=self.kernel, seed=seed,
                )
            except Exception:
                pass

        # Demographic parity
        try:
            dp = DemographicParityOptimizer(self.items, self.groups, kernel=self.kernel)
            results["demographic_parity"] = dp.optimize(k, seed=seed)
        except Exception:
            pass

        # Equalized odds
        try:
            eo = EqualizedOddsSelection(self.items, self.groups, kernel=self.kernel)
            results["equalized_odds"] = eo.select(k)
        except Exception:
            pass

        # Counterfactual
        try:
            cf = CounterfactualFairness(self.items, self.groups, kernel=self.kernel)
            results["counterfactual"] = cf.select(k)
        except Exception:
            pass

        return results


def make_fair_items_internal(
    embeddings: np.ndarray, quality: np.ndarray, groups: np.ndarray,
) -> List[FairItem]:
    return [
        FairItem(index=i, embedding=embeddings[i], quality=float(quality[i]), group=int(groups[i]))
        for i in range(len(embeddings))
    ]


def _build_groups(items: List[FairItem]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for item in items:
        groups.setdefault(item.group, []).append(item.index)
    return groups
