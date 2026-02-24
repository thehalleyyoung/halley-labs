"""Auction theory for diversity-aware subset selection.

Implements combinatorial auctions, multi-item mechanisms, and revenue-optimal
auctions (Myerson) adapted to the diversity elicitation setting. Items are
candidate LLM responses and bidders are quality/diversity aspects that compete
for representation in the final selected subset.

Mathematical foundations:
- Winner Determination Problem (WDP) solved via ILP
- VCG payments for incentive compatibility
- Myerson's optimal auction with virtual valuations
- English / sealed-bid / second-price auctions with diversity constraints
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

from .kernels import Kernel, RBFKernel
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BidderType(Enum):
    """Types of quality-aspect bidders."""
    COHERENCE = "coherence"
    NOVELTY = "novelty"
    RELEVANCE = "relevance"
    FLUENCY = "fluency"
    DIVERSITY = "diversity"
    COVERAGE = "coverage"
    SPECIFICITY = "specificity"
    INFORMATIVENESS = "informativeness"


@dataclass
class Item:
    """A candidate response (item in the auction)."""
    index: int
    embedding: np.ndarray
    quality_score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class Bid:
    """A bid placed by a bidder on a bundle of items."""
    bidder_id: int
    bundle: FrozenSet[int]  # set of item indices
    value: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class Bidder:
    """An aspect-bidder that values subsets of candidate responses."""
    bidder_id: int
    bidder_type: BidderType
    valuation_fn: Callable[[FrozenSet[int], List[Item]], float]
    budget: float = float("inf")
    _reported_bids: List[Bid] = field(default_factory=list)

    def compute_value(self, bundle: FrozenSet[int], items: List[Item]) -> float:
        return self.valuation_fn(bundle, items)

    def submit_bid(self, bundle: FrozenSet[int], items: List[Item]) -> Bid:
        value = self.compute_value(bundle, items)
        bid = Bid(bidder_id=self.bidder_id, bundle=bundle, value=value)
        self._reported_bids.append(bid)
        return bid


@dataclass
class AuctionResult:
    """Result of running an auction mechanism."""
    allocation: Dict[int, FrozenSet[int]]  # bidder -> allocated bundle
    payments: Dict[int, float]
    social_welfare: float
    revenue: float
    selected_items: Set[int]
    is_incentive_compatible: bool
    efficiency_ratio: float  # actual / optimal welfare
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Valuation functions for aspect-bidders
# ---------------------------------------------------------------------------

def coherence_valuation(bundle: FrozenSet[int], items: List[Item]) -> float:
    """Coherence bidder values bundles with semantically related items."""
    if len(bundle) == 0:
        return 0.0
    item_list = [items[i] for i in bundle]
    embeddings = np.array([it.embedding for it in item_list])
    n = len(embeddings)
    if n == 1:
        return item_list[0].quality_score
    # Pairwise cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    sim = normed @ normed.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_sim = np.mean(sim[mask])
    mean_quality = np.mean([it.quality_score for it in item_list])
    return float(mean_sim * mean_quality * n)


def novelty_valuation(bundle: FrozenSet[int], items: List[Item]) -> float:
    """Novelty bidder values bundles with dissimilar items."""
    if len(bundle) == 0:
        return 0.0
    item_list = [items[i] for i in bundle]
    embeddings = np.array([it.embedding for it in item_list])
    n = len(embeddings)
    if n == 1:
        return item_list[0].quality_score * 0.5
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    sim = normed @ normed.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_dissim = 1.0 - np.mean(sim[mask])
    return float(mean_dissim * n)


def relevance_valuation(
    query_embedding: np.ndarray,
) -> Callable[[FrozenSet[int], List[Item]], float]:
    """Create relevance valuation conditioned on a query."""
    def _val(bundle: FrozenSet[int], items: List[Item]) -> float:
        if len(bundle) == 0:
            return 0.0
        item_list = [items[i] for i in bundle]
        embeddings = np.array([it.embedding for it in item_list])
        # Cosine similarity to query
        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-12:
            return 0.0
        q = query_embedding / q_norm
        norms = np.linalg.norm(embeddings, axis=1)
        norms = np.maximum(norms, 1e-12)
        sims = (embeddings @ q) / norms
        return float(np.sum(sims))
    return _val


def coverage_valuation(
    reference_points: np.ndarray,
    radius: float = 1.0,
) -> Callable[[FrozenSet[int], List[Item]], float]:
    """Coverage bidder values bundles that cover reference points."""
    def _val(bundle: FrozenSet[int], items: List[Item]) -> float:
        if len(bundle) == 0:
            return 0.0
        item_list = [items[i] for i in bundle]
        embeddings = np.array([it.embedding for it in item_list])
        covered = 0
        for ref in reference_points:
            dists = np.linalg.norm(embeddings - ref, axis=1)
            if np.min(dists) <= radius:
                covered += 1
        return float(covered)
    return _val


def diversity_valuation_logdet(
    kernel: Optional[Kernel] = None,
) -> Callable[[FrozenSet[int], List[Item]], float]:
    """Log-determinant diversity valuation (submodular)."""
    if kernel is None:
        kernel = RBFKernel(bandwidth=1.0)

    def _val(bundle: FrozenSet[int], items: List[Item]) -> float:
        if len(bundle) == 0:
            return 0.0
        item_list = [items[i] for i in bundle]
        embeddings = np.array([it.embedding for it in item_list])
        K = kernel.gram_matrix(embeddings)
        return log_det_safe(K)
    return _val


def informativeness_valuation(bundle: FrozenSet[int], items: List[Item]) -> float:
    """Values bundles proportional to total quality."""
    if len(bundle) == 0:
        return 0.0
    return float(sum(items[i].quality_score for i in bundle))


# ---------------------------------------------------------------------------
# Combinatorial auction solver (WDP)
# ---------------------------------------------------------------------------

class WinnerDetermination:
    """Winner Determination Problem for combinatorial auctions.

    Solves the allocation problem:
      max  sum_{i,S} x_{i,S} * v_i(S)
      s.t. sum_S x_{i,S} <= 1  for each bidder i
           sum_{i: j in S} x_{i,S} <= 1  for each item j (if no sharing)
           x_{i,S} in {0, 1}

    Uses exhaustive search for small instances and greedy for larger ones.
    """

    def __init__(self, items: List[Item], bidders: List[Bidder],
                 allow_sharing: bool = False, max_bundle_size: int = 0):
        self.items = items
        self.bidders = bidders
        self.allow_sharing = allow_sharing
        self.n_items = len(items)
        self.n_bidders = len(bidders)
        self.max_bundle_size = max_bundle_size if max_bundle_size > 0 else self.n_items

    def _enumerate_bundles(self, max_size: int) -> List[FrozenSet[int]]:
        """Enumerate all bundles up to max_size."""
        bundles = [frozenset()]
        item_indices = list(range(self.n_items))
        for size in range(1, min(max_size, self.n_items) + 1):
            for combo in itertools.combinations(item_indices, size):
                bundles.append(frozenset(combo))
        return bundles

    def _compute_all_values(
        self, bundles: List[FrozenSet[int]]
    ) -> Dict[Tuple[int, FrozenSet[int]], float]:
        """Compute bidder valuations for all bundles."""
        values: Dict[Tuple[int, FrozenSet[int]], float] = {}
        for bidder in self.bidders:
            for bundle in bundles:
                val = bidder.compute_value(bundle, self.items)
                values[(bidder.bidder_id, bundle)] = val
        return values

    def solve_exact(self) -> Tuple[Dict[int, FrozenSet[int]], float]:
        """Exact ILP-style enumeration for small instances.

        Exhaustively tries all allocations. Complexity O(2^(n*m)).
        Only feasible for small n_items * n_bidders.
        """
        bundles = self._enumerate_bundles(self.max_bundle_size)
        values = self._compute_all_values(bundles)
        non_empty = [b for b in bundles if len(b) > 0]

        best_allocation: Dict[int, FrozenSet[int]] = {}
        best_welfare = -float("inf")

        # Generate all possible allocations via recursive search
        def _search(
            bidder_idx: int,
            used_items: Set[int],
            current_alloc: Dict[int, FrozenSet[int]],
            current_welfare: float,
        ) -> None:
            nonlocal best_allocation, best_welfare
            if bidder_idx >= self.n_bidders:
                if current_welfare > best_welfare:
                    best_welfare = current_welfare
                    best_allocation = dict(current_alloc)
                return
            bidder = self.bidders[bidder_idx]
            # Option: allocate empty bundle
            current_alloc[bidder.bidder_id] = frozenset()
            _search(bidder_idx + 1, used_items, current_alloc, current_welfare)
            # Try non-empty bundles
            for bundle in non_empty:
                if not self.allow_sharing and bundle & used_items:
                    continue
                val = values[(bidder.bidder_id, bundle)]
                if val <= 0:
                    continue
                current_alloc[bidder.bidder_id] = bundle
                new_used = used_items | bundle
                _search(bidder_idx + 1, new_used, current_alloc,
                        current_welfare + val)
            current_alloc[bidder.bidder_id] = frozenset()

        _search(0, set(), {}, 0.0)
        return best_allocation, best_welfare

    def solve_greedy(self) -> Tuple[Dict[int, FrozenSet[int]], float]:
        """Greedy approximation for larger instances.

        Iteratively assigns the highest marginal-value bundle to each bidder.
        Provides a 1/sqrt(m) approximation in general.
        """
        bundles = self._enumerate_bundles(min(self.max_bundle_size, 6))
        values = self._compute_all_values(bundles)

        allocation: Dict[int, FrozenSet[int]] = {
            b.bidder_id: frozenset() for b in self.bidders
        }
        used_items: Set[int] = set()
        total_welfare = 0.0

        # Collect (bidder, bundle, value) and sort descending
        candidates = []
        for bidder in self.bidders:
            for bundle in bundles:
                if len(bundle) == 0:
                    continue
                val = values[(bidder.bidder_id, bundle)]
                if val > 0:
                    candidates.append((bidder.bidder_id, bundle, val))
        candidates.sort(key=lambda x: x[2] / max(len(x[1]), 1), reverse=True)

        assigned_bidders: Set[int] = set()
        for bid_id, bundle, val in candidates:
            if bid_id in assigned_bidders:
                continue
            if not self.allow_sharing and bundle & used_items:
                continue
            allocation[bid_id] = bundle
            used_items |= bundle
            total_welfare += val
            assigned_bidders.add(bid_id)

        return allocation, total_welfare

    def solve(self) -> Tuple[Dict[int, FrozenSet[int]], float]:
        """Choose best solver based on instance size."""
        total_complexity = self.n_items * self.n_bidders
        if total_complexity <= 20:
            return self.solve_exact()
        return self.solve_greedy()


# ---------------------------------------------------------------------------
# VCG mechanism for combinatorial diversity auction
# ---------------------------------------------------------------------------

class VCGCombinatorialAuction:
    """Vickrey-Clarke-Groves mechanism for diverse subset selection.

    Properties:
    - Incentive compatible (dominant strategy)
    - Allocatively efficient
    - Individual rational
    - Not budget balanced in general
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        allow_sharing: bool = False,
        max_bundle_size: int = 0,
        reserve_price: float = 0.0,
    ):
        self.items = items
        self.bidders = bidders
        self.allow_sharing = allow_sharing
        self.max_bundle_size = max_bundle_size
        self.reserve_price = reserve_price
        self.wdp = WinnerDetermination(
            items, bidders, allow_sharing, max_bundle_size
        )

    def _compute_vcg_payment(
        self,
        bidder_id: int,
        allocation: Dict[int, FrozenSet[int]],
        total_welfare: float,
    ) -> float:
        """Compute VCG payment for a bidder.

        Payment_i = max welfare without i - (total welfare - i's value).
        """
        # Welfare excluding bidder i's value
        bidder_value = 0.0
        for b in self.bidders:
            if b.bidder_id == bidder_id:
                bidder_value = b.compute_value(
                    allocation.get(bidder_id, frozenset()), self.items
                )
                break
        welfare_without_i_value = total_welfare - bidder_value

        # Solve WDP without bidder i
        remaining_bidders = [b for b in self.bidders if b.bidder_id != bidder_id]
        if len(remaining_bidders) == 0:
            return max(bidder_value - self.reserve_price, 0.0)

        wdp_no_i = WinnerDetermination(
            self.items, remaining_bidders, self.allow_sharing,
            self.max_bundle_size,
        )
        _, welfare_no_i = wdp_no_i.solve()

        # VCG payment = externality imposed on others
        payment = welfare_no_i - welfare_without_i_value
        return max(payment, self.reserve_price)

    def run(self) -> AuctionResult:
        """Run the VCG combinatorial auction."""
        allocation, total_welfare = self.wdp.solve()
        payments: Dict[int, float] = {}
        for bidder in self.bidders:
            bid = bidder.bidder_id
            payments[bid] = self._compute_vcg_payment(bid, allocation, total_welfare)

        selected_items: Set[int] = set()
        for bundle in allocation.values():
            selected_items |= bundle

        revenue = sum(payments.values())

        # Check IC via Clarke pivot rule verification
        is_ic = self._verify_ic(allocation, payments)

        # Efficiency: compare to optimal
        _, optimal_welfare = self.wdp.solve()
        eff = total_welfare / optimal_welfare if optimal_welfare > 0 else 1.0

        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=selected_items,
            is_incentive_compatible=is_ic,
            efficiency_ratio=eff,
            metadata={"mechanism": "VCG_combinatorial"},
        )

    def _verify_ic(
        self,
        allocation: Dict[int, FrozenSet[int]],
        payments: Dict[int, float],
    ) -> bool:
        """Verify IC: no bidder benefits from misreporting."""
        for bidder in self.bidders:
            bid = bidder.bidder_id
            true_value = bidder.compute_value(
                allocation.get(bid, frozenset()), self.items
            )
            utility_truthful = true_value - payments.get(bid, 0.0)
            # Check if any other allocation gives higher utility
            for other_bidder in self.bidders:
                if other_bidder.bidder_id == bid:
                    continue
                other_bundle = allocation.get(other_bidder.bidder_id, frozenset())
                alt_value = bidder.compute_value(other_bundle, self.items)
                alt_payment = payments.get(other_bidder.bidder_id, 0.0)
                if alt_value - alt_payment > utility_truthful + 1e-10:
                    return False
        return True


# ---------------------------------------------------------------------------
# Second-Price Auction with Reserve
# ---------------------------------------------------------------------------

class SecondPriceAuction:
    """Second-price sealed-bid auction (Vickrey auction) for item selection.

    Each candidate response is auctioned to bidders representing quality aspects.
    The winner pays the second-highest bid.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        reserve_price: float = 0.0,
        diversity_penalty: float = 0.0,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.bidders = bidders
        self.reserve_price = reserve_price
        self.diversity_penalty = diversity_penalty
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _compute_diversity_adjusted_value(
        self,
        item_idx: int,
        selected_so_far: List[int],
        base_value: float,
    ) -> float:
        """Adjust bid value based on similarity to already selected items."""
        if len(selected_so_far) == 0 or self.diversity_penalty <= 0:
            return base_value
        item_emb = self.items[item_idx].embedding
        max_sim = 0.0
        for sel_idx in selected_so_far:
            sim = self.kernel.evaluate(item_emb, self.items[sel_idx].embedding)
            max_sim = max(max_sim, sim)
        adjusted = base_value - self.diversity_penalty * max_sim
        return max(adjusted, 0.0)

    def run_single_item(
        self, item_idx: int, selected_so_far: Optional[List[int]] = None,
    ) -> Tuple[int, float, float]:
        """Run second-price auction for a single item.

        Returns (winning_bidder_id, payment, value).
        """
        if selected_so_far is None:
            selected_so_far = []
        bids: List[Tuple[int, float]] = []
        for bidder in self.bidders:
            bundle = frozenset([item_idx])
            raw_value = bidder.compute_value(bundle, self.items)
            adj_value = self._compute_diversity_adjusted_value(
                item_idx, selected_so_far, raw_value
            )
            bids.append((bidder.bidder_id, adj_value))
        bids.sort(key=lambda x: x[1], reverse=True)
        if len(bids) == 0 or bids[0][1] < self.reserve_price:
            return -1, 0.0, 0.0
        winner_id = bids[0][0]
        winner_value = bids[0][1]
        second_price = bids[1][1] if len(bids) > 1 else self.reserve_price
        payment = max(second_price, self.reserve_price)
        return winner_id, payment, winner_value

    def run_sequential(self, k: int) -> AuctionResult:
        """Select k items via sequential second-price auctions."""
        selected: List[int] = []
        allocation: Dict[int, FrozenSet[int]] = {
            b.bidder_id: frozenset() for b in self.bidders
        }
        payments: Dict[int, float] = {b.bidder_id: 0.0 for b in self.bidders}
        total_welfare = 0.0

        for _ in range(min(k, len(self.items))):
            best_item = -1
            best_value = -float("inf")
            best_winner = -1
            best_payment = 0.0

            for idx in range(len(self.items)):
                if idx in selected:
                    continue
                winner_id, payment, value = self.run_single_item(idx, selected)
                if value > best_value and winner_id >= 0:
                    best_item = idx
                    best_value = value
                    best_winner = winner_id
                    best_payment = payment

            if best_item < 0:
                break

            selected.append(best_item)
            old_bundle = allocation.get(best_winner, frozenset())
            allocation[best_winner] = old_bundle | frozenset([best_item])
            payments[best_winner] = payments.get(best_winner, 0.0) + best_payment
            total_welfare += best_value

        revenue = sum(payments.values())
        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected),
            is_incentive_compatible=True,  # SP is DSIC
            efficiency_ratio=1.0,
            metadata={"mechanism": "second_price_sequential", "k": k},
        )


# ---------------------------------------------------------------------------
# English (Ascending Price) Auction
# ---------------------------------------------------------------------------

class EnglishAuction:
    """Ascending price auction for diverse item selection.

    Prices increase incrementally. Bidders drop out when price exceeds
    their value. Winner pays the dropout price of the last remaining bidder.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        price_increment: float = 0.01,
        max_rounds: int = 10000,
        diversity_bonus: float = 0.0,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.bidders = bidders
        self.price_increment = price_increment
        self.max_rounds = max_rounds
        self.diversity_bonus = diversity_bonus
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _bidder_value_for_item(
        self, bidder: Bidder, item_idx: int, selected: List[int],
    ) -> float:
        """Compute bidder's value for item, including diversity bonus."""
        base_val = bidder.compute_value(frozenset([item_idx]), self.items)
        if self.diversity_bonus > 0 and len(selected) > 0:
            emb = self.items[item_idx].embedding
            min_dist = float("inf")
            for s_idx in selected:
                d = np.linalg.norm(emb - self.items[s_idx].embedding)
                min_dist = min(min_dist, d)
            base_val += self.diversity_bonus * min_dist
        return base_val

    def auction_single_item(
        self, item_idx: int, selected: List[int],
    ) -> Tuple[int, float]:
        """Run ascending auction for one item.

        Returns (winner_bidder_id, final_price).
        """
        active = list(range(len(self.bidders)))
        values = [
            self._bidder_value_for_item(self.bidders[i], item_idx, selected)
            for i in active
        ]
        price = 0.0
        for _ in range(self.max_rounds):
            if len(active) <= 1:
                break
            new_active = [i for i in active if values[i] >= price]
            if len(new_active) <= 1:
                if len(new_active) == 1:
                    active = new_active
                break
            active = new_active
            price += self.price_increment

        if len(active) == 0:
            return -1, 0.0
        winner_idx = max(active, key=lambda i: values[i])
        return self.bidders[winner_idx].bidder_id, price

    def run(self, k: int) -> AuctionResult:
        """Select k items via sequential English auctions."""
        selected: List[int] = []
        allocation: Dict[int, FrozenSet[int]] = {
            b.bidder_id: frozenset() for b in self.bidders
        }
        payments: Dict[int, float] = {b.bidder_id: 0.0 for b in self.bidders}
        total_welfare = 0.0

        available = list(range(len(self.items)))
        for _ in range(min(k, len(available))):
            best_item = -1
            best_price = -1.0
            best_winner = -1

            for idx in available:
                if idx in selected:
                    continue
                winner_id, price = self.auction_single_item(idx, selected)
                if winner_id >= 0 and price > best_price:
                    best_item = idx
                    best_price = price
                    best_winner = winner_id

            if best_item < 0:
                break

            selected.append(best_item)
            old_b = allocation.get(best_winner, frozenset())
            allocation[best_winner] = old_b | frozenset([best_item])
            payments[best_winner] = payments.get(best_winner, 0.0) + best_price
            total_welfare += best_price

        revenue = sum(payments.values())
        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected),
            is_incentive_compatible=True,
            efficiency_ratio=1.0,
            metadata={"mechanism": "english_auction", "k": k},
        )


# ---------------------------------------------------------------------------
# Sealed-Bid Auction with Diversity Constraints
# ---------------------------------------------------------------------------

class SealedBidDiversityAuction:
    """First-price sealed-bid auction with diversity constraints.

    Bidders submit sealed bids. Winner is the highest bidder, but
    allocation is constrained to maintain minimum diversity level.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        min_diversity: float = 0.0,
        diversity_metric: str = "logdet",
        kernel: Optional[Kernel] = None,
        max_bundle_size: int = 5,
    ):
        self.items = items
        self.bidders = bidders
        self.min_diversity = min_diversity
        self.diversity_metric = diversity_metric
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.max_bundle_size = max_bundle_size

    def _compute_diversity(self, indices: List[int]) -> float:
        """Compute diversity of selected items."""
        if len(indices) < 2:
            return 0.0
        embeddings = np.array([self.items[i].embedding for i in indices])
        if self.diversity_metric == "logdet":
            K = self.kernel.gram_matrix(embeddings)
            return log_det_safe(K)
        elif self.diversity_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normed = embeddings / norms
            sim = normed @ normed.T
            n = len(indices)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            return float(1.0 - np.mean(sim[mask]))
        elif self.diversity_metric == "pairwise_dist":
            n = len(indices)
            total_dist = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total_dist += np.linalg.norm(
                        embeddings[i] - embeddings[j]
                    )
                    count += 1
            return total_dist / max(count, 1)
        else:
            raise ValueError(f"Unknown diversity metric: {self.diversity_metric}")

    def _collect_bids(self, k: int) -> List[Tuple[int, FrozenSet[int], float]]:
        """Collect sealed bids from all bidders for bundles up to size k."""
        all_bids: List[Tuple[int, FrozenSet[int], float]] = []
        item_indices = list(range(len(self.items)))
        max_size = min(k, self.max_bundle_size, len(self.items))

        for bidder in self.bidders:
            for size in range(1, max_size + 1):
                # Sample bundles if too many combinations
                n_combos = math.comb(len(item_indices), size)
                if n_combos > 100:
                    rng = np.random.RandomState(bidder.bidder_id)
                    for _ in range(100):
                        chosen = rng.choice(item_indices, size=size, replace=False)
                        bundle = frozenset(chosen.tolist())
                        val = bidder.compute_value(bundle, self.items)
                        all_bids.append((bidder.bidder_id, bundle, val))
                else:
                    for combo in itertools.combinations(item_indices, size):
                        bundle = frozenset(combo)
                        val = bidder.compute_value(bundle, self.items)
                        all_bids.append((bidder.bidder_id, bundle, val))
        return all_bids

    def run(self, k: int) -> AuctionResult:
        """Run sealed-bid auction with diversity constraint."""
        all_bids = self._collect_bids(k)
        all_bids.sort(key=lambda x: x[2], reverse=True)

        allocation: Dict[int, FrozenSet[int]] = {}
        selected: Set[int] = set()
        total_welfare = 0.0

        for bidder_id, bundle, value in all_bids:
            if bidder_id in allocation:
                continue
            candidate_selected = selected | bundle
            if len(candidate_selected) > k:
                continue
            # Check diversity constraint
            if len(candidate_selected) >= 2:
                div = self._compute_diversity(list(candidate_selected))
                if div < self.min_diversity:
                    continue
            allocation[bidder_id] = bundle
            selected = candidate_selected
            total_welfare += value
            if len(selected) >= k:
                break

        # First-price payments (pay what you bid)
        payments = {
            bid_id: self.bidders[0].compute_value(bundle, self.items)
            for bid_id, bundle in allocation.items()
        }
        for bidder in self.bidders:
            if bidder.bidder_id in allocation:
                payments[bidder.bidder_id] = bidder.compute_value(
                    allocation[bidder.bidder_id], self.items
                )

        revenue = sum(payments.values())
        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=selected,
            is_incentive_compatible=False,  # first-price is not DSIC
            efficiency_ratio=1.0,
            metadata={"mechanism": "sealed_bid_diversity", "k": k},
        )


# ---------------------------------------------------------------------------
# Myerson's Optimal Auction for Diversity
# ---------------------------------------------------------------------------

class MyersonOptimalAuction:
    """Myerson's optimal (revenue-maximizing) auction adapted for diversity.

    Key ideas:
    1. Compute virtual valuations phi_i(v_i)
    2. Allocate to maximize virtual surplus
    3. Charge payments that make truthful reporting optimal

    Assumes independent private values drawn from known distributions.
    """

    def __init__(
        self,
        items: List[Item],
        n_bidders: int,
        value_distributions: Optional[List[Callable[[], float]]] = None,
        cdf_functions: Optional[List[Callable[[float], float]]] = None,
        pdf_functions: Optional[List[Callable[[float], float]]] = None,
        reserve_multiplier: float = 1.0,
        diversity_weight: float = 0.0,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.n_bidders = n_bidders
        self.diversity_weight = diversity_weight
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.reserve_multiplier = reserve_multiplier

        # Default: uniform [0, 1] distributions
        if value_distributions is None:
            self._rng = np.random.RandomState(42)
            self.value_distributions = [
                lambda: self._rng.uniform(0, 1) for _ in range(n_bidders)
            ]
        else:
            self.value_distributions = value_distributions

        if cdf_functions is None:
            self.cdf_functions = [lambda v: min(max(v, 0.0), 1.0)] * n_bidders
        else:
            self.cdf_functions = cdf_functions

        if pdf_functions is None:
            self.pdf_functions = [lambda v: 1.0] * n_bidders
        else:
            self.pdf_functions = pdf_functions

    def virtual_valuation(self, bidder_idx: int, value: float) -> float:
        """Compute virtual valuation for bidder i with value v.

        phi_i(v) = v - (1 - F_i(v)) / f_i(v)

        For uniform [0,1]: phi(v) = 2v - 1
        """
        cdf = self.cdf_functions[bidder_idx]
        pdf = self.pdf_functions[bidder_idx]
        F_v = cdf(value)
        f_v = pdf(value)
        if f_v < 1e-12:
            return value
        return value - (1.0 - F_v) / f_v

    def ironed_virtual_valuation(
        self, bidder_idx: int, value: float, n_grid: int = 1000,
    ) -> float:
        """Compute ironed virtual valuation for non-regular distributions.

        Uses the Myerson ironing procedure:
        1. Compute H(q) = integral of phi(F^{-1}(q)) dq
        2. Take convex hull of H
        3. Ironed phi = derivative of convex hull
        """
        grid = np.linspace(0.001, 0.999, n_grid)
        cdf = self.cdf_functions[bidder_idx]
        pdf = self.pdf_functions[bidder_idx]

        # Numerically invert CDF
        val_grid = np.linspace(0, 2, n_grid)
        cdf_vals = np.array([cdf(v) for v in val_grid])

        # Compute virtual valuations on grid
        phi_vals = np.array([self.virtual_valuation(bidder_idx, v) for v in val_grid])

        # Compute H(q) = integral_0^q phi(F^{-1}(s)) ds
        H = np.zeros(n_grid)
        dq = 1.0 / n_grid
        for i in range(1, n_grid):
            q = grid[i] if i < len(grid) else 0.999
            # Find value at quantile q
            idx = np.searchsorted(cdf_vals, q)
            idx = min(idx, n_grid - 1)
            phi_at_q = phi_vals[idx]
            H[i] = H[i - 1] + phi_at_q * dq

        # Convex hull of H
        H_convex = self._convex_hull_lower(H)

        # Ironed virtual valuation = derivative of convex hull
        # Find where input value sits
        v_quantile = cdf(value)
        q_idx = int(v_quantile * (n_grid - 1))
        q_idx = max(0, min(q_idx, n_grid - 2))

        ironed_phi = (H_convex[q_idx + 1] - H_convex[q_idx]) / dq
        return ironed_phi

    @staticmethod
    def _convex_hull_lower(H: np.ndarray) -> np.ndarray:
        """Compute lower convex envelope of sequence H."""
        n = len(H)
        hull = np.copy(H)
        # Graham scan style
        for i in range(1, n - 1):
            # If H[i] is above the line from H[i-1] to H[i+1], project down
            interp = H[i - 1] + (H[i + 1] - H[i - 1]) / 2.0
            if H[i] > interp:
                hull[i] = interp
        return hull

    def optimal_reserve_price(self, bidder_idx: int) -> float:
        """Compute optimal reserve price for bidder.

        Reserve r* satisfies phi(r*) = 0, i.e., r* = F^{-1}(1/2) for uniform.
        """
        # Binary search for phi(r) = 0
        lo, hi = 0.0, 10.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            phi = self.virtual_valuation(bidder_idx, mid)
            if phi < 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0 * self.reserve_multiplier

    def run_single_item(
        self,
        item_idx: int,
        realized_values: Optional[List[float]] = None,
    ) -> Tuple[int, float, float]:
        """Run Myerson auction for a single item.

        Returns (winner_idx, payment, virtual_surplus).
        """
        if realized_values is None:
            realized_values = [
                self.value_distributions[i]() for i in range(self.n_bidders)
            ]
        # Compute virtual valuations
        virtual_vals = [
            self.virtual_valuation(i, realized_values[i])
            for i in range(self.n_bidders)
        ]
        # Compute reserve prices
        reserves = [
            self.optimal_reserve_price(i) for i in range(self.n_bidders)
        ]

        # Winner: highest positive virtual valuation above reserve
        eligible = [
            (i, virtual_vals[i])
            for i in range(self.n_bidders)
            if realized_values[i] >= reserves[i] and virtual_vals[i] > 0
        ]
        if len(eligible) == 0:
            return -1, 0.0, 0.0

        eligible.sort(key=lambda x: x[1], reverse=True)
        winner_idx = eligible[0][0]
        virtual_surplus = eligible[0][1]

        # Payment: minimum value winner would need to still win
        if len(eligible) > 1:
            second_virtual = eligible[1][1]
            # Find v such that phi(v) = second_virtual
            payment = self._inverse_virtual(winner_idx, second_virtual)
        else:
            payment = reserves[winner_idx]

        return winner_idx, payment, virtual_surplus

    def _inverse_virtual(self, bidder_idx: int, target_phi: float) -> float:
        """Find value v such that phi_i(v) = target_phi via binary search."""
        lo, hi = 0.0, 10.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            phi = self.virtual_valuation(bidder_idx, mid)
            if phi < target_phi:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def run(
        self,
        k: int,
        n_simulations: int = 100,
        seed: int = 42,
    ) -> AuctionResult:
        """Run Myerson optimal auction, selecting k items.

        Monte Carlo simulation over value draws.
        """
        rng = np.random.RandomState(seed)
        total_revenue = 0.0
        total_welfare = 0.0
        selection_counts = np.zeros(len(self.items))
        all_payments: Dict[int, float] = {i: 0.0 for i in range(self.n_bidders)}

        for _ in range(n_simulations):
            selected: List[int] = []
            sim_revenue = 0.0
            sim_welfare = 0.0

            for step in range(min(k, len(self.items))):
                # Draw values
                values = [rng.uniform(0, 1) for _ in range(self.n_bidders)]
                # Add diversity bonus
                for bidder_idx in range(self.n_bidders):
                    if self.diversity_weight > 0 and len(selected) > 0:
                        item_idx = step  # simplified
                        emb = self.items[min(step, len(self.items) - 1)].embedding
                        min_dist = min(
                            np.linalg.norm(emb - self.items[s].embedding)
                            for s in selected
                        )
                        values[bidder_idx] += self.diversity_weight * min_dist

                winner, payment, virt_surplus = self.run_single_item(
                    min(step, len(self.items) - 1), values
                )
                if winner >= 0:
                    selected.append(min(step, len(self.items) - 1))
                    sim_revenue += payment
                    sim_welfare += virt_surplus
                    all_payments[winner] += payment / n_simulations

            for s in selected:
                selection_counts[s] += 1
            total_revenue += sim_revenue
            total_welfare += sim_welfare

        avg_revenue = total_revenue / n_simulations
        avg_welfare = total_welfare / n_simulations

        # Most frequently selected items
        top_k = np.argsort(selection_counts)[-k:][::-1]
        selected_items = set(top_k.tolist())

        return AuctionResult(
            allocation={0: frozenset(selected_items)},
            payments=all_payments,
            social_welfare=avg_welfare,
            revenue=avg_revenue,
            selected_items=selected_items,
            is_incentive_compatible=True,  # Myerson is BIC
            efficiency_ratio=1.0,
            metadata={
                "mechanism": "myerson_optimal",
                "n_simulations": n_simulations,
                "selection_counts": selection_counts.tolist(),
            },
        )


# ---------------------------------------------------------------------------
# Revenue analysis utilities
# ---------------------------------------------------------------------------

class RevenueAnalyzer:
    """Analyze and compare revenue properties of different auction mechanisms."""

    def __init__(self, items: List[Item], bidders: List[Bidder]):
        self.items = items
        self.bidders = bidders

    def revenue_equivalence_check(
        self,
        mechanisms: List[Tuple[str, Callable[[], AuctionResult]]],
        n_trials: int = 50,
        seed: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """Check revenue equivalence theorem across mechanisms.

        Revenue equivalence states that all standard auctions with
        symmetric independent private values yield same expected revenue.
        """
        rng = np.random.RandomState(seed)
        results: Dict[str, List[float]] = {name: [] for name, _ in mechanisms}

        for trial in range(n_trials):
            for name, run_fn in mechanisms:
                result = run_fn()
                results[name].append(result.revenue)

        analysis: Dict[str, Dict[str, float]] = {}
        for name in results:
            revenues = results[name]
            analysis[name] = {
                "mean_revenue": float(np.mean(revenues)),
                "std_revenue": float(np.std(revenues)),
                "median_revenue": float(np.median(revenues)),
                "min_revenue": float(np.min(revenues)),
                "max_revenue": float(np.max(revenues)),
            }

        # Pairwise revenue comparison
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r_i = np.array(results[names[i]])
                r_j = np.array(results[names[j]])
                diff = r_i - r_j
                analysis[f"{names[i]}_vs_{names[j]}"] = {
                    "mean_diff": float(np.mean(diff)),
                    "std_diff": float(np.std(diff)),
                    "pct_higher_first": float(np.mean(diff > 0)),
                }
        return analysis

    def individual_rationality_check(
        self, result: AuctionResult,
    ) -> Dict[int, bool]:
        """Check individual rationality: no bidder pays more than their value."""
        ir_status: Dict[int, bool] = {}
        for bidder in self.bidders:
            bid = bidder.bidder_id
            bundle = result.allocation.get(bid, frozenset())
            value = bidder.compute_value(bundle, self.items)
            payment = result.payments.get(bid, 0.0)
            ir_status[bid] = payment <= value + 1e-10
        return ir_status

    def budget_balance_check(self, result: AuctionResult) -> Dict[str, float]:
        """Check budget balance properties."""
        total_value = sum(
            bidder.compute_value(
                result.allocation.get(bidder.bidder_id, frozenset()),
                self.items,
            )
            for bidder in self.bidders
        )
        total_payment = sum(result.payments.values())
        return {
            "total_value": total_value,
            "total_payments": total_payment,
            "surplus": total_value - total_payment,
            "is_weakly_balanced": total_payment >= 0,
            "is_strongly_balanced": abs(total_payment - total_value) < 1e-10,
        }

    def efficiency_analysis(
        self, result: AuctionResult,
    ) -> Dict[str, float]:
        """Analyze allocative efficiency."""
        actual_welfare = result.social_welfare
        # Compute optimal welfare (brute force for small instances)
        wdp = WinnerDetermination(self.items, self.bidders)
        _, optimal_welfare = wdp.solve()
        return {
            "actual_welfare": actual_welfare,
            "optimal_welfare": optimal_welfare,
            "efficiency_ratio": (
                actual_welfare / optimal_welfare if optimal_welfare > 0 else 1.0
            ),
            "welfare_loss": optimal_welfare - actual_welfare,
        }


# ---------------------------------------------------------------------------
# Diversity-aware auction framework
# ---------------------------------------------------------------------------

class DiversityAuctionFramework:
    """Unified framework combining auction theory with diversity optimization.

    Orchestrates multiple auction mechanisms and selects the best
    allocation according to a quality-diversity tradeoff.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        quality_weight: float = 0.5,
        diversity_weight: float = 0.5,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.bidders = bidders
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _score_allocation(self, selected: Set[int]) -> float:
        """Score an allocation based on quality-diversity tradeoff."""
        if len(selected) == 0:
            return 0.0
        indices = sorted(selected)
        # Quality
        quality = np.mean([self.items[i].quality_score for i in indices])
        # Diversity
        if len(indices) < 2:
            diversity = 0.0
        else:
            embeddings = np.array([self.items[i].embedding for i in indices])
            K = self.kernel.gram_matrix(embeddings)
            diversity = log_det_safe(K)
        return self.quality_weight * quality + self.diversity_weight * diversity

    def run_all_mechanisms(
        self, k: int, verbose: bool = False,
    ) -> Dict[str, AuctionResult]:
        """Run all available auction mechanisms and return results."""
        results: Dict[str, AuctionResult] = {}

        # 1. VCG Combinatorial
        vcg = VCGCombinatorialAuction(
            self.items, self.bidders, max_bundle_size=min(k, 4),
        )
        results["vcg"] = vcg.run()

        # 2. Second-Price Sequential
        sp = SecondPriceAuction(
            self.items, self.bidders, diversity_penalty=0.5,
            kernel=self.kernel,
        )
        results["second_price"] = sp.run_sequential(k)

        # 3. English Auction
        eng = EnglishAuction(
            self.items, self.bidders, diversity_bonus=0.3,
            kernel=self.kernel,
        )
        results["english"] = eng.run(k)

        # 4. Sealed-Bid with Diversity
        sb = SealedBidDiversityAuction(
            self.items, self.bidders, min_diversity=0.1,
            kernel=self.kernel, max_bundle_size=min(k, 4),
        )
        results["sealed_bid"] = sb.run(k)

        if verbose:
            for name, res in results.items():
                score = self._score_allocation(res.selected_items)
                print(
                    f"{name}: welfare={res.social_welfare:.4f}, "
                    f"revenue={res.revenue:.4f}, "
                    f"score={score:.4f}, IC={res.is_incentive_compatible}"
                )

        return results

    def select_best(
        self, k: int,
    ) -> Tuple[str, AuctionResult]:
        """Run all mechanisms and return the best one."""
        results = self.run_all_mechanisms(k)
        best_name = ""
        best_score = -float("inf")
        best_result = None
        for name, res in results.items():
            score = self._score_allocation(res.selected_items)
            if score > best_score:
                best_score = score
                best_name = name
                best_result = res
        assert best_result is not None
        return best_name, best_result


# ---------------------------------------------------------------------------
# Auction-based diverse subset selection (end-to-end)
# ---------------------------------------------------------------------------

def create_aspect_bidders(
    n_items: int,
    dim: int,
    items: List[Item],
    query_embedding: Optional[np.ndarray] = None,
    kernel: Optional[Kernel] = None,
) -> List[Bidder]:
    """Create a standard set of aspect-bidders for diversity auctions."""
    bidders = []
    bid_id = 0

    # Coherence bidder
    bidders.append(Bidder(
        bidder_id=bid_id,
        bidder_type=BidderType.COHERENCE,
        valuation_fn=coherence_valuation,
    ))
    bid_id += 1

    # Novelty bidder
    bidders.append(Bidder(
        bidder_id=bid_id,
        bidder_type=BidderType.NOVELTY,
        valuation_fn=novelty_valuation,
    ))
    bid_id += 1

    # Coverage bidder
    ref_points = np.random.randn(20, dim) * 0.5
    bidders.append(Bidder(
        bidder_id=bid_id,
        bidder_type=BidderType.COVERAGE,
        valuation_fn=coverage_valuation(ref_points, radius=1.5),
    ))
    bid_id += 1

    # Diversity bidder (log-det)
    bidders.append(Bidder(
        bidder_id=bid_id,
        bidder_type=BidderType.DIVERSITY,
        valuation_fn=diversity_valuation_logdet(kernel),
    ))
    bid_id += 1

    # Informativeness bidder
    bidders.append(Bidder(
        bidder_id=bid_id,
        bidder_type=BidderType.INFORMATIVENESS,
        valuation_fn=informativeness_valuation,
    ))
    bid_id += 1

    # Relevance bidder (if query provided)
    if query_embedding is not None:
        bidders.append(Bidder(
            bidder_id=bid_id,
            bidder_type=BidderType.RELEVANCE,
            valuation_fn=relevance_valuation(query_embedding),
        ))
        bid_id += 1

    return bidders


def run_diversity_auction(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    k: int = 5,
    mechanism: str = "vcg",
    query_embedding: Optional[np.ndarray] = None,
    kernel: Optional[Kernel] = None,
    **kwargs,
) -> AuctionResult:
    """End-to-end diversity auction.

    Args:
        embeddings: (n, d) candidate embeddings
        quality_scores: (n,) quality scores
        k: number to select
        mechanism: "vcg", "second_price", "english", "sealed_bid", "myerson", "best"
        query_embedding: optional query for relevance
        kernel: kernel for diversity computation
    """
    n, d = embeddings.shape
    items = [
        Item(index=i, embedding=embeddings[i], quality_score=float(quality_scores[i]))
        for i in range(n)
    ]
    bidders = create_aspect_bidders(n, d, items, query_embedding, kernel)
    kernel = kernel or RBFKernel(bandwidth=1.0)

    if mechanism == "vcg":
        auction = VCGCombinatorialAuction(
            items, bidders, max_bundle_size=min(k, 4),
            reserve_price=kwargs.get("reserve_price", 0.0),
        )
        return auction.run()
    elif mechanism == "second_price":
        auction = SecondPriceAuction(
            items, bidders,
            reserve_price=kwargs.get("reserve_price", 0.0),
            diversity_penalty=kwargs.get("diversity_penalty", 0.5),
            kernel=kernel,
        )
        return auction.run_sequential(k)
    elif mechanism == "english":
        auction = EnglishAuction(
            items, bidders,
            diversity_bonus=kwargs.get("diversity_bonus", 0.3),
            kernel=kernel,
        )
        return auction.run(k)
    elif mechanism == "sealed_bid":
        auction = SealedBidDiversityAuction(
            items, bidders,
            min_diversity=kwargs.get("min_diversity", 0.1),
            kernel=kernel,
            max_bundle_size=min(k, 4),
        )
        return auction.run(k)
    elif mechanism == "myerson":
        auction = MyersonOptimalAuction(
            items, n_bidders=len(bidders),
            diversity_weight=kwargs.get("diversity_weight", 0.3),
            kernel=kernel,
        )
        return auction.run(k, n_simulations=kwargs.get("n_simulations", 100))
    elif mechanism == "best":
        framework = DiversityAuctionFramework(
            items, bidders,
            quality_weight=kwargs.get("quality_weight", 0.5),
            diversity_weight=kwargs.get("diversity_weight_fw", 0.5),
            kernel=kernel,
        )
        _, result = framework.select_best(k)
        return result
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


# ---------------------------------------------------------------------------
# Multi-unit auction for multiple copies / slots
# ---------------------------------------------------------------------------

class MultiUnitUniformPriceAuction:
    """Multi-unit auction where k identical "slots" are auctioned.

    Each slot is filled with a candidate response. Uniform price is the
    (k+1)-th highest bid.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        k: int = 5,
    ):
        self.items = items
        self.bidders = bidders
        self.k = k

    def run(self) -> AuctionResult:
        """Run uniform price auction for k slots."""
        # Each bidder bids on each item
        all_bids: List[Tuple[int, int, float]] = []  # (bidder_id, item_idx, value)
        for bidder in self.bidders:
            for item in self.items:
                val = bidder.compute_value(frozenset([item.index]), self.items)
                all_bids.append((bidder.bidder_id, item.index, val))

        # Sort by value descending
        all_bids.sort(key=lambda x: x[2], reverse=True)

        # Select top-k unique items
        selected: List[int] = []
        winning_bids: List[Tuple[int, int, float]] = []
        for bidder_id, item_idx, val in all_bids:
            if item_idx not in selected and len(selected) < self.k:
                selected.append(item_idx)
                winning_bids.append((bidder_id, item_idx, val))

        # Uniform price = (k+1)-th highest bid
        if len(all_bids) > self.k:
            uniform_price = all_bids[self.k][2]
        else:
            uniform_price = 0.0
        uniform_price = max(uniform_price, 0.0)

        allocation: Dict[int, FrozenSet[int]] = {}
        payments: Dict[int, float] = {}
        for bidder_id, item_idx, val in winning_bids:
            old = allocation.get(bidder_id, frozenset())
            allocation[bidder_id] = old | frozenset([item_idx])
            payments[bidder_id] = payments.get(bidder_id, 0.0) + uniform_price

        total_welfare = sum(v for _, _, v in winning_bids)
        revenue = sum(payments.values())

        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected),
            is_incentive_compatible=False,  # Uniform price not always IC
            efficiency_ratio=1.0,
            metadata={"mechanism": "uniform_price", "k": self.k,
                       "uniform_price": uniform_price},
        )


# ---------------------------------------------------------------------------
# Discriminatory (pay-as-bid) multi-unit auction
# ---------------------------------------------------------------------------

class DiscriminatoryAuction:
    """Discriminatory (pay-as-bid) auction for k items.

    Winners pay their own bid. Not incentive compatible but commonly used.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        k: int = 5,
        shading_factor: float = 0.8,
    ):
        self.items = items
        self.bidders = bidders
        self.k = k
        self.shading_factor = shading_factor

    def run(self) -> AuctionResult:
        """Run discriminatory auction."""
        all_bids: List[Tuple[int, int, float, float]] = []
        for bidder in self.bidders:
            for item in self.items:
                true_val = bidder.compute_value(frozenset([item.index]), self.items)
                # Rational bidders shade their bids
                bid = true_val * self.shading_factor
                all_bids.append((bidder.bidder_id, item.index, bid, true_val))

        all_bids.sort(key=lambda x: x[2], reverse=True)

        selected: List[int] = []
        winning_bids: List[Tuple[int, int, float, float]] = []
        for bidder_id, item_idx, bid, true_val in all_bids:
            if item_idx not in selected and len(selected) < self.k:
                selected.append(item_idx)
                winning_bids.append((bidder_id, item_idx, bid, true_val))

        allocation: Dict[int, FrozenSet[int]] = {}
        payments: Dict[int, float] = {}
        for bidder_id, item_idx, bid, _ in winning_bids:
            old = allocation.get(bidder_id, frozenset())
            allocation[bidder_id] = old | frozenset([item_idx])
            payments[bidder_id] = payments.get(bidder_id, 0.0) + bid

        total_welfare = sum(tv for _, _, _, tv in winning_bids)
        revenue = sum(payments.values())

        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected),
            is_incentive_compatible=False,
            efficiency_ratio=1.0,
            metadata={"mechanism": "discriminatory", "k": self.k},
        )


# ---------------------------------------------------------------------------
# Auction with externalities (diversity = externality)
# ---------------------------------------------------------------------------

class ExternalityAuction:
    """Auction where diversity creates positive externalities.

    Each selected item generates positive externalities for bidders
    who value diversity. Prices are adjusted to internalize externalities.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        externality_rate: float = 0.1,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.bidders = bidders
        self.externality_rate = externality_rate
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _compute_externality(
        self, item_idx: int, selected: List[int],
    ) -> float:
        """Compute positive externality from adding item to selection."""
        if len(selected) == 0:
            return 0.0
        emb = self.items[item_idx].embedding
        total_ext = 0.0
        for s_idx in selected:
            dist = np.linalg.norm(emb - self.items[s_idx].embedding)
            total_ext += self.externality_rate * dist
        return total_ext

    def run(self, k: int) -> AuctionResult:
        """Run externality-aware auction."""
        selected: List[int] = []
        allocation: Dict[int, FrozenSet[int]] = {
            b.bidder_id: frozenset() for b in self.bidders
        }
        payments: Dict[int, float] = {b.bidder_id: 0.0 for b in self.bidders}
        total_welfare = 0.0

        for _ in range(min(k, len(self.items))):
            best_item = -1
            best_score = -float("inf")
            best_winner = -1
            best_payment = 0.0

            for idx in range(len(self.items)):
                if idx in selected:
                    continue
                externality = self._compute_externality(idx, selected)
                # Find highest bidder
                max_bid = -float("inf")
                second_bid = -float("inf")
                max_bidder = -1
                for bidder in self.bidders:
                    val = bidder.compute_value(frozenset([idx]), self.items)
                    adjusted_val = val + externality
                    if adjusted_val > max_bid:
                        second_bid = max_bid
                        max_bid = adjusted_val
                        max_bidder = bidder.bidder_id
                    elif adjusted_val > second_bid:
                        second_bid = adjusted_val

                total_score = max_bid
                if total_score > best_score:
                    best_score = total_score
                    best_item = idx
                    best_winner = max_bidder
                    best_payment = max(second_bid, 0.0)

            if best_item < 0:
                break

            selected.append(best_item)
            old_b = allocation.get(best_winner, frozenset())
            allocation[best_winner] = old_b | frozenset([best_item])
            # Pigouvian subsidy: reduce payment by externality amount
            ext = self._compute_externality(best_item, selected[:-1])
            actual_payment = max(best_payment - ext, 0.0)
            payments[best_winner] = payments.get(best_winner, 0.0) + actual_payment
            total_welfare += best_score

        revenue = sum(payments.values())
        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected),
            is_incentive_compatible=True,
            efficiency_ratio=1.0,
            metadata={"mechanism": "externality_auction", "k": k},
        )


# ---------------------------------------------------------------------------
# Position auction (for ranked diversity)
# ---------------------------------------------------------------------------

class PositionAuction:
    """Generalized Second-Price (GSP) position auction.

    Positions have decreasing click-through rates (or visibility weights).
    Adapted for diversity: higher-ranked items contribute more to visible diversity.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        position_weights: Optional[List[float]] = None,
        k: int = 5,
    ):
        self.items = items
        self.bidders = bidders
        self.k = k
        if position_weights is not None:
            self.position_weights = position_weights
        else:
            # Logarithmic decay
            self.position_weights = [1.0 / np.log2(i + 2) for i in range(k)]

    def run(self) -> AuctionResult:
        """Run GSP auction for k positions."""
        # Each bidder bids on each item
        item_bids: Dict[int, List[Tuple[int, float]]] = {
            item.index: [] for item in self.items
        }
        for bidder in self.bidders:
            for item in self.items:
                val = bidder.compute_value(frozenset([item.index]), self.items)
                item_bids[item.index].append((bidder.bidder_id, val))

        # Score each item by max bid
        item_scores: List[Tuple[int, float, int]] = []
        for item_idx, bids in item_bids.items():
            if len(bids) == 0:
                continue
            bids.sort(key=lambda x: x[1], reverse=True)
            best_bidder = bids[0][0]
            best_value = bids[0][1]
            item_scores.append((item_idx, best_value, best_bidder))

        item_scores.sort(key=lambda x: x[1], reverse=True)

        selected: List[int] = []
        allocation: Dict[int, FrozenSet[int]] = {}
        payments: Dict[int, float] = {}
        total_welfare = 0.0

        for pos in range(min(self.k, len(item_scores))):
            item_idx, value, winner_id = item_scores[pos]
            selected.append(item_idx)
            weight = self.position_weights[pos] if pos < len(self.position_weights) else 0.1

            old_b = allocation.get(winner_id, frozenset())
            allocation[winner_id] = old_b | frozenset([item_idx])

            # GSP payment: next-position's bid * current weight
            if pos + 1 < len(item_scores):
                next_value = item_scores[pos + 1][1]
            else:
                next_value = 0.0
            payment = next_value * weight
            payments[winner_id] = payments.get(winner_id, 0.0) + payment
            total_welfare += value * weight

        revenue = sum(payments.values())
        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected),
            is_incentive_compatible=False,  # GSP is not DSIC
            efficiency_ratio=1.0,
            metadata={"mechanism": "position_auction", "k": self.k},
        )


# ---------------------------------------------------------------------------
# Combinatorial clock auction
# ---------------------------------------------------------------------------

class CombinatorialClockAuction:
    """Combinatorial clock auction (CCA) for diverse selection.

    Phase 1: Clock phase - prices increase, bidders indicate demand
    Phase 2: Supplementary round - sealed bids
    Phase 3: Winner determination with activity rules
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        initial_price: float = 0.01,
        price_increment: float = 0.05,
        max_clock_rounds: int = 100,
    ):
        self.items = items
        self.bidders = bidders
        self.initial_price = initial_price
        self.price_increment = price_increment
        self.max_clock_rounds = max_clock_rounds

    def _demand_at_price(
        self, bidder: Bidder, prices: Dict[int, float],
    ) -> FrozenSet[int]:
        """Compute bidder's demanded bundle at given prices."""
        # Greedy demand: items with positive surplus, sorted by value/price
        surpluses: List[Tuple[int, float]] = []
        for item in self.items:
            val = bidder.compute_value(frozenset([item.index]), self.items)
            price = prices.get(item.index, 0.0)
            surplus = val - price
            if surplus > 0:
                surpluses.append((item.index, surplus))
        surpluses.sort(key=lambda x: x[1], reverse=True)
        demanded = frozenset(idx for idx, _ in surpluses[:5])
        return demanded

    def clock_phase(self) -> Tuple[Dict[int, float], Dict[int, FrozenSet[int]]]:
        """Run the clock phase: iteratively increase prices."""
        prices = {item.index: self.initial_price for item in self.items}
        demands: Dict[int, FrozenSet[int]] = {}

        for round_num in range(self.max_clock_rounds):
            # Compute demands
            new_demands = {}
            for bidder in self.bidders:
                new_demands[bidder.bidder_id] = self._demand_at_price(bidder, prices)
            demands = new_demands

            # Check for excess demand
            item_demand = {item.index: 0 for item in self.items}
            for bundle in demands.values():
                for idx in bundle:
                    item_demand[idx] = item_demand.get(idx, 0) + 1

            excess = any(d > 1 for d in item_demand.values())
            if not excess:
                break

            # Increase prices for over-demanded items
            for idx, d in item_demand.items():
                if d > 1:
                    prices[idx] += self.price_increment

        return prices, demands

    def run(self, k: int) -> AuctionResult:
        """Run full CCA mechanism."""
        prices, demands = self.clock_phase()

        # Resolve conflicts greedily
        selected: List[int] = []
        allocation: Dict[int, FrozenSet[int]] = {}
        payments: Dict[int, float] = {}
        total_welfare = 0.0

        # Sort bidders by total demand value
        bidder_values = []
        for bidder in self.bidders:
            demand = demands.get(bidder.bidder_id, frozenset())
            val = bidder.compute_value(demand, self.items)
            bidder_values.append((bidder, demand, val))
        bidder_values.sort(key=lambda x: x[2], reverse=True)

        used_items: Set[int] = set()
        for bidder, demand, val in bidder_values:
            feasible = frozenset(idx for idx in demand if idx not in used_items)
            if len(feasible) == 0:
                continue
            # Trim to k
            if len(used_items) + len(feasible) > k:
                n_can_take = k - len(used_items)
                feasible_list = sorted(feasible,
                    key=lambda i: prices.get(i, 0), reverse=True)[:n_can_take]
                feasible = frozenset(feasible_list)

            allocation[bidder.bidder_id] = feasible
            payment = sum(prices.get(idx, 0.0) for idx in feasible)
            payments[bidder.bidder_id] = payment
            total_welfare += val
            used_items |= feasible
            selected.extend(feasible)

            if len(used_items) >= k:
                break

        revenue = sum(payments.values())
        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected[:k]),
            is_incentive_compatible=False,
            efficiency_ratio=1.0,
            metadata={"mechanism": "combinatorial_clock", "k": k,
                       "final_prices": prices},
        )


# ---------------------------------------------------------------------------
# Proportional Share Auction
# ---------------------------------------------------------------------------

class ProportionalShareAuction:
    """Proportional share auction where payments are proportional to bids.

    Each bidder pays proportional to their share of total bids.
    Items are allocated proportionally to bid amounts.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        k: int = 5,
    ):
        self.items = items
        self.bidders = bidders
        self.k = k

    def run(self) -> AuctionResult:
        """Run proportional share auction."""
        n_items = len(self.items)
        n_bidders = len(self.bidders)

        # Each bidder submits bids on each item
        bid_matrix = np.zeros((n_bidders, n_items))
        for bi, bidder in enumerate(self.bidders):
            for item in self.items:
                val = bidder.compute_value(frozenset([item.index]), self.items)
                bid_matrix[bi, item.index] = max(val, 0)

        # Item scores = sum of bids
        item_scores = np.sum(bid_matrix, axis=0)
        top_k = np.argsort(item_scores)[-self.k:][::-1]
        selected = set(top_k.tolist())

        # Payments proportional to share
        allocation: Dict[int, FrozenSet[int]] = {}
        payments: Dict[int, float] = {}
        total_welfare = 0.0

        for bi, bidder in enumerate(self.bidders):
            bid_id = bidder.bidder_id
            allocated_items: Set[int] = set()
            payment = 0.0
            for item_idx in top_k:
                total_bid = np.sum(bid_matrix[:, item_idx])
                if total_bid > 1e-12:
                    share = bid_matrix[bi, item_idx] / total_bid
                    if share > 0.5:  # majority bidder gets the item
                        allocated_items.add(int(item_idx))
                    payment += share * bid_matrix[bi, item_idx]
            allocation[bid_id] = frozenset(allocated_items)
            payments[bid_id] = payment
            total_welfare += sum(bid_matrix[bi, idx] for idx in allocated_items)

        revenue = sum(payments.values())
        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=selected,
            is_incentive_compatible=False,
            efficiency_ratio=1.0,
            metadata={"mechanism": "proportional_share", "k": self.k},
        )


# ---------------------------------------------------------------------------
# Dutch (Descending Price) Auction
# ---------------------------------------------------------------------------

class DutchAuction:
    """Dutch (descending price) auction for diverse item selection.

    Price starts high and decreases. First bidder to accept wins.
    Strategically equivalent to first-price sealed-bid.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        start_price: float = 10.0,
        price_decrement: float = 0.01,
        min_price: float = 0.0,
        diversity_bonus: float = 0.0,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.bidders = bidders
        self.start_price = start_price
        self.price_decrement = price_decrement
        self.min_price = min_price
        self.diversity_bonus = diversity_bonus
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _bidder_willingness_to_pay(
        self, bidder: Bidder, item_idx: int, selected: List[int],
    ) -> float:
        """Maximum price a bidder would accept."""
        base_val = bidder.compute_value(frozenset([item_idx]), self.items)
        if self.diversity_bonus > 0 and len(selected) > 0:
            emb = self.items[item_idx].embedding
            min_dist = min(
                np.linalg.norm(emb - self.items[s].embedding) for s in selected
            )
            base_val += self.diversity_bonus * min_dist
        return base_val

    def auction_single_item(
        self, item_idx: int, selected: List[int],
    ) -> Tuple[int, float]:
        """Run descending auction for one item."""
        price = self.start_price
        wtp = [
            self._bidder_willingness_to_pay(b, item_idx, selected)
            for b in self.bidders
        ]
        # Price descends; first bidder whose WTP >= price wins
        while price >= self.min_price:
            for bi, bidder in enumerate(self.bidders):
                if wtp[bi] >= price:
                    return bidder.bidder_id, price
            price -= self.price_decrement

        return -1, 0.0

    def run(self, k: int) -> AuctionResult:
        """Select k items via sequential Dutch auctions."""
        selected: List[int] = []
        allocation: Dict[int, FrozenSet[int]] = {
            b.bidder_id: frozenset() for b in self.bidders
        }
        payments: Dict[int, float] = {b.bidder_id: 0.0 for b in self.bidders}
        total_welfare = 0.0

        # Score items by aggregate value
        item_scores = []
        for item in self.items:
            score = sum(
                b.compute_value(frozenset([item.index]), self.items)
                for b in self.bidders
            )
            item_scores.append((item.index, score))
        item_scores.sort(key=lambda x: x[1], reverse=True)

        for item_idx, _ in item_scores:
            if len(selected) >= k:
                break
            if item_idx in selected:
                continue
            winner_id, price = self.auction_single_item(item_idx, selected)
            if winner_id >= 0:
                selected.append(item_idx)
                old_b = allocation.get(winner_id, frozenset())
                allocation[winner_id] = old_b | frozenset([item_idx])
                payments[winner_id] = payments.get(winner_id, 0.0) + price
                total_welfare += price

        revenue = sum(payments.values())
        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected),
            is_incentive_compatible=False,
            efficiency_ratio=1.0,
            metadata={"mechanism": "dutch_auction", "k": k},
        )


# ---------------------------------------------------------------------------
# All-Pay Auction
# ---------------------------------------------------------------------------

class AllPayAuction:
    """All-pay auction where all bidders pay their bids.

    Used for modeling competitive diversity: all aspect-bidders
    expend effort, but only the winner gets the item.
    """

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        k: int = 5,
    ):
        self.items = items
        self.bidders = bidders
        self.k = k

    def _optimal_bid(self, bidder: Bidder, item_idx: int) -> float:
        """Compute optimal all-pay bid (shade by 1/n)."""
        true_val = bidder.compute_value(frozenset([item_idx]), self.items)
        n_bidders = len(self.bidders)
        # In equilibrium, bid = v * (n-1)/n * F(v)^{n-1}
        # Simplified: shade proportional to number of competitors
        return true_val * (n_bidders - 1) / n_bidders

    def run(self) -> AuctionResult:
        """Run all-pay auction for k items."""
        all_bids: List[Tuple[int, int, float, float]] = []
        for bidder in self.bidders:
            for item in self.items:
                true_val = bidder.compute_value(frozenset([item.index]), self.items)
                bid = self._optimal_bid(bidder, item.index)
                all_bids.append((bidder.bidder_id, item.index, bid, true_val))

        # Each bidder pays their bid regardless of winning
        total_payments: Dict[int, float] = {b.bidder_id: 0.0 for b in self.bidders}
        for bidder_id, item_idx, bid, _ in all_bids:
            total_payments[bidder_id] += bid / len(self.items)  # normalize

        # Winners: highest bid per item
        item_winners: Dict[int, Tuple[int, float]] = {}
        for bidder_id, item_idx, bid, true_val in all_bids:
            if item_idx not in item_winners or bid > item_winners[item_idx][1]:
                item_winners[item_idx] = (bidder_id, bid)

        # Select top-k items by winning bid
        sorted_items = sorted(item_winners.items(), key=lambda x: x[1][1], reverse=True)
        selected: List[int] = [idx for idx, _ in sorted_items[:self.k]]

        allocation: Dict[int, FrozenSet[int]] = {}
        for item_idx in selected:
            winner_id = item_winners[item_idx][0]
            old = allocation.get(winner_id, frozenset())
            allocation[winner_id] = old | frozenset([item_idx])

        total_welfare = sum(
            item_winners[idx][1] for idx in selected
        )
        revenue = sum(total_payments.values())

        return AuctionResult(
            allocation=allocation,
            payments=total_payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=set(selected),
            is_incentive_compatible=False,
            efficiency_ratio=1.0,
            metadata={"mechanism": "all_pay", "k": self.k},
        )


# ---------------------------------------------------------------------------
# Double Auction for Diversity Trading
# ---------------------------------------------------------------------------

class DiversityDoubleAuction:
    """Double auction where diversity-demanders and diversity-suppliers trade.

    Buyers want diverse subsets; sellers offer individual responses.
    """

    def __init__(
        self,
        items: List[Item],
        n_buyers: int = 3,
        buyer_valuations: Optional[List[Callable[[FrozenSet[int], List[Item]], float]]] = None,
        seller_costs: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.items = items
        self.n_items = len(items)
        self.n_buyers = n_buyers
        self.rng = np.random.RandomState(seed)

        if buyer_valuations is not None:
            self.buyer_valuations = buyer_valuations
        else:
            self.buyer_valuations = [
                lambda bundle, items: sum(items[i].quality_score for i in bundle)
                for _ in range(n_buyers)
            ]

        if seller_costs is not None:
            self.seller_costs = seller_costs
        else:
            self.seller_costs = self.rng.uniform(0.1, 0.5, size=self.n_items)

    def _find_clearing_price(
        self,
        buyer_bids: List[float],
        seller_asks: List[float],
    ) -> Optional[float]:
        """Find market-clearing price."""
        buyer_bids_sorted = sorted(buyer_bids, reverse=True)
        seller_asks_sorted = sorted(seller_asks)

        for i in range(min(len(buyer_bids_sorted), len(seller_asks_sorted))):
            if buyer_bids_sorted[i] >= seller_asks_sorted[i]:
                return (buyer_bids_sorted[i] + seller_asks_sorted[i]) / 2
        return None

    def run(self, k: int = 5) -> AuctionResult:
        """Run double auction."""
        # Buyers submit bids for each item
        buyer_bids: Dict[int, Dict[int, float]] = {}
        for b in range(self.n_buyers):
            buyer_bids[b] = {}
            for item in self.items:
                bundle = frozenset([item.index])
                val = self.buyer_valuations[b](bundle, self.items)
                # Shade bid
                buyer_bids[b][item.index] = val * 0.9

        # Sellers submit asks
        seller_asks = {i: float(self.seller_costs[i]) for i in range(self.n_items)}

        # Match
        trades: List[Tuple[int, int, float]] = []  # (buyer, item, price)
        for item_idx in range(self.n_items):
            bids = [(b, buyer_bids[b][item_idx]) for b in range(self.n_buyers)]
            bids.sort(key=lambda x: x[1], reverse=True)
            ask = seller_asks[item_idx]

            if bids[0][1] >= ask:
                price = (bids[0][1] + ask) / 2
                trades.append((bids[0][0], item_idx, price))

        # Select top-k trades by price
        trades.sort(key=lambda x: x[2], reverse=True)
        trades = trades[:k]

        selected = set(item_idx for _, item_idx, _ in trades)
        allocation: Dict[int, FrozenSet[int]] = {}
        payments: Dict[int, float] = {}
        for buyer, item_idx, price in trades:
            old = allocation.get(buyer, frozenset())
            allocation[buyer] = old | frozenset([item_idx])
            payments[buyer] = payments.get(buyer, 0.0) + price

        total_welfare = sum(p for _, _, p in trades)
        revenue = sum(payments.values())

        return AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=total_welfare,
            revenue=revenue,
            selected_items=selected,
            is_incentive_compatible=False,
            efficiency_ratio=1.0,
            metadata={"mechanism": "double_auction", "n_trades": len(trades)},
        )


# ---------------------------------------------------------------------------
# Auction simulation and analysis
# ---------------------------------------------------------------------------

class AuctionSimulator:
    """Monte Carlo simulation of auction outcomes for statistical analysis."""

    def __init__(
        self,
        n_items: int,
        n_bidders: int,
        dim: int,
        k: int = 5,
        kernel: Optional[Kernel] = None,
    ):
        self.n_items = n_items
        self.n_bidders = n_bidders
        self.dim = dim
        self.k = k
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def _generate_instance(
        self, seed: int,
    ) -> Tuple[List[Item], List[Bidder]]:
        """Generate a random auction instance."""
        rng = np.random.RandomState(seed)
        embeddings = rng.randn(self.n_items, self.dim)
        qualities = rng.uniform(0.3, 0.9, size=self.n_items)
        items = [
            Item(index=i, embedding=embeddings[i], quality_score=float(qualities[i]))
            for i in range(self.n_items)
        ]
        bidders = create_aspect_bidders(self.n_items, self.dim, items, kernel=self.kernel)
        return items, bidders

    def simulate(
        self,
        mechanism: str,
        n_simulations: int = 100,
        base_seed: int = 42,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation of an auction mechanism."""
        revenues, welfares, efficiencies = [], [], []
        ic_flags, ir_flags = [], []
        diversities = []

        for sim in range(n_simulations):
            seed = base_seed + sim
            items, bidders = self._generate_instance(seed)
            embeddings = np.array([item.embedding for item in items])
            quality = np.array([item.quality_score for item in items])

            try:
                result = run_diversity_auction(
                    embeddings, quality, k=self.k,
                    mechanism=mechanism, kernel=self.kernel,
                )
                revenues.append(result.revenue)
                welfares.append(result.social_welfare)
                ic_flags.append(float(result.is_incentive_compatible))
                efficiencies.append(result.efficiency_ratio)

                sel = sorted(result.selected_items)
                if len(sel) >= 2:
                    K = self.kernel.gram_matrix(embeddings[sel])
                    div = log_det_safe(K)
                else:
                    div = 0.0
                diversities.append(div)

                # IR check
                analyzer = RevenueAnalyzer(items, bidders)
                ir = analyzer.individual_rationality_check(result)
                ir_flags.append(float(all(ir.values())))
            except Exception:
                continue

        return {
            "mechanism": mechanism,
            "n_simulations": len(revenues),
            "revenue": {
                "mean": float(np.mean(revenues)) if revenues else 0,
                "std": float(np.std(revenues)) if revenues else 0,
                "median": float(np.median(revenues)) if revenues else 0,
            },
            "welfare": {
                "mean": float(np.mean(welfares)) if welfares else 0,
                "std": float(np.std(welfares)) if welfares else 0,
            },
            "diversity": {
                "mean": float(np.mean(diversities)) if diversities else 0,
                "std": float(np.std(diversities)) if diversities else 0,
            },
            "ic_rate": float(np.mean(ic_flags)) if ic_flags else 0,
            "ir_rate": float(np.mean(ir_flags)) if ir_flags else 0,
        }

    def compare_mechanisms(
        self,
        mechanisms: Optional[List[str]] = None,
        n_simulations: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple mechanisms via simulation."""
        if mechanisms is None:
            mechanisms = ["vcg", "second_price", "english", "sealed_bid"]
        results = {}
        for mech in mechanisms:
            results[mech] = self.simulate(mech, n_simulations)
        return results


# ---------------------------------------------------------------------------
# Optimal reserve price search
# ---------------------------------------------------------------------------

class ReservePriceOptimizer:
    """Find optimal reserve price for diversity auctions."""

    def __init__(
        self,
        items: List[Item],
        bidders: List[Bidder],
        k: int = 5,
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.bidders = bidders
        self.k = k
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def grid_search(
        self,
        reserve_range: Tuple[float, float] = (0.0, 2.0),
        n_grid: int = 20,
        n_trials: int = 10,
    ) -> Tuple[float, float]:
        """Find revenue-maximizing reserve price via grid search."""
        grid = np.linspace(reserve_range[0], reserve_range[1], n_grid)
        best_reserve = 0.0
        best_revenue = -float("inf")

        for reserve in grid:
            revenues = []
            for trial in range(n_trials):
                auction = SecondPriceAuction(
                    self.items, self.bidders,
                    reserve_price=reserve,
                    kernel=self.kernel,
                )
                result = auction.run_sequential(self.k)
                revenues.append(result.revenue)
            mean_rev = float(np.mean(revenues))
            if mean_rev > best_revenue:
                best_revenue = mean_rev
                best_reserve = float(reserve)

        return best_reserve, best_revenue

    def binary_search(
        self,
        lo: float = 0.0,
        hi: float = 5.0,
        n_iterations: int = 20,
        n_trials: int = 10,
    ) -> Tuple[float, float]:
        """Find optimal reserve via golden section search."""
        golden = (math.sqrt(5) - 1) / 2

        for _ in range(n_iterations):
            x1 = hi - golden * (hi - lo)
            x2 = lo + golden * (hi - lo)

            r1 = self._evaluate_reserve(x1, n_trials)
            r2 = self._evaluate_reserve(x2, n_trials)

            if r1 < r2:
                lo = x1
            else:
                hi = x2

        best_reserve = (lo + hi) / 2
        best_revenue = self._evaluate_reserve(best_reserve, n_trials)
        return best_reserve, best_revenue

    def _evaluate_reserve(self, reserve: float, n_trials: int) -> float:
        """Evaluate revenue at a given reserve price."""
        revenues = []
        for trial in range(n_trials):
            auction = SecondPriceAuction(
                self.items, self.bidders,
                reserve_price=reserve,
                kernel=self.kernel,
            )
            result = auction.run_sequential(self.k)
            revenues.append(result.revenue)
        return float(np.mean(revenues))


# ---------------------------------------------------------------------------
# Welfare decomposition
# ---------------------------------------------------------------------------

class WelfareDecomposition:
    """Decompose social welfare into quality and diversity components."""

    def __init__(
        self,
        items: List[Item],
        kernel: Optional[Kernel] = None,
    ):
        self.items = items
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def decompose(
        self, selected_indices: Set[int],
    ) -> Dict[str, float]:
        """Decompose welfare of selected items."""
        if len(selected_indices) == 0:
            return {"quality": 0.0, "diversity": 0.0, "total": 0.0, "synergy": 0.0}

        indices = sorted(selected_indices)
        # Quality component
        quality = float(np.mean([self.items[i].quality_score for i in indices]))

        # Diversity component
        diversity = 0.0
        if len(indices) >= 2:
            embs = np.array([self.items[i].embedding for i in indices])
            K = self.kernel.gram_matrix(embs)
            diversity = log_det_safe(K)

        # Individual quality sum
        sum_quality = sum(self.items[i].quality_score for i in indices)

        # Synergy: super-additive value
        total = quality + max(diversity, 0)
        sum_individual = sum(
            self.items[i].quality_score for i in indices
        ) / len(indices)
        synergy = total - sum_individual

        return {
            "quality": quality,
            "diversity": diversity,
            "total": total,
            "synergy": synergy,
            "n_selected": len(indices),
        }

    def marginal_contribution_analysis(
        self,
        selected_indices: List[int],
    ) -> Dict[int, Dict[str, float]]:
        """Compute marginal contribution of each selected item."""
        contributions: Dict[int, Dict[str, float]] = {}
        full_decomp = self.decompose(set(selected_indices))

        for i in selected_indices:
            without_i = [j for j in selected_indices if j != i]
            without_decomp = self.decompose(set(without_i))
            contributions[i] = {
                "marginal_quality": full_decomp["quality"] - without_decomp["quality"],
                "marginal_diversity": full_decomp["diversity"] - without_decomp["diversity"],
                "marginal_total": full_decomp["total"] - without_decomp["total"],
                "item_quality": self.items[i].quality_score,
            }

        return contributions
