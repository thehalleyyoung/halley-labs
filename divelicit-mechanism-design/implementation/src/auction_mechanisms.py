"""Auction-based diverse elicitation mechanisms.

Implements several auction and mechanism-design approaches for eliciting
diverse responses from multiple agents/bidders:

- Diversity Auction: sealed-bid second-price with diversity bonus
- VCG with Diversity: Vickrey-Clarke-Groves with diversity welfare
- Combinatorial Auction: bundle-based bidding with diversity coverage
- Ascending Clock Auction: dynamic pricing with diversity incentives
- All-Pay Contest: effort-based competition with diversity scoring
- Budget-Balanced Mechanism: AGV transfers for diverse reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Bidder:
    """A bidder participating in an auction."""

    id: str
    embedding: np.ndarray
    budget: float = float("inf")
    active: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class Item:
    """An item that can be auctioned."""

    id: str
    embedding: np.ndarray
    reserve_price: float = 0.0
    quality: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Lot:
    """A bundle of items in a combinatorial auction."""

    id: str
    items: List[Item]
    region_label: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def centroid(self) -> np.ndarray:
        """Centroid embedding of all items in the lot."""
        if not self.items:
            return np.zeros(1)
        return np.mean([it.embedding for it in self.items], axis=0)

    @property
    def size(self) -> int:
        return len(self.items)


@dataclass
class Entry:
    """An entry in an all-pay contest."""

    id: str
    embedding: np.ndarray
    bid: float
    content: Optional[str] = None
    quality_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class AuctionResult:
    """Result of a diversity auction."""

    winners: List[str]
    payments: Dict[str, float]
    items_won: Dict[str, List[str]]
    diversity_bonuses: Dict[str, float]
    total_welfare: float
    total_revenue: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class VCGResult:
    """Result of a VCG mechanism with diversity."""

    allocation: Dict[str, List[str]]
    payments: Dict[str, float]
    social_welfare: float
    diversity_welfare: float
    individual_welfare: Dict[str, float]
    metadata: Dict = field(default_factory=dict)


@dataclass
class CAResult:
    """Result of a combinatorial auction."""

    winning_lots: List[str]
    lot_winners: Dict[str, str]
    payments: Dict[str, float]
    total_welfare: float
    coverage_score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class AscendingResult:
    """Result of an ascending clock auction."""

    winners: Dict[str, str]
    final_prices: Dict[str, float]
    rounds_completed: int
    dropout_history: List[Tuple[int, str, str]]
    total_revenue: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class ContestResult:
    """Result of an all-pay contest."""

    rankings: List[str]
    scores: Dict[str, float]
    quality_scores: Dict[str, float]
    diversity_scores: Dict[str, float]
    prizes: Dict[str, float]
    total_effort: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class BBResult:
    """Result of a budget-balanced mechanism."""

    allocation: List[str]
    transfers: Dict[str, float]
    budget_surplus: float
    reports_used: Dict[str, np.ndarray]
    diversity_score: float
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors.

    Returns 1 - cosine_similarity.  Returns 1.0 when either vector has
    zero norm (maximally distant).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    similarity = np.dot(a, b) / (norm_a * norm_b)
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(1.0 - similarity)


def _diversity_contribution(
    embedding: np.ndarray,
    selected_embeddings: List[np.ndarray],
) -> float:
    """Diversity contribution of *embedding* relative to already-selected set.

    Defined as the minimum cosine distance to the nearest member of
    *selected_embeddings*.  Returns 1.0 when the selected set is empty
    (maximum possible contribution).
    """
    if not selected_embeddings:
        return 1.0
    distances = [_cosine_distance(embedding, s) for s in selected_embeddings]
    return float(min(distances))


def _greedy_welfare_max(
    items: List[Item],
    bidders: List[Bidder],
    diversity_fn: Callable[[np.ndarray, List[np.ndarray]], float],
) -> Tuple[Dict[str, List[str]], float]:
    """Greedy welfare-maximising allocation.

    Iteratively assigns each item to the bidder whose marginal welfare gain
    (valuation + diversity contribution) is highest.  Valuations are modelled
    as the negative cosine distance between a bidder's embedding and the
    item's embedding (bidders prefer items close to their own embedding).

    Returns
    -------
    allocation : dict mapping bidder id -> list of item ids
    total_welfare : float
    """
    allocation: Dict[str, List[str]] = {b.id: [] for b in bidders}
    selected_embeddings: List[np.ndarray] = []
    total_welfare = 0.0

    remaining_items = list(items)
    np.random.shuffle(remaining_items)

    for item in remaining_items:
        best_bidder: Optional[str] = None
        best_gain = -np.inf

        for bidder in bidders:
            if not bidder.active:
                continue
            # Valuation: similarity between bidder and item
            valuation = 1.0 - _cosine_distance(bidder.embedding, item.embedding)
            div_gain = diversity_fn(item.embedding, selected_embeddings)
            marginal = valuation + div_gain
            if marginal > best_gain:
                best_gain = marginal
                best_bidder = bidder.id

        if best_bidder is not None and best_gain > 0:
            allocation[best_bidder].append(item.id)
            selected_embeddings.append(item.embedding)
            total_welfare += best_gain

    return allocation, total_welfare


def _compute_vcg_payment(
    winner_id: str,
    bidders: List[Bidder],
    allocation: Dict[str, List[str]],
    items: List[Item],
    diversity_fn: Callable[[np.ndarray, List[np.ndarray]], float],
) -> float:
    """VCG payment for one bidder.

    Payment = (welfare of others without *winner_id*) - (welfare of others
    with *winner_id* present).  This captures the externality the winner
    imposes on the remaining bidders.
    """
    item_map: Dict[str, Item] = {it.id: it for it in items}

    # --- welfare of others WITH winner present (current allocation) ---
    others_welfare_with = 0.0
    selected_with: List[np.ndarray] = []
    for bid in bidders:
        if bid.id == winner_id:
            continue
        for iid in allocation.get(bid.id, []):
            it = item_map[iid]
            val = 1.0 - _cosine_distance(bid.embedding, it.embedding)
            div = diversity_fn(it.embedding, selected_with)
            others_welfare_with += val + div
            selected_with.append(it.embedding)

    # --- welfare of others WITHOUT winner (re-optimise) ---
    other_bidders = [b for b in bidders if b.id != winner_id]
    _, others_welfare_without = _greedy_welfare_max(items, other_bidders, diversity_fn)

    payment = others_welfare_without - others_welfare_with
    return max(payment, 0.0)


# ---------------------------------------------------------------------------
# 1. Diversity Auction  (sealed-bid second-price + diversity bonus)
# ---------------------------------------------------------------------------

def diversity_auction(
    items: List[Item],
    bidders: List[Bidder],
    diversity_bonus: float = 0.1,
) -> AuctionResult:
    """Run a sealed-bid second-price auction with diversity bonus.

    Each bidder submits a bid for every item equal to its valuation (similarity
    between bidder and item embeddings).  A *diversity_bonus* is added when
    an item would increase the diversity of the already-selected winner set.
    Items are processed greedily in descending order of augmented bid.

    Payments follow second-price logic: the winner pays the second-highest
    augmented bid.

    Parameters
    ----------
    items : list of Item
    bidders : list of Bidder
    diversity_bonus : float
        Multiplier for the diversity contribution added to raw bids.

    Returns
    -------
    AuctionResult
    """
    if not items or not bidders:
        return AuctionResult(
            winners=[], payments={}, items_won={},
            diversity_bonuses={}, total_welfare=0.0, total_revenue=0.0,
        )

    selected_embeddings: List[np.ndarray] = []
    winners: List[str] = []
    payments: Dict[str, float] = {b.id: 0.0 for b in bidders}
    items_won: Dict[str, List[str]] = {b.id: [] for b in bidders}
    diversity_bonuses: Dict[str, float] = {b.id: 0.0 for b in bidders}
    total_welfare = 0.0
    total_revenue = 0.0

    # Build bid matrix: (item, bidder) -> raw valuation
    bid_matrix: Dict[str, Dict[str, float]] = {}
    for item in items:
        bid_matrix[item.id] = {}
        for bidder in bidders:
            val = 1.0 - _cosine_distance(bidder.embedding, item.embedding)
            bid_matrix[item.id][bidder.id] = max(val, 0.0)

    # Rank items by maximum raw bid (process most-wanted first)
    item_order = sorted(
        items,
        key=lambda it: max(bid_matrix[it.id].values()),
        reverse=True,
    )

    for item in item_order:
        if item.quality < 0:
            continue

        augmented: List[Tuple[str, float, float]] = []
        for bidder in bidders:
            if not bidder.active:
                continue
            raw_bid = bid_matrix[item.id][bidder.id]
            if raw_bid < item.reserve_price:
                continue
            div_contrib = _diversity_contribution(item.embedding, selected_embeddings)
            bonus = diversity_bonus * div_contrib
            aug_bid = raw_bid + bonus
            augmented.append((bidder.id, aug_bid, bonus))

        if not augmented:
            continue

        augmented.sort(key=lambda x: x[1], reverse=True)
        winner_id, winner_aug, winner_bonus = augmented[0]

        # Second-price: pay second-highest augmented bid (or reserve)
        if len(augmented) >= 2:
            second_price = augmented[1][1]
        else:
            second_price = item.reserve_price

        payment = max(second_price, item.reserve_price)

        if winner_id not in winners:
            winners.append(winner_id)
        payments[winner_id] += payment
        items_won[winner_id].append(item.id)
        diversity_bonuses[winner_id] += winner_bonus
        selected_embeddings.append(item.embedding)
        total_welfare += winner_aug
        total_revenue += payment

    return AuctionResult(
        winners=winners,
        payments=payments,
        items_won=items_won,
        diversity_bonuses=diversity_bonuses,
        total_welfare=total_welfare,
        total_revenue=total_revenue,
    )


# ---------------------------------------------------------------------------
# 2. VCG with Diversity
# ---------------------------------------------------------------------------

def vcg_with_diversity(
    items: List[Item],
    valuations: Dict[str, Dict[str, float]],
    diversity_fn: Callable[[np.ndarray, List[np.ndarray]], float],
) -> VCGResult:
    """Vickrey-Clarke-Groves mechanism with diversity in welfare.

    Computes the social-welfare-maximising allocation where welfare is the
    sum of bidders' valuations *plus* a diversity term evaluated via
    *diversity_fn*.  Then VCG payments are computed so that each bidder
    pays the externality they impose on everyone else.

    Parameters
    ----------
    items : list of Item
    valuations : dict  bidder_id -> {item_id: value}
        Explicit per-item valuations from each bidder.
    diversity_fn : callable(embedding, selected_embeddings) -> float

    Returns
    -------
    VCGResult
    """
    bidder_ids = list(valuations.keys())
    item_map: Dict[str, Item] = {it.id: it for it in items}

    # --- Find welfare-maximising allocation (greedy) ---
    allocation: Dict[str, List[str]] = {bid: [] for bid in bidder_ids}
    selected_embeddings: List[np.ndarray] = []
    individual_welfare: Dict[str, float] = {bid: 0.0 for bid in bidder_ids}
    total_diversity = 0.0
    total_welfare = 0.0

    assigned_items: set = set()

    # Enumerate all (item, bidder) gains
    candidates: List[Tuple[float, str, str, float]] = []
    for it in items:
        for bid in bidder_ids:
            val = valuations[bid].get(it.id, 0.0)
            if val > 0:
                candidates.append((val, bid, it.id, val))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for _, bid, iid, val in candidates:
        if iid in assigned_items:
            continue
        emb = item_map[iid].embedding
        div_val = diversity_fn(emb, selected_embeddings)
        marginal = val + div_val
        if marginal <= 0:
            continue
        allocation[bid].append(iid)
        assigned_items.add(iid)
        selected_embeddings.append(emb)
        individual_welfare[bid] += val
        total_diversity += div_val
        total_welfare += marginal

    # --- Compute VCG payments ---
    payments: Dict[str, float] = {}
    for bid in bidder_ids:
        # Welfare of others with bid present
        others_welfare_with = 0.0
        sel_with: List[np.ndarray] = []
        for other in bidder_ids:
            if other == bid:
                continue
            for iid in allocation[other]:
                emb = item_map[iid].embedding
                v = valuations[other].get(iid, 0.0)
                d = diversity_fn(emb, sel_with)
                others_welfare_with += v + d
                sel_with.append(emb)

        # Welfare of others without bid (re-allocate)
        other_bidders = [b for b in bidder_ids if b != bid]
        other_vals = {b: valuations[b] for b in other_bidders}

        sel_without: List[np.ndarray] = []
        others_welfare_without = 0.0
        assigned_wo: set = set()

        cands_wo: List[Tuple[float, str, str]] = []
        for it in items:
            for ob in other_bidders:
                v = other_vals[ob].get(it.id, 0.0)
                if v > 0:
                    cands_wo.append((v, ob, it.id))
        cands_wo.sort(key=lambda x: x[0], reverse=True)

        for v, ob, iid in cands_wo:
            if iid in assigned_wo:
                continue
            emb = item_map[iid].embedding
            d = diversity_fn(emb, sel_without)
            m = v + d
            if m <= 0:
                continue
            assigned_wo.add(iid)
            sel_without.append(emb)
            others_welfare_without += m

        payments[bid] = max(others_welfare_without - others_welfare_with, 0.0)

    return VCGResult(
        allocation=allocation,
        payments=payments,
        social_welfare=total_welfare,
        diversity_welfare=total_diversity,
        individual_welfare=individual_welfare,
    )


# ---------------------------------------------------------------------------
# 3. Combinatorial Auction with Diversity
# ---------------------------------------------------------------------------

def _lot_diversity_score(
    lot: Lot,
    selected_lots: List[Lot],
) -> float:
    """Diversity score of a lot relative to already-selected lots.

    Measures how different the lot's centroid is from centroids of lots
    already selected.  Returns 1.0 when no lots have been selected yet.
    """
    if not selected_lots:
        return 1.0
    selected_centroids = [sl.centroid for sl in selected_lots]
    return _diversity_contribution(lot.centroid, selected_centroids)


def _lots_overlap(lot_a: Lot, lot_b: Lot) -> bool:
    """Check whether two lots share any items."""
    ids_a = {it.id for it in lot_a.items}
    ids_b = {it.id for it in lot_b.items}
    return bool(ids_a & ids_b)


def combinatorial_auction_diverse(
    lots: List[Lot],
    bidders: List[Bidder],
) -> CAResult:
    """Combinatorial auction with diversity bonus.

    Bidders bid on bundles (*lots*) of items.  A bidder's bid on a lot equals
    the average similarity between the bidder's embedding and the items in the
    lot.  Winner determination uses a greedy approximation to set packing:
    lots are sorted by augmented value (bid + diversity bonus) and accepted
    when they do not conflict with previously accepted lots.

    Parameters
    ----------
    lots : list of Lot
    bidders : list of Bidder

    Returns
    -------
    CAResult
    """
    if not lots or not bidders:
        return CAResult(
            winning_lots=[], lot_winners={}, payments={},
            total_welfare=0.0, coverage_score=0.0,
        )

    # Compute bids: each bidder bids on each lot
    # bid = mean similarity of bidder embedding to item embeddings in lot
    bids: Dict[str, Dict[str, float]] = {}
    for lot in lots:
        bids[lot.id] = {}
        for bidder in bidders:
            if lot.size == 0:
                bids[lot.id][bidder.id] = 0.0
                continue
            sims = [
                1.0 - _cosine_distance(bidder.embedding, it.embedding)
                for it in lot.items
            ]
            bids[lot.id][bidder.id] = float(np.mean(sims))

    # Greedy set-packing with diversity bonus
    selected_lots_objs: List[Lot] = []
    winning_lots: List[str] = []
    lot_winners: Dict[str, str] = {}
    payments: Dict[str, float] = {b.id: 0.0 for b in bidders}
    total_welfare = 0.0

    # Create (lot, bidder, augmented_value, raw_bid) tuples
    lot_map: Dict[str, Lot] = {l.id: l for l in lots}
    entries: List[Tuple[float, str, str, float]] = []
    for lot in lots:
        div_score = _lot_diversity_score(lot, [])
        for bidder in bidders:
            raw = bids[lot.id][bidder.id]
            aug = raw + 0.15 * div_score  # diversity coefficient
            entries.append((aug, lot.id, bidder.id, raw))

    entries.sort(key=lambda x: x[0], reverse=True)
    used_lots: set = set()

    for aug_val, lid, bid, raw in entries:
        if lid in used_lots:
            continue
        lot_obj = lot_map[lid]

        # Check overlap with already-selected lots
        conflict = False
        for sl in selected_lots_objs:
            if _lots_overlap(lot_obj, sl):
                conflict = True
                break
        if conflict:
            continue

        # Recompute diversity bonus with current selection
        div_bonus = _lot_diversity_score(lot_obj, selected_lots_objs)
        effective_val = raw + 0.15 * div_bonus

        if effective_val <= 0:
            continue

        # Second-price-like payment: find second-highest bid for this lot
        lot_bids_sorted = sorted(
            bids[lid].values(), reverse=True,
        )
        if len(lot_bids_sorted) >= 2:
            second_price = lot_bids_sorted[1]
        else:
            second_price = 0.0

        used_lots.add(lid)
        selected_lots_objs.append(lot_obj)
        winning_lots.append(lid)
        lot_winners[lid] = bid
        payments[bid] += max(second_price, 0.0)
        total_welfare += effective_val

    # Coverage score: fraction of distinct regions covered
    if winning_lots:
        all_regions = {l.region_label for l in lots if l.region_label is not None}
        covered = {
            lot_map[lid].region_label
            for lid in winning_lots
            if lot_map[lid].region_label is not None
        }
        coverage_score = len(covered) / max(len(all_regions), 1)
    else:
        coverage_score = 0.0

    return CAResult(
        winning_lots=winning_lots,
        lot_winners=lot_winners,
        payments=payments,
        total_welfare=total_welfare,
        coverage_score=coverage_score,
    )


# ---------------------------------------------------------------------------
# 4. Ascending Clock Auction
# ---------------------------------------------------------------------------

def ascending_clock_auction(
    items: List[Item],
    bidders: List[Bidder],
    max_rounds: int = 100,
    base_increment: float = 0.05,
    diversity_slowdown: float = 0.5,
) -> AscendingResult:
    """Ascending clock auction with diversity-aware price increments.

    Prices rise every round.  Items whose embeddings are in *under-
    represented* regions of the selected set receive slower price increases
    (controlled by *diversity_slowdown*), incentivising bidders to stay in
    the market for diverse items.

    A bidder remains active on an item while its valuation exceeds the
    current price.  When only one bidder remains for an item the item is
    sold at the current price.

    Parameters
    ----------
    items : list of Item
    bidders : list of Bidder
    max_rounds : int
    base_increment : float
        Default per-round price increase.
    diversity_slowdown : float in [0, 1]
        Factor by which the increment is reduced for under-represented items.

    Returns
    -------
    AscendingResult
    """
    if not items or not bidders:
        return AscendingResult(
            winners={}, final_prices={}, rounds_completed=0,
            dropout_history=[], total_revenue=0.0,
        )

    # Initialise per-item prices and active bidder sets
    prices: Dict[str, float] = {it.id: it.reserve_price for it in items}
    active_bidders: Dict[str, List[str]] = {
        it.id: [b.id for b in bidders if b.active] for it in items
    }
    item_map: Dict[str, Item] = {it.id: it for it in items}
    bidder_map: Dict[str, Bidder] = {b.id: b for b in bidders}

    # Precompute valuations (similarity)
    valuations: Dict[str, Dict[str, float]] = {}
    for bidder in bidders:
        valuations[bidder.id] = {}
        for item in items:
            valuations[bidder.id][item.id] = max(
                1.0 - _cosine_distance(bidder.embedding, item.embedding), 0.0,
            )

    winners: Dict[str, str] = {}
    final_prices: Dict[str, float] = {}
    dropout_history: List[Tuple[int, str, str]] = []
    sold_embeddings: List[np.ndarray] = []

    unsold = {it.id for it in items}
    round_num = 0

    for round_num in range(1, max_rounds + 1):
        newly_sold: List[str] = []

        for iid in list(unsold):
            item = item_map[iid]
            current_price = prices[iid]

            # Determine price increment based on diversity need
            div_contrib = _diversity_contribution(item.embedding, sold_embeddings)
            # High diversity contribution → slow increment (encourage buying)
            slowdown = 1.0 - diversity_slowdown * div_contrib
            increment = base_increment * max(slowdown, 0.1)
            prices[iid] = current_price + increment

            # Check which bidders drop out at new price
            remaining = []
            for bid in active_bidders[iid]:
                bidder = bidder_map[bid]
                val = valuations[bid][iid]
                if val >= prices[iid] and bidder.budget >= prices[iid]:
                    remaining.append(bid)
                else:
                    dropout_history.append((round_num, bid, iid))
            active_bidders[iid] = remaining

            # Sell if exactly one bidder left
            if len(remaining) == 1:
                winner_id = remaining[0]
                winners[iid] = winner_id
                final_prices[iid] = prices[iid]
                sold_embeddings.append(item.embedding)
                newly_sold.append(iid)
            elif len(remaining) == 0:
                # No bidders left; item unsold
                newly_sold.append(iid)
                final_prices[iid] = 0.0

        for iid in newly_sold:
            unsold.discard(iid)

        if not unsold:
            break

    # Handle remaining unsold items: sell to highest-valuation remaining bidder
    for iid in unsold:
        remaining = active_bidders[iid]
        if remaining:
            best = max(remaining, key=lambda b: valuations[b][iid])
            winners[iid] = best
            final_prices[iid] = prices[iid]
            sold_embeddings.append(item_map[iid].embedding)
        else:
            final_prices[iid] = 0.0

    total_revenue = sum(
        final_prices[iid] for iid in winners
    )

    return AscendingResult(
        winners=winners,
        final_prices=final_prices,
        rounds_completed=round_num,
        dropout_history=dropout_history,
        total_revenue=total_revenue,
    )


# ---------------------------------------------------------------------------
# 5. All-Pay Contest
# ---------------------------------------------------------------------------

def _rank_prize_share(rank: int, n_winners: int, prize_pool: float) -> float:
    """Harmonic prize share for rank *rank* (1-indexed) among *n_winners*.

    Winner 1 gets 1/H_k of the pool, winner 2 gets (1/2)/H_k, etc., where
    H_k = sum(1/i for i=1..k).
    """
    if rank > n_winners or rank < 1:
        return 0.0
    harmonic = sum(1.0 / i for i in range(1, n_winners + 1))
    return prize_pool * (1.0 / rank) / harmonic


def all_pay_contest(
    entries: List[Entry],
    judge_fn: Callable[[Entry], float],
    diversity_weight: float = 0.3,
    n_winners: int = 3,
) -> ContestResult:
    """All-pay contest with diversity-weighted scoring.

    Every entrant pays its bid (effort cost), but only top-ranked entries
    receive prizes.  The final score combines quality (from *judge_fn*) and
    diversity contribution.

    Parameters
    ----------
    entries : list of Entry
    judge_fn : callable(Entry) -> float
        Returns a quality score for an entry.
    diversity_weight : float in [0, 1]
        Weight given to diversity in the composite score.
    n_winners : int
        Number of prize slots.

    Returns
    -------
    ContestResult
    """
    if not entries:
        return ContestResult(
            rankings=[], scores={}, quality_scores={},
            diversity_scores={}, prizes={}, total_effort=0.0,
        )

    n_winners = min(n_winners, len(entries))
    quality_weight = 1.0 - diversity_weight

    # Evaluate quality scores
    quality_scores: Dict[str, float] = {}
    for entry in entries:
        qs = judge_fn(entry)
        entry.quality_score = qs
        quality_scores[entry.id] = qs

    # Normalise quality scores to [0, 1]
    q_vals = list(quality_scores.values())
    q_min, q_max = min(q_vals), max(q_vals)
    q_range = q_max - q_min if q_max > q_min else 1.0
    norm_quality: Dict[str, float] = {
        eid: (v - q_min) / q_range for eid, v in quality_scores.items()
    }

    # Compute diversity scores iteratively (greedy ordering)
    # First pass: tentative diversity contribution for each entry against
    # all others, then refine via greedy selection.
    diversity_scores: Dict[str, float] = {}
    all_embeddings = [e.embedding for e in entries]
    for entry in entries:
        others = [emb for e, emb in zip(entries, all_embeddings) if e.id != entry.id]
        diversity_scores[entry.id] = _diversity_contribution(entry.embedding, others)

    # Normalise diversity scores to [0, 1]
    d_vals = list(diversity_scores.values())
    d_min, d_max = min(d_vals), max(d_vals)
    d_range = d_max - d_min if d_max > d_min else 1.0
    norm_diversity: Dict[str, float] = {
        eid: (v - d_min) / d_range for eid, v in diversity_scores.items()
    }

    # Composite scores
    scores: Dict[str, float] = {}
    for entry in entries:
        scores[entry.id] = (
            quality_weight * norm_quality[entry.id]
            + diversity_weight * norm_diversity[entry.id]
        )

    # Rank entries by composite score (descending)
    ranked = sorted(entries, key=lambda e: scores[e.id], reverse=True)
    rankings = [e.id for e in ranked]

    # Prize allocation: total prize pool = sum of all bids
    total_effort = sum(e.bid for e in entries)
    prize_pool = total_effort  # redistribution model

    prizes: Dict[str, float] = {}
    for rank_idx, entry in enumerate(ranked):
        rank = rank_idx + 1
        prizes[entry.id] = _rank_prize_share(rank, n_winners, prize_pool)

    return ContestResult(
        rankings=rankings,
        scores=scores,
        quality_scores=quality_scores,
        diversity_scores=diversity_scores,
        prizes=prizes,
        total_effort=total_effort,
    )


# ---------------------------------------------------------------------------
# 6. Budget-Balanced Mechanism (AGV / expected externality)
# ---------------------------------------------------------------------------

def _pairwise_externality_matrix(
    reports: Dict[str, np.ndarray],
    diversity_bonus: float,
) -> np.ndarray:
    """Build the pairwise externality matrix.

    Entry (i, j) is the externality that agent *i*'s report imposes on
    agent *j*, measured as the reduction in diversity-adjusted value that
    agent *j* experiences due to agent *i*'s presence.

    The matrix is *not* symmetric in general.
    """
    agent_ids = sorted(reports.keys())
    n = len(agent_ids)
    ext = np.zeros((n, n), dtype=np.float64)

    for i, ai in enumerate(agent_ids):
        for j, aj in enumerate(agent_ids):
            if i == j:
                continue
            # Diversity loss: how much does ai's report reduce aj's contribution?
            others_without_i = [
                reports[ak] for ak in agent_ids if ak != ai and ak != aj
            ]
            others_with_i = others_without_i + [reports[ai]]

            dj_without = _diversity_contribution(reports[aj], others_without_i)
            dj_with = _diversity_contribution(reports[aj], others_with_i)
            ext[i, j] = diversity_bonus * max(dj_without - dj_with, 0.0)

    return ext


def budget_balanced_mechanism(
    agents: List[str],
    reports: Dict[str, np.ndarray],
    diversity_bonus: float = 0.1,
    quality_scores: Optional[Dict[str, float]] = None,
) -> BBResult:
    """Budget-balanced (AGV) mechanism for diverse reporting.

    Implements an expected-externality / d'Aspremont-Gérard-Varet style
    mechanism.  Each agent's transfer is set so that the sum of all
    transfers equals zero (budget balance), while incentivising reports
    that increase overall diversity.

    The transfer for agent *i* is:

        t_i  =  (sum over j≠i of externality j imposes on others excluding i)
              - (1/(n-1)) * (sum over j≠i of sum over k≠j of externality j→k)

    This ensures Σ t_i = 0.

    Parameters
    ----------
    agents : list of str
        Agent identifiers.
    reports : dict agent_id -> np.ndarray
        Reported embeddings from each agent.
    diversity_bonus : float
        Scaling for the diversity component of externalities.
    quality_scores : dict, optional
        External quality scores per agent (folded into the allocation).

    Returns
    -------
    BBResult
    """
    n = len(agents)
    if n < 2:
        transfers = {a: 0.0 for a in agents}
        div_score = 0.0
        if n == 1:
            div_score = 1.0
        return BBResult(
            allocation=agents,
            transfers=transfers,
            budget_surplus=0.0,
            reports_used=reports,
            diversity_score=div_score,
        )

    agent_ids = sorted(agents)
    idx = {a: i for i, a in enumerate(agent_ids)}

    # Build externality matrix
    ext = _pairwise_externality_matrix(reports, diversity_bonus)

    # Quality component (additive to value)
    if quality_scores is None:
        quality_scores = {a: 0.0 for a in agent_ids}

    # Compute transfers using AGV formula
    transfers: Dict[str, float] = {}
    for i, ai in enumerate(agent_ids):
        # Benefit term: sum of externalities others impose on the rest,
        # excluding agent i's perspective
        benefit = 0.0
        for j, aj in enumerate(agent_ids):
            if j == i:
                continue
            # Sum of externalities j imposes on agents other than i and j
            for k, ak in enumerate(agent_ids):
                if k == i or k == j:
                    continue
                benefit += ext[j, k]

        # Cost term: average externality budget
        cost = 0.0
        for j, aj in enumerate(agent_ids):
            if j == i:
                continue
            for k, ak in enumerate(agent_ids):
                if k == j:
                    continue
                cost += ext[j, k]

        if n > 1:
            transfers[ai] = benefit - cost / (n - 1)
        else:
            transfers[ai] = 0.0

        # Add quality-based component
        transfers[ai] += quality_scores.get(ai, 0.0) * diversity_bonus

    # Enforce exact budget balance by redistributing residual equally
    residual = sum(transfers.values())
    per_agent_adj = residual / n
    for ai in agent_ids:
        transfers[ai] -= per_agent_adj

    budget_surplus = sum(transfers.values())

    # Determine allocation: include all agents whose net utility is non-negative
    allocation: List[str] = []
    all_embeddings: List[np.ndarray] = []
    for ai in agent_ids:
        div = _diversity_contribution(reports[ai], all_embeddings)
        q = quality_scores.get(ai, 0.0)
        net = q + diversity_bonus * div + transfers[ai]
        if net >= 0 or len(allocation) < max(n // 2, 1):
            allocation.append(ai)
            all_embeddings.append(reports[ai])

    # Overall diversity score of the selected allocation
    if len(allocation) <= 1:
        diversity_score = 1.0 if allocation else 0.0
    else:
        sel = [reports[a] for a in allocation]
        pairwise = []
        for p in range(len(sel)):
            for q_idx in range(p + 1, len(sel)):
                pairwise.append(_cosine_distance(sel[p], sel[q_idx]))
        diversity_score = float(np.mean(pairwise)) if pairwise else 0.0

    return BBResult(
        allocation=allocation,
        transfers=transfers,
        budget_surplus=round(budget_surplus, 12),
        reports_used={a: reports[a] for a in allocation},
        diversity_score=diversity_score,
    )


# ---------------------------------------------------------------------------
# Convenience: run all mechanisms on the same input
# ---------------------------------------------------------------------------

def run_all_mechanisms(
    items: List[Item],
    bidders: List[Bidder],
    diversity_bonus: float = 0.1,
    judge_fn: Optional[Callable[[Entry], float]] = None,
) -> Dict[str, object]:
    """Run every mechanism and return a dict of results keyed by name.

    This is a convenience wrapper useful for benchmarking.  Missing inputs
    are constructed from the available data where possible.

    Parameters
    ----------
    items : list of Item
    bidders : list of Bidder
    diversity_bonus : float
    judge_fn : callable, optional
        Quality judge for the all-pay contest.  Defaults to using
        ``item.quality`` as score.

    Returns
    -------
    dict  mechanism_name -> result dataclass
    """
    results: Dict[str, object] = {}

    # 1. Diversity Auction
    results["diversity_auction"] = diversity_auction(items, bidders, diversity_bonus)

    # 2. VCG
    valuations: Dict[str, Dict[str, float]] = {}
    for bidder in bidders:
        valuations[bidder.id] = {}
        for item in items:
            valuations[bidder.id][item.id] = max(
                1.0 - _cosine_distance(bidder.embedding, item.embedding), 0.0,
            )
    results["vcg"] = vcg_with_diversity(
        items, valuations, _diversity_contribution,
    )

    # 3. Combinatorial (create single-item lots)
    lots = [
        Lot(id=f"lot_{it.id}", items=[it], region_label=it.metadata.get("region"))
        for it in items
    ]
    results["combinatorial"] = combinatorial_auction_diverse(lots, bidders)

    # 4. Ascending clock
    results["ascending_clock"] = ascending_clock_auction(items, bidders)

    # 5. All-pay contest (entries from items)
    if judge_fn is None:
        def judge_fn(e: Entry) -> float:
            return e.quality_score

    contest_entries = [
        Entry(
            id=it.id,
            embedding=it.embedding,
            bid=it.quality * 0.1,
            quality_score=it.quality,
        )
        for it in items
    ]
    results["all_pay_contest"] = all_pay_contest(
        contest_entries, judge_fn, diversity_weight=0.3,
    )

    # 6. Budget-balanced
    agent_reports = {b.id: b.embedding for b in bidders}
    results["budget_balanced"] = budget_balanced_mechanism(
        [b.id for b in bidders],
        agent_reports,
        diversity_bonus,
    )

    return results
