"""
Multi-item auction system: combinatorial, sequential, double auctions,
spectrum auctions, Walrasian equilibrium, and CEEI.
"""

import numpy as np
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


@dataclass
class Bid:
    """A bid on a bundle of items."""
    bidder_id: int
    bundle: Tuple[int, ...]
    value: float


@dataclass
class AuctionAllocation:
    """Result of an auction."""
    assignment: Dict[int, List[Tuple[int, ...]]]  # bidder -> list of won bundles
    payments: Dict[int, float]
    social_welfare: float
    revenue: float


class CombinatorialAuction:
    """
    Combinatorial auction with winner determination via ILP relaxation
    and branch-and-bound.
    """

    def __init__(self, items: List[int]):
        self.items = items
        self.n_items = len(items)
        self.bids: List[Bid] = []

    def add_bidder(self, bidder_id: int, valuations: Dict[Tuple[int, ...], float]):
        """Add a bidder with valuations over bundles."""
        for bundle, value in valuations.items():
            self.bids.append(Bid(bidder_id=bidder_id, bundle=tuple(sorted(bundle)), value=value))

    def _solve_lp_relaxation(self, bids: List[Bid]) -> Tuple[np.ndarray, float]:
        """Solve LP relaxation of winner determination."""
        from scipy.optimize import linprog

        n_bids = len(bids)
        if n_bids == 0:
            return np.array([]), 0.0

        # Maximize sum of values (minimize negation)
        c = np.array([-b.value for b in bids])

        # Each item can be allocated at most once
        A_ub = np.zeros((self.n_items, n_bids))
        for j, bid in enumerate(bids):
            for item in bid.bundle:
                if item < self.n_items:
                    A_ub[item][j] = 1.0
        b_ub = np.ones(self.n_items)

        bounds = [(0, 1)] * n_bids

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return result.x, -result.fun
        return np.zeros(n_bids), 0.0

    def solve_winner_determination(self) -> AuctionAllocation:
        """
        Solve winner determination using branch and bound with LP relaxation.
        """
        n_bids = len(self.bids)
        if n_bids == 0:
            return AuctionAllocation({}, {}, 0.0, 0.0)

        best_solution = np.zeros(n_bids)
        best_value = 0.0

        # Branch and bound
        stack = [(list(range(n_bids)), [])]  # (remaining_bid_indices, fixed_decisions)

        while stack:
            remaining, fixed = stack.pop()

            # Build current solution
            current = np.zeros(n_bids)
            for idx, val in fixed:
                current[idx] = val

            # Check feasibility of fixed decisions
            item_usage = defaultdict(float)
            for idx, val in fixed:
                if val == 1:
                    for item in self.bids[idx].bundle:
                        item_usage[item] += 1
                        if item_usage[item] > 1.0 + 1e-10:
                            continue

            feasible = True
            for item in item_usage:
                if item_usage[item] > 1.0 + 1e-10:
                    feasible = False
                    break

            if not feasible:
                continue

            if not remaining:
                # All decided
                value = sum(self.bids[idx].value for idx, val in fixed if val == 1)
                if value > best_value:
                    best_value = value
                    best_solution = current.copy()
                continue

            # LP relaxation bound
            # Create sub-problem with remaining bids
            sub_bids = [self.bids[i] for i in remaining]
            # Remove items already allocated
            allocated_items = set()
            for idx, val in fixed:
                if val == 1:
                    allocated_items.update(self.bids[idx].bundle)

            available_sub_bids = [b for b in sub_bids
                                  if not any(item in allocated_items for item in b.bundle)]

            fixed_value = sum(self.bids[idx].value for idx, val in fixed if val == 1)

            if available_sub_bids:
                lp_sol, lp_bound = self._solve_lp_relaxation(available_sub_bids)
                upper_bound = fixed_value + lp_bound
            else:
                upper_bound = fixed_value

            if upper_bound <= best_value:
                continue  # Prune

            # Branch on first remaining bid
            branch_idx = remaining[0]
            rest = remaining[1:]

            # Check if we can include this bid
            can_include = True
            for item in self.bids[branch_idx].bundle:
                if item in allocated_items:
                    can_include = False
                    break

            if can_include:
                stack.append((rest, fixed + [(branch_idx, 1)]))
            stack.append((rest, fixed + [(branch_idx, 0)]))

        # Build allocation from solution
        assignment = defaultdict(list)
        payments = defaultdict(float)
        for i, val in enumerate(best_solution):
            if val > 0.5:
                bid = self.bids[i]
                assignment[bid.bidder_id].append(bid.bundle)
                payments[bid.bidder_id] += bid.value  # First-price

        return AuctionAllocation(
            assignment=dict(assignment),
            payments=dict(payments),
            social_welfare=best_value,
            revenue=sum(payments.values())
        )

    def compute_vcg_payments(self) -> Dict[int, float]:
        """Compute VCG payments for the combinatorial auction."""
        main_alloc = self.solve_winner_determination()
        bidder_ids = set(b.bidder_id for b in self.bids)
        payments = {}

        for bidder_i in bidder_ids:
            # Welfare of others in main allocation
            others_welfare = sum(
                sum(next((b.value for b in self.bids
                          if b.bidder_id == bid_id and b.bundle == bundle), 0.0)
                    for bundle in bundles)
                for bid_id, bundles in main_alloc.assignment.items()
                if bid_id != bidder_i
            )

            # Optimal welfare without bidder_i
            bids_without = [b for b in self.bids if b.bidder_id != bidder_i]
            temp_auction = CombinatorialAuction(self.items)
            temp_auction.bids = bids_without
            alloc_without = temp_auction.solve_winner_determination()

            payments[bidder_i] = alloc_without.social_welfare - others_welfare

        return payments


@dataclass
class Order:
    """Order in a double auction."""
    order_id: int
    trader_id: int
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    timestamp: float

    def __lt__(self, other):
        if self.side == 'buy':
            return self.price > other.price  # Higher buy prices have priority
        return self.price < other.price  # Lower sell prices have priority


class DoubleAuction:
    """Continuous double auction with order book and matching engine."""

    def __init__(self):
        self.buy_orders: List[Order] = []  # Max-heap by price
        self.sell_orders: List[Order] = []  # Min-heap by price
        self.trades: List[Dict] = []
        self.next_order_id = 0
        self.time = 0.0

    def submit_order(self, trader_id: int, side: str, price: float,
                     quantity: float) -> List[Dict]:
        """Submit an order and return any resulting trades."""
        self.time += 1.0
        order = Order(self.next_order_id, trader_id, side, price, quantity, self.time)
        self.next_order_id += 1

        new_trades = []
        remaining_qty = quantity

        if side == 'buy':
            # Match against sell orders
            while remaining_qty > 0 and self.sell_orders:
                best_sell = self.sell_orders[0]
                if best_sell.price <= price:
                    trade_qty = min(remaining_qty, best_sell.quantity)
                    trade_price = best_sell.price  # Price-time priority
                    trade = {
                        'buyer': trader_id,
                        'seller': best_sell.trader_id,
                        'price': trade_price,
                        'quantity': trade_qty,
                        'timestamp': self.time
                    }
                    new_trades.append(trade)
                    self.trades.append(trade)
                    remaining_qty -= trade_qty
                    best_sell.quantity -= trade_qty
                    if best_sell.quantity <= 1e-10:
                        heapq.heappop(self.sell_orders)
                else:
                    break
            if remaining_qty > 0:
                order.quantity = remaining_qty
                # Store as negative price for max-heap simulation
                heapq.heappush(self.buy_orders, Order(
                    order.order_id, trader_id, 'buy', -price, remaining_qty, self.time))

        else:  # sell
            while remaining_qty > 0 and self.buy_orders:
                best_buy = self.buy_orders[0]
                actual_buy_price = -best_buy.price
                if actual_buy_price >= price:
                    trade_qty = min(remaining_qty, best_buy.quantity)
                    trade_price = actual_buy_price
                    trade = {
                        'buyer': best_buy.trader_id,
                        'seller': trader_id,
                        'price': trade_price,
                        'quantity': trade_qty,
                        'timestamp': self.time
                    }
                    new_trades.append(trade)
                    self.trades.append(trade)
                    remaining_qty -= trade_qty
                    best_buy.quantity -= trade_qty
                    if best_buy.quantity <= 1e-10:
                        heapq.heappop(self.buy_orders)
                else:
                    break
            if remaining_qty > 0:
                heapq.heappush(self.sell_orders, Order(
                    order.order_id, trader_id, 'sell', price, remaining_qty, self.time))

        return new_trades

    def get_spread(self) -> Optional[Tuple[float, float]]:
        """Get current bid-ask spread."""
        if not self.buy_orders or not self.sell_orders:
            return None
        best_bid = -self.buy_orders[0].price
        best_ask = self.sell_orders[0].price
        return (best_bid, best_ask)

    def get_midprice(self) -> Optional[float]:
        spread = self.get_spread()
        if spread:
            return (spread[0] + spread[1]) / 2
        return None


class SequentialAuction:
    """Multi-round ascending auction with activity rules."""

    def __init__(self, items: List[int], n_rounds: int = 20):
        self.items = items
        self.n_rounds = n_rounds
        self.current_prices = {item: 0.0 for item in items}
        self.price_increment = 1.0
        self.bidder_demands: Dict[int, Set[int]] = {}
        self.history: List[Dict] = []

    def set_initial_prices(self, prices: Dict[int, float]):
        self.current_prices.update(prices)

    def run_round(self, demands: Dict[int, Set[int]]) -> Dict[int, float]:
        """
        Run one round of ascending auction.
        demands: {bidder_id: set of demanded items at current prices}
        Returns updated prices.
        """
        self.bidder_demands = demands

        # Find over-demanded items (more than one bidder demands)
        item_demand = defaultdict(set)
        for bidder_id, items in demands.items():
            for item in items:
                item_demand[item].add(bidder_id)

        # Raise prices on over-demanded items
        for item, bidders in item_demand.items():
            if len(bidders) > 1:
                self.current_prices[item] += self.price_increment

        self.history.append({
            'prices': dict(self.current_prices),
            'demands': {k: list(v) for k, v in demands.items()}
        })

        return dict(self.current_prices)

    def run_auction(self, value_functions: Dict[int, Dict[int, float]]) -> AuctionAllocation:
        """
        Run full ascending auction with straightforward bidding.
        value_functions: {bidder_id: {item_id: value}}
        """
        bidder_ids = list(value_functions.keys())

        for round_num in range(self.n_rounds):
            demands = {}
            for bidder_id in bidder_ids:
                demanded = set()
                for item in self.items:
                    val = value_functions[bidder_id].get(item, 0.0)
                    if val >= self.current_prices[item]:
                        demanded.add(item)
                demands[bidder_id] = demanded

            old_prices = dict(self.current_prices)
            self.run_round(demands)

            # Check convergence
            if self.current_prices == old_prices:
                break

        # Final allocation: assign each item to highest-value bidder willing to pay
        assignment = defaultdict(list)
        payments = defaultdict(float)
        total_welfare = 0.0

        for item in self.items:
            best_bidder = None
            best_surplus = -np.inf
            for bidder_id in bidder_ids:
                val = value_functions[bidder_id].get(item, 0.0)
                surplus = val - self.current_prices[item]
                if surplus >= 0 and surplus > best_surplus:
                    best_surplus = surplus
                    best_bidder = bidder_id

            if best_bidder is not None:
                assignment[best_bidder].append((item,))
                payments[best_bidder] += self.current_prices[item]
                total_welfare += value_functions[best_bidder].get(item, 0.0)

        return AuctionAllocation(
            assignment=dict(assignment),
            payments=dict(payments),
            social_welfare=total_welfare,
            revenue=sum(payments.values())
        )


class SpectrumAuction:
    """
    Simplified FCC-style combinatorial clock auction.
    Two phases: clock phase (ascending) and supplementary round.
    """

    def __init__(self, items: List[int], reserve_prices: Optional[Dict[int, float]] = None):
        self.items = items
        self.reserve_prices = reserve_prices or {i: 0.0 for i in items}
        self.clock_prices = dict(self.reserve_prices)
        self.clock_increment = 0.1
        self.clock_history: List[Dict] = []

    def run_clock_phase(self, value_functions: Dict[int, Dict[Tuple[int, ...], float]],
                        n_rounds: int = 30) -> Dict[int, float]:
        """
        Run clock phase: prices rise on over-demanded items.
        Returns final clock prices.
        """
        bidder_ids = list(value_functions.keys())

        for round_num in range(n_rounds):
            demands = {}
            for bidder_id in bidder_ids:
                # Find most profitable bundle at current prices
                best_bundle = ()
                best_profit = 0.0
                for r in range(1, len(self.items) + 1):
                    for bundle in combinations(self.items, r):
                        val = value_functions[bidder_id].get(bundle, 0.0)
                        cost = sum(self.clock_prices[item] for item in bundle)
                        profit = val - cost
                        if profit > best_profit:
                            best_profit = profit
                            best_bundle = bundle
                demands[bidder_id] = set(best_bundle)

            # Check aggregate demand
            item_demand_count = defaultdict(int)
            for bidder_id, items in demands.items():
                for item in items:
                    item_demand_count[item] += 1

            # Raise prices on over-demanded items
            any_excess = False
            for item, count in item_demand_count.items():
                if count > 1:
                    self.clock_prices[item] *= (1 + self.clock_increment)
                    any_excess = True

            self.clock_history.append({
                'round': round_num,
                'prices': dict(self.clock_prices),
                'demands': {k: list(v) for k, v in demands.items()}
            })

            if not any_excess:
                break

        return dict(self.clock_prices)

    def run_supplementary_round(self, supplementary_bids: Dict[int, Dict[Tuple[int, ...], float]]) -> AuctionAllocation:
        """
        Run supplementary round: bidders submit package bids,
        winner determination solves combinatorial allocation.
        """
        ca = CombinatorialAuction(self.items)
        for bidder_id, bids in supplementary_bids.items():
            ca.add_bidder(bidder_id, bids)
        return ca.solve_winner_determination()


def find_walrasian_equilibrium(n_goods: int, n_agents: int,
                                utility_functions: List[Callable],
                                endowments: np.ndarray,
                                tol: float = 1e-6,
                                max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find Walrasian (competitive) equilibrium for divisible goods.
    Uses tatonnement process.

    utility_functions[i](x) -> utility for agent i consuming bundle x
    endowments: (n_agents, n_goods) initial endowments
    Returns: (prices, allocations)
    """
    from typing import Callable
    prices = np.ones(n_goods)
    prices[0] = 1.0  # Numeraire

    step_size = 0.01

    for iteration in range(max_iter):
        # Compute demands at current prices
        demands = np.zeros((n_agents, n_goods))
        for i in range(n_agents):
            wealth = np.dot(prices, endowments[i])
            # Maximize utility subject to budget constraint
            # Gradient ascent on utility
            x = endowments[i].copy()
            for _ in range(100):
                # Compute numerical gradient of utility
                grad = np.zeros(n_goods)
                eps = 1e-6
                u0 = utility_functions[i](x)
                for g in range(n_goods):
                    x_plus = x.copy()
                    x_plus[g] += eps
                    grad[g] = (utility_functions[i](x_plus) - u0) / eps

                # Project gradient onto budget hyperplane
                # Move towards higher utility, respecting budget
                # Marshallian demand: MU_g / p_g should be equal for all goods
                mu_per_price = grad / (prices + 1e-10)
                avg_mu = np.mean(mu_per_price)

                # Adjust: increase goods with high MU/price, decrease others
                adjustment = (mu_per_price - avg_mu) * step_size
                x += adjustment
                x = np.maximum(x, 0)

                # Enforce budget constraint
                cost = np.dot(prices, x)
                if cost > 0:
                    x *= wealth / cost

            demands[i] = x

        # Excess demand
        total_demand = demands.sum(axis=0)
        total_supply = endowments.sum(axis=0)
        excess = total_demand - total_supply

        # Check convergence
        if np.max(np.abs(excess)) < tol:
            return prices, demands

        # Tatonnement: adjust prices
        prices[1:] += step_size * excess[1:]  # Don't adjust numeraire
        prices = np.maximum(prices, 1e-10)

    return prices, demands


def compute_ceei(n_agents: int, n_goods: int,
                  utility_functions: List,
                  budget: float = 1.0,
                  n_iter: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Competitive Equilibrium from Equal Incomes (CEEI).
    Each agent has equal budget; find market-clearing prices.

    Returns: (prices, allocations)
    """
    prices = np.ones(n_goods)
    step_size = 0.005

    for iteration in range(n_iter):
        allocations = np.zeros((n_agents, n_goods))
        for i in range(n_agents):
            # Find demand: maximize utility subject to budget
            x = np.ones(n_goods) * budget / (n_goods * np.mean(prices) + 1e-10)
            for _ in range(50):
                grad = np.zeros(n_goods)
                eps = 1e-6
                u0 = utility_functions[i](x)
                for g in range(n_goods):
                    x_p = x.copy()
                    x_p[g] += eps
                    grad[g] = (utility_functions[i](x_p) - u0) / eps

                mu_per_price = grad / (prices + 1e-10)
                avg = np.mean(mu_per_price)
                adjustment = (mu_per_price - avg) * step_size
                x += adjustment
                x = np.maximum(x, 0)
                cost = np.dot(prices, x)
                if cost > 1e-10:
                    x *= budget / cost

            allocations[i] = x

        # Excess demand
        total_demand = allocations.sum(axis=0)
        supply = np.ones(n_goods)  # Unit supply of each good
        excess = total_demand - supply

        if np.max(np.abs(excess)) < 1e-6:
            return prices, allocations

        prices += step_size * excess
        prices = np.maximum(prices, 1e-10)

    return prices, allocations


class AuctionAnalyzer:
    """Compute auction performance metrics."""

    @staticmethod
    def compute_efficiency(allocation: AuctionAllocation,
                           optimal_welfare: float) -> float:
        """Allocative efficiency = achieved welfare / optimal welfare."""
        if optimal_welfare < 1e-10:
            return 1.0
        return allocation.social_welfare / optimal_welfare

    @staticmethod
    def compute_revenue_ratio(allocation: AuctionAllocation,
                               optimal_revenue: float) -> float:
        """Revenue as fraction of theoretical optimal."""
        if optimal_revenue < 1e-10:
            return 1.0
        return allocation.revenue / optimal_revenue

    @staticmethod
    def compute_envy(allocation: AuctionAllocation,
                     valuations: Dict[int, Dict[Tuple[int, ...], float]]) -> Dict[int, float]:
        """
        Compute envy: for each agent, max utility they'd get from another's bundle
        minus their own utility.
        """
        envy = {}
        for agent_i in valuations:
            own_bundles = allocation.assignment.get(agent_i, [])
            own_value = sum(valuations[agent_i].get(b, 0.0) for b in own_bundles)
            own_payment = allocation.payments.get(agent_i, 0.0)
            own_utility = own_value - own_payment

            max_envy = 0.0
            for agent_j in valuations:
                if agent_j == agent_i:
                    continue
                j_bundles = allocation.assignment.get(agent_j, [])
                j_value = sum(valuations[agent_i].get(b, 0.0) for b in j_bundles)
                j_payment = allocation.payments.get(agent_j, 0.0)
                j_utility = j_value - j_payment

                envy_ij = j_utility - own_utility
                max_envy = max(max_envy, envy_ij)

            envy[agent_i] = max_envy
        return envy

    @staticmethod
    def compute_regret(allocation: AuctionAllocation,
                       valuations: Dict[int, Dict[Tuple[int, ...], float]]) -> Dict[int, float]:
        """
        Compute regret: difference between best possible utility and achieved utility.
        """
        regret = {}
        for agent_i in valuations:
            own_bundles = allocation.assignment.get(agent_i, [])
            own_value = sum(valuations[agent_i].get(b, 0.0) for b in own_bundles)
            own_payment = allocation.payments.get(agent_i, 0.0)
            own_utility = own_value - own_payment

            # Best possible utility: max value - 0 payment
            max_value = max(valuations[agent_i].values()) if valuations[agent_i] else 0.0
            regret[agent_i] = max(0, max_value - own_utility)

        return regret

    @staticmethod
    def compute_price_of_anarchy(equilibrium_welfare: float,
                                  optimal_welfare: float) -> float:
        """Price of anarchy = optimal / equilibrium welfare."""
        if equilibrium_welfare < 1e-10:
            return float('inf')
        return optimal_welfare / equilibrium_welfare

    @staticmethod
    def compute_gini_coefficient(utilities: List[float]) -> float:
        """Compute Gini coefficient of utility distribution."""
        n = len(utilities)
        if n == 0:
            return 0.0
        sorted_u = sorted(utilities)
        total = sum(sorted_u)
        if total < 1e-10:
            return 0.0
        cumulative = 0.0
        gini_sum = 0.0
        for i, u in enumerate(sorted_u):
            cumulative += u
            gini_sum += cumulative
        gini = (2.0 * gini_sum) / (n * total) - (n + 1.0) / n
        return 1.0 - gini  # Normalize so 0 = perfect equality


def generate_random_combinatorial_valuations(n_bidders: int, n_items: int,
                                              synergy: float = 0.2,
                                              rng=None) -> Dict[int, Dict[Tuple[int, ...], float]]:
    """Generate random combinatorial valuations with synergies."""
    if rng is None:
        rng = np.random.default_rng()

    valuations = {}
    for bidder in range(n_bidders):
        val = {}
        item_values = rng.uniform(1, 10, size=n_items)
        for r in range(1, min(n_items + 1, 5)):  # Limit bundle size for tractability
            for bundle in combinations(range(n_items), r):
                base_value = sum(item_values[i] for i in bundle)
                # Synergy bonus
                synergy_bonus = synergy * base_value * (r - 1) * rng.uniform(0.5, 1.5)
                val[bundle] = base_value + synergy_bonus
        val[()] = 0.0
        valuations[bidder] = val

    return valuations
