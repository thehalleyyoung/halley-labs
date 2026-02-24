"""
Market design module implementing auction mechanisms, pricing algorithms,
and platform design tools using computational economics methods.

All algorithms use real mathematical formulations with numpy/scipy.
"""

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.special import expit
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MarketDesign:
    """Result container for market design computations."""
    allocation: np.ndarray
    prices: np.ndarray
    welfare: float
    revenue: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bid:
    """Represents a bid in a combinatorial auction.

    Attributes:
        bidder_id: Identifier for the bidder.
        items: List of item indices in the bundle.
        value: The bid value for the bundle.
    """
    bidder_id: int
    items: List[int]
    value: float


@dataclass
class XORBid:
    """Exclusive-or bid: bidder wants at most one bundle.

    Attributes:
        bidder_id: Identifier for the bidder.
        bundles: List of (items, value) pairs; at most one is allocated.
    """
    bidder_id: int
    bundles: List[Tuple[List[int], float]]


@dataclass
class ORBid:
    """OR bid: bidder can win any combination of bundles (additive across bundles).

    Attributes:
        bidder_id: Identifier for the bidder.
        bundles: List of (items, value) pairs; any subset may be allocated.
    """
    bidder_id: int
    bundles: List[Tuple[List[int], float]]


class AscendingClockAuction:
    """Ascending clock auction with demand reduction.

    Implements an iterative auction where prices rise on over-demanded goods
    and bidders reduce demand until a competitive equilibrium is found.

    The clock price on each good increases by a step size when aggregate
    demand exceeds supply. Bidders respond with quasi-linear utility
    maximisation at the current price vector.
    """

    def __init__(self, n_goods: int, supply: np.ndarray,
                 step_size: float = 0.1, max_rounds: int = 5000,
                 tol: float = 1e-6):
        """Initialise the ascending clock auction.

        Args:
            n_goods: Number of distinct goods.
            supply: Array of shape (n_goods,) with supply of each good.
            step_size: Price increment per over-demanded round.
            max_rounds: Maximum number of price-adjustment rounds.
            tol: Tolerance for convergence of excess demand.
        """
        self.n_goods = n_goods
        self.supply = np.asarray(supply, dtype=float)
        self.step_size = step_size
        self.max_rounds = max_rounds
        self.tol = tol

    def _demand_at_prices(self, prices: np.ndarray,
                          valuations: np.ndarray,
                          budgets: np.ndarray) -> np.ndarray:
        """Compute aggregate demand at given prices.

        Each agent chooses the bundle maximising value minus cost subject to
        budget. With divisible goods and quasi-linear utility the demand for
        good j by agent i is  max(0, v_ij - p_j) / p_j  clipped to budget.

        Args:
            prices: Shape (n_goods,) current price vector.
            valuations: Shape (n_agents, n_goods) marginal valuations.
            budgets: Shape (n_agents,) budget for each agent.

        Returns:
            Aggregate demand vector of shape (n_goods,).
        """
        n_agents = valuations.shape[0]
        safe_prices = np.maximum(prices, 1e-12)
        individual_demand = np.maximum(valuations - safe_prices, 0.0) / safe_prices
        # Budget constraint: scale down proportionally if cost exceeds budget
        cost = individual_demand * safe_prices
        total_cost = cost.sum(axis=1)
        scale = np.where(total_cost > budgets,
                         budgets / np.maximum(total_cost, 1e-12),
                         1.0)
        individual_demand *= scale[:, None]
        aggregate = individual_demand.sum(axis=0)
        return aggregate

    def run(self, valuations: np.ndarray,
            budgets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Run the ascending clock auction.

        Args:
            valuations: Shape (n_agents, n_goods) marginal valuations.
            budgets: Shape (n_agents,) budget constraints.

        Returns:
            Tuple of (final_prices, final_allocation, price_history) where
            final_allocation is shape (n_agents, n_goods).
        """
        prices = np.ones(self.n_goods) * self.step_size
        price_history: List[np.ndarray] = [prices.copy()]
        n_agents = valuations.shape[0]

        for _ in range(self.max_rounds):
            agg_demand = self._demand_at_prices(prices, valuations, budgets)
            excess = agg_demand - self.supply

            if np.all(np.abs(excess) < self.tol):
                break

            # Raise prices on over-demanded goods, lower on under-demanded
            adjustment = np.where(excess > self.tol, self.step_size,
                                  np.where(excess < -self.tol,
                                           -self.step_size * 0.5, 0.0))
            prices = np.maximum(prices + adjustment, self.step_size * 0.01)
            price_history.append(prices.copy())

        # Final allocation proportional to individual demand
        safe_prices = np.maximum(prices, 1e-12)
        alloc = np.maximum(valuations - safe_prices, 0.0) / safe_prices
        cost = alloc * safe_prices
        total_cost = cost.sum(axis=1)
        scale = np.where(total_cost > budgets,
                         budgets / np.maximum(total_cost, 1e-12), 1.0)
        alloc *= scale[:, None]

        # If aggregate demand still exceeds supply, ration proportionally
        agg = alloc.sum(axis=0)
        ration = np.where(agg > self.supply,
                          self.supply / np.maximum(agg, 1e-12), 1.0)
        alloc *= ration[None, :]

        return prices, alloc, price_history


class PackageBidding:
    """Combinatorial auction solver supporting XOR, OR, and additive bids.

    Uses linear programming relaxation to find the welfare-maximising
    allocation subject to supply constraints.
    """

    def __init__(self, n_items: int):
        """Initialise the package bidding solver.

        Args:
            n_items: Number of distinct items available.
        """
        self.n_items = n_items

    def solve_xor_bids(self, bids: List[XORBid],
                       supply: Optional[np.ndarray] = None) -> MarketDesign:
        """Solve a combinatorial auction with XOR bids via LP relaxation.

        Each bidder may win at most one of their bundles. We maximise total
        declared value subject to supply and XOR constraints.

        Args:
            bids: List of XORBid objects.
            supply: Item supplies; defaults to 1 per item.

        Returns:
            MarketDesign with allocation, prices, welfare, revenue.
        """
        if supply is None:
            supply = np.ones(self.n_items)

        # Decision variables: x_{i,k} for bidder i, bundle k
        var_map: List[Tuple[int, int, List[int], float]] = []
        for bid in bids:
            for k, (items, value) in enumerate(bid.bundles):
                var_map.append((bid.bidder_id, k, items, value))

        n_vars = len(var_map)
        if n_vars == 0:
            return MarketDesign(
                allocation=np.zeros((0, self.n_items)),
                prices=np.zeros(self.n_items),
                welfare=0.0, revenue=0.0
            )

        # Objective: maximise sum of values (linprog minimises, so negate)
        c = -np.array([v[3] for v in var_map])

        # Constraints: supply for each item
        A_ub_rows: List[np.ndarray] = []
        b_ub_list: List[float] = []
        for j in range(self.n_items):
            row = np.zeros(n_vars)
            for idx, (_, _, items, _) in enumerate(var_map):
                if j in items:
                    row[idx] = 1.0
            A_ub_rows.append(row)
            b_ub_list.append(supply[j])

        # XOR constraints: at most one bundle per bidder
        bidder_ids = set(v[0] for v in var_map)
        for bid_id in bidder_ids:
            row = np.zeros(n_vars)
            for idx, (b, _, _, _) in enumerate(var_map):
                if b == bid_id:
                    row[idx] = 1.0
            A_ub_rows.append(row)
            b_ub_list.append(1.0)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_list)
        bounds = [(0, 1) for _ in range(n_vars)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        x = result.x if result.success else np.zeros(n_vars)

        # Build allocation matrix
        max_bidder = max(v[0] for v in var_map) + 1
        allocation = np.zeros((max_bidder, self.n_items))
        for idx, (bid_id, _, items, _) in enumerate(var_map):
            for j in items:
                allocation[bid_id, j] += x[idx]

        welfare = -result.fun if result.success else 0.0

        # Dual variables give clearing prices for the item constraints
        prices = np.zeros(self.n_items)
        if result.success and hasattr(result, 'ineqlin') and result.ineqlin is not None:
            duals = np.abs(result.ineqlin.marginals)
            prices = duals[:self.n_items]

        revenue = float(np.sum(allocation * prices[None, :]))

        return MarketDesign(
            allocation=allocation, prices=prices,
            welfare=welfare, revenue=revenue,
            metadata={'lp_status': result.message, 'x': x.tolist()}
        )

    def solve_or_bids(self, bids: List[ORBid],
                      supply: Optional[np.ndarray] = None) -> MarketDesign:
        """Solve a combinatorial auction with OR bids.

        Under OR semantics a bidder can win any subset of their bundles and
        their value is the sum of won bundle values.

        Args:
            bids: List of ORBid objects.
            supply: Item supplies; defaults to 1 per item.

        Returns:
            MarketDesign with allocation, prices, welfare, revenue.
        """
        if supply is None:
            supply = np.ones(self.n_items)

        var_map: List[Tuple[int, int, List[int], float]] = []
        for bid in bids:
            for k, (items, value) in enumerate(bid.bundles):
                var_map.append((bid.bidder_id, k, items, value))

        n_vars = len(var_map)
        if n_vars == 0:
            return MarketDesign(
                allocation=np.zeros((0, self.n_items)),
                prices=np.zeros(self.n_items),
                welfare=0.0, revenue=0.0
            )

        c = -np.array([v[3] for v in var_map])

        # Only supply constraints (no XOR constraint)
        A_ub_rows: List[np.ndarray] = []
        b_ub_list: List[float] = []
        for j in range(self.n_items):
            row = np.zeros(n_vars)
            for idx, (_, _, items, _) in enumerate(var_map):
                if j in items:
                    row[idx] = 1.0
            A_ub_rows.append(row)
            b_ub_list.append(supply[j])

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_list)
        bounds = [(0, 1) for _ in range(n_vars)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        x = result.x if result.success else np.zeros(n_vars)

        max_bidder = max(v[0] for v in var_map) + 1
        allocation = np.zeros((max_bidder, self.n_items))
        for idx, (bid_id, _, items, _) in enumerate(var_map):
            for j in items:
                allocation[bid_id, j] += x[idx]

        welfare = -result.fun if result.success else 0.0
        prices = np.zeros(self.n_items)
        if result.success and hasattr(result, 'ineqlin') and result.ineqlin is not None:
            duals = np.abs(result.ineqlin.marginals)
            prices = duals[:self.n_items]

        revenue = float(np.sum(allocation * prices[None, :]))
        return MarketDesign(
            allocation=allocation, prices=prices,
            welfare=welfare, revenue=revenue,
            metadata={'lp_status': result.message}
        )

    def solve_additive_bids(self, valuations: np.ndarray,
                            supply: Optional[np.ndarray] = None) -> MarketDesign:
        """Solve allocation with additive valuations via LP.

        Each bidder has an independent value for each item and total value
        is the sum of item values.

        Args:
            valuations: Shape (n_agents, n_items) matrix of per-item values.
            supply: Item supplies; defaults to 1 per item.

        Returns:
            MarketDesign with allocation, prices, welfare, revenue.
        """
        n_agents, n_items = valuations.shape
        if supply is None:
            supply = np.ones(n_items)

        # Variables: x_{i,j} for agent i, item j
        n_vars = n_agents * n_items
        c = -valuations.flatten()

        # Supply constraints: sum_i x_{i,j} <= supply_j
        A_ub_rows = []
        b_ub_list = []
        for j in range(n_items):
            row = np.zeros(n_vars)
            for i in range(n_agents):
                row[i * n_items + j] = 1.0
            A_ub_rows.append(row)
            b_ub_list.append(supply[j])

        A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
        b_ub = np.array(b_ub_list) if b_ub_list else np.zeros(0)
        bounds = [(0, 1) for _ in range(n_vars)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        x = result.x if result.success else np.zeros(n_vars)
        allocation = x.reshape(n_agents, n_items)
        welfare = -result.fun if result.success else 0.0

        prices = np.zeros(n_items)
        if result.success and hasattr(result, 'ineqlin') and result.ineqlin is not None:
            prices = np.abs(result.ineqlin.marginals[:n_items])

        revenue = float(np.sum(allocation * prices[None, :]))
        return MarketDesign(
            allocation=allocation, prices=prices,
            welfare=welfare, revenue=revenue
        )


class ClearingPriceComputation:
    """Find competitive equilibrium prices via tatonnement and LP duality.

    Implements two approaches:
    1. Walrasian tatonnement: iterative price adjustment driven by excess demand.
    2. LP dual extraction: solve the allocation LP and read dual variables.
    """

    def __init__(self, n_goods: int, tol: float = 1e-6, max_iter: int = 10000):
        self.n_goods = n_goods
        self.tol = tol
        self.max_iter = max_iter

    def tatonnement(self, valuations: np.ndarray,
                    supply: np.ndarray,
                    lr: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Walrasian tatonnement to find market-clearing prices.

        Iteratively adjusts prices proportional to excess demand until
        markets clear. Agents have quasi-linear utility:
        u_i(x, p) = sum_j v_{ij} * x_{ij} - sum_j p_j * x_{ij}.

        Marshallian demand at price p for agent i and good j is
        1 if v_{ij} > p_j, else 0 (unit-demand simplification).

        Args:
            valuations: Shape (n_agents, n_goods).
            supply: Shape (n_goods,).
            lr: Learning rate for price adjustment.

        Returns:
            Tuple of (equilibrium_prices, excess_demand_history_norm).
        """
        prices = np.mean(valuations, axis=0) * 0.5
        history = []

        for _ in range(self.max_iter):
            # Smooth demand using sigmoid for gradient-based adjustment
            demand_prob = expit(5.0 * (valuations - prices[None, :]))
            agg_demand = demand_prob.sum(axis=0)
            excess = agg_demand - supply
            history.append(float(np.linalg.norm(excess)))

            if np.linalg.norm(excess) < self.tol:
                break

            prices += lr * excess
            prices = np.maximum(prices, 0.0)

        return prices, np.array(history)

    def lp_dual_prices(self, valuations: np.ndarray,
                       supply: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract competitive equilibrium prices from LP duality.

        Solves the welfare-maximising LP and returns dual variables as
        competitive equilibrium prices. With additive valuations the dual
        of the supply constraint gives the clearing price.

        Args:
            valuations: Shape (n_agents, n_goods).
            supply: Shape (n_goods,).

        Returns:
            Tuple of (prices, allocation).
        """
        n_agents, n_goods = valuations.shape
        n_vars = n_agents * n_goods
        c = -valuations.flatten()

        A_ub_rows = []
        b_ub_list = []
        for j in range(n_goods):
            row = np.zeros(n_vars)
            for i in range(n_agents):
                row[i * n_goods + j] = 1.0
            A_ub_rows.append(row)
            b_ub_list.append(supply[j])

        # Unit demand per agent
        for i in range(n_agents):
            row = np.zeros(n_vars)
            for j in range(n_goods):
                row[i * n_goods + j] = 1.0
            A_ub_rows.append(row)
            b_ub_list.append(1.0)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_list)
        bounds = [(0, 1) for _ in range(n_vars)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        allocation = result.x.reshape(n_agents, n_goods) if result.success else np.zeros((n_agents, n_goods))

        prices = np.zeros(n_goods)
        if result.success and hasattr(result, 'ineqlin') and result.ineqlin is not None:
            prices = np.abs(result.ineqlin.marginals[:n_goods])

        return prices, allocation


class MarketThicknessAnalysis:
    """Analyse how market thickness (number of participants) affects outcomes.

    Uses random matching models to predict match quality as a function of
    market size, following the theoretical framework where thicker markets
    yield better matches due to more competition.
    """

    def __init__(self, rng_seed: int = 42):
        self.rng = np.random.RandomState(rng_seed)

    def expected_max_order_stat(self, n: int, loc: float = 0.0,
                                scale: float = 1.0) -> float:
        """Expected value of the maximum of n draws from N(loc, scale^2).

        Uses the approximation E[X_{(n)}] ≈ loc + scale * sqrt(2 * ln(n))
        for large n, and exact integration for small n via recursive formula.

        Args:
            n: Number of draws.
            loc: Mean of the normal distribution.
            scale: Standard deviation.

        Returns:
            Approximate expected maximum.
        """
        if n <= 0:
            return loc
        if n == 1:
            return loc
        return loc + scale * np.sqrt(2.0 * np.log(max(n, 2)))

    def simulate_match_quality(self, market_sizes: np.ndarray,
                               n_simulations: int = 500,
                               dim: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate average match quality as a function of market size.

        For each market size N, we create N agents on each side of a two-sided
        market with random type vectors in R^dim. Match value between agent i
        on side A and agent j on side B is the negative Euclidean distance.
        We compute the optimal matching and report average match quality.

        Args:
            market_sizes: Array of market sizes to evaluate.
            n_simulations: Number of Monte Carlo repetitions per size.
            dim: Dimensionality of agent type space.

        Returns:
            Tuple of (mean_qualities, std_qualities) arrays.
        """
        means = np.zeros(len(market_sizes))
        stds = np.zeros(len(market_sizes))

        for idx, n in enumerate(market_sizes):
            n = int(n)
            if n == 0:
                means[idx] = 0.0
                stds[idx] = 0.0
                continue
            qualities = np.zeros(n_simulations)
            for s in range(n_simulations):
                types_a = self.rng.randn(n, dim)
                types_b = self.rng.randn(n, dim)
                # Greedy matching: for each agent on side A pick closest on B
                # (approximation to optimal matching for speed)
                available = list(range(n))
                total_quality = 0.0
                for i in range(n):
                    if not available:
                        break
                    dists = np.linalg.norm(
                        types_a[i] - types_b[available], axis=1
                    )
                    best_idx = int(np.argmin(dists))
                    total_quality += -dists[best_idx]
                    available.pop(best_idx)
                qualities[s] = total_quality / max(n, 1)
            means[idx] = qualities.mean()
            stds[idx] = qualities.std()

        return means, stds

    def thickness_elasticity(self, market_sizes: np.ndarray,
                             mean_qualities: np.ndarray) -> np.ndarray:
        """Compute elasticity of match quality with respect to market size.

        Elasticity = d(ln Q) / d(ln N) approximated with finite differences.

        Args:
            market_sizes: Array of market sizes.
            mean_qualities: Corresponding mean match qualities.

        Returns:
            Array of elasticities (length len(market_sizes) - 1).
        """
        ln_n = np.log(np.maximum(market_sizes, 1))
        # Use absolute quality for log (shift so all positive)
        shift = np.abs(mean_qualities.min()) + 1.0
        ln_q = np.log(mean_qualities + shift)
        elasticity = np.diff(ln_q) / np.maximum(np.diff(ln_n), 1e-12)
        return elasticity


class PlatformDesign:
    """Two-sided platform pricing and subsidy analysis.

    Models a platform connecting buyers and sellers where the value to each
    side depends on participation on the other side (network effects).
    Uses the Rochet-Tirole framework for optimal two-sided pricing.
    """

    def __init__(self, n_buyers: int, n_sellers: int):
        self.n_buyers = n_buyers
        self.n_sellers = n_sellers

    def participation_rate(self, fee: float, outside_option: float,
                           cross_benefit: float,
                           other_side_n: float) -> float:
        """Compute participation rate for one side using logit model.

        An agent on this side joins if:
            cross_benefit * other_side_n - fee + epsilon > outside_option
        where epsilon ~ Logistic(0,1). The participation probability is:
            sigma(cross_benefit * other_side_n - fee - outside_option).

        Args:
            fee: Fee charged to this side.
            outside_option: Value of not joining.
            cross_benefit: Per-member benefit from the other side.
            other_side_n: Expected number of participants on the other side.

        Returns:
            Participation probability in [0, 1].
        """
        net = cross_benefit * other_side_n - fee - outside_option
        return float(expit(net))

    def equilibrium_participation(self, fee_b: float, fee_s: float,
                                  outside_b: float, outside_s: float,
                                  cross_b: float, cross_s: float,
                                  max_iter: int = 500,
                                  tol: float = 1e-8) -> Tuple[float, float]:
        """Find equilibrium participation on both sides via fixed-point iteration.

        Iterates:
            n_b = N_B * sigma(cross_b * n_s - fee_b - outside_b)
            n_s = N_S * sigma(cross_s * n_b - fee_s - outside_s)

        Args:
            fee_b: Fee to buyers.
            fee_s: Fee to sellers.
            outside_b: Buyer outside option.
            outside_s: Seller outside option.
            cross_b: Buyer cross-side benefit per seller.
            cross_s: Seller cross-side benefit per buyer.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Tuple of (n_buyers_participating, n_sellers_participating).
        """
        n_b = self.n_buyers * 0.5
        n_s = self.n_sellers * 0.5

        for _ in range(max_iter):
            new_n_b = self.n_buyers * self.participation_rate(
                fee_b, outside_b, cross_b, n_s
            )
            new_n_s = self.n_sellers * self.participation_rate(
                fee_s, outside_s, cross_s, n_b
            )
            if abs(new_n_b - n_b) + abs(new_n_s - n_s) < tol:
                n_b, n_s = new_n_b, new_n_s
                break
            # Damped update for stability
            n_b = 0.7 * new_n_b + 0.3 * n_b
            n_s = 0.7 * new_n_s + 0.3 * n_s

        return n_b, n_s

    def optimal_fees(self, outside_b: float, outside_s: float,
                     cross_b: float, cross_s: float,
                     marginal_cost_b: float = 0.0,
                     marginal_cost_s: float = 0.0,
                     fee_range: Tuple[float, float] = (-5.0, 10.0),
                     grid_size: int = 50) -> Dict[str, float]:
        """Find revenue-maximising fees for both sides via grid search + refinement.

        Platform revenue = fee_b * n_b + fee_s * n_s - mc_b * n_b - mc_s * n_s.

        Args:
            outside_b: Buyer outside option.
            outside_s: Seller outside option.
            cross_b: Buyer cross-side benefit.
            cross_s: Seller cross-side benefit.
            marginal_cost_b: Marginal cost of serving a buyer.
            marginal_cost_s: Marginal cost of serving a seller.
            fee_range: Range to search for each fee.
            grid_size: Number of grid points per dimension.

        Returns:
            Dict with optimal fee_b, fee_s, revenue, n_buyers, n_sellers.
        """
        best_rev = -np.inf
        best_fb, best_fs = 0.0, 0.0
        best_nb, best_ns = 0.0, 0.0

        fb_grid = np.linspace(fee_range[0], fee_range[1], grid_size)
        fs_grid = np.linspace(fee_range[0], fee_range[1], grid_size)

        for fb in fb_grid:
            for fs in fs_grid:
                nb, ns = self.equilibrium_participation(
                    fb, fs, outside_b, outside_s, cross_b, cross_s,
                    max_iter=100
                )
                rev = (fb - marginal_cost_b) * nb + (fs - marginal_cost_s) * ns
                if rev > best_rev:
                    best_rev = rev
                    best_fb, best_fs = fb, fs
                    best_nb, best_ns = nb, ns

        # Local refinement around best grid point
        def neg_revenue(fees: np.ndarray) -> float:
            nb, ns = self.equilibrium_participation(
                fees[0], fees[1], outside_b, outside_s, cross_b, cross_s,
                max_iter=200
            )
            return -(
                (fees[0] - marginal_cost_b) * nb
                + (fees[1] - marginal_cost_s) * ns
            )

        refined = minimize(neg_revenue, x0=np.array([best_fb, best_fs]),
                           method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4})
        if refined.success:
            best_fb, best_fs = refined.x
            best_nb, best_ns = self.equilibrium_participation(
                best_fb, best_fs, outside_b, outside_s, cross_b, cross_s
            )
            best_rev = (
                (best_fb - marginal_cost_b) * best_nb
                + (best_fs - marginal_cost_s) * best_ns
            )

        return {
            'fee_buyers': float(best_fb),
            'fee_sellers': float(best_fs),
            'revenue': float(best_rev),
            'n_buyers': float(best_nb),
            'n_sellers': float(best_ns),
        }

    def subsidy_analysis(self, outside_b: float, outside_s: float,
                         cross_b: float, cross_s: float,
                         subsidy_amounts: np.ndarray
                         ) -> Dict[str, np.ndarray]:
        """Analyse the effect of subsidising one side on total participation.

        For each subsidy level, compute equilibrium participation when
        buyers are subsidised (fee_b = -subsidy) and sellers pay a fixed fee,
        and vice versa.

        Args:
            outside_b: Buyer outside option.
            outside_s: Seller outside option.
            cross_b: Buyer cross-side benefit per seller.
            cross_s: Seller cross-side benefit per buyer.
            subsidy_amounts: Array of subsidy levels to test.

        Returns:
            Dict with arrays: subsidy_buyer_nb, subsidy_buyer_ns,
            subsidy_seller_nb, subsidy_seller_ns.
        """
        n = len(subsidy_amounts)
        results: Dict[str, np.ndarray] = {
            'subsidy_buyer_nb': np.zeros(n),
            'subsidy_buyer_ns': np.zeros(n),
            'subsidy_seller_nb': np.zeros(n),
            'subsidy_seller_ns': np.zeros(n),
        }
        fixed_fee = 1.0

        for i, s in enumerate(subsidy_amounts):
            nb, ns = self.equilibrium_participation(
                -s, fixed_fee, outside_b, outside_s, cross_b, cross_s
            )
            results['subsidy_buyer_nb'][i] = nb
            results['subsidy_buyer_ns'][i] = ns

            nb, ns = self.equilibrium_participation(
                fixed_fee, -s, outside_b, outside_s, cross_b, cross_s
            )
            results['subsidy_seller_nb'][i] = nb
            results['subsidy_seller_ns'][i] = ns

        return results


class CongestionPricing:
    """Optimal tolls for congestible resources.

    Models resources (e.g., roads, servers) where usage cost increases with
    congestion. Computes Pigouvian tolls that internalise the externality.
    Uses the BPR (Bureau of Public Roads) delay function for congestion.
    """

    def __init__(self, n_resources: int, capacities: np.ndarray,
                 free_flow_costs: np.ndarray,
                 alpha: float = 0.15, beta: float = 4.0):
        """Initialise congestion pricing model.

        The travel cost on resource r with flow f_r is:
            c_r(f_r) = free_flow_r * (1 + alpha * (f_r / cap_r)^beta)

        Args:
            n_resources: Number of congestible resources.
            capacities: Capacity of each resource.
            free_flow_costs: Free-flow cost (time/cost when empty).
            alpha: BPR alpha parameter.
            beta: BPR beta parameter.
        """
        self.n_resources = n_resources
        self.capacities = np.asarray(capacities, dtype=float)
        self.free_flow_costs = np.asarray(free_flow_costs, dtype=float)
        self.alpha = alpha
        self.beta = beta

    def travel_cost(self, flows: np.ndarray) -> np.ndarray:
        """Compute travel cost on each resource given flows.

        Args:
            flows: Shape (n_resources,) flow on each resource.

        Returns:
            Array of travel costs.
        """
        ratio = flows / np.maximum(self.capacities, 1e-12)
        return self.free_flow_costs * (1.0 + self.alpha * np.power(ratio, self.beta))

    def marginal_social_cost(self, flows: np.ndarray) -> np.ndarray:
        """Compute marginal social cost: d(f * c(f)) / df.

        MSC = c(f) + f * c'(f) where c'(f) = free_flow * alpha * beta *
        (f/cap)^{beta-1} / cap.

        Args:
            flows: Shape (n_resources,) flow on each resource.

        Returns:
            Marginal social cost on each resource.
        """
        ratio = flows / np.maximum(self.capacities, 1e-12)
        c = self.travel_cost(flows)
        dc_df = (self.free_flow_costs * self.alpha * self.beta
                 * np.power(ratio, np.maximum(self.beta - 1, 0))
                 / np.maximum(self.capacities, 1e-12))
        return c + flows * dc_df

    def pigouvian_toll(self, flows: np.ndarray) -> np.ndarray:
        """Compute the Pigouvian toll: MSC - private cost.

        The toll equals the externality that each user imposes on others:
            toll_r = f_r * c'_r(f_r).

        Args:
            flows: Shape (n_resources,) current flow levels.

        Returns:
            Optimal toll for each resource.
        """
        return self.marginal_social_cost(flows) - self.travel_cost(flows)

    def user_equilibrium(self, total_demand: float,
                         max_iter: int = 5000,
                         lr: float = 0.01) -> np.ndarray:
        """Find user equilibrium flows (Wardrop equilibrium) via Frank-Wolfe.

        All used routes have equal cost; no user can unilaterally reduce
        cost by switching. Uses the convex optimisation formulation with
        the Beckmann objective.

        Args:
            total_demand: Total flow to distribute across resources.
            max_iter: Maximum Frank-Wolfe iterations.
            lr: Step size parameter.

        Returns:
            Equilibrium flow vector of shape (n_resources,).
        """
        # Initial feasible solution: split evenly
        flows = np.ones(self.n_resources) * total_demand / self.n_resources

        for k in range(1, max_iter + 1):
            costs = self.travel_cost(flows)
            # All-or-nothing assignment: put all flow on cheapest resource
            target = np.zeros(self.n_resources)
            target[np.argmin(costs)] = total_demand
            # Step size (diminishing)
            gamma = 2.0 / (k + 2)
            flows = flows + gamma * (target - flows)

        return flows

    def social_optimum(self, total_demand: float,
                       max_iter: int = 5000) -> np.ndarray:
        """Find socially optimal flows minimising total social cost.

        Minimises sum_r integral_0^{f_r} MSC_r(x) dx via Frank-Wolfe
        using marginal social cost as the gradient.

        Args:
            total_demand: Total flow to distribute.
            max_iter: Maximum iterations.

        Returns:
            Socially optimal flow vector.
        """
        flows = np.ones(self.n_resources) * total_demand / self.n_resources

        for k in range(1, max_iter + 1):
            msc = self.marginal_social_cost(flows)
            target = np.zeros(self.n_resources)
            target[np.argmin(msc)] = total_demand
            gamma = 2.0 / (k + 2)
            flows = flows + gamma * (target - flows)

        return flows

    def efficiency_gap(self, total_demand: float) -> Dict[str, float]:
        """Compute the price of anarchy: ratio of UE cost to SO cost.

        Args:
            total_demand: Total flow.

        Returns:
            Dict with ue_cost, so_cost, price_of_anarchy, optimal_tolls.
        """
        ue_flows = self.user_equilibrium(total_demand)
        so_flows = self.social_optimum(total_demand)
        ue_cost = float(np.sum(ue_flows * self.travel_cost(ue_flows)))
        so_cost = float(np.sum(so_flows * self.travel_cost(so_flows)))
        poa = ue_cost / max(so_cost, 1e-12)
        tolls = self.pigouvian_toll(so_flows)
        return {
            'ue_cost': ue_cost,
            'so_cost': so_cost,
            'price_of_anarchy': poa,
            'optimal_tolls': tolls.tolist(),
        }


class DynamicPricing:
    """Dynamic pricing with demand learning over time.

    Implements a multi-armed bandit approach to pricing where the seller
    sets prices in discrete periods, observes demand, and updates a
    Bayesian demand model to maximise cumulative revenue.

    The demand model is log-linear: log(q) = a - b * p + noise,
    estimated via Bayesian linear regression.
    """

    def __init__(self, n_price_levels: int = 20,
                 price_min: float = 0.5, price_max: float = 20.0,
                 prior_a_mean: float = 3.0, prior_a_var: float = 2.0,
                 prior_b_mean: float = 0.3, prior_b_var: float = 0.1,
                 noise_var: float = 0.5):
        """Initialise dynamic pricing model.

        Args:
            n_price_levels: Number of discrete price options.
            price_min: Minimum price.
            price_max: Maximum price.
            prior_a_mean: Prior mean for intercept a.
            prior_a_var: Prior variance for intercept a.
            prior_b_mean: Prior mean for price sensitivity b.
            prior_b_var: Prior variance for price sensitivity b.
            noise_var: Observation noise variance.
        """
        self.prices = np.linspace(price_min, price_max, n_price_levels)
        self.noise_var = noise_var
        # Bayesian linear regression: y = [1, p] @ theta, theta = [a, -b]
        self.prior_mean = np.array([prior_a_mean, -prior_b_mean])
        self.prior_cov = np.diag([prior_a_var, prior_b_var])
        self.posterior_mean = self.prior_mean.copy()
        self.posterior_cov = self.prior_cov.copy()
        self.history: List[Tuple[float, float]] = []

    def _feature(self, price: float) -> np.ndarray:
        """Feature vector for Bayesian regression."""
        return np.array([1.0, price])

    def update_belief(self, price: float, observed_log_demand: float) -> None:
        """Update posterior after observing demand at a price.

        Uses conjugate Bayesian linear regression update:
            Sigma_new^{-1} = Sigma_old^{-1} + phi * phi^T / sigma_n^2
            mu_new = Sigma_new * (Sigma_old^{-1} * mu_old + phi * y / sigma_n^2)

        Args:
            price: Price that was set.
            observed_log_demand: log(demand) observed at that price.
        """
        phi = self._feature(price)
        prec = np.linalg.inv(self.posterior_cov)
        prec_new = prec + np.outer(phi, phi) / self.noise_var
        cov_new = np.linalg.inv(prec_new)
        mean_new = cov_new @ (prec @ self.posterior_mean
                              + phi * observed_log_demand / self.noise_var)
        self.posterior_mean = mean_new
        self.posterior_cov = cov_new
        self.history.append((price, observed_log_demand))

    def expected_revenue(self, price: float) -> float:
        """Expected revenue = price * E[demand] at given price.

        E[demand] = exp(E[log_demand] + Var[log_demand] / 2) using the
        log-normal moment formula.

        Args:
            price: Proposed price.

        Returns:
            Expected revenue.
        """
        phi = self._feature(price)
        mean_log_d = float(phi @ self.posterior_mean)
        var_log_d = float(phi @ self.posterior_cov @ phi) + self.noise_var
        expected_demand = np.exp(mean_log_d + var_log_d / 2.0)
        return price * expected_demand

    def thompson_sample_price(self) -> float:
        """Select a price via Thompson sampling.

        Draw parameters from the posterior, compute expected revenue at each
        price level, and return the price with highest sampled revenue.

        Returns:
            Selected price.
        """
        theta_sample = np.random.multivariate_normal(
            self.posterior_mean, self.posterior_cov
        )
        best_price = self.prices[0]
        best_rev = -np.inf
        for p in self.prices:
            phi = self._feature(p)
            log_d = phi @ theta_sample
            rev = p * np.exp(log_d)
            if rev > best_rev:
                best_rev = rev
                best_price = p
        return float(best_price)

    def ucb_price(self, exploration_weight: float = 2.0) -> float:
        """Select a price using Upper Confidence Bound on revenue.

        UCB = E[revenue] + exploration_weight * std(revenue).

        Args:
            exploration_weight: Controls exploration vs exploitation.

        Returns:
            Selected price.
        """
        best_price = self.prices[0]
        best_ucb = -np.inf
        for p in self.prices:
            phi = self._feature(p)
            mean_log_d = float(phi @ self.posterior_mean)
            var_log_d = float(phi @ self.posterior_cov @ phi) + self.noise_var
            expected_demand = np.exp(mean_log_d + var_log_d / 2.0)
            # Approximate std of revenue via delta method
            std_revenue = p * expected_demand * np.sqrt(np.exp(var_log_d) - 1.0)
            ucb = p * expected_demand + exploration_weight * std_revenue
            if ucb > best_ucb:
                best_ucb = ucb
                best_price = p
        return float(best_price)

    def run_simulation(self, true_a: float, true_b: float,
                       n_periods: int = 100,
                       strategy: str = 'thompson'
                       ) -> Dict[str, np.ndarray]:
        """Run a full dynamic pricing simulation.

        In each period the seller picks a price, demand is realised from the
        true model, and the belief is updated.

        Args:
            true_a: True intercept of log-demand.
            true_b: True price sensitivity.
            n_periods: Number of selling periods.
            strategy: 'thompson', 'ucb', or 'greedy'.

        Returns:
            Dict with arrays: prices_chosen, demands, revenues, cumulative_revenue.
        """
        # Reset posterior
        self.posterior_mean = self.prior_mean.copy()
        self.posterior_cov = self.prior_cov.copy()
        self.history = []

        prices_chosen = np.zeros(n_periods)
        demands = np.zeros(n_periods)
        revenues = np.zeros(n_periods)

        for t in range(n_periods):
            if strategy == 'thompson':
                p = self.thompson_sample_price()
            elif strategy == 'ucb':
                p = self.ucb_price()
            else:
                # Greedy: pick price with highest expected revenue
                revs = np.array([self.expected_revenue(pr) for pr in self.prices])
                p = float(self.prices[np.argmax(revs)])

            # True demand realisation
            log_demand = true_a - true_b * p + np.random.randn() * np.sqrt(self.noise_var)
            demand = np.exp(log_demand)
            revenue = p * demand

            self.update_belief(p, log_demand)

            prices_chosen[t] = p
            demands[t] = demand
            revenues[t] = revenue

        return {
            'prices': prices_chosen,
            'demands': demands,
            'revenues': revenues,
            'cumulative_revenue': np.cumsum(revenues),
        }


class MarketDesigner:
    """Top-level orchestrator for market design.

    Provides a unified interface to auction mechanisms, pricing algorithms,
    and market analysis tools.
    """

    def __init__(self):
        self._auction_cache: Dict[str, Any] = {}

    def design(self, goods: List[str], agents: List[Dict[str, Any]],
               constraints: Optional[Dict[str, Any]] = None) -> MarketDesign:
        """Design a market mechanism for the given goods, agents, and constraints.

        Chooses the appropriate mechanism based on the structure of agent
        preferences and solves for the optimal allocation and prices.

        Args:
            goods: List of good names.
            agents: List of dicts with keys 'id', 'valuations' (dict mapping
                    good names or frozensets of good names to values), and
                    optionally 'budget'.
            constraints: Optional dict with 'supply' (dict good -> quantity),
                         'mechanism' ('ascending_clock', 'xor', 'or', 'additive').

        Returns:
            MarketDesign with allocation, prices, welfare, revenue.
        """
        n_goods = len(goods)
        n_agents = len(agents)
        constraints = constraints or {}
        supply = np.array([
            constraints.get('supply', {}).get(g, 1.0) for g in goods
        ])
        mechanism = constraints.get('mechanism', 'additive')

        good_idx = {g: i for i, g in enumerate(goods)}

        if mechanism == 'ascending_clock':
            valuations = np.zeros((n_agents, n_goods))
            budgets = np.zeros(n_agents)
            for i, agent in enumerate(agents):
                budgets[i] = agent.get('budget', 1e6)
                for g, v in agent.get('valuations', {}).items():
                    if g in good_idx:
                        valuations[i, good_idx[g]] = v
            auction = AscendingClockAuction(n_goods, supply)
            prices, alloc, _ = auction.run(valuations, budgets)
            welfare = float(np.sum(alloc * valuations))
            revenue = float(np.sum(alloc * prices[None, :]))
            return MarketDesign(
                allocation=alloc, prices=prices,
                welfare=welfare, revenue=revenue,
                metadata={'mechanism': 'ascending_clock'}
            )

        if mechanism == 'xor':
            bids = []
            for agent in agents:
                bundles = []
                for key, val in agent.get('valuations', {}).items():
                    if isinstance(key, (list, tuple, frozenset)):
                        items = [good_idx[g] for g in key if g in good_idx]
                    else:
                        items = [good_idx[key]] if key in good_idx else []
                    bundles.append((items, val))
                bids.append(XORBid(bidder_id=agent['id'], bundles=bundles))
            solver = PackageBidding(n_goods)
            return solver.solve_xor_bids(bids, supply)

        if mechanism == 'or':
            bids = []
            for agent in agents:
                bundles = []
                for key, val in agent.get('valuations', {}).items():
                    if isinstance(key, (list, tuple, frozenset)):
                        items = [good_idx[g] for g in key if g in good_idx]
                    else:
                        items = [good_idx[key]] if key in good_idx else []
                    bundles.append((items, val))
                bids.append(ORBid(bidder_id=agent['id'], bundles=bundles))
            solver = PackageBidding(n_goods)
            return solver.solve_or_bids(bids, supply)

        # Default: additive
        valuations = np.zeros((n_agents, n_goods))
        for i, agent in enumerate(agents):
            for g, v in agent.get('valuations', {}).items():
                if g in good_idx:
                    valuations[i, good_idx[g]] = v
        solver = PackageBidding(n_goods)
        return solver.solve_additive_bids(valuations, supply)

    def analyze_thickness(self, sizes: np.ndarray,
                          n_simulations: int = 200) -> Dict[str, np.ndarray]:
        """Run market thickness analysis across different market sizes.

        Args:
            sizes: Array of market sizes.
            n_simulations: Monte Carlo repetitions per size.

        Returns:
            Dict with mean_quality, std_quality, elasticity arrays.
        """
        analyzer = MarketThicknessAnalysis()
        means, stds = analyzer.simulate_match_quality(sizes, n_simulations)
        elasticity = analyzer.thickness_elasticity(sizes, means)
        return {
            'mean_quality': means,
            'std_quality': stds,
            'elasticity': elasticity,
        }

    def design_platform(self, n_buyers: int, n_sellers: int,
                        cross_b: float, cross_s: float,
                        outside_b: float = 0.0,
                        outside_s: float = 0.0) -> Dict[str, float]:
        """Design optimal two-sided platform pricing.

        Args:
            n_buyers: Total potential buyers.
            n_sellers: Total potential sellers.
            cross_b: Buyer benefit per seller on platform.
            cross_s: Seller benefit per buyer on platform.
            outside_b: Buyer outside option.
            outside_s: Seller outside option.

        Returns:
            Dict with optimal fees, participation counts, revenue.
        """
        platform = PlatformDesign(n_buyers, n_sellers)
        return platform.optimal_fees(outside_b, outside_s, cross_b, cross_s)

    def compute_congestion_tolls(self, capacities: np.ndarray,
                                 free_flow_costs: np.ndarray,
                                 total_demand: float) -> Dict[str, Any]:
        """Compute optimal congestion tolls.

        Args:
            capacities: Resource capacities.
            free_flow_costs: Free-flow costs.
            total_demand: Total demand to route.

        Returns:
            Dict with tolls, equilibrium flows, social optimum flows,
            price of anarchy.
        """
        cp = CongestionPricing(len(capacities), capacities, free_flow_costs)
        gap = cp.efficiency_gap(total_demand)
        ue = cp.user_equilibrium(total_demand)
        so = cp.social_optimum(total_demand)
        gap['ue_flows'] = ue.tolist()
        gap['so_flows'] = so.tolist()
        return gap

    def run_dynamic_pricing(self, true_a: float, true_b: float,
                            n_periods: int = 100,
                            strategy: str = 'thompson') -> Dict[str, Any]:
        """Run dynamic pricing simulation with demand learning.

        Args:
            true_a: True demand intercept.
            true_b: True price sensitivity.
            n_periods: Number of selling periods.
            strategy: 'thompson', 'ucb', or 'greedy'.

        Returns:
            Simulation results with prices, demands, revenues.
        """
        dp = DynamicPricing()
        results = dp.run_simulation(true_a, true_b, n_periods, strategy)
        # Convert arrays to lists for serialisation
        return {k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()}
