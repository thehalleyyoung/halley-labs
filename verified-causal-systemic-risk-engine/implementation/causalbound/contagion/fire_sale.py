"""
Fire-Sale Externality Model
=============================

Models the indirect contagion channel through asset fire sales. When
distressed institutions are forced to liquidate assets, the resulting
price impact imposes mark-to-market losses on all holders of similar
assets, potentially triggering further sales.

This model captures:
    - Price impact functions (linear, square-root, power-law)
    - Forced liquidation dynamics
    - Common asset exposure / portfolio overlap effects
    - Amplification through feedback loops

References:
    - Cifuentes, R., Ferrucci, G., & Shin, H.S. (2005). Liquidity risk
      and contagion. Journal of the European Economic Association, 3(2-3).
    - Greenwood, R., Landier, A., & Thesmar, D. (2015). Vulnerable banks.
      Journal of Financial Economics, 115(3), 471-485.
    - Cont, R. & Schaanning, E. (2017). Fire sales, indirect contagion and
      systemic stress testing. Norges Bank Working Paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import optimize


class PriceImpactType(Enum):
    """Price impact function types."""
    LINEAR = "linear"
    SQRT = "sqrt"
    POWER = "power"
    EXPONENTIAL = "exponential"


@dataclass
class AssetHoldings:
    """Portfolio holdings across institutions and assets.

    The holdings matrix H has shape (n_institutions, n_assets) where
    H[i, a] is the amount of asset a held by institution i.
    """
    holdings: np.ndarray  # (n_institutions, n_assets)
    asset_prices: np.ndarray  # (n_assets,) current prices
    asset_market_depths: np.ndarray  # (n_assets,) market depth
    asset_names: Optional[List[str]] = None
    institution_names: Optional[List[int]] = None


@dataclass
class FireSaleResult:
    """Results from a fire-sale simulation."""
    price_impacts: np.ndarray  # price change per asset
    final_prices: np.ndarray  # post-fire-sale prices
    total_liquidation: np.ndarray  # total sold per asset
    institution_losses: Dict[int, float]  # mark-to-market loss per institution
    defaulted_institutions: Set[int]
    n_rounds: int
    amplification_factor: float  # total loss / direct loss
    system_loss: float
    round_history: Optional[List[Dict[str, Any]]] = None


@dataclass
class OverlapMatrix:
    """Portfolio overlap between institutions."""
    cosine_similarity: np.ndarray  # (n_inst, n_inst) cosine similarity
    leverage_weighted: np.ndarray  # overlap weighted by leverage
    vulnerability_scores: np.ndarray  # per-institution vulnerability
    systemic_overlap: float  # system-wide overlap measure


class FireSaleModel:
    """Fire-sale externality model for systemic risk.

    Models the indirect contagion channel where forced asset liquidation
    by distressed institutions depresses prices, imposing mark-to-market
    losses on all holders of the same assets.

    Example:
        >>> model = FireSaleModel()
        >>> holdings = AssetHoldings(
        ...     holdings=np.random.rand(10, 5) * 1e8,
        ...     asset_prices=np.ones(5) * 100,
        ...     asset_market_depths=np.ones(5) * 1e9,
        ... )
        >>> result = model.simulate_fire_sale(graph, {0, 1}, holdings)
    """

    def __init__(
        self,
        price_impact_type: PriceImpactType = PriceImpactType.LINEAR,
        price_impact_coeff: float = 1e-10,
        power_exponent: float = 0.5,
        max_rounds: int = 100,
        leverage_threshold: float = 30.0,
        capital_attr: str = "capital",
        size_attr: str = "size",
    ):
        """Initialise the fire-sale model.

        Args:
            price_impact_type: Functional form of price impact.
            price_impact_coeff: Price impact coefficient (lambda).
            power_exponent: Exponent for power-law price impact.
            max_rounds: Maximum simulation rounds.
            leverage_threshold: Leverage ratio triggering forced liquidation.
            capital_attr: Node attribute for capital.
            size_attr: Node attribute for total assets.
        """
        self.price_impact_type = price_impact_type
        self.price_impact_coeff = price_impact_coeff
        self.power_exponent = power_exponent
        self.max_rounds = max_rounds
        self.leverage_threshold = leverage_threshold
        self.capital_attr = capital_attr
        self.size_attr = size_attr

    def compute_price_impact(
        self,
        sell_volume: np.ndarray,
        market_depth: np.ndarray,
    ) -> np.ndarray:
        """Compute price impact from selling pressure.

        Args:
            sell_volume: Volume sold per asset.
            market_depth: Market depth (liquidity) per asset.

        Returns:
            Fractional price change per asset (negative = price decline).
        """
        safe_depth = np.maximum(market_depth, 1.0)
        ratio = sell_volume / safe_depth

        if self.price_impact_type == PriceImpactType.LINEAR:
            impact = -self.price_impact_coeff * ratio
        elif self.price_impact_type == PriceImpactType.SQRT:
            impact = -self.price_impact_coeff * np.sqrt(ratio)
        elif self.price_impact_type == PriceImpactType.POWER:
            impact = -self.price_impact_coeff * np.power(ratio, self.power_exponent)
        elif self.price_impact_type == PriceImpactType.EXPONENTIAL:
            impact = -(1 - np.exp(-self.price_impact_coeff * ratio))
        else:
            impact = -self.price_impact_coeff * ratio

        return np.clip(impact, -1.0, 0.0)

    def simulate_fire_sale(
        self,
        graph: nx.DiGraph,
        initial_defaults: set,
        asset_holdings: AssetHoldings,
        track_history: bool = True,
    ) -> FireSaleResult:
        """Simulate fire-sale dynamics from initial defaults.

        Institutions that default or become over-leveraged are forced to
        liquidate assets. The resulting price impact imposes losses on
        all holders, potentially triggering further liquidations.

        Args:
            graph: Financial network (for institution attributes).
            initial_defaults: Set of initially defaulting nodes.
            asset_holdings: Portfolio holdings and asset data.
            track_history: Whether to record round-by-round state.

        Returns:
            FireSaleResult with price impacts, losses, and dynamics.
        """
        nodes = list(graph.nodes())
        n_inst = len(nodes)
        n_assets = asset_holdings.holdings.shape[1]
        node_to_idx = {nd: i for i, nd in enumerate(nodes)}

        # State variables
        holdings = asset_holdings.holdings.copy()
        prices = asset_holdings.asset_prices.copy()
        initial_prices = prices.copy()
        market_depth = asset_holdings.asset_market_depths.copy()

        capital = np.array([
            graph.nodes[nd].get(self.capital_attr, 1e8) for nd in nodes
        ])
        sizes = np.array([
            graph.nodes[nd].get(self.size_attr, 1e9) for nd in nodes
        ])

        defaulted = set()
        for nd in initial_defaults:
            if nd in node_to_idx:
                defaulted.add(nd)

        total_liquidation = np.zeros(n_assets)
        inst_losses = {nd: 0.0 for nd in nodes}
        history: List[Dict[str, Any]] = []
        direct_loss = 0.0

        for rnd in range(self.max_rounds):
            # Determine which institutions must liquidate
            liquidating = set(defaulted)

            # Check leverage constraints
            for idx, nd in enumerate(nodes):
                if nd in defaulted:
                    continue
                portfolio_value = float(holdings[idx] @ prices)
                leverage = portfolio_value / max(capital[idx], 1.0)
                if leverage > self.leverage_threshold:
                    liquidating.add(nd)

            if not liquidating and rnd > 0:
                break

            # Compute forced sales
            sell_volume = np.zeros(n_assets)
            for nd in liquidating:
                if nd not in node_to_idx:
                    continue
                idx = node_to_idx[nd]
                if nd in defaulted:
                    # Fully liquidate
                    sell_volume += holdings[idx]
                    if rnd == 0:
                        direct_loss += float(holdings[idx] @ prices)
                    holdings[idx] = 0
                else:
                    # Partial liquidation to restore leverage
                    portfolio_value = float(holdings[idx] @ prices)
                    target_leverage = self.leverage_threshold * 0.8
                    excess = portfolio_value - capital[idx] * target_leverage
                    if excess > 0:
                        fraction_to_sell = min(0.5, excess / max(portfolio_value, 1.0))
                        sale = holdings[idx] * fraction_to_sell
                        sell_volume += sale
                        holdings[idx] -= sale

            # Compute price impact
            price_change = self.compute_price_impact(sell_volume, market_depth)
            prices = prices * (1.0 + price_change)
            prices = np.maximum(prices, initial_prices * 0.01)  # Floor at 1%

            total_liquidation += sell_volume

            # Mark-to-market losses for all institutions
            new_defaults = set()
            for idx, nd in enumerate(nodes):
                if nd in defaulted:
                    continue
                mtm_loss = float(holdings[idx] @ (-price_change * initial_prices))
                inst_losses[nd] += mtm_loss
                capital[idx] -= mtm_loss

                if capital[idx] <= 0:
                    new_defaults.add(nd)
                    defaulted.add(nd)

            if track_history:
                history.append({
                    "round": rnd,
                    "n_liquidating": len(liquidating),
                    "sell_volume_total": float(sell_volume.sum()),
                    "avg_price_change": float(price_change.mean()),
                    "new_defaults": len(new_defaults),
                    "total_defaults": len(defaulted),
                })

            if not new_defaults and rnd > 0:
                break

        # Final metrics
        price_impacts = (prices - initial_prices) / initial_prices
        system_loss = float(sum(inst_losses.values()))
        amplification = system_loss / max(direct_loss, 1.0) if direct_loss > 0 else 1.0

        return FireSaleResult(
            price_impacts=price_impacts,
            final_prices=prices,
            total_liquidation=total_liquidation,
            institution_losses=inst_losses,
            defaulted_institutions=defaulted,
            n_rounds=rnd + 1,
            amplification_factor=amplification,
            system_loss=system_loss,
            round_history=history if track_history else None,
        )

    def compute_indirect_contagion(
        self,
        price_impacts: np.ndarray,
        portfolio_overlaps: OverlapMatrix,
        holdings: AssetHoldings,
    ) -> Dict[int, float]:
        """Compute indirect contagion through common asset exposure.

        Given price impacts from one set of fire sales, compute the
        mark-to-market losses for all institutions based on their
        portfolio overlaps with the selling institutions.

        Args:
            price_impacts: Fractional price changes per asset.
            portfolio_overlaps: Overlap matrix between institutions.
            holdings: Current portfolio holdings.

        Returns:
            Mapping of institution index to indirect loss.
        """
        n_inst = holdings.holdings.shape[0]
        losses: Dict[int, float] = {}

        for i in range(n_inst):
            # Loss = sum over assets of (holdings * price_change * original_price)
            loss = float(-holdings.holdings[i] @ (price_impacts * holdings.asset_prices))
            losses[i] = max(0.0, loss)

        return losses

    def compute_portfolio_overlap(
        self,
        holdings: AssetHoldings,
        leverages: Optional[np.ndarray] = None,
    ) -> OverlapMatrix:
        """Compute portfolio overlap matrix between institutions.

        Uses cosine similarity of portfolio holdings vectors, optionally
        weighted by leverage to capture amplification potential.

        Args:
            holdings: Portfolio holdings data.
            leverages: Leverage ratio per institution.

        Returns:
            OverlapMatrix with similarity measures.
        """
        H = holdings.holdings.copy()
        n_inst = H.shape[0]

        # Value-weighted holdings
        H_valued = H * holdings.asset_prices[np.newaxis, :]

        # Cosine similarity
        norms = np.linalg.norm(H_valued, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        H_normed = H_valued / norms
        cosine_sim = H_normed @ H_normed.T

        # Leverage-weighted overlap
        if leverages is None:
            leverages = np.ones(n_inst) * 10.0
        lev_weights = np.sqrt(np.outer(leverages, leverages))
        leverage_weighted = cosine_sim * lev_weights

        # Per-institution vulnerability score
        vulnerability = np.zeros(n_inst)
        for i in range(n_inst):
            portfolio_value = float(H_valued[i].sum())
            if portfolio_value > 0:
                # Concentration of portfolio in illiquid assets
                asset_illiquidity = 1.0 / np.maximum(
                    holdings.asset_market_depths, 1.0
                )
                weighted_illiq = float(H_valued[i] @ asset_illiquidity)
                vulnerability[i] = weighted_illiq * leverages[i]

        # System-wide overlap
        off_diag = cosine_sim[~np.eye(n_inst, dtype=bool)]
        systemic_overlap = float(off_diag.mean()) if len(off_diag) > 0 else 0.0

        return OverlapMatrix(
            cosine_similarity=cosine_sim,
            leverage_weighted=leverage_weighted,
            vulnerability_scores=vulnerability,
            systemic_overlap=systemic_overlap,
        )

    def get_amplification_factor(
        self,
        graph: nx.DiGraph,
        asset_holdings: AssetHoldings,
        n_simulations: int = 100,
        shock_fraction: float = 0.05,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate the fire-sale amplification factor.

        Compares total losses (direct + fire-sale induced) to direct losses
        alone, measuring how much fire sales amplify the initial shock.

        Args:
            graph: Financial network.
            asset_holdings: Portfolio holdings.
            n_simulations: Number of Monte Carlo runs.
            shock_fraction: Fraction of nodes initially shocked.
            seed: Random seed.

        Returns:
            Dictionary with amplification statistics.
        """
        rng = np.random.default_rng(seed)
        nodes = list(graph.nodes())
        n = len(nodes)
        n_shock = max(1, int(n * shock_fraction))

        amplifications = []
        system_losses = []
        direct_losses = []

        for _ in range(n_simulations):
            shock_nodes = set(rng.choice(nodes, size=n_shock, replace=False).tolist())

            # Direct loss (without fire sales)
            direct = sum(
                graph.nodes[nd].get(self.size_attr, 1e9)
                for nd in shock_nodes
            )

            # Total loss with fire sales
            result = self.simulate_fire_sale(
                graph, shock_nodes, asset_holdings, track_history=False
            )

            if direct > 0:
                amp = result.system_loss / direct
            else:
                amp = 1.0

            amplifications.append(amp)
            system_losses.append(result.system_loss)
            direct_losses.append(direct)

        amp_arr = np.array(amplifications)

        return {
            "mean_amplification": float(amp_arr.mean()),
            "median_amplification": float(np.median(amp_arr)),
            "max_amplification": float(amp_arr.max()),
            "std_amplification": float(amp_arr.std()),
            "p95_amplification": float(np.percentile(amp_arr, 95)),
            "mean_system_loss": float(np.mean(system_losses)),
            "mean_direct_loss": float(np.mean(direct_losses)),
        }

    @staticmethod
    def generate_random_holdings(
        n_institutions: int,
        n_assets: int,
        total_assets_range: Tuple[float, float] = (1e8, 1e11),
        concentration: float = 0.3,
        seed: Optional[int] = None,
    ) -> AssetHoldings:
        """Generate random but realistic portfolio holdings.

        Args:
            n_institutions: Number of institutions.
            n_assets: Number of tradeable assets.
            total_assets_range: Range of total asset values.
            concentration: Portfolio concentration parameter.
            seed: Random seed.

        Returns:
            AssetHoldings with random portfolios.
        """
        rng = np.random.default_rng(seed)

        # Institution sizes (log-normal)
        log_sizes = rng.uniform(
            np.log(total_assets_range[0]),
            np.log(total_assets_range[1]),
            size=n_institutions,
        )
        sizes = np.exp(log_sizes)

        # Dirichlet-distributed portfolio weights with concentration
        alpha = np.ones(n_assets) * (1.0 / concentration)
        holdings = np.zeros((n_institutions, n_assets))
        for i in range(n_institutions):
            weights = rng.dirichlet(alpha)
            holdings[i] = sizes[i] * weights

        # Asset prices (around 100)
        prices = rng.lognormal(mean=np.log(100), sigma=0.5, size=n_assets)

        # Market depths (proportional to total holdings of each asset)
        total_per_asset = holdings.sum(axis=0)
        market_depths = total_per_asset * rng.uniform(2.0, 10.0, size=n_assets)

        return AssetHoldings(
            holdings=holdings,
            asset_prices=prices,
            asset_market_depths=market_depths,
            asset_names=[f"asset_{i}" for i in range(n_assets)],
            institution_names=list(range(n_institutions)),
        )
