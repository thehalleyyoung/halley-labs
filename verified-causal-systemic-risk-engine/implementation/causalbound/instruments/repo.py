"""Repurchase agreement model with haircut dynamics and fire-sale pricing.

Implements repo valuation, margin call mechanics, fire-sale price impact,
rollover risk, and CPD discretization for junction-tree inference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats, interpolate
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class CollateralAsset:
    """Description of a collateral asset in a repo transaction."""

    asset_id: str
    asset_class: str
    market_value: float
    face_value: float
    coupon_rate: float = 0.0
    maturity: float = 10.0
    credit_rating: str = "AAA"
    liquidity_score: float = 1.0
    volatility: float = 0.02
    is_eligible: bool = True

    def stressed_value(self, stress_factor: float) -> float:
        """Compute collateral value under stress.

        Parameters
        ----------
        stress_factor : float
            Multiplicative stress shock (e.g., 0.9 = 10% drop).

        Returns
        -------
        float
            Stressed market value.
        """
        return self.market_value * stress_factor


@dataclass
class RepoTerms:
    """Contractual terms for a repo agreement."""

    initial_margin: float = 0.02
    variation_margin_threshold: float = 0.01
    margin_call_frequency: float = 1.0 / 365.0
    substitution_allowed: bool = True
    rehypothecation_allowed: bool = False
    close_out_days: int = 2
    minimum_denomination: float = 1_000_000.0


class RepoModel:
    """Repurchase Agreement model for CausalBound.

    A repo is a secured short-term borrowing: the borrower sells a security
    (collateral) and agrees to repurchase it at a later date at a slightly
    higher price.

    Parameters
    ----------
    principal : float
        Cash amount borrowed / lent.
    repo_rate_annualized : float
        Annualized repo rate (decimal).
    tenor : float
        Term in years (e.g., 1/365 for overnight).
    collateral : CollateralAsset or list
        Collateral asset(s).
    terms : RepoTerms, optional
        Contract terms.
    is_borrower : bool
        True if this entity is the cash borrower.
    """

    def __init__(
        self,
        principal: float = 10_000_000.0,
        repo_rate_annualized: float = 0.02,
        tenor: float = 7.0 / 365.0,
        collateral: Optional[Union[CollateralAsset, List[CollateralAsset]]] = None,
        terms: Optional[RepoTerms] = None,
        is_borrower: bool = True,
    ) -> None:
        self.principal = principal
        self.repo_rate = repo_rate_annualized
        self.tenor = tenor
        self.is_borrower = is_borrower
        self.terms = terms if terms is not None else RepoTerms()

        if collateral is None:
            self.collateral = [
                CollateralAsset(
                    asset_id="GENERIC_BOND",
                    asset_class="government_bond",
                    market_value=principal * (1.0 + self.terms.initial_margin),
                    face_value=principal * (1.0 + self.terms.initial_margin),
                )
            ]
        elif isinstance(collateral, CollateralAsset):
            self.collateral = [collateral]
        else:
            self.collateral = list(collateral)

    @property
    def total_collateral_value(self) -> float:
        """Total market value of posted collateral."""
        return sum(c.market_value for c in self.collateral)

    @property
    def haircut(self) -> float:
        """Effective haircut = 1 - principal / collateral_value."""
        cv = self.total_collateral_value
        if cv < 1e-12:
            return 1.0
        return 1.0 - self.principal / cv

    def repo_rate_model(
        self,
        haircut: Optional[float] = None,
        collateral_quality: Optional[float] = None,
        market_stress: float = 0.0,
        fed_funds_rate: float = 0.02,
    ) -> float:
        """Model the repo rate as a function of haircut and collateral quality.

        repo_rate = fed_funds - spread(quality) + stress_premium + haircut_premium

        Parameters
        ----------
        haircut : float, optional
            Haircut level. If None, uses contract haircut.
        collateral_quality : float, optional
            Quality score in [0, 1]. 1 = highest quality (treasuries).
        market_stress : float
            Stress indicator in [0, 1]. 0 = normal, 1 = crisis.
        fed_funds_rate : float
            Federal funds target rate.

        Returns
        -------
        float
            Modeled repo rate.
        """
        h = haircut if haircut is not None else self.haircut
        q = collateral_quality if collateral_quality is not None else 1.0

        # GC spread for high-quality collateral
        gc_spread = 0.0010 * (1.0 - q)

        # Specialness premium (negative for special collateral)
        specialness = -0.0005 * q

        # Stress premium increases with market stress
        stress_premium = 0.005 * market_stress ** 2

        # Haircut impact: higher haircut implies cheaper funding
        haircut_adj = -0.001 * h

        rate = fed_funds_rate + gc_spread + specialness + stress_premium + haircut_adj
        return max(rate, 0.0)

    def repurchase_price(self) -> float:
        """Compute the repurchase price at maturity.

        Returns
        -------
        float
            Price to repurchase the collateral.
        """
        return self.principal * (1.0 + self.repo_rate * self.tenor)

    def margin_call(
        self,
        price_change: float,
        haircut: Optional[float] = None,
    ) -> Dict[str, float]:
        """Determine if a margin call is triggered.

        A margin call occurs when the collateral value falls below the
        required coverage ratio.

        Parameters
        ----------
        price_change : float
            Relative change in collateral price (e.g., -0.05 = 5% drop).
        haircut : float, optional
            Current haircut level.

        Returns
        -------
        dict
            'triggered': bool as 0/1,
            'margin_call_amount': required additional collateral,
            'new_coverage_ratio': updated coverage ratio,
            'excess_collateral': amount above/below requirement.
        """
        h = haircut if haircut is not None else self.terms.initial_margin
        required_collateral = self.principal * (1.0 + h)
        new_collateral = self.total_collateral_value * (1.0 + price_change)
        coverage_ratio = new_collateral / self.principal if self.principal > 0 else 0.0
        shortfall = required_collateral - new_collateral
        threshold = self.terms.variation_margin_threshold * self.principal

        triggered = shortfall > threshold
        margin_call_amount = max(shortfall, 0.0) if triggered else 0.0

        return {
            "triggered": float(triggered),
            "margin_call_amount": margin_call_amount,
            "new_coverage_ratio": coverage_ratio,
            "excess_collateral": -shortfall,
        }

    def fire_sale_impact(
        self,
        volume: float,
        market_depth: float,
        price_elasticity: float = 0.5,
        current_price: float = 100.0,
    ) -> Dict[str, float]:
        """Compute fire-sale price impact from forced collateral liquidation.

        Uses a square-root market impact model:
            impact = gamma * sqrt(volume / ADV) * sigma

        Parameters
        ----------
        volume : float
            Volume to be liquidated (in notional).
        market_depth : float
            Average daily volume (ADV) of the collateral.
        price_elasticity : float
            Kyle's lambda parameter for market impact.
        current_price : float
            Current collateral price.

        Returns
        -------
        dict
            'price_impact_pct': percentage price impact,
            'realized_price': price after impact,
            'loss_from_impact': loss due to fire-sale,
            'effective_recovery': fraction of value recovered.
        """
        if market_depth < 1e-12:
            return {
                "price_impact_pct": 1.0,
                "realized_price": 0.0,
                "loss_from_impact": volume,
                "effective_recovery": 0.0,
            }

        participation_rate = volume / market_depth
        avg_vol = 0.02

        # Almgren-Chriss temporary impact
        temporary_impact = price_elasticity * avg_vol * np.sqrt(participation_rate)
        # Permanent impact
        permanent_impact = 0.1 * price_elasticity * participation_rate

        total_impact = min(temporary_impact + permanent_impact, 0.99)
        realized_price = current_price * (1.0 - total_impact)
        loss = volume * total_impact
        recovery = 1.0 - total_impact

        return {
            "price_impact_pct": total_impact,
            "realized_price": realized_price,
            "loss_from_impact": loss,
            "effective_recovery": recovery,
        }

    def haircut_dynamics(
        self,
        market_vol: NDArray[np.float64],
        credit_spread: NDArray[np.float64],
        time_grid: Optional[NDArray] = None,
    ) -> NDArray[np.float64]:
        """Model haircut dynamics over time.

        Haircuts increase with market volatility and credit spread.

        h(t) = h_base + alpha * vol(t) + beta * spread(t) + gamma * vol(t) * spread(t)

        Parameters
        ----------
        market_vol : NDArray
            Market volatility path.
        credit_spread : NDArray
            Credit spread path in decimal.
        time_grid : NDArray, optional
            Time grid.

        Returns
        -------
        NDArray
            Haircut path over time.
        """
        h_base = self.terms.initial_margin
        alpha = 2.0
        beta = 5.0
        gamma = 10.0

        haircuts = (
            h_base
            + alpha * market_vol
            + beta * credit_spread
            + gamma * market_vol * credit_spread
        )
        return np.clip(haircuts, h_base, 0.50)

    def collateral_substitution_cost(
        self,
        old_asset: CollateralAsset,
        new_asset: CollateralAsset,
    ) -> float:
        """Compute cost of collateral substitution.

        Parameters
        ----------
        old_asset : CollateralAsset
            Asset being returned.
        new_asset : CollateralAsset
            Replacement asset.

        Returns
        -------
        float
            Net substitution cost (positive = more expensive).
        """
        if not self.terms.substitution_allowed:
            return float("inf")

        # Opportunity cost from quality difference
        quality_diff = old_asset.liquidity_score - new_asset.liquidity_score
        quality_cost = quality_diff * 0.001 * self.principal

        # Transaction cost
        bid_ask = 0.0005 * (old_asset.market_value + new_asset.market_value)

        # Haircut differential
        old_haircut = 0.02 / max(old_asset.liquidity_score, 0.1)
        new_haircut = 0.02 / max(new_asset.liquidity_score, 0.1)
        haircut_cost = (new_haircut - old_haircut) * self.principal

        return quality_cost + bid_ask + haircut_cost

    def rollover_probability(
        self,
        market_stress: float,
        tenor: Optional[float] = None,
        counterparty_credit: float = 1.0,
        collateral_quality: float = 1.0,
    ) -> float:
        """Estimate rollover probability.

        Parameters
        ----------
        market_stress : float
            Market stress level in [0, 1].
        tenor : float, optional
            Remaining tenor.
        counterparty_credit : float
            Counterparty credit quality in [0, 1].
        collateral_quality : float
            Collateral quality in [0, 1].

        Returns
        -------
        float
            Probability of successful rollover in [0, 1].
        """
        T = tenor if tenor is not None else self.tenor

        # Base rollover probability from logistic model
        z = (
            2.0 * counterparty_credit
            + 1.5 * collateral_quality
            - 3.0 * market_stress
            - 0.5 * np.log1p(T * 365.0)
            + 1.0
        )
        base_prob = 1.0 / (1.0 + np.exp(-z))

        # Crisis adjustment: sharp cliff in rollover at high stress
        if market_stress > 0.7:
            crisis_factor = np.exp(-5.0 * (market_stress - 0.7))
            base_prob *= crisis_factor

        return float(np.clip(base_prob, 0.0, 1.0))

    def funding_liquidity_risk(
        self,
        haircut_path: NDArray[np.float64],
        collateral_value_path: NDArray[np.float64],
        available_cash: float,
    ) -> Dict[str, NDArray]:
        """Assess funding liquidity risk from margin spirals.

        Parameters
        ----------
        haircut_path : NDArray
            Haircut values over time.
        collateral_value_path : NDArray
            Collateral values over time.
        available_cash : float
            Initial available cash buffer.

        Returns
        -------
        dict
            'margin_calls': margin call amounts per period,
            'cumulative_funding_need': cumulative cash needed,
            'cash_buffer': remaining cash buffer,
            'default_indicator': whether cash buffer is exhausted.
        """
        n = len(haircut_path)
        margin_calls = np.zeros(n)
        cumulative_need = np.zeros(n)
        cash = available_cash

        for t in range(n):
            required = self.principal * (1.0 + haircut_path[t])
            current_coll = collateral_value_path[t]
            shortfall = max(required - current_coll, 0.0)

            if shortfall > self.terms.variation_margin_threshold * self.principal:
                margin_calls[t] = shortfall
                cash -= shortfall

            cumulative_need[t] = max(-cash, 0.0) + np.sum(margin_calls[:t + 1])

        cash_buffer = np.maximum(
            available_cash - np.cumsum(margin_calls), 0.0
        )
        default_indicator = (cash_buffer <= 0).astype(float)

        return {
            "margin_calls": margin_calls,
            "cumulative_funding_need": cumulative_need,
            "cash_buffer": cash_buffer,
            "default_indicator": default_indicator,
        }

    def to_cpd(
        self,
        discretization: Dict,
    ) -> Dict[str, NDArray]:
        """Encode repo dynamics as a CPD for junction-tree inference.

        Conditions on collateral price change and market stress to produce
        distributions over funding outcomes.

        Parameters
        ----------
        discretization : dict
            Keys: 'price_change_bins', 'stress_bins', 'outcome_bins'.

        Returns
        -------
        dict
            'cpd': NDArray of shape (n_price, n_stress, n_outcome),
            plus bin edge arrays.
        """
        pc_bins = discretization.get(
            "price_change_bins", np.linspace(-0.30, 0.10, 15)
        )
        stress_bins = discretization.get(
            "stress_bins", np.linspace(0, 1, 10)
        )
        outcome_bins = discretization.get(
            "outcome_bins", np.array([0, 1, 2, 3])
        )

        n_pc = len(pc_bins) - 1
        n_stress = len(stress_bins) - 1
        n_outcome = len(outcome_bins) - 1

        cpd = np.zeros((n_pc, n_stress, n_outcome))

        for i in range(n_pc):
            pc_mid = 0.5 * (pc_bins[i] + pc_bins[i + 1])
            for j in range(n_stress):
                stress_mid = 0.5 * (stress_bins[j] + stress_bins[j + 1])

                mc = self.margin_call(pc_mid)
                rollover = self.rollover_probability(stress_mid)

                # Outcome encoding:
                #   0: no issue (no margin call, rollover succeeds)
                #   1: margin call but manageable
                #   2: rollover failure / forced liquidation
                if mc["triggered"] < 0.5 and rollover > 0.7:
                    cpd[i, j, 0] = rollover
                    cpd[i, j, min(1, n_outcome - 1)] = 1.0 - rollover
                elif mc["triggered"] > 0.5 and rollover > 0.3:
                    cpd[i, j, 0] = 0.1
                    cpd[i, j, min(1, n_outcome - 1)] = rollover * 0.8
                    if n_outcome > 2:
                        cpd[i, j, min(2, n_outcome - 1)] = 1.0 - 0.1 - rollover * 0.8
                        cpd[i, j, min(2, n_outcome - 1)] = max(cpd[i, j, min(2, n_outcome - 1)], 0.0)
                else:
                    if n_outcome > 2:
                        cpd[i, j, min(2, n_outcome - 1)] = 0.8
                    cpd[i, j, min(1, n_outcome - 1)] = 0.15
                    cpd[i, j, 0] = 0.05

                # Normalize
                row_sum = cpd[i, j, :].sum()
                if row_sum > 0:
                    cpd[i, j, :] /= row_sum
                else:
                    cpd[i, j, :] = 1.0 / n_outcome

        return {
            "cpd": cpd,
            "price_change_bins": pc_bins,
            "stress_bins": stress_bins,
            "outcome_bins": outcome_bins,
        }

    def simulate_repo_run(
        self,
        n_periods: int = 30,
        n_scenarios: int = 500,
        stress_path: Optional[NDArray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, NDArray]:
        """Simulate a repo run scenario.

        Models the cascading dynamics of haircut increases, margin calls,
        and forced liquidation.

        Parameters
        ----------
        n_periods : int
            Number of daily periods.
        n_scenarios : int
            Monte Carlo scenarios.
        stress_path : NDArray, optional
            Exogenous stress path. If None, simulated.
        rng : Generator, optional
            RNG.

        Returns
        -------
        dict
            'collateral_values', 'haircuts', 'margin_calls',
            'rollover_probs', 'default_times'.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        dt = 1.0 / 365.0
        init_value = self.total_collateral_value
        vol = np.mean([c.volatility for c in self.collateral])

        coll_values = np.zeros((n_scenarios, n_periods))
        haircuts = np.zeros((n_scenarios, n_periods))
        margin_calls = np.zeros((n_scenarios, n_periods))
        rollover_probs = np.zeros((n_scenarios, n_periods))
        default_times = np.full(n_scenarios, np.inf)

        if stress_path is None:
            base_stress = 0.3
            stress_drift = 0.01
            stress_vol = 0.05
        else:
            base_stress = stress_path[0] if len(stress_path) > 0 else 0.3

        for s in range(n_scenarios):
            price = init_value
            stress = base_stress
            defaulted = False

            for t in range(n_periods):
                if stress_path is not None and t < len(stress_path):
                    stress = stress_path[t]
                else:
                    dw_stress = rng.standard_normal() * np.sqrt(dt)
                    stress = stress + stress_drift * dt + stress_vol * dw_stress
                    stress = np.clip(stress, 0.0, 1.0)

                dw_price = rng.standard_normal() * np.sqrt(dt)
                ret = -0.5 * vol ** 2 * dt + vol * dw_price
                # Stress-induced return adjustment
                ret -= 0.1 * max(stress - 0.5, 0) * dt
                price *= np.exp(ret)

                coll_values[s, t] = price
                h = self.terms.initial_margin + 2.0 * stress * vol
                h = np.clip(h, self.terms.initial_margin, 0.5)
                haircuts[s, t] = h

                required = self.principal * (1.0 + h)
                shortfall = max(required - price, 0.0)
                if shortfall > self.terms.variation_margin_threshold * self.principal:
                    margin_calls[s, t] = shortfall

                rollover_probs[s, t] = self.rollover_probability(stress)

                if not defaulted and rollover_probs[s, t] < 0.1:
                    default_times[s] = t * dt
                    defaulted = True

        return {
            "collateral_values": coll_values,
            "haircuts": haircuts,
            "margin_calls": margin_calls,
            "rollover_probs": rollover_probs,
            "default_times": default_times,
        }

    def secured_funding_spread(
        self,
        unsecured_spread: float,
        haircut: Optional[float] = None,
        collateral_quality: float = 1.0,
    ) -> float:
        """Compute secured funding spread from unsecured spread.

        Parameters
        ----------
        unsecured_spread : float
            Unsecured borrowing spread.
        haircut : float, optional
            Haircut level.
        collateral_quality : float
            Quality in [0, 1].

        Returns
        -------
        float
            Secured funding spread.
        """
        h = haircut if haircut is not None else self.haircut
        # Secured spread = unsecured * (1 - collateral benefit)
        benefit = collateral_quality * (1.0 - h) * 0.9
        return unsecured_spread * (1.0 - benefit)
