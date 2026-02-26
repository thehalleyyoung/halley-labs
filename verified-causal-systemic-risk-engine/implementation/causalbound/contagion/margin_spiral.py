"""
Margin Spiral Model
====================

Models the procyclical dynamics of margin calls in cleared and bilateral
derivatives markets. Under stress, rising volatility triggers increased
margin requirements, forcing institutions to raise liquidity, which can
amplify market stress in a self-reinforcing spiral.

Covers:
    - Variation margin (VM) calls from mark-to-market changes
    - Initial margin (IM) increases under stress (procyclicality)
    - Funding liquidity feedback loops
    - CCP waterfall implications

References:
    - Brunnermeier, M. & Pedersen, L.H. (2009). Market liquidity and
      funding liquidity. Review of Financial Studies, 22(6), 2201-2238.
    - Murphy, D., Vasios, M., & Vause, N. (2014). An investigation into
      the procyclicality of risk-based initial margin models. Bank of
      England Financial Stability Paper No. 29.
    - BCBS-CPMI-IOSCO (2022). Review of margining practices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class MarginModelType(Enum):
    """Initial margin model types."""
    HISTORICAL_VaR = "hist_var"
    PARAMETRIC_VaR = "param_var"
    EWMA = "ewma"
    FILTERED_HS = "filtered_hs"


@dataclass
class Position:
    """A derivatives position between two counterparties."""
    institution_id: int
    counterparty_id: int
    notional: float
    mark_to_market: float  # current MTM value
    asset_class: str = "rates"
    delta: float = 1.0  # sensitivity to underlying
    initial_margin: float = 0.0
    variation_margin: float = 0.0
    margin_period_of_risk: int = 10  # days


@dataclass
class MarginParams:
    """Parameters for margin computation."""
    confidence_level: float = 0.99  # VaR confidence
    lookback_window: int = 250  # historical lookback in days
    margin_period_of_risk: int = 10  # days
    floor_percentile: float = 0.25  # anti-procyclicality floor
    stressed_period_weight: float = 0.25  # weight on stressed returns
    ewma_lambda: float = 0.94  # EWMA decay factor
    im_model: MarginModelType = MarginModelType.HISTORICAL_VaR
    vm_frequency: str = "daily"  # daily or intraday
    minimum_transfer_amount: float = 5e5  # MTA
    threshold: float = 0.0  # IM threshold


@dataclass
class FundingCapacity:
    """Funding capacity of an institution."""
    institution_id: int
    cash_reserves: float
    credit_lines_available: float
    repo_capacity: float
    eligible_collateral: float
    total_capacity: float = 0.0

    def __post_init__(self) -> None:
        self.total_capacity = (
            self.cash_reserves
            + self.credit_lines_available
            + self.repo_capacity
        )


@dataclass
class MarginSpiralResult:
    """Results from a margin spiral simulation."""
    total_margin_calls: float
    total_vm_calls: float
    total_im_increase: float
    im_increase_pct: float  # percentage increase from baseline
    n_funding_failures: int
    failed_institutions: List[int]
    round_by_round: List[Dict[str, Any]]
    procyclicality_index: float
    max_single_day_call: float
    system_funding_shortfall: float


class MarginSpiralModel:
    """Margin spiral dynamics model.

    Simulates the feedback loop between market stress, margin increases,
    and funding liquidity. Models both variation margin (daily P&L
    settlement) and initial margin (collateral for potential future
    exposure) under stress scenarios.

    Example:
        >>> model = MarginSpiralModel()
        >>> positions = [Position(0, 1, 1e9, 1e7), Position(1, 0, 1e9, -1e7)]
        >>> market_moves = np.random.normal(0, 0.02, (10, 1))
        >>> params = MarginParams()
        >>> result = model.simulate_margin_spiral(positions, market_moves, params)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def simulate_margin_spiral(
        self,
        positions: List[Position],
        market_moves: np.ndarray,
        margin_params: MarginParams,
        funding_capacities: Optional[Dict[int, FundingCapacity]] = None,
    ) -> MarginSpiralResult:
        """Simulate margin spiral dynamics over a stress scenario.

        Args:
            positions: List of derivatives positions.
            market_moves: Array of market moves (n_days, n_risk_factors).
            margin_params: Margin calculation parameters.
            funding_capacities: Funding capacity per institution.

        Returns:
            MarginSpiralResult with spiral dynamics details.
        """
        n_days = market_moves.shape[0]
        n_factors = market_moves.shape[1] if market_moves.ndim > 1 else 1
        if market_moves.ndim == 1:
            market_moves = market_moves.reshape(-1, 1)

        # Identify all institutions
        institutions = set()
        for pos in positions:
            institutions.add(pos.institution_id)
            institutions.add(pos.counterparty_id)
        inst_list = sorted(institutions)

        # Initialise funding
        if funding_capacities is None:
            funding_capacities = {}
            for inst_id in inst_list:
                total_notional = sum(
                    pos.notional for pos in positions
                    if pos.institution_id == inst_id
                )
                funding_capacities[inst_id] = FundingCapacity(
                    institution_id=inst_id,
                    cash_reserves=total_notional * 0.05,
                    credit_lines_available=total_notional * 0.10,
                    repo_capacity=total_notional * 0.08,
                    eligible_collateral=total_notional * 0.15,
                )

        # Compute baseline IM
        baseline_ims = self._compute_initial_margins(
            positions, np.zeros((margin_params.lookback_window, n_factors)),
            margin_params, is_baseline=True,
        )

        # Build return history for margin computation
        return_history = self.rng.normal(0, 0.01, (margin_params.lookback_window, n_factors))

        total_vm = 0.0
        total_im_increase = 0.0
        total_margin_calls = 0.0
        max_single_day = 0.0
        failed_institutions: List[int] = []
        round_data: List[Dict[str, Any]] = []
        baseline_im_total = sum(baseline_ims.values())

        funding_remaining = {
            inst_id: cap.total_capacity
            for inst_id, cap in funding_capacities.items()
        }

        for day in range(n_days):
            move = market_moves[day]

            # Update return history (rolling window)
            return_history = np.vstack([return_history[1:], move.reshape(1, -1)])

            # Compute variation margin calls
            vm_calls = self.compute_margin_calls(positions, move)
            day_vm_total = sum(abs(v) for v in vm_calls.values())

            # Update positions MTM
            for pos in positions:
                pos.mark_to_market += pos.delta * pos.notional * float(move.mean())

            # Compute new initial margins (stress-adjusted)
            current_ims = self._compute_initial_margins(
                positions, return_history, margin_params, is_baseline=False,
            )

            # IM increases
            day_im_increase = 0.0
            for inst_id in inst_list:
                im_new = current_ims.get(inst_id, 0.0)
                im_old = baseline_ims.get(inst_id, 0.0)
                if im_new > im_old:
                    day_im_increase += (im_new - im_old)

            # Total margin calls this day
            day_total = day_vm_total + day_im_increase
            max_single_day = max(max_single_day, day_total)

            # Check funding capacity
            n_failures_today = 0
            for inst_id in inst_list:
                call = abs(vm_calls.get(inst_id, 0.0))
                im_call = max(0, current_ims.get(inst_id, 0.0) - baseline_ims.get(inst_id, 0.0))
                total_call = call + im_call

                if total_call > margin_params.minimum_transfer_amount:
                    funding_remaining[inst_id] -= total_call
                    if funding_remaining[inst_id] < 0 and inst_id not in failed_institutions:
                        failed_institutions.append(inst_id)
                        n_failures_today += 1

            total_vm += day_vm_total
            total_im_increase += day_im_increase
            total_margin_calls += day_total

            # Update baseline for next day
            baseline_ims = current_ims.copy()

            round_data.append({
                "day": day,
                "vm_calls": day_vm_total,
                "im_increase": day_im_increase,
                "total_calls": day_total,
                "n_failures": n_failures_today,
                "cumulative_failures": len(failed_institutions),
                "market_move": float(move.mean()),
            })

        # Procyclicality measure
        procyclicality = self.procyclicality_measure(
            margin_params,
            market_moves[:, 0] if n_factors >= 1 else market_moves.flatten(),
        )

        im_increase_pct = (
            total_im_increase / baseline_im_total * 100
            if baseline_im_total > 0 else 0.0
        )

        system_shortfall = sum(
            -funding_remaining[inst_id]
            for inst_id in failed_institutions
            if funding_remaining[inst_id] < 0
        )

        return MarginSpiralResult(
            total_margin_calls=total_margin_calls,
            total_vm_calls=total_vm,
            total_im_increase=total_im_increase,
            im_increase_pct=im_increase_pct,
            n_funding_failures=len(failed_institutions),
            failed_institutions=failed_institutions,
            round_by_round=round_data,
            procyclicality_index=procyclicality,
            max_single_day_call=max_single_day,
            system_funding_shortfall=system_shortfall,
        )

    def compute_margin_calls(
        self,
        positions: List[Position],
        market_moves: np.ndarray,
    ) -> Dict[int, float]:
        """Compute variation margin calls from market moves.

        Positive values = institution must pay margin.
        Negative values = institution receives margin.

        Args:
            positions: List of positions.
            market_moves: Market move vector for this period.

        Returns:
            Dictionary of institution_id -> net VM call.
        """
        vm_calls: Dict[int, float] = {}
        move_scalar = float(market_moves.mean()) if market_moves.ndim > 0 else float(market_moves)

        for pos in positions:
            pnl = pos.delta * pos.notional * move_scalar
            # Institution's VM call (pays if losing)
            inst_id = pos.institution_id
            cp_id = pos.counterparty_id

            vm_calls[inst_id] = vm_calls.get(inst_id, 0.0) - pnl
            vm_calls[cp_id] = vm_calls.get(cp_id, 0.0) + pnl

        return vm_calls

    def funding_liquidity_impact(
        self,
        margin_calls: Dict[int, float],
        funding_capacity: Dict[int, FundingCapacity],
    ) -> Dict[int, Dict[str, Any]]:
        """Assess the funding liquidity impact of margin calls.

        Determines which institutions can meet their margin calls and
        which face funding shortfalls.

        Args:
            margin_calls: Net margin calls per institution.
            funding_capacity: Available funding per institution.

        Returns:
            Dictionary with funding status per institution.
        """
        results: Dict[int, Dict[str, Any]] = {}

        for inst_id, call in margin_calls.items():
            if call <= 0:
                results[inst_id] = {
                    "call": call,
                    "can_meet": True,
                    "shortfall": 0.0,
                    "funding_usage_pct": 0.0,
                    "stress_level": "none",
                }
                continue

            capacity = funding_capacity.get(inst_id)
            if capacity is None:
                results[inst_id] = {
                    "call": call,
                    "can_meet": False,
                    "shortfall": call,
                    "funding_usage_pct": 100.0,
                    "stress_level": "critical",
                }
                continue

            can_meet = call <= capacity.total_capacity
            shortfall = max(0.0, call - capacity.total_capacity)
            usage_pct = (call / capacity.total_capacity * 100) if capacity.total_capacity > 0 else 100.0

            if usage_pct < 30:
                stress = "low"
            elif usage_pct < 60:
                stress = "moderate"
            elif usage_pct < 90:
                stress = "high"
            else:
                stress = "critical"

            # Waterfall: cash -> credit lines -> repo
            remaining_call = call
            cash_used = min(remaining_call, capacity.cash_reserves)
            remaining_call -= cash_used
            credit_used = min(remaining_call, capacity.credit_lines_available)
            remaining_call -= credit_used
            repo_used = min(remaining_call, capacity.repo_capacity)
            remaining_call -= repo_used

            results[inst_id] = {
                "call": call,
                "can_meet": can_meet,
                "shortfall": shortfall,
                "funding_usage_pct": usage_pct,
                "stress_level": stress,
                "cash_used": cash_used,
                "credit_lines_used": credit_used,
                "repo_used": repo_used,
                "unfunded": remaining_call,
            }

        return results

    def procyclicality_measure(
        self,
        margin_params: MarginParams,
        volatility_path: np.ndarray,
    ) -> float:
        """Measure the procyclicality of the margin model.

        Computes how much margin requirements increase during stress periods
        relative to calm periods, measuring the tendency of margins to
        amplify market stress.

        The procyclicality index is defined as:
            PI = max(IM_stressed) / mean(IM_calm) - 1

        Higher values indicate more procyclical margin models.

        Args:
            margin_params: Margin model parameters.
            volatility_path: Time series of returns/volatility.

        Returns:
            Procyclicality index (0 = no procyclicality).
        """
        n = len(volatility_path)
        if n < 50:
            return 0.0

        window = margin_params.lookback_window
        confidence = margin_params.confidence_level

        # Compute rolling VaR-based margin estimates
        margin_series = np.zeros(max(0, n - window))
        for t in range(window, n):
            historical_returns = volatility_path[t - window:t]
            vol = np.std(historical_returns)

            if margin_params.im_model == MarginModelType.HISTORICAL_VaR:
                var = np.percentile(
                    np.abs(historical_returns),
                    confidence * 100,
                )
                margin_series[t - window] = var * np.sqrt(margin_params.margin_period_of_risk)
            elif margin_params.im_model == MarginModelType.PARAMETRIC_VaR:
                z = stats.norm.ppf(confidence)
                margin_series[t - window] = (
                    z * vol * np.sqrt(margin_params.margin_period_of_risk)
                )
            elif margin_params.im_model == MarginModelType.EWMA:
                ewma_vol = self._compute_ewma_vol(
                    historical_returns, margin_params.ewma_lambda
                )
                z = stats.norm.ppf(confidence)
                margin_series[t - window] = (
                    z * ewma_vol * np.sqrt(margin_params.margin_period_of_risk)
                )
            else:
                margin_series[t - window] = vol * 3.0

        if len(margin_series) < 10:
            return 0.0

        # Split into calm and stressed periods (by realized volatility)
        vols = np.array([
            np.std(volatility_path[max(0, t - 20):t])
            for t in range(window, n)
        ])
        vol_threshold = np.percentile(vols, 75)
        calm_mask = vols < vol_threshold
        stressed_mask = ~calm_mask

        calm_margins = margin_series[calm_mask[:len(margin_series)]]
        stressed_margins = margin_series[stressed_mask[:len(margin_series)]]

        if len(calm_margins) == 0 or len(stressed_margins) == 0:
            return 0.0

        mean_calm = float(np.mean(calm_margins))
        max_stressed = float(np.max(stressed_margins))

        if mean_calm <= 0:
            return 0.0

        return (max_stressed / mean_calm) - 1.0

    def _compute_initial_margins(
        self,
        positions: List[Position],
        return_history: np.ndarray,
        params: MarginParams,
        is_baseline: bool = False,
    ) -> Dict[int, float]:
        """Compute initial margin requirements for each institution."""
        institution_ims: Dict[int, float] = {}

        # Group positions by institution
        inst_positions: Dict[int, List[Position]] = {}
        for pos in positions:
            if pos.institution_id not in inst_positions:
                inst_positions[pos.institution_id] = []
            inst_positions[pos.institution_id].append(pos)

        for inst_id, pos_list in inst_positions.items():
            total_notional = sum(pos.notional for pos in pos_list)

            if is_baseline or len(return_history) < 10:
                # Baseline IM: percentage of notional
                im = total_notional * 0.03  # 3% notional
            else:
                # Stress-adjusted IM
                vol = np.std(return_history[:, 0]) if return_history.shape[1] > 0 else 0.01
                z = stats.norm.ppf(params.confidence_level)
                mpor_factor = np.sqrt(params.margin_period_of_risk)
                im = total_notional * z * vol * mpor_factor

                # Anti-procyclicality floor
                floor = total_notional * 0.01  # minimum 1%
                im = max(im, floor)

            institution_ims[inst_id] = im

        return institution_ims

    def _compute_ewma_vol(
        self, returns: np.ndarray, lambda_param: float
    ) -> float:
        """Compute EWMA volatility from return series.

        Uses the exponentially weighted moving average formula:
            sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2

        Args:
            returns: Return series.
            lambda_param: Decay factor (0.94 is standard RiskMetrics).

        Returns:
            EWMA volatility estimate.
        """
        n = len(returns)
        if n == 0:
            return 0.01

        variance = float(returns[0] ** 2)
        for t in range(1, n):
            variance = lambda_param * variance + (1 - lambda_param) * returns[t] ** 2

        return float(np.sqrt(max(variance, 1e-12)))
