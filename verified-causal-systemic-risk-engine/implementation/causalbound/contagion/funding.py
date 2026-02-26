"""
Funding Liquidity Model
========================

Models funding liquidity risk in financial networks, including wholesale
funding runs, credit line revocation, maturity transformation risk, and
liquidity coverage ratio effects.

This module captures the funding channel of contagion where loss of
confidence leads to withdrawal of short-term funding, forcing fire
sales or default even for solvent institutions.

References:
    - Diamond, D.W. & Dybvig, P.H. (1983). Bank runs, deposit insurance,
      and liquidity. Journal of Political Economy, 91(3), 401-419.
    - Brunnermeier, M. (2009). Deciphering the liquidity and credit crunch
      2007-2008. Journal of Economic Perspectives, 23(1), 77-100.
    - BCBS (2013). Basel III: The Liquidity Coverage Ratio and liquidity
      risk monitoring tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy import stats


class FundingType(Enum):
    """Types of wholesale funding."""
    UNSECURED_INTERBANK = "unsecured_interbank"
    REPO = "repo"
    COMMERCIAL_PAPER = "commercial_paper"
    CERTIFICATE_OF_DEPOSIT = "certificate_of_deposit"
    COVERED_BOND = "covered_bond"
    DEPOSITS = "deposits"


class StressLevel(Enum):
    """Stress severity levels."""
    NORMAL = "normal"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


@dataclass
class FundingProfile:
    """Funding structure of a financial institution."""
    institution_id: int
    funding_sources: Dict[FundingType, float]  # type -> amount
    maturity_buckets: Dict[str, float]  # e.g. "overnight": 1e9, "1w": 5e8
    total_funding: float = 0.0
    weighted_avg_maturity: float = 0.0  # in days
    stable_funding: float = 0.0  # funding unlikely to run
    volatile_funding: float = 0.0  # funding that may run

    def __post_init__(self) -> None:
        self.total_funding = sum(self.funding_sources.values())
        # Classify stability
        stable_types = {FundingType.DEPOSITS, FundingType.COVERED_BOND}
        self.stable_funding = sum(
            v for k, v in self.funding_sources.items()
            if k in stable_types
        )
        self.volatile_funding = self.total_funding - self.stable_funding

        # Compute weighted average maturity
        maturity_map = {
            "overnight": 1, "1w": 7, "2w": 14, "1m": 30,
            "3m": 90, "6m": 180, "1y": 365, "gt1y": 730,
        }
        total_weighted = 0.0
        for bucket, amount in self.maturity_buckets.items():
            days = maturity_map.get(bucket, 30)
            total_weighted += days * amount
        if self.total_funding > 0:
            self.weighted_avg_maturity = total_weighted / self.total_funding


@dataclass
class CreditEvent:
    """A credit event that may trigger funding withdrawal."""
    institution_id: int
    event_type: str  # "downgrade", "loss_announcement", "default", "rumour"
    severity: float  # 0 to 1
    timestamp: int  # day index


@dataclass
class LCRComponents:
    """Liquidity Coverage Ratio components."""
    high_quality_liquid_assets: float  # HQLA
    level1_assets: float  # cash, central bank reserves, sovereigns
    level2a_assets: float  # covered bonds, corporate bonds (AA-)
    level2b_assets: float  # equities, lower-rated corporates
    total_net_outflows_30d: float  # expected outflows minus inflows
    lcr: float = 0.0

    def __post_init__(self) -> None:
        if self.total_net_outflows_30d > 0:
            self.lcr = self.high_quality_liquid_assets / self.total_net_outflows_30d
        else:
            self.lcr = float("inf")


@dataclass
class FundingWithdrawalResult:
    """Results from funding withdrawal simulation."""
    total_withdrawal: float
    withdrawal_by_type: Dict[FundingType, float]
    surviving_institutions: Set[int]
    failed_institutions: Set[int]
    institution_shortfalls: Dict[int, float]
    rounds: int
    system_shortfall: float
    withdrawal_fraction: float  # total withdrawal / total system funding
    round_history: List[Dict[str, Any]]


class FundingLiquidityModel:
    """Funding liquidity withdrawal model.

    Simulates the dynamics of wholesale funding withdrawal in response
    to credit events, reputational contagion, and information cascades.
    Models bank run dynamics in the interbank market.

    Example:
        >>> model = FundingLiquidityModel()
        >>> events = [CreditEvent(0, "downgrade", 0.5, 0)]
        >>> result = model.simulate_funding_withdrawal(graph, events)
    """

    def __init__(
        self,
        run_threshold: float = 0.3,
        contagion_rate: float = 0.2,
        max_rounds: int = 30,
        seed: Optional[int] = None,
    ):
        """Initialise the funding liquidity model.

        Args:
            run_threshold: Fraction of funding loss triggering a run.
            contagion_rate: Rate at which funding stress spreads to neighbours.
            max_rounds: Maximum simulation rounds per day.
            seed: Random seed.
        """
        self.run_threshold = run_threshold
        self.contagion_rate = contagion_rate
        self.max_rounds = max_rounds
        self.rng = np.random.default_rng(seed)

    def simulate_funding_withdrawal(
        self,
        graph: nx.DiGraph,
        credit_events: List[CreditEvent],
        funding_profiles: Optional[Dict[int, FundingProfile]] = None,
    ) -> FundingWithdrawalResult:
        """Simulate funding withdrawal dynamics following credit events.

        Args:
            graph: Financial network (edges = funding relationships).
            credit_events: List of credit events triggering withdrawals.
            funding_profiles: Funding profiles per institution.

        Returns:
            FundingWithdrawalResult with withdrawal dynamics.
        """
        nodes = list(graph.nodes())
        n = len(nodes)

        if funding_profiles is None:
            funding_profiles = self._generate_default_profiles(graph, nodes)

        # Track funding state
        current_funding = {
            nd: funding_profiles[nd].total_funding
            for nd in nodes if nd in funding_profiles
        }
        initial_funding = dict(current_funding)
        total_system_funding = sum(current_funding.values())

        failed: Set[int] = set()
        survived: Set[int] = set(nodes)
        shortfalls: Dict[int, float] = {nd: 0.0 for nd in nodes}
        withdrawal_by_type: Dict[FundingType, float] = {ft: 0.0 for ft in FundingType}
        total_withdrawal = 0.0
        history: List[Dict[str, Any]] = []

        # Stress levels per institution
        stress = {nd: 0.0 for nd in nodes}

        # Apply credit events as initial stress
        for event in credit_events:
            nd = event.institution_id
            if nd in stress:
                stress[nd] = max(stress[nd], event.severity)

        # Simulate day by day
        n_days = max(1, max((e.timestamp for e in credit_events), default=0) + 1)
        n_days = min(n_days + 10, 60)  # extend for contagion

        for day in range(n_days):
            # Apply events for this day
            for event in credit_events:
                if event.timestamp == day:
                    nd = event.institution_id
                    if nd in stress:
                        stress[nd] = max(stress[nd], event.severity)

            day_withdrawal = 0.0
            day_failures = set()

            for rnd in range(self.max_rounds):
                round_withdrawal = 0.0
                new_failures = set()

                for nd in nodes:
                    if nd in failed:
                        continue
                    if nd not in funding_profiles:
                        continue

                    s = stress[nd]
                    if s <= 0:
                        continue

                    profile = funding_profiles[nd]
                    # Compute withdrawal based on stress and funding structure
                    withdrawal = self._compute_withdrawal(
                        profile, s, current_funding.get(nd, 0.0)
                    )

                    if withdrawal > 0:
                        current_funding[nd] = max(0, current_funding[nd] - withdrawal)
                        round_withdrawal += withdrawal

                        # Track by funding type
                        for ft, amount in profile.funding_sources.items():
                            fraction = amount / max(profile.total_funding, 1.0)
                            withdrawal_by_type[ft] += withdrawal * fraction

                        # Check for funding failure
                        lost_fraction = 1 - current_funding[nd] / max(initial_funding.get(nd, 1.0), 1.0)
                        if lost_fraction > self.run_threshold:
                            new_failures.add(nd)
                            failed.add(nd)
                            survived.discard(nd)
                            shortfalls[nd] = initial_funding.get(nd, 0.0) - current_funding[nd]

                # Contagion: stress spreads to counterparties
                for nd in list(new_failures) + [n for n in nodes if stress[n] > 0.1]:
                    if nd in failed:
                        for neighbor in graph.successors(nd):
                            if neighbor not in failed and neighbor in stress:
                                # Stress proportional to exposure
                                exposure = graph.edges[nd, neighbor].get("weight", 0.0)
                                neighbor_size = graph.nodes[neighbor].get("size", 1e9)
                                exposure_ratio = exposure / max(neighbor_size, 1.0)
                                stress_increment = self.contagion_rate * stress[nd] * min(exposure_ratio * 10, 1.0)
                                stress[neighbor] = min(1.0, stress[neighbor] + stress_increment)

                day_withdrawal += round_withdrawal
                total_withdrawal += round_withdrawal

                if round_withdrawal < 1e3 and not new_failures:
                    break

            history.append({
                "day": day,
                "withdrawal": day_withdrawal,
                "cumulative_withdrawal": total_withdrawal,
                "n_failed": len(failed),
                "new_failures": len(day_failures),
                "max_stress": max(stress.values()) if stress else 0.0,
                "avg_stress": float(np.mean(list(stress.values()))) if stress else 0.0,
            })

            # Decay stress slightly for non-failed institutions
            for nd in nodes:
                if nd not in failed:
                    stress[nd] *= 0.95

        withdrawal_fraction = total_withdrawal / max(total_system_funding, 1.0)
        system_shortfall = sum(shortfalls.values())

        return FundingWithdrawalResult(
            total_withdrawal=total_withdrawal,
            withdrawal_by_type=withdrawal_by_type,
            surviving_institutions=survived,
            failed_institutions=failed,
            institution_shortfalls=shortfalls,
            rounds=n_days,
            system_shortfall=system_shortfall,
            withdrawal_fraction=withdrawal_fraction,
            round_history=history,
        )

    def compute_rollover_risk(
        self,
        funding_profile: FundingProfile,
        stress_level: float,
    ) -> Dict[str, Any]:
        """Compute rollover risk for a given stress level.

        Estimates the probability and magnitude of funding rollover failure
        based on the institution's maturity profile and stress conditions.

        Args:
            funding_profile: Institution's funding structure.
            stress_level: Current stress level (0 to 1).

        Returns:
            Dictionary with rollover risk metrics.
        """
        maturity_map = {
            "overnight": 1, "1w": 7, "2w": 14, "1m": 30,
            "3m": 90, "6m": 180, "1y": 365, "gt1y": 730,
        }

        # Run-off rates by maturity under stress
        base_runoff = {
            "overnight": 0.10, "1w": 0.08, "2w": 0.06, "1m": 0.04,
            "3m": 0.02, "6m": 0.01, "1y": 0.005, "gt1y": 0.001,
        }

        total_at_risk = 0.0
        expected_runoff = 0.0
        rollover_details: Dict[str, Dict[str, float]] = {}

        for bucket, amount in funding_profile.maturity_buckets.items():
            base_rate = base_runoff.get(bucket, 0.03)
            # Stress multiplier (higher stress -> higher runoff, short maturity most affected)
            maturity_days = maturity_map.get(bucket, 30)
            stress_mult = 1 + stress_level * (3.0 * np.exp(-maturity_days / 30))
            adjusted_rate = min(1.0, base_rate * stress_mult)

            runoff = amount * adjusted_rate
            expected_runoff += runoff
            total_at_risk += amount if maturity_days <= 30 else 0

            rollover_details[bucket] = {
                "amount": amount,
                "maturity_days": maturity_days,
                "base_runoff_rate": base_rate,
                "stress_adjusted_rate": adjusted_rate,
                "expected_runoff": runoff,
            }

        # Rollover failure probability (simplified)
        volatile_fraction = funding_profile.volatile_funding / max(funding_profile.total_funding, 1.0)
        failure_prob = min(1.0, stress_level * volatile_fraction * 2)

        return {
            "total_at_risk_30d": total_at_risk,
            "expected_runoff": expected_runoff,
            "runoff_fraction": expected_runoff / max(funding_profile.total_funding, 1.0),
            "failure_probability": failure_prob,
            "weighted_avg_maturity": funding_profile.weighted_avg_maturity,
            "volatile_fraction": volatile_fraction,
            "details_by_bucket": rollover_details,
        }

    def credit_line_usage(
        self,
        graph: nx.DiGraph,
        stress_events: List[CreditEvent],
        credit_line_fractions: Optional[Dict[int, float]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Model credit line drawdowns under stress.

        During stress, institutions draw on committed credit lines from
        their counterparties, potentially straining lenders' liquidity.

        Args:
            graph: Financial network.
            stress_events: Credit events triggering drawdowns.
            credit_line_fractions: Fraction of exposure that is committed credit line.

        Returns:
            Dictionary with credit line usage per institution.
        """
        nodes = list(graph.nodes())

        # Default: 20% of interbank exposure is committed credit lines
        if credit_line_fractions is None:
            credit_line_fractions = {nd: 0.20 for nd in nodes}

        # Compute credit lines available to each institution
        credit_lines: Dict[int, float] = {}
        for nd in nodes:
            total_cl = 0.0
            for pred in graph.predecessors(nd):
                exposure = graph.edges[pred, nd].get("weight", 0.0)
                cl_frac = credit_line_fractions.get(pred, 0.20)
                total_cl += exposure * cl_frac
            credit_lines[nd] = total_cl

        # Determine stressed institutions and their drawdown behaviour
        stressed_nodes = {e.institution_id: e.severity for e in stress_events}

        results: Dict[int, Dict[str, Any]] = {}
        for nd in nodes:
            severity = stressed_nodes.get(nd, 0.0)
            available = credit_lines.get(nd, 0.0)

            # Drawdown probability increases with stress
            drawdown_prob = min(1.0, severity * 1.5)
            drawdown_fraction = min(1.0, severity * 0.8)
            drawn_amount = available * drawdown_fraction * drawdown_prob

            # Impact on lenders
            lender_impacts: Dict[int, float] = {}
            for pred in graph.predecessors(nd):
                exposure = graph.edges[pred, nd].get("weight", 0.0)
                cl_frac = credit_line_fractions.get(pred, 0.20)
                cl_amount = exposure * cl_frac
                lender_drain = cl_amount * drawdown_fraction * drawdown_prob
                lender_impacts[pred] = lender_drain

            results[nd] = {
                "credit_lines_available": available,
                "stress_severity": severity,
                "drawdown_probability": drawdown_prob,
                "expected_drawdown": drawn_amount,
                "drawdown_fraction": drawdown_fraction,
                "lender_impacts": lender_impacts,
            }

        return results

    def liquidity_coverage_impact(
        self,
        lcr: LCRComponents,
        outflows: Dict[str, float],
        stress_multiplier: float = 1.5,
    ) -> Dict[str, Any]:
        """Assess impact of stress on Liquidity Coverage Ratio.

        Computes how stressed outflows affect the LCR and whether
        the institution can maintain regulatory compliance.

        Args:
            lcr: Current LCR components.
            outflows: Additional stressed outflows by category.
            stress_multiplier: Multiplier for expected outflows under stress.

        Returns:
            Dictionary with LCR impact analysis.
        """
        # Basel III LCR outflow categories and run-off rates
        outflow_rates = {
            "retail_stable": 0.05,
            "retail_less_stable": 0.10,
            "unsecured_wholesale_operational": 0.25,
            "unsecured_wholesale_non_operational": 0.40,
            "secured_funding_l1": 0.00,
            "secured_funding_l2a": 0.15,
            "secured_funding_other": 1.00,
            "derivatives": 1.00,
            "committed_facilities": 0.05,
        }

        total_stressed_outflows = 0.0
        outflow_details: Dict[str, Dict[str, float]] = {}

        for category, amount in outflows.items():
            base_rate = outflow_rates.get(category, 0.50)
            stressed_rate = min(1.0, base_rate * stress_multiplier)
            stressed_outflow = amount * stressed_rate

            total_stressed_outflows += stressed_outflow
            outflow_details[category] = {
                "amount": amount,
                "base_rate": base_rate,
                "stressed_rate": stressed_rate,
                "stressed_outflow": stressed_outflow,
            }

        new_total_outflows = lcr.total_net_outflows_30d + total_stressed_outflows

        # HQLA haircuts under stress
        hqla_haircut = 0.0
        haircut_l2a = lcr.level2a_assets * 0.15  # 15% haircut
        haircut_l2b = lcr.level2b_assets * 0.50  # 50% haircut
        hqla_haircut = haircut_l2a + haircut_l2b
        stressed_hqla = lcr.high_quality_liquid_assets - hqla_haircut

        # Apply cap: L2 assets <= 40% of total HQLA
        l2_cap = stressed_hqla * 0.4
        l2_total = (lcr.level2a_assets - haircut_l2a) + (lcr.level2b_assets - haircut_l2b)
        if l2_total > l2_cap:
            stressed_hqla -= (l2_total - l2_cap)

        stressed_lcr = stressed_hqla / max(new_total_outflows, 1.0)
        lcr_change = stressed_lcr - lcr.lcr

        # Compliance check
        compliant = stressed_lcr >= 1.0
        buffer = stressed_hqla - new_total_outflows

        return {
            "baseline_lcr": lcr.lcr,
            "stressed_lcr": stressed_lcr,
            "lcr_change": lcr_change,
            "lcr_change_pct": lcr_change / max(lcr.lcr, 0.01) * 100,
            "compliant": compliant,
            "hqla_after_stress": stressed_hqla,
            "total_outflows_stressed": new_total_outflows,
            "liquidity_buffer": buffer,
            "hqla_haircut": hqla_haircut,
            "outflow_details": outflow_details,
            "additional_outflows": total_stressed_outflows,
        }

    def _compute_withdrawal(
        self,
        profile: FundingProfile,
        stress: float,
        current_funding: float,
    ) -> float:
        """Compute funding withdrawal for a single institution.

        Withdrawal increases with stress and is concentrated in volatile
        (short-term, unsecured) funding.
        """
        if stress <= 0 or current_funding <= 0:
            return 0.0

        # Withdrawal rate increases non-linearly with stress
        # Volatile funding runs faster than stable funding
        volatile_rate = min(1.0, stress * 0.5)
        stable_rate = min(1.0, stress * 0.05)

        volatile_fraction = profile.volatile_funding / max(profile.total_funding, 1.0)
        stable_fraction = 1 - volatile_fraction

        weighted_rate = volatile_rate * volatile_fraction + stable_rate * stable_fraction
        withdrawal = current_funding * weighted_rate * 0.1  # 10% per round

        return min(withdrawal, current_funding)

    def _generate_default_profiles(
        self,
        graph: nx.DiGraph,
        nodes: List[int],
    ) -> Dict[int, FundingProfile]:
        """Generate default funding profiles from graph attributes."""
        profiles: Dict[int, FundingProfile] = {}

        for nd in nodes:
            size = graph.nodes[nd].get("size", 1e9)
            inst_type = graph.nodes[nd].get("institution_type", "bank")

            # Different funding mixes by institution type
            if inst_type in ("bank", "BANK"):
                sources = {
                    FundingType.DEPOSITS: size * 0.40,
                    FundingType.UNSECURED_INTERBANK: size * 0.15,
                    FundingType.REPO: size * 0.20,
                    FundingType.COMMERCIAL_PAPER: size * 0.10,
                    FundingType.COVERED_BOND: size * 0.10,
                    FundingType.CERTIFICATE_OF_DEPOSIT: size * 0.05,
                }
                maturities = {
                    "overnight": size * 0.15,
                    "1w": size * 0.10,
                    "1m": size * 0.15,
                    "3m": size * 0.20,
                    "6m": size * 0.15,
                    "1y": size * 0.15,
                    "gt1y": size * 0.10,
                }
            elif inst_type in ("fund", "FUND"):
                sources = {
                    FundingType.REPO: size * 0.40,
                    FundingType.COMMERCIAL_PAPER: size * 0.30,
                    FundingType.UNSECURED_INTERBANK: size * 0.20,
                    FundingType.CERTIFICATE_OF_DEPOSIT: size * 0.10,
                }
                maturities = {
                    "overnight": size * 0.30,
                    "1w": size * 0.20,
                    "1m": size * 0.20,
                    "3m": size * 0.15,
                    "6m": size * 0.10,
                    "1y": size * 0.05,
                }
            else:
                sources = {
                    FundingType.DEPOSITS: size * 0.30,
                    FundingType.REPO: size * 0.25,
                    FundingType.UNSECURED_INTERBANK: size * 0.20,
                    FundingType.COVERED_BOND: size * 0.15,
                    FundingType.COMMERCIAL_PAPER: size * 0.10,
                }
                maturities = {
                    "overnight": size * 0.10,
                    "1w": size * 0.10,
                    "1m": size * 0.15,
                    "3m": size * 0.20,
                    "6m": size * 0.20,
                    "1y": size * 0.15,
                    "gt1y": size * 0.10,
                }

            profiles[nd] = FundingProfile(
                institution_id=nd,
                funding_sources=sources,
                maturity_buckets=maturities,
            )

        return profiles
