"""
Complete mechanism design engine implementing VCG, Groves, Myerson, and related mechanisms.
"""

import numpy as np
from itertools import combinations, permutations, product
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field


@dataclass
class Allocation:
    """Represents an allocation of items to agents."""
    assignment: Dict[int, List[int]]  # agent_id -> list of item indices
    social_welfare: float = 0.0

    def items_for(self, agent_id: int) -> List[int]:
        return self.assignment.get(agent_id, [])


@dataclass
class PaymentResult:
    """Payments each agent must make."""
    payments: Dict[int, float]

    def total_revenue(self) -> float:
        return sum(self.payments.values())


class VCGMechanism:
    """
    Vickrey-Clarke-Groves mechanism with Clarke pivot rule.
    Implements the full VCG mechanism for combinatorial allocation.
    """

    def __init__(self, n_items: int):
        self.n_items = n_items

    def _enumerate_allocations(self, agent_ids: List[int]) -> List[Dict[int, List[int]]]:
        """Enumerate all possible allocations of items to agents."""
        items = list(range(self.n_items))
        allocations = []
        # Each item can go to any agent or remain unallocated (agent=-1)
        choices = agent_ids + [-1]
        for assignment_tuple in product(choices, repeat=self.n_items):
            alloc = {aid: [] for aid in agent_ids}
            for item_idx, assigned_to in enumerate(assignment_tuple):
                if assigned_to != -1:
                    alloc[assigned_to].append(item_idx)
            allocations.append(alloc)
        return allocations

    def _compute_welfare(self, allocation: Dict[int, List[int]],
                         valuations: Dict[int, Dict[tuple, float]]) -> float:
        """Compute total social welfare of an allocation."""
        total = 0.0
        for agent_id, items in allocation.items():
            bundle = tuple(sorted(items))
            total += valuations.get(agent_id, {}).get(bundle, 0.0)
        return total

    def allocate(self, valuations: Dict[int, Dict[tuple, float]]) -> Allocation:
        """
        Find welfare-maximizing allocation.
        valuations: {agent_id: {bundle_tuple: value}}
        """
        agent_ids = list(valuations.keys())
        best_alloc = None
        best_welfare = -np.inf

        all_allocations = self._enumerate_allocations(agent_ids)
        for alloc in all_allocations:
            w = self._compute_welfare(alloc, valuations)
            if w > best_welfare:
                best_welfare = w
                best_alloc = alloc

        return Allocation(assignment=best_alloc, social_welfare=best_welfare)

    def compute_payments(self, valuations: Dict[int, Dict[tuple, float]],
                         allocation: Allocation) -> PaymentResult:
        """
        Compute VCG payments using Clarke pivot rule.
        Payment_i = max welfare without i - welfare of others in current allocation.
        """
        agent_ids = list(valuations.keys())
        payments = {}

        for agent_i in agent_ids:
            # Welfare of others in current allocation
            others_welfare_with_i = 0.0
            for agent_j in agent_ids:
                if agent_j != agent_i:
                    bundle = tuple(sorted(allocation.items_for(agent_j)))
                    others_welfare_with_i += valuations[agent_j].get(bundle, 0.0)

            # Optimal welfare without agent i
            remaining_agents = [a for a in agent_ids if a != agent_i]
            remaining_valuations = {a: valuations[a] for a in remaining_agents}

            if remaining_agents:
                alloc_without_i = self._allocate_subset(remaining_valuations, remaining_agents)
                max_welfare_without_i = alloc_without_i.social_welfare
            else:
                max_welfare_without_i = 0.0

            # Clarke pivot payment
            payments[agent_i] = max_welfare_without_i - others_welfare_with_i

        return PaymentResult(payments=payments)

    def _allocate_subset(self, valuations: Dict[int, Dict[tuple, float]],
                         agent_ids: List[int]) -> Allocation:
        """Allocate items to a subset of agents."""
        best_alloc = None
        best_welfare = -np.inf

        all_allocations = self._enumerate_allocations(agent_ids)
        for alloc in all_allocations:
            w = self._compute_welfare(alloc, valuations)
            if w > best_welfare:
                best_welfare = w
                best_alloc = alloc

        return Allocation(assignment=best_alloc, social_welfare=best_welfare)

    def verify_strategyproofness(self, valuations: Dict[int, Dict[tuple, float]],
                                 agent_id: int,
                                 possible_reports: List[Dict[tuple, float]]) -> Tuple[bool, Optional[Dict]]:
        """
        Verify that truthful reporting is dominant strategy for agent_id.
        Tests all possible misreports.
        """
        true_valuation = valuations[agent_id]
        true_alloc = self.allocate(valuations)
        true_payments = self.compute_payments(valuations, true_alloc)
        true_bundle = tuple(sorted(true_alloc.items_for(agent_id)))
        true_utility = true_valuation.get(true_bundle, 0.0) - true_payments.payments[agent_id]

        for report in possible_reports:
            modified_vals = dict(valuations)
            modified_vals[agent_id] = report
            new_alloc = self.allocate(modified_vals)
            new_payments = self.compute_payments(modified_vals, new_alloc)
            new_bundle = tuple(sorted(new_alloc.items_for(agent_id)))
            # Utility under true valuation but misreported allocation
            new_utility = true_valuation.get(new_bundle, 0.0) - new_payments.payments[agent_id]

            if new_utility > true_utility + 1e-10:
                return False, {
                    'misreport': report,
                    'true_utility': true_utility,
                    'misreport_utility': new_utility
                }

        return True, None


class GrovesScheme:
    """
    Generic Groves mechanism with configurable h-functions.
    Payment_i = h_i(v_{-i}) - sum_{j != i} v_j(f(v))
    where f is the efficient allocation rule.
    """

    def __init__(self, n_items: int, h_functions: Optional[Dict[int, Callable]] = None):
        self.n_items = n_items
        self.vcg = VCGMechanism(n_items)
        self.h_functions = h_functions or {}

    def set_h_function(self, agent_id: int, h_func: Callable):
        """Set the redistribution function for an agent."""
        self.h_functions[agent_id] = h_func

    def compute_payments(self, valuations: Dict[int, Dict[tuple, float]],
                         allocation: Allocation) -> PaymentResult:
        """Compute Groves payments with custom h-functions."""
        agent_ids = list(valuations.keys())
        payments = {}

        for agent_i in agent_ids:
            # Sum of others' values in current allocation
            others_value = 0.0
            for agent_j in agent_ids:
                if agent_j != agent_i:
                    bundle = tuple(sorted(allocation.items_for(agent_j)))
                    others_value += valuations[agent_j].get(bundle, 0.0)

            # h_i function of others' reports
            others_reports = {a: valuations[a] for a in agent_ids if a != agent_i}
            if agent_i in self.h_functions:
                h_value = self.h_functions[agent_i](others_reports)
            else:
                # Default: Clarke pivot (VCG)
                remaining = [a for a in agent_ids if a != agent_i]
                if remaining:
                    alloc_without = self.vcg._allocate_subset(
                        {a: valuations[a] for a in remaining}, remaining)
                    h_value = alloc_without.social_welfare
                else:
                    h_value = 0.0

            payments[agent_i] = h_value - others_value

        return PaymentResult(payments=payments)

    def allocate(self, valuations: Dict[int, Dict[tuple, float]]) -> Allocation:
        """Use efficient allocation (same as VCG)."""
        return self.vcg.allocate(valuations)


class ExpectedExternality:
    """
    Bayesian optimal mechanism for known prior distributions.
    Computes expected externality payments when type distributions are known.
    """

    def __init__(self, n_agents: int, n_items: int):
        self.n_agents = n_agents
        self.n_items = n_items

    def compute_expected_externality(self, agent_id: int,
                                     type_distributions: Dict[int, np.ndarray],
                                     allocation_rule: Callable,
                                     n_samples: int = 10000) -> float:
        """
        Compute expected externality of agent_id on others via Monte Carlo.
        type_distributions: {agent_id: array of possible type values}
        allocation_rule: function(types) -> allocation
        """
        rng = np.random.default_rng(42)
        agent_ids = list(type_distributions.keys())

        total_externality = 0.0
        for _ in range(n_samples):
            # Sample types for all agents except agent_id
            sampled_types = {}
            for aid in agent_ids:
                idx = rng.integers(0, len(type_distributions[aid]))
                sampled_types[aid] = type_distributions[aid][idx]

            # Allocation with agent_id
            alloc_with = allocation_rule(sampled_types)

            # Welfare of others with agent_id present
            others_welfare_with = sum(
                sampled_types[a] * alloc_with.get(a, 0)
                for a in agent_ids if a != agent_id
            )

            # Allocation without agent_id
            types_without = {a: sampled_types[a] for a in agent_ids if a != agent_id}
            alloc_without = allocation_rule(types_without)

            others_welfare_without = sum(
                types_without[a] * alloc_without.get(a, 0)
                for a in agent_ids if a != agent_id
            )

            total_externality += others_welfare_without - others_welfare_with

        return total_externality / n_samples

    def compute_bayesian_optimal_payments(self, types: Dict[int, float],
                                          type_distributions: Dict[int, np.ndarray],
                                          allocation_rule: Callable,
                                          n_samples: int = 5000) -> Dict[int, float]:
        """
        Compute expected externality payments for all agents.
        """
        payments = {}
        for agent_id in types:
            payments[agent_id] = self.compute_expected_externality(
                agent_id, type_distributions, allocation_rule, n_samples
            )
        return payments


class MyersonAuction:
    """
    Optimal single-item auction (Myerson 1981).
    Computes virtual values, handles ironing for irregular distributions.
    """

    def __init__(self, distributions: Dict[int, Tuple[str, dict]]):
        """
        distributions: {agent_id: (dist_type, params)}
        dist_type: 'uniform', 'exponential', 'normal_truncated'
        """
        self.distributions = distributions
        self.agent_ids = list(distributions.keys())
        self._ironed_virtual = {}

    def _cdf(self, agent_id: int, v: float) -> float:
        dist_type, params = self.distributions[agent_id]
        if dist_type == 'uniform':
            a, b = params['low'], params['high']
            if v <= a:
                return 0.0
            elif v >= b:
                return 1.0
            return (v - a) / (b - a)
        elif dist_type == 'exponential':
            lam = params['lambda']
            return 1.0 - np.exp(-lam * max(0, v))
        elif dist_type == 'normal_truncated':
            mu, sigma = params['mu'], params['sigma']
            lo, hi = params.get('low', mu - 4 * sigma), params.get('high', mu + 4 * sigma)
            from scipy.stats import norm
            if v <= lo:
                return 0.0
            if v >= hi:
                return 1.0
            return (norm.cdf(v, mu, sigma) - norm.cdf(lo, mu, sigma)) / \
                   (norm.cdf(hi, mu, sigma) - norm.cdf(lo, mu, sigma))
        return 0.0

    def _pdf(self, agent_id: int, v: float) -> float:
        dist_type, params = self.distributions[agent_id]
        if dist_type == 'uniform':
            a, b = params['low'], params['high']
            if a <= v <= b:
                return 1.0 / (b - a)
            return 0.0
        elif dist_type == 'exponential':
            lam = params['lambda']
            if v < 0:
                return 0.0
            return lam * np.exp(-lam * v)
        elif dist_type == 'normal_truncated':
            mu, sigma = params['mu'], params['sigma']
            lo, hi = params.get('low', mu - 4 * sigma), params.get('high', mu + 4 * sigma)
            from scipy.stats import norm
            if v < lo or v > hi:
                return 0.0
            denom = norm.cdf(hi, mu, sigma) - norm.cdf(lo, mu, sigma)
            if denom < 1e-15:
                return 0.0
            return norm.pdf(v, mu, sigma) / denom
        return 0.0

    def virtual_value(self, agent_id: int, v: float) -> float:
        """Compute virtual value: phi(v) = v - (1 - F(v)) / f(v)."""
        f_v = self._pdf(agent_id, v)
        if f_v < 1e-15:
            return v
        F_v = self._cdf(agent_id, v)
        return v - (1.0 - F_v) / f_v

    def _compute_ironed_virtual_values(self, agent_id: int, n_points: int = 1000) -> Callable:
        """
        Iron virtual values for irregular distributions.
        Uses convex hull approach on the integrated virtual value function.
        """
        dist_type, params = self.distributions[agent_id]
        if dist_type == 'uniform':
            lo, hi = params['low'], params['high']
        elif dist_type == 'exponential':
            lo, hi = 0.0, params.get('high', 10.0 / params['lambda'])
        elif dist_type == 'normal_truncated':
            mu, sigma = params['mu'], params['sigma']
            lo = params.get('low', mu - 4 * sigma)
            hi = params.get('high', mu + 4 * sigma)
        else:
            lo, hi = 0.0, 1.0

        quantiles = np.linspace(0.001, 0.999, n_points)
        values = np.zeros(n_points)
        virtual_vals = np.zeros(n_points)

        # Map quantiles to values via inverse CDF (bisection)
        for i, q in enumerate(quantiles):
            v_lo, v_hi = lo, hi
            for _ in range(100):
                v_mid = (v_lo + v_hi) / 2
                if self._cdf(agent_id, v_mid) < q:
                    v_lo = v_mid
                else:
                    v_hi = v_mid
            values[i] = (v_lo + v_hi) / 2
            virtual_vals[i] = self.virtual_value(agent_id, values[i])

        # Compute integrated virtual value H(q) = integral_0^q phi(F^{-1}(s)) ds
        H = np.zeros(n_points + 1)
        for i in range(n_points):
            dq = quantiles[i] - (quantiles[i - 1] if i > 0 else 0.0)
            H[i + 1] = H[i] + virtual_vals[i] * dq

        # Convex hull (ironing): compute greatest convex minorant of H
        # Using a simple left-to-right sweep
        hull_H = np.copy(H)
        hull_indices = [0]
        for i in range(1, len(H)):
            while len(hull_indices) >= 2:
                j = hull_indices[-1]
                k = hull_indices[-2]
                # Check if slope from k to i >= slope from k to j
                slope_ki = (H[i] - H[k]) / (i - k) if i != k else 0
                slope_kj = (H[j] - H[k]) / (j - k) if j != k else 0
                if slope_ki <= slope_kj:
                    hull_indices.pop()
                else:
                    break
            hull_indices.append(i)

        # Reconstruct ironed virtual values from hull slopes
        ironed = np.zeros(n_points)
        hull_set = set(hull_indices)
        hi_idx = 0
        for i in range(n_points):
            # Find the hull segment containing quantile i+1
            while hi_idx < len(hull_indices) - 1 and hull_indices[hi_idx + 1] <= i + 1:
                hi_idx += 1
            if hi_idx < len(hull_indices) - 1:
                k = hull_indices[hi_idx]
                j = hull_indices[hi_idx + 1]
                if j != k:
                    ironed[i] = (H[j] - H[k]) / (j - k) * n_points
                else:
                    ironed[i] = virtual_vals[i]
            else:
                ironed[i] = virtual_vals[i]

        # Check if ironing was needed (virtual values already monotone)
        is_monotone = all(virtual_vals[i] <= virtual_vals[i + 1] + 1e-10
                          for i in range(len(virtual_vals) - 1))
        if is_monotone:
            ironed = virtual_vals

        def ironed_virtual(v):
            idx = np.searchsorted(values, v)
            idx = min(max(idx, 0), n_points - 1)
            return ironed[idx]

        return ironed_virtual

    def get_ironed_virtual_value(self, agent_id: int, v: float) -> float:
        """Get ironed virtual value for agent at value v."""
        if agent_id not in self._ironed_virtual:
            self._ironed_virtual[agent_id] = self._compute_ironed_virtual_values(agent_id)
        return self._ironed_virtual[agent_id](v)

    def optimal_auction(self, bids: Dict[int, float]) -> Tuple[Optional[int], Dict[int, float]]:
        """
        Run optimal auction: allocate to highest non-negative virtual value bidder.
        Returns (winner, payments).
        """
        virtual_vals = {}
        for agent_id, bid in bids.items():
            virtual_vals[agent_id] = self.get_ironed_virtual_value(agent_id, bid)

        # Find winner: highest non-negative virtual value
        winner = None
        best_virtual = 0.0
        for agent_id, vv in virtual_vals.items():
            if vv > best_virtual:
                best_virtual = vv
                winner = agent_id

        payments = {agent_id: 0.0 for agent_id in bids}
        if winner is not None:
            # Payment = infimum of bids at which winner still wins
            # Binary search for the threshold
            v_lo = 0.0
            v_hi = bids[winner]
            for _ in range(200):
                v_mid = (v_lo + v_hi) / 2
                vv_mid = self.get_ironed_virtual_value(winner, v_mid)

                # Check if winner still wins at bid v_mid
                still_wins = True
                for other_id, other_vv in virtual_vals.items():
                    if other_id != winner and other_vv >= vv_mid:
                        still_wins = False
                        break
                if vv_mid < 0:
                    still_wins = False

                if still_wins:
                    v_hi = v_mid
                else:
                    v_lo = v_mid

            payments[winner] = v_hi

        return winner, payments

    def expected_revenue(self, n_samples: int = 50000) -> float:
        """Compute expected revenue of optimal auction via Monte Carlo."""
        rng = np.random.default_rng(42)
        total_revenue = 0.0

        for _ in range(n_samples):
            bids = {}
            for agent_id in self.agent_ids:
                dist_type, params = self.distributions[agent_id]
                if dist_type == 'uniform':
                    bids[agent_id] = rng.uniform(params['low'], params['high'])
                elif dist_type == 'exponential':
                    bids[agent_id] = rng.exponential(1.0 / params['lambda'])
                elif dist_type == 'normal_truncated':
                    mu, sigma = params['mu'], params['sigma']
                    lo = params.get('low', mu - 4 * sigma)
                    hi = params.get('high', mu + 4 * sigma)
                    while True:
                        v = rng.normal(mu, sigma)
                        if lo <= v <= hi:
                            bids[agent_id] = v
                            break

            winner, payments = self.optimal_auction(bids)
            total_revenue += sum(payments.values())

        return total_revenue / n_samples


class BudgetBalance:
    """Analyze budget balance properties of mechanisms."""

    @staticmethod
    def check_weak_budget_balance(payments: PaymentResult) -> bool:
        """Check if total payments >= 0 (no deficit)."""
        return payments.total_revenue() >= -1e-10

    @staticmethod
    def check_strong_budget_balance(payments: PaymentResult) -> bool:
        """Check if total payments == 0."""
        return abs(payments.total_revenue()) < 1e-10

    @staticmethod
    def compute_worst_case_deficit(mechanism, valuation_profiles: List[Dict[int, Dict[tuple, float]]]) -> float:
        """Compute worst-case deficit across valuation profiles."""
        worst_deficit = 0.0
        for valuations in valuation_profiles:
            alloc = mechanism.allocate(valuations)
            payments = mechanism.compute_payments(valuations, alloc)
            revenue = payments.total_revenue()
            if revenue < worst_deficit:
                worst_deficit = revenue
        return worst_deficit

    @staticmethod
    def green_laffont_impossibility(n_agents: int, n_items: int) -> str:
        """
        Green-Laffont theorem: no Groves mechanism is budget balanced
        in unrestricted quasi-linear domains with 3+ agents.
        """
        if n_agents >= 3:
            return ("Green-Laffont impossibility applies: no Groves mechanism can be "
                    "simultaneously efficient, strategy-proof, and strongly budget balanced "
                    f"with {n_agents} agents and unrestricted preferences.")
        elif n_agents == 2:
            return ("With 2 agents, budget balance may be achievable depending on the domain. "
                    "The AGV (expected externality) mechanism achieves ex-ante budget balance.")
        return "Single agent: budget balance is trivially achievable."

    @staticmethod
    def compute_vcg_redistribution(payments: PaymentResult, n_agents: int) -> Dict[int, float]:
        """
        Compute VCG redistribution using Bailey-Cavallo mechanism.
        Redistribute (n-1)/n of the minimum VCG payment to all agents equally.
        """
        agent_ids = list(payments.payments.keys())
        if not agent_ids:
            return {}

        min_payment = min(payments.payments.values())
        # Redistribute fraction of surplus
        redistribution_amount = max(0, min_payment) * (n_agents - 1) / n_agents
        per_agent = redistribution_amount / n_agents if n_agents > 0 else 0

        adjusted = {}
        for aid in agent_ids:
            adjusted[aid] = payments.payments[aid] - per_agent

        return adjusted


def verify_strategy_proofness(mechanism, type_space: List[Dict[int, Dict[tuple, float]]],
                               agents: List[int]) -> Tuple[bool, Optional[Dict]]:
    """
    Verify strategy-proofness by brute force over type space.
    For each agent, check that truthful reporting is optimal against all type profiles.
    """
    for true_profile in type_space:
        for agent_id in agents:
            true_alloc = mechanism.allocate(true_profile)
            true_payments = mechanism.compute_payments(true_profile, true_alloc)
            true_bundle = tuple(sorted(true_alloc.items_for(agent_id)))
            true_utility = true_profile[agent_id].get(true_bundle, 0.0) - \
                           true_payments.payments[agent_id]

            # Try all possible misreports from type space
            for alt_profile in type_space:
                if alt_profile[agent_id] is true_profile[agent_id]:
                    continue
                # Agent reports alt type, others report truthfully
                misreport_profile = dict(true_profile)
                misreport_profile[agent_id] = alt_profile[agent_id]

                mis_alloc = mechanism.allocate(misreport_profile)
                mis_payments = mechanism.compute_payments(misreport_profile, mis_alloc)
                mis_bundle = tuple(sorted(mis_alloc.items_for(agent_id)))
                # Utility under TRUE valuation
                mis_utility = true_profile[agent_id].get(mis_bundle, 0.0) - \
                              mis_payments.payments[agent_id]

                if mis_utility > true_utility + 1e-10:
                    return False, {
                        'agent': agent_id,
                        'true_type': true_profile[agent_id],
                        'misreport': alt_profile[agent_id],
                        'true_utility': true_utility,
                        'misreport_utility': mis_utility
                    }

    return True, None


def check_individual_rationality(mechanism, valuations: Dict[int, Dict[tuple, float]],
                                  outside_option: float = 0.0) -> Dict[int, bool]:
    """
    Check if mechanism satisfies individual rationality for each agent.
    IR: utility from participation >= outside_option.
    """
    alloc = mechanism.allocate(valuations)
    payments = mechanism.compute_payments(valuations, alloc)

    ir_results = {}
    for agent_id in valuations:
        bundle = tuple(sorted(alloc.items_for(agent_id)))
        utility = valuations[agent_id].get(bundle, 0.0) - payments.payments[agent_id]
        ir_results[agent_id] = utility >= outside_option - 1e-10

    return ir_results


class RevenueBound:
    """Compute theoretical revenue bounds for mechanisms."""

    @staticmethod
    def second_price_expected_revenue(n_bidders: int, dist_type: str = 'uniform',
                                       params: Optional[dict] = None) -> float:
        """
        Expected revenue of second-price auction with n i.i.d. bidders.
        For uniform[0,1]: E[revenue] = (n-1)/(n+1)
        """
        if dist_type == 'uniform':
            lo = params.get('low', 0.0) if params else 0.0
            hi = params.get('high', 1.0) if params else 1.0
            # E[2nd order statistic from Uniform[a,b]] = a + (b-a)*(n-1)/(n+1)
            return lo + (hi - lo) * (n_bidders - 1) / (n_bidders + 1)
        elif dist_type == 'exponential':
            lam = params.get('lambda', 1.0) if params else 1.0
            # E[2nd order stat] = (H_n - 1)/lambda where H_n = harmonic number
            h_n = sum(1.0 / k for k in range(1, n_bidders + 1))
            return (h_n - 1.0) / lam
        return 0.0

    @staticmethod
    def myerson_optimal_reserve(dist_type: str = 'uniform',
                                 params: Optional[dict] = None) -> float:
        """
        Compute optimal reserve price for a single bidder.
        For uniform[0,1]: reserve = 0.5
        """
        if dist_type == 'uniform':
            lo = params.get('low', 0.0) if params else 0.0
            hi = params.get('high', 1.0) if params else 1.0
            return (lo + hi) / 2.0
        elif dist_type == 'exponential':
            lam = params.get('lambda', 1.0) if params else 1.0
            return 1.0 / lam
        return 0.0


class MechanismComparator:
    """Compare different mechanisms on efficiency and revenue."""

    def __init__(self, mechanisms: Dict[str, Any]):
        self.mechanisms = mechanisms

    def compare_efficiency(self, valuation_profiles: List[Dict[int, Dict[tuple, float]]]) -> Dict[str, float]:
        """Compare mechanisms by average social welfare."""
        results = {}
        for name, mechanism in self.mechanisms.items():
            total_welfare = 0.0
            for valuations in valuation_profiles:
                alloc = mechanism.allocate(valuations)
                total_welfare += alloc.social_welfare
            results[name] = total_welfare / len(valuation_profiles)
        return results

    def compare_revenue(self, valuation_profiles: List[Dict[int, Dict[tuple, float]]]) -> Dict[str, float]:
        """Compare mechanisms by average revenue."""
        results = {}
        for name, mechanism in self.mechanisms.items():
            total_revenue = 0.0
            for valuations in valuation_profiles:
                alloc = mechanism.allocate(valuations)
                payments = mechanism.compute_payments(valuations, alloc)
                total_revenue += payments.total_revenue()
            results[name] = total_revenue / len(valuation_profiles)
        return results

    def compare_ir_satisfaction(self, valuation_profiles: List[Dict[int, Dict[tuple, float]]]) -> Dict[str, float]:
        """Compare mechanisms by fraction of IR-satisfying instances."""
        results = {}
        for name, mechanism in self.mechanisms.items():
            ir_count = 0
            total = 0
            for valuations in valuation_profiles:
                ir_results = check_individual_rationality(mechanism, valuations)
                ir_count += sum(1 for v in ir_results.values() if v)
                total += len(ir_results)
            results[name] = ir_count / max(total, 1)
        return results


def generate_random_valuations(n_agents: int, n_items: int, rng=None,
                                additive: bool = True) -> Dict[int, Dict[tuple, float]]:
    """Generate random valuations for testing."""
    if rng is None:
        rng = np.random.default_rng()

    valuations = {}
    for agent_id in range(n_agents):
        val = {}
        if additive:
            # Additive valuations: value of bundle = sum of item values
            item_values = rng.uniform(0, 10, size=n_items)
            for r in range(n_items + 1):
                for bundle in combinations(range(n_items), r):
                    val[bundle] = sum(item_values[i] for i in bundle)
        else:
            # General valuations
            for r in range(n_items + 1):
                for bundle in combinations(range(n_items), r):
                    if r == 0:
                        val[bundle] = 0.0
                    else:
                        val[bundle] = rng.uniform(0, 10 * r)
        valuations[agent_id] = val

    return valuations


class OnlineMechanism:
    """Online mechanism design for sequential arrivals."""

    def __init__(self, n_items: int, n_rounds: int):
        self.n_items = n_items
        self.n_rounds = n_rounds
        self.available_items = list(range(n_items))
        self.allocations = {}
        self.payments = {}
        self.round = 0

    def process_arrival(self, agent_id: int, valuations: Dict[tuple, float]) -> Tuple[tuple, float]:
        """
        Process a single agent arrival. Greedy allocation with posted prices.
        """
        if not self.available_items:
            self.allocations[agent_id] = ()
            self.payments[agent_id] = 0.0
            return (), 0.0

        # Find best available bundle
        best_bundle = ()
        best_surplus = 0.0

        for r in range(1, len(self.available_items) + 1):
            for bundle in combinations(self.available_items, r):
                val = valuations.get(bundle, 0.0)
                # Simple posted price: fraction of remaining rounds
                price = val * (self.n_rounds - self.round) / (2 * self.n_rounds)
                surplus = val - price
                if surplus > best_surplus:
                    best_surplus = surplus
                    best_bundle = bundle
                    best_price = price

        if best_bundle:
            for item in best_bundle:
                self.available_items.remove(item)
            self.allocations[agent_id] = best_bundle
            self.payments[agent_id] = best_price
            self.round += 1
            return best_bundle, best_price
        else:
            self.allocations[agent_id] = ()
            self.payments[agent_id] = 0.0
            self.round += 1
            return (), 0.0

    def get_competitive_ratio(self, offline_welfare: float) -> float:
        """Compute competitive ratio vs offline optimal."""
        online_welfare = sum(
            self.payments.get(aid, 0.0)  # Use surplus as proxy
            for aid in self.allocations
        )
        if offline_welfare < 1e-10:
            return 1.0
        return online_welfare / offline_welfare


class AutomatedMechanismDesign:
    """
    Automated mechanism design: search for optimal mechanism
    given objectives and constraints.
    """

    def __init__(self, n_agents: int, n_outcomes: int, type_space_sizes: List[int]):
        self.n_agents = n_agents
        self.n_outcomes = n_outcomes
        self.type_space_sizes = type_space_sizes

    def solve_revenue_maximization(self, type_priors: List[np.ndarray],
                                    utilities: np.ndarray,
                                    ic_constraints: bool = True,
                                    ir_constraints: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for revenue-maximizing mechanism via LP relaxation.
        type_priors[i]: probability distribution over agent i's types
        utilities[i, t, o]: utility of agent i with type t for outcome o
        Returns: (allocation_rule, payment_rule)
        """
        from scipy.optimize import linprog

        # Build type profiles
        type_profiles = list(product(*[range(s) for s in self.type_space_sizes]))
        n_profiles = len(type_profiles)

        # Decision variables: p(o | type_profile) for each outcome and profile
        n_vars = n_profiles * self.n_outcomes + n_profiles * self.n_agents
        # First n_profiles * n_outcomes vars: allocation probabilities
        # Next n_profiles * n_agents vars: payments

        # Objective: maximize expected revenue = maximize sum of payments
        c = np.zeros(n_vars)
        n_alloc_vars = n_profiles * self.n_outcomes
        for prof_idx, profile in enumerate(type_profiles):
            prob = np.prod([type_priors[i][profile[i]] for i in range(self.n_agents)])
            for i in range(self.n_agents):
                pay_idx = n_alloc_vars + prof_idx * self.n_agents + i
                c[pay_idx] = -prob  # Negate for minimization

        # Constraints
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []

        # Probability simplex: sum of allocation probs = 1 for each profile
        for prof_idx in range(n_profiles):
            row = np.zeros(n_vars)
            for o in range(self.n_outcomes):
                row[prof_idx * self.n_outcomes + o] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)

        if ic_constraints:
            # IC: for each agent i, type t, misreport t'
            for i in range(self.n_agents):
                for t in range(self.type_space_sizes[i]):
                    for t_prime in range(self.type_space_sizes[i]):
                        if t == t_prime:
                            continue
                        # E[u_i(t, o) - p_i | t] >= E[u_i(t, o) - p_i | t']
                        row = np.zeros(n_vars)
                        # Average over others' types
                        for prof_idx, profile in enumerate(type_profiles):
                            if profile[i] != t:
                                continue
                            prob_others = np.prod([
                                type_priors[j][profile[j]]
                                for j in range(self.n_agents) if j != i
                            ])
                            for o in range(self.n_outcomes):
                                row[prof_idx * self.n_outcomes + o] += \
                                    prob_others * utilities[i, t, o]
                            row[n_alloc_vars + prof_idx * self.n_agents + i] -= prob_others

                        # Misreport t' -> find matching profiles
                        for prof_idx, profile in enumerate(type_profiles):
                            if profile[i] != t_prime:
                                continue
                            # Others have same types but agent i reported t'
                            others_match = tuple(profile[j] for j in range(self.n_agents) if j != i)
                            prob_others = np.prod([
                                type_priors[j][profile[j]]
                                for j in range(self.n_agents) if j != i
                            ])
                            for o in range(self.n_outcomes):
                                row[prof_idx * self.n_outcomes + o] -= \
                                    prob_others * utilities[i, t, o]
                            row[n_alloc_vars + prof_idx * self.n_agents + i] += prob_others

                        A_ub.append(-row)  # >= becomes <= with negation
                        b_ub.append(0.0)

        if ir_constraints:
            # IR: E[u_i(t, o) - p_i | t] >= 0
            for i in range(self.n_agents):
                for t in range(self.type_space_sizes[i]):
                    row = np.zeros(n_vars)
                    for prof_idx, profile in enumerate(type_profiles):
                        if profile[i] != t:
                            continue
                        prob_others = np.prod([
                            type_priors[j][profile[j]]
                            for j in range(self.n_agents) if j != i
                        ])
                        for o in range(self.n_outcomes):
                            row[prof_idx * self.n_outcomes + o] += \
                                prob_others * utilities[i, t, o]
                        row[n_alloc_vars + prof_idx * self.n_agents + i] -= prob_others

                    A_ub.append(-row)
                    b_ub.append(0.0)

        # Bounds: allocation probs in [0,1], payments unbounded below
        bounds = [(0, 1)] * n_alloc_vars + [(None, None)] * (n_profiles * self.n_agents)

        A_ub_arr = np.array(A_ub) if A_ub else None
        b_ub_arr = np.array(b_ub) if b_ub else None
        A_eq_arr = np.array(A_eq) if A_eq else None
        b_eq_arr = np.array(b_eq) if b_eq else None

        result = linprog(c, A_ub=A_ub_arr, b_ub=b_ub_arr,
                         A_eq=A_eq_arr, b_eq=b_eq_arr, bounds=bounds,
                         method='highs')

        if result.success:
            alloc_vars = result.x[:n_alloc_vars].reshape(n_profiles, self.n_outcomes)
            pay_vars = result.x[n_alloc_vars:].reshape(n_profiles, self.n_agents)
            return alloc_vars, pay_vars
        else:
            return np.zeros((n_profiles, self.n_outcomes)), np.zeros((n_profiles, self.n_agents))
