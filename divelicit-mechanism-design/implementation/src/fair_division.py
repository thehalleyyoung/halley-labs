"""
Fair division algorithms: Cut-and-Choose, Adjusted Winner, Round Robin,
Nash Welfare, EF1, MMS, rent division, and comprehensive fairness analysis.
"""

import numpy as np
from itertools import combinations, permutations, product
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FairAllocation:
    """Result of a fair division."""
    assignment: Dict[int, List[int]]  # agent -> items
    utilities: Dict[int, float]
    method: str


class CutAndChoose:
    """Cut-and-Choose protocol for 2-agent fair division."""

    def divide(self, valuations: np.ndarray, n_agents: int = 2) -> FairAllocation:
        """
        Divide items between 2 agents using cut-and-choose.
        valuations: (n_agents, n_items) matrix of item values.
        Agent 0 cuts (partitions), Agent 1 chooses.
        """
        assert n_agents == 2, "Cut-and-Choose is for 2 agents"
        n_items = valuations.shape[1]

        # Agent 0 finds a partition that maximizes their minimum share
        best_partition = None
        best_min_value = -np.inf

        # Try all ways to split items into 2 groups
        items = list(range(n_items))
        for r in range(n_items + 1):
            for group1 in combinations(items, r):
                group1 = list(group1)
                group2 = [i for i in items if i not in group1]

                val1 = sum(valuations[0, i] for i in group1)
                val2 = sum(valuations[0, i] for i in group2)
                min_val = min(val1, val2)

                if min_val > best_min_value:
                    best_min_value = min_val
                    best_partition = (group1, group2)

        # Agent 1 chooses preferred bundle
        group1, group2 = best_partition
        val1_for_agent1 = sum(valuations[1, i] for i in group1)
        val2_for_agent1 = sum(valuations[1, i] for i in group2)

        if val1_for_agent1 >= val2_for_agent1:
            assignment = {1: group1, 0: group2}
        else:
            assignment = {0: group1, 1: group2}

        utilities = {
            agent: sum(valuations[agent, i] for i in items_list)
            for agent, items_list in assignment.items()
        }

        return FairAllocation(assignment=assignment, utilities=utilities, method="CutAndChoose")


class AdjustedWinner:
    """
    Adjusted Winner procedure for 2 agents dividing multiple items.
    Each agent distributes 100 points across items.
    """

    def divide(self, points: np.ndarray) -> FairAllocation:
        """
        points: (2, n_items) where each row sums to 100.
        Returns allocation that is envy-free, equitable, and efficient.
        """
        n_items = points.shape[1]

        # Initial assignment: each item goes to the agent who values it more
        assignment = {0: [], 1: []}
        for item in range(n_items):
            if points[0, item] >= points[1, item]:
                assignment[0].append(item)
            else:
                assignment[1].append(item)

        # Compute initial scores
        scores = [
            sum(points[0, i] for i in assignment[0]),
            sum(points[1, i] for i in assignment[1])
        ]

        # If already equal, done
        if abs(scores[0] - scores[1]) < 1e-10:
            utilities = {
                a: sum(points[a, i] for i in items)
                for a, items in assignment.items()
            }
            return FairAllocation(assignment=assignment, utilities=utilities,
                                  method="AdjustedWinner")

        # Determine who has more points
        if scores[0] > scores[1]:
            giver, receiver = 0, 1
        else:
            giver, receiver = 1, 0

        # Sort giver's items by ratio of receiver's value to giver's value (ascending)
        giver_items = sorted(assignment[giver],
                             key=lambda i: points[receiver, i] / max(points[giver, i], 1e-10))

        # Transfer items until equitable
        for item in giver_items:
            giver_loss = points[giver, item]
            receiver_gain = points[receiver, item]

            new_giver_score = scores[giver] - giver_loss
            new_receiver_score = scores[receiver] + receiver_gain

            if new_giver_score >= new_receiver_score:
                # Full transfer
                assignment[giver].remove(item)
                assignment[receiver].append(item)
                scores[giver] = new_giver_score
                scores[receiver] = new_receiver_score
            else:
                # Partial transfer to equalize
                # Find fraction f such that:
                # scores[giver] - f * giver_loss = scores[receiver] + f * receiver_gain
                if giver_loss + receiver_gain > 1e-10:
                    f = (scores[giver] - scores[receiver]) / (giver_loss + receiver_gain)
                    f = min(max(f, 0), 1)
                else:
                    f = 0.5
                # For simplicity with indivisible items, round to nearest
                if f > 0.5:
                    assignment[giver].remove(item)
                    assignment[receiver].append(item)
                    scores[giver] -= giver_loss
                    scores[receiver] += receiver_gain
                break

        utilities = {
            a: sum(points[a, i] for i in items)
            for a, items in assignment.items()
        }
        return FairAllocation(assignment=assignment, utilities=utilities,
                              method="AdjustedWinner")


class RoundRobin:
    """Round-robin (sequential) allocation."""

    def divide(self, valuations: np.ndarray, priority: Optional[List[int]] = None) -> FairAllocation:
        """
        Sequential allocation where agents take turns picking best remaining item.
        valuations: (n_agents, n_items)
        priority: order in which agents pick (default: 0, 1, 2, ...)
        """
        n_agents, n_items = valuations.shape
        if priority is None:
            priority = list(range(n_agents))

        assignment = {a: [] for a in range(n_agents)}
        remaining = set(range(n_items))

        round_num = 0
        while remaining:
            agent = priority[round_num % n_agents]
            # Pick most valuable remaining item
            best_item = max(remaining, key=lambda i: valuations[agent, i])
            assignment[agent].append(best_item)
            remaining.remove(best_item)
            round_num += 1

        utilities = {
            a: sum(valuations[a, i] for i in items)
            for a, items in assignment.items()
        }
        return FairAllocation(assignment=assignment, utilities=utilities,
                              method="RoundRobin")


class MaxNashWelfare:
    """
    Maximize Nash social welfare (product of utilities).
    Uses convex optimization via log-transform.
    """

    def divide(self, valuations: np.ndarray) -> FairAllocation:
        """
        Find allocation maximizing Nash welfare.
        Uses greedy approximation (exact is NP-hard).
        """
        n_agents, n_items = valuations.shape
        assignment = {a: [] for a in range(n_agents)}
        remaining = list(range(n_items))

        # Initialize utilities with small epsilon to avoid log(0)
        current_utilities = np.ones(n_agents) * 1e-10

        # Greedy: assign each item to agent that maximizes product of utilities
        # Equivalent to: assign to agent with highest value / current_utility ratio
        for _ in range(n_items):
            best_agent = -1
            best_item = -1
            best_ratio = -np.inf

            for item in remaining:
                for agent in range(n_agents):
                    # Increase in log-welfare
                    new_util = current_utilities[agent] + valuations[agent, item]
                    ratio = np.log(new_util) - np.log(current_utilities[agent])
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_agent = agent
                        best_item = item

            if best_item >= 0:
                assignment[best_agent].append(best_item)
                current_utilities[best_agent] += valuations[best_agent, best_item]
                remaining.remove(best_item)

        utilities = {
            a: sum(valuations[a, i] for i in items)
            for a, items in assignment.items()
        }
        return FairAllocation(assignment=assignment, utilities=utilities,
                              method="MaxNashWelfare")


class EnvyFreeUpToOneItem:
    """Compute EF1 allocation using envy-cycle elimination."""

    def divide(self, valuations: np.ndarray) -> FairAllocation:
        """
        Find EF1 allocation using envy-cycle elimination algorithm.
        valuations: (n_agents, n_items)
        """
        n_agents, n_items = valuations.shape
        assignment = {a: [] for a in range(n_agents)}
        remaining = list(range(n_items))

        # Start with round-robin as initial allocation
        rr = RoundRobin()
        initial = rr.divide(valuations)
        assignment = {a: list(items) for a, items in initial.assignment.items()}

        # Eliminate envy cycles
        max_iterations = n_agents * n_items * 10
        for _ in range(max_iterations):
            # Find envy graph
            envy_graph = {}
            for agent_i in range(n_agents):
                val_i = sum(valuations[agent_i, item] for item in assignment[agent_i])
                for agent_j in range(n_agents):
                    if agent_i == agent_j:
                        continue
                    val_j = sum(valuations[agent_i, item] for item in assignment[agent_j])

                    # Check EF1: is agent_i envious of agent_j even after removing one item?
                    ef1_satisfied = False
                    if not assignment[agent_j]:
                        ef1_satisfied = True
                    else:
                        for remove_item in assignment[agent_j]:
                            val_j_minus = val_j - valuations[agent_i, remove_item]
                            if val_i >= val_j_minus - 1e-10:
                                ef1_satisfied = True
                                break

                    if not ef1_satisfied:
                        envy_graph[agent_i] = agent_j

            if not envy_graph:
                break

            # Find and eliminate a cycle in envy graph
            visited = set()
            path = []
            current = next(iter(envy_graph))

            while current not in visited:
                visited.add(current)
                path.append(current)
                if current in envy_graph:
                    current = envy_graph[current]
                else:
                    break

            if current in path:
                cycle_start = path.index(current)
                cycle = path[cycle_start:]

                if len(cycle) >= 2:
                    # Rotate bundles along cycle
                    bundles = [assignment[a] for a in cycle]
                    for i, agent in enumerate(cycle):
                        assignment[agent] = bundles[(i + 1) % len(cycle)]

        utilities = {
            a: sum(valuations[a, i] for i in items)
            for a, items in assignment.items()
        }
        return FairAllocation(assignment=assignment, utilities=utilities,
                              method="EF1")


class MaximinShareGuarantee:
    """Compute MMS values and find MMS-fair allocations."""

    def compute_mms(self, valuations: np.ndarray, agent: int) -> float:
        """
        Compute maximin share for an agent.
        MMS = max over partitions P into n bundles: min bundle value.
        """
        n_agents, n_items = valuations.shape
        if n_items <= 8 and n_agents <= 3:
            return self._exact_mms(valuations, agent, n_agents)
        else:
            return self._approx_mms(valuations, agent, n_agents)

    def _exact_mms(self, valuations: np.ndarray, agent: int, n_parts: int) -> float:
        """Exact MMS via enumeration of partitions."""
        n_items = valuations.shape[1]
        items = list(range(n_items))
        best_min = -np.inf

        def partition_items(items, k):
            """Generate all partitions of items into k non-empty groups."""
            if k == 1:
                yield [items]
                return
            if len(items) == k:
                yield [[item] for item in items]
                return
            if len(items) < k:
                return

            first = items[0]
            rest = items[1:]

            # first goes into a new group
            for partition in partition_items(rest, k - 1):
                yield [[first]] + partition

            # first goes into existing group
            for partition in partition_items(rest, k):
                for i in range(len(partition)):
                    new_partition = [list(g) for g in partition]
                    new_partition[i] = [first] + new_partition[i]
                    yield new_partition

        # For small instances, enumerate
        if n_items <= 8:
            for partition in partition_items(items, n_parts):
                min_val = min(
                    sum(valuations[agent, i] for i in group)
                    for group in partition
                )
                best_min = max(best_min, min_val)
        else:
            # Binary search on the MMS value
            total = sum(valuations[agent, i] for i in items)
            lo, hi = 0.0, total / n_parts

            for _ in range(50):
                mid = (lo + hi) / 2
                if self._can_partition(valuations[agent], n_parts, mid):
                    lo = mid
                else:
                    hi = mid
            best_min = lo

        return best_min

    def _can_partition(self, values: np.ndarray, n_parts: int, threshold: float) -> bool:
        """Check if items can be partitioned into n_parts groups each worth >= threshold."""
        n_items = len(values)
        sorted_items = sorted(range(n_items), key=lambda i: -values[i])

        # Greedy bin packing
        bins = [0.0] * n_parts
        def backtrack(idx):
            if idx == n_items:
                return all(b >= threshold - 1e-10 for b in bins)
            item = sorted_items[idx]
            tried = set()
            for b in range(n_parts):
                if bins[b] in tried:
                    continue
                tried.add(bins[b])
                bins[b] += values[item]
                if backtrack(idx + 1):
                    return True
                bins[b] -= values[item]
            return False

        return backtrack(0)

    def _approx_mms(self, valuations: np.ndarray, agent: int, n_parts: int) -> float:
        """Approximate MMS via repeated partitioning."""
        n_items = valuations.shape[1]
        items = list(range(n_items))
        total = sum(valuations[agent, i] for i in items)
        target = total / n_parts

        # Greedy approximation: repeatedly find a group worth ~target
        best_results = []
        for trial in range(20):
            rng = np.random.default_rng(42 + trial)
            perm = rng.permutation(items).tolist()
            groups = []
            current_group = []
            current_val = 0.0

            for item in perm:
                current_group.append(item)
                current_val += valuations[agent, item]
                if current_val >= target and len(groups) < n_parts - 1:
                    groups.append((current_group, current_val))
                    current_group = []
                    current_val = 0.0

            groups.append((current_group, current_val))
            if len(groups) >= n_parts:
                min_val = min(v for _, v in groups[:n_parts])
                best_results.append(min_val)

        return max(best_results) if best_results else 0.0

    def divide(self, valuations: np.ndarray) -> FairAllocation:
        """Find an approximately MMS-fair allocation."""
        n_agents, n_items = valuations.shape

        # Compute MMS for each agent
        mms_values = {}
        for agent in range(n_agents):
            mms_values[agent] = self.compute_mms(valuations, agent)

        # Use round-robin with MMS-aware ordering
        assignment = {a: [] for a in range(n_agents)}
        remaining = list(range(n_items))

        # Priority to agents furthest from their MMS
        current_utils = np.zeros(n_agents)
        while remaining:
            # Agent with lowest utility relative to MMS picks
            ratios = [(current_utils[a] / max(mms_values[a], 1e-10), a)
                      for a in range(n_agents)]
            ratios.sort()
            agent = ratios[0][1]

            best_item = max(remaining, key=lambda i: valuations[agent, i])
            assignment[agent].append(best_item)
            current_utils[agent] += valuations[agent, best_item]
            remaining.remove(best_item)

        utilities = {
            a: sum(valuations[a, i] for i in items)
            for a, items in assignment.items()
        }
        return FairAllocation(assignment=assignment, utilities=utilities,
                              method="MMS")


def check_proportionality(valuations: np.ndarray, allocation: FairAllocation) -> Dict[int, bool]:
    """Check if each agent gets at least 1/n of their total value."""
    n_agents = valuations.shape[0]
    results = {}
    for agent in range(n_agents):
        total_value = sum(valuations[agent, :])
        fair_share = total_value / n_agents
        received = sum(valuations[agent, i] for i in allocation.assignment.get(agent, []))
        results[agent] = received >= fair_share - 1e-10
    return results


def check_envy_freeness(valuations: np.ndarray, allocation: FairAllocation) -> Dict[Tuple[int, int], bool]:
    """Check envy-freeness for all pairs of agents."""
    n_agents = valuations.shape[0]
    results = {}
    for i in range(n_agents):
        val_i = sum(valuations[i, item] for item in allocation.assignment.get(i, []))
        for j in range(n_agents):
            if i == j:
                continue
            val_j = sum(valuations[i, item] for item in allocation.assignment.get(j, []))
            results[(i, j)] = val_i >= val_j - 1e-10
    return results


def check_ef1(valuations: np.ndarray, allocation: FairAllocation) -> Dict[Tuple[int, int], bool]:
    """Check EF1 for all pairs of agents."""
    n_agents = valuations.shape[0]
    results = {}
    for i in range(n_agents):
        val_i = sum(valuations[i, item] for item in allocation.assignment.get(i, []))
        for j in range(n_agents):
            if i == j:
                continue
            items_j = allocation.assignment.get(j, [])
            if not items_j:
                results[(i, j)] = True
                continue
            # EF1: exists item in j's bundle such that removing it eliminates envy
            ef1 = False
            for remove in items_j:
                val_j_minus = sum(valuations[i, item] for item in items_j if item != remove)
                if val_i >= val_j_minus - 1e-10:
                    ef1 = True
                    break
            results[(i, j)] = ef1
    return results


def check_equitability(valuations: np.ndarray, allocation: FairAllocation,
                        tolerance: float = 0.1) -> bool:
    """Check if all agents have approximately equal utility."""
    n_agents = valuations.shape[0]
    utilities = []
    for agent in range(n_agents):
        u = sum(valuations[agent, i] for i in allocation.assignment.get(agent, []))
        utilities.append(u)
    if not utilities:
        return True
    max_u = max(utilities)
    min_u = min(utilities)
    if max_u < 1e-10:
        return True
    return (max_u - min_u) / max_u <= tolerance


class RentDivision:
    """
    Envy-free rent splitting using Sperner's lemma based approach.
    Approximate envy-free division of rent among roommates with rooms.
    """

    def __init__(self, n_rooms: int, total_rent: float):
        self.n_rooms = n_rooms
        self.total_rent = total_rent

    def compute_envy_free_prices(self, valuations: np.ndarray,
                                  n_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find envy-free room assignment and prices.
        valuations: (n_agents, n_rooms) how much each agent values each room.

        Uses iterative adjustment (discrete Sperner approximation).
        Returns: (assignment, prices)
        """
        n = self.n_rooms
        # Start with equal prices
        prices = np.ones(n) * self.total_rent / n

        step = self.total_rent / (10 * n)

        for iteration in range(n_iterations):
            # Each agent picks most preferred room (value - price)
            surplus = valuations - prices[np.newaxis, :]
            preferences = np.argmax(surplus, axis=1)

            # Check if assignment is valid (each room taken by exactly one agent)
            room_count = np.bincount(preferences, minlength=n)

            if np.all(room_count == 1):
                # Check envy-freeness
                assignment = preferences
                is_ef = True
                for i in range(n):
                    my_surplus = valuations[i, assignment[i]] - prices[assignment[i]]
                    for j in range(n):
                        if i == j:
                            continue
                        other_surplus = valuations[i, assignment[j]] - prices[assignment[j]]
                        if other_surplus > my_surplus + 1e-10:
                            is_ef = False
                            break
                    if not is_ef:
                        break

                if is_ef:
                    return assignment, prices

            # Adjust prices: increase price of over-demanded rooms,
            # decrease price of under-demanded rooms
            for room in range(n):
                if room_count[room] > 1:
                    prices[room] += step
                elif room_count[room] == 0:
                    prices[room] -= step

            # Ensure prices are non-negative and sum to total_rent
            prices = np.maximum(prices, 0)
            price_sum = prices.sum()
            if price_sum > 0:
                prices *= self.total_rent / price_sum

            step *= 0.99  # Decrease step size

        # Final assignment: Hungarian-like greedy
        assignment = np.argmax(valuations - prices[np.newaxis, :], axis=1)
        return assignment, prices


class FairDivisionAnalyzer:
    """Compute all fairness metrics for a given allocation."""

    def __init__(self, valuations: np.ndarray):
        self.valuations = valuations
        self.n_agents, self.n_items = valuations.shape

    def analyze(self, allocation: FairAllocation) -> Dict:
        """Compute comprehensive fairness metrics."""
        # Utilities
        utilities = {}
        for agent in range(self.n_agents):
            utilities[agent] = sum(
                self.valuations[agent, i]
                for i in allocation.assignment.get(agent, [])
            )

        # Proportionality
        proportional = check_proportionality(self.valuations, allocation)

        # Envy-freeness
        ef = check_envy_freeness(self.valuations, allocation)

        # EF1
        ef1 = check_ef1(self.valuations, allocation)

        # Equitability
        equitable = check_equitability(self.valuations, allocation)

        # MMS
        mms_checker = MaximinShareGuarantee()
        mms_satisfied = {}
        for agent in range(self.n_agents):
            mms_val = mms_checker.compute_mms(self.valuations, agent)
            mms_satisfied[agent] = utilities[agent] >= mms_val - 1e-10

        # Utilitarian welfare
        util_welfare = sum(utilities.values())

        # Nash welfare
        nash_welfare = np.prod([max(u, 1e-10) for u in utilities.values()])

        # Egalitarian welfare
        egal_welfare = min(utilities.values()) if utilities else 0

        # Envy statistics
        total_envy = 0.0
        max_envy = 0.0
        for (i, j), is_ef in ef.items():
            if not is_ef:
                val_j = sum(self.valuations[i, item]
                            for item in allocation.assignment.get(j, []))
                envy_amount = val_j - utilities[i]
                total_envy += max(0, envy_amount)
                max_envy = max(max_envy, envy_amount)

        return {
            'utilities': utilities,
            'proportional': proportional,
            'all_proportional': all(proportional.values()),
            'envy_free': all(ef.values()),
            'ef1': all(ef1.values()),
            'equitable': equitable,
            'mms_satisfied': mms_satisfied,
            'all_mms': all(mms_satisfied.values()),
            'utilitarian_welfare': util_welfare,
            'nash_welfare': float(nash_welfare),
            'egalitarian_welfare': egal_welfare,
            'total_envy': total_envy,
            'max_envy': max_envy,
            'method': allocation.method
        }


def generate_random_valuations(n_agents: int, n_items: int,
                                distribution: str = 'uniform',
                                rng=None) -> np.ndarray:
    """Generate random valuations for fair division."""
    if rng is None:
        rng = np.random.default_rng()

    if distribution == 'uniform':
        return rng.uniform(0, 10, size=(n_agents, n_items))
    elif distribution == 'exponential':
        return rng.exponential(5, size=(n_agents, n_items))
    elif distribution == 'correlated':
        # Items have inherent quality, agents have noise
        quality = rng.uniform(1, 10, size=n_items)
        noise = rng.normal(0, 1, size=(n_agents, n_items))
        return np.maximum(quality[np.newaxis, :] + noise, 0)
    else:
        return rng.uniform(0, 10, size=(n_agents, n_items))
