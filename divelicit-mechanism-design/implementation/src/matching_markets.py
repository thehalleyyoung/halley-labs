"""
Two-sided matching markets: Gale-Shapley, Top Trading Cycles,
Stable Roommates, School Choice, and market simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Matching:
    """Result of a matching."""
    pairs: Dict[int, int]  # proposer -> acceptor (or agent -> agent)
    stable: bool
    method: str

    def partner(self, agent: int) -> Optional[int]:
        if agent in self.pairs:
            return self.pairs[agent]
        for k, v in self.pairs.items():
            if v == agent:
                return k
        return None


class GaleShapley:
    """Gale-Shapley deferred acceptance algorithm."""

    def propose_dispose(self, proposer_prefs: Dict[int, List[int]],
                        acceptor_prefs: Dict[int, List[int]]) -> Matching:
        """
        Run Gale-Shapley deferred acceptance.
        proposer_prefs: {proposer_id: [ranked acceptors]}
        acceptor_prefs: {acceptor_id: [ranked proposers]}
        Returns proposer-optimal stable matching.
        """
        # Build acceptor ranking lookup
        acceptor_rank = {}
        for acceptor, prefs in acceptor_prefs.items():
            acceptor_rank[acceptor] = {p: i for i, p in enumerate(prefs)}

        # Initialize
        free_proposers = list(proposer_prefs.keys())
        proposals = {p: 0 for p in proposer_prefs}  # Next to propose to
        current_match = {}  # acceptor -> proposer
        proposer_match = {}  # proposer -> acceptor

        while free_proposers:
            proposer = free_proposers.pop(0)
            prefs = proposer_prefs[proposer]

            if proposals[proposer] >= len(prefs):
                continue  # Exhausted all options

            acceptor = prefs[proposals[proposer]]
            proposals[proposer] += 1

            if acceptor not in current_match:
                # Acceptor is free
                current_match[acceptor] = proposer
                proposer_match[proposer] = acceptor
            else:
                current_holder = current_match[acceptor]
                rankings = acceptor_rank.get(acceptor, {})

                proposer_rank = rankings.get(proposer, float('inf'))
                holder_rank = rankings.get(current_holder, float('inf'))

                if proposer_rank < holder_rank:
                    # Acceptor prefers new proposer
                    current_match[acceptor] = proposer
                    proposer_match[proposer] = acceptor
                    del proposer_match[current_holder]
                    free_proposers.append(current_holder)
                else:
                    # Rejected, try next
                    free_proposers.append(proposer)

        return Matching(pairs=proposer_match, stable=True, method="GaleShapley")


class TopTradingCycles:
    """Top Trading Cycles algorithm for house allocation."""

    def find_allocation(self, initial_endowment: Dict[int, int],
                        preferences: Dict[int, List[int]]) -> Dict[int, int]:
        """
        TTC algorithm.
        initial_endowment: {agent: house_they_own}
        preferences: {agent: [houses in preference order]}
        Returns: {agent: house_they_get}
        """
        remaining_agents = set(initial_endowment.keys())
        remaining_houses = set(initial_endowment.values())
        owner_of = {h: a for a, h in initial_endowment.items()}
        allocation = {}

        while remaining_agents:
            # Build directed graph: each agent points to owner of their top remaining house
            graph = {}
            for agent in remaining_agents:
                for house in preferences[agent]:
                    if house in remaining_houses:
                        graph[agent] = owner_of[house]
                        break

            # Find cycles
            visited = set()
            in_cycle = set()

            for start in remaining_agents:
                if start in visited:
                    continue
                path = []
                current = start
                while current not in visited and current in graph:
                    visited.add(current)
                    path.append(current)
                    current = graph[current]

                if current in path:
                    # Found a cycle
                    cycle_start = path.index(current)
                    cycle = path[cycle_start:]

                    # Execute trades in cycle
                    for agent in cycle:
                        # Agent gets the house they're pointing to
                        target_agent = graph[agent]
                        allocation[agent] = initial_endowment[target_agent]
                        in_cycle.add(agent)

            # Remove cycle agents
            for agent in in_cycle:
                house = initial_endowment[agent]
                remaining_agents.discard(agent)
                remaining_houses.discard(house)
                if house in owner_of:
                    del owner_of[house]

            if not in_cycle:
                # No cycles found, remaining agents keep their houses
                for agent in list(remaining_agents):
                    allocation[agent] = initial_endowment[agent]
                break

        return allocation


class StableRoommates:
    """Irving's algorithm for the stable roommate problem."""

    def find_stable_matching(self, preferences: Dict[int, List[int]]) -> Optional[Matching]:
        """
        Irving's algorithm for stable roommate problem.
        preferences: {agent: [ranked other agents]}
        Returns stable matching or None if no stable matching exists.
        """
        agents = list(preferences.keys())
        n = len(agents)
        if n % 2 != 0:
            return None

        # Phase 1: Proposal (similar to GS)
        prefs = {a: list(p) for a, p in preferences.items()}
        proposals = {a: 0 for a in agents}
        held_by = {}  # agent -> who is holding them
        holding = {}  # agent -> who they're holding

        free = list(agents)
        while free:
            proposer = free.pop(0)
            if proposals[proposer] >= len(prefs[proposer]):
                continue

            target = prefs[proposer][proposals[proposer]]
            proposals[proposer] += 1

            if target not in holding:
                holding[target] = proposer
                held_by[proposer] = target
            else:
                current = holding[target]
                target_prefs = prefs[target]
                if target_prefs.index(proposer) < target_prefs.index(current):
                    holding[target] = proposer
                    held_by[proposer] = target
                    del held_by[current]
                    free.append(current)
                    # Remove all after proposer in target's list
                    idx = target_prefs.index(proposer)
                    to_remove = target_prefs[idx + 1:]
                    prefs[target] = target_prefs[:idx + 1]
                    for r in to_remove:
                        if target in prefs.get(r, []):
                            prefs[r].remove(target)
                else:
                    free.append(proposer)
                    # Remove target from proposer's list
                    if target in prefs[proposer]:
                        prefs[proposer].remove(target)
                    if proposer in prefs.get(target, []):
                        prefs[target].remove(proposer)

        # Phase 2: Find and eliminate rotations
        def find_rotation():
            for a in agents:
                if len(prefs.get(a, [])) > 1:
                    break
            else:
                return None

            p = [a]
            q = [prefs[a][1] if len(prefs[a]) > 1 else None]
            if q[0] is None:
                return None

            visited = {a}
            while True:
                last_q = q[-1]
                if not prefs.get(last_q, []):
                    return None
                next_p = prefs[last_q][-1]
                if next_p in visited:
                    idx = p.index(next_p)
                    return list(zip(p[idx:], q[idx:]))
                visited.add(next_p)
                p.append(next_p)
                next_q = prefs[next_p][1] if len(prefs.get(next_p, [])) > 1 else None
                if next_q is None:
                    return None
                q.append(next_q)

        max_iter = n * n
        for _ in range(max_iter):
            # Check if done
            all_single = all(len(prefs.get(a, [])) <= 1 for a in agents)
            if all_single:
                break

            rotation = find_rotation()
            if rotation is None:
                break

            # Eliminate rotation
            for pi, qi in rotation:
                # Remove qi from pi's list and vice versa
                if qi in prefs.get(pi, []):
                    idx = prefs[pi].index(qi)
                    to_remove = prefs[pi][idx:]
                    prefs[pi] = prefs[pi][:idx]
                    for r in to_remove:
                        if pi in prefs.get(r, []):
                            prefs[r].remove(pi)

        # Build matching
        pairs = {}
        matched = set()
        for a in agents:
            if a in matched:
                continue
            if prefs.get(a, []):
                partner = prefs[a][0]
                if partner not in matched:
                    pairs[a] = partner
                    matched.add(a)
                    matched.add(partner)

        if len(pairs) * 2 < n:
            return None

        return Matching(pairs=pairs, stable=True, method="StableRoommates")


class SchoolChoice:
    """School choice mechanisms."""

    def boston_mechanism(self, student_prefs: Dict[int, List[int]],
                        school_capacities: Dict[int, int],
                        school_priorities: Dict[int, List[int]]) -> Dict[int, int]:
        """
        Boston mechanism (immediate acceptance).
        Not strategy-proof but commonly used.
        """
        assignment = {}
        remaining_students = set(student_prefs.keys())
        remaining_capacity = dict(school_capacities)

        n_schools = len(school_capacities)
        for round_num in range(n_schools):
            # Each student applies to their (round_num+1)-th choice
            applications = defaultdict(list)
            for student in remaining_students:
                prefs = student_prefs[student]
                if round_num < len(prefs):
                    school = prefs[round_num]
                    if remaining_capacity.get(school, 0) > 0:
                        applications[school].append(student)

            # Schools accept in priority order up to capacity
            for school, applicants in applications.items():
                priorities = school_priorities.get(school, [])
                priority_map = {s: i for i, s in enumerate(priorities)}

                sorted_applicants = sorted(applicants,
                                           key=lambda s: priority_map.get(s, float('inf')))

                for student in sorted_applicants:
                    if remaining_capacity[school] > 0:
                        assignment[student] = school
                        remaining_students.discard(student)
                        remaining_capacity[school] -= 1

        return assignment

    def serial_dictatorship(self, student_prefs: Dict[int, List[int]],
                            school_capacities: Dict[int, int],
                            priority_order: List[int]) -> Dict[int, int]:
        """Serial dictatorship: students pick in order of priority."""
        assignment = {}
        remaining_capacity = dict(school_capacities)

        for student in priority_order:
            if student not in student_prefs:
                continue
            for school in student_prefs[student]:
                if remaining_capacity.get(school, 0) > 0:
                    assignment[student] = school
                    remaining_capacity[school] -= 1
                    break

        return assignment

    def deferred_acceptance(self, student_prefs: Dict[int, List[int]],
                            school_capacities: Dict[int, int],
                            school_priorities: Dict[int, List[int]]) -> Dict[int, int]:
        """
        Student-proposing deferred acceptance (student-optimal stable matching).
        Strategy-proof for students.
        """
        priority_rank = {}
        for school, priorities in school_priorities.items():
            priority_rank[school] = {s: i for i, s in enumerate(priorities)}

        proposals = {s: 0 for s in student_prefs}
        tentative = defaultdict(list)  # school -> [students]
        free_students = list(student_prefs.keys())

        while free_students:
            student = free_students.pop(0)
            prefs = student_prefs[student]

            if proposals[student] >= len(prefs):
                continue

            school = prefs[proposals[student]]
            proposals[student] += 1

            tentative[school].append(student)

            # If over capacity, reject lowest priority
            capacity = school_capacities.get(school, 1)
            if len(tentative[school]) > capacity:
                ranks = priority_rank.get(school, {})
                tentative[school].sort(key=lambda s: ranks.get(s, float('inf')))
                rejected = tentative[school][capacity:]
                tentative[school] = tentative[school][:capacity]
                free_students.extend(rejected)

        assignment = {}
        for school, students in tentative.items():
            for student in students:
                assignment[student] = school

        return assignment


def find_blocking_pairs(matching: Matching,
                        prefs_a: Dict[int, List[int]],
                        prefs_b: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """
    Find blocking pairs in a two-sided matching.
    A pair (a, b) blocks if both prefer each other to current partner.
    """
    blocking = []

    # Build reverse matching
    reverse = {}
    for a, b in matching.pairs.items():
        reverse[b] = a

    rank_a = {a: {p: i for i, p in enumerate(prefs)} for a, prefs in prefs_a.items()}
    rank_b = {b: {p: i for i, p in enumerate(prefs)} for b, prefs in prefs_b.items()}

    for a in prefs_a:
        current_b = matching.pairs.get(a)
        a_current_rank = rank_a[a].get(current_b, float('inf')) if current_b is not None else float('inf')

        for b in prefs_b:
            if b == current_b:
                continue
            current_a = reverse.get(b)
            b_current_rank = rank_b[b].get(current_a, float('inf')) if current_a is not None else float('inf')

            a_prefers_b = rank_a[a].get(b, float('inf')) < a_current_rank
            b_prefers_a = rank_b[b].get(a, float('inf')) < b_current_rank

            if a_prefers_b and b_prefers_a:
                blocking.append((a, b))

    return blocking


def check_matching_strategy_proofness(mechanism_func, agent_id: int,
                                       true_prefs: Dict[int, List[int]],
                                       other_side_prefs: Dict[int, List[int]],
                                       all_alternatives: List[int]) -> Tuple[bool, Optional[Dict]]:
    """
    Check if truthful reporting is optimal for agent_id.
    Try all possible preference misreports.
    """
    from itertools import permutations

    true_result = mechanism_func(true_prefs, other_side_prefs)
    true_partner = true_result.pairs.get(agent_id)
    true_rank = true_prefs[agent_id].index(true_partner) if true_partner in true_prefs[agent_id] else len(true_prefs[agent_id])

    for misreport in permutations(all_alternatives):
        misreport = list(misreport)
        if misreport == true_prefs[agent_id]:
            continue

        modified_prefs = dict(true_prefs)
        modified_prefs[agent_id] = misreport

        mis_result = mechanism_func(modified_prefs, other_side_prefs)
        mis_partner = mis_result.pairs.get(agent_id)

        if mis_partner is None:
            continue

        mis_rank = true_prefs[agent_id].index(mis_partner) if mis_partner in true_prefs[agent_id] else len(true_prefs[agent_id])

        if mis_rank < true_rank:
            return False, {
                'agent': agent_id,
                'true_pref': true_prefs[agent_id],
                'misreport': misreport,
                'true_partner': true_partner,
                'new_partner': mis_partner,
                'improvement': true_rank - mis_rank
            }

    return True, None


class MatchingMarketSimulator:
    """Generate random preference profiles and run experiments."""

    def __init__(self, n_proposers: int, n_acceptors: int, seed: int = 42):
        self.n_proposers = n_proposers
        self.n_acceptors = n_acceptors
        self.rng = np.random.default_rng(seed)

    def generate_uniform_preferences(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """Generate uniformly random preferences."""
        proposers = list(range(self.n_proposers))
        acceptors = list(range(self.n_proposers, self.n_proposers + self.n_acceptors))

        prop_prefs = {}
        for p in proposers:
            prop_prefs[p] = list(self.rng.permutation(acceptors))

        acc_prefs = {}
        for a in acceptors:
            acc_prefs[a] = list(self.rng.permutation(proposers))

        return prop_prefs, acc_prefs

    def generate_correlated_preferences(self, correlation: float = 0.5) -> Tuple[Dict, Dict]:
        """Generate preferences with correlation (common desirability)."""
        proposers = list(range(self.n_proposers))
        acceptors = list(range(self.n_proposers, self.n_proposers + self.n_acceptors))

        # Acceptor quality
        acceptor_quality = self.rng.uniform(0, 1, size=self.n_acceptors)
        prop_prefs = {}
        for p in proposers:
            noise = self.rng.uniform(0, 1, size=self.n_acceptors)
            scores = correlation * acceptor_quality + (1 - correlation) * noise
            prop_prefs[p] = [acceptors[i] for i in np.argsort(-scores)]

        proposer_quality = self.rng.uniform(0, 1, size=self.n_proposers)
        acc_prefs = {}
        for a in acceptors:
            noise = self.rng.uniform(0, 1, size=self.n_proposers)
            scores = correlation * proposer_quality + (1 - correlation) * noise
            acc_prefs[a] = [proposers[i] for i in np.argsort(-scores)]

        return prop_prefs, acc_prefs

    def run_stability_experiment(self, n_trials: int = 100) -> Dict:
        """Run experiments measuring stability and average rank."""
        gs = GaleShapley()
        results = {
            'all_stable': True,
            'avg_proposer_rank': 0.0,
            'avg_acceptor_rank': 0.0,
            'n_trials': n_trials
        }

        total_p_rank = 0.0
        total_a_rank = 0.0
        count = 0

        for trial in range(n_trials):
            self.rng = np.random.default_rng(42 + trial)
            prop_prefs, acc_prefs = self.generate_uniform_preferences()
            matching = gs.propose_dispose(prop_prefs, acc_prefs)

            # Check stability
            blocking = find_blocking_pairs(matching, prop_prefs, acc_prefs)
            if blocking:
                results['all_stable'] = False

            # Average rank of partner
            for p, a in matching.pairs.items():
                p_rank = prop_prefs[p].index(a) if a in prop_prefs[p] else self.n_acceptors
                a_rank = acc_prefs[a].index(p) if p in acc_prefs[a] else self.n_proposers
                total_p_rank += p_rank
                total_a_rank += a_rank
                count += 1

        if count > 0:
            results['avg_proposer_rank'] = total_p_rank / count
            results['avg_acceptor_rank'] = total_a_rank / count

        return results
