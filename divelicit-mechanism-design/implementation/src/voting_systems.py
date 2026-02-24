"""
Complete voting theory implementation with all major voting systems,
impossibility theorem demonstrations, and proportional representation methods.
"""

import numpy as np
from itertools import permutations, combinations
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict


@dataclass
class ElectionResult:
    """Result of an election."""
    winner: int
    ranking: List[int]
    scores: Dict[int, float]
    explanation: str


class PluralityVoting:
    """First-past-the-post voting."""

    def elect(self, ballots: List[List[int]]) -> int:
        """Each voter's top choice gets one vote. Most votes wins."""
        counts = Counter(b[0] for b in ballots if b)
        if not counts:
            return -1
        return counts.most_common(1)[0][0]

    def rank(self, ballots: List[List[int]]) -> List[int]:
        """Full ranking by plurality scores."""
        counts = Counter(b[0] for b in ballots if b)
        candidates = set()
        for b in ballots:
            candidates.update(b)
        for c in candidates:
            if c not in counts:
                counts[c] = 0
        return [c for c, _ in counts.most_common()]

    def explain(self, ballots: List[List[int]]) -> str:
        counts = Counter(b[0] for b in ballots if b)
        lines = ["Plurality Voting Results:"]
        for c, v in counts.most_common():
            lines.append(f"  Candidate {c}: {v} first-place votes")
        winner = self.elect(ballots)
        lines.append(f"Winner: Candidate {winner}")
        return "\n".join(lines)


class BordaCount:
    """Borda count voting system."""

    def _compute_scores(self, ballots: List[List[int]]) -> Dict[int, float]:
        candidates = set()
        for b in ballots:
            candidates.update(b)
        n = len(candidates)
        scores = {c: 0.0 for c in candidates}
        for ballot in ballots:
            for rank_pos, candidate in enumerate(ballot):
                scores[candidate] += (n - 1 - rank_pos)
        return scores

    def elect(self, ballots: List[List[int]]) -> int:
        scores = self._compute_scores(ballots)
        if not scores:
            return -1
        return max(scores, key=scores.get)

    def rank(self, ballots: List[List[int]]) -> List[int]:
        scores = self._compute_scores(ballots)
        return sorted(scores, key=scores.get, reverse=True)

    def explain(self, ballots: List[List[int]]) -> str:
        scores = self._compute_scores(ballots)
        lines = ["Borda Count Results:"]
        for c in sorted(scores, key=scores.get, reverse=True):
            lines.append(f"  Candidate {c}: {scores[c]:.1f} points")
        lines.append(f"Winner: Candidate {self.elect(ballots)}")
        return "\n".join(lines)


class InstantRunoffVoting:
    """Instant-runoff voting (ranked choice voting)."""

    def elect(self, ballots: List[List[int]]) -> int:
        candidates = set()
        for b in ballots:
            candidates.update(b)
        active = set(candidates)
        current_ballots = [list(b) for b in ballots]
        rounds_info = []

        while len(active) > 1:
            # Count first-place votes among active candidates
            counts = Counter()
            for ballot in current_ballots:
                for c in ballot:
                    if c in active:
                        counts[c] += 1
                        break

            # Check for majority
            total = sum(counts.values())
            for c, v in counts.most_common():
                if v > total / 2:
                    return c

            # Eliminate candidate with fewest votes
            if not counts:
                break
            min_votes = min(counts.values())
            eliminated = [c for c, v in counts.items() if v == min_votes]
            # Break ties by eliminating the one with lowest Borda score
            if len(eliminated) > 1:
                borda = BordaCount()._compute_scores(current_ballots)
                eliminated.sort(key=lambda c: borda.get(c, 0))
            active.discard(eliminated[0])

        if active:
            return active.pop()
        return -1

    def rank(self, ballots: List[List[int]]) -> List[int]:
        """Rank by order of elimination (last eliminated = highest rank)."""
        candidates = set()
        for b in ballots:
            candidates.update(b)
        active = set(candidates)
        current_ballots = [list(b) for b in ballots]
        elimination_order = []

        while len(active) > 1:
            counts = Counter()
            for ballot in current_ballots:
                for c in ballot:
                    if c in active:
                        counts[c] += 1
                        break

            if not counts:
                break

            total = sum(counts.values())
            majority_winner = None
            for c, v in counts.most_common():
                if v > total / 2:
                    majority_winner = c
                    break

            if majority_winner:
                remaining = sorted(active - {majority_winner},
                                   key=lambda c: counts.get(c, 0))
                elimination_order.extend(remaining)
                elimination_order.append(majority_winner)
                break

            min_votes = min(counts.values())
            eliminated = [c for c, v in counts.items() if v == min_votes]
            eliminated.sort()
            active.discard(eliminated[0])
            elimination_order.append(eliminated[0])

        if active and not elimination_order:
            elimination_order.extend(active)

        # Remaining active candidates
        for c in active:
            if c not in elimination_order:
                elimination_order.append(c)

        return list(reversed(elimination_order))

    def explain(self, ballots: List[List[int]]) -> str:
        candidates = set()
        for b in ballots:
            candidates.update(b)
        active = set(candidates)
        current_ballots = [list(b) for b in ballots]
        lines = ["Instant Runoff Voting:"]
        round_num = 1

        while len(active) > 1:
            counts = Counter()
            for ballot in current_ballots:
                for c in ballot:
                    if c in active:
                        counts[c] += 1
                        break
            total = sum(counts.values())
            lines.append(f"  Round {round_num}: {dict(counts)}")

            for c, v in counts.most_common():
                if v > total / 2:
                    lines.append(f"Winner: Candidate {c} with majority")
                    return "\n".join(lines)

            if not counts:
                break
            min_votes = min(counts.values())
            eliminated = min((c for c, v in counts.items() if v == min_votes))
            lines.append(f"  Eliminated: Candidate {eliminated}")
            active.discard(eliminated)
            round_num += 1

        if active:
            winner = active.pop()
            lines.append(f"Winner: Candidate {winner}")
        return "\n".join(lines)


def _pairwise_matrix(ballots: List[List[int]], candidates: List[int]) -> np.ndarray:
    """Compute pairwise comparison matrix. M[i][j] = # voters preferring i to j."""
    n = len(candidates)
    cand_idx = {c: i for i, c in enumerate(candidates)}
    matrix = np.zeros((n, n), dtype=int)
    for ballot in ballots:
        for i, ci in enumerate(ballot):
            for cj in ballot[i + 1:]:
                if ci in cand_idx and cj in cand_idx:
                    matrix[cand_idx[ci]][cand_idx[cj]] += 1
    return matrix


class CopelandMethod:
    """Copeland's pairwise comparison method."""

    def _compute_scores(self, ballots: List[List[int]]) -> Dict[int, float]:
        candidates = sorted(set(c for b in ballots for c in b))
        n = len(candidates)
        matrix = _pairwise_matrix(ballots, candidates)
        scores = {}
        for i, ci in enumerate(candidates):
            s = 0.0
            for j in range(n):
                if i == j:
                    continue
                if matrix[i][j] > matrix[j][i]:
                    s += 1.0
                elif matrix[i][j] == matrix[j][i]:
                    s += 0.5
            scores[ci] = s
        return scores

    def elect(self, ballots: List[List[int]]) -> int:
        scores = self._compute_scores(ballots)
        return max(scores, key=scores.get) if scores else -1

    def rank(self, ballots: List[List[int]]) -> List[int]:
        scores = self._compute_scores(ballots)
        return sorted(scores, key=scores.get, reverse=True)

    def explain(self, ballots: List[List[int]]) -> str:
        scores = self._compute_scores(ballots)
        lines = ["Copeland Method Results:"]
        for c in sorted(scores, key=scores.get, reverse=True):
            lines.append(f"  Candidate {c}: {scores[c]:.1f} Copeland score")
        lines.append(f"Winner: Candidate {self.elect(ballots)}")
        return "\n".join(lines)


class SchulzeMethod:
    """Schulze method (beatpath)."""

    def _strongest_paths(self, ballots: List[List[int]]) -> Tuple[np.ndarray, List[int]]:
        candidates = sorted(set(c for b in ballots for c in b))
        n = len(candidates)
        d = _pairwise_matrix(ballots, candidates)

        # Floyd-Warshall for widest paths
        p = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if d[i][j] > d[j][i]:
                        p[i][j] = d[i][j]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j and i != k and j != k:
                        p[i][j] = max(p[i][j], min(p[i][k], p[k][j]))

        return p, candidates

    def elect(self, ballots: List[List[int]]) -> int:
        p, candidates = self._strongest_paths(ballots)
        n = len(candidates)
        # Winner: candidate i where p[i][j] >= p[j][i] for all j
        for i in range(n):
            is_winner = True
            for j in range(n):
                if i != j and p[i][j] < p[j][i]:
                    is_winner = False
                    break
            if is_winner:
                return candidates[i]
        return candidates[0] if candidates else -1

    def rank(self, ballots: List[List[int]]) -> List[int]:
        p, candidates = self._strongest_paths(ballots)
        n = len(candidates)
        # Count how many candidates each beats
        wins = {}
        for i in range(n):
            w = sum(1 for j in range(n) if i != j and p[i][j] > p[j][i])
            wins[candidates[i]] = w
        return sorted(wins, key=wins.get, reverse=True)

    def explain(self, ballots: List[List[int]]) -> str:
        p, candidates = self._strongest_paths(ballots)
        lines = ["Schulze Method (Beatpath):"]
        n = len(candidates)
        for i in range(n):
            for j in range(n):
                if i != j:
                    lines.append(f"  Path {candidates[i]}->{candidates[j]}: strength {p[i][j]}")
        lines.append(f"Winner: Candidate {self.elect(ballots)}")
        return "\n".join(lines)


class KemenyYoung:
    """Kemeny-Young method: find ranking minimizing Kendall tau distance."""

    def _kendall_score(self, ranking: List[int], ballots: List[List[int]],
                       candidates: List[int]) -> int:
        """Count total pairwise agreements between ranking and ballots."""
        cand_rank = {c: i for i, c in enumerate(ranking)}
        score = 0
        for ballot in ballots:
            ballot_rank = {c: i for i, c in enumerate(ballot)}
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    ci, cj = candidates[i], candidates[j]
                    if ci in cand_rank and cj in cand_rank and ci in ballot_rank and cj in ballot_rank:
                        if (cand_rank[ci] < cand_rank[cj]) == (ballot_rank[ci] < ballot_rank[cj]):
                            score += 1
        return score

    def rank(self, ballots: List[List[int]]) -> List[int]:
        candidates = sorted(set(c for b in ballots for c in b))
        if len(candidates) <= 8:
            # Exact: enumerate all permutations
            best_ranking = None
            best_score = -1
            for perm in permutations(candidates):
                score = self._kendall_score(list(perm), ballots, candidates)
                if score > best_score:
                    best_score = score
                    best_ranking = list(perm)
            return best_ranking
        else:
            # Heuristic: start with Copeland ranking, local search
            copeland = CopelandMethod()
            current = copeland.rank(ballots)
            current_score = self._kendall_score(current, ballots, candidates)
            improved = True
            while improved:
                improved = False
                for i in range(len(current) - 1):
                    for j in range(i + 1, len(current)):
                        candidate = list(current)
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        score = self._kendall_score(candidate, ballots, candidates)
                        if score > current_score:
                            current = candidate
                            current_score = score
                            improved = True
            return current

    def elect(self, ballots: List[List[int]]) -> int:
        ranking = self.rank(ballots)
        return ranking[0] if ranking else -1

    def explain(self, ballots: List[List[int]]) -> str:
        ranking = self.rank(ballots)
        return f"Kemeny-Young optimal ranking: {ranking}\nWinner: Candidate {ranking[0]}"


def find_condorcet_winner(ballots: List[List[int]]) -> Optional[int]:
    """Find Condorcet winner if one exists."""
    candidates = sorted(set(c for b in ballots for c in b))
    n = len(candidates)
    matrix = _pairwise_matrix(ballots, candidates)
    n_voters = len(ballots)

    for i in range(n):
        is_condorcet = True
        for j in range(n):
            if i != j and matrix[i][j] <= n_voters / 2:
                is_condorcet = False
                break
        if is_condorcet:
            return candidates[i]
    return None


def find_condorcet_cycle(ballots: List[List[int]]) -> Optional[List[int]]:
    """Find a Condorcet cycle if one exists."""
    candidates = sorted(set(c for b in ballots for c in b))
    n = len(candidates)
    matrix = _pairwise_matrix(ballots, candidates)

    # Build directed graph of pairwise defeats
    defeats = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > matrix[j][i]:
                defeats[candidates[i]].add(candidates[j])

    # DFS for cycle
    def find_cycle_dfs():
        visited = set()
        path = []
        path_set = set()

        def dfs(node):
            visited.add(node)
            path.append(node)
            path_set.add(node)
            for neighbor in defeats.get(node, []):
                if neighbor in path_set:
                    # Found cycle
                    idx = path.index(neighbor)
                    return path[idx:] + [neighbor]
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
            path.pop()
            path_set.discard(node)
            return None

        for c in candidates:
            if c not in visited:
                result = dfs(c)
                if result:
                    return result
        return None

    return find_cycle_dfs()


def demonstrate_arrow_impossibility(n_candidates: int = 3, n_voters: int = 3) -> Dict:
    """
    Demonstrate Arrow's impossibility theorem.
    For 3+ candidates, find a profile where some reasonable SWF violates
    at least one of: unanimity, IIA, or non-dictatorship.
    """
    # Classic Condorcet paradox profile
    if n_candidates >= 3:
        # 3 voters with cyclic preferences
        ballots = [
            list(range(n_candidates)),  # A > B > C
            list(range(1, n_candidates)) + [0],  # B > C > A
            [n_candidates - 1] + list(range(n_candidates - 1)),  # C > A > B
        ]
        # Pad with more voters if needed
        while len(ballots) < n_voters:
            ballots.append(ballots[len(ballots) % 3])

        methods = {
            'Plurality': PluralityVoting(),
            'Borda': BordaCount(),
            'Copeland': CopelandMethod(),
        }

        violations = {}
        for name, method in methods.items():
            ranking = method.rank(ballots)

            # Check IIA violation: remove a candidate and see if relative order changes
            for removed in range(n_candidates):
                reduced_ballots = [
                    [c for c in b if c != removed] for b in ballots
                ]
                reduced_ranking = method.rank(reduced_ballots)
                # Check if pairwise ordering of remaining candidates is preserved
                original_order = [c for c in ranking if c != removed]
                if original_order != reduced_ranking:
                    violations[name] = {
                        'type': 'IIA violation',
                        'removed_candidate': removed,
                        'original_order': original_order,
                        'new_order': reduced_ranking,
                    }
                    break

        return {
            'profile': ballots,
            'condorcet_winner': find_condorcet_winner(ballots),
            'condorcet_cycle': find_condorcet_cycle(ballots),
            'violations': violations,
            'theorem': ("Arrow's impossibility: No ranked voting system for 3+ candidates "
                        "can simultaneously satisfy unanimity (Pareto), independence of "
                        "irrelevant alternatives (IIA), and non-dictatorship.")
        }
    return {'theorem': 'Arrow\'s theorem requires 3+ candidates.'}


def find_gibbard_satterthwaite_manipulation(method, ballots: List[List[int]],
                                             candidates: List[int]) -> Optional[Dict]:
    """
    Find a strategic manipulation for a given voting method.
    Returns a manipulable profile if found.
    """
    true_winner = method.elect(ballots)

    for voter_idx in range(len(ballots)):
        true_pref = ballots[voter_idx]
        # Try all possible misreports
        for misreport in permutations(candidates):
            misreport = list(misreport)
            if misreport == true_pref:
                continue
            modified = list(ballots)
            modified[voter_idx] = misreport

            new_winner = method.elect(modified)
            if new_winner != true_winner:
                # Check if voter prefers new winner
                true_rank_of_new = true_pref.index(new_winner) if new_winner in true_pref else len(true_pref)
                true_rank_of_old = true_pref.index(true_winner) if true_winner in true_pref else len(true_pref)
                if true_rank_of_new < true_rank_of_old:
                    return {
                        'voter': voter_idx,
                        'true_preference': true_pref,
                        'misreport': misreport,
                        'true_winner': true_winner,
                        'new_winner': new_winner,
                        'explanation': (f"Voter {voter_idx} can change winner from "
                                        f"{true_winner} to {new_winner} by misreporting")
                    }
    return None


# Proportional representation methods

class DHondt:
    """D'Hondt (Jefferson) method for proportional seat allocation."""

    def allocate_seats(self, votes: Dict[int, int], n_seats: int,
                       threshold: float = 0.0) -> Dict[int, int]:
        """
        Allocate seats proportionally using D'Hondt method.
        votes: {party_id: vote_count}
        """
        total_votes = sum(votes.values())
        # Apply threshold
        eligible = {p: v for p, v in votes.items()
                    if v / total_votes >= threshold}
        if not eligible:
            return {p: 0 for p in votes}

        seats = {p: 0 for p in eligible}
        for _ in range(n_seats):
            # Find party with highest quotient
            quotients = {p: eligible[p] / (seats[p] + 1) for p in eligible}
            winner = max(quotients, key=quotients.get)
            seats[winner] += 1

        # Add zero seats for ineligible parties
        for p in votes:
            if p not in seats:
                seats[p] = 0
        return seats


class SainteLague:
    """Sainte-Laguë (Webster) method for proportional seat allocation."""

    def allocate_seats(self, votes: Dict[int, int], n_seats: int,
                       threshold: float = 0.0) -> Dict[int, int]:
        total_votes = sum(votes.values())
        eligible = {p: v for p, v in votes.items()
                    if v / total_votes >= threshold}
        if not eligible:
            return {p: 0 for p in votes}

        seats = {p: 0 for p in eligible}
        for _ in range(n_seats):
            quotients = {p: eligible[p] / (2 * seats[p] + 1) for p in eligible}
            winner = max(quotients, key=quotients.get)
            seats[winner] += 1

        for p in votes:
            if p not in seats:
                seats[p] = 0
        return seats


class STV:
    """Single Transferable Vote for multi-winner elections."""

    def elect(self, ballots: List[List[int]], n_seats: int) -> List[int]:
        """
        Run STV election.
        Returns list of elected candidates.
        """
        candidates = set(c for b in ballots for c in b)
        active = set(candidates)
        # Droop quota
        quota = len(ballots) / (n_seats + 1) + 1

        elected = []
        # Each ballot has weight 1.0 initially
        weights = [1.0] * len(ballots)
        current_ballots = [list(b) for b in ballots]

        while len(elected) < n_seats and active:
            # Count first preferences
            counts = defaultdict(float)
            for i, ballot in enumerate(current_ballots):
                for c in ballot:
                    if c in active:
                        counts[c] += weights[i]
                        break

            if not counts:
                break

            # Check if any candidate meets quota
            elected_this_round = False
            for c in sorted(counts, key=counts.get, reverse=True):
                if counts[c] >= quota and len(elected) < n_seats:
                    elected.append(c)
                    active.discard(c)
                    elected_this_round = True

                    # Transfer surplus
                    surplus = counts[c] - quota
                    if surplus > 0 and counts[c] > 0:
                        transfer_ratio = surplus / counts[c]
                        for i, ballot in enumerate(current_ballots):
                            for bc in ballot:
                                if bc == c:
                                    weights[i] *= transfer_ratio
                                    break

            if not elected_this_round:
                # Eliminate candidate with fewest votes
                min_c = min(counts, key=counts.get)
                active.discard(min_c)

        return elected


class VotingSimulator:
    """Generate random voting profiles from various models."""

    def __init__(self, n_candidates: int, n_voters: int, seed: int = 42):
        self.n_candidates = n_candidates
        self.n_voters = n_voters
        self.rng = np.random.default_rng(seed)

    def impartial_culture(self) -> List[List[int]]:
        """Each voter has a uniformly random preference ordering."""
        candidates = list(range(self.n_candidates))
        ballots = []
        for _ in range(self.n_voters):
            perm = list(self.rng.permutation(candidates))
            ballots.append(perm)
        return ballots

    def mallows_model(self, phi: float = 0.5, reference: Optional[List[int]] = None) -> List[List[int]]:
        """
        Mallows model: probability proportional to phi^(Kendall distance from reference).
        phi=1: uniform, phi->0: all agree with reference.
        """
        if reference is None:
            reference = list(range(self.n_candidates))

        ballots = []
        for _ in range(self.n_voters):
            # Repeated insertion model for Mallows sampling
            perm = []
            for i in range(self.n_candidates):
                # Insert reference[i] at position chosen with geometric-like distribution
                probs = np.array([phi ** j for j in range(len(perm) + 1)])
                probs = probs / probs.sum()
                pos = self.rng.choice(len(perm) + 1, p=probs)
                perm.insert(pos, reference[i])
            ballots.append(perm)
        return ballots

    def plackett_luce(self, strengths: Optional[np.ndarray] = None) -> List[List[int]]:
        """
        Plackett-Luce model: probability of ranking proportional to strength parameters.
        At each step, probability of choosing candidate c is strength[c] / sum(remaining).
        """
        if strengths is None:
            strengths = self.rng.exponential(1.0, size=self.n_candidates)

        candidates = list(range(self.n_candidates))
        ballots = []
        for _ in range(self.n_voters):
            remaining = list(candidates)
            remaining_strengths = np.array([strengths[c] for c in remaining])
            perm = []
            for _ in range(self.n_candidates):
                probs = remaining_strengths / remaining_strengths.sum()
                idx = self.rng.choice(len(remaining), p=probs)
                perm.append(remaining[idx])
                remaining.pop(idx)
                remaining_strengths = np.delete(remaining_strengths, idx)
            ballots.append(perm)
        return ballots

    def single_peaked(self, axis: Optional[List[int]] = None) -> List[List[int]]:
        """
        Generate single-peaked preferences.
        Each voter has a peak on a 1D axis and preferences decrease with distance.
        """
        if axis is None:
            axis = list(range(self.n_candidates))

        ballots = []
        for _ in range(self.n_voters):
            peak_idx = self.rng.integers(0, self.n_candidates)
            distances = [abs(i - peak_idx) for i in range(self.n_candidates)]
            # Add small noise to break ties
            noisy_dist = [d + self.rng.uniform(0, 0.01) for d in distances]
            ranking = sorted(axis, key=lambda c: noisy_dist[axis.index(c)])
            ballots.append(ranking)
        return ballots


def condorcet_efficiency(method, n_candidates: int = 4, n_voters: int = 100,
                          n_trials: int = 100, seed: int = 42) -> float:
    """
    Compute Condorcet efficiency: fraction of elections where method
    selects the Condorcet winner (when one exists).
    """
    sim = VotingSimulator(n_candidates, n_voters, seed)
    condorcet_exists = 0
    condorcet_selected = 0

    for trial in range(n_trials):
        sim.rng = np.random.default_rng(seed + trial)
        ballots = sim.impartial_culture()
        cw = find_condorcet_winner(ballots)
        if cw is not None:
            condorcet_exists += 1
            winner = method.elect(ballots)
            if winner == cw:
                condorcet_selected += 1

    if condorcet_exists == 0:
        return 1.0  # Vacuously true
    return condorcet_selected / condorcet_exists


def compute_all_condorcet_efficiencies(n_candidates: int = 4, n_voters: int = 50,
                                        n_trials: int = 100, seed: int = 42) -> Dict[str, float]:
    """Compute Condorcet efficiency for all voting methods."""
    methods = {
        'Plurality': PluralityVoting(),
        'Borda': BordaCount(),
        'IRV': InstantRunoffVoting(),
        'Copeland': CopelandMethod(),
        'Schulze': SchulzeMethod(),
    }
    results = {}
    for name, method in methods.items():
        results[name] = condorcet_efficiency(method, n_candidates, n_voters, n_trials, seed)
    return results
