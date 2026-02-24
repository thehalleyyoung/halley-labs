"""
Collective decision-making mechanisms for divergence-aware governance.

Implements judgment aggregation, liquid democracy, quadratic voting,
conviction voting, participatory budgeting, and futarchy using
real mathematical algorithms built on numpy and scipy.
"""

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import softmax
from typing import List, Dict, Tuple, Optional, Any


class JudgmentAggregation:
    """Aggregate individual judgments into collective judgments.

    Implements majority rule with doctrinal paradox detection,
    premise-based procedures, and conclusion-based procedures
    following the framework of List & Pettit (2002).

    Each judgment set is a binary vector over a set of propositions.
    A consistency constraint matrix encodes logical relationships
    among propositions so that paradoxes can be detected.

    Attributes:
        n_propositions: Number of propositions in the agenda.
        constraints: Matrix encoding logical constraints among propositions.
            Each row represents a constraint. A judgment vector j is
            consistent iff constraints @ j <= constraint_bounds for every row.
        constraint_bounds: Right-hand side of the constraint inequalities.
        premise_indices: Indices of propositions treated as premises.
        conclusion_indices: Indices of propositions treated as conclusions.
    """

    def __init__(
        self,
        n_propositions: int,
        constraints: Optional[np.ndarray] = None,
        constraint_bounds: Optional[np.ndarray] = None,
        premise_indices: Optional[List[int]] = None,
        conclusion_indices: Optional[List[int]] = None,
    ):
        """Initialise a JudgmentAggregation instance.

        Args:
            n_propositions: Number of binary propositions in the agenda.
            constraints: A (k, n_propositions) matrix of linear constraints.
                A judgment vector j is *consistent* when
                ``constraints @ j <= constraint_bounds`` holds element-wise.
                When ``None``, no consistency constraints are imposed.
            constraint_bounds: A length-k vector of upper bounds for the
                constraint inequalities.  Required when *constraints* is given.
            premise_indices: Proposition indices treated as premises.
                Defaults to all indices except the last one.
            conclusion_indices: Proposition indices treated as conclusions.
                Defaults to the last index only.
        """
        self.n_propositions = n_propositions
        if constraints is not None:
            self.constraints = np.asarray(constraints, dtype=float)
            self.constraint_bounds = np.asarray(constraint_bounds, dtype=float)
        else:
            self.constraints = None
            self.constraint_bounds = None
        self.premise_indices = (
            premise_indices if premise_indices is not None
            else list(range(n_propositions - 1))
        )
        self.conclusion_indices = (
            conclusion_indices if conclusion_indices is not None
            else [n_propositions - 1]
        )

    def _check_consistency(self, judgment: np.ndarray) -> bool:
        """Return True if *judgment* satisfies all constraints.

        Args:
            judgment: Binary vector of length n_propositions.

        Returns:
            True when the judgment is consistent with the constraint matrix,
            or when no constraints have been set.
        """
        if self.constraints is None:
            return True
        lhs = self.constraints @ judgment
        return bool(np.all(lhs <= self.constraint_bounds + 1e-9))

    def aggregate(
        self, individual_judgments: np.ndarray
    ) -> Dict[str, Any]:
        """Aggregate judgments using majority rule.

        Applies proposition-wise majority voting and checks the resulting
        collective judgment for consistency (the *doctrinal paradox*).

        Args:
            individual_judgments: A (n_voters, n_propositions) binary matrix.

        Returns:
            A dictionary with keys:
                ``collective``: the majority-rule judgment vector,
                ``paradox``: boolean indicating a doctrinal paradox,
                ``support``: proportion of voters supporting each proposition.
        """
        judgments = np.asarray(individual_judgments, dtype=float)
        n_voters = judgments.shape[0]
        support = judgments.mean(axis=0)
        collective = (support > 0.5).astype(float)
        # Break exact ties toward rejection (0).
        tie_mask = np.isclose(support, 0.5)
        collective[tie_mask] = 0.0
        paradox = not self._check_consistency(collective)
        return {
            "collective": collective,
            "paradox": paradox,
            "support": support,
        }

    def premise_based(
        self, individual_judgments: np.ndarray
    ) -> Dict[str, Any]:
        """Premise-based procedure for judgment aggregation.

        Applies majority rule to premise propositions only, then derives
        the conclusion propositions by finding the closest consistent
        judgment vector (in Hamming distance) that agrees with the
        majority result on all premises.

        Args:
            individual_judgments: A (n_voters, n_propositions) binary matrix.

        Returns:
            Dictionary with ``collective``, ``premise_majority``,
            ``derived_conclusions``, and ``consistent`` flag.
        """
        judgments = np.asarray(individual_judgments, dtype=float)
        support = judgments.mean(axis=0)
        premise_majority = (support[self.premise_indices] > 0.5).astype(float)

        best_judgment = np.zeros(self.n_propositions)
        for idx, pi in enumerate(self.premise_indices):
            best_judgment[pi] = premise_majority[idx]

        # Search over all 2^|C| assignments to conclusions.
        n_conc = len(self.conclusion_indices)
        best_dist = float("inf")
        best_conc = np.zeros(n_conc)
        for bits in range(2 ** n_conc):
            candidate = best_judgment.copy()
            conc_vals = np.array(
                [(bits >> k) & 1 for k in range(n_conc)], dtype=float
            )
            for k, ci in enumerate(self.conclusion_indices):
                candidate[ci] = conc_vals[k]
            if self._check_consistency(candidate):
                dist = np.sum(np.abs(candidate - support))
                if dist < best_dist:
                    best_dist = dist
                    best_conc = conc_vals.copy()
                    best_judgment = candidate.copy()

        derived = {}
        for k, ci in enumerate(self.conclusion_indices):
            derived[ci] = best_conc[k]
            best_judgment[ci] = best_conc[k]

        return {
            "collective": best_judgment,
            "premise_majority": premise_majority,
            "derived_conclusions": derived,
            "consistent": self._check_consistency(best_judgment),
        }

    def conclusion_based(
        self, individual_judgments: np.ndarray
    ) -> Dict[str, Any]:
        """Conclusion-based procedure for judgment aggregation.

        Each voter first derives their own conclusions from their full
        judgment set, then majority rule is applied only to conclusions.
        Premises in the collective are set to match the majority on premises,
        but inconsistencies between premise and conclusion majorities are
        tolerated.

        Args:
            individual_judgments: A (n_voters, n_propositions) binary matrix.

        Returns:
            Dictionary with ``collective``, ``conclusion_majority``,
            ``premise_majority``, and ``consistent`` flag.
        """
        judgments = np.asarray(individual_judgments, dtype=float)
        support = judgments.mean(axis=0)

        conclusion_majority = (
            support[self.conclusion_indices] > 0.5
        ).astype(float)
        premise_majority = (
            support[self.premise_indices] > 0.5
        ).astype(float)

        collective = np.zeros(self.n_propositions)
        for idx, pi in enumerate(self.premise_indices):
            collective[pi] = premise_majority[idx]
        for idx, ci in enumerate(self.conclusion_indices):
            collective[ci] = conclusion_majority[idx]

        return {
            "collective": collective,
            "conclusion_majority": conclusion_majority,
            "premise_majority": premise_majority,
            "consistent": self._check_consistency(collective),
        }


class LiquidDemocracy:
    """Delegative voting with transitive proxy chains and cycle detection.

    Voters may either vote directly or delegate their vote to another voter.
    Delegations are transitive: if A delegates to B and B delegates to C,
    then A's weight flows to C.  Cycles are detected via depth-first search
    and broken by revoking all delegations in the cycle (those voters'
    weights remain with themselves).

    Attributes:
        n_voters: Number of participants.
    """

    def __init__(self, n_voters: int):
        """Initialise with the number of voters.

        Args:
            n_voters: Total number of voters / delegates.
        """
        self.n_voters = n_voters

    def _detect_cycles(
        self, delegations: Dict[int, int]
    ) -> List[List[int]]:
        """Find all delegation cycles using iterative DFS.

        Args:
            delegations: Mapping from voter index to delegate index.

        Returns:
            List of cycles, where each cycle is a list of voter indices.
        """
        visited: Dict[int, int] = {}
        cycles: List[List[int]] = []
        WHITE, GREY, BLACK = 0, 1, 2

        for node in range(self.n_voters):
            if visited.get(node, WHITE) != WHITE:
                continue
            stack = [node]
            path: List[int] = []
            path_set: set = set()
            while stack:
                v = stack[-1]
                if visited.get(v, WHITE) == WHITE:
                    visited[v] = GREY
                    path.append(v)
                    path_set.add(v)
                    nxt = delegations.get(v)
                    if nxt is not None and nxt != v:
                        if visited.get(nxt, WHITE) == WHITE:
                            stack.append(nxt)
                        elif visited.get(nxt, WHITE) == GREY and nxt in path_set:
                            idx = path.index(nxt)
                            cycles.append(path[idx:])
                        # If BLACK, skip
                    # No delegation or self-delegation: nothing to push
                else:
                    stack.pop()
                    if path and path[-1] == v:
                        path.pop()
                        path_set.discard(v)
                    visited[v] = BLACK
        return cycles

    def resolve(
        self,
        delegations: Dict[int, int],
        direct_votes: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        """Resolve delegations and compute weighted votes.

        Args:
            delegations: Maps voter index to the index they delegate to.
                Voters not in this dict (or who delegate to themselves) vote
                directly.
            direct_votes: Maps voter index to their vote vector.  Voters who
                delegate need not appear here; if they do, their direct vote
                is used only when delegation is revoked.

        Returns:
            Dictionary with:
                ``weights``: array of effective weight per voter,
                ``cycles``: list of detected cycles,
                ``effective_votes``: dict mapping voter index to vote vector,
                ``result``: weighted vote tally across all options.
        """
        cycles = self._detect_cycles(delegations)
        cycle_members: set = set()
        for cycle in cycles:
            cycle_members.update(cycle)

        clean_delegations = {
            k: v
            for k, v in delegations.items()
            if k not in cycle_members and k != v
        }

        # Follow chains to find terminal delegate for each voter.
        terminal: Dict[int, int] = {}
        for voter in range(self.n_voters):
            cur = voter
            seen: set = set()
            while cur in clean_delegations and cur not in seen:
                seen.add(cur)
                cur = clean_delegations[cur]
            terminal[voter] = cur

        weights = np.zeros(self.n_voters)
        for voter in range(self.n_voters):
            weights[terminal[voter]] += 1.0

        # Determine effective vote for each voter.
        vote_dim = None
        for v in direct_votes.values():
            vote_dim = len(v)
            break
        if vote_dim is None:
            vote_dim = 1

        effective_votes: Dict[int, np.ndarray] = {}
        for voter in range(self.n_voters):
            t = terminal[voter]
            if t in direct_votes:
                effective_votes[voter] = direct_votes[t]
            elif voter in direct_votes:
                effective_votes[voter] = direct_votes[voter]
            else:
                effective_votes[voter] = np.zeros(vote_dim)

        result = np.zeros(vote_dim)
        counted: set = set()
        for voter in range(self.n_voters):
            t = terminal[voter]
            if t not in counted and t in direct_votes:
                result += weights[t] * direct_votes[t]
                counted.add(t)
            elif voter not in clean_delegations and voter in direct_votes and voter not in counted:
                result += weights[voter] * direct_votes[voter]
                counted.add(voter)

        return {
            "weights": weights,
            "cycles": cycles,
            "effective_votes": effective_votes,
            "result": result,
        }


class QuadraticVoting:
    """Optimal allocation of voice credits using quadratic cost.

    Each voter receives a budget of voice credits.  The cost of casting
    *v* votes on a single issue is *v^2* credits, incentivising voters
    to spread influence across issues they care about rather than
    concentrating on a single one.

    Attributes:
        n_issues: Number of issues to vote on.
        budget: Voice-credit budget per voter.
    """

    def __init__(self, n_issues: int, budget: float = 100.0):
        """Initialise quadratic voting mechanism.

        Args:
            n_issues: Number of issues on the ballot.
            budget: Total voice credits available to each voter.
        """
        self.n_issues = n_issues
        self.budget = budget

    def optimal_allocation(
        self, utilities: np.ndarray
    ) -> np.ndarray:
        """Compute the optimal vote allocation for a single voter.

        Maximises total utility sum_i u_i * v_i subject to the quadratic
        budget constraint sum_i v_i^2 <= budget, solved via KKT conditions.

        The closed-form solution is v_i = u_i / (2 * lambda) where lambda
        is chosen so that sum v_i^2 = budget.

        Args:
            utilities: Length-n_issues vector of marginal utilities.

        Returns:
            Optimal vote vector (may contain negative entries for
            opposition).
        """
        u = np.asarray(utilities, dtype=float)
        norm_sq = np.dot(u, u)
        if norm_sq < 1e-12:
            return np.zeros(self.n_issues)
        # lambda = sqrt(norm_sq / (4 * budget))
        lam = np.sqrt(norm_sq / (4.0 * self.budget))
        votes = u / (2.0 * lam)
        return votes

    def tally(
        self, all_utilities: np.ndarray
    ) -> Dict[str, Any]:
        """Run a full quadratic vote across all voters.

        Each row of *all_utilities* is a voter's utility vector.
        Optimal allocations are computed, then the net vote on each issue
        is summed.

        Args:
            all_utilities: (n_voters, n_issues) matrix of utilities.

        Returns:
            Dictionary with:
                ``allocations``: (n_voters, n_issues) vote allocations,
                ``net_votes``: length-n_issues net vote tally,
                ``outcomes``: boolean vector (True = issue passes),
                ``credits_used``: credits spent by each voter.
        """
        utils = np.asarray(all_utilities, dtype=float)
        n_voters = utils.shape[0]
        allocations = np.zeros_like(utils)
        credits_used = np.zeros(n_voters)

        for i in range(n_voters):
            alloc = self.optimal_allocation(utils[i])
            allocations[i] = alloc
            credits_used[i] = np.sum(alloc ** 2)

        net_votes = allocations.sum(axis=0)
        outcomes = net_votes > 0

        return {
            "allocations": allocations,
            "net_votes": net_votes,
            "outcomes": outcomes,
            "credits_used": credits_used,
        }


class ConvictionVoting:
    """Continuous preference signalling with exponential decay.

    Voters stake tokens on proposals over discrete time steps.
    Conviction for proposal *p* at time *t* is:

        C_p(t) = alpha * C_p(t-1) + S_p(t)

    where *alpha* is the decay parameter and *S_p(t)* is the total
    stake on proposal *p* at time *t*.  A proposal triggers (passes)
    when its conviction exceeds a threshold derived from the requested
    funding relative to the total pool.

    Attributes:
        n_proposals: Number of active proposals.
        alpha: Decay / momentum parameter in [0, 1).
        total_supply: Total token supply (for threshold calculation).
    """

    def __init__(
        self,
        n_proposals: int,
        alpha: float = 0.9,
        total_supply: float = 1e6,
    ):
        """Initialise conviction voting.

        Args:
            n_proposals: Number of proposals.
            alpha: Exponential decay rate.
            total_supply: Total token supply used in threshold formula.
        """
        self.n_proposals = n_proposals
        self.alpha = alpha
        self.total_supply = total_supply
        self.conviction = np.zeros(n_proposals)

    def threshold(
        self, requested_fraction: float, beta: float = 0.2
    ) -> float:
        """Compute the conviction threshold for a funding request.

        Uses the formula from Commons Stack / 1Hive:
            T = total_supply * beta / (1 - requested_fraction)^2

        Args:
            requested_fraction: Fraction of the common pool requested,
                must be in (0, 1).
            beta: Sensitivity parameter controlling threshold steepness.

        Returns:
            Conviction threshold value.
        """
        denom = (1.0 - requested_fraction) ** 2
        if denom < 1e-12:
            return float("inf")
        return self.total_supply * beta / denom

    def step(self, stakes: np.ndarray) -> np.ndarray:
        """Advance one time step given current staking distribution.

        Updates the internal conviction state according to the exponential
        moving average formula.

        Args:
            stakes: Length-n_proposals vector of total stake per proposal
                at this time step.

        Returns:
            Updated conviction vector.
        """
        s = np.asarray(stakes, dtype=float)
        self.conviction = self.alpha * self.conviction + s
        return self.conviction.copy()

    def simulate(
        self,
        stake_history: np.ndarray,
        requested_fractions: np.ndarray,
        beta: float = 0.2,
    ) -> Dict[str, Any]:
        """Simulate conviction voting over multiple time steps.

        Args:
            stake_history: (T, n_proposals) matrix of stakes over time.
            requested_fractions: Length-n_proposals vector of requested
                fractions for each proposal.
            beta: Threshold sensitivity parameter.

        Returns:
            Dictionary with:
                ``conviction_history``: (T, n_proposals) conviction over time,
                ``thresholds``: threshold per proposal,
                ``triggered``: boolean vector indicating which proposals
                    crossed their threshold,
                ``trigger_times``: time step at which each proposal first
                    triggered (-1 if never).
        """
        stakes = np.asarray(stake_history, dtype=float)
        fracs = np.asarray(requested_fractions, dtype=float)
        T = stakes.shape[0]

        self.conviction = np.zeros(self.n_proposals)
        conviction_history = np.zeros((T, self.n_proposals))
        thresholds = np.array(
            [self.threshold(f, beta) for f in fracs]
        )

        trigger_times = -np.ones(self.n_proposals, dtype=int)
        triggered = np.zeros(self.n_proposals, dtype=bool)

        for t in range(T):
            self.step(stakes[t])
            conviction_history[t] = self.conviction.copy()
            for p in range(self.n_proposals):
                if not triggered[p] and self.conviction[p] >= thresholds[p]:
                    triggered[p] = True
                    trigger_times[p] = t

        return {
            "conviction_history": conviction_history,
            "thresholds": thresholds,
            "triggered": triggered,
            "trigger_times": trigger_times,
        }

    def steady_state_conviction(self, constant_stake: np.ndarray) -> np.ndarray:
        """Compute steady-state conviction for a constant stake vector.

        When stakes are constant the geometric series yields:
            C_inf = S / (1 - alpha)

        Args:
            constant_stake: Length-n_proposals constant stake vector.

        Returns:
            Steady-state conviction vector.
        """
        s = np.asarray(constant_stake, dtype=float)
        return s / (1.0 - self.alpha)


class ParticipatoryBudgeting:
    """Knapsack-based allocation with voter preference aggregation.

    Voters submit approval ballots over a set of projects, each with a
    cost.  The mechanism selects the subset of projects that maximises
    aggregated voter support subject to the total budget constraint.
    This is a variant of the 0-1 knapsack problem solved via LP
    relaxation and rounding.

    Attributes:
        budget: Total available budget.
        project_costs: Array of per-project costs.
        n_projects: Number of candidate projects.
    """

    def __init__(
        self, budget: float, project_costs: np.ndarray
    ):
        """Initialise participatory budgeting.

        Args:
            budget: Total budget available for project funding.
            project_costs: Length-n_projects vector of project costs.
        """
        self.budget = budget
        self.project_costs = np.asarray(project_costs, dtype=float)
        self.n_projects = len(self.project_costs)

    def aggregate_preferences(
        self, ballots: np.ndarray, method: str = "approval"
    ) -> np.ndarray:
        """Aggregate voter ballots into project scores.

        Args:
            ballots: (n_voters, n_projects) matrix.  For approval voting
                entries are 0/1.  For score voting entries are real-valued.
            method: ``'approval'`` sums binary approvals; ``'score'`` sums
                raw scores; ``'rank'`` uses Borda-count conversion.

        Returns:
            Length-n_projects score vector.
        """
        b = np.asarray(ballots, dtype=float)
        if method == "approval":
            return b.sum(axis=0)
        elif method == "score":
            return b.sum(axis=0)
        elif method == "rank":
            n_voters, n_proj = b.shape
            borda = np.zeros(n_proj)
            for i in range(n_voters):
                order = np.argsort(b[i])
                for rank_pos, proj_idx in enumerate(order):
                    borda[proj_idx] += rank_pos
            return borda
        else:
            return b.sum(axis=0)

    def knapsack_greedy(self, scores: np.ndarray) -> np.ndarray:
        """Greedy 0-1 knapsack: sort by score/cost ratio, fill greedily.

        Args:
            scores: Length-n_projects score vector.

        Returns:
            Binary selection vector.
        """
        ratios = np.where(
            self.project_costs > 0, scores / self.project_costs, 0.0
        )
        order = np.argsort(-ratios)
        selected = np.zeros(self.n_projects, dtype=float)
        remaining = self.budget
        for idx in order:
            if self.project_costs[idx] <= remaining + 1e-9:
                selected[idx] = 1.0
                remaining -= self.project_costs[idx]
        return selected

    def knapsack_lp(self, scores: np.ndarray) -> np.ndarray:
        """LP-relaxation based selection with deterministic rounding.

        Solves the LP relaxation of the 0-1 knapsack, then rounds the
        fractional solution greedily.

        Args:
            scores: Length-n_projects aggregate scores.

        Returns:
            Binary selection vector.
        """
        c = -scores  # linprog minimises
        A_ub = self.project_costs.reshape(1, -1)
        b_ub = np.array([self.budget])
        bounds = [(0.0, 1.0)] * self.n_projects

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            return self.knapsack_greedy(scores)

        fractional = res.x
        # Round: select all with x >= 1-eps, then greedily add remainder.
        selected = np.zeros(self.n_projects, dtype=float)
        remaining = self.budget
        order = np.argsort(-fractional)
        for idx in order:
            if fractional[idx] > 1e-6 and self.project_costs[idx] <= remaining + 1e-9:
                selected[idx] = 1.0
                remaining -= self.project_costs[idx]
        return selected

    def allocate(
        self,
        ballots: np.ndarray,
        method: str = "approval",
        solver: str = "lp",
    ) -> Dict[str, Any]:
        """Run the full participatory budgeting pipeline.

        Args:
            ballots: (n_voters, n_projects) preference matrix.
            method: Aggregation method (see ``aggregate_preferences``).
            solver: ``'lp'`` for LP-relaxation, ``'greedy'`` for greedy.

        Returns:
            Dictionary with:
                ``selected``: binary project selection vector,
                ``scores``: aggregate project scores,
                ``total_cost``: total cost of selected projects,
                ``budget_remaining``: unspent budget.
        """
        scores = self.aggregate_preferences(ballots, method)
        if solver == "lp":
            selected = self.knapsack_lp(scores)
        else:
            selected = self.knapsack_greedy(scores)

        total_cost = np.dot(selected, self.project_costs)
        return {
            "selected": selected,
            "scores": scores,
            "total_cost": total_cost,
            "budget_remaining": self.budget - total_cost,
        }

    def proportional_allocation(
        self, ballots: np.ndarray
    ) -> Dict[str, Any]:
        """Equal-shares / proportional participatory budgeting.

        Distributes the budget equally among voters, then iteratively
        funds projects whose per-supporter share is affordable.

        Args:
            ballots: (n_voters, n_projects) binary approval matrix.

        Returns:
            Dictionary with ``selected``, ``voter_spending``,
            ``total_cost``.
        """
        b = np.asarray(ballots, dtype=float)
        n_voters = b.shape[0]
        voter_budget = np.full(n_voters, self.budget / n_voters)
        selected = np.zeros(self.n_projects, dtype=float)
        funded = np.zeros(self.n_projects, dtype=bool)

        for _ in range(self.n_projects):
            best_project = -1
            best_share = float("inf")
            for p in range(self.n_projects):
                if funded[p]:
                    continue
                supporters = np.where(b[:, p] > 0.5)[0]
                if len(supporters) == 0:
                    continue
                available = voter_budget[supporters].sum()
                if available < self.project_costs[p] - 1e-9:
                    continue
                share = self.project_costs[p] / len(supporters)
                if share < best_share:
                    best_share = share
                    best_project = p

            if best_project < 0:
                break

            supporters = np.where(b[:, best_project] > 0.5)[0]
            cost_remaining = self.project_costs[best_project]
            per_supporter = cost_remaining / len(supporters)

            # Iterative adjustment when some supporters can't pay full share
            can_pay = voter_budget[supporters] >= per_supporter - 1e-9
            while not np.all(can_pay) and cost_remaining > 1e-9:
                cannot = supporters[~can_pay]
                cost_remaining -= voter_budget[cannot].sum()
                voter_budget[cannot] = 0.0
                supporters = supporters[can_pay]
                if len(supporters) == 0:
                    break
                per_supporter = cost_remaining / len(supporters)
                can_pay = voter_budget[supporters] >= per_supporter - 1e-9

            if len(supporters) == 0:
                continue

            voter_budget[supporters] -= per_supporter
            voter_budget = np.maximum(voter_budget, 0.0)
            selected[best_project] = 1.0
            funded[best_project] = True

        total_cost = np.dot(selected, self.project_costs)
        return {
            "selected": selected,
            "voter_spending": self.budget / n_voters - voter_budget,
            "total_cost": total_cost,
        }


class Futarchy:
    """Prediction-market-based governance simulation.

    Implements Robin Hanson's futarchy concept: 'Vote on values, bet on
    beliefs.'  For each policy proposal two conditional prediction
    markets are simulated (policy adopted vs. rejected).  The policy
    whose conditional market price for the welfare metric is higher gets
    adopted.

    The market micro-structure uses a logarithmic market scoring rule
    (LMSR) as the automated market maker.

    Attributes:
        n_policies: Number of policy proposals.
        liquidity: LMSR liquidity parameter *b*.
    """

    def __init__(self, n_policies: int, liquidity: float = 100.0):
        """Initialise futarchy mechanism.

        Args:
            n_policies: Number of alternative policy proposals.
            liquidity: LMSR liquidity parameter (higher = deeper market).
        """
        self.n_policies = n_policies
        self.liquidity = liquidity

    def lmsr_cost(
        self, quantities: np.ndarray, b: Optional[float] = None
    ) -> float:
        """Compute the LMSR cost function C(q).

        C(q) = b * log(sum_i exp(q_i / b))

        Args:
            quantities: Vector of outstanding shares per outcome.
            b: Liquidity parameter. Defaults to ``self.liquidity``.

        Returns:
            Cost function value.
        """
        if b is None:
            b = self.liquidity
        q = np.asarray(quantities, dtype=float)
        # Numerically stable log-sum-exp.
        max_q = np.max(q / b)
        return b * (max_q + np.log(np.sum(np.exp(q / b - max_q))))

    def lmsr_price(
        self, quantities: np.ndarray, outcome: int, b: Optional[float] = None
    ) -> float:
        """Instantaneous price for one share of *outcome*.

        price_i = exp(q_i / b) / sum_j exp(q_j / b)

        Args:
            quantities: Current outstanding share vector.
            outcome: Index of the outcome to price.
            b: Liquidity parameter.

        Returns:
            Price in [0, 1].
        """
        if b is None:
            b = self.liquidity
        q = np.asarray(quantities, dtype=float)
        return float(softmax(q / b)[outcome])

    def lmsr_trade_cost(
        self,
        quantities: np.ndarray,
        outcome: int,
        amount: float,
        b: Optional[float] = None,
    ) -> float:
        """Cost to buy *amount* shares of *outcome* at current state.

        Args:
            quantities: Current outstanding shares.
            outcome: Which outcome to buy shares of.
            amount: Number of shares to purchase (negative to sell).
            b: Liquidity parameter.

        Returns:
            Cost in currency units (negative means trader receives money).
        """
        q = np.asarray(quantities, dtype=float)
        cost_before = self.lmsr_cost(q, b)
        q_after = q.copy()
        q_after[outcome] += amount
        cost_after = self.lmsr_cost(q_after, b)
        return cost_after - cost_before

    def simulate_market(
        self,
        true_prob: float,
        n_traders: int = 50,
        n_rounds: int = 100,
        noise: float = 0.1,
    ) -> Dict[str, Any]:
        """Simulate an LMSR prediction market converging to true_prob.

        Traders receive noisy private signals about the true probability
        and trade to maximise expected log-wealth.

        Args:
            true_prob: True probability of the outcome (in [0, 1]).
            n_traders: Number of traders.
            n_rounds: Number of trading rounds.
            noise: Standard deviation of signal noise.

        Returns:
            Dictionary with:
                ``price_history``: list of prices after each trade,
                ``final_price``: terminal market price,
                ``quantities``: final outstanding shares.
        """
        rng = np.random.default_rng(42)
        quantities = np.zeros(2)
        price_history = []

        for _round in range(n_rounds):
            trader_idx = rng.integers(0, n_traders)
            signal = np.clip(
                true_prob + rng.normal(0, noise), 0.01, 0.99
            )
            current_price = self.lmsr_price(quantities, 0)
            # Kelly-criterion inspired sizing.
            if signal > current_price + 0.01:
                edge = signal - current_price
                amount = self.liquidity * edge * 0.5
                quantities[0] += amount
            elif signal < current_price - 0.01:
                edge = current_price - signal
                amount = self.liquidity * edge * 0.5
                quantities[1] += amount

            price_history.append(self.lmsr_price(quantities, 0))

        return {
            "price_history": price_history,
            "final_price": price_history[-1] if price_history else 0.5,
            "quantities": quantities,
        }

    def evaluate_policies(
        self,
        welfare_signals: np.ndarray,
        n_rounds: int = 200,
        noise: float = 0.1,
    ) -> Dict[str, Any]:
        """Run futarchy: simulate conditional markets and pick the best policy.

        For each policy, two markets are run:
        - "welfare | policy adopted"  with true signal welfare_signals[i]
        - "welfare | policy rejected" with a baseline signal of 0.5

        The policy with the highest spread (adopted price - rejected price)
        is selected.

        Args:
            welfare_signals: Length-n_policies vector of true welfare
                probabilities conditional on adoption.
            n_rounds: Trading rounds per market.
            noise: Trader signal noise.

        Returns:
            Dictionary with:
                ``selected_policy``: index of chosen policy,
                ``adopted_prices``: final prices in adoption markets,
                ``rejected_prices``: final prices in rejection markets,
                ``spreads``: adopted - rejected per policy.
        """
        w = np.asarray(welfare_signals, dtype=float)
        adopted_prices = np.zeros(self.n_policies)
        rejected_prices = np.zeros(self.n_policies)

        for i in range(self.n_policies):
            adopt_result = self.simulate_market(
                w[i], n_rounds=n_rounds, noise=noise
            )
            adopted_prices[i] = adopt_result["final_price"]
            reject_result = self.simulate_market(
                0.5, n_rounds=n_rounds, noise=noise
            )
            rejected_prices[i] = reject_result["final_price"]

        spreads = adopted_prices - rejected_prices
        selected = int(np.argmax(spreads))

        return {
            "selected_policy": selected,
            "adopted_prices": adopted_prices,
            "rejected_prices": rejected_prices,
            "spreads": spreads,
        }

    def combinatorial_futarchy(
        self,
        n_outcomes: int,
        correlations: np.ndarray,
        n_rounds: int = 200,
    ) -> Dict[str, Any]:
        """Combinatorial prediction market over correlated policy outcomes.

        Models correlations among policy outcomes using a multivariate
        normal copula, then runs separate LMSR markets for each outcome
        and adjusts prices by the correlation structure.

        Args:
            n_outcomes: Number of welfare outcomes to track.
            correlations: (n_outcomes, n_outcomes) correlation matrix.
            n_rounds: Trading rounds per market.

        Returns:
            Dictionary with ``marginal_prices``, ``joint_adjustment``,
            ``adjusted_prices``.
        """
        rng = np.random.default_rng(123)
        corr = np.asarray(correlations, dtype=float)
        # Ensure PSD via eigenvalue clipping.
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Draw correlated signals.
        mean = np.full(n_outcomes, 0.5)
        raw_signals = rng.multivariate_normal(mean, corr_psd * 0.01, size=1)[0]
        true_probs = np.clip(raw_signals, 0.05, 0.95)

        marginal_prices = np.zeros(n_outcomes)
        for i in range(n_outcomes):
            result = self.simulate_market(
                true_probs[i], n_rounds=n_rounds, noise=0.1
            )
            marginal_prices[i] = result["final_price"]

        # Adjust marginals by off-diagonal correlation effects.
        joint_adjustment = np.zeros(n_outcomes)
        for i in range(n_outcomes):
            for j in range(n_outcomes):
                if i != j:
                    joint_adjustment[i] += (
                        corr_psd[i, j]
                        * (marginal_prices[j] - 0.5)
                        * 0.1
                    )
        adjusted_prices = np.clip(marginal_prices + joint_adjustment, 0.0, 1.0)

        return {
            "marginal_prices": marginal_prices,
            "joint_adjustment": joint_adjustment,
            "adjusted_prices": adjusted_prices,
        }


def divergence_aware_aggregation(
    individual_judgments: np.ndarray,
    constraints: Optional[np.ndarray] = None,
    constraint_bounds: Optional[np.ndarray] = None,
    premise_indices: Optional[List[int]] = None,
    conclusion_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run all three judgment aggregation procedures and compare.

    This is the main entry point for divergence-aware collective
    decision analysis.  It applies majority rule, premise-based,
    and conclusion-based procedures, highlights paradoxes, and
    computes inter-method divergence.

    Args:
        individual_judgments: (n_voters, n_propositions) binary matrix.
        constraints: Consistency constraint matrix.
        constraint_bounds: Constraint bounds vector.
        premise_indices: Indices for premise propositions.
        conclusion_indices: Indices for conclusion propositions.

    Returns:
        Dictionary with results from all three methods plus a
        ``divergence_matrix`` measuring pairwise Hamming distances.
    """
    n_prop = individual_judgments.shape[1]
    ja = JudgmentAggregation(
        n_prop, constraints, constraint_bounds,
        premise_indices, conclusion_indices,
    )

    majority = ja.aggregate(individual_judgments)
    premise = ja.premise_based(individual_judgments)
    conclusion = ja.conclusion_based(individual_judgments)

    methods = {
        "majority": majority["collective"],
        "premise_based": premise["collective"],
        "conclusion_based": conclusion["collective"],
    }
    names = list(methods.keys())
    div_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            div_matrix[i, j] = np.sum(
                np.abs(methods[names[i]] - methods[names[j]])
            )

    return {
        "majority": majority,
        "premise_based": premise,
        "conclusion_based": conclusion,
        "divergence_matrix": div_matrix,
        "method_names": names,
    }


def run_collective_pipeline(
    judgments: np.ndarray,
    delegations: Dict[int, int],
    direct_votes: Dict[int, np.ndarray],
    utilities: np.ndarray,
    stake_history: np.ndarray,
    requested_fractions: np.ndarray,
    budget: float,
    project_costs: np.ndarray,
    ballots: np.ndarray,
    welfare_signals: np.ndarray,
    constraints: Optional[np.ndarray] = None,
    constraint_bounds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute the full collective decision pipeline.

    Orchestrates judgment aggregation, liquid democracy, quadratic
    voting, conviction voting, participatory budgeting, and futarchy
    in a single call.

    Args:
        judgments: Binary judgment matrix for JudgmentAggregation.
        delegations: Delegation map for LiquidDemocracy.
        direct_votes: Direct vote vectors for LiquidDemocracy.
        utilities: Utility matrix for QuadraticVoting.
        stake_history: Stake matrix for ConvictionVoting.
        requested_fractions: Funding requests for ConvictionVoting.
        budget: Budget for ParticipatoryBudgeting.
        project_costs: Project costs for ParticipatoryBudgeting.
        ballots: Voter ballots for ParticipatoryBudgeting.
        welfare_signals: Welfare signals for Futarchy.
        constraints: Optional consistency constraints.
        constraint_bounds: Optional constraint bounds.

    Returns:
        Nested dictionary with results keyed by mechanism name.
    """
    results = {}

    # Judgment aggregation
    results["judgment_aggregation"] = divergence_aware_aggregation(
        judgments, constraints, constraint_bounds,
    )

    # Liquid democracy
    n_voters = max(
        max(delegations.keys(), default=0),
        max(direct_votes.keys(), default=0),
    ) + 1
    ld = LiquidDemocracy(n_voters)
    results["liquid_democracy"] = ld.resolve(delegations, direct_votes)

    # Quadratic voting
    n_issues = utilities.shape[1]
    qv = QuadraticVoting(n_issues)
    results["quadratic_voting"] = qv.tally(utilities)

    # Conviction voting
    n_proposals = stake_history.shape[1]
    cv = ConvictionVoting(n_proposals)
    results["conviction_voting"] = cv.simulate(
        stake_history, requested_fractions
    )

    # Participatory budgeting
    pb = ParticipatoryBudgeting(budget, project_costs)
    results["participatory_budgeting"] = pb.allocate(ballots)

    # Futarchy
    n_policies = len(welfare_signals)
    fu = Futarchy(n_policies)
    results["futarchy"] = fu.evaluate_policies(welfare_signals)

    return results
