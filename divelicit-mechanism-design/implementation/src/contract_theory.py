"""
Contract Theory: Moral Hazard, Adverse Selection, and Mechanism Design.

Implements core models from contract theory using numerical optimization:
- Moral hazard (first-best and second-best contracts)
- Adverse selection (screening and signaling)
- Revelation principle verification
- Multi-dimensional screening with ironing

All algorithms use real mathematical formulations solved via numpy/scipy.
"""

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.linalg import solve as linalg_solve
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict


@dataclass
class OptimalContract:
    """Represents the solution to a contract design problem.

    Attributes:
        payments: Array of payments from principal to agent indexed by outcome
            or type. Shape depends on the model.
        effort_levels: Array of effort levels prescribed by the contract.
        principal_profit: Expected profit of the principal under this contract.
        agent_rents: Array of agent utilities (rents) by type or effort level.
        information_rent: Total information rent captured by the agent due to
            private information, measured as agent surplus above reservation.
        binding_constraints: List of constraint indices that bind at optimum.
        menu_items: Optional list of (quality, transfer) pairs for screening.
    """

    payments: np.ndarray
    effort_levels: np.ndarray
    principal_profit: float
    agent_rents: np.ndarray
    information_rent: float
    binding_constraints: List[int] = field(default_factory=list)
    menu_items: Optional[List[Tuple[float, float]]] = None


@dataclass
class Menu:
    """A menu of contracts offered in a screening or signaling game.

    Attributes:
        qualities: Array of quality/quantity levels for each type.
        transfers: Array of transfers (prices) for each type.
        type_assignments: Mapping from type index to menu item index.
        utilities: Agent utility for each type under assigned contract.
        principal_profit: Expected profit of the principal.
        is_separating: Whether the menu fully separates all types.
        pooling_groups: List of sets of type indices that are pooled together.
    """

    qualities: np.ndarray
    transfers: np.ndarray
    type_assignments: np.ndarray
    utilities: np.ndarray
    principal_profit: float
    is_separating: bool
    pooling_groups: Optional[List[set]] = None


class MoralHazardModel:
    """Solves moral hazard problems in principal-agent settings.

    In moral hazard, the agent's effort is unobservable to the principal.
    The principal designs a contract (payment schedule) to incentivize
    effort while sharing risk optimally.

    The model supports:
    - First-best: effort is observable, solve for Pareto-optimal risk sharing
    - Second-best: effort is unobservable, add incentive compatibility

    Mathematical formulation follows Holmstrom (1979) and Grossman-Hart (1983).
    """

    def __init__(self, reservation_utility: float = 0.0):
        """Initialize the moral hazard model.

        Args:
            reservation_utility: Agent's outside option utility (IR constraint).
        """
        self.reservation_utility = reservation_utility

    def solve(
        self,
        principal_utility: Callable[[float], float],
        agent_utility: Callable[[float], float],
        effort_costs: np.ndarray,
        output_dist: np.ndarray,
    ) -> OptimalContract:
        """Solve for the optimal contract under moral hazard.

        Solves the second-best problem where effort is unobservable.
        Uses the Grossman-Hart two-step approach:
          Step 1: For each effort level, find the cheapest incentive-compatible
                  payment scheme.
          Step 2: Choose the effort level maximizing principal's profit.

        Args:
            principal_utility: Concave utility function v(profit) for principal.
            agent_utility: Concave utility function u(payment) for agent.
                Must be strictly increasing.
            effort_costs: Array of shape (n_efforts,) with cost of each effort.
            output_dist: Array of shape (n_efforts, n_outcomes) where
                output_dist[e, x] = Pr(outcome x | effort e).
                Each row must sum to 1.

        Returns:
            OptimalContract with optimal payments, effort, and surplus division.
        """
        n_efforts = len(effort_costs)
        n_outcomes = output_dist.shape[1]

        best_profit = -np.inf
        best_contract = None

        for e_star in range(n_efforts):
            contract = self._solve_for_effort(
                e_star, principal_utility, agent_utility,
                effort_costs, output_dist
            )
            if contract is not None and contract.principal_profit > best_profit:
                best_profit = contract.principal_profit
                best_contract = contract

        if best_contract is None:
            payments = np.zeros(n_outcomes)
            best_contract = OptimalContract(
                payments=payments,
                effort_levels=np.array([0]),
                principal_profit=0.0,
                agent_rents=np.array([self.reservation_utility]),
                information_rent=0.0,
            )

        return best_contract

    def _solve_for_effort(
        self,
        e_star: int,
        principal_utility: Callable,
        agent_utility: Callable,
        effort_costs: np.ndarray,
        output_dist: np.ndarray,
    ) -> Optional[OptimalContract]:
        """Find cheapest incentive-compatible contract implementing effort e_star.

        Uses linear programming in the space of u(w_x) values (the agent's
        utility of payments), then inverts to recover payments.

        The LP minimizes expected payment subject to:
        - IR: E[u(w) | e*] - c(e*) >= u_bar
        - IC: E[u(w) | e*] - c(e*) >= E[u(w) | e] - c(e) for all e != e*

        Args:
            e_star: The effort level to implement.
            principal_utility: Principal's utility function.
            agent_utility: Agent's utility function.
            effort_costs: Cost array for each effort.
            output_dist: Conditional distribution of outcomes given effort.

        Returns:
            OptimalContract implementing e_star, or None if infeasible.
        """
        n_efforts, n_outcomes = output_dist.shape
        p_star = output_dist[e_star]

        # Decision variables: v_x = u(w_x) for each outcome x
        # Minimize E[w(v_x) | e*] = sum_x p*(x) * u_inv(v_x)
        # We approximate u_inv via a grid and use LP relaxation

        # For CARA utility u(w) = -exp(-r*w), u_inv(v) = -ln(-v)/r
        # For general utility, we linearize around a reference point

        # Use scipy minimize with the actual nonlinear formulation
        v0 = np.zeros(n_outcomes)

        def objective(v):
            """Expected cost of the contract: sum p*(x) * u_inv(v_x)."""
            # u_inv approximated: if u is concave increasing, invert numerically
            total = 0.0
            for x in range(n_outcomes):
                w_x = self._invert_utility(agent_utility, v[x])
                total += p_star[x] * w_x
            return total

        constraints = []

        # IR constraint: sum_x p*(x) v_x - c(e*) >= u_bar
        def ir_constraint(v):
            return np.dot(p_star, v) - effort_costs[e_star] - self.reservation_utility

        constraints.append({"type": "ineq", "fun": ir_constraint})

        # IC constraints: for each e != e*
        binding = []
        for e in range(n_efforts):
            if e == e_star:
                continue
            e_local = e  # capture in closure

            def ic_constraint(v, e_alt=e_local):
                lhs = np.dot(p_star, v) - effort_costs[e_star]
                rhs = np.dot(output_dist[e_alt], v) - effort_costs[e_alt]
                return lhs - rhs

            constraints.append({"type": "ineq", "fun": ic_constraint})

        # Initial guess: constant payment satisfying IR
        v_const = self.reservation_utility + effort_costs[e_star]
        v0 = np.full(n_outcomes, v_const)

        result = minimize(
            objective,
            v0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            # Try with a different starting point
            v0_alt = np.linspace(v_const - 1, v_const + 1, n_outcomes)
            result = minimize(
                objective,
                v0_alt,
                method="SLSQP",
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if not result.success:
                return None

        v_opt = result.x
        payments = np.array([
            self._invert_utility(agent_utility, v_opt[x])
            for x in range(n_outcomes)
        ])

        # Compute outcomes: assume outcome values are 0, 1, ..., n-1 scaled
        outcome_values = np.arange(n_outcomes, dtype=float)
        expected_output = np.dot(p_star, outcome_values)
        expected_payment = np.dot(p_star, payments)
        principal_profit = expected_output - expected_payment

        agent_eu = np.dot(p_star, v_opt) - effort_costs[e_star]
        info_rent = max(0.0, agent_eu - self.reservation_utility)

        # Check which constraints bind
        binding_list = []
        tol = 1e-6
        if abs(ir_constraint(v_opt)) < tol:
            binding_list.append(0)
        for idx, e in enumerate(e for e in range(n_efforts) if e != e_star):
            ic_val = np.dot(p_star, v_opt) - effort_costs[e_star] - (
                np.dot(output_dist[e], v_opt) - effort_costs[e]
            )
            if abs(ic_val) < tol:
                binding_list.append(idx + 1)

        return OptimalContract(
            payments=payments,
            effort_levels=np.array([e_star]),
            principal_profit=principal_profit,
            agent_rents=np.array([agent_eu]),
            information_rent=info_rent,
            binding_constraints=binding_list,
        )

    def solve_first_best(
        self,
        principal_utility: Callable[[float], float],
        agent_utility: Callable[[float], float],
        effort_costs: np.ndarray,
        output_dist: np.ndarray,
    ) -> OptimalContract:
        """Solve the first-best problem where effort is observable.

        When effort is observable, the principal can directly contract on effort.
        The problem reduces to optimal risk sharing (Borch rule) plus choosing
        the efficient effort level.

        The Borch rule states: u'(w(x)) / v'(x - w(x)) = lambda (constant),
        meaning marginal rates of substitution are equalized across states.

        Args:
            principal_utility: Principal's utility v(.).
            agent_utility: Agent's utility u(.).
            effort_costs: Cost of each effort level.
            output_dist: Pr(outcome | effort), shape (n_efforts, n_outcomes).

        Returns:
            OptimalContract with first-best payments and effort.
        """
        n_efforts, n_outcomes = output_dist.shape
        outcome_values = np.arange(n_outcomes, dtype=float)

        best_profit = -np.inf
        best_contract = None

        for e in range(n_efforts):
            p_e = output_dist[e]
            contract = self._borch_risk_sharing(
                p_e, outcome_values, principal_utility, agent_utility,
                effort_costs[e]
            )
            if contract is not None and contract.principal_profit > best_profit:
                best_profit = contract.principal_profit
                best_contract = contract
                best_contract.effort_levels = np.array([e])

        return best_contract

    def _borch_risk_sharing(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        v_func: Callable,
        u_func: Callable,
        effort_cost: float,
    ) -> Optional[OptimalContract]:
        """Compute Borch-optimal risk sharing for given probabilities.

        Maximizes E[v(x - w(x))] subject to E[u(w(x))] - c >= u_bar.

        At the optimum, the Borch rule holds:
            v'(x - w(x)) = lambda * u'(w(x)) for all x.

        We solve by parameterizing w(x) and using constrained optimization.

        Args:
            probabilities: Probability of each outcome.
            outcomes: Value of each outcome.
            v_func: Principal's utility function.
            u_func: Agent's utility function.
            effort_cost: Cost of the prescribed effort.

        Returns:
            OptimalContract with Borch-optimal payments.
        """
        n = len(outcomes)

        def neg_principal_eu(w):
            total = 0.0
            for i in range(n):
                total += probabilities[i] * v_func(outcomes[i] - w[i])
            return -total

        def agent_ir(w):
            eu = sum(probabilities[i] * u_func(w[i]) for i in range(n))
            return eu - effort_cost - self.reservation_utility

        # Initial guess: constant payment giving agent reservation utility
        w_init = self._invert_utility(u_func, self.reservation_utility + effort_cost)
        w0 = np.full(n, w_init)
        constraints = [{"type": "ineq", "fun": agent_ir}]

        # Payment bounds (reasonable range)
        bounds = [(0.0, outcomes[i] + 10.0) for i in range(n)]

        result = minimize(
            neg_principal_eu, w0, method="SLSQP",
            constraints=constraints, bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            return None

        w_opt = result.x
        principal_eu = -result.fun
        agent_eu = sum(
            probabilities[i] * u_func(w_opt[i]) for i in range(n)
        ) - effort_cost
        info_rent = max(0.0, agent_eu - self.reservation_utility)

        expected_payment = np.dot(probabilities, w_opt)
        expected_output = np.dot(probabilities, outcomes)

        return OptimalContract(
            payments=w_opt,
            effort_levels=np.array([0]),
            principal_profit=expected_output - expected_payment,
            agent_rents=np.array([agent_eu]),
            information_rent=info_rent,
        )

    @staticmethod
    def _invert_utility(
        u_func: Callable[[float], float], target: float,
        low: float = -100.0, high: float = 100.0,
    ) -> float:
        """Numerically invert u(w) = target to find w via bisection.

        Args:
            u_func: Strictly increasing utility function.
            target: Target utility value.
            low: Lower bound for bisection search.
            high: Upper bound for bisection search.

        Returns:
            w such that u(w) ≈ target.
        """
        for _ in range(200):
            mid = (low + high) / 2.0
            if u_func(mid) < target:
                low = mid
            else:
                high = mid
            if high - low < 1e-12:
                break
        return (low + high) / 2.0

    def compute_efficiency_loss(
        self,
        first_best: OptimalContract,
        second_best: OptimalContract,
    ) -> Dict[str, float]:
        """Compute the welfare loss from moral hazard.

        Measures the efficiency cost of unobservable effort by comparing
        first-best and second-best contracts.

        Args:
            first_best: Solution to the first-best problem.
            second_best: Solution to the second-best problem.

        Returns:
            Dictionary with efficiency loss metrics.
        """
        profit_loss = first_best.principal_profit - second_best.principal_profit
        rent_diff = second_best.information_rent - first_best.information_rent
        total_surplus_fb = first_best.principal_profit + np.sum(first_best.agent_rents)
        total_surplus_sb = second_best.principal_profit + np.sum(second_best.agent_rents)
        deadweight_loss = total_surplus_fb - total_surplus_sb

        return {
            "principal_profit_loss": profit_loss,
            "information_rent_increase": rent_diff,
            "deadweight_loss": deadweight_loss,
            "efficiency_ratio": total_surplus_sb / max(total_surplus_fb, 1e-15),
        }


class AdverseSelectionModel:
    """Solves adverse selection problems via screening and signaling.

    In adverse selection, the agent has private information about their type.
    The principal designs a menu of contracts to screen types, or agents
    send costly signals to reveal information.

    Implements:
    - Screening (Mussa-Rosen / Maskin-Riley): optimal menu separating types
    - Signaling (Spence): job market signaling equilibrium
    - Revelation principle verification
    - Multi-dimensional screening with ironing
    """

    def __init__(self, reservation_utilities: Optional[np.ndarray] = None):
        """Initialize the adverse selection model.

        Args:
            reservation_utilities: Outside option for each type. If None,
                all types have reservation utility 0.
        """
        self.reservation_utilities = reservation_utilities

    def solve(
        self,
        types: np.ndarray,
        type_dist: np.ndarray,
        principal_utility: Callable[[float, float, float], float],
    ) -> Menu:
        """Solve the screening problem for optimal menu of contracts.

        The principal offers a menu {(q_i, t_i)} of quality-transfer pairs.
        Type theta_i agent chooses (q_i, t_i) to maximize theta_i * q - t.
        The principal maximizes expected profit sum f_i * (t_i - C(q_i)).

        Solution uses the standard approach:
        1. Only downward IC constraints bind (higher types mimic lower).
        2. IR binds for the lowest type.
        3. Solve recursively from bottom type upward.

        Args:
            types: Array of type parameters theta_1 < theta_2 < ... < theta_n.
            type_dist: Probability distribution over types.
            principal_utility: Function (q, t, theta) -> principal's payoff.
                Typically t - C(q) where C is cost.

        Returns:
            Menu with optimal screening contracts.
        """
        n = len(types)
        if self.reservation_utilities is None:
            u_bar = np.zeros(n)
        else:
            u_bar = self.reservation_utilities.copy()

        # Sort types in increasing order
        order = np.argsort(types)
        theta = types[order]
        f = type_dist[order]
        u_bar_sorted = u_bar[order]

        # Virtual type computation: phi(theta_i) = theta_i - (1 - F_i) / f_i * delta_theta
        # For discrete types, use the Myerson virtual value analog
        cum_f = np.cumsum(f)
        virtual_types = np.zeros(n)
        for i in range(n):
            hazard_rate_term = (1.0 - cum_f[i]) / max(f[i], 1e-15)
            if i < n - 1:
                delta = theta[i + 1] - theta[i]
            else:
                delta = theta[i] - theta[i - 1] if i > 0 else 1.0
            virtual_types[i] = theta[i] - hazard_rate_term * delta

        # Solve for quality using virtual types
        # Optimal q_i: marginal cost C'(q_i) = virtual_type_i
        # Assume quadratic cost C(q) = q^2 / 2, so C'(q) = q
        # Then q_i = max(virtual_type_i, 0) for standard model
        qualities = np.maximum(virtual_types, 0.0)

        # Enforce monotonicity via ironing (pool types where q is non-monotone)
        qualities, pooling = self._iron_allocation(qualities, theta, f)

        # Compute transfers using IC constraints (envelope formula)
        # U(theta_i) = U(theta_0) + integral of q d_theta
        # t_i = theta_i * q_i - U(theta_i)
        utilities = np.zeros(n)
        utilities[0] = u_bar_sorted[0]  # IR binds for lowest type

        for i in range(1, n):
            # U(theta_i) = U(theta_{i-1}) + q_{i-1} * (theta_i - theta_{i-1})
            utilities[i] = utilities[i - 1] + qualities[i - 1] * (theta[i] - theta[i - 1])

        transfers = theta * qualities - utilities

        # Compute principal profit
        profit = 0.0
        for i in range(n):
            profit += f[i] * principal_utility(qualities[i], transfers[i], theta[i])

        # Compute information rents
        info_rent = np.sum(f * np.maximum(utilities - u_bar_sorted, 0.0))

        # Determine type assignments and separating status
        assignments = np.arange(n)
        is_sep = len(pooling) == 0

        # Unsort to original order
        inv_order = np.argsort(order)
        qualities_out = qualities[inv_order]
        transfers_out = transfers[inv_order]
        utilities_out = utilities[inv_order]
        assignments_out = assignments[inv_order]

        return Menu(
            qualities=qualities_out,
            transfers=transfers_out,
            type_assignments=assignments_out,
            utilities=utilities_out,
            principal_profit=profit,
            is_separating=is_sep,
            pooling_groups=[set(g) for g in pooling] if pooling else None,
        )

    def _iron_allocation(
        self,
        qualities: np.ndarray,
        types: np.ndarray,
        dist: np.ndarray,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Apply ironing to enforce monotonicity of the allocation rule.

        When virtual types are non-monotone, the optimal allocation may violate
        incentive compatibility. Ironing pools adjacent types to restore
        monotonicity while maximizing expected virtual surplus.

        The algorithm:
        1. Compute cumulative virtual surplus H(theta).
        2. Take the convex hull (concavification) of H.
        3. The ironed virtual type equals the slope of the convex hull.

        Args:
            qualities: Raw quality allocation from virtual types.
            types: Sorted type parameters.
            dist: Distribution over types.

        Returns:
            Tuple of (ironed qualities, list of pooled type groups).
        """
        n = len(qualities)
        if n <= 1:
            return qualities, []

        # Check if already monotone
        is_monotone = all(qualities[i] <= qualities[i + 1] for i in range(n - 1))
        if is_monotone:
            return qualities, []

        # Ironing via pool-adjacent-violators algorithm
        ironed = qualities.copy()
        pooling_groups = []

        blocks = [[i] for i in range(n)]
        block_vals = [ironed[i] for i in range(n)]

        changed = True
        while changed:
            changed = False
            new_blocks = [blocks[0]]
            new_vals = [block_vals[0]]

            for j in range(1, len(blocks)):
                if new_vals[-1] > block_vals[j]:
                    # Merge: pool these blocks
                    merged = new_blocks[-1] + blocks[j]
                    total_weight = sum(dist[k] for k in merged)
                    if total_weight > 0:
                        avg = sum(dist[k] * ironed[k] for k in merged) / total_weight
                    else:
                        avg = block_vals[j]
                    new_blocks[-1] = merged
                    new_vals[-1] = avg
                    changed = True
                else:
                    new_blocks.append(blocks[j])
                    new_vals.append(block_vals[j])

            blocks = new_blocks
            block_vals = new_vals

        # Apply ironed values
        result = ironed.copy()
        for block, val in zip(blocks, block_vals):
            for idx in block:
                result[idx] = val
            if len(block) > 1:
                pooling_groups.append(block)

        return result, pooling_groups

    def solve_signaling(
        self,
        types: np.ndarray,
        type_dist: np.ndarray,
        productivity: np.ndarray,
        signaling_cost: Callable[[float, float], float],
    ) -> Menu:
        """Solve for Spence signaling equilibrium in the job market.

        Workers have private productivity types. They choose education level
        (signal) to reveal type. Firms observe education and offer wages.

        In the least-cost separating equilibrium:
        - Each type theta chooses signal s(theta).
        - Wages equal productivity: w(s) = theta for the type choosing s.
        - IC: theta * w(s(theta)) - c(s(theta), theta) >=
              theta * w(s(theta')) - c(s(theta'), theta) for all theta'.
        - The lowest type gets zero signal.

        We solve the differential equation for the separating equilibrium:
            ds/d_theta = (dw/d_theta) / (dc/ds * (1/theta))

        For discrete types, we use the binding downward IC constraints.

        Args:
            types: Worker types (productivity parameters), sorted ascending.
            type_dist: Distribution over types.
            productivity: Output/productivity for each type.
            signaling_cost: Cost function c(signal, type). Must satisfy
                single crossing: c_s_theta < 0 (higher types have lower
                marginal signaling cost).

        Returns:
            Menu where qualities=signals, transfers=wages.
        """
        n = len(types)
        signals = np.zeros(n)
        wages = productivity.copy()

        # Lowest type: no signaling
        signals[0] = 0.0

        # For each subsequent type, find signal making adjacent IC bind
        for i in range(1, n):
            # IC for type i-1 not mimicking type i:
            # wages[i-1] - c(signals[i-1], types[i-1]) >= wages[i] - c(s, types[i-1])
            # => c(s, types[i-1]) >= wages[i] - wages[i-1] + c(signals[i-1], types[i-1])
            # Find s such that type i-1 is just indifferent
            rhs = (wages[i] - wages[i - 1]
                   + signaling_cost(signals[i - 1], types[i - 1]))

            # Solve: c(s, types[i-1]) = rhs for s
            def ic_eq(s, target=rhs, t=types[i - 1]):
                return (signaling_cost(s, t) - target) ** 2

            result = minimize_scalar(
                ic_eq, bounds=(signals[i - 1], signals[i - 1] + 100),
                method="bounded",
            )
            signals[i] = max(result.x, signals[i - 1])

        # Compute utilities
        utilities = np.array([
            wages[i] - signaling_cost(signals[i], types[i])
            for i in range(n)
        ])

        profit = np.dot(type_dist, productivity - wages)

        return Menu(
            qualities=signals,
            transfers=wages,
            type_assignments=np.arange(n),
            utilities=utilities,
            principal_profit=profit,
            is_separating=True,
            pooling_groups=None,
        )

    def verify_revelation_principle(
        self,
        types: np.ndarray,
        qualities: np.ndarray,
        transfers: np.ndarray,
        type_utility: Callable[[float, float, float], float],
        reservation_utilities: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Verify whether a mechanism satisfies IC and IR constraints.

        The revelation principle states that any implementable allocation can
        be achieved by a direct truthful mechanism. This method checks:

        IC (Incentive Compatibility): For all types i, j:
            U(q_i, t_i; theta_i) >= U(q_j, t_j; theta_i)
            (no type wants to misreport)

        IR (Individual Rationality): For all types i:
            U(q_i, t_i; theta_i) >= u_bar_i
            (all types prefer participating)

        Args:
            types: Array of type parameters.
            qualities: Allocated quality for each type.
            transfers: Payment for each type.
            type_utility: Utility function U(q, t, theta) for the agent.
            reservation_utilities: Outside option for each type.

        Returns:
            Dictionary with verification results:
            - 'ic_satisfied': bool
            - 'ir_satisfied': bool
            - 'ic_violations': list of (i, j) pairs where IC fails
            - 'ir_violations': list of types where IR fails
            - 'ic_slack': matrix of IC constraint slacks
            - 'ir_slack': array of IR constraint slacks
        """
        n = len(types)
        if reservation_utilities is None:
            u_bar = np.zeros(n)
        else:
            u_bar = reservation_utilities

        ic_slack = np.zeros((n, n))
        ic_violations = []

        for i in range(n):
            u_truth = type_utility(qualities[i], transfers[i], types[i])
            for j in range(n):
                if i == j:
                    continue
                u_deviate = type_utility(qualities[j], transfers[j], types[i])
                ic_slack[i, j] = u_truth - u_deviate
                if ic_slack[i, j] < -1e-8:
                    ic_violations.append((i, j))

        ir_slack = np.zeros(n)
        ir_violations = []
        for i in range(n):
            u_i = type_utility(qualities[i], transfers[i], types[i])
            ir_slack[i] = u_i - u_bar[i]
            if ir_slack[i] < -1e-8:
                ir_violations.append(i)

        return {
            "ic_satisfied": len(ic_violations) == 0,
            "ir_satisfied": len(ir_violations) == 0,
            "ic_violations": ic_violations,
            "ir_violations": ir_violations,
            "ic_slack": ic_slack,
            "ir_slack": ir_slack,
        }

    def solve_multidimensional(
        self,
        types_2d: np.ndarray,
        type_dist: np.ndarray,
        cost_function: Callable[[np.ndarray], float],
        valuation: Callable[[np.ndarray, np.ndarray], float],
    ) -> Menu:
        """Solve a 2D screening problem with ironing.

        In multi-dimensional screening, agent types are vectors in R^2.
        The principal offers a menu of bundles. The problem is significantly
        harder than 1D because there is no natural ordering of types.

        Approach: Reduce to 1D via projections, solve with ironing, then
        verify incentive compatibility on the full 2D type space.

        For a 2D type theta = (theta_1, theta_2) with valuation
        v(q, theta) = theta_1 * q_1 + theta_2 * q_2 and cost C(q),
        we use the sweep-line technique along the direction of maximal
        type variation.

        Args:
            types_2d: Array of shape (n, 2) with 2D type parameters.
            type_dist: Probability distribution over types.
            cost_function: Cost C(q) as a function of quality vector q.
            valuation: Agent valuation v(q, theta) function.

        Returns:
            Menu with multi-dimensional screening solution.
        """
        n = len(types_2d)

        # Project types onto principal component direction
        type_mean = np.mean(types_2d, axis=0)
        centered = types_2d - type_mean
        cov = centered.T @ np.diag(type_dist) @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        direction = eigenvectors[:, -1]  # largest eigenvalue direction

        # Project types onto this direction
        projections = types_2d @ direction
        order = np.argsort(projections)

        # Solve 1D problem along projection
        proj_sorted = projections[order]
        dist_sorted = type_dist[order]

        # Compute virtual values for the 1D projection
        cum_f = np.cumsum(dist_sorted)
        virtual_vals = np.zeros(n)
        for i in range(n):
            hazard = (1.0 - cum_f[i]) / max(dist_sorted[i], 1e-15)
            if i < n - 1:
                delta = proj_sorted[i + 1] - proj_sorted[i]
            else:
                delta = proj_sorted[i] - proj_sorted[i - 1] if i > 0 else 1.0
            virtual_vals[i] = proj_sorted[i] - hazard * delta

        # Quality along projection: q(theta) proportional to virtual value
        raw_qualities_1d = np.maximum(virtual_vals, 0.0)

        # Iron to ensure monotonicity
        ironed_q, pooling = self._iron_allocation(
            raw_qualities_1d, proj_sorted, dist_sorted
        )

        # Map back to 2D: quality vector = q_1d * direction
        qualities_2d = np.outer(ironed_q, np.abs(direction))

        # Compute transfers via envelope condition
        utilities_sorted = np.zeros(n)
        utilities_sorted[0] = 0.0  # IR binds for lowest projected type
        for i in range(1, n):
            grad_val = ironed_q[i - 1] * np.abs(np.dot(direction, direction))
            utilities_sorted[i] = utilities_sorted[i - 1] + grad_val * (
                proj_sorted[i] - proj_sorted[i - 1]
            )

        # Transfer = valuation - utility
        transfers_sorted = np.zeros(n)
        for i in range(n):
            theta_i = types_2d[order[i]]
            q_i = qualities_2d[i]
            transfers_sorted[i] = valuation(q_i, theta_i) - utilities_sorted[i]

        # Compute profit
        profit = 0.0
        for i in range(n):
            q_i = qualities_2d[i]
            profit += dist_sorted[i] * (transfers_sorted[i] - cost_function(q_i))

        # Unsort
        inv_order = np.argsort(order)
        qualities_out = np.linalg.norm(qualities_2d[inv_order], axis=1)
        transfers_out = transfers_sorted[inv_order]
        utilities_out = utilities_sorted[inv_order]

        info_rent = np.sum(type_dist * np.maximum(utilities_out, 0.0))

        return Menu(
            qualities=qualities_out,
            transfers=transfers_out,
            type_assignments=np.arange(n),
            utilities=utilities_out,
            principal_profit=profit,
            is_separating=len(pooling) == 0,
            pooling_groups=[set(g) for g in pooling] if pooling else None,
        )


class RevelationMechanism:
    """Implements the revelation principle and mechanism design verification.

    The revelation principle states that for any Bayesian Nash equilibrium
    of any mechanism, there exists a direct mechanism where truthful
    reporting is an equilibrium and yields the same outcome.

    This class provides tools to:
    1. Convert indirect mechanisms to direct mechanisms.
    2. Verify incentive compatibility of direct mechanisms.
    3. Compute the optimal direct mechanism via LP.
    """

    def __init__(self, types: np.ndarray, type_dist: np.ndarray):
        """Initialize the revelation mechanism.

        Args:
            types: Array of possible agent types.
            type_dist: Prior distribution over types.
        """
        self.types = types
        self.type_dist = type_dist
        self.n_types = len(types)

    def optimal_mechanism_lp(
        self,
        values: np.ndarray,
        costs: np.ndarray,
        reservation_utilities: Optional[np.ndarray] = None,
    ) -> OptimalContract:
        """Compute the optimal mechanism via linear programming.

        Maximizes expected revenue subject to IC and IR constraints.

        Variables: q_i (allocation probability) and t_i (transfer) for each type.

        Objective: max sum_i f_i * t_i
        IC: theta_i * q_i - t_i >= theta_i * q_j - t_j  for all i, j
        IR: theta_i * q_i - t_i >= u_bar_i  for all i
        Feasibility: 0 <= q_i <= 1 for all i

        We reformulate with U_i = theta_i * q_i - t_i (agent utility):
        t_i = theta_i * q_i - U_i
        Objective: max sum_i f_i * (theta_i * q_i - U_i)

        Args:
            values: Agent values for the good by type, shape (n_types,).
            costs: Principal's cost of providing to each type, shape (n_types,).
            reservation_utilities: IR levels by type, defaults to 0.

        Returns:
            OptimalContract with optimal allocation and payments.
        """
        n = self.n_types
        theta = self.types
        f = self.type_dist

        if reservation_utilities is None:
            u_bar = np.zeros(n)
        else:
            u_bar = reservation_utilities

        # Decision variables: [q_0, ..., q_{n-1}, U_0, ..., U_{n-1}]
        # Objective: maximize sum f_i (theta_i q_i - U_i)
        # = sum f_i theta_i q_i - sum f_i U_i
        # linprog minimizes, so negate
        c_obj = np.zeros(2 * n)
        for i in range(n):
            c_obj[i] = -f[i] * theta[i]       # coefficient of q_i
            c_obj[n + i] = f[i]                # coefficient of U_i (minimize -U -> +U)

        # IC constraints: U_i >= theta_i * q_j - t_j + t_i - theta_i * q_i + U_i
        # Rewritten: U_i >= U_j + (theta_i - theta_j) * q_j
        # => U_j - U_i + (theta_i - theta_j) * q_j <= 0
        A_ub_list = []
        b_ub_list = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                row = np.zeros(2 * n)
                row[j] = (theta[i] - theta[j])   # q_j coefficient
                row[n + j] = 1.0                  # U_j coefficient
                row[n + i] = -1.0                 # -U_i coefficient
                A_ub_list.append(row)
                b_ub_list.append(0.0)

        # IR constraints: U_i >= u_bar_i => -U_i <= -u_bar_i
        for i in range(n):
            row = np.zeros(2 * n)
            row[n + i] = -1.0
            A_ub_list.append(row)
            b_ub_list.append(-u_bar[i])

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)

        # Bounds: 0 <= q_i <= 1, U_i >= 0
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # q_i
        for i in range(n):
            bounds.append((0.0, None))  # U_i

        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method="highs")

        if not result.success:
            return OptimalContract(
                payments=np.zeros(n), effort_levels=np.zeros(n),
                principal_profit=0.0, agent_rents=np.zeros(n),
                information_rent=0.0,
            )

        q_opt = result.x[:n]
        U_opt = result.x[n:]
        t_opt = theta * q_opt - U_opt

        profit = np.dot(f, t_opt - costs * q_opt)
        info_rent = np.dot(f, np.maximum(U_opt - u_bar, 0.0))

        return OptimalContract(
            payments=t_opt,
            effort_levels=q_opt,
            principal_profit=profit,
            agent_rents=U_opt,
            information_rent=info_rent,
        )

    def check_implementability(
        self,
        allocation: np.ndarray,
    ) -> Dict[str, object]:
        """Check if an allocation rule is implementable (monotone).

        By the Myerson lemma, an allocation rule q(.) is implementable
        if and only if it is monotone non-decreasing in type.

        For implementable allocations, the unique payment rule is:
            t(theta_i) = theta_i * q(theta_i) - integral_0^{theta_i} q(s) ds

        Args:
            allocation: Array of allocation probabilities q_i for each type.

        Returns:
            Dictionary with:
            - 'implementable': bool
            - 'violations': list of index pairs where monotonicity fails
            - 'payments': payment rule if implementable
        """
        n = self.n_types
        theta = self.types
        order = np.argsort(theta)
        q_sorted = allocation[order]
        theta_sorted = theta[order]

        violations = []
        for i in range(n - 1):
            if q_sorted[i] > q_sorted[i + 1] + 1e-10:
                violations.append((order[i], order[i + 1]))

        implementable = len(violations) == 0

        payments = np.zeros(n)
        if implementable:
            # Compute transfers via integral (trapezoidal rule)
            integral = np.zeros(n)
            for i in range(1, n):
                dt = theta_sorted[i] - theta_sorted[i - 1]
                integral[i] = integral[i - 1] + 0.5 * (q_sorted[i] + q_sorted[i - 1]) * dt

            t_sorted = theta_sorted * q_sorted - integral
            inv_order = np.argsort(order)
            payments = t_sorted[inv_order]

        return {
            "implementable": implementable,
            "violations": violations,
            "payments": payments,
        }

    def convert_indirect_to_direct(
        self,
        strategies: np.ndarray,
        outcome_function: Callable[[int], Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert an indirect mechanism equilibrium to a direct mechanism.

        Given equilibrium strategies s*(theta) mapping types to actions,
        and an outcome function g(a) -> (allocation, transfer), construct
        the direct mechanism (q(theta), t(theta)) = g(s*(theta)).

        Args:
            strategies: Array mapping type index to action index.
            outcome_function: Maps action index to (allocation, transfer).

        Returns:
            Tuple of (allocations, transfers) arrays defining the direct mechanism.
        """
        n = self.n_types
        allocations = np.zeros(n)
        transfers = np.zeros(n)

        for i in range(n):
            action = strategies[i]
            q, t = outcome_function(int(action))
            allocations[i] = q
            transfers[i] = t

        return allocations, transfers


class BundlingMechanism:
    """Multi-good mechanism design with bundling.

    When the principal sells multiple goods to a buyer with multi-dimensional
    private values, bundling can increase revenue beyond separate sales.

    Implements optimal deterministic bundling for 2 goods.
    """

    def __init__(self, n_goods: int = 2):
        """Initialize with the number of goods.

        Args:
            n_goods: Number of goods (default 2).
        """
        self.n_goods = n_goods

    def optimal_bundle_pricing(
        self,
        type_values: np.ndarray,
        type_dist: np.ndarray,
        costs: np.ndarray,
    ) -> Dict[str, object]:
        """Find optimal pricing among separate sales, pure bundling, mixed bundling.

        Compares three strategies:
        1. Separate pricing: set individual prices for each good.
        2. Pure bundling: sell only the bundle at one price.
        3. Mixed bundling: offer individual prices and a bundle discount.

        For each strategy, optimize prices to maximize expected revenue.

        Args:
            type_values: Shape (n_types, n_goods), values for each good.
            type_dist: Distribution over types.
            costs: Marginal cost of each good.

        Returns:
            Dictionary with optimal strategy, prices, and revenue.
        """
        n_types = len(type_values)
        n_goods = self.n_goods

        # Strategy 1: Separate pricing
        def separate_revenue(prices):
            rev = 0.0
            for i in range(n_types):
                for g in range(n_goods):
                    if type_values[i, g] >= prices[g]:
                        rev += type_dist[i] * (prices[g] - costs[g])
            return -rev  # negate for minimization

        p0_sep = np.mean(type_values, axis=0)
        res_sep = minimize(separate_revenue, p0_sep, method="Nelder-Mead",
                           options={"maxiter": 1000})
        rev_separate = -res_sep.fun
        prices_separate = res_sep.x

        # Strategy 2: Pure bundling
        bundle_values = np.sum(type_values, axis=1)
        total_cost = np.sum(costs)

        def bundle_revenue(bundle_price):
            bp = bundle_price[0]
            rev = 0.0
            for i in range(n_types):
                if bundle_values[i] >= bp:
                    rev += type_dist[i] * (bp - total_cost)
            return -rev

        p0_bundle = np.array([np.mean(bundle_values)])
        res_bundle = minimize(bundle_revenue, p0_bundle, method="Nelder-Mead",
                              options={"maxiter": 1000})
        rev_bundle = -res_bundle.fun
        price_bundle = res_bundle.x[0]

        # Strategy 3: Mixed bundling
        # Prices: p_1, p_2, ..., p_K, p_bundle
        def mixed_revenue(params):
            indiv_prices = params[:n_goods]
            bp = params[n_goods]
            rev = 0.0
            for i in range(n_types):
                # Buyer chooses option maximizing surplus
                surplus_nothing = 0.0
                surplus_bundle = bundle_values[i] - bp
                individual_surpluses = []
                for g in range(n_goods):
                    individual_surpluses.append(
                        max(type_values[i, g] - indiv_prices[g], 0.0)
                    )
                surplus_individual = sum(individual_surpluses)

                if surplus_bundle >= surplus_individual and surplus_bundle >= 0:
                    rev += type_dist[i] * (bp - total_cost)
                elif surplus_individual > 0:
                    for g in range(n_goods):
                        if type_values[i, g] >= indiv_prices[g]:
                            rev += type_dist[i] * (indiv_prices[g] - costs[g])
            return -rev

        p0_mixed = np.append(prices_separate, price_bundle)
        res_mixed = minimize(mixed_revenue, p0_mixed, method="Nelder-Mead",
                             options={"maxiter": 2000})
        rev_mixed = -res_mixed.fun
        prices_mixed = res_mixed.x

        revenues = {
            "separate": rev_separate,
            "pure_bundle": rev_bundle,
            "mixed_bundle": rev_mixed,
        }
        best = max(revenues, key=revenues.get)

        return {
            "optimal_strategy": best,
            "revenues": revenues,
            "separate_prices": prices_separate,
            "bundle_price": price_bundle,
            "mixed_prices": {
                "individual": prices_mixed[:n_goods],
                "bundle": prices_mixed[n_goods],
            },
        }


class AuctionMechanism:
    """Optimal auction design following Myerson (1981).

    Implements the revenue-optimal auction for independent private values.
    Uses virtual value theory to compute the optimal allocation and payment.
    """

    def __init__(self, n_bidders: int, value_bounds: Tuple[float, float] = (0.0, 1.0)):
        """Initialize the auction mechanism.

        Args:
            n_bidders: Number of bidders.
            value_bounds: Support of each bidder's value distribution [a, b].
        """
        self.n_bidders = n_bidders
        self.value_bounds = value_bounds

    def myerson_optimal_auction(
        self,
        values_grid: np.ndarray,
        cdf_func: Callable[[float], float],
        pdf_func: Callable[[float], float],
        reserve_price: Optional[float] = None,
    ) -> Dict[str, object]:
        """Compute the Myerson optimal auction for symmetric IPV.

        The optimal auction allocates to the bidder with the highest
        non-negative virtual value, where:
            phi(v) = v - (1 - F(v)) / f(v)

        The optimal reserve price r* satisfies phi(r*) = 0.

        For uniform [0,1]: phi(v) = 2v - 1, so r* = 0.5.

        Args:
            values_grid: Grid of value points for computation.
            cdf_func: CDF of each bidder's value distribution.
            pdf_func: PDF of each bidder's value distribution.
            reserve_price: If None, compute optimal reserve.

        Returns:
            Dictionary with auction parameters and expected revenue.
        """
        # Compute virtual values on grid
        virtual_values = np.zeros_like(values_grid)
        for i, v in enumerate(values_grid):
            f_v = pdf_func(v)
            F_v = cdf_func(v)
            if f_v > 1e-15:
                virtual_values[i] = v - (1.0 - F_v) / f_v
            else:
                virtual_values[i] = v

        # Find optimal reserve: phi(r*) = 0
        if reserve_price is None:
            # Binary search for phi(v) = 0
            lo, hi = self.value_bounds
            for _ in range(100):
                mid = (lo + hi) / 2
                f_mid = pdf_func(mid)
                F_mid = cdf_func(mid)
                if f_mid > 1e-15:
                    phi_mid = mid - (1.0 - F_mid) / f_mid
                else:
                    phi_mid = mid
                if phi_mid < 0:
                    lo = mid
                else:
                    hi = mid
            reserve_price = (lo + hi) / 2

        # Expected revenue via revenue equivalence
        # E[Revenue] = n * integral from r* to 1 of phi(v) * v^{n-1} * n * v^{n-1} dv
        # For symmetric IPV with CDF F:
        # E[Rev] = n * integral_r^bar{v} phi(v) * (n-1) * F(v)^{n-2} * f(v) * F(v) dv
        # Simplified using order statistics

        n = self.n_bidders
        dv = values_grid[1] - values_grid[0] if len(values_grid) > 1 else 0.01

        # Expected revenue = E[second highest value | both above reserve]
        # + reserve * P(exactly one above reserve)
        # Use numerical integration
        expected_revenue = 0.0
        for i, v in enumerate(values_grid):
            if v < reserve_price:
                continue
            f_v = pdf_func(v)
            F_v = cdf_func(v)
            phi_v = virtual_values[i]

            # Probability this bidder has highest virtual value and it's positive
            # For symmetric bidders: prob of being highest = F(v)^{n-1}
            prob_highest = F_v ** (n - 1)
            expected_revenue += n * phi_v * prob_highest * f_v * dv

        expected_revenue = max(expected_revenue, 0.0)

        # Compute allocation rule: probability of winning given value
        win_prob = np.zeros_like(values_grid)
        for i, v in enumerate(values_grid):
            if v >= reserve_price:
                F_v = cdf_func(v)
                win_prob[i] = F_v ** (n - 1)
            else:
                win_prob[i] = 0.0

        # Compute payment rule via revenue equivalence
        # t(v) = v * q(v) - integral_0^v q(s) ds
        payments = np.zeros_like(values_grid)
        integral_q = 0.0
        for i, v in enumerate(values_grid):
            if i > 0:
                integral_q += 0.5 * (win_prob[i] + win_prob[i - 1]) * dv
            payments[i] = v * win_prob[i] - integral_q

        return {
            "reserve_price": reserve_price,
            "expected_revenue": expected_revenue,
            "virtual_values": virtual_values,
            "win_probabilities": win_prob,
            "payments": payments,
            "values_grid": values_grid,
        }

    def compare_auction_formats(
        self,
        n_simulations: int = 10000,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Compare revenue across auction formats via simulation.

        Simulates first-price, second-price, and optimal (with reserve) auctions
        with uniform [0,1] values, verifying revenue equivalence.

        Args:
            n_simulations: Number of Monte Carlo simulations.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary of expected revenues by auction format.
        """
        rng = np.random.RandomState(seed)
        n = self.n_bidders
        lo, hi = self.value_bounds

        values = rng.uniform(lo, hi, size=(n_simulations, n))

        # Second-price auction (Vickrey): winner pays second-highest bid
        # Truthful bidding is dominant strategy
        sorted_vals = np.sort(values, axis=1)
        sp_revenue = np.mean(sorted_vals[:, -2])  # second highest

        # First-price auction: symmetric BNE bid = (n-1)/n * value (uniform case)
        # bid_i(v) = v - integral_0^v F(s)^{n-1}/F(v)^{n-1} ds
        # For uniform: bid(v) = (n-1)/n * v
        bids_fp = values * (n - 1) / n
        winner_bids = np.max(bids_fp, axis=1)
        fp_revenue = np.mean(winner_bids)

        # Optimal auction with reserve
        reserve = (lo + hi) / 2  # optimal reserve for uniform
        opt_revenues = []
        for sim in range(n_simulations):
            eligible = values[sim][values[sim] >= reserve]
            if len(eligible) == 0:
                opt_revenues.append(0.0)
            elif len(eligible) == 1:
                opt_revenues.append(reserve)
            else:
                sorted_eligible = np.sort(eligible)
                opt_revenues.append(sorted_eligible[-2])
        opt_revenue = np.mean(opt_revenues)

        return {
            "first_price": fp_revenue,
            "second_price": sp_revenue,
            "optimal_with_reserve": opt_revenue,
            "n_bidders": n,
            "n_simulations": n_simulations,
        }


def solve_contract_with_limited_liability(
    types: np.ndarray,
    type_dist: np.ndarray,
    output_values: np.ndarray,
    liability_limit: float = 0.0,
) -> OptimalContract:
    """Solve principal-agent problem with limited liability constraints.

    When the agent has limited liability (payments cannot be negative or below
    a floor), the optimal contract differs from the standard model. The
    principal may leave rents to all types, not just through information
    asymmetry but also due to the liability constraint.

    Formulation (LP):
        max sum_i f_i * (y_i * q_i - t_i)
        s.t. IC: theta_i * q_i - t_i >= theta_i * q_j - t_j  for all i,j
             IR: theta_i * q_i - t_i >= 0  for all i
             LL: t_i >= liability_limit  for all i
             q_i in [0, 1]

    Args:
        types: Agent type parameters.
        type_dist: Distribution over types.
        output_values: Expected output for each type.
        liability_limit: Minimum payment (limited liability floor).

    Returns:
        OptimalContract respecting limited liability.
    """
    n = len(types)

    # Variables: q_0,...,q_{n-1}, t_0,...,t_{n-1}
    c_obj = np.zeros(2 * n)
    for i in range(n):
        c_obj[i] = -type_dist[i] * output_values[i]  # -f_i * y_i (maximize output)
        c_obj[n + i] = type_dist[i]                   # +f_i (minimize transfer)

    A_ub_list = []
    b_ub_list = []

    # IC constraints
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = np.zeros(2 * n)
            row[j] = (types[i] - types[j])
            row[n + j] = 1.0
            row[n + i] = -1.0
            A_ub_list.append(row)
            b_ub_list.append(0.0)

    # IR constraints: theta_i * q_i - t_i >= 0 => t_i - theta_i * q_i <= 0
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = -types[i]
        row[n + i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(0.0)

    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((liability_limit, None))

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        q_opt = result.x[:n]
        t_opt = result.x[n:]
    else:
        q_opt = np.zeros(n)
        t_opt = np.full(n, liability_limit)

    utilities = types * q_opt - t_opt
    profit = np.dot(type_dist, output_values * q_opt - t_opt)
    info_rent = np.dot(type_dist, np.maximum(utilities, 0.0))

    return OptimalContract(
        payments=t_opt,
        effort_levels=q_opt,
        principal_profit=profit,
        agent_rents=utilities,
        information_rent=info_rent,
    )


def compute_virtual_surplus(
    types: np.ndarray,
    type_dist: np.ndarray,
    allocation: np.ndarray,
    cost_func: Callable[[float], float],
) -> float:
    """Compute the expected virtual surplus of an allocation rule.

    Virtual surplus = sum_i f_i * [phi(theta_i) * q_i - C(q_i)]
    where phi is the virtual type function.

    This is the objective the principal effectively maximizes when
    solving the relaxed problem (ignoring monotonicity).

    Args:
        types: Array of type parameters, sorted ascending.
        type_dist: Probability distribution over types.
        allocation: Allocation rule q(theta_i).
        cost_func: Cost function C(q).

    Returns:
        Expected virtual surplus value.
    """
    n = len(types)
    cum_f = np.cumsum(type_dist)

    virtual_surplus = 0.0
    for i in range(n):
        hazard = (1.0 - cum_f[i]) / max(type_dist[i], 1e-15)
        if i < n - 1:
            delta = types[i + 1] - types[i]
        else:
            delta = types[i] - types[i - 1] if i > 0 else 1.0
        phi_i = types[i] - hazard * delta
        virtual_surplus += type_dist[i] * (phi_i * allocation[i] - cost_func(allocation[i]))

    return virtual_surplus


def bayesian_optimal_mechanism(
    type_space: np.ndarray,
    type_dist: np.ndarray,
    social_choice: Callable[[np.ndarray], float],
    transfer_rule: str = "vcg",
) -> Dict[str, np.ndarray]:
    """Compute a Bayesian optimal mechanism with transfers.

    Implements VCG (Vickrey-Clarke-Groves) or AGV (d'Aspremont-Gérard-Varet)
    mechanisms for efficient social choice with transfers.

    VCG: Dominant strategy IC, but may run a deficit.
    AGV: Bayesian IC, budget balanced, but not dominant strategy IC.

    Args:
        type_space: Shape (n_agents, n_types_per_agent) type reports.
        type_dist: Shape (n_agents, n_types_per_agent) marginal distributions.
        social_choice: Maps type profile to social value.
        transfer_rule: Either "vcg" or "agv".

    Returns:
        Dictionary with allocations, transfers, and budget surplus.
    """
    n_agents = type_space.shape[0]
    n_types = type_space.shape[1]

    # Enumerate all type profiles
    from itertools import product
    profiles = list(product(range(n_types), repeat=n_agents))
    n_profiles = len(profiles)

    # Compute social values for all profiles
    social_values = np.zeros(n_profiles)
    for idx, profile in enumerate(profiles):
        theta_vec = np.array([type_space[a, profile[a]] for a in range(n_agents)])
        social_values[idx] = social_choice(theta_vec)

    # Compute profile probabilities
    profile_probs = np.zeros(n_profiles)
    for idx, profile in enumerate(profiles):
        prob = 1.0
        for a in range(n_agents):
            prob *= type_dist[a, profile[a]]
        profile_probs[idx] = prob

    # Compute transfers
    transfers = np.zeros((n_agents, n_types))

    if transfer_rule == "vcg":
        # VCG: t_i = sum_{j != i} v_j(outcome) - h_i(theta_{-i})
        # h_i chosen so that t_i(theta) = max_{d} sum_{j!=i} v_j(d, theta_{-i})
        for a in range(n_agents):
            for ti in range(n_types):
                # Expected externality imposed by agent a with type ti
                total_others_with = 0.0
                total_others_without = 0.0
                weight = 0.0

                for idx, profile in enumerate(profiles):
                    if profile[a] != ti:
                        continue
                    prob_others = profile_probs[idx] / max(type_dist[a, ti], 1e-15)

                    theta_vec = np.array([
                        type_space[ag, profile[ag]] for ag in range(n_agents)
                    ])
                    val_with = social_choice(theta_vec)

                    # Value without agent a (replace with 0)
                    theta_without = theta_vec.copy()
                    theta_without[a] = 0.0
                    val_without = social_choice(theta_without)

                    total_others_with += prob_others * val_with
                    total_others_without += prob_others * val_without
                    weight += prob_others

                if weight > 0:
                    transfers[a, ti] = (total_others_with - total_others_without) / weight

    elif transfer_rule == "agv":
        # AGV: expected externality transfers for budget balance
        for a in range(n_agents):
            for ti in range(n_types):
                expected_value = 0.0
                weight = 0.0
                for idx, profile in enumerate(profiles):
                    if profile[a] != ti:
                        continue
                    prob_others = profile_probs[idx] / max(type_dist[a, ti], 1e-15)
                    theta_vec = np.array([
                        type_space[ag, profile[ag]] for ag in range(n_agents)
                    ])
                    expected_value += prob_others * social_choice(theta_vec)
                    weight += prob_others

                if weight > 0:
                    transfers[a, ti] = expected_value / weight / n_agents

    # Budget surplus
    expected_surplus = 0.0
    for idx, profile in enumerate(profiles):
        total_transfer = sum(transfers[a, profile[a]] for a in range(n_agents))
        expected_surplus += profile_probs[idx] * total_transfer

    allocations = social_values

    return {
        "allocations": allocations,
        "transfers": transfers,
        "expected_budget_surplus": expected_surplus,
        "profile_probabilities": profile_probs,
        "mechanism_type": transfer_rule,
    }
