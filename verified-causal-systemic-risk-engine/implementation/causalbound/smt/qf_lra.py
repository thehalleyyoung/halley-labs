"""
QFLRAEncoder: specialized QF_LRA (quantifier-free linear real arithmetic)
encoding for LP bound verification.

Encodes LP feasibility, dual optimality, column-generation termination
(no negative reduced cost), and Farkas infeasibility certificates as
Z3 QF_LRA formulas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import z3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rv(val: float) -> z3.RatNumRef:
    """Convert a Python float to a Z3 rational literal."""
    return z3.RealVal(str(val))


def _var(name: str) -> z3.ArithRef:
    return z3.Real(name)


# ---------------------------------------------------------------------------
# QFLRAEncoder
# ---------------------------------------------------------------------------

class QFLRAEncoder:
    """
    Specialized encoder for QF_LRA verification of LP certificates.

    Translates primal/dual feasibility, optimality gaps, reduced-cost
    non-negativity, and Farkas infeasibility lemma into Z3 assertions.

    All ``encode_*`` methods return a ``z3.BoolRef`` conjunction ready
    for assertion into an incremental solver.

    Parameters
    ----------
    epsilon : float
        Numerical tolerance for approximate equality checks.
    prefix : str
        Variable-name prefix to prevent collisions.
    """

    def __init__(
        self,
        epsilon: float = 1e-9,
        prefix: str = "lra",
    ) -> None:
        self.epsilon = epsilon
        self._prefix = prefix
        self._var_cache: Dict[str, z3.ExprRef] = {}

    # ------------------------------------------------------------------
    # Variable helpers
    # ------------------------------------------------------------------

    def _real(self, name: str) -> z3.ArithRef:
        key = f"{self._prefix}_{name}"
        if key not in self._var_cache:
            self._var_cache[key] = z3.Real(key)
        return self._var_cache[key]  # type: ignore[return-value]

    def _eps(self) -> z3.RatNumRef:
        return _rv(self.epsilon)

    # ------------------------------------------------------------------
    # 1. LP feasibility:  Ax <= b, x >= 0
    # ------------------------------------------------------------------

    def encode_lp_feasible(
        self,
        A: List[List[float]],
        b: List[float],
        x: List[float],
    ) -> z3.BoolRef:
        """
        Assert that *x* is a feasible solution to Ax ≤ b, x ≥ 0.

        Parameters
        ----------
        A : list of list of float   Constraint matrix (m × n).
        b : list of float           Right-hand side (length m).
        x : list of float           Candidate solution (length n).

        Returns
        -------
        z3.BoolRef
            Conjunction asserting feasibility.
        """
        m = len(A)
        n = len(x)
        eps = self._eps()

        x_vars = [self._real(f"x_{j}") for j in range(n)]

        clauses: List[z3.BoolRef] = []

        # Fix x to its given values
        for j in range(n):
            clauses.append(x_vars[j] == _rv(x[j]))

        # Non-negativity
        for j in range(n):
            clauses.append(x_vars[j] >= _rv(0.0))

        # Ax <= b  (with tolerance)
        for i in range(m):
            row = A[i]
            lhs = z3.Sum(
                [_rv(row[j]) * x_vars[j] for j in range(min(len(row), n))]
            )
            clauses.append(lhs <= _rv(b[i]) + eps)

        return z3.And(*clauses)

    def encode_lp_feasible_equality(
        self,
        A: List[List[float]],
        b: List[float],
        x: List[float],
    ) -> z3.BoolRef:
        """Assert feasibility for equality constraints Ax = b, x ≥ 0."""
        m = len(A)
        n = len(x)
        eps = self._eps()

        x_vars = [self._real(f"xeq_{j}") for j in range(n)]
        clauses: List[z3.BoolRef] = []

        for j in range(n):
            clauses.append(x_vars[j] == _rv(x[j]))
            clauses.append(x_vars[j] >= _rv(0.0))

        for i in range(m):
            row = A[i]
            lhs = z3.Sum(
                [_rv(row[j]) * x_vars[j] for j in range(min(len(row), n))]
            )
            clauses.append(lhs >= _rv(b[i]) - eps)
            clauses.append(lhs <= _rv(b[i]) + eps)

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 2. LP optimality (primal-dual)
    # ------------------------------------------------------------------

    def encode_lp_optimal(
        self,
        c: List[float],
        A: List[List[float]],
        b: List[float],
        x: List[float],
        dual: List[float],
    ) -> z3.BoolRef:
        """
        Assert primal-dual optimality:

        1. Primal feasibility:  Ax ≤ b, x ≥ 0
        2. Dual feasibility:    Aᵀy ≥ c, y ≥ 0
        3. Strong duality:      cᵀx = bᵀy  (within tolerance)

        Parameters
        ----------
        c : list of float   Objective coefficients (length n).
        A : list of list of float   Constraint matrix (m × n).
        b : list of float   RHS (length m).
        x : list of float   Primal solution (length n).
        dual : list of float   Dual solution (length m).
        """
        m = len(A)
        n = len(c)
        eps = self._eps()

        x_vars = [self._real(f"opt_x_{j}") for j in range(n)]
        y_vars = [self._real(f"opt_y_{i}") for i in range(m)]

        clauses: List[z3.BoolRef] = []

        # Fix primal values
        for j in range(n):
            clauses.append(x_vars[j] == _rv(x[j]))
            clauses.append(x_vars[j] >= _rv(0.0))

        # Fix dual values
        for i in range(m):
            clauses.append(y_vars[i] == _rv(dual[i]))
            clauses.append(y_vars[i] >= _rv(0.0))

        # Primal feasibility: Ax <= b
        for i in range(m):
            row = A[i]
            lhs = z3.Sum(
                [_rv(row[j]) * x_vars[j] for j in range(min(len(row), n))]
            )
            clauses.append(lhs <= _rv(b[i]) + eps)

        # Dual feasibility: Aᵀy >= c
        for j in range(n):
            col_sum = z3.Sum(
                [_rv(A[i][j]) * y_vars[i] for i in range(m) if j < len(A[i])]
            )
            clauses.append(col_sum >= _rv(c[j]) - eps)

        # Strong duality: cᵀx = bᵀy
        primal_obj = z3.Sum([_rv(c[j]) * x_vars[j] for j in range(n)])
        dual_obj = z3.Sum([_rv(b[i]) * y_vars[i] for i in range(m)])
        clauses.append(primal_obj >= dual_obj - eps)
        clauses.append(primal_obj <= dual_obj + eps)

        return z3.And(*clauses)

    def encode_weak_duality(
        self,
        c: List[float],
        A: List[List[float]],
        b: List[float],
        x: List[float],
        dual: List[float],
    ) -> z3.BoolRef:
        """
        Assert weak duality only: cᵀx ≥ bᵀy for a minimisation problem
        (or cᵀx ≤ bᵀy for maximisation, encoded symmetrically here).
        """
        n = len(c)
        m = len(b)
        eps = self._eps()

        x_vars = [self._real(f"wd_x_{j}") for j in range(n)]
        y_vars = [self._real(f"wd_y_{i}") for i in range(m)]

        clauses: List[z3.BoolRef] = []
        for j in range(n):
            clauses.append(x_vars[j] == _rv(x[j]))
        for i in range(m):
            clauses.append(y_vars[i] == _rv(dual[i]))

        primal_obj = z3.Sum([_rv(c[j]) * x_vars[j] for j in range(n)])
        dual_obj = z3.Sum([_rv(b[i]) * y_vars[i] for i in range(m)])

        # Weak duality: primal >= dual (for minimisation)
        clauses.append(primal_obj >= dual_obj - eps)

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 3. Complementary slackness
    # ------------------------------------------------------------------

    def encode_complementary_slackness(
        self,
        A: List[List[float]],
        b: List[float],
        x: List[float],
        dual: List[float],
    ) -> z3.BoolRef:
        """
        Assert complementary slackness conditions:

        For each constraint i:  y_i * (b_i - A_i·x) = 0
        For each variable j:   x_j * (Aᵀ_j·y - c_j) = 0

        (The second set requires *c* but we encode only the first here
        since c is not always available in this signature.)
        """
        m = len(A)
        n = len(x)
        eps = self._eps()

        x_vars = [self._real(f"cs_x_{j}") for j in range(n)]
        y_vars = [self._real(f"cs_y_{i}") for i in range(m)]

        clauses: List[z3.BoolRef] = []
        for j in range(n):
            clauses.append(x_vars[j] == _rv(x[j]))
        for i in range(m):
            clauses.append(y_vars[i] == _rv(dual[i]))

        # Primal complementary slackness
        for i in range(m):
            row = A[i]
            slack = _rv(b[i]) - z3.Sum(
                [_rv(row[j]) * x_vars[j] for j in range(min(len(row), n))]
            )
            product = y_vars[i] * slack
            clauses.append(product >= -eps)
            clauses.append(product <= eps)

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 4. No negative reduced cost  (column-generation termination)
    # ------------------------------------------------------------------

    def encode_no_negative_reduced_cost(
        self,
        dual: List[float],
        columns: List[List[float]],
        costs: Optional[List[float]] = None,
    ) -> z3.BoolRef:
        """
        Assert that no column has negative reduced cost.

        For column generation: the reduced cost of column *j* is
        ``c_j - yᵀ·a_j``.  If all reduced costs ≥ 0, the current
        restricted master is optimal.

        Parameters
        ----------
        dual : list of float   Dual values (length m).
        columns : list of list of float   Candidate columns (each length m).
        costs : list of float, optional
            Original cost coefficients (one per column).
            Defaults to zero if not provided.
        """
        m = len(dual)
        k = len(columns)
        eps = self._eps()

        y_vars = [self._real(f"rc_y_{i}") for i in range(m)]
        clauses: List[z3.BoolRef] = []

        for i in range(m):
            clauses.append(y_vars[i] == _rv(dual[i]))

        for j in range(k):
            col = columns[j]
            col_len = min(len(col), m)
            dual_contribution = z3.Sum(
                [_rv(col[i]) * y_vars[i] for i in range(col_len)]
            )
            cost_j = _rv(costs[j]) if costs and j < len(costs) else _rv(0.0)
            reduced_cost = cost_j - dual_contribution
            rc_var = self._real(f"rc_{j}")
            clauses.append(rc_var == reduced_cost)
            clauses.append(rc_var >= -eps)

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 5. Farkas lemma (infeasibility certificate)
    # ------------------------------------------------------------------

    def encode_farkas(
        self,
        A: List[List[float]],
        b: List[float],
        y: List[float],
    ) -> z3.BoolRef:
        """
        Encode a Farkas infeasibility certificate.

        Farkas' lemma: the system Ax ≤ b has no solution iff ∃ y ≥ 0
        such that Aᵀy = 0 and bᵀy < 0.

        Parameters
        ----------
        A : list of list of float   Constraint matrix (m × n).
        b : list of float   RHS (length m).
        y : list of float   Farkas multipliers (length m).
        """
        m = len(A)
        n = max((len(row) for row in A), default=0)
        eps = self._eps()

        y_vars = [self._real(f"fk_y_{i}") for i in range(m)]
        clauses: List[z3.BoolRef] = []

        # Fix y values and enforce non-negativity
        for i in range(m):
            clauses.append(y_vars[i] == _rv(y[i]))
            clauses.append(y_vars[i] >= _rv(0.0))

        # Aᵀy = 0  (for each column j)
        for j in range(n):
            col_sum = z3.Sum(
                [_rv(A[i][j]) * y_vars[i] for i in range(m) if j < len(A[i])]
            )
            clauses.append(col_sum >= -eps)
            clauses.append(col_sum <= eps)

        # bᵀy < 0
        rhs_sum = z3.Sum([_rv(b[i]) * y_vars[i] for i in range(m)])
        clauses.append(rhs_sum < _rv(0.0))

        return z3.And(*clauses)

    def encode_farkas_alternative(
        self,
        A: List[List[float]],
        b: List[float],
        y: List[float],
    ) -> z3.BoolRef:
        """
        Alternative Farkas encoding for equality-constrained systems.

        For Ax = b, x ≥ 0:  infeasible iff ∃ y with Aᵀy ≥ 0, bᵀy < 0.
        """
        m = len(A)
        n = max((len(row) for row in A), default=0)
        eps = self._eps()

        y_vars = [self._real(f"fka_y_{i}") for i in range(m)]
        clauses: List[z3.BoolRef] = []

        for i in range(m):
            clauses.append(y_vars[i] == _rv(y[i]))

        # Aᵀy >= 0
        for j in range(n):
            col_sum = z3.Sum(
                [_rv(A[i][j]) * y_vars[i] for i in range(m) if j < len(A[i])]
            )
            clauses.append(col_sum >= -eps)

        # bᵀy < 0
        rhs_sum = z3.Sum([_rv(b[i]) * y_vars[i] for i in range(m)])
        clauses.append(rhs_sum < _rv(0.0))

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 6. Bound certification  (lower/upper via dual)
    # ------------------------------------------------------------------

    def encode_bound_certificate(
        self,
        c: List[float],
        A: List[List[float]],
        b: List[float],
        dual: List[float],
        claimed_bound: float,
        is_lower: bool = True,
    ) -> z3.BoolRef:
        """
        Certify that *claimed_bound* is a valid lower (or upper) bound
        on the LP objective.

        For a **lower bound** on min cᵀx:
          dual feasible (Aᵀy ≥ c, y ≥ 0) implies bᵀy ≤ cᵀx,
          so bᵀy is a lower bound.

        For an **upper bound** on max cᵀx:
          symmetric argument.
        """
        m = len(A)
        n = len(c)
        eps = self._eps()

        y_vars = [self._real(f"bc_y_{i}") for i in range(m)]
        clauses: List[z3.BoolRef] = []

        for i in range(m):
            clauses.append(y_vars[i] == _rv(dual[i]))
            clauses.append(y_vars[i] >= _rv(0.0))

        # Dual feasibility: Aᵀy >= c
        for j in range(n):
            col_sum = z3.Sum(
                [_rv(A[i][j]) * y_vars[i] for i in range(m) if j < len(A[i])]
            )
            clauses.append(col_sum >= _rv(c[j]) - eps)

        dual_obj = z3.Sum([_rv(b[i]) * y_vars[i] for i in range(m)])
        bound_var = self._real("bc_bound")
        clauses.append(bound_var == _rv(claimed_bound))

        if is_lower:
            # Lower bound: dual objective <= claimed bound
            clauses.append(dual_obj >= bound_var - eps)
        else:
            clauses.append(dual_obj <= bound_var + eps)

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 7. LP relaxation gap
    # ------------------------------------------------------------------

    def encode_lp_gap(
        self,
        primal_obj: float,
        dual_obj: float,
        tolerance: float,
    ) -> z3.BoolRef:
        """
        Assert that the duality gap is within *tolerance*.
        """
        pv = self._real("gap_primal")
        dv = self._real("gap_dual")
        tol = self._real("gap_tol")
        gap = self._real("gap_val")

        return z3.And(
            pv == _rv(primal_obj),
            dv == _rv(dual_obj),
            tol == _rv(tolerance),
            gap == pv - dv,
            gap >= _rv(0.0),
            gap <= tol,
        )

    # ------------------------------------------------------------------
    # 8. Variable bounds
    # ------------------------------------------------------------------

    def encode_variable_bounds(
        self,
        lower_bounds: List[float],
        upper_bounds: List[float],
        values: List[float],
    ) -> z3.BoolRef:
        """
        Assert ``lower[j] ≤ x[j] ≤ upper[j]`` for each variable j.
        """
        n = len(values)
        clauses: List[z3.BoolRef] = []
        for j in range(n):
            xj = self._real(f"vb_x_{j}")
            clauses.append(xj == _rv(values[j]))
            if j < len(lower_bounds):
                clauses.append(xj >= _rv(lower_bounds[j]))
            if j < len(upper_bounds):
                clauses.append(xj <= _rv(upper_bounds[j]))
        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 9. Basis verification
    # ------------------------------------------------------------------

    def encode_basis_feasibility(
        self,
        A: List[List[float]],
        b: List[float],
        basis_indices: List[int],
        basis_values: List[float],
    ) -> z3.BoolRef:
        """
        Assert that the basic feasible solution corresponding to
        *basis_indices* is feasible: the basis columns times the
        basis values equal b, and all basis values are non-negative.
        """
        m = len(b)
        k = len(basis_indices)
        eps = self._eps()

        clauses: List[z3.BoolRef] = []
        bv = [self._real(f"bas_v_{j}") for j in range(k)]

        for j in range(k):
            clauses.append(bv[j] == _rv(basis_values[j]))
            clauses.append(bv[j] >= _rv(0.0))

        # B * xB = b
        for i in range(m):
            row_sum = z3.Sum(
                [
                    _rv(A[i][basis_indices[j]]) * bv[j]
                    for j in range(k)
                    if basis_indices[j] < len(A[i])
                ]
            )
            clauses.append(row_sum >= _rv(b[i]) - eps)
            clauses.append(row_sum <= _rv(b[i]) + eps)

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_variable_count(self) -> int:
        return len(self._var_cache)

    def clear_cache(self) -> None:
        self._var_cache.clear()
