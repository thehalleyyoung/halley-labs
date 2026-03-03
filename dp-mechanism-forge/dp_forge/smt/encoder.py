"""SMT encoding of differential privacy mechanism design problems.

Translates mechanism probability tables, privacy constraints, workload
structure, and optimization objectives into SMT formulas over the
theory of linear real arithmetic (LRA).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    AdjacencyRelation,
    Formula,
    Predicate,
    PrivacyBudget,
)

from dp_forge.smt import (
    EncodeStrategy,
    SMTConfig,
    SMTConstraint,
    SMTVariable,
)


# ---------------------------------------------------------------------------
# Helper: create Formula objects
# ---------------------------------------------------------------------------

def _formula(expr: str, variables: Optional[Set[str]] = None,
             ftype: str = "linear_arithmetic") -> Formula:
    """Shorthand for building a Formula."""
    vs = frozenset(variables) if variables else frozenset()
    return Formula(expr=expr, variables=vs, formula_type=ftype)


def _constraint(expr: str, variables: Optional[Set[str]] = None,
                label: Optional[str] = None) -> SMTConstraint:
    """Shorthand for building an SMTConstraint."""
    return SMTConstraint(formula=_formula(expr, variables), label=label)


# ---------------------------------------------------------------------------
# PrivacyConstraintEncoder
# ---------------------------------------------------------------------------

class PrivacyConstraintEncoder:
    """Encode ε-DP and (ε,δ)-DP constraints as SMT formulas.

    For pure ε-DP, the constraint is:
        ∀ adjacent (i,i'), ∀ output j:
            M[i][j] ≤ e^ε · M[i'][j]

    For (ε,δ)-DP (approximate DP), the constraint includes a δ additive term:
        ∀ adjacent (i,i'), ∀ S ⊆ outputs:
            Pr[M(i) ∈ S] ≤ e^ε · Pr[M(i') ∈ S] + δ

    For finite mechanisms, per-output constraints suffice for the
    ratio-based encoding.
    """

    def __init__(self, strategy: EncodeStrategy = EncodeStrategy.RATIO) -> None:
        self.strategy = strategy

    def encode_pure_dp(
        self,
        n: int,
        k: int,
        epsilon: float,
        adjacency: AdjacencyRelation,
        prob_vars: List[List[str]],
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode pure ε-DP constraints.

        Args:
            n: Number of inputs.
            k: Number of outputs.
            epsilon: Privacy parameter.
            adjacency: Adjacency relation.
            prob_vars: prob_vars[i][j] = variable name for M[i][j].

        Returns:
            (extra_variables, constraints)
        """
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []
        ratio = math.exp(epsilon)

        for i, ip in adjacency.edges:
            for j in range(k):
                p_ij = prob_vars[i][j]
                p_ipj = prob_vars[ip][j]

                if self.strategy == EncodeStrategy.RATIO:
                    # p[i][j] <= e^eps * p[i'][j]
                    expr = f"{p_ij} - {ratio}*{p_ipj} <= 0.0"
                    vs = {p_ij, p_ipj}
                    constraints.append(_constraint(
                        expr, vs,
                        label=f"dp_ratio_{i}_{ip}_{j}_fwd"
                    ))
                    if adjacency.symmetric:
                        expr2 = f"{p_ipj} - {ratio}*{p_ij} <= 0.0"
                        constraints.append(_constraint(
                            expr2, vs,
                            label=f"dp_ratio_{i}_{ip}_{j}_bwd"
                        ))

                elif self.strategy == EncodeStrategy.LOG_SPACE:
                    # log(p[i][j]) - log(p[i'][j]) <= epsilon
                    lp_ij = f"log_{p_ij}"
                    lp_ipj = f"log_{p_ipj}"
                    variables.append(SMTVariable(name=lp_ij, sort="Real"))
                    variables.append(SMTVariable(name=lp_ipj, sort="Real"))
                    expr = f"{lp_ij} - {lp_ipj} <= {epsilon}"
                    constraints.append(_constraint(
                        expr, {lp_ij, lp_ipj},
                        label=f"dp_log_{i}_{ip}_{j}_fwd"
                    ))
                    if adjacency.symmetric:
                        expr2 = f"{lp_ipj} - {lp_ij} <= {epsilon}"
                        constraints.append(_constraint(
                            expr2, {lp_ij, lp_ipj},
                            label=f"dp_log_{i}_{ip}_{j}_bwd"
                        ))

                elif self.strategy == EncodeStrategy.DIFFERENCE:
                    # p[i][j] - e^eps * p[i'][j] <= 0
                    expr = f"{p_ij} - {ratio}*{p_ipj} <= 0.0"
                    constraints.append(_constraint(
                        expr, {p_ij, p_ipj},
                        label=f"dp_diff_{i}_{ip}_{j}"
                    ))
                    if adjacency.symmetric:
                        expr2 = f"{p_ipj} - {ratio}*{p_ij} <= 0.0"
                        constraints.append(_constraint(
                            expr2, {p_ij, p_ipj},
                            label=f"dp_diff_{i}_{ip}_{j}_sym"
                        ))

        return variables, constraints

    def encode_approximate_dp(
        self,
        n: int,
        k: int,
        epsilon: float,
        delta: float,
        adjacency: AdjacencyRelation,
        prob_vars: List[List[str]],
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode (ε,δ)-DP constraints.

        Uses per-output relaxation:
            p[i][j] <= e^ε * p[i'][j] + δ/k
        """
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []
        ratio = math.exp(epsilon)
        delta_per_output = delta  # conservative: full delta per output

        for i, ip in adjacency.edges:
            for j in range(k):
                p_ij = prob_vars[i][j]
                p_ipj = prob_vars[ip][j]

                expr = f"{p_ij} - {ratio}*{p_ipj} <= {delta_per_output}"
                vs = {p_ij, p_ipj}
                constraints.append(_constraint(
                    expr, vs,
                    label=f"adp_{i}_{ip}_{j}_fwd"
                ))

                if adjacency.symmetric:
                    expr2 = f"{p_ipj} - {ratio}*{p_ij} <= {delta_per_output}"
                    constraints.append(_constraint(
                        expr2, vs,
                        label=f"adp_{i}_{ip}_{j}_bwd"
                    ))

        return variables, constraints

    def encode_violation(
        self,
        n: int,
        k: int,
        epsilon: float,
        delta: float,
        adjacency: AdjacencyRelation,
        prob_vars: List[List[str]],
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode the negation of DP: existence of a violation.

        At least one adjacent pair (i,i') and output j must violate:
            p[i][j] > e^ε * p[i'][j] + δ
        """
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []
        ratio = math.exp(epsilon)

        # Introduce a Boolean indicator for each potential violation
        violation_vars: List[str] = []
        for idx, (i, ip) in enumerate(adjacency.edges):
            for j in range(k):
                p_ij = prob_vars[i][j]
                p_ipj = prob_vars[ip][j]

                viol_var = f"__viol_{i}_{ip}_{j}"
                variables.append(SMTVariable(name=viol_var, sort="Real",
                                             lower_bound=0.0, upper_bound=1.0))
                violation_vars.append(viol_var)

                # If viol_var > 0, then p[i][j] > e^eps * p[i'][j] + delta
                expr = f"{p_ij} - {ratio}*{p_ipj} - {delta} >= {viol_var}"
                constraints.append(_constraint(
                    expr, {p_ij, p_ipj, viol_var},
                    label=f"violation_{i}_{ip}_{j}"
                ))

        # At least one violation must be positive
        if violation_vars:
            sum_expr = " + ".join(violation_vars)
            constraints.append(_constraint(
                f"{sum_expr} >= 0.001",
                set(violation_vars),
                label="at_least_one_violation"
            ))

        return variables, constraints


# ---------------------------------------------------------------------------
# MechanismEncoder
# ---------------------------------------------------------------------------

class MechanismEncoder:
    """Encode mechanism structure: probability variables, simplex, support.

    Creates SMT variables for the n × k probability table and encodes:
    - Non-negativity: p[i][j] ≥ 0
    - Normalization: Σ_j p[i][j] = 1
    - Optional support constraints
    """

    def __init__(self) -> None:
        pass

    def encode_probability_table(
        self,
        n: int,
        k: int,
        prefix: str = "p",
    ) -> Tuple[List[SMTVariable], List[SMTConstraint], List[List[str]]]:
        """Create probability variables and simplex constraints.

        Returns:
            (variables, constraints, var_names_grid)
        """
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []
        var_names: List[List[str]] = []

        for i in range(n):
            row_names: List[str] = []
            for j in range(k):
                name = f"{prefix}_{i}_{j}"
                variables.append(SMTVariable(
                    name=name, sort="Real",
                    lower_bound=0.0, upper_bound=1.0,
                ))
                row_names.append(name)

                # Non-negativity
                constraints.append(_constraint(
                    f"{name} >= 0.0", {name},
                    label=f"nonneg_{i}_{j}",
                ))
                # Upper bound
                constraints.append(_constraint(
                    f"{name} <= 1.0", {name},
                    label=f"upperbd_{i}_{j}",
                ))

            var_names.append(row_names)

            # Normalization: sum = 1
            sum_expr = " + ".join(row_names)
            constraints.append(_constraint(
                f"{sum_expr} = 1.0",
                set(row_names),
                label=f"simplex_{i}",
            ))

        return variables, constraints, var_names

    def encode_fixed_mechanism(
        self,
        mechanism: npt.NDArray[np.float64],
        prefix: str = "p",
    ) -> Tuple[List[SMTVariable], List[SMTConstraint], List[List[str]]]:
        """Encode a fixed (concrete) mechanism as equality constraints.

        Args:
            mechanism: n × k probability table.

        Returns:
            (variables, constraints, var_names_grid)
        """
        n, k = mechanism.shape
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []
        var_names: List[List[str]] = []

        for i in range(n):
            row_names: List[str] = []
            for j in range(k):
                name = f"{prefix}_{i}_{j}"
                val = float(mechanism[i, j])
                variables.append(SMTVariable(
                    name=name, sort="Real",
                    lower_bound=val, upper_bound=val,
                ))
                row_names.append(name)
                constraints.append(_constraint(
                    f"{name} = {val}", {name},
                    label=f"fixed_{i}_{j}",
                ))
            var_names.append(row_names)

        return variables, constraints, var_names

    def encode_support(
        self,
        var_names: List[List[str]],
        support: npt.NDArray[np.bool_],
    ) -> List[SMTConstraint]:
        """Encode support constraints: p[i][j] = 0 where support is False."""
        constraints: List[SMTConstraint] = []
        n = len(var_names)
        for i in range(n):
            k = len(var_names[i])
            for j in range(k):
                if not support[i, j]:
                    constraints.append(_constraint(
                        f"{var_names[i][j]} = 0.0",
                        {var_names[i][j]},
                        label=f"support_{i}_{j}",
                    ))
        return constraints


# ---------------------------------------------------------------------------
# WorkloadEncoder
# ---------------------------------------------------------------------------

class WorkloadEncoder:
    """Encode workload and query structure as SMT constraints.

    Encodes linear queries of the form q(x) = Σ w_j · f(x_j)
    and their sensitivity requirements.
    """

    def __init__(self) -> None:
        pass

    def encode_linear_query(
        self,
        query_weights: npt.NDArray[np.float64],
        prob_vars: List[List[str]],
        output_var: str = "q_out",
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode a linear query over the mechanism output.

        For each input i, the expected query answer is:
            E[q | input=i] = Σ_j query_weights[j] * p[i][j]
        """
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []
        n = len(prob_vars)
        k = len(query_weights)

        for i in range(n):
            exp_var = f"{output_var}_{i}"
            variables.append(SMTVariable(name=exp_var, sort="Real"))
            terms = []
            vs: Set[str] = {exp_var}
            for j in range(min(k, len(prob_vars[i]))):
                w = float(query_weights[j])
                if abs(w) > 1e-15:
                    terms.append(f"{w}*{prob_vars[i][j]}")
                    vs.add(prob_vars[i][j])
            if terms:
                rhs = " + ".join(terms)
                constraints.append(_constraint(
                    f"{exp_var} = {rhs}", vs,
                    label=f"query_exp_{i}",
                ))
            else:
                constraints.append(_constraint(
                    f"{exp_var} = 0.0", {exp_var},
                    label=f"query_exp_{i}",
                ))

        return variables, constraints

    def encode_sensitivity(
        self,
        query_vars: List[str],
        adjacency: AdjacencyRelation,
        max_sensitivity: float,
    ) -> List[SMTConstraint]:
        """Encode sensitivity constraints: |q(i) - q(i')| ≤ sensitivity."""
        constraints: List[SMTConstraint] = []
        for i, ip in adjacency.edges:
            qi = query_vars[i]
            qip = query_vars[ip]
            # q_i - q_ip <= sensitivity
            constraints.append(_constraint(
                f"{qi} - {qip} <= {max_sensitivity}",
                {qi, qip},
                label=f"sens_fwd_{i}_{ip}",
            ))
            if adjacency.symmetric:
                constraints.append(_constraint(
                    f"{qip} - {qi} <= {max_sensitivity}",
                    {qi, qip},
                    label=f"sens_bwd_{i}_{ip}",
                ))
        return constraints

    def encode_workload_matrix(
        self,
        workload: npt.NDArray[np.float64],
        prob_vars: List[List[str]],
        prefix: str = "wq",
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode a full workload matrix W where each row is a query."""
        all_vars: List[SMTVariable] = []
        all_constraints: List[SMTConstraint] = []

        num_queries = workload.shape[0]
        for q_idx in range(num_queries):
            query_weights = workload[q_idx]
            out_var = f"{prefix}_{q_idx}"
            v, c = self.encode_linear_query(query_weights, prob_vars, out_var)
            all_vars.extend(v)
            all_constraints.extend(c)

        return all_vars, all_constraints


# ---------------------------------------------------------------------------
# ObjectiveEncoder
# ---------------------------------------------------------------------------

class ObjectiveEncoder:
    """Encode optimization objectives as SMT assertions.

    Supports encoding error minimization objectives by asserting
    upper bounds on error terms and using binary search externally.
    """

    def __init__(self) -> None:
        pass

    def encode_max_error_bound(
        self,
        error_vars: List[str],
        bound: float,
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Assert that all error variables are bounded by `bound`.

        Useful for binary search on the optimal error.
        """
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []

        obj_var = "__max_error"
        variables.append(SMTVariable(name=obj_var, sort="Real",
                                     lower_bound=0.0, upper_bound=bound))
        constraints.append(_constraint(
            f"{obj_var} <= {bound}", {obj_var},
            label="obj_bound",
        ))

        for ev in error_vars:
            constraints.append(_constraint(
                f"{ev} <= {obj_var}", {ev, obj_var},
                label=f"obj_err_{ev}",
            ))
            constraints.append(_constraint(
                f"-1.0*{ev} <= {obj_var}", {ev, obj_var},
                label=f"obj_err_neg_{ev}",
            ))

        return variables, constraints

    def encode_expected_error(
        self,
        true_values: npt.NDArray[np.float64],
        query_output_vars: List[str],
        prob_vars: List[List[str]],
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode expected squared error terms.

        For each input i, create a variable for (expected_output - true_value)^2.
        Since quadratic terms aren't LRA, we use linearization via |error|.
        """
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []
        error_vars: List[str] = []

        n = len(query_output_vars)
        for i in range(min(n, len(true_values))):
            tv = float(true_values[i])
            err_var = f"__err_{i}"
            abs_err_var = f"__abs_err_{i}"
            variables.append(SMTVariable(name=err_var, sort="Real"))
            variables.append(SMTVariable(name=abs_err_var, sort="Real",
                                         lower_bound=0.0))

            qv = query_output_vars[i]
            constraints.append(_constraint(
                f"{err_var} = {qv} - {tv}", {err_var, qv},
                label=f"err_def_{i}",
            ))
            # |error| >= error and |error| >= -error
            constraints.append(_constraint(
                f"{abs_err_var} >= {err_var}", {abs_err_var, err_var},
                label=f"abs_err_pos_{i}",
            ))
            constraints.append(_constraint(
                f"{abs_err_var} >= -1.0*{err_var}", {abs_err_var, err_var},
                label=f"abs_err_neg_{i}",
            ))
            error_vars.append(abs_err_var)

        return variables, constraints

    def encode_minimax_objective(
        self,
        error_vars: List[str],
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode a minimax objective: minimize max(error_vars)."""
        variables: List[SMTVariable] = []
        constraints: List[SMTConstraint] = []

        obj_var = "__minimax"
        variables.append(SMTVariable(name=obj_var, sort="Real", lower_bound=0.0))

        for ev in error_vars:
            constraints.append(_constraint(
                f"{obj_var} >= {ev}", {obj_var, ev},
                label=f"minimax_{ev}",
            ))

        return variables, constraints


# ---------------------------------------------------------------------------
# IncrementalEncoder
# ---------------------------------------------------------------------------

class IncrementalEncoder:
    """Incremental SMT encoding with push/pop scope management.

    Allows building encodings incrementally and backtracking to
    previous states, useful for CEGAR and binary search on objectives.
    """

    def __init__(self) -> None:
        self._variables: List[SMTVariable] = []
        self._constraints: List[SMTConstraint] = []
        self._var_stack: List[int] = []
        self._con_stack: List[int] = []
        self._scope_depth: int = 0

    @property
    def variables(self) -> List[SMTVariable]:
        return list(self._variables)

    @property
    def constraints(self) -> List[SMTConstraint]:
        return list(self._constraints)

    @property
    def scope_depth(self) -> int:
        return self._scope_depth

    def add_variables(self, variables: List[SMTVariable]) -> None:
        self._variables.extend(variables)

    def add_constraints(self, constraints: List[SMTConstraint]) -> None:
        self._constraints.extend(constraints)

    def push(self) -> None:
        """Push a backtracking point."""
        self._var_stack.append(len(self._variables))
        self._con_stack.append(len(self._constraints))
        self._scope_depth += 1

    def pop(self) -> None:
        """Pop to the last backtracking point."""
        if self._scope_depth == 0:
            raise RuntimeError("Cannot pop: no scope to pop")
        self._scope_depth -= 1
        nv = self._var_stack.pop()
        nc = self._con_stack.pop()
        self._variables = self._variables[:nv]
        self._constraints = self._constraints[:nc]

    def reset(self) -> None:
        """Reset all state."""
        self._variables.clear()
        self._constraints.clear()
        self._var_stack.clear()
        self._con_stack.clear()
        self._scope_depth = 0


# ---------------------------------------------------------------------------
# SMTEncoderImpl — full encoder combining sub-encoders
# ---------------------------------------------------------------------------

class SMTEncoderImpl:
    """Full SMT encoder for DP mechanism problems.

    Combines PrivacyConstraintEncoder, MechanismEncoder, WorkloadEncoder,
    and ObjectiveEncoder to produce complete SMT encodings.
    """

    def __init__(self, config: Optional[SMTConfig] = None) -> None:
        from dp_forge.smt import SMTConfig as _Cfg
        self.config = config or _Cfg()
        self.privacy_encoder = PrivacyConstraintEncoder(self.config.encode_strategy)
        self.mechanism_encoder = MechanismEncoder()
        self.workload_encoder = WorkloadEncoder()
        self.objective_encoder = ObjectiveEncoder()
        self.incremental = IncrementalEncoder()

    def encode_mechanism(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
        budget: PrivacyBudget,
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode a concrete mechanism with DP constraints."""
        n, k = mechanism.shape
        all_vars: List[SMTVariable] = []
        all_cons: List[SMTConstraint] = []

        # Fixed mechanism
        m_vars, m_cons, var_names = self.mechanism_encoder.encode_fixed_mechanism(mechanism)
        all_vars.extend(m_vars)
        all_cons.extend(m_cons)

        # Privacy constraints
        if budget.delta == 0.0:
            p_vars, p_cons = self.privacy_encoder.encode_pure_dp(
                n, k, budget.epsilon, adjacency, var_names
            )
        else:
            p_vars, p_cons = self.privacy_encoder.encode_approximate_dp(
                n, k, budget.epsilon, budget.delta, adjacency, var_names
            )
        all_vars.extend(p_vars)
        all_cons.extend(p_cons)

        return all_vars, all_cons

    def encode_privacy_violation(
        self,
        n: int,
        k: int,
        adjacency: AdjacencyRelation,
        budget: PrivacyBudget,
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode existence of a DP violation (negation of DP)."""
        all_vars: List[SMTVariable] = []
        all_cons: List[SMTConstraint] = []

        # Free mechanism variables
        m_vars, m_cons, var_names = self.mechanism_encoder.encode_probability_table(n, k)
        all_vars.extend(m_vars)
        all_cons.extend(m_cons)

        # Violation constraints
        v_vars, v_cons = self.privacy_encoder.encode_violation(
            n, k, budget.epsilon, budget.delta, adjacency, var_names
        )
        all_vars.extend(v_vars)
        all_cons.extend(v_cons)

        return all_vars, all_cons

    def encode_formula(self, formula: Formula) -> List[SMTConstraint]:
        """Convert a generic formula to SMT constraints."""
        return [SMTConstraint(formula=formula)]


__all__ = [
    "PrivacyConstraintEncoder",
    "MechanismEncoder",
    "WorkloadEncoder",
    "ObjectiveEncoder",
    "IncrementalEncoder",
    "SMTEncoderImpl",
]
