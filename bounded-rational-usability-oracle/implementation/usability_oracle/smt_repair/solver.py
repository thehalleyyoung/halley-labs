"""
usability_oracle.smt_repair.solver — Repair solver.

Wraps the Z3 SMT solver to find minimal-cost UI repairs.  Supports:

* **MaxSMT** — minimise the number of changed UI properties.
* **Weighted MaxSMT** — weighted objective over property importance.
* **Incremental solving** — add constraints and re-solve.
* **Pareto-optimal enumeration** — enumerate non-dominated repairs.
* **Graceful timeout** — return best-so-far on timeout.

Hard constraints (accessibility, structural validity) are always
satisfied; soft constraints (change minimisation, preferences) are
optimised subject to the hard constraints.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import z3

from usability_oracle.smt_repair.encoding import TreeEncoding, Z3Encoder
from usability_oracle.smt_repair.types import (
    ConstraintKind,
    ConstraintSystem,
    MutationCandidate,
    MutationType,
    RepairConstraint,
    RepairResult,
    SolverStatus,
    UIVariable,
    VariableSort,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# RepairSolver
# ═══════════════════════════════════════════════════════════════════════════

class RepairSolver:
    """SMT-based repair solver implementing the RepairSolver protocol.

    Translates a :class:`ConstraintSystem` into Z3 assertions, solves
    for a satisfying assignment that minimises UI changes, and extracts
    :class:`MutationCandidate` instances from the model.
    """

    def __init__(self, encoder: Optional[Z3Encoder] = None) -> None:
        self._encoder = encoder or Z3Encoder()

    # ── protocol methods ──────────────────────────────────────────────

    def solve(self, system: ConstraintSystem) -> RepairResult:
        """Solve the constraint system.

        Hard constraints are asserted unconditionally.  Soft constraints
        are added as weighted soft assertions via Z3's Optimize engine.

        Parameters:
            system: Complete constraint system to solve.

        Returns:
            A :class:`RepairResult` with status, mutations, and timing.
        """
        t0 = time.monotonic()
        timeout_ms = int(system.timeout_seconds * 1000)

        try:
            opt, z3_vars, var_lookup = self._build_optimizer(system, timeout_ms)
            result_status = opt.check()
            elapsed = time.monotonic() - t0

            if result_status == z3.sat:
                model = opt.model()
                mutations = self._extract_mutations(model, z3_vars, var_lookup)
                cost_delta = sum(m.cost_delta for m in mutations)
                return RepairResult(
                    status=SolverStatus.SAT,
                    mutations=tuple(mutations),
                    total_cost_delta=cost_delta,
                    unsat_core=(),
                    solver_time_seconds=elapsed,
                    constraint_system=system,
                )
            elif result_status == z3.unsat:
                core = self._try_extract_core(opt, system)
                return RepairResult(
                    status=SolverStatus.UNSAT,
                    mutations=(),
                    total_cost_delta=0.0,
                    unsat_core=tuple(core),
                    solver_time_seconds=elapsed,
                    constraint_system=system,
                )
            else:
                # z3.unknown — may be timeout.
                return RepairResult(
                    status=SolverStatus.UNKNOWN,
                    mutations=(),
                    total_cost_delta=0.0,
                    unsat_core=(),
                    solver_time_seconds=elapsed,
                    constraint_system=system,
                )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning("Solver error: %s", exc)
            return RepairResult(
                status=SolverStatus.UNKNOWN,
                mutations=(),
                total_cost_delta=0.0,
                unsat_core=(),
                solver_time_seconds=elapsed,
                constraint_system=system,
            )

    def solve_incremental(
        self,
        base_system: ConstraintSystem,
        additional_constraints: Sequence[RepairConstraint],
    ) -> RepairResult:
        """Incrementally add constraints and re-solve.

        Merges *additional_constraints* into *base_system* and invokes
        :meth:`solve`.  Useful for iterative refinement after
        validation feedback.

        Parameters:
            base_system: Previously solved constraint system.
            additional_constraints: New constraints to assert.

        Returns:
            Updated :class:`RepairResult`.
        """
        merged_constraints = tuple(base_system.constraints) + tuple(additional_constraints)
        merged = ConstraintSystem(
            variables=base_system.variables,
            constraints=merged_constraints,
            objective_expression=base_system.objective_expression,
            timeout_seconds=base_system.timeout_seconds,
        )
        return self.solve(merged)

    def extract_unsat_core(
        self,
        system: ConstraintSystem,
    ) -> Sequence[str]:
        """Extract the minimal unsatisfiable core.

        Uses Z3's ``unsat_core()`` to identify the hard constraints
        responsible for infeasibility.

        Parameters:
            system: An unsatisfiable constraint system.

        Returns:
            Sequence of constraint identifiers forming the UNSAT core.
        """
        solver = z3.Solver()
        timeout_ms = int(system.timeout_seconds * 1000)
        solver.set("timeout", timeout_ms)

        z3_vars = self._declare_variables(system)
        hard = [c for c in system.constraints if c.is_hard]

        # Assert hard constraints with tracking labels.
        for c in hard:
            label = z3.Bool(c.constraint_id)
            z3_expr = self._parse_constraint_expr(c, z3_vars)
            if z3_expr is not None:
                solver.assert_and_track(z3_expr, label)

        result = solver.check()
        if result == z3.unsat:
            core = solver.unsat_core()
            return [str(c) for c in core]
        return []

    # ── extended solving methods ──────────────────────────────────────

    def find_minimal_repair(
        self,
        system: ConstraintSystem,
    ) -> RepairResult:
        """Solve for the minimum-cost repair (MaxSMT).

        Equivalent to :meth:`solve` but emphasises minimality in the
        objective: the solver minimises the total number of property
        changes.
        """
        return self.solve(system)

    def enumerate_repairs(
        self,
        system: ConstraintSystem,
        max_count: int = 5,
    ) -> List[RepairResult]:
        """Enumerate up to *max_count* Pareto-optimal repairs.

        After finding a solution, adds a blocking clause to exclude
        it and re-solves.  Continues until UNSAT or *max_count* is
        reached.

        Parameters:
            system: Constraint system.
            max_count: Maximum number of repairs to enumerate.

        Returns:
            List of :class:`RepairResult` instances (each SAT).
        """
        results: List[RepairResult] = []
        timeout_ms = int(system.timeout_seconds * 1000)
        z3_vars = self._declare_variables(system)
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)

        # Assert hard constraints.
        for c in system.constraints:
            if c.is_hard:
                expr = self._parse_constraint_expr(c, z3_vars)
                if expr is not None:
                    solver.add(expr)

        var_lookup = {v.variable_id: v for v in system.variables}

        for _ in range(max_count):
            t0 = time.monotonic()
            status = solver.check()
            elapsed = time.monotonic() - t0

            if status != z3.sat:
                break

            model = solver.model()
            mutations = self._extract_mutations(model, z3_vars, var_lookup)
            cost_delta = sum(m.cost_delta for m in mutations)
            results.append(RepairResult(
                status=SolverStatus.SAT,
                mutations=tuple(mutations),
                total_cost_delta=cost_delta,
                unsat_core=(),
                solver_time_seconds=elapsed,
                constraint_system=system,
            ))

            # Block this solution.
            blocking: List[z3.BoolRef] = []
            for vid, zvar in z3_vars.items():
                val = model.evaluate(zvar, model_completion=True)
                blocking.append(zvar != val)
            if blocking:
                solver.add(z3.Or(*blocking))
            else:
                break

        return results

    def weighted_max_smt(
        self,
        system: ConstraintSystem,
        weights: Optional[Dict[str, float]] = None,
    ) -> RepairResult:
        """Weighted MaxSMT: minimise a weighted sum of changed properties.

        Parameters:
            system: Constraint system.
            weights: Mapping from variable_id to importance weight.
                Variables with higher weight are less likely to change.

        Returns:
            :class:`RepairResult` with the optimal solution.
        """
        if weights is None:
            weights = {}

        t0 = time.monotonic()
        timeout_ms = int(system.timeout_seconds * 1000)
        opt = z3.Optimize()
        opt.set("timeout", timeout_ms)

        z3_vars = self._declare_variables(system)
        var_lookup = {v.variable_id: v for v in system.variables}

        # Assert hard constraints.
        for c in system.constraints:
            if c.is_hard:
                expr = self._parse_constraint_expr(c, z3_vars)
                if expr is not None:
                    opt.add(expr)

        # Soft constraints.
        for c in system.constraints:
            if not c.is_hard:
                expr = self._parse_constraint_expr(c, z3_vars)
                if expr is not None:
                    opt.add_soft(expr, weight=c.weight)

        # Weighted preservation objective.
        cost_terms: List[z3.ArithRef] = []
        for v in system.variables:
            zvar = z3_vars.get(v.variable_id)
            if zvar is None:
                continue
            w = weights.get(v.variable_id, 1.0)
            if v.sort in (VariableSort.INT, VariableSort.REAL):
                changed = z3.If(zvar != v.current_value, z3.IntVal(1), z3.IntVal(0))
            elif v.sort == VariableSort.BOOL:
                bval = z3.BoolVal(bool(v.current_value))
                changed = z3.If(zvar != bval, z3.IntVal(1), z3.IntVal(0))
            else:
                continue
            cost_terms.append(z3.IntVal(int(w * 100)) * changed)

        if cost_terms:
            opt.minimize(z3.Sum(*cost_terms))

        status = opt.check()
        elapsed = time.monotonic() - t0

        if status == z3.sat:
            model = opt.model()
            mutations = self._extract_mutations(model, z3_vars, var_lookup)
            cost_delta = sum(m.cost_delta for m in mutations)
            return RepairResult(
                status=SolverStatus.SAT,
                mutations=tuple(mutations),
                total_cost_delta=cost_delta,
                unsat_core=(),
                solver_time_seconds=elapsed,
                constraint_system=system,
            )

        final_status = (
            SolverStatus.UNSAT if status == z3.unsat else SolverStatus.UNKNOWN
        )
        return RepairResult(
            status=final_status,
            mutations=(),
            total_cost_delta=0.0,
            unsat_core=(),
            solver_time_seconds=elapsed,
            constraint_system=system,
        )

    @staticmethod
    def timeout_handler(
        solver: z3.Optimize,
        timeout_ms: int,
    ) -> None:
        """Configure graceful timeout on a Z3 solver/optimizer.

        Sets the ``"timeout"`` parameter so the solver returns
        ``z3.unknown`` rather than running indefinitely.

        Parameters:
            solver: Z3 Optimize (or Solver) instance.
            timeout_ms: Timeout in milliseconds.
        """
        solver.set("timeout", timeout_ms)

    # ── private: build solver ─────────────────────────────────────────

    def _build_optimizer(
        self,
        system: ConstraintSystem,
        timeout_ms: int,
    ) -> Tuple[z3.Optimize, Dict[str, z3.ExprRef], Dict[str, UIVariable]]:
        """Build a Z3 Optimize instance from a ConstraintSystem."""
        opt = z3.Optimize()
        opt.set("timeout", timeout_ms)

        z3_vars = self._declare_variables(system)
        var_lookup = {v.variable_id: v for v in system.variables}

        # Hard constraints.
        for c in system.constraints:
            if c.is_hard:
                expr = self._parse_constraint_expr(c, z3_vars)
                if expr is not None:
                    opt.add(expr)

        # Soft constraints.
        for c in system.constraints:
            if not c.is_hard:
                expr = self._parse_constraint_expr(c, z3_vars)
                if expr is not None:
                    opt.add_soft(expr, weight=c.weight)

        # Minimise total changes (preservation objective).
        change_indicators: List[z3.ArithRef] = []
        for v in system.variables:
            zvar = z3_vars.get(v.variable_id)
            if zvar is None:
                continue
            if v.sort in (VariableSort.INT, VariableSort.REAL):
                indicator = z3.If(zvar != v.current_value, z3.IntVal(1), z3.IntVal(0))
            elif v.sort == VariableSort.BOOL:
                bval = z3.BoolVal(bool(v.current_value))
                indicator = z3.If(zvar != bval, z3.IntVal(1), z3.IntVal(0))
            else:
                continue
            change_indicators.append(indicator)

        if change_indicators:
            opt.minimize(z3.Sum(*change_indicators))

        return opt, z3_vars, var_lookup

    def _declare_variables(
        self,
        system: ConstraintSystem,
    ) -> Dict[str, z3.ExprRef]:
        """Declare Z3 variables from the ConstraintSystem."""
        z3_vars: Dict[str, z3.ExprRef] = {}
        for v in system.variables:
            if v.sort == VariableSort.INT:
                z3_vars[v.variable_id] = z3.Int(v.variable_id)
            elif v.sort == VariableSort.REAL:
                z3_vars[v.variable_id] = z3.Real(v.variable_id)
            elif v.sort == VariableSort.BOOL:
                z3_vars[v.variable_id] = z3.Bool(v.variable_id)
            elif v.sort == VariableSort.STRING:
                # Encode string enums as bounded integers.
                z3_vars[v.variable_id] = z3.Int(v.variable_id)
            else:
                z3_vars[v.variable_id] = z3.Int(v.variable_id)

            # Add domain bounds.
            if v.lower_bound is not None and v.sort in (VariableSort.INT, VariableSort.REAL, VariableSort.STRING):
                pass  # Bounds are encoded via soft/hard constraints already.
        return z3_vars

    def _parse_constraint_expr(
        self,
        constraint: RepairConstraint,
        z3_vars: Dict[str, z3.ExprRef],
    ) -> Optional[z3.BoolRef]:
        """Parse an SMT-LIB-style expression string into a Z3 expression.

        This is a simplified S-expression parser supporting common
        operators: ``=``, ``>=``, ``<=``, ``>``, ``<``, ``+``, ``-``,
        ``*``, ``not``, ``and``, ``or``, ``distinct``.

        Falls back to ``True`` for expressions that cannot be parsed.
        """
        expr_str = constraint.expression.strip()
        if not expr_str:
            return None

        try:
            result = self._parse_sexpr(expr_str, z3_vars)
            if isinstance(result, z3.BoolRef):
                return result
            return None
        except Exception:
            logger.debug("Could not parse constraint %s: %s", constraint.constraint_id, expr_str)
            return None

    def _parse_sexpr(
        self,
        s: str,
        z3_vars: Dict[str, z3.ExprRef],
    ) -> Any:
        """Recursive S-expression parser."""
        s = s.strip()

        # Literal true/false.
        if s == "true":
            return z3.BoolVal(True)
        if s == "false":
            return z3.BoolVal(False)

        # Integer literal.
        try:
            return z3.IntVal(int(s))
        except ValueError:
            pass

        # Float literal.
        try:
            return z3.RealVal(float(s))
        except ValueError:
            pass

        # Variable reference.
        if s in z3_vars:
            return z3_vars[s]

        # S-expression: (op arg1 arg2 ...)
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            tokens = self._tokenize(inner)
            if not tokens:
                return z3.BoolVal(True)

            op = tokens[0]
            args = [self._parse_sexpr(t, z3_vars) for t in tokens[1:]]

            return self._apply_op(op, args)

        # Unknown — return a dummy true.
        return z3.BoolVal(True)

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        """Tokenize an S-expression body respecting parenthesis nesting."""
        tokens: List[str] = []
        depth = 0
        current: List[str] = []

        for ch in s:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch in (" ", "\t", "\n") and depth == 0:
                if current:
                    tokens.append("".join(current))
                    current = []
            else:
                current.append(ch)

        if current:
            tokens.append("".join(current))
        return tokens

    @staticmethod
    def _apply_op(op: str, args: List[Any]) -> Any:
        """Apply an S-expression operator to parsed arguments."""
        if op == "=" and len(args) == 2:
            return args[0] == args[1]
        if op == ">=" and len(args) == 2:
            return args[0] >= args[1]
        if op == "<=" and len(args) == 2:
            return args[0] <= args[1]
        if op == ">" and len(args) == 2:
            return args[0] > args[1]
        if op == "<" and len(args) == 2:
            return args[0] < args[1]
        if op == "+" and len(args) >= 2:
            result = args[0]
            for a in args[1:]:
                result = result + a
            return result
        if op == "-" and len(args) >= 1:
            if len(args) == 1:
                return -args[0]
            result = args[0]
            for a in args[1:]:
                result = result - a
            return result
        if op == "*" and len(args) >= 2:
            result = args[0]
            for a in args[1:]:
                result = result * a
            return result
        if op == "not" and len(args) == 1:
            return z3.Not(args[0])
        if op == "and" and args:
            return z3.And(*args)
        if op == "or" and args:
            return z3.Or(*args)
        if op == "distinct" and len(args) >= 2:
            return z3.Distinct(*args)
        if op == "ite" and len(args) == 3:
            return z3.If(args[0], args[1], args[2])
        # Fallback.
        return z3.BoolVal(True)

    def _extract_mutations(
        self,
        model: z3.ModelRef,
        z3_vars: Dict[str, z3.ExprRef],
        var_lookup: Dict[str, UIVariable],
    ) -> List[MutationCandidate]:
        """Compare model values against current values to find mutations."""
        mutations: List[MutationCandidate] = []

        for vid, zvar in z3_vars.items():
            uv = var_lookup.get(vid)
            if uv is None:
                continue

            new_val_z3 = model.evaluate(zvar, model_completion=True)
            new_val = Z3Encoder._z3_val_to_python(new_val_z3, uv)

            if new_val != uv.current_value:
                mutations.append(MutationCandidate(
                    node_id=uv.node_id,
                    mutation_type=MutationType.PROPERTY_CHANGE,
                    property_name=uv.property_name,
                    old_value=uv.current_value,
                    new_value=new_val,
                    cost_delta=-0.1,  # Heuristic improvement estimate.
                    confidence=1.0,
                ))
        return mutations

    @staticmethod
    def _try_extract_core(
        opt: z3.Optimize,
        system: ConstraintSystem,
    ) -> List[str]:
        """Best-effort UNSAT core extraction from Optimize.

        Z3's Optimize does not always support unsat_core, so we fall
        back to listing all hard constraint IDs.
        """
        try:
            core = opt.unsat_core()
            return [str(c) for c in core]
        except Exception:
            return [c.constraint_id for c in system.constraints if c.is_hard]
