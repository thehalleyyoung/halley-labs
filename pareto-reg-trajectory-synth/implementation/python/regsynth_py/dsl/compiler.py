"""Compiler that transforms a RegSynth DSL AST into a constraint problem.

The emitted *ConstraintProblem* can be serialised to JSON for downstream
solvers or to SMT-LIB2 for formal verification.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from regsynth_py.dsl.ast_nodes import (
    ASTNode,
    BinaryOp,
    ComposeMode,
    CompositionDecl,
    ConstraintDecl,
    Declaration,
    Expression,
    FrameworkType,
    Identifier,
    JurisdictionDecl,
    Literal,
    ObligationDecl,
    ObligationType,
    Program,
    RiskLevel,
    StrategyDecl,
    TemporalExpr,
    UnaryOp,
)


# ---------------------------------------------------------------------------
# Constraint representation
# ---------------------------------------------------------------------------

class Direction(enum.Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Variable:
    """A decision variable in the constraint problem."""

    name: str
    domain: tuple[Any, ...]  # e.g. (0, 1) for binary, (0.0, inf) for cost

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "domain": list(self.domain)}

    def to_smt2(self) -> str:
        if self.domain == (0, 1):
            return f"(declare-const {_smt_id(self.name)} Bool)"
        lo, hi = self.domain
        if isinstance(lo, float) or isinstance(hi, float):
            return f"(declare-const {_smt_id(self.name)} Real)"
        return f"(declare-const {_smt_id(self.name)} Int)"


@dataclass
class Constraint:
    """A single constraint over one or more variables."""

    name: str
    variables: list[str]
    relation: str  # human-readable or SMT fragment

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "variables": self.variables, "relation": self.relation}

    def to_smt2(self) -> str:
        return f"(assert {self.relation})  ; {self.name}"


@dataclass
class ObjectiveFunction:
    """A linear objective over a subset of variables."""

    name: str
    variables: list[str]
    coefficients: list[float]
    direction: Direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variables": self.variables,
            "coefficients": self.coefficients,
            "direction": self.direction.value,
        }

    def to_smt2(self) -> str:
        terms = " ".join(
            f"(* {c} {_smt_id(v)})" for c, v in zip(self.coefficients, self.variables)
        )
        total = f"(+ {terms})" if len(self.variables) > 1 else terms
        directive = "minimize" if self.direction is Direction.MINIMIZE else "maximize"
        return f"({directive} {total})  ; {self.name}"


@dataclass
class ConstraintProblem:
    """Complete constraint model ready for solving."""

    variables: list[Variable] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    objectives: list[ObjectiveFunction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": [v.to_dict() for v in self.variables],
            "constraints": [c.to_dict() for c in self.constraints],
            "objectives": [o.to_dict() for o in self.objectives],
        }


# ---------------------------------------------------------------------------
# Risk level numeric mapping (higher = worse)
# ---------------------------------------------------------------------------

_RISK_VALUE: dict[RiskLevel, int] = {
    RiskLevel.MINIMAL: 1,
    RiskLevel.LIMITED: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.UNACCEPTABLE: 4,
}


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_date(value: str) -> Optional[date]:
    if not _ISO_DATE_RE.match(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _date_ordinal(d: date) -> int:
    return d.toordinal()


def _smt_id(name: str) -> str:
    sanitised = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"|{sanitised}|" if sanitised[0].isdigit() else sanitised


# ---------------------------------------------------------------------------
# Expression evaluation (for constant-folding literals)
# ---------------------------------------------------------------------------

def _eval_literal(expr: Expression) -> Optional[float]:
    if isinstance(expr, Literal):
        if expr.literal_type in ("int", "float"):
            return float(expr.value)
    return None


# ---------------------------------------------------------------------------
# Obligation encoding
# ---------------------------------------------------------------------------

class ObligationEncoding:
    """Encodes a single obligation declaration into variables & constraints."""

    def encode(self, obligation: ObligationDecl, jurisdiction_map: dict[str, JurisdictionDecl]) -> tuple[list[Variable], list[Constraint]]:
        variables: list[Variable] = []
        constraints: list[Constraint] = []
        prefix = f"obl_{obligation.name}"

        # Binary selection variable
        sel_var = Variable(name=f"{prefix}_selected", domain=(0, 1))
        variables.append(sel_var)

        # Cost variable (non-negative real)
        cost_var = Variable(name=f"{prefix}_cost", domain=(0.0, float("inf")))
        variables.append(cost_var)

        # If obligation has a risk level, encode threshold constraint
        if obligation.risk_level is not None:
            risk_val = _RISK_VALUE.get(obligation.risk_level, 0)
            risk_var = Variable(name=f"{prefix}_risk", domain=(1, 4))
            variables.append(risk_var)
            constraints.append(Constraint(
                name=f"{prefix}_risk_assign",
                variables=[risk_var.name],
                relation=f"(= {_smt_id(risk_var.name)} {risk_val})",
            ))

        # Temporal ordering constraint
        if obligation.temporal is not None:
            t_vars, t_cons = self._encode_temporal(obligation.temporal, prefix)
            variables.extend(t_vars)
            constraints.extend(t_cons)

        # Jurisdiction-derived constraints
        if obligation.jurisdiction and obligation.jurisdiction in jurisdiction_map:
            jur = jurisdiction_map[obligation.jurisdiction]
            if jur.enforcement_date is not None:
                parsed = _parse_date(jur.enforcement_date)
                if parsed is not None:
                    deadline_var = Variable(name=f"{prefix}_jur_deadline", domain=(0, 99999))
                    variables.append(deadline_var)
                    constraints.append(Constraint(
                        name=f"{prefix}_jur_deadline_val",
                        variables=[deadline_var.name],
                        relation=f"(= {_smt_id(deadline_var.name)} {_date_ordinal(parsed)})",
                    ))

        # Articles cardinality (at least one article required)
        if obligation.articles:
            constraints.append(Constraint(
                name=f"{prefix}_articles_nonempty",
                variables=[sel_var.name],
                relation=f"(=> (= {_smt_id(sel_var.name)} 1) true)",
            ))

        return variables, constraints

    def _encode_temporal(self, texpr: TemporalExpr, prefix: str) -> tuple[list[Variable], list[Constraint]]:
        variables: list[Variable] = []
        constraints: list[Constraint] = []
        time_var = Variable(name=f"{prefix}_time", domain=(0, 99999))
        variables.append(time_var)

        if texpr.operator == "BEFORE" and texpr.deadline:
            parsed = _parse_date(texpr.deadline)
            if parsed:
                constraints.append(Constraint(
                    name=f"{prefix}_before_{texpr.deadline}",
                    variables=[time_var.name],
                    relation=f"(<= {_smt_id(time_var.name)} {_date_ordinal(parsed)})",
                ))
        elif texpr.operator == "AFTER" and texpr.deadline:
            parsed = _parse_date(texpr.deadline)
            if parsed:
                constraints.append(Constraint(
                    name=f"{prefix}_after_{texpr.deadline}",
                    variables=[time_var.name],
                    relation=f"(>= {_smt_id(time_var.name)} {_date_ordinal(parsed)})",
                ))
        elif texpr.operator == "WITHIN" and texpr.deadline:
            parsed = _parse_date(texpr.deadline)
            if parsed:
                today_ord = _date_ordinal(date.today())
                constraints.append(Constraint(
                    name=f"{prefix}_within_{texpr.deadline}",
                    variables=[time_var.name],
                    relation=f"(and (>= {_smt_id(time_var.name)} {today_ord}) (<= {_smt_id(time_var.name)} {_date_ordinal(parsed)}))",
                ))
        elif texpr.operator == "EVERY" and texpr.recurrence:
            # Encode recurrence as a modular constraint placeholder
            constraints.append(Constraint(
                name=f"{prefix}_every_{texpr.recurrence}",
                variables=[time_var.name],
                relation=f"(>= {_smt_id(time_var.name)} 0)",
            ))

        return variables, constraints


# ---------------------------------------------------------------------------
# Composition expander
# ---------------------------------------------------------------------------

class CompositionExpander:
    """Expands a composition declaration into constraints over strategy variables."""

    def expand(
        self,
        composition: CompositionDecl,
        strategies: dict[str, StrategyDecl],
        obligation_sel_vars: dict[str, str],
        strategy_sel_vars: dict[str, str],
    ) -> list[Constraint]:
        mode = composition.mode
        strat_names = composition.strategies
        constraints: list[Constraint] = []

        resolved_strats = {s: strategies[s] for s in strat_names if s in strategies}

        if mode == ComposeMode.UNION:
            constraints.extend(self._expand_union(composition.name, resolved_strats, obligation_sel_vars, strategy_sel_vars))
        elif mode == ComposeMode.INTERSECT:
            constraints.extend(self._expand_intersect(composition.name, resolved_strats, obligation_sel_vars, strategy_sel_vars))
        elif mode == ComposeMode.SEQUENCE:
            constraints.extend(self._expand_sequence(composition.name, resolved_strats, strategy_sel_vars))
        elif mode == ComposeMode.OVERRIDE:
            constraints.extend(self._expand_override(composition.name, strat_names, resolved_strats, obligation_sel_vars, strategy_sel_vars))

        return constraints

    def _expand_union(
        self, comp_name: str, strats: dict[str, StrategyDecl],
        obl_vars: dict[str, str], strat_vars: dict[str, str],
    ) -> list[Constraint]:
        constraints: list[Constraint] = []
        # An obligation is selected if ANY contributing strategy is selected
        all_obls: set[str] = set()
        for sd in strats.values():
            all_obls.update(sd.obligations)

        for obl_name in all_obls:
            obl_sel = obl_vars.get(obl_name)
            if obl_sel is None:
                continue
            contributing = [
                strat_vars[sn] for sn, sd in strats.items()
                if obl_name in sd.obligations and sn in strat_vars
            ]
            if not contributing:
                continue
            or_clause = " ".join(f"(= {_smt_id(sv)} 1)" for sv in contributing)
            if len(contributing) > 1:
                or_clause = f"(or {or_clause})"
            constraints.append(Constraint(
                name=f"{comp_name}_union_{obl_name}",
                variables=[obl_sel] + contributing,
                relation=f"(=> {or_clause} (= {_smt_id(obl_sel)} 1))",
            ))
        return constraints

    def _expand_intersect(
        self, comp_name: str, strats: dict[str, StrategyDecl],
        obl_vars: dict[str, str], strat_vars: dict[str, str],
    ) -> list[Constraint]:
        constraints: list[Constraint] = []
        if not strats:
            return constraints
        # An obligation is selected only if ALL strategies include it AND are selected
        strat_list = list(strats.values())
        common_obls = set(strat_list[0].obligations)
        for sd in strat_list[1:]:
            common_obls &= set(sd.obligations)

        for obl_name in common_obls:
            obl_sel = obl_vars.get(obl_name)
            if obl_sel is None:
                continue
            all_strat_sels = [strat_vars[sn] for sn in strats if sn in strat_vars]
            and_clause = " ".join(f"(= {_smt_id(sv)} 1)" for sv in all_strat_sels)
            if len(all_strat_sels) > 1:
                and_clause = f"(and {and_clause})"
            constraints.append(Constraint(
                name=f"{comp_name}_intersect_{obl_name}",
                variables=[obl_sel] + all_strat_sels,
                relation=f"(= (= {_smt_id(obl_sel)} 1) {and_clause})",
            ))
        return constraints

    def _expand_sequence(
        self, comp_name: str, strats: dict[str, StrategyDecl],
        strat_vars: dict[str, str],
    ) -> list[Constraint]:
        constraints: list[Constraint] = []
        ordered = list(strats.keys())
        for i in range(len(ordered) - 1):
            earlier = ordered[i]
            later = ordered[i + 1]
            ev = strat_vars.get(earlier)
            lv = strat_vars.get(later)
            if ev and lv:
                time_early = f"strat_{earlier}_end_time"
                time_late = f"strat_{later}_start_time"
                constraints.append(Constraint(
                    name=f"{comp_name}_seq_{earlier}_before_{later}",
                    variables=[ev, lv, time_early, time_late],
                    relation=f"(=> (and (= {_smt_id(ev)} 1) (= {_smt_id(lv)} 1)) (<= {_smt_id(time_early)} {_smt_id(time_late)}))",
                ))
        return constraints

    def _expand_override(
        self, comp_name: str, ordered_names: list[str],
        strats: dict[str, StrategyDecl], obl_vars: dict[str, str],
        strat_vars: dict[str, str],
    ) -> list[Constraint]:
        constraints: list[Constraint] = []
        # Later strategies override earlier ones for shared obligations
        seen_obls: dict[str, str] = {}  # obligation -> controlling strategy
        for sname in ordered_names:
            sd = strats.get(sname)
            if sd is None:
                continue
            sv = strat_vars.get(sname)
            if sv is None:
                continue
            for obl_name in sd.obligations:
                if obl_name in seen_obls:
                    prev_strat = seen_obls[obl_name]
                    prev_sv = strat_vars.get(prev_strat)
                    obl_sel = obl_vars.get(obl_name)
                    if prev_sv and obl_sel:
                        # If the later strategy is selected, the earlier one's
                        # version of this obligation is suppressed
                        constraints.append(Constraint(
                            name=f"{comp_name}_override_{obl_name}_{prev_strat}_by_{sname}",
                            variables=[obl_sel, sv, prev_sv],
                            relation=f"(=> (= {_smt_id(sv)} 1) (= {_smt_id(prev_sv)} 0))",
                        ))
                seen_obls[obl_name] = sname
        return constraints


# ---------------------------------------------------------------------------
# Main Compiler
# ---------------------------------------------------------------------------

class Compiler:
    """Transforms a RegSynth Program AST into a *ConstraintProblem*."""

    def __init__(self) -> None:
        self._problem = ConstraintProblem()
        self._jurisdiction_map: dict[str, JurisdictionDecl] = {}
        self._obligation_map: dict[str, ObligationDecl] = {}
        self._strategy_map: dict[str, StrategyDecl] = {}
        self._composition_map: dict[str, CompositionDecl] = {}
        self._constraint_map: dict[str, ConstraintDecl] = {}
        self._obl_sel_vars: dict[str, str] = {}
        self._strat_sel_vars: dict[str, str] = {}
        self._obl_encoder = ObligationEncoding()
        self._comp_expander = CompositionExpander()

    # -- public entry point ------------------------------------------------

    def compile(self, program: Program) -> ConstraintProblem:
        self._problem = ConstraintProblem()
        self._jurisdiction_map.clear()
        self._obligation_map.clear()
        self._strategy_map.clear()
        self._composition_map.clear()
        self._constraint_map.clear()
        self._obl_sel_vars.clear()
        self._strat_sel_vars.clear()

        # Classify declarations
        for decl in program.declarations:
            if isinstance(decl, JurisdictionDecl):
                self._jurisdiction_map[decl.name] = decl
            elif isinstance(decl, ObligationDecl):
                self._obligation_map[decl.name] = decl
            elif isinstance(decl, StrategyDecl):
                self._strategy_map[decl.name] = decl
            elif isinstance(decl, CompositionDecl):
                self._composition_map[decl.name] = decl
            elif isinstance(decl, ConstraintDecl):
                self._constraint_map[decl.name] = decl

        self.compile_jurisdictions(list(self._jurisdiction_map.values()))
        self.compile_obligations(list(self._obligation_map.values()))
        self.compile_strategies(list(self._strategy_map.values()))
        self.compile_compositions(list(self._composition_map.values()))
        self.compile_constraints(list(self._constraint_map.values()))
        self._problem.objectives = self.build_objectives(program)

        return self._problem

    # -- jurisdictions -----------------------------------------------------

    def compile_jurisdictions(self, decls: list[JurisdictionDecl]) -> None:
        for jur in decls:
            # Each jurisdiction gets a selection variable for multi-jurisdiction mode
            jur_var = Variable(name=f"jur_{jur.name}_active", domain=(0, 1))
            self._problem.variables.append(jur_var)

            if jur.enforcement_date:
                parsed = _parse_date(jur.enforcement_date)
                if parsed:
                    deadline_var = Variable(name=f"jur_{jur.name}_deadline", domain=(0, 99999))
                    self._problem.variables.append(deadline_var)
                    self._problem.constraints.append(Constraint(
                        name=f"jur_{jur.name}_deadline_fix",
                        variables=[deadline_var.name],
                        relation=f"(= {_smt_id(deadline_var.name)} {_date_ordinal(parsed)})",
                    ))

    # -- obligations -------------------------------------------------------

    def compile_obligations(self, decls: list[ObligationDecl]) -> list[tuple[list[Variable], list[Constraint]]]:
        results: list[tuple[list[Variable], list[Constraint]]] = []
        for obl in decls:
            variables, constraints = self._obl_encoder.encode(obl, self._jurisdiction_map)
            self._problem.variables.extend(variables)
            self._problem.constraints.extend(constraints)
            sel_name = f"obl_{obl.name}_selected"
            self._obl_sel_vars[obl.name] = sel_name
            results.append((variables, constraints))
        return results

    # -- strategies --------------------------------------------------------

    def compile_strategies(self, decls: list[StrategyDecl]) -> None:
        for strat in decls:
            sel_var = Variable(name=f"strat_{strat.name}_selected", domain=(0, 1))
            self._problem.variables.append(sel_var)
            self._strat_sel_vars[strat.name] = sel_var.name

            # If strategy selected, its obligations must be selected
            for obl_name in strat.obligations:
                obl_sel = self._obl_sel_vars.get(obl_name)
                if obl_sel:
                    self._problem.constraints.append(Constraint(
                        name=f"strat_{strat.name}_requires_{obl_name}",
                        variables=[sel_var.name, obl_sel],
                        relation=f"(=> (= {_smt_id(sel_var.name)} 1) (= {_smt_id(obl_sel)} 1))",
                    ))

            # Cost linkage
            if strat.cost is not None:
                cost_val = _eval_literal(strat.cost)
                if cost_val is not None:
                    cost_var = Variable(name=f"strat_{strat.name}_cost", domain=(0.0, float("inf")))
                    self._problem.variables.append(cost_var)
                    self._problem.constraints.append(Constraint(
                        name=f"strat_{strat.name}_cost_val",
                        variables=[sel_var.name, cost_var.name],
                        relation=f"(= {_smt_id(cost_var.name)} (ite (= {_smt_id(sel_var.name)} 1) {cost_val} 0.0))",
                    ))

            # Sequence timing variables
            start_var = Variable(name=f"strat_{strat.name}_start_time", domain=(0, 99999))
            end_var = Variable(name=f"strat_{strat.name}_end_time", domain=(0, 99999))
            self._problem.variables.extend([start_var, end_var])
            self._problem.constraints.append(Constraint(
                name=f"strat_{strat.name}_time_order",
                variables=[start_var.name, end_var.name],
                relation=f"(<= {_smt_id(start_var.name)} {_smt_id(end_var.name)})",
            ))

    # -- compositions ------------------------------------------------------

    def compile_compositions(self, decls: list[CompositionDecl]) -> None:
        for comp in decls:
            new_constraints = self._comp_expander.expand(
                comp, self._strategy_map, self._obl_sel_vars, self._strat_sel_vars,
            )
            self._problem.constraints.extend(new_constraints)

    # -- user constraints --------------------------------------------------

    def compile_constraints(self, decls: list[ConstraintDecl]) -> None:
        for cdecl in decls:
            ctype = cdecl.constraint_type
            params = cdecl.parameters

            if ctype == "budget":
                self._compile_budget_constraint(cdecl, params)
            elif ctype == "timeline":
                self._compile_timeline_constraint(cdecl, params)
            elif ctype == "coverage":
                self._compile_coverage_constraint(cdecl, params)
            elif ctype == "risk_threshold":
                self._compile_risk_threshold(cdecl, params)
            elif ctype == "mutual_exclusion":
                self._compile_mutual_exclusion(cdecl, params)
            elif ctype == "dependency":
                self._compile_dependency(cdecl, params)
            elif ctype == "cardinality":
                self._compile_cardinality(cdecl, params)
            elif ctype == "temporal_order":
                self._compile_temporal_order(cdecl, params)
            else:
                # Generic: encode parameters as equality constraints
                for pname, pexpr in params.items():
                    val = _eval_literal(pexpr)
                    if val is not None:
                        var = Variable(name=f"constraint_{cdecl.name}_{pname}", domain=(val, val))
                        self._problem.variables.append(var)

    def _compile_budget_constraint(self, cdecl: ConstraintDecl, params: dict[str, Expression]) -> None:
        max_cost = _eval_literal(params.get("max_cost", Literal(value=0, literal_type="int")))
        if max_cost is None:
            return
        cost_vars = [v.name for v in self._problem.variables if v.name.endswith("_cost")]
        if cost_vars:
            sum_expr = " ".join(_smt_id(cv) for cv in cost_vars)
            if len(cost_vars) > 1:
                sum_expr = f"(+ {sum_expr})"
            self._problem.constraints.append(Constraint(
                name=f"constraint_{cdecl.name}_budget",
                variables=cost_vars,
                relation=f"(<= {sum_expr} {max_cost})",
            ))

    def _compile_timeline_constraint(self, cdecl: ConstraintDecl, params: dict[str, Expression]) -> None:
        max_days = _eval_literal(params.get("max_days", Literal(value=0, literal_type="int")))
        if max_days is None:
            return
        time_vars = [v.name for v in self._problem.variables if v.name.endswith("_time") or v.name.endswith("_end_time")]
        today_ord = _date_ordinal(date.today())
        for tv in time_vars:
            self._problem.constraints.append(Constraint(
                name=f"constraint_{cdecl.name}_timeline_{tv}",
                variables=[tv],
                relation=f"(<= {_smt_id(tv)} {today_ord + int(max_days)})",
            ))

    def _compile_coverage_constraint(self, cdecl: ConstraintDecl, params: dict[str, Expression]) -> None:
        min_cov = _eval_literal(params.get("min_coverage", Literal(value=0, literal_type="float")))
        if min_cov is None:
            return
        sel_vars = [v.name for v in self._problem.variables if v.name.endswith("_selected") and v.name.startswith("obl_")]
        total = len(sel_vars)
        if total == 0:
            return
        required = int(min_cov * total) if min_cov <= 1.0 else int(min_cov)
        sum_expr = " ".join(_smt_id(sv) for sv in sel_vars)
        if len(sel_vars) > 1:
            sum_expr = f"(+ {sum_expr})"
        self._problem.constraints.append(Constraint(
            name=f"constraint_{cdecl.name}_coverage",
            variables=sel_vars,
            relation=f"(>= {sum_expr} {required})",
        ))

    def _compile_risk_threshold(self, cdecl: ConstraintDecl, params: dict[str, Expression]) -> None:
        max_risk = _eval_literal(params.get("max_risk", Literal(value=4, literal_type="int")))
        if max_risk is None:
            return
        risk_vars = [v.name for v in self._problem.variables if v.name.endswith("_risk")]
        for rv in risk_vars:
            self._problem.constraints.append(Constraint(
                name=f"constraint_{cdecl.name}_risk_{rv}",
                variables=[rv],
                relation=f"(<= {_smt_id(rv)} {int(max_risk)})",
            ))

    def _compile_mutual_exclusion(self, cdecl: ConstraintDecl, params: dict[str, Expression]) -> None:
        obls_expr = params.get("obligations")
        obl_names: list[str] = []
        if isinstance(obls_expr, Literal) and isinstance(obls_expr.value, str):
            obl_names = [s.strip() for s in obls_expr.value.split(",") if s.strip()]
        sel_vars = [self._obl_sel_vars[n] for n in obl_names if n in self._obl_sel_vars]
        if len(sel_vars) < 2:
            return
        sum_expr = " ".join(_smt_id(sv) for sv in sel_vars)
        sum_expr = f"(+ {sum_expr})"
        self._problem.constraints.append(Constraint(
            name=f"constraint_{cdecl.name}_mutex",
            variables=sel_vars,
            relation=f"(<= {sum_expr} 1)",
        ))

    def _compile_dependency(self, cdecl: ConstraintDecl, params: dict[str, Expression]) -> None:
        src_expr = params.get("source")
        tgt_expr = params.get("target")
        src_name = src_expr.value if isinstance(src_expr, Literal) and isinstance(src_expr.value, str) else None
        tgt_name = tgt_expr.value if isinstance(tgt_expr, Literal) and isinstance(tgt_expr.value, str) else None
        if src_name and tgt_name:
            src_sel = self._obl_sel_vars.get(src_name)
            tgt_sel = self._obl_sel_vars.get(tgt_name)
            if src_sel and tgt_sel:
                self._problem.constraints.append(Constraint(
                    name=f"constraint_{cdecl.name}_dep",
                    variables=[src_sel, tgt_sel],
                    relation=f"(=> (= {_smt_id(tgt_sel)} 1) (= {_smt_id(src_sel)} 1))",
                ))

    def _compile_cardinality(self, cdecl: ConstraintDecl, params: dict[str, Expression]) -> None:
        min_val = _eval_literal(params.get("min", Literal(value=0, literal_type="int")))
        max_val = _eval_literal(params.get("max", Literal(value=9999, literal_type="int")))
        sel_vars = [v.name for v in self._problem.variables if v.name.endswith("_selected") and v.name.startswith("obl_")]
        if not sel_vars:
            return
        sum_expr = " ".join(_smt_id(sv) for sv in sel_vars)
        if len(sel_vars) > 1:
            sum_expr = f"(+ {sum_expr})"
        constraints_to_add: list[Constraint] = []
        if min_val is not None:
            constraints_to_add.append(Constraint(
                name=f"constraint_{cdecl.name}_card_min",
                variables=sel_vars,
                relation=f"(>= {sum_expr} {int(min_val)})",
            ))
        if max_val is not None:
            constraints_to_add.append(Constraint(
                name=f"constraint_{cdecl.name}_card_max",
                variables=sel_vars,
                relation=f"(<= {sum_expr} {int(max_val)})",
            ))
        self._problem.constraints.extend(constraints_to_add)

    def _compile_temporal_order(self, cdecl: ConstraintDecl, params: dict[str, Expression]) -> None:
        before_expr = params.get("before")
        after_expr = params.get("after")
        before_name = before_expr.value if isinstance(before_expr, Literal) and isinstance(before_expr.value, str) else None
        after_name = after_expr.value if isinstance(after_expr, Literal) and isinstance(after_expr.value, str) else None
        if before_name and after_name:
            before_tv = f"obl_{before_name}_time"
            after_tv = f"obl_{after_name}_time"
            self._problem.constraints.append(Constraint(
                name=f"constraint_{cdecl.name}_temporal_order",
                variables=[before_tv, after_tv],
                relation=f"(<= {_smt_id(before_tv)} {_smt_id(after_tv)})",
            ))

    # -- objectives --------------------------------------------------------

    def build_objectives(self, program: Program) -> list[ObjectiveFunction]:
        objectives: list[ObjectiveFunction] = []

        # 1) Cost minimization
        cost_vars = [v.name for v in self._problem.variables if v.name.endswith("_cost")]
        if cost_vars:
            objectives.append(ObjectiveFunction(
                name="minimize_total_cost",
                variables=cost_vars,
                coefficients=[1.0] * len(cost_vars),
                direction=Direction.MINIMIZE,
            ))

        # 2) Coverage maximization
        sel_vars = [v.name for v in self._problem.variables if v.name.endswith("_selected") and v.name.startswith("obl_")]
        if sel_vars:
            objectives.append(ObjectiveFunction(
                name="maximize_coverage",
                variables=sel_vars,
                coefficients=[1.0] * len(sel_vars),
                direction=Direction.MAXIMIZE,
            ))

        # 3) Risk minimization (weight by risk value)
        risk_vars = [v.name for v in self._problem.variables if v.name.endswith("_risk")]
        if risk_vars:
            objectives.append(ObjectiveFunction(
                name="minimize_risk",
                variables=risk_vars,
                coefficients=[1.0] * len(risk_vars),
                direction=Direction.MINIMIZE,
            ))

        # 4) Timeline minimization
        time_vars = [v.name for v in self._problem.variables if v.name.endswith("_end_time")]
        if time_vars:
            objectives.append(ObjectiveFunction(
                name="minimize_timeline",
                variables=time_vars,
                coefficients=[1.0] * len(time_vars),
                direction=Direction.MINIMIZE,
            ))

        return objectives

    # -- temporal encoding -------------------------------------------------

    def encode_temporal(self, temporal: TemporalExpr) -> Constraint:
        var_name = f"temporal_{id(temporal)}"
        if temporal.operator == "BEFORE" and temporal.deadline:
            parsed = _parse_date(temporal.deadline)
            bound = _date_ordinal(parsed) if parsed else 0
            return Constraint(
                name=f"temporal_before_{temporal.deadline}",
                variables=[var_name],
                relation=f"(<= {_smt_id(var_name)} {bound})",
            )
        if temporal.operator == "AFTER" and temporal.deadline:
            parsed = _parse_date(temporal.deadline)
            bound = _date_ordinal(parsed) if parsed else 0
            return Constraint(
                name=f"temporal_after_{temporal.deadline}",
                variables=[var_name],
                relation=f"(>= {_smt_id(var_name)} {bound})",
            )
        if temporal.operator == "WITHIN" and temporal.deadline:
            parsed = _parse_date(temporal.deadline)
            bound = _date_ordinal(parsed) if parsed else 0
            today = _date_ordinal(date.today())
            return Constraint(
                name=f"temporal_within_{temporal.deadline}",
                variables=[var_name],
                relation=f"(and (>= {_smt_id(var_name)} {today}) (<= {_smt_id(var_name)} {bound}))",
            )
        return Constraint(
            name=f"temporal_{temporal.operator}",
            variables=[var_name],
            relation=f"(>= {_smt_id(var_name)} 0)",
        )

    # -- risk encoding -----------------------------------------------------

    def encode_risk_constraint(self, risk_level: RiskLevel, jurisdiction: str) -> Constraint:
        threshold = _RISK_VALUE.get(risk_level, 4)
        jur_var = f"jur_{jurisdiction}_active"
        risk_var = f"risk_{jurisdiction}_level"
        return Constraint(
            name=f"risk_{jurisdiction}_{risk_level.value}",
            variables=[jur_var, risk_var],
            relation=f"(=> (= {_smt_id(jur_var)} 1) (<= {_smt_id(risk_var)} {threshold}))",
        )

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return self._problem.to_dict()

    def to_smt2(self) -> str:
        lines: list[str] = [
            "; RegSynth compiled constraint problem",
            "(set-logic QF_LIA)",
            "",
        ]
        # Variables
        for var in self._problem.variables:
            lines.append(var.to_smt2())
        lines.append("")

        # Constraints
        for con in self._problem.constraints:
            lines.append(con.to_smt2())
        lines.append("")

        # Objectives (as comments + soft assertions for OMT solvers)
        for obj in self._problem.objectives:
            lines.append(obj.to_smt2())
        lines.append("")

        lines.append("(check-sat)")
        lines.append("(get-model)")
        return "\n".join(lines)

    def summary(self) -> str:
        nv = len(self._problem.variables)
        nc = len(self._problem.constraints)
        no = len(self._problem.objectives)
        parts = [
            f"RegSynth Compiled Problem",
            f"  Variables:   {nv}",
            f"  Constraints: {nc}",
            f"  Objectives:  {no} (Pareto multi-objective)",
        ]
        if self._problem.objectives:
            parts.append("  Objective breakdown:")
            for obj in self._problem.objectives:
                parts.append(f"    - {obj.name} ({obj.direction.value}, {len(obj.variables)} vars)")
        return "\n".join(parts)
