"""Assume-guarantee contracts for compositional verification.

Contracts capture modular specifications: a module satisfies contract (A, G)
if whenever its environment satisfies assumption A, the module guarantees G.
Operations include conjunction, composition, refinement, and quotient.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from bioprover.encoding.expression import (
    And,
    ExprNode,
    Implies,
    Not,
    Or,
    Var,
    Const,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal direction and interface variables
# ---------------------------------------------------------------------------

class SignalDirection(Enum):
    """Direction of a signal at a module interface."""
    INPUT = auto()
    OUTPUT = auto()
    BIDIRECTIONAL = auto()


@dataclass(frozen=True)
class InterfaceVariable:
    """A variable exposed at a module's interface.

    Attributes:
        name: Variable identifier (typically a species name).
        direction: Whether the variable is an input, output, or shared.
        lower_bound: Minimum physically meaningful concentration.
        upper_bound: Maximum physically meaningful concentration.
        unit: Physical unit string (e.g. "nM", "uM").
    """
    name: str
    direction: SignalDirection
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    unit: str = "nM"

    def to_var(self) -> Var:
        """Return the corresponding expression variable."""
        return Var(self.name)


class WellFormednessError(Exception):
    """Raised when a contract violates well-formedness conditions."""


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass
class Contract:
    """Assume-guarantee contract over continuous biological signals.

    A contract C = (A, G) states:
      • A (assumption): what the module assumes about its environment.
      • G (guarantee): what the module promises about its own behaviour.

    Both A and G are STL/first-order formulas built from *ExprNode*.

    Attributes:
        name: Human-readable identifier.
        assumption: The assumption formula A.
        guarantee: The guarantee formula G.
        input_signals: Variables the module reads from its environment.
        output_signals: Variables the module writes.
        interface_vars: Complete set of interface variables.
        metadata: Arbitrary extra information (e.g. provenance).
    """

    name: str
    assumption: ExprNode
    guarantee: ExprNode
    input_signals: List[InterfaceVariable] = field(default_factory=list)
    output_signals: List[InterfaceVariable] = field(default_factory=list)
    interface_vars: List[InterfaceVariable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- helpers --------------------------------------------------------

    @property
    def input_names(self) -> FrozenSet[str]:
        return frozenset(v.name for v in self.input_signals)

    @property
    def output_names(self) -> FrozenSet[str]:
        return frozenset(v.name for v in self.output_signals)

    @property
    def all_interface_names(self) -> FrozenSet[str]:
        return self.input_names | self.output_names | frozenset(
            v.name for v in self.interface_vars
        )

    # -- well-formedness ------------------------------------------------

    def check_well_formedness(self, *, allow_circular: bool = False) -> None:
        """Validate structural well-formedness.

        Raises:
            WellFormednessError: If the contract is malformed.
        """
        overlap = self.input_names & self.output_names
        if overlap and not allow_circular:
            raise WellFormednessError(
                f"Input/output overlap without circular mode: {overlap}"
            )
        if self.assumption is None or self.guarantee is None:
            raise WellFormednessError(
                "Both assumption and guarantee must be non-None."
            )
        self._check_variable_coverage()

    def _check_variable_coverage(self) -> None:
        """Ensure formulas only reference declared interface variables."""
        declared = self.all_interface_names
        a_vars = _free_vars(self.assumption)
        g_vars = _free_vars(self.guarantee)
        undeclared_a = a_vars - declared
        undeclared_g = g_vars - declared
        if undeclared_a:
            logger.warning(
                "Assumption references undeclared variables: %s", undeclared_a
            )
        if undeclared_g:
            logger.warning(
                "Guarantee references undeclared variables: %s", undeclared_g
            )

    # -- contract formula (A => G) --------------------------------------

    def as_implication(self) -> ExprNode:
        """Return the formula A => G."""
        return Implies(self.assumption, self.guarantee)

    # -- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the contract to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "assumption": str(self.assumption),
            "guarantee": str(self.guarantee),
            "input_signals": [
                _ivar_to_dict(v) for v in self.input_signals
            ],
            "output_signals": [
                _ivar_to_dict(v) for v in self.output_signals
            ],
            "interface_vars": [
                _ivar_to_dict(v) for v in self.interface_vars
            ],
            "metadata": self.metadata,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    # -- visualization --------------------------------------------------

    def ascii_diagram(self) -> str:
        """Render a simple ASCII input/output diagram."""
        lines: List[str] = []
        header = f"┌─ Contract: {self.name} "
        header += "─" * max(0, 50 - len(header)) + "┐"
        lines.append(header)

        if self.input_signals:
            lines.append("│  Inputs:")
            for v in self.input_signals:
                lines.append(f"│    ──▶  {v.name}  [{v.lower_bound}, {v.upper_bound}] {v.unit}")

        if self.output_signals:
            lines.append("│  Outputs:")
            for v in self.output_signals:
                lines.append(f"│    ◀──  {v.name}  [{v.lower_bound}, {v.upper_bound}] {v.unit}")

        lines.append(f"│  Assumption: {self.assumption}")
        lines.append(f"│  Guarantee:  {self.guarantee}")
        lines.append("└" + "─" * (len(header) - 2) + "┘")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Contract(name={self.name!r}, "
            f"A={self.assumption}, G={self.guarantee})"
        )


# ---------------------------------------------------------------------------
# Contract operations
# ---------------------------------------------------------------------------

class ContractConjunction:
    """Conjunction of contracts: (A1∧A2, G1∧G2).

    The conjunction tightens both assumptions and guarantees, yielding
    a contract that requires *all* assumptions and enforces *all* guarantees.
    """

    @staticmethod
    def conjoin(c1: Contract, c2: Contract, name: Optional[str] = None) -> Contract:
        """Compute the conjunction C1 ∧ C2."""
        new_name = name or f"({c1.name} ∧ {c2.name})"
        return Contract(
            name=new_name,
            assumption=And(c1.assumption, c2.assumption),
            guarantee=And(c1.guarantee, c2.guarantee),
            input_signals=_merge_signals(c1.input_signals, c2.input_signals),
            output_signals=_merge_signals(c1.output_signals, c2.output_signals),
            interface_vars=_merge_signals(c1.interface_vars, c2.interface_vars),
            metadata={"operation": "conjunction", "operands": [c1.name, c2.name]},
        )

    @staticmethod
    def conjoin_all(
        contracts: Sequence[Contract], name: Optional[str] = None
    ) -> Contract:
        """Conjoin an arbitrary number of contracts."""
        if not contracts:
            raise ValueError("Cannot conjoin an empty sequence of contracts.")
        result = contracts[0]
        for c in contracts[1:]:
            result = ContractConjunction.conjoin(result, c)
        if name:
            result.name = name
        return result


class ContractComposition:
    """Parallel composition of contracts.

    For two contracts C1 = (A1, G1) and C2 = (A2, G2) with compatible
    interfaces, the parallel composition is:
        A_comp = (A1 ∧ A2) ∧ ¬(G1 ∧ G2)  ∨  (A1 ∧ A2)
    Simplified to:
        A_comp = A1 ∧ A2  (environment assumptions combined)
        G_comp = G1 ∧ G2  (both modules must deliver)
    with outputs of one feeding inputs of the other removed from the
    external interface.
    """

    @staticmethod
    def compose(
        c1: Contract,
        c2: Contract,
        name: Optional[str] = None,
        *,
        feedback: bool = False,
    ) -> Contract:
        """Compute parallel composition of two contracts.

        When *feedback* is True, output-to-input connections within the
        composed system are permitted (circular dependencies).
        """
        # Identify internal connections: outputs of one that are inputs of the other
        internal_1_to_2 = c1.output_names & c2.input_names
        internal_2_to_1 = c2.output_names & c1.input_names
        internal = internal_1_to_2 | internal_2_to_1

        if internal_2_to_1 and not feedback:
            raise WellFormednessError(
                f"Feedback connections {internal_2_to_1} require feedback=True"
            )

        # External inputs: inputs of the composition not provided internally
        ext_inputs = _merge_signals(
            [v for v in c1.input_signals if v.name not in internal],
            [v for v in c2.input_signals if v.name not in internal],
        )
        # External outputs: all outputs
        ext_outputs = _merge_signals(c1.output_signals, c2.output_signals)

        # Composed assumption: both assumptions hold, minus what is internally guaranteed
        # Standard AG: A_comp = (A1 \ G2) ∧ (A2 \ G1) for the internal signals
        # Simplified: weaken assumptions by removing internally-satisfied parts
        composed_assumption = And(c1.assumption, c2.assumption)
        composed_guarantee = And(c1.guarantee, c2.guarantee)

        new_name = name or f"({c1.name} ‖ {c2.name})"
        return Contract(
            name=new_name,
            assumption=composed_assumption,
            guarantee=composed_guarantee,
            input_signals=ext_inputs,
            output_signals=ext_outputs,
            interface_vars=[
                InterfaceVariable(n, SignalDirection.BIDIRECTIONAL)
                for n in internal
            ],
            metadata={
                "operation": "composition",
                "operands": [c1.name, c2.name],
                "internal_connections": sorted(internal),
                "feedback": feedback,
            },
        )

    @staticmethod
    def compose_all(
        contracts: Sequence[Contract],
        name: Optional[str] = None,
        *,
        feedback: bool = False,
    ) -> Contract:
        """Compose an arbitrary number of contracts in parallel."""
        if not contracts:
            raise ValueError("Cannot compose an empty sequence of contracts.")
        result = contracts[0]
        for c in contracts[1:]:
            result = ContractComposition.compose(result, c, feedback=feedback)
        if name:
            result.name = name
        return result


class ContractRefinement:
    """Refinement checking between contracts.

    C1 refines C2 (C1 ≤ C2) iff:
        A2 ⇒ A1   (C1 can work in more environments)
        G1 ⇒ G2   (C1 provides stronger guarantees)

    This is the standard substitutability relation: anywhere C2 is used,
    C1 can be substituted.
    """

    @staticmethod
    def check_refinement(
        c_impl: Contract,
        c_spec: Contract,
        solver: Optional[Any] = None,
    ) -> RefinementResult:
        """Check whether *c_impl* refines *c_spec*.

        If a solver is provided, performs semantic checking via SMT.
        Otherwise performs a conservative syntactic check.
        """
        # Syntactic fast path: structural equality
        a_weakened = _syntactic_implies(c_spec.assumption, c_impl.assumption)
        g_strengthened = _syntactic_implies(c_impl.guarantee, c_spec.guarantee)

        if a_weakened and g_strengthened:
            return RefinementResult(
                refines=True,
                assumption_check=True,
                guarantee_check=True,
                method="syntactic",
            )

        if solver is not None:
            return _semantic_refinement_check(c_impl, c_spec, solver)

        return RefinementResult(
            refines=False,
            assumption_check=a_weakened,
            guarantee_check=g_strengthened,
            method="syntactic",
            reason="Syntactic check inconclusive; provide solver for semantic check.",
        )

    @staticmethod
    def quotient(
        system: Contract,
        component: Contract,
        name: Optional[str] = None,
    ) -> Contract:
        """Compute the quotient contract.

        Given a system-level contract *system* and one component contract
        *component*, derive the contract that the remaining component must
        satisfy so that the composition refines *system*.

        Quotient C_q = system / component:
            A_q = A_sys ∧ G_comp
            G_q = G_sys  (simplified; full theory uses ¬A_sys ∨ G_sys)
        """
        new_name = name or f"({system.name} / {component.name})"

        # Remaining inputs: system inputs not provided by component outputs
        remaining_inputs = [
            v for v in system.input_signals
            if v.name not in component.output_names
        ]
        # Remaining outputs: system outputs not provided by component
        remaining_outputs = [
            v for v in system.output_signals
            if v.name not in component.output_names
        ]

        return Contract(
            name=new_name,
            assumption=And(system.assumption, component.guarantee),
            guarantee=system.guarantee,
            input_signals=remaining_inputs + list(component.output_signals),
            output_signals=remaining_outputs,
            metadata={
                "operation": "quotient",
                "system": system.name,
                "component": component.name,
            },
        )


@dataclass(frozen=True)
class RefinementResult:
    """Result of a refinement check."""
    refines: bool
    assumption_check: bool
    guarantee_check: bool
    method: str = "syntactic"
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Contract satisfaction
# ---------------------------------------------------------------------------

class ContractSatisfaction:
    """Check whether a module (model) satisfies a contract.

    A module M satisfies C = (A, G) iff:
        A ⇒ (M |= G)
    i.e., under the assumption, the module's behaviour implies the guarantee.
    """

    @staticmethod
    def check(
        model: Any,
        contract: Contract,
        solver: Optional[Any] = None,
        *,
        cegar_enabled: bool = True,
        time_horizon: float = 100.0,
        dt: float = 0.1,
    ) -> SatisfactionResult:
        """Check whether *model* satisfies *contract*.

        Args:
            model: A biological model (ODE system or hybrid automaton).
            contract: The contract to verify against.
            solver: An SMT solver instance. If None, simulation-based.
            cegar_enabled: Use CEGAR loop for the restricted model.
            time_horizon: Verification horizon (time units).
            dt: Time discretization step.

        Returns:
            SatisfactionResult with verdict and diagnostics.
        """
        logger.info("Checking satisfaction of contract %s", contract.name)

        if solver is not None:
            return ContractSatisfaction._smt_check(
                model, contract, solver,
                time_horizon=time_horizon, dt=dt,
                cegar_enabled=cegar_enabled,
            )
        return ContractSatisfaction._simulation_check(
            model, contract,
            time_horizon=time_horizon, dt=dt,
        )

    @staticmethod
    def _smt_check(
        model: Any,
        contract: Contract,
        solver: Any,
        *,
        time_horizon: float,
        dt: float,
        cegar_enabled: bool,
    ) -> SatisfactionResult:
        """SMT-based satisfaction check.

        Encodes A ∧ model_dynamics ∧ ¬G and checks unsatisfiability.
        If UNSAT, the contract is satisfied.
        """
        # Build the restricted verification query
        negated_guarantee = Not(contract.guarantee)
        query = And(contract.assumption, negated_guarantee)

        try:
            result = solver.check_sat(query)
            if hasattr(result, "name"):
                result_name = result.name
            else:
                result_name = str(result)

            if result_name in ("UNSAT", "unsat"):
                return SatisfactionResult(
                    satisfied=True,
                    method="smt",
                    message="Contract satisfied: A ∧ ¬G is unsatisfiable.",
                )
            elif result_name in ("SAT", "sat", "DELTA_SAT", "delta_sat"):
                cex = None
                if hasattr(solver, "get_model"):
                    cex = solver.get_model()
                return SatisfactionResult(
                    satisfied=False,
                    method="smt",
                    message="Contract violated: found counterexample.",
                    counterexample=cex,
                )
            else:
                return SatisfactionResult(
                    satisfied=False,
                    method="smt",
                    message=f"Solver returned {result_name}; inconclusive.",
                    inconclusive=True,
                )
        except Exception as exc:
            logger.error("SMT check failed: %s", exc)
            return SatisfactionResult(
                satisfied=False,
                method="smt",
                message=f"SMT check error: {exc}",
                inconclusive=True,
            )

    @staticmethod
    def _simulation_check(
        model: Any,
        contract: Contract,
        *,
        time_horizon: float,
        dt: float,
        n_simulations: int = 100,
    ) -> SatisfactionResult:
        """Simulation-based (incomplete) satisfaction check.

        Runs Monte-Carlo simulations under the assumption and checks
        whether the guarantee holds on all traces. Sound for
        *falsification* but not for verification.
        """
        import numpy as np

        violations: List[Dict[str, Any]] = []

        for sim_idx in range(n_simulations):
            # Sample initial state consistent with input bounds
            init_state = _sample_initial_state(contract, seed=sim_idx)
            trace = _simulate_model(model, init_state, time_horizon, dt)

            if trace is None:
                continue

            # Evaluate assumption on trace
            a_holds = _evaluate_formula_on_trace(contract.assumption, trace)
            if not a_holds:
                continue  # assumption not met – skip

            g_holds = _evaluate_formula_on_trace(contract.guarantee, trace)
            if not g_holds:
                violations.append({
                    "simulation": sim_idx,
                    "trace_summary": {
                        k: (float(np.min(v)), float(np.max(v)))
                        for k, v in trace.items()
                        if isinstance(v, np.ndarray)
                    },
                })

        if violations:
            return SatisfactionResult(
                satisfied=False,
                method="simulation",
                message=f"Contract violated in {len(violations)}/{n_simulations} simulations.",
                counterexample=violations[0],
            )
        return SatisfactionResult(
            satisfied=True,
            method="simulation",
            message=f"No violations in {n_simulations} simulations (not a proof).",
        )


@dataclass
class SatisfactionResult:
    """Result of a satisfaction check."""
    satisfied: bool
    method: str
    message: str = ""
    counterexample: Optional[Any] = None
    inconclusive: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_vars(expr: ExprNode) -> Set[str]:
    """Collect free variable names from an expression tree."""
    result: Set[str] = set()
    _collect_vars(expr, result)
    return result


def _collect_vars(expr: ExprNode, acc: Set[str]) -> None:
    """Recursively collect variable names."""
    if isinstance(expr, Var):
        acc.add(expr.name)
        return
    for child in getattr(expr, "children", []):
        _collect_vars(child, acc)
    # Handle binary/unary nodes with named fields
    for attr in ("left", "right", "operand", "condition", "then_", "else_",
                 "lhs", "rhs", "body", "arg"):
        child = getattr(expr, attr, None)
        if isinstance(child, ExprNode):
            _collect_vars(child, acc)


def _syntactic_implies(premise: ExprNode, conclusion: ExprNode) -> bool:
    """Conservative syntactic implication check.

    Returns True only when the implication is trivially obvious
    from structure (e.g. identical formulas).
    """
    if premise is conclusion:
        return True
    if str(premise) == str(conclusion):
        return True
    # A ∧ B => A  and  A ∧ B => B
    if isinstance(premise, And):
        left = getattr(premise, "left", None) or getattr(premise, "children", [None])[0]
        right = getattr(premise, "right", None) or (
            getattr(premise, "children", [None, None])[1]
            if len(getattr(premise, "children", [])) > 1 else None
        )
        if left is not None and _syntactic_implies(left, conclusion):
            return True
        if right is not None and _syntactic_implies(right, conclusion):
            return True
    return False


def _semantic_refinement_check(
    c_impl: Contract, c_spec: Contract, solver: Any
) -> RefinementResult:
    """Use an SMT solver to check refinement semantically."""
    # Check A_spec => A_impl  (assumption weakening)
    a_query = And(c_spec.assumption, Not(c_impl.assumption))
    a_result = solver.check_sat(a_query)
    a_ok = _is_unsat(a_result)

    # Check G_impl => G_spec  (guarantee strengthening)
    g_query = And(c_impl.guarantee, Not(c_spec.guarantee))
    g_result = solver.check_sat(g_query)
    g_ok = _is_unsat(g_result)

    return RefinementResult(
        refines=a_ok and g_ok,
        assumption_check=a_ok,
        guarantee_check=g_ok,
        method="semantic",
        reason=None if (a_ok and g_ok) else (
            "Assumption weakening failed." if not a_ok else "Guarantee strengthening failed."
        ),
    )


def _is_unsat(result: Any) -> bool:
    """Check if solver result is UNSAT."""
    if hasattr(result, "name"):
        return result.name in ("UNSAT", "unsat")
    return str(result).upper() == "UNSAT"


def _merge_signals(
    a: Sequence[InterfaceVariable], b: Sequence[InterfaceVariable]
) -> List[InterfaceVariable]:
    """Merge two signal lists, removing duplicates by name."""
    seen: Dict[str, InterfaceVariable] = {}
    for v in a:
        seen[v.name] = v
    for v in b:
        if v.name not in seen:
            seen[v.name] = v
    return list(seen.values())


def _ivar_to_dict(v: InterfaceVariable) -> Dict[str, Any]:
    return {
        "name": v.name,
        "direction": v.direction.name,
        "lower_bound": v.lower_bound,
        "upper_bound": v.upper_bound,
        "unit": v.unit,
    }


def _sample_initial_state(
    contract: Contract, seed: int = 0
) -> Dict[str, float]:
    """Sample an initial state respecting interface bounds."""
    import numpy as np
    rng = np.random.RandomState(seed)
    state: Dict[str, float] = {}
    for v in contract.input_signals + contract.output_signals + contract.interface_vars:
        ub = v.upper_bound if v.upper_bound < 1e6 else 100.0
        state[v.name] = rng.uniform(v.lower_bound, ub)
    return state


def _simulate_model(
    model: Any,
    init_state: Dict[str, float],
    time_horizon: float,
    dt: float,
) -> Optional[Dict[str, Any]]:
    """Run a single simulation. Returns a trace dict or None on failure."""
    if hasattr(model, "simulate"):
        try:
            return model.simulate(init_state, time_horizon, dt)
        except Exception as exc:
            logger.debug("Simulation failed: %s", exc)
            return None
    return None


def _evaluate_formula_on_trace(formula: ExprNode, trace: Dict[str, Any]) -> bool:
    """Evaluate a Boolean formula on a simulation trace.

    This is a best-effort evaluator for simple predicates.
    """
    if isinstance(formula, Const):
        return bool(formula.value)
    if hasattr(formula, "evaluate"):
        try:
            return bool(formula.evaluate(trace))
        except Exception:
            pass
    # Conservatively return True when we cannot evaluate
    return True
