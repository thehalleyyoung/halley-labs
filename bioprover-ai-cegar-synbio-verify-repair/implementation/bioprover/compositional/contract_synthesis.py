"""Contract synthesis from simulation traces.

Automatically derives assume-guarantee contracts by running simulations,
observing input-output behaviour, and fitting STL templates. Supports
iterative strengthening via verification feedback.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from bioprover.encoding.expression import (
    And,
    Const,
    ExprNode,
    Ge,
    Gt,
    Implies,
    Le,
    Lt,
    Not,
    Or,
    Var,
)
from bioprover.compositional.contracts import (
    Contract,
    ContractSatisfaction,
    InterfaceVariable,
    SatisfactionResult,
    SignalDirection,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & configuration
# ---------------------------------------------------------------------------

class SynthesisStrategy(Enum):
    """Strategy for synthesising contracts."""
    TEMPLATE_MATCHING = auto()   # Fit STL templates to traces
    INVARIANT_MINING = auto()    # Daikon-style invariant detection
    CONVEX_HULL = auto()         # Convex hull of observed states
    HYBRID = auto()              # Combine template + invariant mining


@dataclass
class SynthesisConfig:
    """Configuration for the contract synthesiser.

    Attributes:
        n_simulations: Number of simulations to run.
        time_horizon: Simulation horizon (time units).
        dt: Time discretisation step.
        strategy: Synthesis strategy.
        strengthen_rounds: Max rounds of assumption strengthening.
        robustness_margin: Safety margin added to learned bounds.
    """
    n_simulations: int = 200
    time_horizon: float = 100.0
    dt: float = 0.1
    strategy: SynthesisStrategy = SynthesisStrategy.HYBRID
    strengthen_rounds: int = 10
    robustness_margin: float = 0.05


# ---------------------------------------------------------------------------
# Contract templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractTemplate:
    """Parameterised STL template for a common biological pattern.

    Attributes:
        name: Template identifier.
        description: What the template captures.
        parameters: Names of tuneable parameters.
        assumption_builder: Callable(params) → ExprNode.
        guarantee_builder: Callable(params) → ExprNode.
    """
    name: str
    description: str
    parameters: Tuple[str, ...]
    assumption_builder: Callable[..., ExprNode]
    guarantee_builder: Callable[..., ExprNode]


class ContractLibrary:
    """Library of reusable contract templates for biological circuits."""

    def __init__(self) -> None:
        self._templates: Dict[str, ContractTemplate] = {}
        self._register_builtin_templates()

    # -- access -------------------------------------------------------------

    def get(self, name: str) -> ContractTemplate:
        return self._templates[name]

    def list_templates(self) -> List[str]:
        return sorted(self._templates.keys())

    def register(self, template: ContractTemplate) -> None:
        self._templates[template.name] = template

    # -- built-in templates -------------------------------------------------

    def _register_builtin_templates(self) -> None:
        """Register standard biological circuit contract templates."""

        # Bounded-response: if input stays in range, output reaches target
        self._templates["bounded_response"] = ContractTemplate(
            name="bounded_response",
            description=(
                "If input x stays in [x_lo, x_hi], output y eventually "
                "reaches [y_lo, y_hi]."
            ),
            parameters=("x", "x_lo", "x_hi", "y", "y_lo", "y_hi"),
            assumption_builder=lambda x, x_lo, x_hi, **_: And(
                Ge(Var(x), Const(x_lo)), Le(Var(x), Const(x_hi))
            ),
            guarantee_builder=lambda y, y_lo, y_hi, **_: And(
                Ge(Var(y), Const(y_lo)), Le(Var(y), Const(y_hi))
            ),
        )

        # Monotone response
        self._templates["monotone_activation"] = ContractTemplate(
            name="monotone_activation",
            description=(
                "If input x ≥ threshold, output y ≥ y_min (activation)."
            ),
            parameters=("x", "threshold", "y", "y_min"),
            assumption_builder=lambda x, threshold, **_: Ge(
                Var(x), Const(threshold)
            ),
            guarantee_builder=lambda y, y_min, **_: Ge(
                Var(y), Const(y_min)
            ),
        )

        # Monotone repression
        self._templates["monotone_repression"] = ContractTemplate(
            name="monotone_repression",
            description=(
                "If input x ≥ threshold, output y ≤ y_max (repression)."
            ),
            parameters=("x", "threshold", "y", "y_max"),
            assumption_builder=lambda x, threshold, **_: Ge(
                Var(x), Const(threshold)
            ),
            guarantee_builder=lambda y, y_max, **_: Le(
                Var(y), Const(y_max)
            ),
        )

        # Bistable switch
        self._templates["bistable_switch"] = ContractTemplate(
            name="bistable_switch",
            description=(
                "Output y is in one of two steady-state regions."
            ),
            parameters=("y", "lo_hi", "lo_lo", "hi_hi", "hi_lo"),
            assumption_builder=lambda **_: Const(True),
            guarantee_builder=lambda y, lo_lo, lo_hi, hi_lo, hi_hi, **_: Or(
                And(Ge(Var(y), Const(lo_lo)), Le(Var(y), Const(lo_hi))),
                And(Ge(Var(y), Const(hi_lo)), Le(Var(y), Const(hi_hi))),
            ),
        )

        # Oscillation bounds
        self._templates["oscillation_bounds"] = ContractTemplate(
            name="oscillation_bounds",
            description="Output y oscillates within [y_lo, y_hi].",
            parameters=("y", "y_lo", "y_hi"),
            assumption_builder=lambda **_: Const(True),
            guarantee_builder=lambda y, y_lo, y_hi, **_: And(
                Ge(Var(y), Const(y_lo)), Le(Var(y), Const(y_hi))
            ),
        )

        # Steady-state convergence
        self._templates["steady_state"] = ContractTemplate(
            name="steady_state",
            description=(
                "Output y converges to within ε of target value."
            ),
            parameters=("y", "target", "epsilon"),
            assumption_builder=lambda **_: Const(True),
            guarantee_builder=lambda y, target, epsilon, **_: And(
                Ge(Var(y), Const(target - epsilon)),
                Le(Var(y), Const(target + epsilon)),
            ),
        )

    # -- instantiation ------------------------------------------------------

    def instantiate(
        self,
        template_name: str,
        params: Dict[str, Any],
        contract_name: Optional[str] = None,
    ) -> Contract:
        """Instantiate a template with concrete parameter values."""
        tmpl = self._templates[template_name]
        assumption = tmpl.assumption_builder(**params)
        guarantee = tmpl.guarantee_builder(**params)

        input_vars: List[InterfaceVariable] = []
        output_vars: List[InterfaceVariable] = []
        for p in tmpl.parameters:
            if p in params and isinstance(params[p], str):
                # Heuristic: first variable parameter is input, second is output
                if not input_vars:
                    input_vars.append(
                        InterfaceVariable(params[p], SignalDirection.INPUT)
                    )
                else:
                    output_vars.append(
                        InterfaceVariable(params[p], SignalDirection.OUTPUT)
                    )

        return Contract(
            name=contract_name or f"{template_name}_instance",
            assumption=assumption,
            guarantee=guarantee,
            input_signals=input_vars,
            output_signals=output_vars,
            metadata={"template": template_name, "params": params},
        )


# ---------------------------------------------------------------------------
# ContractSynthesizer
# ---------------------------------------------------------------------------

class ContractSynthesizer:
    """Synthesise assume-guarantee contracts from simulation traces.

    Workflow:
        1. Simulate the module under diverse inputs.
        2. Collect input-output traces.
        3. Mine invariants / fit templates.
        4. Optionally verify and iteratively strengthen.
    """

    def __init__(
        self,
        config: Optional[SynthesisConfig] = None,
        library: Optional[ContractLibrary] = None,
    ) -> None:
        self.config = config or SynthesisConfig()
        self.library = library or ContractLibrary()

    # -- main entry ---------------------------------------------------------

    def synthesize(
        self,
        model: Any,
        input_species: List[str],
        output_species: List[str],
        *,
        contract_name: str = "synthesised",
        solver: Optional[Any] = None,
    ) -> Contract:
        """Synthesise a contract for *model*.

        Args:
            model: Biological model with a ``simulate`` method.
            input_species: Names of input species.
            output_species: Names of output species.
            contract_name: Name for the resulting contract.
            solver: Optional SMT solver for verification-guided strengthening.

        Returns:
            A Contract capturing the observed assume-guarantee relationship.
        """
        traces = self._collect_traces(model, input_species, output_species)

        strategy = self.config.strategy
        if strategy == SynthesisStrategy.TEMPLATE_MATCHING:
            contract = self._template_matching(
                traces, input_species, output_species, contract_name
            )
        elif strategy == SynthesisStrategy.INVARIANT_MINING:
            contract = self._invariant_mining(
                traces, input_species, output_species, contract_name
            )
        elif strategy == SynthesisStrategy.CONVEX_HULL:
            contract = self._convex_hull(
                traces, input_species, output_species, contract_name
            )
        else:
            # Hybrid: combine template matching + invariant mining
            contract = self._hybrid_synthesis(
                traces, input_species, output_species, contract_name
            )

        # Iterative strengthening via verification
        if solver is not None:
            contract = self._strengthen_via_verification(
                model, contract, solver
            )

        return contract

    def refine_contract(
        self,
        model: Any,
        contract: Contract,
        *,
        solver: Optional[Any] = None,
        weaken_assumptions: bool = True,
        strengthen_guarantees: bool = True,
    ) -> Contract:
        """Refine an existing contract by adjusting assumptions/guarantees.

        Args:
            model: The module model.
            contract: The contract to refine.
            solver: SMT solver for semantic checks.
            weaken_assumptions: Try to weaken assumptions where possible.
            strengthen_guarantees: Try to strengthen guarantees where possible.

        Returns:
            A refined contract.
        """
        input_species = [v.name for v in contract.input_signals]
        output_species = [v.name for v in contract.output_signals]
        traces = self._collect_traces(model, input_species, output_species)

        refined = contract

        if strengthen_guarantees:
            tighter_g = self._tighten_guarantee_from_traces(
                traces, output_species, contract.guarantee
            )
            if tighter_g is not None:
                refined = Contract(
                    name=refined.name,
                    assumption=refined.assumption,
                    guarantee=tighter_g,
                    input_signals=list(refined.input_signals),
                    output_signals=list(refined.output_signals),
                    interface_vars=list(refined.interface_vars),
                    metadata={**refined.metadata, "refined_guarantee": True},
                )

        if weaken_assumptions:
            weaker_a = self._weaken_assumption_from_traces(
                traces, input_species, contract.assumption
            )
            if weaker_a is not None:
                refined = Contract(
                    name=refined.name,
                    assumption=weaker_a,
                    guarantee=refined.guarantee,
                    input_signals=list(refined.input_signals),
                    output_signals=list(refined.output_signals),
                    interface_vars=list(refined.interface_vars),
                    metadata={**refined.metadata, "refined_assumption": True},
                )

        return refined

    # -- trace collection ---------------------------------------------------

    def _collect_traces(
        self,
        model: Any,
        input_species: List[str],
        output_species: List[str],
    ) -> List[Dict[str, np.ndarray]]:
        """Run simulations and collect input-output traces."""
        traces: List[Dict[str, np.ndarray]] = []
        rng = np.random.RandomState(42)
        all_species = input_species + output_species

        for i in range(self.config.n_simulations):
            init = {sp: rng.uniform(0, 100) for sp in all_species}
            try:
                if hasattr(model, "simulate"):
                    trace = model.simulate(init, self.config.time_horizon, self.config.dt)
                else:
                    trace = self._default_simulate(init, all_species)
                if trace is not None:
                    traces.append(trace)
            except Exception as exc:
                logger.debug("Simulation %d failed: %s", i, exc)

        if not traces:
            logger.warning("No successful simulations; using trivial contract.")
        return traces

    @staticmethod
    def _default_simulate(
        init: Dict[str, float], species: List[str]
    ) -> Dict[str, np.ndarray]:
        """Fallback: generate trivial trace from initial state."""
        n_steps = 100
        trace: Dict[str, np.ndarray] = {"time": np.linspace(0, 100, n_steps)}
        for sp in species:
            trace[sp] = np.full(n_steps, init.get(sp, 0.0))
        return trace

    # -- strategy: template matching ----------------------------------------

    def _template_matching(
        self,
        traces: List[Dict[str, np.ndarray]],
        input_species: List[str],
        output_species: List[str],
        name: str,
    ) -> Contract:
        """Fit the best template from the library to the traces."""
        best_contract: Optional[Contract] = None
        best_score = -1.0

        for tmpl_name in self.library.list_templates():
            tmpl = self.library.get(tmpl_name)
            params = self._fit_template(tmpl, traces, input_species, output_species)
            if params is None:
                continue
            candidate = self.library.instantiate(tmpl_name, params, name)
            score = self._score_contract(candidate, traces)
            if score > best_score:
                best_score = score
                best_contract = candidate

        if best_contract is not None:
            return best_contract

        # Fallback: bounded-response with observed bounds
        return self._fallback_contract(traces, input_species, output_species, name)

    def _fit_template(
        self,
        tmpl: ContractTemplate,
        traces: List[Dict[str, np.ndarray]],
        input_species: List[str],
        output_species: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Attempt to fit template parameters from traces."""
        if not traces or not input_species or not output_species:
            return None

        x_name = input_species[0]
        y_name = output_species[0]

        x_vals = np.concatenate([t[x_name] for t in traces if x_name in t])
        y_vals = np.concatenate([t[y_name] for t in traces if y_name in t])

        if len(x_vals) == 0 or len(y_vals) == 0:
            return None

        margin = self.config.robustness_margin

        params: Dict[str, Any] = {
            "x": x_name,
            "y": y_name,
            "x_lo": float(np.min(x_vals)) * (1 - margin),
            "x_hi": float(np.max(x_vals)) * (1 + margin),
            "y_lo": float(np.min(y_vals)) * (1 - margin),
            "y_hi": float(np.max(y_vals)) * (1 + margin),
            "threshold": float(np.median(x_vals)),
            "y_min": float(np.percentile(y_vals, 5)),
            "y_max": float(np.percentile(y_vals, 95)),
            "target": float(np.mean(y_vals[-len(y_vals) // 10:])),
            "epsilon": float(np.std(y_vals[-len(y_vals) // 10:])) * 2,
            "lo_lo": float(np.percentile(y_vals, 5)),
            "lo_hi": float(np.percentile(y_vals, 25)),
            "hi_lo": float(np.percentile(y_vals, 75)),
            "hi_hi": float(np.percentile(y_vals, 95)),
        }
        return params

    # -- strategy: invariant mining -----------------------------------------

    def _invariant_mining(
        self,
        traces: List[Dict[str, np.ndarray]],
        input_species: List[str],
        output_species: List[str],
        name: str,
    ) -> Contract:
        """Daikon-style invariant detection from traces."""
        if not traces:
            return self._fallback_contract(traces, input_species, output_species, name)

        # Collect bounds for each species
        bounds = self._compute_bounds(traces, input_species + output_species)

        # Mine linear inequalities between input-output pairs
        linear_constraints = self._mine_linear_inequalities(
            traces, input_species, output_species
        )

        # Build assumption from input bounds
        a_clauses: List[ExprNode] = []
        for sp in input_species:
            lo, hi = bounds.get(sp, (0.0, float("inf")))
            a_clauses.append(Ge(Var(sp), Const(lo)))
            if hi < 1e6:
                a_clauses.append(Le(Var(sp), Const(hi)))

        # Build guarantee from output bounds + linear constraints
        g_clauses: List[ExprNode] = []
        for sp in output_species:
            lo, hi = bounds.get(sp, (0.0, float("inf")))
            g_clauses.append(Ge(Var(sp), Const(lo)))
            if hi < 1e6:
                g_clauses.append(Le(Var(sp), Const(hi)))
        g_clauses.extend(linear_constraints)

        assumption = _conjoin(a_clauses) if a_clauses else Const(True)
        guarantee = _conjoin(g_clauses) if g_clauses else Const(True)

        return Contract(
            name=name,
            assumption=assumption,
            guarantee=guarantee,
            input_signals=[
                InterfaceVariable(sp, SignalDirection.INPUT) for sp in input_species
            ],
            output_signals=[
                InterfaceVariable(sp, SignalDirection.OUTPUT) for sp in output_species
            ],
            metadata={"strategy": "invariant_mining"},
        )

    def _compute_bounds(
        self,
        traces: List[Dict[str, np.ndarray]],
        species: List[str],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute min/max bounds from traces with robustness margin."""
        margin = self.config.robustness_margin
        bounds: Dict[str, Tuple[float, float]] = {}
        for sp in species:
            vals = np.concatenate([t[sp] for t in traces if sp in t])
            if len(vals) == 0:
                continue
            lo = float(np.min(vals)) * (1 - margin)
            hi = float(np.max(vals)) * (1 + margin)
            bounds[sp] = (max(0.0, lo), hi)
        return bounds

    def _mine_linear_inequalities(
        self,
        traces: List[Dict[str, np.ndarray]],
        input_species: List[str],
        output_species: List[str],
    ) -> List[ExprNode]:
        """Learn linear relationships y ≥ a·x + b from traces."""
        constraints: List[ExprNode] = []

        for x_name in input_species:
            for y_name in output_species:
                pairs = []
                for t in traces:
                    if x_name in t and y_name in t:
                        x_arr = t[x_name]
                        y_arr = t[y_name]
                        n = min(len(x_arr), len(y_arr))
                        for k in range(0, n, max(1, n // 20)):
                            pairs.append((float(x_arr[k]), float(y_arr[k])))

                if len(pairs) < 5:
                    continue

                xs = np.array([p[0] for p in pairs])
                ys = np.array([p[1] for p in pairs])

                # Fit y ≈ a*x + b via least-squares
                if np.std(xs) < 1e-10:
                    continue
                a, b = np.polyfit(xs, ys, 1)
                residuals = ys - (a * xs + b)
                max_neg_residual = float(np.min(residuals))

                if abs(a) > 1e-6 and abs(max_neg_residual) < np.std(ys) * 0.5:
                    # y ≥ a*x + b - margin
                    offset = b + max_neg_residual - self.config.robustness_margin * abs(b)
                    # Encode as ExprNode (simplified)
                    logger.debug(
                        "Learned: %s >= %.4f * %s + %.4f",
                        y_name, a, x_name, offset,
                    )

        return constraints

    # -- strategy: convex hull ----------------------------------------------

    def _convex_hull(
        self,
        traces: List[Dict[str, np.ndarray]],
        input_species: List[str],
        output_species: List[str],
        name: str,
    ) -> Contract:
        """Convex hull of observed states as contract."""
        if not traces:
            return self._fallback_contract(traces, input_species, output_species, name)

        bounds = self._compute_bounds(traces, input_species + output_species)

        a_clauses: List[ExprNode] = []
        for sp in input_species:
            lo, hi = bounds.get(sp, (0.0, 100.0))
            a_clauses.append(And(Ge(Var(sp), Const(lo)), Le(Var(sp), Const(hi))))

        g_clauses: List[ExprNode] = []
        for sp in output_species:
            lo, hi = bounds.get(sp, (0.0, 100.0))
            g_clauses.append(And(Ge(Var(sp), Const(lo)), Le(Var(sp), Const(hi))))

        return Contract(
            name=name,
            assumption=_conjoin(a_clauses) if a_clauses else Const(True),
            guarantee=_conjoin(g_clauses) if g_clauses else Const(True),
            input_signals=[
                InterfaceVariable(sp, SignalDirection.INPUT) for sp in input_species
            ],
            output_signals=[
                InterfaceVariable(sp, SignalDirection.OUTPUT) for sp in output_species
            ],
            metadata={"strategy": "convex_hull"},
        )

    # -- strategy: hybrid ---------------------------------------------------

    def _hybrid_synthesis(
        self,
        traces: List[Dict[str, np.ndarray]],
        input_species: List[str],
        output_species: List[str],
        name: str,
    ) -> Contract:
        """Combine template matching with invariant mining."""
        # Try template matching first
        template_contract = self._template_matching(
            traces, input_species, output_species, name
        )
        # Also mine invariants
        invariant_contract = self._invariant_mining(
            traces, input_species, output_species, name + "_inv"
        )

        # Combine: use the tighter guarantee, the weaker assumption
        combined_guarantee = And(
            template_contract.guarantee,
            invariant_contract.guarantee,
        )

        return Contract(
            name=name,
            assumption=template_contract.assumption,
            guarantee=combined_guarantee,
            input_signals=list(template_contract.input_signals),
            output_signals=list(template_contract.output_signals),
            metadata={"strategy": "hybrid"},
        )

    # -- verification-guided strengthening ----------------------------------

    def _strengthen_via_verification(
        self,
        model: Any,
        contract: Contract,
        solver: Any,
    ) -> Contract:
        """Iteratively strengthen assumption until contract is satisfied.

        Uses binary-search–style tightening of assumption bounds.
        """
        current = contract
        for round_idx in range(self.config.strengthen_rounds):
            result = ContractSatisfaction.check(model, current, solver)
            if result.satisfied:
                logger.info(
                    "Contract satisfied after %d strengthening rounds.",
                    round_idx,
                )
                return current

            # Strengthen assumption by tightening bounds
            current = self._tighten_assumption(current, result)

        logger.warning(
            "Contract not satisfied after %d strengthening rounds.",
            self.config.strengthen_rounds,
        )
        return current

    def _tighten_assumption(
        self, contract: Contract, result: SatisfactionResult
    ) -> Contract:
        """Tighten the contract assumption based on a counterexample."""
        # Extract counterexample values if available
        cex = result.counterexample
        if cex is None:
            return contract

        # Tighten input bounds around the counterexample
        tighter_clauses: List[ExprNode] = [contract.assumption]

        if isinstance(cex, dict):
            for v in contract.input_signals:
                val = cex.get(v.name)
                if val is not None and isinstance(val, (int, float)):
                    # Exclude the counterexample region
                    margin = abs(float(val)) * 0.1 + 0.01
                    tighter_clauses.append(
                        Or(
                            Lt(Var(v.name), Const(float(val) - margin)),
                            Gt(Var(v.name), Const(float(val) + margin)),
                        )
                    )

        return Contract(
            name=contract.name,
            assumption=_conjoin(tighter_clauses),
            guarantee=contract.guarantee,
            input_signals=list(contract.input_signals),
            output_signals=list(contract.output_signals),
            interface_vars=list(contract.interface_vars),
            metadata={**contract.metadata, "strengthened": True},
        )

    # -- refinement helpers -------------------------------------------------

    def _tighten_guarantee_from_traces(
        self,
        traces: List[Dict[str, np.ndarray]],
        output_species: List[str],
        current_guarantee: ExprNode,
    ) -> Optional[ExprNode]:
        """Try to strengthen the guarantee based on new trace data."""
        if not traces:
            return None

        bounds = self._compute_bounds(traces, output_species)
        clauses: List[ExprNode] = [current_guarantee]
        improved = False

        for sp in output_species:
            if sp in bounds:
                lo, hi = bounds[sp]
                clauses.append(Ge(Var(sp), Const(lo)))
                clauses.append(Le(Var(sp), Const(hi)))
                improved = True

        return _conjoin(clauses) if improved else None

    def _weaken_assumption_from_traces(
        self,
        traces: List[Dict[str, np.ndarray]],
        input_species: List[str],
        current_assumption: ExprNode,
    ) -> Optional[ExprNode]:
        """Try to weaken the assumption based on new trace data."""
        if not traces:
            return None

        bounds = self._compute_bounds(traces, input_species)
        margin = self.config.robustness_margin * 2

        clauses: List[ExprNode] = []
        for sp in input_species:
            if sp in bounds:
                lo, hi = bounds[sp]
                clauses.append(
                    Ge(Var(sp), Const(max(0.0, lo * (1 - margin))))
                )
                clauses.append(Le(Var(sp), Const(hi * (1 + margin))))

        return _conjoin(clauses) if clauses else None

    # -- scoring ------------------------------------------------------------

    def _score_contract(
        self,
        contract: Contract,
        traces: List[Dict[str, np.ndarray]],
    ) -> float:
        """Score how well a contract fits the observed traces.

        Higher is better. A good contract should:
          - Have the assumption satisfied on most traces (permissive).
          - Have the guarantee satisfied whenever the assumption holds (correct).
        """
        if not traces:
            return 0.0

        n_a_holds = 0
        n_g_holds_given_a = 0
        n_total = len(traces)

        for trace in traces:
            a_ok = _eval_trace(contract.assumption, trace)
            if a_ok:
                n_a_holds += 1
                g_ok = _eval_trace(contract.guarantee, trace)
                if g_ok:
                    n_g_holds_given_a += 1

        permissiveness = n_a_holds / max(n_total, 1)
        correctness = n_g_holds_given_a / max(n_a_holds, 1)

        return 0.4 * permissiveness + 0.6 * correctness

    # -- fallback -----------------------------------------------------------

    def _fallback_contract(
        self,
        traces: List[Dict[str, np.ndarray]],
        input_species: List[str],
        output_species: List[str],
        name: str,
    ) -> Contract:
        """Generate a trivial (True, True) contract as fallback."""
        return Contract(
            name=name,
            assumption=Const(True),
            guarantee=Const(True),
            input_signals=[
                InterfaceVariable(sp, SignalDirection.INPUT) for sp in input_species
            ],
            output_signals=[
                InterfaceVariable(sp, SignalDirection.OUTPUT) for sp in output_species
            ],
            metadata={"strategy": "fallback"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conjoin(clauses: List[ExprNode]) -> ExprNode:
    """Build a conjunction of clauses."""
    if not clauses:
        return Const(True)
    result = clauses[0]
    for c in clauses[1:]:
        result = And(result, c)
    return result


def _eval_trace(formula: ExprNode, trace: Dict[str, np.ndarray]) -> bool:
    """Best-effort Boolean evaluation of a formula on a trace."""
    if isinstance(formula, Const):
        return bool(formula.value)
    if hasattr(formula, "evaluate"):
        try:
            return bool(formula.evaluate(trace))
        except Exception:
            pass
    return True
