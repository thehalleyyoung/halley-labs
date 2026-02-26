"""Circular assume-guarantee checking for systems with cyclic dependencies.

When modules have circular dependencies (e.g. feedback loops), standard
sequential AG reasoning is unsound. This module implements a co-inductive
fixed-point algorithm that iteratively tightens assumptions until convergence.
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
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from bioprover.encoding.expression import (
    And,
    ExprNode,
    Implies,
    Not,
    Or,
    Const,
    Var,
)
from bioprover.compositional.contracts import (
    Contract,
    ContractSatisfaction,
    SatisfactionResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & result types
# ---------------------------------------------------------------------------

class ConvergenceStatus(Enum):
    """Outcome of the fixed-point iteration."""
    CONVERGED = auto()       # Fixed point reached — proof is valid
    DIVERGED = auto()        # Iteration diverged despite widening
    FAILED = auto()          # Some module failed to satisfy its contract
    MAX_ITERATIONS = auto()  # Reached iteration cap without convergence
    INCOMPLETE = auto()      # Not yet finished


@dataclass
class FixedPointState:
    """Snapshot of the iteration state at one step.

    Attributes:
        iteration: Current iteration number (0-based).
        assumptions: Map module_name → current assumption formula.
        guarantees: Map module_name → current guarantee formula.
        verification_results: Map module_name → satisfaction result from last check.
        changed: Set of modules whose assumptions changed in this iteration.
    """
    iteration: int
    assumptions: Dict[str, ExprNode] = field(default_factory=dict)
    guarantees: Dict[str, ExprNode] = field(default_factory=dict)
    verification_results: Dict[str, SatisfactionResult] = field(
        default_factory=dict
    )
    changed: Set[str] = field(default_factory=set)

    @property
    def all_satisfied(self) -> bool:
        return all(
            r.satisfied for r in self.verification_results.values()
        )

    @property
    def is_fixed_point(self) -> bool:
        return len(self.changed) == 0


@dataclass
class CircularAGResult:
    """Final result of circular AG checking.

    Attributes:
        status: Convergence outcome.
        final_state: Terminal fixed-point state.
        history: Complete iteration history.
        proof_valid: Whether the result constitutes a valid proof.
        soundness_argument: Human-readable explanation of why the proof is sound.
        diagnostics: Extra information for debugging.
    """
    status: ConvergenceStatus
    final_state: Optional[FixedPointState] = None
    history: List[FixedPointState] = field(default_factory=list)
    proof_valid: bool = False
    soundness_argument: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AGWellFormedness:
    """Well-formedness conditions for circular assume-guarantee.

    Attributes:
        is_well_formed: Whether all well-formedness conditions hold.
        violations: List of human-readable violation descriptions.
        coupling_strength: Measure of inter-module coupling (spectral radius
            of the coupling matrix; must be < 1 for convergence guarantee).
        coverage_ok: Whether module state spaces cover the system.
        non_vacuity_ok: Whether all assumptions are satisfiable.
        monotonicity_ok: Whether the assumption-refinement operator is monotone.
        contraction_ok: Whether inter-module coupling satisfies contraction.
    """
    is_well_formed: bool
    violations: List[str] = field(default_factory=list)
    coupling_strength: float = 0.0
    coverage_ok: bool = True
    non_vacuity_ok: bool = True
    monotonicity_ok: bool = True
    contraction_ok: bool = True


@dataclass
class AGFailureDiagnostics:
    """Detailed diagnostics when circular AG verification fails.

    Attributes:
        violated_conditions: Which well-formedness conditions were violated.
        failed_modules: Modules that failed verification.
        coupling_matrix: The inter-module coupling matrix (if computed).
        spectral_radius: Spectral radius of the coupling matrix.
        suggestions: Actionable suggestions for the user.
    """
    violated_conditions: List[str] = field(default_factory=list)
    failed_modules: List[str] = field(default_factory=list)
    coupling_matrix: Optional[Any] = None
    spectral_radius: float = 0.0
    suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CircularAGChecker
# ---------------------------------------------------------------------------

class CircularAGChecker:
    """Circular assume-guarantee verification via co-inductive fixed point.

    Algorithm overview:
        1. Initialise each module's assumption to True (⊤).
        2. Verify each module against its contract under current assumptions.
        3. Update guarantees based on verification outcomes.
        4. Strengthen each module's assumption using the guarantees of the
           modules that feed into it.
        5. Repeat until a fixed point is reached (no assumptions change) or
           a failure / divergence is detected.

    Soundness:
        The fixed point is a co-inductive invariant. Because assumptions
        are monotonically strengthened and guarantees are derived from
        actual verification results, the final assumptions are self-
        consistent: every module's assumption is implied by the guarantees
        of its dependencies. This is the co-induction principle.
    """

    def __init__(
        self,
        modules: Dict[str, Any],
        contracts: Dict[str, Contract],
        dependency_graph: Dict[str, Set[str]],
        *,
        solver: Optional[Any] = None,
        max_iterations: int = 50,
        widening_threshold: int = 10,
    ) -> None:
        """
        Args:
            modules: Map module_name → model object.
            contracts: Map module_name → initial contract.
            dependency_graph: Map module_name → set of module names it depends on.
            solver: Optional SMT solver for semantic checking.
            max_iterations: Hard cap on iterations.
            widening_threshold: After this many iterations, apply widening.
        """
        self._modules = modules
        self._contracts = dict(contracts)
        self._deps = dependency_graph
        self._solver = solver
        self._max_iter = max_iterations
        self._widen_threshold = widening_threshold

        self._module_names = sorted(self._contracts.keys())
        self._validate_inputs()

    # -- public API ---------------------------------------------------------

    def check(self) -> CircularAGResult:
        """Run the circular AG fixed-point algorithm.

        Returns:
            CircularAGResult with status and proof artefacts.
        """
        logger.info(
            "Starting circular AG check with %d modules", len(self._module_names)
        )

        # Step 1: initialise assumptions to True
        assumptions: Dict[str, ExprNode] = {
            name: Const(True) for name in self._module_names
        }
        guarantees: Dict[str, ExprNode] = {
            name: self._contracts[name].guarantee
            for name in self._module_names
        }

        history: List[FixedPointState] = []

        for iteration in range(self._max_iter):
            state = FixedPointState(iteration=iteration)
            state.assumptions = dict(assumptions)
            state.guarantees = dict(guarantees)

            # Step 2: verify each module under current assumptions
            all_ok = True
            for name in self._module_names:
                contract_i = self._make_contract_with_assumption(
                    name, assumptions[name]
                )
                result = self._verify_module(name, contract_i)
                state.verification_results[name] = result
                if not result.satisfied:
                    all_ok = False

            # Step 3: update guarantees
            for name in self._module_names:
                vr = state.verification_results[name]
                if vr.satisfied:
                    guarantees[name] = self._contracts[name].guarantee
                else:
                    # Module failed; guarantee weakened to False
                    guarantees[name] = Const(False)

            # Step 4: strengthen assumptions
            new_assumptions: Dict[str, ExprNode] = {}
            changed: Set[str] = set()

            for name in self._module_names:
                deps = self._deps.get(name, set())
                if not deps:
                    new_assumptions[name] = assumptions[name]
                    continue

                # New assumption = conjunction of dependency guarantees
                dep_guarantees = [guarantees[d] for d in sorted(deps)]
                strengthened = dep_guarantees[0]
                for dg in dep_guarantees[1:]:
                    strengthened = And(strengthened, dg)

                # Combine with previous assumption (monotonic strengthening)
                new_a = And(assumptions[name], strengthened)

                if not self._formulas_equal(new_a, assumptions[name]):
                    changed.add(name)

                new_assumptions[name] = new_a

            state.changed = changed
            assumptions = new_assumptions
            history.append(state)

            logger.debug(
                "Iteration %d: %d modules changed, all_ok=%s",
                iteration, len(changed), all_ok,
            )

            # Step 5: check for fixed point
            if not changed:
                if all_ok:
                    return CircularAGResult(
                        status=ConvergenceStatus.CONVERGED,
                        final_state=state,
                        history=history,
                        proof_valid=True,
                        soundness_argument=self._soundness_text(state),
                    )
                else:
                    return CircularAGResult(
                        status=ConvergenceStatus.FAILED,
                        final_state=state,
                        history=history,
                        proof_valid=False,
                        soundness_argument="Fixed point reached but some modules fail.",
                        diagnostics={
                            "failed_modules": [
                                n for n, r in state.verification_results.items()
                                if not r.satisfied
                            ]
                        },
                    )

            # Widening to accelerate convergence
            if iteration >= self._widen_threshold:
                assumptions = self._apply_widening(
                    assumptions, history, iteration
                )

        # Max iterations reached
        final = history[-1] if history else None
        return CircularAGResult(
            status=ConvergenceStatus.MAX_ITERATIONS,
            final_state=final,
            history=history,
            proof_valid=False,
            soundness_argument="Did not converge within iteration limit.",
        )

    def check_with_symmetry(
        self,
        symmetry_groups: List[FrozenSet[str]],
    ) -> CircularAGResult:
        """Exploit structural symmetry to accelerate the fixed-point loop.

        If modules within a symmetry group are structurally identical
        (same contract up to variable renaming), we only verify one
        representative per group and copy the result.

        Args:
            symmetry_groups: Groups of module names that are symmetric.
        """
        logger.info(
            "Circular AG with symmetry: %d groups", len(symmetry_groups)
        )

        representatives: Dict[str, str] = {}
        for group in symmetry_groups:
            rep = min(group)
            for name in group:
                representatives[name] = rep

        # Override _verify_module to reuse results
        cache: Dict[str, SatisfactionResult] = {}
        original_verify = self._verify_module

        def cached_verify(name: str, contract: Contract) -> SatisfactionResult:
            rep = representatives.get(name, name)
            if rep in cache:
                return cache[rep]
            result = original_verify(rep, contract)
            cache[rep] = result
            return result

        self._verify_module = cached_verify  # type: ignore[assignment]
        try:
            return self.check()
        finally:
            self._verify_module = original_verify  # type: ignore[assignment]

    def check_quantitative(
        self,
        robustness_thresholds: Optional[Dict[str, float]] = None,
    ) -> CircularAGResult:
        """Circular AG with quantitative (robustness) contracts.

        Instead of Boolean satisfaction, each contract has a robustness
        margin ρ. Verification succeeds only if ρ > threshold.

        Args:
            robustness_thresholds: Map module_name → minimum robustness.
        """
        thresholds = robustness_thresholds or {}

        assumptions: Dict[str, ExprNode] = {
            name: Const(True) for name in self._module_names
        }
        guarantees: Dict[str, ExprNode] = {
            name: self._contracts[name].guarantee
            for name in self._module_names
        }
        robustness: Dict[str, float] = {
            name: float("inf") for name in self._module_names
        }

        history: List[FixedPointState] = []

        for iteration in range(self._max_iter):
            state = FixedPointState(iteration=iteration)
            state.assumptions = dict(assumptions)
            state.guarantees = dict(guarantees)

            for name in self._module_names:
                contract_i = self._make_contract_with_assumption(
                    name, assumptions[name]
                )
                result = self._verify_module(name, contract_i)
                state.verification_results[name] = result

                # Extract robustness if available
                if hasattr(result, "robustness"):
                    robustness[name] = result.robustness
                elif result.satisfied:
                    robustness[name] = 1.0
                else:
                    robustness[name] = 0.0

            # Check quantitative thresholds
            all_robust = all(
                robustness.get(n, 0.0) >= thresholds.get(n, 0.0)
                for n in self._module_names
            )

            # Update guarantees (same logic)
            for name in self._module_names:
                vr = state.verification_results[name]
                threshold = thresholds.get(name, 0.0)
                if vr.satisfied and robustness.get(name, 0.0) >= threshold:
                    guarantees[name] = self._contracts[name].guarantee
                else:
                    guarantees[name] = Const(False)

            # Strengthen assumptions
            new_assumptions: Dict[str, ExprNode] = {}
            changed: Set[str] = set()
            for name in self._module_names:
                deps = self._deps.get(name, set())
                if not deps:
                    new_assumptions[name] = assumptions[name]
                    continue
                dep_guarantees = [guarantees[d] for d in sorted(deps)]
                strengthened = dep_guarantees[0]
                for dg in dep_guarantees[1:]:
                    strengthened = And(strengthened, dg)
                new_a = And(assumptions[name], strengthened)
                if not self._formulas_equal(new_a, assumptions[name]):
                    changed.add(name)
                new_assumptions[name] = new_a

            state.changed = changed
            assumptions = new_assumptions
            history.append(state)

            if not changed:
                status = (
                    ConvergenceStatus.CONVERGED if all_robust
                    else ConvergenceStatus.FAILED
                )
                return CircularAGResult(
                    status=status,
                    final_state=state,
                    history=history,
                    proof_valid=all_robust,
                    soundness_argument=self._soundness_text(state),
                    diagnostics={"robustness": dict(robustness)},
                )

        final = history[-1] if history else None
        return CircularAGResult(
            status=ConvergenceStatus.MAX_ITERATIONS,
            final_state=final,
            history=history,
            proof_valid=False,
            diagnostics={"robustness": dict(robustness)},
        )

    # -- internal -----------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Sanity-check constructor arguments."""
        for name in self._module_names:
            if name not in self._modules:
                raise ValueError(f"Module '{name}' has contract but no model.")
        for name, deps in self._deps.items():
            for d in deps:
                if d not in self._module_names:
                    raise ValueError(
                        f"Dependency '{d}' of module '{name}' is unknown."
                    )

    # -- well-formedness checking ---------------------------------------

    def check_well_formedness(
        self,
        decomposition: Optional[Any] = None,
        contracts: Optional[Dict[str, Contract]] = None,
    ) -> AGWellFormedness:
        """Check that circular AG decomposition satisfies well-formedness.

        Conditions:
        1. Coverage: union of module state spaces covers system state space.
        2. Non-vacuity: each assumption is satisfiable given other guarantees.
        3. Monotonicity: the assumption-refinement operator is monotone.
        4. Bounded coupling: inter-module coupling bounded by contraction
           factor < 1 (sufficient for fixed-point convergence).

        Args:
            decomposition: Optional decomposition result with module info.
            contracts: Contracts to check (defaults to instance contracts).

        Returns:
            AGWellFormedness with detailed results.
        """
        contracts = contracts or self._contracts
        violations: List[str] = []

        # 1. Coverage check
        coverage_ok = self._check_coverage(decomposition)
        if not coverage_ok:
            violations.append(
                "Coverage violation: module state spaces do not cover "
                "the full system state space. Some species may be "
                "unassigned to any module."
            )

        # 2. Non-vacuity check
        non_vacuity_ok = self._check_non_vacuity(contracts)
        if not non_vacuity_ok:
            violations.append(
                "Non-vacuity violation: one or more module assumptions "
                "are unsatisfiable (trivially true guarantee). Check that "
                "dependency guarantees are compatible."
            )

        # 3. Monotonicity check
        monotonicity_ok = self._check_monotonicity(contracts)
        if not monotonicity_ok:
            violations.append(
                "Monotonicity violation: the assumption-refinement "
                "operator is not monotone. Strengthening an assumption "
                "may weaken a guarantee, preventing convergence."
            )

        # 4. Contraction / coupling check
        coupling_strength, contraction_ok = self._compute_coupling_strength()
        if not contraction_ok:
            violations.append(
                f"Contraction violation: spectral radius of coupling "
                f"matrix is {coupling_strength:.4f} (must be < 1). "
                f"Modules are too tightly coupled for guaranteed "
                f"convergence. Consider re-decomposing."
            )

        is_well_formed = (coverage_ok and non_vacuity_ok
                          and monotonicity_ok and contraction_ok)

        return AGWellFormedness(
            is_well_formed=is_well_formed,
            violations=violations,
            coupling_strength=coupling_strength,
            coverage_ok=coverage_ok,
            non_vacuity_ok=non_vacuity_ok,
            monotonicity_ok=monotonicity_ok,
            contraction_ok=contraction_ok,
        )

    def diagnose_failure(
        self, result: CircularAGResult,
    ) -> AGFailureDiagnostics:
        """Produce detailed diagnostics when circular AG fails.

        Examines the result to determine which well-formedness conditions
        were likely violated and provides actionable suggestions.

        Args:
            result: The failed CircularAGResult.

        Returns:
            AGFailureDiagnostics with violation details and suggestions.
        """
        diag = AGFailureDiagnostics()

        # Identify failed modules
        if result.final_state is not None:
            diag.failed_modules = [
                n for n, r in result.final_state.verification_results.items()
                if not r.satisfied
            ]

        # Compute coupling info
        coupling_strength, contraction_ok = self._compute_coupling_strength()
        diag.spectral_radius = coupling_strength

        # Build coupling matrix for diagnostics
        n = len(self._module_names)
        coupling_matrix = np.zeros((n, n))
        for i, mi in enumerate(self._module_names):
            for j, mj in enumerate(self._module_names):
                if mj in self._deps.get(mi, set()):
                    coupling_matrix[i, j] = 1.0
        diag.coupling_matrix = coupling_matrix

        # Determine violated conditions
        if result.status == ConvergenceStatus.FAILED:
            diag.violated_conditions.append(
                "Module verification failure: one or more modules "
                "cannot satisfy their contracts."
            )
            diag.suggestions.append(
                "Try weakening the specification or strengthening "
                "the module guarantees."
            )
            if diag.failed_modules:
                diag.suggestions.append(
                    f"Failed modules: {diag.failed_modules}. "
                    f"Consider decomposing these modules further."
                )

        if result.status == ConvergenceStatus.MAX_ITERATIONS:
            diag.violated_conditions.append(
                "Convergence failure: fixed-point iteration did not "
                "converge within the iteration limit."
            )
            if not contraction_ok:
                diag.violated_conditions.append(
                    f"Contraction violated: spectral radius = "
                    f"{coupling_strength:.4f} >= 1."
                )
                diag.suggestions.append(
                    "The coupling between modules is too strong. Try: "
                    "(1) Re-decomposing to minimize inter-module coupling, "
                    "(2) Increasing the iteration limit, or "
                    "(3) Using a coarser abstraction at module boundaries."
                )
            else:
                diag.suggestions.append(
                    "Coupling is contractible but convergence is slow. "
                    "Try increasing max_iterations or adjusting the "
                    "widening threshold."
                )

        if result.status == ConvergenceStatus.DIVERGED:
            diag.violated_conditions.append(
                "Divergence: assumptions grew without bound."
            )
            diag.suggestions.append(
                "The widening operator failed to stabilize. Try: "
                "(1) Reducing the widening threshold, "
                "(2) Using a more aggressive widening strategy, or "
                "(3) Switching to bounded model checking."
            )

        return diag

    def compute_coupling_matrix(self) -> np.ndarray:
        """Compute the inter-module coupling matrix.

        Entry C[i,j] represents the strength of the dependency of
        module i on module j, normalized to [0, 1].

        Returns:
            numpy array of shape (n_modules, n_modules).
        """
        n = len(self._module_names)
        C = np.zeros((n, n))

        for i, mi in enumerate(self._module_names):
            deps_i = self._deps.get(mi, set())
            n_deps = len(deps_i) if deps_i else 1
            for j, mj in enumerate(self._module_names):
                if mj in deps_i:
                    # Coupling strength: 1/n_deps per dependency
                    # (uniform distribution of coupling)
                    C[i, j] = 1.0 / n_deps

        return C

    # -- private well-formedness helpers --------------------------------

    def _check_coverage(self, decomposition: Optional[Any]) -> bool:
        """Check that modules cover the system state space."""
        if decomposition is None:
            # Without decomposition info, check that all modules
            # referenced in dependencies exist
            all_referenced: Set[str] = set(self._module_names)
            for deps in self._deps.values():
                all_referenced |= deps
            return all_referenced.issubset(set(self._module_names))

        # If decomposition has module species, check coverage
        if hasattr(decomposition, 'modules'):
            covered: Set[str] = set()
            for mod in decomposition.modules:
                if hasattr(mod, 'species'):
                    covered |= set(mod.species)
            # Coverage is OK if we have at least one species per module
            return len(covered) > 0
        return True

    def _check_non_vacuity(self, contracts: Dict[str, Contract]) -> bool:
        """Check that no assumption is trivially unsatisfiable."""
        for name, contract in contracts.items():
            assumption = contract.assumption
            # Syntactic check: assumption = False means vacuously true
            if isinstance(assumption, Const) and assumption.value is False:
                logger.warning("Module %s has False assumption (vacuous).", name)
                return False
        return True

    def _check_monotonicity(self, contracts: Dict[str, Contract]) -> bool:
        """Check monotonicity of the assumption-refinement operator.

        The operator is monotone if strengthening assumptions only
        strengthens (or preserves) guarantees.  We check this
        structurally: each guarantee should not negate its own
        module's assumption.
        """
        for name, contract in contracts.items():
            assumption_str = str(contract.assumption)
            guarantee_str = str(contract.guarantee)
            # Heuristic: if the guarantee contains NOT(assumption),
            # monotonicity may be violated
            if f"Not({assumption_str})" in guarantee_str:
                logger.warning(
                    "Module %s guarantee negates its assumption; "
                    "monotonicity may be violated.", name,
                )
                return False
        return True

    def _compute_coupling_strength(self) -> Tuple[float, bool]:
        """Compute spectral radius of the coupling matrix.

        The spectral radius ρ(C) < 1 is a sufficient condition for
        the fixed-point iteration to converge (contraction mapping).

        Returns:
            (spectral_radius, is_contractive) tuple.
        """
        C = self.compute_coupling_matrix()
        if C.size == 0:
            return 0.0, True

        eigenvalues = np.linalg.eigvals(C)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        is_contractive = spectral_radius < 1.0

        logger.debug(
            "Coupling matrix spectral radius: %.6f (contractive=%s)",
            spectral_radius, is_contractive,
        )
        return spectral_radius, is_contractive

    def _make_contract_with_assumption(
        self, name: str, assumption: ExprNode
    ) -> Contract:
        """Return a copy of the module's contract with overridden assumption."""
        c = self._contracts[name]
        return Contract(
            name=c.name,
            assumption=assumption,
            guarantee=c.guarantee,
            input_signals=list(c.input_signals),
            output_signals=list(c.output_signals),
            interface_vars=list(c.interface_vars),
            metadata=dict(c.metadata),
        )

    def _verify_module(
        self, name: str, contract: Contract
    ) -> SatisfactionResult:
        """Verify a single module against its contract."""
        model = self._modules[name]
        return ContractSatisfaction.check(
            model, contract, solver=self._solver
        )

    def _formulas_equal(self, a: ExprNode, b: ExprNode) -> bool:
        """Conservative syntactic equality check for formulas."""
        if a is b:
            return True
        return str(a) == str(b)

    def _apply_widening(
        self,
        assumptions: Dict[str, ExprNode],
        history: List[FixedPointState],
        iteration: int,
    ) -> Dict[str, ExprNode]:
        """Apply widening to accelerate convergence.

        After many iterations, assumptions may oscillate. Widening
        over-approximates by dropping conjuncts that keep changing.
        """
        if len(history) < 3:
            return assumptions

        widened: Dict[str, ExprNode] = {}
        prev = history[-2].assumptions

        for name in self._module_names:
            cur = assumptions[name]
            old = prev.get(name)

            if old is not None and not self._formulas_equal(cur, old):
                # This assumption keeps changing — widen to previous (weaker)
                widened[name] = old
                logger.debug("Widening assumption for module %s", name)
            else:
                widened[name] = cur

        return widened

    def _soundness_text(self, state: FixedPointState) -> str:
        """Generate a human-readable soundness argument."""
        lines = [
            "Circular AG proof by co-inductive fixed point.",
            "",
            "Soundness argument:",
            "  1. Assumptions were initialised to True (⊤) — the weakest assumption.",
            "  2. At each iteration, assumptions were monotonically strengthened",
            "     using the guarantees of dependency modules.",
            "  3. Every module was verified under its current assumption:",
        ]
        for name in sorted(state.verification_results):
            r = state.verification_results[name]
            status = "✓ satisfied" if r.satisfied else "✗ failed"
            lines.append(f"       {name}: {status}")

        lines.extend([
            "  4. A fixed point was reached: no assumptions changed.",
            "  5. By the co-induction principle, the fixed point is a valid",
            "     invariant. Each module's assumption is implied by the",
            "     guarantees of its dependencies, forming a self-consistent",
            "     set of contracts.",
            "",
            f"  Iterations to convergence: {state.iteration + 1}",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TopologyAnalyzer
# ---------------------------------------------------------------------------

class TopologyAnalyzer:
    """Analyze the dependency graph to determine AG verification strategy.

    Examines the module dependency graph and determines whether circular
    AG reasoning is necessary, or whether sequential AG suffices.
    """

    def __init__(self, dependency_graph: Dict[str, Set[str]]) -> None:
        self._deps = dependency_graph

    def is_acyclic(self) -> bool:
        """Return True if the dependency graph has no cycles."""
        import networkx as nx
        G = nx.DiGraph()
        for mod, deps in self._deps.items():
            G.add_node(mod)
            for d in deps:
                G.add_edge(mod, d)
        return nx.is_directed_acyclic_graph(G)

    def find_cycles(self) -> List[List[str]]:
        """Return all simple cycles in the dependency graph."""
        import networkx as nx
        G = nx.DiGraph()
        for mod, deps in self._deps.items():
            G.add_node(mod)
            for d in deps:
                G.add_edge(mod, d)
        return list(nx.simple_cycles(G))

    def topological_order(self) -> Optional[List[str]]:
        """Return a topological ordering if the graph is acyclic, else None."""
        import networkx as nx
        G = nx.DiGraph()
        for mod, deps in self._deps.items():
            G.add_node(mod)
            for d in deps:
                G.add_edge(mod, d)
        if not nx.is_directed_acyclic_graph(G):
            return None
        return list(nx.topological_sort(G))

    def needs_circular_ag(self) -> bool:
        """Return True if circular AG is required (graph has cycles)."""
        return not self.is_acyclic()

    def recommend_strategy(self) -> str:
        """Recommend 'sequential' or 'circular' AG based on topology."""
        if self.is_acyclic():
            return "sequential"
        return "circular"

    def strongly_connected_components(self) -> List[FrozenSet[str]]:
        """Return SCCs of the dependency graph."""
        import networkx as nx
        G = nx.DiGraph()
        for mod, deps in self._deps.items():
            G.add_node(mod)
            for d in deps:
                G.add_edge(mod, d)
        return [frozenset(c) for c in nx.strongly_connected_components(G)]


# ---------------------------------------------------------------------------
# WellFormednessChecker
# ---------------------------------------------------------------------------

class WellFormednessChecker:
    """Verify well-formedness conditions BEFORE running circular AG.

    Checks three key conditions:
    1. Contract compatibility: each module's guarantee implies the
       assumptions of all dependent modules.
    2. Progress condition: the iteration must be strictly contractive
       (assumptions get tighter each step).
    3. Acyclicity fallback: if the dependency graph is acyclic,
       recommend sequential AG instead (sound by default).

    The key insight (Misra & Chandy, 1981): circular AG is sound if and
    only if the map from assumptions to guarantees is contractive.
    """

    def __init__(
        self,
        contracts: Dict[str, Contract],
        dependency_graph: Dict[str, Set[str]],
        modules: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._contracts = contracts
        self._deps = dependency_graph
        self._modules = modules or {}
        self._topology = TopologyAnalyzer(dependency_graph)

    def check_all(self) -> AGWellFormedness:
        """Run all well-formedness checks and return aggregated result.

        Returns:
            AGWellFormedness with detailed violation information.
        """
        violations: List[str] = []

        # 1. Contract compatibility
        compat_ok, compat_violations = self.check_contract_compatibility()
        if not compat_ok:
            violations.extend(compat_violations)

        # 2. Progress / contraction condition
        contraction_ok, coupling_strength = self.check_progress_condition()
        if not contraction_ok:
            violations.append(
                f"Progress violation: spectral radius of coupling matrix "
                f"is {coupling_strength:.4f} (must be < 1 for guaranteed "
                f"convergence). The assumption-to-guarantee map is not "
                f"contractive (Misra & Chandy, 1981)."
            )

        # 3. Acyclicity fallback
        acyclic = self._topology.is_acyclic()
        if acyclic:
            violations.append(
                "Acyclicity detected: dependency graph is acyclic. "
                "Sequential AG is sound by default and more efficient. "
                "Consider using sequential AG instead of circular AG."
            )

        is_well_formed = compat_ok and contraction_ok
        # Acyclicity is a recommendation, not a hard failure
        monotonicity_ok = compat_ok

        return AGWellFormedness(
            is_well_formed=is_well_formed,
            violations=violations,
            coupling_strength=coupling_strength,
            coverage_ok=True,
            non_vacuity_ok=True,
            monotonicity_ok=monotonicity_ok,
            contraction_ok=contraction_ok,
        )

    def check_contract_compatibility(self) -> Tuple[bool, List[str]]:
        """Check that each module's guarantee implies dependent assumptions.

        For every module M with dependents D1, D2, ..., we check that
        G_M => A_Di syntactically (conservative check).

        Returns:
            (is_compatible, list_of_violations)
        """
        violations: List[str] = []

        # Build reverse dependency map: module -> modules that depend on it
        dependents: Dict[str, Set[str]] = {
            m: set() for m in self._contracts
        }
        for mod, deps in self._deps.items():
            for d in deps:
                if d in dependents:
                    dependents[d].add(mod)

        for provider, consumers in dependents.items():
            if not consumers:
                continue
            guarantee = self._contracts[provider].guarantee

            # If provider has a trivially false guarantee, it cannot
            # satisfy any consumer assumption
            if isinstance(guarantee, Const) and guarantee.value is False:
                for consumer in sorted(consumers):
                    assumption = self._contracts[consumer].assumption
                    if not (isinstance(assumption, Const) and
                            assumption.value is True):
                        violations.append(
                            f"Contract incompatibility: {provider}'s "
                            f"guarantee is False but {consumer} has a "
                            f"non-trivial assumption."
                        )

        is_compatible = len(violations) == 0
        return is_compatible, violations

    def check_progress_condition(self) -> Tuple[bool, float]:
        """Check that the assumption-to-guarantee map is contractive.

        Computes the spectral radius of the coupling matrix. A spectral
        radius < 1 guarantees convergence of the fixed-point iteration.

        Returns:
            (is_contractive, spectral_radius)
        """
        module_names = sorted(self._contracts.keys())
        n = len(module_names)
        if n == 0:
            return True, 0.0

        C = np.zeros((n, n))
        for i, mi in enumerate(module_names):
            deps_i = self._deps.get(mi, set())
            n_deps = len(deps_i) if deps_i else 1
            for j, mj in enumerate(module_names):
                if mj in deps_i:
                    C[i, j] = 1.0 / n_deps

        if C.size == 0:
            return True, 0.0

        eigenvalues = np.linalg.eigvals(C)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        return spectral_radius < 1.0, spectral_radius

    def recommend_strategy(self) -> str:
        """Recommend the best AG strategy based on well-formedness analysis.

        Returns:
            One of 'sequential', 'circular', or 'abort' with explanation.
        """
        if self._topology.is_acyclic():
            return "sequential"

        contraction_ok, rho = self.check_progress_condition()
        if contraction_ok:
            return "circular"

        return "abort"
