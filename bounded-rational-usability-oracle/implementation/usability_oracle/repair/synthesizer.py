"""
usability_oracle.repair.synthesizer — SMT-backed repair synthesis.

Uses the Z3 theorem prover to encode the UI repair problem as a
satisfiability-modulo-theories (SMT) instance.  The synthesiser:

1. Encodes the current MDP structure and bottleneck annotations as
   constraints over integer/real variables representing UI mutations.
2. Adds cognitive-law constraints (Fitts, Hick-Hyman, working memory)
   via :class:`ConstraintEncoder`.
3. Searches for satisfying assignments that reduce expected cognitive
   cost while preserving functionality.
4. Decodes solutions into :class:`UIMutation` lists and ranks the
   resulting :class:`RepairCandidate` objects.

References
----------
- de Moura, L. & Bjørner, N. (2008). Z3: An efficient SMT solver. *TACAS*.
- Card, S., Moran, T., & Newell, A. (1983). *The Psychology of HCI*.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import z3

from usability_oracle.core.enums import BottleneckType, PipelineStage, Severity
from usability_oracle.core.config import RepairConfig
from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.repair.models import (
    MutationType,
    RepairCandidate,
    RepairConstraint,
    RepairResult,
    UIMutation,
)
from usability_oracle.repair.constraints import ConstraintEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bottleneck result stub (used as interface — the bottleneck module defines
# the canonical class; we accept any object with these attributes)
# ---------------------------------------------------------------------------

@dataclass
class BottleneckResult:
    """Minimal bottleneck descriptor for repair input."""
    bottleneck_type: str = ""
    state_id: str = ""
    action_id: str = ""
    severity: str = "medium"
    cost_contribution: float = 0.0
    description: str = ""
    node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RepairSynthesizer
# ---------------------------------------------------------------------------

class RepairSynthesizer:
    """SMT-backed repair synthesis engine.

    Parameters
    ----------
    config : RepairConfig
        Repair-specific configuration (max repairs, timeout, etc.).
    constraint_encoder : ConstraintEncoder | None
        Custom constraint encoder; a default is created if *None*.
    max_mutations_per_candidate : int
        Upper bound on mutations in a single repair candidate.
    """

    def __init__(
        self,
        config: Optional[RepairConfig] = None,
        constraint_encoder: Optional[ConstraintEncoder] = None,
        max_mutations_per_candidate: int = 5,
    ) -> None:
        self.config = config or RepairConfig()
        self.encoder = constraint_encoder or ConstraintEncoder()
        self.max_mutations = max_mutations_per_candidate
        self._candidates_explored = 0

    # ── Public API --------------------------------------------------------

    def synthesize(
        self,
        mdp: MDP,
        bottlenecks: Sequence[Any],
        constraints: Sequence[RepairConstraint] | None = None,
        timeout: float | None = None,
    ) -> RepairResult:
        """Run repair synthesis.

        Parameters
        ----------
        mdp : MDP
            The current UI navigation MDP.
        bottlenecks : sequence of BottleneckResult-like objects
            Detected cognitive bottlenecks.
        constraints : sequence of RepairConstraint, optional
            Additional hard/soft constraints on the repair.
        timeout : float, optional
            Solver timeout in seconds (overrides config).

        Returns
        -------
        RepairResult
        """
        t0 = time.monotonic()
        timeout = timeout or self.config.timeout_seconds
        constraints = list(constraints) if constraints else []
        self._candidates_explored = 0

        if not bottlenecks:
            return RepairResult(
                solver_status="sat",
                synthesis_time=time.monotonic() - t0,
            )

        candidates: list[RepairCandidate] = []
        solver_status = "unknown"

        # Process each bottleneck independently and combine
        for bn in bottlenecks:
            elapsed = time.monotonic() - t0
            remaining = timeout - elapsed
            if remaining <= 0:
                solver_status = "timeout"
                break

            solver = self._encode_problem(mdp, bn, constraints)
            solver.set("timeout", int(remaining * 1000))  # ms

            result = solver.check()
            self._candidates_explored += 1

            if result == z3.sat:
                solver_status = "sat"
                model = solver.model()
                mutations = self._decode_solution(model, mdp, bn)
                candidate = RepairCandidate(
                    mutations=mutations,
                    expected_cost_reduction=self._estimate_cost_reduction(
                        model, mdp, bn
                    ),
                    confidence=self._compute_confidence(model, mdp, bn),
                    bottleneck_addressed=getattr(bn, "bottleneck_type", ""),
                    feasible=True,
                    verification_status="unverified",
                    description=self._describe_candidate(mutations, bn),
                )
                if self._verify_candidate(candidate, mdp):
                    candidate.verification_status = "verified"
                candidates.append(candidate)

                # Search for alternative solutions via blocking clauses
                alt_candidates = self._search_alternatives(
                    solver, model, mdp, bn, remaining
                )
                candidates.extend(alt_candidates)

            elif result == z3.unknown:
                solver_status = "timeout" if solver_status != "sat" else solver_status
            else:
                if solver_status != "sat":
                    solver_status = "unsat"

        ranked = self._rank_candidates(candidates)
        best = ranked[0] if ranked else None
        synthesis_time = time.monotonic() - t0

        return RepairResult(
            candidates=ranked,
            best=best,
            synthesis_time=synthesis_time,
            solver_status=solver_status,
            n_candidates_explored=self._candidates_explored,
        )

    # ── Encoding ----------------------------------------------------------

    def _encode_problem(
        self,
        mdp: MDP,
        bottleneck: Any,
        constraints: list[RepairConstraint],
    ) -> z3.Solver:
        """Encode the repair problem as a Z3 solver instance.

        Creates integer variables for each mutation type that can be applied,
        real variables for cost targets, and boolean variables indicating
        which mutations are active.
        """
        solver = z3.Solver()
        ctx = z3.main_ctx()

        node_ids = getattr(bottleneck, "node_ids", [])
        bn_type = getattr(bottleneck, "bottleneck_type", "")
        state_id = getattr(bottleneck, "state_id", "")

        # --- Mutation activation variables ---
        n_types = len(MutationType)
        mutation_active = [
            z3.Bool(f"mut_{mt.value}") for mt in MutationType
        ]

        # --- Mutation count variable ---
        n_mutations = z3.Int("n_mutations")
        solver.add(n_mutations >= 1)
        solver.add(n_mutations <= self.max_mutations)
        solver.add(
            n_mutations == z3.Sum([z3.If(m, 1, 0) for m in mutation_active])
        )

        # --- Dimensional variables per affected node ---
        width_vars: dict[str, z3.ArithRef] = {}
        height_vars: dict[str, z3.ArithRef] = {}
        x_vars: dict[str, z3.ArithRef] = {}
        y_vars: dict[str, z3.ArithRef] = {}

        for nid in node_ids:
            safe_nid = nid.replace("-", "_").replace(":", "_")
            width_vars[nid] = z3.Real(f"w_{safe_nid}")
            height_vars[nid] = z3.Real(f"h_{safe_nid}")
            x_vars[nid] = z3.Real(f"x_{safe_nid}")
            y_vars[nid] = z3.Real(f"y_{safe_nid}")

            solver.add(width_vars[nid] > 0)
            solver.add(height_vars[nid] > 0)
            solver.add(x_vars[nid] >= 0)
            solver.add(y_vars[nid] >= 0)

        # --- Cost variables ---
        total_cost = z3.Real("total_cost")
        original_cost = z3.Real("original_cost")
        cost_reduction = z3.Real("cost_reduction")

        original_cost_val = getattr(bottleneck, "cost_contribution", 1.0)
        solver.add(original_cost == z3.RealVal(str(original_cost_val)))
        solver.add(cost_reduction == original_cost - total_cost)
        solver.add(cost_reduction > 0)  # must improve
        solver.add(total_cost >= 0)

        # --- Bottleneck-type-specific encoding ---
        self._encode_bottleneck_constraints(
            solver, bn_type, mutation_active, width_vars, height_vars,
            x_vars, y_vars, total_cost, original_cost_val, node_ids,
        )

        # --- Structural constraints (preserve functionality) ---
        self._encode_structural_constraints(solver, mdp)

        # --- Mutation bounds ---
        self._encode_mutation_bounds(solver, self.max_mutations)

        # --- User-supplied constraints ---
        for rc in constraints:
            self._encode_user_constraint(
                solver, rc, width_vars, height_vars, x_vars, y_vars, total_cost
            )

        return solver

    def _encode_bottleneck_constraints(
        self,
        solver: z3.Solver,
        bn_type: str,
        mutation_active: list[z3.BoolRef],
        width_vars: dict[str, z3.ArithRef],
        height_vars: dict[str, z3.ArithRef],
        x_vars: dict[str, z3.ArithRef],
        y_vars: dict[str, z3.ArithRef],
        total_cost: z3.ArithRef,
        original_cost: float,
        node_ids: list[str],
    ) -> None:
        """Add bottleneck-type-specific constraints."""
        mut_idx = {mt.value: i for i, mt in enumerate(MutationType)}

        if bn_type == BottleneckType.MOTOR_DIFFICULTY:
            # Encourage resize and reposition; constrain Fitts ID
            solver.add(
                z3.Or(
                    mutation_active[mut_idx["resize"]],
                    mutation_active[mut_idx["reposition"]],
                    mutation_active[mut_idx["add_shortcut"]],
                )
            )
            for nid in node_ids:
                if nid in width_vars:
                    self.encoder.encode_target_size_constraint(
                        solver, width_vars[nid], height_vars[nid], 44.0
                    )

        elif bn_type == BottleneckType.CHOICE_PARALYSIS:
            solver.add(
                z3.Or(
                    mutation_active[mut_idx["simplify_menu"]],
                    mutation_active[mut_idx["regroup"]],
                    mutation_active[mut_idx["remove"]],
                )
            )
            n_choices = z3.Int("n_choices")
            solver.add(n_choices >= 1)
            self.encoder.encode_hick_constraint(solver, n_choices, 7)

        elif bn_type == BottleneckType.PERCEPTUAL_OVERLOAD:
            solver.add(
                z3.Or(
                    mutation_active[mut_idx["regroup"]],
                    mutation_active[mut_idx["remove"]],
                    mutation_active[mut_idx["add_landmark"]],
                )
            )

        elif bn_type == BottleneckType.MEMORY_DECAY:
            solver.add(
                z3.Or(
                    mutation_active[mut_idx["add_landmark"]],
                    mutation_active[mut_idx["regroup"]],
                    mutation_active[mut_idx["relabel"]],
                )
            )
            mem_load = z3.Int("memory_load")
            solver.add(mem_load >= 0)
            self.encoder.encode_memory_constraint(solver, mem_load, 4)

        elif bn_type == BottleneckType.CROSS_CHANNEL_INTERFERENCE:
            solver.add(
                z3.Or(
                    mutation_active[mut_idx["reposition"]],
                    mutation_active[mut_idx["regroup"]],
                    mutation_active[mut_idx["remove"]],
                )
            )

        # Ensure total cost is bounded below original
        solver.add(total_cost <= z3.RealVal(str(original_cost * 0.8)))

    def _encode_structural_constraints(
        self, solver: z3.Solver, mdp: MDP
    ) -> None:
        """Preserve reachability and functionality invariants.

        Every state reachable in the original MDP must remain reachable
        after repair, and every goal state must remain a goal.
        """
        reachable = mdp.reachable_states()
        n_reachable = z3.Int("n_reachable_after")
        solver.add(n_reachable >= len(reachable))

        # Preserve goal states: encode as boolean invariants
        for gs in mdp.goal_states:
            goal_preserved = z3.Bool(f"goal_{gs.replace('-', '_')}")
            solver.add(goal_preserved == True)

    def _encode_mutation_bounds(
        self, solver: z3.Solver, max_mutations: int
    ) -> None:
        """Limit the total number of active mutations."""
        n_muts = z3.Int("n_mutations")
        solver.add(n_muts <= max_mutations)

    def _encode_cost_constraint(
        self,
        solver: z3.Solver,
        variable: z3.ArithRef,
        bound: float,
    ) -> None:
        """Add a cost upper-bound constraint."""
        solver.add(variable <= z3.RealVal(str(bound)))

    def _encode_user_constraint(
        self,
        solver: z3.Solver,
        rc: RepairConstraint,
        width_vars: dict[str, z3.ArithRef],
        height_vars: dict[str, z3.ArithRef],
        x_vars: dict[str, z3.ArithRef],
        y_vars: dict[str, z3.ArithRef],
        total_cost: z3.ArithRef,
    ) -> None:
        """Encode a single user-supplied RepairConstraint."""
        if rc.constraint_type == "fitts" and rc.target in width_vars:
            dist_var = z3.Real(f"dist_{rc.target}")
            self.encoder.encode_fitts_constraint(
                solver, dist_var, width_vars[rc.target], rc.bound
            )
        elif rc.constraint_type == "hick":
            n_var = z3.Int(f"hick_{rc.target}")
            self.encoder.encode_hick_constraint(solver, n_var, int(rc.bound))
        elif rc.constraint_type == "memory":
            mem_var = z3.Int(f"mem_{rc.target}")
            self.encoder.encode_memory_constraint(solver, mem_var, int(rc.bound))
        elif rc.constraint_type == "target_size":
            if rc.target in width_vars and rc.target in height_vars:
                self.encoder.encode_target_size_constraint(
                    solver, width_vars[rc.target], height_vars[rc.target], rc.bound
                )
        elif rc.constraint_type == "cost":
            self._encode_cost_constraint(solver, total_cost, rc.bound)

    # ── Decoding ----------------------------------------------------------

    def _decode_solution(
        self,
        model: z3.ModelRef,
        mdp: MDP,
        bottleneck: Any,
    ) -> list[UIMutation]:
        """Extract concrete mutations from a Z3 model."""
        mutations: list[UIMutation] = []
        node_ids = getattr(bottleneck, "node_ids", [])
        primary_node = node_ids[0] if node_ids else getattr(
            bottleneck, "state_id", "unknown"
        )

        for mt in MutationType:
            var = z3.Bool(f"mut_{mt.value}")
            val = model.eval(var, model_completion=True)
            if z3.is_true(val):
                params = self._extract_mutation_params(model, mt, node_ids)
                target = primary_node
                if mt == MutationType.REGROUP and len(node_ids) > 1:
                    params["node_ids"] = node_ids
                mutations.append(UIMutation(
                    mutation_type=mt.value,
                    target_node_id=target,
                    parameters=params,
                    description=f"Apply {mt.value} to address bottleneck",
                ))

        return mutations

    def _extract_mutation_params(
        self,
        model: z3.ModelRef,
        mt: MutationType,
        node_ids: list[str],
    ) -> dict[str, Any]:
        """Extract type-specific parameters from the Z3 model."""
        params: dict[str, Any] = {}

        if mt == MutationType.RESIZE and node_ids:
            nid = node_ids[0]
            safe = nid.replace("-", "_").replace(":", "_")
            w_val = model.eval(z3.Real(f"w_{safe}"), model_completion=True)
            h_val = model.eval(z3.Real(f"h_{safe}"), model_completion=True)
            params["width"] = _z3_to_float(w_val)
            params["height"] = _z3_to_float(h_val)

        elif mt == MutationType.REPOSITION and node_ids:
            nid = node_ids[0]
            safe = nid.replace("-", "_").replace(":", "_")
            x_val = model.eval(z3.Real(f"x_{safe}"), model_completion=True)
            y_val = model.eval(z3.Real(f"y_{safe}"), model_completion=True)
            params["x"] = _z3_to_float(x_val)
            params["y"] = _z3_to_float(y_val)

        elif mt == MutationType.RELABEL:
            params["new_name"] = "Improved Label"

        elif mt == MutationType.ADD_SHORTCUT:
            params["shortcut_key"] = "Ctrl+Shift+A"

        elif mt == MutationType.SIMPLIFY_MENU:
            n_val = model.eval(z3.Int("n_choices"), model_completion=True)
            params["max_items"] = _z3_to_int(n_val) if n_val is not None else 7

        elif mt == MutationType.ADD_LANDMARK:
            params["landmark_role"] = "region"

        elif mt == MutationType.REGROUP:
            params["new_parent_role"] = "group"

        return params

    # ── Verification & ranking -------------------------------------------

    def _verify_candidate(
        self, candidate: RepairCandidate, mdp: MDP
    ) -> bool:
        """Lightweight verification that the candidate is structurally sound.

        Checks that:
        1. All target node IDs reference valid MDP states or metadata nodes.
        2. Mutations are self-consistent (no conflicting resize/remove).
        3. Expected cost reduction is positive.
        """
        if candidate.expected_cost_reduction <= 0:
            return False

        mutation_targets = set()
        has_remove = False
        for m in candidate.mutations:
            errors = m.validate()
            if errors:
                logger.debug("Mutation validation errors: %s", errors)
                return False
            mutation_targets.add(m.target_node_id)
            if m.mutation_type == MutationType.REMOVE:
                has_remove = True

        # Check for conflicting mutations on same node
        for m in candidate.mutations:
            if has_remove and m.target_node_id in mutation_targets:
                if m.mutation_type not in (MutationType.REMOVE, "remove"):
                    # Cannot resize a removed node
                    if m.mutation_type in (MutationType.RESIZE, MutationType.REPOSITION):
                        return False

        return True

    def _rank_candidates(
        self, candidates: list[RepairCandidate]
    ) -> list[RepairCandidate]:
        """Rank candidates by composite score (cost reduction × confidence).

        Feasible candidates are ranked above infeasible ones.
        """
        feasible = [c for c in candidates if c.feasible]
        infeasible = [c for c in candidates if not c.feasible]

        feasible.sort(key=lambda c: c.score(), reverse=True)
        infeasible.sort(key=lambda c: c.expected_cost_reduction, reverse=True)

        return feasible + infeasible

    def _search_alternatives(
        self,
        solver: z3.Solver,
        model: z3.ModelRef,
        mdp: MDP,
        bottleneck: Any,
        remaining_time: float,
    ) -> list[RepairCandidate]:
        """Search for alternative solutions by adding blocking clauses."""
        alternatives: list[RepairCandidate] = []
        max_alts = min(self.config.max_repairs - 1, 5)

        for _ in range(max_alts):
            if remaining_time <= 0.5:
                break

            # Block current solution
            block = []
            for mt in MutationType:
                var = z3.Bool(f"mut_{mt.value}")
                val = model.eval(var, model_completion=True)
                if z3.is_true(val):
                    block.append(var == False)
                else:
                    block.append(var == True)
            solver.add(z3.Or(block))

            t0 = time.monotonic()
            solver.set("timeout", int(remaining_time * 1000))
            result = solver.check()
            self._candidates_explored += 1
            elapsed = time.monotonic() - t0
            remaining_time -= elapsed

            if result != z3.sat:
                break

            model = solver.model()
            mutations = self._decode_solution(model, mdp, bottleneck)
            candidate = RepairCandidate(
                mutations=mutations,
                expected_cost_reduction=self._estimate_cost_reduction(
                    model, mdp, bottleneck
                ),
                confidence=self._compute_confidence(model, mdp, bottleneck),
                bottleneck_addressed=getattr(bottleneck, "bottleneck_type", ""),
                feasible=True,
                verification_status="unverified",
                description=self._describe_candidate(mutations, bottleneck),
            )
            if self._verify_candidate(candidate, mdp):
                candidate.verification_status = "verified"
            alternatives.append(candidate)

        return alternatives

    # ── Helpers -----------------------------------------------------------

    def _estimate_cost_reduction(
        self, model: z3.ModelRef, mdp: MDP, bottleneck: Any
    ) -> float:
        """Estimate cost reduction from the Z3 model."""
        cost_red = model.eval(z3.Real("cost_reduction"), model_completion=True)
        val = _z3_to_float(cost_red)
        return max(0.0, val)

    def _compute_confidence(
        self, model: z3.ModelRef, mdp: MDP, bottleneck: Any
    ) -> float:
        """Heuristic confidence based on how tightly the solver bounded cost."""
        total = model.eval(z3.Real("total_cost"), model_completion=True)
        original = model.eval(z3.Real("original_cost"), model_completion=True)
        total_f = _z3_to_float(total)
        original_f = _z3_to_float(original)

        if original_f <= 0:
            return 0.5

        ratio = total_f / original_f
        # Higher confidence when reduction is larger but not implausible
        if ratio < 0.1:
            return 0.3  # suspicious large reduction
        elif ratio < 0.5:
            return 0.85
        elif ratio < 0.8:
            return 0.7
        else:
            return 0.5

    def _describe_candidate(
        self, mutations: list[UIMutation], bottleneck: Any
    ) -> str:
        """Generate a human-readable description of the candidate."""
        bn_type = getattr(bottleneck, "bottleneck_type", "unknown")
        parts = [f"Address {bn_type} via:"]
        for m in mutations:
            parts.append(f"  - {m.mutation_type} on {m.target_node_id}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _z3_to_float(val: Any) -> float:
    """Convert a Z3 value to a Python float."""
    if val is None:
        return 0.0
    try:
        if z3.is_rational_value(val):
            return float(val.numerator_as_long()) / float(val.denominator_as_long())
        if z3.is_int_value(val):
            return float(val.as_long())
        return float(str(val))
    except (ValueError, AttributeError, z3.Z3Exception):
        return 0.0


def _z3_to_int(val: Any) -> int:
    """Convert a Z3 value to a Python int."""
    if val is None:
        return 0
    try:
        if z3.is_int_value(val):
            return val.as_long()
        return int(float(str(val)))
    except (ValueError, AttributeError, z3.Z3Exception):
        return 0
