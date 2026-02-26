"""
Orientation rules for causal graphs.

Implements Meek's rules (R1–R4) for CPDAG orientation, Zhang's rules
(R1–R10) for PAG orientation, and domain-specific financial causality
rules for systemic risk modelling.

References
----------
- Meek (1995). Causal inference and causal explanation with background
  knowledge.
- Zhang (2008). On the completeness of orientation rules for causal
  discovery in the presence of latent confounders and selection variables.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np

from .dag import DAGRepresentation, EdgeType


# ──────────────────────────────────────────────────────────────────────
# Domain rule specifications
# ──────────────────────────────────────────────────────────────────────

class RulePriority(Enum):
    """Priority levels for conflict resolution among orientation rules."""
    STRUCTURAL = 0      # Graph-theoretic (Meek / Zhang) – highest
    DOMAIN_HARD = 1     # Hard domain constraints
    DOMAIN_SOFT = 2     # Soft domain preferences
    DATA_DRIVEN = 3     # Data-driven orientations – lowest


@dataclass
class OrientationConstraint:
    """A single orientation constraint.

    If ``direction`` is ``(u, v)`` the constraint says *u* → *v* must hold.
    ``forbidden`` lists edges that must *not* exist.
    """
    direction: Optional[Tuple[str, str]] = None
    forbidden: List[Tuple[str, str]] = field(default_factory=list)
    priority: RulePriority = RulePriority.DOMAIN_HARD
    description: str = ""
    source: str = ""  # e.g. "meek_r1", "domain_margin_call"


# ──────────────────────────────────────────────────────────────────────
# Financial domain knowledge
# ──────────────────────────────────────────────────────────────────────

# Pre-defined financial causal axioms
FINANCIAL_AXIOMS: List[OrientationConstraint] = [
    # 1. Margin calls are caused by price moves
    OrientationConstraint(
        direction=("price_move", "margin_call"),
        priority=RulePriority.DOMAIN_HARD,
        description="Margin calls are caused by price moves, not vice versa.",
        source="domain_margin_call",
    ),
    OrientationConstraint(
        forbidden=[("margin_call", "price_move")],
        priority=RulePriority.DOMAIN_HARD,
        description="Margin calls do not cause price moves directly.",
        source="domain_margin_call_forbidden",
    ),
    # 2. Defaults are caused by insolvency
    OrientationConstraint(
        direction=("insolvency", "default"),
        priority=RulePriority.DOMAIN_HARD,
        description="Defaults are caused by insolvency, not by observation.",
        source="domain_default_insolvency",
    ),
    OrientationConstraint(
        forbidden=[("default", "insolvency")],
        priority=RulePriority.DOMAIN_HARD,
        description="Default does not cause insolvency (reverse not causal).",
        source="domain_default_forbidden",
    ),
    # 3. Fire sales are caused by forced liquidation
    OrientationConstraint(
        direction=("forced_liquidation", "fire_sale"),
        priority=RulePriority.DOMAIN_HARD,
        description="Fire sales are caused by forced liquidation.",
        source="domain_fire_sale",
    ),
    OrientationConstraint(
        direction=("margin_call", "forced_liquidation"),
        priority=RulePriority.DOMAIN_HARD,
        description="Forced liquidation follows margin calls.",
        source="domain_forced_liq",
    ),
    # 4. Funding withdrawals caused by credit deterioration
    OrientationConstraint(
        direction=("credit_deterioration", "funding_withdrawal"),
        priority=RulePriority.DOMAIN_HARD,
        description="Funding withdrawals are caused by credit deterioration.",
        source="domain_funding",
    ),
    OrientationConstraint(
        direction=("funding_withdrawal", "liquidity_crisis"),
        priority=RulePriority.DOMAIN_HARD,
        description="Liquidity crises follow funding withdrawals.",
        source="domain_liquidity",
    ),
    # 5. Contagion chain
    OrientationConstraint(
        direction=("fire_sale", "price_move"),
        priority=RulePriority.DOMAIN_SOFT,
        description="Fire sales cause price drops (contagion feedback).",
        source="domain_contagion",
    ),
    OrientationConstraint(
        direction=("default", "credit_deterioration"),
        priority=RulePriority.DOMAIN_SOFT,
        description="Defaults of counterparties deteriorate credit.",
        source="domain_credit_contagion",
    ),
]


def _get_financial_axioms_for_variables(
    variables: Set[str],
) -> List[OrientationConstraint]:
    """Filter financial axioms to those involving present variables.

    Also generates pattern-matched constraints for variables whose names
    contain known financial keywords.
    """
    applicable: List[OrientationConstraint] = []

    # Direct matches
    for axiom in FINANCIAL_AXIOMS:
        if axiom.direction is not None:
            u, v = axiom.direction
            if u in variables and v in variables:
                applicable.append(axiom)
        for fu, fv in axiom.forbidden:
            if fu in variables and fv in variables:
                applicable.append(axiom)
                break

    # Pattern-matched constraints
    keyword_patterns = {
        "margin": "margin_call",
        "default": "default",
        "fire_sale": "fire_sale",
        "liquidat": "forced_liquidation",
        "price": "price_move",
        "insolvency": "insolvency",
        "equity": "equity",
        "funding": "funding_withdrawal",
        "credit": "credit_deterioration",
        "liquidity": "liquidity_crisis",
    }

    var_to_role: Dict[str, str] = {}
    for var in variables:
        var_lower = var.lower()
        for keyword, role in keyword_patterns.items():
            if keyword in var_lower:
                var_to_role[var] = role
                break

    # Generate constraints from matched roles
    role_pairs_directed = [
        ("price_move", "margin_call"),
        ("margin_call", "forced_liquidation"),
        ("forced_liquidation", "fire_sale"),
        ("fire_sale", "price_move"),
        ("insolvency", "default"),
        ("credit_deterioration", "funding_withdrawal"),
        ("funding_withdrawal", "liquidity_crisis"),
    ]

    role_to_vars: Dict[str, List[str]] = {}
    for var, role in var_to_role.items():
        role_to_vars.setdefault(role, []).append(var)

    for cause_role, effect_role in role_pairs_directed:
        for cause_var in role_to_vars.get(cause_role, []):
            for effect_var in role_to_vars.get(effect_role, []):
                applicable.append(OrientationConstraint(
                    direction=(cause_var, effect_var),
                    priority=RulePriority.DOMAIN_SOFT,
                    description=f"Pattern: {cause_role} -> {effect_role}",
                    source="pattern_match",
                ))

    return applicable


# ──────────────────────────────────────────────────────────────────────
# Meek's orientation rules for CPDAGs
# ──────────────────────────────────────────────────────────────────────

class MeekRules:
    """Meek's four orientation rules for completing a CPDAG.

    Given a partially oriented DAG (with some directed and some
    undirected edges), these rules orient undirected edges so that
    no new v-structures or cycles are introduced.
    """

    @staticmethod
    def apply(dag: nx.DiGraph, undirected: Set[FrozenSet[str]]) -> Tuple[nx.DiGraph, Set[FrozenSet[str]]]:
        """Apply Meek's rules until convergence.

        Parameters
        ----------
        dag : nx.DiGraph
            The directed edges already known.
        undirected : set of frozensets
            The remaining undirected edges.

        Returns
        -------
        (dag, undirected) after orientation.
        """
        changed = True
        while changed:
            changed = False
            changed |= MeekRules._rule_r1(dag, undirected)
            changed |= MeekRules._rule_r2(dag, undirected)
            changed |= MeekRules._rule_r3(dag, undirected)
            changed |= MeekRules._rule_r4(dag, undirected)
        return dag, undirected

    @staticmethod
    def _rule_r1(dag: nx.DiGraph, undirected: Set[FrozenSet[str]]) -> bool:
        """R1: If A -> B — C and A not adj C, orient B -> C.

        Prevents creation of new v-structure A -> B <- C.
        """
        changed = False
        to_orient: List[Tuple[str, str]] = []

        for edge in list(undirected):
            b, c = tuple(edge)
            for direction in [(b, c), (c, b)]:
                node_b, node_c = direction
                for a in dag.predecessors(node_b):
                    if not dag.has_edge(a, node_c) and not dag.has_edge(node_c, a):
                        if frozenset({a, node_c}) not in undirected:
                            to_orient.append((node_b, node_c))
                            break

        for b, c in to_orient:
            edge = frozenset({b, c})
            if edge in undirected:
                undirected.discard(edge)
                dag.add_edge(b, c)
                changed = True
        return changed

    @staticmethod
    def _rule_r2(dag: nx.DiGraph, undirected: Set[FrozenSet[str]]) -> bool:
        """R2: If A -> B -> C and A — C, orient A -> C.

        Prevents creation of cycle A -> B -> C -> A.
        """
        changed = False
        to_orient: List[Tuple[str, str]] = []

        for edge in list(undirected):
            a, c = tuple(edge)
            for direction in [(a, c), (c, a)]:
                node_a, node_c = direction
                # Look for A -> B -> C
                for b in dag.successors(node_a):
                    if dag.has_edge(b, node_c):
                        to_orient.append((node_a, node_c))
                        break

        for a, c in to_orient:
            edge = frozenset({a, c})
            if edge in undirected:
                undirected.discard(edge)
                dag.add_edge(a, c)
                changed = True
        return changed

    @staticmethod
    def _rule_r3(dag: nx.DiGraph, undirected: Set[FrozenSet[str]]) -> bool:
        """R3: If A — B, A — C, A — D, B -> D, C -> D, B not adj C,
        orient A -> D.

        Prevents creation of new v-structure through D.
        """
        changed = False
        to_orient: List[Tuple[str, str]] = []

        nodes = list(dag.nodes)
        for a in nodes:
            undirected_neighbors = []
            for edge in undirected:
                if a in edge:
                    other = (edge - {a}).pop()
                    undirected_neighbors.append(other)

            for d in undirected_neighbors:
                # Find B, C: both undirected neighbors of A, both parents of D,
                # B and C not adjacent
                parent_candidates = [
                    n for n in undirected_neighbors
                    if n != d and dag.has_edge(n, d)
                ]
                for i in range(len(parent_candidates)):
                    for j in range(i + 1, len(parent_candidates)):
                        b, c = parent_candidates[i], parent_candidates[j]
                        if (not dag.has_edge(b, c) and not dag.has_edge(c, b)
                                and frozenset({b, c}) not in undirected):
                            to_orient.append((a, d))

        for a, d in to_orient:
            edge = frozenset({a, d})
            if edge in undirected:
                undirected.discard(edge)
                dag.add_edge(a, d)
                changed = True
        return changed

    @staticmethod
    def _rule_r4(dag: nx.DiGraph, undirected: Set[FrozenSet[str]]) -> bool:
        """R4: If A — B, B -> C, C -> D, A — D, A not adj C,
        orient A -> D.

        Prevents creation of cycle A — D -> ... -> B — A.
        """
        changed = False
        to_orient: List[Tuple[str, str]] = []

        for edge in list(undirected):
            a, d = tuple(edge)
            for direction in [(a, d), (d, a)]:
                node_a, node_d = direction
                # Find B: A — B and B -> C -> D
                for other_edge in undirected:
                    if node_a not in other_edge:
                        continue
                    b = (other_edge - {node_a}).pop()
                    if b == node_d:
                        continue
                    for c in dag.successors(b):
                        if c == node_a:
                            continue
                        if dag.has_edge(c, node_d):
                            if (not dag.has_edge(node_a, c) and
                                    not dag.has_edge(c, node_a) and
                                    frozenset({node_a, c}) not in undirected):
                                to_orient.append((node_a, node_d))

        for a, d in to_orient:
            edge = frozenset({a, d})
            if edge in undirected:
                undirected.discard(edge)
                dag.add_edge(a, d)
                changed = True
        return changed


# ──────────────────────────────────────────────────────────────────────
# OrientationRules  (main public class)
# ──────────────────────────────────────────────────────────────────────

class OrientationRules:
    """Orientation rules engine for causal graphs.

    Combines structural rules (Meek / Zhang) with domain-specific
    financial causality constraints, resolving conflicts by priority.

    Parameters
    ----------
    use_financial_axioms : bool
        If ``True`` (default), automatically include financial domain
        axioms when applicable.
    custom_rules : list[OrientationConstraint]
        Additional user-supplied orientation constraints.
    """

    def __init__(
        self,
        use_financial_axioms: bool = True,
        custom_rules: Optional[List[OrientationConstraint]] = None,
    ) -> None:
        self.use_financial_axioms = use_financial_axioms
        self.custom_rules: List[OrientationConstraint] = list(custom_rules or [])
        self._applied_rules: List[OrientationConstraint] = []
        self._conflicts: List[Tuple[OrientationConstraint, OrientationConstraint]] = []

    # ── public API ────────────────────────────────────────────────────

    def apply_domain_rules(
        self,
        pag: Any,
        domain_knowledge: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Apply domain-specific orientation rules to a PAG.

        Parameters
        ----------
        pag : PAG
            The partially oriented graph.
        domain_knowledge : dict, optional
            Additional domain info (e.g. ``{"forbidden": [("A","B")],
            "required": [("C","D")]``).
        """
        from .causal_discovery import PAG as PAGClass, Mark

        constraints = list(self.custom_rules)

        # Collect financial axioms if enabled
        if self.use_financial_axioms:
            variables = set(pag.variables)
            constraints.extend(_get_financial_axioms_for_variables(variables))

        # Add from domain_knowledge dict
        if domain_knowledge:
            for u, v in domain_knowledge.get("required", []):
                constraints.append(OrientationConstraint(
                    direction=(u, v),
                    priority=RulePriority.DOMAIN_HARD,
                    description=f"Required: {u} -> {v}",
                    source="user_required",
                ))
            for u, v in domain_knowledge.get("forbidden", []):
                constraints.append(OrientationConstraint(
                    forbidden=[(u, v)],
                    priority=RulePriority.DOMAIN_HARD,
                    description=f"Forbidden: {u} -> {v}",
                    source="user_forbidden",
                ))
            for u, v in domain_knowledge.get("soft_required", []):
                constraints.append(OrientationConstraint(
                    direction=(u, v),
                    priority=RulePriority.DOMAIN_SOFT,
                    description=f"Soft required: {u} -> {v}",
                    source="user_soft",
                ))

        # Resolve conflicts
        constraints = self.resolve_conflicts(constraints)

        # Apply each constraint
        for constraint in constraints:
            self._apply_single_constraint(pag, constraint)

        self._applied_rules = constraints
        return pag

    def apply_meek_rules(self, dag: DAGRepresentation) -> DAGRepresentation:
        """Apply Meek's orientation rules to a partially oriented DAG.

        Treats edges present only in one direction as directed, and
        edges present in both as undirected.  Returns a new DAG with
        as many edges oriented as possible.
        """
        G = dag.to_networkx()
        undirected: Set[FrozenSet[str]] = set()

        # Identify undirected edges (where neither direction is "definitive")
        # For simplicity: edges that appear only in one direction are directed;
        # we allow the caller to mark undirected edges via bidirected set
        for pair in dag._bidirected:
            undirected.add(pair)

        G, undirected = MeekRules.apply(G, undirected)

        # Build result
        result = DAGRepresentation(dag.nodes)
        for u, v in G.edges:
            if frozenset({u, v}) not in undirected:
                result.add_edge(u, v)
        for pair in undirected:
            u, v = tuple(pair)
            result.add_edge(u, v, EdgeType.BIDIRECTED)

        return result

    def apply_zhang_rules(self, pag: Any) -> Any:
        """Apply Zhang's FCI orientation rules (R1–R10) to a PAG.

        This is typically called from the FCI algorithm itself, but can
        also be applied independently to refine a PAG.
        """
        from .causal_discovery import FastCausalInference
        fci = FastCausalInference()
        return fci._apply_fci_rules(pag)

    def check_acyclicity(self, dag: DAGRepresentation) -> bool:
        """Return ``True`` if the DAG is acyclic."""
        return dag.is_dag()

    def resolve_conflicts(
        self, rules: List[OrientationConstraint]
    ) -> List[OrientationConstraint]:
        """Resolve conflicts among orientation rules by priority.

        If two rules give contradictory orientations for the same edge,
        the higher-priority rule wins.  Conflicts are recorded in
        ``self._conflicts``.
        """
        self._conflicts = []
        edge_to_rule: Dict[FrozenSet[str], OrientationConstraint] = {}
        resolved: List[OrientationConstraint] = []

        # Sort by priority (lower enum value = higher priority)
        sorted_rules = sorted(rules, key=lambda r: r.priority.value)

        for rule in sorted_rules:
            conflict_found = False
            if rule.direction is not None:
                key = frozenset(rule.direction)
                existing = edge_to_rule.get(key)
                if existing is not None:
                    # Check if conflicting direction
                    if (existing.direction is not None and
                            existing.direction != rule.direction and
                            existing.direction == (rule.direction[1], rule.direction[0])):
                        self._conflicts.append((existing, rule))
                        conflict_found = True
                        # Higher priority already recorded; skip this rule
                        continue
                edge_to_rule[key] = rule

            # Check forbidden edges for conflicts with required edges
            for fu, fv in rule.forbidden:
                fkey = frozenset({fu, fv})
                existing = edge_to_rule.get(fkey)
                if existing is not None and existing.direction == (fu, fv):
                    self._conflicts.append((existing, rule))
                    conflict_found = True

            if not conflict_found:
                resolved.append(rule)

        if self._conflicts:
            warnings.warn(
                f"{len(self._conflicts)} orientation rule conflict(s) resolved "
                f"by priority."
            )

        return resolved

    @property
    def conflicts(self) -> List[Tuple[OrientationConstraint, OrientationConstraint]]:
        return list(self._conflicts)

    @property
    def applied_rules(self) -> List[OrientationConstraint]:
        return list(self._applied_rules)

    # ── internal ──────────────────────────────────────────────────────

    def _apply_single_constraint(self, pag: Any, constraint: OrientationConstraint) -> None:
        """Apply a single orientation constraint to the PAG."""
        from .causal_discovery import Mark

        if constraint.direction is not None:
            u, v = constraint.direction
            if pag.has_edge(u, v):
                pag.set_mark(u, v, Mark.ARROW)
                pag.set_mark(v, u, Mark.TAIL)
            elif pag.has_edge(v, u):
                # Need to flip – but only if allowed
                pag.add_edge(u, v, mark_at_u=Mark.TAIL, mark_at_v=Mark.ARROW)
                pag.remove_edge(v, u)

        for fu, fv in constraint.forbidden:
            if pag.has_edge(fu, fv):
                mark = pag.get_mark(fu, fv)
                if mark is not None:
                    # Don't remove; just ensure it's not oriented fu -> fv
                    if pag.is_directed(fu, fv):
                        # Flip to fv -> fu if possible
                        pag.set_mark(fu, fv, Mark.TAIL)
                        pag.set_mark(fv, fu, Mark.ARROW)

    # ── convenience constructors ──────────────────────────────────────

    @classmethod
    def financial_default(cls) -> "OrientationRules":
        """Return an ``OrientationRules`` instance with standard financial axioms."""
        return cls(use_financial_axioms=True)

    @classmethod
    def structural_only(cls) -> "OrientationRules":
        """Return an ``OrientationRules`` instance with no domain rules."""
        return cls(use_financial_axioms=False)

    def add_rule(
        self,
        cause: str,
        effect: str,
        priority: RulePriority = RulePriority.DOMAIN_HARD,
        description: str = "",
    ) -> "OrientationRules":
        """Convenience: add a directed orientation rule."""
        self.custom_rules.append(OrientationConstraint(
            direction=(cause, effect),
            priority=priority,
            description=description,
            source="user_custom",
        ))
        return self

    def add_forbidden(
        self,
        u: str,
        v: str,
        priority: RulePriority = RulePriority.DOMAIN_HARD,
        description: str = "",
    ) -> "OrientationRules":
        """Convenience: add a forbidden-edge rule."""
        self.custom_rules.append(OrientationConstraint(
            forbidden=[(u, v)],
            priority=priority,
            description=description,
            source="user_custom",
        ))
        return self

    def __repr__(self) -> str:
        return (
            f"OrientationRules(financial={self.use_financial_axioms}, "
            f"custom_rules={len(self.custom_rules)}, "
            f"applied={len(self._applied_rules)}, "
            f"conflicts={len(self._conflicts)})"
        )
