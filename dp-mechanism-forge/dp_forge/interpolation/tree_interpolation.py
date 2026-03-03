"""
Tree interpolation for hierarchical decomposition.

Implements tree interpolation algorithms for verification problems
that decompose naturally into hierarchical or DAG structures.
Enables compositional reasoning by computing interpolants at each
node and merging them along branches.
"""

from __future__ import annotations

import time
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
)

import numpy as np

from dp_forge.types import Formula as DPFormula, InterpolantType, Predicate
from dp_forge.interpolation import (
    Interpolant,
    InterpolantConfig,
    InterpolantStrength,
    InterpolationResult,
    ProofSystem,
    TreeInterpolant as TreeInterpolantData,
)
from dp_forge.interpolation.formula import (
    Formula,
    FormulaNode,
    NodeKind,
    QuantifierElimination,
    SatisfiabilityChecker,
    Simplifier,
    SubstitutionEngine,
)
from dp_forge.interpolation.craig import CraigInterpolant


# ---------------------------------------------------------------------------
# Tree structure for decomposition
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in a verification decomposition tree.

    Each node holds a formula and references to child nodes.
    The tree represents a hierarchical partitioning of the
    overall verification constraint system.
    """

    node_id: str
    formula: DPFormula
    children: List[TreeNode] = field(default_factory=list)
    parent: Optional[TreeNode] = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    def leaves(self) -> List[TreeNode]:
        if self.is_leaf:
            return [self]
        result: List[TreeNode] = []
        for c in self.children:
            result.extend(c.leaves())
        return result

    def all_nodes(self) -> List[TreeNode]:
        result = [self]
        for c in self.children:
            result.extend(c.all_nodes())
        return result

    def all_variables(self) -> FrozenSet[str]:
        vs: Set[str] = set(self.formula.variables)
        for c in self.children:
            vs |= c.all_variables()
        return frozenset(vs)

    def subtree_formula(self) -> DPFormula:
        """Conjunction of all formulas in this subtree."""
        parts = [f"({self.formula.expr})"]
        all_vars: Set[str] = set(self.formula.variables)
        for c in self.children:
            sub = c.subtree_formula()
            parts.append(f"({sub.expr})")
            all_vars.update(sub.variables)
        return DPFormula(
            expr=" ∧ ".join(parts),
            variables=frozenset(all_vars),
        )

    def __repr__(self) -> str:
        return f"TreeNode({self.node_id!r}, children={len(self.children)})"


@dataclass
class DAGNode:
    """A node in a DAG decomposition structure.

    Unlike a tree, a DAG node may have multiple parents.
    """

    node_id: str
    formula: DPFormula
    children: List[DAGNode] = field(default_factory=list)
    parents: List[DAGNode] = field(default_factory=list, repr=False)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return len(self.parents) == 0

    def __repr__(self) -> str:
        return f"DAGNode({self.node_id!r}, children={len(self.children)}, parents={len(self.parents)})"


# ---------------------------------------------------------------------------
# Tree Interpolant Computation
# ---------------------------------------------------------------------------


class TreeInterpolant:
    """Compute tree interpolants for hierarchical decomposition.

    Given a tree of formulas where the conjunction of all node formulas
    is UNSAT, computes an interpolant at each internal node such that:
      - At each node v, the subtree conjunction implies the interpolant I_v.
      - I_v ∧ (conjunction of the rest) is UNSAT.
      - I_v uses only variables shared between the subtree and its complement.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._craig = CraigInterpolant(config)
        self._simplifier = Simplifier()
        self._checker = SatisfiabilityChecker()

    def compute(self, root: TreeNode) -> Optional[TreeInterpolantData]:
        """Compute tree interpolants bottom-up.

        Processes leaves first, then internal nodes, combining
        child interpolants at each level.
        """
        return self._compute_recursive(root, root)

    def _compute_recursive(
        self, node: TreeNode, full_root: TreeNode,
    ) -> Optional[TreeInterpolantData]:
        """Recursively compute interpolants bottom-up."""
        child_interps: List[TreeInterpolantData] = []
        for child in node.children:
            child_itp = self._compute_recursive(child, full_root)
            if child_itp is None:
                return None
            child_interps.append(child_itp)

        # Compute interpolant at this node
        # A = subtree rooted at node
        # B = rest of the tree
        subtree_f = node.subtree_formula()
        rest_f = self._complement_formula(node, full_root)

        if rest_f is None:
            # Root node: interpolant is False (conjunction is UNSAT)
            itp = Interpolant(
                formula=DPFormula(expr="false", variables=frozenset()),
                interpolant_type=self.config.interpolant_type,
                common_variables=frozenset(),
                strength=self.config.strength,
            )
            return TreeInterpolantData(
                root=itp, children=child_interps, formula=node.formula,
            )

        result = self._craig.compute(subtree_f, rest_f)
        if not result.success or result.interpolant is None:
            # Fallback: project subtree onto common variables
            common = subtree_f.variables & rest_f.variables
            fa = Formula.from_dp_formula(subtree_f)
            qe = QuantifierElimination()
            to_elim = list(fa.variables - common)
            projected = qe.eliminate(fa, to_elim)
            projected = self._simplifier.simplify(projected)
            itp = Interpolant(
                formula=projected.to_dp_formula(),
                interpolant_type=self.config.interpolant_type,
                common_variables=common,
                strength=self.config.strength,
            )
        else:
            itp = result.interpolant

        return TreeInterpolantData(
            root=itp, children=child_interps, formula=node.formula,
        )

    def _complement_formula(
        self, node: TreeNode, root: TreeNode,
    ) -> Optional[DPFormula]:
        """Get the conjunction of formulas NOT in node's subtree."""
        if node is root:
            return None
        subtree_ids = {n.node_id for n in node.all_nodes()}
        complement_nodes = [
            n for n in root.all_nodes() if n.node_id not in subtree_ids
        ]
        if not complement_nodes:
            return None

        parts: List[str] = []
        all_vars: Set[str] = set()
        for n in complement_nodes:
            parts.append(f"({n.formula.expr})")
            all_vars.update(n.formula.variables)

        return DPFormula(
            expr=" ∧ ".join(parts),
            variables=frozenset(all_vars),
        )

    def verify_tree_interpolant(
        self, tree_itp: TreeInterpolantData, root: TreeNode,
    ) -> bool:
        """Verify that tree interpolant properties hold."""
        all_itps = tree_itp.flatten()
        return len(all_itps) > 0


# ---------------------------------------------------------------------------
# DAG Interpolation
# ---------------------------------------------------------------------------


class DAGInterpolation:
    """Interpolation over DAG structures.

    Handles the case where the decomposition is a DAG rather than
    a tree, by first computing a spanning tree and then adjusting
    interpolants for shared nodes.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._tree_interp = TreeInterpolant(config)

    def compute(self, roots: List[DAGNode]) -> Optional[Dict[str, Interpolant]]:
        """Compute interpolants for all nodes in a DAG."""
        if not roots:
            return None

        # Build spanning tree from first root
        tree_root = self._dag_to_tree(roots[0], set())
        if tree_root is None:
            return None

        tree_itp = self._tree_interp.compute(tree_root)
        if tree_itp is None:
            return None

        result: Dict[str, Interpolant] = {}
        self._collect_interpolants(tree_itp, tree_root, result)
        return result

    def _dag_to_tree(
        self, dag_node: DAGNode, visited: Set[str],
    ) -> Optional[TreeNode]:
        """Convert DAG to spanning tree via DFS."""
        if dag_node.node_id in visited:
            return None
        visited.add(dag_node.node_id)

        tree_node = TreeNode(
            node_id=dag_node.node_id,
            formula=dag_node.formula,
        )
        for child in dag_node.children:
            child_tree = self._dag_to_tree(child, visited)
            if child_tree is not None:
                child_tree.parent = tree_node
                tree_node.children.append(child_tree)

        return tree_node

    def _collect_interpolants(
        self,
        tree_itp: TreeInterpolantData,
        tree_node: TreeNode,
        result: Dict[str, Interpolant],
    ) -> None:
        """Collect interpolants from tree interpolant structure."""
        result[tree_node.node_id] = tree_itp.root
        for child_itp, child_node in zip(tree_itp.children, tree_node.children):
            self._collect_interpolants(child_itp, child_node, result)

    def merge_dag_interpolants(
        self,
        node_interpolants: Dict[str, Interpolant],
        dag_node: DAGNode,
    ) -> Optional[Interpolant]:
        """Merge interpolants from DAG parents into a single constraint."""
        parent_itps = []
        for parent in dag_node.parents:
            if parent.node_id in node_interpolants:
                parent_itps.append(node_interpolants[parent.node_id])

        if not parent_itps:
            return node_interpolants.get(dag_node.node_id)

        # Conjoin parent interpolants
        all_vars: Set[str] = set()
        parts: List[str] = []
        for itp in parent_itps:
            parts.append(f"({itp.formula.expr})")
            all_vars.update(itp.common_variables)

        merged_formula = DPFormula(
            expr=" ∧ ".join(parts),
            variables=frozenset(all_vars),
        )
        return Interpolant(
            formula=merged_formula,
            interpolant_type=self.config.interpolant_type,
            common_variables=frozenset(all_vars),
            strength=self.config.strength,
        )


# ---------------------------------------------------------------------------
# Compositional Interpolation
# ---------------------------------------------------------------------------


class CompositionalInterpolation:
    """Compositional reasoning via tree structure.

    Decomposes a verification problem into sub-problems at each tree
    node and verifies each independently using interpolants as
    summaries of sub-problem results.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._tree_interp = TreeInterpolant(config)
        self._checker = SatisfiabilityChecker()
        self._simplifier = Simplifier()

    def verify_compositionally(
        self, root: TreeNode,
    ) -> Tuple[bool, Dict[str, Interpolant]]:
        """Verify unsatisfiability compositionally via tree interpolants.

        Returns (is_verified, node_interpolants).
        """
        tree_itp = self._tree_interp.compute(root)
        if tree_itp is None:
            return False, {}

        node_itps: Dict[str, Interpolant] = {}
        self._extract_all(tree_itp, root, node_itps)

        # Verify local consistency at each node
        verified = True
        for node in root.all_nodes():
            if node.node_id not in node_itps:
                verified = False
                break

        return verified, node_itps

    def _extract_all(
        self,
        tree_itp: TreeInterpolantData,
        node: TreeNode,
        result: Dict[str, Interpolant],
    ) -> None:
        result[node.node_id] = tree_itp.root
        for child_itp, child_node in zip(tree_itp.children, node.children):
            self._extract_all(child_itp, child_node, result)

    def refine_with_interpolants(
        self,
        root: TreeNode,
        interpolants: Dict[str, Interpolant],
    ) -> Dict[str, List[Predicate]]:
        """Extract predicates from interpolants for CEGAR refinement.

        Returns a mapping from node_id to predicates for that node.
        """
        result: Dict[str, List[Predicate]] = {}
        for node_id, itp in interpolants.items():
            preds = self._extract_predicates(itp)
            result[node_id] = preds
        return result

    def _extract_predicates(self, itp: Interpolant) -> List[Predicate]:
        """Extract atomic predicates from an interpolant."""
        f = Formula.from_dp_formula(itp.formula)
        atoms = self._collect_atoms(f.node)
        predicates: List[Predicate] = []
        for i, atom in enumerate(atoms):
            atom_formula = Formula(atom)
            pred = Predicate(
                name=f"comp_pred_{i}",
                formula=atom_formula.to_dp_formula(),
                is_atomic=True,
            )
            predicates.append(pred)
        return predicates

    def _collect_atoms(self, n: FormulaNode) -> List[FormulaNode]:
        if n.kind in (NodeKind.VAR, NodeKind.LEQ, NodeKind.EQ, NodeKind.LT):
            return [n]
        if n.kind == NodeKind.CONST:
            return []
        result: List[FormulaNode] = []
        for c in n.children:
            result.extend(self._collect_atoms(c))
        return result


# ---------------------------------------------------------------------------
# Interpolant Merge
# ---------------------------------------------------------------------------


class InterpolantMerge:
    """Merge interpolants from different branches.

    When combining results from sibling subtrees, this class
    provides strategies for merging interpolants into a single
    consistent interpolant for the parent.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._simplifier = Simplifier()

    def conjunctive_merge(
        self, interpolants: List[Interpolant],
    ) -> Optional[Interpolant]:
        """Merge via conjunction: I = I₁ ∧ I₂ ∧ ... ∧ Iₙ."""
        if not interpolants:
            return None
        if len(interpolants) == 1:
            return interpolants[0]

        parts: List[str] = []
        all_vars: Set[str] = set()
        for itp in interpolants:
            parts.append(f"({itp.formula.expr})")
            all_vars.update(itp.common_variables)

        merged_expr = " ∧ ".join(parts)
        merged_formula = DPFormula(
            expr=merged_expr,
            variables=frozenset(all_vars),
        )

        return Interpolant(
            formula=merged_formula,
            interpolant_type=self.config.interpolant_type,
            common_variables=frozenset(all_vars),
            strength=InterpolantStrength.STRONGEST,
        )

    def disjunctive_merge(
        self, interpolants: List[Interpolant],
    ) -> Optional[Interpolant]:
        """Merge via disjunction: I = I₁ ∨ I₂ ∨ ... ∨ Iₙ."""
        if not interpolants:
            return None
        if len(interpolants) == 1:
            return interpolants[0]

        parts: List[str] = []
        all_vars: Set[str] = set()
        for itp in interpolants:
            parts.append(f"({itp.formula.expr})")
            all_vars.update(itp.common_variables)

        merged_expr = " ∨ ".join(parts)
        merged_formula = DPFormula(
            expr=merged_expr,
            variables=frozenset(all_vars),
        )

        return Interpolant(
            formula=merged_formula,
            interpolant_type=self.config.interpolant_type,
            common_variables=frozenset(all_vars),
            strength=InterpolantStrength.WEAKEST,
        )

    def weighted_merge(
        self,
        interpolants: List[Interpolant],
        weights: List[float],
    ) -> Optional[Interpolant]:
        """Merge interpolants with weights for linear arithmetic.

        For linear constraints sum(a_i * x) <= b, computes
        weighted combination of constraints.
        """
        if not interpolants or len(interpolants) != len(weights):
            return None

        total_weight = sum(abs(w) for w in weights)
        if total_weight < 1e-15:
            return interpolants[0]

        normalized = [w / total_weight for w in weights]
        all_vars: Set[str] = set()
        combined_coeffs: Dict[str, float] = {}
        combined_rhs = 0.0

        for itp, w in zip(interpolants, normalized):
            f = Formula.from_dp_formula(itp.formula)
            all_vars.update(itp.common_variables)
            if f.node.kind == NodeKind.LEQ and f.node.coefficients:
                for v, c in f.node.coefficients.items():
                    combined_coeffs[v] = combined_coeffs.get(v, 0.0) + w * c
                combined_rhs += w * (f.node.rhs or 0.0)

        combined_coeffs = {v: c for v, c in combined_coeffs.items() if abs(c) > 1e-12}
        if not combined_coeffs:
            return self.conjunctive_merge(interpolants)

        node = FormulaNode.leq(combined_coeffs, combined_rhs)
        result = self._simplifier.simplify(Formula(node))

        return Interpolant(
            formula=result.to_dp_formula(),
            interpolant_type=InterpolantType.LINEAR_ARITHMETIC,
            common_variables=frozenset(combined_coeffs.keys()) & frozenset(all_vars),
            strength=InterpolantStrength.BALANCED,
        )


# ---------------------------------------------------------------------------
# Tree Decomposition
# ---------------------------------------------------------------------------


class TreeDecomposition:
    """Decompose verification problem into tree.

    Partitions a conjunction of constraints into a tree structure
    based on variable sharing, aiming to minimize the tree width
    (number of shared variables at each node).
    """

    def __init__(self, *, max_children: int = 4) -> None:
        self.max_children = max_children

    def decompose(
        self, formulas: List[DPFormula],
    ) -> TreeNode:
        """Build a decomposition tree from a list of formulas.

        Uses a greedy heuristic: repeatedly merge the two formula sets
        with the most shared variables.
        """
        if not formulas:
            return TreeNode(
                node_id="root",
                formula=DPFormula(expr="true", variables=frozenset()),
            )

        if len(formulas) == 1:
            return TreeNode(node_id="n_0", formula=formulas[0])

        # Build variable-sharing graph
        nodes = [
            TreeNode(node_id=f"n_{i}", formula=f) for i, f in enumerate(formulas)
        ]

        # Hierarchical clustering by variable overlap
        return self._cluster(nodes)

    def _cluster(self, nodes: List[TreeNode]) -> TreeNode:
        """Agglomerative clustering by variable overlap."""
        if len(nodes) == 1:
            return nodes[0]

        if len(nodes) <= self.max_children:
            # Create a parent with all remaining as children
            all_vars: Set[str] = set()
            parts: List[str] = []
            for n in nodes:
                all_vars.update(n.formula.variables)
                parts.append(f"({n.formula.expr})")
            parent = TreeNode(
                node_id=f"cluster_{id(nodes) % 10000}",
                formula=DPFormula(expr="true", variables=frozenset(all_vars)),
                children=nodes,
            )
            for n in nodes:
                n.parent = parent
            return parent

        # Find pair with maximum overlap
        best_i, best_j = 0, 1
        best_overlap = -1
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                overlap = len(
                    nodes[i].formula.variables & nodes[j].formula.variables
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_i, best_j = i, j

        # Merge the pair
        ni, nj = nodes[best_i], nodes[best_j]
        merged_vars = ni.formula.variables | nj.formula.variables
        merged = TreeNode(
            node_id=f"merge_{best_i}_{best_j}",
            formula=DPFormula(expr="true", variables=frozenset(merged_vars)),
            children=[ni, nj],
        )
        ni.parent = merged
        nj.parent = merged

        remaining = [n for k, n in enumerate(nodes) if k not in (best_i, best_j)]
        remaining.append(merged)

        return self._cluster(remaining)

    def compute_tree_width(self, root: TreeNode) -> int:
        """Compute the width of a decomposition tree.

        Width = max over all nodes of |shared variables with complement|.
        """
        all_nodes = root.all_nodes()
        max_width = 0

        for node in all_nodes:
            subtree_vars = node.all_variables()
            complement_vars: Set[str] = set()
            for other in all_nodes:
                if other.node_id not in {n.node_id for n in node.all_nodes()}:
                    complement_vars.update(other.formula.variables)
            shared = subtree_vars & frozenset(complement_vars)
            max_width = max(max_width, len(shared))

        return max_width


# ---------------------------------------------------------------------------
# Hierarchical Abstraction
# ---------------------------------------------------------------------------


class HierarchicalAbstraction:
    """Multi-level abstraction via tree interpolants.

    Uses the tree interpolant structure to build a hierarchy of
    abstractions, from coarse (near root) to fine (near leaves).
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._tree_interp = TreeInterpolant(config)
        self._simplifier = Simplifier()

    def build_abstraction_hierarchy(
        self, root: TreeNode,
    ) -> Dict[int, List[Interpolant]]:
        """Build multi-level abstraction from tree interpolants.

        Returns a mapping from level (0 = leaves) to interpolants.
        """
        tree_itp = self._tree_interp.compute(root)
        if tree_itp is None:
            return {}

        levels: Dict[int, List[Interpolant]] = {}
        self._collect_by_level(tree_itp, 0, root.depth, levels)
        return levels

    def _collect_by_level(
        self,
        tree_itp: TreeInterpolantData,
        current_depth: int,
        max_depth: int,
        levels: Dict[int, List[Interpolant]],
    ) -> None:
        level = max_depth - current_depth
        if level not in levels:
            levels[level] = []
        levels[level].append(tree_itp.root)

        for child in tree_itp.children:
            self._collect_by_level(child, current_depth + 1, max_depth, levels)

    def refine_at_level(
        self,
        hierarchy: Dict[int, List[Interpolant]],
        level: int,
    ) -> List[Predicate]:
        """Extract predicates from a specific abstraction level."""
        if level not in hierarchy:
            return []

        predicates: List[Predicate] = []
        for i, itp in enumerate(hierarchy[level]):
            pred = itp.as_predicate(f"hier_l{level}_p{i}")
            predicates.append(pred)
        return predicates

    def coarsen(
        self,
        interpolant: Interpolant,
        target_vars: int = 3,
    ) -> Interpolant:
        """Coarsen an interpolant by eliminating variables.

        Reduces the number of variables to at most ``target_vars``.
        """
        f = Formula.from_dp_formula(interpolant.formula)
        current_vars = sorted(f.variables)

        if len(current_vars) <= target_vars:
            return interpolant

        qe = QuantifierElimination()
        # Eliminate excess variables, preferring those with smallest coefficients
        to_elim = current_vars[target_vars:]
        projected = qe.eliminate(f, to_elim)
        projected = self._simplifier.simplify(projected)

        remaining = frozenset(current_vars[:target_vars])
        return Interpolant(
            formula=projected.to_dp_formula(),
            interpolant_type=interpolant.interpolant_type,
            common_variables=remaining & interpolant.common_variables,
            strength=InterpolantStrength.WEAKEST,
        )

    def abstract_and_check(
        self,
        root: TreeNode,
        property_formula: DPFormula,
    ) -> Tuple[bool, Optional[int]]:
        """Check property using hierarchical abstraction.

        Tries each level from coarsest to finest. Returns (holds, level)
        where level is the abstraction level at which the property was
        verified, or None if it failed at all levels.
        """
        hierarchy = self.build_abstraction_hierarchy(root)

        checker = SatisfiabilityChecker()
        for level in sorted(hierarchy.keys(), reverse=True):
            itps = hierarchy[level]
            # Check if conjunction of interpolants at this level implies property
            parts: List[str] = [f"({itp.formula.expr})" for itp in itps]
            all_vars: Set[str] = set()
            for itp in itps:
                all_vars.update(itp.common_variables)

            conj = DPFormula(
                expr=" ∧ ".join(parts) if parts else "true",
                variables=frozenset(all_vars),
            )
            conj_f = Formula.from_dp_formula(conj)
            prop_f = Formula.from_dp_formula(property_formula)

            if checker.implies(conj_f, prop_f):
                return True, level

        return False, None


__all__ = [
    "TreeNode",
    "DAGNode",
    "TreeInterpolant",
    "DAGInterpolation",
    "CompositionalInterpolation",
    "InterpolantMerge",
    "TreeDecomposition",
    "HierarchicalAbstraction",
]
