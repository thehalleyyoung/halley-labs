"""
Treewidth sub-package — tree decomposition and treewidth computation.

Provides data structures and algorithm interfaces for computing tree
decompositions, elimination orderings, and treewidth bounds.  These are
used by the FPT dynamic-programming solver (ALG 7) to exploit bounded
treewidth in the moral graph of the DAG.
"""

from causalcert.treewidth.types import (
    EliminationOrdering,
    TreeBag,
    TreeDecomposition,
)
from causalcert.treewidth.protocols import (
    BagProcessor,
    DecompositionAlgorithm,
)
from causalcert.treewidth.elimination import (
    best_heuristic_ordering,
    compute_treewidth_lower_bound,
    compute_treewidth_upper_bound,
    min_degree_ordering,
    min_fill_ordering,
    min_width_ordering,
    max_cardinality_search,
    ordering_to_decomposition,
    triangulate,
    detect_perfect_elimination_ordering,
    degeneracy_lower_bound,
)
from causalcert.treewidth.decomposition import (
    compute_tree_decomposition,
    compute_tree_decomposition_from_adj,
    compute_treewidth_bounds,
    compute_treewidth_bounds_from_adj,
    validate_tree_decomposition,
    decomposition_from_chordal_graph,
    reroot_decomposition,
    simplify_decomposition,
    width_of_decomposition,
)
from causalcert.treewidth.nice import (
    NiceNodeType,
    NiceTreeDecomposition,
    NiceTreeNode,
    to_nice_decomposition,
    validate_nice_decomposition,
)
from causalcert.treewidth.bags import (
    NO_EDGE,
    FORWARD,
    BACKWARD,
    BagState,
    StateTable,
    enumerate_bag_states,
    is_acyclic_in_bag,
)
from causalcert.treewidth.dp import (
    CIConstraint,
    TreewidthDP,
    compute_min_edit_distance,
    compute_min_edit_distance_with_witness,
)
from causalcert.treewidth.separator import (
    Separator,
    CliqueTree,
    Atom,
    enumerate_minimal_separators,
    is_minimal_separator,
    is_safe_separator,
    find_safe_separators,
    build_clique_tree,
    atom_decomposition,
)

__all__ = [
    # types
    "TreeBag",
    "TreeDecomposition",
    "EliminationOrdering",
    # protocols
    "DecompositionAlgorithm",
    "BagProcessor",
    # elimination
    "best_heuristic_ordering",
    "compute_treewidth_lower_bound",
    "compute_treewidth_upper_bound",
    "min_degree_ordering",
    "min_fill_ordering",
    "min_width_ordering",
    "max_cardinality_search",
    "ordering_to_decomposition",
    "triangulate",
    "detect_perfect_elimination_ordering",
    "degeneracy_lower_bound",
    # decomposition
    "compute_tree_decomposition",
    "compute_tree_decomposition_from_adj",
    "compute_treewidth_bounds",
    "compute_treewidth_bounds_from_adj",
    "validate_tree_decomposition",
    "decomposition_from_chordal_graph",
    "reroot_decomposition",
    "simplify_decomposition",
    "width_of_decomposition",
    # nice
    "NiceNodeType",
    "NiceTreeDecomposition",
    "NiceTreeNode",
    "to_nice_decomposition",
    "validate_nice_decomposition",
    # bags
    "NO_EDGE",
    "FORWARD",
    "BACKWARD",
    "BagState",
    "StateTable",
    "enumerate_bag_states",
    "is_acyclic_in_bag",
    # dp
    "CIConstraint",
    "TreewidthDP",
    "compute_min_edit_distance",
    "compute_min_edit_distance_with_witness",
    # separator
    "Separator",
    "CliqueTree",
    "Atom",
    "enumerate_minimal_separators",
    "is_minimal_separator",
    "is_safe_separator",
    "find_safe_separators",
    "build_clique_tree",
    "atom_decomposition",
]
