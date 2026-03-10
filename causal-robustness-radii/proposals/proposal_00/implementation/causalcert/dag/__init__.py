"""
DAG sub-package — graph representation, d-separation, and structural operations.

Provides :class:`CausalDAG`, the central graph object, together with algorithms
for d-separation (including incremental updates), ancestral set extraction,
moral graphs, Markov equivalence classes, and edit-distance utilities.
"""

from causalcert.dag.graph import CausalDAG
from causalcert.dag.dsep import DSeparationOracle
from causalcert.dag.incremental import IncrementalDSep
from causalcert.dag.ancestors import ancestors, descendants, topological_order
from causalcert.dag.moral import moral_graph, treewidth, tree_decomposition
from causalcert.dag.mec import to_cpdag, mec_size_bound
from causalcert.dag.edit import edit_distance, apply_edit, k_neighbourhood
from causalcert.dag.validation import is_dag, find_cycle
from causalcert.dag.conversions import to_dot, from_dot, to_json, from_json
from causalcert.dag.paths import (
    all_directed_paths,
    backdoor_paths,
    causal_paths,
    has_directed_path,
    reachability_matrix,
    shortest_directed_path,
)
from causalcert.dag.equivalence import (
    enumerate_mec,
    is_markov_equivalent,
    mec_size,
    reversible_edges,
    sample_from_mec,
    skeleton,
    v_structures,
)
from causalcert.dag.causal_order import (
    causal_order,
    causal_layers,
    is_valid_causal_order,
    topological_sort_kahn,
)

__all__ = [
    "CausalDAG",
    "DSeparationOracle",
    "IncrementalDSep",
    "ancestors",
    "descendants",
    "topological_order",
    "moral_graph",
    "treewidth",
    "tree_decomposition",
    "to_cpdag",
    "mec_size_bound",
    "edit_distance",
    "apply_edit",
    "k_neighbourhood",
    "is_dag",
    "find_cycle",
    "to_dot",
    "from_dot",
    "to_json",
    "from_json",
    # paths
    "all_directed_paths",
    "backdoor_paths",
    "causal_paths",
    "has_directed_path",
    "reachability_matrix",
    "shortest_directed_path",
    # equivalence
    "enumerate_mec",
    "is_markov_equivalent",
    "mec_size",
    "reversible_edges",
    "sample_from_mec",
    "skeleton",
    "v_structures",
    # causal_order
    "causal_order",
    "causal_layers",
    "is_valid_causal_order",
    "topological_sort_kahn",
]
