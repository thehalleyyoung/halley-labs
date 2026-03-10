"""
Nice tree decomposition conversion and utilities.

A *nice* tree decomposition is a rooted binary tree where every node is
one of four types:

* **Leaf** — a single-vertex bag with no children.
* **Introduce** — adds exactly one vertex relative to its single child.
* **Forget** — removes exactly one vertex relative to its single child.
* **Join** — has two children with identical bags.

Nice decompositions simplify the formulation of dynamic-programming
algorithms on tree decompositions.
"""

from __future__ import annotations

import enum
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import networkx as nx

from causalcert.types import NodeId, NodeSet
from causalcert.treewidth.types import TreeBag, TreeDecomposition


# ---------------------------------------------------------------------------
# Node type enumeration
# ---------------------------------------------------------------------------


class NiceNodeType(enum.Enum):
    """Type of a node in a nice tree decomposition."""

    LEAF = "leaf"
    """Leaf bag containing a single vertex and no children."""

    INTRODUCE = "introduce"
    """Introduce bag: one child, adds one vertex."""

    FORGET = "forget"
    """Forget bag: one child, removes one vertex."""

    JOIN = "join"
    """Join bag: two children with identical bags."""


# ---------------------------------------------------------------------------
# Nice tree node
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NiceTreeNode:
    """A node in a nice tree decomposition.

    Attributes
    ----------
    node_id : int
        Unique identifier within the nice decomposition.
    bag : NodeSet
        Vertices in this bag.
    node_type : NiceNodeType
        The type of this node.
    special_vertex : NodeId | None
        The vertex being introduced or forgotten; ``None`` for leaf and join.
    children_ids : tuple[int, ...]
        IDs of child nodes (0 for leaf, 1 for introduce/forget, 2 for join).
    """

    node_id: int
    bag: NodeSet
    node_type: NiceNodeType
    special_vertex: NodeId | None = None
    children_ids: tuple[int, ...] = ()

    @property
    def is_leaf(self) -> bool:
        """Return True if this is a leaf node."""
        return self.node_type == NiceNodeType.LEAF

    @property
    def is_introduce(self) -> bool:
        """Return True if this is an introduce node."""
        return self.node_type == NiceNodeType.INTRODUCE

    @property
    def is_forget(self) -> bool:
        """Return True if this is a forget node."""
        return self.node_type == NiceNodeType.FORGET

    @property
    def is_join(self) -> bool:
        """Return True if this is a join node."""
        return self.node_type == NiceNodeType.JOIN

    @property
    def width(self) -> int:
        """Width contribution (``|bag| - 1``)."""
        return max(0, len(self.bag) - 1)


# ---------------------------------------------------------------------------
# Nice tree decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NiceTreeDecomposition:
    """A nice tree decomposition.

    Attributes
    ----------
    nodes : tuple[NiceTreeNode, ...]
        All nodes in the nice decomposition, indexed by ``node_id``.
    root_id : int
        ID of the root node.
    width : int
        Width of the decomposition.
    n_vertices : int
        Number of vertices in the original graph.
    """

    nodes: tuple[NiceTreeNode, ...]
    root_id: int
    width: int
    n_vertices: int = 0

    def node(self, node_id: int) -> NiceTreeNode:
        """Retrieve a node by ID.

        Parameters
        ----------
        node_id : int
            Node identifier.

        Returns
        -------
        NiceTreeNode

        Raises
        ------
        KeyError
            If no node with that ID exists.
        """
        for nd in self.nodes:
            if nd.node_id == node_id:
                return nd
        raise KeyError(f"No node with id {node_id}.")

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the nice decomposition."""
        return len(self.nodes)

    def leaves(self) -> list[NiceTreeNode]:
        """Return all leaf nodes."""
        return [n for n in self.nodes if n.is_leaf]

    def postorder(self) -> list[NiceTreeNode]:
        """Return nodes in post-order (children before parents).

        Returns
        -------
        list[NiceTreeNode]
            Nodes in bottom-up traversal order.
        """
        node_map = {n.node_id: n for n in self.nodes}
        visited: set[int] = set()
        result: list[NiceTreeNode] = []

        def _dfs(nid: int) -> None:
            if nid in visited:
                return
            visited.add(nid)
            nd = node_map[nid]
            for cid in nd.children_ids:
                _dfs(cid)
            result.append(nd)

        _dfs(self.root_id)
        return result


# ---------------------------------------------------------------------------
# Conversion: arbitrary TD → nice TD
# ---------------------------------------------------------------------------


def to_nice_decomposition(
    td: TreeDecomposition,
) -> NiceTreeDecomposition:
    """Convert an arbitrary tree decomposition to a nice tree decomposition.

    The conversion proceeds as follows:

    1. Root the tree at ``td.root_id``.
    2. At each node, if it has more than two children, introduce binary
       join nodes.
    3. Between a parent and each child, insert introduce/forget chains so
       that each step adds or removes exactly one vertex.
    4. At leaves, insert a chain down to a single-vertex bag.
    5. At the root, add forget nodes until the root bag is empty.

    Parameters
    ----------
    td : TreeDecomposition
        Arbitrary tree decomposition.

    Returns
    -------
    NiceTreeDecomposition
        An equivalent nice tree decomposition.
    """
    if not td.bags:
        return NiceTreeDecomposition(nodes=(), root_id=0, width=0, n_vertices=0)

    bag_map: dict[int, TreeBag] = {b.bag_id: b for b in td.bags}
    nodes: list[NiceTreeNode] = []
    counter = [0]

    def _new_id() -> int:
        nid = counter[0]
        counter[0] += 1
        return nid

    def _build_leaf_chain(bag_verts: frozenset[NodeId]) -> int:
        """Build a chain: leaf(single vertex) → introduce → … → full bag."""
        verts = sorted(bag_verts)
        if not verts:
            nid = _new_id()
            nodes.append(NiceTreeNode(
                node_id=nid, bag=frozenset(), node_type=NiceNodeType.LEAF,
            ))
            return nid

        # Start with a leaf containing the first vertex
        leaf_id = _new_id()
        nodes.append(NiceTreeNode(
            node_id=leaf_id,
            bag=frozenset({verts[0]}),
            node_type=NiceNodeType.LEAF,
        ))

        prev_id = leaf_id
        current_bag = frozenset({verts[0]})

        # Introduce remaining vertices one by one
        for v in verts[1:]:
            new_bag = current_bag | frozenset({v})
            nid = _new_id()
            nodes.append(NiceTreeNode(
                node_id=nid,
                bag=new_bag,
                node_type=NiceNodeType.INTRODUCE,
                special_vertex=v,
                children_ids=(prev_id,),
            ))
            prev_id = nid
            current_bag = new_bag

        return prev_id

    def _introduce_chain(
        target_bag: frozenset[NodeId],
        child_bag: frozenset[NodeId],
        child_id: int,
    ) -> int:
        """Insert introduce nodes for vertices in target_bag − child_bag."""
        to_introduce = sorted(target_bag - child_bag)
        prev_id = child_id
        current_bag = child_bag

        for v in to_introduce:
            new_bag = current_bag | frozenset({v})
            nid = _new_id()
            nodes.append(NiceTreeNode(
                node_id=nid,
                bag=new_bag,
                node_type=NiceNodeType.INTRODUCE,
                special_vertex=v,
                children_ids=(prev_id,),
            ))
            prev_id = nid
            current_bag = new_bag

        return prev_id

    def _forget_chain(
        target_bag: frozenset[NodeId],
        child_bag: frozenset[NodeId],
        child_id: int,
    ) -> int:
        """Insert forget nodes for vertices in child_bag − target_bag."""
        to_forget = sorted(child_bag - target_bag)
        prev_id = child_id
        current_bag = child_bag

        for v in to_forget:
            new_bag = current_bag - frozenset({v})
            nid = _new_id()
            nodes.append(NiceTreeNode(
                node_id=nid,
                bag=new_bag,
                node_type=NiceNodeType.FORGET,
                special_vertex=v,
                children_ids=(prev_id,),
            ))
            prev_id = nid
            current_bag = new_bag

        return prev_id

    def _transition_chain(
        target_bag: frozenset[NodeId],
        child_bag: frozenset[NodeId],
        child_id: int,
    ) -> int:
        """Build introduce then forget chain from child_bag to target_bag."""
        # First introduce all missing vertices, then forget extras
        union_bag = target_bag | child_bag
        mid_id = _introduce_chain(union_bag, child_bag, child_id)
        return _forget_chain(target_bag, union_bag, mid_id)

    def _process_node(bag_id: int) -> int:
        """Process a bag and its sub-tree, returning the nice-node ID at top."""
        bag = bag_map[bag_id]
        children = list(bag.children_ids)

        if not children:
            # Leaf — build an introduce chain
            return _build_leaf_chain(bag.vertices)

        # Recursively process children
        child_nice_ids: list[int] = []
        for cid in children:
            c_top = _process_node(cid)
            c_bag = bag_map[cid].vertices
            # Transition from child bag to this bag
            nice_id = _transition_chain(bag.vertices, c_bag, c_top)
            child_nice_ids.append(nice_id)

        if len(child_nice_ids) == 1:
            return child_nice_ids[0]

        # Multiple children: create binary join tree
        # All children now have the same bag (= bag.vertices)
        while len(child_nice_ids) > 1:
            left = child_nice_ids.pop(0)
            right = child_nice_ids.pop(0)
            nid = _new_id()
            nodes.append(NiceTreeNode(
                node_id=nid,
                bag=bag.vertices,
                node_type=NiceNodeType.JOIN,
                children_ids=(left, right),
            ))
            child_nice_ids.append(nid)

        return child_nice_ids[0]

    # Process the tree from the root
    root_nice_id = _process_node(td.root_id)

    # Add forget chain at the root to empty bag
    root_bag = bag_map[td.root_id].vertices
    if root_bag:
        final_id = _forget_chain(frozenset(), root_bag, root_nice_id)
    else:
        final_id = root_nice_id

    width = 0
    for nd in nodes:
        w = len(nd.bag) - 1 if nd.bag else 0
        width = max(width, w)

    return NiceTreeDecomposition(
        nodes=tuple(nodes),
        root_id=final_id,
        width=width,
        n_vertices=td.n_vertices,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_nice_decomposition(
    ntd: NiceTreeDecomposition,
) -> tuple[bool, list[str]]:
    """Validate that a nice tree decomposition satisfies all nice-TD properties.

    Checks:
    - Every leaf has an empty or single-vertex bag and no children.
    - Every introduce node has exactly one child, with the child's bag being
      the parent's bag minus the special vertex.
    - Every forget node has exactly one child, with the child's bag being
      the parent's bag plus the special vertex.
    - Every join node has exactly two children with identical bags equal to
      the parent's bag.

    Parameters
    ----------
    ntd : NiceTreeDecomposition
        Nice tree decomposition to validate.

    Returns
    -------
    tuple[bool, list[str]]
        ``(valid, errors)``.
    """
    errors: list[str] = []
    node_map = {n.node_id: n for n in ntd.nodes}

    for nd in ntd.nodes:
        if nd.node_type == NiceNodeType.LEAF:
            if nd.children_ids:
                errors.append(f"Leaf node {nd.node_id} has children.")
            if len(nd.bag) > 1:
                errors.append(
                    f"Leaf node {nd.node_id} has {len(nd.bag)} vertices "
                    f"(expected 0 or 1)."
                )

        elif nd.node_type == NiceNodeType.INTRODUCE:
            if len(nd.children_ids) != 1:
                errors.append(
                    f"Introduce node {nd.node_id} has "
                    f"{len(nd.children_ids)} children (expected 1)."
                )
            elif nd.children_ids[0] in node_map:
                child = node_map[nd.children_ids[0]]
                expected = nd.bag - frozenset({nd.special_vertex})
                if child.bag != expected:
                    errors.append(
                        f"Introduce node {nd.node_id}: child bag "
                        f"{child.bag} != expected {expected}."
                    )
                if nd.special_vertex not in nd.bag:
                    errors.append(
                        f"Introduce node {nd.node_id}: special vertex "
                        f"{nd.special_vertex} not in bag."
                    )

        elif nd.node_type == NiceNodeType.FORGET:
            if len(nd.children_ids) != 1:
                errors.append(
                    f"Forget node {nd.node_id} has "
                    f"{len(nd.children_ids)} children (expected 1)."
                )
            elif nd.children_ids[0] in node_map:
                child = node_map[nd.children_ids[0]]
                expected = nd.bag | frozenset({nd.special_vertex})
                if child.bag != expected:
                    errors.append(
                        f"Forget node {nd.node_id}: child bag "
                        f"{child.bag} != expected {expected}."
                    )
                if nd.special_vertex in nd.bag:
                    errors.append(
                        f"Forget node {nd.node_id}: special vertex "
                        f"{nd.special_vertex} still in bag."
                    )

        elif nd.node_type == NiceNodeType.JOIN:
            if len(nd.children_ids) != 2:
                errors.append(
                    f"Join node {nd.node_id} has "
                    f"{len(nd.children_ids)} children (expected 2)."
                )
            else:
                left = node_map.get(nd.children_ids[0])
                right = node_map.get(nd.children_ids[1])
                if left and left.bag != nd.bag:
                    errors.append(
                        f"Join node {nd.node_id}: left child bag "
                        f"{left.bag} != {nd.bag}."
                    )
                if right and right.bag != nd.bag:
                    errors.append(
                        f"Join node {nd.node_id}: right child bag "
                        f"{right.bag} != {nd.bag}."
                    )

    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Traversal helpers
# ---------------------------------------------------------------------------


def postorder_traversal(ntd: NiceTreeDecomposition) -> list[int]:
    """Return node IDs in post-order (children before parents).

    Parameters
    ----------
    ntd : NiceTreeDecomposition
        Nice tree decomposition.

    Returns
    -------
    list[int]
        Node IDs in post-order.
    """
    return [nd.node_id for nd in ntd.postorder()]


def count_by_type(ntd: NiceTreeDecomposition) -> dict[NiceNodeType, int]:
    """Count nodes of each type in the nice decomposition.

    Parameters
    ----------
    ntd : NiceTreeDecomposition
        Nice tree decomposition.

    Returns
    -------
    dict[NiceNodeType, int]
        Counts keyed by node type.
    """
    counts: dict[NiceNodeType, int] = {t: 0 for t in NiceNodeType}
    for nd in ntd.nodes:
        counts[nd.node_type] += 1
    return counts


def nice_decomposition_summary(ntd: NiceTreeDecomposition) -> str:
    """Return a human-readable summary of the nice decomposition.

    Parameters
    ----------
    ntd : NiceTreeDecomposition
        Nice tree decomposition.

    Returns
    -------
    str
        Multi-line summary string.
    """
    counts = count_by_type(ntd)
    lines = [
        f"Nice tree decomposition: {ntd.n_nodes} nodes, width {ntd.width}",
        f"  Vertices: {ntd.n_vertices}",
        f"  Leaves:     {counts[NiceNodeType.LEAF]}",
        f"  Introduce:  {counts[NiceNodeType.INTRODUCE]}",
        f"  Forget:     {counts[NiceNodeType.FORGET]}",
        f"  Join:       {counts[NiceNodeType.JOIN]}",
    ]
    return "\n".join(lines)


__all__ = [
    "NiceNodeType",
    "NiceTreeNode",
    "NiceTreeDecomposition",
    "to_nice_decomposition",
    "validate_nice_decomposition",
    "postorder_traversal",
    "count_by_type",
    "nice_decomposition_summary",
]
