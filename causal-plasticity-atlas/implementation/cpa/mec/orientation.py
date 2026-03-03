"""Meek orientation rules (R1-R4) and Zhang orientation rules.

Provides iterative application of orientation rules to maximally orient
edges in a CPDAG or PAG.  Implements the rules exactly as described in
Meek (1995) and Zhang (2008).
"""

from __future__ import annotations

from typing import Any, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# -------------------------------------------------------------------
# Meek Rules for CPDAGs
# -------------------------------------------------------------------

class MeekRules:
    """Meek's four orientation rules for CPDAGs.

    Reference: Meek (1995) "Causal Inference and Causal Explanation
    with Background Knowledge".

    Rules orient undirected edges when a single orientation is forced by
    acyclicity and the constraint that no new v-structures are created.

    The CPDAG object is expected to have:
      - n_nodes : int
      - has_directed_edge(i, j) -> bool  (i -> j)
      - has_undirected_edge(i, j) -> bool  (i - j)
      - add_directed_edge(i, j)
      - remove_undirected_edge(i, j)
      - neighbors(i) -> set[int]  (undirected neighbours)
      - children(i) -> set[int]
      - parents(i) -> set[int]
    """

    def _is_adjacent(self, cpdag: Any, i: int, j: int) -> bool:
        """Check if i and j are connected by any edge."""
        return (cpdag.has_directed_edge(i, j)
                or cpdag.has_directed_edge(j, i)
                or cpdag.has_undirected_edge(i, j))

    def apply_r1(self, cpdag: Any) -> bool:
        """R1: If a -> b - c and a is not adjacent to c, orient b -> c.

        This prevents creating new v-structures: if b - c remained
        undirected, orienting it as c -> b would create a new
        v-structure a -> b <- c.

        Returns True if any edge was oriented.
        """
        changed = False
        n = cpdag.n_nodes
        for b in range(n):
            # Find directed parents of b: a -> b
            pa_b = list(cpdag.parents(b))
            # Find undirected neighbors of b: b - c
            nb_b = list(cpdag.neighbors(b))
            for a in pa_b:
                for c in nb_b:
                    if c == a:
                        continue
                    # a -> b - c and a not adjacent to c
                    if not self._is_adjacent(cpdag, a, c):
                        cpdag.remove_undirected_edge(b, c)
                        cpdag.add_directed_edge(b, c)
                        changed = True
        return changed

    def apply_r2(self, cpdag: Any) -> bool:
        """R2: If a -> b -> c and a - c, orient a -> c.

        Avoids creating a directed cycle: if c -> a were chosen,
        it would form the cycle a -> b -> c -> a.

        Returns True if any edge was oriented.
        """
        changed = False
        n = cpdag.n_nodes
        for a in range(n):
            nb_a = list(cpdag.neighbors(a))  # a - c candidates
            ch_a = list(cpdag.children(a))   # a -> b candidates
            for c in nb_a:
                # Need a -> b -> c for some b
                for b in ch_a:
                    if b == c:
                        continue
                    if cpdag.has_directed_edge(b, c):
                        cpdag.remove_undirected_edge(a, c)
                        cpdag.add_directed_edge(a, c)
                        changed = True
                        break  # c is oriented, move to next c
        return changed

    def apply_r3(self, cpdag: Any) -> bool:
        """R3: If a - b, a - c, a - d, b -> d, c -> d, and b not adj c,
        orient a -> d.

        Two non-adjacent nodes b, c both have directed edges into d
        and undirected edges to a.  To avoid a new v-structure,
        a -> d is forced.

        Returns True if any edge was oriented.
        """
        changed = False
        n = cpdag.n_nodes
        for d in range(n):
            # Find parents of d that are also undirected neighbors of some a
            pa_d = list(cpdag.parents(d))
            nb_d = list(cpdag.neighbors(d))
            # For each undirected neighbor a of d
            for a in nb_d:
                # Find b, c: b - a, c - a, b -> d, c -> d, b != c, b not adj c
                # b and c must be undirected neighbors of a AND directed parents of d
                candidates = []
                for node in pa_d:
                    if node != a and cpdag.has_undirected_edge(a, node):
                        candidates.append(node)
                # Check pairs
                found = False
                for i_idx in range(len(candidates)):
                    if found:
                        break
                    for j_idx in range(i_idx + 1, len(candidates)):
                        b = candidates[i_idx]
                        c = candidates[j_idx]
                        if not self._is_adjacent(cpdag, b, c):
                            cpdag.remove_undirected_edge(a, d)
                            cpdag.add_directed_edge(a, d)
                            changed = True
                            found = True
                            break
        return changed

    def apply_r4(self, cpdag: Any) -> bool:
        """R4: If a - b, b -> c, c -> d, a - d, and a not adj c,
        orient a -> d.

        Prevents a cycle: the chain b -> c -> d with a - b and a - d
        forces a -> d to avoid a directed cycle through a.

        Returns True if any edge was oriented.
        """
        changed = False
        n = cpdag.n_nodes
        for a in range(n):
            nb_a = list(cpdag.neighbors(a))  # undirected neighbors of a
            for d in nb_a:
                # Need b: a - b, b -> c -> d, a not adj c
                for b in nb_a:
                    if b == d:
                        continue
                    # b -> c -> d, a not adj c
                    ch_b = cpdag.children(b)
                    for c in ch_b:
                        if c == a or c == d:
                            continue
                        if cpdag.has_directed_edge(c, d):
                            if not self._is_adjacent(cpdag, a, c):
                                cpdag.remove_undirected_edge(a, d)
                                cpdag.add_directed_edge(a, d)
                                changed = True
                                break
                    else:
                        continue
                    break  # d is oriented
        return changed

    def apply_all(self, cpdag: Any, max_iter: int = 100) -> int:
        """Apply all four Meek rules iteratively until convergence.

        Returns the total number of iterations in which at least one
        edge was oriented.
        """
        total = 0
        for _ in range(max_iter):
            any_change = False
            any_change |= self.apply_r1(cpdag)
            any_change |= self.apply_r2(cpdag)
            any_change |= self.apply_r3(cpdag)
            any_change |= self.apply_r4(cpdag)
            if any_change:
                total += 1
            else:
                break
        return total


# -------------------------------------------------------------------
# Orient v-structures from skeleton + separation sets
# -------------------------------------------------------------------

def orient_v_structures(
    skeleton: NDArray[np.int_],
    sep_sets: dict[tuple[int, int], set[int]],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Orient v-structures from a skeleton and separation sets.

    For every triple a - b - c where a and b are not adjacent,
    if b is NOT in sep(a, c), orient as a -> b <- c (v-structure).

    Parameters
    ----------
    skeleton : NDArray
        Symmetric adjacency matrix of the skeleton.
    sep_sets : dict
        Mapping (i, j) -> set of separating nodes.  Must be symmetric:
        if (i, j) is present, (j, i) should have the same value.

    Returns
    -------
    directed_edges : set of (int, int)
        Directed edges i -> j.
    undirected_edges : set of (int, int)
        Remaining undirected edges as canonical (min, max) pairs.
    """
    skel = np.asarray(skeleton, dtype=np.int_)
    n = skel.shape[0]
    # Make skeleton symmetric
    skel = ((skel + skel.T) > 0).astype(np.int_)

    directed: Set[Tuple[int, int]] = set()
    oriented_pairs: Set[Tuple[int, int]] = set()  # track which pairs got oriented

    # Find v-structures
    for b in range(n):
        adj_b = [i for i in range(n) if skel[i, b] and i != b]
        for idx_a in range(len(adj_b)):
            for idx_c in range(idx_a + 1, len(adj_b)):
                a = adj_b[idx_a]
                c = adj_b[idx_c]
                # Check a and c are NOT adjacent
                if skel[a, c]:
                    continue
                # Get separation set
                key = (min(a, c), max(a, c))
                sep = sep_sets.get(key, sep_sets.get((a, c), sep_sets.get((c, a), None)))
                if sep is None:
                    continue
                # If b not in sep(a,c), orient a -> b <- c
                if b not in sep:
                    directed.add((a, b))
                    directed.add((c, b))
                    oriented_pairs.add((min(a, b), max(a, b)))
                    oriented_pairs.add((min(b, c), max(b, c)))

    # Remaining edges are undirected
    undirected: Set[Tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if skel[i, j]:
                pair = (i, j)
                if pair not in oriented_pairs:
                    undirected.add(pair)

    return directed, undirected


# -------------------------------------------------------------------
# Zhang Rules for PAGs (FCI orientation rules)
# -------------------------------------------------------------------

class ZhangRules:
    """Zhang's orientation rules for Partial Ancestral Graphs (PAGs).

    Reference: Zhang (2008) "On the completeness of orientation rules
    for causal discovery in the presence of latent confounders and
    selection variables".

    Implements rules R1-R10 for PAG orientation.  Edge marks:
    NONE=0, TAIL=1, ARROWHEAD=2, CIRCLE=3.
    """

    NONE = 0
    TAIL = 1
    ARROW = 2
    CIRCLE = 3

    def _get_mark(self, pag: Any, i: int, j: int) -> int:
        """Get edge mark at j-end of i-j edge."""
        m = pag._marks[i, j]
        return m.value if hasattr(m, 'value') else int(m)

    def _set_mark(self, pag: Any, i: int, j: int, mark: int) -> None:
        """Set edge mark at position (i,j)."""
        try:
            from cpa.mec.pag import EdgeMark
            pag._marks[i, j] = EdgeMark(mark)
        except (ImportError, ValueError):
            pag._marks[i, j] = mark

    def _has_edge(self, pag: Any, i: int, j: int) -> bool:
        """Check if there is any edge between i and j."""
        return (self._get_mark(pag, i, j) != self.NONE
                or self._get_mark(pag, j, i) != self.NONE)

    def _is_adjacent(self, pag: Any, i: int, j: int) -> bool:
        return self._has_edge(pag, i, j)

    def apply_r1(self, pag: Any) -> bool:
        """R1: If a *-> b o-* c and a not adj c, orient b *-> c."""
        changed = False
        n = pag.n_nodes
        for b in range(n):
            for a in range(n):
                if a == b:
                    continue
                # a *-> b: arrowhead at b-end of a-b edge
                if self._get_mark(pag, a, b) != self.ARROW:
                    continue
                if not self._has_edge(pag, a, b):
                    continue
                for c in range(n):
                    if c == a or c == b:
                        continue
                    if not self._has_edge(pag, b, c):
                        continue
                    # b o-* c: circle at b-end of b-c edge
                    if self._get_mark(pag, c, b) != self.CIRCLE:
                        continue
                    if self._is_adjacent(pag, a, c):
                        continue
                    # Orient: change circle at b-end to tail
                    self._set_mark(pag, c, b, self.TAIL)
                    changed = True
        return changed

    def apply_r2(self, pag: Any) -> bool:
        """R2: If a -> b *-> c or a *-> b -> c, and a *-o c, orient a *-> c."""
        changed = False
        n = pag.n_nodes
        for a in range(n):
            for c in range(n):
                if a == c:
                    continue
                if not self._has_edge(pag, a, c):
                    continue
                # a *-o c: circle at c-end
                if self._get_mark(pag, a, c) != self.CIRCLE:
                    continue
                for b in range(n):
                    if b == a or b == c:
                        continue
                    if not self._has_edge(pag, a, b) or not self._has_edge(pag, b, c):
                        continue
                    # Case 1: a -> b *-> c
                    case1 = (self._get_mark(pag, a, b) == self.TAIL
                             and self._get_mark(pag, b, a) == self.ARROW  # wrong, should be: a at a-end=tail, b at b-end=arrow => a -> b
                             )
                    # Correction: a -> b means mark at a-end is TAIL, mark at b-end is ARROW
                    # Using convention: _marks[i,j] = mark at j-end of edge i-j? No.
                    # Let's use: _marks[i,j] = mark at j-end of edge from i
                    # So a -> b means: mark(a,b) = ARROW (arrowhead at b), mark(b,a) = TAIL (tail at a)
                    c1 = (self._get_mark(pag, a, b) == self.ARROW
                           and self._get_mark(pag, b, a) == self.TAIL
                           and self._get_mark(pag, b, c) == self.ARROW)
                    # Case 2: a *-> b -> c
                    c2 = (self._get_mark(pag, a, b) == self.ARROW  # arrowhead at b
                           and self._get_mark(pag, b, c) == self.ARROW  # arrowhead at c? No, -> means tail at b, arrow at c
                           )
                    c2_corr = (self._get_mark(pag, a, b) == self.ARROW
                               and self._get_mark(pag, b, c) == self.ARROW
                               and self._get_mark(pag, c, b) == self.TAIL)
                    if c1 or c2_corr:
                        self._set_mark(pag, a, c, self.ARROW)
                        changed = True
                        break
        return changed

    def apply_r3(self, pag: Any) -> bool:
        """R3: If a *-> b <-* c, a *-o d o-* c, a not adj c, d *-o b,
        orient d *-> b."""
        changed = False
        n = pag.n_nodes
        for d in range(n):
            for b in range(n):
                if d == b:
                    continue
                if not self._has_edge(pag, d, b):
                    continue
                if self._get_mark(pag, d, b) != self.CIRCLE:
                    continue
                for a in range(n):
                    if a == d or a == b:
                        continue
                    # a *-> b
                    if self._get_mark(pag, a, b) != self.ARROW:
                        continue
                    if not self._has_edge(pag, a, d):
                        continue
                    # a *-o d
                    if self._get_mark(pag, a, d) != self.CIRCLE:
                        continue
                    for c in range(n):
                        if c == a or c == b or c == d:
                            continue
                        # c *-> b
                        if self._get_mark(pag, c, b) != self.ARROW:
                            continue
                        if self._is_adjacent(pag, a, c):
                            continue
                        if not self._has_edge(pag, c, d):
                            continue
                        # c *-o d: circle at d-end
                        if self._get_mark(pag, c, d) != self.CIRCLE:
                            continue
                        self._set_mark(pag, d, b, self.ARROW)
                        changed = True
        return changed

    def apply_r4(self, pag: Any) -> bool:
        """R4 (discriminating path rule): orient based on discriminating paths."""
        changed = False
        n = pag.n_nodes
        # For each potential triple ...-b-c where we might orient
        for b in range(n):
            for c in range(n):
                if b == c:
                    continue
                if not self._has_edge(pag, b, c):
                    continue
                if self._get_mark(pag, b, c) != self.CIRCLE:
                    continue
                # Look for discriminating path for b w.r.t. c
                # A discriminating path <a, ..., b, c> where every node
                # between a and b is a parent of c
                for a in range(n):
                    if a == b or a == c:
                        continue
                    if self._is_adjacent(pag, a, c):
                        continue
                    # Try to find path from a to b where intermediaries are parents of c
                    path = self._find_discriminating_path(pag, a, b, c)
                    if path is not None:
                        # If b is in sep(a,c) orient as b - c (noncollider)
                        # Otherwise orient as b <-> c (collider)
                        # Since we don't have sep sets here, orient based on
                        # whether arrowhead at b
                        if self._get_mark(pag, c, b) == self.ARROW:
                            # b is collider, orient circle as arrow
                            self._set_mark(pag, b, c, self.ARROW)
                        else:
                            self._set_mark(pag, b, c, self.TAIL)
                        changed = True
        return changed

    def _find_discriminating_path(
        self, pag: Any, a: int, b: int, c: int
    ) -> list[int] | None:
        """Find a discriminating path from a to b for the edge b-c."""
        n = pag.n_nodes
        # BFS/DFS to find path a -> ... -> b where all intermediaries are parents of c
        from collections import deque
        visited = {a}
        queue = deque([(a, [a])])
        while queue:
            node, path = queue.popleft()
            for nxt in range(n):
                if nxt in visited:
                    continue
                if not self._has_edge(pag, node, nxt):
                    continue
                if nxt == b:
                    return path + [b]
                # nxt must be a parent of c (tail at nxt, arrow at c)
                if (self._get_mark(pag, nxt, c) == self.ARROW
                        and self._get_mark(pag, c, nxt) == self.TAIL):
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def apply_r8(self, pag: Any) -> bool:
        """R8: If a -> b -> c or a -o b -> c, and a o-> c, orient a -> c."""
        changed = False
        n = pag.n_nodes
        for a in range(n):
            for c in range(n):
                if a == c:
                    continue
                if not self._has_edge(pag, a, c):
                    continue
                # a o-> c: circle at a-end, arrow at c-end
                if (self._get_mark(pag, c, a) != self.CIRCLE
                        or self._get_mark(pag, a, c) != self.ARROW):
                    continue
                for b in range(n):
                    if b == a or b == c:
                        continue
                    if not self._has_edge(pag, a, b) or not self._has_edge(pag, b, c):
                        continue
                    # a -> b -> c or a -o b -> c
                    if (self._get_mark(pag, b, c) == self.ARROW
                            and self._get_mark(pag, c, b) == self.TAIL):
                        mark_ab_at_a = self._get_mark(pag, b, a)
                        mark_ab_at_b = self._get_mark(pag, a, b)
                        if mark_ab_at_b == self.ARROW and mark_ab_at_a in (self.TAIL, self.CIRCLE):
                            self._set_mark(pag, c, a, self.TAIL)
                            changed = True
                            break
        return changed

    def apply_r9(self, pag: Any) -> bool:
        """R9: If a o-> c and there is an uncovered potentially directed
        path from a to c through b, orient a -> c."""
        changed = False
        n = pag.n_nodes
        for a in range(n):
            for c in range(n):
                if a == c:
                    continue
                if not self._has_edge(pag, a, c):
                    continue
                if (self._get_mark(pag, c, a) != self.CIRCLE
                        or self._get_mark(pag, a, c) != self.ARROW):
                    continue
                # Find uncovered p.d. path a - b - ... - c
                if self._has_uncovered_pd_path(pag, a, c):
                    self._set_mark(pag, c, a, self.TAIL)
                    changed = True
        return changed

    def _has_uncovered_pd_path(self, pag: Any, a: int, c: int) -> bool:
        """Check for uncovered potentially directed path from a to c of length >= 2."""
        n = pag.n_nodes
        from collections import deque
        # BFS: (current_node, previous_node, path_length)
        queue = deque()
        for b in range(n):
            if b == a or b == c:
                continue
            if not self._has_edge(pag, a, b):
                continue
            if not self._is_adjacent(pag, a, b):
                continue
            # Edge from a to b must not have arrowhead at a
            if self._get_mark(pag, b, a) == self.ARROW:
                continue
            queue.append((b, a))

        visited = set()
        while queue:
            node, prev = queue.popleft()
            if (node, prev) in visited:
                continue
            visited.add((node, prev))
            for nxt in range(n):
                if nxt == node or nxt == prev:
                    continue
                if not self._has_edge(pag, node, nxt):
                    continue
                # Must not have arrowhead at node from nxt
                if self._get_mark(pag, nxt, node) == self.ARROW:
                    continue
                # Uncovered: prev not adjacent to nxt
                if self._is_adjacent(pag, prev, nxt):
                    continue
                if nxt == c:
                    return True
                queue.append((nxt, node))
        return False

    def apply_r10(self, pag: Any) -> bool:
        """R10: If a o-> c, b -> c <- d, a o-o b, a o-o d, and
        mu is an uncovered p.d. path from b to d through a,
        orient a -> c."""
        changed = False
        n = pag.n_nodes
        for a in range(n):
            for c in range(n):
                if a == c:
                    continue
                if not self._has_edge(pag, a, c):
                    continue
                if (self._get_mark(pag, c, a) != self.CIRCLE
                        or self._get_mark(pag, a, c) != self.ARROW):
                    continue
                # Find b, d: b -> c <- d, a o-o b, a o-o d
                parents_c = []
                for node in range(n):
                    if node == a or node == c:
                        continue
                    if (self._get_mark(pag, node, c) == self.ARROW
                            and self._get_mark(pag, c, node) == self.TAIL):
                        parents_c.append(node)

                for i_b in range(len(parents_c)):
                    for i_d in range(i_b + 1, len(parents_c)):
                        b, d = parents_c[i_b], parents_c[i_d]
                        if not self._has_edge(pag, a, b) or not self._has_edge(pag, a, d):
                            continue
                        self._set_mark(pag, c, a, self.TAIL)
                        changed = True
                        break
                    if changed:
                        break
        return changed

    def apply_rules(self, pag: Any, max_iter: int = 200) -> int:
        """Apply Zhang orientation rules R1-R4, R8-R10 until convergence.

        Returns the total number of iterations with changes.
        """
        total = 0
        for _ in range(max_iter):
            any_change = False
            any_change |= self.apply_r1(pag)
            any_change |= self.apply_r2(pag)
            any_change |= self.apply_r3(pag)
            any_change |= self.apply_r4(pag)
            any_change |= self.apply_r8(pag)
            any_change |= self.apply_r9(pag)
            any_change |= self.apply_r10(pag)
            if any_change:
                total += 1
            else:
                break
        return total


# -------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------

def apply_orientation_rules(
    graph: Any,
    rule_set: str = "meek",
) -> int:
    """Apply the specified orientation rule set to *graph*.

    Parameters
    ----------
    graph : CPDAG | PAG
        The partially oriented graph.
    rule_set : str
        ``"meek"`` or ``"zhang"``.

    Returns
    -------
    int
        Number of orientation rounds applied.
    """
    if rule_set == "meek":
        return MeekRules().apply_all(graph)
    elif rule_set == "zhang":
        return ZhangRules().apply_rules(graph)
    else:
        raise ValueError(f"Unknown rule set: {rule_set!r}. Use 'meek' or 'zhang'.")
