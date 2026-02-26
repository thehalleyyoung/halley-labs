"""
Constraint Encoder
===================

Encodes various constraint families arising from a causal DAG into the
rows of the LP constraint matrix  A x = b  (or A x <= b).

Constraint types:
  - Normalisation (probabilities sum to 1)
  - Markov / d-separation (conditional-independence structure)
  - Marginal consistency (observed marginals match)
  - Interventional (do-operator removes incoming edges)
  - Moment matching (observed moments)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Types
# ---------------------------------------------------------------------------

class ConstraintType(Enum):
    NORMALIZATION = auto()
    MARKOV = auto()
    MARGINAL_CONSISTENCY = auto()
    INTERVENTION = auto()
    MOMENT_MATCHING = auto()
    DSEPARATION = auto()
    NONNEGATIVITY = auto()


@dataclass
class ConstraintBlock:
    """A named block of rows in the constraint matrix."""
    name: str
    ctype: ConstraintType
    row_start: int
    row_end: int   # exclusive
    variables: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def num_rows(self) -> int:
        return self.row_end - self.row_start


# ---------------------------------------------------------------------------
#  Constraint encoder
# ---------------------------------------------------------------------------

class ConstraintEncoder:
    """
    Builds the constraint matrix  A, b  for the causal polytope LP.

    All constraints are equality constraints of the form  A x = b .
    Inequality constraints (bounds) are handled separately.

    Parameters
    ----------
    dag : DAGSpec
        The (possibly mutilated) DAG.
    """

    def __init__(self, dag):
        self.dag = dag
        self._topo: List[str] = dag.topological_order()
        self._strides: Dict[str, int] = self._compute_strides()
        self._total_vars: int = self._compute_total_vars()
        self._rows: List[np.ndarray] = []
        self._rhs: List[float] = []
        self._names: List[str] = []
        self._blocks: List[ConstraintBlock] = []
        self._row_count: int = 0

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def add_normalization_constraints(self) -> ConstraintBlock:
        """
        Add the normalisation constraint: sum_x P(x) = 1 .

        This is a single row with all coefficients equal to 1.
        """
        row = np.ones(self._total_vars, dtype=np.float64)
        block = self._add_constraint_row(
            row, 1.0, "normalization", ConstraintType.NORMALIZATION
        )
        logger.debug("Added normalisation constraint (1 row)")
        return block

    def add_markov_constraints(self) -> List[ConstraintBlock]:
        """
        Add conditional-independence constraints implied by the DAG
        Markov property.

        For each node X with parents pa(X) and each non-descendant Z
        that is not a parent:

            P(X=x | pa(X)=pa) = P(X=x | pa(X)=pa, Z=z)  for all x, pa, z

        Equivalently:
            sum_{z} P(x, pa, z) * P(x', pa', z') - ... = 0
        encoded as linear equalities in the joint distribution vector.
        """
        blocks: List[ConstraintBlock] = []

        for node in self._topo:
            parents = self.dag.parents(node)
            non_desc = self._non_descendants(node)
            # Remove parents from non-descendants
            non_parents_non_desc = [v for v in non_desc if v not in parents]

            if not non_parents_non_desc or not parents:
                continue

            for z_var in non_parents_non_desc:
                block = self._add_markov_ci_constraint(node, parents, z_var)
                if block is not None:
                    blocks.append(block)

        logger.debug("Added %d Markov CI constraint blocks", len(blocks))
        return blocks

    def add_observed_marginal_constraints(self, observed) -> List[ConstraintBlock]:
        """
        Add constraints ensuring the joint distribution's marginals
        match the observed marginals.

        For each observed marginal P_obs(S=s):
            sum_{x: x_S = s} P(x) = P_obs(S=s)

        Parameters
        ----------
        observed : ObservedMarginals
            Object containing observed marginal tables.
        """
        blocks: List[ConstraintBlock] = []

        for var_set in observed.variable_sets():
            marginal = observed.get_marginal(var_set)
            if marginal is None:
                continue

            vars_list = sorted(var_set)
            block = self._add_marginal_block(vars_list, marginal, var_set)
            if block is not None:
                blocks.append(block)

        logger.debug("Added %d marginal consistency blocks", len(blocks))
        return blocks

    def add_intervention_constraints(
        self, interv_var: str, interv_val: int
    ) -> ConstraintBlock:
        """
        Add constraints for do(interv_var = interv_val).

        Under the intervention, P(interv_var=v) = delta(v, interv_val).
        So:
            sum_{x: x_{interv_var} = interv_val} P(x) = 1
            sum_{x: x_{interv_var} != interv_val} P(x) = 0

        Equivalently, for each v != interv_val:
            sum_{x: x_{interv_var} = v} P(x) = 0
        """
        start = self._row_count
        card = self.dag.card[interv_var]

        for v in range(card):
            if v == interv_val:
                continue
            row = np.zeros(self._total_vars, dtype=np.float64)
            for flat_idx in range(self._total_vars):
                assign = self._flat_to_assignment(flat_idx)
                if assign[interv_var] == v:
                    row[flat_idx] = 1.0
            self._add_row(row, 0.0, f"do({interv_var}={interv_val})_not_{v}")

        block = ConstraintBlock(
            name=f"intervention_{interv_var}={interv_val}",
            ctype=ConstraintType.INTERVENTION,
            row_start=start,
            row_end=self._row_count,
            variables=frozenset({interv_var}),
        )
        self._blocks.append(block)
        logger.debug(
            "Added intervention constraint do(%s=%d): %d rows",
            interv_var, interv_val, block.num_rows,
        )
        return block

    def add_dseparation_constraints(
        self,
        x_set: FrozenSet[str],
        y_set: FrozenSet[str],
        z_set: FrozenSet[str],
    ) -> Optional[ConstraintBlock]:
        """
        Add a conditional independence constraint  X _||_ Y | Z
        (if it holds by d-separation in the DAG).

            P(X=x, Y=y | Z=z) = P(X=x | Z=z) * P(Y=y | Z=z)

        Encoded as:
            P(x,y,z) * P(z) - P(x,z) * P(y,z) = 0   (bilinear, linearised)

        For the LP we use a first-order linearisation around the current
        marginals, or we enumerate and add linear rows.
        """
        if not self._check_dsep(x_set, y_set, z_set):
            logger.debug(
                "d-separation does not hold for %s _||_ %s | %s",
                x_set, y_set, z_set,
            )
            return None

        start = self._row_count
        x_vars = sorted(x_set)
        y_vars = sorted(y_set)
        z_vars = sorted(z_set)

        x_cards = [self.dag.card[v] for v in x_vars]
        y_cards = [self.dag.card[v] for v in y_vars]
        z_cards = [self.dag.card[v] for v in z_vars]

        x_range = list(itertools.product(*[range(c) for c in x_cards]))
        y_range = list(itertools.product(*[range(c) for c in y_cards]))
        z_range = list(itertools.product(*[range(c) for c in z_cards]))

        for z_assign in z_range:
            z_dict = dict(zip(z_vars, z_assign))

            for x_assign in x_range:
                x_dict = dict(zip(x_vars, x_assign))

                for y_assign in y_range:
                    y_dict = dict(zip(y_vars, y_assign))

                    # Row encoding:
                    # P(x,y,z) - (1/|Y|)*P(x,z) - (1/|X|)*P(y,z) + (1/(|X|*|Y|))*P(z) = 0
                    # This is a linear relaxation of the CI constraint.
                    row = np.zeros(self._total_vars, dtype=np.float64)

                    for flat_idx in range(self._total_vars):
                        a = self._flat_to_assignment(flat_idx)

                        matches_xyz = all(
                            a[v] == x_dict[v] for v in x_vars
                        ) and all(
                            a[v] == y_dict[v] for v in y_vars
                        ) and all(
                            a[v] == z_dict[v] for v in z_vars
                        )
                        matches_xz = all(
                            a[v] == x_dict[v] for v in x_vars
                        ) and all(
                            a[v] == z_dict[v] for v in z_vars
                        )
                        matches_yz = all(
                            a[v] == y_dict[v] for v in y_vars
                        ) and all(
                            a[v] == z_dict[v] for v in z_vars
                        )

                        n_y = len(y_range)
                        n_x = len(x_range)

                        if matches_xyz:
                            row[flat_idx] += 1.0
                        if matches_xz:
                            row[flat_idx] -= 1.0 / max(n_y, 1)
                        if matches_yz:
                            row[flat_idx] -= 1.0 / max(n_x, 1)

                    name = (
                        f"dsep_{x_vars}={x_assign}_"
                        f"{y_vars}={y_assign}|{z_vars}={z_assign}"
                    )
                    self._add_row(row, 0.0, name)

        block = ConstraintBlock(
            name=f"dsep_{x_set}_{y_set}|{z_set}",
            ctype=ConstraintType.DSEPARATION,
            row_start=start,
            row_end=self._row_count,
            variables=x_set | y_set | z_set,
        )
        self._blocks.append(block)
        return block

    def add_moment_constraints(
        self,
        variable: str,
        moments: List[float],
    ) -> ConstraintBlock:
        """
        Add constraints matching the first k moments of a variable.

        E[X^k] = moments[k-1]   for k = 1, ..., len(moments)
        """
        start = self._row_count
        card = self.dag.card[variable]

        for k, mk in enumerate(moments, 1):
            row = np.zeros(self._total_vars, dtype=np.float64)

            for flat_idx in range(self._total_vars):
                assign = self._flat_to_assignment(flat_idx)
                val = assign[variable]
                row[flat_idx] = float(val ** k)

            self._add_row(row, mk, f"moment_{variable}_k{k}")

        block = ConstraintBlock(
            name=f"moments_{variable}",
            ctype=ConstraintType.MOMENT_MATCHING,
            row_start=start,
            row_end=self._row_count,
            variables=frozenset({variable}),
        )
        self._blocks.append(block)
        return block

    def add_conditional_probability_constraint(
        self,
        child: str,
        parents: List[str],
        cpt: np.ndarray,
    ) -> ConstraintBlock:
        """
        Add constraints encoding a known conditional probability table (CPT):
            P(child | parents) = cpt

        The CPT is indexed as cpt[parent_config, child_val].
        """
        start = self._row_count
        child_card = self.dag.card[child]
        parent_cards = [self.dag.card[p] for p in parents]

        if not parents:
            # Marginal constraint on child
            for c_val in range(child_card):
                row = np.zeros(self._total_vars, dtype=np.float64)
                for flat_idx in range(self._total_vars):
                    assign = self._flat_to_assignment(flat_idx)
                    if assign[child] == c_val:
                        row[flat_idx] = 1.0
                self._add_row(row, float(cpt[c_val]), f"cpt_{child}={c_val}")
        else:
            parent_configs = list(itertools.product(*[range(c) for c in parent_cards]))

            for pi, pa_config in enumerate(parent_configs):
                pa_dict = dict(zip(parents, pa_config))

                for c_val in range(child_card):
                    # P(child=c_val, pa=pa_config) = cpt[pi, c_val] * P(pa=pa_config)
                    # Linearised:  P(child=c_val, pa=pa_config) - cpt * P(pa=pa_config) = 0
                    row = np.zeros(self._total_vars, dtype=np.float64)
                    cpt_val = float(cpt[pi, c_val]) if cpt.ndim > 1 else float(cpt[c_val])

                    for flat_idx in range(self._total_vars):
                        assign = self._flat_to_assignment(flat_idx)
                        matches_pa = all(assign[p] == pa_dict[p] for p in parents)
                        if matches_pa:
                            if assign[child] == c_val:
                                row[flat_idx] += 1.0
                            row[flat_idx] -= cpt_val

                    name = f"cpt_{child}={c_val}|{parents}={pa_config}"
                    self._add_row(row, 0.0, name)

        block = ConstraintBlock(
            name=f"cpt_{child}|{parents}",
            ctype=ConstraintType.MARKOV,
            row_start=start,
            row_end=self._row_count,
            variables=frozenset({child} | set(parents)),
        )
        self._blocks.append(block)
        return block

    def build_constraint_matrix(
        self,
    ) -> Tuple[sparse.csr_matrix, np.ndarray, List[str]]:
        """
        Build and return the full constraint matrix A, rhs b, and names.

        Returns
        -------
        A : csr_matrix of shape (m, total_vars)
        b : ndarray of shape (m,)
        names : list of constraint names
        """
        if not self._rows:
            m = 0
            A = sparse.csr_matrix((m, self._total_vars), dtype=np.float64)
            b = np.array([], dtype=np.float64)
            return A, b, []

        A_dense = np.vstack(self._rows)
        b = np.array(self._rhs, dtype=np.float64)

        # Convert to sparse
        A = sparse.csr_matrix(A_dense, dtype=np.float64)

        logger.info(
            "Built constraint matrix: %d rows x %d cols, nnz=%d",
            A.shape[0], A.shape[1], A.nnz,
        )
        return A, b, list(self._names)

    def build_incremental(
        self,
        new_rows: List[np.ndarray],
        new_rhs: List[float],
        new_names: List[str],
        existing_A: sparse.spmatrix,
        existing_b: np.ndarray,
    ) -> Tuple[sparse.csr_matrix, np.ndarray, List[str]]:
        """
        Incrementally add rows to an existing constraint matrix.
        """
        if not new_rows:
            return existing_A, existing_b, list(self._names)

        new_A = sparse.csr_matrix(np.vstack(new_rows), dtype=np.float64)
        combined_A = sparse.vstack([existing_A, new_A], format="csr")
        combined_b = np.concatenate([existing_b, np.array(new_rhs, dtype=np.float64)])
        combined_names = list(self._names) + list(new_names)

        return combined_A, combined_b, combined_names

    def remove_block(self, block: ConstraintBlock) -> None:
        """
        Remove a constraint block (mark for rebuilding).
        Note: actual removal happens on next build_constraint_matrix().
        """
        rows_to_keep = []
        rhs_to_keep = []
        names_to_keep = []

        for i in range(len(self._rows)):
            if i < block.row_start or i >= block.row_end:
                rows_to_keep.append(self._rows[i])
                rhs_to_keep.append(self._rhs[i])
                names_to_keep.append(self._names[i])

        self._rows = rows_to_keep
        self._rhs = rhs_to_keep
        self._names = names_to_keep
        self._row_count = len(self._rows)

        # Update block indices
        removed_count = block.num_rows
        updated_blocks = []
        for b in self._blocks:
            if b.name == block.name:
                continue
            if b.row_start >= block.row_end:
                updated_blocks.append(ConstraintBlock(
                    name=b.name,
                    ctype=b.ctype,
                    row_start=b.row_start - removed_count,
                    row_end=b.row_end - removed_count,
                    variables=b.variables,
                ))
            elif b.row_end <= block.row_start:
                updated_blocks.append(b)
        self._blocks = updated_blocks

    def get_blocks(self) -> List[ConstraintBlock]:
        return list(self._blocks)

    def get_block_by_name(self, name: str) -> Optional[ConstraintBlock]:
        for b in self._blocks:
            if b.name == name:
                return b
        return None

    @property
    def num_constraints(self) -> int:
        return self._row_count

    @property
    def num_variables(self) -> int:
        return self._total_vars

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _add_markov_ci_constraint(
        self, node: str, parents: List[str], z_var: str
    ) -> Optional[ConstraintBlock]:
        """
        Encode X _||_ Z | pa(X) as linear constraints.

        P(X=x, Z=z | pa) = P(X=x | pa) * P(Z=z | pa)

        Linearised:
            P(x, z, pa) - (1/|Z|) * sum_z' P(x, z', pa) = 0
        for each (x, z, pa) assignment.
        """
        start = self._row_count
        card_x = self.dag.card[node]
        card_z = self.dag.card[z_var]
        pa_cards = [self.dag.card[p] for p in parents]

        if not pa_cards:
            return None

        pa_configs = list(itertools.product(*[range(c) for c in pa_cards]))

        for pa_config in pa_configs:
            pa_dict = dict(zip(parents, pa_config))

            for x_val in range(card_x):
                for z_val in range(card_z):
                    row = np.zeros(self._total_vars, dtype=np.float64)

                    for flat_idx in range(self._total_vars):
                        assign = self._flat_to_assignment(flat_idx)

                        # Check parent match
                        if not all(assign[p] == pa_dict[p] for p in parents):
                            continue

                        if assign[node] == x_val:
                            if assign[z_var] == z_val:
                                # P(x, z, pa)
                                row[flat_idx] += 1.0
                            # Subtract (1/|Z|) * P(x, z', pa) for all z'
                            row[flat_idx] -= 1.0 / card_z

                    # Only add if row is non-trivial
                    if np.any(np.abs(row) > 1e-15):
                        name = f"markov_{node}={x_val},{z_var}={z_val}|{parents}={pa_config}"
                        self._add_row(row, 0.0, name)

        if self._row_count == start:
            return None

        block = ConstraintBlock(
            name=f"markov_{node}_||_{z_var}|{parents}",
            ctype=ConstraintType.MARKOV,
            row_start=start,
            row_end=self._row_count,
            variables=frozenset({node, z_var} | set(parents)),
        )
        self._blocks.append(block)
        return block

    def _add_marginal_block(
        self,
        vars_list: List[str],
        marginal: np.ndarray,
        var_set: FrozenSet[str],
    ) -> Optional[ConstraintBlock]:
        """Add marginal consistency constraints for a set of variables."""
        start = self._row_count
        cards = [self.dag.card[v] for v in vars_list]
        configs = list(itertools.product(*[range(c) for c in cards]))

        if marginal.ndim == 1:
            # Flatten matches configs
            for ci, config in enumerate(configs):
                if ci >= len(marginal):
                    break
                row = np.zeros(self._total_vars, dtype=np.float64)
                config_dict = dict(zip(vars_list, config))

                for flat_idx in range(self._total_vars):
                    assign = self._flat_to_assignment(flat_idx)
                    if all(assign[v] == config_dict[v] for v in vars_list):
                        row[flat_idx] = 1.0

                self._add_row(
                    row, float(marginal[ci]),
                    f"marginal_{vars_list}={config}",
                )
        else:
            # Multi-dimensional marginal
            flat_marg = marginal.ravel()
            for ci, config in enumerate(configs):
                if ci >= len(flat_marg):
                    break
                row = np.zeros(self._total_vars, dtype=np.float64)
                config_dict = dict(zip(vars_list, config))

                for flat_idx in range(self._total_vars):
                    assign = self._flat_to_assignment(flat_idx)
                    if all(assign[v] == config_dict[v] for v in vars_list):
                        row[flat_idx] = 1.0

                self._add_row(
                    row, float(flat_marg[ci]),
                    f"marginal_{vars_list}={config}",
                )

        if self._row_count == start:
            return None

        block = ConstraintBlock(
            name=f"marginal_{var_set}",
            ctype=ConstraintType.MARGINAL_CONSISTENCY,
            row_start=start,
            row_end=self._row_count,
            variables=var_set,
        )
        self._blocks.append(block)
        return block

    def _add_constraint_row(
        self,
        row: np.ndarray,
        rhs: float,
        name: str,
        ctype: ConstraintType,
    ) -> ConstraintBlock:
        """Add a single constraint row and create a block for it."""
        start = self._row_count
        self._add_row(row, rhs, name)
        block = ConstraintBlock(
            name=name,
            ctype=ctype,
            row_start=start,
            row_end=self._row_count,
        )
        self._blocks.append(block)
        return block

    def _add_row(self, row: np.ndarray, rhs: float, name: str) -> None:
        """Append a single row to the internal storage."""
        self._rows.append(row.copy())
        self._rhs.append(rhs)
        self._names.append(name)
        self._row_count += 1

    def _check_dsep(
        self,
        x_set: FrozenSet[str],
        y_set: FrozenSet[str],
        z_set: FrozenSet[str],
    ) -> bool:
        """
        Check d-separation: X _||_ Y | Z in the DAG.

        Uses the Bayes-Ball algorithm.
        """
        # Bayes-Ball: start from X, see if we can reach Y without being blocked by Z
        observed = set(z_set)
        ancestors_of_observed = set()
        for z in observed:
            ancestors_of_observed |= set(self.dag.ancestors(z))

        # BFS/DFS with direction tracking
        visited_up: Set[str] = set()
        visited_down: Set[str] = set()
        reachable: Set[str] = set()

        # Queue entries: (node, direction) where direction is "up" or "down"
        queue: List[Tuple[str, str]] = []
        for x in x_set:
            queue.append((x, "up"))

        while queue:
            node, direction = queue.pop()

            if direction == "up" and node not in visited_up:
                visited_up.add(node)
                if node not in observed:
                    reachable.add(node)
                    # Visit parents (going up)
                    for parent in self.dag.parents(node):
                        queue.append((parent, "up"))
                    # Visit children (going down)
                    for child in self.dag.children(node):
                        queue.append((child, "down"))
                else:
                    # Node is observed: can pass through to parents
                    if node in ancestors_of_observed or node in observed:
                        for parent in self.dag.parents(node):
                            queue.append((parent, "up"))

            elif direction == "down" and node not in visited_down:
                visited_down.add(node)
                if node not in observed:
                    reachable.add(node)
                    # Visit children (going down)
                    for child in self.dag.children(node):
                        queue.append((child, "down"))
                else:
                    # Node is observed: can go up through parents
                    for parent in self.dag.parents(node):
                        queue.append((parent, "up"))

        # d-separated if no Y node is reachable
        return not bool(y_set & reachable)

    def _non_descendants(self, node: str) -> List[str]:
        """Return non-descendants of node in the DAG."""
        desc = self.dag.descendants(node)
        return [n for n in self.dag.nodes if n != node and n not in desc]

    def _flat_to_assignment(self, flat_idx: int) -> Dict[str, int]:
        assignment: Dict[str, int] = {}
        remaining = flat_idx
        for node in self._topo:
            card = self.dag.card[node]
            stride = self._strides[node]
            assignment[node] = (remaining // stride) % card
        return assignment

    def _assignment_to_flat(self, assignment: Dict[str, int]) -> int:
        idx = 0
        for node in self._topo:
            idx += assignment.get(node, 0) * self._strides[node]
        return idx

    def _compute_strides(self) -> Dict[str, int]:
        strides: Dict[str, int] = {}
        s = 1
        for node in reversed(self._topo):
            strides[node] = s
            s *= self.dag.card[node]
        return strides

    def _compute_total_vars(self) -> int:
        total = 1
        for n in self.dag.nodes:
            total *= self.dag.card[n]
        return total
