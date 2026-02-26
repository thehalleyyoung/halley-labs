"""
Multi-dimensional potential table for junction-tree inference.

Stores factors (CPDs, messages, clique potentials) as dense or sparse
numpy tensors indexed by variable name.  All standard factor-graph
operations—multiply, marginalize, reduce, normalize—are provided in
both linear and numerically-stable log-space variants.
"""

from __future__ import annotations

import copy
import itertools
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class PotentialTable:
    """Multi-dimensional potential (factor) table over discrete variables.

    Parameters
    ----------
    variables : list[str]
        Ordered variable names; axis *i* of ``values`` corresponds to
        ``variables[i]``.
    cardinalities : dict[str, int]
        Number of states for every variable that appears in ``variables``.
    values : np.ndarray or None
        Dense tensor of potentials.  If *None* a uniform table is created.
    log_space : bool
        If *True* the stored ``values`` represent **log-potentials**.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        variables: List[str],
        cardinalities: Dict[str, int],
        values: Optional[NDArray] = None,
        log_space: bool = False,
    ) -> None:
        self.variables: List[str] = list(variables)
        self.cardinalities: Dict[str, int] = {
            v: cardinalities[v] for v in self.variables
        }
        self._var_to_axis: Dict[str, int] = {
            v: i for i, v in enumerate(self.variables)
        }
        self.log_space: bool = log_space

        shape = tuple(self.cardinalities[v] for v in self.variables)
        if values is not None:
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(
                    f"Shape mismatch: expected {shape}, got {arr.shape}"
                )
            self.values: NDArray = arr
        else:
            if log_space:
                self.values = np.zeros(shape, dtype=np.float64)
            else:
                self.values = np.ones(shape, dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  Accessors
    # ------------------------------------------------------------------ #

    @property
    def ndim(self) -> int:
        return len(self.variables)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.values.shape

    @property
    def size(self) -> int:
        return int(self.values.size)

    @property
    def variable_set(self) -> FrozenSet[str]:
        return frozenset(self.variables)

    def axis_of(self, variable: str) -> int:
        """Return the axis index for *variable*."""
        return self._var_to_axis[variable]

    def get_entry(self, assignment: Dict[str, int]) -> float:
        """Return the potential value at a full or partial assignment.

        For a partial assignment the remaining axes are summed out
        (marginalised) before returning a scalar.
        """
        idx: List[Union[int, slice]] = [slice(None)] * self.ndim
        for var, val in assignment.items():
            if var in self._var_to_axis:
                idx[self._var_to_axis[var]] = val
        result = self.values[tuple(idx)]
        if isinstance(result, np.ndarray):
            return float(result.sum())
        return float(result)

    def set_entry(self, assignment: Dict[str, int], value: float) -> None:
        """Set the potential at a complete assignment."""
        idx = tuple(assignment[v] for v in self.variables)
        self.values[idx] = value

    # ------------------------------------------------------------------ #
    #  Core algebraic operations
    # ------------------------------------------------------------------ #

    def multiply(self, other: "PotentialTable") -> "PotentialTable":
        """Point-wise multiplication (addition in log-space).

        The result table's variable list is the **union** of both
        operands' variables, with axes aligned by variable name.
        """
        if self.log_space != other.log_space:
            raise ValueError("Cannot multiply tables in different spaces")

        new_vars = list(self.variables)
        for v in other.variables:
            if v not in self._var_to_axis:
                new_vars.append(v)

        new_cards: Dict[str, int] = {}
        new_cards.update(self.cardinalities)
        new_cards.update(other.cardinalities)

        # Validate overlapping cardinalities
        for v in self.variables:
            if v in other.cardinalities:
                if self.cardinalities[v] != other.cardinalities[v]:
                    raise ValueError(
                        f"Cardinality mismatch for '{v}': "
                        f"{self.cardinalities[v]} vs {other.cardinalities[v]}"
                    )

        new_shape = tuple(new_cards[v] for v in new_vars)
        new_var_idx = {v: i for i, v in enumerate(new_vars)}

        # Broadcast self into the new shape
        self_idx = [np.newaxis] * len(new_vars)
        for v in self.variables:
            self_idx[new_var_idx[v]] = slice(None)
        a = self.values.reshape(
            tuple(
                self.cardinalities[v] if isinstance(self_idx[i], slice) else 1
                for i, v in enumerate(new_vars)
            )
        )

        other_idx = [np.newaxis] * len(new_vars)
        for v in other.variables:
            other_idx[new_var_idx[v]] = slice(None)
        b = other.values.reshape(
            tuple(
                other.cardinalities[v]
                if isinstance(other_idx[i], slice)
                else 1
                for i, v in enumerate(new_vars)
            )
        )

        if self.log_space:
            new_values = a + b
        else:
            new_values = a * b

        result = PotentialTable(
            new_vars,
            new_cards,
            np.broadcast_to(new_values, new_shape).copy(),
            log_space=self.log_space,
        )
        return result

    def divide(self, other: "PotentialTable") -> "PotentialTable":
        """Point-wise division (subtraction in log-space).

        ``other.variables`` must be a subset of ``self.variables``.
        Zero-division is handled gracefully (result is zero).
        """
        if self.log_space != other.log_space:
            raise ValueError("Cannot divide tables in different spaces")

        for v in other.variables:
            if v not in self._var_to_axis:
                raise ValueError(
                    f"Division requires other.variables ⊆ self.variables; "
                    f"'{v}' not in self"
                )

        new_var_idx = {v: i for i, v in enumerate(self.variables)}
        shape_b = [1] * self.ndim
        for v in other.variables:
            shape_b[new_var_idx[v]] = other.cardinalities[v]
        b = other.values.reshape(tuple(shape_b))

        if self.log_space:
            new_values = self.values - b
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                new_values = np.where(b != 0, self.values / b, 0.0)

        return PotentialTable(
            list(self.variables),
            dict(self.cardinalities),
            new_values,
            log_space=self.log_space,
        )

    def marginalize(self, variables: Iterable[str]) -> "PotentialTable":
        """Sum out (marginalize) the given variables.

        In log-space this uses the *log-sum-exp* trick for stability.
        """
        vars_to_remove = [v for v in variables if v in self._var_to_axis]
        if not vars_to_remove:
            return self.copy()

        axes = tuple(self._var_to_axis[v] for v in vars_to_remove)
        remaining = [v for v in self.variables if v not in vars_to_remove]
        remaining_cards = {v: self.cardinalities[v] for v in remaining}

        if self.log_space:
            new_values = _logsumexp_axes(self.values, axes)
        else:
            new_values = self.values.sum(axis=axes)

        return PotentialTable(
            remaining, remaining_cards, new_values, log_space=self.log_space
        )

    def max_marginalize(self, variables: Iterable[str]) -> "PotentialTable":
        """Max-marginalize the given variables (for MAP inference)."""
        vars_to_remove = [v for v in variables if v in self._var_to_axis]
        if not vars_to_remove:
            return self.copy()

        remaining = [v for v in self.variables if v not in vars_to_remove]
        remaining_cards = {v: self.cardinalities[v] for v in remaining}

        result = self.values.copy()
        # Max over axes in reverse sorted order to keep indices stable
        for v in sorted(vars_to_remove, key=lambda u: self._var_to_axis[u], reverse=True):
            ax = self._var_to_axis[v]
            result = result.max(axis=ax)

        return PotentialTable(
            remaining, remaining_cards, result, log_space=self.log_space
        )

    def reduce(self, evidence: Dict[str, int]) -> "PotentialTable":
        """Condition (reduce) on observed evidence.

        Returns a new table with the observed variables removed and the
        corresponding slices selected.
        """
        idx: List[Union[int, slice]] = [slice(None)] * self.ndim
        observed_vars = []
        for var, val in evidence.items():
            if var in self._var_to_axis:
                idx[self._var_to_axis[var]] = val
                observed_vars.append(var)

        if not observed_vars:
            return self.copy()

        new_values = self.values[tuple(idx)]
        remaining = [v for v in self.variables if v not in observed_vars]
        remaining_cards = {v: self.cardinalities[v] for v in remaining}

        if new_values.ndim == 0:
            new_values = new_values.reshape(())

        return PotentialTable(
            remaining, remaining_cards, np.asarray(new_values).copy(),
            log_space=self.log_space,
        )

    def normalize(self) -> "PotentialTable":
        """Return a copy normalised so entries sum to 1 (or log-normalised)."""
        result = self.copy()
        if self.log_space:
            z = _logsumexp_all(result.values)
            result.values = result.values - z
        else:
            z = result.values.sum()
            if z > 0:
                result.values = result.values / z
        return result

    def normalize_inplace(self) -> float:
        """Normalise in-place and return the partition value."""
        if self.log_space:
            z = _logsumexp_all(self.values)
            self.values -= z
            return float(z)
        else:
            z = float(self.values.sum())
            if z > 0:
                self.values /= z
            return z

    # ------------------------------------------------------------------ #
    #  Log-space conversion
    # ------------------------------------------------------------------ #

    def to_log_space(self) -> "PotentialTable":
        """Convert from linear to log-space."""
        if self.log_space:
            return self.copy()
        with np.errstate(divide="ignore"):
            log_vals = np.log(np.maximum(self.values, 1e-300))
        return PotentialTable(
            list(self.variables),
            dict(self.cardinalities),
            log_vals,
            log_space=True,
        )

    def from_log_space(self) -> "PotentialTable":
        """Convert from log-space to linear."""
        if not self.log_space:
            return self.copy()
        return PotentialTable(
            list(self.variables),
            dict(self.cardinalities),
            np.exp(self.values),
            log_space=False,
        )

    # ------------------------------------------------------------------ #
    #  Tensor contraction
    # ------------------------------------------------------------------ #

    def contract(
        self, other: "PotentialTable", contract_vars: Iterable[str]
    ) -> "PotentialTable":
        """Tensor contraction: multiply then marginalize ``contract_vars``.

        More memory-efficient than multiply().marginalize() for large
        tables because we fuse the two operations via einsum.
        """
        contract_set = set(contract_vars)
        all_vars = list(self.variables)
        for v in other.variables:
            if v not in self._var_to_axis:
                all_vars.append(v)

        cards: Dict[str, int] = {}
        cards.update(self.cardinalities)
        cards.update(other.cardinalities)

        # Build einsum subscript string (supports up to 52 variables)
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if len(all_vars) > len(letters):
            # Fallback to multiply+marginalize
            return self.multiply(other).marginalize(contract_set)

        var_to_letter = {v: letters[i] for i, v in enumerate(all_vars)}

        subs_self = "".join(var_to_letter[v] for v in self.variables)
        subs_other = "".join(var_to_letter[v] for v in other.variables)
        result_vars = [v for v in all_vars if v not in contract_set]
        subs_result = "".join(var_to_letter[v] for v in result_vars)

        subscripts = f"{subs_self},{subs_other}->{subs_result}"

        if self.log_space:
            a = np.exp(self.values)
            b = np.exp(other.values)
            result_vals = np.einsum(subscripts, a, b)
            with np.errstate(divide="ignore"):
                result_vals = np.log(np.maximum(result_vals, 1e-300))
        else:
            result_vals = np.einsum(subscripts, self.values, other.values)

        result_cards = {v: cards[v] for v in result_vars}
        return PotentialTable(
            result_vars, result_cards, result_vals, log_space=self.log_space
        )

    # ------------------------------------------------------------------ #
    #  Sparse support
    # ------------------------------------------------------------------ #

    @classmethod
    def from_sparse(
        cls,
        variables: List[str],
        cardinalities: Dict[str, int],
        entries: Dict[Tuple[int, ...], float],
        log_space: bool = False,
    ) -> "PotentialTable":
        """Build a potential table from a sparse dict of non-default entries.

        Parameters
        ----------
        entries : dict mapping tuples of state indices to potential values.
        """
        shape = tuple(cardinalities[v] for v in variables)
        if log_space:
            values = np.full(shape, -np.inf, dtype=np.float64)
        else:
            values = np.zeros(shape, dtype=np.float64)
        for idx, val in entries.items():
            values[idx] = val
        return cls(variables, cardinalities, values, log_space=log_space)

    def to_sparse(self, threshold: float = 1e-12) -> Dict[Tuple[int, ...], float]:
        """Return non-negligible entries as a sparse dict."""
        if self.log_space:
            mask = self.values > (np.max(self.values) - 30)  # within exp(-30)
        else:
            mask = np.abs(self.values) > threshold
        indices = np.argwhere(mask)
        return {tuple(int(x) for x in idx): float(self.values[tuple(idx)]) for idx in indices}

    def sparsity(self) -> float:
        """Fraction of entries that are effectively zero."""
        if self.log_space:
            near_zero = self.values < (np.max(self.values) - 30)
        else:
            near_zero = np.abs(self.values) < 1e-12
        return float(near_zero.sum()) / max(self.size, 1)

    # ------------------------------------------------------------------ #
    #  Adaptive-discretization helpers
    # ------------------------------------------------------------------ #

    def expand_variable(
        self, variable: str, new_cardinality: int
    ) -> "PotentialTable":
        """Return a new table where *variable* has been expanded to a
        larger cardinality by linear interpolation along that axis.
        """
        if variable not in self._var_to_axis:
            raise ValueError(f"Variable '{variable}' not in table")
        axis = self._var_to_axis[variable]
        old_card = self.cardinalities[variable]
        if new_cardinality <= old_card:
            return self.copy()

        old_coords = np.linspace(0, 1, old_card)
        new_coords = np.linspace(0, 1, new_cardinality)

        # Interpolate along axis
        new_shape = list(self.shape)
        new_shape[axis] = new_cardinality
        new_values = np.empty(new_shape, dtype=np.float64)

        # Move target axis to last position for convenience
        moved = np.moveaxis(self.values, axis, -1)
        new_moved_shape = list(moved.shape)
        new_moved_shape[-1] = new_cardinality
        new_moved = np.empty(new_moved_shape, dtype=np.float64)

        it = np.nditer(
            moved[..., 0], flags=["multi_index"], op_flags=["readonly"]
        )
        while not it.finished:
            idx = it.multi_index
            old_slice = moved[idx]
            new_moved[idx] = np.interp(new_coords, old_coords, old_slice)
            it.iternext()

        new_values = np.moveaxis(new_moved, -1, axis)
        new_cards = dict(self.cardinalities)
        new_cards[variable] = new_cardinality
        return PotentialTable(
            list(self.variables), new_cards, new_values, log_space=self.log_space
        )

    def coarsen_variable(
        self, variable: str, new_cardinality: int
    ) -> "PotentialTable":
        """Reduce the cardinality of *variable* by summing adjacent bins."""
        if variable not in self._var_to_axis:
            raise ValueError(f"Variable '{variable}' not in table")
        axis = self._var_to_axis[variable]
        old_card = self.cardinalities[variable]
        if new_cardinality >= old_card:
            return self.copy()

        bin_edges = np.linspace(0, old_card, new_cardinality + 1).astype(int)
        slices: List[NDArray] = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            sl = np.take(self.values, range(lo, hi), axis=axis)
            if self.log_space:
                slices.append(_logsumexp_axes(sl, (axis,)))
            else:
                slices.append(sl.sum(axis=axis, keepdims=False))

        new_values = np.stack(slices, axis=axis)
        new_cards = dict(self.cardinalities)
        new_cards[variable] = new_cardinality
        return PotentialTable(
            list(self.variables), new_cards, new_values, log_space=self.log_space
        )

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def copy(self) -> "PotentialTable":
        """Deep copy."""
        return PotentialTable(
            list(self.variables),
            dict(self.cardinalities),
            self.values.copy(),
            log_space=self.log_space,
        )

    def reorder(self, new_order: List[str]) -> "PotentialTable":
        """Return a copy with axes reordered to match ``new_order``."""
        if set(new_order) != set(self.variables):
            raise ValueError("new_order must contain exactly the same variables")
        perm = [self._var_to_axis[v] for v in new_order]
        new_values = np.transpose(self.values, perm)
        return PotentialTable(
            new_order, dict(self.cardinalities), new_values.copy(),
            log_space=self.log_space,
        )

    def allclose(self, other: "PotentialTable", atol: float = 1e-8) -> bool:
        """Check if two tables are numerically close."""
        if set(self.variables) != set(other.variables):
            return False
        reordered = other.reorder(self.variables)
        return bool(np.allclose(self.values, reordered.values, atol=atol))

    def entropy(self) -> float:
        """Shannon entropy of the (normalised) distribution."""
        p = self.normalize()
        if p.log_space:
            p = p.from_log_space()
        vals = p.values.ravel()
        vals = vals[vals > 0]
        return float(-np.sum(vals * np.log(vals)))

    def kl_divergence(self, other: "PotentialTable") -> float:
        """KL(self || other) after normalising both distributions."""
        p = self.normalize()
        q = other.normalize()
        if p.log_space:
            p = p.from_log_space()
        if q.log_space:
            q = q.from_log_space()
        q_reord = q.reorder(p.variables)
        pv = p.values.ravel()
        qv = q_reord.values.ravel()
        mask = pv > 0
        pv = pv[mask]
        qv = np.maximum(qv[mask], 1e-300)
        return float(np.sum(pv * (np.log(pv) - np.log(qv))))

    def expected_value(self, variable: str, bin_centers: NDArray) -> float:
        """Compute E[variable] given bin centres for that variable.

        The table is normalised first, then marginalised to the single
        variable, and the expectation is computed as a dot product.
        """
        marg = self.normalize().marginalize(
            [v for v in self.variables if v != variable]
        )
        if marg.log_space:
            marg = marg.from_log_space()
        probs = marg.values.ravel()
        if len(bin_centers) != len(probs):
            raise ValueError("bin_centers length must match variable cardinality")
        return float(np.dot(probs, bin_centers))

    def tail_probability(
        self, variable: str, threshold: float, bin_edges: NDArray
    ) -> float:
        """Compute P(variable > threshold) from discretized table.

        Parameters
        ----------
        bin_edges : array of shape (cardinality+1,) giving bin boundaries.
        """
        marg = self.normalize().marginalize(
            [v for v in self.variables if v != variable]
        )
        if marg.log_space:
            marg = marg.from_log_space()
        probs = marg.values.ravel()
        # Identify bins whose midpoint exceeds threshold
        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mask = midpoints > threshold
        return float(probs[mask].sum())

    def __repr__(self) -> str:
        space = "log" if self.log_space else "linear"
        return (
            f"PotentialTable(vars={self.variables}, "
            f"shape={self.shape}, space={space})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PotentialTable):
            return NotImplemented
        return self.allclose(other)


# ------------------------------------------------------------------ #
#  Module-level numerical helpers
# ------------------------------------------------------------------ #

def _logsumexp_axes(arr: NDArray, axes: Tuple[int, ...]) -> NDArray:
    """Stable log-sum-exp over multiple axes simultaneously."""
    max_val = arr.max()
    if np.isinf(max_val) and max_val < 0:
        shape = list(arr.shape)
        for ax in sorted(axes, reverse=True):
            shape.pop(ax)
        return np.full(shape or (), -np.inf, dtype=np.float64)
    shifted = arr - max_val
    summed = np.exp(shifted).sum(axis=axes)
    return np.log(summed) + max_val


def _logsumexp_all(arr: NDArray) -> float:
    """Stable log-sum-exp over all elements."""
    max_val = float(arr.max())
    if np.isinf(max_val) and max_val < 0:
        return -np.inf
    return float(max_val + np.log(np.exp(arr - max_val).sum()))


def multiply_potentials(tables: Sequence[PotentialTable]) -> PotentialTable:
    """Multiply a sequence of potential tables together."""
    if not tables:
        raise ValueError("Cannot multiply empty sequence of potentials")
    result = tables[0].copy()
    for t in tables[1:]:
        result = result.multiply(t)
    return result


def marginalize_to(
    table: PotentialTable, target_vars: Iterable[str]
) -> PotentialTable:
    """Marginalize *table* down to only ``target_vars``."""
    target = set(target_vars)
    to_remove = [v for v in table.variables if v not in target]
    return table.marginalize(to_remove)
