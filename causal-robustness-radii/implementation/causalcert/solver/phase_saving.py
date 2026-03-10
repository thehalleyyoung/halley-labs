"""
Phase saving and polarity heuristics for CDCL-based robustness radius search.

In CDCL solvers the *phase* of a variable is whether it was last assigned
true (edit applied) or false (edit not applied).  Phase saving records
the polarity each variable had when it was unassigned during backtracking
and re-uses that polarity the next time the variable is chosen as a
decision.  This dramatically reduces the number of conflicts after a
restart because the solver tends to re-enter the same region of the search
space.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from causalcert.types import StructuralEdit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polarity enum-like constants
# ---------------------------------------------------------------------------

class Polarity:
    """Polarity constants for edit variables."""

    POSITIVE = True    # apply edit
    NEGATIVE = False   # do not apply edit


# ---------------------------------------------------------------------------
# Phase saver
# ---------------------------------------------------------------------------

class PhaseSaver:
    """Phase saving and polarity selection for edit variables.

    Tracks the last polarity each edit variable was assigned and re-uses it
    on the next decision.  Supports random phase injection for diversification
    and solution-guided initialisation.

    Parameters
    ----------
    edits : Sequence[StructuralEdit]
        All candidate edit variables.
    default_polarity : bool
        Default polarity for variables never seen before.
        ``True`` means *apply the edit*; ``False`` means *do not apply*.
    random_freq : float
        Probability of choosing a random polarity instead of the saved phase
        on each decision.  0.0 disables random injection.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        edits: Sequence[StructuralEdit],
        default_polarity: bool = False,
        random_freq: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._default = default_polarity
        self._random_freq = max(0.0, min(1.0, random_freq))
        self._rng = random.Random(seed)

        # Saved phase for each edit
        self._phase: dict[StructuralEdit, bool] = {e: default_polarity for e in edits}

        # Solution-guided phases (from best known solution)
        self._solution_phase: dict[StructuralEdit, bool] | None = None

        # Statistics
        self._n_saved: int = 0
        self._n_random: int = 0
        self._n_solution: int = 0

    # -- phase recording ----------------------------------------------------

    def save_phase(self, edit: StructuralEdit, polarity: bool) -> None:
        """Record the polarity of *edit* (called during backtracking).

        Parameters
        ----------
        edit : StructuralEdit
            The edit variable being unassigned.
        polarity : bool
            The polarity it had before backtracking.
        """
        self._phase[edit] = polarity
        self._n_saved += 1

    def save_phases_bulk(
        self, assignments: dict[StructuralEdit, bool],
    ) -> None:
        """Save phases for many variables at once.

        Parameters
        ----------
        assignments : dict[StructuralEdit, bool]
            Mapping from edits to their current polarity.
        """
        for edit, pol in assignments.items():
            self._phase[edit] = pol
        self._n_saved += len(assignments)

    # -- polarity selection -------------------------------------------------

    def get_polarity(self, edit: StructuralEdit) -> bool:
        """Return the polarity to use when branching on *edit*.

        Applies random injection and solution-guided priority.

        Parameters
        ----------
        edit : StructuralEdit
            The edit to decide on.

        Returns
        -------
        bool
            ``True`` to apply the edit, ``False`` to skip it.
        """
        # Random injection
        if self._random_freq > 0.0 and self._rng.random() < self._random_freq:
            self._n_random += 1
            return self._rng.choice([True, False])

        # Solution-guided phase
        if self._solution_phase is not None and edit in self._solution_phase:
            self._n_solution += 1
            return self._solution_phase[edit]

        # Saved phase
        return self._phase.get(edit, self._default)

    # -- solution-guided initialisation -------------------------------------

    def set_solution_guide(
        self, solution_edits: frozenset[StructuralEdit],
    ) -> None:
        """Initialise phases from a known (possibly suboptimal) solution.

        Edits in *solution_edits* get positive polarity; all others negative.

        Parameters
        ----------
        solution_edits : frozenset[StructuralEdit]
            Edits present in the guiding solution.
        """
        self._solution_phase = {}
        for edit in self._phase:
            self._solution_phase[edit] = edit in solution_edits
        logger.debug(
            "Phase saver: solution guide set with %d active edits",
            len(solution_edits),
        )

    def clear_solution_guide(self) -> None:
        """Remove solution-guided phases."""
        self._solution_phase = None

    # -- reset --------------------------------------------------------------

    def reset_all(self, polarity: bool | None = None) -> None:
        """Reset all saved phases to the default (or to *polarity*).

        Parameters
        ----------
        polarity : bool | None
            Override polarity.  ``None`` uses the constructor default.
        """
        val = polarity if polarity is not None else self._default
        for edit in self._phase:
            self._phase[edit] = val

    def reset_variable(self, edit: StructuralEdit) -> None:
        """Reset a single variable's phase to the default."""
        self._phase[edit] = self._default

    # -- configuration ------------------------------------------------------

    def set_random_freq(self, freq: float) -> None:
        """Update the random-injection frequency.

        Parameters
        ----------
        freq : float
            New frequency in [0, 1].
        """
        self._random_freq = max(0.0, min(1.0, freq))

    # -- register new variable ----------------------------------------------

    def register(self, edit: StructuralEdit) -> None:
        """Register a new edit variable with default polarity."""
        if edit not in self._phase:
            self._phase[edit] = self._default

    def register_many(self, edits: Sequence[StructuralEdit]) -> None:
        """Register multiple edit variables."""
        for e in edits:
            self.register(e)

    # -- statistics ---------------------------------------------------------

    @property
    def n_saved(self) -> int:
        """Total phase-save operations."""
        return self._n_saved

    @property
    def n_random(self) -> int:
        """Total random-polarity decisions."""
        return self._n_random

    @property
    def n_solution_guided(self) -> int:
        """Total solution-guided polarity decisions."""
        return self._n_solution

    @property
    def phase_map(self) -> dict[StructuralEdit, bool]:
        """Current saved phases (read-only copy)."""
        return dict(self._phase)

    def positive_edits(self) -> list[StructuralEdit]:
        """Return edits whose saved phase is positive (apply)."""
        return [e for e, p in self._phase.items() if p]

    def negative_edits(self) -> list[StructuralEdit]:
        """Return edits whose saved phase is negative (skip)."""
        return [e for e, p in self._phase.items() if not p]

    def __repr__(self) -> str:  # pragma: no cover
        n_pos = sum(1 for p in self._phase.values() if p)
        return (
            f"PhaseSaver(vars={len(self._phase)}, pos={n_pos}, "
            f"random_freq={self._random_freq:.2f})"
        )
