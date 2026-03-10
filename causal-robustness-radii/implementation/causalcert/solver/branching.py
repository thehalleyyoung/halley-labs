"""
Advanced branching heuristics for CDCL-based robustness radius search.

Implements LRB (Learning Rate Branching), CHB (Conflict History-Based),
EVSIDS (Exponential VSIDS), random branching, and a unified
:class:`BranchingEngine` that selects among them.

All heuristics maintain a score per *edit variable* (add / delete /
reverse an edge).  The highest-scoring unassigned variable is chosen as
the next decision.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from causalcert.solver.phase_saving import PhaseSaver
from causalcert.solver.clause_database import EditLiteral
from causalcert.types import StructuralEdit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EVSIDS (Exponential VSIDS)
# ---------------------------------------------------------------------------

class EVSIDS:
    """Exponential VSIDS branching heuristic.

    An improved variant of VSIDS where the bump value grows exponentially
    so that recent conflicts dominate.

    Parameters
    ----------
    edits : Sequence[StructuralEdit]
        Candidate edit variables.
    decay : float
        Multiplicative decay factor in (0, 1).
    """

    def __init__(
        self,
        edits: Sequence[StructuralEdit],
        decay: float = 0.95,
    ) -> None:
        self._scores: dict[StructuralEdit, float] = {e: 0.0 for e in edits}
        self._decay = decay
        self._bump_inc: float = 1.0
        self._n_bumps: int = 0

    def bump(self, edit: StructuralEdit) -> None:
        """Increase the activity of *edit*."""
        if edit in self._scores:
            self._scores[edit] += self._bump_inc
            self._n_bumps += 1

    def bump_many(self, edits: frozenset[StructuralEdit]) -> None:
        """Bump all edits in a conflict."""
        for e in edits:
            self.bump(e)
        self._bump_inc /= self._decay
        if self._bump_inc > 1e100:
            self._rescale()

    def _rescale(self) -> None:
        """Rescale all scores to prevent floating-point overflow."""
        factor = 1.0 / self._bump_inc
        for e in self._scores:
            self._scores[e] *= factor
        self._bump_inc = 1.0

    def decay(self) -> None:
        """Apply decay (implicit via bump increment growth)."""
        pass  # decay is embedded in bump_inc growth

    def pick(self, assigned: set[StructuralEdit]) -> StructuralEdit | None:
        """Return the highest-scoring unassigned edit.

        Parameters
        ----------
        assigned : set[StructuralEdit]
            Currently assigned variables.

        Returns
        -------
        StructuralEdit | None
            Best unassigned edit, or ``None`` if all are assigned.
        """
        best: StructuralEdit | None = None
        best_score = -1.0
        for e, s in self._scores.items():
            if e not in assigned and s > best_score:
                best_score = s
                best = e
        return best

    def score(self, edit: StructuralEdit) -> float:
        """Return the current activity score of *edit*."""
        return self._scores.get(edit, 0.0)

    def register(self, edit: StructuralEdit) -> None:
        """Register a new variable."""
        if edit not in self._scores:
            self._scores[edit] = 0.0


# ---------------------------------------------------------------------------
# LRB (Learning Rate Branching)
# ---------------------------------------------------------------------------

class LRB:
    """Learning Rate Branching heuristic.

    LRB estimates the "learning rate" of each variable — how often it
    participates in conflict derivation relative to how often it has been
    assigned.  Variables with high learning rates are preferred for branching.

    Parameters
    ----------
    edits : Sequence[StructuralEdit]
        Candidate edit variables.
    alpha : float
        EMA smoothing factor.
    step_decay : float
        Decay applied to step counters.
    """

    def __init__(
        self,
        edits: Sequence[StructuralEdit],
        alpha: float = 0.4,
        step_decay: float = 0.95,
    ) -> None:
        self._alpha = alpha
        self._step_decay = step_decay

        # Counters
        self._participated: dict[StructuralEdit, float] = {e: 0.0 for e in edits}
        self._reasoned: dict[StructuralEdit, float] = {e: 0.0 for e in edits}
        self._assigned_count: dict[StructuralEdit, float] = {e: 0.0 for e in edits}

        # Exponential moving average of learning rate
        self._ema: dict[StructuralEdit, float] = {e: 0.0 for e in edits}

        self._total_conflicts: int = 0

    def on_conflict(
        self,
        involved: frozenset[StructuralEdit],
        reason_edits: frozenset[StructuralEdit],
    ) -> None:
        """Update scores after a conflict.

        Parameters
        ----------
        involved : frozenset[StructuralEdit]
            Edits that participated in the conflict derivation.
        reason_edits : frozenset[StructuralEdit]
            Edits that appeared as reasons (antecedents) during analysis.
        """
        self._total_conflicts += 1
        for e in involved:
            if e in self._participated:
                self._participated[e] += 1.0
        for e in reason_edits:
            if e in self._reasoned:
                self._reasoned[e] += 1.0

    def on_assign(self, edit: StructuralEdit) -> None:
        """Notify that *edit* was assigned."""
        if edit in self._assigned_count:
            self._assigned_count[edit] += 1.0

    def on_unassign(self, edit: StructuralEdit) -> None:
        """Notify that *edit* was unassigned (backtrack).

        Computes the learning rate for the interval this variable was assigned
        and updates the EMA.
        """
        if edit not in self._participated:
            return

        interval_assigns = max(1.0, self._assigned_count[edit])
        learning_rate = self._participated[edit] / interval_assigns

        old_ema = self._ema[edit]
        self._ema[edit] = self._alpha * learning_rate + (1.0 - self._alpha) * old_ema

        # Reset interval counters
        self._participated[edit] = 0.0
        self._reasoned[edit] = 0.0

    def decay(self) -> None:
        """Decay step counters."""
        for e in self._assigned_count:
            self._assigned_count[e] *= self._step_decay
            self._participated[e] *= self._step_decay
            self._reasoned[e] *= self._step_decay

    def pick(self, assigned: set[StructuralEdit]) -> StructuralEdit | None:
        """Return the edit with the highest learning rate EMA."""
        best: StructuralEdit | None = None
        best_score = -1.0
        for e, s in self._ema.items():
            if e not in assigned and s > best_score:
                best_score = s
                best = e
        return best

    def score(self, edit: StructuralEdit) -> float:
        """Return the EMA learning rate of *edit*."""
        return self._ema.get(edit, 0.0)

    def register(self, edit: StructuralEdit) -> None:
        """Register a new variable."""
        if edit not in self._ema:
            self._ema[edit] = 0.0
            self._participated[edit] = 0.0
            self._reasoned[edit] = 0.0
            self._assigned_count[edit] = 0.0


# ---------------------------------------------------------------------------
# CHB (Conflict History-Based)
# ---------------------------------------------------------------------------

class CHB:
    """Conflict History-Based branching heuristic.

    CHB maintains an EMA of the conflict "reward" for each variable.
    The reward for variable *v* at conflict *c* is ``1 / (c - last_conflict[v])``
    where *last_conflict[v]* is the last conflict at which *v* participated.

    Parameters
    ----------
    edits : Sequence[StructuralEdit]
        Candidate edit variables.
    alpha_init : float
        Initial EMA smoothing factor.
    alpha_min : float
        Minimum (converged) smoothing factor.
    alpha_decay : float
        Reduction applied to alpha after each conflict.
    """

    def __init__(
        self,
        edits: Sequence[StructuralEdit],
        alpha_init: float = 0.4,
        alpha_min: float = 0.06,
        alpha_decay: float = 1e-6,
    ) -> None:
        self._alpha = alpha_init
        self._alpha_min = alpha_min
        self._alpha_decay = alpha_decay

        self._q: dict[StructuralEdit, float] = {e: 0.0 for e in edits}
        self._last_conflict: dict[StructuralEdit, int] = {e: 0 for e in edits}
        self._conflict_counter: int = 0

    def on_conflict(self, involved: frozenset[StructuralEdit]) -> None:
        """Update CHB scores after a conflict.

        Parameters
        ----------
        involved : frozenset[StructuralEdit]
            Edits participating in the conflict derivation.
        """
        self._conflict_counter += 1
        for e in involved:
            if e not in self._q:
                continue
            last = self._last_conflict[e]
            interval = max(1, self._conflict_counter - last)
            reward = 1.0 / interval
            self._q[e] = (1.0 - self._alpha) * self._q[e] + self._alpha * reward
            self._last_conflict[e] = self._conflict_counter

        # Decay alpha
        self._alpha = max(self._alpha_min, self._alpha - self._alpha_decay)

    def pick(self, assigned: set[StructuralEdit]) -> StructuralEdit | None:
        """Return the highest-scoring unassigned edit."""
        best: StructuralEdit | None = None
        best_score = -1.0
        for e, s in self._q.items():
            if e not in assigned and s > best_score:
                best_score = s
                best = e
        return best

    def score(self, edit: StructuralEdit) -> float:
        """Return the CHB score of *edit*."""
        return self._q.get(edit, 0.0)

    def register(self, edit: StructuralEdit) -> None:
        """Register a new variable."""
        if edit not in self._q:
            self._q[edit] = 0.0
            self._last_conflict[edit] = 0


# ---------------------------------------------------------------------------
# Random branching
# ---------------------------------------------------------------------------

class RandomBranching:
    """Uniform random branching for diversification.

    Parameters
    ----------
    edits : Sequence[StructuralEdit]
        Candidate edit variables.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        edits: Sequence[StructuralEdit],
        seed: int | None = None,
    ) -> None:
        self._edits = list(edits)
        self._rng = random.Random(seed)

    def pick(self, assigned: set[StructuralEdit]) -> StructuralEdit | None:
        """Pick a random unassigned edit."""
        unassigned = [e for e in self._edits if e not in assigned]
        if not unassigned:
            return None
        return self._rng.choice(unassigned)

    def register(self, edit: StructuralEdit) -> None:
        """Register a new variable."""
        if edit not in self._edits:
            self._edits.append(edit)


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class BranchingStrategy:
    """Named constants for branching strategies."""

    EVSIDS = "evsids"
    LRB = "lrb"
    CHB = "chb"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Unified branching engine
# ---------------------------------------------------------------------------

class BranchingEngine:
    """Unified branching engine with strategy selection.

    Parameters
    ----------
    edits : Sequence[StructuralEdit]
        Candidate edit variables.
    strategy : str
        One of ``"evsids"``, ``"lrb"``, ``"chb"``, ``"random"``.
    phase_saver : PhaseSaver | None
        Optional phase saver for polarity decisions.
    seed : int | None
        Random seed for random branching and tie-breaking.
    evsids_decay : float
        Decay factor for EVSIDS.
    lrb_alpha : float
        EMA alpha for LRB.
    chb_alpha : float
        Initial alpha for CHB.
    """

    def __init__(
        self,
        edits: Sequence[StructuralEdit],
        strategy: str = BranchingStrategy.EVSIDS,
        phase_saver: PhaseSaver | None = None,
        seed: int | None = None,
        evsids_decay: float = 0.95,
        lrb_alpha: float = 0.4,
        chb_alpha: float = 0.4,
    ) -> None:
        self._strategy = strategy
        self._phase_saver = phase_saver
        self._rng = random.Random(seed)

        self._evsids = EVSIDS(edits, evsids_decay)
        self._lrb = LRB(edits, lrb_alpha)
        self._chb = CHB(edits, chb_alpha)
        self._random = RandomBranching(edits, seed)

        self._n_decisions: int = 0

    # -- decision -----------------------------------------------------------

    def decide(self, assigned: set[StructuralEdit]) -> EditLiteral | None:
        """Pick the next branching literal.

        Parameters
        ----------
        assigned : set[StructuralEdit]
            Currently assigned edit variables.

        Returns
        -------
        EditLiteral | None
            The literal to branch on, or ``None`` if all variables are assigned.
        """
        edit = self._pick_variable(assigned)
        if edit is None:
            return None

        polarity = True  # default: apply the edit
        if self._phase_saver is not None:
            polarity = self._phase_saver.get_polarity(edit)

        self._n_decisions += 1
        return EditLiteral(edit, polarity)

    def _pick_variable(self, assigned: set[StructuralEdit]) -> StructuralEdit | None:
        """Select the next variable using the active strategy."""
        if self._strategy == BranchingStrategy.EVSIDS:
            return self._evsids.pick(assigned)
        elif self._strategy == BranchingStrategy.LRB:
            return self._lrb.pick(assigned)
        elif self._strategy == BranchingStrategy.CHB:
            return self._chb.pick(assigned)
        elif self._strategy == BranchingStrategy.RANDOM:
            return self._random.pick(assigned)
        else:
            return self._evsids.pick(assigned)

    # -- conflict notification ---------------------------------------------

    def on_conflict(
        self,
        involved: frozenset[StructuralEdit],
        reason_edits: frozenset[StructuralEdit] | None = None,
    ) -> None:
        """Notify all heuristics of a conflict.

        Parameters
        ----------
        involved : frozenset[StructuralEdit]
            Edits participating in the conflict.
        reason_edits : frozenset[StructuralEdit] | None
            Edits from antecedent clauses (for LRB).
        """
        self._evsids.bump_many(involved)
        self._chb.on_conflict(involved)
        if reason_edits is not None:
            self._lrb.on_conflict(involved, reason_edits)
        else:
            self._lrb.on_conflict(involved, frozenset())

    # -- assignment / unassignment notifications ---------------------------

    def on_assign(self, edit: StructuralEdit) -> None:
        """Notify that *edit* was assigned."""
        self._lrb.on_assign(edit)

    def on_unassign(self, edit: StructuralEdit) -> None:
        """Notify that *edit* was unassigned (backtrack)."""
        self._lrb.on_unassign(edit)

    # -- periodic decay -----------------------------------------------------

    def decay(self) -> None:
        """Apply periodic decay to all heuristics."""
        self._lrb.decay()

    # -- variable registration ---------------------------------------------

    def register(self, edit: StructuralEdit) -> None:
        """Register a new edit variable with all heuristics."""
        self._evsids.register(edit)
        self._lrb.register(edit)
        self._chb.register(edit)
        self._random.register(edit)

    # -- strategy switching -------------------------------------------------

    def set_strategy(self, strategy: str) -> None:
        """Switch the active branching strategy.

        Parameters
        ----------
        strategy : str
            One of ``"evsids"``, ``"lrb"``, ``"chb"``, ``"random"``.
        """
        self._strategy = strategy
        logger.debug("Branching engine switched to %s", strategy)

    # -- queries ------------------------------------------------------------

    @property
    def strategy(self) -> str:
        """Currently active strategy."""
        return self._strategy

    @property
    def n_decisions(self) -> int:
        """Total branching decisions made."""
        return self._n_decisions

    def scores(
        self, assigned: set[StructuralEdit] | None = None,
    ) -> dict[StructuralEdit, float]:
        """Return scores from the active heuristic.

        Parameters
        ----------
        assigned : set[StructuralEdit] | None
            If given, only include unassigned variables.

        Returns
        -------
        dict[StructuralEdit, float]
            Edit → score mapping.
        """
        if self._strategy == BranchingStrategy.EVSIDS:
            src = self._evsids._scores
        elif self._strategy == BranchingStrategy.LRB:
            src = self._lrb._ema
        elif self._strategy == BranchingStrategy.CHB:
            src = self._chb._q
        else:
            return {}

        if assigned is not None:
            return {e: s for e, s in src.items() if e not in assigned}
        return dict(src)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BranchingEngine(strategy={self._strategy!r}, "
            f"decisions={self._n_decisions})"
        )
