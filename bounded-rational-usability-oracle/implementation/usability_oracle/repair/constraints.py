"""
usability_oracle.repair.constraints — Cognitive-law constraint encoding for Z3.

Provides :class:`ConstraintEncoder` which translates human-factors laws
(Fitts' Law, Hick-Hyman Law, working-memory limits, target-sizing
guidelines, layout non-overlap, hierarchy depth) into Z3 arithmetic
constraints over the repair search space.

Each method takes a Z3 ``Solver`` and the relevant Z3 variables, and
adds the appropriate assertions.  All constraints use *real* or
*integer* Z3 arithmetic — no bit-vectors.

References
----------
- Fitts, P. M. (1954). The information capacity of the human motor
  system in controlling the amplitude of movement. *J. Exp. Psych.*
- Hick, W. E. (1952). On the rate of gain of information. *QJEP*.
- Cowan, N. (2001). The magical number 4 in short-term memory. *BBS*.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import z3

logger = logging.getLogger(__name__)


class ConstraintEncoder:
    """Encodes cognitive-law and layout constraints as Z3 assertions.

    All public methods follow the pattern::

        encode_<law>_constraint(solver, <relevant vars>, <bound>) -> None

    They mutate the solver in-place by calling ``solver.add(...)``.
    """

    # ── Fitts' Law --------------------------------------------------------

    def encode_fitts_constraint(
        self,
        solver: z3.Solver,
        distance_var: z3.ArithRef,
        width_var: z3.ArithRef,
        max_id: float,
    ) -> None:
        """Encode Fitts' Law index-of-difficulty upper bound.

        Fitts' Law:  ID = log₂(1 + D / W)

        Since Z3 does not have a native log₂, we encode the equivalent
        exponential form:

            1 + D / W  ≤  2^max_id

        Which is linear when rearranged:

            D / W  ≤  2^max_id − 1
            D  ≤  (2^max_id − 1) · W

        Parameters
        ----------
        solver : z3.Solver
        distance_var : z3.ArithRef
            Real variable representing movement distance D (pixels).
        width_var : z3.ArithRef
            Real variable representing target width W (pixels).
        max_id : float
            Maximum allowed index of difficulty (bits).
        """
        if max_id <= 0:
            logger.warning("max_id must be positive, got %f; skipping", max_id)
            return

        upper = 2.0 ** max_id - 1.0
        upper_z3 = z3.RealVal(str(upper))

        # D ≤ (2^max_id - 1) * W
        solver.add(distance_var <= upper_z3 * width_var)
        # D ≥ 0, W > 0 (physical constraints)
        solver.add(distance_var >= 0)
        solver.add(width_var > 0)

        logger.debug(
            "Fitts constraint: D ≤ %.2f * W  (max ID = %.2f bits)", upper, max_id
        )

    def encode_fitts_time_constraint(
        self,
        solver: z3.Solver,
        distance_var: z3.ArithRef,
        width_var: z3.ArithRef,
        time_var: z3.ArithRef,
        a: float = 0.05,
        b: float = 0.15,
        max_time: float = 2.0,
    ) -> None:
        """Encode Fitts' Law movement time upper bound.

        MT = a + b · log₂(1 + D/W)

        Linearised as:
            D ≤ (2^((max_time − a) / b) − 1) · W

        Parameters
        ----------
        solver : z3.Solver
        distance_var, width_var : z3.ArithRef
        time_var : z3.ArithRef
            Variable representing movement time.
        a, b : float
            Fitts' Law parameters.
        max_time : float
            Maximum movement time in seconds.
        """
        if b <= 0:
            return
        max_id = (max_time - a) / b
        if max_id <= 0:
            return

        upper = 2.0 ** max_id - 1.0
        solver.add(distance_var <= z3.RealVal(str(upper)) * width_var)
        solver.add(distance_var >= 0)
        solver.add(width_var > 0)
        solver.add(time_var <= z3.RealVal(str(max_time)))
        solver.add(time_var >= z3.RealVal(str(a)))

    # ── Hick-Hyman Law ----------------------------------------------------

    def encode_hick_constraint(
        self,
        solver: z3.Solver,
        n_choices_var: z3.ArithRef,
        max_choices: int,
    ) -> None:
        """Encode Hick-Hyman Law constraint on the number of choices.

        Hick-Hyman Law:  RT = a + b · log₂(n + 1)

        Reducing n directly reduces decision time.  We impose a hard upper
        bound on the number of simultaneously visible choices.

        Parameters
        ----------
        solver : z3.Solver
        n_choices_var : z3.ArithRef
            Integer variable for the number of choices.
        max_choices : int
            Maximum allowed simultaneous choices (typically 5–9).
        """
        if max_choices < 1:
            max_choices = 1

        solver.add(n_choices_var >= 1)
        solver.add(n_choices_var <= max_choices)

        logger.debug("Hick constraint: n ≤ %d", max_choices)

    def encode_hick_time_constraint(
        self,
        solver: z3.Solver,
        n_choices_var: z3.ArithRef,
        time_var: z3.ArithRef,
        a: float = 0.2,
        b: float = 0.15,
        max_time: float = 1.5,
    ) -> None:
        """Encode Hick-Hyman reaction-time upper bound.

        RT = a + b · log₂(n + 1) ≤ max_time
        ⟹  n + 1 ≤ 2^((max_time - a) / b)
        ⟹  n ≤ 2^((max_time - a) / b) - 1
        """
        if b <= 0:
            return
        exponent = (max_time - a) / b
        if exponent <= 0:
            return
        max_n = int(2.0 ** exponent) - 1
        max_n = max(1, max_n)

        solver.add(n_choices_var >= 1)
        solver.add(n_choices_var <= max_n)
        solver.add(time_var <= z3.RealVal(str(max_time)))
        solver.add(time_var >= z3.RealVal(str(a)))

    # ── Working Memory ----------------------------------------------------

    def encode_memory_constraint(
        self,
        solver: z3.Solver,
        load_var: z3.ArithRef,
        max_load: int,
    ) -> None:
        """Encode Cowan's working-memory capacity limit.

        The number of items the user must hold in working memory at any
        point during the task must not exceed *max_load* (typically 3–5).

        Parameters
        ----------
        solver : z3.Solver
        load_var : z3.ArithRef
            Integer variable for working-memory load (chunks).
        max_load : int
            Maximum chunks (Cowan's estimate: 4 ± 1).
        """
        if max_load < 1:
            max_load = 1

        solver.add(load_var >= 0)
        solver.add(load_var <= max_load)

        logger.debug("Memory constraint: load ≤ %d chunks", max_load)

    def encode_memory_decay_constraint(
        self,
        solver: z3.Solver,
        load_var: z3.ArithRef,
        time_var: z3.ArithRef,
        max_load: int = 4,
        half_life: float = 7.0,
    ) -> None:
        """Encode memory decay: effective capacity decreases over time.

        Effective capacity = max_load · 2^(-t / half_life)

        We linearise by discretising time into intervals and encoding
        step-wise capacity bounds.
        """
        solver.add(load_var >= 0)
        solver.add(time_var >= 0)

        # Piecewise linear approximation at key time points
        time_points = [0.0, half_life / 2, half_life, half_life * 1.5, half_life * 2]
        for t in time_points:
            effective = max_load * (2.0 ** (-t / half_life))
            effective_int = max(1, int(math.ceil(effective)))
            t_z3 = z3.RealVal(str(t))
            next_t = t + half_life / 2
            next_z3 = z3.RealVal(str(next_t))
            solver.add(
                z3.Implies(
                    z3.And(time_var >= t_z3, time_var < next_z3),
                    load_var <= effective_int,
                )
            )

    # ── Target Size -------------------------------------------------------

    def encode_target_size_constraint(
        self,
        solver: z3.Solver,
        width_var: z3.ArithRef,
        height_var: z3.ArithRef,
        min_size: float,
    ) -> None:
        """Ensure target meets minimum size (WCAG / platform guidelines).

        Both width and height must be at least *min_size* pixels.
        This satisfies the WCAG 2.5.8 Target Size (Minimum) criterion
        as well as Apple/Google HIG touch-target recommendations.

        Parameters
        ----------
        solver : z3.Solver
        width_var, height_var : z3.ArithRef
            Real variables for target dimensions.
        min_size : float
            Minimum dimension in pixels (typically 24 or 44).
        """
        min_z3 = z3.RealVal(str(max(0.0, min_size)))
        solver.add(width_var >= min_z3)
        solver.add(height_var >= min_z3)

        logger.debug("Target size constraint: W,H ≥ %.0f px", min_size)

    def encode_target_area_constraint(
        self,
        solver: z3.Solver,
        width_var: z3.ArithRef,
        height_var: z3.ArithRef,
        min_area: float,
    ) -> None:
        """Ensure target area is above a threshold.

        W · H ≥ min_area

        This is a quadratic constraint; Z3 handles it via nonlinear
        real arithmetic (NRA).
        """
        solver.add(width_var * height_var >= z3.RealVal(str(min_area)))
        solver.add(width_var > 0)
        solver.add(height_var > 0)

    # ── Layout (non-overlap) ----------------------------------------------

    def encode_layout_constraint(
        self,
        solver: z3.Solver,
        positions: list[tuple[z3.ArithRef, z3.ArithRef, z3.ArithRef, z3.ArithRef]],
        no_overlap: bool = True,
    ) -> None:
        """Encode layout constraints for a set of positioned elements.

        Each entry in *positions* is a tuple (x, y, width, height) of
        Z3 real variables.  If *no_overlap* is True, pairwise
        non-overlap constraints are added.

        Parameters
        ----------
        solver : z3.Solver
        positions : list of (x, y, w, h) tuples of z3.ArithRef
        no_overlap : bool
            If True, add pairwise non-overlap constraints.
        """
        # Positivity constraints
        for x, y, w, h in positions:
            solver.add(x >= 0)
            solver.add(y >= 0)
            solver.add(w > 0)
            solver.add(h > 0)

        if not no_overlap:
            return

        # Pairwise non-overlap: for any two rectangles A and B,
        # at least one separating condition must hold:
        #   A.right ≤ B.left  OR  B.right ≤ A.left  OR
        #   A.bottom ≤ B.top  OR  B.bottom ≤ A.top
        n = len(positions)
        for i in range(n):
            ax, ay, aw, ah = positions[i]
            for j in range(i + 1, n):
                bx, by, bw, bh = positions[j]
                solver.add(
                    z3.Or(
                        ax + aw <= bx,  # A right of B
                        bx + bw <= ax,  # B right of A
                        ay + ah <= by,  # A below B
                        by + bh <= ay,  # B below A
                    )
                )

        logger.debug(
            "Layout constraint: %d elements, no_overlap=%s", n, no_overlap
        )

    def encode_containment_constraint(
        self,
        solver: z3.Solver,
        child: tuple[z3.ArithRef, z3.ArithRef, z3.ArithRef, z3.ArithRef],
        parent: tuple[z3.ArithRef, z3.ArithRef, z3.ArithRef, z3.ArithRef],
    ) -> None:
        """Ensure child rectangle is fully contained within parent."""
        cx, cy, cw, ch = child
        px, py, pw, ph = parent

        solver.add(cx >= px)
        solver.add(cy >= py)
        solver.add(cx + cw <= px + pw)
        solver.add(cy + ch <= py + ph)

    # ── Hierarchy Depth ---------------------------------------------------

    def encode_hierarchy_constraint(
        self,
        solver: z3.Solver,
        depth_var: z3.ArithRef,
        max_depth: int,
    ) -> None:
        """Limit menu / navigation hierarchy depth.

        Deep hierarchies increase navigation time and cognitive load.
        The *max_depth* bound typically comes from UX guidelines
        (≤ 3 levels for primary navigation).

        Parameters
        ----------
        solver : z3.Solver
        depth_var : z3.ArithRef
            Integer variable for hierarchy depth.
        max_depth : int
            Maximum allowed depth.
        """
        if max_depth < 1:
            max_depth = 1

        solver.add(depth_var >= 1)
        solver.add(depth_var <= max_depth)

        logger.debug("Hierarchy constraint: depth ≤ %d", max_depth)

    # ── Total Cost --------------------------------------------------------

    def encode_total_cost_constraint(
        self,
        solver: z3.Solver,
        cost_vars: list[z3.ArithRef],
        max_total: float,
    ) -> None:
        """Bound the sum of component costs.

        ∑ cost_i  ≤  max_total

        Parameters
        ----------
        solver : z3.Solver
        cost_vars : list of z3.ArithRef
            Real variables for individual cost components.
        max_total : float
            Upper bound on total cost.
        """
        if not cost_vars:
            return

        # Non-negativity
        for cv in cost_vars:
            solver.add(cv >= 0)

        total = z3.Sum(cost_vars) if len(cost_vars) > 1 else cost_vars[0]
        solver.add(total <= z3.RealVal(str(max_total)))

        logger.debug(
            "Total cost constraint: sum of %d vars ≤ %.2f",
            len(cost_vars), max_total,
        )

    def encode_cost_improvement_constraint(
        self,
        solver: z3.Solver,
        original_cost: float,
        new_cost_var: z3.ArithRef,
        min_improvement_ratio: float = 0.1,
    ) -> None:
        """Ensure the new cost is at least *min_improvement_ratio* better.

        new_cost ≤ original_cost · (1 − min_improvement_ratio)
        """
        bound = original_cost * (1.0 - min_improvement_ratio)
        solver.add(new_cost_var <= z3.RealVal(str(bound)))
        solver.add(new_cost_var >= 0)
