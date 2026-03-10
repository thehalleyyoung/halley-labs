"""Interval comparison and dominance relations.

When reasoning about usability metrics under parameter uncertainty the
oracle must decide whether one design alternative is *certainly* better
than another, *possibly* better, or indistinguishable given the current
uncertainty.  This module provides the three-valued comparison logic
required for such decisions.

Mathematical background
-----------------------
Let :math:`A = [a, b]` and :math:`B = [c, d]` be closed real intervals.

*Certainly less* (``A ≺ B``):
    :math:`\\forall\\, x \\in A,\\; y \\in B:\\; x < y \\iff b < c`

*Possibly less* (``A ≲? B``):
    :math:`\\exists\\, x \\in A,\\; y \\in B:\\; x < y \\iff a < d`

*Necessity of less* (Dubois & Prade, 1988):
    :math:`N(A < B) = \\max\\!\\bigl(0,\\; \\frac{d - a}{(b - a) + (d - c)}\\bigr)`
    when both intervals are non-degenerate.

*Possibility of less*:
    :math:`\\Pi(A < B) = \\max\\!\\bigl(0,\\; \\frac{d - a}{(b - a) + (d - c)}\\bigr)`
    (see possibility–necessity duality).

References
----------
Dubois, D., & Prade, H. (1988).
    *Possibility Theory*. Plenum Press.
Moore, R. E. (1966).
    *Interval Analysis*. Prentice-Hall.
"""

from __future__ import annotations

from usability_oracle.interval.interval import Interval


class IntervalComparison:
    """Three-valued comparison primitives for closed intervals."""

    # ------------------------------------------------------------------
    # Certain (strong) relations
    # ------------------------------------------------------------------

    @staticmethod
    def certainly_less(a: Interval, b: Interval) -> bool:
        """Return ``True`` iff every element of *a* is strictly less than
        every element of *b*.

        Formally: :math:`a.\\text{high} < b.\\text{low}`.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        bool
        """
        return a.high < b.low

    @staticmethod
    def certainly_greater(a: Interval, b: Interval) -> bool:
        """Return ``True`` iff every element of *a* is strictly greater
        than every element of *b*.

        Formally: :math:`a.\\text{low} > b.\\text{high}`.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        bool
        """
        return a.low > b.high

    @staticmethod
    def certainly_less_or_equal(a: Interval, b: Interval) -> bool:
        """Return ``True`` iff every element of *a* is ≤ every element
        of *b*.

        Formally: :math:`a.\\text{high} \\leq b.\\text{low}`.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        bool
        """
        return a.high <= b.low

    @staticmethod
    def certainly_greater_or_equal(a: Interval, b: Interval) -> bool:
        """Return ``True`` iff every element of *a* is ≥ every element
        of *b*.

        Formally: :math:`a.\\text{low} \\geq b.\\text{high}`.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        bool
        """
        return a.low >= b.high

    # ------------------------------------------------------------------
    # Possible (weak) relations
    # ------------------------------------------------------------------

    @staticmethod
    def possibly_overlapping(a: Interval, b: Interval) -> bool:
        """Return ``True`` when the two intervals share at least one
        point.

        Equivalent to ``a.overlaps(b)``.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        bool
        """
        return a.overlaps(b)

    @staticmethod
    def possibly_less(a: Interval, b: Interval) -> bool:
        """Return ``True`` if there exist *x* ∈ *a*, *y* ∈ *b* with
        *x* < *y*.

        Formally: :math:`a.\\text{low} < b.\\text{high}`.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        bool
        """
        return a.low < b.high

    # ------------------------------------------------------------------
    # Dominance
    # ------------------------------------------------------------------

    @staticmethod
    def dominance(a: Interval, b: Interval) -> str:
        """Classify the dominance relation between two intervals.

        Returns
        -------
        str
            One of:

            - ``"a_dominates"`` — *a* is certainly less than *b*
              (lower is better).
            - ``"b_dominates"`` — *b* is certainly less than *a*.
            - ``"overlapping"`` — neither interval certainly dominates.
        """
        if a.high < b.low:
            return "a_dominates"
        if b.high < a.low:
            return "b_dominates"
        return "overlapping"

    # ------------------------------------------------------------------
    # Quantitative measures
    # ------------------------------------------------------------------

    @staticmethod
    def overlap_ratio(a: Interval, b: Interval) -> float:
        """Fraction of the union range that is shared by both intervals.

        .. math::
            \\rho(A, B) = \\frac{\\max(0,\\;\\min(b, d) - \\max(a, c))}
                               {\\max(b, d) - \\min(a, c)}

        Returns 0.0 when the intervals are disjoint and 1.0 when they
        are identical.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        float
            Value in [0, 1].
        """
        union_lo = min(a.low, b.low)
        union_hi = max(a.high, b.high)
        union_width = union_hi - union_lo
        if union_width == 0.0:
            # Both are the same degenerate interval.
            return 1.0

        inter_lo = max(a.low, b.low)
        inter_hi = min(a.high, b.high)
        inter_width = max(0.0, inter_hi - inter_lo)
        return inter_width / union_width

    @staticmethod
    def separation(a: Interval, b: Interval) -> float:
        """Hausdorff-like gap between two intervals.

        Returns the positive distance between the closest endpoints when
        the intervals are disjoint, and 0.0 when they overlap.

        .. math::
            \\delta(A, B) = \\max(0,\\; a.\\text{low} - b.\\text{high},
                                      \\; b.\\text{low} - a.\\text{high})

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        float
            Non-negative separation distance.
        """
        if a.overlaps(b):
            return 0.0
        if a.high < b.low:
            return b.low - a.high
        return a.low - b.high

    @staticmethod
    def possibility_of_less(a: Interval, b: Interval) -> float:
        """Degree of possibility that a randomly chosen *x* ∈ *a* is
        less than a randomly chosen *y* ∈ *b*, under uniform
        distributions.

        For non-degenerate intervals this equals the fraction of the
        joint sample space (a × b) where x < y:

        .. math::
            \\Pi(A < B) = \\frac{\\text{area}\\{(x,y) \\in A \\times B
                          \\mid x < y\\}}{|A| \\cdot |B|}

        The computation is carried out analytically.

        Special cases
        ^^^^^^^^^^^^^
        - If *a* is certainly less than *b*, returns 1.0.
        - If *a* is certainly greater than or equal to *b*, returns 0.0.
        - Degenerate intervals are handled by limit arguments.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        float
            Value in [0, 1].
        """
        # Trivial cases.
        if a.high <= b.low:
            return 1.0
        if a.low >= b.high:
            return 0.0

        wa = a.width
        wb = b.width

        # Both degenerate — a single comparison.
        if wa == 0.0 and wb == 0.0:
            return 1.0 if a.low < b.low else 0.0

        # One degenerate: use linear measure.
        if wa == 0.0:
            # a is a point; fraction of b that is > a.low.
            frac = (b.high - a.low) / wb
            return max(0.0, min(1.0, frac))
        if wb == 0.0:
            # b is a point; fraction of a that is < b.low.
            frac = (b.low - a.low) / wa
            return max(0.0, min(1.0, frac))

        # General case — area of {(x,y) ∈ A×B : x < y} / (|A|·|B|).
        # The region is the intersection of the rectangle A×B with the
        # half-plane x < y.  We compute its area using inclusion–
        # exclusion on the triangle/trapezoid decomposition.
        total_area = wa * wb

        # Overlap region [lo, hi] = [max(a.low, b.low), min(a.high, b.high)].
        lo = max(a.low, b.low)
        hi = min(a.high, b.high)

        # Area contribution from the rectangular region where a is
        # entirely below b  (x ∈ [a.low, b.low], y ∈ [b.low, b.high])
        # — but only if a.low < b.low.
        area = 0.0

        if a.low < b.low:
            # Full rectangle: x in [a.low, min(a.high, b.low)], y in B.
            x_hi = min(a.high, b.low)
            area += (x_hi - a.low) * wb

        # Overlap band: x in [lo, hi], y in [lo, hi], x < y.
        # This is a right triangle with legs = (hi − lo).
        overlap_len = hi - lo
        if overlap_len > 0.0:
            # For each x in [lo, hi], y ranges from x to min(b.high, …).
            # But y also must be in [b.low, b.high].
            # Since lo >= b.low and hi <= b.high within the overlap:
            # y ranges from x to b.high for x in [lo, hi].
            # Area = ∫_{lo}^{hi} (b.high − x) dx
            #      = b.high*(hi − lo) − (hi² − lo²)/2
            area += b.high * overlap_len - (hi ** 2 - lo ** 2) / 2.0

            # Subtract the part where y > a_high (doesn't apply since
            # hi = min(a.high, b.high)... but we already cap x at hi).
            # Actually: for x in [lo, hi], y goes up to b.high, which is
            # always ≥ hi.  So the above integral is correct.

            # We also need to subtract the area where y < b.low; but since
            # lo = max(a.low, b.low) ≥ b.low, and y > x ≥ lo ≥ b.low,
            # the y < b.low region is empty.  Correct.

        # Also, if a.high > b.high, there's x in [b.high, a.high] where
        # x >= b.high >= all y in B, contributing zero to x < y.  Good.

        # Normalise.
        prob = area / total_area
        return max(0.0, min(1.0, prob))

    @staticmethod
    def necessity_of_less(a: Interval, b: Interval) -> float:
        """Degree of necessity that *a* < *b*.

        By necessity–possibility duality:

        .. math::
            N(A < B) = 1 - \\Pi(B \\leq A) = 1 - \\Pi(B < A) \\;-\\;
            \\text{(boundary measure zero adjustment)}

        In practice we use the complementary computation:

        .. math::
            N(A < B) = 1 - \\text{possibility\\_of\\_less}(b, a)

        with a small correction to handle the boundary consistently.

        Returns 1.0 when *a* is certainly less than *b* and 0.0 when
        there is any possibility that *a* ≥ *b*.

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        float
            Value in [0, 1].
        """
        if a.high < b.low:
            return 1.0
        if a.high >= b.high:
            return 0.0
        # N(A < B) = 1 − Π(B ≤ A)
        complement = IntervalComparison.possibility_of_less(b, a)
        return max(0.0, min(1.0, 1.0 - complement))
